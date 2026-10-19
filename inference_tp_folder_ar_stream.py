"""Folder AR inference with asynchronous VAE decoding and a live browser view.

This entry keeps the DiT autoregressive path of ``inference_tp_folder_ar.py`` but does not wait
for the whole video before running the VAE:

    DiT chunk k (GPU0)  ──►  latent snapshot  ──►  async VAE window (GPU1)
                                                   │
                                                   ├──► paced MJPEG stream (browser)
                                                   └──► [--package-video] MP4 encoder (CPU x264)

The VAE decodes ``current chunk + previous chunk`` as soon as the chunk's last denoising step is
written back.  Overlapping windows are de-duplicated by a monotonic pixel cursor, so the browser
receives 25, 32, 32 ... frames per chunk while the next chunk is still being generated.

Example (two audio files, TP=1, VAE on the second GPU):

    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=1 inference_tp_folder_ar_stream.py \
        --distilled-checkpoint-path checkpoints/avatarforever-ltx-2.3-22b.safetensors \
        --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
        --audio-dir audio --max-files 2 \
        --output-dir output/folder_ar_tp1_stream \
        --height 192 --width 352 --frame-rate 20 --fast-infer \
        --decode-device auto --port 8080

Open ``http://<host>:8080/`` while the process is running; finished MP4s are linked at the bottom
of the page when ``--package-video`` is enabled.  MP4 packaging is off by default: the browser
stream and audio are unaffected, and skipping it removes the per-job x264/mux cost.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch

# Heavy project imports intentionally stay below the CUDA/device setup in ``main``.

STREAMING_DEFAULT_PROMPT = (
    "A friendly young Chinese man sits in a softly lit home office in a medium close-up shot, "
    "facing the camera with natural eye contact. Realistic conversational video-call framing, "
    "subtle head motion, natural blinking, soft facial expressions, clear lip sync, gentle breathing, "
    "and small relaxed hand gestures while speaking.\n\n"
    "Natural audio-driven speaking motion with accurate lip synchronization, smooth and continuous facial "
    "animation, subtle head movement, natural blinking, gentle breathing, and relaxed upper-body motion. "
    "Expressions and gestures should respond naturally to the rhythm, tone, and emotion of the speech "
    "while remaining restrained and realistic.\n\n"
    "Maintain strong temporal consistency across all frames. Keep facial appearance, identity, pose, body "
    "structure, clothing details, lighting, and background stable throughout the video. Avoid sudden motion "
    "changes, excessive gestures, unnatural expression shifts, frame-to-frame appearance variation, "
    "flickering, jitter, ghosting, texture instability, temporal artifacts, or deformation. All motion "
    "should be coherent, fluid, stable, and naturally driven by the audio."
)

DEFAULT_CONFIG_FILENAME = "stream_config.yaml"
CONFIG_ENV_VAR = "AVATAR_STREAM_CONFIG"
DEFAULT_FRONTEND_CONFIG_FILENAME = "frontend_config.yaml"
FRONTEND_CONFIG_ENV_VAR = "AVATAR_FRONTEND_CONFIG"

FRONTEND_CONFIG_DEFAULTS: dict[str, Any] = {
    "audio_stall_ms": 60,
    "audio_drift_seconds": 0.20,
    "status_poll_ms": 100,
    "sync_interval_ms": 50,
    "status_timeout_ms": 2500,
    "stream_reconnect_ms": 3000,
    "audio_retry_backoff_ms": 5000,
    "mjpeg_keepalive_seconds": 0.25,
    "sound_enabled": True,
}

# name -> (kind, minimum, maximum)
FRONTEND_CONFIG_RULES: dict[str, tuple[str, float, float]] = {
    "audio_stall_ms": ("int", 0, 5000),
    "audio_drift_seconds": ("float", 0.0, 10.0),
    "status_poll_ms": ("int", 20, 10000),
    "sync_interval_ms": ("int", 10, 5000),
    "status_timeout_ms": ("int", 100, 60000),
    "stream_reconnect_ms": ("int", 0, 60000),
    "audio_retry_backoff_ms": ("int", 0, 60000),
    "mjpeg_keepalive_seconds": ("float", 0.05, 60.0),
}


def _default_config_path() -> Path:
    """Config file shipped next to this entry point (auto-loaded when present)."""
    return Path(__file__).resolve().parent / DEFAULT_CONFIG_FILENAME


def _preparse_config_args(argv: list[str]) -> argparse.Namespace:
    """Read --config/--no-config before the full parser checks required options."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre.add_argument("--no-config", action="store_true")
    known, _ = pre.parse_known_args(argv)
    return known


def _load_config_file(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is part of the runtime image
        raise SystemExit("读取 YAML 配置需要 PyYAML，请先执行：pip install pyyaml") from exc
    try:
        text = path.expanduser().read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"无法读取配置文件 {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SystemExit(f"配置文件 {path} 不是合法 YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(f"配置文件 {path} 的顶层必须是键值映射（YAML mapping）。")
    return data


def _apply_config_defaults(
    parser: argparse.ArgumentParser,
    values: dict[str, Any],
    source: Path,
) -> list[str]:
    """Turn a flat YAML mapping into argparse defaults; CLI arguments still win.

    Keys may use either the command-line spelling (``max-files``) or the argparse destination
    spelling (``max_files``).  Boolean-optional switches also accept a ``no-`` prefix, e.g.
    ``no-trim-to-full-chunks: false`` means "keep trimming on".
    """
    actions = {action.dest: action for action in parser._actions if action.dest != "help"}
    resolved: dict[str, Any] = {}
    for raw_key, value in values.items():
        key = str(raw_key).strip().replace("-", "_")
        if key in ("config", "no_config"):
            continue
        negate = False
        if key not in actions and key.startswith("no_") and key[3:] in actions:
            key = key[3:]
            negate = True
        if key not in actions:
            parser.error(f"配置文件 {source} 中存在未知选项：{raw_key!r}")
        if negate:
            if not isinstance(value, bool):
                parser.error(f"配置文件 {source} 中的 {raw_key!r} 必须是 true/false。")
            value = not value
        resolved[key] = value

    for key, value in resolved.items():
        action = actions[key]
        # The base parser marks the two model paths required; a config value satisfies them instead.
        if action.required:
            action.required = False
        parser.set_defaults(**{key: value})
    return sorted(resolved)


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("configuration")
    group.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            f"YAML config file. Default: ${CONFIG_ENV_VAR} if set, otherwise "
            f"{DEFAULT_CONFIG_FILENAME} next to this script when it exists."
        ),
    )
    group.add_argument(
        "--no-config",
        action="store_true",
        help="Ignore the default YAML config; explicit --config still loads.",
    )
    group.add_argument(
        "--frontend-config",
        type=Path,
        default=None,
        help=(
            f"Frontend YAML config. Default: ${FRONTEND_CONFIG_ENV_VAR} if set, otherwise "
            f"{DEFAULT_FRONTEND_CONFIG_FILENAME} next to this script when it exists."
        ),
    )
    group.add_argument(
        "--no-frontend-config",
        action="store_true",
        help="Use built-in frontend defaults and ignore frontend_config.yaml.",
    )


def _load_frontend_config(
    path: Path | None,
    *,
    disabled: bool,
) -> tuple[dict[str, Any], Path | None]:
    """Load and validate the browser-side timing/audio config.

    Returns ``(config, source_path)``.  Missing files fall back to the built-in defaults so the
    stream entry still starts if the frontend config was not shipped.
    """
    config = dict(FRONTEND_CONFIG_DEFAULTS)
    if disabled:
        return config, None

    source = path
    if source is None:
        env_path = os.environ.get(FRONTEND_CONFIG_ENV_VAR)
        if env_path:
            source = Path(env_path)
        else:
            candidate = Path(__file__).resolve().parent / DEFAULT_FRONTEND_CONFIG_FILENAME
            if candidate.is_file():
                source = candidate
    if source is None:
        return config, None

    source = source.expanduser()
    if not source.is_file():
        raise SystemExit(f"前端配置文件不存在：{source}")
    values = _load_config_file(source)
    for raw_key, value in values.items():
        key = str(raw_key).strip().replace("-", "_")
        if key == "sound_enabled":
            if not isinstance(value, bool):
                raise SystemExit(f"前端配置 {source} 中的 {raw_key!r} 必须是 true/false。")
            config[key] = value
            continue
        rule = FRONTEND_CONFIG_RULES.get(key)
        if rule is None:
            valid = ", ".join(sorted([*FRONTEND_CONFIG_RULES, "sound_enabled"]))
            raise SystemExit(
                f"前端配置 {source} 中存在未知选项：{raw_key!r}；可用选项：{valid}"
            )
        kind, minimum, maximum = rule
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SystemExit(f"前端配置 {source} 中的 {raw_key!r} 必须是数字。")
        value = int(value) if kind == "int" else float(value)
        if not minimum <= value <= maximum:
            raise SystemExit(
                f"前端配置 {source} 中的 {raw_key!r}={value} 超出范围 "
                f"[{minimum}, {maximum}]。"
            )
        config[key] = value
    return config, source


def _resolve_decode_device(spec: str | None, main_device: torch.device) -> torch.device:
    if spec in (None, "", "auto"):
        count = torch.cuda.device_count()
        if count > 1:
            return torch.device("cuda", (int(main_device.index or 0) + 1) % count)
        return main_device
    if spec == "same":
        return main_device
    return torch.device(spec)


def _ipv4_addresses() -> list[str]:
    """Best-effort list of non-loopback IPv4 addresses of this VM/container."""
    values: list[str] = []
    try:
        values = subprocess.check_output(["hostname", "-I"], text=True, timeout=2).split()
    except Exception:  # noqa: BLE001
        try:
            values = socket.gethostbyname_ex(socket.gethostname())[2]
        except OSError:
            values = []
    result: list[str] = []
    for value in values:
        if ":" in value or value.startswith("127.") or value in result:
            continue
        result.append(value)
    return result


def _print_access_info(*, bind_host: str, port: int, public_host: str | None) -> str:
    """Print local, candidate and SSH-tunnel addresses; return the URL to show in summaries."""
    local_url = f"http://127.0.0.1:{port}/"
    candidates = _ipv4_addresses()
    vm_url = f"http://{candidates[0]}:{port}/" if candidates else local_url
    primary_url = f"http://{public_host}:{port}/" if public_host else vm_url

    print(f"前端监听：{bind_host}:{port}")
    if public_host:
        print(f"前端地址（--public-host）：{primary_url}")
    else:
        print(f"主机浏览器访问：{primary_url}")
    print(f"VM 内自测：{local_url}")
    print(f"状态接口：{local_url}status.json")

    if candidates:
        print("其它候选地址：")
        for address in candidates[1:7]:
            print(f"  http://{address}:{port}/")

    print("备用访问（直连 VM IP 不通时）：在本地主机执行")
    ssh_connection = os.environ.get("SSH_CONNECTION", "").split()
    server_ip = ssh_connection[2] if len(ssh_connection) >= 3 else "<vm-ip>"
    print(f"  ssh -N -L {port}:127.0.0.1:{port} <user>@{server_ip}")
    print(f"  然后浏览器打开 {local_url}")
    print("  如果使用 VS Code Remote/网页终端，请使用其 Ports/端口转发功能；")
    print("  如果是 VirtualBox/VMware NAT，请在虚拟机设置里把主机端口转发到 guest 的该端口。")
    return primary_url


def _add_stream_arguments(parser) -> None:  # noqa: ANN001
    stream = parser.add_argument_group("streaming")
    stream.add_argument("--host", default="0.0.0.0", help="HTTP bind address (default: 0.0.0.0).")
    stream.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080; 0 picks a free port).")
    stream.add_argument("--public-host", default=None, help="Host shown in the printed URL.")
    stream.add_argument(
        "--decode-device",
        default="auto",
        help="VAE device: auto (second visible GPU if available), same (DiT device), or cuda:N.",
    )
    stream.add_argument(
        "--decode-window-latent-frames",
        type=int,
        default=None,
        help="VAE decode window in latent frames. Default: 2 x --ar-video-chunk-size.",
    )
    stream.add_argument(
        "--no-stream-pacing",
        action="store_true",
        help="Publish decoded frames as fast as possible instead of pacing them at --frame-rate.",
    )
    stream.add_argument("--jpeg-quality", type=int, default=85, help="MJPEG JPEG quality (default: 85).")
    stream.add_argument(
        "--package-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Encode and mux an MP4 for every job (default: off). Live MJPEG/audio streaming and "
            "cross-job continuity are unaffected; without it no /video/<job> files exist."
        ),
    )
    stream.add_argument(
        "--linger-seconds",
        type=float,
        default=300.0,
        help="Keep the web server alive after the last job (default: 300 s; 0 exits immediately).",
    )
    watch = parser.add_argument_group("continuous audio watching")
    watch.add_argument(
        "--watch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep watching --audio-dir and process new audio files as they appear.",
    )
    watch.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Directory scan interval in seconds when --watch is on (default: 1.0).",
    )
    watch.add_argument(
        "--stable-seconds",
        type=float,
        default=1.0,
        help="Only process a new file after its size/mtime stay unchanged for this long (default: 1.0).",
    )
    watch.add_argument(
        "--watch-sort",
        choices=("name", "mtime"),
        default="name",
        help="Order for new files in --watch mode (default: name).",
    )
    watch.add_argument(
        "--idle-exit-seconds",
        type=float,
        default=0.0,
        help="Exit watching after this many seconds without a new file; 0 means never (default: 0).",
    )


def main() -> None:  # noqa: PLR0915
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise SystemExit(
            "inference_tp_folder_ar_stream.py currently supports TP=1 only "
            f"(got WORLD_SIZE={world_size})."
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    from inference import build_arg_parser
    from inference_tp_folder_ar import (
        _add_folder_arguments,
        _attach_job_file_handler,
        _audio_duration_seconds,
        _detach_job_file_handler,
        _frames_for_duration,
        _job_log_path,
        _list_audio_files,
        _output_path,
        _relax_required_args,
        _release_pipeline,
    )
    from ltx_pipelines.utils.autoregressive import build_progressive_chunk_ranges
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.model_ledger import ModelLedger
    from ltx_tp.tp_shim import TPContext, shard_transformer_model_
    from util import (
        build_first_frame_images,
        build_tiling_config,
        encode_first_frame_channel_condition,
        prepare_padded_audio,
    )

    from stream_server import ServerState, start_stream_server
    from streaming_vae import AsyncChunkVAEDecoder, FrameBroadcaster, StreamingCrossVideoARPipeline

    parser = build_arg_parser()
    _relax_required_args(parser)
    _add_folder_arguments(parser)
    _add_stream_arguments(parser)
    _add_config_arguments(parser)
    # The base parser defaults --prompt to inference.DEFAULT_PROMPT.  This entry is scene-driven,
    # so use the home-office prompt above unless the user explicitly passes --prompt.
    parser.set_defaults(prompt=STREAMING_DEFAULT_PROMPT)

    # Layer order: built-in defaults < YAML config < command-line flags.
    pre = _preparse_config_args(sys.argv[1:])
    config_path: Path | None = pre.config
    if config_path is None and not pre.no_config:
        env_config = os.environ.get(CONFIG_ENV_VAR)
        if env_config:
            config_path = Path(env_config)
        elif _default_config_path().is_file():
            config_path = _default_config_path()
    if config_path is not None:
        config_path = config_path.expanduser()
        config_values = _load_config_file(config_path)
        applied = _apply_config_defaults(parser, config_values, config_path)
        print(f"已加载配置：{config_path}（{len(applied)} 项；命令行参数优先）")
        if applied:
            print("  " + ", ".join(applied))

    args = parser.parse_args()
    missing = [
        flag
        for flag, value in (
            ("--distilled-checkpoint-path", args.distilled_checkpoint_path),
            ("--gemma-root", args.gemma_root),
        )
        if value in (None, "")
    ]
    if missing:
        parser.error("缺少必填参数（命令行或配置文件）：" + ", ".join(missing))
    if args.window_chunks < 1:
        parser.error("--window-chunks must be >= 1")
    if args.decode_window_latent_frames is not None and args.decode_window_latent_frames < 1:
        parser.error("--decode-window-latent-frames must be >= 1")

    # The async path must keep the DiT/Gemma/VAE modules resident across chunks and jobs.
    args.fast_infer = True
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.info("async streaming entry: forcing --fast-infer")

    decode_device = _resolve_decode_device(args.decode_device, device)
    logging.info("DiT device=%s, async VAE device=%s", device, decode_device)

    frontend_config, frontend_config_path = _load_frontend_config(
        args.frontend_config,
        disabled=args.no_frontend_config,
    )
    if frontend_config_path is not None:
        print(f"已加载前端配置：{frontend_config_path}（{len(frontend_config)} 项）")
    else:
        print("前端配置：使用内置默认值")

    broadcaster = FrameBroadcaster(
        quality=args.jpeg_quality,
        keepalive_seconds=float(frontend_config["mjpeg_keepalive_seconds"]),
    )
    state = ServerState(broadcaster=broadcaster, frontend_config=frontend_config)
    state.update(package_video=bool(args.package_video))
    httpd, port = start_stream_server(host=args.host, port=args.port, state=state)
    url = _print_access_info(bind_host=args.host, port=port, public_host=args.public_host)
    logging.info("stream server listening on %s:%d", args.host, port)

    tp = TPContext(rank=local_rank, world=world_size, device=device, verbose=True)
    original_transformer = ModelLedger.transformer

    def tp_transformer(self, extra_config=None):  # noqa: ANN001, ANN202
        module = original_transformer(self, extra_config=extra_config)
        if not getattr(module, "_tp_sharded", False):
            shard_transformer_model_(module, tp)
            module._tp_sharded = True
        return module

    ModelLedger.transformer = tp_transformer

    pipeline = StreamingCrossVideoARPipeline(
        distilled_checkpoint_path=str(args.distilled_checkpoint_path),
        spatial_upsampler_path=None,
        gemma_root=str(args.gemma_root),
        loras=[],
        device=device,
    )
    images = build_first_frame_images(
        args.first_frame_condition_image_path,
        strength=args.first_frame_image_strength,
        crf=args.first_frame_image_crf,
    )
    first_frame_channel_condition_latent = encode_first_frame_channel_condition(
        pipeline,
        image_path=args.first_frame_condition_image_path,
        enabled=args.first_frame_channel_condition,
        height=args.height,
        width=args.width,
        crf=args.first_frame_image_crf,
    )
    derive_channel_condition_from_first_chunk = (
        args.first_frame_channel_condition and first_frame_channel_condition_latent is None
    )
    tiling_config = build_tiling_config(args)
    prompt = args.prompt or STREAMING_DEFAULT_PROMPT

    audio_dir = Path(args.audio_dir).expanduser()
    try:
        files = _list_audio_files(audio_dir)
    except Exception as exc:  # noqa: BLE001
        state.update(phase="error", error=str(exc))
        raise
    if args.max_files is not None:
        files = files[: args.max_files]
    state.set_audio_files(files)
    state.update(
        phase="loading",
        message="模型已加载，准备生成",
        job_total=0 if args.watch else len(files),
        watch=bool(args.watch),
        waiting_for_next=False,
        live_job_done=False,
        processed_files=0,
        generating_job_index=None,
        generating_job_name=None,
        num_frames=0,
        frames_decoded=0,
        frames_published=0,
        package_video=bool(args.package_video),
    )
    print(f"文件夹模式：{args.audio_dir} -> {len(files)} 个音频文件（按文件名排序）")
    for position, path in enumerate(files, 1):
        print(f"  {position:3d}. {path.name}")
    if args.watch:
        print(f"持续监听模式：每 {args.poll_interval:.1f}s 扫描 {audio_dir}；新文件稳定 {args.stable_seconds:.1f}s 后处理")
    elif not files:
        print("没有可处理的音频文件，退出。")
        state.update(phase="done", message="没有音频文件")

    summary: list[dict[str, Any]] = []
    sink_video = sink_audio = None
    history_video = history_audio = None
    history_video_ranges: list[tuple[int, int]] = []
    history_audio_ranges: list[tuple[int, int]] = []
    summary_path = args.summary_path or (args.output_dir / "streaming_summary.json")
    first_start_event = threading.Event()
    first_start_event.set()
    pending_playback_events: list[threading.Event] = []
    processed_paths: set[Path] = set()
    candidate_states: dict[Path, tuple[tuple[int, int], float]] = {}

    def write_summary() -> None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    def process_job(
        audio_path: Path,
        index: int,
        *,
        job_total: int,
        is_last: bool,
        start_event: threading.Event,
        wait_for_playback: bool,
    ) -> AsyncChunkVAEDecoder:
        nonlocal sink_video, sink_audio, history_video, history_audio
        nonlocal history_video_ranges, history_audio_ranges

        duration = _audio_duration_seconds(audio_path)
        num_frames = args.num_frames or _frames_for_duration(duration, args.frame_rate)
        expanded = False
        # Expand non-final videos to the next full AR chunk boundary. In watch mode every job is
        # treated as potentially non-final because the next file is not known yet.
        if (
            args.trim_to_full_chunks
            and (args.watch or not is_last)
            and args.num_frames is None
        ):
            latent_frames = (num_frames - 1) // 8 + 1
            full_latent_frames = (
                (latent_frames + args.ar_video_chunk_size - 1) // args.ar_video_chunk_size
            ) * args.ar_video_chunk_size
            if full_latent_frames > latent_frames:
                num_frames = 8 * (full_latent_frames - 1) + 1
                expanded = True

        latent_frames = (num_frames - 1) // 8 + 1
        chunk_ranges = build_progressive_chunk_ranges(latent_frames, args.ar_video_chunk_size)
        history_indices = list(
            range(max(0, len(chunk_ranges) - (args.window_chunks - 1)), len(chunk_ranges))
        )

        pipeline.set_cross_video_context(
            sink_video=sink_video if args.carry_history else None,
            sink_audio=sink_audio if args.carry_history else None,
            history_video=history_video if args.carry_history else None,
            history_audio=history_audio if args.carry_history else None,
            history_video_ranges=history_video_ranges if args.carry_history else [],
            history_audio_ranges=history_audio_ranges if args.carry_history else [],
            window_chunks=args.window_chunks,
        )
        sink_first_chunk = args.ar_sink_first_chunk and (
            not args.carry_history or index == 1 or not args.sink_first_video_chunk
        )
        effective_sink = sink_video if args.carry_history else None
        effective_history = history_video if args.carry_history else None

        output_path = _output_path(args, index=index, audio_stem=audio_path.stem)
        log_path = _job_log_path(output_path)
        padded_audio_path = prepare_padded_audio(
            audio_path,
            video_duration=num_frames / args.frame_rate,
            output_dir=args.output_dir,
            index=index,
        )
        state.register_audio(index, padded_audio_path)
        video_chunks_number = num_frames
        job_log_handler = _attach_job_file_handler(log_path)
        start_time = time.time()

        def on_job_start(session: AsyncChunkVAEDecoder) -> None:
            state.current_session = session
            state.update(
                phase="generating",
                job_index=session.job_id,
                job_name=session.job_name,
                audio=session.audio_path,
                num_frames=session.total_frames,
                fps=session.fps,
                frames_decoded=0,
                frames_published=0,
                decoded_chunks=0,
                last_decode_ms=0,
                decode_window=[],
                live_job_done=False,
                waiting_for_next=False,
                error=None,
            )

        def on_job_done(session: AsyncChunkVAEDecoder) -> None:
            state.update(live_job_done=True, frames_published=session.total_frames)
            if state.status.get("waiting_for_next"):
                state.update(phase="waiting", message="等待下一段音频")

        session = AsyncChunkVAEDecoder(
            device=decode_device,
            dtype=pipeline.dtype,
            chunk_size=args.ar_video_chunk_size,
            total_latent_frames=latent_frames,
            total_frames=num_frames,
            fps=args.frame_rate,
            seed=args.seed,
            window_latent_frames=args.decode_window_latent_frames,
            broadcaster=broadcaster,
            pace_frames=not args.no_stream_pacing,
            jpeg_quality=args.jpeg_quality,
            on_progress=state.update,
            job_id=index,
            job_name=audio_path.name,
            audio_path=str(audio_path),
            start_event=start_event,
            on_job_start=on_job_start,
            on_job_done=on_job_done,
        )
        pipeline.set_streaming_session(session)
        state.update(
            phase="generating",
            generating_job_index=index,
            generating_job_name=audio_path.name,
            generating_num_frames=num_frames,
            generating_chunks=len(chunk_ranges),
            message=f"正在生成 job {index}",
            waiting_for_next=False,
            error=None,
        )

        context = {
            "index": index,
            "audio": str(audio_path),
            "num_frames": num_frames,
            "latent_frames": latent_frames,
            "chunks": len(chunk_ranges),
            "trimmed_to_full_chunks": False,
            "expanded_to_full_chunks": expanded,
            "padded_audio": str(padded_audio_path),
            "sink_video_tokens": int(effective_sink.shape[1]) if effective_sink is not None else 0,
            "carried_video_tokens": int(effective_history.shape[1]) if effective_history is not None else 0,
            "carried_chunks": len(history_video_ranges) if args.carry_history else 0,
            "sink_source": "initial video chunk 0" if effective_sink is not None else "-",
            "history_source": (
                f"previous video chunks {history_indices}"
                if (effective_history is not None and index > 1)
                else "-"
            ),
            "sink_first_chunk_in_video": bool(sink_first_chunk),
            "output": str(output_path) if args.package_video else None,
            "packaged": bool(args.package_video),
        }
        print(
            f"[job {index}] {audio_path.name}: {num_frames} 帧 / {len(chunk_ranges)} chunks；"
            f"DiT={device}, VAE={decode_device}, window={session.window_latent_frames} latent frames"
            + ("（已扩展到完整 chunk 边界，音轨补静音）" if expanded else "")
        )

        try:
            with torch.inference_mode():
                video, audio = pipeline(
                    prompt=prompt,
                    seed=args.seed,
                    height=args.height,
                    width=args.width,
                    num_frames=num_frames,
                    frame_rate=args.frame_rate,
                    images=images,
                    audio_path=str(padded_audio_path),
                    audio_start_time=0.0,
                    audio_max_duration=num_frames / args.frame_rate,
                    tiling_config=tiling_config,
                    enhance_prompt=False,
                    stage_mode="one-stage",
                    stage1_sigmas=args.stage1_sigmas,
                    ar_video_chunk_size=args.ar_video_chunk_size,
                    ar_history_chunk_count=args.ar_history_chunk_count,
                    ar_sink_first_chunk=sink_first_chunk,
                    ar_relative_positions=args.ar_relative_positions,
                    ar_history_feature_cache=args.ar_history_feature_cache,
                    ar_first_frame_prefix_condition=args.first_frame_prefix_condition,
                    ar_first_frame_condition_position=args.first_frame_condition_position,
                    first_frame_channel_condition_latent=first_frame_channel_condition_latent,
                    first_frame_channel_condition_init=args.first_frame_channel_condition_init,
                    first_frame_channel_condition_mode=args.first_frame_channel_condition_mode,
                    ar_first_frame_channel_condition_from_first_chunk=derive_channel_condition_from_first_chunk,
                    ar_channel_condition_current_chunk_only=args.first_frame_channel_condition,
                    fast_infer=True,
                )
            generation_end = time.time()
            if args.package_video:
                state.update(phase="encoding", message="DiT 完成，正在收尾 VAE/MP4")
                encode_video(
                    video=video,
                    fps=args.frame_rate,
                    audio=audio,
                    output_path=str(output_path),
                    video_chunks_number=video_chunks_number,
                    crf=args.video_crf,
                    preset=args.video_preset,
                )
            else:
                state.update(phase="encoding", message="DiT 完成，正在收尾 VAE")
            if wait_for_playback:
                session.finish()
            else:
                session.finish_decode()
        except Exception as exc:
            session.abort(exc)
            state.update(phase="error", error=str(exc))
            logging.exception("Job %d failed", index)
            raise

        pipeline.set_streaming_session(None)
        if args.package_video:
            state.add_video(index=index, audio=str(audio_path), output=output_path)
            state.update(message=f"job {index} 已生成")
        else:
            state.update(message=f"job {index} 已生成（未打包视频）")

        tail_video, tail_audio, tail_video_ranges, tail_audio_ranges = pipeline.extract_chunk_tokens(
            chunk_size=args.ar_video_chunk_size,
            chunk_indices=history_indices,
        )
        if index == 1 and args.sink_first_video_chunk:
            sink_video, sink_audio, _, _ = pipeline.extract_chunk_tokens(
                chunk_size=args.ar_video_chunk_size,
                chunk_indices=[0],
            )
        history_video, history_audio = tail_video, tail_audio
        history_video_ranges, history_audio_ranges = tail_video_ranges, tail_audio_ranges

        total_seconds = time.time() - start_time
        peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
        if decode_device != device:
            decode_peak_gib = torch.cuda.max_memory_allocated(decode_device) / 1024**3
        else:
            decode_peak_gib = peak_gib
        context.update(
            {
                "ar_seconds": round(generation_end - start_time, 3),
                "total_seconds_incl_encode": round(total_seconds, 3),
                "peak_dit_gpu_memory_GiB": round(peak_gib, 2),
                "peak_vae_gpu_memory_GiB": round(decode_peak_gib, 2),
                "dit_device": str(device),
                "vae_device": str(decode_device),
                "decode_window_latent_frames": session.window_latent_frames,
                "carried_out_video_tokens": int(tail_video.shape[1]) if tail_video is not None else 0,
                "carried_out_audio_tokens": int(tail_audio.shape[1]) if tail_audio is not None else 0,
                "carried_chunks_in_window_per_chunk": (
                    pipeline.last_window_stats or {}
                ).get("carried_chunks_in_window_per_chunk"),
                "window_sizes_per_chunk": (pipeline.last_window_stats or {}).get(
                    "selected_chunk_count_per_chunk"
                ),
                "log": str(log_path),
            }
        )
        summary.append(context)
        write_summary()
        _detach_job_file_handler(job_log_handler)
        if args.package_video:
            print(
                f"    完成：{output_path.name}（{total_seconds:.2f}s，"
                f"DiT峰值 {peak_gib:.1f} GiB，VAE峰值 {decode_peak_gib:.1f} GiB）"
            )
        else:
            print(
                f"    完成：job {index} 未打包 MP4（{total_seconds:.2f}s，"
                f"DiT峰值 {peak_gib:.1f} GiB，VAE峰值 {decode_peak_gib:.1f} GiB）"
            )
        return session

    def wait_for_next_audio() -> Path | None:
        nonlocal candidate_states
        idle_since = time.time()
        announced = False
        while True:
            if args.max_files is not None and len(processed_paths) >= args.max_files:
                return None
            now = time.time()
            current_files = set(_list_audio_files(audio_dir))
            for path in list(candidate_states):
                if path not in current_files:
                    candidate_states.pop(path, None)
            ready: list[Path] = []
            changed = False
            for path in sorted(current_files):
                if path in processed_paths:
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                key = (int(stat.st_size), int(stat.st_mtime_ns))
                previous = candidate_states.get(path)
                if previous is None or previous[0] != key:
                    candidate_states[path] = (key, now)
                    changed = True
                    continue
                if now - previous[1] >= args.stable_seconds:
                    ready.append(path)
            if changed:
                idle_since = now
            if ready:
                if args.watch_sort == "mtime":
                    ready.sort(key=lambda path: (path.stat().st_mtime, path.name))
                else:
                    ready.sort(key=lambda path: path.name)
                path = ready[0]
                candidate_states.pop(path, None)
                return path
            if not announced:
                state.update(
                    phase="waiting",
                    waiting_for_next=True,
                    generating_job_index=None,
                    message="等待下一段音频",
                )
                print(f"等待新音频：{audio_dir}（每 {args.poll_interval:.1f}s 扫描）")
                announced = True
            if args.idle_exit_seconds > 0 and now - idle_since >= args.idle_exit_seconds:
                print(f"已 {args.idle_exit_seconds:.0f}s 没有新音频，停止监听。")
                return None
            time.sleep(max(0.1, args.poll_interval))

    try:
        if args.watch:
            pending = list(files)
            index = 0
            while True:
                if args.max_files is not None and index >= args.max_files:
                    break
                if pending:
                    audio_path = pending.pop(0)
                else:
                    audio_path = wait_for_next_audio()
                    if audio_path is None:
                        break
                if audio_path in processed_paths:
                    continue
                # Allow one job to be generated/queued while the previous one is still playing.
                while len(pending_playback_events) >= 2:
                    pending_playback_events.pop(0).wait()
                index += 1
                processed_paths.add(audio_path)
                state.register_audio(index, audio_path)
                start_event = pending_playback_events[-1] if pending_playback_events else first_start_event
                session = process_job(
                    audio_path,
                    index,
                    job_total=0,
                    is_last=False,
                    start_event=start_event,
                    wait_for_playback=False,
                )
                pending_playback_events.append(session.playback_done_event)
                state.update(processed_files=index)
            for event in pending_playback_events:
                event.wait()
            state.update(
                phase="done",
                waiting_for_next=False,
                live_job_done=True,
                message="监听结束，全部音频处理完成",
            )
        else:
            for index, audio_path in enumerate(files, 1):
                process_job(
                    audio_path,
                    index,
                    job_total=len(files),
                    is_last=index == len(files),
                    start_event=first_start_event,
                    wait_for_playback=True,
                )
            state.update(phase="done", waiting_for_next=False, live_job_done=True, message="全部音频处理完成")

        write_summary()
        print(f"汇总写入：{summary_path}")
        print(f"前端仍可访问：{url}（{args.linger_seconds:.0f}s 后退出）")
        if args.linger_seconds > 0:
            deadline = time.time() + args.linger_seconds
            try:
                while time.time() < deadline:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("收到 Ctrl+C，退出。")
    finally:
        _release_pipeline(pipeline, device)
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    main()
