"""Folder-driven A2V inference with a cross-video autoregressive chunk history.

Differences from the other entry points:

* audio is read **only** from a folder (``--audio-dir``, default ``audio``); there is no
  interactive prompt, no ``--audio-path``, and the process exits when the folder is done;
* the AR chunk history is **not reset between audio files**:
    - the sink anchor is the **chunk 0 of the initial video** (video #1) and stays pinned for
      every later video (``--sink-first-video-chunk``, on by default);
    - the extra history is the **last N chunks of the previous video** (``--history-chunks``);
    - inside one video the usual AR chaining applies (chunk ``k`` also sees chunk ``k-1``).

Example:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \\
        inference_tp_folder_ar.py \\
        --distilled-checkpoint-path checkpoints/avatarforever-ltx-2.3-22b.safetensors \\
        --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \\
        --audio-dir audio --output-dir output/cross_video_ar \\
        --height 192 --width 352 --frame-rate 20 --fast-infer \\
        --prompt '...'
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import av
import torch
import torch.distributed as dist

AUDIO_EXTENSIONS = frozenset(
    {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aif", ".aiff"}
)


# --------------------------------------------------------------------------- helpers


def _audio_duration_seconds(audio_path: Path) -> float:
    with av.open(str(audio_path)) as container:
        durations = [
            float(stream.duration * stream.time_base)
            for stream in container.streams
            if stream.type == "audio" and stream.duration is not None and stream.time_base is not None
        ]
        if durations:
            duration = max(durations)
        elif container.duration is not None:
            duration = float(container.duration) / av.time_base
        else:
            duration = 0.0
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Could not determine a positive audio duration: {audio_path}")
    return duration


def _frames_for_duration(duration: float, frame_rate: float) -> int:
    raw_frames = max(1, math.ceil(duration * frame_rate))
    return 8 * math.ceil((raw_frames - 1) / 8) + 1


def _list_audio_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"音频目录不存在或不是目录：{directory}")
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS),
        key=lambda path: path.name,
    )


def _relax_required_args(parser: argparse.ArgumentParser) -> None:
    for action in parser._actions:
        if "--audio-path" in action.option_strings:
            action.required = False
            action.default = None
        elif "--num-frames" in action.option_strings:
            # Auto-infer the frame count from each audio file unless the user fixes it.
            action.default = None
            action.help = (
                "Fixed frame count for every audio file. If omitted, infer from each audio duration "
                "and round up to 8n+1."
            )


def _add_folder_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("audio"),
        help="Folder with the audio files to process (sorted by name). Default: audio",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Only process the first N audio files.")
    parser.add_argument(
        "--window-chunks",
        type=int,
        default=2,
        help=(
            "Rolling context window in chunks, counted including the chunk being generated (default 2 = "
            "current + 1 previous). The window slides over the global chunk sequence, so it spans the video "
            "boundary; only the initial video's chunk 0 stays pinned as sink."
        ),
    )
    parser.add_argument(
        "--trim-to-full-chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Expand every non-final video to the next full AR chunk boundary, so carried history chunks are "
            "never partial and the video is at least as long as the audio (default on). Set "
            "--no-trim-to-full-chunks to keep the natural 8n+1 length everywhere."
        ),
    )
    parser.add_argument(
        "--sink-first-video-chunk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pin chunk 0 of the initial video as the global sink anchor for all later videos (default on).",
    )
    parser.add_argument(
        "--carry-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Carry the rolling chunk window across videos (default on). With --no-carry-history every audio "
            "file is generated independently (no carried chunks, no global sink) while keeping the same "
            "in-video window, which is the clean baseline for A/B comparisons."
        ),
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Where to write the JSON summary (default: <output-dir>/cross_video_summary.json).",
    )


def _output_path(args: argparse.Namespace, *, index: int, audio_stem: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = args.output_dir / f"{audio_stem}_{index:04d}_{timestamp}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _job_log_path(output_path: Path) -> Path:
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{output_path.stem}.log"


def _attach_job_file_handler(log_path: Path) -> logging.FileHandler:
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def _detach_job_file_handler(handler: logging.FileHandler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.flush()
    handler.close()


def _release_pipeline(pipeline: Any, device: torch.device) -> None:
    if pipeline is not None:
        del pipeline
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# --------------------------------------------------------------------------- main


def main() -> None:  # noqa: PLR0915
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = local_rank == 0

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl", device_id=device)

    # Heavy project imports stay after the process group init (see inference_tp.py).
    from inference import DEFAULT_PROMPT, build_arg_parser
    from ltx_core.model.video_vae import get_video_chunks_number
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

    from cross_video_ar_pipeline import CrossVideoARPipeline

    parser = build_arg_parser()
    _relax_required_args(parser)
    _add_folder_arguments(parser)
    args = parser.parse_args()
    if args.window_chunks < 1:
        parser.error("--window-chunks must be >= 1")
    if args.window_chunks > max(1, args.ar_video_chunk_size) + 8:
        parser.error("--window-chunks is unreasonably large (memory grows with the window)")
    window_chunks = int(args.window_chunks)
    carried_chunks = window_chunks - 1

    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format=f"%(asctime)s | rank{local_rank} | %(levelname)s | %(name)s | %(message)s",
    )

    tp = TPContext(rank=local_rank, world=world_size, device=device, verbose=is_main)

    original_transformer = ModelLedger.transformer

    def tp_transformer(self, extra_config=None):  # noqa: ANN001, ANN202
        module = original_transformer(self, extra_config=extra_config)
        if not getattr(module, "_tp_sharded", False):
            shard_transformer_model_(module, tp)
            module._tp_sharded = True
        return module

    ModelLedger.transformer = tp_transformer

    pipeline = CrossVideoARPipeline(
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
    prompt = args.prompt or DEFAULT_PROMPT

    files: list[Path] = []
    if is_main:
        try:
            files = _list_audio_files(Path(args.audio_dir).expanduser())
        except Exception as exc:  # noqa: BLE001
            print(f"读取音频目录失败：{exc}")
        if args.max_files is not None:
            files = files[: args.max_files]
    file_count = len(files)
    if world_size > 1:
        counts = [file_count]
        dist.broadcast_object_list(counts, src=0)
        file_count = counts[0]

    if is_main:
        print(f"文件夹模式：{args.audio_dir} -> {len(files)} 个音频文件（按文件名排序）")
        for position, path in enumerate(files, 1):
            print(f"  {position:3d}. {path.name}")
        if not files:
            print("没有可处理的音频文件，退出。")
        print(
            f"跨视频 AR：sink={'-' if not args.sink_first_video_chunk else '初始视频 chunk 0'}，"
            f"滚动窗口={window_chunks} 个 chunk（含正在生成的 chunk），"
            f"即携带上一个视频尾部 {carried_chunks} 个 chunk；"
            f"非末个视频扩展到完整 chunk 边界={'是' if args.trim_to_full_chunks else '否'}"
            "（本入口的窗口由 --window-chunks 决定，--ar-history-chunk-count 不生效）"
        )

    summary: list[dict[str, Any]] = []
    sink_video: torch.Tensor | None = None
    sink_audio: torch.Tensor | None = None
    history_video: torch.Tensor | None = None
    history_audio: torch.Tensor | None = None
    history_video_ranges: list[tuple[int, int]] = []
    history_audio_ranges: list[tuple[int, int]] = []

    try:
        for index in range(1, file_count + 1):
            if is_main:
                audio_path = files[index - 1]
                duration = _audio_duration_seconds(audio_path)
                num_frames = args.num_frames or _frames_for_duration(duration, args.frame_rate)
                expanded = False
                if args.trim_to_full_chunks and index < file_count and args.num_frames is None:
                    # Extend the video to the next full chunk so the history carried into the next
                    # video is always made of full chunks.
                    latent_frames = (num_frames - 1) // 8 + 1
                    full_latent_frames = (
                        (latent_frames + args.ar_video_chunk_size - 1) // args.ar_video_chunk_size
                    ) * args.ar_video_chunk_size
                    if full_latent_frames > latent_frames:
                        num_frames = 8 * (full_latent_frames - 1) + 1
                        expanded = True
                job = {
                    "audio_path": str(audio_path),
                    "audio_stem": audio_path.stem,
                    "audio_duration": duration,
                    "num_frames": num_frames,
                    "expanded": expanded,
                    "padded_audio_path": str(
                        prepare_padded_audio(
                            audio_path,
                            video_duration=num_frames / args.frame_rate,
                            output_dir=args.output_dir,
                            index=index,
                        )
                    ),
                }
            else:
                job = None
            if world_size > 1:
                payload = [job]
                dist.broadcast_object_list(payload, src=0)
                job = payload[0]

            num_frames = int(job["num_frames"])
            latent_frames = (num_frames - 1) // 8 + 1
            chunk_ranges = build_progressive_chunk_ranges(latent_frames, args.ar_video_chunk_size)
            history_indices = list(
                range(max(0, len(chunk_ranges) - carried_chunks), len(chunk_ranges))
            )

            pipeline.set_cross_video_context(
                sink_video=sink_video if args.carry_history else None,
                sink_audio=sink_audio if args.carry_history else None,
                history_video=history_video if args.carry_history else None,
                history_audio=history_audio if args.carry_history else None,
                history_video_ranges=history_video_ranges if args.carry_history else [],
                history_audio_ranges=history_audio_ranges if args.carry_history else [],
                window_chunks=window_chunks,
            )
            # From video #2 on the sink role is taken over by the pinned global prefix chunk.
            sink_first_chunk = args.ar_sink_first_chunk and (
                not args.carry_history or index == 1 or not args.sink_first_video_chunk
            )
            effective_sink = sink_video if args.carry_history else None
            effective_history = history_video if args.carry_history else None

            output_path = _output_path(args, index=index, audio_stem=job["audio_stem"])
            log_path = _job_log_path(output_path)
            padded_audio_path = Path(job["padded_audio_path"])
            video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
            start_time = time.time()
            job_log_handler = _attach_job_file_handler(log_path) if is_main else None
            context = {
                "index": index,
                "audio": job["audio_path"],
                "num_frames": num_frames,
                "latent_frames": latent_frames,
                "chunks": len(chunk_ranges),
                "trimmed_to_full_chunks": False,
                "expanded_to_full_chunks": bool(job["expanded"]),
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
                "output": str(output_path),
            }

            try:
                if is_main:
                    logging.info("Video %d/%d started: %s", index, len(files), audio_path.name)
                    for key, value in context.items():
                        logging.info("  %s: %s", key, value)
                    print(
                        f"[{index}/{len(files)}] {audio_path.name}: {num_frames} 帧 / {len(chunk_ranges)} chunks；"
                        f"sink={context['sink_video_tokens']} tokens, "
                        f"携带历史={context['carried_chunks']} chunks/{context['carried_video_tokens']} tokens"
                        + ("（已扩展到完整 chunk 边界，音轨补静音）" if job["expanded"] else "")
                    )

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
                        fast_infer=args.fast_infer,
                    )
                    generation_end = time.time()
                    if is_main:
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
                        for _ in video:
                            pass
                        del video
            except Exception:
                if is_main:
                    logging.exception("Video %d failed", index)
                    if job_log_handler is not None:
                        _detach_job_file_handler(job_log_handler)
                        job_log_handler = None
                raise

            # Roll the context forward: the tail of this video becomes the next video's history.
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

            if world_size > 1:
                dist.barrier()

            if is_main:
                total_seconds = time.time() - start_time
                peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
                context.update(
                    {
                        "ar_seconds": round(generation_end - start_time, 3),
                        "total_seconds_incl_encode": round(total_seconds, 3),
                        "peak_gpu_memory_GiB": round(peak_gib, 2),
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
                logging.info("Video %d done in %.2fs (peak %.1f GiB)", index, total_seconds, peak_gib)
                if job_log_handler is not None:
                    _detach_job_file_handler(job_log_handler)
                    job_log_handler = None
                print(f"    完成：{output_path.name}（{total_seconds:.2f}s，峰值 {peak_gib:.1f} GiB）")
                stats = pipeline.last_window_stats or {}
                print(
                    "    窗口统计：每个 chunk 的窗口大小="
                    f"{stats.get('selected_chunk_count_per_chunk')}，其中来自上一个视频的 chunk 数="
                    f"{stats.get('carried_chunks_in_window_per_chunk')}"
                )
    finally:
        if is_main and summary:
            summary_path = args.summary_path or (args.output_dir / "cross_video_summary.json")
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"汇总写入：{summary_path}")
        _release_pipeline(pipeline, device)
        if is_main:
            print("模型已从显卡卸载，程序退出。")
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
