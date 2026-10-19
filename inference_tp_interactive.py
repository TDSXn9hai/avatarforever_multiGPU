"""Interactive, persistent-model TP inference for Avatar-Forever.

The model is loaded once and kept on GPU while the process accepts multiple
audio/prompt jobs from rank 0. Job metadata is broadcast to the other ranks so
all ranks run the same collectives. Type one of the exit commands at any
interactive prompt to release the model and terminate.

Example:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
        inference_tp_interactive.py \
        --distilled-checkpoint-path checkpoints/avatarforever-ltx-2.3-22b.safetensors \
        --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized

Startup arguments retain the normal inference settings. ``--audio-path`` and
``--prompt`` are entered interactively after model loading. ``--num-frames``
is optional; when omitted, it is inferred from each audio file duration.

Batch mode:
    Add ``--audio-dir /audio`` (plus optionally ``--prompt "<text>"``) to process every
    audio file in that folder, sorted by name, one after another without asking between
    jobs; the process exits when the folder is exhausted. Output files are prefixed with
    the source audio stem (``<stem>_<index>_<timestamp>.mp4``) so each video maps back to
    its segment. When ``--prompt`` is omitted the prompt is asked once before the batch
    starts and reused for every file; ``--num-frames``, when given, is applied to every
    file, otherwise the frame count is inferred per file.
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import sys
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import av
import torch
import torch.distributed as dist


EXIT_COMMANDS = frozenset({":q", ":quit", "/exit", "/quit", "exit", "quit"})

# Extensions accepted by --audio-dir; anything else in the folder is ignored.
AUDIO_EXTENSIONS = frozenset(
    {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aif", ".aiff"}
)


def _set_optional_interactive_arguments(parser: argparse.ArgumentParser) -> None:
    for action in parser._actions:
        if "--audio-path" in action.option_strings:
            action.required = False
            action.default = None
        elif "--prompt" in action.option_strings:
            action.default = None
        elif "--num-frames" in action.option_strings:
            action.default = None
            action.help = (
                "Fixed output frame count for every job. If omitted, infer from each audio duration "
                "and round up to 8n+1."
            )


def _add_batch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help=(
            "Batch mode: process every audio file in this folder (sorted by name) one after another, "
            "then exit. The prompt is taken from --prompt, or asked once at startup when --prompt is "
            "omitted. Without this option the entry point keeps its interactive behaviour."
        ),
    )


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


def _is_exit_command(value: str) -> bool:
    return value.strip().lower() in EXIT_COMMANDS


def _read_multiline_prompt(default_prompt: str) -> str | None:
    print(
        "请输入提示词。支持多行和空行；输入单独一行 :done 结束，"
        "直接输入 :q、exit 或 quit 退出。"
    )
    lines: list[str] = []
    while True:
        try:
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            print()
            return None

        if line == "":
            print()
            return None

        line = line.rstrip("\r\n")
        if _is_exit_command(line):
            return None
        if line.strip().lower() in {":done", ":end"}:
            break
        lines.append(line)

    prompt = "\n".join(lines).strip()
    return prompt or default_prompt


def _make_job(
    audio_path: Path,
    prompt: str,
    frame_rate: float,
    fixed_num_frames: int | None,
    *,
    duration: float | None = None,
) -> dict[str, Any]:
    """Build the job payload that is broadcast to every rank."""
    duration = _audio_duration_seconds(audio_path) if duration is None else duration
    num_frames = fixed_num_frames or _frames_for_duration(duration, frame_rate)
    return {
        "exit": False,
        "audio_path": str(audio_path),
        "audio_stem": audio_path.stem,
        "prompt": prompt,
        "num_frames": num_frames,
        "audio_duration": duration,
    }


def _list_audio_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"音频目录不存在或不是目录：{directory}")
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS),
        key=lambda path: path.name,
    )


def _read_batch_jobs(
    audio_dir: Path,
    *,
    default_prompt: str,
    prompt: str | None,
    frame_rate: float,
    fixed_num_frames: int | None,
) -> Iterator[dict[str, Any]]:
    """Yield one job per audio file in ``audio_dir`` (rank 0 only), then an exit job."""
    try:
        files = _list_audio_files(audio_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"读取音频目录失败：{exc}")
        yield {"exit": True}
        return

    if not files:
        print(f"目录中没有可用音频文件（支持 {', '.join(sorted(AUDIO_EXTENSIONS))}）：{audio_dir}")
        yield {"exit": True}
        return

    if prompt is None:
        print(f"批量模式：将处理 {audio_dir} 下的 {len(files)} 个音频文件：")
        for index, path in enumerate(files, 1):
            print(f"  {index:3d}. {path.name}")
        prompt = _read_multiline_prompt(default_prompt)
        if prompt is None:
            yield {"exit": True}
            return

    print(f"批量模式：共 {len(files)} 个音频文件，提示词已固定，逐个生成。")
    for index, path in enumerate(files, 1):
        try:
            job = _make_job(path, prompt, frame_rate, fixed_num_frames)
        except Exception as exc:  # noqa: BLE001
            print(f"[batch] {index}/{len(files)} 读取失败，跳过：{path.name}（{exc}）")
            continue
        print(
            f"[batch] {index}/{len(files)}：{path.name}，时长 {job['audio_duration']:.3f}s，"
            f"{job['num_frames']} 帧"
        )
        yield job

    print(f"批量模式：{len(files)} 个音频文件已全部处理。")
    yield {"exit": True}


def _read_job_from_rank0(
    default_prompt: str,
    frame_rate: float,
    fixed_num_frames: int | None,
) -> dict[str, Any]:
    while True:
        try:
            audio_input = input(
                "\n模型已加载并常驻显卡。\n"
                "请输入音频路径（输入 :q、exit 或 quit 退出）："
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return {"exit": True}

        if _is_exit_command(audio_input):
            return {"exit": True}
        if not audio_input:
            print("音频路径不能为空，请重新输入。")
            continue

        audio_path = Path(audio_input).expanduser()
        if not audio_path.is_file():
            print(f"音频文件不存在或不可读：{audio_path}")
            continue

        try:
            duration = _audio_duration_seconds(audio_path)
        except Exception as exc:  # noqa: BLE001
            print(f"读取音频时长失败：{exc}")
            continue

        prompt = _read_multiline_prompt(default_prompt)
        if prompt is None:
            return {"exit": True}

        job = _make_job(audio_path, prompt, frame_rate, fixed_num_frames, duration=duration)
        print(
            f"已读取音频：{audio_path}，时长 {duration:.3f}s；"
            f"本次生成 {job['num_frames']} 帧，视频时长约 {job['num_frames'] / frame_rate:.3f}s。"
        )
        return job


def _broadcast_job(job: dict[str, Any], world_size: int) -> dict[str, Any]:
    if world_size == 1:
        return job
    payload = [job]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _output_path_for_job(
    args: argparse.Namespace,
    *,
    job_index: int,
    audio_stem: str | None = None,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Batch runs carry the source audio stem so each video stays traceable to its segment.
    label = f"{audio_stem}_{job_index:04d}" if audio_stem else f"interactive_{job_index:04d}"
    if args.output_path is None:
        path = args.output_dir / f"{label}_{timestamp}.mp4"
    else:
        configured = Path(args.output_path)
        if "{index" in str(configured) or "{timestamp" in str(configured):
            path = Path(
                str(configured).format(
                    index=job_index,
                    timestamp=timestamp,
                    stem=audio_stem or "interactive",
                )
            )
        else:
            path = configured.with_name(
                f"{configured.stem}_{label}_{timestamp}{configured.suffix or '.mp4'}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _job_log_path(output_path: Path) -> Path:
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{output_path.stem}.log"


def _attach_job_file_handler(log_path: Path) -> logging.FileHandler:
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logging.getLogger().addHandler(handler)
    return handler


def _detach_job_file_handler(handler: logging.FileHandler) -> None:
    root_logger = logging.getLogger()
    root_logger.removeHandler(handler)
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


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = local_rank == 0

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl", device_id=device)

    # Keep project imports after process-group initialization; see inference_tp.py.
    from inference import DEFAULT_PROMPT, build_arg_parser
    from ltx_core.model.video_vae import get_video_chunks_number
    from ltx_pipelines import ARA2VidDistilledPipeline
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.model_ledger import ModelLedger
    from ltx_tp.tp_shim import TPContext, shard_transformer_model_
    from util import (
        build_first_frame_images,
        build_tiling_config,
        encode_first_frame_channel_condition,
    )

    parser = build_arg_parser()
    _set_optional_interactive_arguments(parser)
    _add_batch_arguments(parser)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format=f"%(asctime)s | rank{local_rank} | %(levelname)s | %(name)s | %(message)s",
    )

    tp = TPContext(rank=local_rank, world=world_size, device=device, verbose=is_main)
    if is_main:
        logging.info(
            "Starting persistent interactive inference: TP=%d; audio and prompt are entered per job",
            world_size,
        )

    original_transformer = ModelLedger.transformer

    def tp_transformer(self, extra_config=None):  # noqa: ANN001, ANN202
        module = original_transformer(self, extra_config=extra_config)
        if not getattr(module, "_tp_sharded", False):
            shard_transformer_model_(module, tp)
            module._tp_sharded = True
        return module

    ModelLedger.transformer = tp_transformer

    pipeline = ARA2VidDistilledPipeline(
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

    if is_main:
        print("模型加载完成，权重将常驻显卡。")
        print("退出命令：:q、:quit、/exit、/quit、exit、quit。")

    job_index = 0
    batch_jobs: Iterator[dict[str, Any]] | None = None
    if is_main and args.audio_dir is not None:
        batch_jobs = _read_batch_jobs(
            Path(args.audio_dir).expanduser(),
            default_prompt=DEFAULT_PROMPT,
            prompt=args.prompt,
            frame_rate=args.frame_rate,
            fixed_num_frames=args.num_frames,
        )

    try:
        while True:
            if is_main:
                if batch_jobs is not None:
                    job = next(batch_jobs, {"exit": True})
                else:
                    job = _read_job_from_rank0(
                        DEFAULT_PROMPT,
                        args.frame_rate,
                        args.num_frames,
                    )
            else:
                job = None
            job = _broadcast_job(job, world_size)
            if job["exit"]:
                if is_main:
                    if batch_jobs is not None:
                        print("批量任务处理完毕，正在卸载模型并释放显存。")
                    else:
                        print("收到退出指令，正在卸载模型并释放显存。")
                break

            job_index += 1
            num_frames = int(job["num_frames"])
            args.num_frames = num_frames
            video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
            output_path = _output_path_for_job(
                args,
                job_index=job_index,
                # Only batch runs rename outputs after the source audio; the interactive
                # mode keeps its documented `interactive_<index>_<timestamp>.mp4` scheme.
                audio_stem=job.get("audio_stem") if batch_jobs is not None else None,
            )
            log_path = _job_log_path(output_path)
            start_time = time.time()
            job_log_handler = _attach_job_file_handler(log_path) if is_main else None

            try:
                if is_main:
                    logging.info("Job %d started", job_index)
                    logging.info("Audio path: %s", job["audio_path"])
                    logging.info("Audio duration: %.3fs", job["audio_duration"])
                    logging.info("Num frames: %d", num_frames)
                    logging.info("Frame rate: %.3f", args.frame_rate)
                    logging.info("Prompt:\n%s", job["prompt"])
                    logging.info("Output path: %s", output_path)
                    logging.info("Job log path: %s", log_path)

                with torch.inference_mode():
                    video, audio = pipeline(
                        prompt=job["prompt"],
                        seed=args.seed,
                        height=args.height,
                        width=args.width,
                        num_frames=num_frames,
                        frame_rate=args.frame_rate,
                        images=images,
                        audio_path=job["audio_path"],
                        audio_start_time=0.0,
                        audio_max_duration=num_frames / args.frame_rate,
                        tiling_config=tiling_config,
                        enhance_prompt=False,
                        stage_mode="one-stage",
                        stage1_sigmas=args.stage1_sigmas,
                        ar_video_chunk_size=args.ar_video_chunk_size,
                        ar_history_chunk_count=args.ar_history_chunk_count,
                        ar_sink_first_chunk=args.ar_sink_first_chunk,
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
                    logging.exception("Job %d failed", job_index)
                    if job_log_handler is not None:
                        _detach_job_file_handler(job_log_handler)
                        job_log_handler = None
                raise

            if world_size > 1:
                dist.barrier()

            if is_main:
                total_seconds = time.time() - start_time
                peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
                logging.info("Job %d AR generation finished in %.2fs", job_index, generation_end - start_time)
                logging.info(
                    "Job %d total including encode %.2fs, peak GPU memory %.1f GiB",
                    job_index,
                    total_seconds,
                    peak_gib,
                )
                logging.info("Job %d log saved to %s", job_index, log_path)
                if job_log_handler is not None:
                    _detach_job_file_handler(job_log_handler)
                    job_log_handler = None
                print(f"生成完成，视频路径：{output_path}")
                print(f"本轮日志路径：{log_path}")
                print(f"本次耗时：{total_seconds:.2f}s；模型仍常驻显卡。")

            continue_job = True
            if is_main and batch_jobs is None:
                try:
                    answer = input("再来一个？输入 y 继续，输入 n 或退出命令结束：").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    answer = "n"
                continue_job = answer in {"y", "yes", "继续", "是"}
            if world_size > 1:
                decision = [continue_job]
                dist.broadcast_object_list(decision, src=0)
                continue_job = decision[0]
            if not continue_job:
                break
    finally:
        _release_pipeline(pipeline, device)
        if is_main:
            print("模型已从显卡卸载，程序退出。")
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
