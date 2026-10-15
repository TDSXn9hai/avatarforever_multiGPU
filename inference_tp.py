"""Tensor-parallel (TP) entry point for Avatar-Forever A2V inference.

Mirrors ``inference.py`` but shards the diffusion transformer across ``WORLD_SIZE``
ranks (launched with ``torchrun``). Only rank 0 writes the output video; the other
ranks drain the generator so that every rank executes the same collectives.

Note: the project imports (``ltx_pipelines`` et al.) happen *inside* ``main`` after
``dist.init_process_group``. Importing them first makes NCCL init segfault on this
driver/library combination (reproduced with a minimal 2-rank all_reduce test).

Example:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 inference_tp.py \
        --distilled-checkpoint-path checkpoints/avatarforever-ltx-2.3-22b.safetensors \
        --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
        --audio-path eng_test_0.wav --output-path output/result_tp2.mp4
"""

from __future__ import annotations

import logging
import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = local_rank == 0

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl", device_id=device)

    # Heavy project imports: only safe once the process group exists (see module docstring).
    from inference import build_arg_parser
    from ltx_core.model.video_vae import get_video_chunks_number
    from ltx_pipelines import ARA2VidDistilledPipeline
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.model_ledger import ModelLedger
    from ltx_tp.tp_shim import TPContext, shard_transformer_model_
    from util import (
        build_first_frame_images,
        build_tiling_config,
        encode_first_frame_channel_condition,
        resolve_output_path,
    )

    args = build_arg_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format=f"%(asctime)s | rank{local_rank} | %(levelname)s | %(name)s | %(message)s",
    )

    tp = TPContext(rank=local_rank, world=world_size, device=device, verbose=is_main)
    if is_main:
        logging.info("Tensor parallel world size = %d (hidden state replicated, weights sharded)", world_size)

    # Shard the DiT as soon as the ledger builds it (weights are loaded single-GPU first).
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
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    output_path = resolve_output_path(args)

    start_time = time.time()
    with torch.inference_mode():
        video, audio = pipeline(
            prompt=args.prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            images=images,
            audio_path=str(args.audio_path),
            audio_start_time=0.0,
            audio_max_duration=args.num_frames / args.frame_rate,
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
            # Drain the generator so both ranks stay in lockstep on the collectives.
            for _ in video:
                pass
            del video

    if world_size > 1:
        dist.barrier()

    if is_main:
        encode_end = time.time()
        peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
        logging.info("AR generation finished in %.2fs", generation_end - start_time)
        logging.info("Total (incl. encode) %.2fs, peak GPU memory %.1f GiB", encode_end - start_time, peak_gib)
        logging.info("Saved video to %s", output_path)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
