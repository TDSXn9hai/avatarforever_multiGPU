from __future__ import annotations

import argparse
import contextlib
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import load_image_conditioning

if TYPE_CHECKING:
    from ltx_core.model.video_vae import VideoEncoder
    from ltx_pipelines import ARA2VidDistilledPipeline


def optional_int(value: str) -> int | None:
    if value.lower() in {"none", "null", ""}:
        return None
    return int(value)


def valid_num_frames(value: str) -> int:
    frames = int(value)
    if frames <= 0 or (frames - 1) % 8 != 0:
        raise argparse.ArgumentTypeError(
            f"num_frames must be positive and follow 8n+1 (for example 161, 2001, or 8001), got {frames}."
        )
    return frames


def build_loras(lora_path: Path | None, strength: float) -> list[LoraPathStrengthAndSDOps]:
    if lora_path is None:
        return []
    return [
        LoraPathStrengthAndSDOps(
            str(lora_path),
            strength,
            LTXV_LORA_COMFY_RENAMING_MAP,
        )
    ]


def build_first_frame_images(
    image_path: Path | None,
    *,
    strength: float,
    crf: int,
) -> list[ImageConditioningInput]:
    if image_path is None:
        return []
    return [
        ImageConditioningInput(
            path=str(image_path),
            frame_idx=0,
            strength=strength,
            crf=crf,
        )
    ]


def _video_encoder_for_pipeline(pipeline: ARA2VidDistilledPipeline) -> VideoEncoder:
    if hasattr(pipeline, "model_ledger"):
        return pipeline.model_ledger.video_encoder()
    return pipeline.stage_1_model_ledger.video_encoder()


def _clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def encode_first_frame_channel_condition(
    pipeline: ARA2VidDistilledPipeline,
    *,
    image_path: Path | None,
    enabled: bool,
    height: int,
    width: int,
    crf: int,
) -> torch.Tensor | None:
    if not enabled or image_path is None:
        return None

    video_encoder = _video_encoder_for_pipeline(pipeline)
    image = load_image_conditioning(
        image_path=str(image_path),
        height=height,
        width=width,
        dtype=pipeline.dtype,
        device=pipeline.device,
        crf=crf,
    )
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=pipeline.dtype)
        if pipeline.device.type == "cuda"
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        latent = video_encoder(image)

    latent = latent[:1].detach().contiguous()
    del image, video_encoder
    _clear_cuda_cache()
    return latent


def build_tiling_config(args: argparse.Namespace) -> TilingConfig:
    return TilingConfig(
        spatial_config=SpatialTilingConfig(
            tile_size_in_pixels=args.spatial_tile_size,
            tile_overlap_in_pixels=args.spatial_tile_overlap,
        ),
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=args.temporal_tile_size,
            tile_overlap_in_frames=args.temporal_tile_overlap,
        ),
    )


def prepare_padded_audio(
    audio_path: Path,
    *,
    video_duration: float,
    output_dir: Path,
    index: int,
) -> Path:
    """Write a WAV padded/truncated to exactly ``video_duration`` seconds.

    The video frame count is rounded up to ``8n+1`` (and may be expanded to a full AR
    chunk boundary), so the source audio is often shorter than the video.  Using this
    file as the pipeline input keeps the generated audio track and the MP4 duration in
    sync; the frontend also serves it through ``/audio/<index>``.
    """
    import wave

    import av
    import numpy as np

    chunks: list[np.ndarray] = []
    with av.open(str(audio_path)) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        sample_rate = int(stream.rate)
        channels = int(stream.channels or 1)
        for frame in container.decode(stream):
            array = frame.to_ndarray()
            if array.dtype == np.int16:
                array = array.astype(np.float32) / 32768.0
            elif array.dtype == np.int32:
                array = array.astype(np.float32) / 2147483648.0
            elif array.dtype == np.uint8:
                array = (array.astype(np.float32) - 128.0) / 128.0
            else:
                array = array.astype(np.float32)
            if array.ndim == 1:
                array = array[None, :]
            chunks.append(array)

    if not chunks:
        raise ValueError(f"No audio samples decoded from {audio_path}")
    waveform = np.concatenate(chunks, axis=-1)
    if waveform.shape[0] == 1 and channels > 1:
        waveform = np.repeat(waveform, channels, axis=0)
    channels = int(waveform.shape[0])
    target_samples = max(1, round(float(video_duration) * sample_rate))
    current_samples = int(waveform.shape[-1])
    if current_samples < target_samples:
        waveform = np.pad(
            waveform,
            ((0, 0), (0, target_samples - current_samples)),
            mode="constant",
        )
    elif current_samples > target_samples:
        waveform = waveform[..., :target_samples]

    out_dir = Path(output_dir) / "audio_padded"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{Path(audio_path).stem}_{int(index):04d}_{timestamp}.wav"
    pcm16 = np.clip(waveform.T, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    with wave.open(str(out_path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16.tobytes())
    return out_path


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output_path is not None:
        output_path = args.output_path
    else:
        history_count = "none" if args.ar_history_chunk_count is None else str(args.ar_history_chunk_count)
        output_path = args.output_dir / (
            f"f{args.num_frames}_chunk{args.ar_video_chunk_size}_hist{history_count}.mp4"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path
