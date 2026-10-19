"""Cross-video autoregressive context for the Avatar-Forever A2V distilled pipeline.

The stock AR pipeline keeps the chunk history inside one video: chunk *k* is denoised with
``{sink, k-history..k-1, k}`` of the same video, and everything is reset for the next call.

This module makes the chunk history **global across videos**:

* ``sink``  – chunk 0 of the **initial video**, always visible as a pinned prefix;
* ``carried`` – the trailing chunks of the previous video, injected into the *chunk index
  space* so that the normal ``history_chunk_count`` window keeps rolling over them.

Resulting visible set for chunk *j* of any later video::

    [sink] + {the newest (window_chunks - 1) chunks} + [chunk j]

where "the newest chunks" are taken from the global chunk sequence (previous video's tail
first, then the current video's own chunks), so a chunk drops out of the window exactly when
it stops being recent — i.e. a real rolling window that spans the video boundary.

Implementation notes
--------------------
* The sink is prepended as an ordinary "global prefix condition" (``conditioning_prefix_token_count``
  with ``denoise_mask = 0``), which upstream already supports and strips again before decoding.
* The carried chunks are *not* part of that always-visible prefix: they are prepended to the
  chunk token ranges, and ``select_inference_chunk_indices`` then decides per chunk whether they
  are still inside the window. This is why ``_cross_video_ar_loop`` below is a port of
  ``ltx_pipelines.utils.autoregressive.autoregressive_euler_denoising_loop`` — the only change is
  the extended chunk index space (carried chunks + current video chunks) and the explicit target
  token offsets. Everything else (window selection, masking, channel-condition alignment, RoPE
  handling, write-back) reuses the upstream helpers unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import torch
from tqdm import tqdm

from ltx_core.components.patchifiers import AudioLatentShape, VideoLatentShape
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import LatentState
from ltx_pipelines import ARA2VidDistilledPipeline
from ltx_pipelines.utils.autoregressive import (
    ARFeatureCache,
    _first_frame_audio_token_count,
    _first_latent_frame_prefix,
    _repeated_first_latent_frame_channel_condition,
    _select_ar_chunk_state,
    build_aligned_audio_chunk_ranges,
    build_progressive_chunk_ranges,
    select_inference_chunk_indices,
)
from ltx_pipelines.utils.helpers import (
    apply_video_channel_condition_to_state,
    noise_audio_state,
    noise_video_state,
    post_process_latent,
)

logger = logging.getLogger(__name__)


def _canonical_video_positions(
    *,
    components,
    output_shape,
    total_latent_frames: int,
    device,
    dtype,
) -> torch.Tensor:
    """Position grid for a video of ``total_latent_frames`` frames, built by the upstream tools.

    ``select_inference_chunk_state`` + ``_select_ar_chunk_state`` implement relative positions by
    taking ``positions[:, :, :window_tokens]``, i.e. the *head* of the grid. The head is only a
    valid sliding window when it is the canonical, frame-increasing sequence of a single video —
    which is why the sink/history blocks cannot simply reuse (or duplicate) the target grid. The
    spliced state is therefore laid out like one longer video: ``[sink | carried | target]`` frames
    numbered 0..N-1 in row order.
    """
    from ltx_core.types import VideoPixelShape

    pixel_frames = 8 * (total_latent_frames - 1) + 1
    extended_pixel_shape = VideoPixelShape(
        batch=output_shape.batch,
        frames=pixel_frames,
        width=output_shape.width,
        height=output_shape.height,
        fps=output_shape.fps,
    )
    latent_shape = VideoLatentShape.from_pixel_shape(
        shape=extended_pixel_shape,
        latent_channels=components.video_latent_channels,
        scale_factors=components.video_scale_factors,
    )
    tools = VideoLatentTools(components.video_patchifier, latent_shape, extended_pixel_shape.fps)
    positions = tools.create_initial_state(device=device, dtype=dtype).positions
    if positions.shape[2] != total_latent_frames * (positions.shape[2] // total_latent_frames):
        raise ValueError("unexpected position grid size")
    return positions


def _canonical_audio_positions(
    *,
    components,
    output_shape,
    total_audio_tokens: int,
    device,
    dtype,
) -> torch.Tensor:
    """Audio counterpart of :func:`_canonical_video_positions` (1-D temporal grid)."""
    base_duration = output_shape.frames / output_shape.fps
    base_shape = AudioLatentShape.from_duration(
        batch=output_shape.batch, duration=base_duration, channels=8, mel_bins=16
    )
    base_tokens = max(
        1,
        AudioLatentTools(components.audio_patchifier, base_shape)
        .create_initial_state(device=device, dtype=dtype)
        .latent.shape[1],
    )
    # Keep the same token rate as the current job, then pad the duration a little for safety.
    extended_duration = base_duration * (total_audio_tokens / base_tokens) * 1.1
    extended_shape = AudioLatentShape.from_duration(
        batch=output_shape.batch, duration=extended_duration, channels=8, mel_bins=16
    )
    tools = AudioLatentTools(components.audio_patchifier, extended_shape)
    positions = tools.create_initial_state(device=device, dtype=dtype).positions
    if positions.shape[2] < total_audio_tokens:
        raise ValueError(
            f"audio position grid too short: {positions.shape[2]} < {total_audio_tokens}"
        )
    return positions[:, :, :total_audio_tokens]


def _prepend_state_tokens(state: LatentState, tokens: torch.Tensor | None, *, label: str) -> LatentState:
    """Prepend clean context tokens and mark them as an always-visible prefix.

    The position grid is left untouched here: the caller rebuilds it once after every splice so the
    head of the grid stays the canonical frame-increasing sequence of the spliced state.
    """
    if tokens is None or tokens.shape[1] == 0:
        return state
    if state.attention_mask is not None:
        raise NotImplementedError("cross-video context requires attention_mask=None (full attention)")

    batch, token_count, channels = tokens.shape
    if batch != state.latent.shape[0] or channels != state.latent.shape[-1]:
        raise ValueError(
            f"{label} prefix token shape {tuple(tokens.shape)} is incompatible with the latent state "
            f"{tuple(state.latent.shape)}."
        )

    tokens = tokens.to(device=state.latent.device, dtype=state.latent.dtype)
    mask_shape = (batch, token_count, *state.denoise_mask.shape[2:])
    return LatentState(
        latent=torch.cat([tokens, state.latent], dim=1),
        denoise_mask=torch.cat([state.denoise_mask.new_zeros(mask_shape), state.denoise_mask], dim=1),
        positions=state.positions,
        clean_latent=torch.cat([tokens, state.clean_latent], dim=1),
        attention_mask=None,
        conditioning_prefix_token_count=state.conditioning_prefix_token_count + token_count,
        channel_condition=state.channel_condition,
    )


def _cross_video_ar_loop(  # noqa: PLR0913
    *,
    sigmas: torch.Tensor,
    video_state: LatentState,
    audio_state: LatentState,
    stepper,
    denoise_fn_builder,
    total_video_latent_frames: int,
    video_chunk_size: int,
    video_target_token_count: int,
    audio_target_token_count: int,
    carried_video_ranges: list[tuple[int, int]],
    carried_audio_ranges: list[tuple[int, int]],
    window_chunks: int,
    sink_first_chunk: bool,
    relative_positions: bool,
    history_feature_cache: bool,
    derive_channel_condition_from_first_chunk: bool,
    channel_condition_current_chunk_only: bool,
    first_frame_prefix_condition: bool,
    first_frame_condition_position: str,
    rope_max_temporal_index: int | None,
    on_chunk_ready=None,
) -> tuple[LatentState, LatentState, dict]:
    """``autoregressive_euler_denoising_loop`` with the chunk index space extended by carried chunks."""
    if first_frame_condition_position not in ("prepend", "append"):
        raise ValueError(
            f"first_frame_condition_position must be 'prepend' or 'append', got {first_frame_condition_position!r}"
        )
    if total_video_latent_frames <= 0:
        raise ValueError(f"total_video_latent_frames must be positive, got {total_video_latent_frames}")
    if video_chunk_size <= 0:
        raise ValueError(f"video_chunk_size must be positive, got {video_chunk_size}")
    if video_target_token_count <= 0 or video_target_token_count % total_video_latent_frames != 0:
        raise ValueError(
            "Video target token count must be a positive multiple of the latent-frame count. "
            f"target_tokens={video_target_token_count}, latent_frames={total_video_latent_frames}"
        )

    video_tokens_per_frame = video_target_token_count // total_video_latent_frames
    video_chunk_ranges = build_progressive_chunk_ranges(total_video_latent_frames, video_chunk_size)
    audio_chunk_ranges = build_aligned_audio_chunk_ranges(
        video_chunk_ranges,
        total_video_frames=total_video_latent_frames,
        total_audio_steps=audio_target_token_count,
    )

    video_prefix_token_count = int(video_state.conditioning_prefix_token_count)
    audio_prefix_token_count = int(audio_state.conditioning_prefix_token_count)
    video_target_start = video_prefix_token_count + sum(end - start for start, end in carried_video_ranges)
    audio_target_start = audio_prefix_token_count + sum(end - start for start, end in carried_audio_ranges)

    # ``carried_*_ranges`` are relative to the carried tensors; make them absolute in the state
    # (the sink/prefix sits in front of the carried chunks).
    carried_video_abs = [
        (video_prefix_token_count + start, video_prefix_token_count + end) for start, end in carried_video_ranges
    ]
    carried_audio_abs = [
        (audio_prefix_token_count + start, audio_prefix_token_count + end) for start, end in carried_audio_ranges
    ]

    all_video_ranges = list(carried_video_abs) + [
        (
            video_target_start + frame_start * video_tokens_per_frame,
            video_target_start + frame_end * video_tokens_per_frame,
        )
        for frame_start, frame_end in video_chunk_ranges
    ]
    all_audio_ranges = list(carried_audio_abs) + [
        (audio_target_start + start, audio_target_start + end) for start, end in audio_chunk_ranges
    ]
    chunk_index_offset = len(carried_video_ranges)
    num_chunks = len(all_video_ranges)
    if len(all_audio_ranges) != num_chunks:
        raise ValueError(
            f"carried video/audio chunk counts must match: {len(carried_video_ranges)} vs {len(carried_audio_ranges)}"
        )

    total_steps = len(video_chunk_ranges) * max(0, len(sigmas) - 1)
    progress = tqdm(total=total_steps, desc="Generating", unit="step")
    carried_per_chunk: list[int] = []
    window_sizes: list[int] = []
    try:
        first_frame_prefix_latent: torch.Tensor | None = None
        first_frame_prefix_positions: torch.Tensor | None = None
        first_frame_audio_prefix_latent: torch.Tensor | None = None
        first_frame_audio_prefix_positions: torch.Tensor | None = None
        first_frame_audio_prefix_token_count = 0
        if first_frame_prefix_condition:
            first_frame_audio_prefix_token_count = _first_frame_audio_token_count(
                total_video_latent_frames=total_video_latent_frames,
                total_audio_steps=audio_target_token_count,
            )
            first_frame_prefix_latent, first_frame_prefix_positions = _first_latent_frame_prefix(
                state=video_state,
                prefix_token_count=video_target_start,
                token_count=video_tokens_per_frame,
            )
            first_frame_audio_prefix_latent, first_frame_audio_prefix_positions = _first_latent_frame_prefix(
                state=audio_state,
                prefix_token_count=audio_target_start,
                token_count=first_frame_audio_prefix_token_count,
            )

        for chunk_idx, ((frame_start, frame_end), (audio_start, audio_end)) in enumerate(
            zip(video_chunk_ranges, audio_chunk_ranges, strict=True)
        ):
            if audio_end <= audio_start:
                raise ValueError(
                    "AR audio chunk mapping produced an empty chunk. "
                    "Reduce ar_video_chunk_size or disable audio generation for this configuration."
                )

            current_index = chunk_index_offset + chunk_idx
            selected_chunk_indices = select_inference_chunk_indices(
                current_chunk_idx=current_index,
                num_chunks=num_chunks,
                history_chunk_count=max(0, window_chunks - 1),
                sink_first_chunk=sink_first_chunk,
            )
            carried_per_chunk.append(sum(1 for idx in selected_chunk_indices if idx < chunk_index_offset))
            window_sizes.append(len(selected_chunk_indices))
            use_first_frame_prefix = (
                first_frame_prefix_condition
                and first_frame_prefix_latent is not None
                and first_frame_audio_prefix_latent is not None
                and chunk_idx > 0
                and 0 not in selected_chunk_indices
            )
            chunk_denoise_fn = denoise_fn_builder()
            chunk_feature_cache = ARFeatureCache() if history_feature_cache else None

            for step_idx, _sigma in enumerate(sigmas[:-1]):
                current_video_state, current_video_slice = _select_ar_chunk_state(
                    video_state,
                    chunk_token_ranges=all_video_ranges,
                    selected_chunk_indices=selected_chunk_indices,
                    current_chunk_idx=current_index,
                    prefix_token_count=video_prefix_token_count,
                    relative_positions=relative_positions,
                    channel_condition_current_chunk_only=channel_condition_current_chunk_only,
                    video_tokens_per_frame=video_tokens_per_frame,
                    current_temporal_index=frame_end - 1,
                    rope_max_temporal_index=rope_max_temporal_index,
                    dynamic_prefix_latent=first_frame_prefix_latent if use_first_frame_prefix else None,
                    dynamic_prefix_positions=first_frame_prefix_positions if use_first_frame_prefix else None,
                    dynamic_condition_position=first_frame_condition_position,
                )
                current_audio_state, current_audio_slice = _select_ar_chunk_state(
                    audio_state,
                    chunk_token_ranges=all_audio_ranges,
                    selected_chunk_indices=selected_chunk_indices,
                    current_chunk_idx=current_index,
                    prefix_token_count=audio_prefix_token_count,
                    relative_positions=relative_positions,
                    dynamic_prefix_latent=(
                        first_frame_audio_prefix_latent if use_first_frame_prefix else None
                    ),
                    dynamic_prefix_positions=(
                        first_frame_audio_prefix_positions if use_first_frame_prefix else None
                    ),
                    dynamic_condition_position=first_frame_condition_position,
                )

                if chunk_feature_cache is None:
                    denoised_video, denoised_audio = chunk_denoise_fn(
                        current_video_state,
                        current_audio_state,
                        sigmas,
                        step_idx,
                    )
                else:
                    denoised_video, denoised_audio = chunk_denoise_fn(
                        current_video_state,
                        current_audio_state,
                        sigmas,
                        step_idx,
                        ar_feature_cache=chunk_feature_cache,
                        video_current_slice=current_video_slice,
                        audio_current_slice=current_audio_slice,
                    )
                denoised_video = post_process_latent(
                    denoised_video, current_video_state.denoise_mask, current_video_state.clean_latent
                )
                denoised_audio = post_process_latent(
                    denoised_audio, current_audio_state.denoise_mask, current_audio_state.clean_latent
                )

                video_step = stepper.step(current_video_state.latent, denoised_video, sigmas, step_idx)
                video_step = post_process_latent(
                    video_step, current_video_state.denoise_mask, current_video_state.clean_latent
                )
                video_start, video_end = all_video_ranges[current_index]
                new_video_latent = video_state.latent.clone()
                new_video_latent[:, video_start:video_end] = video_step[:, current_video_slice]
                video_state = replace(video_state, latent=new_video_latent)

                audio_step = stepper.step(current_audio_state.latent, denoised_audio, sigmas, step_idx)
                audio_step = post_process_latent(
                    audio_step, current_audio_state.denoise_mask, current_audio_state.clean_latent
                )
                audio_start_abs, audio_end_abs = all_audio_ranges[current_index]
                new_audio_latent = audio_state.latent.clone()
                new_audio_latent[:, audio_start_abs:audio_end_abs] = audio_step[:, current_audio_slice]
                audio_state = replace(audio_state, latent=new_audio_latent)

                progress.update(1)

            if (
                derive_channel_condition_from_first_chunk
                and chunk_idx == 0
                and video_state.channel_condition is None
            ):
                video_state = replace(
                    video_state,
                    channel_condition=_repeated_first_latent_frame_channel_condition(
                        state=video_state,
                        prefix_token_count=video_target_start,
                        total_video_latent_frames=total_video_latent_frames,
                        video_tokens_per_frame=video_tokens_per_frame,
                    ),
                )
            if first_frame_prefix_condition and chunk_idx == 0:
                first_frame_prefix_latent, first_frame_prefix_positions = _first_latent_frame_prefix(
                    state=video_state,
                    prefix_token_count=video_target_start,
                    token_count=video_tokens_per_frame,
                )
                first_frame_audio_prefix_latent, first_frame_audio_prefix_positions = _first_latent_frame_prefix(
                    state=audio_state,
                    prefix_token_count=audio_target_start,
                    token_count=first_frame_audio_prefix_token_count,
                )
            if on_chunk_ready is not None:
                on_chunk_ready(
                    video_state=video_state,
                    chunk_idx=chunk_idx,
                    frame_start=frame_start,
                    frame_end=frame_end,
                )
    finally:
        progress.close()

    stats = {
        "window_chunks": window_chunks,
        "carried_chunks_available": chunk_index_offset,
        "carried_chunks_in_window_per_chunk": carried_per_chunk,
        "selected_chunk_count_per_chunk": window_sizes,
        "video_target_start_token": video_target_start,
    }
    return video_state, audio_state, stats


class CrossVideoARPipeline(ARA2VidDistilledPipeline):
    """AR A2V pipeline whose rolling chunk window spans video boundaries."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cv_sink: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None)
        self._cv_history: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None)
        self._cv_history_ranges: tuple[list[tuple[int, int]], list[tuple[int, int]]] = ([], [])
        self._cv_window_chunks: int = 2
        self._cv_video_tools = None
        self._cv_audio_tools = None
        self._cv_audio_token_count: int | None = None
        self._cv_video_target_start: int = 0
        self.last_window_stats: dict | None = None
        # Optional hook used by the async/streaming entry. Receives one completed target chunk
        # as soon as its last denoising step has been written back into ``video_state``.
        self.streaming_chunk_callback = None

    # ------------------------------------------------------------------ public API

    def set_cross_video_context(
        self,
        *,
        sink_video: torch.Tensor | None,
        sink_audio: torch.Tensor | None,
        history_video: torch.Tensor | None,
        history_audio: torch.Tensor | None,
        history_video_ranges: list[tuple[int, int]],
        history_audio_ranges: list[tuple[int, int]],
        window_chunks: int,
    ) -> None:
        """Set the context used for the *next* video.

        ``sink_*`` is chunk 0 of the initial video (pinned prefix), ``history_*`` are the carried
        trailing chunks of the previous video together with their token ranges inside that tensor.
        """
        self._cv_sink = (sink_video, sink_audio)
        self._cv_history = (history_video, history_audio)
        self._cv_history_ranges = (list(history_video_ranges), list(history_audio_ranges))
        self._cv_window_chunks = max(1, int(window_chunks))

    def clear_cross_video_context(self) -> None:
        self.set_cross_video_context(
            sink_video=None,
            sink_audio=None,
            history_video=None,
            history_audio=None,
            history_video_ranges=[],
            history_audio_ranges=[],
            window_chunks=self._cv_window_chunks,
        )

    def set_streaming_chunk_callback(self, callback) -> None:
        """Register ``callback(chunk_idx, frame_start, frame_end, latent)`` or ``None``."""
        self.streaming_chunk_callback = callback

    def _emit_streaming_chunk(
        self,
        *,
        video_state: LatentState,
        chunk_idx: int,
        frame_start: int,
        frame_end: int,
    ) -> None:
        """Unpatchify the just-finished target chunk and hand a CPU snapshot to the caller."""
        if self.streaming_chunk_callback is None or self._cv_video_tools is None:
            return

        # ``video_state`` is patchified and may carry the [sink | carried | target] prefix.  Clearing
        # conditioning with the target start gives exactly the current video's latent sequence.
        target_state = replace(video_state, conditioning_prefix_token_count=self._cv_video_target_start)
        target_state = self._cv_video_tools.clear_conditioning(target_state)
        latent = self._cv_video_tools.unpatchify(target_state).latent
        chunk_latent = latent[:, :, frame_start:frame_end].detach().to("cpu", copy=True).contiguous()
        self.streaming_chunk_callback(
            chunk_idx=chunk_idx,
            frame_start=frame_start,
            frame_end=frame_end,
            latent=chunk_latent,
        )

    def extract_chunk_tokens(
        self,
        *,
        chunk_size: int,
        chunk_indices: list[int],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, list[tuple[int, int]], list[tuple[int, int]]]:
        """Patchified video/audio tokens of the given chunks of the last video, plus their token ranges."""
        video_grid = self.last_final_video_latent
        audio_grid = self.last_audio_latent
        if video_grid is None or audio_grid is None:
            raise RuntimeError("no generated latents available; run the pipeline once before extracting context")
        if self._cv_video_tools is None or self._cv_audio_tools is None:
            raise RuntimeError("latent tools are unavailable; the AR denoising path has not run yet")

        latent_frames = int(video_grid.shape[2])
        ranges = build_progressive_chunk_ranges(latent_frames, chunk_size)
        selected = [ranges[index] for index in chunk_indices]
        if not selected:
            return None, None, [], []

        video_frames = torch.cat([video_grid[:, :, start:end] for start, end in selected], dim=2)
        video_tokens = self._cv_video_tools.patchifier.patchify(
            video_frames.to(device=self.device, dtype=self.dtype)
        )

        audio_tokens_all = self._cv_audio_tools.patchifier.patchify(
            audio_grid.to(device=self.device, dtype=self.dtype)
        )
        if self._cv_audio_token_count is not None and audio_tokens_all.shape[1] != self._cv_audio_token_count:
            raise ValueError(
                "audio token count mismatch between the AR state and the patchified audio latent: "
                f"{self._cv_audio_token_count} vs {audio_tokens_all.shape[1]}."
            )
        audio_ranges_all = build_aligned_audio_chunk_ranges(
            ranges,
            total_video_frames=latent_frames,
            total_audio_steps=audio_tokens_all.shape[1],
        )
        selected_audio = [audio_ranges_all[index] for index in chunk_indices]
        audio_tokens = torch.cat(
            [audio_tokens_all[:, start:end] for start, end in selected_audio],
            dim=1,
        )

        tokens_per_frame = video_tokens.shape[1] // sum(end - start for start, end in selected)
        video_ranges: list[tuple[int, int]] = []
        cursor = 0
        for start, end in selected:
            length = (end - start) * tokens_per_frame
            video_ranges.append((cursor, cursor + length))
            cursor += length
        audio_ranges: list[tuple[int, int]] = []
        cursor = 0
        for start, end in selected_audio:
            audio_ranges.append((cursor, cursor + (end - start)))
            cursor += end - start
        return (
            video_tokens.detach().to("cpu"),
            audio_tokens.detach().to("cpu"),
            video_ranges,
            audio_ranges,
        )

    # ------------------------------------------------------------------ denoising

    def _denoise_video_only_ar(self, **kwargs) -> LatentState:  # noqa: PLR0915
        """Mirror of the upstream AR path with the cross-video sink/history spliced in."""
        ar_video_chunk_size = kwargs["ar_video_chunk_size"]
        window_chunks = self._cv_window_chunks
        sink_video, sink_audio = self._cv_sink
        history_video, history_audio = self._cv_history
        history_video_ranges, history_audio_ranges = self._cv_history_ranges

        video_state, video_tools = noise_video_state(
            output_shape=kwargs["output_shape"],
            noiser=kwargs["noiser"],
            conditionings=kwargs["conditionings"],
            components=self.pipeline_components,
            dtype=kwargs["dtype"],
            device=self.device,
            noise_scale=kwargs.get("noise_scale", 1.0),
            initial_latent=kwargs.get("initial_video_latent"),
        )
        self._cv_video_tools = video_tools
        video_target_token_count = int(video_state.latent.shape[1] - video_state.conditioning_prefix_token_count)

        # Layout: [sink | carried history | target]. The sink sits in the always-visible prefix; the
        # carried chunks stay outside it so the rolling window can drop them when they age out.
        if history_video is not None and history_video.shape[1] > 0:
            video_state = LatentState(
                latent=torch.cat(
                    [
                        video_state.latent[:, : video_state.conditioning_prefix_token_count],
                        history_video.to(device=video_state.latent.device, dtype=video_state.latent.dtype),
                        video_state.latent[:, video_state.conditioning_prefix_token_count :],
                    ],
                    dim=1,
                ),
                denoise_mask=torch.cat(
                    [
                        video_state.denoise_mask[:, : video_state.conditioning_prefix_token_count],
                        video_state.denoise_mask.new_zeros(
                            history_video.shape[0], history_video.shape[1], *video_state.denoise_mask.shape[2:]
                        ),
                        video_state.denoise_mask[:, video_state.conditioning_prefix_token_count :],
                    ],
                    dim=1,
                ),
                positions=torch.cat(
                    [
                        video_state.positions[:, :, : video_state.conditioning_prefix_token_count],
                        video_state.positions[:, :, :1].expand(
                            -1, -1, history_video.shape[1], -1
                        ),
                        video_state.positions[:, :, video_state.conditioning_prefix_token_count :],
                    ],
                    dim=2,
                ),
                clean_latent=torch.cat(
                    [
                        video_state.clean_latent[:, : video_state.conditioning_prefix_token_count],
                        history_video.to(device=video_state.latent.device, dtype=video_state.latent.dtype),
                        video_state.clean_latent[:, video_state.conditioning_prefix_token_count :],
                    ],
                    dim=1,
                ),
                attention_mask=None,
                conditioning_prefix_token_count=video_state.conditioning_prefix_token_count,
                channel_condition=video_state.channel_condition,
            )
        if sink_video is not None and sink_video.shape[1] > 0:
            video_state = _prepend_state_tokens(video_state, sink_video, label="video sink")

        # Rebuild the position grid as one canonical video laid out like [sink | carried | target],
        # so the relative-position window (the head of the grid) keeps increasing across chunks.
        if history_video is not None or sink_video is not None:
            tokens_per_frame = video_target_token_count // kwargs["total_video_latent_frames"]
            total_spliced_frames = int(video_state.latent.shape[1]) // tokens_per_frame
            video_state = replace(
                video_state,
                positions=_canonical_video_positions(
                    components=self.pipeline_components,
                    output_shape=kwargs["output_shape"],
                    total_latent_frames=total_spliced_frames,
                    device=self.device,
                    dtype=kwargs["dtype"],
                ),
            )

        # Align the channel condition with the full layout, then shrink ``n`` back to the sink so the
        # carried chunks are selected per chunk instead of being visible unconditionally.
        video_state = replace(
            video_state,
            conditioning_prefix_token_count=(
                video_state.conditioning_prefix_token_count + sum(e - s for s, e in history_video_ranges)
            ),
        )
        video_state = apply_video_channel_condition_to_state(
            video_state=video_state,
            video_tools=video_tools,
            channel_condition=kwargs.get("video_channel_condition"),
        )
        sink_token_count = int(video_state.conditioning_prefix_token_count) - sum(
            e - s for s, e in history_video_ranges
        )
        video_state = replace(video_state, conditioning_prefix_token_count=sink_token_count)
        video_target_start = sink_token_count + sum(e - s for s, e in history_video_ranges)
        self._cv_video_target_start = video_target_start
        if history_video is not None and video_target_start != video_state.latent.shape[1] - video_target_token_count:
            raise ValueError("cross-video video state layout mismatch")

        audio_state, audio_tools = noise_audio_state(
            output_shape=kwargs["output_shape"],
            noiser=kwargs["noiser"],
            conditionings=[],
            components=self.pipeline_components,
            dtype=kwargs["dtype"],
            device=self.device,
            noise_scale=0.0,
            initial_latent=kwargs.get("initial_audio_latent"),
        )
        self._cv_audio_tools = audio_tools
        self._cv_audio_token_count = int(audio_state.latent.shape[1])
        audio_target_token_count = int(audio_state.latent.shape[1])
        if history_audio is not None and history_audio.shape[1] > 0:
            history_audio = history_audio.to(device=audio_state.latent.device, dtype=audio_state.latent.dtype)
            audio_state = LatentState(
                latent=torch.cat([history_audio, audio_state.latent], dim=1),
                denoise_mask=torch.cat(
                    [
                        audio_state.denoise_mask.new_zeros(
                            history_audio.shape[0], history_audio.shape[1], *audio_state.denoise_mask.shape[2:]
                        ),
                        audio_state.denoise_mask,
                    ],
                    dim=1,
                ),
                positions=torch.cat(
                    [
                        audio_state.positions[:, :, :1].expand(-1, -1, history_audio.shape[1], -1),
                        audio_state.positions,
                    ],
                    dim=2,
                ),
                clean_latent=torch.cat([history_audio, audio_state.clean_latent], dim=1),
                attention_mask=None,
                conditioning_prefix_token_count=audio_state.conditioning_prefix_token_count,
                channel_condition=None,
            )
        if sink_audio is not None and sink_audio.shape[1] > 0:
            audio_state = _prepend_state_tokens(audio_state, sink_audio, label="audio sink")
        audio_state = replace(audio_state, denoise_mask=torch.zeros_like(audio_state.denoise_mask))
        if history_audio is not None or sink_audio is not None:
            audio_state = replace(
                audio_state,
                positions=_canonical_audio_positions(
                    components=self.pipeline_components,
                    output_shape=kwargs["output_shape"],
                    total_audio_tokens=int(audio_state.latent.shape[1]),
                    device=self.device,
                    dtype=kwargs["dtype"],
                ),
            )

        logger.info(
            "cross-video context: window=%d chunks, sink video/audio tokens=%d/%d, "
            "carried video/audio tokens=%d/%d",
            window_chunks,
            int(video_state.conditioning_prefix_token_count),
            int(audio_state.conditioning_prefix_token_count),
            sum(e - s for s, e in history_video_ranges),
            sum(e - s for s, e in history_audio_ranges),
        )

        video_state, _, self.last_window_stats = _cross_video_ar_loop(
            sigmas=kwargs["sigmas"],
            video_state=video_state,
            audio_state=audio_state,
            stepper=kwargs["stepper"],
            denoise_fn_builder=kwargs["denoise_fn_builder"],
            total_video_latent_frames=kwargs["total_video_latent_frames"],
            video_chunk_size=ar_video_chunk_size,
            video_target_token_count=video_target_token_count,
            audio_target_token_count=audio_target_token_count,
            carried_video_ranges=history_video_ranges,
            carried_audio_ranges=history_audio_ranges,
            window_chunks=window_chunks,
            sink_first_chunk=kwargs["ar_sink_first_chunk"],
            relative_positions=kwargs["ar_relative_positions"],
            history_feature_cache=kwargs["ar_history_feature_cache"],
            derive_channel_condition_from_first_chunk=kwargs[
                "derive_channel_condition_from_first_chunk"
            ],
            channel_condition_current_chunk_only=kwargs["channel_condition_current_chunk_only"],
            first_frame_prefix_condition=kwargs["ar_first_frame_prefix_condition"],
            first_frame_condition_position=kwargs["ar_first_frame_condition_position"],
            rope_max_temporal_index=kwargs["ar_rope_max_temporal_index"],
            on_chunk_ready=(
                self._emit_streaming_chunk if self.streaming_chunk_callback is not None else None
            ),
        )
        # Strip the sink + carried chunks before unpatchifying so the decoded video only has target frames.
        video_state = replace(
            video_state,
            conditioning_prefix_token_count=(
                int(video_state.conditioning_prefix_token_count)
                + sum(e - s for s, e in history_video_ranges)
            ),
        )
        video_state = video_tools.clear_conditioning(video_state)
        return video_tools.unpatchify(video_state)
