"""Asynchronous VAE decoding and frame broadcasting for the AR A2V pipeline.

The stock AR entry runs ``DiT chunk 0..N`` and only then decodes the complete latent with the
VAE.  This module keeps that DiT path untouched but exposes every completed chunk through a
small callback.  A worker thread decodes ``current chunk + previous chunk(s)`` on its own CUDA
stream (optionally on a second GPU) while the main thread continues the next DiT chunk.

Decoding a sliding latent window is what makes the output streamable: a VAE window starting at
latent frame ``s`` produces video frames ``8*s .. 8*(e-1)`` for the window ``[s, e)``.  Windows
overlap, so the first frames of a new window are duplicates and are dropped by a monotonic
pixel cursor.  At 352x192 / chunk size 4 this gives 25, 32, 32, ... new frames per chunk, i.e.
the same video cadence as the serial full decode while frames become available much earlier.
"""

from __future__ import annotations

import io
import logging
import queue
import threading
import time
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image

from cross_video_ar_pipeline import CrossVideoARPipeline

logger = logging.getLogger(__name__)


def _to_uint8_rgb(decoded: torch.Tensor) -> torch.Tensor:
    """``[B, 3, T, H, W]`` in [-1, 1] -> ``[T, H, W, 3]`` uint8."""
    if decoded.shape[1] != 3:
        raise ValueError(f"expected 3 decoded channels, got shape {tuple(decoded.shape)}")
    frames = (((decoded + 1.0) / 2.0).clamp_(0.0, 1.0) * 255.0).to(torch.uint8)
    return frames[0].permute(1, 2, 3, 0).contiguous()


class FrameBroadcaster:
    """Fan out decoded frames to MJPEG subscribers.

    JPEG encoding is skipped entirely when nobody is watching, so headless benchmark runs do not
    pay the CPU cost.  Slow clients get their oldest queued frame dropped instead of blocking the
    decoder thread.
    """

    def __init__(self, quality: int = 85, keepalive_seconds: float = 0.25) -> None:
        self.quality = int(quality)
        self.keepalive_seconds = max(0.05, float(keepalive_seconds))
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue[bytes | None]] = []
        self._latest: bytes | None = None
        self._frame_index = 0
        self._last_publish_at = 0.0
        self._closed = False
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop,
            name=f"mjpeg-keepalive-{id(self):x}",
            daemon=True,
        )
        self._keepalive_thread.start()

    @property
    def frame_index(self) -> int:
        with self._lock:
            return self._frame_index

    @property
    def latest(self) -> bytes | None:
        with self._lock:
            return self._latest

    def has_subscribers(self) -> bool:
        with self._lock:
            return bool(self._subscribers)

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def subscribe(self) -> queue.Queue[bytes | None]:
        q: queue.Queue[bytes | None] = queue.Queue(maxsize=120)
        with self._lock:
            if self._closed:
                q.put_nowait(None)
            else:
                self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[bytes | None]) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def publish(self, frame: torch.Tensor) -> None:
        with self._lock:
            self._frame_index += 1
            subscribers = list(self._subscribers)
        if not subscribers:
            return

        array = frame.detach().cpu().numpy()
        if array.shape[0] != 1:
            raise ValueError(f"FrameBroadcaster expects one frame, got {array.shape[0]}")
        buffer = io.BytesIO()
        Image.fromarray(array[0], mode="RGB").save(
            buffer,
            format="JPEG",
            quality=self.quality,
        )
        payload = buffer.getvalue()

        with self._lock:
            self._latest = payload
            self._last_publish_at = time.monotonic()
        for q in subscribers:
            try:
                q.put_nowait(payload)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(payload)
                except (queue.Empty, queue.Full):
                    pass

    def close(self) -> None:
        with self._lock:
            self._closed = True
            subscribers = list(self._subscribers)
            self._subscribers.clear()
        for q in subscribers:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass

    def _keepalive_loop(self) -> None:
        """Repeat the latest JPEG during a silent gap so MJPEG clients never time out."""
        sleep_for = min(0.1, self.keepalive_seconds / 2.0)
        while True:
            time.sleep(sleep_for)
            with self._lock:
                if self._closed:
                    return
                latest = self._latest
                subscribers = list(self._subscribers)
                last_publish_at = self._last_publish_at
            if not latest or not subscribers:
                continue
            if time.monotonic() - last_publish_at < self.keepalive_seconds:
                continue
            for q in subscribers:
                try:
                    q.put_nowait(latest)
                except queue.Full:
                    try:
                        q.get_nowait()
                        q.put_nowait(latest)
                    except (queue.Empty, queue.Full):
                        pass
            with self._lock:
                self._last_publish_at = time.monotonic()


class AsyncChunkVAEDecoder:
    """Decode AR chunks on a worker thread as soon as they are finished.

    Args:
        device: CUDA device used by the VAE (can differ from the DiT device).
        chunk_size: AR chunk size in latent frames.
        total_latent_frames: Number of latent frames in the current video.
        total_frames: Number of pixel frames in the current video.
        window_latent_frames: Decode window; defaults to two AR chunks.  A larger window gives
            more temporal context at the cost of a bigger VAE forward.
        pace_frames: Publish to the browser at ``fps`` instead of as fast as decoding finishes.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        chunk_size: int,
        total_latent_frames: int,
        total_frames: int,
        fps: float,
        seed: int = 42,
        window_latent_frames: int | None = None,
        broadcaster: FrameBroadcaster | None = None,
        pace_frames: bool = True,
        jpeg_quality: int = 85,
        on_progress=None,
        job_id: int = 1,
        job_name: str = "",
        audio_path: str = "",
        start_event: threading.Event | None = None,
        on_job_start=None,
        on_job_done=None,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.chunk_size = int(chunk_size)
        self.total_latent_frames = int(total_latent_frames)
        self.total_frames = int(total_frames)
        self.fps = float(fps)
        self.seed = int(seed)
        self.window_latent_frames = int(
            window_latent_frames if window_latent_frames is not None else 2 * self.chunk_size
        )
        if self.window_latent_frames < self.chunk_size:
            self.window_latent_frames = self.chunk_size
        # The last frame of a window is decoded without any future latent context.  If the next
        # window starts before it (window > chunk size), hold that frame back and publish the
        # better-context version from the next window.  The final window of the video publishes
        # everything because no future context exists there anyway.
        self._hold_last_frame = self.window_latent_frames > self.chunk_size
        self.broadcaster = broadcaster
        self.pace_frames = bool(pace_frames)
        self.jpeg_quality = int(jpeg_quality)
        self.on_progress = on_progress
        self.job_id = int(job_id)
        self.job_name = job_name
        self.audio_path = audio_path
        self.start_event = start_event
        self.on_job_start = on_job_start
        self.on_job_done = on_job_done
        self.playback_done_event = threading.Event()

        self._decoder = None
        self._generator: torch.Generator | None = None
        self._stream: torch.cuda.Stream | None = None
        self._decoder_lock = threading.Lock()
        self._started = False

        self._pending: queue.Queue[Any] = queue.Queue()
        self._publish_queue: queue.Queue[Any] = queue.Queue()
        self._latent_frames: dict[int, torch.Tensor] = {}
        self._frames: list[torch.Tensor] = []
        self._cond = threading.Condition()
        self._next_pixel = 0
        self._decode_done = False
        self._error: BaseException | None = None

        self._decode_thread = threading.Thread(
            target=self._decode_loop,
            name=f"async-vae-{id(self):x}",
            daemon=True,
        )
        self._publisher_thread = threading.Thread(
            target=self._publisher_loop,
            name=f"frame-publisher-{id(self):x}",
            daemon=True,
        )

        self.decoded_chunks = 0
        self.frames_decoded = 0
        self.frames_published = 0
        self.last_decode_ms = 0.0
        self.last_window = (0, 0)

    # ------------------------------------------------------------------ lifecycle

    def attach_decoder(self, decoder: torch.nn.Module) -> None:
        """Move the real VAE decoder to ``self.device`` and start the worker threads."""
        with self._decoder_lock:
            if self._decoder is not None:
                return
            self._decoder = decoder.to(self.device)
            self._decoder.eval()
            self._generator = torch.Generator(device=self.device).manual_seed(self.seed)
            self._stream = torch.cuda.Stream(device=self.device)
            self._started = True
        self._decode_thread.start()
        self._publisher_thread.start()
        logger.info(
            "async VAE: decoder=%s window=%d latent frames total=%d video_frames=%d pace=%s",
            self.device,
            self.window_latent_frames,
            self.total_latent_frames,
            self.total_frames,
            self.pace_frames,
        )

    def submit_chunk(self, *, chunk_idx: int, frame_start: int, frame_end: int, latent: torch.Tensor) -> None:
        """Called by the DiT chunk callback (main thread).  Must not block on decoding."""
        if not self._started:
            raise RuntimeError("submit_chunk() called before attach_decoder()")
        if latent.ndim != 5 or latent.shape[2] != frame_end - frame_start:
            raise ValueError(
                f"unexpected chunk latent shape {tuple(latent.shape)} for frames "
                f"[{frame_start}, {frame_end})"
            )
        self._pending.put((int(chunk_idx), int(frame_start), int(frame_end), latent))

    def finish(self) -> None:
        """Wait for decoding and playback (batch-mode compatibility helper)."""
        self.finish_decode()
        self.finish_playback()

    def finish_decode(self) -> None:
        """Wait for all submitted chunks to decode; playback may still be running."""
        if not self._started:
            return
        self._pending.put(None)
        self._decode_thread.join()
        if self._error is not None:
            raise RuntimeError(f"async VAE decoding failed: {self._error}") from self._error
        if len(self._frames) != self.total_frames:
            raise RuntimeError(
                f"async VAE produced {len(self._frames)} frames, expected {self.total_frames}"
            )

    def finish_playback(self) -> None:
        """Wait until this job's paced frames have all been published."""
        if not self._started:
            return
        self._publisher_thread.join()
        if self._error is not None:
            raise RuntimeError(f"async VAE decoding failed: {self._error}") from self._error

    def first_frame(self) -> torch.Tensor | None:
        """Return a copy of the first decoded ``[1, 3, 1, H, W]`` frame in [-1, 1]."""
        with self._cond:
            if not self._frames:
                return None
            return self._frames[0].detach().clone()

    def last_frame(self) -> torch.Tensor | None:
        """Return a copy of the last decoded ``[1, 3, 1, H, W]`` frame in [-1, 1]."""
        with self._cond:
            if not self._frames:
                return None
            return self._frames[-1].detach().clone()

    def abort(self, error: BaseException | None = None) -> None:
        with self._cond:
            if self._error is None:
                self._error = error or RuntimeError("aborted")
            self._decode_done = True
            self._cond.notify_all()
        for q in (self._pending, self._publish_queue):
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        if self.start_event is not None:
            self.start_event.set()
        self.playback_done_event.set()

    # ------------------------------------------------------------------ consumer

    def iter_frames(self) -> Iterator[torch.Tensor]:
        """Yield decoded ``[1, 3, 1, H, W]`` frames in order, blocking until they exist.

        The shape/dtype intentionally matches what ``VideoDecoder`` would return, because the
        upstream ``decode_video()`` helper runs its own float[-1, 1] -> uint8 conversion over
        ``tiled_decode()`` output.  Returning uint8 here would make that helper convert twice.
        """
        index = 0
        while index < self.total_frames:
            with self._cond:
                while index >= len(self._frames) and not self._decode_done and self._error is None:
                    self._cond.wait(timeout=0.5)
                if self._error is not None:
                    raise RuntimeError(f"async VAE decoding failed: {self._error}") from self._error
                if index >= len(self._frames):
                    if self._decode_done:
                        raise RuntimeError(
                            f"async VAE finished with {len(self._frames)}/{self.total_frames} frames"
                        )
                    continue
                frame = self._frames[index]
            index += 1
            yield frame

    def snapshot(self) -> dict[str, Any]:
        with self._cond:
            decoded = self.frames_decoded
            queued = len(self._frames) - self.frames_published
        return {
            "job_id": self.job_id,
            "job_name": self.job_name,
            "decode_device": str(self.device),
            "decode_window_latent_frames": self.window_latent_frames,
            "decoded_chunks": self.decoded_chunks,
            "frames_decoded": decoded,
            "frames_published": self.frames_published,
            "decode_backlog": max(0, queued),
            "last_decode_ms": round(self.last_decode_ms, 1),
            "last_window": list(self.last_window),
        }

    # ------------------------------------------------------------------ worker

    def _decode_loop(self) -> None:
        try:
            torch.cuda.set_device(self.device)
            while True:
                item = self._pending.get()
                if item is None:
                    break
                chunk_idx, frame_start, frame_end, latent = item
                for offset in range(frame_end - frame_start):
                    self._latent_frames[frame_start + offset] = latent[:, :, offset : offset + 1].contiguous()
                self.decoded_chunks += 1
                self._decode_new_frames(frame_end)
                if self.on_progress is not None:
                    self.on_progress(**{
                        "decoded_chunks": self.decoded_chunks,
                        "last_decode_ms": round(self.last_decode_ms, 1),
                        "decode_window": list(self.last_window),
                    })
        except BaseException as exc:  # noqa: BLE001
            logger.exception("async VAE worker failed")
            with self._cond:
                self._error = exc
                self._decode_done = True
                self._cond.notify_all()
        finally:
            with self._cond:
                self._decode_done = True
                self._cond.notify_all()
            try:
                self._publish_queue.put_nowait(None)
            except queue.Full:
                self._publish_queue.put(None)

    def _decode_new_frames(self, completed_end: int) -> None:
        start = max(0, completed_end - self.window_latent_frames)
        missing = [i for i in range(start, completed_end) if i not in self._latent_frames]
        if missing:
            raise RuntimeError(f"latent window [{start}, {completed_end}) is missing frames {missing}")

        window = torch.cat([self._latent_frames[i] for i in range(start, completed_end)], dim=2)
        began = time.perf_counter()
        with torch.inference_mode(), torch.cuda.stream(self._stream):
            sample = window.to(device=self.device, dtype=self.dtype, non_blocking=False)
            decoded = self._decoder(sample, generator=self._generator)
            decoded_cpu = decoded.detach().to("cpu")
            uint8_frames = _to_uint8_rgb(decoded).cpu()
        self.last_decode_ms = (time.perf_counter() - began) * 1000.0
        self.last_window = (start, completed_end)

        global_start = 8 * start
        frame_limit = decoded_cpu.shape[2]
        if self._hold_last_frame and completed_end < self.total_latent_frames:
            # Hold back one latent step (8 pixel frames): those frames still benefit from the next
            # latent frame, and the next window (which starts at least one chunk earlier) will
            # re-decode them with that future context.
            frame_limit = max(1, frame_limit - 8)
        with self._cond:
            for offset in range(frame_limit):
                pixel_index = global_start + offset
                if pixel_index < self._next_pixel:
                    continue
                if pixel_index > self._next_pixel:
                    raise RuntimeError(
                        f"decoded frame gap: got {pixel_index}, expected {self._next_pixel}"
                    )
                # The final MP4 path re-uses the upstream float -> uint8 conversion.
                self._frames.append(decoded_cpu[:, :, offset : offset + 1])
                self.frames_decoded += 1
                self._next_pixel += 1
                # The browser gets the already-converted RGB frame.
                self._publish_queue.put((pixel_index, uint8_frames[offset : offset + 1]))
            self._cond.notify_all()

    def _publisher_loop(self) -> None:
        if self.start_event is not None:
            self.start_event.wait()
        if self.on_job_start is not None:
            self.on_job_start(self)
        first_time: float | None = None
        first_index = 0
        try:
            while True:
                item = self._publish_queue.get()
                if item is None:
                    break
                pixel_index, frame = item
                # Headless runs should not wait for playback pacing.  As soon as a browser
                # subscribes, subsequent frames are paced at the target fps.
                if (
                    self.pace_frames
                    and self.fps > 0
                    and self.broadcaster is not None
                    and self.broadcaster.has_subscribers()
                ):
                    if first_time is None:
                        first_time = time.monotonic()
                        first_index = pixel_index
                    target = first_time + (pixel_index - first_index) / self.fps
                    delay = target - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                if self.broadcaster is not None:
                    self.broadcaster.publish(frame)
                self.frames_published = pixel_index + 1
                if self.on_progress is not None:
                    self.on_progress(frames_published=self.frames_published)
        except BaseException as exc:  # noqa: BLE001
            logger.exception("frame publisher failed")
            with self._cond:
                if self._error is None:
                    self._error = exc
                self._cond.notify_all()
        finally:
            self.playback_done_event.set()
            if self.on_job_done is not None:
                self.on_job_done(self)


class StreamingVideoDecoderProxy:
    """Stand-in for ``VideoDecoder`` used by the upstream pipeline's final decode call.

    The real decoder has already been run chunk-by-chunk by :class:`AsyncChunkVAEDecoder`.
    This proxy only replays the decoded frames in order, so the upstream code path (which builds
    the final MP4) does not trigger a second, serial full-video decode.
    """

    def __init__(self, session: AsyncChunkVAEDecoder) -> None:
        self.session = session

    def tiled_decode(self, latent, tiling_config=None, timestep=None, generator=None):  # noqa: ANN001
        yield from self.session.iter_frames()

    def __call__(self, latent, timestep=None, generator=None):  # noqa: ANN001
        raise RuntimeError("StreamingVideoDecoderProxy only supports tiled_decode()")


class StreamingCrossVideoARPipeline(CrossVideoARPipeline):
    """Cross-video AR pipeline with an optional async VAE session."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._stream_session: AsyncChunkVAEDecoder | None = None
        self._stream_real_decoder: torch.nn.Module | None = None

    def set_streaming_session(self, session: AsyncChunkVAEDecoder | None) -> None:
        self._stream_session = session
        self.set_streaming_chunk_callback(session.submit_chunk if session is not None else None)

    def _get_fast_modules(self, *, stage_mode, transformer_extra_config=None):  # noqa: ANN001
        modules = super()._get_fast_modules(
            stage_mode=stage_mode,
            transformer_extra_config=transformer_extra_config,
        )
        if self._stream_session is None:
            return modules

        real_decoder = self._stream_real_decoder
        if real_decoder is None:
            real_decoder = modules.video_decoder
            self._stream_real_decoder = real_decoder
        self._stream_session.attach_decoder(real_decoder)
        modules.video_decoder = StreamingVideoDecoderProxy(self._stream_session)
        return modules
