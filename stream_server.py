"""Tiny dependency-free HTTP server for the async/streaming inference entry."""

from __future__ import annotations

import json
import logging
import queue
import re
import socket
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from streaming_vae import FrameBroadcaster

logger = logging.getLogger(__name__)

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class ServerState:
    """Shared state between the inference thread and HTTP handlers."""

    def __init__(
        self,
        *,
        broadcaster: FrameBroadcaster,
        frontend_config: dict[str, Any] | None = None,
    ) -> None:
        self.broadcaster = broadcaster
        self.frontend_config = dict(frontend_config or {})
        self._lock = threading.Lock()
        self._audio_files: dict[int, str] = {}
        self._video_files: dict[int, str] = {}
        self.current_session = None
        self.started_at = time.time()
        self.status: dict[str, Any] = {
            "phase": "loading",
            "message": "正在加载模型",
            "job_index": 0,
            "job_total": 0,
            "job_name": None,
            "audio": None,
            "num_frames": 0,
            "latent_frames": 0,
            "chunks": 0,
            "fps": 0,
            "frames_decoded": 0,
            "frames_published": 0,
            "decoded_chunks": 0,
            "last_decode_ms": 0,
            "decode_window": [],
            "decode_device": None,
            "package_video": False,
            "videos": [],
            "error": None,
            "elapsed": 0,
        }

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            self.status.update(kwargs)

    def set_audio_files(self, files: list[Path]) -> None:
        with self._lock:
            self._audio_files = {index: str(path) for index, path in enumerate(files, 1)}

    def register_audio(self, index: int, path: Path) -> None:
        with self._lock:
            self._audio_files[int(index)] = str(path)

    def audio_path(self, index: int) -> Path | None:
        with self._lock:
            value = self._audio_files.get(index)
        return Path(value) if value is not None else None

    def add_video(self, *, index: int, audio: str, output: Path) -> None:
        with self._lock:
            self._video_files[index] = str(output)
            videos = list(self.status.get("videos") or [])
            videos = [item for item in videos if item.get("index") != index]
            videos.append(
                {
                    "index": index,
                    "audio": audio,
                    "output": str(output),
                    "url": f"/video/{index}",
                    "audio_url": f"/audio/{index}",
                }
            )
            self.status["videos"] = sorted(videos, key=lambda item: item["index"])

    def video_path(self, index: int) -> Path | None:
        with self._lock:
            value = self._video_files.get(index)
        return Path(value) if value is not None else None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            snapshot = dict(self.status)
            snapshot["elapsed"] = round(time.time() - self.started_at, 2)
            snapshot["videos"] = list(snapshot.get("videos") or [])
            session = self.current_session
        if session is not None:
            try:
                snapshot.update(session.snapshot())
            except Exception:  # noqa: BLE001
                logger.debug("could not snapshot async decoder", exc_info=True)
        snapshot["stream_clients"] = self.broadcaster.subscriber_count()
        return snapshot


class _Handler(BaseHTTPRequestHandler):
    server_version = "AvatarStream/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("%s - %s", self.address_string(), fmt % args)

    @property
    def state(self) -> ServerState:
        return self.server.state  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._serve_frontend()
        elif path in ("/frontend_config.json", "/frontend-config.json"):
            self._serve_frontend_config()
        elif path == "/status.json":
            self._serve_json()
        elif path == "/health":
            self._send_bytes(b"ok\n", "text/plain; charset=utf-8")
        elif path == "/stream.mjpg":
            self._serve_mjpeg()
        elif path.startswith("/audio/"):
            self._serve_media(path, kind="audio")
        elif path.startswith("/video/"):
            self._serve_media(path, kind="video")
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "not found")

    # ------------------------------------------------------------------ endpoints

    def _serve_frontend(self) -> None:
        html_path = Path(__file__).resolve().parent / "frontend" / "stream_player.html"
        if not html_path.is_file():
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "frontend file missing")
            return
        self._send_bytes(html_path.read_bytes(), "text/html; charset=utf-8")

    def _serve_frontend_config(self) -> None:
        payload = json.dumps(self.state.frontend_config, ensure_ascii=False).encode("utf-8")
        self._send_bytes(payload, "application/json; charset=utf-8", cache=False)

    def _serve_json(self) -> None:
        payload = json.dumps(self.state.snapshot(), ensure_ascii=False).encode("utf-8")
        self._send_bytes(payload, "application/json; charset=utf-8", cache=False)

    def _serve_mjpeg(self) -> None:
        q = self.state.broadcaster.subscribe()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        try:
            while True:
                try:
                    payload = q.get(timeout=5.0)
                except queue.Empty:
                    continue
                if payload is None:
                    break
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.state.broadcaster.unsubscribe(q)

    def _serve_media(self, path: str, *, kind: str) -> None:
        try:
            index = int(path.rsplit("/", 1)[-1])
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "bad index")
            return
        target = self.state.audio_path(index) if kind == "audio" else self.state.video_path(index)
        if target is None or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, f"{kind} not ready")
            return
        if kind == "audio":
            content_type = {
                ".wav": "audio/wav",
                ".mp3": "audio/mpeg",
                ".flac": "audio/flac",
                ".m4a": "audio/mp4",
                ".aac": "audio/aac",
                ".ogg": "audio/ogg",
                ".opus": "audio/opus",
            }.get(target.suffix.lower(), "application/octet-stream")
        else:
            content_type = "video/mp4"
        self._serve_file(target, content_type)

    def _serve_file(self, path: Path, content_type: str) -> None:
        size = path.stat().st_size
        start = 0
        end = size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header:
            match = _RANGE_RE.match(range_header.strip())
            if match:
                start_text, end_text = match.groups()
                if start_text:
                    start = int(start_text)
                if end_text:
                    end = min(int(end_text), size - 1)
                if start > end or start >= size:
                    self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    return
                status = HTTPStatus.PARTIAL_CONTENT

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    def _send_bytes(self, payload: bytes, content_type: str, *, cache: bool = False) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)


class StreamHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], state: ServerState) -> None:
        self.state = state
        self.extra_servers: list[ThreadingHTTPServer] = []
        super().__init__(address, _Handler)

    def shutdown(self) -> None:
        super().shutdown()
        for server in self.extra_servers:
            server.shutdown()

    def server_close(self) -> None:
        super().server_close()
        for server in self.extra_servers:
            server.server_close()


class StreamHTTPServerV6(StreamHTTPServer):
    address_family = socket.AF_INET6


def start_stream_server(
    *,
    host: str,
    port: int,
    state: ServerState,
) -> tuple[StreamHTTPServer, int]:
    """Listen on IPv4 and (when possible) IPv6 loopback.

    Port-forward UIs often render the remote target as ``localhost:PORT``.  On hosts where
    ``localhost`` resolves to ``::1`` first, an IPv4-only listener makes the forwarder accept
    the local connection and then close it without a response (the browser reports
    ``ERR_EMPTY_RESPONSE``).  Listening on ``127.0.0.1`` and ``[::1]`` avoids that.
    """
    servers: list[StreamHTTPServer] = []
    primary: StreamHTTPServer

    if host in ("", "0.0.0.0", "::"):
        primary = StreamHTTPServer(("0.0.0.0", int(port)), state)
        servers.append(primary)
        try:
            ipv6 = StreamHTTPServerV6(("::1", int(primary.server_address[1])), state)
            servers.append(ipv6)
        except OSError:
            logger.warning("IPv6 loopback listener unavailable; IPv4 only", exc_info=True)
    elif ":" in host:
        primary = StreamHTTPServerV6((host, int(port)), state)
        servers.append(primary)
    else:
        primary = StreamHTTPServer((host, int(port)), state)
        servers.append(primary)

    primary.extra_servers = [server for server in servers if server is not primary]
    for index, server in enumerate(servers):
        thread = threading.Thread(
            target=server.serve_forever,
            name=f"stream-http-{index}",
            daemon=True,
        )
        thread.start()
    return primary, int(primary.server_address[1])
