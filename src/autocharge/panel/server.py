"""Loopback-only standard-library operator server. MuJoCo/GL stay on one worker."""

import concurrent.futures
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .session import Session, plain

STATIC = Path(__file__).parent / "static"


class Worker:
    def __init__(self, output):
        self.commands = queue.Queue()
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.current = dict(status="loading", records=[], frames=0)
        self.jpeg = None
        self.frame_meta = {}
        self.output = output
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, name, body):
        future = concurrent.futures.Future()
        self.commands.put((name, body, future))
        return future.result(timeout=90)

    def run(self):
        session = Session(self.output)
        try:
            try:
                session.configure(dict(level=1, seed=42))
            except Exception as exc:
                session.error = str(exc)
                session.status = "invalid"
            last_render = last_state = 0.0
            while not self.closed.is_set():
                start = time.monotonic()
                changed = False
                while True:
                    try:
                        name, body, future = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        result = session.action(name, body)
                        # Publish committed scene before acknowledging the command.
                        # A render must not leave polls reporting the previous scene.
                        with self.lock:
                            self.current = session.state()
                        future.set_result(self.current if isinstance(result,dict) and "status" in result else result)
                    except Exception as exc:
                        session.error = str(exc)
                        future.set_exception(ValueError(str(exc)))
                    changed = True
                try:
                    session.tick(10)
                except Exception as exc:
                    session.status = "paused"
                    session.error = f"Simulation stopped: {exc}"
                now = time.monotonic()
                if changed or now - last_render > 0.15:
                    try:
                        frame = session.render()
                        with self.lock:
                            self.jpeg = frame
                            self.frame_meta = {
                                "X-Autocharge-Attempt": session.playing.id
                                if session.playing
                                else session.record.id,
                                "X-Autocharge-Camera": session.camera,
                                "X-Autocharge-Index": str(session.play_index)
                                if session.playing
                                else "live",
                            }
                    except Exception as exc:
                        session.frame_error = f"Rendering unavailable: {exc}"
                    last_render = time.monotonic()
                if changed or now - last_state > 0.1:
                    with self.lock:
                        self.current = session.state()
                    last_state = now
                self.closed.wait(max(0, 0.02 - (time.monotonic() - start)))
        finally:
            session.close()

    def close(self):
        self.closed.set()
        self.thread.join(timeout=10)


def make_server(host="127.0.0.1", port=8765, output="outputs/panel"):
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("Operator panel must bind to loopback")
    worker = Worker(output)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, data, kind="application/json", extra=None):
            if isinstance(data, dict):
                data = json.dumps(data, default=plain).encode()
            if isinstance(data, str):
                data = data.encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/state":
                with worker.lock:
                    state = worker.current
                self.reply(200, state)
            elif path == "/api/frame":
                with worker.lock:
                    frame = worker.jpeg
                    metadata = dict(worker.frame_meta)
                if frame:
                    self.reply(200, frame, "image/jpeg", extra=metadata)
                else:
                    self.reply(
                        503, dict(error="Viewport initializing or renderer unavailable")
                    )
            elif path in (
                "/sample-layouts/box_avoidance.json",
                "/sample-layouts/stone_slope.json",
            ):
                self.reply(
                    200,
                    (
                        Path(__file__).resolve().parents[3]
                        / "layouts"
                        / path.rsplit("/", 1)[1]
                    ).read_bytes(),
                    "application/json",
                )
            elif path in ("/", "/app.js", "/style.css"):
                name = "index.html" if path == "/" else path[1:]
                kinds = {
                    "index.html": "text/html; charset=utf-8",
                    "app.js": "text/javascript; charset=utf-8",
                    "style.css": "text/css; charset=utf-8",
                }
                self.reply(200, (STATIC / name).read_bytes(), kinds[name])
            else:
                self.reply(404, dict(error="Not found"))

        def do_POST(self):
            if not self.path.startswith("/api/"):
                self.reply(404, dict(error="Not found"))
                return
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                self.reply(403, dict(error="Same-origin commands only"))
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.reply(415, dict(error="JSON required"))
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 100000:
                    raise ValueError("Invalid command size")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("Object required")
                result = worker.submit(self.path.removeprefix("/api/"), body)
                self.reply(200, result)
            except (
                ValueError,
                TypeError,
                KeyError,
                concurrent.futures.TimeoutError,
            ) as exc:
                self.reply(400, dict(error=str(exc)))

    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except Exception:
        worker.close()
        raise
    server.worker = worker
    return server
