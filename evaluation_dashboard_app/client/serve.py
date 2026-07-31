"""Run the bbox API against the local workspace.

The server code is reused verbatim; the only thing the client changes is where it looks
for data and where it writes caches, both via environment variables read by
``backend.app_paths``.

It runs in a background thread of this process rather than a child process. The
original concern with in-process hosting was ``scene_result.pkl``: unpickled scenes are
huge and the cache holds one at a time. A pulled workspace has no pickles unless the
user explicitly chose the ``raw`` tier, because the DevOps answers are pre-baked, so
that pressure is gone and a thread keeps the packaged app single-process.
"""

from __future__ import annotations

import socket
import threading
from typing import Any

from client import config


def find_free_port(preferred: int = 8765, attempts: int = 40) -> int:
    """First free port at or above ``preferred``, falling back to any free port."""
    for offset in range(attempts):
        candidate = preferred + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            return candidate
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class LocalServer:
    """A bbox API bound to loopback, serving the client workspace."""

    def __init__(self, port: int | None = None, host: str = "127.0.0.1") -> None:
        config.apply_server_env()
        self.host = host
        self.port = port or find_free_port()
        self._httpd: Any = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> str:
        # Imported after apply_server_env() so module-level path defaults see the
        # workspace rather than the process working directory.
        from http.server import ThreadingHTTPServer

        from client.webapi import build_handler

        self._httpd = ThreadingHTTPServer((self.host, self.port), build_handler())
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="evaldash-local-api", daemon=True
        )
        self._thread.start()
        return self.base_url

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        self._thread = None

    def serve_forever(self) -> None:
        """Block until interrupted, for the CLI's foreground ``serve``."""
        if self._httpd is None:
            self.start()
        try:
            while self._thread is not None and self._thread.is_alive():
                self._thread.join(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def __enter__(self) -> "LocalServer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
