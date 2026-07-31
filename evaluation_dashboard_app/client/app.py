"""Desktop shell: a window over the locally served viewer.

Two backends, in order of preference:

1. ``pywebview`` -- a real application window with no browser chrome. Optional, because
   it needs a native webview (WebKitGTK on Linux, WebView2 on Windows, WKWebView on
   macOS) that we cannot assume is present.
2. the system browser -- always available, and the viewer is an ordinary web page, so
   nothing is lost but the window frame.

The fallback is not a degraded mode to apologise for: the same HTML, the same local
server, the same offline behaviour. It is only a different frame around it.
"""

from __future__ import annotations

import os
import sys
import threading
import webbrowser

from client import config, serve

WINDOW_TITLE = "Evaluation Dashboard - Local"
DEFAULT_SIZE = (1600, 1000)


def _has_pywebview() -> bool:
    if os.environ.get("EVALDASH_FORCE_BROWSER") == "1":
        return False
    try:
        import webview  # noqa: F401
    except Exception:
        return False
    return True


PAGES = {"home": "", "explorer": "explorer", "viewer": "viewer"}


def _page_url(base_url: str, page: str) -> str:
    suffix = PAGES.get(page, "")
    return f"{base_url}/{suffix}" if suffix else f"{base_url}/"


def launch(port: int | None = None, page: str = "home", prefer_browser: bool = False) -> int:
    if not config.local_runs():
        # Not a warning any more: the home page is where you download a run, so an
        # empty workspace is the expected first-run state.
        print("workspace is empty - use the app's home page to download a run", file=sys.stderr)
    server = serve.LocalServer(port=port)
    base_url = server.start()
    url = _page_url(base_url, page)
    print(f"serving   {base_url}")
    print(f"workspace {config.workspace_dir()}")

    if not prefer_browser and _has_pywebview():
        return _run_window(server, url)
    return _run_browser(server, url)


def _run_window(server: serve.LocalServer, url: str) -> int:
    import webview

    print(f"opening   {url} (native window)")
    try:
        webview.create_window(WINDOW_TITLE, url, width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1])
        webview.start()
    except Exception as exc:
        # A missing/broken native webview only shows up at start(); fall back rather
        # than leaving the user with a started server and no visible UI.
        print(f"native window unavailable ({exc}); falling back to the browser", file=sys.stderr)
        return _run_browser(server, url)
    finally:
        server.stop()
    return 0


def _run_browser(server: serve.LocalServer, url: str) -> int:
    print(f"opening   {url} (system browser)")
    # Open on a timer so the server is already accepting connections.
    threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    print("\nCtrl-C to stop.")
    server.serve_forever()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the packaged, double-clickable app.

    Takes no arguments in the normal case: double-clicking cannot supply any. If the
    workspace has no runs yet, the page explains that and the CLI is still available
    from the same executable via ``--cli``.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--cli":
        from client.cli import main as cli_main

        return cli_main(args[1:])
    if args:
        from client.cli import main as cli_main

        return cli_main(args)
    return launch()


if __name__ == "__main__":
    raise SystemExit(main())
