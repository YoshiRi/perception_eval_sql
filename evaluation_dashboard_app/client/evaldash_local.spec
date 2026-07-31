# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the double-clickable local client.

Build from the repo root:

    pyinstaller client/evaldash_local.spec --noconfirm

Produces ``dist/evaldash-local`` -- a single file that serves the viewer over loopback
and opens a window. The bundled ``static/`` tree is found through
``backend.app_paths.app_root()``, which resolves to PyInstaller's ``_MEIPASS`` when
frozen, so no path configuration is needed at runtime.
"""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs, copy_metadata

REPO_ROOT = Path(os.path.abspath(SPECPATH)).parent  # noqa: F821 - SPECPATH is injected

# The viewer is served from disk at request time, so every asset it can ask for has to
# be inside the bundle. Ship the whole directory rather than an allowlist that would
# silently drift as renderers are added.
datas = [
    (str(REPO_ROOT / "static" / name), "static")
    for name in os.listdir(REPO_ROOT / "static")
    if name.endswith((".js", ".css", ".html"))
]

# duckdb's __init__ resolves its own version through importlib.metadata, which fails
# with PackageNotFoundError unless the .dist-info is inside the bundle. Metadata is not
# collected automatically, so it has to be requested explicitly.
for _package in ("duckdb", "pandas", "numpy"):
    try:
        datas += copy_metadata(_package)
    except Exception:
        pass

# duckdb ships a large compiled extension; let PyInstaller find its binaries rather
# than guessing at filenames that change between versions.
binaries = collect_dynamic_libs("duckdb")

hiddenimports = [
    "duckdb",
    "yaml",
    "backend.app_paths",
    "backend.export_api",
    "backend.local_bbox_api",
    "backend.prebake",
    "backend.prebake_cli",
    "backend.workflow_api",
    "client.app",
    "client.cli",
    "client.config",
    "client.remote",
    "client.serve",
    "client.sync",
    "client.t4",
    "client.webapi",
]

# Only present when build_app.sh was given --server; config.py imports it optionally.
if (REPO_ROOT / "client" / "_defaults.py").exists():
    hiddenimports.append("client._defaults")

# pywebview is optional: when it is installed at build time the window backend is
# bundled, and when it is not the app falls back to the system browser. Only backends
# that actually import here are named -- listing an absent one (no qtpy, say) just
# produces build noise and cannot help at runtime.
try:
    import importlib

    import webview  # noqa: F401

    hiddenimports.append("webview")
    for _backend in ("gtk", "qt", "cocoa", "winforms", "edgechromium"):
        try:
            importlib.import_module(f"webview.platforms.{_backend}")
        except Exception:
            continue
        hiddenimports.append(f"webview.platforms.{_backend}")
except ImportError:
    pass

# The client needs duckdb, pandas/numpy (duckdb's .df()), and the stdlib. Everything
# below is reachable from the environment but unused, and excluding it is what keeps
# the bundle at a sane size.
excludes = [
    # Dashboard-only: Streamlit, plotting, reporting, queue and DB stacks.
    "streamlit",
    "matplotlib",
    "plotly",
    "kaleido",
    "reportlab",
    "weasyprint",
    "bokeh",
    "shapely",
    "psycopg2",
    "rq",
    "redis",
    "docker",
    "polars",
    "sqlalchemy",
    # Parquet goes through DuckDB's own reader, so Arrow is dead weight.
    "pyarrow",
    "numba",
    "llvmlite",
    "tables",
    "openpyxl",
    "xlrd",
    "lxml",
    # Two Qt bindings are installed here and PyInstaller refuses to bundle both. The
    # window uses the GTK backend, so drop Qt entirely rather than picking one.
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "qtpy",
    # Credential/crypto stack pulled in transitively; the client only speaks HTTPS
    # through the stdlib.
    "keyring",
    "cryptography",
    "nacl",
    "bcrypt",
    "paramiko",
    "botocore",
    "boto3",
    "googleapiclient",
    # Notebook/science extras.
    "IPython",
    "notebook",
    "jupyter",
    "zmq",
    "scipy",
    "sklearn",
    "PIL",
    "tkinter",
    "tests",
]

a = Analysis(  # noqa: F821
    [str(REPO_ROOT / "client" / "app.py")],
    pathex=[str(REPO_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="evaldash-local",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    # A GUI-only build would hide the CLI and swallow tracebacks; the app is usable
    # both ways from one executable, so keep a console on Windows too.
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
