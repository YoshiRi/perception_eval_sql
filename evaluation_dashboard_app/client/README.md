# evaldash-local

A local client for the perception evaluation dashboard. It downloads already-generated
result files from a running dashboard server into a workspace on your machine, then
serves the **existing** bbox explorer and viewer against them — so you can dig into
scenario details offline, without the VPN, the server, or a Streamlit session.

It reuses the dashboard's own code: the same `backend/local_bbox_api.py`, the same
`static/` renderers, the same DuckDB queries. The client adds a transport (pull files),
a workspace, and a window.

---

## Why the download is small

A large run is ~11 GB on the server, but the viewer needs almost none of that. About
95% of a run is `scene_result.pkl` files (27–133 MB each, one per scenario), which exist
only so the DevOps routes can recompute gate verdicts. The server pre-computes those
verdicts instead (see [Pre-baking](#pre-baking)), so the client never needs the pickles.

Measured on a real 11 GB run:

| Tier | Size | What you get |
|---|---:|---|
| `minimal` | 487 MB | Role parquet + run metadata. Explorer, viewer, statistics, curves. |
| `criteria` | 489 MB | + scenario YAML/sidecars + pre-baked gate and frame verdicts. **Default.** |
| `full` | 489 MB + | + pre-baked true-negative objects (the preview overlay). |
| `raw` | 9,055 MB | + `scene_result.pkl`. Only if you want to recompute from source. |

`criteria` is 18× smaller than the run and loses nothing the viewer displays.

---

## Server setup (once)

The export routes are **closed by default**. Generate a token and enable them:

```bash
./deploy/11_ENABLE_EXPORT_API.sh --apply     # writes EVAL_EXPORT_TOKEN to deploy/.env.local
./deploy/10_RESTART_STREAMLIT.sh
```

The routes ride on the existing `/bbox-api/` nginx mapping — no new port, no compose
change. Then verify through the real edge:

```bash
./deploy/12_VERIFY_EXPORT_API.sh https://your-dashboard <token>
# add a Cloudflare service token if the dashboard is behind Access:
./deploy/12_VERIFY_EXPORT_API.sh https://your-dashboard <token> <cf-id> <cf-secret>
```

That script checks the things which only differ in production: that nginx forwards the
bearer token, that it forwards `Range` (so a 465 MB pull can resume), and that you are
getting JSON rather than a Cloudflare sign-in page.

Without the token, every export route answers `503`. With a wrong token, `401`.

### Pre-baking

To let clients skip the pickles, pre-compute the DevOps answers on the server, where the
evaluator libraries already live:

```bash
# what still needs work, across every run (seconds; reads no pickles)
docker compose exec streamlit1 python3 -m backend.prebake_cli --report

# one run
docker compose exec streamlit1 python3 -m backend.prebake_cli --run <run_name>

# everything, detached and resumable
docker compose exec -d streamlit1 sh -c \
  'python3 -m backend.prebake_cli --all --role devops >/app/data/prebake.log 2>&1'
```

Cost per scenario is bimodal: instant when a scenario has no `scene_result.pkl`, minutes
when it does. `--report` tells you the split before you commit, `--max-seconds N`
time-boxes a batch, and Ctrl-C finishes the current scenario and stops. Interrupting is
always safe — entries are written one at a time and a re-run skips what exists, so work
resumes rather than restarting.

Output lands in `<run>/<role>/.export_prebake/` at roughly 5 KB per scenario (~570 KB for
108 scenarios) and is picked up automatically by both the server and any client that
pulls it.

Without pre-baking the client still works — the DevOps criteria panels just report
`available: false` the same way the server does when a pickle is missing.

---

## Install

**Option A — packaged app (recommended).**

```bash
./client/build_app.sh              # -> dist/evaldash-local (single file, ~190 MB)
./client/build_app.sh --desktop    # also installs a Linux .desktop launcher
```

Double-click `dist/evaldash-local` to open the viewer. The same executable is the CLI:
`dist/evaldash-local --cli <command>`.

For a native window rather than a browser tab, install pywebview **before** building:

```bash
pip install pywebview
sudo apt install gir1.2-webkit2-4.0 python3-gi   # Linux only
```

Without it the app opens your system browser instead — same page, same local server,
same offline behaviour, just a different frame.

**Option B — from source.** Needs `duckdb` and `pandas`; `PyYAML` is optional.

```bash
python3 -m client --help
```

---

## Use

The app is self-sufficient: **double-click it**, and its home page lets you enter the
server URL and token, browse runs with their per-tier download sizes, download with a
progress bar, and jump into the explorer. No terminal needed.

The CLI does the same things, and is what you want for scripting:

```bash
# once
evaldash-local login --server https://your-dashboard --token <EVAL_EXPORT_TOKEN>

# what is on the server, and what each tier would cost to pull
evaldash-local runs
evaldash-local runs -q pilot4.4

# pull a run
evaldash-local pull <run_name> --role devops --tier criteria
evaldash-local pull <run_name> --tier minimal --dry-run    # show the plan only

# inspect
evaldash-local open              # window (or browser) on the home page
evaldash-local open --page explorer
evaldash-local serve             # server only, no window
evaldash-local ls                # local runs
evaldash-local rm <run_name>
evaldash-local doctor            # diagnose this build and the connection

# offline 3D point clouds (see below)
evaldash-local t4 scenarios <dataset_id>
evaldash-local t4 fetch <dataset_id> --scenario <name>
evaldash-local t4 ls
```

`doctor` exits 0 on a healthy build even when nothing is configured yet; it exits 1 only
for real defects (missing duckdb or viewer assets, a configured server that will not
answer).

`--server` / `--token` on `login` are stored in `~/.evaldash/config.json` (mode 0600).
`EVALDASH_SERVER` and `EVALDASH_TOKEN` override them per-invocation.

### Behind Cloudflare Access

Pass a service token; it is sent as `CF-Access-Client-Id` / `CF-Access-Client-Secret`,
mirroring how the dashboard authenticates its own calls to the T4 visualizer:

```bash
evaldash-local login --server https://dash.example.com --token <export-token> \
  --cf-client-id <id> --cf-client-secret <secret>
```

If a Cloudflare sign-in page comes back instead of JSON, the client says so explicitly
rather than failing on a JSON parse error.

### Interrupted downloads

Pulls are resumable and incremental. A partial file is kept as `<name>.evaldash-part`
and the next `pull` continues from that offset via an HTTP `Range` request; re-running
after a completed pull transfers nothing.

- Files are verified against the server's sha256 as they land, and a mismatch is
  discarded rather than kept.
- A later pull trusts size + the recorded checksum. To re-hash everything on disk
  (catches same-size corruption or hand-edited files), use `--verify`.
- `--prune` deletes local files that no longer exist on the server.

---

## Workspace layout

```
~/.evaldash/                        # or $EVALDASH_HOME
  config.json                       # server URL + token (0600)
  workspace/<run>/                  # data root handed to the bbox API
    .evaldash_manifest.json         # what this client downloaded, for diffing
    metadata.yaml
    devops/
      current.parquet
      .export_prebake/<route>/*.json.gz
      <suite>/<scenario>/scenario.yaml
  cache/                            # DuckDB query caches, regenerable
  t4/<dataset_id>/<scenario>/       # cached 3D scenes
    manifest.json  page.html  meta.json
    frames/<i>.bin  frames/<i>.hdr.json
    lanelet/<i>.json  overlay/<i>__<hash>.json  caminfo/<i>__<hash>.json
```

Uninstalling is `rm -rf ~/.evaldash` plus the executable.

---

## 3D point clouds

Two different things get called "3D" here.

**Bounding-box geometry** — what `local_bbox_viewer.html` draws — comes entirely from
the parquet, in base_link frame, rendered by bespoke canvas/SVG code with no three.js and
no external assets. This travels with the `minimal` tier and works fully offline.

**Point clouds** come from `t4-server`, a separate FastAPI service in a different repo
(`evaluator_result_parser`) that reads T4 datasets from a shared mount. The dashboard has
never proxied it — the browser fetches `frame.bin` from it directly, which is why
`T4_VISUALIZER_BROWSER_BASE_URL` exists.

The client can either talk to it live, or cache a scenario for offline use:

```bash
evaldash-local login --server https://dash --token <t> --t4-base-url http://t4host:8000
evaldash-local t4 scenarios <dataset_id>            # what is available
evaldash-local t4 fetch <dataset_id> --scenario <name>
evaldash-local t4 fetch <dataset_id> --scenario <name> --frames 0-49   # a slice
evaldash-local t4 ls                                # what is cached
evaldash-local t4 rm <dataset_id> <scenario>
```

`fetch` sizes the scene from one frame and asks before committing, because the numbers are
large: frames are `float32[4]` per point with no decimation option, so one frame is
roughly 1.6–3.2 MB and a 100-frame scene is 200–300 MB. Fetching is resumable and
incremental — an already-cached frame is skipped unless you pass `--force` — and
`--no-camera` / `--no-lanelet` trim what is stored.

Once cached, the client serves the same `/viewer/three*` paths locally, byte-for-byte,
including the `X-T4V-*` headers. Because every URL the viewer page fetches is
root-relative, the page works unchanged. A request for something that was never fetched
returns a 404 naming the exact command to run rather than silently rendering an empty
scene.

Cached scenes live in `~/.evaldash/t4/<dataset_id>/<scenario>/`. Frame-window queries are
computed from the cached frame count rather than stored, so scrubbing works offline.

---

## Environment variables

| Variable | Side | Purpose |
|---|---|---|
| `EVAL_EXPORT_TOKEN` | server | Enables the export routes. Unset = closed. |
| `EVAL_DASHBOARD_DATA_ROOT` | both | Run directory root. |
| `EVAL_BBOX_CACHE_DIR` | both | Where derived caches go. |
| `EVAL_APP_ROOT` | both | Where `static/` lives. Auto-detected; rarely needed. |
| `EVAL_LIB_PATHS` | server | `os.pathsep` list of evaluator library paths for unpickling. |
| `EVAL_PREBAKE_READ` | both | `0` forces the live pickle path, ignoring pre-baked answers. |
| `EVALDASH_HOME` | client | Workspace location (default `~/.evaldash`). |
| `EVALDASH_SERVER` / `EVALDASH_TOKEN` | client | Override stored connection settings. |
| `EVALDASH_FORCE_BROWSER` | client | `1` skips the native window. |

---

## Notes and limits

- The client's own HTTP server binds `127.0.0.1` and deliberately leaves the export
  routes closed, so it is not a file server for your laptop.
- `pull` never writes outside the workspace; the server confines every request to the
  resolved run directory on top of its data-root sandbox.
- `raw` tier downloads pickles, and reading them requires the evaluator libraries
  (`EVAL_LIB_PATHS`) plus `pickle.load` on downloaded files. Prefer pre-baking.
- Pulls of a run whose parquet the server later regenerates will re-transfer that
  parquet — it is one file, and its checksum changed.
