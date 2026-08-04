# evaldash-local

A local client for the perception evaluation dashboard. It downloads already-generated
result files from a running dashboard server into a workspace on your machine, then
serves the **existing** bbox explorer and viewer against them — so you can dig into
scenario details offline, without the VPN, the server, or a Streamlit session.

It reuses the dashboard's own code: the same `backend/local_bbox_api.py`, the same
`static/` renderers, the same DuckDB queries. The client adds a transport (pull files),
a workspace, and a window.

---

## Quick start (from a fresh clone)

You need Python 3.10+ and the URL of the dashboard. **No token in most deployments** —
access reuses the dashboard's own identity model, so if you can reach the dashboard you
can use the client.

```bash
git clone <repo> && cd <repo>/evaluation_dashboard_app

python3 -m venv .venv && source .venv/bin/activate
pip install -r client/requirements.txt          # duckdb, pandas, numpy, PyYAML

python3 -m client doctor                        # sanity check; should exit 0
python3 -m client open                          # opens the home page in your browser
```

On the home page: enter the dashboard URL, press **Connect**, pick a run, press
**Download**, then **Open**. Nothing else is required — no Streamlit, no Docker, no
database, and normally no token.

**Want a real application window instead of a browser tab, and a double-clickable file?**

```bash
pip install pywebview pyinstaller
sudo apt install gir1.2-webkit2-4.0 python3-gi   # Linux: the native webview
./client/build_app.sh --server http://your-dashboard    # -> dist/evaldash-local
```

Passing `--server` bakes the URL into the executable, so the app connects on launch and
the recipient types nothing at all. Combined with identity-based access, double-clicking
is the entire setup.

Then double-click `dist/evaldash-local`. That single file is self-contained: it needs
neither the repo nor a Python install, so it is the thing to hand to a teammate. The same
executable is also the CLI, via `dist/evaldash-local --cli <command>`.

Everything the client writes lives in `~/.evaldash/`; uninstalling is deleting that
directory and the executable.

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

**Usually nothing to do.** The export routes reuse the dashboard's own access model
rather than adding a second secret:

| How the request arrives | Result |
|---|---|
| Direct hit (LAN, container port) | **Allowed.** Anyone who can reach this port can already list run files via `/api/parquets` and query them through the existing routes, so a separate secret would be friction without a boundary. |
| Via Cloudflare, authenticated | **Allowed**, and the identity is recorded. |
| Via Cloudflare, no identity | **Refused.** Failing closed here is what stops the edge being bypassed by setting one header. |
| Valid `EVAL_EXPORT_TOKEN` presented | **Allowed.** Useful for automation. |

A forged `Cf-Access-Authenticated-User-Email` on a direct hit is ignored: `lib/auth.py`
only believes that header alongside the other `Cf-*` signals, and these routes reuse that
rule rather than reimplementing it.

Two knobs tighten this if exposure changes — both need a container **recreate**, not a
restart:

- `EVAL_EXPORT_REQUIRE_TOKEN=1` — demand a token from everyone.
- `EVAL_EXPORT_ALLOW_DIRECT=0` — refuse non-Cloudflare requests.

To issue a token for the first, or for automation:

```bash
./deploy/11_ENABLE_EXPORT_API.sh --apply     # appends EVAL_EXPORT_TOKEN to deploy/.env
cd deploy && docker compose --env-file .env up -d --no-build streamlit1
```

Two traps worth knowing, both of which cost time to discover:

* **`deploy/.env` is the only env file that reaches the containers.** Compose is invoked
  as `docker compose --env-file .env` (`deploy/04_START.sh:34`). `deploy/.env.local` is
  only a naming convention mentioned in a `.env.example` comment — nothing loads it, so a
  token written there silently has no effect.
* **A restart is not enough.** `docker compose restart` (what
  `10_RESTART_STREAMLIT.sh` does) reuses the existing container config and never
  re-reads `env_file`. A new variable needs `up -d`, which recreates the container.

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

A wrong token is rejected outright (`401`) rather than falling back to the identity
rules — presenting a bad credential is an error, not something to paper over.

The workflow routes (`/api/workflow_*`, `backend/workflow_api.py`) are mounted next to the
export routes and authorize through exactly the same table, so nothing extra is needed to
let a client start runs. They additionally need the task queue the dashboard already uses
(`USE_TASK_QUEUE`, `DATABASE_URL`, `REDIS_URL`, a worker); where that is absent, starting
is refused with that reason and only the read routes answer. Whoever can download runs can
start them, so if that is not the intent, `EVAL_EXPORT_REQUIRE_TOKEN=1` gates both.

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

### Starting workflows

Downloading covers runs that already exist. The **Workflow** page (`/workflow`, linked
from the home page) and the `workflow` subcommands start new ones: the request goes to the
dashboard, which queues the same worker job the dashboard's own Workflow page would.

```bash
# which catalogs are available (--refresh also asks the evaluator API)
evaldash-local workflow catalogs

# start a perception run; the preset name resolves the catalog and integration ids
evaldash-local workflow start --target beta/v4.3.2 --catalog "Performance Test"

# see the exact parameters the worker would get, without queueing anything
evaldash-local workflow start --target beta/v4.3.2 --catalog "Performance Test" --dry-run

# traffic-light recognition, and the release spec sheet
evaldash-local workflow start --kind tlr --target beta/v4.3.2 --catalog "J6Gen2_TLR_Regression"
evaldash-local workflow start --kind release --target beta/v4.3.2 --metadata-file trend.yaml

# watch and control
evaldash-local workflow list
evaldash-local workflow status <task-id>       # exits 1 if the task failed
evaldash-local workflow logs <task-id> -n 50
evaldash-local workflow cancel <task-id>
```

`status` and `logs` accept the short id printed by `list`. Task ids and run folders line
up with `pull`, so a finished workflow is fetched with
`evaldash-local pull $(evaldash-local workflow status <id> --json | jq -r .run_name)`.

The server needs `USE_TASK_QUEUE=true`, `DATABASE_URL`, `REDIS_URL` and a running worker
for any of this; without them `workflow list` still works and `start` says so plainly.
Authorization is the same as for downloads — no separate token — and the run is attributed
to the Cloudflare identity that started it, so it shows up as yours in the dashboard.

### Which server it talks to

Three sources, highest priority first:

| Source | Set by | Persists? |
|---|---|---|
| `EVALDASH_SERVER` env | the shell, per invocation | no — a transient override, never written to disk |
| Saved setting | the home page's **Connect**, or `login --server` | yes, `~/.evaldash/config.json` (mode 0600) |
| Build default | `build_app.sh --server URL` | baked into the executable |

The environment wins deliberately: `connect()` caches the resolved build default on
first use, so nearly every install ends up with a saved URL, and an override that lost
to it would be useless.

Change it any time from the home page, or:

```bash
evaldash-local login --server https://other-dashboard   # change and save
evaldash-local login --reset                            # forget it, fall back to the default
evaldash-local doctor                                   # shows the URL and where it came from
```

**Reset matters more than it sounds.** A saved URL shadows the baked-in one, so
rebuilding the app with a different `--server` would otherwise keep talking to the old
host. Reset (button on the home page, or `--reset`) clears the saved server and token;
downloaded runs are untouched.

A failed `login` leaves the previous setting intact and exits 1, so a typo cannot strand
the app.

### Behind Cloudflare Access

Access refuses unauthenticated requests at the edge, before the dashboard's own auth is
ever consulted, so the export token does nothing for it. Two credentials work.

**Browser sign-in** — nothing to store, right for a person's laptop:

```bash
evaldash-local login --server https://dash.example.com --cf-login
```

That runs `cloudflared access login` (install `cloudflared` first) and the session it
mints is cached by `cloudflared` under `~/.cloudflared`. Every later command reads it
back automatically and sends it as a `CF_Authorization` cookie — including resumed
downloads, which keep their HTTP `Range` behaviour through the tunnel. The session
expires on the Access policy's schedule, typically daily; re-run the same command.

**Service token** — no browser, right for a shared or unattended machine. It is sent as
`CF-Access-Client-Id` / `CF-Access-Client-Secret`, mirroring how the dashboard
authenticates its own calls to the T4 visualizer, and takes precedence over any browser
session:

```bash
evaldash-local login --server https://dash.example.com --token <export-token> \
  --cf-client-id <id> --cf-client-secret <secret>
```

When neither is available the client says so in those words — naming Cloudflare Access
and the two ways to fix it — instead of failing on a JSON parse error against a sign-in
page. Note the API is mounted at `/bbox-api` behind nginx; `login` probes for that and
saves the resolved URL, so a bare hostname is fine to type.

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

The home page's **3D point clouds & cameras** card does all of this without a terminal:
save the t4-server URL (with a reachability check), list a dataset's scenarios, see the
size before committing — the download button only appears after the estimate — fetch
with a progress bar, and open, resume or delete cached scenes. The CLI equivalents:

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

## Troubleshooting

**`Failed to load module "canberra-gtk-module"`** — harmless, and the window works. GTK
inside the PyInstaller bundle cannot see the system module directory, so it reports the
optional sound-event module as failed. The package is usually installed system-wide
already; the bundled GTK just does not look there. Silencing it would mean redirecting
file descriptor 2, which would hide real errors too, so the launcher prints a note above
the lines instead.

**The window opens but the run list is empty** — nothing has been downloaded yet. Enter
the server URL and token on the home page and click Download.

**"Export API is disabled on this server" (HTTP 503)** — `EVAL_EXPORT_TOKEN` is not set
*inside the container*. Check with:

```bash
cd deploy && docker compose --env-file .env exec streamlit1 sh -c 'echo $EVAL_EXPORT_TOKEN'
```

If it is empty but present in `deploy/.env`, the container predates the variable:
recreate it (`up -d`), do not merely restart it.

**DevOps criteria panels say "unavailable"** — that run has not been pre-baked. Check
coverage with `python3 -m backend.prebake_cli --report`.

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
