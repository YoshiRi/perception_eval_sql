---
name: local-dashboard-dev
description: Run, build, validate, and troubleshoot this evaluation dashboard repository locally. Use when starting the Streamlit app, working on pages/backend/static assets, running or packaging the evaldash-local client, using client/build_app.sh, checking generated client files, running local smoke tests, or using Docker/deploy scripts for this app.
---

# Local dashboard development

Use this for repository development tasks. For server-side evaluation workflows
(starting evaluator jobs, releases, trends, analysis packages), use `scripts/evalctl.py`
and the task-specific eval skills instead.

## Agent setup command

When a user needs to hand an agent a single public-safe setup command, use:

```bash
mkdir -p ~/work && cd ~/work && \
  { [ -d perception_eval_sql/.git ] || git clone https://github.com/tier4/perception_eval_sql.git; } && \
  cd perception_eval_sql/evaluation_dashboard_app && \
  printf 'Repository ready. Read AGENTS.md before running dashboard tasks.\n'
```

This clones the public repo only when it is missing, avoids pulling over an existing
checkout, and lands the agent in the dashboard app directory.

Natural-language prompt for an agent:

> Check that the repository URL is `https://github.com/tier4/perception_eval_sql.git`.
> If the repo is not already cloned, clone it. Then enter
> `perception_eval_sql/evaluation_dashboard_app`, read `AGENTS.md`, and follow the
> matching `.claude/skills/` playbook for the task before making changes or running
> dashboard commands.

## Local Streamlit app

Start the dashboard from the repo root:

```bash
streamlit run Overview.py
```

Use this path for UI/page work in `Overview.py`, `pages/`, `lib/`, `backend/`, and
`static/`. Local Streamlit does not provide the same container-only controls as the
Docker deployment; pages that require container context should show guidance instead
of attempting host Docker operations.

For dependency setup, follow `Readme.en.md` / `Readme.md`. The core local packages are
listed in `requirements.txt`; Docker-specific dependencies are in
`requirements-docker.txt`.

## Local client from source

The local client is documented in `client/README.md`. It downloads generated run files
from a dashboard server into a local workspace and serves the existing bbox/TLR/static
viewer routes from `127.0.0.1`.

Fresh-source workflow:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r client/requirements.txt
python3 -m client doctor
python3 -m client open
```

Useful client commands:

```bash
python3 -m client --help
python3 -m client doctor
python3 -m client open
python3 -m client serve --port 8765
python3 -m client runs
python3 -m client pull <run_name> --tier criteria
python3 -m client t4 scenarios <dataset_id>
```

Everything the client writes should live under `~/.evaldash/` by default, or
`EVALDASH_HOME` when set.

## Build packaged client

Build from the repo root:

```bash
pip install pyinstaller
# Optional native window support:
pip install pywebview
./client/build_app.sh
```

Common build variants:

```bash
./client/build_app.sh --server <dashboard-url>
./client/build_app.sh --server <dashboard-url> --t4-url <t4-server-url>
./client/build_app.sh --desktop
```

The output is `dist/evaldash-local`. The build script performs a real smoke test:
it starts the packaged client server, checks `/api/health`, static viewer assets,
`/api/client/state`, `/workflow`, `/api/parquets`, and `/api/tlr_dirs`.

Important generated file rule:

- `client/build_app.sh --server ...` or `--t4-url ...` writes `client/_defaults.py`.
  It is build-local generated state. Do not commit it unless the user explicitly
  asks for a committed default. Check `git status --short` after packaging.
- Running `client/build_app.sh` without baked defaults removes `client/_defaults.py`.
  If that file is tracked or intentionally present, inspect before accepting the
  change.

## Docker deployment scripts

For local or server Docker deployment, prefer the numbered scripts in `deploy/`:

```bash
cd deploy
./01_SETUP_ENV.sh
./02_BUILD.sh
./03_INIT_DB.sh
./04_START.sh
./06_STATUS.sh
./07_LOGS.sh
./09_RESTART_WORKER.sh
./10_RESTART_STREAMLIT.sh
```

Operational rules:

- `deploy/04_START.sh` runs `docker compose --env-file .env up -d`, scales workers from
  `EVAL_COMPOSE_SCALE_WORKER` unless overridden, and starts
  `EVAL_COMPOSE_STREAMLIT_REPLICAS` (1..3) Streamlit replicas, removing any above that
  count and recreating nginx last.
- Streamlit replica count is set only via `EVAL_COMPOSE_STREAMLIT_REPLICAS` in
  `deploy/.env`. All scripts share `deploy/_compose_lib.sh`, which derives the compose
  profiles and the nginx upstream list from it. Never start a replica with a bare
  `docker compose --profile ...`: it then sits outside the scripts' view and keeps
  running a stale environment.
- `deploy/04_START.sh` checks for active queued/running tasks before restart; do not
  bypass with `--force` unless the user explicitly accepts that risk.
- `deploy/09_RESTART_WORKER.sh` is the narrow restart for worker/lib/backend changes.
  Use `--idle-only` when active work may be running.
- `deploy/10_RESTART_STREAMLIT.sh` recreates the Streamlit replicas (and nginx after
  them), so it does pick up new `.env` values. Workers and queued tasks are untouched.
- A hand-typed `docker compose restart` still never rereads `env_file`; use
  `docker compose --env-file .env up -d --no-build <service>` if you bypass the scripts.
- Never print full `.env` values, tokens, Cloudflare credentials, or client secrets.

## Export API and client server setup

For local-client export routes on a deployed dashboard, use the existing scripts:

```bash
cd deploy
./11_ENABLE_EXPORT_API.sh --apply
./12_VERIFY_EXPORT_API.sh <base-url> [token] [cf-client-id] [cf-client-secret]
```

Rules:

- `deploy/.env` is the env file compose actually uses.
- `deploy/.env.local` is not automatically loaded.
- After `11_ENABLE_EXPORT_API.sh --apply`, recreate the Streamlit container; a
  restart is not enough.
- Use `client/README.md` for detailed Cloudflare Access and T4 client behavior.

## Verification choices

Choose checks by blast radius:

- Python/client logic: run focused tests such as `pytest tests/test_client_t4.py` or
  targeted `pytest` files.
- Source client sanity: `python3 -m client doctor`.
- Packaged client: `./client/build_app.sh` and rely on its smoke test.
- Streamlit UI changes: start `streamlit run Overview.py` and inspect the affected
  page when browser access is available.
- Docker/deploy changes: `cd deploy && ./06_STATUS.sh`, then targeted logs with
  `./07_LOGS.sh <service>`.

Before finishing, report which checks ran and whether any were skipped because they
needed network, Docker, Cloudflare credentials, a browser, or private dependencies.
