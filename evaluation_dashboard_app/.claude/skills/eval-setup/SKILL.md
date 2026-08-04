---
name: eval-setup
description: Configure and verify the connection to the evaluation dashboard server (URL, token, queue, workers). Use when evalctl commands fail with auth/connection errors, or when setting up a new machine to drive evaluations.
---

# Eval server setup & doctor

Two environment variables drive everything:

- `EVAL_DASHBOARD_URL` — e.g. `http://eval-server:8502` (the backend API port, not
  the Streamlit UI port)
- `EVAL_EXPORT_TOKEN` — only if the server demands a token (`doctor` will say)

## Verify

```bash
python scripts/evalctl.py doctor
```

Interpreting it:

- `UNREACHABLE` → wrong URL/port, VPN, or the server is down. Distinguish
  connection-refused (wrong port / not running) from timeout (network/VPN).
- `FAIL authorized` → the reason is printed. "requires an export token" → get the
  token value from the server's `EVAL_EXPORT_TOKEN` env (ask the server admin),
  export it locally, re-run doctor.
- `FAIL task queue` → server-side configuration (USE_TASK_QUEUE/DATABASE_URL/
  REDIS_URL); nothing to fix client-side — report it to whoever runs the server.
- `workers alive: 0` → workflows would queue but never run; the server's RQ worker
  container needs a restart.

## Persist

Once working, persist the two variables in the user's shell profile (ask before
editing their dotfiles), or prefix per-command. Never print the token back in full.
