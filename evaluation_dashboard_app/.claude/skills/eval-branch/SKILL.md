---
name: eval-branch
description: Start a perception/TLR evaluator workflow for a git branch or tag on the evaluation dashboard server, validate the target first, and optionally watch it to completion. Use when asked to "evaluate branch X", "run the evaluator on ...", or "start a perception eval".
---

# Evaluate a branch

Everything goes through `scripts/evalctl.py` (stdlib-only). The server URL comes from
`EVAL_DASHBOARD_URL` and the token from `EVAL_EXPORT_TOKEN`; if they are not set, ask
the user or run the eval-setup skill first.

## Steps

1. **Preflight yourself only if something fails** — `evalctl start` already checks
   auth, queue, and worker liveness and refuses with the server's own reason.
2. Normalize the target: users say "evaluate v4.3.2" but branches are usually
   `beta/v4.3.2`. If unsure which exists, a dry run reports the `target_check`:

   ```bash
   python scripts/evalctl.py start beta/v4.3.2 --dry-run
   ```

   `target_check.exists: false` means the branch is not in the pilot-auto repo —
   try the other spelling (or `--tag` for tags) before bothering the user.
3. Start it. Catalog defaults to the first server preset; pick by name when the user
   asks for something specific (e.g. `--catalog "Performance Test"`):

   ```bash
   python scripts/evalctl.py start beta/v4.3.2 --description "why this run"
   ```

   For TLR: add `--kind tlr`.
4. Report the `task_id`, `run_name` and output path back to the user.
5. Only `--watch` (blocks, polls every 20 s) when the user explicitly wants to wait;
   evaluator runs take hours. Otherwise tell them to check later with the
   workflow-status skill.

## Failure handling

- "was not found" → the branch really doesn't exist; show the user the exact ref
  that was checked, suggest close matches if you can list them (`git ls-remote`).
- "No RQ worker is listening" → the server-side worker is down; do NOT `--force`
  unless the user says so — the task would sit pending.
- Auth errors → run `python scripts/evalctl.py doctor` and show the result.
