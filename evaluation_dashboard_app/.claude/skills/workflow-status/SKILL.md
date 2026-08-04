---
name: workflow-status
description: Check, watch, diagnose, or cancel evaluation-dashboard workflow tasks. Use when asked "how is my eval going", "did the release finish", "why did the workflow fail", or "cancel that run".
---

# Workflow status & diagnosis

```bash
python scripts/evalctl.py status                    # recent tasks, all users
python scripts/evalctl.py status --mine user@tier4.jp
python scripts/evalctl.py status <task_id>          # one task + result_summary
python scripts/evalctl.py status <task_id> --log    # + log tail (4000 chars)
python scripts/evalctl.py status <task_id> --watch  # poll until it finishes
python scripts/evalctl.py cancel <task_id>
```

## Diagnosing a failed task

Start with the structured triage bundle — it pre-extracts what you'd otherwise regex
out of the log:

```bash
python scripts/evalctl.py triage <task_id>          # human-readable
python scripts/evalctl.py --json triage <task_id>   # for parsing
```

It carries the evaluator job/build/test statuses, the failed cases with reasons, the
report/catalog/commit links, the error-mentioning log lines, and the log tail. Fall
back to `status <task_id> --log` only when you need more of the raw log. Common
patterns:

- Evaluator job rejected / build failure right at the start → the branch existed but
  does not build; the log names the failing phase. Point the user at the evaluator
  report URL if the log contains one.
- Long silence then timeout → evaluator-side queue congestion; the run can simply be
  restarted (same command the user used originally).
- Download/parquet errors after the evaluator succeeded → server-side disk or schema
  issue; the eval results still exist on the evaluator service, so a restart with
  `--performance-job-id <job>` (release) reuses them instead of re-running hours of
  simulation. Job ids are in the task's `result_summary` or log.
- Cancelled tasks are marked failed with "Cancelled by ..." — that's intentional.

Increase `--log-chars` when the tail isn't enough. Use `--json` on any command when
you want to parse the output.
