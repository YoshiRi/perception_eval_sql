# Agent guide: driving the evaluation dashboard

This repo ships `scripts/evalctl.py`, a stdlib-only CLI that drives the dashboard  
server's workflow/export API. Any coding agent (Codex, Claude Code, etc.) should use  
it instead of calling the HTTP API by hand. Claude Code users get richer guidance via  
the skills in `.claude/skills/`; this file is the condensed version.

Setup: `EVAL_DASHBOARD_URL` (backend API base URL) and, if the server demands one,
`EVAL_EXPORT_TOKEN`. Verify with `python scripts/evalctl.py doctor`.

Human-facing companion doc (what users can ask for, in plain language):
`docs/AGENT_TASKS.md`.

## Commands

```bash
python scripts/evalctl.py doctor                      # connectivity/auth/queue/worker check
python scripts/evalctl.py start <branch> [--kind tlr] [--catalog NAME] [--dry-run]
python scripts/evalctl.py release <branch> [--dry-run] [--yes] [--set key=value]
python scripts/evalctl.py status [task_id] [--log] [--watch] [--json]
python scripts/evalctl.py cancel <task_id>
python scripts/evalctl.py trends [--topic T] [--limit N] [--json]
python scripts/evalctl.py runs [--q substring]
python scripts/evalctl.py fetch <run> [--tier criteria]
python scripts/evalctl.py analyze <run>               # LLM analysis package, single run
python scripts/evalctl.py compare <base> <candidate>  # regression package, two runs
```

## Rules

- `start`/`release` preflight the server and validate the git target automatically;
a "was not found" refusal means the branch/tag really does not exist — try the
`beta/` prefix or `--tag` before asking the user.
- `release` is expensive and writes trend history. Always show the user the
auto-filled metadata (`--dry-run` first); `date` must be today JST as `YYYY.MM.DD`
(autofilled), `pilot_auto_version` like `Pilot.Auto v4.5.0`. Only pass `--yes`
after the user approved the metadata.
- Evaluator/release runs take hours: report the task id and return; don't block on
`--watch` unless asked.
- To analyze results: `analyze`/`compare` download a ZIP with `llm_instructions.md`,
`analysis_data_brief.md`, a report blueprint, and CSV evidence tables — read the
instructions first, then write the report yourself from the tables.
- On failures, read `status <id> --log` and diagnose before reporting; add `--json`
to any command for parseable output.

