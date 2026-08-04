# Agent guide: driving the evaluation dashboard

This repo ships `scripts/evalctl.py`, a stdlib-only CLI that drives the dashboard  
server's workflow/export API. Any coding agent (Codex, Claude Code, etc.) should use  
it instead of calling the HTTP API by hand. Claude Code users get richer guidance via  
the skills in `.claude/skills/`; this file is the condensed version.

Setup: `EVAL_DASHBOARD_URL` (backend API base URL) and, if the server demands one,
`EVAL_EXPORT_TOKEN`. Verify with `python scripts/evalctl.py doctor`. If the server is
behind Cloudflare Access, `evalctl` signs in on its own — see the Cloudflare rule
below.

Human-facing companion doc (what users can ask for, in plain language):
`docs/AGENT_TASKS.md`.

**Before doing one of these tasks, read the matching playbook** — they hold the
judgment this file omits (failure-log patterns, metadata sanity checks, report
structure), and they are plain markdown, not Claude-specific:

| Task | Read first |
|---|---|
| evaluate a branch | `.claude/skills/eval-branch/SKILL.md` |
| release spec-sheet | `.claude/skills/release-specsheet/SKILL.md` |
| check/diagnose/cancel tasks | `.claude/skills/workflow-status/SKILL.md` |
| trend history / metric trends | `.claude/skills/trend-report/SKILL.md` |
| analyze or compare runs | `.claude/skills/analyze-run/SKILL.md` |
| connection/auth problems | `.claude/skills/eval-setup/SKILL.md` |
| local app/client build/development | `.claude/skills/local-dashboard-dev/SKILL.md` |

## Commands

```bash
python scripts/evalctl.py doctor                      # connectivity/auth/queue/worker check
python scripts/evalctl.py login [--force]             # Cloudflare Access sign-in (browser)
python scripts/evalctl.py start <branch> [--kind tlr] [--catalog NAME] [--dry-run]
python scripts/evalctl.py release <branch> [--dry-run] [--yes] [--set key=value]
python scripts/evalctl.py status [task_id] [--log] [--watch] [--json]
python scripts/evalctl.py cancel <task_id>
python scripts/evalctl.py trends [--topic T] [--limit N] [--json]
python scripts/evalctl.py runs [--q substring]
python scripts/evalctl.py fetch <run> [--tier criteria]
python scripts/evalctl.py analyze <run>               # LLM analysis package, single run
python scripts/evalctl.py compare <base> <candidate>  # regression package, two runs
python scripts/evalctl.py analyze <path> --kind tlr   # TLR packages (compare works too)
python scripts/evalctl.py report <run> [--kind specsheet] [--candidate-run B]  # official PDFs
python scripts/evalctl.py triage <task_id>            # structured failure root-cause bundle
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
- **Cloudflare Access**: a `302` to `*.cloudflareaccess.com` or `HTTP 403: error code:
1010` is the edge, not the app — `EVAL_EXPORT_TOKEN` is irrelevant to it. `evalctl`
recovers by itself: service token (`CF_ACCESS_CLIENT_ID`/`CF_ACCESS_CLIENT_SECRET`)
if set, else a cached `cloudflared` session, else an automatic
`cloudflared access login`. Since that last step needs a human at a browser, ask the
user to run `python scripts/evalctl.py login` rather than triggering it blind; in an
unattended context use a service token, or `--no-cf-login` to fail fast instead of
hanging. A `1010` that persists *after* a valid sign-in is the browser-integrity check
rejecting the HTTP client, not the user — do not retry the login. Details:
`.claude/skills/eval-setup/SKILL.md`.
- Behind nginx / a Cloudflare tunnel the API is mounted at `/bbox-api`, not the host
root; `evalctl` finds it after a 404/405 and prints the `EVAL_DASHBOARD_URL` to make
permanent.

