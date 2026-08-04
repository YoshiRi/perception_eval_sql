# Running evaluation tasks by talking to Claude Code / Codex

The dashboard's workflows (evaluator runs, release spec-sheets, trend queries,
detection analysis) can be driven entirely by a coding agent. You describe what you
want in plain language; the agent runs `scripts/evalctl.py` against the dashboard
server and reports back. This page is the phrasebook: what you can ask for, what
happens, and what the agent will ask you in return.

## One-time setup

Open a terminal (or ask the agent to check for you):

```bash
export EVAL_DASHBOARD_URL=http://<server>:<api-port>   # backend API, not the Streamlit UI port
export EVAL_EXPORT_TOKEN=<token>                       # only if the server requires one
python scripts/evalctl.py doctor
```

Or just say: **"check the eval server connection"** — the agent runs the doctor and
explains any failure (wrong URL, missing token, queue disabled, worker down).

## What you can say

### Evaluate a branch

> "Evaluate branch beta/v4.5.0"
> "Run a perception eval on my branch, TLR kind"

The agent validates the branch actually exists in the pilot-auto repo (typos are
caught in seconds instead of failing hours later inside the evaluator), starts the
workflow, and gives you the task id and run name. Runs take hours — you don't need
to keep the session open.

### Release a spec-sheet

> "Release the specsheet for v4.5.0"
> "Run a release for beta/v4.5.0, description: Q3 milestone release"

The agent drafts the trend metadata for you — today's date (JST, `YYYY.MM.DD`),
`Pilot.Auto vX.Y.Z` from the branch name, and `release_group` / `data_count` /
`topic_name` carried forward from the most recent past release — then **shows it to
you for approval before starting**. Correct anything in plain language ("data count
should be 180", "new release group 2026Q3"). It never starts a release without your
confirmation.

### Check on tasks

> "How is my eval going?" / "Did the release finish?"
> "Why did task 3f2a… fail?"
> "Cancel that run"

For failures the agent reads the task log and diagnoses (branch built but evaluator
phase failed, queue congestion, download errors…) rather than just saying "failed".
If evaluator jobs already succeeded before a later step broke, it can restart the
release reusing those job ids instead of re-running hours of simulation.

### Analyze results

> "Analyze run eval_beta_v4_5_0_20260804"
> "Compare my branch's run against the latest release run — did it regress?"

The agent pulls the same curated evidence package the Detection Stats page builds
(class metrics, scene hotspots, FN frames, distance-band rates; degradations and FP
diffs for comparisons) and writes the analysis report itself, verdict first.

### Trend history

> "How has mAP trended over the last 10 releases?"
> "Compare the last two releases' pass rates"
> "Make a chart of recall by release for the obstacle topic"

Backed by the trends API — the same data as the Trend Insights page, so numbers
match the dashboard exactly.

## The usual chain

The commands compose. A typical end-to-end request:

> "Evaluate branch beta/v4.6.0, and when it's done, compare it against the latest
> release and tell me if anything regressed."

## Ground rules the agents follow

- **Releases always show you the metadata first.** No `--yes` without your approval.
- **Long runs are fire-and-forget**: the agent reports the task id and returns; ask
  again later for status.
- A refused start with "was not found" means the branch/tag genuinely doesn't exist
  — the agent will try the `beta/` prefix or tag form before bothering you.
- Everything the agent does is also available to you directly:
  `python scripts/evalctl.py --help`, or raw HTTP (`/api/workflow_*`,
  `/api/analysis_package`) for CI jobs.

## Where the agent instructions live

- `.claude/skills/` — per-task guidance for Claude Code (eval-branch,
  release-specsheet, workflow-status, trend-report, analyze-run, eval-setup)
- `AGENTS.md` — condensed rules for Codex and other agents
- `scripts/evalctl.py` — the CLI both of them drive
