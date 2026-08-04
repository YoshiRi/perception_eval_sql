---
name: release-specsheet
description: Start the release spec-sheet workflow with auto-filled trend metadata (today's date, version from the branch name, conventions carried forward from past releases). Use when asked to "release the specsheet", "run a release for ...", or "make the release spec sheet".
---

# Release spec-sheet workflow

This is the expensive one (schedules evaluator jobs, builds the spec-sheet PDF, adds a
point to trend history), so metadata correctness matters and the user confirms before
it starts.

## Metadata rules (how past releases wrote it)

- `date`: today, JST, dotted format `YYYY.MM.DD`. `evalctl` fills this automatically.
- `pilot_auto_version`: `Pilot.Auto vX.Y.Z`, derived from the branch/tag name.
- `release_group`, `data_count`, `topic_name`: carried forward from the newest past
  release via the trends API. Before trusting the autofill, LOOK at recent history:

  ```bash
  python scripts/evalctl.py trends --limit 5
  ```

  and check the carried-forward values still make sense for this release (a new
  quarter means a new `release_group`; `data_count` changes when the dataset grew).
- `description`: one short line saying what this release is. Ask the user if they
  did not say.

## Steps

1. Dry-run first and show the user the metadata + target check:

   ```bash
   python scripts/evalctl.py release beta/v4.5.0 --dry-run
   ```

2. Apply any user overrides (`--version`, `--release-group`, `--data-count`,
   `--topic`, `--description`, `--date`, `--set key=value`). A prepared YAML file
   bypasses autofill entirely: `--metadata-file meta.yaml`.
3. Start for real. The command prints the final metadata and asks for confirmation;
   pass `--yes` only after the user has seen and approved the metadata:

   ```bash
   python scripts/evalctl.py release beta/v4.5.0 --description "..." --yes
   ```

4. Report the task id. Release runs take hours — don't `--watch` unless asked.
   To reuse already-finished evaluator jobs instead of scheduling new ones:
   `--performance-job-id` / `--devops-job-id`.

## After it finishes

Check the outcome with the workflow-status skill; the completed task's
`result_summary` carries the produced paths. The new release then appears in
`evalctl trends` and on the dashboard's Trend Insights page.
