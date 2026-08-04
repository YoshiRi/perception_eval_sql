---
name: analyze-run
description: Produce a detection-performance analysis report for one run, or a regression comparison between two runs (base vs candidate), from the dashboard's curated evidence tables. Use when asked to "analyze run X", "write a detection report", "compare run A vs run B", or "did branch X regress against the release".
---

# Analyze a run / compare two runs

The server prepares an **analysis package**: curated evidence tables (class metrics,
scene hotspots, FN frames, distance-band rates; for comparisons also per-object
degradations, FP diffs, critical cases, consecutive failures) plus instructions and a
report blueprint. You are the LLM those instructions are written for.

```bash
python scripts/evalctl.py runs --q <name>       # find exact run names first
python scripts/evalctl.py analyze <run>                      # single run
python scripts/evalctl.py compare <base_run> <candidate_run> # regression check
```

Both extract into `analysis_<run>/` or `compare_<base>_vs_<candidate>/` (choose
`--dest` in your scratchpad). `--role devops` switches tier; `--exclude-polygons`
matches the spec-sheet metric convention.

## Writing the report

1. Read `llm_instructions.md`, `analysis_data_brief.md`, and
   `recommended_report_blueprint.md` from the extracted package FIRST — they define
   the expected report structure and what each table means. Follow them.
2. Read the CSVs under `tables/` (schema in `manifest.json`).
3. Interpretation is your job: connect signals across tables (e.g. recall drop
   concentrated in a distance band + one scene hotspot), judge trade-offs, and be
   explicit about what is NOT degraded. Quote concrete numbers.
4. For a compare, the verdict comes first: regression / improvement / mixed, with
   the two or three decisive numbers, then the details.

Typical chain: after `eval-branch` finishes, compare its run against the newest
release run (find both with `evalctl runs`), and report the verdict.
