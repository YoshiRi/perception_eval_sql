---
name: trend-report
description: Summarize or analyze release/trend history from the evaluation dashboard — metric trends across releases, latest-vs-previous comparisons, pass-rate history. Use when asked "how did metrics change", "trend of mAP", "compare the last two releases' summary numbers", or for report material.
---

# Trend report

The trends API returns exactly what the dashboard's Trend Insights page charts: one
entry per release with metadata and summary-derived metrics.

```bash
python scripts/evalctl.py trends --limit 20                 # compact table
python scripts/evalctl.py trends --json --limit 20          # full structured data
python scripts/evalctl.py trends --topic obstacle --json
python scripts/evalctl.py trends --cases --json             # + devops per-case rows
python scripts/evalctl.py trends --summaries --json         # + raw summary payloads (big)
```

Per release you get: `version`, `date`, `description`, `data_count`, `roles`, and
`metrics` — full-dataset `mAP`/`precision`/`recall`/error metrics, `usecase_*`
planning metrics, devops `overall_pass_rate` + `scenario_count`, and `recall_bands`
(0.2.0+ releases only; older ones legitimately lack them — say "not recorded", don't
treat as zero).

## Writing the report

- Compare like with like: filter to one `topic` before trending, and mention
  `data_count` when it changed between releases (metric shifts may be dataset shifts).
- `metrics.errors` on an entry means that summary could not be parsed — exclude it
  from numeric trends and note it.
- Lead with the delta of the newest release vs the previous one, then the longer
  trend. Absolute values without the previous release as context are not useful.
- For visual output, build a chart/HTML artifact from the `--json` data rather than
  screenshotting the dashboard.
