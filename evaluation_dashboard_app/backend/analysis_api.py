"""Analysis-package export: the Detection Stats LLM evidence ZIP over HTTP.

The Detection Stats page can bundle curated evidence tables (class metrics, scene
hotspots, FN frames, distance-band rates; for comparisons the degradation/FP-diff
analyses) with instructions for an LLM. These routes build the identical package
server-side so a coding agent can pull it with one call instead of a human clicking
through Streamlit. Mounted and authorized like the export routes; lib/ imports stay
lazy for the packaged client, which ships backend/ without lib/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

try:
    from backend import export_api
except ImportError:  # pragma: no cover - frozen client puts backend/ on sys.path
    import export_api  # type: ignore[no-redef]


class AnalysisError(export_api.ExportError):
    """An analysis-package request that cannot be honoured as sent."""


_FILTER_KEYS = ("topic_name", "label", "suites", "visibility")


def _tier_dir(run_dir: Path, role: str) -> Path:
    """The directory whose parquet the page would analyze for this run+role."""
    candidate = run_dir / role
    if candidate.is_dir():
        return candidate
    # Some runs (local evals) keep parquet at the top level instead of role tiers.
    if any(run_dir.glob("*.parquet")):
        return run_dir
    raise AnalysisError(
        f"Run '{run_dir.name}' has no '{role}' tier and no top-level parquet."
    )


def _pick_parquet(tier_dir: Path) -> Path:
    from lib.detection_eval_sql import default_parquet_index, list_parquets_in_run

    parquets = list_parquets_in_run(tier_dir)
    if not parquets:
        raise AnalysisError(f"No parquet files under {tier_dir.name}/.")
    return Path(parquets[default_parquet_index(parquets)])


def _filters(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("filters")
    if raw in (None, ""):
        return {}
    if not isinstance(raw, dict):
        raise AnalysisError("filters must be an object, e.g. {\"label\": [\"car\"]}.")
    unknown = set(raw) - set(_FILTER_KEYS)
    if unknown:
        raise AnalysisError(
            f"Unknown filter(s): {', '.join(sorted(unknown))}. Known: {', '.join(_FILTER_KEYS)}"
        )
    return dict(raw)


def _prepare_view(con: Any, run_name: str, role: str, view: str,
                  *, exclude_polygons: bool) -> dict[str, Any]:
    """Resolve one run to an eval_flat view; returns source info for the metadata."""
    from lib.detection_eval_sql import create_view_eval_flat

    run_dir = export_api._resolve_run(run_name)
    source = _pick_parquet(_tier_dir(run_dir, role))
    create_view_eval_flat(con, str(source), view, exclude_polygons=exclude_polygons)
    return {"run": run_dir.name, "role": role, "parquet": source.name}


def build_analysis_package_bytes(payload: dict[str, Any]) -> tuple[bytes, str]:
    """Build the ZIP the page's "Prepare LLM report ZIP" button builds. Returns
    ``(zip_bytes, filename)``. Split from the HTTP handler so tests and other
    callers can use it without a socket."""
    import duckdb

    from lib.detection_eval_sql import build_filter_clause, kpi_row_for_view
    from lib.detection_llm_package import (
        build_compare_llm_analysis_tables,
        build_llm_analysis_package,
        build_single_llm_analysis_tables,
    )

    mode = str(payload.get("mode") or "single").lower()
    role = str(payload.get("role") or "performance")
    exclude_polygons = payload.get("exclude_polygons") is True
    # Matches the page's LLM-package call sites: distance filtering is disabled so the
    # distance-band tables keep every band.
    filter_clause = build_filter_clause(_filters(payload), enable_dist_h=False)

    con = duckdb.connect()
    try:
        if mode == "single":
            run_name = str(payload.get("run") or "")
            source = _prepare_view(con, run_name, role, "view_eval_flat",
                                   exclude_polygons=exclude_polygons)
            tables = build_single_llm_analysis_tables(
                con, view="view_eval_flat", filter_clause=filter_clause
            )
            metadata: dict[str, Any] = {
                "mode": "single",
                "scope": source,
                "filters": _filters(payload),
                "kpis": {source["run"]: kpi_row_for_view(con, "view_eval_flat", filter_clause)},
            }
            filename = f"analysis_{source['run']}.zip"
        elif mode == "compare":
            base_name = str(payload.get("base_run") or "")
            candidate_name = str(payload.get("candidate_run") or "")
            base = _prepare_view(con, base_name, role, "view_base",
                                 exclude_polygons=exclude_polygons)
            candidate = _prepare_view(con, candidate_name, role, "view_candidate",
                                      exclude_polygons=exclude_polygons)
            if base["run"] == candidate["run"]:
                raise AnalysisError("base_run and candidate_run are the same run.")
            tables = build_compare_llm_analysis_tables(
                con,
                base_view="view_base",
                candidate_view="view_candidate",
                base_filter=filter_clause,
                candidate_filter=filter_clause,
            )
            metadata = {
                "mode": "compare",
                "comparison": {"base": base, "candidate": candidate},
                "filters": _filters(payload),
                "kpis": {
                    base["run"]: kpi_row_for_view(con, "view_base", filter_clause),
                    candidate["run"]: kpi_row_for_view(con, "view_candidate", filter_clause),
                },
            }
            filename = f"compare_{base['run']}_vs_{candidate['run']}.zip"
        else:
            raise AnalysisError(f"Unknown mode '{mode}'. Expected single or compare.")
    finally:
        con.close()
    return build_llm_analysis_package(tables=tables, metadata=metadata), filename


def analysis_package(handler: Any, payload: dict[str, Any], *, head_only: bool = False) -> None:
    """Stream the analysis ZIP for one run (mode: single) or two (mode: compare)."""
    export_api.require_auth(handler)
    data, filename = build_analysis_package_bytes(payload)
    handler.send_response(200)
    handler.send_header("Content-Type", "application/zip")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Content-Disposition", f'attachment; filename="{filename}"')
    # Same contract as export_file: once headers are sent, the dispatcher must not
    # overwrite the response with a JSON error.
    handler._export_stream_started = True
    handler.end_headers()
    if not head_only:
        handler.wfile.write(data)


STREAM_ROUTES: dict[str, Callable[..., None]] = {
    "/api/analysis_package": analysis_package,
}
