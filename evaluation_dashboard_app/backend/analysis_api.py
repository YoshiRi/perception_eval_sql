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
    tier = _tier_dir(run_dir, role)
    source = _pick_parquet(tier)
    create_view_eval_flat(con, str(source), view, exclude_polygons=exclude_polygons)
    return {"run": run_dir.name, "role": role, "parquet": source.name, "tier_dir": str(tier)}


# --------------------------------------------------------------------- prediction


_PREDICTION_TABLES = {"label_summary": "Prediction label summary",
                      "distance_summary": "Prediction distance summary"}


def _prediction_source(tier_dir: Path) -> Path | None:
    for name in ("future.parquet", "future.csv"):
        candidate = tier_dir / name
        if candidate.is_file():
            return candidate
    return None


def _load_prediction_summaries(tier_dir: Path) -> dict[str, Any]:
    """minADE/minFDE summary tables for a run tier, cache-first.

    Reads the Prediction Evaluation page's artifact cache when it is fresh; otherwise
    computes from future.parquet with the same specsheet-aligned builder. Returns
    ``{"tables": {...}}`` or ``{"error": reason}`` -- prediction data is an optional
    bonus in a detection package and must never sink it.
    """
    import json

    import pandas as pd

    source = _prediction_source(tier_dir)
    if source is None:
        return {}
    cache_dir = tier_dir / ".dashboard_cache" / "prediction_eval_cache"
    manifest_path = cache_dir / "manifest.json"
    try:
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if int(manifest.get("future_mtime_ns", -1)) == source.stat().st_mtime_ns:
                tables = {
                    title: pd.read_parquet(cache_dir / f"{name}.parquet")
                    for name, title in _PREDICTION_TABLES.items()
                    if (cache_dir / f"{name}.parquet").is_file()
                }
                if tables:
                    return {"tables": tables}
        from lib.prediction_eval import build_specsheet_aligned_prediction_artifacts

        if source.suffix == ".parquet":
            future_df = pd.read_parquet(source)
        else:
            future_df = pd.read_csv(source)
        artifacts = build_specsheet_aligned_prediction_artifacts(future_df)
        return {"tables": {
            title: artifacts[name] for name, title in _PREDICTION_TABLES.items()
            if isinstance(artifacts.get(name), pd.DataFrame)
        }}
    except Exception as exc:
        return {"error": str(exc)}


def _merge_prediction(tables: dict[str, Any], metadata: dict[str, Any],
                      tier_dir: str, *, suffix: str = "") -> None:
    result = _load_prediction_summaries(Path(tier_dir))
    for title, frame in (result.get("tables") or {}).items():
        tables[f"{title}{suffix}"] = frame
    if result.get("error"):
        metadata.setdefault("prediction_errors", {})[suffix.strip(" -") or "run"] = result["error"]


# ---------------------------------------------------------------------------- TLR


def _resolve_tlr_dir(path_text: str) -> Path:
    """A TLR result dir may be nested (run/role or deeper); confine it to the data root."""
    text = str(path_text or "").strip().replace("\\", "/").lstrip("/")
    if not text:
        raise AnalysisError("A TLR path (relative to the data root) is required.")
    root = export_api._data_root()
    candidate = (root / text).resolve()
    if candidate != root and root not in candidate.parents:
        raise AnalysisError(f"Path is outside the data root: {path_text}")
    if not candidate.is_dir():
        raise AnalysisError(f"No such directory: {path_text}")
    return candidate


def _build_tlr_package(payload: dict[str, Any], mode: str) -> tuple[bytes, str]:
    from lib.tlr_llm_package import (
        build_tlr_compare_tables,
        build_tlr_llm_analysis_package,
        build_tlr_llm_analysis_tables,
        load_tlr_analyzer,
    )

    def _load(field: str) -> tuple[Any, Path]:
        path = _resolve_tlr_dir(str(payload.get(field) or ""))
        try:
            return load_tlr_analyzer(str(path)), path
        except ValueError as exc:
            raise AnalysisError(str(exc)) from exc

    if mode == "single":
        analyzer, path = _load("run")
        tables = build_tlr_llm_analysis_tables(analyzer)
        metadata: dict[str, Any] = {
            "mode": "single", "kind": "tlr",
            "scope": {"path": path.name, "stats": analyzer.get_summary_stats()},
        }
        filename = f"tlr_analysis_{path.name}.zip"
    elif mode == "compare":
        base, base_path = _load("base_run")
        candidate, candidate_path = _load("candidate_run")
        if base_path == candidate_path:
            raise AnalysisError("base_run and candidate_run are the same directory.")
        tables = build_tlr_compare_tables(base, candidate)
        metadata = {
            "mode": "compare", "kind": "tlr",
            "comparison": {
                "base": {"path": base_path.name, "stats": base.get_summary_stats()},
                "candidate": {"path": candidate_path.name,
                              "stats": candidate.get_summary_stats()},
            },
        }
        filename = f"tlr_compare_{base_path.name}_vs_{candidate_path.name}.zip"
    else:
        raise AnalysisError(f"Unknown mode '{mode}'. Expected single or compare.")
    return build_tlr_llm_analysis_package(tables=tables, metadata=metadata), filename


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
    kind = str(payload.get("kind") or "detection").lower()
    if kind == "tlr":
        return _build_tlr_package(payload, mode)
    if kind != "detection":
        raise AnalysisError(f"Unknown kind '{kind}'. Expected detection or tlr.")
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
            _merge_prediction(tables, metadata, source.pop("tier_dir"))
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
            _merge_prediction(tables, metadata, base.pop("tier_dir"), suffix=" - base")
            _merge_prediction(tables, metadata, candidate.pop("tier_dir"), suffix=" - candidate")
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
