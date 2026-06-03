from __future__ import annotations

import io
import html
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from lib.score_schema import (
    SCORE_BLOCK_SIZE,
    SCORE_NUM_COLS,
    SCORE_VIEW_METRIC_COLS,
    build_score_view,
    infer_score_criteria_count,
    score_identity_cols,
)
from lib.parquet_schema import is_detection_stats_parquet
from lib.summary_compare import build_summary_delta

PRODUCT_LABEL_JA_DEFAULT = {
    "Occlusion-Case": "遮蔽ケース",
    "False-Positive-Grass": "草誤検知（草停止）",
    "False-Positive-Ground": "地面誤検知",
    "False-Positive-Splash": "水しぶき 誤検知",
    "False-Positive-Exhaust-Fog": "排ガス・霧 誤検知",
    "Missed-Detection-Animal": "動物ロスト（犬）",
    "Missed-Detection-Falling-Object": "落下物未検知",
    "Missed-Detection-Pedestrian-Child": "歩行者未検知：子供",
    "Missed-Detection-Pedestrian-Umbrella": "歩行者未検知：傘",
    "Missed-Detection-Pedestrian-Crouching": "歩行者未検知：しゃがむ",
    "Missed-Detection-Pedestrian-Near-Structure": "歩行者未検知：構造物に近い",
    "False-Positive-Truck": "トラック誤検知",
    "Pose-Estimation-Yaw-Error": "Yawおかしい",
    "Long-Range-Detection-Failure": "遠方見えない",
    "Ghost-Object": "ミサイル",
    "Sudden-Fast-Vehicle-Ghost": "高速車両の突然出現・急ブレーキ誘発",
    "Misclassification-Structure-Grass-as-Pedestrian": "構造物・草を人に誤検知",
    "Misclassification-Structure-Grass-as-Vehicle": "構造物・草を車両に誤検知",
    "Misclassification-Bike-Motorcycle": "自転車・バイクのミスラベル",
    "Missed-Detection-Unridden-Bike": "人の乗ってないバイク自転車ロスト",
    "Missed-Detection-Traffic-Cone": "カラーコーンが認識できない",
    "Missed-Detection-Other": "その他ロスト",
}

_COMPARE_RUN_COLORS = ["#312e81", "#0f766e", "#e86a33", "#6b8e23", "#9b59b6", "#1abc9c"]
_OVERVIEW_COMPARE_COLORS = ["#31356E", "#008E9B", "#E86A33", "#6B8E23", "#9B59B6", "#1ABC9C"]
_CRITERIA_COLS = SCORE_VIEW_METRIC_COLS
_NUM_COLS = SCORE_NUM_COLS
_BLOCK_SIZE = SCORE_BLOCK_SIZE
_DEFAULT_MAX_EVAL_RANGE = 50
_DISTANCE_BIN_CASE = """CASE
    WHEN dist_h < 10 THEN '[0,10)'
    WHEN dist_h < 20 THEN '[10,20)'
    WHEN dist_h < 30 THEN '[20,30)'
    WHEN dist_h < 40 THEN '[30,40)'
    WHEN dist_h < 50 THEN '[40,50)'
    WHEN dist_h < 60 THEN '[50,60)'
    WHEN dist_h < 70 THEN '[60,70)'
    WHEN dist_h < 80 THEN '[70,80)'
    WHEN dist_h < 90 THEN '[80,90)'
    WHEN dist_h < 100 THEN '[90,100)'
    WHEN dist_h < 110 THEN '[100,110)'
    WHEN dist_h < 120 THEN '[110,120)'
    WHEN dist_h < 130 THEN '[120,130)'
    WHEN dist_h < 140 THEN '[130,140)'
    WHEN dist_h < 150 THEN '[140,150)'
    ELSE '[150,inf)'
END"""


def make_report_filename(
    run_names: Sequence[str],
    *,
    now: Optional[datetime] = None,
    prefix: str = "overview_report",
) -> str:
    ts = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    slug = _slugify(run_names[0] if run_names else "report")
    return f"{prefix}_{slug}_{ts}.pdf"


def build_overview_pdf_report(
    *,
    mode: str,
    run_records: Sequence[dict],
    run_labels: Sequence[str],
    filters: Optional[dict] = None,
    product_label_map: Optional[dict] = None,
    generated_at: Optional[datetime] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> bytes:
    reportlab_import_error = _ensure_reportlab_available()
    if reportlab_import_error is not None:
        raise RuntimeError(reportlab_import_error)

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    product_label_map = product_label_map or PRODUCT_LABEL_JA_DEFAULT
    generated_at = generated_at or datetime.now()
    filters = filters or {}

    def _notify(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=28,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#0f172a"),
    )
    section_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading1"],
        fontSize=16,
        leading=21,
        spaceAfter=8,
        textColor=colors.HexColor("#0f172a"),
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#334155"),
    )
    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#475569"),
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="Overview PDF Report",
    )
    content_width = doc.width
    story: List[Any] = []
    _notify("Preparing cover and active filter summary")

    run_names = [Path(str(r.get("path", ""))).name or f"Run {lbl}" for r, lbl in zip(run_records, run_labels)]
    story.extend(
        [
            Paragraph("Evaluation Dashboard Report", title_style),
            Spacer(1, 8),
            Paragraph(
                f"Generated {generated_at.strftime('%Y-%m-%d %H:%M:%S')} · "
                f"{'Compare mode' if mode == 'Compare Mode' else 'Single mode'}",
                body_style,
            ),
            Spacer(1, 8),
            _styled_table(
                [["Run", "Label", "Directory"]] + [
                    [f"Run {lbl}", name, str(record.get("path", ""))]
                    for record, lbl, name in zip(run_records, run_labels, run_names)
                ],
                content_width,
            ),
            Spacer(1, 12),
            Paragraph(
                f"Perception labels: {_summarize_filter_values(filters.get('perception_labels'))}<br/>"
                f"Product labels: {_summarize_filter_values(filters.get('product_labels'))}",
                body_style,
            ),
            Spacer(1, 16),
        ]
    )

    _notify("Building Overview section")
    overview_section = _build_overview_section(run_records, run_labels, product_label_map)
    _notify("Building TP Summary section")
    tp_section = _build_tp_summary_section(run_records, run_labels, product_label_map)
    _notify("Building Criteria Based Score section")
    criteria_section = _build_criteria_section(run_records, run_labels)
    _notify("Building Detection Stats section")
    detection_section = _build_detection_section(run_records, run_labels)

    sections = [
        ("Overview", overview_section),
        ("TP Summary", tp_section),
        ("Criteria Based Score", criteria_section),
        ("Detection Stats", detection_section),
    ]

    available_sections = 0
    for idx, (title, payload) in enumerate(sections):
        story.append(Paragraph(title, section_style))
        story.append(Paragraph(payload["summary"], body_style))
        story.append(Spacer(1, 8))
        if payload.get("flowables"):
            for flowable in payload["flowables"]:
                story.append(flowable)
                story.append(Spacer(1, 8))
        if payload.get("tables"):
            for table in payload["tables"]:
                story.append(_styled_table(table, content_width))
                story.append(Spacer(1, 8))
        figs = payload.get("figures", [])
        if figs:
            exported_any_fig = False
            for fig, caption in figs:
                try:
                    story.append(_plotly_figure_to_image(fig, content_width, ImageReader))
                    story.append(Spacer(1, 4))
                    story.append(Paragraph(caption, caption_style))
                    exported_any_fig = True
                except Exception as exc:
                    story.append(
                        Paragraph(
                            f"Chart export unavailable for this figure: {str(exc)}",
                            caption_style,
                        )
                    )
                story.append(Spacer(1, 12))
            if exported_any_fig:
                available_sections += 1
        else:
            if payload.get("tables"):
                available_sections += 1
            story.append(Paragraph(payload.get("fallback_note", "Section unavailable."), caption_style))
            story.append(Spacer(1, 12))
        if idx != len(sections) - 1:
            story.append(PageBreak())

    if available_sections == 0:
        story.append(
            Paragraph(
                "No report sections were available for export. Check Summary.csv, Score.csv, and parquet data for the selected run(s).",
                body_style,
            )
        )

    def _draw_page_number(canvas, document):
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawRightString(document.pagesize[0] - document.rightMargin, 18, f"Page {document.page}")

    _notify("Assembling PDF pages")
    doc.build(story, onFirstPage=_draw_page_number, onLaterPages=_draw_page_number)
    _notify("Finalizing PDF bytes")
    return buffer.getvalue()


def _build_tp_summary_section(
    run_records: Sequence[dict],
    run_labels: Sequence[str],
    product_label_map: dict,
) -> dict:
    available = [r for r in run_records if r.get("summary") is not None]
    if not available:
        return {
            "summary": "Summary.csv is not available for the selected run set.",
            "figures": [],
            "tables": [],
            "fallback_note": "TP Summary skipped because Summary.csv is missing.",
        }

    summaries = [r["summary"] for r in run_records if r.get("summary") is not None]
    labels = [run_labels[i] for i, r in enumerate(run_records) if r.get("summary") is not None]
    figures: List[Tuple[go.Figure, str]] = []
    tables: List[list[list[str]]] = []

    metrics_table = [["Run", "Rows", "TP mean", "XRMS mean", "YRMS mean", "XSTD mean", "YSTD mean"]]
    for lbl, df in zip(labels, summaries):
        metrics_table.append(
            [
                lbl,
                f"{len(df):,}",
                _fmt_number(df["TP"].mean()),
                _fmt_number(df["xrms"].mean()),
                _fmt_number(df["yrms"].mean()),
                _fmt_number(df["xstd"].mean()),
                _fmt_number(df["ystd"].mean()),
            ]
        )
    tables.append(metrics_table)

    if len(summaries) >= 2:
        baseline_lbl = labels[0]
        for cand_idx in range(1, len(summaries)):
            cand_lbl = labels[cand_idx]
            delta_df = build_summary_delta(summaries[0], summaries[cand_idx])
            if delta_df.empty:
                figures.append(
                    (
                        _make_text_placeholder_figure(
                            f"No overlapping Summary rows for delta ({cand_lbl} vs {baseline_lbl})."
                        ),
                        f"Delta view is empty because baseline {baseline_lbl} and candidate {cand_lbl} do not share Summary keys.",
                    )
                )
            else:
                figures.extend(_build_tp_default_compare_figures(delta_df, cand_lbl))
    else:
        figures.extend(_build_tp_default_single_figures(summaries[0]))

    return {
        "summary": "This section follows the default TP Summary page view as closely as possible using the current Overview-selected runs and filters.",
        "figures": figures,
        "tables": tables,
        "fallback_note": "No TP Summary figures were available after filtering.",
    }


def _build_overview_section(
    run_records: Sequence[dict],
    run_labels: Sequence[str],
    product_label_map: dict,
) -> dict:
    summary_runs = [(run_labels[i], r["summary"]) for i, r in enumerate(run_records) if r.get("summary") is not None]
    if not summary_runs:
        return {
            "summary": "Overview metrics are unavailable because Summary.csv is missing for the selected run set.",
            "figures": [],
            "tables": [],
            "fallback_note": "Overview section skipped because Summary.csv is missing.",
        }

    tables: List[list[list[str]]] = []
    figures: List[Tuple[go.Figure, str]] = []
    flowables: List[Any] = []

    metric_card_rows = [["Run", "TP mean", "XRMS", "YRMS", "XSTD", "YSTD"]]
    for lbl, df in summary_runs:
        metric_card_rows.append(
            [
                lbl,
                _fmt_number(df["TP"].mean()),
                _fmt_number(df["xrms"].mean()),
                _fmt_number(df["yrms"].mean()),
                _fmt_number(df["xstd"].mean()),
                _fmt_number(df["ystd"].mean()),
            ]
        )
    tables.append(metric_card_rows)
    flowables.extend(_build_overview_metric_cards(summary_runs))

    summaries = [df for _, df in summary_runs]
    labels = [lbl for lbl, _ in summary_runs]
    fig_perception = _build_tp_mean_by_label_compare_figure(summaries, labels, "perception_label")
    if fig_perception is not None:
        figures.append((fig_perception, "Overview page result: TP mean by Perception Label."))
    fig_product = _build_tp_mean_by_label_compare_figure(
        summaries,
        labels,
        "product_label",
        label_jp_map=product_label_map,
    )
    if fig_product is not None:
        figures.append((fig_product, "Overview page result: TP mean by Product Label."))

    return {
        "summary": "This section mirrors the Overview page first: summary metrics and TP mean by label using the current Overview run selection and label filters.",
        "flowables": flowables,
        "figures": figures,
        "tables": tables,
        "fallback_note": "Overview figures were unavailable after filtering.",
    }


def _build_overview_metric_cards(summary_runs: Sequence[Tuple[str, pd.DataFrame]]) -> List[Any]:
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    card_cells: List[Any] = []
    for idx, (lbl, df) in enumerate(summary_runs):
        accent = _compare_color(idx)
        rows = [
            [f"Run {lbl}"],
            [f"TP mean  {_fmt_number(df['TP'].mean())}"],
            [f"XRMS  {_fmt_number(df['xrms'].mean())}    YRMS  {_fmt_number(df['yrms'].mean())}"],
            [f"XSTD  {_fmt_number(df['xstd'].mean())}    YSTD  {_fmt_number(df['ystd'].mean())}"],
        ]
        t = Table(rows, colWidths=[220])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(accent)),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 11),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
                    ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#0f172a")),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 1), (-1, -1), 10),
                    ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#cbd5e1")),
                    ("ROUNDEDCORNERS", [10, 10, 10, 10]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        card_cells.append(t)

    if not card_cells:
        return []

    cards_per_row = 2
    grid_rows: List[List[Any]] = []
    for start in range(0, len(card_cells), cards_per_row):
        row = card_cells[start : start + cards_per_row]
        if len(row) < cards_per_row:
            row = row + ["" for _ in range(cards_per_row - len(row))]
        grid_rows.append(row)

    grid = Table(grid_rows, colWidths=[260, 260], hAlign="LEFT")
    grid.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return [grid]


def _build_criteria_section(run_records: Sequence[dict], run_labels: Sequence[str]) -> dict:
    score_runs = [(run_labels[i], r) for i, r in enumerate(run_records) if r.get("score") is not None]
    if not score_runs:
        return {
            "summary": "Score.csv is not available for the selected run set.",
            "figures": [],
            "tables": [],
            "fallback_note": "Criteria section skipped because Score.csv is missing.",
        }

    criteria_count = min(infer_score_criteria_count(rec["score"]) for _, rec in score_runs)
    if criteria_count <= 0:
        return {
            "summary": "Score.csv was loaded, but no criteria blocks were detected.",
            "figures": [],
            "tables": [],
            "fallback_note": "Criteria section skipped because no criteria blocks were found.",
        }

    criteria_idx = 0
    views: List[Tuple[str, pd.DataFrame]] = []
    for lbl, rec in score_runs:
        df_view = _build_score_view(rec["score"], criteria_idx)
        if not df_view.empty:
            df_view["Run"] = lbl
            views.append((lbl, df_view))

    if not views:
        return {
            "summary": "Criteria data was present but could not be shaped into a report view.",
            "figures": [],
            "tables": [],
            "fallback_note": "Criteria section skipped because the selected rows were empty.",
        }

    combined = pd.concat([df for _, df in views], ignore_index=True)
    tables = [
        [["Run", "Rows", "Pass rate mean", "Pass rate median", "NM mean"]]
        + [
            [
                lbl,
                f"{len(df):,}",
                _fmt_number(df["pass_rate"].mean()),
                _fmt_number(df["pass_rate"].median()),
                _fmt_number(df["nm"].mean()),
            ]
            for lbl, df in views
        ]
    ]

    figures: List[Tuple[go.Figure, str]] = []
    if len(views) >= 2:
        figures.extend(_build_criteria_default_compare_figures(views))
        scenario_table = _build_criteria_compare_table(views)
    else:
        figures.extend(_build_criteria_default_single_figures(views[0][1]))
        scenario_table = _build_criteria_single_table(views[0][1])
    if scenario_table:
        tables.append(scenario_table)

    return {
        "summary": "This section follows the default Criteria Based Score page setup: criteria0, metric=pass_rate, and group_by=GT_OBJ.",
        "figures": figures,
        "tables": tables,
        "fallback_note": "Criteria charts were unavailable for the selected run set.",
    }


def _build_detection_section(run_records: Sequence[dict], run_labels: Sequence[str]) -> dict:
    con = duckdb.connect()
    parquet_paths: List[Tuple[str, str]] = []
    for rec, lbl in zip(run_records, run_labels):
        for file in sorted(Path(rec["path"]).glob("*.parquet")):
            if is_detection_stats_parquet(con, str(file)):
                parquet_paths.append((lbl, str(file)))
                break

    if not parquet_paths:
        return {
            "summary": "No object-level detection parquet files were found in the selected run set.",
            "figures": [],
            "tables": [],
            "fallback_note": "Detection Stats skipped because compatible parquet data is missing.",
        }

    views: List[Tuple[str, str]] = []
    try:
        for idx, (lbl, pq) in enumerate(parquet_paths):
            view_name = "pdf_eval_flat" if idx == 0 else f"pdf_eval_flat_{idx}"
            _create_eval_flat_view(con, pq, view_name)
            views.append((lbl, view_name))

        tables = [[["Run", "TP", "FP", "FN", "TPR", "Precision", "F1"]]]
        figures: List[Tuple[go.Figure, str]] = []

        kpi_rows = [["Run", "TP", "FP", "FN", "TPR", "Precision", "F1"]]
        for lbl, view in views:
            kpi = _kpi_row_for_view(con, view)
            if kpi is None:
                continue
            kpi_rows.append(
                [
                    lbl,
                    f"{kpi['tp']:,}",
                    f"{kpi['fp']:,}",
                    f"{kpi['fn']:,}",
                    _fmt_percent(kpi["tpr"]),
                    _fmt_percent(kpi["precision"]),
                    _fmt_percent(kpi["f1"]),
                ]
            )
        tables = [kpi_rows] if len(kpi_rows) > 1 else []

        dataset_rows = [["Run", "Distinct datasets"]]
        for lbl, view in views:
            n_ds = con.execute(f"SELECT COUNT(DISTINCT t4dataset_id) FROM {view}").fetchone()[0]
            dataset_rows.append([lbl, f"{int(n_ds or 0):,}"])
        if len(dataset_rows) > 1:
            tables.append(dataset_rows)

        df_status = _query_status_counts(con, views)
        if not df_status.empty:
            fig_status = _build_detection_status_figure(df_status)
            _apply_detection_theme(fig_status, "Detection status distribution by label")
            figures.append((fig_status, "Stacked TP/FP/FN counts per label from the first parquet file in each selected run."))

        figures.extend(_build_detection_distance_figures(con, views))
        figures.extend(_build_detection_tpr_figures(con, views))
        figures.extend(_build_detection_mean_error_figures(con, views))
        figures.extend(_build_detection_perception_diff_figures(con, views))

        return {
            "summary": (
                "This section follows the default Detection Stats view as closely as possible: "
                "summary KPIs, status distribution, distance panels, TP rate, mean error, and compare-mode perception diff."
            ),
            "figures": figures,
            "tables": tables,
            "fallback_note": "Detection charts were unavailable for the selected parquet data.",
        }
    finally:
        con.close()


def _build_tp_mean_by_label_compare_figure(
    df_list: Sequence[pd.DataFrame],
    run_labels: Sequence[str],
    label_col: str,
    *,
    label_jp_map: Optional[dict] = None,
) -> Optional[go.Figure]:
    if not df_list or not run_labels or label_col not in df_list[0].columns:
        return None
    all_labels = set()
    groups = []
    for df in df_list:
        if label_col not in df.columns:
            return None
        xdf = df[df[label_col].notna() & (df[label_col].astype(str).str.strip() != "")]
        g = xdf.groupby(label_col)["TP"].mean() if not xdf.empty else pd.Series(dtype=float)
        groups.append(g)
        all_labels.update(g.index)
    if not all_labels:
        return None
    all_labels = sorted(all_labels)
    labels_disp = [label_jp_map.get(l, l) for l in all_labels] if label_jp_map else all_labels
    traces = []
    for idx, (g, lbl) in enumerate(zip(groups, run_labels)):
        vals = [g.get(label, float("nan")) for label in all_labels]
        traces.append(
            go.Bar(
                name=lbl,
                x=labels_disp,
                y=vals,
                marker=dict(color=_OVERVIEW_COMPARE_COLORS[idx % len(_OVERVIEW_COMPARE_COLORS)]),
                text=[f"{x:.2f}" if pd.notna(x) else "N/A" for x in vals],
                textposition="auto",
            )
        )
    fig = go.Figure(traces)
    fig.update_layout(
        title=f"TP mean by {label_col.replace('_', ' ')}",
        barmode="group",
        xaxis_title=label_col.replace("_", " ").title(),
        yaxis_title="TP mean",
        height=420,
        margin=dict(t=70, b=55, l=55, r=25),
        legend_title="Run",
        template="plotly_white",
    )
    return fig


def _build_tp_default_single_figures(df: pd.DataFrame) -> List[Tuple[go.Figure, str]]:
    df_f = df.copy()
    for column in ("vx", "vy"):
        if column in df_f.columns and not df_f.empty:
            q1, q99 = df_f[column].quantile([0.01, 0.99]).values
            df_f[column] = df_f[column].clip(q1, q99)
    figures: List[Tuple[go.Figure, str]] = []
    fig_rms = px.scatter(
        df_f,
        x="xrms",
        y="yrms",
        color="TP",
        hover_data=["id"],
        labels={"xrms": "X RMS", "yrms": "Y RMS", "TP": "TP"},
        color_continuous_scale="Viridis",
    )
    fig_rms.update_traces(marker=dict(size=8, opacity=0.7))
    _apply_tp_clean_theme(fig_rms)
    figures.append((fig_rms, "Default TP Summary RMS scatter from the selected Summary.csv rows."))

    fig_vel = px.scatter(
        df_f,
        x="vx",
        y="vy",
        color="TP",
        hover_data=["id"],
        labels={"vx": "Vx", "vy": "Vy", "TP": "TP"},
        color_continuous_scale="Plasma",
        title="Vx vs Vy",
    )
    _apply_tp_clean_theme(fig_vel)
    figures.append((fig_vel, "Default TP Summary velocity scatter with outlier clipping enabled."))

    figures.append((_build_tp_distribution_figure(df_f, "TP"), "Default TP distribution view (metric = TP)."))
    figures.append((_build_tp_violin_figure(df_f, "TP"), "Default TP density violin for metric = TP."))
    return figures


def _build_tp_default_compare_figures(df_delta: pd.DataFrame, candidate_label: str) -> List[Tuple[go.Figure, str]]:
    figures: List[Tuple[go.Figure, str]] = []
    tp_col = "TP_delta"
    fig_rms_x = px.scatter(
        df_delta,
        x="xrms_B",
        y="xrms",
        color=tp_col,
        hover_data=["id", "xrms_delta", "yrms_delta"],
        labels={
            "xrms_B": f"X RMS ({candidate_label})",
            "xrms": "X RMS (A)",
            tp_col: "Delta TP",
            "xrms_delta": "Delta X RMS",
            "yrms_delta": "Delta Y RMS",
        },
        title=f"Scatter: X RMS ({candidate_label}) vs X RMS (A)",
        color_continuous_scale="Viridis",
    )
    fig_rms_x.update_traces(marker=dict(size=8, opacity=0.6))
    _apply_tp_clean_theme(fig_rms_x)
    figures.append(
        (fig_rms_x, f"TP Summary compare ({candidate_label} vs baseline): X RMS scatter, colored by TP delta.")
    )

    fig_rms_y = px.scatter(
        df_delta,
        x="yrms_B",
        y="yrms",
        color=tp_col,
        hover_data=["id", "xrms_delta", "yrms_delta"],
        labels={
            "yrms_B": f"Y RMS ({candidate_label})",
            "yrms": "Y RMS (A)",
            tp_col: "Delta TP",
            "xrms_delta": "Delta X RMS",
            "yrms_delta": "Delta Y RMS",
        },
        title=f"Scatter: Y RMS ({candidate_label}) vs Y RMS (A)",
        color_continuous_scale="Viridis",
    )
    fig_rms_y.update_traces(marker=dict(size=8, opacity=0.6))
    _apply_tp_clean_theme(fig_rms_y)
    figures.append(
        (fig_rms_y, f"TP Summary compare ({candidate_label} vs baseline): Y RMS scatter, colored by TP delta.")
    )

    figures.append(
        (
            _build_tp_distribution_figure(df_delta, "TP_delta"),
            f"TP Summary compare ({candidate_label} vs baseline): TP delta distribution.",
        )
    )
    figures.append(
        (
            _build_tp_violin_figure(df_delta, "TP_delta"),
            f"TP Summary compare ({candidate_label} vs baseline): TP delta violin.",
        )
    )
    return figures


def _build_tp_distribution_figure(df: pd.DataFrame, metric: str) -> go.Figure:
    fig = px.histogram(
        df,
        x=metric,
        nbins=40,
        color_discrete_sequence=["#0d9488"],
        marginal="box",
        opacity=0.88,
    )
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        bargap=0.04,
        xaxis_title=metric,
        yaxis_title="Count",
        paper_bgcolor="rgba(248,250,252,0.9)",
        plot_bgcolor="rgba(255,255,255,0.95)",
        font=dict(family="system-ui, sans-serif", size=12, color="#334155"),
        margin=dict(t=36, b=48, l=56, r=28),
    )
    return fig


def _build_tp_violin_figure(df: pd.DataFrame, metric: str) -> go.Figure:
    fig = px.violin(
        df,
        y=metric,
        box=True,
        points="all",
        color_discrete_sequence=["#312e81"],
    )
    fig.update_layout(
        template="plotly_white",
        yaxis_title=metric,
        showlegend=False,
        paper_bgcolor="rgba(248,250,252,0.9)",
        plot_bgcolor="rgba(255,255,255,0.95)",
        font=dict(family="system-ui, sans-serif", size=12, color="#334155"),
        margin=dict(t=36, b=48, l=56, r=28),
    )
    return fig


def _apply_tp_clean_theme(fig: go.Figure) -> None:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(248,250,252,0.9)",
        plot_bgcolor="rgba(255,255,255,0.95)",
        font=dict(family="system-ui, sans-serif", size=12, color="#334155"),
        margin=dict(t=48, b=48, l=56, r=28),
    )


def _build_criteria_default_single_figures(df_view: pd.DataFrame) -> List[Tuple[go.Figure, str]]:
    figures: List[Tuple[go.Figure, str]] = []
    metric = "pass_rate"
    group_by = "GT_OBJ" if df_view["GT_OBJ"].notna().any() else "Option"
    fig_hist = px.histogram(
        df_view,
        x=metric,
        color=group_by,
        nbins=30,
        marginal="box",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    _apply_criteria_theme(fig_hist, f"{metric} · histogram")
    figures.append((fig_hist, "Default Criteria page distribution chart for criteria0 and metric = pass_rate."))

    df_avg = df_view.groupby(group_by, as_index=False)[metric].mean().sort_values(metric, ascending=False)
    fig_bar = px.bar(
        df_avg,
        x=group_by,
        y=metric,
        text_auto=".2f",
        color=group_by,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    _apply_criteria_theme(fig_bar, f"Mean {metric}")
    fig_bar.update_layout(showlegend=False)
    figures.append((fig_bar, f"Default grouped mean chart by {group_by}."))

    fig_box = px.box(
        df_view,
        x=group_by,
        y="pass_rate",
        points="all",
        color=group_by,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    _apply_criteria_theme(fig_box, "Pass rate by group")
    fig_box.update_layout(showlegend=False)
    figures.append((fig_box, f"Default pass-rate overview by {group_by}."))
    return figures


def _build_criteria_default_compare_figures(views: Sequence[Tuple[str, pd.DataFrame]]) -> List[Tuple[go.Figure, str]]:
    figures: List[Tuple[go.Figure, str]] = []
    metric = "pass_rate"
    group_by = "GT_OBJ"
    run_order = [lbl for lbl, _ in views]
    combined = pd.concat([df.assign(Run=lbl) for lbl, df in views], ignore_index=True)
    combined["Run"] = pd.Categorical(combined["Run"], categories=run_order, ordered=True)
    px_map = {lbl: _COMPARE_RUN_COLORS[i % len(_COMPARE_RUN_COLORS)] for i, (lbl, _) in enumerate(views)}

    fig_hist = px.histogram(
        combined,
        x=metric,
        color="Run",
        color_discrete_map=px_map,
        category_orders={"Run": run_order},
        nbins=30,
        barmode="overlay",
        opacity=0.55,
        marginal="box",
    )
    _apply_criteria_theme(fig_hist, f"{metric} · row-level distribution")
    figures.append((fig_hist, "Default compare overlay view for pass-rate distribution."))

    df_avg = combined.groupby([group_by, "Run"], as_index=False)[metric].mean()
    obj_means = df_avg.groupby(group_by, as_index=False)[metric].mean().sort_values(metric, ascending=False)
    obj_order = [x for x in obj_means[group_by].tolist() if x in set(df_avg[group_by])]
    df_avg[group_by] = pd.Categorical(df_avg[group_by], categories=obj_order, ordered=True)
    df_avg = df_avg.sort_values([group_by, "Run"])
    fig_bar = px.bar(
        df_avg,
        x=group_by,
        y=metric,
        color="Run",
        color_discrete_map=px_map,
        category_orders={group_by: obj_order, "Run": run_order},
        barmode="group",
        text_auto=".2f",
    )
    _apply_criteria_theme(fig_bar, f"Mean {metric} by {group_by}")
    figures.append((fig_bar, f"Default compare grouped mean view by {group_by}."))

    fig_box = px.box(
        combined,
        x=group_by,
        y="pass_rate",
        color="Run",
        color_discrete_map=px_map,
        category_orders={group_by: obj_order, "Run": run_order},
        points="all",
    )
    _apply_criteria_theme(fig_box, "Pass rate overview")
    figures.append((fig_box, f"Default compare pass-rate overview by {group_by}."))

    scenario_delta = _build_criteria_compare_delta_figure(views)
    if scenario_delta is not None:
        base_l = run_order[0]
        if len(run_order) == 2:
            cap = f"Default compare per-scenario delta view for candidate {run_order[1]} vs baseline {base_l}."
        else:
            rest = ", ".join(run_order[1:])
            cap = (
                f"Default compare per-scenario delta vs baseline {base_l} "
                f"for candidates {rest} (grouped bars)."
            )
        figures.append((scenario_delta, cap))
    return figures


def _build_criteria_single_table(df_view: pd.DataFrame) -> List[List[str]]:
    key_cols = score_identity_cols(df_view)
    scenario_metric = df_view.groupby(key_cols, as_index=False)["pass_rate"].mean().sort_values("pass_rate", ascending=False).head(20)
    rows = [key_cols + ["Pass rate mean"]]
    for _, row in scenario_metric.iterrows():
        rows.append([_shorten_scenario_name(str(row[c])) for c in key_cols] + [_fmt_number(row["pass_rate"])])
    first_w = 0.56 if len(key_cols) > 1 else 0.72
    rest_w = (1.0 - first_w) / len(key_cols)
    return {"rows": rows, "col_width_weights": [first_w] + [rest_w] * len(key_cols)}


def _build_criteria_compare_table(views: Sequence[Tuple[str, pd.DataFrame]]) -> List[List[str]]:
    labels = [lbl for lbl, _ in views]
    key_cols = score_identity_cols(views[0][1])
    merges = []
    for lbl, df in views:
        g = df.groupby(key_cols, as_index=False)["pass_rate"].mean()
        merges.append(g.rename(columns={"pass_rate": f"pr_{lbl}"}))
    per_scenario = merges[0]
    for g in merges[1:]:
        per_scenario = per_scenario.merge(g, on=key_cols, how="inner")
    base = labels[0]
    delta_cols: List[str] = []
    for cand in labels[1:]:
        dcol = f"delta_{cand}"
        per_scenario[dcol] = per_scenario[f"pr_{cand}"] - per_scenario[f"pr_{base}"]
        delta_cols.append(dcol)
    rank_key = per_scenario[delta_cols].abs().max(axis=1)
    per_scenario = per_scenario.reindex(rank_key.sort_values(ascending=False).index).head(20)
    header: List[str] = key_cols + [f"Pass rate ({base})"]
    for cand in labels[1:]:
        header.extend([f"Pass rate ({cand})", f"Δ({cand} - {base})"])
    rows = [header]
    for _, row in per_scenario.iterrows():
        cells: List[str] = [_shorten_scenario_name(str(row[c])) for c in key_cols] + [_fmt_number(row[f"pr_{base}"])]
        for cand in labels[1:]:
            cells.extend([_fmt_number(row[f"pr_{cand}"]), _fmt_number(row[f"delta_{cand}"])])
        rows.append(cells)
    ncols = len(header)
    scen_w = 0.28 if ncols > 5 else 0.44
    rest_w = (1.0 - scen_w) / max(ncols - 1, 1)
    weights = [scen_w] + [rest_w] * (ncols - 1)
    return {"rows": rows, "col_width_weights": weights}


def _build_criteria_compare_delta_figure(views: Sequence[Tuple[str, pd.DataFrame]]) -> Optional[go.Figure]:
    if len(views) < 2:
        return None
    labels = [lbl for lbl, _ in views]
    base = labels[0]
    key_cols = score_identity_cols(views[0][1])
    merges = []
    for lbl, df in views:
        g = df.groupby(key_cols, as_index=False)["pass_rate"].mean()
        merges.append(g.rename(columns={"pass_rate": f"pr_{lbl}"}))
    per_scenario = merges[0]
    for g in merges[1:]:
        per_scenario = per_scenario.merge(g, on=key_cols, how="inner")
    if per_scenario.empty:
        return None
    long_rows: List[dict] = []
    delta_cols: List[str] = []
    for cand in labels[1:]:
        dcol = f"delta_{cand}"
        per_scenario[dcol] = per_scenario[f"pr_{cand}"] - per_scenario[f"pr_{base}"]
        delta_cols.append(dcol)
    rank_key = per_scenario[delta_cols].abs().max(axis=1)
    vis = per_scenario.reindex(rank_key.sort_values(ascending=False).index).head(20)
    if "Dataset" in key_cols:
        scenario_labels = vis["Scenario"].astype(str) + " [" + vis["Dataset"].astype(str) + "]"
    else:
        scenario_labels = vis["Scenario"].astype(str)
    scen_order = [_shorten_scenario_name(str(s)) for s in scenario_labels.tolist()]
    for _, row in vis.iterrows():
        scen_raw = f"{row['Scenario']} [{row['Dataset']}]" if "Dataset" in key_cols else row["Scenario"]
        scen_disp = _shorten_scenario_name(str(scen_raw))
        for cand in labels[1:]:
            long_rows.append(
                {
                    "Scenario": scen_disp,
                    "vs_baseline": f"Δ({cand} - {base})",
                    "delta": float(row[f"delta_{cand}"]),
                }
            )
    melted = pd.DataFrame(long_rows)
    if melted.empty:
        return None
    legend_order = [f"Δ({cand} - {base})" for cand in labels[1:]]
    color_map = {
        leg: _COMPARE_RUN_COLORS[(i + 1) % len(_COMPARE_RUN_COLORS)]
        for i, leg in enumerate(legend_order)
    }
    fig = px.bar(
        melted,
        x="Scenario",
        y="delta",
        color="vs_baseline",
        color_discrete_map=color_map,
        category_orders={"Scenario": scen_order, "vs_baseline": legend_order},
        barmode="group",
        text_auto=".2f",
    )
    fig.update_layout(coloraxis_showscale=False, legend_title_text="")
    _apply_criteria_theme(fig, "Pass rate delta by scenario")
    return fig


def _build_detection_status_figure(df_status: pd.DataFrame) -> go.Figure:
    status_colors = {"TP": "#2d8f47", "FN": "#d73027", "FP": "#E86A33", "TN": "#4A90D9"}
    if "run" in df_status.columns and df_status["run"].nunique() > 1:
        fig = px.bar(
            df_status,
            x="label",
            y="num",
            color="status",
            barmode="stack",
            facet_col="run",
            color_discrete_map=status_colors,
            title="Status Distribution per Label",
            labels={"num": "Count", "label": "Label", "status": "Status"},
        )
        fig.for_each_annotation(lambda ann: ann.update(text=ann.text.replace("run=", "")))
        return fig
    if df_status["label"].nunique() > 6:
        return px.bar(
            df_status,
            y="label",
            x="num",
            color="status",
            barmode="stack",
            title="Status Distribution per Label",
            labels={"num": "Count", "label": "Label", "status": "Status"},
            color_discrete_map=status_colors,
            orientation="h",
        )
    return px.bar(
        df_status,
        x="label",
        y="num",
        color="status",
        barmode="stack",
        title="Status Distribution per Label",
        labels={"num": "Count", "label": "Label", "status": "Status"},
        color_discrete_map=status_colors,
    )


def _build_detection_distance_figures(
    con: duckdb.DuckDBPyConnection,
    views: Sequence[Tuple[str, str]],
) -> List[Tuple[go.Figure, str]]:
    figures: List[Tuple[go.Figure, str]] = []
    labels = [lbl for lbl, _ in views]
    if len(views) == 1:
        df_both = _query_distance_rates_single(con, views[0][1])
        if not df_both.empty:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=df_both["bin_label"],
                    y=df_both["tpr"],
                    name="TP rate",
                    marker_color=_COMPARE_RUN_COLORS[0],
                    hovertemplate="%{x}<br>TP rate: %{y:.2%}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=df_both["bin_label"],
                    y=df_both["fpr"],
                    name="FP rate",
                    marker_color=_COMPARE_RUN_COLORS[2],
                    hovertemplate="%{x}<br>FP rate: %{y:.2%}<extra></extra>",
                )
            )
            _apply_detection_theme(fig, "TP & FP rate by distance")
            fig.update_layout(
                xaxis_title="Distance bin",
                yaxis_title="Rate",
                yaxis_range=[0, 1],
                barmode="group",
                xaxis=dict(tickangle=-35, categoryorder="array", categoryarray=df_both["bin_label"].tolist()),
                hovermode="x unified",
            )
            fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(0,0,0,0.25)")
            figures.append((fig, "Detection Stats distance panel in bar-chart mode across the full 0-150+ range."))
        df_oc = _query_object_counts_single(con, views[0][1])
        if not df_oc.empty:
            align_x = sorted(df_oc["bin_label"].unique(), key=_distance_bin_sort_key)
            pivot_oc = df_oc.pivot_table(index="bin_label", columns="label", values="n", aggfunc="sum", fill_value=0).reindex(align_x, fill_value=0)
            fig_oc = go.Figure()
            for j, lab in enumerate(pivot_oc.columns):
                c = _compare_color(j)
                fig_oc.add_trace(
                    go.Bar(
                        x=align_x,
                        y=pivot_oc[lab].values,
                        name=str(lab),
                        marker_color=c,
                        hovertemplate=f"{lab}<br>%{{x}}<br>Count: %{{y:.0f}}<extra></extra>",
                    )
                )
            _apply_detection_theme(fig_oc, "Object count by distance bin")
            fig_oc.update_layout(
                xaxis_title="Distance bin",
                yaxis_title="Count",
                barmode="group",
                xaxis=dict(tickangle=-35, categoryorder="array", categoryarray=align_x),
                hovermode="x unified",
            )
            figures.append((fig_oc, "Detection Stats object-count-by-distance panel in bar-chart mode across the full 0-150+ range."))
        return figures

    df_tpr = _query_distance_rates_compare(con, views, metric="tpr")
    if not df_tpr.empty:
        fig_tpr = go.Figure()
        for i, lbl in enumerate(labels):
            d = df_tpr[df_tpr["run"] == lbl].sort_values("bin_order")
            c = _compare_color(i)
            fig_tpr.add_trace(
                go.Bar(
                    x=d["bin_label"],
                    y=d["tpr"],
                    name=lbl,
                    marker_color=c,
                    hovertemplate=f"{lbl}<br>%{{x}}<br>TP rate: %{{y:.2%}}<extra></extra>",
                )
            )
        align_x = df_tpr[df_tpr["run"] == labels[0]].sort_values("bin_order")["bin_label"].tolist()
        _apply_detection_theme(fig_tpr, "TP rate by distance")
        fig_tpr.update_layout(
            xaxis_title="Distance bin",
            yaxis_title="TP rate",
            yaxis_range=[0, 1],
            barmode="group",
            xaxis=dict(tickangle=-35, categoryorder="array", categoryarray=align_x),
            hovermode="x unified",
        )
        fig_tpr.add_hline(y=0.5, line_dash="dash", line_color="rgba(0,0,0,0.25)")
        figures.append((fig_tpr, "Detection Stats compare distance panel in bar-chart mode: TP rate by distance."))

    df_fpr = _query_distance_rates_compare(con, views, metric="fpr")
    if not df_fpr.empty:
        fig_fpr = go.Figure()
        for i, lbl in enumerate(labels):
            d = df_fpr[df_fpr["run"] == lbl].sort_values("bin_order")
            c = _compare_color(i)
            fig_fpr.add_trace(
                go.Bar(
                    x=d["bin_label"],
                    y=d["fpr"],
                    name=lbl,
                    marker_color=c,
                    hovertemplate=f"{lbl}<br>%{{x}}<br>FP rate: %{{y:.2%}}<extra></extra>",
                )
            )
        align_x = df_fpr[df_fpr["run"] == labels[0]].sort_values("bin_order")["bin_label"].tolist()
        _apply_detection_theme(fig_fpr, "FP rate by distance")
        fig_fpr.update_layout(
            xaxis_title="Distance bin",
            yaxis_title="FP rate",
            yaxis_range=[0, 1],
            barmode="group",
            xaxis=dict(tickangle=-35, categoryorder="array", categoryarray=align_x),
            hovermode="x unified",
        )
        fig_fpr.add_hline(y=0.5, line_dash="dash", line_color="rgba(0,0,0,0.25)")
        figures.append((fig_fpr, "Detection Stats compare distance panel in bar-chart mode: FP rate by distance."))

    df_oc = _query_object_counts_compare(con, views)
    if not df_oc.empty:
        align_x = sorted(df_oc["bin_label"].unique(), key=_distance_bin_sort_key)
        pivot_oc = df_oc.pivot_table(index="bin_label", columns="run", values="n", aggfunc="sum", fill_value=0).reindex(align_x, fill_value=0)
        fig_oc = go.Figure()
        for j, rl in enumerate([r for r in labels if r in pivot_oc.columns]):
            c = _compare_color(j)
            fig_oc.add_trace(
                go.Bar(
                    x=align_x,
                    y=pivot_oc[rl].values,
                    name=str(rl),
                    marker_color=c,
                    hovertemplate=f"{rl}<br>%{{x}}<br>Count: %{{y:.0f}}<extra></extra>",
                )
            )
        _apply_detection_theme(fig_oc, "Object count by distance bin")
        fig_oc.update_layout(
            xaxis_title="Distance bin",
            yaxis_title="Count",
            barmode="group",
            xaxis=dict(tickangle=-35, categoryorder="array", categoryarray=align_x),
            hovermode="x unified",
        )
        figures.append((fig_oc, "Detection Stats compare object-count-by-distance panel in bar-chart mode."))
    return figures


def _build_detection_tpr_figures(
    con: duckdb.DuckDBPyConnection,
    views: Sequence[Tuple[str, str]],
) -> List[Tuple[go.Figure, str]]:
    figures: List[Tuple[go.Figure, str]] = []
    labels = [lbl for lbl, _ in views]
    if len(views) == 1:
        df_tpr = _query_tpr_by_label(con, views[0][1], _DEFAULT_MAX_EVAL_RANGE)
        if df_tpr.empty:
            return figures
        fig = px.bar(
            df_tpr,
            x="label",
            y="tpr",
            title=f"Total TP rate within {_DEFAULT_MAX_EVAL_RANGE} [m]",
            labels={"tpr": "TP Rate", "label": "Label"},
            color_discrete_sequence=[_COMPARE_RUN_COLORS[0]],
        )
        fig.update_traces(marker_color=_COMPARE_RUN_COLORS[0])
        _apply_detection_theme(fig, f"Total TP rate within {_DEFAULT_MAX_EVAL_RANGE} [m]")
        fig.update_layout(yaxis_range=[0, 1.2])
        fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(0,0,0,0.2)")
        figures.append((fig, "Default Detection Stats TP-rate panel: bar chart per object class."))
        return figures

    dfs = []
    for lbl, view_name in views:
        df = _query_tpr_by_label(con, view_name, _DEFAULT_MAX_EVAL_RANGE)
        if df.empty:
            continue
        df["run"] = lbl
        dfs.append(df)
    if not dfs:
        return figures
    df_all = pd.concat(dfs, ignore_index=True)
    cats = sorted(df_all["label"].astype(str).unique())
    fig = _tpr_spider_compare_figure(df_all, cats, "TP rate (<=50 m)", labels, height=360)
    figures.append((fig, "Default compare Detection Stats TP-rate panel: spider chart per object class."))
    return figures


def _build_detection_mean_error_figures(
    con: duckdb.DuckDBPyConnection,
    views: Sequence[Tuple[str, str]],
) -> List[Tuple[go.Figure, str]]:
    figures: List[Tuple[go.Figure, str]] = []
    labels = [lbl for lbl, _ in views]
    if not _views_have_error_columns(con, [view for _, view in views]):
        return figures
    if len(views) == 1:
        df = _query_mean_error_by_label(con, views[0][1], _DEFAULT_MAX_EVAL_RANGE)
        if df.empty:
            return figures
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["label"], y=df["mean_abs_x_error"], name="X Error", marker_color=_compare_color(0)))
        fig.add_trace(go.Bar(x=df["label"], y=df["mean_abs_y_error"], name="Y Error", marker_color=_compare_color(1)))
        fig.add_trace(go.Bar(x=df["label"], y=df["mean_abs_yaw_error"], name="Yaw Error", marker_color=_compare_color(2)))
        _apply_detection_theme(fig, f"Mean Error within {_DEFAULT_MAX_EVAL_RANGE} [m]")
        fig.update_layout(xaxis_title="Label", yaxis_title="Error [m] or [rad]", barmode="group")
        figures.append((fig, "Default Detection Stats mean-error panel: grouped bars for X/Y/Yaw."))
        return figures

    dfs = []
    for lbl, view_name in views:
        df = _query_mean_error_by_label(con, view_name, _DEFAULT_MAX_EVAL_RANGE)
        if df.empty:
            continue
        df["run"] = lbl
        dfs.append(df)
    if not dfs:
        return figures
    df_err_melt = pd.concat(dfs, ignore_index=True)
    cats = sorted(df_err_melt["label"].astype(str).unique())
    err_specs = [
        ("Mean |x error| (within 50 m)", "mean_abs_x_error", "Mean |x error| (m)", ".3f"),
        ("Mean |y error| (within 50 m)", "mean_abs_y_error", "Mean |y error| (m)", ".3f"),
        ("Mean |yaw error| (within 50 m)", "mean_abs_yaw_error", "Mean |yaw error| (rad)", ".4f"),
    ]
    for chart_title, col, hover_lbl, tfmt in err_specs:
        figures.append(
            (
                _scalar_metric_spider_compare_figure(df_err_melt, cats, chart_title, labels, col, hover_lbl, height=400, tickformat=tfmt),
                f"Default compare Detection Stats mean-error panel: spider chart for {hover_lbl}.",
            )
        )
    return figures


def _query_distance_rates_single(con: duckdb.DuckDBPyConnection, view_name: str) -> pd.DataFrame:
    query = f"""
    WITH stats AS (
        SELECT
            distance_bin,
            COUNT(*) FILTER (WHERE source='GT' AND status IN ('TP','FN')) AS gt_total,
            COUNT(*) FILTER (WHERE source='GT' AND status='TP') AS tp_gt,
            COUNT(*) FILTER (WHERE source='EST' AND status IN ('TP','FP')) AS est_total,
            COUNT(*) FILTER (WHERE source='EST' AND status='FP') AS fp_est
        FROM {view_name}
        GROUP BY distance_bin
    )
    SELECT
        distance_bin,
        CASE WHEN gt_total > 0 THEN CAST(tp_gt AS DOUBLE) / gt_total ELSE 0 END AS tpr,
        CASE WHEN est_total > 0 THEN CAST(fp_est AS DOUBLE) / est_total ELSE 0 END AS fpr
    FROM stats
    """
    df = con.execute(query).df()
    return _decorate_distance_bins(df)


def _query_distance_rates_compare(
    con: duckdb.DuckDBPyConnection,
    views: Sequence[Tuple[str, str]],
    *,
    metric: str,
) -> pd.DataFrame:
    frames = []
    for lbl, view_name in views:
        query = f"""
        WITH stats AS (
            SELECT
                distance_bin,
                COUNT(*) FILTER (WHERE source='GT' AND status IN ('TP','FN')) AS gt_total,
                COUNT(*) FILTER (WHERE source='GT' AND status='TP') AS tp_gt,
                COUNT(*) FILTER (WHERE source='EST' AND status IN ('TP','FP')) AS est_total,
                COUNT(*) FILTER (WHERE source='EST' AND status='FP') AS fp_est
            FROM {view_name}
            GROUP BY distance_bin
        )
        SELECT
            distance_bin,
            CASE
                WHEN {'gt_total' if metric == 'tpr' else 'est_total'} > 0
                THEN CAST({'tp_gt' if metric == 'tpr' else 'fp_est'} AS DOUBLE) / {'gt_total' if metric == 'tpr' else 'est_total'}
                ELSE 0
            END AS {metric}
        FROM stats
        """
        df = con.execute(query).df()
        if df.empty:
            continue
        df["run"] = lbl
        frames.append(_decorate_distance_bins(df))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _query_object_counts_single(con: duckdb.DuckDBPyConnection, view_name: str) -> pd.DataFrame:
    query = f"""
    SELECT distance_bin, label, COUNT(*) AS n
    FROM {view_name}
    GROUP BY distance_bin, label
    """
    return _decorate_distance_bins(con.execute(query).df())


def _query_object_counts_compare(con: duckdb.DuckDBPyConnection, views: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    frames = []
    for lbl, view_name in views:
        query = f"""
        SELECT distance_bin, COUNT(*) AS n
        FROM {view_name}
        GROUP BY distance_bin
        """
        df = con.execute(query).df()
        if df.empty:
            continue
        df["run"] = lbl
        frames.append(_decorate_distance_bins(df))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _query_tpr_by_label(con: duckdb.DuckDBPyConnection, view_name: str, max_range: int) -> pd.DataFrame:
    query = f"""
    SELECT
        label,
        CASE
            WHEN COUNT(*) FILTER (WHERE source='GT' AND status IN ('TP','FN')) > 0
            THEN CAST(COUNT(*) FILTER (WHERE source='GT' AND status='TP') AS DOUBLE)
                 / COUNT(*) FILTER (WHERE source='GT' AND status IN ('TP','FN'))
            ELSE 0
        END AS tpr
    FROM {view_name}
    WHERE dist_h < {int(max_range)}
    GROUP BY label
    ORDER BY label
    """
    return con.execute(query).df()


def _query_mean_error_by_label(con: duckdb.DuckDBPyConnection, view_name: str, max_range: int) -> pd.DataFrame:
    query = f"""
    SELECT
        label,
        AVG(ABS(CAST(x_error AS DOUBLE))) FILTER (WHERE status = 'TP' AND x_error IS NOT NULL) AS mean_abs_x_error,
        AVG(ABS(CAST(y_error AS DOUBLE))) FILTER (WHERE status = 'TP' AND y_error IS NOT NULL) AS mean_abs_y_error,
        AVG(ABS(CAST(yaw_error AS DOUBLE))) FILTER (WHERE status = 'TP' AND yaw_error IS NOT NULL) AS mean_abs_yaw_error
    FROM {view_name}
    WHERE dist_h < {int(max_range)}
    GROUP BY label
    ORDER BY label
    """
    return con.execute(query).df()


def _build_detection_perception_diff_figures(
    con: duckdb.DuckDBPyConnection,
    views: Sequence[Tuple[str, str]],
) -> List[Tuple[go.Figure, str]]:
    if len(views) < 2:
        return []
    figures: List[Tuple[go.Figure, str]] = []
    base_view = views[0][1]
    for lbl, comp_view in views[1:]:
        df_obj = _query_perception_diff_objects(con, base_view, comp_view)
        if df_obj.empty:
            continue
        h_imp = _baobab_hierarchy_from_objects(df_obj, "improved", f"Improved ({lbl} vs A)", 15, 10)
        h_deg = _baobab_hierarchy_from_objects(df_obj, "degraded", f"Degraded ({lbl} vs A)", 15, 10)
        if not h_imp.empty and "n" in h_imp.columns:
            fig_imp = px.sunburst(
                h_imp,
                path=["root", "scen_g", "fr_display", "label"],
                values="n",
                color="n",
                color_continuous_scale=[[0.0, "#f7fcf5"], [1.0, "#1a9850"]],
                title=f"Sunburst: improved (n = {int(h_imp['n'].sum())} GT objects)",
            )
            _apply_detection_theme(fig_imp, f"Sunburst: improved ({lbl} vs A)")
            figures.append((fig_imp, f"Perception diff sunburst for improved objects: {lbl} vs baseline A."))
        if not h_deg.empty and "n" in h_deg.columns:
            fig_deg = px.sunburst(
                h_deg,
                path=["root", "scen_g", "fr_display", "label"],
                values="n",
                color="n",
                color_continuous_scale=[[0.0, "#fff5f0"], [1.0, "#d73027"]],
                title=f"Sunburst: degraded (n = {int(h_deg['n'].sum())} GT objects)",
            )
            _apply_detection_theme(fig_deg, f"Sunburst: degraded ({lbl} vs A)")
            figures.append((fig_deg, f"Perception diff sunburst for degraded objects: {lbl} vs baseline A."))

        df_by_label, scen_agg, df_frame_sorted = _query_perception_diff_lens_tables(con, base_view, comp_view)
        root_lens = f"{lbl} vs A"
        if not df_by_label.empty:
            tdf_l = _comparison_lens_treemap_df(
                df_by_label["label"],
                df_by_label["improved_cnt"],
                df_by_label["degraded_cnt"],
                root_lens,
            )
            fig_l = _comparison_lens_treemap_figure(tdf_l, "By class")
            if fig_l is not None:
                figures.append((fig_l, f"Perception diff comparison lens by class: {lbl} vs baseline A."))
        if not scen_agg.empty:
            tdf_s = _comparison_lens_treemap_df(
                scen_agg["scenario_name"].astype(str),
                scen_agg["improved_cnt"],
                scen_agg["degraded_cnt"],
                root_lens,
            )
            fig_s = _comparison_lens_treemap_figure(tdf_s, "By scenario")
            if fig_s is not None:
                figures.append((fig_s, f"Perception diff comparison lens by scenario: {lbl} vs baseline A."))
        if not df_frame_sorted.empty:
            fr_cap = 36
            fr_top = df_frame_sorted.head(fr_cap).copy()
            nms = (fr_top["scenario_name"].astype(str).str.slice(0, 26) + "\n· f" + fr_top["frame_index"].astype(str)).tolist()
            ims = fr_top["improved_cnt"].astype(float).tolist()
            dgs = fr_top["degraded_cnt"].astype(float).tolist()
            rest = df_frame_sorted.iloc[fr_cap:]
            if not rest.empty:
                io = float(rest["improved_cnt"].sum())
                do = float(rest["degraded_cnt"].sum())
                if io > 0 or do > 0:
                    nms.append(f"Other frames\n({len(rest)} frames)")
                    ims.append(io)
                    dgs.append(do)
            tdf_f = _comparison_lens_treemap_df(pd.Series(nms), pd.Series(ims), pd.Series(dgs), root_lens)
            fig_f = _comparison_lens_treemap_figure(tdf_f, "By frame")
            if fig_f is not None:
                figures.append((fig_f, f"Perception diff comparison lens by frame: {lbl} vs baseline A."))
    return figures


def _query_perception_diff_objects(
    con: duckdb.DuckDBPyConnection,
    base_view: str,
    comp_view: str,
) -> pd.DataFrame:
    query = f"""
    WITH base_gt AS (
        SELECT
            t4dataset_id,
            frame_index,
            uuid AS gt_uuid,
            COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_base,
            COALESCE(MAX(try_cast(suite_name AS VARCHAR)), '') AS suite_name,
            COALESCE(MAX(try_cast(scenario_name AS VARCHAR)), '') AS scenario_name,
            COALESCE(MAX(try_cast(t4dataset_name AS VARCHAR)), '') AS t4dataset_name
        FROM {base_view}
        WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL
        GROUP BY 1, 2, 3
    ),
    comp_gt AS (
        SELECT
            t4dataset_id,
            frame_index,
            uuid AS gt_uuid,
            COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_comp,
            COALESCE(MAX(try_cast(suite_name AS VARCHAR)), '') AS suite_name,
            COALESCE(MAX(try_cast(scenario_name AS VARCHAR)), '') AS scenario_name,
            COALESCE(MAX(try_cast(t4dataset_name AS VARCHAR)), '') AS t4dataset_name
        FROM {comp_view}
        WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL
        GROUP BY 1, 2, 3
    ),
    joined AS (
        SELECT
            COALESCE(CAST(b.t4dataset_id AS VARCHAR), CAST(c.t4dataset_id AS VARCHAR)) AS t4dataset_id,
            COALESCE(CAST(b.frame_index AS VARCHAR), CAST(c.frame_index AS VARCHAR)) AS frame_index,
            COALESCE(b.gt_uuid, c.gt_uuid) AS gt_uuid,
            COALESCE(b.tp_base, FALSE) AS tp_base,
            COALESCE(c.tp_comp, FALSE) AS tp_comp,
            COALESCE(b.suite_name, c.suite_name, '') AS suite_name,
            COALESCE(b.scenario_name, c.scenario_name, '') AS scenario_name,
            COALESCE(b.t4dataset_name, c.t4dataset_name, '') AS t4dataset_name
        FROM base_gt b
        FULL OUTER JOIN comp_gt c
            ON b.t4dataset_id = c.t4dataset_id
           AND b.frame_index = c.frame_index
           AND b.gt_uuid = c.gt_uuid
    ),
    obj_attrs AS (
        SELECT
            t4dataset_id,
            frame_index,
            uuid,
            MAX(CAST(label AS VARCHAR)) AS label,
            MAX(dist_h) AS dist_h
        FROM {base_view}
        WHERE source = 'GT'
        GROUP BY 1, 2, 3
    )
    SELECT
        j.t4dataset_id,
        j.frame_index,
        j.gt_uuid,
        COALESCE(e.label, '') AS label,
        COALESCE(e.dist_h, 0.0) AS dist_h,
        {_DISTANCE_BIN_CASE.replace("dist_h", "COALESCE(e.dist_h, 0.0)")} AS distance_bin,
        j.suite_name,
        j.scenario_name,
        j.t4dataset_name,
        CASE
            WHEN NOT j.tp_base AND j.tp_comp THEN 'improved'
            WHEN j.tp_base AND NOT j.tp_comp THEN 'degraded'
            WHEN j.tp_base AND j.tp_comp THEN 'both_tp'
            ELSE 'both_fn'
        END AS change_type,
        j.tp_base,
        j.tp_comp
    FROM joined j
    LEFT JOIN obj_attrs e
        ON CAST(j.t4dataset_id AS VARCHAR) = CAST(e.t4dataset_id AS VARCHAR)
       AND j.frame_index = CAST(e.frame_index AS VARCHAR)
       AND j.gt_uuid = e.uuid
    ORDER BY change_type, j.t4dataset_id, j.frame_index
    """
    try:
        return con.execute(query).df()
    except Exception:
        return pd.DataFrame()


def _query_perception_diff_lens_tables(
    con: duckdb.DuckDBPyConnection,
    base_view: str,
    comp_view: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    query_label = f"""
    WITH base_gt AS (
        SELECT t4dataset_id, frame_index, uuid AS gt_uuid, COALESCE(MAX(try_cast(label AS VARCHAR)), '') AS label,
               COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_base
        FROM {base_view}
        WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL
        GROUP BY 1, 2, 3
    ),
    comp_gt AS (
        SELECT t4dataset_id, frame_index, uuid AS gt_uuid, COALESCE(MAX(try_cast(label AS VARCHAR)), '') AS label,
               COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_comp
        FROM {comp_view}
        WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL
        GROUP BY 1, 2, 3
    ),
    joined AS (
        SELECT COALESCE(b.label, c.label) AS label, COALESCE(b.tp_base, FALSE) AS tp_base, COALESCE(c.tp_comp, FALSE) AS tp_comp
        FROM base_gt b FULL OUTER JOIN comp_gt c
          ON b.t4dataset_id = c.t4dataset_id AND b.frame_index = c.frame_index AND b.gt_uuid = c.gt_uuid
    )
    SELECT label,
           CAST(COUNT(*) FILTER (WHERE NOT tp_base AND tp_comp) AS DOUBLE) AS improved_cnt,
           CAST(COUNT(*) FILTER (WHERE tp_base AND NOT tp_comp) AS DOUBLE) AS degraded_cnt
    FROM joined
    GROUP BY label
    """
    query_frame = f"""
    WITH base_gt AS (
        SELECT t4dataset_id, frame_index, uuid AS gt_uuid, COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_base,
               COALESCE(MAX(try_cast(scenario_name AS VARCHAR)), '') AS scenario_name
        FROM {base_view}
        WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL
        GROUP BY 1,2,3
    ),
    comp_gt AS (
        SELECT t4dataset_id, frame_index, uuid AS gt_uuid, COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_comp,
               COALESCE(MAX(try_cast(scenario_name AS VARCHAR)), '') AS scenario_name
        FROM {comp_view}
        WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL
        GROUP BY 1,2,3
    ),
    joined AS (
        SELECT COALESCE(CAST(b.t4dataset_id AS VARCHAR), CAST(c.t4dataset_id AS VARCHAR)) AS t4dataset_id,
               COALESCE(CAST(b.frame_index AS VARCHAR), CAST(c.frame_index AS VARCHAR)) AS frame_index,
               COALESCE(b.tp_base, FALSE) AS tp_base,
               COALESCE(c.tp_comp, FALSE) AS tp_comp,
               COALESCE(b.scenario_name, c.scenario_name, '') AS scenario_name
        FROM base_gt b FULL OUTER JOIN comp_gt c
          ON b.t4dataset_id = c.t4dataset_id AND b.frame_index = c.frame_index AND b.gt_uuid = c.gt_uuid
    )
    SELECT t4dataset_id, frame_index, scenario_name,
           CAST(COUNT(*) FILTER (WHERE NOT tp_base AND tp_comp) AS DOUBLE) AS improved_cnt,
           CAST(COUNT(*) FILTER (WHERE tp_base AND NOT tp_comp) AS DOUBLE) AS degraded_cnt
    FROM joined
    GROUP BY t4dataset_id, frame_index, scenario_name
    ORDER BY degraded_cnt DESC, improved_cnt DESC
    """
    query_scenario = f"""
    WITH base_gt AS (
        SELECT t4dataset_id, frame_index, uuid AS gt_uuid, COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_base,
               COALESCE(MAX(try_cast(scenario_name AS VARCHAR)), '') AS scenario_name
        FROM {base_view}
        WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL
        GROUP BY 1,2,3
    ),
    comp_gt AS (
        SELECT t4dataset_id, frame_index, uuid AS gt_uuid, COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_comp,
               COALESCE(MAX(try_cast(scenario_name AS VARCHAR)), '') AS scenario_name
        FROM {comp_view}
        WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL
        GROUP BY 1,2,3
    ),
    joined AS (
        SELECT COALESCE(b.scenario_name, c.scenario_name, '') AS scenario_name,
               COALESCE(b.tp_base, FALSE) AS tp_base,
               COALESCE(c.tp_comp, FALSE) AS tp_comp
        FROM base_gt b FULL OUTER JOIN comp_gt c
          ON b.t4dataset_id = c.t4dataset_id AND b.frame_index = c.frame_index AND b.gt_uuid = c.gt_uuid
    )
    SELECT scenario_name,
           CAST(COUNT(*) FILTER (WHERE NOT tp_base AND tp_comp) AS DOUBLE) AS improved_cnt,
           CAST(COUNT(*) FILTER (WHERE tp_base AND NOT tp_comp) AS DOUBLE) AS degraded_cnt
    FROM joined
    GROUP BY scenario_name
    ORDER BY degraded_cnt DESC, improved_cnt DESC
    """
    try:
        df_label = con.execute(query_label).df()
    except Exception:
        df_label = pd.DataFrame()
    try:
        df_scenario = con.execute(query_scenario).df()
    except Exception:
        df_scenario = pd.DataFrame()
    try:
        df_frame = con.execute(query_frame).df()
    except Exception:
        df_frame = pd.DataFrame()
    return df_label, df_scenario, df_frame


def _baobab_hierarchy_from_objects(
    df_obj: pd.DataFrame,
    change_type: str,
    root_label: str,
    max_scenarios: int,
    max_frames: int,
) -> pd.DataFrame:
    if df_obj.empty or "change_type" not in df_obj.columns:
        return pd.DataFrame()
    sub = df_obj[df_obj["change_type"] == change_type].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["scenario_name"] = sub["scenario_name"].fillna("").astype(str).replace("", "(no scenario)")
    sub["label"] = sub["label"].fillna("").astype(str).replace("", "(no label)")
    sub["frame_key"] = sub["t4dataset_id"].astype(str) + "|f" + sub["frame_index"].astype(str)
    leaf = sub.groupby(["scenario_name", "frame_key", "label"], dropna=False).size().reset_index(name="n")
    scen_tot = leaf.groupby("scenario_name")["n"].sum().sort_values(ascending=False)
    top_scen = set(scen_tot.head(max_scenarios).index.tolist())
    leaf["scen_g"] = leaf["scenario_name"].where(leaf["scenario_name"].isin(top_scen), "Other scenarios")
    out_parts = []
    for _, g in leaf.groupby("scen_g"):
        fr_tot = g.groupby("frame_key")["n"].sum().sort_values(ascending=False)
        top_fr = set(fr_tot.head(max_frames).index.tolist())
        g2 = g.copy()
        g2["fr_g"] = g2["frame_key"].where(g2["frame_key"].isin(top_fr), "Other frames")
        agg = g2.groupby(["scen_g", "fr_g", "label"], as_index=False)["n"].sum()
        out_parts.append(agg)
    out = pd.concat(out_parts, ignore_index=True)
    out["root"] = root_label
    out["fr_display"] = out["fr_g"].astype(str)
    return out


def _comparison_lens_treemap_df(names: pd.Series, improved: pd.Series, degraded: pd.Series, root_label: str) -> pd.DataFrame:
    rows = []
    for name, imp, deg in zip(names.astype(str), improved.astype(float), degraded.astype(float)):
        if imp > 0:
            rows.append({"root": root_label, "side": "Improved", "item": name, "n": float(imp)})
        if deg > 0:
            rows.append({"root": root_label, "side": "Degraded", "item": name, "n": float(deg)})
    if not rows:
        return pd.DataFrame(columns=["root", "side", "item", "n"])
    return pd.DataFrame(rows)


def _comparison_lens_treemap_figure(tdf: pd.DataFrame, title: str) -> Optional[go.Figure]:
    if tdf.empty or "n" not in tdf.columns:
        return None
    fig = px.treemap(
        tdf,
        path=["root", "side", "item"],
        values="n",
        color="side",
        color_discrete_map={"Improved": "#1a9850", "Degraded": "#d73027"},
    )
    fig.update_traces(
        textfont_size=12,
        textinfo="label+value+percent parent",
        hovertemplate=("<b>%{label}</b><br>GT objects: %{value:.0f}<br>% of parent: %{percentParent}<extra></extra>"),
        marker_line_width=1.5,
        marker_line_color="rgba(255,255,255,0.45)",
        root_color="rgba(240,240,245,0.95)",
    )
    _apply_detection_theme(fig, title)
    fig.update_layout(height=430, margin=dict(t=20, l=2, r=2, b=2), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def _views_have_error_columns(con: duckdb.DuckDBPyConnection, view_names: Sequence[str]) -> bool:
    if not view_names:
        return False
    sample_df = con.execute(f"SELECT * FROM {view_names[0]} LIMIT 1").df()
    return all(col in sample_df.columns for col in ["x_error", "y_error", "yaw_error"])


def _decorate_distance_bins(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "distance_bin" not in df.columns:
        return df
    df = df.copy()
    df["bin_order"] = df["distance_bin"].map(_distance_bin_sort_key)
    df["bin_label"] = df["distance_bin"]
    return df.sort_values("bin_order")


def _distance_bin_sort_key(label: str) -> int:
    try:
        return _distance_bin_order().index(str(label))
    except ValueError:
        return len(_distance_bin_order()) + 1


def _compare_color(index: int) -> str:
    return _COMPARE_RUN_COLORS[index % len(_COMPARE_RUN_COLORS)]


def _tpr_spider_compare_figure(
    df_all: pd.DataFrame,
    categories: List[str],
    title: str,
    run_order: List[str],
    *,
    height: int = 440,
) -> go.Figure:
    fig = go.Figure()
    for i, run_lbl in enumerate(run_order):
        sub = df_all[df_all["run"] == run_lbl].drop_duplicates("label").set_index("label")
        r_vals = [float(sub.loc[c, "tpr"]) if c in sub.index else 0.0 for c in categories]
        r_closed = r_vals + r_vals[:1]
        theta = categories + categories[:1]
        c = _compare_color(i)
        fig.add_trace(
            go.Scatterpolar(
                r=r_closed,
                theta=theta,
                name=str(run_lbl),
                line=dict(color=c, width=2),
                fillcolor=f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.12)",
                fill="toself",
                hovertemplate="%{theta}<br>TP rate: %{r:.2%}<extra></extra>",
            )
        )
    _apply_detection_theme(fig, title)
    fig.update_layout(
        height=height,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%", gridcolor="rgba(0,0,0,0.08)"),
            angularaxis=dict(tickfont=dict(size=10)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
    )
    return fig


def _scalar_metric_spider_compare_figure(
    df_all: pd.DataFrame,
    categories: List[str],
    title: str,
    run_order: List[str],
    value_col: str,
    hover_metric: str,
    *,
    height: int = 380,
    tickformat: str = ".3f",
) -> go.Figure:
    fig = go.Figure()
    max_r = 0.0
    traces_r: List[List[float]] = []
    for run_lbl in run_order:
        sub = df_all[df_all["run"] == run_lbl].drop_duplicates("label").set_index("label")
        r_vals = [float(sub.loc[c, value_col]) if c in sub.index and pd.notna(sub.loc[c, value_col]) else 0.0 for c in categories]
        traces_r.append(r_vals)
        if r_vals:
            max_r = max(max_r, max(r_vals))
    r_max = max(max_r * 1.08, 1.0)
    for i, run_lbl in enumerate(run_order):
        r_vals = traces_r[i]
        r_closed = r_vals + r_vals[:1]
        theta = categories + categories[:1]
        c = _compare_color(i)
        fig.add_trace(
            go.Scatterpolar(
                r=r_closed,
                theta=theta,
                name=str(run_lbl),
                line=dict(color=c, width=2),
                fillcolor=f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.12)",
                fill="toself",
                hovertemplate="%{theta}<br>" + hover_metric + ": %{r:" + tickformat + "}<extra></extra>",
            )
        )
    _apply_detection_theme(fig, title)
    fig.update_layout(
        height=height,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, r_max], tickformat=tickformat, gridcolor="rgba(0,0,0,0.08)"),
            angularaxis=dict(tickfont=dict(size=9)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
    )
    return fig


def _make_text_placeholder_figure(text: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=text, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=16, color="#475569"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=240,
        template="plotly_white",
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(248,250,252,0.9)",
        plot_bgcolor="rgba(255,255,255,0.95)",
    )
    return fig


def _build_score_view(df_raw: pd.DataFrame, criteria_idx: int) -> pd.DataFrame:
    return build_score_view(df_raw, criteria_idx)


def _create_eval_flat_view(con: duckdb.DuckDBPyConnection, parquet_path: str, view_name: str) -> None:
    query = f"""
    CREATE OR REPLACE VIEW {view_name} AS
    WITH src AS (
        SELECT * FROM parquet_scan('{parquet_path}')
        UNION BY NAME
        SELECT CAST(NULL AS VARCHAR) AS visibility,
               CAST(NULL AS VARCHAR) AS suite_name,
               CAST(NULL AS VARCHAR) AS scenario_name,
               CAST(NULL AS VARCHAR) AS t4dataset_name
        WHERE FALSE
    ),
    base AS (
        SELECT
            * REPLACE (coalesce(CAST(visibility AS VARCHAR), 'not available') AS visibility),
            sqrt(CAST(x AS DOUBLE)*CAST(x AS DOUBLE) + CAST(y AS DOUBLE)*CAST(y AS DOUBLE)) AS dist_h
        FROM src
        WHERE x IS NOT NULL AND y IS NOT NULL
    )
    SELECT
        *,
        {_DISTANCE_BIN_CASE} AS distance_bin
    FROM base
    """
    con.execute(query)


def _kpi_row_for_view(con: duckdb.DuckDBPyConnection, view_name: str) -> Optional[dict]:
    query = f"""
    SELECT
        COUNT(*) FILTER (WHERE source = 'GT' AND status = 'TP') AS tp_gt,
        COUNT(*) FILTER (WHERE source = 'GT' AND status = 'FN') AS fn,
        COUNT(*) FILTER (WHERE source = 'EST' AND status = 'TP') AS tp_est,
        COUNT(*) FILTER (WHERE source = 'EST' AND status = 'FP') AS fp
    FROM {view_name}
    WHERE dist_h < 50
    """
    row = con.execute(query).fetchone()
    if not row:
        return None
    tp_gt, fn, tp_est, fp = [int(x or 0) for x in row]
    gt_total = tp_gt + fn
    est_total = tp_est + fp
    tpr = (tp_gt / gt_total) if gt_total > 0 else None
    precision = (tp_est / est_total) if est_total > 0 else None
    recall = tpr
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    return {
        "tp": tp_gt,
        "fp": fp,
        "fn": fn,
        "tpr": tpr,
        "precision": precision,
        "f1": f1,
    }


def _query_status_counts(con: duckdb.DuckDBPyConnection, views: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    parts = [
        f"SELECT '{lbl}' AS run, label, status, COUNT(*) AS num "
        f"FROM {view_name} WHERE dist_h < 50 GROUP BY label, status"
        for lbl, view_name in views
    ]
    if not parts:
        return pd.DataFrame()
    query = " UNION ALL ".join(parts) + " ORDER BY run, label, status"
    return con.execute(query).df()


def _query_distance_tpr(con: duckdb.DuckDBPyConnection, views: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    frames = []
    for lbl, view_name in views:
        query = f"""
        WITH stats AS (
            SELECT
                distance_bin,
                COUNT(*) FILTER (WHERE source='GT' AND status IN ('TP','FN')) AS gt_total,
                COUNT(*) FILTER (WHERE source='GT' AND status='TP') AS tp_gt
            FROM {view_name}
            WHERE dist_h < 150
            GROUP BY distance_bin
        )
        SELECT
            '{lbl}' AS run,
            distance_bin,
            CASE WHEN gt_total > 0 THEN CAST(tp_gt AS DOUBLE) / gt_total ELSE 0 END AS tpr
        FROM stats
        """
        frames.append(con.execute(query).df())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _apply_criteria_theme(fig: go.Figure, title: str) -> None:
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(size=16, color="#0f172a"), x=0, xanchor="left", pad=dict(t=8, b=12)),
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", size=12, color="#334155"),
        paper_bgcolor="rgba(248, 250, 252, 0.92)",
        plot_bgcolor="rgba(255, 255, 255, 0.95)",
        margin=dict(l=56, r=28, t=72, b=52),
        height=420,
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="system-ui"),
        legend=dict(
            title_text="",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.7)",
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.25)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.25)", zeroline=False)


def _apply_detection_theme(fig: go.Figure, title: str) -> None:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1f2937")),
        font=dict(family='"Inter", "Segoe UI", sans-serif', size=11),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.6)",
        margin=dict(t=48, b=40, l=52, r=24),
        height=390,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(
        tickfont=dict(size=11),
        title_font=dict(size=12),
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=True,
        zerolinecolor="rgba(0,0,0,0.15)",
    )
    fig.update_yaxes(
        tickfont=dict(size=11),
        title_font=dict(size=12),
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=True,
        zerolinecolor="rgba(0,0,0,0.15)",
    )


def _plotly_figure_to_image(fig: go.Figure, content_width: float, image_reader_cls):
    from reportlab.platypus import Image

    png_bytes = fig.to_image(format="png", width=1400, height=800, scale=2)
    image_buffer = io.BytesIO(png_bytes)
    reader = image_reader_cls(image_buffer)
    img_width, img_height = reader.getSize()
    target_width = content_width
    target_height = target_width * (img_height / img_width)
    image_buffer.seek(0)
    return Image(image_buffer, width=target_width, height=target_height)


def _styled_table(rows: Any, content_width: float):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Table, TableStyle

    col_width_weights = None
    if isinstance(rows, dict):
        col_width_weights = rows.get("col_width_weights")
        rows = rows.get("rows", [])
    if not rows:
        rows = [["No data"]]
    ncols = max(len(row) for row in rows)
    styles = getSampleStyleSheet()
    header_style = styles["BodyText"].clone("table_header")
    header_style.fontName = "Helvetica-Bold"
    header_style.fontSize = 8.5
    header_style.leading = 10
    body_style = styles["BodyText"].clone("table_body")
    body_style.fontName = "Helvetica"
    body_style.fontSize = 8.2
    body_style.leading = 9.6
    body_style.textColor = colors.HexColor("#0f172a")
    normalized = []
    for row_idx, row in enumerate(rows):
        padded = list(row) + [""] * (ncols - len(row))
        cell_style = header_style if row_idx == 0 else body_style
        normalized.append([
            _table_paragraph(cell, cell_style)
            for cell in padded
        ])
    if col_width_weights and len(col_width_weights) == ncols:
        total = sum(col_width_weights) or 1.0
        col_widths = [content_width * (w / total) for w in col_width_weights]
    else:
        col_width = content_width / ncols
        col_widths = [col_width] * ncols
    table = Table(normalized, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _distance_bin_order() -> List[str]:
    return [
        "[0,10)",
        "[10,20)",
        "[20,30)",
        "[30,40)",
        "[40,50)",
        "[50,60)",
        "[60,70)",
        "[70,80)",
        "[80,90)",
        "[90,100)",
        "[100,110)",
        "[110,120)",
        "[120,130)",
        "[130,140)",
        "[140,150)",
        "[150,inf)",
    ]


def _ensure_reportlab_available() -> Optional[str]:
    try:
        import reportlab  # noqa: F401
    except ImportError:
        return "PDF export requires the `reportlab` package to be installed."
    return None


def _slugify(value: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_") or "report"


def _fmt_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.2f}"


def _fmt_percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{100.0 * float(value):.1f}%"


def _summarize_filter_values(values: Optional[Iterable[Any]], *, empty_label: str = "All") -> str:
    if values is None:
        return empty_label
    vals = [str(v) for v in values if str(v).strip() != ""]
    if not vals:
        return empty_label
    if len(vals) <= 6:
        return ", ".join(vals)
    return ", ".join(vals[:6]) + f", ... (+{len(vals) - 6} more)"


def _shorten_scenario_name(value: str, *, max_len: int = 52) -> str:
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _table_paragraph(value: Any, style: Any):
    from reportlab.platypus import Paragraph

    text = html.escape("" if value is None else str(value)).replace("\n", "<br/>")
    return Paragraph(text, style)
