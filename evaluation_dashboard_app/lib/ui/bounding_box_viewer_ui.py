"""Bounding box viewer: static HTML legends for BEV status colors."""

from __future__ import annotations

# Swatch hues mirror the box colors the viewers draw (static/bbox_viewer.css --sw-*),
# so they stay fixed in both themes; only the chip surface/border/text follow tokens.
_SW_GT_TP = "#00cc66"
_SW_GT_FN = "#ff9933"
_SW_EST_TP = "#66b3ff"
_SW_EST_FP = "#ff6666"


def bev_status_legend_markup() -> str:
    """Single-run status color legend (GT/TP, GT/FN, EST/TP, EST/FP)."""
    return (
        '<div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap; '
        "margin-bottom:10px; padding:10px 14px; background:var(--t4-surface-2); border-radius:8px; font-size:0.9em; "
        "border:1px solid var(--t4-border);\">"
        "<span style=\"font-weight:700;color:var(--t4-text);\">Status:</span> "
        f'<span style="background:{_SW_GT_TP};color:#000;padding:2px 8px;border-radius:4px;">GT/TP</span> '
        f'<span style="background:{_SW_GT_FN};color:#000;padding:2px 8px;border-radius:4px;">GT/FN</span> '
        f'<span style="background:{_SW_EST_TP};color:#000;padding:2px 8px;border-radius:4px;">EST/TP</span> '
        f'<span style="background:{_SW_EST_FP};color:#fff;padding:2px 8px;border-radius:4px;">EST/FP</span>'
        "</div>"
    )


def bev_overlay_line_and_status_legend_markup(line_hint_html: str) -> str:
    """
    Multi-run overlay: line style per run + same status colors.
    line_hint_html: pre-built inner HTML (e.g. <strong>Run A</strong> — solid …).
    """
    return (
        '<div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap; '
        "margin-bottom:10px; padding:10px 14px; background:var(--t4-surface-2); border-radius:8px; font-size:0.9em; "
        "border:1px solid var(--t4-border);\">"
        '<span style="font-weight:700;color:var(--t4-text);">Line = Run:</span> '
        f'<span style="color:var(--t4-text-2);">{line_hint_html}</span>'
        ' &nbsp;&nbsp; '
        '<span style="font-weight:700;color:var(--t4-text);">Color = Status:</span> '
        f'<span style="background:{_SW_GT_TP};color:#000;padding:2px 8px;border-radius:4px;">GT/TP</span> '
        f'<span style="background:{_SW_GT_FN};color:#000;padding:2px 8px;border-radius:4px;">GT/FN</span> '
        f'<span style="background:{_SW_EST_TP};color:#000;padding:2px 8px;border-radius:4px;">EST/TP</span> '
        f'<span style="background:{_SW_EST_FP};color:#fff;padding:2px 8px;border-radius:4px;">EST/FP</span>'
        "</div>"
    )
