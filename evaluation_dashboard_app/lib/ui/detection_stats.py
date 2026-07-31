"""Detection Stats page: CSS injection, KPI cards, section headers, spot loaders."""

from __future__ import annotations

from contextlib import contextmanager

import streamlit as st


def inject_detection_stats_styles() -> None:
    """Section headers, loading banner, spot loader (inject once per page run)."""
    st.markdown(
        """
<style>
.section-header { border-left: 4px solid var(--t4-accent-2); padding-left: 12px; font-weight: 700; font-size: 1.02rem; color: var(--t4-text); margin: 1.35rem 0 0.65rem 0; letter-spacing: -0.02em; }
.section-block { margin-bottom: 1.5rem; }
.run-chip { display: inline-block; background: var(--t4-chip-bg); border: 1px solid var(--t4-border); border-radius: 999px; padding: 0.35rem 0.85rem; font-size: 0.875rem; margin: 0.25rem 0.25rem 0.25rem 0; color: var(--t4-text-2); }
.run-chip strong { color: var(--t4-text); }
@keyframes ds-load-shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
@keyframes ds-load-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.78; }
}
/* Glow/beacon tints ride on the teal accent tokens (translucent in dark) so the
   pulse tints the canvas instead of flashing a bright block on #0e1117. */
@keyframes ds-banner-glow {
  0%, 100% {
    box-shadow:
      0 0 0 1px var(--t4-accent-2-border),
      0 4px 14px var(--t4-accent-2-soft),
      0 0 28px var(--t4-accent-2-soft);
  }
  50% {
    box-shadow:
      0 0 0 2px var(--t4-accent-2-border),
      0 6px 22px var(--t4-accent-2-border),
      0 0 40px var(--t4-accent-2-soft);
  }
}
@keyframes ds-dot-beacon {
  0%, 100% { transform: scale(1); box-shadow: 0 0 0 0 var(--t4-accent-2-border); }
  55% { transform: scale(1.12); box-shadow: 0 0 0 14px transparent; }
}
@keyframes ds-banner-bg-shift {
  0% { background-position: 0% 40%; }
  100% { background-position: 100% 60%; }
}
.ds-page-loading-banner {
  display: flex; align-items: flex-start; gap: 1rem;
  padding: 1rem 1.2rem; margin: 0 0 1.15rem 0;
  border-radius: 14px;
  border: 2px solid var(--t4-accent-2);
  background: linear-gradient(125deg, var(--t4-accent-2-soft) 0%, var(--t4-accent-2-border) 22%, var(--t4-accent-soft) 48%, var(--t4-surface-2) 72%, var(--t4-surface) 100%);
  background-size: 240% 240%;
  animation: ds-banner-glow 2.2s ease-in-out infinite, ds-banner-bg-shift 6s ease-in-out infinite alternate;
}
.ds-page-loading-banner .ds-plb-head {
  display: flex; align-items: center; flex-wrap: wrap; gap: 0.5rem 0.65rem;
}
.ds-page-loading-banner .ds-plb-badge {
  flex-shrink: 0;
  font-size: 0.62rem; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--t4-accent-on);
  background: linear-gradient(135deg, var(--t4-accent-2) 0%, var(--t4-accent) 100%);
  padding: 0.28rem 0.55rem; border-radius: 6px;
  box-shadow: var(--t4-shadow-sm);
  animation: ds-load-pulse 1.4s ease-in-out infinite;
}
.ds-page-loading-banner .ds-plb-text {
  flex: 1; min-width: 0;
  font-size: 1.08rem; font-weight: 800; color: var(--t4-text); letter-spacing: -0.02em;
  line-height: 1.25;
  text-shadow: 0 1px 0 var(--t4-overlay);
}
.ds-page-loading-banner .ds-plb-sub {
  display: block; font-size: 0.82rem; font-weight: 600; color: var(--t4-text-2); margin-top: 0.35rem;
  line-height: 1.4;
}
.ds-plb-shimmer-wrap {
  height: 7px; border-radius: 999px; overflow: hidden;
  background: var(--t4-accent-2-soft); margin-top: 0.65rem;
  border: 1px solid var(--t4-accent-2-border);
}
.ds-plb-shimmer {
  height: 100%; width: 100%;
  background: linear-gradient(
    90deg,
    transparent 0%,
    var(--t4-accent-2-soft) 38%,
    var(--t4-accent-2) 50%,
    var(--t4-accent-2-soft) 62%,
    transparent 100%
  );
  background-size: 200% 100%;
  animation: ds-load-shimmer 1.35s ease-in-out infinite;
}
.ds-plb-dot {
  width: 14px; height: 14px; border-radius: 50%;
  background: radial-gradient(circle at 30% 30%, var(--t4-accent-2), var(--t4-accent));
  flex-shrink: 0; margin-top: 0.15rem;
  border: 2px solid var(--t4-surface);
  animation: ds-dot-beacon 1.5s ease-out infinite;
}
@keyframes ds-spot-bar-slide {
  0% { transform: translateX(-130%); }
  100% { transform: translateX(400%); }
}
.ds-spot-loader {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin: 0.2rem 0 0.7rem 0;
  padding: 0.5rem 0.75rem;
  border-radius: 10px;
  border: 1px solid var(--t4-accent-2-border);
  background: linear-gradient(100deg, var(--t4-accent-2-soft) 0%, var(--t4-surface-2) 55%, var(--t4-accent-soft) 100%);
  box-shadow: var(--t4-shadow-sm);
}
.ds-spot-loader .ds-spot-ping {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--t4-accent-2);
  flex-shrink: 0;
  animation: ds-dot-beacon 1.35s ease-out infinite;
}
.ds-spot-loader .ds-spot-working {
  font-size: 0.58rem;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--t4-accent-on);
  background: linear-gradient(135deg, var(--t4-accent-2), var(--t4-accent));
  padding: 0.2rem 0.45rem;
  border-radius: 4px;
  flex-shrink: 0;
}
.ds-spot-loader .ds-spot-label {
  font-size: 0.8rem;
  font-weight: 700;
  color: var(--t4-text);
  letter-spacing: -0.015em;
  flex: 1 1 120px;
  min-width: 0;
}
.ds-spot-loader .ds-spot-bar {
  flex: 1 1 72px;
  max-width: 168px;
  height: 5px;
  border-radius: 999px;
  overflow: hidden;
  background: var(--t4-accent-2-soft);
}
.ds-spot-loader .ds-spot-bar-inner {
  display: block;
  height: 100%;
  width: 34%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--t4-accent-2), var(--t4-accent));
  animation: ds-spot-bar-slide 1s ease-in-out infinite;
}
@media (prefers-reduced-motion: reduce) {
  .ds-page-loading-banner { animation: none; background-size: auto; }
  .ds-plb-shimmer { animation: none; opacity: 0.85; }
  .ds-plb-dot { animation: none; }
  .ds-plb-badge { animation: none; }
  .ds-spot-loader .ds-spot-bar-inner { animation: none; transform: none; width: 100%; opacity: 0.4; }
  .ds-spot-loader .ds-spot-ping { animation: none; }
}
</style>

        """,
        unsafe_allow_html=True,
    )


def inject_detection_stats_kpi_styles() -> None:
    """KPI card grid styles (call before each kpi-wrap block that needs it)."""
    st.markdown(
        """
<style>
.kpi-wrap { display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: stretch; margin-bottom: 1.5rem; }
.kpi-card {
    background: var(--t4-card-bg);
    border: 1px solid var(--t4-border);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    min-width: 360px;
    min-height: 200px;
    box-shadow: var(--t4-shadow-sm);
    display: flex;
    flex-direction: column;
}
.kpi-title { font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: var(--t4-text-3); margin-bottom: 1rem; }
.kpi-row { display: flex; gap: 2rem; margin-bottom: 0.85rem; }
.kpi-row:last-child { margin-bottom: 0; }
.kpi-cell { display: flex; flex-direction: column; align-items: flex-start; min-width: 4.5rem; min-height: 2.6rem; }
.kpi-label { font-size: 0.8rem; color: var(--t4-muted); text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 0.25rem; }
.kpi-value { font-size: 1.5rem; font-weight: 700; color: var(--t4-text); font-variant-numeric: tabular-nums; line-height: 1.2; }
.kpi-delta-inline { display: block; font-size: 0.8rem; font-weight: 600; margin-top: 0.2rem; font-variant-numeric: tabular-nums; min-height: 1.1rem; }
.kpi-delta-inline.delta-pos { color: var(--t4-ok); }
.kpi-delta-inline.delta-neg { color: var(--t4-bad); }
.kpi-empty { font-size: 1rem; color: var(--t4-muted); font-style: italic; }

/* KPI Comparison Analysis */
.kpi-analysis {
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin: 0.5rem 0 1.5rem 0;
    font-size: 0.92rem;
    line-height: 1.6;
    color: var(--t4-text);
    border-left: 4px solid var(--t4-neutral-border);
}
.kpi-analysis-good {
    background: var(--t4-ok-bg);
    border-left-color: var(--t4-ok);
}
.kpi-analysis-bad {
    background: var(--t4-bad-bg);
    border-left-color: var(--t4-bad);
}
.kpi-analysis-warn {
    background: var(--t4-warn-bg);
    border-left-color: var(--t4-warn);
}
.kpi-analysis-neutral {
    background: var(--t4-neutral-bg);
    border-left-color: var(--t4-neutral);
}
.kpi-analysis-verdict { font-size: 0.98rem; }
.kpi-analysis-verdict.kpi-analysis-good { color: var(--t4-ok); }
.kpi-analysis-verdict.kpi-analysis-bad { color: var(--t4-bad); }
.kpi-analysis-verdict.kpi-analysis-warn { color: var(--t4-warn); }
.kpi-analysis-verdict.kpi-analysis-neutral { color: var(--t4-neutral); }
.kpi-analysis-recommendation {
    margin: 0.45rem 0 0.15rem 0;
    color: var(--t4-text-2);
}
.kpi-analysis-note {
    font-size: 0.84rem;
    color: var(--t4-muted);
    font-style: italic;
}
</style>

        """,
        unsafe_allow_html=True,
    )


def _pct_str(v):
    if v is None:
        return "—"
    p = min(100.0, v * 100)
    return f"{p:.0f}%" if abs(p - round(p)) < 0.05 else f"{p:.1f}%"


def _metric_cell(label: str, value: str, delta_str: str = "", delta_positive: bool | None = None) -> str:
    delta_span = ""
    if delta_str:
        cls = "kpi-delta-inline delta-pos" if delta_positive is True else "kpi-delta-inline delta-neg" if delta_positive is False else "kpi-delta-inline"
        delta_span = f'<span class="{cls}">{delta_str}</span>'
    return f'<div class="kpi-cell"><span class="kpi-label">{label}</span><span class="kpi-value">{value}</span>{delta_span}</div>'


def render_kpi_card(title: str, kpi: dict, css_id: str = "", deltas: dict | None = None) -> str:
    """deltas: optional dict with KPI keys (B - A). Shown inline in card."""
    if not kpi:
        return f'<div class="kpi-card" id="{css_id}"><div class="kpi-title">{title}</div><div class="kpi-empty">No data</div></div>'
    d = deltas or {}

    def _cell(label: str, val: str, delta_key: str, lower_is_better: bool = False):
        delta_val = d.get(delta_key)
        if delta_val is None:
            return _metric_cell(label, val)
        if delta_key in ("tpr", "fpr", "precision", "recall") and isinstance(delta_val, (int, float)):
            delta_str = f"{delta_val * 100:+.1f}%" if abs(delta_val) <= 1 else f"{delta_val:+.1f}%"
        elif delta_key == "f1":
            delta_str = f"{delta_val:+.3f}"
        else:
            delta_str = f"{delta_val:+d}" if isinstance(delta_val, int) else f"{delta_val:+.3f}"
        good = (delta_val >= 0 and not lower_is_better) or (delta_val <= 0 and lower_is_better)
        return _metric_cell(label, val, delta_str, good)

    row1 = "".join([
        _cell("GT", str(kpi.get("gt", "—")), "gt"),
        _cell("TP", str(kpi.get("tp", "—")), "tp"),
        _cell("FP", str(kpi.get("fp", "—")), "fp", lower_is_better=True),
        _cell("FN", str(kpi.get("fn", "—")), "fn", lower_is_better=True),
    ])
    f1_val = f"{kpi['f1']:.3f}" if kpi.get("f1") is not None else "—"
    row2 = "".join([
        _cell("Recall / TP rate", _pct_str(kpi.get("recall", kpi.get("tpr"))), "recall"),
        _cell("FP rate", _pct_str(kpi.get("fpr")), "fpr", lower_is_better=True),
        _cell("Precision", _pct_str(kpi.get("precision")), "precision"),
        _cell("F1", f1_val, "f1"),
    ])
    return f'''<div class="kpi-card" id="{css_id}">
        <div class="kpi-title">{title}</div>
        <div class="kpi-row">{row1}</div>
        <div class="kpi-row">{row2}</div>
    </div>'''


def section_header_html(title: str, caption: str = "") -> str:
    """HTML for a styled section header with optional caption."""
    if caption:
        return f'<div class="section-header">{title}</div><p style="margin-top: 0.25rem; margin-bottom: 0.75rem; font-size: 0.9rem; color: var(--t4-muted);">{caption}</p>'
    return f'<div class="section-header">{title}</div>'


def ds_spot_loading_markup(_label: str) -> str:
    """Spot loader HTML disabled (was: “Working here” + label); returns empty string."""
    return ""


@contextmanager
def ds_spot_loading(_label: str):
    """Spot loader context manager disabled (no-op); kept for call-site compatibility."""
    yield

def detection_stats_page_loading_banner_markup() -> str:
    """Top-of-page banner while queries and charts stream in."""
    return """
    <div class="ds-page-loading-banner" role="status" aria-live="polite">
      <span class="ds-plb-dot" aria-hidden="true"></span>
      <div style="flex:1;min-width:0;">
        <div class="ds-plb-head">
          <span class="ds-plb-badge">In progress</span>
          <span class="ds-plb-text">Crunching detection stats…</span>
        </div>
        <span class="ds-plb-sub">Hang tight — large Parquet files can take a moment.</span>
        <div class="ds-plb-shimmer-wrap"><div class="ds-plb-shimmer"></div></div>
      </div>
    </div>
    """
