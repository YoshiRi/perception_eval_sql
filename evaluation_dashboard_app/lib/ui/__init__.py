"""Streamlit HTML/CSS helpers split out from page scripts."""

from lib.ui.bounding_box_viewer_ui import (
    bev_overlay_line_and_status_legend_markup,
    bev_status_legend_markup,
)
from lib.ui.criteria_banners import (
    criteria_absolute_gate_section_hr_markup,
    criteria_compare_workspace_intro_markup,
    criteria_failing_scenarios_heading_markup,
    gate_compare_overlap_intro_markup,
    gate_verdict_banner_html,
    hover_html_for_scenarios,
)
from lib.ui.detection_stats import (
    detection_stats_page_loading_banner_markup,
    ds_spot_loading,
    ds_spot_loading_markup,
    inject_detection_stats_kpi_styles,
    inject_detection_stats_styles,
    render_kpi_card,
    section_header_html,
)
from lib.ui.download_ui import (
    ImpressiveProgressHUD,
    impressive_progress_hud_markup,
    render_detailed_scenario_download_panel,
    render_download_hero,
    render_download_status_table_intro,
    render_download_task_section_header,
    render_job_archives_summary_panel,
    render_job_json_summary_panel,
    render_recent_scenario_downloads_intro,
    render_scenario_download_summary_panel,
)
from lib.ui.styles_download import inject_download_page_styles
from lib.ui.styles_global import inject_app_page_styles
from lib.ui.task_history import get_task_list_current_user, render_task_detail_content, render_task_list
from lib.ui.task_result_summary import render_summary_table, render_task_result_summary

__all__ = [
    "ImpressiveProgressHUD",
    "bev_overlay_line_and_status_legend_markup",
    "bev_status_legend_markup",
    "criteria_absolute_gate_section_hr_markup",
    "criteria_compare_workspace_intro_markup",
    "criteria_failing_scenarios_heading_markup",
    "detection_stats_page_loading_banner_markup",
    "ds_spot_loading",
    "ds_spot_loading_markup",
    "gate_compare_overlap_intro_markup",
    "gate_verdict_banner_html",
    "hover_html_for_scenarios",
    "impressive_progress_hud_markup",
    "inject_app_page_styles",
    "inject_detection_stats_kpi_styles",
    "inject_detection_stats_styles",
    "inject_download_page_styles",
    "render_detailed_scenario_download_panel",
    "render_download_hero",
    "render_download_status_table_intro",
    "render_download_task_section_header",
    "render_job_archives_summary_panel",
    "render_job_json_summary_panel",
    "render_recent_scenario_downloads_intro",
    "render_scenario_download_summary_panel",
    "get_task_list_current_user",
    "render_summary_table",
    "render_task_detail_content",
    "render_task_list",
    "render_task_result_summary",
    "render_kpi_card",
    "section_header_html",
]
