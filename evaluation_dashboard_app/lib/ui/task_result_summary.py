"""Shared task result-summary renderers used by background task pages."""

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


def render_summary_table(rows: Optional[List[Dict[str, Any]]]) -> None:
    """Render a summary table from rows (e.g. Scenario Name, Scenario ID, Status) when present."""
    if not rows:
        return
    try:
        df = pd.DataFrame(rows)
        st.subheader("Download Status")
        st.dataframe(df, width="stretch")
    except Exception:
        pass


def render_task_result_summary(summary: Dict[str, Any]) -> None:
    """Render a result summary block from task result_summary JSON."""
    job = summary.get("job", "")
    if job == "download_results":
        total = summary.get("total", 0)
        success = summary.get("success", 0)
        failed = summary.get("failed", 0)
        out = summary.get("output_path", "")
        st.subheader("Summary")
        st.write(f"- Total scenarios processed: **{total}**")
        st.write(f"- Successfully downloaded: **{success}**")
        if failed:
            st.write(f"- Failed: **{failed}**")
        st.write(f"- Output directory: `{out}`")
        if success > 0:
            st.info("To generate the final summary CSV files, go to the **Eval Results** tab and run the evaluation.")
        render_summary_table(summary.get("rows"))
    elif job == "download_scenarios":
        total = summary.get("total", 0)
        success = summary.get("success", 0)
        failed = summary.get("failed", 0)
        out = summary.get("output_path", "")
        st.subheader("Summary")
        st.write(f"- Total scenarios: **{total}**")
        st.write(f"- Successfully downloaded: **{success}**")
        if failed:
            st.write(f"- Failed: **{failed}**")
        st.write(f"- Result JSON files: **{total}** downloaded.")
        st.write(f"- Output directory: `{out}`")
        if success > 0:
            st.info("To generate summary CSV files, go to the **Eval Results** tab and run the evaluation.")
        render_summary_table(summary.get("rows"))
    elif job == "run_eval_dirs":
        dirs = summary.get("directories_processed", 0)
        path = summary.get("summary_path", "")
        srows = summary.get("summary_rows", 0)
        scrows = summary.get("score_rows", 0)
        st.subheader("Eval Summary")
        st.write(f"- Directories processed: **{dirs}**")
        st.write(f"- Generated Summary.csv (**{srows}** rows) and Score.csv (**{scrows}** rows) in `{path}`")
    elif job == "generate_summary_csv":
        path = summary.get("summary_path", "")
        srows = summary.get("summary_rows", 0)
        scrows = summary.get("score_rows", 0)
        st.subheader("Summary")
        st.write(f"- Generated Summary.csv (**{srows}** rows) and Score.csv (**{scrows}** rows) in `{path}`")
    elif job == "build_parquet":
        path = summary.get("output_path", "")
        st.subheader("Summary")
        st.write(f"- Output: `{path}`")
    elif job == "download_and_eval":
        dl_summary = summary.get("download_summary", {})
        eval_summary_data = summary.get("eval_summary", {})
        parquet_path = summary.get("parquet_path", "")
        errors = summary.get("errors", [])

        st.subheader("Download + Eval + Parquet Summary")

        dl_success = summary.get("download_success", False)
        if dl_success:
            st.write("✅ **Download: SUCCESS**")
            st.write(
                f"   - Total: **{dl_summary.get('total', 0)}**, "
                f"Success: **{dl_summary.get('success', 0)}**, "
                f"Failed: **{dl_summary.get('failed', 0)}**"
            )
        else:
            st.write("❌ **Download: FAILED**")
            if errors:
                for err in errors:
                    st.write(f"   - {err}")

        if eval_summary_data:
            st.write("✅ **Eval: SUCCESS**")
            st.write(f"   - Directories processed: **{eval_summary_data.get('directories_processed', 0)}**")
            st.write(
                f"   - Summary.csv: **{eval_summary_data.get('summary_rows', 0)}** rows, "
                f"Score.csv: **{eval_summary_data.get('score_rows', 0)}** rows"
            )

        if parquet_path:
            st.write(f"✅ **Parquet: SUCCESS** → `{parquet_path}`")

        if errors:
            st.error("Errors during execution:")
            for err in errors:
                st.write(f"- {err}")
    elif job == "run_evaluator_and_process":
        evaluator_job_id = summary.get("evaluator_job_id", "")
        evaluator_report_url = summary.get("evaluator_report_url", "")
        evaluator_status = summary.get("evaluator_status", "unknown")
        evaluator_build_status = summary.get("evaluator_build_status", "")
        evaluator_test_status = summary.get("evaluator_test_status", "")
        evaluator_fail_message = summary.get("evaluator_fail_message", "")
        evaluator_case_totals = summary.get("evaluator_case_totals", {})
        evaluator_suites = summary.get("evaluator_suites", [])
        evaluator_failed_cases = summary.get("evaluator_failed_cases", [])
        dl_summary = summary.get("download_summary", {})
        download_rows = summary.get("download_rows", [])
        eval_summary_data = summary.get("eval_summary", {})
        parquet_path = summary.get("parquet_path", "")

        st.subheader("Run Evaluator + Download + Eval + Parquet Summary")

        st.write("🎯 **Evaluator**")
        st.write(f"   - Job ID: `{evaluator_job_id}`")
        st.write(f"   - Status: **{evaluator_status}**")
        if evaluator_build_status:
            st.write(f"   - Build: **{evaluator_build_status}**")
        if evaluator_test_status:
            st.write(f"   - Test: **{evaluator_test_status}**")
        if evaluator_case_totals:
            st.write(
                "   - Case results: "
                f"**{evaluator_case_totals.get('success', 0)}** success, "
                f"**{evaluator_case_totals.get('failed', 0)}** failed, "
                f"**{evaluator_case_totals.get('canceled', 0)}** canceled "
                f"(total **{evaluator_case_totals.get('total', 0)}**)"
            )
        if evaluator_fail_message:
            st.write(f"   - Message: `{evaluator_fail_message}`")
        if evaluator_report_url:
            st.markdown(f"   - Report: [Open]({evaluator_report_url})")
        if evaluator_suites:
            st.caption("Evaluator suite summary")
            st.dataframe(pd.DataFrame(evaluator_suites), width="stretch", hide_index=True)
        if evaluator_failed_cases:
            st.caption("Failed cases from evaluator")
            st.dataframe(pd.DataFrame(evaluator_failed_cases), width="stretch", hide_index=True)

        dl_total = dl_summary.get("total", 0)
        dl_success = dl_summary.get("success", 0)
        dl_failed = dl_summary.get("failed", 0)
        st.write("📥 **Download**")
        st.write(f"   - Total: **{dl_total}**, Success: **{dl_success}**, Failed: **{dl_failed}**")
        if download_rows:
            render_summary_table(download_rows)

        if eval_summary_data:
            st.write("🧮 **Evaluation**")
            st.write(f"   - Directories processed: **{eval_summary_data.get('directories_processed', 0)}**")
            st.write(
                f"   - Success: **{eval_summary_data.get('success', 0)}**, "
                f"Failed: **{eval_summary_data.get('failed', 0)}**"
            )
            st.write(
                f"   - Summary.csv: **{eval_summary_data.get('summary_rows', 0)}** rows, "
                f"Score.csv: **{eval_summary_data.get('score_rows', 0)}** rows"
            )

        if parquet_path:
            st.write("📦 **Parquet**")
            st.write(f"   - Output: `{parquet_path}`")

        if evaluator_report_url:
            st.markdown(f"### [📊 View Evaluator Report]({evaluator_report_url})")
    elif job == "run_release_specsheet_workflow":
        st.subheader("Release Specsheet Summary")
        st.write(f"📁 **Release root:** `{summary.get('release_root', '')}`")
        st.write(f"🏷️ **Version:** `{summary.get('version', '')}`")
        evaluator_jobs = summary.get("evaluator_jobs", {})
        if evaluator_jobs:
            rows = []
            for role, payload in evaluator_jobs.items():
                rows.append(
                    {
                        "role": role,
                        "job_id": payload.get("job_id", ""),
                        "status": payload.get("status", ""),
                        "catalog_id": payload.get("catalog_id", ""),
                        "suite_count": payload.get("suite_count", ""),
                        "description": payload.get("description", ""),
                        "report_url": payload.get("report_url", ""),
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        analysis_artifacts = summary.get("analysis_artifacts", {})
        if analysis_artifacts:
            st.write("🔎 **Detailed analysis artifacts:**")
            rows = []
            for role, payload in analysis_artifacts.items():
                download = payload.get("download", {}) if isinstance(payload.get("download"), dict) else {}
                eval_summary = payload.get("eval", {}) if isinstance(payload.get("eval"), dict) else {}
                warnings = payload.get("warnings", []) if isinstance(payload.get("warnings"), list) else []
                rows.append(
                    {
                        "role": role,
                        "path": payload.get("path", ""),
                        "download_success": download.get("success", ""),
                        "download_total": download.get("total", ""),
                        "summary_rows": eval_summary.get("summary_rows", ""),
                        "score_rows": eval_summary.get("score_rows", ""),
                        "parquet_path": payload.get("parquet_path", ""),
                        "warnings": "; ".join(str(item) for item in warnings[:3]),
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        specsheet_pdf = summary.get("specsheet_pdf", "")
        if specsheet_pdf:
            st.write(f"✅ **Specsheet PDF:** `{specsheet_pdf}`")
    elif job == "prepare_pr_test_branch":
        st.subheader("PR Test Branch Summary")
        st.write(f"- Pilot checkout: `{summary.get('pilot_checkout', '')}`")
        st.write(f"- Base pilot branch: `{summary.get('pilot_base_branch', '')}` @ `{summary.get('pilot_base_sha', '')}`")
        st.write(f"- Sub repo: `{summary.get('sub_repo', '')}` at `{summary.get('sub_repo_path', '')}`")
        selected_sub_base = summary.get("selected_sub_base", {})
        if isinstance(selected_sub_base, dict) and selected_sub_base:
            st.write(
                "- Selected sub repo base: "
                f"`{selected_sub_base.get('ref', '')}` @ `{selected_sub_base.get('sha', '')}` "
                f"({selected_sub_base.get('kind', '')})"
            )
        st.write(f"- Test sub repo branch: `{summary.get('sub_branch', '')}` @ `{summary.get('sub_new_sha', '')}`")
        st.write(f"- Test pilot branch: `{summary.get('pilot_branch', '')}` @ `{summary.get('pilot_new_sha', '')}`")
        branch_urls = summary.get("branch_urls", {})
        if isinstance(branch_urls, dict):
            pilot_url = str(branch_urls.get("pilot") or "")
            sub_repo_url = str(branch_urls.get("sub_repo") or "")
            if pilot_url:
                st.markdown(f"- Pilot branch link: [Open pilot branch]({pilot_url})")
            if sub_repo_url:
                st.markdown(f"- Sub repo branch link: [Open sub repo branch]({sub_repo_url})")
        diff_urls = summary.get("diff_urls", {})
        if isinstance(diff_urls, dict):
            pilot_diff_url = str(diff_urls.get("pilot") or "")
            sub_repo_diff_url = str(diff_urls.get("sub_repo") or "")
            if pilot_diff_url:
                st.markdown(f"- Pilot diff: [View pilot diff]({pilot_diff_url})")
            if sub_repo_diff_url:
                st.markdown(f"- Sub repo diff: [View sub repo diff]({sub_repo_diff_url})")
        st.write(f"- Restore status: **{summary.get('restore_status', 'pending')}**")
        changed_files = summary.get("changed_files", [])
        if changed_files:
            st.caption("Changed files staged from the source diff")
            st.dataframe(pd.DataFrame({"file": changed_files}), width="stretch", hide_index=True)
        push_commands = summary.get("push_commands", [])
        if push_commands and summary.get("push_status") != "completed":
            st.caption("Push commands")
            st.code("\n".join(str(cmd) for cmd in push_commands), language="bash")
        repos_diff = str(summary.get("repos_diff") or "").strip()
        if repos_diff:
            st.caption("autoware.repos diff")
            st.code(repos_diff, language="diff")
        steps = summary.get("steps", [])
        if steps:
            st.caption("Recorded steps")
            st.dataframe(pd.DataFrame(steps), width="stretch", hide_index=True)
    else:
        st.json(summary)
