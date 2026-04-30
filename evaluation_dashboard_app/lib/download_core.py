"""
Core download logic (no Streamlit) for use by the worker.
API-authenticated downloads and file organization using server-side webauto auth.
"""

import glob
import json
import logging
import os
import shutil
import urllib.parse
from pathlib import Path
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

logger = logging.getLogger(__name__)

# Default environment for evaluator API
DEFAULT_ENVIRONMENT = "default"
API_BASE_URL = "https://evaluation.ci.web.auto/v3"


def _make_evaluator_session(environment: str = DEFAULT_ENVIRONMENT):
    """Build authenticated session for evaluation.ci.web.auto API (no Streamlit)."""
    os.environ["AUTH_PROFILE"] = environment
    import webautoauth.requests
    from webautoauth.token import HttpService, TokenSource, load_config

    config = load_config()
    token_source = TokenSource(HttpService(config))
    session = webautoauth.requests.make_session(token_source)
    presigned = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    presigned.mount("http://", HTTPAdapter(max_retries=retries))
    presigned.mount("https://", HTTPAdapter(max_retries=retries))
    return session, presigned


def get_evaluator_session(environment: str = DEFAULT_ENVIRONMENT):
    """Public API: same session as worker. Returns (session, presigned) for the given environment."""
    os.environ["AUTH_PROFILE"] = environment
    return _make_evaluator_session(environment)


def _is_internal_url(url: str) -> bool:
    WEBAUTO_URL = "https://evaluation.ci.web.auto/v3/"
    FMS_URL = "https://fms.web.auto/api/v1/"
    return url.startswith(WEBAUTO_URL) or url.startswith(FMS_URL)


def _download_file(
    url: str,
    output_file: str | Path,
    *,
    chunk_size: int = 1024 * 1024,
    timeout: int = 10,
    skip_large_file: bool = False,
    large_file_mb: float = 50.0,
) -> int:
    """Download URL to file. Returns bytes written. Returns 0 if skipped (large file)."""
    output_file = Path(output_file)
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    total_size = int(r.headers.get("content-length", 0))
    if skip_large_file and total_size > 0 and (total_size / (1024 * 1024)) >= large_file_mb:
        logger.info("Skipping large file %s (%.1f MB)", output_file, total_size / (1024 * 1024))
        return 0
    downloaded = 0
    with open(output_file, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
    return downloaded


def _resp_json(resp: Any) -> Dict[str, Any]:
    """Get JSON from session get/post response (may be Response or dict)."""
    if hasattr(resp, "content"):
        return json.loads(resp.content)
    return resp


def _response_reason(resp: Any) -> str:
    """Extract a short reason from error response body for log messages."""
    if not hasattr(resp, "content") or not resp.content:
        return ""
    try:
        raw = resp.content.decode("utf-8", errors="replace").strip()
        if not raw:
            return ""
        try:
            data = json.loads(raw)
            code = data.get("code") or data.get("error_code", "")
            message = data.get("message") or data.get("error", "")
            if code and message:
                return f" {code}: {message}"
            if message:
                return f" {message}"
            if code:
                return f" {code}"
        except (json.JSONDecodeError, TypeError):
            pass
        return f" {raw[:500]}" if len(raw) <= 500 else f" {raw[:497]}..."
    except Exception:
        return ""


def get_case_simulation_log_info(
    auth_session: Any,
    api_base_url: str,
    project_id: str,
    job_id: str,
    suite_id: str = "",
    suite_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Paginate case/reports API and return list of log info dicts (no Streamlit)."""
    url = f"{api_base_url}/projects/{project_id}/jobs/{job_id}/test/case/reports"
    headers = {"Content-Type": "application/json", "accept": "application/json"}
    result: List[Dict[str, Any]] = []
    next_token = ""
    while True:
        params = {"next_token": next_token, "size": 100}
        if next_token:
            resp = auth_session.get(f"{url}?{urllib.parse.urlencode(params)}", headers=headers)
        else:
            resp = auth_session.get(url, headers=headers)
        if hasattr(resp, "status_code") and resp.status_code != 200:
            raise RuntimeError(f"API error {resp.status_code}: {getattr(resp, 'content', resp)}")
        data = _resp_json(resp)
        next_token = data.get("next_token", "")
        for report in data.get("reports", []):
            suite_info = report.get("suite", {})
            sid = suite_info.get("id", "")
            sname = suite_info.get("display_name", "")
            if suite_ids:
                if sid not in suite_ids:
                    continue
            elif suite_id and sid != suite_id:
                continue
            if "simulation_archive" not in report.get("logs", {}):
                continue
            if "simulation_result_json" not in report.get("logs", {}):
                continue
            scenario_params = report.get("scenario_parameters") or {}
            result.append({
                "suite_id": sid,
                "suite_name": sname,
                "archive_id": report["logs"]["simulation_archive"]["id"],
                "result_json_id": report["logs"]["simulation_result_json"]["id"],
                "scenario_name": report["scenario"]["display_name"],
                "scenario_id": report["scenario"]["id"],
                "scenario_ver": report["scenario"]["version_id"],
                "t4_dataset_id": scenario_params.get("t4_dataset_id", ""),
                "t4_dataset_version_id": scenario_params.get("t4_dataset_version_id", ""),
            })
        if not next_token:
            break
    return result


def _safe_path_component(value: str) -> str:
    cleaned = value.replace(os.sep, "_").replace("/", "_").strip()
    return cleaned if cleaned else "unknown"


def _get_output_path_for_log(
    output_path: str,
    log_info: Dict[str, Any],
    suite_id: str,
    suite_ids: Optional[List[str]],
) -> str:
    if suite_id and not suite_ids:
        return output_path
    sid = log_info.get("suite_id", "unknown")
    sname = log_info.get("suite_name", "")
    if sname:
        suite_dir = f"{_safe_path_component(sname)}_{_safe_path_component(sid)}"
    else:
        suite_dir = _safe_path_component(sid)
    return os.path.join(output_path, suite_dir)


def download_archive_log(
    auth_session: Any,
    presigned_session: requests.Session,
    api_base_url: str,
    project_id: str,
    log_info: Dict[str, Any],
    log_type: str,
    fmt: str,
    output_path: str,
    *,
    skip_large_file: bool = False,
    large_file_mb: float = 50.0,
    scenario_name_counts: Optional[Dict[str, int]] = None,
    on_warning: Optional[Callable[[str], None]] = None,
) -> bool:
    """Download one log (archive or result JSON) to output_path. Returns True on success."""
    url = f"{api_base_url}/projects/{project_id}/logs/{log_info[log_type]}/download"
    post_obj = {"expiration_time": 600, "filename": "suite_log.zip"}
    headers = {"Content-Type": "application/json", "accept": "application/json"}
    try:
        resp = auth_session.post(url, headers=headers, data=json.dumps(post_obj).encode("utf-8"))
    except Exception as e:
        msg = f"Failed to get download URL for {log_info.get('scenario_name')}: {e}"
        logger.warning("%s", msg)
        if on_warning:
            on_warning(msg)
        return False
    if hasattr(resp, "status_code") and resp.status_code != 200:
        reason = _response_reason(resp)
        log_id = log_info.get(log_type, "")
        scenario = log_info.get("scenario_name", "")
        msg = f"Can not get log_id {log_id}: {resp.status_code}{reason}"
        if scenario:
            msg = f"[{scenario}] {msg}"
        logger.warning("%s", msg)
        if on_warning:
            on_warning(msg)
        return False
    content = _resp_json(resp)
    download_url = content.get("url")
    if not download_url:
        return False
    os.makedirs(output_path, exist_ok=True)
    safe_scenario = _safe_path_component(log_info["scenario_name"])
    need_suffix = (
        scenario_name_counts is None
        or scenario_name_counts.get(log_info["scenario_name"], 0) > 1
    )
    if need_suffix:
        t4_id = log_info.get("t4_dataset_id", "") or ""
        fallback_id = log_info.get(log_type, "") or ""
        unique_id = (t4_id[:8] if t4_id else "") or (fallback_id[:8] if fallback_id else "")
        dl_filename = f"{safe_scenario}_{unique_id}.{fmt}" if unique_id else f"{safe_scenario}.{fmt}"
    else:
        dl_filename = f"{safe_scenario}.{fmt}"
    output_file = os.path.join(output_path, dl_filename)
    try:
        size = _download_file(
            download_url,
            output_file,
            skip_large_file=skip_large_file,
            large_file_mb=large_file_mb,
        )
        if size <= 0:
            return False
        if log_type == "archive_id" and log_info.get("t4_dataset_id"):
            stem = Path(dl_filename).stem
            try:
                sidecar_dir = Path(output_path) / stem
                sidecar_dir.mkdir(parents=True, exist_ok=True)
                sidecar_path = sidecar_dir / "t4_metadata.json"
                with open(sidecar_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "t4_dataset_id": log_info.get("t4_dataset_id", ""),
                            "t4_dataset_version_id": log_info.get("t4_dataset_version_id", ""),
                        },
                        f,
                        indent=0,
                    )
            except Exception as e:
                logger.warning("Could not write t4_metadata.json for %s: %s", stem, e)
                if on_warning:
                    on_warning(f"Could not write t4_metadata.json for {stem}: {e}")
        return True
    except Exception as e:
        msg = f"Download failed {dl_filename}: {e}"
        logger.warning("%s", msg)
        if on_warning:
            on_warning(msg)
        return False


def extract_archives(
    phase: str,
    output_path: str,
    keep_zip_files: bool = False,
    *,
    on_warning: Optional[Callable[[str], None]] = None,
) -> None:
    """Extract zip archives under output_path, keep only the given phase and scene_result.pkl.
    Skips corrupt or incomplete zip files (e.g. from interrupted download) so one bad file
    does not fail the whole job.
    """
    archive_paths = glob.glob(os.path.join(output_path, "*.zip"))
    for archive_path in archive_paths:
        dir_path = archive_path.replace(".zip", "")
        try:
            shutil.unpack_archive(archive_path, dir_path)
        except (shutil.ReadError, ValueError, OSError) as e:
            msg = f"Skipping corrupt or incomplete archive {os.path.basename(archive_path)}: {e}"
            logger.warning("%s", msg)
            if on_warning:
                on_warning(msg)
            if not keep_zip_files:
                try:
                    os.remove(archive_path)
                except OSError:
                    pass
            continue
        if not keep_zip_files:
            os.remove(archive_path)
        if not os.path.isdir(dir_path) or not os.listdir(dir_path):
            # Empty extraction directory, skip further processing
            logger.warning(f"No files found after extracting {os.path.basename(archive_path)}")
        else:
            for sub_dir_path in os.listdir(dir_path):
                if Path(sub_dir_path).name == "scenario.yaml":
                    continue
                full_path = os.path.join(dir_path, sub_dir_path)
                if Path(sub_dir_path).name != phase:
                    if os.path.isdir(full_path):
                        shutil.rmtree(full_path)
                else:
                    result_file = os.path.join(full_path, "scene_result.pkl")
                    if os.path.exists(result_file):
                        shutil.move(result_file, os.path.join(dir_path, "scene_result.pkl"))
                    shutil.rmtree(full_path)


def organize_files_into_directories(folder_path: str) -> None:
    """Create a directory per file (name without extension) and move file into it as result.json."""
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            new_dir_name = os.path.splitext(filename)[0]
            new_dir_path = os.path.join(folder_path, new_dir_name)
            os.makedirs(new_dir_path, exist_ok=True)
            shutil.move(file_path, os.path.join(new_dir_path, "result.json"))


def run_download_results(
    project_id: str,
    job_id: str,
    suite_id: Optional[str],
    output_path: str,
    download_type: str = "archives",
    phase: str = "first",
    *,
    skip_large_file: bool = False,
    large_file_mb: float = 50.0,
    keep_zip_files: bool = False,
    suite_ids: Optional[List[str]] = None,
    on_progress: Optional[Callable[[str], None]] = None,
    on_warning: Optional[Callable[[str], None]] = None,
) -> tuple:
    """
    Download job results (archives or result JSON), extract and organize.
    Requires server-side webauto auth (AUTH_PROFILE / ~/.webauto).
    Returns (failure_count, total_attempted, rows) where rows is a list of {"Scenario Name", "Scenario ID", "Status"}.
    """
    env = os.environ.get("EVALUATOR_ENVIRONMENT", DEFAULT_ENVIRONMENT)
    base_url = API_BASE_URL
    if env in ("dev", "stg"):
        base_url = f"https://scenario.ci.{env}.web.auto/v3"
    auth_session, presigned = _make_evaluator_session(env)
    suite_id = suite_id or ""
    log_dicts = get_case_simulation_log_info(
        auth_session, base_url, project_id, job_id, suite_id=suite_id, suite_ids=suite_ids
    )
    if not log_dicts:
        raise RuntimeError("No case reports found for this job/suite")
    os.makedirs(output_path, exist_ok=True)
    scenario_name_counts = dict(Counter(log_info["scenario_name"] for log_info in log_dicts))
    suite_output_paths: set = set()
    failure_count = 0
    total_attempted = 0
    rows: List[Dict[str, Any]] = []
    for i, log_info in enumerate(log_dicts):
        if on_progress:
            on_progress(f"Downloading {i+1}/{len(log_dicts)}: {log_info['scenario_name']}")
        if "second" in log_info.get("scenario_name", ""):
            continue
        total_attempted += 1
        out_dir = _get_output_path_for_log(output_path, log_info, suite_id, suite_ids)
        if download_type == "archives":
            ok = download_archive_log(
                auth_session,
                presigned,
                base_url,
                project_id,
                log_info,
                "archive_id",
                "zip",
                out_dir,
                skip_large_file=skip_large_file,
                large_file_mb=large_file_mb,
                scenario_name_counts=scenario_name_counts,
                on_warning=on_warning,
            )
        else:
            ok = download_archive_log(
                auth_session,
                presigned,
                base_url,
                project_id,
                log_info,
                "result_json_id",
                "json",
                out_dir,
                scenario_name_counts=scenario_name_counts,
                on_warning=on_warning,
            )
        if ok:
            suite_output_paths.add(out_dir)
        status = "success" if ok else "failed"
        if not ok:
            failure_count += 1
        rows.append({
            "Scenario Name": log_info.get("scenario_name", ""),
            "Scenario ID": log_info.get("scenario_id", ""),
            "Status": status,
        })
    if on_progress:
        on_progress("Extracting archives..." if download_type == "archives" else "Organizing files...")
    for out_dir in sorted(suite_output_paths):
        if download_type == "archives":
            extract_archives(phase, out_dir, keep_zip_files=keep_zip_files, on_warning=on_warning)
        else:
            organize_files_into_directories(out_dir)
    return (failure_count, total_attempted, rows)


def run_download_scenarios(
    project_id: str,
    job_id: str,
    suite_id: Optional[str],
    output_dir: str,
    overwrite: bool = False,
    *,
    scenario_name_filter: Optional[str] = None,
    selected_ids: Optional[List[str]] = None,
    suite_ids: Optional[List[str]] = None,
    on_progress: Optional[Callable[[str], None]] = None,
    on_warning: Optional[Callable[[str], None]] = None,
) -> tuple:
    """
    Download scenarios from job to output_dir (YAML per scenario).
    Requires server-side webauto auth. After scenarios, also downloads result JSON and organizes.
    Returns (failure_count, total_attempted) for the result JSON download phase.
    """
    import yaml
    from lib.WebAPI import scenarioAPI

    env = os.environ.get("EVALUATOR_ENVIRONMENT", DEFAULT_ENVIRONMENT)
    base_url = API_BASE_URL
    if env in ("dev", "stg"):
        base_url = f"https://scenario.ci.{env}.web.auto/v3"
    os.environ["AUTH_PROFILE"] = env
    auth_session, presigned = _make_evaluator_session(env)
    suite_id = suite_id or ""
    log_dicts = get_case_simulation_log_info(
        auth_session, base_url, project_id, job_id, suite_id=suite_id, suite_ids=suite_ids
    )
    if scenario_name_filter or selected_ids:
        filtered = []
        for log_info in log_dicts:
            if scenario_name_filter and scenario_name_filter.lower() not in log_info["scenario_name"].lower():
                continue
            if selected_ids and log_info["scenario_id"] not in selected_ids:
                continue
            filtered.append(log_info)
        log_dicts = filtered
    if not log_dicts:
        raise RuntimeError("No scenarios match the criteria")
    os.makedirs(output_dir, exist_ok=True)
    scenario_api = scenarioAPI(project_id)
    for i, log_info in enumerate(log_dicts):
        scenario_name = log_info["scenario_name"]
        scenario_id = log_info["scenario_id"]
        if on_progress:
            on_progress(f"Downloading scenario {i+1}/{len(log_dicts)}: {scenario_name}")
        out_dir = _get_output_path_for_log(output_dir, log_info, suite_id, suite_ids)
        scenario_dir = os.path.join(out_dir, scenario_name)
        yaml_path = os.path.join(scenario_dir, "scenario.yaml")
        if os.path.exists(yaml_path) and not overwrite:
            continue
        os.makedirs(scenario_dir, exist_ok=True)
        try:
            scenario = scenario_api.get_latest_scenario(scenario_id)
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    json.loads(scenario["scenario_format"]),
                    stream=f,
                    allow_unicode=True,
                    sort_keys=False,
                )
        except Exception as e:
            msg = f"Failed to download scenario {scenario_name}: {e}"
            logger.warning("%s", msg)
            if on_warning:
                on_warning(msg)
            raise
    # Download result JSON files and organize (same as tab1 "Result JSON only")
    suite_output_paths = {_get_output_path_for_log(output_dir, log_info, suite_id, suite_ids) for log_info in log_dicts}
    scenario_name_counts = dict(Counter(log_info["scenario_name"] for log_info in log_dicts))
    failure_count = 0
    rows: List[Dict[str, Any]] = []
    for log_info in log_dicts:
        out_dir = _get_output_path_for_log(output_dir, log_info, suite_id, suite_ids)
        ok = download_archive_log(
            auth_session,
            presigned,
            base_url,
            project_id,
            log_info,
            "result_json_id",
            "json",
            out_dir,
            scenario_name_counts=scenario_name_counts,
            on_warning=on_warning,
        )
        status = "success" if ok else "failed"
        if not ok:
            failure_count += 1
        rows.append({
            "Scenario Name": log_info.get("scenario_name", ""),
            "Scenario ID": log_info.get("scenario_id", ""),
            "Status": status,
        })
    for out_dir in sorted(suite_output_paths):
        organize_files_into_directories(out_dir)
    total_attempted = len(log_dicts)
    return (failure_count, total_attempted, rows)


def run_download_and_eval(
    project_id: str,
    job_id: str,
    suite_id: Optional[str],
    output_path: str,
    download_type: str = "archives",
    phase: str = "perception.object_recognition.tracking.objects",
    *,
    skip_large_file: bool = False,
    large_file_mb: float = 50.0,
    keep_zip_files: bool = False,
    suite_ids: Optional[List[str]] = None,
    run_eval: bool = True,
    generate_parquet: bool = True,
    eval_recursive: bool = True,
    eval_overwrite: bool = False,
    on_progress: Optional[Callable[[str], None]] = None,
    on_warning: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Combined workflow: Download results, then optionally run eval and generate parquet.
    
    Returns dict with:
        - download_success: bool
        - download_summary: dict with success/fail counts
        - eval_summary: dict with directories_processed, etc. (if run_eval=True)
        - parquet_path: str (if generate_parquet=True)
    """
    from lib import eval_summary
    
    # Try to import parquet generation
    pkl_archive_to_parquet = None
    try:
        from lib.perception_catalog_io import pkl_archive_to_parquet as _p2p
        pkl_archive_to_parquet = _p2p
    except ImportError:
        pass
    
    result: Dict[str, Any] = {
        "download_success": False,
        "download_summary": {},
        "eval_summary": {},
        "parquet_path": "",
        "errors": [],
    }
    
    # Step 1: Download
    if on_progress:
        on_progress("Starting download phase...")
    
    try:
        failure_count, total_attempted, rows = run_download_results(
            project_id=project_id,
            job_id=job_id,
            suite_id=suite_id,
            output_path=output_path,
            download_type=download_type,
            phase=phase,
            skip_large_file=skip_large_file,
            large_file_mb=large_file_mb,
            keep_zip_files=keep_zip_files,
            suite_ids=suite_ids,
            on_progress=on_progress,
            on_warning=on_warning,
        )
        success_count = total_attempted - failure_count
        result["download_summary"] = {
            "total": total_attempted,
            "success": success_count,
            "failed": failure_count,
            "rows": rows,
        }
        
        # Check if download was successful (at least some files downloaded)
        result["download_success"] = success_count > 0
        if failure_count > 0 and success_count == 0:
            result["errors"].append(f"Download failed: {failure_count} of {total_attempted} scenarios failed")
            return result
        if success_count == 0:
            result["errors"].append("Download: No scenarios were successfully downloaded")
            return result
            
    except Exception as e:
        result["errors"].append(f"Download exception: {e}")
        return result
    
    # Step 2: Run eval (if requested and download succeeded)
    if run_eval and result["download_success"]:
        if on_progress:
            on_progress("Download complete. Starting eval phase...")
        
        try:
            eval_root = output_path
            target_dirs = eval_summary.find_eval_result_dirs(eval_root, recursive=eval_recursive)
            if target_dirs:
                total = len(target_dirs)
                for i, result_dir in enumerate(target_dirs):
                    if on_progress:
                        on_progress(f"Eval: Processing {i+1}/{total}: {result_dir}")
                    eval_summary.run_eval_result_for_dir(result_dir, overwrite=eval_overwrite)
                
                # Generate summary CSVs
                csv_info = eval_summary.generate_summary_and_score_csv(eval_root)
                result["eval_summary"] = {
                    "directories_processed": total,
                    "summary_path": csv_info.get("summary_path", eval_root),
                    "summary_rows": csv_info.get("summary_rows", 0),
                    "score_rows": csv_info.get("score_rows", 0),
                }
            else:
                if on_warning:
                    on_warning("No eval result directories found")
        except Exception as e:
            result["errors"].append(f"Eval exception: {e}")
    
    # Step 3: Generate parquet (if requested and download succeeded)
    if generate_parquet and result["download_success"] and pkl_archive_to_parquet:
        if on_progress:
            on_progress("Generating parquet...")
        
        try:
            parquet_path = pkl_archive_to_parquet(
                output_path,
                on_progress=None,
                on_skip=None,
                project_id=project_id,
                job_id=job_id,
            )
            result["parquet_path"] = parquet_path
        except Exception as e:
            result["errors"].append(f"Parquet exception: {e}")
    
    return result
