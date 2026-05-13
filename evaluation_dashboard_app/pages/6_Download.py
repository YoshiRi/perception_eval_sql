import streamlit as st
import html
import json
import os
import time
import urllib.parse
import requests
import shutil
import glob
import yaml
import tempfile
import traceback
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from collections import Counter
from typing import Text, Optional, List, Dict, Any

# JST for task time display (UTC+9)
_JST = timezone(timedelta(hours=9))


def _to_jst(dt: Any) -> Optional[datetime]:
    """Convert datetime to JST for display. Naive datetimes are assumed UTC."""
    if dt is None:
        return None
    if not hasattr(dt, "astimezone"):
        return None
    try:
        if getattr(dt, "tzinfo", None) is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(_JST)
    except Exception:
        return None
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from lib.WebAPI import scenarioAPI
from lib.user_config import UserConfig
from lib.path_utils import get_data_root, resolve_under_data_root, to_data_relative
from lib.eval_summary import find_eval_result_dirs, run_eval_result_for_dir, generate_summary_and_score_csv
from lib.page_chrome import inject_app_page_styles
from lib.ui.download_ui import (
    ImpressiveProgressHUD,
    render_detailed_scenario_download_panel,
    render_download_hero,
    render_download_status_table_intro,
    render_download_task_section_header,
    render_job_archives_summary_panel,
    render_job_json_summary_panel,
    render_recent_scenario_downloads_intro,
    render_scenario_download_summary_panel,
)
from lib.ui.task_history import get_task_list_current_user, render_task_list
from lib.ui.styles_download import inject_download_page_styles
from lib.db import (
    create_task,
    delete_task,
    get_task,
    is_task_queue_enabled,
    list_recent_tasks,
    update_task_rq_job_id,
)
from lib import download_core
from lib import evaluator_api
from lib.auth import get_current_user_id, is_auth_enabled

try:
    from lib.perception_catalog_io import pkl_archive_to_parquet
    CATALOG_IO_AVAILABLE = True
except ImportError:
    CATALOG_IO_AVAILABLE = False

# Task queue panel: time window + row cap (must match header + list_recent_tasks)
_TASK_LIST_SINCE_DAYS = 7
_TASK_LIST_MAX_ROWS = 200

def _parse_rq_timeout_sec(raw: Optional[str], *, default: int, minimum: int) -> int:
    if raw is None or not str(raw).strip():
        return default
    try:
        return max(minimum, int(str(raw).strip(), 10))
    except ValueError:
        return default


# RQ kills the work horse when job_timeout is exceeded. The library default (~180s) is too
# short for downloads, run_eval_dirs, and parquet. All enqueued jobs use a long default.
_RQ_JOB_TIMEOUT_DEFAULT_SEC = 7 * 24 * 3600
_RQ_DEFAULT_JOB_TIMEOUT_SEC = _parse_rq_timeout_sec(
    os.environ.get("RQ_JOB_TIMEOUT_SEC"),
    default=_RQ_JOB_TIMEOUT_DEFAULT_SEC,
    minimum=60,
)
_bp_raw = os.environ.get("RQ_BUILD_PARQUET_TIMEOUT_SEC")
if _bp_raw is not None and str(_bp_raw).strip():
    _BUILD_PARQUET_JOB_TIMEOUT_SEC = _parse_rq_timeout_sec(
        _bp_raw,
        default=_RQ_DEFAULT_JOB_TIMEOUT_SEC,
        minimum=300,
    )
else:
    _BUILD_PARQUET_JOB_TIMEOUT_SEC = _RQ_DEFAULT_JOB_TIMEOUT_SEC

_APP_ROOT = Path(__file__).resolve().parents[1]
_CATALOGS_FILENAME = "catalogs.json"
_LEGACY_CATALOGS_PATH = Path("/home/leigu/EvaluatorRunnerUITest/catalogs.json")


def _catalog_preset_candidate_paths() -> List[Path]:
    """Return catalog preset paths in priority order."""
    paths: List[Path] = []
    env_path = os.environ.get("EVAL_CATALOGS_PATH")
    if env_path:
        paths.append(Path(env_path).expanduser())

    paths.extend(
        [
            _APP_ROOT / _CATALOGS_FILENAME,
            Path.cwd() / _CATALOGS_FILENAME,
            _LEGACY_CATALOGS_PATH,
        ]
    )

    unique_paths: List[Path] = []
    seen = set()
    for path in paths:
        key = os.fspath(path)
        if key not in seen:
            unique_paths.append(path)
            seen.add(key)
    return unique_paths


def _load_catalog_presets() -> tuple[List[Dict[str, Any]], Optional[Path], Optional[str]]:
    """Load evaluator catalog presets from the first available catalogs.json."""
    required_keys = {"display_name", "catalog_id", "integration_id"}
    for path in _catalog_preset_candidate_paths():
        if not path.is_file():
            continue

        try:
            with path.open("r", encoding="utf-8") as f:
                presets = json.load(f)
            if not isinstance(presets, list):
                raise ValueError("catalog preset file must contain a JSON list")

            valid_presets = [
                preset
                for preset in presets
                if isinstance(preset, dict) and required_keys.issubset(preset)
            ]
            return valid_presets, path, None
        except Exception as exc:
            return [], path, str(exc)

    return [], None, None


def _enqueue_task(
    task_type: str,
    parameters: Dict[str, Any],
    job_timeout: Optional[int] = None,
) -> Optional[str]:
    """Create task in Postgres and enqueue to RQ. Returns task_id or None on failure.
    job_timeout: RQ max runtime in seconds; when omitted, uses RQ_JOB_TIMEOUT_SEC (default 7 days).
    """
    session_id = get_current_user_id() if is_auth_enabled() else None
    task_id = create_task(task_type, parameters, session_id=session_id)
    if not task_id:
        return None
    try:
        from redis import Redis
        from rq import Queue
        from worker.tasks import run_job
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        conn = Redis.from_url(redis_url)
        queue = Queue(os.environ.get("RQ_QUEUE", "default"), connection=conn)
        effective_timeout = (
            job_timeout if job_timeout is not None else _RQ_DEFAULT_JOB_TIMEOUT_SEC
        )
        job = queue.enqueue(
            run_job,
            task_id,
            task_type,
            parameters,
            job_timeout=effective_timeout,
        )
        rid = getattr(job, "id", None)
        if rid:
            update_task_rq_job_id(task_id, str(rid))
        return task_id
    except Exception:
        return None

# Initialize or load user config
_user_config = UserConfig(warning_fn=st.warning)

def get_config_value(key, default=None):
    return _user_config.get(key, default)

def set_config_value(key, value):
    _user_config.set(key, value)

# Environment for Evaluator API: only "default" is used in practice.
# Change here to "dev" or "stg" if needed; no UI is exposed for this.
ENVIRONMENT = "default"

# Try to import the authentication module, with fallback handling
# https://github.com/tier4/webauto-auth-py.git
try:
    import webautoauth.requests
    from webautoauth.token import HttpService
    from webautoauth.token import load_config
    from webautoauth.token import TokenSource
    AUTH_AVAILABLE = True
except ImportError:
    st.warning("webautoauth not available. Authentication features will be limited.")
    AUTH_AVAILABLE = False

# Rest of your classes remain mostly the same...
class AuthcliHelper:
    def __init__(self, environment: str = "default"):
        if not AUTH_AVAILABLE:
            st.error("webautoauth is required for authentication. Please install it.")
            raise ImportError("webautoauth is not available")
            
        os.environ["AUTH_PROFILE"] = environment
        self.__headers = {
            "Content-Type": "application/json",
            "accept": "application/json",
        }
        # web.auto resource
        config = load_config()
        token_source = TokenSource(HttpService(config))
        self.__auth_session = webautoauth.requests.make_session(token_source)
        self.__next_token = ""

        # A session to retrieve a pre-signed URL
        self.__presigned_session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
        )
        self.__presigned_session.mount("http://", HTTPAdapter(max_retries=retries))
        self.__presigned_session.mount("https://", HTTPAdapter(max_retries=retries))

    def __is_internal_url(self, url: str):
        WEBAUTO_URL = "https://evaluation.ci.web.auto/v3/"
        FMS_URL = "https://fms.web.auto/api/v1/"
        return url.startswith(WEBAUTO_URL) or url.startswith(FMS_URL)

    def get(self, url: str):
        params = {
            "next_token": self.__next_token,
            "size": 100,
        }

        # Choose session
        if self.__is_internal_url(url):
            response = self.__auth_session.get(
                f"{url}?{urllib.parse.urlencode(params)}", headers=self.__headers
            )
        else:
            # Use a session to retrieve a pre-signed URL
            response = self.__presigned_session.get(url)

        if response.status_code == 200:
            data = json.loads(response.content)
            return data
        elif response.status_code == 403:
            raise Exception(f"403 Permission denied: {response.content}")
        else:
            raise Exception(f"{response.status_code} Authorization error: {response.content}")

    def post(self, url: str, data: dict = None):
        params = {
            "next_token": self.__next_token,
            "size": 100,
        }

        # Set header
        if self.__is_internal_url(url):
            response = self.__auth_session.post(
                f"{url}?{urllib.parse.urlencode(params)}",
                headers=self.__headers,
                data=json.dumps(data).encode("utf-8"),
            )
        else:
            # Use a session to retrieve a pre-signed URL
            response = self.__presigned_session.post(url)

        if response.status_code == 200:
            data = json.loads(response.content)
            return data
        elif response.status_code == 403:
            raise Exception(f"403 Permission denied: {response.content}")
        else:
            raise Exception(f"{response.status_code} Authorization error: {response.content}")

def download_file(
    url: str,
    output_file: str | Path,
    *,
    chunk_size: int = 1024 * 1024,
    timeout: int = 10,
    min_progress_mb: float = 20.0,
    skip_large_file: bool = False,
    large_file_mb: float = 50.0,
) -> int:
    """
    Download a file with optional progress display.

    Progress is shown only if the total file size is >= min_progress_mb.

    If skip_large_file is True and file size >= large_file_mb, skip and return 0.

    Returns:
        int: downloaded size in bytes
    """
    output_file = Path(output_file)

    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()

    total_size = int(r.headers.get("content-length", 0))
    show_progress = total_size >= min_progress_mb * 1024 * 1024

    # Check for large file skip
    if skip_large_file and total_size > 0 and (total_size / (1024 * 1024)) >= large_file_mb:
        print(
            f"Skipping download of {output_file}: file size {total_size/1024/1024:.1f} MB exceeds threshold ({large_file_mb} MB)"
        )
        return 0

    downloaded = 0
    start_time = time.time()

    print(f"Downloading file to {output_file} with total size {total_size/1024/1024:.1f} MB")
    with open(output_file, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue

            f.write(chunk)
            downloaded += len(chunk)

            if not show_progress or total_size == 0:
                continue

            elapsed = time.time() - start_time
            speed = downloaded / max(elapsed, 1e-6)
            percent = downloaded / total_size * 100
            eta = (total_size - downloaded) / max(speed, 1e-6)

            print(
                f"\r{percent:6.2f}% "
                f"({downloaded / 1e6:.1f}/{total_size / 1e6:.1f} MB) "
                f"ETA {eta:5.1f}s",
                end="",
                flush=True,
            )

    if show_progress:
        print("\nDownload complete")

    return downloaded

class JobResult:
    def __init__(
        self,
        environment: Text,
        project_id: Text,
        job_id: Text,
        suite_id: Text,
        suite_ids: Optional[List[str]],
        output_path: Text,
    ):
        self.__environment = environment
        self.__project_id = project_id
        self.__job_id = job_id
        self.__suite_id = suite_id
        self.__suite_ids = [sid for sid in (suite_ids or []) if sid]
        self.__output_path = output_path
        self.__api_base_url = "https://evaluation.ci.web.auto/v3"
        if self.__environment in ["dev", "stg"]:
            self.__api_base_url = "https://scenario.ci." + self.__environment + ".web.auto/v3"
        self.__session, self.__presigned = download_core.get_evaluator_session(environment)

    def _safe_path_component(self, value: str) -> str:
        return download_core._safe_path_component(value)

    def _get_output_path_for_log(self, log_info: Dict[str, Any]) -> str:
        return download_core._get_output_path_for_log(
            self.__output_path, log_info, self.__suite_id or "", self.__suite_ids
        )

    def download_archive_and_unzip(self, phase, skip_large_file=False, large_file_mb=50.0, keep_zip_files=False):
        log_dicts = self.get_case_simulation_log_info()
        log_dicts = [log for log in log_dicts if "second" not in log.get("scenario_name", "")]
        st.write(f"Found {len(log_dicts)} logs")
        scenario_name_counts = Counter(log_info["scenario_name"] for log_info in log_dicts)
        remain_list = []
        download_rows = []
        suite_output_paths = set()
        _n_arch = len(log_dicts)
        _hud_arch = ImpressiveProgressHUD()
        for i, log_info in enumerate(log_dicts):
            if st.session_state.get("stop_downloads"):
                st.warning("Download stopped by user.")
                break
            _hud_arch.show(
                fraction=(i / _n_arch) if _n_arch else 1.0,
                headline=f"Archive {i + 1} of {_n_arch}",
                detail=log_info["scenario_name"],
                foot="Pulling evaluator archives and writing ZIPs to disk…",
            )
            try:
                output_dir = self._get_output_path_for_log(log_info)
                suite_output_paths.add(output_dir)
                ok = self.download_archive_log(
                    log_info,
                    "archive_id",
                    "zip",
                    output_path=output_dir,
                    skip_large_file=skip_large_file,
                    large_file_mb=large_file_mb,
                    keep_zip_files=keep_zip_files,
                    scenario_name_counts=scenario_name_counts,
                )
                status = "downloaded" if ok else "failed"
                detail = "ok" if ok else "download failed"
            except Exception as e:
                status = "failed"
                detail = str(e)
            download_rows.append(
                {
                    "suite_name": log_info.get("suite_name", ""),
                    "suite_id": log_info.get("suite_id", ""),
                    "archive_id": log_info["archive_id"],
                    "result_json_id": log_info["result_json_id"],
                    "scenario_name": log_info["scenario_name"],
                    "scenario_id": log_info["scenario_id"],
                    "scenario_ver": log_info["scenario_ver"],
                    "t4_dataset_id": log_info.get("t4_dataset_id", ""),
                    "status": status,
                    "detail": detail,
                }
            )
            remain_list.append(log_info)

        if _n_arch:
            if st.session_state.get("stop_downloads"):
                _hud_arch.show(
                    fraction=len(remain_list) / _n_arch,
                    headline="Download stopped",
                    detail=f"{len(remain_list)} of {_n_arch} archives processed before stop.",
                )
            else:
                _hud_arch.show(
                    fraction=1.0,
                    headline="Archive download pass complete",
                    detail="Preparing extraction phase…",
                )
        _hud_arch.clear()

        if download_rows:
            try:
                import pandas as pd

                render_download_status_table_intro()
                st.dataframe(pd.DataFrame(download_rows), width="stretch")
            except Exception as e:
                st.warning(f"Could not render download table: {e}")
        
        with st.spinner("Extracting archives..."):
            for output_dir in sorted(suite_output_paths):
                self.extract_archives(phase, output_dir, keep_zip_files)
        return remain_list

    def download_result_json(self):
        log_dicts = self.get_case_simulation_log_info()
        st.write(f"Found {len(log_dicts)} result JSON files")
        scenario_name_counts = Counter(log_info["scenario_name"] for log_info in log_dicts)
        suite_output_paths = set()
        _n_json = len(log_dicts)
        _hud_json = ImpressiveProgressHUD()
        _json_done = 0
        for i, log_info in enumerate(log_dicts):
            if st.session_state.get("stop_downloads"):
                st.warning("Download stopped by user.")
                break
            _hud_json.show(
                fraction=(i / _n_json) if _n_json else 1.0,
                headline=f"Result JSON {i + 1} of {_n_json}",
                detail=log_info["scenario_name"],
                foot="Fetching per-scenario result JSON…",
            )
            output_dir = self._get_output_path_for_log(log_info)
            suite_output_paths.add(output_dir)
            self.download_archive_log(
                log_info, "result_json_id", "json", output_path=output_dir,
                scenario_name_counts=scenario_name_counts,
            )
            _json_done = i + 1
        if _n_json:
            if st.session_state.get("stop_downloads") and _json_done < _n_json:
                _hud_json.show(
                    fraction=_json_done / _n_json,
                    headline="Download stopped",
                    detail=f"{_json_done} of {_n_json} JSON files retrieved before stop.",
                )
            else:
                _hud_json.show(fraction=1.0, headline="JSON downloads complete", detail="Organizing files…")
        _hud_json.clear()

        with st.spinner("Organizing files..."):
            for output_dir in sorted(suite_output_paths):
                self.organize_files_into_directories(output_dir)
        return log_dicts

    def get_case_simulation_log_info(self) -> list:
        _hud = ImpressiveProgressHUD()
        _hud.show(
            indeterminate=True,
            headline="Fetching simulation logs",
            detail="Resolving suites and scenarios from the Evaluator API…",
        )
        try:
            result = download_core.get_case_simulation_log_info(
                self.__session,
                self.__api_base_url,
                self.__project_id,
                self.__job_id,
                suite_id=self.__suite_id or "",
                suite_ids=self.__suite_ids or None,
            )
        finally:
            _hud.clear()
        return result

    def download_archive_log(
        self,
        log_info,
        type,
        format,
        output_path: Optional[str] = None,
        skip_large_file=False,
        large_file_mb=50.0,
        keep_zip_files=False,
        scenario_name_counts: Optional[Dict[str, int]] = None,
    ) -> bool:
        output_dir = output_path or self.__output_path

        def on_warning(msg: str) -> None:
            st.warning(msg)

        ok = download_core.download_archive_log(
            self.__session,
            self.__presigned,
            self.__api_base_url,
            self.__project_id,
            log_info,
            type,
            format,
            output_dir,
            skip_large_file=skip_large_file,
            large_file_mb=large_file_mb,
            scenario_name_counts=scenario_name_counts,
            on_warning=on_warning,
        )
        # if ok:
        #     st.success(f"Downloaded: {log_info.get('scenario_name', '')}.{format}")
        return ok

    def extract_archives(self, phase, output_path: str, keep_zip_files=False):
        archive_paths = glob.glob(os.path.join(output_path, "*.zip"))
        st.write(f"Found {len(archive_paths)} archives to extract")
        with st.spinner("Extracting archives..."):
            download_core.extract_archives(phase, output_path, keep_zip_files=keep_zip_files)
        st.success("Extraction complete!")

    def organize_files_into_directories(self, folder_path):
        """Scan all files in the given folder, create a directory per file, move file into it as result.json."""
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        st.write(f"Organizing {len(files)} files into directories...")
        with st.spinner("Organizing files..."):
            download_core.organize_files_into_directories(folder_path)
        st.success("File organization complete!")

    def _download_scenario_file(self, url: str, output_path: str) -> bool:
        """Download a single scenario file (internal API or direct URL)."""
        try:
            if not url.startswith(("http://", "https://")):
                st.error(f"Invalid URL format: {url}")
                return False
            if download_core._is_internal_url(url):
                headers = {"Content-Type": "application/json", "accept": "application/json"}
                resp = self.__session.post(
                    url,
                    headers=headers,
                    data=json.dumps({"expiration_time": 600}).encode("utf-8"),
                )
                if getattr(resp, "status_code", None) != 200:
                    st.error(f"API error for {url}: {getattr(resp, 'status_code', 'unknown')}")
                    return False
                content = json.loads(resp.content)
                if "url" in content:
                    return download_file(content["url"], output_path)
                st.error(f"Unexpected response format for URL: {url}")
                return False
            return download_file(url, output_path)
        except Exception as e:
            st.error(f"Download error for {url}: {str(e)}")
            return False
            
    def _download_additional_resources(self, yaml_content: dict, scenario_dir: str):
        """Download additional resources referenced in scenario YAML"""
        try:
            # Check for Ego vehicle
            if "ego" in yaml_content and "vehicle" in yaml_content["ego"]:
                vehicle_data = yaml_content["ego"]["vehicle"]
                if "url" in vehicle_data:
                    vehicle_url = vehicle_data["url"]
                    vehicle_file_name = os.path.basename(vehicle_url)
                    vehicle_path = os.path.join(scenario_dir, vehicle_file_name)
                    
                    if self._download_scenario_file(vehicle_url, vehicle_path):
                        yaml_content["ego"]["vehicle"]["url"] = vehicle_file_name
            
            # Check for NPC vehicles
            if "npcs" in yaml_content:
                for i, npc in enumerate(yaml_content["npcs"]):
                    if "vehicle" in npc and "url" in npc["vehicle"]:
                        npc_url = npc["vehicle"]["url"]
                        npc_file_name = os.path.basename(npc_url)
                        npc_path = os.path.join(scenario_dir, npc_file_name)
                        
                        if self._download_scenario_file(npc_url, npc_path):
                            yaml_content["npcs"][i]["vehicle"]["url"] = npc_file_name
            
            # Save updated YAML
            yaml_file_path = os.path.join(scenario_dir, "scenario.yaml")
            with open(yaml_file_path, 'w') as f:
                yaml.dump(yaml_content, f, default_flow_style=False)
                
        except Exception as e:
            st.warning(f"Could not download additional resources: {str(e)}")

    def download_scenarios(
        self,
        output_dir: str,
        scenario_name_filter: Optional[str] = None,
        overwrite: bool = False,
        selected_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Download scenarios from the job results.
        
        Args:
            output_dir: Directory to save downloaded scenarios
            scenario_name_filter: Optional filter for scenario names (substring match)
            overwrite: Whether to overwrite existing files
            selected_ids: Optional list of specific scenario IDs to download
            
        Returns:
            List of downloaded scenario information
        """
        log_dicts = self.get_case_simulation_log_info()
        downloaded_scenarios = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        st.write(f"Found {len(log_dicts)} scenarios in job results")
        
        # Apply filters if specified
        if scenario_name_filter or selected_ids:
            filtered_logs = []
            for log_info in log_dicts:
                include = True
                
                if scenario_name_filter and scenario_name_filter.lower() not in log_info["scenario_name"].lower():
                    include = False
                
                if selected_ids and log_info["scenario_id"] not in selected_ids:
                    include = False
                
                if include:
                    filtered_logs.append(log_info)
            
            log_dicts = filtered_logs
            st.write(f"Filtered to {len(log_dicts)} scenarios")
        
        if not log_dicts:
            st.warning("No scenarios match the filter criteria")
            return []
        
        # Download each scenario
        _n = len(log_dicts)
        _hud = ImpressiveProgressHUD()
        scenario_api = scenarioAPI(self.__project_id)
        for i, log_info in enumerate(log_dicts):
            scenario_name = log_info["scenario_name"]
            scenario_id = log_info["scenario_id"]
            _hud.show(
                fraction=i / _n,
                headline=f"Scenario {i + 1} of {_n}",
                detail=scenario_name,
                foot="Downloading YAML and assets into your output tree…",
            )
            
            scenario = scenario_api.get_latest_scenario(scenario_id) # sometimes by scenario_name
            print("scenario", scenario)
            try:
                # First get scenario metadata
                
                output_dir = self._get_output_path_for_log(log_info)
                # Check if already exists
                scenario_dir = os.path.join(output_dir, f"{scenario_name}")
                yaml_file_path = os.path.join(scenario_dir, "scenario.yaml")
                scenario_dir_exists = os.path.exists(scenario_dir)
                yaml_exists = os.path.exists(yaml_file_path)

                if scenario_dir_exists and yaml_exists and not overwrite:
                    st.info(f"Skipping existing scenario: {scenario_name}")
                    downloaded_scenarios.append({
                        "name": scenario_name,
                        "id": scenario_id,
                        "path": scenario_dir,
                        "status": "skipped"
                    })
                    continue
                
                # Create scenario directory
                os.makedirs(scenario_dir, exist_ok=True)
                
                # 1. Download scenario YAML
                yaml_file_path = os.path.join(scenario_dir, "scenario.yaml")
                with open(
                    file=yaml_file_path,
                    mode="w",
                    encoding="utf-8",
                ) as f:
                    yaml.dump(
                        data=json.loads(scenario["scenario_format"]),
                        stream=f,
                        allow_unicode=True,
                        sort_keys=False,
                    )

                
                # # 2. Download map if referenced in YAML
                # try:
                #     with open(yaml_file_path, 'r') as f:
                #         yaml_content = yaml.safe_load(f)
                    
                #     if "map" in yaml_content and "url" in yaml_content["map"]:
                #         map_url = yaml_content["map"]["url"]
                #         map_file_name = os.path.basename(map_url)
                #         map_file_path = os.path.join(scenario_dir, map_file_name)
                        
                #         if not self._download_scenario_file(map_url, map_file_path):
                #             st.warning(f"Failed to download map for {scenario_name}")
                #         else:
                #             # Update YAML with local map path
                #             yaml_content["map"]["url"] = map_file_name
                #             with open(yaml_file_path, 'w') as f:
                #                 yaml.dump(yaml_content, f, default_flow_style=False)
                
                # except (yaml.YAMLError, KeyError) as e:
                #     st.warning(f"Error processing YAML for {scenario_name}: {str(e)}")
                
                # # 3. Download additional resources if needed
                # self._download_additional_resources(yaml_content, scenario_dir)
                
                downloaded_scenarios.append({
                    "name": scenario_name,
                    "id": scenario_id,
                    "path": scenario_dir,
                    "status": "success"
                })
                
                st.success(f"✓ Downloaded scenario: {scenario_name}")
                
            except Exception as e:
                st.error(f"✗ Failed to download scenario {scenario_name}: {str(e)}")
                downloaded_scenarios.append({
                    "name": scenario_name,
                    "id": scenario_id,
                    "path": "",
                    "status": "failed",
                    "error": str(e)
                })
        
        _hud.show(
            fraction=1.0,
            headline="Scenario downloads finished",
            detail=f"Processed {_n} scenario(s).",
            foot="",
        )
        _hud.clear()
        
        # Summary
        success_count = sum(1 for s in downloaded_scenarios if s["status"] == "success")
        skipped_count = sum(1 for s in downloaded_scenarios if s["status"] == "skipped")
        failed_count = sum(1 for s in downloaded_scenarios if s["status"] == "failed")
        render_scenario_download_summary_panel(
            success=success_count,
            skipped=skipped_count,
            failed=failed_count,
        )
        
        return downloaded_scenarios


st.set_page_config(
    page_title="Autoware Evaluator Downloader",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_app_page_styles()
inject_download_page_styles()
render_download_hero(queue_enabled=is_task_queue_enabled())



# Task queue status (production deployment); per-user when auth is enabled
_current_user = None
if is_task_queue_enabled():
    _current_user = get_task_list_current_user()
    render_download_task_section_header(
        since_days=_TASK_LIST_SINCE_DAYS,
        max_rows=_TASK_LIST_MAX_ROWS,
    )
    _use_fragment = getattr(st, "fragment", None) is not None
    if _use_fragment:
        try:
            @st.fragment(run_every=timedelta(seconds=3))
            def _task_list_poll():
                _t = list_recent_tasks(
                    limit=_TASK_LIST_MAX_ROWS,
                    session_id=_current_user,
                    since_days=_TASK_LIST_SINCE_DAYS,
                )
                render_task_list(_t, _current_user)
            _task_list_poll()
        except (TypeError, AttributeError):
            _use_fragment = False
    if not _use_fragment:
        tasks = list_recent_tasks(
            limit=_TASK_LIST_MAX_ROWS,
            session_id=_current_user,
            since_days=_TASK_LIST_SINCE_DAYS,
        )
        has_active = render_task_list(tasks, _current_user)
        if st.button("Refresh task list", key="refresh_tasks"):
            st.rerun()
        if has_active:
            st.info("You have running tasks. Refresh the page to see latest status and logs.")


# Initialize session state
if 'downloaded_scenarios' not in st.session_state:
    st.session_state.downloaded_scenarios = []
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Download Results"
if "stop_downloads" not in st.session_state:
    st.session_state.stop_downloads = False
if "suite_options" not in st.session_state:
    st.session_state.suite_options = []


def _run_eval_result_worker(result_dir: str, overwrite: bool) -> Dict[str, Any]:
    """Worker wrapper so the main thread owns Streamlit progress updates."""
    return run_eval_result_for_dir(result_dir, overwrite=overwrite)


def _parse_api_dt(value: Any) -> Optional[datetime]:
    """Parse evaluator API timestamps into timezone-aware datetimes."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if getattr(value, "tzinfo", None) is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    try:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if getattr(dt, "tzinfo", None) is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _format_jst_time(value: Any, *, include_seconds: bool = False) -> str:
    """Format timestamps for display in JST."""
    dt = _to_jst(_parse_api_dt(value))
    if not dt:
        return "—"
    return dt.strftime("%Y-%m-%d %H:%M:%S JST" if include_seconds else "%Y-%m-%d %H:%M JST")


def _format_jst_time_compact(value: Any) -> str:
    """Compact timestamp for dense recent-job rows."""
    dt = _to_jst(_parse_api_dt(value))
    if not dt:
        return "—"
    return dt.strftime("%m-%d %H:%M")


def _format_jst_time_title(value: Any) -> str:
    """Readable timestamp for fallback job titles."""
    dt = _to_jst(_parse_api_dt(value))
    if not dt:
        return "unknown time"
    return f"{dt.year}/{dt.month}/{dt.day} {dt.hour}:{dt.minute:02d}:{dt.second:02d}"


def _format_relative_time(value: Any) -> str:
    """Human-friendly age/duration from a timestamp until now."""
    dt = _parse_api_dt(value)
    if not dt:
        return "—"
    now = datetime.now(timezone.utc)
    secs = max(0, int((now - dt.astimezone(timezone.utc)).total_seconds()))
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


def _format_duration(start_value: Any, end_value: Any) -> str:
    """Format elapsed duration between two evaluator timestamps."""
    start = _parse_api_dt(start_value)
    end = _parse_api_dt(end_value)
    if not start or not end:
        return "—"
    secs = max(0, int((end - start).total_seconds()))
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m {secs % 60}s"
    return f"{secs // 3600}h {(secs % 3600) // 60}m"


def _extract_git_target(report: Dict[str, Any]) -> str:
    """Return a compact branch/tag label from evaluator job report metadata."""
    source = ((report.get("event") or {}).get("source") or {})
    git_ref = str(source.get("git_ref") or "").strip()
    if git_ref.startswith("refs/heads/"):
        return git_ref[len("refs/heads/"):]
    if git_ref.startswith("refs/tags/"):
        return git_ref[len("refs/tags/"):]
    return git_ref or str(source.get("git_sha") or "").strip()[:12] or "—"


def _extract_catalog_url(report: Dict[str, Any]) -> str:
    """Return a best-effort catalog URL for linking from recent evaluator jobs."""
    catalog = report.get("catalog") or {}
    direct_url = str(
        catalog.get("web_url")
        or catalog.get("url")
        or catalog.get("catalog_url")
        or ""
    ).strip()
    if direct_url:
        return direct_url

    project_id = str(report.get("project_id") or "").strip()
    catalog_id = str(
        catalog.get("catalog_id")
        or catalog.get("id")
        or ""
    ).strip()
    if project_id and catalog_id:
        return f"https://evaluation.tier4.jp/evaluation/vehicle_catalogs/{catalog_id}?project_id={project_id}"
    return ""


def _extract_job_title(report: Dict[str, Any]) -> str:
    """Prefer evaluator description for display title, with a readable fallback."""
    description = str(report.get("description") or "").strip()
    if description:
        return description
    started_like = report.get("started_at") or report.get("scheduled_at") or report.get("finished_at")
    return f"no description (Started at {_format_jst_time_title(started_like)})"


def _extract_case_totals(report: Dict[str, Any]) -> Dict[str, int]:
    """Return total/success/failed/canceled counts from job report."""
    test = report.get("test") or {}
    result = test.get("available_case_results") or test.get("case_results") or {}
    return {
        "total": int(result.get("total_count", 0) or 0),
        "success": int(result.get("success_count", 0) or 0),
        "failed": int(result.get("failure_count", 0) or 0),
        "canceled": int(result.get("cancellation_count", 0) or 0),
    }


def _extract_failed_case_rows(case_reports: List[Dict[str, Any]], *, limit: int = 50) -> List[Dict[str, Any]]:
    """Normalize failed case rows for display tables."""
    rows: List[Dict[str, Any]] = []
    for report in case_reports:
        status = str(report.get("status") or "").strip().lower()
        result_status = str(((report.get("result") or {}).get("status") or "")).strip().lower()
        if status not in evaluator_api.FAILED_JOB_STATUSES and result_status not in evaluator_api.FAILED_JOB_STATUSES:
            continue
        logs = report.get("logs") or {}
        rows.append(
            {
                "Suite": ((report.get("suite") or {}).get("display_name") or ""),
                "Scenario": ((report.get("scenario") or {}).get("display_name") or ""),
                "Status": report.get("status", ""),
                "Fail message": report.get("fail_message", ""),
                "Cause": ", ".join(report.get("failure_cause_labels", []) or []),
                "Archive log": "yes" if ((logs.get("simulation_archive") or {}).get("id")) else "no",
                "Result JSON": "yes" if ((logs.get("simulation_result_json") or {}).get("id")) else "no",
            }
        )
    rows.sort(key=lambda row: (row["Suite"], row["Scenario"], row["Fail message"]))
    return rows[:limit]


def _extract_suite_rows(suite_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize suite summary rows for display tables."""
    rows = [
        {
            "Suite": row.get("name", ""),
            "Total": int(row.get("all", 0) or 0),
            "Success": int(row.get("success", 0) or 0),
            "Failed": int(row.get("fail", 0) or 0),
            "Canceled": int(row.get("cancel", 0) or 0),
            "Simulation": row.get("simulation", ""),
            "Report": row.get("url", ""),
        }
        for row in suite_rows or []
    ]
    rows.sort(key=lambda row: (-row["Failed"], row["Suite"]))
    return rows


def _extract_suite_selection_options(suite_rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Build suite picker options from evaluator suite summary rows."""
    options: List[Dict[str, str]] = []
    seen_ids = set()
    for row in suite_rows or []:
        report_url = str(row.get("url") or "").strip()
        suite_id = ""
        if "/tests/" in report_url:
            tail = report_url.split("/tests/", 1)[1]
            suite_id = tail.split("?", 1)[0].split("/", 1)[0].strip()
        if not suite_id or suite_id in seen_ids:
            continue
        seen_ids.add(suite_id)
        suite_name = str(row.get("name") or suite_id).strip()
        options.append({"id": suite_id, "label": f"{suite_name} ({suite_id})"})
    return options


def _status_color_variant(status: str) -> str:
    """Map evaluator status to a style token used by the recent-job cards."""
    normalized = evaluator_api.normalize_job_status(status)
    if normalized in evaluator_api.SUCCESS_JOB_STATUSES:
        return "success"
    if normalized in ("canceled", "cancelled", "aborted"):
        return "canceled"
    if normalized in evaluator_api.FAILED_JOB_STATUSES:
        return "failed"
    if normalized in ("started", "running", "pending", "queued", "created"):
        return "running"
    return "unknown"


def _status_display_label(status: str) -> str:
    """Short status label for compact list rows."""
    normalized = evaluator_api.normalize_job_status(status)
    if normalized in ("succeeded", "success"):
        return "success"
    if normalized in ("failed", "failure", "error"):
        return "failed"
    if normalized in ("canceled", "cancelled", "aborted"):
        return "canceled"
    if normalized in ("started", "running"):
        return "running"
    if normalized in ("pending", "queued", "created"):
        return "queued"
    return normalized or "unknown"


def _summarize_recent_job(report: Dict[str, Any]) -> Dict[str, Any]:
    """Compact summary for one evaluator job card."""
    status = evaluator_api.extract_job_status(report)
    totals = _extract_case_totals(report)
    source = ((report.get("event") or {}).get("source") or {})
    git_url = str(source.get("git_web_url") or source.get("git_url") or "").strip()
    source_repo_label = git_url.rstrip("/").split("/")[-1] if git_url else "—"
    git_ref_label = _extract_git_target(report)
    return {
        "job_id": report.get("job_id") or report.get("id") or "",
        "title": _extract_job_title(report),
        "status": status,
        "status_variant": _status_color_variant(status),
        "build_status": ((report.get("build") or {}).get("status") or ""),
        "test_status": ((report.get("test") or {}).get("status") or ""),
        "target": git_ref_label,
        "catalog": ((report.get("catalog") or {}).get("display_name") or ""),
        "catalog_url": _extract_catalog_url(report),
        "description": report.get("description", ""),
        "source_label": git_ref_label,
        "source_repo_label": source_repo_label,
        "scheduled_at": report.get("scheduled_at"),
        "started_at": report.get("started_at"),
        "finished_at": report.get("finished_at"),
        "duration": _format_duration(report.get("started_at"), report.get("finished_at")),
        "created_label": _format_relative_time(report.get("scheduled_at") or report.get("started_at")),
        "report_url": evaluator_api.get_job_report_url(report.get("project_id", ""), report.get("job_id") or report.get("id") or ""),
        "fail_message": report.get("fail_message", ""),
        "total": totals["total"],
        "success": totals["success"],
        "failed": totals["failed"],
        "canceled": totals["canceled"],
        "git_sha": str(source.get("git_sha") or "")[:12],
        "git_ref_url": source.get("git_ref_url", ""),
        "git_commit_url": source.get("git_commit_url", ""),
        "source_url": git_url,
    }


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_recent_evaluator_jobs(project_id: str, environment: str, limit: int) -> List[Dict[str, Any]]:
    """Fetch recent evaluator jobs and normalize them for list rendering."""
    if not project_id:
        return []
    os.environ["AUTH_PROFILE"] = environment or ENVIRONMENT
    api = evaluator_api.EvaluationRunAPI()
    reports = api.get_report_list(project_id, status="all", max_results=max(1, int(limit)))
    reports = sorted(
        reports,
        key=lambda report: _parse_api_dt(report.get("scheduled_at") or report.get("started_at") or report.get("finished_at")) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    normalized = []
    for report in reports[:limit]:
        normalized.append(_summarize_recent_job(report))
    return normalized


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_evaluator_job_detail(project_id: str, environment: str, job_id: str) -> Dict[str, Any]:
    """Fetch deep evaluator detail for one job on demand."""
    if not project_id or not job_id:
        return {}
    os.environ["AUTH_PROFILE"] = environment or ENVIRONMENT
    api = evaluator_api.EvaluationRunAPI()
    report = api.get_job_report(project_id, job_id)
    suite_rows = api.get_suite_summary(project_id, job_id, use_available_case_results=True)
    case_reports = api.get_case_reports(project_id, job_id)
    summary = _summarize_recent_job(report)
    return {
        **summary,
        "suite_rows": _extract_suite_rows(suite_rows),
        "failed_case_rows": _extract_failed_case_rows(case_reports),
        "raw_report": report,
    }


def _inject_recent_evaluator_jobs_styles() -> None:
    """Task-adjacent styles for the recent evaluator jobs section."""
    st.markdown(
        """
        <style>
        .evj-card {
            border-radius: 16px;
            padding: 0.7rem 0.85rem;
            border: 1px solid rgba(148, 163, 184, 0.22);
            background: rgba(255, 255, 255, 0.92);
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        }
        .evj-card--running {
            border-color: rgba(245, 158, 11, 0.28);
            background: linear-gradient(180deg, rgba(255, 251, 235, 0.98), rgba(255,255,255,0.98));
        }
        .evj-card--success {
            border-color: rgba(16, 185, 129, 0.24);
            background: linear-gradient(180deg, rgba(236, 253, 245, 0.98), rgba(255,255,255,0.98));
        }
        .evj-card--failed {
            border-color: rgba(239, 68, 68, 0.24);
            background: linear-gradient(180deg, rgba(254, 242, 242, 0.98), rgba(255,255,255,0.98));
        }
        .evj-top, .evj-meta, .evj-stats {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }
        .evj-top { justify-content: space-between; }
        .evj-row {
            display: grid;
            grid-template-columns: minmax(180px, 1.35fr) minmax(86px, 0.5fr) minmax(108px, 0.7fr) minmax(180px, 1.25fr) minmax(190px, 1.15fr);
            gap: 8px;
            align-items: center;
        }
        .evj-title {
            font-size: 0.9rem;
            font-weight: 800;
            color: #0f172a;
            margin: 0;
            word-break: break-word;
        }
        .evj-title a {
            color: inherit;
            text-decoration: none;
        }
        .evj-title a:hover {
            text-decoration: underline;
        }
        .evj-name {
            min-width: 0;
        }
        .evj-name .evj-title,
        .evj-name .evj-name-sub,
        .evj-ref-cell,
        .evj-ref-cell .evj-name-sub {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .evj-name-sub {
            margin-top: 0.15rem;
            font-size: 0.74rem;
            color: #64748b;
        }
        .evj-status {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 0.24rem 0.5rem;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 800;
            text-transform: lowercase;
            letter-spacing: 0.01em;
            border: 1px solid transparent;
        }
        .evj-status--running { color: #9a6700; background: #fff7db; border-color: rgba(245, 158, 11, 0.28); }
        .evj-status--success { color: #047857; background: #dcfce7; border-color: rgba(16, 185, 129, 0.28); }
        .evj-status--failed { color: #b91c1c; background: #fee2e2; border-color: rgba(239, 68, 68, 0.28); }
        .evj-status--canceled { color: #7c3aed; background: #f3e8ff; border-color: rgba(124, 58, 237, 0.24); }
        .evj-status--unknown { color: #475569; background: #f1f5f9; border-color: rgba(148, 163, 184, 0.28); }
        .evj-status-mark {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 14px;
            height: 14px;
            border-radius: 999px;
            font-size: 0.62rem;
            font-weight: 900;
            line-height: 1;
            border: 1px solid currentColor;
            flex: 0 0 auto;
        }
        .evj-status-mark--success {
            background: rgba(4, 120, 87, 0.08);
        }
        .evj-status-mark--failed {
            background: rgba(185, 28, 28, 0.08);
        }
        .evj-status-mark--canceled {
            background: rgba(124, 58, 237, 0.08);
        }
        .evj-status-mark--unknown {
            background: rgba(71, 85, 105, 0.08);
        }
        .evj-status-mark--running {
            position: relative;
            border-radius: 999px;
            border: 1.5px solid rgba(154, 103, 0, 0.18);
            border-top-color: currentColor;
            border-right-color: currentColor;
            background: transparent;
            animation: evj-spin 0.9s linear infinite;
        }
        .evj-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            display: inline-block;
            background: currentColor;
            opacity: 0.88;
        }
        .evj-dot--pulse {
            animation: evj-pulse 1.4s ease-in-out infinite;
        }
        @keyframes evj-pulse {
            0% { transform: scale(0.9); opacity: 0.55; }
            50% { transform: scale(1.2); opacity: 1; }
            100% { transform: scale(0.9); opacity: 0.55; }
        }
        @keyframes evj-spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .evj-meta {
            color: #475569;
            font-size: 0.82rem;
        }
        .evj-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-top: 0.7rem;
        }
        .evj-toolbar-note {
            margin: 0.15rem 0 0.35rem;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            color: #64748b;
            text-transform: uppercase;
        }
        .evj-pager-note {
            margin-top: 0.28rem;
            font-size: 0.76rem;
            color: #475569;
            white-space: nowrap;
        }
        .evj-cell {
            min-width: 0;
            font-size: 0.78rem;
            color: #334155;
        }
        .evj-cell a {
            color: #0f766e;
            text-decoration: none;
            font-weight: 700;
        }
        .evj-cell a:hover {
            text-decoration: underline;
        }
        .evj-cell strong {
            color: #0f172a;
        }
        .evj-cell--nowrap {
            white-space: nowrap;
        }
        .evj-detail {
            margin-top: 1rem;
            padding: 1rem 1rem 0.8rem;
            border-radius: 18px;
            border: 1px solid rgba(15, 118, 110, 0.14);
            background:
                radial-gradient(circle at top right, rgba(45, 212, 191, 0.10), transparent 24%),
                linear-gradient(180deg, rgba(255,255,255,0.99), rgba(247,250,252,0.99));
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06);
        }
        .evj-stat {
            flex: 1 1 80px;
            min-width: 72px;
            padding: 0.55rem 0.7rem;
            border-radius: 14px;
            background: rgba(248, 250, 252, 0.92);
            border: 1px solid rgba(148, 163, 184, 0.16);
        }
        .evj-inline-stats {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            font-size: 0.76rem;
            color: #334155;
        }
        [class*="st-key-recent_eval_view_"] button,
        [class*="st-key-recent_eval_run_"] button,
        [class*="st-key-recent_eval_jobs_prev"] button,
        [class*="st-key-recent_eval_jobs_next"] button,
        [class*="st-key-refresh_recent_eval_jobs"] button {
            min-height: 2rem;
            padding: 0.18rem 0.58rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            box-shadow: none;
        }
        [class*="st-key-recent_eval_view_"] button,
        [class*="st-key-recent_eval_jobs_prev"] button,
        [class*="st-key-recent_eval_jobs_next"] button,
        [class*="st-key-refresh_recent_eval_jobs"] button {
            border-color: rgba(148, 163, 184, 0.34);
            color: #334155;
            background: #ffffff;
        }
        [class*="st-key-recent_eval_view_"] button:hover,
        [class*="st-key-recent_eval_jobs_prev"] button:hover,
        [class*="st-key-recent_eval_jobs_next"] button:hover,
        [class*="st-key-refresh_recent_eval_jobs"] button:hover {
            border-color: rgba(15, 118, 110, 0.28);
            color: #0f766e;
            background: #f8fffd;
        }
        [class*="st-key-recent_eval_run_"] button {
            border-color: rgba(13, 148, 136, 0.22);
            background: linear-gradient(180deg, #f0fdfa, #ecfeff);
            color: #0f766e;
        }
        [class*="st-key-recent_eval_run_"] button:hover {
            border-color: rgba(13, 148, 136, 0.34);
            background: linear-gradient(180deg, #ccfbf1, #ecfeff);
            color: #115e59;
        }
        .evj-stat-label {
            display: block;
            font-size: 0.68rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #64748b;
            font-weight: 800;
            margin-bottom: 0.14rem;
        }
        .evj-stat-value {
            display: block;
            font-size: 1rem;
            font-weight: 800;
            color: #0f172a;
        }
        .evj-desc {
            margin-top: 0.55rem;
            font-size: 0.86rem;
            color: #334155;
        }
        .evj-empty {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: #f8fafc;
            border: 1px dashed rgba(148, 163, 184, 0.4);
            color: #475569;
        }
        @media (max-width: 1080px) {
            .evj-row {
                grid-template-columns: 1fr;
                gap: 8px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_recent_evaluator_job_card(job: Dict[str, Any]) -> None:
    """Render one recent evaluator job as a single-row list item."""
    variant = html.escape(job.get("status_variant", "unknown"))
    status = html.escape(_status_display_label(job.get("status", "unknown") or "unknown"))
    title_text = html.escape(job.get("title", "—"))
    description = html.escape(job.get("description", "") or "")
    catalog = html.escape(job.get("catalog", "") or "—")
    catalog_url = html.escape(job.get("catalog_url", "") or "")
    scheduled = html.escape(_format_jst_time_compact(job.get("scheduled_at")))
    duration = html.escape(job.get("duration", "—"))
    job_id = html.escape(str(job.get("job_id", "")))
    build_status = html.escape(job.get("build_status", "") or "—")
    test_status = html.escape(job.get("test_status", "") or "—")
    created_label = html.escape(job.get("created_label", "—"))
    git_sha = html.escape(job.get("git_sha", "") or "—")
    source_label = html.escape(job.get("source_label", "") or "—")
    report_url = html.escape(job.get("report_url", "") or "")
    source_url = html.escape(job.get("git_ref_url", "") or job.get("source_url", "") or "")
    status_variant = job.get("status_variant", "unknown")
    status_mark = {
        "running": '<span class="evj-status-mark evj-status-mark--running" aria-hidden="true"></span>',
        "success": '<span class="evj-status-mark evj-status-mark--success" aria-hidden="true">✓</span>',
        "failed": '<span class="evj-status-mark evj-status-mark--failed" aria-hidden="true">!</span>',
        "canceled": '<span class="evj-status-mark evj-status-mark--canceled" aria-hidden="true">×</span>',
    }.get(status_variant, '<span class="evj-status-mark evj-status-mark--unknown" aria-hidden="true">?</span>')
    meta_line = f"id {job_id[:8]}"
    counts = (
        f'S <strong>{int(job.get("success", 0))}</strong> · '
        f'F <strong>{int(job.get("failed", 0))}</strong> · '
        f'C <strong>{int(job.get("canceled", 0))}</strong> / '
        f'<strong>{int(job.get("total", 0))}</strong>'
    )
    title_html = f'<a href="{report_url}" target="_blank" rel="noopener noreferrer">{title_text}</a>' if report_url else title_text
    source_html = (
        f'<a href="{source_url}" target="_blank" rel="noopener noreferrer">{source_label}</a>'
        if source_url else source_label
    )
    catalog_html = (
        f'<a href="{catalog_url}" target="_blank" rel="noopener noreferrer">{catalog}</a>'
        if catalog_url else catalog
    )
    st.markdown(
        f"""
        <div class="evj-card evj-card--{variant}">
          <div class="evj-row">
            <div class="evj-name">
              <div class="evj-title">{title_html}</div>
              <div class="evj-name-sub">{meta_line}</div>
            </div>
            <div class="evj-cell evj-cell--nowrap">
              <span class="evj-status evj-status--{variant}">{status_mark}{status}</span>
            </div>
            <div class="evj-cell">
              <strong>{scheduled}</strong><br><span class="evj-name-sub">{duration} · {created_label}</span>
            </div>
            <div class="evj-cell evj-ref-cell">
              <strong>{catalog_html}</strong><br><span class="evj-name-sub">{source_html}</span>
            </div>
            <div class="evj-cell">
              <span class="evj-name-sub">build {build_status} · test {test_status} · {git_sha}</span><br>
              <span class="evj-inline-stats">{counts}</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_recent_evaluator_job_detail(project_id: str, environment: str, job: Dict[str, Any]) -> None:
    """Render detailed evaluator-job information inside an expander."""
    job_id = str(job.get("job_id", "") or "")
    if not job_id:
        st.warning("Missing job id.")
        return
    try:
        detail = _fetch_evaluator_job_detail(project_id, environment, job_id)
    except Exception as e:
        st.error(f"Could not fetch evaluator details: {e}")
        return

    st.markdown("**Overview**")
    top_cols = st.columns(4)
    top_cols[0].metric("Total", int(detail.get("total", 0)))
    top_cols[1].metric("Success", int(detail.get("success", 0)))
    top_cols[2].metric("Failed", int(detail.get("failed", 0)))
    top_cols[3].metric("Canceled", int(detail.get("canceled", 0)))

    overview_left, overview_right = st.columns([1.3, 1.1])
    with overview_left:
        st.write(f"Status: `{detail.get('status', 'unknown')}`")
        st.write(f"Title: `{detail.get('title', '—')}`")
        st.write(f"Build/Test: `{detail.get('build_status', '—')}` / `{detail.get('test_status', '—')}`")
        st.write(f"Ref: `{detail.get('target', '—')}`")
        st.write(f"Catalog: `{detail.get('catalog', '—')}`")
        st.write(f"Repo: `{detail.get('source_repo_label', '—')}`")
    with overview_right:
        st.write(f"Scheduled: `{_format_jst_time(detail.get('scheduled_at'), include_seconds=True)}`")
        st.write(f"Started: `{_format_jst_time(detail.get('started_at'), include_seconds=True)}`")
        st.write(f"Finished: `{_format_jst_time(detail.get('finished_at'), include_seconds=True)}`")
        st.write(f"Duration: `{detail.get('duration', '—')}`")
        st.write(f"SHA: `{detail.get('git_sha', '—')}`")

    action_cols = st.columns([1.2, 1.2, 4])
    report_url = detail.get("report_url", "")
    catalog_url = detail.get("catalog_url", "")
    source_url = detail.get("source_url", "") or detail.get("git_ref_url", "")
    with action_cols[0]:
        if report_url:
            st.link_button("Open report", report_url, use_container_width=True)
    with action_cols[1]:
        if catalog_url:
            st.link_button("Open catalog", catalog_url, use_container_width=True)
    with action_cols[2]:
        if source_url:
            st.link_button("Open source", source_url, use_container_width=True)

    if detail.get("fail_message"):
        st.warning(detail.get("fail_message"))

    suite_rows = detail.get("suite_rows") or []
    with st.expander(f"Suites ({len(suite_rows)})", expanded=bool(suite_rows)):
        if suite_rows:
            st.dataframe(pd.DataFrame(suite_rows), width="stretch", hide_index=True)
        else:
            st.caption("No suite summary available.")

    failed_case_rows = detail.get("failed_case_rows") or []
    with st.expander(f"Failed Cases ({len(failed_case_rows)})", expanded=bool(failed_case_rows)):
        if failed_case_rows:
            st.dataframe(pd.DataFrame(failed_case_rows), width="stretch", hide_index=True)
        else:
            st.caption("No failed cases in the current report.")

    with st.expander("Raw JSON", expanded=False):
        st.json(detail.get("raw_report", {}))


def _render_recent_evaluator_job_run_dialog(
    project_id: str,
    environment: str,
    job: Dict[str, Any],
    *,
    output_path_default: str,
    download_type_default: str,
    phase_default: str,
    skip_large_file_default: bool,
    large_file_mb_default: float,
    keep_zip_files_default: bool,
) -> None:
    """Render the dialog used to enqueue Download + Eval + Parquet from a recent job row."""
    job_id = str(job.get("job_id", "") or "")
    if not job_id:
        st.error("Missing evaluator job id.")
        return

    detail = _fetch_evaluator_job_detail(project_id, environment, job_id)
    suite_options = _extract_suite_selection_options(detail.get("suite_rows") or [])
    suite_label_to_id = {opt["label"]: opt["id"] for opt in suite_options}
    suite_labels = [opt["label"] for opt in suite_options]

    st.caption("Confirm the workflow options for this evaluator job, then start a background task.")
    summary_cols = st.columns([1.45, 1.15, 1.35, 1.05])
    summary_cols[0].markdown(f"**Title**  \n`{detail.get('title', '—')}`")
    summary_cols[1].markdown(f"**Status**  \n`{detail.get('status', 'unknown')}`")
    summary_cols[2].markdown(f"**Catalog**  \n`{detail.get('catalog', '—')}`")
    summary_cols[3].markdown(f"**Cases**  \n`{int(detail.get('total', 0))}`")

    with st.form(key=f"recent_eval_run_form_{job_id}", border=False):
        run_output_path = st.text_input(
            "Output path",
            value=output_path_default,
            help="Folder under the data directory. This uses the same safe path rules as the main download workflow.",
        )

        selected_suite_labels = st.multiselect(
            "Suites to download (optional)",
            options=suite_labels,
            default=[],
            help="Leave empty to download all suites from this evaluator job.",
        )

        run_download_type = st.radio(
            "Download type",
            ["Archives (ZIP)", "Result JSON only"],
            index=0 if download_type_default == "Archives (ZIP)" else 1,
            horizontal=True,
        )

        run_phase = ""
        run_skip_large_file = False
        run_large_file_mb = 50.0
        run_keep_zip_files = False
        if run_download_type == "Archives (ZIP)":
            run_phase = st.text_input(
                "Phase to extract",
                value=phase_default,
                help="Enter the phase name to extract from archives.",
            )
            opt_cols = st.columns([1.2, 1.3, 1.2])
            with opt_cols[0]:
                run_skip_large_file = st.checkbox(
                    "Skip large files",
                    value=skip_large_file_default,
                    help="Skip unusually large archives during download.",
                )
            with opt_cols[1]:
                run_large_file_mb = st.number_input(
                    "Skip threshold (MB)",
                    min_value=1.0,
                    max_value=5000.0,
                    step=1.0,
                    value=float(large_file_mb_default),
                )
            with opt_cols[2]:
                run_keep_zip_files = st.checkbox(
                    "Keep ZIP files",
                    value=keep_zip_files_default,
                    help="Keep downloaded ZIPs after extraction.",
                )

        run_cols = st.columns([1.25, 1.25, 1.1])
        with run_cols[0]:
            run_eval = st.checkbox(
                "Run evaluation",
                value=True,
                help="Run eval_result and generate Summary.csv / Score.csv after download.",
            )
        with run_cols[1]:
            generate_parquet = st.checkbox(
                "Generate parquet",
                value=CATALOG_IO_AVAILABLE,
                disabled=not CATALOG_IO_AVAILABLE,
                help="Build scene_result.parquet from .pkl files." if CATALOG_IO_AVAILABLE else "Install perception_catalog_analyzer to enable this.",
            )
        with run_cols[2]:
            eval_recursive = st.checkbox(
                "Recursive eval",
                value=True,
                help="Search subdirectories for evaluation result folders.",
            )

        action_cols = st.columns([1.15, 1.15, 3.7])
        cancel_clicked = action_cols[0].form_submit_button("Cancel", use_container_width=True)
        start_clicked = action_cols[1].form_submit_button("Start", type="primary", use_container_width=True)

    if cancel_clicked:
        st.session_state.pop("recent_eval_jobs_run_selected", None)
        st.rerun()

    if not start_clicked:
        return

    resolved_output, path_err = resolve_under_data_root(run_output_path, allow_create=True)
    if path_err:
        st.error(f"Output path is invalid: {path_err}")
        return

    selected_suite_ids = [suite_label_to_id[label] for label in selected_suite_labels]
    resolved_path_str = str(resolved_output)
    set_config_value("output_path", to_data_relative(resolved_output))
    set_config_value("environment", environment)
    set_config_value("project_id", project_id)
    set_config_value("job_id", job_id)
    set_config_value("suite_id", "")
    set_config_value("suite_ids", selected_suite_ids)
    set_config_value("download_type", run_download_type)
    if run_download_type == "Archives (ZIP)":
        set_config_value("phase", run_phase)
        set_config_value("skip_large_file", run_skip_large_file)
        set_config_value("large_file_mb", run_large_file_mb)
        set_config_value("keep_zip_files", run_keep_zip_files)

    params = {
        "output_path": resolved_path_str,
        "project_id": project_id,
        "job_id": job_id,
        "suite_id": "",
        "suite_ids": selected_suite_ids or None,
        "download_type": "archives" if run_download_type == "Archives (ZIP)" else "result_json",
        "phase": run_phase if run_download_type == "Archives (ZIP)" else "",
        "skip_large_file": run_skip_large_file if run_download_type == "Archives (ZIP)" else False,
        "large_file_mb": run_large_file_mb if run_download_type == "Archives (ZIP)" else 50.0,
        "keep_zip_files": run_keep_zip_files if run_download_type == "Archives (ZIP)" else False,
        "run_eval": run_eval,
        "generate_parquet": generate_parquet,
        "eval_recursive": eval_recursive,
        "eval_overwrite": False,
    }
    task_id = _enqueue_task("download_and_eval", params)
    if not task_id:
        st.error("Failed to enqueue task. Check REDIS_URL and DATABASE_URL.")
        return

    st.session_state["recent_eval_jobs_flash"] = (
        f"Queued Download + Eval + Parquet for `{detail.get('title', job_id)}`. "
        f"Task id: `{task_id}`."
    )
    st.session_state.pop("recent_eval_jobs_run_selected", None)
    st.rerun()


def _render_recent_evaluator_jobs_section(
    project_id: str,
    environment: str,
    *,
    output_path_default: str,
    download_type_default: str,
    phase_default: str,
    skip_large_file_default: bool,
    large_file_mb_default: float,
    keep_zip_files_default: bool,
) -> None:
    """Render a direct evaluator-jobs browser above the download tabs."""
    _inject_recent_evaluator_jobs_styles()
    show_section = st.toggle(
        "Show recent evaluator jobs",
        value=st.session_state.get("recent_eval_jobs_show", False),
        key="recent_eval_jobs_show",
        help="Load recent evaluator jobs only when you want to browse them.",
    )
    if not show_section:
        return

    st.subheader("Recent evaluator jobs")
    st.caption("Compact browser for recent evaluator jobs. Select one job to inspect detailed suite and failed-case information.")
    flash_message = st.session_state.pop("recent_eval_jobs_flash", None)
    if flash_message:
        st.success(flash_message)

    control_cols = st.columns([0.95, 1.45, 1.65, 0.9])
    with control_cols[0]:
        st.markdown('<div class="evj-toolbar-note">Rows</div>', unsafe_allow_html=True)
        limit = int(
            st.selectbox(
                "Rows",
                options=[6, 12, 20, 30],
                index=1,
                key="recent_eval_jobs_limit",
                help="How many recent evaluator jobs to fetch for this project.",
                label_visibility="collapsed",
            )
        )
    with control_cols[1]:
        st.markdown('<div class="evj-toolbar-note">Status</div>', unsafe_allow_html=True)
        status_filter = st.multiselect(
            "Status",
            options=["running", "success", "failed", "canceled", "unknown"],
            default=[],
            key="recent_eval_jobs_status_filter",
            help="Leave empty to show all recent jobs.",
            label_visibility="collapsed",
            placeholder="All statuses",
        )
    with control_cols[2]:
        st.markdown('<div class="evj-toolbar-note">Branch Or Tag</div>', unsafe_allow_html=True)
        branch_filter = st.text_input(
            "Branch/tag contains",
            value=st.session_state.get("recent_eval_jobs_branch_filter", ""),
            key="recent_eval_jobs_branch_filter",
            help="Optional substring filter for branch or tag name.",
            label_visibility="collapsed",
            placeholder="Filter by branch or tag",
        ).strip()
    with control_cols[3]:
        st.markdown('<div class="evj-toolbar-note">Actions</div>', unsafe_allow_html=True)
        if st.button("Refresh", key="refresh_recent_eval_jobs", use_container_width=True):
            _fetch_recent_evaluator_jobs.clear()
            _fetch_evaluator_job_detail.clear()
            st.rerun()

    page_key = "recent_eval_jobs_page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1

    def _render_job_list() -> None:
        if not project_id:
            st.info("Enter a project id in the sidebar to browse recent evaluator jobs.")
            return
        current_page = max(1, int(st.session_state.get(page_key, 1)))
        fetch_limit = max(limit * (current_page + 1), limit + 1)
        try:
            jobs = _fetch_recent_evaluator_jobs(project_id, environment, fetch_limit)
        except Exception as e:
            st.error(f"Could not fetch recent evaluator jobs: {e}")
            return

        if branch_filter:
            branch_filter_lower = branch_filter.lower()
            jobs = [job for job in jobs if branch_filter_lower in str(job.get("target", "")).lower()]
        if status_filter:
            selected = set(status_filter)
            jobs = [job for job in jobs if job.get("status_variant") in selected or evaluator_api.normalize_job_status(job.get("status", "")) in selected]

        if not jobs:
            st.session_state[page_key] = 1
            st.markdown('<div class="evj-empty">No recent evaluator jobs matched the current filters.</div>', unsafe_allow_html=True)
            return

        total_loaded = len(jobs)
        has_next_page = total_loaded > current_page * limit
        max_known_page = max(1, (total_loaded + limit - 1) // limit)
        if current_page > max_known_page:
            current_page = max_known_page
            st.session_state[page_key] = current_page
        start_idx = (current_page - 1) * limit
        end_idx = start_idx + limit
        visible_jobs = jobs[start_idx:end_idx]
        if not visible_jobs and current_page > 1:
            current_page = max(1, current_page - 1)
            st.session_state[page_key] = current_page
            start_idx = (current_page - 1) * limit
            end_idx = start_idx + limit
            visible_jobs = jobs[start_idx:end_idx]
            has_next_page = total_loaded > current_page * limit

        pager_cols = st.columns([0.9, 1.1, 5.2, 0.9, 1.2])
        with pager_cols[0]:
            if st.button("Back", key="recent_eval_jobs_prev", use_container_width=True, disabled=current_page <= 1):
                st.session_state[page_key] = max(1, current_page - 1)
                st.rerun()
        with pager_cols[1]:
            st.markdown(
                f"<div class='evj-pager-note'><strong>{current_page}</strong></div>",
                unsafe_allow_html=True,
            )
        with pager_cols[3]:
            if st.button("More", key="recent_eval_jobs_next", use_container_width=True, disabled=not has_next_page):
                st.session_state[page_key] = current_page + 1
                st.rerun()
        with pager_cols[4]:
            st.markdown(
                f"<div class='evj-pager-note' style='text-align:right;'>{len(visible_jobs)} shown · {total_loaded}+ loaded</div>",
                unsafe_allow_html=True,
            )

        selected_job_id = st.session_state.get("recent_eval_jobs_selected")
        if selected_job_id and not any(str(job.get("job_id", "")) == str(selected_job_id) for job in jobs):
            st.session_state.pop("recent_eval_jobs_selected", None)
            selected_job_id = None

        selected_run_job_id = st.session_state.get("recent_eval_jobs_run_selected")
        if selected_run_job_id and not any(str(job.get("job_id", "")) == str(selected_run_job_id) for job in jobs):
            st.session_state.pop("recent_eval_jobs_run_selected", None)
            selected_run_job_id = None

        st.markdown('<div class="evj-list">', unsafe_allow_html=True)
        for job in visible_jobs:
            row_cols = st.columns([9.4, 2.0])
            with row_cols[0]:
                _render_recent_evaluator_job_card(job)
            with row_cols[1]:
                action_cols = st.columns([1.0, 1.0], gap="small")
                with action_cols[0]:
                    if st.button("Details", key=f"recent_eval_view_{job['job_id']}", use_container_width=True):
                        st.session_state["recent_eval_jobs_selected"] = str(job["job_id"])
                        _fetch_evaluator_job_detail.clear()
                        st.rerun()
                with action_cols[1]:
                    if st.button("Run", key=f"recent_eval_run_{job['job_id']}", use_container_width=True):
                        st.session_state["recent_eval_jobs_run_selected"] = str(job["job_id"])
                        _fetch_evaluator_job_detail.clear()
                        st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        selected_job_id = st.session_state.get("recent_eval_jobs_selected")
        if selected_job_id:
            selected_job = next((job for job in jobs if str(job.get("job_id", "")) == str(selected_job_id)), None)
            if selected_job:
                if callable(getattr(st, "dialog", None)):
                    try:
                        @st.dialog(f"Job details · {selected_job.get('title', '—')}", width="large")
                        def _recent_eval_job_dialog() -> None:
                            _render_recent_evaluator_job_detail(project_id, environment, selected_job)
                            if st.button("Close", key="recent_eval_jobs_close_detail", use_container_width=True):
                                st.session_state.pop("recent_eval_jobs_selected", None)
                                st.rerun()

                        _recent_eval_job_dialog()
                    finally:
                        st.session_state.pop("recent_eval_jobs_selected", None)
                else:
                    st.markdown('<div class="evj-detail">', unsafe_allow_html=True)
                    hdr_cols = st.columns([4.4, 1.1])
                    with hdr_cols[0]:
                        st.subheader(f"Job details · {selected_job.get('title', '—')}")
                    with hdr_cols[1]:
                        if st.button("Close", key="recent_eval_jobs_close_detail_fallback", use_container_width=True):
                            st.session_state.pop("recent_eval_jobs_selected", None)
                            st.rerun()
                    _render_recent_evaluator_job_detail(project_id, environment, selected_job)
                    st.markdown("</div>", unsafe_allow_html=True)

        selected_run_job_id = st.session_state.get("recent_eval_jobs_run_selected")
        if selected_run_job_id:
            selected_run_job = next((job for job in jobs if str(job.get("job_id", "")) == str(selected_run_job_id)), None)
            if selected_run_job:
                if callable(getattr(st, "dialog", None)):
                    try:
                        @st.dialog(f"Download + Eval + Parquet · {selected_run_job.get('title', '—')}", width="large")
                        def _recent_eval_run_dialog() -> None:
                            _render_recent_evaluator_job_run_dialog(
                                project_id,
                                environment,
                                selected_run_job,
                                output_path_default=output_path_default,
                                download_type_default=download_type_default,
                                phase_default=phase_default,
                                skip_large_file_default=skip_large_file_default,
                                large_file_mb_default=large_file_mb_default,
                                keep_zip_files_default=keep_zip_files_default,
                            )

                        _recent_eval_run_dialog()
                    finally:
                        if st.session_state.get("recent_eval_jobs_run_selected") == str(selected_run_job_id):
                            st.session_state.pop("recent_eval_jobs_run_selected", None)
                else:
                    st.markdown('<div class="evj-detail">', unsafe_allow_html=True)
                    hdr_cols = st.columns([4.4, 1.1])
                    with hdr_cols[0]:
                        st.subheader(f"Download + Eval + Parquet · {selected_run_job.get('title', '—')}")
                    with hdr_cols[1]:
                        if st.button("Close", key="recent_eval_jobs_close_run_fallback", use_container_width=True):
                            st.session_state.pop("recent_eval_jobs_run_selected", None)
                            st.rerun()
                    _render_recent_evaluator_job_run_dialog(
                        project_id,
                        environment,
                        selected_run_job,
                        output_path_default=output_path_default,
                        download_type_default=download_type_default,
                        phase_default=phase_default,
                        skip_large_file_default=skip_large_file_default,
                        large_file_mb_default=large_file_mb_default,
                        keep_zip_files_default=keep_zip_files_default,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

    _render_job_list()


# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    environment = ENVIRONMENT  # Not exposed in UI; change ENVIRONMENT constant above if needed

    project_id = st.text_input(
        "Project ID",
        value=get_config_value("project_id", "x2_dev"),
        help="Enter the project ID"
    )
    set_config_value("project_id", project_id)

    def extract_job_id_from_url(input_str):
        import re
        import urllib.parse

        # Try to extract job id from known URL patterns
        try:
            parsed_url = urllib.parse.urlparse(input_str)
            # Only if it's http(s) and /reports/xxxx pattern
            if parsed_url.scheme in ("http", "https"):
                match = re.search(r'/reports/([a-f0-9\-]{16,})', parsed_url.path)
                if match:
                    return match.group(1)
            # If not, return as is
            return input_str
        except Exception:
            return input_str

    def extract_suite_id_from_url(input_str):
        import re
        import urllib.parse

        # Try to extract suite id from known URL patterns (e.g. evaluation.tier4.jp/evaluation/suites/<uuid>?project_id=...)
        try:
            parsed_url = urllib.parse.urlparse(input_str)
            if parsed_url.scheme in ("http", "https"):
                # Match /suites/<uuid> in path (UUID is 8-4-4-4-12 hex)
                match = re.search(r'/suites/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', parsed_url.path, re.IGNORECASE)
                if match:
                    return match.group(1)
            return input_str
        except Exception:
            return input_str

    # Initialize session state
    if "job_id" not in st.session_state:
        st.session_state.job_id = get_config_value("job_id", "")

    # Callback when input changes
    def on_job_id_change():
        raw = st.session_state.job_id
        parsed = extract_job_id_from_url(raw)

        # Update text input with extracted ID
        st.session_state.job_id = parsed

        # Save config
        set_config_value("job_id", parsed)


    # Text input bound to session state
    st.text_input(
        "Job ID/Report URL",
        key="job_id",
        help="Enter the job ID or paste a full Evaluator report URL",
        on_change=on_job_id_change,
    )
    set_config_value("job_id", st.session_state.job_id)

    # Suite ID: same URL-based extraction as Job ID
    if "suite_id" not in st.session_state:
        st.session_state.suite_id = get_config_value("suite_id", "")

    def on_suite_id_change():
        raw = st.session_state.suite_id
        parsed = extract_suite_id_from_url(raw)
        st.session_state.suite_id = parsed
        set_config_value("suite_id", parsed)

    st.text_input(
        "Suite ID/Suite URL (leave empty to download all suites)",
        key="suite_id",
        help="Enter the suite ID or paste a full Evaluator suite URL (e.g. https://evaluation.tier4.jp/evaluation/suites/<suite_id>?project_id=...)",
        on_change=on_suite_id_change,
    )
    suite_id = st.session_state.suite_id
    set_config_value("suite_id", suite_id)

    _default_output = "download"
    _stored = get_config_value("output_path", _default_output)
    _display_output = to_data_relative(_stored) if _stored else _default_output
    output_path = st.text_input(
        "Output Path",
        value=_display_output,
        help="Folder under the data directory (e.g. download or v4.3.1_test4). Path is limited to the server data root."
    )
    set_config_value("output_path", output_path)

    fetch_suites = st.button("Fetch suites from job", help="Retrieve suites for the current Project ID and Job ID")
    if fetch_suites:
        if not all([project_id, st.session_state.job_id]):
            st.error("Please fill in Project ID and Job ID before fetching suites.")
        else:
            _resolved, _err = resolve_under_data_root(output_path, allow_create=True)
            if _err:
                st.error(f"Output path is invalid: {_err}.")
            else:
                try:
                    with st.spinner("Fetching suites..."):
                        job_result = JobResult(
                            environment=environment,
                            project_id=project_id,
                            job_id=st.session_state.job_id,
                            suite_id="",
                            suite_ids=None,
                            output_path=str(_resolved)
                        )
                        log_dicts = job_result.get_case_simulation_log_info()
                    suite_map = {}
                    for log_info in log_dicts:
                        sid = log_info.get("suite_id", "")
                        sname = log_info.get("suite_name", "")
                        if sid:
                            suite_map[sid] = sname or sid
                    st.session_state.suite_options = [
                        {"id": sid, "name": suite_map[sid]} for sid in sorted(suite_map, key=lambda s: suite_map[s].lower())
                    ]
                    if st.session_state.suite_options:
                        st.success(f"Found {len(st.session_state.suite_options)} suites.")
                    else:
                        st.info("No suites found for this job.")
                except Exception as e:
                    st.error(f"Failed to fetch suites: {str(e)}")

    suite_options = st.session_state.suite_options
    suite_id_map = {f"{opt['name']} ({opt['id']})": opt["id"] for opt in suite_options}
    saved_suite_ids = get_config_value("suite_ids", []) or []
    default_suite_labels = [
        label for label, sid in suite_id_map.items() if sid in saved_suite_ids
    ]
    suite_labels = sorted(suite_id_map.keys())
    selected_suite_labels = st.multiselect(
        "Suites to download (optional)",
        options=suite_labels,
        default=default_suite_labels,
        help="Pick one or more suites from the job. Leave empty to download all suites.",
        disabled=not suite_labels,
    )
    suite_ids = [suite_id_map[label] for label in selected_suite_labels]
    set_config_value("suite_ids", suite_ids)
    
    download_type = st.radio(
        "Download Type",
        ["Archives (ZIP)", "Result JSON only"],
        index=["Archives (ZIP)", "Result JSON only"].index(get_config_value("download_type", "Archives (ZIP)"))
    )
    set_config_value("download_type", download_type)

    if download_type == "Archives (ZIP)":
        phase = st.text_input(
            "Phase to extract",
            value=get_config_value(
                "phase", "perception.object_recognition.tracking.objects"
            ),
            help="Enter the phase name to extract from archives",
        )
        set_config_value("phase", phase)

        # Add options to control skipping large files
        col_large_skip1, col_large_skip2 = st.columns([2, 3])
        with col_large_skip1:
            skip_large_file = st.checkbox(
                "Skip large files?",
                value=get_config_value("skip_large_file", False),
                help="Large archive is in an abnormal state."
                    "Skipping them is the correct behavior. Uncheck only if you explicitly want to download them."
            )
            set_config_value("skip_large_file", skip_large_file)
        with col_large_skip2:
            large_file_mb = st.number_input(
                "Skip threshold (MB)",
                value=float(get_config_value("large_file_mb", 50.0)),
                min_value=1.0,
                max_value=5000.0,
                step=1.0,
                help="ZIP files larger than this size will be skipped if 'Skip large files?' is checked."
            )
            set_config_value("large_file_mb", large_file_mb)
    else:
        skip_large_file = False
        large_file_mb = 50.0  # Doesn't apply

_render_recent_evaluator_jobs_section(
    project_id,
    environment,
    output_path_default=output_path,
    download_type_default=download_type,
    phase_default=phase if download_type == "Archives (ZIP)" else get_config_value(
        "phase", "perception.object_recognition.tracking.objects"
    ),
    skip_large_file_default=skip_large_file,
    large_file_mb_default=large_file_mb,
    keep_zip_files_default=bool(get_config_value("keep_zip_files", False)) if download_type == "Archives (ZIP)" else False,
)

st.markdown('<p class="dl-tabs-rail">Pick a workflow</p>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(
    ["📥 Download Results", "🗺️ Download Scenarios", "📊 View Downloads", "🧮 Eval Results"]
)



with tab1:
    st.header("Download Job Results")
        
    # Main content area
    stop_col, open_folder_col, _ = st.columns([1, 1, 4])
    with stop_col:
        if st.button("Stop Downloads", key="stop_downloads_btn"):
            st.session_state.stop_downloads = True
            st.warning("Stop requested. Current download will finish then halt.")

    # Button to open the output folder in file explorer
    import platform
    if open_folder_col.button("Open Output Folder", key="open_folder_btn"):
        import subprocess
        _resolved, _err = resolve_under_data_root(output_path, allow_create=False, allow_missing=True)
        folder_path = str(_resolved) if _resolved else None
        if _err or not folder_path or not os.path.isdir(folder_path):
            st.warning(f"Output folder does not exist yet: `{output_path}`. Run a download first to create it.")
        else:
            try:
                if platform.system() == "Windows":
                    os.startfile(folder_path)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.Popen(["open", folder_path])
                else:  # Linux or others
                    subprocess.Popen(["xdg-open", folder_path])
            except Exception as e:
                st.error(f"Could not open folder: {e}")

    # Add "Keep ZIP files" option, default False
    keep_zip_files = False
    if download_type == "Archives (ZIP)":
        keep_zip_files = st.checkbox(
            "Keep ZIP files after extract?",
            value=get_config_value("keep_zip_files", False),
            help="If checked, downloaded ZIP archives will be kept after extraction (default: not kept)."
        )
        set_config_value("keep_zip_files", keep_zip_files)

    selected_suite_ids = suite_ids or ([suite_id] if suite_id else [])

    if st.button("List Available Logs Info"):
        if not all([project_id, st.session_state.job_id]):
            st.error("Please fill in all required fields: Project ID and Job ID")
            st.stop()
        _resolved, _err = resolve_under_data_root(output_path, allow_create=True)
        if _err:
            st.error(f"Output path is invalid: {_err}.")
            st.stop()
        try:
            # Initialize JobResult
            job_result = JobResult(
                environment=environment,
                project_id=project_id,
                job_id=st.session_state.job_id,
                suite_id=suite_id,
                suite_ids=selected_suite_ids,
                output_path=str(_resolved)
            )
            log_dicts = job_result.get_case_simulation_log_info()
            st.subheader("Simulation Logs Info")
            if log_dicts:
                display_keys = [
                    "suite_name",
                    "suite_id",
                    "scenario_name", 
                    "archive_id", 
                    "result_json_id", 
                    "scenario_id", 
                    "scenario_ver"
                ]
                # Flatten to DataFrame for selected keys
                table_data = [
                    {k: log.get(k, "") for k in display_keys} 
                    for log in log_dicts
                ]
                df = pd.DataFrame(table_data)
                st.dataframe(df)
            else:
                st.info("No log info found for the current suite/job/project combination.")
        except Exception as e:
            st.error(f"Failed to retrieve log info: {str(e)}")

    if st.button("Download Results", type="primary"):
        st.session_state.stop_downloads = False
        if not all([project_id, st.session_state.job_id]):
            st.error("Please fill in all required fields: Project ID and Job ID")
            st.stop()
        resolved_output, path_err = resolve_under_data_root(output_path, allow_create=True)
        if path_err:
            st.error(f"Output path is invalid: {path_err}. Use a path under the server data root.")
            st.stop()
        resolved_path_str = str(resolved_output)
        set_config_value("output_path", to_data_relative(resolved_output))
        set_config_value("environment", environment)
        set_config_value("project_id", project_id)
        set_config_value("job_id", st.session_state.job_id)
        set_config_value("suite_id", suite_id)
        set_config_value("suite_ids", selected_suite_ids)
        set_config_value("download_type", download_type)
        if download_type == "Archives (ZIP)":
            set_config_value("phase", phase)
            set_config_value("skip_large_file", skip_large_file)
            set_config_value("large_file_mb", large_file_mb)
            set_config_value("keep_zip_files", keep_zip_files)

        if is_task_queue_enabled():
            params = {
                "output_path": resolved_path_str,
                "project_id": project_id,
                "job_id": st.session_state.job_id,
                "suite_id": suite_id or "",
                "suite_ids": selected_suite_ids or None,
                "download_type": "archives" if download_type == "Archives (ZIP)" else "result_json",
                "phase": phase if download_type == "Archives (ZIP)" else "",
                "skip_large_file": skip_large_file,
                "large_file_mb": large_file_mb,
                "keep_zip_files": keep_zip_files,
            }
            task_id = _enqueue_task("download_results", params)
            if task_id:
                st.success("Task queued. It will appear in the **Task status** section below; the list updates automatically.")
            else:
                st.error("Failed to enqueue task. Check REDIS_URL and DATABASE_URL.")
            st.stop()

        # Create output directory (inline path)
        os.makedirs(resolved_path_str, exist_ok=True)
        try:
            job_result = JobResult(
                environment=environment,
                project_id=project_id,
                job_id=st.session_state.job_id,
                suite_id=suite_id,
                suite_ids=selected_suite_ids,
                output_path=resolved_path_str,
            )
            download_successful = False
            if download_type == "Archives (ZIP)":
                with st.expander("Downloading Archives", expanded=True):
                    remain_list = job_result.download_archive_and_unzip(
                        phase,
                        skip_large_file=skip_large_file,
                        large_file_mb=large_file_mb,
                        keep_zip_files=keep_zip_files,
                    )
                    st.success(f"✅ Downloaded and extracted {len(remain_list)} archives")
                    download_successful = len(remain_list) > 0
                    render_job_archives_summary_panel(
                        scenarios_processed=len(remain_list),
                        output_rel=to_data_relative(resolved_path_str),
                    )
            else:  # Result JSON only
                with st.expander("Downloading Result JSON", expanded=True):
                    log_dicts = job_result.download_result_json()
                    st.success(f"✅ Downloaded {len(log_dicts)} JSON files")
                    download_successful = len(log_dicts) > 0
                    render_job_json_summary_panel(
                        json_files=len(log_dicts),
                        output_rel=to_data_relative(resolved_path_str),
                    )
            
            # Show file tree
            with st.expander("📁 File Structure"):
                for root, dirs, files in os.walk(resolved_path_str):
                    level = root.replace(resolved_path_str, '').count(os.sep)
                    indent = ' ' * 4 * level
                    st.text(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 4 * (level + 1)
                    for file in files:
                        st.text(f"{subindent}{file}")
            
            # Suggest next step if download succeeded
            if download_successful:
                st.info("🎉 Download complete! To generate the final summary CSV files, go to the **'Eval Results'** tab and run the evaluation.")
                        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

    # === Combined Download + Eval + Parquet Button ===
    st.divider()
    st.subheader("🚀 Combined Workflow: Download + Eval + Parquet")
    st.caption("Download results, run evaluation, and generate parquet in one click. Eval only runs if download succeeds.")
    
    # Options for combined workflow
    col_combo1, col_combo2 = st.columns(2)
    with col_combo1:
        combined_run_eval = st.checkbox(
            "Run evaluation (eval_result + Summary/Score CSV)",
            value=True,
            key="combined_run_eval",
            help="Run eval_result on downloaded directories and generate Summary.csv/Score.csv"
        )
    with col_combo2:
        combined_generate_parquet = st.checkbox(
            "Generate parquet",
            value=CATALOG_IO_AVAILABLE,
            key="combined_generate_parquet",
            help="Build scene_result.parquet from .pkl files" if CATALOG_IO_AVAILABLE else "Install perception_catalog_analyzer to enable",
            disabled=not CATALOG_IO_AVAILABLE,
        )
    
    combined_eval_recursive = st.checkbox(
        "Search subdirectories for eval",
        value=True,
        key="combined_eval_recursive",
        help="Recursively search for result directories"
    )
    
    if st.button("📥 Download + Eval + Parquet", type="primary", key="download_and_eval_btn"):
        st.session_state.stop_downloads = False
        if not all([project_id, st.session_state.job_id]):
            st.error("Please fill in all required fields: Project ID and Job ID")
            st.stop()
        resolved_output, path_err = resolve_under_data_root(output_path, allow_create=True)
        if path_err:
            st.error(f"Output path is invalid: {path_err}. Use a path under the server data root.")
            st.stop()
        resolved_path_str = str(resolved_output)
        set_config_value("output_path", to_data_relative(resolved_output))
        set_config_value("environment", environment)
        set_config_value("project_id", project_id)
        set_config_value("job_id", st.session_state.job_id)
        set_config_value("suite_id", suite_id)
        set_config_value("suite_ids", selected_suite_ids)
        set_config_value("download_type", download_type)
        if download_type == "Archives (ZIP)":
            set_config_value("phase", phase)
            set_config_value("skip_large_file", skip_large_file)
            set_config_value("large_file_mb", large_file_mb)
            set_config_value("keep_zip_files", keep_zip_files)

        if is_task_queue_enabled():
            # Enqueue combined task
            params = {
                "output_path": resolved_path_str,
                "project_id": project_id,
                "job_id": st.session_state.job_id,
                "suite_id": suite_id or "",
                "suite_ids": selected_suite_ids or None,
                "download_type": "archives" if download_type == "Archives (ZIP)" else "result_json",
                "phase": phase if download_type == "Archives (ZIP)" else "",
                "skip_large_file": skip_large_file,
                "large_file_mb": large_file_mb,
                "keep_zip_files": keep_zip_files,
                "run_eval": combined_run_eval,
                "generate_parquet": combined_generate_parquet,
                "eval_recursive": combined_eval_recursive,
                "eval_overwrite": False,
            }
            task_id = _enqueue_task("download_and_eval", params)
            if task_id:
                st.success("Combined task queued. It will appear in the **Task status** section below; the list updates automatically.")
                st.info("The task will: 1) Download results → 2) Run eval (if download succeeds) → 3) Generate parquet (if download succeeds)")
            else:
                st.error("Failed to enqueue task. Check REDIS_URL and DATABASE_URL.")
            st.stop()

        # Inline execution (non-task-queue mode)
        os.makedirs(resolved_path_str, exist_ok=True)
        try:
            job_result = JobResult(
                environment=environment,
                project_id=project_id,
                job_id=st.session_state.job_id,
                suite_id=suite_id,
                suite_ids=selected_suite_ids,
                output_path=resolved_path_str,
            )
            
            # Progress containers
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            def inline_progress(msg: str):
                status_placeholder.info(msg)
            
            # Step 1: Download
            progress_placeholder.info("📥 Step 1/3: Downloading results...")
            download_successful = False
            if download_type == "Archives (ZIP)":
                with st.expander("Downloading Archives", expanded=True):
                    remain_list = job_result.download_archive_and_unzip(
                        phase,
                        skip_large_file=skip_large_file,
                        large_file_mb=large_file_mb,
                        keep_zip_files=keep_zip_files,
                    )
                    download_successful = len(remain_list) > 0
                    st.success(f"✅ Downloaded and extracted {len(remain_list)} archives")
            else:
                with st.expander("Downloading Result JSON", expanded=True):
                    log_dicts = job_result.download_result_json()
                    download_successful = len(log_dicts) > 0
                    st.success(f"✅ Downloaded {len(log_dicts)} JSON files")
            
            if not download_successful:
                st.error("❌ Download failed. Cannot continue with evaluation.")
                st.stop()
            
            # Step 2: Run eval
            if combined_run_eval:
                progress_placeholder.info("🧮 Step 2/3: Running evaluation...")
                target_dirs = find_eval_result_dirs(resolved_path_str, recursive=combined_eval_recursive)
                if target_dirs:
                    eval_results = []
                    eval_progress = st.progress(0)
                    for i, result_dir in enumerate(target_dirs):
                        eval_progress.progress((i + 1) / len(target_dirs), f"Evaluating {i+1}/{len(target_dirs)}")
                        eval_results.append(run_eval_result_for_dir(result_dir, overwrite=False))
                    eval_progress.empty()
                    
                    success_eval = sum(1 for r in eval_results if r["status"] == "success")
                    failed_eval = sum(1 for r in eval_results if r["status"] == "failed")
                    st.success(f"✅ Eval complete: {success_eval} success, {failed_eval} failed")
                    
                    # Generate summary CSVs
                    with st.spinner("Generating Summary.csv and Score.csv..."):
                        csv_info = generate_summary_and_score_csv(resolved_path_str)
                    st.success(f"Generated Summary.csv ({csv_info['summary_rows']} rows) and Score.csv ({csv_info['score_rows']} rows)")
                else:
                    st.warning("⚠️ No eval result directories found")
            
            # Step 3: Generate parquet
            if combined_generate_parquet and CATALOG_IO_AVAILABLE:
                progress_placeholder.info("📦 Step 3/3: Generating parquet...")
                pkl_dir = Path(resolved_path_str)
                all_pkl_files = list(pkl_dir.rglob("*.pkl")) + list(pkl_dir.rglob("*.pkl.z"))
                pkl_count = len(all_pkl_files)
                if pkl_count > 0:
                    with st.spinner(f"Processing {pkl_count} pkl files..."):
                        parquet_path = pkl_archive_to_parquet(
                            pkl_dir,
                            on_progress=None,
                            on_skip=None,
                            project_id=project_id,
                            job_id=st.session_state.job_id,
                        )
                    st.success(f"✅ Parquet generated: {parquet_path}")
                else:
                    st.warning("⚠️ No .pkl files found for parquet generation")
            
            progress_placeholder.empty()
            status_placeholder.empty()
            st.success("🎉 Combined workflow complete!")
            
            # Show file tree
            with st.expander("📁 File Structure"):
                for root, dirs, files in os.walk(resolved_path_str):
                    level = root.replace(resolved_path_str, '').count(os.sep)
                    indent = ' ' * 4 * level
                    st.text(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 4 * (level + 1)
                    for file in files:
                        st.text(f"{subindent}{file}")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

    # Information section
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. **Get your IDs:**
            - Project ID, Job ID, and Suite ID can be found in the Autoware Evaluator URL
            - Example URL: `https://evaluation.ci.web.auto/v3/projects/{project_id}/jobs/{job_id}`
        
        2. **Choose download type:**
            - **Archives (ZIP):** Downloads zip files and extracts specific phase data
            - **Result JSON only:** Downloads only the result JSON files
        
        3. **Output:**
            - Files will be saved to the specified output directory
            - Each scenario gets its own folder
        """)


with tab2:
    st.header("Download Scenarios")
    
    col1, col2 = st.columns(2)
    
    with col1:
        scenario_filter = st.text_input(
            "Filter by scenario name (optional)",
            placeholder="e.g., intersection, highway",
            help="Only download scenarios containing this text in their name"
        )
    
    with col2:
        overwrite = st.checkbox(
            "Overwrite existing files",
            value=False,
            help="If checked, will redownload scenarios even if they already exist"
        )
    
    # Advanced options
    with st.expander("Advanced Options"):
        st.write("Specify specific scenario IDs to download (one per line):")
        scenario_ids_text = st.text_area(
            "Scenario IDs",
            placeholder="Enter one scenario ID per line",
            height=100
        )
    
    if st.button("Download Scenarios", type="primary", key="download_scenarios"):
        if not all([project_id, st.session_state.job_id]):
            st.error("Please fill in all required fields: Project ID and Job ID")
            st.stop()
        resolved_output, path_err = resolve_under_data_root(output_path, allow_create=True)
        if path_err:
            st.error(f"Output path is invalid: {path_err}.")
            st.stop()
        out_path = str(resolved_output)

        if is_task_queue_enabled():
            selected_ids = None
            if scenario_ids_text:
                selected_ids = [id.strip() for id in scenario_ids_text.split("\n") if id.strip()]
            params = {
                "output_dir": out_path,
                "output_path": out_path,
                "project_id": project_id,
                "job_id": st.session_state.job_id,
                "suite_id": suite_id or "",
                "suite_ids": selected_suite_ids or None,
                "overwrite": overwrite,
                "scenario_name_filter": scenario_filter or None,
                "selected_ids": selected_ids,
            }
            task_id = _enqueue_task("download_scenarios", params)
            if task_id:
                st.success("Task queued. It will appear in the **Task status** section below; the list updates automatically.")
            else:
                st.error("Failed to enqueue task. Check REDIS_URL and DATABASE_URL.")
            st.stop()

        # Parse scenario IDs (inline path)
        selected_ids = None
        if scenario_ids_text:
            selected_ids = [id.strip() for id in scenario_ids_text.split('\n') if id.strip()]
            st.info(f"Will download {len(selected_ids)} specific scenarios")
        
        try:
            job_result = JobResult(
                environment=environment,
                project_id=project_id,
                job_id=st.session_state.job_id,
                suite_id=suite_id,
                suite_ids=selected_suite_ids,
                output_path=out_path
            )
            with st.expander("Downloading Scenarios", expanded=True):
                downloaded = job_result.download_scenarios(
                    output_dir=out_path,
                    scenario_name_filter=scenario_filter,
                    overwrite=overwrite,
                    selected_ids=selected_ids
                )

                # Also download result JSON files (same as tab1 "Result JSON only")
                log_dicts = job_result.download_result_json()

                # Store in session state
                st.session_state.downloaded_scenarios = downloaded
                
                success_scenarios = [s for s in downloaded if s["status"] == "success"]
                render_detailed_scenario_download_panel(
                    success_rows=[(s["name"], str(s["id"])) for s in success_scenarios],
                    json_file_count=len(log_dicts),
                )
                
        except Exception as e:
            st.error(f"❌ Error downloading scenarios: {str(e)}")
            st.exception(e)

with tab3:
    st.header("View Downloaded Content")
    _tab3_resolved, _tab3_err = resolve_under_data_root(output_path, allow_missing=True)
    _tab3_output = str(_tab3_resolved) if _tab3_resolved else None
    if _tab3_err or not _tab3_output or not os.path.exists(_tab3_output):
        st.info("No downloads yet. Use the other tabs to download content first.")
    else:
        # Show directory structure
        st.subheader("📁 Directory Structure")
        
        # Let user browse directories
        selected_dir = st.selectbox(
            "Browse directory",
            [_tab3_output] + [os.path.join(_tab3_output, d) for d in os.listdir(_tab3_output) 
                            if os.path.isdir(os.path.join(_tab3_output, d))]
        )
        
        if os.path.exists(selected_dir):
            # Show files in selected directory
            st.write(f"**Files in `{selected_dir}`:**")
            
            files = os.listdir(selected_dir)
            if files:
                for file in sorted(files):
                    file_path = os.path.join(selected_dir, file)
                    if os.path.isdir(file_path):
                        st.write(f"📁 **{file}/**")
                        # Show files in subdirectory
                        sub_files = os.listdir(file_path)
                        for sub_file in sorted(sub_files)[:10]:  # Limit to 10 files
                            st.write(f"    └─ {sub_file}")
                        if len(sub_files) > 10:
                            st.write(f"    └─ ... and {len(sub_files) - 10} more files")
                    else:
                        file_size = os.path.getsize(file_path)
                        size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
                        st.write(f"📄 {file} ({size_str})")
            else:
                st.write("No files in this directory")
        
        # Show session state downloads
        if st.session_state.downloaded_scenarios:
            render_recent_scenario_downloads_intro()
            df_data = []
            for scenario in st.session_state.downloaded_scenarios:
                df_data.append({
                    "Scenario Name": scenario["name"],
                    "ID": scenario["id"],
                    "Status": scenario["status"],
                    "Path": scenario.get("path", "N/A")
                })
            
            import pandas as pd
            df = pd.DataFrame(df_data)
            st.dataframe(df, width="stretch")


with tab4:
    st.header("Eval Results")

    # Safe default so tab4 never fails with NameError if sidebar output_path wasn't set (e.g. fragment-only run)
    try:
        _default_eval_root = str(output_path)
    except NameError:
        _default_eval_root = "download"
    _stored_eval = get_config_value("eval_root", _default_eval_root)
    _display_eval = to_data_relative(_stored_eval) if _stored_eval else _default_eval_root
    eval_root = st.text_input(
        "Root directory to evaluate",
        value=_display_eval,
        help="Folder under the data directory (e.g. download or a run name). Path is limited to the server data root. Uploaded files are also saved here.",
    )
    set_config_value("eval_root", eval_root)

    # --- Upload local file (Summary CSV / Score CSV / Parquet) into the root directory above ---
    with st.expander("📤 Upload local file (Summary.csv, Score.csv, or parquet)", expanded=False):
        st.caption(
            "Upload Summary CSV, Score CSV, or parquet file(s). They will be saved into the **Root directory to evaluate** above so the dashboard can use them (Overview, Detection Stats, Bounding Box Viewer)."
        )
        uploaded_files = st.file_uploader(
            "Choose file(s)",
            type=["csv", "parquet"],
            accept_multiple_files=True,
            key="tab4_upload_local",
            help="Files are saved into the root directory to evaluate (above).",
        )
        if st.button("Save uploaded file(s) to root directory", key="tab4_upload_btn"):
            if not uploaded_files:
                st.warning("Please select at least one file to upload.")
            elif not eval_root or not str(eval_root).strip():
                st.warning("Please enter Root directory to evaluate above.")
            else:
                resolved, err = resolve_under_data_root(eval_root.strip(), allow_create=True)
                if err:
                    st.error(f"Cannot use that path: {err}. Use a path under the server data root.")
                else:
                    try:
                        target_dir = Path(resolved)
                        target_dir.mkdir(parents=True, exist_ok=True)
                        saved = []
                        for up in uploaded_files:
                            name_lower = (up.name or "").lower()
                            if name_lower == "summary.csv" or ("summary" in name_lower and name_lower.endswith(".csv")):
                                out_name = "Summary.csv"
                            elif name_lower == "score.csv" or ("score" in name_lower and name_lower.endswith(".csv")):
                                out_name = "Score.csv"
                            elif name_lower.endswith(".parquet"):
                                out_name = up.name
                            else:
                                out_name = up.name
                            out_path = target_dir / out_name
                            with open(out_path, "wb") as f:
                                f.write(up.getvalue())
                            saved.append(out_name)
                        st.success(f"Saved {len(saved)} file(s) to **{to_data_relative(target_dir)}**: {', '.join(saved)}. You can use this run in Overview and other pages.")
                    except Exception as e:
                        st.error(f"Failed to save: {e}")
                        st.exception(e)

    col1, col2, col3 = st.columns(3)
    with col1:
        eval_recursive = st.checkbox(
            "Search subdirectories",
            value=get_config_value("eval_recursive", True),
            help="If checked, search all subdirectories for scenario results",
        )
    with col2:
        eval_overwrite = st.checkbox(
            "Overwrite existing result.txt",
            value=get_config_value("eval_overwrite", False),
            help="If unchecked, directories with result.txt will be skipped",
        )
    with col3:
        eval_parallel = st.checkbox(
            "Run in parallel",
            value=get_config_value("eval_parallel", False),
            help="Temporarily disabled. Parallel execution currently provides no measurable benefit.",    
            disabled=True
        )
        if eval_parallel:
            eval_workers = st.number_input(
                "Eval worker threads",
                min_value=1,
                max_value=16,
                value=get_config_value("eval_workers", 1),
                help="Number of parallel threads used to run eval_result",
            )   
            set_config_value("eval_workers", eval_workers)
        else:
            eval_workers = 1
            set_config_value("eval_workers", eval_workers)
    set_config_value("eval_recursive", eval_recursive)
    set_config_value("eval_overwrite", eval_overwrite)
    set_config_value("eval_parallel", eval_parallel)

    # New option: Only generate summary/score csv
    only_generate_summary = st.checkbox(
        "Only generate Summary.csv and Score.csv",
        value=False,
        help="If checked, skip running eval_result per directory and generate only the summary/score CSVs from existing results."
    )

    notify_when_done = st.checkbox(
        "Notify when eval finishes (browser notification)",
        value=get_config_value("eval_notify_when_done", True),
        help="Show a browser notification when evaluation completes so you can switch to other tabs.",
    )
    set_config_value("eval_notify_when_done", notify_when_done)

    def _emit_eval_finished_notification(message: str):
        import html
        import streamlit.components.v1 as components
        try:
            msg_safe = html.escape(message, quote=True)
            html_fragment = f"""
            <span id="eval-msg" data-msg="{msg_safe}"></span>
            <script>
            (function() {{
              if (!("Notification" in window)) return;
              var el = document.getElementById("eval-msg");
              var text = el ? (el.getAttribute("data-msg") || "Eval finished.") : "Eval finished.";
              function show() {{
                if (Notification.permission === "granted")
                  new Notification("Eval finished", {{ body: text }});
                else if (Notification.permission !== "denied")
                  Notification.requestPermission().then(function(p) {{ if (p === "granted") new Notification("Eval finished", {{ body: text }}); }});
              }}
              if (Notification.permission === "granted") show();
              else if (Notification.permission !== "denied") Notification.requestPermission().then(function(p) {{ if (p === "granted") show(); }});
            }})();
            </script>
            """
            components.html(html_fragment, height=0)
        except Exception:
            pass

    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        run_eval_clicked = st.button(
            "Run eval_result" if not only_generate_summary else "Generate Summary and Score CSV only",
            type="primary",
            key="run_eval_result"
        )
    with btn_col2:
        generate_parquet_clicked = st.button(
            "Generate parquet",
            key="generate_parquet_btn",
            type="primary",
            disabled=not CATALOG_IO_AVAILABLE,
            help="Build scene_result.parquet from .pkl files in the evaluate path above (same as Run eval)." if CATALOG_IO_AVAILABLE else "Install perception_catalog_analyzer to enable.",
        )
    with btn_col3:
        generate_both_clicked = st.button(
            "Generate both",
            key="generate_both_btn",
            type="primary",
            disabled=not CATALOG_IO_AVAILABLE,
            help="Generate both parquet and Summary.csv/Score.csv.",
        )
    # Task queue: enqueue eval/parquet tasks and skip inline execution
    if is_task_queue_enabled() and (run_eval_clicked or generate_parquet_clicked or generate_both_clicked):
        resolved_eval, eval_err = resolve_under_data_root(eval_root, allow_missing=True)
        if eval_err:
            st.error(f"Eval root is invalid: {eval_err}. Use a path under the server data root.")
        else:
            eval_path = str(resolved_eval)
            set_config_value("eval_root", to_data_relative(resolved_eval))
            enqueued = []
            if generate_parquet_clicked or generate_both_clicked:
                if CATALOG_IO_AVAILABLE:
                    tid = _enqueue_task(
                        "build_parquet",
                        {
                            "pkl_dir": eval_path,
                            "project_id": project_id or "",
                            "job_id": st.session_state.job_id or "",
                        },
                        job_timeout=_BUILD_PARQUET_JOB_TIMEOUT_SEC,
                    )
                    if tid:
                        enqueued.append(f"build_parquet ({tid[:8]}...)")
                else:
                    st.warning("Parquet task skipped: perception_catalog_io not available.")
            if run_eval_clicked or generate_both_clicked:
                if only_generate_summary:
                    tid = _enqueue_task("generate_summary_csv", {"eval_root": eval_path})
                else:
                    tid = _enqueue_task("run_eval_dirs", {
                        "eval_root": eval_path,
                        "recursive": eval_recursive,
                        "overwrite": eval_overwrite,
                    })
                if tid:
                    enqueued.append(f"{'generate_summary_csv' if only_generate_summary else 'run_eval_dirs'} ({tid[:8]}...)")
            if enqueued:
                st.success("Tasks queued: " + ", ".join(enqueued) + ". See **Task status** below; the list updates automatically.")
            else:
                st.error("Failed to enqueue. Check REDIS_URL and DATABASE_URL.")
        st.stop()

    if generate_parquet_clicked or generate_both_clicked and CATALOG_IO_AVAILABLE:
        resolved, err = resolve_under_data_root(eval_root, allow_missing=True)
        if err:
            st.error(f"Path is invalid: {err}. Use a path under the server data root.")
        else:
            pkl_dir = Path(resolved)
            if not pkl_dir.is_dir():
                st.warning(f"Not a directory: `{pkl_dir}`.")
            else:
                all_pkl_files = list(pkl_dir.rglob("*.pkl")) + list(pkl_dir.rglob("*.pkl.z"))
                pkl_count = len(all_pkl_files)
                if pkl_count == 0:
                    st.warning(f"No .pkl or .pkl.z files in `{pkl_dir}` or its subdirectories.")
                else:
                    _parquet_hud: Optional[ImpressiveProgressHUD] = None
                    try:
                        skip_log = []
                        def on_skip(path: str, reason: str):
                            skip_log.append((path, reason))

                        st.write(f"Processing {pkl_count} pkl files…")
                        _parquet_hud = ImpressiveProgressHUD()
                        start_time = time.time()

                        def _format_eta(sec: float) -> str:
                            if sec is None or sec < 0 or not float("inf") > sec:
                                return "—"
                            m, s = divmod(int(round(sec)), 60)
                            h, m = divmod(m, 60)
                            if h > 0:
                                return f"{h}h {m}m {s}s"
                            if m > 0:
                                return f"{m}m {s}s"
                            return f"{s}s"

                        def _update_parquet_progress(done: int, total: int):
                            elapsed = time.time() - start_time
                            foot = ""
                            if done > 0 and done < total and elapsed > 0:
                                rate = done / elapsed
                                remaining_sec = (total - done) / rate
                                eta_finish = datetime.now() + timedelta(seconds=remaining_sec)
                                foot = (
                                    f"{done}/{total} files · Elapsed {_format_eta(elapsed)} · "
                                    f"Remaining ~{_format_eta(remaining_sec)} · Finish ~{eta_finish.strftime('%H:%M:%S')}"
                                )
                            elif done >= total:
                                foot = f"{total}/{total} files · {_format_eta(elapsed)} total"
                            else:
                                foot = f"{done}/{total} files · Elapsed {_format_eta(elapsed)}"
                            _parquet_hud.show(
                                fraction=(done / total) if total else 1.0,
                                headline="Building parquet",
                                detail=f"Converting catalog pickles ({done} of {total})",
                                foot=foot,
                            )

                        parquet_path = pkl_archive_to_parquet(
                            pkl_dir,
                            on_skip=on_skip,
                            on_progress=_update_parquet_progress,
                            project_id=project_id or None,
                            job_id=st.session_state.job_id or None,
                        )
                        _update_parquet_progress(pkl_count, pkl_count)
                        st.success(f"Saved: `{parquet_path}`")
                        if skip_log:
                            with st.expander("Skipped pkl files"):
                                for path, reason in skip_log:
                                    st.text(f"{os.fspath(path)}: {reason}")
                    except Exception as e:
                        st.error(f"Parquet generation failed: {e}")
                        st.exception(e)
                    finally:
                        if _parquet_hud is not None:
                            _parquet_hud.clear()

    if run_eval_clicked or generate_both_clicked:
        import pandas as pd
        resolved_eval_root, eval_path_err = resolve_under_data_root(eval_root, allow_missing=True)
        if eval_path_err:
            st.error(f"Eval root path is invalid: {eval_path_err}. Use a path under the server data root.")
            st.stop()
        eval_root_resolved = str(resolved_eval_root)
        eval_root_display = to_data_relative(resolved_eval_root)
        set_config_value("eval_root", eval_root_display)

        if only_generate_summary:
            with st.spinner("Generating Summary.csv and Score.csv..."):
                try:
                    csv_info = generate_summary_and_score_csv(eval_root_resolved)
                    st.success(
                        f"Generated Summary.csv ({csv_info['summary_rows']} rows) and "
                        f"Score.csv ({csv_info['score_rows']} rows) in `{eval_root_display}`"
                    )
                    if notify_when_done:
                        _emit_eval_finished_notification(
                            f"Summary.csv ({csv_info['summary_rows']} rows) and Score.csv ({csv_info['score_rows']} rows) in {eval_root_display}"
                        )
                except Exception as e:
                    st.error(f"Failed to generate CSV files: {e}")
                    if notify_when_done:
                        _emit_eval_finished_notification(f"Summary/Score CSV generation failed: {e}")
        else:
            with st.spinner("Searching for result directories..."):
                target_dirs = find_eval_result_dirs(eval_root_resolved, recursive=eval_recursive)
                print("target_dirs", target_dirs)

            if not target_dirs:
                st.warning("No directories found with scenario.yaml and scene_result.pkl")
                st.stop()

            st.write(f"Found {len(target_dirs)} directories to process")
            _eval_hud = ImpressiveProgressHUD()
            results = []
            total = len(target_dirs)
            start_time = time.time()

            def _format_eta(sec: float) -> str:
                if sec is None or sec < 0 or not float("inf") > sec:
                    return "—"
                m, s = divmod(int(round(sec)), 60)
                h, m = divmod(m, 60)
                if h > 0:
                    return f"{h}h {m}m {s}s"
                if m > 0:
                    return f"{m}m {s}s"
                return f"{s}s"

            def _update_progress_status(done: int, total_dirs: int):
                elapsed = time.time() - start_time
                foot = ""
                if done > 0 and done < total_dirs and elapsed > 0:
                    rate = done / elapsed
                    remaining_sec = (total_dirs - done) / rate
                    eta_finish = datetime.now() + timedelta(seconds=remaining_sec)
                    foot = (
                        f"{done}/{total_dirs} dirs · Elapsed {_format_eta(elapsed)} · "
                        f"Remaining ~{_format_eta(remaining_sec)} · Finish ~{eta_finish.strftime('%H:%M:%S')}"
                    )
                elif done >= total_dirs:
                    foot = f"{total_dirs}/{total_dirs} dirs · {_format_eta(elapsed)} total"
                else:
                    foot = f"{done}/{total_dirs} dirs · Elapsed {_format_eta(elapsed)}"
                _eval_hud.show(
                    fraction=(done / total_dirs) if total_dirs else 1.0,
                    headline="Running perception eval",
                    detail=f"Processing result directories ({done} of {total_dirs})",
                    foot=foot,
                )

            try:
                # sequential evaluation
                if not eval_parallel:
                    for i, result_dir in enumerate(target_dirs):
                        _update_progress_status(i, total)
                        results.append(run_eval_result_for_dir(result_dir, overwrite=eval_overwrite))
                        _update_progress_status(i + 1, total)
                else:
                    max_workers = max(1, min(int(eval_workers), len(target_dirs)))
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_map = {
                            executor.submit(_run_eval_result_worker, result_dir, eval_overwrite): result_dir
                            for result_dir in target_dirs
                        }
                        completed = 0
                        for future in as_completed(future_map):
                            completed += 1
                            _update_progress_status(completed, total)
                            try:
                                results.append(future.result())
                            except Exception as e:
                                result_dir = future_map.get(future, "unknown")
                                results.append(
                                    {"path": result_dir, "status": "failed", "detail": str(e)}
                                )

                _update_progress_status(total, total)
            finally:
                _eval_hud.clear()

            success_count = sum(1 for r in results if r["status"] == "success")
            skipped_count = sum(1 for r in results if r["status"] == "skipped")
            failed_count = sum(1 for r in results if r["status"] == "failed")

            st.subheader("📊 Eval Summary")
            st.write(f"- ✅ Success: {success_count}")
            st.write(f"- ⏭️ Skipped: {skipped_count}")
            st.write(f"- ❌ Failed: {failed_count}")

            if results:
                st.subheader("📋 Details")
                st.dataframe(pd.DataFrame(results), width="stretch")

            # Generate summary CSVs at the eval_root level
            with st.spinner("Generating Summary.csv and Score.csv..."):
                try:
                    csv_info = generate_summary_and_score_csv(eval_root_resolved)
                    st.success(
                        f"Generated Summary.csv ({csv_info['summary_rows']} rows) and "
                        f"Score.csv ({csv_info['score_rows']} rows) in `{eval_root_display}`"
                    )
                    if notify_when_done:
                        _emit_eval_finished_notification(
                            f"Eval done. Success: {success_count}, Skipped: {skipped_count}, Failed: {failed_count}. "
                            f"Summary.csv and Score.csv in {eval_root_display}"
                        )
                except Exception as e:
                    st.error(f"Failed to generate CSV files: {e}")
                    if notify_when_done:
                        _emit_eval_finished_notification(
                            f"Eval run finished with CSV error. Success: {success_count}, Skipped: {skipped_count}, Failed: {failed_count}. {e}"
                        )
