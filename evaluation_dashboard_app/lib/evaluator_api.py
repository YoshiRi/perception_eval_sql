"""
Evaluator API wrapper for job scheduling and status polling.
Based on evaluator_run_api.py from EvaluatorRunnerUITest, extended with polling support.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import requests
import webautoauth.requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

EVALUATION_API_BASE_URL = "https://evaluation.ci.web.auto/v3"
EVALUATION_REPORT_BASE_URL = "https://evaluation.tier4.jp/evaluation/reports"
DEFAULT_WEBAUTO_AUTH_PATH = Path.home() / ".webauto" / "auth.toml"
SUCCESS_JOB_STATUSES = frozenset({"succeeded", "success"})
FAILED_JOB_STATUSES = frozenset(
    {
        "failed",
        "failure",
        "error",
        "canceled",
        "cancelled",
        "aborted",
        "timed_out",
        "timeout",
    }
)
TERMINAL_JOB_STATUSES = SUCCESS_JOB_STATUSES | FAILED_JOB_STATUSES
_TEST_STATUS_PATHS = (("test", "status"),)
_OVERALL_STATUS_PATHS = (
    ("job", "status"),
    ("evaluation", "status"),
    ("status",),
    ("state",),
)
_BUILD_STATUS_PATHS = (("build", "status"),)


@dataclass(frozen=True)
class TestCaseDefinition:
    test_id: str
    project_id: str
    catalog_id: str
    integration_id: str
    suite_ids: list[str]
    catalog_display_name_prefix: str = ""


class EvaluationAPIError(RuntimeError):
    """Raised when the evaluation API returns an unexpected response."""


def normalize_job_status(status: Any) -> str:
    if status is None:
        return ""
    return str(status).strip().lower()


def _get_first_status(report: dict[str, Any], paths: tuple[tuple[str, ...], ...]) -> str:
    for path in paths:
        current: Any = report
        for key in path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)

        status = normalize_job_status(current)
        if status:
            return status

    return ""


def extract_job_status(report: dict[str, Any]) -> str:
    """Return the best evaluator status from known report response shapes."""
    if not isinstance(report, dict):
        return "unknown"

    test_status = _get_first_status(report, _TEST_STATUS_PATHS)
    if test_status:
        return test_status

    overall_status = _get_first_status(report, _OVERALL_STATUS_PATHS)
    if overall_status:
        return overall_status

    build_status = _get_first_status(report, _BUILD_STATUS_PATHS)
    if build_status:
        return f"build:{build_status}"

    return "unknown"


def is_terminal_job_status(status: Any) -> bool:
    return normalize_job_status(status) in TERMINAL_JOB_STATUSES


def is_success_job_status(status: Any) -> bool:
    return normalize_job_status(status) in SUCCESS_JOB_STATUSES


def get_job_completion(report: dict[str, Any]) -> tuple[bool, str]:
    """
    Return (is_completed, status) for an evaluator job report.

    Build success only means the build phase is done; evaluator jobs can still be
    running suites/tests after that. Build failure is terminal because tests cannot
    proceed, but build success must not unlock downloads by itself.
    """
    if not isinstance(report, dict):
        return False, "unknown"

    status = extract_job_status(report)
    test_status = _get_first_status(report, _TEST_STATUS_PATHS)
    if test_status:
        return is_terminal_job_status(test_status), status

    overall_status = _get_first_status(report, _OVERALL_STATUS_PATHS)
    if overall_status and is_terminal_job_status(overall_status):
        return True, status

    build_status = _get_first_status(report, _BUILD_STATUS_PATHS)
    if build_status in FAILED_JOB_STATUSES:
        return True, status

    return False, status


def load_test_cases(path: Path | str) -> dict[str, dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_test_case(test_id: str, source: Any) -> TestCaseDefinition:
    test_cases = normalize_test_case_mapping(source)
    if test_id not in test_cases:
        raise KeyError(f"Unknown test_id: {test_id}")
    data = test_cases[test_id]
    return make_test_case_definition(test_id, data)


def make_test_case_definition(test_id: str, data: dict[str, Any]) -> TestCaseDefinition:
    return TestCaseDefinition(
        test_id=test_id,
        project_id=data["project_id"],
        catalog_id=data["catalog_id"],
        integration_id=data["integration_id"],
        suite_ids=list(data.get("suite_ids", [])),
        catalog_display_name_prefix=data.get("catalog_display_name_prefix", ""),
    )


def normalize_test_case_mapping(source: Any) -> dict[str, dict[str, Any]]:
    """Normalize a test-case source into a mapping keyed by test_id."""
    if isinstance(source, dict):
        return source
    if isinstance(source, (str, Path)):
        return load_test_cases(Path(source))
    raise TypeError("test case source must be a dict or JSON file path")


def normalize_test_case_definition(
    test_case: Any, *, test_id: str = "custom"
) -> TestCaseDefinition:
    """Normalize one test case definition."""
    if isinstance(test_case, TestCaseDefinition):
        return test_case
    if isinstance(test_case, dict):
        return make_test_case_definition(test_id, test_case)
    raise TypeError("test_case must be a TestCaseDefinition or dict")


def get_job_report_url(project_id: str, job_id: str) -> str:
    return f"{EVALUATION_REPORT_BASE_URL}/{job_id}/?project_id={project_id}"


def get_suite_report_url(project_id: str, job_id: str, suite_report_id: str) -> str:
    return f"{EVALUATION_REPORT_BASE_URL}/{job_id}/tests/{suite_report_id}?project_id={project_id}"


def extract_job_id(url: str) -> str:
    if "/reports/" in url:
        url = url.split("/reports/")[1]
        if "/" in url:
            url = url.split("/")[0]
        if "?" in url:
            url = url.split("?")[0]
    return url


def extract_project_id(url: str) -> str:
    if "project_id=" in url:
        return url.split("project_id=")[1]
    return url


def _make_session(auth_path: Path | str | None = DEFAULT_WEBAUTO_AUTH_PATH):
    """Build authenticated session for evaluation.ci.web.auto API."""
    headers = {
        "Content-Type": "application/json",
        "accept": "application/json",
    }
    if auth_path is not None:
        auth_path = Path(auth_path).expanduser().resolve()
        if not auth_path.exists():
            raise FileNotFoundError(f"webauto auth config not found: {auth_path}")
    from webautoauth.token import HttpService, TokenSource, load_config

    config = load_config()
    token_source = TokenSource(HttpService(config))
    session = webautoauth.requests.make_session(token_source)
    presigned = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    presigned.mount("http://", HTTPAdapter(max_retries=retries))
    presigned.mount("https://", HTTPAdapter(max_retries=retries))
    return session, presigned, headers


def get_evaluator_session(environment: str = "default"):
    """Public API: same session as worker. Returns (session, presigned, headers)."""
    import os
    os.environ["AUTH_PROFILE"] = environment
    return _make_session()


class EvaluationRunAPI:
    """Minimal wrapper for scheduling evaluation jobs and collecting reports."""

    def __init__(
        self,
        api_base_url: str = EVALUATION_API_BASE_URL,
        *,
        auth_path: Path | str | None = DEFAULT_WEBAUTO_AUTH_PATH,
        test_cases: Optional[dict[str, dict[str, Any]]] = None,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self._session, self._presigned, self._headers = _make_session(auth_path)
        self.test_cases = test_cases or {}

    def request(self, url: str, params: Optional[dict[str, Any]] = None, method: str = "GET"):
        if method == "GET":
            from urllib.parse import urlencode
            if params:
                return self._session.get(f"{url}?{urlencode(params)}", headers=self._headers)
            return self._session.get(url, headers=self._headers)

        if method == "POST":
            if params is None:
                return self._session.post(url, headers=self._headers)
            return self._session.post(
                url,
                data=json.dumps(params).encode("utf-8"),
                headers=self._headers,
            )

        raise ValueError(f"Unsupported method: {method}")

    def schedule_job(
        self,
        *,
        project_id: str,
        catalog_id: str,
        integration_id: Optional[str] = None,
        target_name: Optional[str] = None,
        source_job_id: Optional[str] = None,
        suite_ids: Optional[list[str]] = None,
        max_retries: int = 1,
        description: str = "no description",
        clean_build: bool = False,
        debug: bool = False,
        release: bool = False,
        record_caret: bool = False,
        log_expiration_time_in_days: float = 14.0,
        is_tag: bool = False,
    ) -> dict[str, Any]:
        if not source_job_id and not target_name:
            raise ValueError("Either target_name or source_job_id must be provided.")
        payload = {
            "build_options": {
                "clean_build": clean_build,
                "debug": debug,
            },
            "catalog_id": catalog_id,
            "description": description,
            "release": release,
            "suite_ids": suite_ids or [],
            "test_options": {
                "max_retries": max_retries,
                "record_caret": record_caret,
                "log_expiration_time": int(log_expiration_time_in_days * 24 * 60 * 60),
            },
        }
        if integration_id:
            payload["integration_id"] = integration_id
        if source_job_id:
            payload["source_job_id"] = str(source_job_id)
        if target_name:
            payload["source"] = {"git_tag" if is_tag else "git_branch": str(target_name)}
        if record_caret:
            payload["build_options"]["developer_option_names"] = [
                "webauto:ci:caret_enabled"
            ]

        url = f"{self.api_base_url}/projects/{project_id}/jobs/schedule"
        response = self.request(url, payload, method="POST")
        if response is None:
            raise EvaluationAPIError("No response returned from evaluation API")
        if response.status_code != 202:
            raise EvaluationAPIError(
                f"Failed to schedule job: status={response.status_code}, body={response.text}"
            )
        return json.loads(response.content)

    def schedule_job_by_test_id(
        self,
        test_id: str,
        *,
        target_name: str,
        test_cases: Any = None,
        max_retries: int = 1,
        description: str = "no description",
        clean_build: bool = False,
        debug: bool = False,
        release: bool = False,
        record_caret: bool = False,
        log_expiration_time_in_days: float = 14.0,
        is_tag: bool = False,
    ) -> dict[str, Any]:
        if test_cases is None:
            if not self.test_cases:
                raise ValueError(
                    "No test case source provided. Pass `test_cases=...` or use schedule_job()."
                )
            source = self.test_cases
        else:
            source = test_cases

        test_case = resolve_test_case(test_id, source)
        return self.schedule_job(
            project_id=test_case.project_id,
            catalog_id=test_case.catalog_id,
            integration_id=test_case.integration_id,
            target_name=target_name,
            suite_ids=test_case.suite_ids,
            max_retries=max_retries,
            description=description,
            clean_build=clean_build,
            debug=debug,
            release=release,
            record_caret=record_caret,
            log_expiration_time_in_days=log_expiration_time_in_days,
            is_tag=is_tag,
        )

    def schedule_job_by_definition(
        self,
        test_case: TestCaseDefinition | dict[str, Any],
        *,
        target_name: str,
        test_id: str = "custom",
        max_retries: int = 1,
        description: str = "no description",
        clean_build: bool = False,
        debug: bool = False,
        release: bool = False,
        record_caret: bool = False,
        log_expiration_time_in_days: float = 14.0,
        is_tag: bool = False,
    ) -> dict[str, Any]:
        definition = normalize_test_case_definition(test_case, test_id=test_id)
        return self.schedule_job(
            project_id=definition.project_id,
            catalog_id=definition.catalog_id,
            integration_id=definition.integration_id,
            target_name=target_name,
            suite_ids=definition.suite_ids,
            max_retries=max_retries,
            description=description,
            clean_build=clean_build,
            debug=debug,
            release=release,
            record_caret=record_caret,
            log_expiration_time_in_days=log_expiration_time_in_days,
            is_tag=is_tag,
        )

    def get_job_status(self, project_id: str, job_id: str) -> dict[str, Any]:
        """Get current job status from the API."""
        url = f"{self.api_base_url}/projects/{project_id}/jobs/{job_id}/report"
        response = self.request(url, {})
        if response is None:
            raise EvaluationAPIError("No response returned from evaluation API")
        if response.status_code != 200:
            raise EvaluationAPIError(
                f"Failed to get job status: status={response.status_code}, body={response.text}"
            )
        return json.loads(response.content)

    def is_job_completed(self, project_id: str, job_id: str) -> tuple[bool, str, dict[str, Any]]:
        """
        Check if a job has completed (success or failure).
        Returns (is_completed, status, report_data).
        Status can be: 'pending', 'running', 'succeeded', 'failed', 'canceled', 'unknown'
        """
        report = self.get_job_status(project_id, job_id)
        
        is_completed, status = get_job_completion(report)
        
        return is_completed, status, report

    def wait_for_job_completion(
        self,
        project_id: str,
        job_id: str,
        poll_interval: float = 60.0,
        max_wait_seconds: float = 0.0,
        on_progress: Optional[Callable[[str], None]] = None,
        on_check: Optional[Callable[[str, float], None]] = None,
    ) -> dict[str, Any]:
        """
        Poll job status until completion or timeout.
        
        Args:
            project_id: Project ID
            job_id: Job ID to wait for
            poll_interval: Seconds between status checks (default 60s)
            max_wait_seconds: Maximum seconds to wait. Values <= 0 disable timeout.
            on_progress: Callback for progress messages (receives message string)
            on_check: Callback after each check (receives status string, elapsed seconds)
        
        Returns:
            Final job report dict
        
        Raises:
            EvaluationAPIError: If timeout or API error
        """
        start_time = time.time()
        last_status = "unknown"
        
        if on_progress:
            on_progress(f"Waiting for evaluator job {job_id} to complete...")
        
        while True:
            elapsed = time.time() - start_time
            
            # Values <= 0 mean "wait indefinitely".
            if max_wait_seconds > 0 and elapsed > max_wait_seconds:
                raise EvaluationAPIError(
                    f"Timeout waiting for job {job_id} after {elapsed:.0f}s"
                )
            
            try:
                is_completed, status, report = self.is_job_completed(project_id, job_id)
                last_status = status
                
                if on_check:
                    on_check(status, elapsed)
                
                if is_completed:
                    if on_progress:
                        on_progress(f"Job {job_id} completed with status: {status}")
                    return report
                
                # Log progress periodically (every 5 minutes or on status change)
                if on_progress and (elapsed < 60 or int(elapsed) % 300 < poll_interval):
                    on_progress(
                        f"Job {job_id} status: {status} (elapsed: {elapsed/3600:.1f}h)"
                    )
                
            except Exception as e:
                if on_progress:
                    on_progress(f"Error checking job status: {e}")
                # Continue polling on transient errors
            
            time.sleep(poll_interval)

    def get_report_list(
        self,
        project_id: str,
        *,
        status: str = "all",
        max_results: Optional[int] = None,
        catalog_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        next_token = ""
        url = f"{self.api_base_url}/projects/{project_id}/jobs/reports"
        while True:
            params = {
                "next_token": next_token,
                "size": 100,
                "status": status,
            }
            if catalog_id is not None:
                params["catalog_id"] = catalog_id

            response = self.request(url, params)
            if response is None:
                raise EvaluationAPIError("No response returned from evaluation API")
            if response.status_code != 200:
                raise EvaluationAPIError(
                    f"Failed to fetch report list: status={response.status_code}, body={response.text}"
                )

            data = json.loads(response.content)
            reports.extend(data.get("reports", []))
            next_token = data.get("next_token", "")
            if next_token == "":
                return reports
            if max_results is not None and len(reports) >= max_results:
                return reports[:max_results]

    def search_report_list(
        self,
        project_id: str,
        *,
        filters: Optional[list[dict[str, Any]]] = None,
        sort: Optional[list[dict[str, Any]]] = None,
        next_token: str = "",
        size: int = 100,
    ) -> dict[str, Any]:
        url = f"{self.api_base_url}/projects/{project_id}/jobs/reports/search"
        payload: dict[str, Any] = {
            "size": max(1, min(int(size), 100)),
        }
        if next_token:
            payload["next_token"] = next_token
        if filters:
            payload["filters"] = filters
        if sort:
            payload["sort"] = sort

        response = self.request(url, payload, method="POST")
        if response is None:
            raise EvaluationAPIError("No response returned from evaluation API")
        if response.status_code != 200:
            raise EvaluationAPIError(
                f"Failed to search report list: status={response.status_code}, body={response.text}"
            )
        return json.loads(response.content)

    def get_suite_reports(self, project_id: str, job_id: str) -> list[dict[str, Any]]:
        return self._get_paginated_reports(
            f"{self.api_base_url}/projects/{project_id}/jobs/{job_id}/test/suite/reports"
        )

    def get_spec_reports(self, project_id: str, job_id: str) -> list[dict[str, Any]]:
        return self._get_paginated_reports(
            f"{self.api_base_url}/projects/{project_id}/jobs/{job_id}/test/spec/reports"
        )

    def get_case_reports(self, project_id: str, job_id: str) -> list[dict[str, Any]]:
        return self._get_paginated_reports(
            f"{self.api_base_url}/projects/{project_id}/jobs/{job_id}/test/case/reports"
        )

    def get_build_reports(self, project_id: str, job_id: str) -> dict[str, Any]:
        url = f"{self.api_base_url}/projects/{project_id}/jobs/{job_id}/build/reports"
        response = self.request(url, {})
        if response is None:
            raise EvaluationAPIError("No response returned from evaluation API")
        if response.status_code != 200:
            raise EvaluationAPIError(
                f"Failed to fetch build reports: status={response.status_code}, body={response.text}"
            )
        return json.loads(response.content)

    def get_job_report(self, project_id: str, job_id: str) -> dict[str, Any]:
        url = f"{self.api_base_url}/projects/{project_id}/jobs/{job_id}/report"
        response = self.request(url, {})
        if response is None:
            raise EvaluationAPIError("No response returned from evaluation API")
        if response.status_code != 200:
            raise EvaluationAPIError(
                f"Failed to fetch job report: status={response.status_code}, body={response.text}"
            )
        return json.loads(response.content)

    def get_suite_summary(
        self,
        project_id: str,
        job_id: str,
        *,
        use_available_case_results: bool = False,
    ) -> list[dict[str, Any]]:
        mode = "available_case_results" if use_available_case_results else "case_results"
        summaries: list[dict[str, Any]] = []
        for suite_report in self.get_suite_reports(project_id, job_id):
            if mode not in suite_report:
                continue

            result = suite_report[mode]
            cancellation_count = result.get("cancellation_count", 0)
            summaries.append(
                {
                    "name": suite_report["suite"]["display_name"],
                    "all": result["total_count"] + cancellation_count,
                    "success": result["success_count"],
                    "fail": result["failure_count"] + cancellation_count,
                    "cancel": cancellation_count,
                    "simulation": suite_report["simulation"]["name"],
                    "url": get_suite_report_url(project_id, job_id, suite_report["id"]),
                }
            )
        return summaries

    def _get_paginated_reports(self, url: str) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        next_token = ""
        while True:
            params = {
                "next_token": next_token,
                "size": 100,
            }
            response = self.request(url, params)
            if response is None:
                raise EvaluationAPIError("No response returned from evaluation API")
            if response.status_code != 200:
                raise EvaluationAPIError(
                    f"Failed to fetch paginated reports: status={response.status_code}, body={response.text}"
                )

            data = json.loads(response.content)
            reports.extend(data.get("reports", []))
            next_token = data.get("next_token", "")
            if next_token == "":
                return reports
