from pathlib import Path

import subprocess

from lib import local_evaluator_debug as led
from lib.local_evaluator_debug import (
    DEFAULT_WORK_ROOT,
    build_webauto_command,
    default_checkout_path,
    default_debug_container_name,
    default_image_name,
    default_prepared_image_name,
    default_sandbox_container_name,
    docker_images,
    find_repos_file,
    find_webauto_ci_config,
    host_path_for_docker_bind,
    host_visible_webauto_path,
    host_webauto_root,
    load_evaluator_ci_phases,
    read_container_file,
    resolve_webauto_bin,
    resolve_agnocast_env,
    sanitize_name,
    should_run_simulation_pretasks,
    should_run_perception_pretask,
    simulation_pretasks_from_webauto_ci_text,
    start_debug_container,
    tail_text,
)


def test_sanitize_name_keeps_branch_parts_filesystem_safe():
    assert sanitize_name("feature/foo bar") == "feature-foo-bar"
    assert sanitize_name("../../bad branch!!") == "bad-branch"
    assert sanitize_name("", "fallback") == "fallback"


def test_default_names_derive_from_branch():
    branch = "users/me/fix build"
    assert default_image_name(branch) == "pilot-auto:evaluation-users-me-fix-build"
    assert default_checkout_path(branch).name == "users-me-fix-build"
    assert default_debug_container_name("pilot-auto:evaluation-users/me") == "eval-debug-pilot-auto-evaluation-users-me"
    assert default_sandbox_container_name("1234567890", branch) == "evaluator-sandbox-12345678"
    assert (
        default_prepared_image_name("firmware.ci.web.auto/x2_dev/main:abc", "1234567890")
        == "local-evaluator-prepared:firmware.ci.web.auto-x2_dev-main-abc-12345678"
    )


def test_default_work_root_is_not_tmp_for_shared_log_links():
    assert "/tmp/webauto-local-evaluator" not in str(DEFAULT_WORK_ROOT)


def test_tail_text_returns_recent_bytes(tmp_path: Path):
    log = tmp_path / "build.log"
    log.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    assert tail_text(log, max_bytes=11) == "beta\ngamma\n"
    assert tail_text(tmp_path / "missing.log") == ""


def test_find_repos_file_prefers_standard_autoware_repos(tmp_path: Path):
    root_repos = tmp_path / "autoware.repos"
    root_repos.write_text("repositories: {}\n", encoding="utf-8")
    other_repos = tmp_path / "other.repos"
    other_repos.write_text("repositories: {}\n", encoding="utf-8")

    assert find_repos_file(tmp_path) == root_repos


def test_find_repos_file_supports_relative_override(tmp_path: Path):
    nested = tmp_path / "configs" / "debug.repos"
    nested.parent.mkdir()
    nested.write_text("repositories: {}\n", encoding="utf-8")

    assert find_repos_file(tmp_path, "configs/debug.repos") == nested


def test_load_evaluator_ci_phases_reads_common_asset_deploy_from_yaml(tmp_path: Path):
    config = tmp_path / ".webauto-ci.yml"
    config.write_text(
        """
version: 2
artifacts:
  - name: main
    build:
      environment_variables:
        AUTOWARE_PATH: /home/autoware/pilot-auto
      phases:
        - name: environment-setup
          user: root
          exec: ./.webauto-ci/main/environment-setup/run.sh
        - name: autoware-setup
          user: autoware
          workdir: /home/autoware/pilot-auto
          exec: ./.webauto-ci/main/autoware-setup/run.sh
        - name: autoware-build
          user: autoware
          workdir: /home/autoware/pilot-auto
          exec: ./.webauto-ci/main/autoware-build/run.sh
          environment_variables:
            PARALLEL_WORKERS: "32"
        - name: asset-deploy
          user: autoware
          workdir: /home/autoware/pilot-auto
          exec: ./.webauto-ci/common/asset-deploy/run.sh
        - name: ecu-system-setup
          user: autoware
          exec: ./.webauto-ci/main/ecu-system-setup/run.sh
""",
        encoding="utf-8",
    )
    for script in [
        ".webauto-ci/main/environment-setup/run.sh",
        ".webauto-ci/main/autoware-setup/run.sh",
        ".webauto-ci/main/autoware-build/run.sh",
        ".webauto-ci/common/asset-deploy/run.sh",
    ]:
        path = tmp_path / script
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    phases = load_evaluator_ci_phases(tmp_path, artifact_name="main")

    assert find_webauto_ci_config(tmp_path) == config
    assert [phase["name"] for phase in phases] == [
        "environment-setup",
        "autoware-setup",
        "autoware-build",
        "asset-deploy",
    ]
    assert phases[-1]["exec"] == "./.webauto-ci/common/asset-deploy/run.sh"
    assert phases[2]["environment_variables"]["AUTOWARE_PATH"] == "/home/autoware/pilot-auto"
    assert phases[2]["environment_variables"]["PARALLEL_WORKERS"] == "32"


def test_host_path_for_docker_bind_maps_dashboard_data_root(monkeypatch):
    monkeypatch.setattr(led, "LOCAL_EVALUATOR_CONTAINER_DATA_ROOT", "/app/data")
    monkeypatch.setattr(led, "LOCAL_EVALUATOR_HOST_DATA_ROOT", "/host/evaluation_dashboard_app/data")

    assert (
        host_path_for_docker_bind(Path("/app/data/local_evaluator_debug/checkouts/branch"))
        == "/host/evaluation_dashboard_app/data/local_evaluator_debug/checkouts/branch"
    )


def test_host_visible_webauto_path_replaces_tmp_with_shared_fallback(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(led, "LOCAL_EVALUATOR_CONTAINER_DATA_ROOT", str(tmp_path / "container_data"))
    monkeypatch.setattr(led, "LOCAL_EVALUATOR_HOST_DATA_ROOT", str(tmp_path / "host_data"))
    fallback = tmp_path / "container_data" / "runs" / "job" / "work"

    assert host_visible_webauto_path("/tmp/webauto-local-sim", fallback) == str(
        tmp_path / "host_data" / "runs" / "job" / "work"
    )
    assert (tmp_path / "host_data" / "runs" / "job" / "work").is_dir()
    assert fallback.is_dir()


def test_host_webauto_root_prefers_configured_host_path(monkeypatch):
    monkeypatch.setattr(led, "LOCAL_EVALUATOR_HOST_WEBAUTO_ROOT", "/host/user/.webauto")

    assert host_webauto_root() == "/host/user/.webauto"


def test_build_webauto_command_appends_local_defaults_and_keeps_quoted_parameters():
    raw = (
        "webauto ci scenario run --project-id x2_dev "
        "--scenario-id abc --scenario-parameters 'a=1,b=2' --simulation-name perception"
    )

    cmd = build_webauto_command(
        raw,
        image_name="pilot-auto:evaluation-branch",
        container_runtime_path="/tmp/checkout",
        work_dir="/tmp/work",
        asset_dir="/tmp/work/assets",
        timeout="45m",
    )

    assert cmd[:4] == ["webauto", "ci", "scenario", "run"]
    assert "a=1,b=2" in cmd
    assert cmd[cmd.index("--docker-image") + 1] == "pilot-auto:evaluation-branch"
    assert cmd[cmd.index("--container-runtime-path") + 1] == "/tmp/checkout"
    assert cmd[cmd.index("--work-dir") + 1] == "/tmp/work"
    assert cmd[cmd.index("--asset-dir") + 1] == "/tmp/work/assets"
    assert cmd[cmd.index("--timeout") + 1] == "45m"


def test_build_webauto_command_overwrites_bind_sensitive_paths():
    raw = (
        "webauto ci scenario run --project-id x2_dev --scenario-id abc "
        "--container-runtime-path /tmp/old-runtime --work-dir=/tmp/old-work --asset-dir /tmp/old-assets"
    )

    cmd = build_webauto_command(
        raw,
        image_name="pilot-auto:evaluation-branch",
        container_runtime_path="/host/runtime",
        work_dir="/host/work",
        asset_dir="/host/work/assets",
        timeout="45m",
    )

    assert cmd[cmd.index("--container-runtime-path") + 1] == "/host/runtime"
    assert "--work-dir=/host/work" in cmd
    assert cmd[cmd.index("--asset-dir") + 1] == "/host/work/assets"


def test_build_webauto_command_omits_empty_container_runtime_path():
    cmd = build_webauto_command(
        "webauto ci scenario run --project-id x2_dev --scenario-id abc",
        image_name="firmware.ci.web.auto/x2_dev/main:abc",
        container_runtime_path="",
        work_dir="/host/work",
        asset_dir="/host/work/assets",
        timeout="45m",
    )

    assert "--container-runtime-path" not in cmd
    assert cmd[cmd.index("--docker-image") + 1] == "firmware.ci.web.auto/x2_dev/main:abc"


def test_should_run_perception_pretask_auto_detects_perception_command(monkeypatch):
    monkeypatch.delenv("LOCAL_EVALUATOR_RUN_PERCEPTION_PRETASK", raising=False)
    monkeypatch.delenv("LOCAL_EVALUATOR_RUN_SIMULATION_PRETASKS", raising=False)

    assert should_run_simulation_pretasks({"webauto_command": "webauto ci scenario run --simulation-name perception"})
    assert should_run_perception_pretask({"webauto_command": "webauto ci scenario run --simulation-name planning"})
    assert not should_run_simulation_pretasks(
        {"webauto_command": "webauto ci scenario run --simulation-name perception", "run_simulation_pretasks": False}
    )


def test_simulation_pretasks_from_webauto_ci_text_selects_named_simulation():
    text = """
simulations:
  - name: planning
    simulator:
      pre_tasks:
        - exec: echo planning
  - name: perception
    simulator:
      pre_tasks:
        - exec: echo first
        - exec: echo second
          mounts:
            - volume: model
"""

    tasks = simulation_pretasks_from_webauto_ci_text(text, "perception")

    assert tasks == [
        {"name": "perception pre-task 1", "exec": "echo first"},
        {"name": "perception pre-task 2", "exec": "echo second"},
    ]


def test_resolve_agnocast_env_auto_disables_when_device_missing(monkeypatch):
    monkeypatch.setattr(led, "agnocast_device_available", lambda: False)

    assert resolve_agnocast_env("auto") == {"ENABLE_AGNOCAST": "0", "AGNOCAST_BRIDGE_MODE": "off"}
    assert resolve_agnocast_env("disable") == {"ENABLE_AGNOCAST": "0", "AGNOCAST_BRIDGE_MODE": "off"}


def test_resolve_agnocast_env_require_fails_when_device_missing(monkeypatch):
    monkeypatch.setattr(led, "agnocast_device_available", lambda: False)

    try:
        resolve_agnocast_env("require")
    except RuntimeError as exc:
        assert "/dev/agnocast" in str(exc)
    else:
        raise AssertionError("Expected require mode to fail without /dev/agnocast")


def test_build_webauto_command_can_substitute_resolved_webauto_binary():
    cmd = build_webauto_command(
        "webauto ci scenario run --project-id p --scenario-id s",
        webauto_bin="/usr/local/bin/webauto",
        image_name="pilot-auto:evaluation-branch",
        container_runtime_path="/tmp/checkout",
        work_dir="/tmp/work",
        asset_dir="/tmp/work/assets",
        timeout="45m",
    )

    assert cmd[:4] == ["/usr/local/bin/webauto", "ci", "scenario", "run"]


def test_resolve_webauto_bin_returns_none_for_missing_absolute_path():
    assert resolve_webauto_bin("/definitely/missing/webauto") is None


def test_build_webauto_command_keeps_explicit_docker_image():
    cmd = build_webauto_command(
        "webauto ci scenario run --project-id p --scenario-id s --docker-image custom:image",
        image_name="pilot-auto:evaluation-branch",
        container_runtime_path="/tmp/checkout",
        work_dir="/tmp/work",
        asset_dir="/tmp/work/assets",
        timeout="45m",
    )

    assert cmd.count("--docker-image") == 1
    assert cmd[cmd.index("--docker-image") + 1] == "custom:image"


def test_start_debug_container_uses_sleep_infinity(monkeypatch):
    calls = []

    def fake_run(cmd, *, timeout=None):
        calls.append(cmd)
        return subprocess.CompletedProcess(["docker", *cmd], 0, "abc123\n", "")

    monkeypatch.setattr(led, "_run_docker", fake_run)

    result = start_debug_container("pilot-auto:evaluation-x", "debug-x")

    assert result["ok"] == "true"
    assert result["container"] == "debug-x"
    assert calls == [["run", "-dit", "--name", "debug-x", "pilot-auto:evaluation-x", "sleep", "infinity"]]


def test_read_container_file_truncates_large_text(monkeypatch):
    def fake_run(cmd, *, timeout=None):
        return subprocess.CompletedProcess(["docker", *cmd], 0, "abcdef", "")

    monkeypatch.setattr(led, "_run_docker", fake_run)

    result = read_container_file("debug-x", "/tmp/file.txt", max_bytes=3)

    assert result["ok"] is True
    assert result["content"] == "abc"
    assert result["truncated"] is True


def test_docker_images_filters_evaluator_image_prefixes(monkeypatch):
    def fake_run(cmd, *, timeout=None):
        return subprocess.CompletedProcess(
            ["docker", *cmd],
            0,
            "\n".join(
                [
                    "firmware.ci.web.auto/x2_dev/main:a8ba4e6a-7f03-4095-af7c-0a200fd795f7",
                    "pilot-auto:evaluation-branch",
                    "ubuntu:22.04",
                    "<none>:<none>",
                ]
            ),
            "",
        )

    monkeypatch.setattr(led, "_run_docker", fake_run)

    assert docker_images() == [
        "firmware.ci.web.auto/x2_dev/main:a8ba4e6a-7f03-4095-af7c-0a200fd795f7",
        "pilot-auto:evaluation-branch",
    ]
