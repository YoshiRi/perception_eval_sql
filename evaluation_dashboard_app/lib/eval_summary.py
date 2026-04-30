"""
Eval and summary logic usable from both Streamlit UI and worker (no Streamlit dependency).
"""

import glob
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from lib.perception_eval_result_summarizer import run_eval_result, generate_score_json


def find_eval_result_dirs(root_dir: str, recursive: bool = True) -> List[str]:
    """Return sorted list of directories under root_dir that contain scenario.yaml and scene_result.pkl."""
    if not os.path.isdir(root_dir):
        return []
    if recursive:
        walker = os.walk(root_dir)
    else:
        walker = [
            (root_dir, [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))], [])
        ]
    result_dirs = []
    for current_dir, subdirs, files in walker:
        if "scenario.yaml" in files and "scene_result.pkl" in files:
            result_dirs.append(current_dir)
    return sorted(result_dirs)


def _run_eval_result_for_dir_inline(result_dir: str, overwrite: bool = False) -> Dict[str, Any]:
    """Run eval_result in the current process and generate score.json for one directory."""
    result_file = os.path.join(result_dir, "result.txt")
    score_file = os.path.join(result_dir, "score.json")
    if os.path.exists(result_file) and not overwrite:
        if os.path.exists(score_file):
            return {"path": result_dir, "status": "skipped", "detail": "result.txt exists"}
        try:
            generate_score_json(result_dir)
            return {"path": result_dir, "status": "success", "detail": "score.json generated"}
        except Exception as e:
            import traceback
            error_output = f"Error: {e}\n{traceback.format_exc()}"
            with open(result_file, "a", encoding="utf-8") as f:
                f.write(f"\n{error_output}")
            return {"path": result_dir, "status": "failed", "detail": str(e)}

    try:
        report_text = run_eval_result(result_dir)
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        generate_score_json(result_dir)
        return {"path": result_dir, "status": "success", "detail": "completed"}
    except Exception as e:
        import traceback
        error_output = f"Error: {e}\n{traceback.format_exc()}"
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(error_output)
        return {"path": result_dir, "status": "failed", "detail": str(e)}


def _signal_detail(returncode: int) -> str:
    """Return a human-readable detail string for a subprocess return code."""
    if returncode < 0:
        sig_num = -returncode
    elif returncode > 128:
        sig_num = returncode - 128
    else:
        return f"exit code {returncode}"
    try:
        sig_name = signal.Signals(sig_num).name
    except ValueError:
        sig_name = f"signal {sig_num}"
    return f"{sig_name} ({sig_num})"


def _write_eval_subprocess_failure(
    result_dir: str,
    message: str,
    stdout: str = "",
    stderr: str = "",
) -> None:
    """Persist native-crash details where the UI and user can inspect them."""
    result_path = Path(result_dir) / "result.txt"
    log_path = Path(result_dir) / "eval_subprocess.log"
    detail = f"Error: {message}\n"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(detail)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(detail)
        if stdout:
            f.write("\n--- stdout ---\n")
            f.write(stdout)
        if stderr:
            f.write("\n--- stderr ---\n")
            f.write(stderr)


def _run_eval_result_for_dir_subprocess(result_dir: str, overwrite: bool = False) -> Dict[str, Any]:
    """Run one scenario eval in a child Python process so native crashes are contained."""
    env = os.environ.copy()
    env.setdefault("PYTHONFAULTHANDLER", "1")
    cmd = [
        sys.executable,
        "-m",
        "lib.eval_summary",
        "__run_eval_dir",
        result_dir,
        "1" if overwrite else "0",
    ]
    completed = subprocess.run(
        cmd,
        cwd=os.fspath(Path(__file__).resolve().parents[1]),
        env=env,
        text=True,
        capture_output=True,
    )
    if completed.returncode == 0:
        for line in reversed(completed.stdout.splitlines()):
            if line.startswith("__EVAL_RESULT_JSON__"):
                try:
                    return json.loads(line.removeprefix("__EVAL_RESULT_JSON__"))
                except json.JSONDecodeError:
                    break
        return {"path": result_dir, "status": "success", "detail": "completed"}

    detail = f"eval subprocess failed with {_signal_detail(completed.returncode)}"
    _write_eval_subprocess_failure(
        result_dir,
        detail,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    return {"path": result_dir, "status": "failed", "detail": detail}


def run_eval_result_for_dir(result_dir: str, overwrite: bool = False) -> Dict[str, Any]:
    """Run eval_result and generate score.json for one directory. Returns status dict."""
    isolated = os.environ.get("EVAL_RUN_ISOLATED_SUBPROCESS", "1").lower()
    if isolated in ("0", "false", "no"):
        return _run_eval_result_for_dir_inline(result_dir, overwrite=overwrite)
    return _run_eval_result_for_dir_subprocess(result_dir, overwrite=overwrite)


def generate_summary_and_score_csv(input_path: str) -> Dict[str, Any]:
    """
    Generate Summary.csv and Score.csv in input_path from each subdirectory's result.txt and score.json.
    Returns dict with summary_path, score_path, summary_rows, score_rows.
    """
    def _infer_suite_name(dir_name: str) -> str:
        base = Path(dir_name).name.rstrip("/")
        parts = base.rsplit("_", 1)
        if len(parts) == 2:
            maybe_uuid = parts[1]
            if len(maybe_uuid) == 36 and maybe_uuid.count("-") == 4:
                return parts[0]
        return base

    result_folders = glob.glob(os.path.join(input_path, "*/"))
    result_folders.sort()
    result_entries: List[Dict[str, str]] = []
    flat_results = False
    for folder in result_folders:
        if os.path.exists(os.path.join(folder, "result.txt")):
            flat_results = True
            result_entries.append({"suite": "", "path": folder})

    if not flat_results:
        for suite_dir in result_folders:
            suite_name = _infer_suite_name(suite_dir)
            suite_cases = glob.glob(os.path.join(suite_dir, "*/"))
            suite_cases.sort()
            for case_dir in suite_cases:
                if os.path.exists(os.path.join(case_dir, "result.txt")):
                    result_entries.append({"suite": suite_name, "path": case_dir})

    summary_lines: List[str] = []
    score_lines: List[str] = []

    for entry in result_entries:
        folder = entry["path"]
        suite_name = entry["suite"]
        result_txt = os.path.join(folder, "result.txt")
        if not os.path.exists(result_txt):
            continue

        data: List[float] = []
        with open(result_txt, "r", encoding="utf-8") as txt:
            found = False
            for input_line in txt:
                if not found:
                    if "TP xave xstd xrms yave ystd yrms vx vy" in input_line:
                        found = True
                else:
                    parts = [p for p in input_line.split(" ") if p.strip() != ""]
                    try:
                        data = [float(s) for s in parts]
                    except ValueError:
                        data = []
                    break

        if not data:
            continue

        scenario_name = Path(folder).name
        tp_percent = data[0] * 100
        x_ave = data[1]
        x_std = data[2]
        x_rms = abs(data[1]) + data[2] * 3
        y_ave = data[4]
        y_std = data[5]
        y_rms = abs(data[4]) + data[5] * 3
        vx = data[7]
        vy = data[8]

        summary_lines.append(
            f"{scenario_name},{tp_percent:.3f},{x_ave:.3f},{x_std:.3f},"
            f"{x_rms:.3f},{y_ave:.3f},{y_std:.3f},{y_rms:.3f},"
            f"{vx:.3f},{vy:.3f},{suite_name}\n"
        )

    for entry in result_entries:
        folder = entry["path"]
        score_json_path = os.path.join(folder, "score.json")
        if not os.path.exists(score_json_path):
            continue

        with open(score_json_path, "r", encoding="utf-8") as f:
            dic = json.load(f)

        line = f"{Path(folder).name},"
        line += f"{dic.get('Option', '')},"
        line += f"{dic.get('criteria0', {}).get('GT_OBJ', '')},"

        dic_items = [(k, v) for k, v in dic.items() if k != "Option" and isinstance(v, dict)]
        num_items = len(dic_items)
        for idx, (k, v) in enumerate(dic_items):
            is_last = idx == (num_items - 1)
            line += f"{k},"
            line += f"{v.get('NM', '')},"
            line += f"{v.get('TP/TN', '')},"
            line += f"{v.get('ADD', '')},"
            line += f"{v.get('AIL', '')},"
            line += f"{v.get('UIL', '')},"
            line += f"{v.get('PFN/PFP', '')},"
            line += f"{v.get('UUID_NUM', '')},"

            nm = v.get("NM", 0)
            try:
                nm_value = float(nm)
            except (TypeError, ValueError):
                nm_value = 0.0
            if nm_value == 0:
                line += "100.0,"
            else:
                try:
                    pass_rate = 100.0 * (
                        float(v.get("TP/TN", 0))
                        + float(v.get("AIL", 0))
                        + float(v.get("ADD", 0))
                    ) / nm_value
                except (TypeError, ValueError, ZeroDivisionError):
                    pass_rate = 0.0
                line += f"{pass_rate:.3f},"

            line += f"{v.get('MAX_DIST_THRESH', '')},"

            obj_cnts = v.get("OBJ_CNTS", {})
            if isinstance(obj_cnts, dict):
                obj_parts = [f"{obj}:{cnt}" for obj, cnt in obj_cnts.items()]
                line += ";".join(obj_parts)
            if not is_last:
                line += ","

        score_lines.append(line + "\n")

    with open(os.path.join(input_path, "Summary.csv"), mode="w", encoding="utf-8") as f:
        f.writelines(summary_lines)
    with open(os.path.join(input_path, "Score.csv"), mode="w", encoding="utf-8") as f:
        f.writelines(score_lines)

    return {
        "summary_path": os.path.join(input_path, "Summary.csv"),
        "score_path": os.path.join(input_path, "Score.csv"),
        "summary_rows": len(summary_lines),
        "score_rows": len(score_lines),
    }


def _main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "__run_eval_dir":
        result_dir = sys.argv[2]
        overwrite = len(sys.argv) >= 4 and sys.argv[3] == "1"
        result = _run_eval_result_for_dir_inline(result_dir, overwrite=overwrite)
        print("__EVAL_RESULT_JSON__" + json.dumps(result, ensure_ascii=False))
        return 0
    print("Usage: python -m lib.eval_summary __run_eval_dir <result_dir> <overwrite:0|1>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
