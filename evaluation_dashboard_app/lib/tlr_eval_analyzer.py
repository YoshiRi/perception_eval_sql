"""
TLR (Traffic Light Recognition) Evaluation Analyzer.
Loads data from result.json (JSONL, preferred) or scene_result.pkl / *.pkl.z from scenario directories.
"""

import json
import math
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency fallback
    yaml = None


def _obj_to_dict(obj: Any) -> Any:
    """Recursively convert an object to dict/list primitives for TLR frame structure."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _obj_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_obj_to_dict(x) for x in obj]
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return {k: _obj_to_dict(v) for k, v in obj.__dict__.items()}
    return obj


class TLREvaluationAnalyzer:
    """Analyzes TLR evaluation results from a directory of scenario folders (result.json or scene_result.pkl / *.pkl.z)."""

    def __init__(self, result_directory: str):
        self.result_directory = result_directory
        self.scenario_results: Dict[str, List[Dict]] = {}
        self.scenario_paths: Dict[str, Path] = {}
        self.scenario_metadata: Dict[str, Dict[str, str]] = {}
        self.criteria_data: Dict[str, Dict] = {}
        self.cached_vehicle_statuses: Dict[str, List[Dict]] = {}
        self.cached_traffic_light_data: Dict[str, List[Dict]] = {}

    def load_all_results(self) -> None:
        """Load all results: prefer result.json (JSONL), then fall back to pkl (scene_result.pkl / *.pkl.z)."""
        if not self.result_directory or not os.path.isdir(self.result_directory):
            return
        self.load_all_results_from_json()
        if not self.scenario_results:
            self.load_all_results_from_pkl()

    def load_all_results_from_pkl(self) -> None:
        """Load from scene_result.pkl or *.pkl.z in each scenario subdirectory."""
        if not self.result_directory or not os.path.isdir(self.result_directory):
            return
        root = Path(self.result_directory)
        # Collect scenario dirs: direct children with scene_result.pkl or any .pkl.z
        for child in root.iterdir():
            if not child.is_dir():
                continue
            frames = self._load_pkl_scenario(child)
            if frames:
                self.scenario_results[child.name] = frames
                self.scenario_paths[child.name] = child
        # Also support flat layout: root contains *.pkl.z (e.g. archive) — each file = one scenario
        if not self.scenario_results:
            for pkl_path in root.glob("*.pkl.z"):
                frames = self._load_single_pkl_file(pkl_path)
                if frames:
                    self.scenario_results[pkl_path.stem] = frames
                    self.scenario_paths[pkl_path.stem] = pkl_path.parent
            for pkl_path in root.glob("*.pkl"):
                if pkl_path.name == "scene_result.pkl":
                    continue  # already handled as child/scene_result.pkl
                frames = self._load_single_pkl_file(pkl_path)
                if frames:
                    self.scenario_results[pkl_path.stem] = frames
                    self.scenario_paths[pkl_path.stem] = pkl_path.parent

    def _load_pkl_scenario(self, scenario_path: Path) -> List[Dict]:
        """Load one scenario dir: scene_result.pkl or first .pkl.z in that dir."""
        pkl_plain = scenario_path / "scene_result.pkl"
        if pkl_plain.exists():
            return self._load_single_pkl_file(pkl_plain)
        for pklz in scenario_path.glob("*.pkl.z"):
            return self._load_single_pkl_file(pklz)
        return []

    def _load_single_pkl_file(self, pkl_path: Path) -> List[Dict]:
        """Load one .pkl or .pkl.z file and return list of frame dicts (TLR format)."""
        path_str = os.fspath(pkl_path)
        is_pklz = path_str.lower().endswith(".pkl.z")
        try:
            if is_pklz:
                try:
                    import joblib
                    data = joblib.load(pkl_path)
                except ImportError:
                    return []
            else:
                import pickle
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
        except Exception:
            return []
        raw_frames = self._pkl_root_to_frame_list(data)
        result = []
        for raw in raw_frames:
            frame_dict = self._frame_to_tlr_dict(raw)
            if frame_dict is not None:
                result.append(frame_dict)
        return result

    def _pkl_root_to_frame_list(self, data: Any) -> List[Any]:
        """Extract list of frame objects from pkl root (list or WebAutoEvaluatorScenarioResult)."""
        if isinstance(data, list):
            return data
        frame_results = getattr(data, "frame_results", None)
        if isinstance(frame_results, dict) and frame_results:
            # Use first topic (e.g. perception.object_recognition.tracking.objects or TLR topic)
            return next(iter(frame_results.values()), [])
        return []

    def _frame_to_tlr_dict(self, frame: Any) -> Dict | None:
        """Convert one frame (dict or object) to TLR dict with keys Frame, Stamp, Result."""
        if frame is None:
            return None
        if isinstance(frame, dict):
            if "Frame" in frame:
                return frame
            # Nested under frame_result or similar
            for key in ("frame_result", "Frame", "frame"):
                if key in frame and isinstance(frame[key], dict) and "Frame" in frame[key]:
                    return frame[key]
            return None
        # Object: try Frame, Stamp, Result attributes (or __dict__)
        frame_val = getattr(frame, "Frame", None) or getattr(frame, "frame", None)
        stamp_val = getattr(frame, "Stamp", None) or getattr(frame, "stamp", None)
        result_val = getattr(frame, "Result", None) or getattr(frame, "result", None)
        if frame_val is None and hasattr(frame, "__dict__"):
            d = frame.__dict__
            frame_val = d.get("Frame") or d.get("frame")
            stamp_val = stamp_val or d.get("Stamp") or d.get("stamp")
            result_val = result_val or d.get("Result") or d.get("result")
        if frame_val is None:
            return None
        return {
            "Frame": _obj_to_dict(frame_val),
            "Stamp": _obj_to_dict(stamp_val) if stamp_val is not None else {},
            "Result": _obj_to_dict(result_val) if result_val is not None else {},
        }

    def load_all_results_from_json(self) -> None:
        """Load all result.json files from scenario subdirectories (JSONL format).
        Supports two layouts:
        - Flat: result_directory / scenario_name / result.json
        - Suite: result_directory / suite_name / testcase_name / result.json (subfolders are suite folders)
        """
        if not self.result_directory or not os.path.isdir(self.result_directory):
            return
        root = Path(self.result_directory)
        for child in root.iterdir():
            if not child.is_dir():
                continue
            result_file = child / "result.json"
            if result_file.exists():
                # Flat: direct child has result.json
                self.scenario_results[child.name] = self._load_result_jsonl(os.fspath(result_file))
                self.scenario_paths[child.name] = child
            else:
                # Suite: child is a suite folder; look for testcase subdirs with result.json
                for testcase_dir in child.iterdir():
                    if not testcase_dir.is_dir():
                        continue
                    tc_result = testcase_dir / "result.json"
                    if tc_result.exists():
                        scenario_key = f"{child.name}/{testcase_dir.name}"
                        self.scenario_results[scenario_key] = self._load_result_jsonl(os.fspath(tc_result))
                        self.scenario_paths[scenario_key] = testcase_dir

    def _load_result_jsonl(self, file_path: str) -> List[Dict]:
        """Load and parse result.json (JSONL format)."""
        results = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        results.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        pass
        return results

    def extract_criteria_data(self) -> None:
        """Extract criteria evaluation data from all scenarios."""
        for scenario_name, results in self.scenario_results.items():
            if not results:
                continue
            last_frame = None
            for result in results:
                if "Frame" in result and "FinalScore" in result["Frame"]:
                    last_frame = result
                    break
            if not last_frame:
                continue
            summary = last_frame.get("Result", {}).get("Summary", "")
            criteria_data = self._parse_summary(summary)
            if criteria_data:
                self.criteria_data[scenario_name] = criteria_data

    def pre_calculate_all_data(self) -> None:
        """Pre-calculate and cache vehicle statuses and traffic light data for all scenarios."""
        for scenario_name, results in self.scenario_results.items():
            if not results:
                continue
            self.cached_vehicle_statuses[scenario_name] = self._calculate_vehicle_status(results)
            self.cached_traffic_light_data[scenario_name] = self._extract_traffic_light_data(results)

    def _parse_summary(self, summary: str) -> Dict[str, Dict]:
        """Parse the Summary string to extract criteria results."""
        criteria_data = {}
        parts = summary.split(",")
        for part in parts:
            part = part.strip()
            if "criteria" in part and "(" in part and ")" in part:
                criteria_match = part.split("(")[0].strip()
                if "criteria" in criteria_match:
                    criteria_num_str = criteria_match.split()[-1].replace("_", "").replace("criteria", "")
                    try:
                        criteria_key = f"criteria_{int(criteria_num_str)}"
                    except ValueError:
                        continue
                    if "(Success)" in part:
                        result = "Success"
                    elif "(Fail)" in part:
                        result = "Fail"
                    else:
                        continue
                    if ":" in part and "->" in part:
                        count_part = part.split(":")[-1].split("->")[0].strip()
                        if "/" in count_part:
                            tp, total = map(int, count_part.split("/"))
                            tp_rate = tp / total if total > 0 else 0.0
                            criteria_data[criteria_key] = {
                                "result": result,
                                "tp": tp,
                                "total": total,
                                "tp_rate": tp_rate,
                            }
        return criteria_data

    def _calculate_vehicle_status(self, results: List[Dict]) -> List[Dict]:
        """Calculate vehicle status for each frame based on speed and yaw rate."""
        statuses = []
        for i in range(len(results)):
            current_time = None
            if i < 2:
                statuses.append({
                    "status": "No Move",
                    "speed_kph": 0.0,
                    "yaw_rate_deg_s": 0.0,
                    "current_time": 0.0,
                })
                continue
            current_frame = results[i]
            previous_frame = results[i - 2]
            current_pos = self._extract_position(current_frame)
            previous_pos = self._extract_position(previous_frame)
            current_yaw = self._extract_yaw(current_frame)
            previous_yaw = self._extract_yaw(previous_frame)
            if current_pos is None or previous_pos is None:
                current_time = self._extract_ros_timestamp(current_frame) or 0.0
                statuses.append({
                    "status": "No Move",
                    "speed_kph": 0.0,
                    "yaw_rate_deg_s": 0.0,
                    "current_time": current_time,
                })
                continue
            current_time = self._extract_ros_timestamp(current_frame)
            previous_time = self._extract_ros_timestamp(previous_frame)
            if current_time is None or previous_time is None:
                statuses.append({
                    "status": "No Move",
                    "speed_kph": 0.0,
                    "yaw_rate_deg_s": 0.0,
                    "current_time": current_time or 0.0,
                })
                continue
            time_diff = current_time - previous_time
            if time_diff <= 0:
                statuses.append({
                    "status": "No Move",
                    "speed_kph": 0.0,
                    "yaw_rate_deg_s": 0.0,
                    "current_time": current_time,
                })
                continue
            dx = current_pos[0] - previous_pos[0]
            dy = current_pos[1] - previous_pos[1]
            distance = math.sqrt(dx * dx + dy * dy)
            speed_mps = distance / time_diff
            speed_kph = speed_mps * 3.6
            yaw_diff = abs(current_yaw - previous_yaw)
            yaw_diff = min(yaw_diff, 2 * math.pi - yaw_diff)
            yaw_rate_rad_s = yaw_diff / time_diff
            yaw_rate_deg_s = yaw_rate_rad_s * 180 / math.pi
            if yaw_rate_deg_s > 1.5:
                status = "Turning"
            elif speed_kph > 0.5:
                status = "Driving"
            else:
                status = "No Move"
            statuses.append({
                "status": status,
                "speed_kph": speed_kph,
                "yaw_rate_deg_s": yaw_rate_deg_s,
                "current_time": current_time,
            })
        return statuses

    def _extract_position(self, frame_data: Dict) -> Tuple[float, float] | None:
        try:
            ego = frame_data.get("Frame", {}).get("Ego", {})
            transform = ego.get("TransformStamped", {}).get("transform", {})
            translation = transform.get("translation", {})
            return (translation.get("x", 0), translation.get("y", 0))
        except Exception:
            return None

    def _extract_yaw(self, frame_data: Dict) -> float:
        try:
            ego = frame_data.get("Frame", {}).get("Ego", {})
            rotation_euler = ego.get("TransformStamped", {}).get("rotation_euler", {})
            return rotation_euler.get("yaw", 0)
        except Exception:
            return 0.0

    def _extract_ros_timestamp(self, frame_data: Dict) -> float | None:
        try:
            stamp = frame_data.get("Stamp", {})
            ros_time = stamp.get("ROS", None)
            return float(ros_time) if ros_time is not None else None
        except Exception:
            return None

    def _extract_traffic_light_data(self, results: List[Dict]) -> List[Dict]:
        """Extract traffic light recognition data from results."""
        traffic_light_data = []
        for result in results:
            frame_data = result.get("Frame", {})
            traffic_light_info = None
            for criteria_key, criteria_data in frame_data.items():
                if not criteria_key.startswith("criteria") or not isinstance(criteria_data, dict):
                    continue
                if "PassFail" not in criteria_data:
                    continue
                pass_fail = criteria_data["PassFail"]
                if "Info" not in pass_fail:
                    continue
                info = pass_fail["Info"]
                tp_info = info.get("TP", "")
                fp_info = info.get("FP", "")
                fn_info = info.get("FN", "")
                tn_info = info.get("TN", "")
                traffic_light_type = self._parse_traffic_light_type(tp_info)
                if traffic_light_type is None and fn_info and "[" in fn_info and "]" in fn_info:
                    traffic_light_type = self._parse_traffic_light_type(fn_info)
                if traffic_light_type is None:
                    traffic_light_type = "unknown"
                traffic_light_info = {
                    "frame": frame_data.get("FrameName", ""),
                    "traffic_light_type": traffic_light_type,
                    "tp": tp_info,
                    "fp": fp_info,
                    "fn": fn_info,
                    "tn": tn_info,
                    "criteria": criteria_key,
                }
                break
            if not traffic_light_info and "FinalScore" in frame_data:
                final_score = frame_data["FinalScore"]
                if "Score" in final_score and "TP" in final_score["Score"]:
                    tp_scores = final_score["Score"]["TP"]
                    best_type, best_score = "unknown", 0.0
                    for tlr_type, score in tp_scores.items():
                        if tlr_type != "ALL" and isinstance(score, (int, float)) and score > best_score:
                            best_type, best_score = tlr_type, score
                    if best_type != "unknown":
                        traffic_light_info = {
                            "frame": frame_data.get("FrameName", ""),
                            "traffic_light_type": best_type,
                            "tp": f"{best_score}",
                            "fp": "0",
                            "fn": "0",
                            "tn": "0",
                            "criteria": "FinalScore",
                        }
            if traffic_light_info:
                traffic_light_data.append(traffic_light_info)
        return traffic_light_data

    def _parse_traffic_light_type(self, tp_info: str) -> str | None:
        if not tp_info or tp_info == "null":
            return None
        if "[" in tp_info and "]" in tp_info:
            type_part = tp_info.split("[")[1].split("]")[0]
            if type_part:
                return type_part
        return None

    def create_criteria_matrix(self) -> pd.DataFrame:
        """Create matrix of criteria vs TP, total frames, TP rate."""
        criteria_list = [f"criteria_{i}" for i in range(21)]
        matrix_data = []
        for criteria in criteria_list:
            tp_total = total_frames = 0
            for _scenario_name, cdata in self.criteria_data.items():
                if criteria in cdata:
                    tp_total += cdata[criteria]["tp"]
                    total_frames += cdata[criteria]["total"]
            tp_rate = tp_total / total_frames if total_frames > 0 else 0.0
            matrix_data.append({
                "Criteria": criteria,
                "Number of TP": tp_total,
                "Number of total frames": total_frames,
                "TP rate": tp_rate,
            })
        return pd.DataFrame(matrix_data)

    def create_vehicle_status_matrix(self) -> pd.DataFrame:
        """Create matrix of vehicle status vs traffic light type (TP rates)."""
        traffic_light_categories = [
            "green(blue) and criteria0-9",
            "yellow and criteria0-9",
            "red and criteria0-9",
            "other types and criteria0-9",
            "green(blue) and other criteria",
            "yellow and other criteria",
            "red and other criteria",
            "other types and other criteria",
            "all types combined",
        ]
        vehicle_status_categories = ["Turning", "Driving", "No Move", "All Status Combined"]
        matrix_data = []
        for status in vehicle_status_categories:
            row_data = {"Vehicle Status": status}
            for tlr_type in traffic_light_categories:
                _, _, tp_rate = self._calculate_status_tlr_data(status, tlr_type)
                row_data[tlr_type] = tp_rate
            matrix_data.append(row_data)
        return pd.DataFrame(matrix_data)

    def create_vehicle_status_counts_matrix(self) -> pd.DataFrame:
        """Create matrix of vehicle status vs traffic light type (raw counts TP / Total)."""
        traffic_light_categories = [
            "green(blue) and criteria0-9",
            "yellow and criteria0-9",
            "red and criteria0-9",
            "other types and criteria0-9",
            "green(blue) and other criteria",
            "yellow and other criteria",
            "red and other criteria",
            "other types and other criteria",
            "all types combined",
        ]
        vehicle_status_categories = ["Turning", "Driving", "No Move", "All Status Combined"]
        matrix_data = []
        for status in vehicle_status_categories:
            row_data = {"Vehicle Status": status}
            for tlr_type in traffic_light_categories:
                tp_count, total_count, _ = self._calculate_status_tlr_data(status, tlr_type)
                row_data[tlr_type] = f"{tp_count} / {total_count}"
            matrix_data.append(row_data)
        return pd.DataFrame(matrix_data)

    def _calculate_status_tlr_data(self, status: str, tlr_type: str) -> Tuple[int, int, float]:
        total_tp = total_frames = 0
        for scenario_name, results in self.scenario_results.items():
            if not results:
                continue
            vehicle_statuses = self.cached_vehicle_statuses.get(scenario_name)
            traffic_light_data = self.cached_traffic_light_data.get(scenario_name)
            if not vehicle_statuses or not traffic_light_data:
                vehicle_statuses = self._calculate_vehicle_status(results)
                traffic_light_data = self._extract_traffic_light_data(results)
            for frame_status_info, tlr_info in zip(vehicle_statuses, traffic_light_data):
                if status != "All Status Combined" and frame_status_info["status"] != status:
                    continue
                if not self._matches_tlr_category(tlr_info["traffic_light_type"], tlr_type):
                    continue
                criteria_key = tlr_info.get("criteria", "")
                if not self._matches_criteria_range(self._is_criteria0_9(criteria_key), tlr_type):
                    continue
                tp_info = tlr_info.get("tp", "")
                fn_info = tlr_info.get("fn", "")
                has_tp = tp_info and tp_info != "0 []" and tp_info != "null"
                has_fn = fn_info and fn_info != "0 []" and fn_info != "null"
                if has_tp or has_fn:
                    total_frames += 1
                    if has_tp:
                        total_tp += 1
        if total_tp == 0 and total_frames == 0:
            tp_rate = 1.0
        elif total_frames > 0:
            tp_rate = total_tp / total_frames
        else:
            tp_rate = 0.0
        return total_tp, total_frames, tp_rate

    def _matches_tlr_category(self, actual_type: str, category: str) -> bool:
        if category == "all types combined":
            return True
        if "green" in category.lower() and actual_type == "green":
            return True
        if "yellow" in category.lower() and actual_type == "yellow":
            return True
        if "red" in category.lower() and actual_type == "red":
            return True
        if "other types" in category.lower() and actual_type not in ["green", "yellow", "red"]:
            return True
        return False

    def _is_criteria0_9(self, criteria_key: str) -> bool:
        if not criteria_key or criteria_key == "FinalScore":
            return False
        try:
            if "_" in criteria_key:
                criteria_num = int(criteria_key.replace("criteria_", ""))
            else:
                criteria_num = int(criteria_key.replace("criteria", ""))
            return 0 <= criteria_num <= 9
        except ValueError:
            return False

    def _matches_criteria_range(self, is_criteria0_9: bool, tlr_type: str) -> bool:
        if "criteria0-9" in tlr_type:
            return is_criteria0_9
        if "other criteria" in tlr_type:
            return not is_criteria0_9
        if "all types combined" in tlr_type:
            return True
        return False

    def create_vehicle_status_critical_priority_matrix(self) -> pd.DataFrame:
        """Create matrix with critical (criteria5-6) and priority (criteria2-4) zones."""
        traffic_light_categories = [
            "green(blue) and criteria5-6(critical zone)",
            "yellow and criteria5-6(critical zone)",
            "red and criteria5-6(critical zone)",
            "other types and criteria5-6(critical zone)",
            "green(blue) and criteria2-4(priority zone)",
            "yellow and criteria2-4(priority zone)",
            "red and criteria2-4(priority zone)",
            "other types and criteria2-4(priority zone)",
            "green(blue) and other criteria",
            "yellow and other criteria",
            "red and other criteria",
            "other types and other criteria",
            "all types combined",
        ]
        vehicle_status_categories = ["Turning", "Driving", "No Move", "All Status Combined"]
        matrix_data = []
        for status in vehicle_status_categories:
            row_data = {"Vehicle Status": status}
            for tlr_type in traffic_light_categories:
                _, _, tp_rate = self._calculate_status_tlr_data_critical_priority(status, tlr_type)
                row_data[tlr_type] = tp_rate
            matrix_data.append(row_data)
        return pd.DataFrame(matrix_data)

    def create_vehicle_status_critical_priority_counts_matrix(self) -> pd.DataFrame:
        """Raw counts for critical/priority matrix."""
        traffic_light_categories = [
            "green(blue) and criteria5-6(critical zone)",
            "yellow and criteria5-6(critical zone)",
            "red and criteria5-6(critical zone)",
            "other types and criteria5-6(critical zone)",
            "green(blue) and criteria2-4(priority zone)",
            "yellow and criteria2-4(priority zone)",
            "red and criteria2-4(priority zone)",
            "other types and criteria2-4(priority zone)",
            "green(blue) and other criteria",
            "yellow and other criteria",
            "red and other criteria",
            "other types and other criteria",
            "all types combined",
        ]
        vehicle_status_categories = ["Turning", "Driving", "No Move", "All Status Combined"]
        matrix_data = []
        for status in vehicle_status_categories:
            row_data = {"Vehicle Status": status}
            for tlr_type in traffic_light_categories:
                tp_count, total_count, _ = self._calculate_status_tlr_data_critical_priority(status, tlr_type)
                row_data[tlr_type] = f"{tp_count} / {total_count}"
            matrix_data.append(row_data)
        return pd.DataFrame(matrix_data)

    def _calculate_status_tlr_data_critical_priority(self, status: str, tlr_type: str) -> Tuple[int, int, float]:
        total_tp = total_frames = 0
        for scenario_name, results in self.scenario_results.items():
            if not results:
                continue
            vehicle_statuses = self.cached_vehicle_statuses.get(scenario_name)
            traffic_light_data = self.cached_traffic_light_data.get(scenario_name)
            if not vehicle_statuses or not traffic_light_data:
                vehicle_statuses = self._calculate_vehicle_status(results)
                traffic_light_data = self._extract_traffic_light_data(results)
            for frame_status_info, tlr_info in zip(vehicle_statuses, traffic_light_data):
                if status != "All Status Combined" and frame_status_info["status"] != status:
                    continue
                if not self._matches_tlr_category_critical_priority(tlr_info["traffic_light_type"], tlr_type):
                    continue
                criteria_key = tlr_info.get("criteria", "")
                criteria_range = self._get_criteria_range_critical_priority(criteria_key)
                if not self._matches_criteria_range_critical_priority(criteria_range, tlr_type):
                    continue
                tp_info = tlr_info.get("tp", "")
                fn_info = tlr_info.get("fn", "")
                has_tp = tp_info and tp_info != "0 []" and tp_info != "null"
                has_fn = fn_info and fn_info != "0 []" and fn_info != "null"
                if has_tp or has_fn:
                    total_frames += 1
                    if has_tp:
                        total_tp += 1
        if total_tp == 0 and total_frames == 0:
            tp_rate = 1.0
        elif total_frames > 0:
            tp_rate = total_tp / total_frames
        else:
            tp_rate = 0.0
        return total_tp, total_frames, tp_rate

    def _matches_tlr_category_critical_priority(self, actual_type: str, category: str) -> bool:
        if category == "all types combined":
            return True
        if "green" in category.lower() and actual_type == "green":
            return True
        if "yellow" in category.lower() and actual_type == "yellow":
            return True
        if "red" in category.lower() and actual_type == "red":
            return True
        if "other types" in category.lower() and actual_type not in ["green", "yellow", "red"]:
            return True
        return False

    def _get_criteria_range_critical_priority(self, criteria_key: str) -> str:
        if not criteria_key or criteria_key == "FinalScore":
            return "other"
        try:
            if criteria_key.startswith("criteria"):
                if "_" in criteria_key:
                    criteria_num = int(criteria_key.replace("criteria_", ""))
                else:
                    criteria_num = int(criteria_key.replace("criteria", ""))
                if 5 <= criteria_num <= 6:
                    return "critical"
                if 2 <= criteria_num <= 4:
                    return "priority"
                return "other"
        except ValueError:
            pass
        return "other"

    def _matches_criteria_range_critical_priority(self, criteria_range: str, tlr_type: str) -> bool:
        if "criteria5-6(critical zone)" in tlr_type:
            return criteria_range == "critical"
        if "criteria2-4(priority zone)" in tlr_type:
            return criteria_range == "priority"
        if "other criteria" in tlr_type:
            return criteria_range == "other"
        if "all types combined" in tlr_type:
            return True
        return False

    def _get_scenario_metadata(self, scenario_name: str) -> Dict[str, str]:
        """Load lightweight scenario metadata such as t4dataset_id from scenario.yaml."""
        if scenario_name in self.scenario_metadata:
            return self.scenario_metadata[scenario_name]

        metadata: Dict[str, str] = {"t4dataset_id": ""}
        scenario_path = self.scenario_paths.get(scenario_name)
        scenario_yaml_path = None
        if scenario_path:
            direct_yaml_path = scenario_path / "scenario.yaml"
            if direct_yaml_path.is_file():
                scenario_yaml_path = direct_yaml_path
            else:
                # Some TLR layouts store result.json in a suffixed sibling dir
                # (for example ``ScenarioName_4038db04``) while ``scenario.yaml``
                # lives in the unsuffixed directory next to it.
                base_name = scenario_path.name
                if re.fullmatch(r".*_[0-9a-fA-F]{8}", base_name):
                    sibling_yaml_path = scenario_path.parent / base_name.rsplit("_", 1)[0] / "scenario.yaml"
                    if sibling_yaml_path.is_file():
                        scenario_yaml_path = sibling_yaml_path
        if scenario_yaml_path and scenario_yaml_path.is_file() and yaml is not None:
            try:
                with open(scenario_yaml_path, "r", encoding="utf-8") as f:
                    scenario_doc = yaml.safe_load(f) or {}
                datasets = scenario_doc.get("Evaluation", {}).get("Datasets", [])
                if isinstance(datasets, list):
                    dataset_ids = []
                    for item in datasets:
                        if isinstance(item, dict):
                            dataset_ids.extend(str(k) for k in item.keys())
                    metadata["t4dataset_id"] = ", ".join(dataset_ids)
            except (OSError, yaml.YAMLError):
                pass

        self.scenario_metadata[scenario_name] = metadata
        return metadata

    def get_vehicle_status_details_df(self) -> pd.DataFrame | None:
        """Return a DataFrame of per-frame vehicle status and TLR info for all scenarios."""
        all_status_data = []
        for scenario_name, results in self.scenario_results.items():
            if not results:
                continue
            scenario_metadata = self._get_scenario_metadata(scenario_name)
            vehicle_statuses = self.cached_vehicle_statuses.get(scenario_name)
            traffic_light_data = self.cached_traffic_light_data.get(scenario_name)
            if not vehicle_statuses or not traffic_light_data:
                vehicle_statuses = self._calculate_vehicle_status(results)
                traffic_light_data = self._extract_traffic_light_data(results)
            for i, (frame_status_info, tlr_info) in enumerate(zip(vehicle_statuses, traffic_light_data)):
                all_status_data.append({
                    "scenario": scenario_name,
                    "t4dataset_id": scenario_metadata.get("t4dataset_id", ""),
                    "frame_index": i,
                    "frame_name": tlr_info.get("frame", ""),
                    "status": frame_status_info["status"],
                    "speed_kph": frame_status_info["speed_kph"],
                    "yaw_rate_deg_s": frame_status_info["yaw_rate_deg_s"],
                    "current_time": frame_status_info["current_time"],
                    "traffic_light_type": tlr_info.get("traffic_light_type", ""),
                    "criteria": tlr_info.get("criteria", ""),
                    "tp": tlr_info.get("tp", ""),
                    "fp": tlr_info.get("fp", ""),
                    "fn": tlr_info.get("fn", ""),
                    "tn": tlr_info.get("tn", ""),
                })
        return pd.DataFrame(all_status_data) if all_status_data else None

    def get_summary_stats(self) -> Dict[str, Any]:
        """Return overall summary statistics for the Streamlit overview."""
        criteria_df = self.create_criteria_matrix()
        total_tp = int(criteria_df["Number of TP"].sum())
        total_frames = int(criteria_df["Number of total frames"].sum())
        overall_tp_rate = total_tp / total_frames if total_frames > 0 else 0.0
        best = criteria_df.loc[criteria_df["TP rate"].idxmax()] if not criteria_df.empty else None
        worst = criteria_df.loc[criteria_df["TP rate"].idxmin()] if not criteria_df.empty else None
        return {
            "num_scenarios": len(self.scenario_results),
            "num_scenarios_with_criteria": len(self.criteria_data),
            "total_tp": total_tp,
            "total_frames": total_frames,
            "overall_tp_rate": overall_tp_rate,
            "best_criteria": best["Criteria"] if best is not None else None,
            "best_tp_rate": float(best["TP rate"]) if best is not None else None,
            "worst_criteria": worst["Criteria"] if worst is not None else None,
            "worst_tp_rate": float(worst["TP rate"]) if worst is not None else None,
        }
