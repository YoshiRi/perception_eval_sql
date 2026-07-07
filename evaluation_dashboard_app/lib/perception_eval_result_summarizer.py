# import pkg_resources
# v1.1.1
import json
import math
import pandas as pd
from pandas import isnull
import re
import os
import glob
import yaml
import sys
from lib.plot_geom_lib import (
    calculate_common_area,
    calc_line_polygon_distance,
    calc_line_vector_intersection_time,
    plot_rectangles_with_common_area,
)

try:
    from perception_eval.tool import PerceptionAnalyzer3D
except Exception as exc:
    PerceptionAnalyzer3D = None
    _perception_eval_import_error = exc

# from perception_analyzer_v import PerceptionAnalyzerV
obj_group_base = {
    "car": ["car", "truck", "bus"],
    "truck": ["car", "truck", "bus"],
    "bus": ["car", "truck", "bus"],
    "bicycle": ["bicycle", "motorbike"],
    "motorbike": ["bicycle", "motorbike"],
    "pedestrian": ["pedestrian"],
    "unknown": ["unknown", "car", "truck", "bus", "bicycle", "motorbike", "pedestrian"],
    "false_positive": ["false_positive"],
}

hsize_of_ped = 0.6
area0_lat_dist = 25.0
# X2
vehicle_width = 2.230
vehicle_front_extension = 5.6754
rear_overhang = 1.498
front_line = [
    [vehicle_width * -0.5, vehicle_front_extension],
    [vehicle_width * 0.5, vehicle_front_extension],
]
vehicle_polygon = [
    [vehicle_width * -0.5, vehicle_front_extension],
    [vehicle_width * 0.5, vehicle_front_extension],
    [vehicle_width * 0.5, -rear_overhang],
    [vehicle_width * -0.5, -rear_overhang],
]
area0 = [
    [vehicle_width * -0.5, -rear_overhang],
    [vehicle_width * 0.5, -rear_overhang],
    [vehicle_width * 0.5, vehicle_front_extension + area0_lat_dist],
    [vehicle_width * -0.5, vehicle_front_extension + area0_lat_dist],
]
area0_left_line = [
    [vehicle_width * -0.5, -rear_overhang],
    [vehicle_width * -0.5, vehicle_front_extension + area0_lat_dist],
]
area0_right_line = [
    [vehicle_width * 0.5, -rear_overhang],
    [vehicle_width * 0.5, vehicle_front_extension + area0_lat_dist],
]


def get_option_and_object_group(result_directory):
    def get_last_float(string):
        """
        Extract the last float number from the given string.

        Args:
        string (str): The input string to search for float numbers.

        Returns:
        float: The last float number found in the string, or None if no float numbers are found.
        """
        # Use regex to find all float numbers in the string
        # This pattern matches integers and decimal numbers, including negative numbers
        float_pattern = r"-?\d+(?:\.\d+)?"
        numbers = re.findall(float_pattern, string)

        # Return the last number as a float if found, otherwise None
        return float(numbers[-1]) if numbers else None

    res = {
        "Option": "",
        "criteria0": {
            "NM": 0,
            "TP/TN": 0,
            "ADD": 0,
            "AIL": 0,
            "UIL": 0,
            "PFN/PFP": 0,
            "UUID_NUM": 0,
            "GT_OBJ": "",
            "MAX_DIST_THRESH": 0,
            "OBJ_CNTS": {},
        },
        "criteria1": {
            "NM": 0,
            "TP/TN": 0,
            "ADD": 0,
            "AIL": 0,
            "UIL": 0,
            "PFN/PFP": 0,
            "UUID_NUM": 0,
            "GT_OBJ": "",
            "MAX_DIST_THRESH": 0,
            "OBJ_CNTS": {},
        },
        "criteria2": {
            "NM": 0,
            "TP/TN": 0,
            "ADD": 0,
            "AIL": 0,
            "UIL": 0,
            "PFN/PFP": 0,
            "UUID_NUM": 0,
            "GT_OBJ": "",
            "MAX_DIST_THRESH": 0,
            "OBJ_CNTS": {},
        },
        "criteria3": {
            "NM": 0,
            "TP/TN": 0,
            "ADD": 0,
            "AIL": 0,
            "UIL": 0,
            "PFN/PFP": 0,
            "UUID_NUM": 0,
            "GT_OBJ": "",
            "MAX_DIST_THRESH": 0,
            "OBJ_CNTS": {},
        },
    }

    obj_group = {}
    matching_option = ""
    criteria_max_dist = [300.0, 300.0, 300.0, 300.0]
    with open(result_directory + "scenario.yaml", "r") as scenario_file:
        scenario_dic = yaml.safe_load(scenario_file)
        try:
            matching_option = scenario_dic["Evaluation"]["PerceptionEvaluationConfig"][
                "evaluation_config_dict"
            ]["matching_label_policy"]
        except:
            pass
        try:
            for i in range(len(criteria_max_dist) - 1):
                if (
                    "-"
                    not in scenario_dic["Evaluation"]["Conditions"]["Criterion"][i]["Filter"][
                        "Distance"
                    ][-2:]
                ):
                    criteria_max_dist[i] = get_last_float(
                        scenario_dic["Evaluation"]["Conditions"]["Criterion"][i]["Filter"][
                            "Distance"
                        ]
                    )
                else:
                    break
        except:
            pass

        if "ALLOW_UNKNOWN" == matching_option:
            res["Option"] = "ALLOW_UNKNOWN"
            for key, value in obj_group_base.items():
                obj_group[key] = value if "unknown" in value else value + ["unknown"]
        elif "ALLOW_ANY" == matching_option:
            res["Option"] = "ALLOW_ANY"
            for key in obj_group_base.keys():
                obj_group[key] = [
                    "unknown",
                    "car",
                    "truck",
                    "bus",
                    "bicycle",
                    "motorbike",
                    "pedestrian",
                ]
        else:
            obj_group = obj_group_base

        for i in range(len(criteria_max_dist)):
            res["criteria" + str(i)]["MAX_DIST_THRESH"] = criteria_max_dist[i]
    return res, obj_group, criteria_max_dist


def create_bbox_movie(result_directory: str, folder_name: str, fps: int = 10):
    """
    Create a movie from bbox comparison images.

    Args:
        result_directory: Directory containing the bbox images
        fps: Frames per second for the output video (default: 10)
    """
    try:
        import cv2
        bbox_dir = os.path.join(result_directory, folder_name)
        if not os.path.exists(bbox_dir):
            print(f"Directory not found: {bbox_dir}")
            return

        # Get all jpg files and sort them
        image_files = sorted([f for f in os.listdir(bbox_dir) if f.endswith(".jpg")])
        if not image_files:
            print("No jpg files found in the directory")
            return

        # Read the first image to get dimensions
        first_image = cv2.imread(os.path.join(bbox_dir, image_files[0]))
        height, width, layers = first_image.shape

        # Define output video file
        output_path = os.path.join(result_directory, f"{folder_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Add each image to the video
        for image_file in image_files:
            image_path = os.path.join(bbox_dir, image_file)
            frame = cv2.imread(image_path)
            video.write(frame)

        # Release the video writer
        video.release()
        print(f"Video created successfully: {output_path}")

        # Try to create a backup in AVI format (more compatible)
        try:
            output_path_avi = os.path.join(result_directory, f"{folder_name}.avi")
            video_avi = cv2.VideoWriter(
                output_path_avi, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height)
            )

            for image_file in image_files:
                image_path = os.path.join(bbox_dir, image_file)
                frame = cv2.imread(image_path)
                video_avi.write(frame)

            video_avi.release()
            print(f"Backup video created successfully: {output_path_avi}")
        except Exception as e:
            print(f"Warning: Could not create AVI backup: {e}")

    except Exception as e:
        print(f"Error creating video: {e}")


def calc_score_group(df, result_directory):
    def get_frame_index(df, i):
        if not isnull(df.loc[(i, "ground_truth"), "frame"]):
            return df.loc[(i, "ground_truth"), "frame"]
        else:
            return df.loc[(i, "estimation"), "frame"]

    print("result_directory", result_directory)
    res, obj_group, criteria_max_dist = get_option_and_object_group(result_directory)
    total_row_num = int(df.shape[0] / 2)
    if total_row_num == 0:
        with open(result_directory + "score.json", "w") as file:
            file.write(json.dumps(res, indent=4))
        return [], [res["criteria" + str(i)]["MAX_DIST_THRESH"] for i in range(len(criteria_max_dist))]
    found_gt, pos = False, []
    (
        act_rect_list,
        act_mio_rect,
        act_pmio_rect,
        est_matching_obj_rect_list,
        est_ail_rect_list,
        est_uil_rect_list,
    ) = ([], [], [], [], [], [])
    mio_dist, pmio_intersection_dist, ttra, mio_area0 = 1000.0, 1000.0, 11.0, False
    point = {"x": 0.0, "y": 0.0, "dist": 0.0, "vx": 0.0, "vy": 0.0, "status": ""}
    num_gt, found_gt = 0, False

    for i in range(total_row_num):
        idx = get_frame_index(df, i)
        if not isnull(df.loc[(i, "ground_truth"), "timestamp"]):
            found_gt = True
            num_gt += 1
            act_label = df.loc[(i, "ground_truth"), "label"]
            act_x = df.loc[(i, "ground_truth"), "x"]
            act_y = df.loc[(i, "ground_truth"), "y"]
            act_dist = math.sqrt(act_x**2 + act_y**2)
            act_vx = df.loc[(i, "ground_truth"), "vx"]
            act_vy = df.loc[(i, "ground_truth"), "vy"]
            point["x"] = -act_y / num_gt + (point["x"] * (num_gt - 1) / num_gt)
            point["y"] = act_x / num_gt + (point["y"] * (num_gt - 1) / num_gt)
            point["dist"] = act_dist / num_gt + (point["dist"] * (num_gt - 1) / num_gt)
            point["vx"] = act_vx / num_gt + (point["vx"] * (num_gt - 1) / num_gt)
            point["vy"] = act_vy / num_gt + (point["vy"] * (num_gt - 1) / num_gt)
            act_rect = [
                [-act_y - hsize_of_ped, act_x - hsize_of_ped],
                [-act_y + hsize_of_ped, act_x - hsize_of_ped],
                [-act_y + hsize_of_ped, act_x + hsize_of_ped],
                [-act_y - hsize_of_ped, act_x + hsize_of_ped],
            ]
            act_rect_list.append(act_rect)
            if calculate_common_area([act_rect], [area0]) > 0.0:
                tmp_mio_dist = calc_line_polygon_distance(front_line, act_rect)
                if tmp_mio_dist < mio_dist:
                    mio_dist = tmp_mio_dist
                    act_mio_rect = act_rect
                    mio_area0 = True
                # print("rmio_dist", mio_dist, -act_vx, act_vy, -act_y, act_x )
            else:
                tmp_pmio_int_dist1, tmp_pmio_dist1, tmp_ttra1 = calc_line_vector_intersection_time(
                    area0_left_line, [-act_y, act_x], [-act_vx, act_vy], vehicle_polygon
                )
                tmp_pmio_int_dist2, tmp_pmio_dist2, tmp_ttra2 = calc_line_vector_intersection_time(
                    area0_right_line, [-act_y, act_x], [-act_vx, act_vy], vehicle_polygon
                )

                if tmp_ttra1 < tmp_ttra2:
                    tmp_pmio_intersection_dist = tmp_pmio_int_dist1
                    tmp_ttra = tmp_ttra1
                    tmp_pmio_dist = tmp_pmio_dist1
                else:
                    tmp_pmio_intersection_dist = tmp_pmio_int_dist2
                    tmp_ttra = tmp_ttra2
                    tmp_pmio_dist = tmp_pmio_dist2
                # print("tmp_ttra", tmp_ttra, "pmio_int_dist", tmp_pmio_intersection_dist, "pmio_dist", tmp_pmio_dist, -act_vx, act_vy, -act_y, act_x)

                if tmp_ttra < ttra:
                    ttra = tmp_ttra
                    if not mio_area0:
                        act_mio_rect = act_rect

                if tmp_ttra < 5.0:
                    if (
                        tmp_pmio_intersection_dist < pmio_intersection_dist
                        and tmp_pmio_intersection_dist < mio_dist
                    ):
                        pmio_intersection_dist = tmp_pmio_intersection_dist
                        act_pmio_rect = act_rect

                if tmp_pmio_intersection_dist == 1000.0:
                    if not mio_area0 and tmp_pmio_dist < mio_dist and ttra >= 10.0:
                        mio_dist = tmp_pmio_dist
                        act_mio_rect = act_rect

            if point["dist"] < criteria_max_dist[0]:
                key = "criteria0"
            elif point["dist"] < criteria_max_dist[1]:
                key = "criteria1"
            elif point["dist"] < criteria_max_dist[2]:
                key = "criteria2"
            elif point["dist"] < criteria_max_dist[3]:
                key = "criteria3"
            else:
                raise ValueError("act_dist is out of range", point)

            if not isnull(df.loc[(i, "estimation"), "timestamp"]):
                est_label = df.loc[(i, "estimation"), "label"]
                est_x = df.loc[(i, "estimation"), "x"]
                est_y = df.loc[(i, "estimation"), "y"]
                if est_label == act_label:
                    est_matching_obj_rect_list.append(
                        [
                            [-est_y - hsize_of_ped, est_x - hsize_of_ped],
                            [-est_y + hsize_of_ped, est_x - hsize_of_ped],
                            [-est_y + hsize_of_ped, est_x + hsize_of_ped],
                            [-est_y - hsize_of_ped, est_x + hsize_of_ped],
                        ]
                    )
                    est_ail_rect_list.append(
                        [
                            [-est_y - hsize_of_ped, est_x - hsize_of_ped],
                            [-est_y + hsize_of_ped, est_x - hsize_of_ped],
                            [-est_y + hsize_of_ped, est_x + hsize_of_ped],
                            [-est_y - hsize_of_ped, est_x + hsize_of_ped],
                        ]
                    )
                    est_uil_rect_list.append(
                        [
                            [-est_y - hsize_of_ped, est_x - hsize_of_ped],
                            [-est_y + hsize_of_ped, est_x - hsize_of_ped],
                            [-est_y + hsize_of_ped, est_x + hsize_of_ped],
                            [-est_y - hsize_of_ped, est_x + hsize_of_ped],
                        ]
                    )
                elif est_label in obj_group[act_label]:
                    est_ail_rect_list.append(
                        [
                            [-est_y - hsize_of_ped, est_x - hsize_of_ped],
                            [-est_y + hsize_of_ped, est_x - hsize_of_ped],
                            [-est_y + hsize_of_ped, est_x + hsize_of_ped],
                            [-est_y - hsize_of_ped, est_x + hsize_of_ped],
                        ]
                    )
                    est_uil_rect_list.append(
                        [
                            [-est_y - hsize_of_ped, est_x - hsize_of_ped],
                            [-est_y + hsize_of_ped, est_x - hsize_of_ped],
                            [-est_y + hsize_of_ped, est_x + hsize_of_ped],
                            [-est_y - hsize_of_ped, est_x + hsize_of_ped],
                        ]
                    )
                else:
                    est_uil_rect_list.append(
                        [
                            [-est_y - hsize_of_ped, est_x - hsize_of_ped],
                            [-est_y + hsize_of_ped, est_x - hsize_of_ped],
                            [-est_y + hsize_of_ped, est_x + hsize_of_ped],
                            [-est_y - hsize_of_ped, est_x + hsize_of_ped],
                        ]
                    )
                    # print(f"est_label: {est_label} {idx}")

        if i == 0:
            continue
        if (
            i == len(df.index) - 1 or get_frame_index(df, i) != get_frame_index(df, i - 1)
        ) and found_gt:
            # print("#Result#")
            # print("act_mio_rect", act_mio_rect)
            # print("act_pmio_rect", act_pmio_rect)
            sum_area = []
            if len(act_mio_rect) > 0:
                sum_area.append(act_mio_rect)
            if len(act_pmio_rect) > 0:
                sum_area.append(act_pmio_rect)
            # print("#Reset#", get_frame_index(df, i))
            if calculate_common_area([act_mio_rect], est_matching_obj_rect_list) > 0.0 and (
                len(act_pmio_rect) == 0
                or calculate_common_area([act_pmio_rect], est_matching_obj_rect_list) > 0.0
            ):
                status = "TP/TN"
            elif calculate_common_area([act_mio_rect], est_ail_rect_list) > 0.0 and (
                len(act_pmio_rect) == 0
                or calculate_common_area([act_pmio_rect], est_ail_rect_list) > 0.0
            ):
                status = "AIL"
            elif calculate_common_area([act_mio_rect], est_uil_rect_list) > 0.0 and (
                len(act_pmio_rect) == 0
                or calculate_common_area([act_pmio_rect], est_uil_rect_list) > 0.0
            ):
                status = "UIL"
            else:
                status = "PFN/PFP"
            plot_rectangles_with_common_area(
                act_rect_list,
                est_ail_rect_list,
                result_directory,
                idx,
                "bbox",
                f"act_obj_bbox_list vs est_obj_bbox_list{int(idx):04d}",
                vehicle_polygon,
            )
            plot_rectangles_with_common_area(
                sum_area,
                est_ail_rect_list,
                result_directory,
                idx,
                "mios_bbox",
                f"act_obj_bbox_list vs est_obj_bbox_list{int(idx):04d}",
                vehicle_polygon,
                area_polygon=area0,
            )
            (
                act_rect_list,
                act_mio_rect,
                act_pmio_rect,
                est_matching_obj_rect_list,
                est_ail_rect_list,
                est_uil_rect_list,
            ) = ([], [], [], [], [], [])
            mio_dist, pmio_intersection_dist, ttra, mio_area0 = 1000.0, 1000.0, 11.0, False
            point = {
                "x": 0.0,
                "y": 0.0,
                "dist": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "status": "",
                "uuid_num": 1,
            }
            num_gt, found_gt = 0, False
            res[key][status] += 1
            res[key]["NM"] += 1
            res[key]["UUID_NUM"] = 1
            point["status"] = status
            pos.append(point)

    with open(result_directory + "score.json", "w") as file:
        file.write(json.dumps(res, indent=4))

    return pos, [res["criteria" + str(i)]["MAX_DIST_THRESH"] for i in range(len(criteria_max_dist))]


def calc_score_single(df, result_directory):
    print("result_directory", result_directory)
    res, obj_group, criteria_max_dist = get_option_and_object_group(result_directory)
    total_row_num = int(df.shape[0] / 2)
    if total_row_num == 0:
        with open(result_directory + "score.json", "w") as file:
            file.write(json.dumps(res, indent=4))
        return [], [res["criteria" + str(i)]["MAX_DIST_THRESH"] for i in range(len(criteria_max_dist))]
    found_gt, pos, prev_frame, uuid_list, obj_idx = False, [], -1, [], 0

    frame_data = {}
    for i in range(total_row_num):
        frame_num = df.loc[(i, "estimation"), "frame"]
        if isnull(frame_num):
            frame_num = df.loc[(i, "ground_truth"), "frame"]

        if frame_num not in frame_data:
            frame_data[frame_num] = {"ground_truth": [], "estimation": []}

        if not isnull(df.loc[(i, "estimation"), "timestamp"]):
            frame_data[frame_num]["estimation"].append(
                {
                    "index": i,
                    "x": df.loc[(i, "estimation"), "x"],
                    "y": df.loc[(i, "estimation"), "y"],
                    "label": df.loc[(i, "estimation"), "label"],
                    "uuid": df.loc[(i, "estimation"), "uuid"],
                    "timestamp": df.loc[(i, "estimation"), "timestamp"],
                }
            )

        if not isnull(df.loc[(i, "ground_truth"), "timestamp"]):
            frame_data[frame_num]["ground_truth"].append(
                {
                    "index": i,
                    "x": df.loc[(i, "ground_truth"), "x"],
                    "y": df.loc[(i, "ground_truth"), "y"],
                    "label": df.loc[(i, "ground_truth"), "label"],
                    "vx": df.loc[(i, "ground_truth"), "vx"],
                    "vy": df.loc[(i, "ground_truth"), "vy"],
                    "frame": df.loc[(i, "ground_truth"), "frame"],
                }
            )

    for frame_num in sorted(frame_data.keys()):
        frame_gt_list = frame_data[frame_num]["ground_truth"]
        frame_est_list = frame_data[frame_num]["estimation"]

        for gt_obj in frame_gt_list:
            i = gt_obj["index"]
            act_uuid = df.loc[(i, "ground_truth"), "uuid"]
            est_uuid = ""
            prev_frame = df.loc[(i, "ground_truth"), "frame"]

            act_x = gt_obj["x"]
            act_y = gt_obj["y"]
            act_dist = math.sqrt(act_x**2 + act_y**2)
            act_vx = gt_obj["vx"]
            act_vy = gt_obj["vy"]
            point = {"x": -act_y, "y": act_x, "dist": act_dist, "vx": -act_vx, "vy": act_vy}

            if act_dist < criteria_max_dist[0]:
                key = "criteria0"
                dist_err_torelance = 2
            elif act_dist < criteria_max_dist[1]:
                key = "criteria1"
                dist_err_torelance = 3
            elif act_dist < criteria_max_dist[2]:
                key = "criteria2"
                dist_err_torelance = 5
            elif act_dist < criteria_max_dist[3]:
                key = "criteria3"
                dist_err_torelance = 5
            else:
                raise ValueError("act_dist is out of range")

            act_label = gt_obj["label"]
            if not found_gt:
                found_gt = True
                res["criteria0"]["GT_OBJ"] = act_label
                res["criteria1"]["GT_OBJ"] = act_label
                res["criteria2"]["GT_OBJ"] = act_label

            if not isnull(df.loc[(i, "estimation"), "timestamp"]):
                est_label = df.loc[(i, "estimation"), "label"]
                est_uuid = df.loc[(i, "estimation"), "uuid"]
                if act_label != "false_positive":
                    est_x = df.loc[(i, "estimation"), "x"]
                    est_y = df.loc[(i, "estimation"), "y"]
                    diff_dist = math.sqrt((act_x - est_x) ** 2 + (act_y - est_y) ** 2)

                    if est_uuid not in uuid_list:
                        uuid_list.append(est_uuid)

                    if act_label == est_label:
                        if diff_dist < dist_err_torelance:
                            status = "TP/TN"
                        else:
                            status = "ADD"
                    elif est_label in obj_group[act_label]:
                        status = "AIL"
                    else:
                        status = "UIL"
                else:
                    status = "PFN/PFP"
                res[key]["OBJ_CNTS"].setdefault(est_label, 0)
                res[key]["OBJ_CNTS"][est_label] += 1
            else:
                if act_label != "false_positive":
                    closest_dist = float("inf")
                    closest_est = None

                    for est_obj in frame_est_list:
                        diff_dist = math.sqrt((act_x - est_obj["x"]) ** 2 + (act_y - est_obj["y"]) ** 2)
                        if diff_dist < closest_dist:
                            closest_dist = diff_dist
                            closest_est = est_obj

                    if closest_est is not None and closest_dist < 1.0:
                        est_label = closest_est["label"]
                        est_uuid = closest_est["uuid"]

                        if est_uuid is not None and est_uuid not in uuid_list:
                            uuid_list.append(est_uuid)

                        if act_label == est_label:
                            if closest_dist < dist_err_torelance:
                                status = "TP/TN"
                            else:
                                status = "ADD"
                        elif est_label in obj_group[act_label]:
                            status = "AIL"
                        else:
                            status = "UIL"

                        res[key]["OBJ_CNTS"].setdefault(est_label, 0)
                        res[key]["OBJ_CNTS"][est_label] += 1
                    else:
                        status = "PFN/PFP"
                else:
                    status = "TP/TN"

            res[key][status] += 1
            res[key]["NM"] += 1
            res[key]["UUID_NUM"] = len(uuid_list)
            point["status"] = status
            point["act_uuid"] = act_uuid
            point["est_uuid"] = est_uuid
            pos.append(point)

    with open(result_directory + "score.json", "w") as file:
        file.write(json.dumps(res, indent=4))

    return pos, [res["criteria" + str(i)]["MAX_DIST_THRESH"] for i in range(len(criteria_max_dist))]


def plot_coordinates(positions, result_directory, dist_thresholds):
    import matplotlib.pyplot as plt
    import numpy as np
    x = []
    y = []
    vx = []
    vy = []
    status = []
    uuid_num = []
    uuid_switch = []

    for obj_idx, pos in enumerate(positions):
        x.append([p["x"] for p in pos])
        y.append([p["y"] for p in pos])
        vx.append([p["vx"] for p in pos])
        vy.append([p["vy"] for p in pos])
        status.append([p["status"] for p in pos])
        uuid_num.append([p["uuid_num"] for p in pos])
        uuid_switch.append(
            [False] + [uuid_num[obj_idx][i - 1] != uuid_num[obj_idx][i] for i in range(1, len(pos))]
        )

    color_map = {
        "TP/TN": "green",
        "ADD": "green",
        "AIL": "yellowgreen",
        "UIL": "orange",
        "PFN/PFP": "red",
    }
    legend_map = {
        "TP/TN": "TP/TN(Perfect true)",
        "ADD": "ADD(Acceptable dist diff)",
        "AIL": "AIL(Acceptable label error)",
        "UIL": "UIL(Unacceptable label error)",
        "PFN/PFP": "PFN/PFP(Perfect false)",
    }

    scatter_circle = []
    scatter_star = []

    # Create figure once before the loop
    plt.figure(figsize=(10, 10))

    # Add circles for distance thresholds (only once)
    for i, radius in enumerate(dist_thresholds):
        circle = plt.Circle((0, 0), radius, fill=False, linestyle="--", color="gray", alpha=0.5)
        plt.gca().add_patch(circle)
        plt.text(
            radius / np.sqrt(2),
            radius / np.sqrt(2),
            f"criteria{i}\n{radius}m",
            color="gray",
            alpha=0.7,
            backgroundcolor="white",
            ha="left",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )

    for obj_idx, pos in enumerate(positions):
        colors = [color_map.get(s, "gray") for s in status[obj_idx]]

        # Create separate arrays for different markers
        x_circle = [x[obj_idx][i] for i in range(len(x[obj_idx])) if not uuid_switch[obj_idx][i]]
        y_circle = [y[obj_idx][i] for i in range(len(y[obj_idx])) if not uuid_switch[obj_idx][i]]
        colors_circle = [colors[i] for i in range(len(colors)) if not uuid_switch[obj_idx][i]]

        x_star = [x[obj_idx][i] for i in range(len(x[obj_idx])) if uuid_switch[obj_idx][i]]
        y_star = [y[obj_idx][i] for i in range(len(y[obj_idx])) if uuid_switch[obj_idx][i]]
        colors_star = [colors[i] for i in range(len(colors)) if uuid_switch[obj_idx][i]]

        # Plot points with different markers
        scatter_circle.append(
            plt.scatter(x_circle, y_circle, c=colors_circle, marker="o", alpha=0.7)
        )
        scatter_star.append(
            plt.scatter(x_star, y_star, c=colors_star, marker="*", s=200, alpha=0.7)
        )
        # s=200 makes stars bigger

        counter = 0
        for xi, yi in zip(x[obj_idx], y[obj_idx]):
            if counter == 0 or counter % 4 != 0:
                px, py = xi, yi
                counter += 1
                continue

            plt.annotate(
                "",
                xy=(xi, yi),
                xytext=(px, py),
                arrowprops=dict(arrowstyle="->", color="blue", lw=0.5),
            )
            px, py = xi, yi
            counter += 1

    # Set axis properties after all objects are plotted
    plt.axis("equal")
    plt.xlabel("Relative Lateral Distance [m]")
    plt.ylabel("Relative Longitudinal Distance [m]")
    plt.title("Object Positions and Status")
    plt.grid(True)

    # Create legend with both marker types
    legend_elements = []
    for status, color in color_map.items():
        # Add color marker
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"{legend_map[status]}",
                markerfacecolor=color,
                markersize=10,
            )
        )
    # Add star marker
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            label="UUID Switch",
            markerfacecolor="gray",
            markersize=15,
        )
    )
    # Add circle marker
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"No UUID Switch",
            markerfacecolor="gray",
            markersize=10,
        )
    )

    plt.legend(handles=legend_elements, title="Status", loc="upper right")
    plt.savefig(f"{result_directory}object_positions_status.jpg", dpi=300)
    plt.axhline(y=0, color="black", linewidth=1.5)
    plt.axvline(x=0, color="black", linewidth=1.5)
    plt.axis("equal")

    # Flatten all x and y coordinates to get proper min/max values
    all_x = [coord for obj_x in x for coord in obj_x]
    all_y = [coord for obj_y in y for coord in obj_y]

    # Only set limits if we have data points
    if all_x and all_y:
        x_margin = (max(all_x) - min(all_x)) * 0.1  # 10% margin
        y_margin = (max(all_y) - min(all_y)) * 0.1  # 10% margin

        plt.xlim(min(all_x) - x_margin, max(all_x) + x_margin)
        plt.ylim(min(all_y) - y_margin, max(all_y) + y_margin)
    else:
        # Fallback to a reasonable default range if no data
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)

    plt.savefig(f"{result_directory}object_positions_status_equal_scale.jpg", dpi=300)
    plt.close()


def _format_summary_line(sum_rat: pd.DataFrame, sum_err: pd.DataFrame, label: str) -> str:
    try:
        return " ".join(
            [
                str(sum_rat.loc[label, "TP"]),
                str(sum_err.loc[(label, "x"), "average"]),
                str(sum_err.loc[(label, "x"), "std"]),
                str(sum_err.loc[(label, "x"), "rms"]),
                str(sum_err.loc[(label, "y"), "average"]),
                str(sum_err.loc[(label, "y"), "std"]),
                str(sum_err.loc[(label, "y"), "rms"]),
                str(sum_err.loc[(label, "vx"), "average"]),
                str(sum_err.loc[(label, "vy"), "average"]),
            ]
        )
    except KeyError:
        return "N/A"


def summarize_eval_result(result_root: str) -> dict:
    if PerceptionAnalyzer3D is None:
        raise ImportError(
            "PerceptionAnalyzer3D is unavailable. "
            "Install/sourcing perception_eval or use calc_score_* to generate score.json."
        ) from _perception_eval_import_error
    result_directory = os.path.join(result_root, "")
    scenario_paths = sorted(glob.glob(os.path.join(result_directory, "*.yaml")))
    if not scenario_paths:
        raise FileNotFoundError(f"No scenario yaml found in {result_directory}")

    scenario_name = scenario_paths[0]
    pickle_path = os.path.join(result_directory, "scene_result.pkl")
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Missing scene_result.pkl in {result_directory}")

    test_id = os.path.basename(os.path.normpath(result_root))

    analyzer = PerceptionAnalyzer3D.from_scenario(result_directory, scenario_name)
    analyzer.add_from_pkl(pickle_path)

    df = analyzer.df
    if df.empty or not isinstance(df.index, pd.MultiIndex):
        return {
            "result_root": result_root,
            "test_id": test_id,
            "scenario_name": scenario_name,
            "num_ground_truth": 0,
            "num_ground_truth_tp": 0,
            "summary_ratio": pd.DataFrame(),
            "summary_error": pd.DataFrame(),
            "frame_table": df,
        }

    sum_rat = analyzer.summarize_ratio()
    sum_err = analyzer.summarize_error()

    summary = {
        "result_root": result_root,
        "test_id": test_id,
        "scenario_name": scenario_name,
        "num_ground_truth": analyzer.get_num_ground_truth(),
        "num_ground_truth_tp": analyzer.get_num_ground_truth(status="TP"),
        "summary_ratio": sum_rat,
        "summary_error": sum_err,
        "frame_table": analyzer.get(
            "timestamp",
            "x",
            "y",
            "yaw",
            "vx",
            "vy",
            "length",
            "width",
            "label",
            "uuid",
            "num_points",
            "status",
            "frame",
        ),
    }
    return summary


def generate_score_json(result_root: str) -> str:
    """
    Generate score.json for a result directory without any plotting.

    Returns:
        str: path to the generated score.json
    """
    if PerceptionAnalyzer3D is None:
        raise ImportError(
            "PerceptionAnalyzer3D is unavailable. "
            "Install/sourcing perception_eval or use calc_score_* to generate score.json."
        ) from _perception_eval_import_error
    result_directory = os.path.join(result_root, "")
    scenario_paths = sorted(glob.glob(os.path.join(result_directory, "*.yaml")))
    if not scenario_paths:
        raise FileNotFoundError(f"No scenario yaml found in {result_directory}")

    scenario_name = scenario_paths[0]
    pickle_path = os.path.join(result_directory, "scene_result.pkl")
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Missing scene_result.pkl in {result_directory}")

    test_id = os.path.basename(os.path.normpath(result_root))

    analyzer = PerceptionAnalyzer3D.from_scenario(result_directory, scenario_name)
    analyzer.add_from_pkl(pickle_path)

    df = analyzer.df
    if df.empty or not isinstance(df.index, pd.MultiIndex):
        res, _, _ = get_option_and_object_group(result_directory)
        with open(result_directory + "score.json", "w") as file:
            file.write(json.dumps(res, indent=4))
        return os.path.join(result_directory, "score.json")

    # Score generation only; skip all plots.
    if "pedestrians_with_umbrella" in test_id:
        calc_score_group(df, result_directory)
    else:
        calc_score_single(df, result_directory)

    return os.path.join(result_directory, "score.json")


def format_eval_report(summary: dict) -> str:
    lines = [
        f"result_root: {summary['result_root']}",
        f"scenario_name: {summary['scenario_name']}",
        f"num_ground_truth: {summary['num_ground_truth']}",
        f"num_ground_truth_TP: {summary['num_ground_truth_tp']}",
        "",
        "TP xave xstd xrms yave ystd yrms vx vy",
        _format_summary_line(summary["summary_ratio"], summary["summary_error"], "ALL"),
        "",
    ]

    if "MultiTarget1" in summary["test_id"]:
        for label in ["car", "bicycle"]:
            lines.append(f"{label} TP xave xstd xrms yave ystd yrms vx vy")
            lines.append(_format_summary_line(summary["summary_ratio"], summary["summary_error"], label))
            lines.append("")

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 256):
        lines.append("summarize_ratio")
        lines.append(summary["summary_ratio"].to_string())
        lines.append("")
        lines.append("summarize_error")
        lines.append(summary["summary_error"].to_string())
        lines.append("")
        lines.append(summary["frame_table"].to_string())

    return "\n".join(lines).rstrip() + "\n"


def run_eval_result(result_root: str) -> str:
    summary = summarize_eval_result(result_root)
    return format_eval_report(summary)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: perception_eval_result_summarizer.py <result_root>")
    print(run_eval_result(sys.argv[1]))
