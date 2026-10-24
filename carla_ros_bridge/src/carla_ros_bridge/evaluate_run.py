#!/usr/bin/env python
#
# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
"""
Offline evaluation of a recorded run produced by `record_evaluation`: computes
trajectory error (ATE/RPE via evo) and cone-map accuracy of the SLAM output
against CARLA ground truth, both for the whole run and per lap (using the
`<bag>_lap_crossings.csv` boundaries written by lap_monitor).

The whole-run trajectory alignment (rotation + translation, no scale --
RGB-D depth is already metric) is computed once via evo's Umeyama alignment,
then reused to both slice per-lap trajectory error and to transform the SLAM
cone map into the ground-truth frame for map-accuracy scoring. This avoids
re-aligning per lap, which would hide/mask drift instead of measuring it.

By default, results are written under ./evaluation_results/<bag_name>_eval
(relative to cwd) rather than next to the bag -- bags typically live under
/tmp and don't survive a container restart, but evaluation_results/ is meant
to be run from the repo checkout (e.g. perception-ws-sw/evaluation_results)
so results persist. Pass --out to override.

Usage:
    ros2 run carla_ros_bridge evaluate_run --bag /tmp/bags/eval_full_stack_01
"""

import argparse
import copy
import csv
import os
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D

GT_TOPIC = "/carla/autopilot/ground_truth_odom"
GT_CONE_TOPIC = "/carla/cone_ground_truth"
SLAM_POSE_TOPIC = "/slam/pose"
SLAM_CONE_TOPIC = "/slam/cones"

GT_COLOR_NAME = {0: "unknown", 1: "blue", 2: "yellow", 3: "orange", 4: "red"}
# matches kColors in orbslam_node.cpp::publishCircuit (type 0/1/2 -> RGBA)
SLAM_COLOR_REF = {"blue": (0.0, 0.0, 1.0), "yellow": (1.0, 1.0, 0.0), "orange": (1.0, 0.65, 0.0)}

# cone_spacing in track.yaml is 2.5m; half that plus margin gives a sane
# nearest-neighbor gate that won't cross-match adjacent cones.
CONE_MATCH_MAX_DIST = 1.5

# Persistent default: relative to cwd, meant to be run from the repo checkout
# so results outlive the (usually /tmp, non-persistent) bag they came from.
DEFAULT_RESULTS_ROOT = "evaluation_results"

RPE_DELTA_FRAMES = 1.0


def _find_carla_msgs_dir():
    """Locate carla_msgs/msg/*.msg to teach rosbags about the custom cone message."""
    here = os.path.dirname(os.path.abspath(__file__))
    # .../carla_ros_bridge/src/carla_ros_bridge/evaluate_run.py -> up to the
    # carla-ros-bridge checkout root, then carla_msgs/msg
    candidate = os.path.normpath(os.path.join(here, "..", "..", "..", "carla_msgs", "msg"))
    if os.path.isdir(candidate):
        return candidate
    # fall back to searching the ROS_PACKAGE install share dir
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(":"):
        candidate = os.path.join(prefix, "share", "carla_msgs", "msg")
        if os.path.isdir(candidate):
            return candidate
    return None


def build_typestore():
    ts = get_typestore(Stores.ROS2_HUMBLE)
    msg_dir = _find_carla_msgs_dir()
    if msg_dir is None:
        raise RuntimeError("Could not locate carla_msgs/msg/*.msg to register custom types")
    types = {}
    for fname, typename in (
        ("CarlaGroundTruthCone.msg", "carla_msgs/msg/CarlaGroundTruthCone"),
        ("CarlaGroundTruthConeArray.msg", "carla_msgs/msg/CarlaGroundTruthConeArray"),
    ):
        text = open(os.path.join(msg_dir, fname)).read()
        types.update(get_types_from_msg(text, typename))
    ts.register(types)
    return ts


def _msg_stamp(topic, msg):
    # MarkerArray has no header of its own -- every Marker inside carries the
    # same stamp (set from the single header passed into publishCircuit()).
    if topic == SLAM_CONE_TOPIC:
        stamp = msg.markers[0].header.stamp
    else:
        stamp = msg.header.stamp
    return stamp.sec + stamp.nanosec * 1e-9


def read_bag(bag_path, typestore):
    """Returns (data, wall_clock_range) where data is topic -> list of
    (sim_stamp_sec_float, msg) sorted by stamp. Header stamps are CARLA
    simulation time (matches lap_crossings.csv), which is a different clock
    domain than the bag's own recv-time and the orbslam PerformanceLogger's
    wall-clock System_Timestamp_us -- wall_clock_range (epoch seconds, from
    the bag's own recv timestamps) is what the performance CSV must be
    filtered against instead of the sim-time run_start/run_end."""
    topics = {GT_TOPIC, GT_CONE_TOPIC, SLAM_POSE_TOPIC, SLAM_CONE_TOPIC}
    data = {t: [] for t in topics}
    wall_min, wall_max = float("inf"), float("-inf")
    with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:
        connections = [c for c in reader.connections if c.topic in topics]
        for connection, bag_ts_ns, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            if connection.topic == SLAM_CONE_TOPIC and not msg.markers:
                continue
            data[connection.topic].append((_msg_stamp(connection.topic, msg), msg))
            wall_s = bag_ts_ns * 1e-9
            wall_min, wall_max = min(wall_min, wall_s), max(wall_max, wall_s)
    for t in data:
        data[t].sort(key=lambda pair: pair[0])
    return data, (wall_min, wall_max)


def load_lap_boundaries(bag_path):
    """Returns sorted list of (lap_num, stamp_sec_float) from the sibling lap_crossings.csv."""
    csv_path = bag_path.rstrip("/") + "_lap_crossings.csv"
    boundaries = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            stamp = float(row["stamp_sec"]) + float(row["stamp_nanosec"]) * 1e-9
            boundaries.append((int(row["lap"]), stamp))
    boundaries.sort(key=lambda p: p[1])
    return boundaries


def _pose_of(msg):
    """Odometry nests a PoseWithCovariance (msg.pose.pose), PoseStamped nests a
    Pose directly (msg.pose) -- normalize both to the inner Pose."""
    pose = msg.pose
    return pose.pose if hasattr(pose, "pose") else pose


def to_pose_trajectory(entries):
    """entries: list of (t, msg) where msg is an Odometry or PoseStamped."""
    t = np.array([e[0] for e in entries])
    poses = [_pose_of(e[1]) for e in entries]
    xyz = np.array([[p.position.x, p.position.y, p.position.z] for p in poses])
    # evo wants wxyz
    quat = np.array([[p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]
                      for p in poses])
    return PoseTrajectory3D(positions_xyz=xyz, orientations_quat_wxyz=quat, timestamps=t)


def compute_trajectory_errors(gt_entries, slam_entries, max_diff=0.05):
    """Returns (traj_gt_sync, traj_slam_aligned, ape_metric, rpe_metric, r, t_vec, s)."""
    traj_gt = to_pose_trajectory(gt_entries)
    traj_slam = to_pose_trajectory(slam_entries)

    traj_gt_sync, traj_slam_sync = sync.associate_trajectories(traj_gt, traj_slam, max_diff=max_diff)

    traj_slam_aligned = copy.deepcopy(traj_slam_sync)
    r, t_vec, s = traj_slam_aligned.align(traj_gt_sync, correct_scale=False)

    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data((traj_gt_sync, traj_slam_aligned))

    rpe_metric = metrics.RPE(metrics.PoseRelation.translation_part,
                              delta=RPE_DELTA_FRAMES, delta_unit=metrics.Unit.frames)
    rpe_metric.process_data((traj_gt_sync, traj_slam_aligned))

    return traj_gt_sync, traj_slam_aligned, ape_metric, rpe_metric, r, t_vec, s


def _segment_stats(values):
    if len(values) == 0:
        return dict(count=0, rmse=float("nan"), mean=float("nan"),
                     std=float("nan"), median=float("nan"), max=float("nan"))
    values = np.asarray(values)
    return dict(
        count=len(values),
        rmse=float(np.sqrt(np.mean(values ** 2))),
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        median=float(np.median(values)),
        max=float(np.max(values)),
    )


def build_ate_rpe_table(traj_gt_sync, ape_metric, rpe_metric, lap_boundaries, run_start, run_end):
    """Slices the already-computed, already-aligned APE/RPE error arrays by lap
    boundary timestamps -- no re-alignment per lap, only re-slicing of residuals."""
    ape_t = traj_gt_sync.timestamps
    rpe_t = traj_gt_sync.timestamps[rpe_metric.delta_ids]

    edges = [run_start] + [b[1] for b in lap_boundaries] + [run_end]
    rows = []
    for lap_idx in range(len(edges) - 1):
        t0, t1 = edges[lap_idx], edges[lap_idx + 1]
        if t1 <= t0:
            # recording stopped exactly at the last lap crossing -> zero-width
            # trailing segment, not a real partial lap
            continue
        label = f"lap_{lap_idx + 1}" if lap_idx < len(lap_boundaries) else "lap_final_partial"
        ape_mask = (ape_t >= t0) & (ape_t < t1)
        rpe_mask = (rpe_t >= t0) & (rpe_t < t1)
        ape_s = _segment_stats(ape_metric.error[ape_mask])
        rpe_s = _segment_stats(rpe_metric.error[rpe_mask])
        rows.append(dict(segment=label, t_start=t0, t_end=t1,
                          ape_rmse=ape_s["rmse"], ape_mean=ape_s["mean"], ape_std=ape_s["std"],
                          ape_median=ape_s["median"], ape_max=ape_s["max"], ape_n=ape_s["count"],
                          rpe_rmse=rpe_s["rmse"], rpe_mean=rpe_s["mean"], rpe_std=rpe_s["std"],
                          rpe_median=rpe_s["median"], rpe_max=rpe_s["max"], rpe_n=rpe_s["count"]))

    ape_s = _segment_stats(ape_metric.error)
    rpe_s = _segment_stats(rpe_metric.error)
    rows.append(dict(segment="whole_run", t_start=run_start, t_end=run_end,
                      ape_rmse=ape_s["rmse"], ape_mean=ape_s["mean"], ape_std=ape_s["std"],
                      ape_median=ape_s["median"], ape_max=ape_s["max"], ape_n=ape_s["count"],
                      rpe_rmse=rpe_s["rmse"], rpe_mean=rpe_s["mean"], rpe_std=rpe_s["std"],
                      rpe_median=rpe_s["median"], rpe_max=rpe_s["max"], rpe_n=rpe_s["count"]))
    return rows


def _slam_cone_color_name(marker_color):
    best_name, best_dist = "unknown", float("inf")
    for name, ref in SLAM_COLOR_REF.items():
        d = (marker_color.r - ref[0]) ** 2 + (marker_color.g - ref[1]) ** 2 + (marker_color.b - ref[2]) ** 2
        if d < best_dist:
            best_dist, best_name = d, name
    return best_name


def latest_before(entries, t_cutoff):
    """Last entry with stamp <= t_cutoff, or the very last entry if none qualify."""
    chosen = None
    for stamp, msg in entries:
        if stamp <= t_cutoff:
            chosen = msg
        else:
            break
    return chosen if chosen is not None else (entries[-1][1] if entries else None)


def transform_points(xyz, r, t_vec, s):
    return (s * (r @ xyz.T)).T + t_vec


def match_cones(gt_xyz, gt_colors, slam_xyz_transformed, slam_colors, max_dist=CONE_MATCH_MAX_DIST):
    """Greedy nearest-neighbor 1:1 matching in the ground-truth frame (XY only,
    cones are on the ground so Z is not discriminative). Returns a dict of stats
    plus the matched/unmatched index lists for plotting."""
    if len(gt_xyz) == 0 or len(slam_xyz_transformed) == 0:
        return dict(n_gt=len(gt_xyz), n_detected=len(slam_xyz_transformed), n_matched=0,
                     precision=float("nan"), recall=float("nan"),
                     pos_err_mean=float("nan"), pos_err_std=float("nan"),
                     color_accuracy=float("nan")), [], list(range(len(slam_xyz_transformed))), list(range(len(gt_xyz)))

    tree = cKDTree(gt_xyz[:, :2])
    dists, idxs = tree.query(slam_xyz_transformed[:, :2], k=1)

    # sort slam detections by distance to their candidate match so the closest
    # pairs claim their ground-truth cone first (greedy 1:1 assignment)
    order = np.argsort(dists)
    gt_taken = set()
    matches = []  # (slam_i, gt_i, dist)
    for slam_i in order:
        gt_i = int(idxs[slam_i])
        if dists[slam_i] > max_dist or gt_i in gt_taken:
            continue
        gt_taken.add(gt_i)
        matches.append((int(slam_i), gt_i, float(dists[slam_i])))

    matched_slam_idx = {m[0] for m in matches}
    matched_gt_idx = {m[1] for m in matches}
    false_positives = [i for i in range(len(slam_xyz_transformed)) if i not in matched_slam_idx]
    false_negatives = [i for i in range(len(gt_xyz)) if i not in matched_gt_idx]

    pos_errs = np.array([m[2] for m in matches])
    color_hits = sum(1 for slam_i, gt_i, _ in matches if slam_colors[slam_i] == gt_colors[gt_i])

    stats = dict(
        n_gt=len(gt_xyz), n_detected=len(slam_xyz_transformed), n_matched=len(matches),
        precision=len(matches) / len(slam_xyz_transformed) if slam_xyz_transformed.size else float("nan"),
        recall=len(matches) / len(gt_xyz) if gt_xyz.size else float("nan"),
        pos_err_mean=float(np.mean(pos_errs)) if len(pos_errs) else float("nan"),
        pos_err_std=float(np.std(pos_errs)) if len(pos_errs) else float("nan"),
        color_accuracy=color_hits / len(matches) if matches else float("nan"),
    )
    return stats, matches, false_positives, false_negatives


def build_cone_map_table(gt_cone_entries, slam_cone_entries, r, t_vec, s,
                          lap_boundaries, run_end):
    gt_msg = gt_cone_entries[-1][1] if gt_cone_entries else None
    gt_xyz = np.array([[c.x, c.y, c.z] for c in gt_msg.cones]) if gt_msg else np.zeros((0, 3))
    gt_colors = [GT_COLOR_NAME.get(c.color, "unknown") for c in gt_msg.cones] if gt_msg else []

    checkpoints = [(f"after_lap_{lap}", t) for lap, t in lap_boundaries]
    checkpoints.append(("final", run_end))

    rows, snapshots = [], {}
    for label, t_cutoff in checkpoints:
        slam_msg = latest_before(slam_cone_entries, t_cutoff)
        if slam_msg is None or not slam_msg.markers:
            slam_xyz = np.zeros((0, 3))
            slam_colors = []
        else:
            slam_xyz = np.array([[m.pose.position.x, m.pose.position.y, m.pose.position.z]
                                  for m in slam_msg.markers])
            slam_colors = [_slam_cone_color_name(m.color) for m in slam_msg.markers]

        slam_xyz_t = transform_points(slam_xyz, r, t_vec, s) if len(slam_xyz) else slam_xyz
        stats, matches, fps, fns = match_cones(gt_xyz, gt_colors, slam_xyz_t, slam_colors)
        stats["segment"] = label
        rows.append(stats)
        snapshots[label] = dict(gt_xyz=gt_xyz, gt_colors=gt_colors, slam_xyz=slam_xyz_t,
                                 slam_colors=slam_colors, matches=matches, fps=fps, fns=fns)
    return rows, snapshots


def load_performance_csv(perf_csv_path, run_start_epoch, run_end_epoch):
    if not perf_csv_path or not os.path.isfile(perf_csv_path):
        return None
    proc_times, states = [], []
    with open(perf_csv_path) as f:
        for row in csv.DictReader(f):
            t = float(row["System_Timestamp_us"]) * 1e-6
            if not (run_start_epoch <= t <= run_end_epoch):
                continue
            proc_times.append(float(row["Processing_Time_ms"]))
            states.append(row["Tracking_State"])
    if not proc_times:
        return None
    proc_times = np.array(proc_times)
    n = len(states)
    state_counts = {}
    for s in states:
        state_counts[s] = state_counts.get(s, 0) + 1
    return dict(
        n_frames=n,
        proc_time_mean_ms=float(np.mean(proc_times)),
        proc_time_std_ms=float(np.std(proc_times)),
        proc_time_p95_ms=float(np.percentile(proc_times, 95)),
        proc_time_max_ms=float(np.max(proc_times)),
        effective_fps=1000.0 / float(np.mean(proc_times)),
        tracking_ok_pct=100.0 * state_counts.get("OK", 0) / n,
        state_counts=state_counts,
    )


def plot_trajectories(traj_gt_sync, traj_slam_aligned, lap_boundaries, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(traj_gt_sync.positions_xyz[:, 0], traj_gt_sync.positions_xyz[:, 1],
            color="black", linewidth=2, label="ground truth")
    ax.plot(traj_slam_aligned.positions_xyz[:, 0], traj_slam_aligned.positions_xyz[:, 1],
            color="tab:red", linewidth=1, linestyle="--", label="SLAM (aligned)")
    for lap, t in lap_boundaries:
        idx = int(np.argmin(np.abs(traj_gt_sync.timestamps - t)))
        xy = traj_gt_sync.positions_xyz[idx, :2]
        ax.scatter(*xy, color="tab:blue", zorder=5)
        # all laps cross near the same start/finish point, so stack labels
        # vertically instead of letting them overlap
        ax.annotate(f"lap {lap}", xy, xytext=(8, 8 + 14 * (lap - 1)),
                    textcoords="offset points", fontsize=8,
                    arrowprops=dict(arrowstyle="-", color="tab:blue", lw=0.5))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Ground truth vs. SLAM trajectory (aligned)")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cone_map(final_snapshot, out_path):
    fig, ax = plt.subplots(figsize=(9, 9))
    gt_xyz, gt_colors = final_snapshot["gt_xyz"], final_snapshot["gt_colors"]
    slam_xyz, slam_colors = final_snapshot["slam_xyz"], final_snapshot["slam_colors"]
    color_map = {"blue": "tab:blue", "yellow": "gold", "orange": "tab:orange",
                 "red": "tab:red", "unknown": "gray"}

    for name in set(gt_colors) or []:
        mask = [c == name for c in gt_colors]
        ax.scatter(gt_xyz[mask, 0], gt_xyz[mask, 1], marker="o", s=60,
                   facecolors="none", edgecolors=color_map.get(name, "gray"),
                   label=f"GT {name}")

    matched_slam = {m[0] for m in final_snapshot["matches"]}
    if len(slam_xyz):
        matched_mask = np.array([i in matched_slam for i in range(len(slam_xyz))])
        colors_arr = np.array([color_map.get(c, "gray") for c in slam_colors])
        if matched_mask.any():
            ax.scatter(slam_xyz[matched_mask, 0], slam_xyz[matched_mask, 1], marker="x", s=50,
                       c=colors_arr[matched_mask], label="SLAM matched")
        if (~matched_mask).any():
            ax.scatter(slam_xyz[~matched_mask, 0], slam_xyz[~matched_mask, 1], marker="x", s=80,
                       c="black", label="SLAM false positive")

    fns = final_snapshot["fns"]
    if fns:
        ax.scatter(gt_xyz[fns, 0], gt_xyz[fns, 1], marker="s", s=100,
                   facecolors="none", edgecolors="red", linewidths=2, label="GT missed")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Final cone map: ground truth vs. SLAM (transformed to GT frame)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_csv(rows, path, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True, help="Path to the bag directory (rosbag2)")
    parser.add_argument("--perf-csv", default=None,
                        help="Path to orbslam PerformanceLogger CSV (defaults to "
                             "orbslam-logs/slam_performance_metrics.csv next to cwd)")
    parser.add_argument("--out", default=None,
                        help=f"Output directory (default: ./{DEFAULT_RESULTS_ROOT}/<bag_name>_eval)")
    parser.add_argument("--max-time-diff", type=float, default=0.05,
                        help="Max seconds between GT/SLAM samples to associate them")
    parsed, _ros_args = parser.parse_known_args(args=sys.argv[1:] if args is None else args)

    bag_path = parsed.bag.rstrip("/")
    out_dir = parsed.out or os.path.join(DEFAULT_RESULTS_ROOT, os.path.basename(bag_path) + "_eval")
    os.makedirs(out_dir, exist_ok=True)
    perf_csv = parsed.perf_csv or "orbslam-logs/slam_performance_metrics.csv"

    lap_csv_src = bag_path + "_lap_crossings.csv"
    if os.path.isfile(lap_csv_src):
        shutil.copy(lap_csv_src, os.path.join(out_dir, "lap_crossings.csv"))

    print(f"Reading bag '{bag_path}' ...")
    typestore = build_typestore()
    data, wall_clock_range = read_bag(bag_path, typestore)
    for topic, entries in data.items():
        print(f"  {topic}: {len(entries)} messages")
        if not entries:
            print(f"ERROR: topic '{topic}' has no messages in this bag.", file=sys.stderr)
            sys.exit(1)

    lap_boundaries = load_lap_boundaries(bag_path)
    run_start = min(data[GT_TOPIC][0][0], data[SLAM_POSE_TOPIC][0][0])
    run_end = max(data[GT_TOPIC][-1][0], data[SLAM_POSE_TOPIC][-1][0])

    print("Computing trajectory alignment + APE/RPE (whole run, once) ...")
    traj_gt_sync, traj_slam_aligned, ape_metric, rpe_metric, r, t_vec, s = compute_trajectory_errors(
        data[GT_TOPIC], data[SLAM_POSE_TOPIC], max_diff=parsed.max_time_diff)
    print(f"  associated poses: {traj_gt_sync.num_poses}, alignment scale: {s:.4f} (expect ~1.0, no correction applied)")

    ate_rpe_rows = build_ate_rpe_table(traj_gt_sync, ape_metric, rpe_metric, lap_boundaries, run_start, run_end)
    write_csv(ate_rpe_rows, os.path.join(out_dir, "ate_rpe_table.csv"),
              fieldnames=["segment", "t_start", "t_end", "ape_rmse", "ape_mean", "ape_std",
                          "ape_median", "ape_max", "ape_n", "rpe_rmse", "rpe_mean", "rpe_std",
                          "rpe_median", "rpe_max", "rpe_n"])

    print("Computing cone-map accuracy per lap boundary ...")
    cone_rows, cone_snapshots = build_cone_map_table(
        data[GT_CONE_TOPIC], data[SLAM_CONE_TOPIC], r, t_vec, s, lap_boundaries, run_end)
    write_csv(cone_rows, os.path.join(out_dir, "cone_map_table.csv"),
              fieldnames=["segment", "n_gt", "n_detected", "n_matched", "precision", "recall",
                          "pos_err_mean", "pos_err_std", "color_accuracy"])

    print("Rendering plots ...")
    plot_trajectories(traj_gt_sync, traj_slam_aligned, lap_boundaries,
                       os.path.join(out_dir, "trajectory_plot.png"))
    plot_cone_map(cone_snapshots["final"], os.path.join(out_dir, "cone_map_plot.png"))

    print("Summarizing SLAM performance log ...")
    perf = load_performance_csv(perf_csv, wall_clock_range[0], wall_clock_range[1])
    if perf:
        write_csv([perf], os.path.join(out_dir, "performance_summary.csv"),
                  fieldnames=["n_frames", "proc_time_mean_ms", "proc_time_std_ms",
                              "proc_time_p95_ms", "proc_time_max_ms", "effective_fps",
                              "tracking_ok_pct"])
    else:
        print(f"  WARNING: no performance data found at '{perf_csv}' for this run's time window", file=sys.stderr)

    print(f"\nDone. Artifacts written to '{out_dir}/':")
    print("  ate_rpe_table.csv, cone_map_table.csv, trajectory_plot.png, cone_map_plot.png, lap_crossings.csv"
          + (", performance_summary.csv" if perf else ""))

    whole = next(row for row in ate_rpe_rows if row["segment"] == "whole_run")
    final_cones = next(row for row in cone_rows if row["segment"] == "final")
    print("\n--- Summary ---")
    print(f"Whole-run ATE RMSE: {whole['ape_rmse']:.3f} m | RPE RMSE: {whole['rpe_rmse']:.3f} m/frame")
    print(f"Final cone map: {final_cones['n_matched']}/{final_cones['n_gt']} matched "
          f"(precision={final_cones['precision']:.2f}, recall={final_cones['recall']:.2f}, "
          f"color_acc={final_cones['color_accuracy']:.2f})")
    if perf:
        print(f"SLAM tracking OK: {perf['tracking_ok_pct']:.1f}% of frames, "
              f"avg processing {perf['proc_time_mean_ms']:.1f} ms ({perf['effective_fps']:.1f} FPS)")


if __name__ == "__main__":
    main()
