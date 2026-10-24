#!/usr/bin/env python
#
# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
"""
Orchestrates one bounded evaluation recording: starts lap_monitor + `ros2 bag
record` for the ground-truth and SLAM topics, and stops both cleanly once
lap_monitor signals it has seen the requested number of lap crossings.

Assumes carla_ros_bridge, orbslam_ros and the CARLA sim (with ground-truth
pseudo-sensors already attached via spawn_ground_truth) are already running.
"""

import argparse
import datetime
import os
import signal
import subprocess
import sys

import rclpy
import ros_compatibility as roscomp
from ros_compatibility.node import CompatibleNode
from std_msgs.msg import Empty

RECORDED_TOPICS = [
    "/carla/autopilot/ground_truth_odom",
    "/carla/cone_ground_truth",
    "/slam/pose",
    "/slam/cones",
]


class RecordEvaluation(CompatibleNode):

    def __init__(self, target_laps, start_x, start_y, bag_path, cross_radius, min_away_dist):
        super(RecordEvaluation, self).__init__("record_evaluation")
        self._done = False
        self.new_subscription(Empty, "/lap_monitor/done", self._done_cb, qos_profile=10)

        self.loginfo(f"Starting lap_monitor (target_laps={target_laps})")
        self._lap_monitor_proc = subprocess.Popen([
            "ros2", "run", "carla_ros_bridge", "lap_monitor", "--ros-args",
            "-p", f"start_x:={start_x}",
            "-p", f"start_y:={start_y}",
            "-p", f"target_laps:={target_laps}",
            "-p", f"cross_radius:={cross_radius}",
            "-p", f"min_away_dist:={min_away_dist}",
            "-p", f"output_csv:={bag_path}_lap_crossings.csv",
        ], start_new_session=True)  # own process group, so _stop() can signal its ros2-run + real-node tree together

        self.loginfo(f"Recording {RECORDED_TOPICS} to {bag_path}")
        self._bag_proc = subprocess.Popen(
            ["ros2", "bag", "record", "-o", bag_path] + RECORDED_TOPICS,
            start_new_session=True,
        )

    def _done_cb(self, msg):
        self.loginfo("lap_monitor signalled done -- stopping recording.")
        self._done = True

    def run(self):
        while not self._done:
            rclpy.spin_once(self, timeout_sec=0.2)

        self._stop(self._bag_proc, "ros2 bag record")
        self._stop(self._lap_monitor_proc, "lap_monitor")

    @staticmethod
    def _stop(proc, name):
        # `ros2 run`/`ros2 bag` spawn the real worker as a child of the process we
        # hold a handle to -- signalling just that one process leaves the actual
        # worker (e.g. the lap_monitor node itself) running as an orphan. Each was
        # started with start_new_session=True, so its whole tree shares one
        # process group; signal that group instead.
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            print(f"{name} didn't exit within 15s of SIGINT, killing its process group", file=sys.stderr)
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-x", type=float, required=True)
    parser.add_argument("--start-y", type=float, required=True)
    parser.add_argument("--laps", type=int, default=5)
    parser.add_argument("--cross-radius", type=float, default=5.0)
    parser.add_argument("--min-away-dist", type=float, default=20.0)
    parser.add_argument("--bag-path", type=str, default=None,
                        help="Defaults to bags/eval_<timestamp> in the current directory")
    parsed, ros_args = parser.parse_known_args(args=sys.argv[1:] if args is None else args)

    bag_path = parsed.bag_path or (
        "bags/eval_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    # ros2 bag record refuses to run if bag_path itself already exists, so only
    # ensure its parent directory is there.
    parent = os.path.dirname(bag_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    roscomp.init("record_evaluation", args=ros_args)
    node = None
    try:
        node = RecordEvaluation(
            parsed.laps, parsed.start_x, parsed.start_y, bag_path,
            parsed.cross_radius, parsed.min_away_dist,
        )
        node.run()
        node.loginfo(f"Done. Bag at '{bag_path}', lap crossings at '{bag_path}_lap_crossings.csv'.")
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.destroy_node()
        roscomp.shutdown()


if __name__ == "__main__":
    main()
