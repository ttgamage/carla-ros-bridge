#!/usr/bin/env python
#
# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
"""
Detects lap crossings from ground-truth odometry (not the SLAM pose, which
may have drifted) so the evaluation bag-recording can be bounded by a
repeatable number of laps instead of a fixed wall-clock duration.

A crossing is counted when the vehicle comes back within `cross_radius` of
the start point, but only after it has previously gotten further than
`min_away_dist` away -- this hysteresis avoids re-triggering on GPS-like
jitter right at the line.
"""

import csv
import math
import sys

import ros_compatibility as roscomp
from ros_compatibility.node import CompatibleNode
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32, Empty


class LapMonitor(CompatibleNode):

    def __init__(self):
        super(LapMonitor, self).__init__("lap_monitor")

        # start_x/start_y are given in raw CARLA world coordinates (e.g. straight
        # out of track.yaml), matching how every other tool in this project
        # refers to track position. carla_common.transforms negates Y when
        # converting a carla.Transform to a ROS pose (left-handed -> right-handed
        # convention), so ground-truth odometry messages have a Y sign flipped
        # relative to that raw CARLA value -- negate once here so callers never
        # have to think about it.
        raw_start_x = self.get_param("start_x", 0.0)
        raw_start_y = self.get_param("start_y", 0.0)
        self.start_x = raw_start_x
        self.start_y = -raw_start_y
        self.cross_radius = self.get_param("cross_radius", 5.0)
        self.min_away_dist = self.get_param("min_away_dist", 20.0)
        self.target_laps = self.get_param("target_laps", 0)  # 0 = unbounded, just log every crossing
        self.output_csv = self.get_param("output_csv", "lap_crossings.csv")

        self._away = False
        self._lap_count = 0

        self._csv_file = open(self.output_csv, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(["lap", "stamp_sec", "stamp_nanosec", "x", "y"])

        self.lap_count_pub = self.new_publisher(Int32, "/lap_monitor/lap_count", qos_profile=10)
        self.done_pub = self.new_publisher(Empty, "/lap_monitor/done", qos_profile=10)

        odom_topic = self.get_param("odom_topic", "/carla/autopilot/ground_truth_odom")
        self.new_subscription(Odometry, odom_topic, self._odom_cb, qos_profile=10)
        self.loginfo(
            f"Watching '{odom_topic}' for crossings of ({raw_start_x:.2f}, {raw_start_y:.2f}) "
            f"[CARLA world coords], cross_radius={self.cross_radius}, min_away_dist={self.min_away_dist}, "
            f"target_laps={self.target_laps or 'unbounded'}"
        )

    def _odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        dist = math.hypot(x - self.start_x, y - self.start_y)

        if dist > self.min_away_dist:
            self._away = True
            return

        if dist < self.cross_radius and self._away:
            self._away = False
            self._lap_count += 1
            stamp = msg.header.stamp
            self._csv_writer.writerow([self._lap_count, stamp.sec, stamp.nanosec, x, y])
            self._csv_file.flush()
            self.loginfo(f"Lap crossing #{self._lap_count} at ({x:.2f}, {y:.2f})")
            self.lap_count_pub.publish(Int32(data=self._lap_count))

            if self.target_laps and self._lap_count >= self.target_laps:
                self.loginfo(f"Reached target of {self.target_laps} laps, signalling done.")
                self.done_pub.publish(Empty())

    def destroy(self):
        self._csv_file.close()
        self.destroy_node()


def main(args=None):
    roscomp.init("lap_monitor", args=args)
    node = None
    try:
        node = LapMonitor()
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.destroy()
        try:
            roscomp.shutdown()
        except Exception:
            # SIGINT reaches this process and its ros2-run parent at the same
            # time (both in the same process group); whichever gets there
            # first already tore down the shared rclpy context.
            pass


if __name__ == "__main__":
    main()
