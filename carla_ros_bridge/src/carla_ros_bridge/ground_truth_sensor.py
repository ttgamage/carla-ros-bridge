#!/usr/bin/env python
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
"""
handle a ground truth pose sensor
"""

import bisect
from collections import deque
from threading import Lock

import numpy as np
import tf2_ros

from carla_ros_bridge.pseudo_actor import PseudoActor

from geometry_msgs.msg import Pose, Transform, TransformStamped
from nav_msgs.msg import Odometry
from perception_interfaces.msg import PerceptionStatus

ODOMETRY_TOPIC = "/ground_truth/odometry"
PERCEPTION_STATUS_TOPIC = "/perception/status"
GT_BASE_FRAME = "gt/base_link"
DEFAULT_MAP_FRAME = "map"

# How far back the pose history reaches. The perception stack announces its map
# origin a few frames after the fact, so this only has to cover that delay plus
# slack for a late-starting perception node.
HISTORY_SECONDS = 120.0


def _stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def _quat_to_array(q):
    return np.array([q.x, q.y, q.z, q.w])


def _slerp(q0, q1, alpha):
    """Spherical interpolation between two xyzw quaternions."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:  # take the short way round
        q1, dot = -q1, -dot
    if dot > 0.9995:  # nearly parallel: linear interpolation is exact enough
        q = q0 + alpha * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(dot)
    return (np.sin((1.0 - alpha) * theta) * q0 + np.sin(alpha * theta) * q1) / np.sin(theta)


def _interpolate_pose(p0, p1, alpha):
    pose = Pose()
    pose.position.x = p0.position.x + alpha * (p1.position.x - p0.position.x)
    pose.position.y = p0.position.y + alpha * (p1.position.y - p0.position.y)
    pose.position.z = p0.position.z + alpha * (p1.position.z - p0.position.z)
    q = _slerp(_quat_to_array(p0.orientation), _quat_to_array(p1.orientation), alpha)
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
    return pose


def _pose_to_transform(pose):
    transform = Transform()
    transform.translation.x = pose.position.x
    transform.translation.y = pose.position.y
    transform.translation.z = pose.position.z
    transform.rotation = pose.orientation
    return transform


class GroundTruthSensor(PseudoActor):

    """
    Pseudo sensor exposing the exact pose of its parent vehicle, independently
    of whatever perception stack is running.

    Every tick it publishes:
      - nav_msgs/Odometry on /ground_truth/odometry (world_frame -> gt/base_link)
      - TF world_frame -> gt/base_link

    and once, as static TF:
      - gt/base_link -> <vehicle frame> (identity), so the bridge's own sensor
        frames hang off the ground truth vehicle in the TF tree.

    It also anchors the perception stack's "map" frame to the CARLA world by
    timestamp: by contract "map" is base_link at PerceptionStatus.map_origin_stamp,
    so world_frame -> map is simply the ground truth pose at that instant. The
    pose is looked up in a short history (both sides run on simulation time) and
    published as static TF; it is re-sent whenever the stack starts a new map.

    gt/base_link coincides with the CARLA vehicle origin; the perception stack's
    base_link must use the same definition for the two to be comparable.
    """

    def __init__(self, uid, name, parent, node):
        """
        Constructor

        :param uid: unique identifier for this object
        :type uid: int
        :param name: name identiying this object
        :type name: string
        :param parent: the parent of this
        :type parent: carla_ros_bridge.Parent
        :param node: node-handle
        :type node: carla_ros_bridge.CarlaRosBridge
        """
        super(GroundTruthSensor, self).__init__(uid=uid,
                                                name=name,
                                                parent=parent,
                                                node=node)

        self.world_frame = node.parameters['world_frame']

        self._history_lock = Lock()
        self._history = deque()  # (stamp_sec, geometry_msgs/Pose), ordered by stamp
        self._anchored_map = None  # (map_id, origin_stamp_sec) last anchored

        # rclpy's StaticTransformBroadcaster does not accumulate: every send replaces
        # the latched /tf_static message of this publisher, so all static transforms
        # are kept here and re-sent together.
        self._static_transforms = {}

        self.odometry_publisher = node.new_publisher(Odometry, ODOMETRY_TOPIC, qos_profile=10)
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(node)
        self._static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(node)
        self.status_subscriber = node.new_subscription(PerceptionStatus,
                                                       PERCEPTION_STATUS_TOPIC,
                                                       self._perception_status_updated,
                                                       qos_profile=10)

        vehicle_link = TransformStamped()
        vehicle_link.header = self.get_msg_header(frame_id=GT_BASE_FRAME)
        vehicle_link.child_frame_id = self.parent.get_prefix()
        vehicle_link.transform.rotation.w = 1.0
        self._send_static_transform(vehicle_link)

    def destroy(self):
        """
        Function to destroy this object.
        :return:
        """
        super(GroundTruthSensor, self).destroy()
        self.node.destroy_subscription(self.status_subscriber)
        self.node.destroy_publisher(self.odometry_publisher)

    @staticmethod
    def get_blueprint_name():
        """
        Get the blueprint identifier for the pseudo sensor
        :return: name
        """
        return "sensor.pseudo.ground_truth"

    def update(self, frame, timestamp):
        """
        Function (override) to update this object.
        """
        try:
            pose = self.parent.get_current_ros_pose()
            twist = self.parent.get_current_ros_twist_rotated()
        except AttributeError:
            # parent actor disappeared, do not publish
            self.node.logwarn(
                "GroundTruthSensor could not publish. parent actor {} not found".format(
                    self.parent.uid))
            return

        header = self.get_msg_header(frame_id=self.world_frame, timestamp=timestamp)

        odometry = Odometry(header=header)
        odometry.child_frame_id = GT_BASE_FRAME
        odometry.pose.pose = pose
        odometry.twist.twist = twist
        self.odometry_publisher.publish(odometry)

        self._tf_broadcaster.sendTransform(TransformStamped(
            header=header,
            child_frame_id=GT_BASE_FRAME,
            transform=_pose_to_transform(pose)))

        stamp_sec = _stamp_to_sec(header.stamp)
        with self._history_lock:
            self._history.append((stamp_sec, pose))
            while self._history and self._history[0][0] < stamp_sec - HISTORY_SECONDS:
                self._history.popleft()

    def _perception_status_updated(self, status):
        if status.state == PerceptionStatus.NOT_INITIALIZED:
            return
        origin_sec = _stamp_to_sec(status.map_origin_stamp)
        map_key = (status.map_id, origin_sec)
        if map_key == self._anchored_map:
            return

        pose = self._pose_at(origin_sec)
        if pose is None:
            # Logged on every status message until it resolves; that is intended,
            # a missing anchor means the map overlay is wrong.
            self.node.logwarn(
                "GroundTruthSensor: no ground truth around map origin t={:.3f}s, "
                "cannot anchor the perception map".format(origin_sec))
            return

        map_frame = status.header.frame_id or DEFAULT_MAP_FRAME
        anchor = TransformStamped()
        anchor.header = self.get_msg_header(frame_id=self.world_frame, timestamp=origin_sec)
        anchor.child_frame_id = map_frame
        anchor.transform = _pose_to_transform(pose)
        self._send_static_transform(anchor)
        self._anchored_map = map_key
        self.node.loginfo(
            "GroundTruthSensor: anchored '{}' (map_id={}) at t={:.3f}s, "
            "position ({:.2f}, {:.2f}, {:.2f}) in '{}'".format(
                map_frame, status.map_id, origin_sec,
                pose.position.x, pose.position.y, pose.position.z, self.world_frame))

    def _pose_at(self, stamp_sec):
        """
        Ground truth pose at stamp_sec, interpolated between the two surrounding
        ticks. In synchronous mode the stamp matches a tick exactly. Returns None
        if stamp_sec is outside the recorded history.
        """
        with self._history_lock:
            history = list(self._history)
        if not history:
            return None
        stamps = [entry[0] for entry in history]
        # tolerate float round-trip through builtin_interfaces/Time
        eps = 1e-6
        if stamp_sec < stamps[0] - eps or stamp_sec > stamps[-1] + eps:
            return None
        i = bisect.bisect_left(stamps, stamp_sec)
        if i < len(stamps) and abs(stamps[i] - stamp_sec) <= eps:
            return history[i][1]
        if i == 0:
            return history[0][1]
        if i == len(stamps):
            return history[-1][1]
        (t0, p0), (t1, p1) = history[i - 1], history[i]
        return _interpolate_pose(p0, p1, (stamp_sec - t0) / (t1 - t0))

    def _send_static_transform(self, transform):
        self._static_transforms[transform.child_frame_id] = transform
        self._static_tf_broadcaster.sendTransform(list(self._static_transforms.values()))
