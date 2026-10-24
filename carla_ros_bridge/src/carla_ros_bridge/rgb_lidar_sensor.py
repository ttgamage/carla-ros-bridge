#!/usr/bin/env python
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
"""
handle an RGB LiDAR pseudo sensor
"""

import math
from threading import Lock

import numpy as np
import tf2_ros

import carla_common.transforms as trans
from carla_ros_bridge.pseudo_actor import PseudoActor

from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2, PointField

LIDAR_ROLE = "rgb_lidar"
CAMERA_ROLE_PREFIX = "rgb_lidar_cam"
UNCOLORED = np.array([255, 0, 255], np.uint8)

FIELDS = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name='rgb', offset=16, datatype=PointField.FLOAT32, count=1),
]
POINT_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4'), ('rgb', '<u4')])


def colorize(points, lidar_to_world, images):
    """
    Colors each LiDAR point (Nx3, CARLA frame) with the pixel of the camera whose
    optical axis is closest to it. images: carla.Image of the same frame.
    Returns Nx3 uint8 RGB, UNCOLORED where no camera sees the point.
    """
    hom = np.c_[points, np.ones(len(points))]
    cam_points = [(hom @ (np.linalg.inv(np.array(img.transform.get_matrix())) @ lidar_to_world).T)[:, :3]
                  for img in images]
    cosines = np.stack([p[:, 0] / np.maximum(np.linalg.norm(p, axis=1), 1e-6) for p in cam_points])
    best = np.argmax(cosines, axis=0)

    colors = np.tile(UNCOLORED, (len(points), 1))
    for i, (img, p) in enumerate(zip(images, cam_points)):
        w, h = img.width, img.height
        f = w / (2.0 * math.tan(math.radians(img.fov) / 2.0))
        idx = np.flatnonzero((best == i) & (p[:, 0] > 0))
        x, y, z = p[idx].T
        u = np.floor(w / 2.0 + f * y / x).astype(int)
        v = np.floor(h / 2.0 - f * z / x).astype(int)
        inside = (u >= 0) & (u <= w) & (v >= 0) & (v <= h)  # image border included
        idx = idx[inside]
        u, v = np.minimum(u[inside], w - 1), np.minimum(v[inside], h - 1)
        pixels = np.frombuffer(img.raw_data, np.uint8).reshape(h, w, 4)
        colors[idx] = pixels[v, u, 2::-1]  # BGRA -> RGB
    return colors


class RgbLidarSensor(PseudoActor):

    """
    Pseudo sensor fusing the parent's LiDAR (role_name LIDAR_ROLE) with its RGB
    cameras (role_name CAMERA_ROLE_PREFIX*): each LiDAR slice is colored with the
    images of the same frame, and every full turn is published as a PointCloud2
    (x, y, z, intensity, rgb as in pcl::PointXYZRGB) on /carla/<parent>/<name>,
    stamped at its last slice. The points of each slice stay in the LiDAR frame
    of that slice, as with a real rotating LiDAR. TF world_frame -> LiDAR frame
    is the pose at the last slice.
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
        :type node: CompatibleNode
        """
        super(RgbLidarSensor, self).__init__(uid=uid,
                                             name=name,
                                             parent=parent,
                                             node=node)
        world = node.carla_world
        rig = [a for a in world.get_actors().filter("sensor.*")
               if a.parent is not None and a.parent.id == parent.carla_actor.id]
        lidars = [a for a in rig if a.type_id == "sensor.lidar.ray_cast"
                  and a.attributes.get("role_name") == LIDAR_ROLE]
        self.cameras = [a for a in rig if a.type_id == "sensor.camera.rgb"
                        and a.attributes.get("role_name", "").startswith(CAMERA_ROLE_PREFIX)]
        if not lidars or not self.cameras:
            raise RuntimeError("{}: no '{}' LiDAR or '{}*' cameras on parent {}".format(
                name, LIDAR_ROLE, CAMERA_ROLE_PREFIX, parent.carla_actor.id))
        self.lidar = lidars[0]

        dt = world.get_settings().fixed_delta_seconds
        self.ticks_per_scan = round(1.0 / (float(self.lidar.attributes["rotation_frequency"]) * dt))

        self._lock = Lock()
        self._frames = {}       # frame -> {role_name: carla sensor data}
        self._first_frame = None
        self._scan = []         # (points, intensity, rgb) of the slices of the current turn
        self._scan_id = None

        self.publisher = node.new_publisher(PointCloud2, self.get_topic_prefix(), qos_profile=10)
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(node)
        for actor in [self.lidar] + self.cameras:
            role = actor.attributes["role_name"]
            actor.listen(lambda data, role=role: self._on_data(role, data))

    def destroy(self):
        """
        Function (override) to destroy this object.
        Stops listening to the rig sensors still alive (stop() on a destroyed one raises).
        """
        for actor in [self.lidar] + self.cameras:
            try:
                actor.stop()
            except RuntimeError:  # already destroyed by its owner, is_alive may still be True
                pass
        self.node.destroy_publisher(self.publisher)
        super(RgbLidarSensor, self).destroy()

    @staticmethod
    def get_blueprint_name():
        """
        Get the blueprint identifier for the pseudo sensor
        :return: name
        """
        return "sensor.pseudo.rgb_lidar"

    def _on_data(self, role, data):
        """Buffers the data by frame; a frame with the LiDAR and every camera is fused, older incomplete ones dropped."""
        with self._lock:
            frame = self._frames.setdefault(data.frame, {})
            frame[role] = data
            if len(frame) < 1 + len(self.cameras):
                return
            stale = [f for f in self._frames if f < data.frame]
            if stale:
                self.node.logwarn("{}: dropped {} incomplete frames".format(self.name, len(stale)))
            for f in stale + [data.frame]:
                del self._frames[f]
            self._add_slice(frame)

    def _add_slice(self, frame):
        """Colors the frame's LiDAR slice and publishes the turn at its last slice."""
        lidar = frame[LIDAR_ROLE]
        if self._first_frame is None:
            self._first_frame = lidar.frame
        scan_id, index = divmod(lidar.frame - self._first_frame, self.ticks_per_scan)
        if scan_id != self._scan_id:
            self._scan = []  # leftover slices of a turn whose last slice was dropped
            self._scan_id = scan_id

        raw = np.frombuffer(lidar.raw_data, np.float32).reshape(-1, 4).copy()  # CARLA reuses the buffer
        images = [frame[c.attributes["role_name"]] for c in self.cameras]
        rgb = colorize(raw[:, :3].astype(np.float64), np.array(lidar.transform.get_matrix()), images)
        self._scan.append((raw, rgb))
        if index == self.ticks_per_scan - 1:
            self._publish(lidar)
            self._scan = []

    def _publish(self, last_slice):
        """Publishes the turn as PointCloud2 (right-handed: y flipped) and the TF of the LiDAR at its last slice."""
        raw = np.concatenate([r for r, _ in self._scan])
        rgb = np.concatenate([c for _, c in self._scan]).astype(np.uint32)
        cloud = np.empty(len(raw), POINT_DTYPE)
        cloud['x'], cloud['y'], cloud['z'], cloud['intensity'] = raw[:, 0], -raw[:, 1], raw[:, 2], raw[:, 3]
        cloud['rgb'] = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]

        header = self.get_msg_header(timestamp=last_slice.timestamp)
        self.publisher.publish(PointCloud2(header=header, height=1, width=len(cloud), fields=FIELDS,
                                           is_bigendian=False, point_step=POINT_DTYPE.itemsize,
                                           row_step=POINT_DTYPE.itemsize * len(cloud), is_dense=True,
                                           data=cloud.tobytes()))

        pose = trans.carla_transform_to_ros_pose(last_slice.transform)
        transform = TransformStamped()
        transform.header.stamp = header.stamp
        transform.header.frame_id = self.node.parameters['world_frame']
        transform.child_frame_id = self.get_prefix()
        transform.transform.translation.x = pose.position.x
        transform.transform.translation.y = pose.position.y
        transform.transform.translation.z = pose.position.z
        transform.transform.rotation = pose.orientation
        self._tf_broadcaster.sendTransform(transform)
