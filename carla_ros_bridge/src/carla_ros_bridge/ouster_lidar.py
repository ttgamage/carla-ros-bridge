#!/usr/bin/env python

"""
Ray-cast LiDAR published as full turns in the Ouster point layout (x y z intensity t reflectivity ring
ambient range), for LiDAR-inertial odometry such as FAST-LIO (lidar_type 3, timestamp_unit 3).

The actor factory creates it, instead of the plain Lidar, for every sensor.lidar.ray_cast whose role_name
starts with OUSTER_ROLE_PREFIX: no extra spawn step. Topic /carla/<parent>/<role_name>, frame
<parent>/<role_name>, header stamp = first slice of the turn, t = ns from it (see ouster_format).
A turn is 1 / (rotation_frequency * step) consecutive simulation frames (see ouster_format.TurnAccumulator).
The bridge parameter ouster_time_mode ('sweep' or 'zero') chooses the per-slice t or every t = 0.
"""

from carla_ros_bridge.ouster_format import (OUSTER_DTYPE, TurnAccumulator, ouster_fields, slice_points)
from carla_ros_bridge.sensor import Sensor

from sensor_msgs.msg import PointCloud2, PointField

OUSTER_ROLE_PREFIX = "ouster"

FIELDS = [PointField(name=name, offset=offset, datatype=datatype, count=1)
          for name, offset, datatype in ouster_fields()]


class OusterLidar(Sensor):

    """
    Actor implementation details for a ray-cast LiDAR published in the Ouster layout
    """

    def __init__(self, uid, name, parent, relative_spawn_pose, node, carla_actor, synchronous_mode):
        """
        Constructor

        :param uid: unique identifier for this object
        :type uid: int
        :param name: name identiying this object
        :type name: string
        :param parent: the parent of this
        :type parent: carla_ros_bridge.Parent
        :param relative_spawn_pose: the spawn pose of this
        :type relative_spawn_pose: geometry_msgs.Pose
        :param node: node-handle
        :type node: CompatibleNode
        :param carla_actor: carla actor object
        :type carla_actor: carla.Actor
        :param synchronous_mode: use in synchronous mode?
        :type synchronous_mode: bool
        """
        super(OusterLidar, self).__init__(uid=uid,
                                          name=name,
                                          parent=parent,
                                          relative_spawn_pose=relative_spawn_pose,
                                          node=node,
                                          carla_actor=carla_actor,
                                          synchronous_mode=synchronous_mode)

        time_mode = node.parameters.get("ouster_time_mode", "sweep")
        self.accumulator = TurnAccumulator(float(carla_actor.attributes["rotation_frequency"]), time_mode)
        self._ticks_logged = None
        node.loginfo("{}: Ouster layout, {} channels, {} Hz, t mode '{}'".format(
            self.get_prefix(), carla_actor.attributes["channels"],
            carla_actor.attributes["rotation_frequency"], time_mode))

        self.lidar_publisher = node.new_publisher(PointCloud2,
                                                  self.get_topic_prefix(),
                                                  qos_profile=10)
        self.listen()

    def destroy(self):
        super(OusterLidar, self).destroy()
        self.node.destroy_publisher(self.lidar_publisher)

    # pylint: disable=arguments-differ
    def sensor_data_updated(self, carla_lidar_measurement):
        """
        Adds the slice of this step to the current turn and publishes the turn when it is complete.

        :param carla_lidar_measurement: carla lidar measurement object
        :type carla_lidar_measurement: carla.LidarMeasurement
        """
        m = carla_lidar_measurement
        counts = [m.get_point_count(channel) for channel in range(m.channels)]
        turn = self.accumulator.add(m.frame, m.timestamp, *slice_points(m.raw_data, counts))
        if turn is None:
            return
        ticks = self.accumulator.ticks_per_turn
        if ticks != self._ticks_logged:  # logged at the first turn published with a new step
            self._ticks_logged = ticks
            self.node.loginfo("{}: {} slices per turn".format(self.get_prefix(), ticks))
            if not self.accumulator.exact:
                self.node.logwarn("{}: 1 / (rotation_frequency * fixed_delta_seconds) is not an integer, "
                                  "turns do not cover exactly 360 deg".format(self.get_prefix()))
        stamp, cloud = turn
        header = self.get_msg_header(timestamp=stamp)
        self.lidar_publisher.publish(PointCloud2(header=header,
                                                 height=1,
                                                 width=len(cloud),
                                                 fields=FIELDS,
                                                 is_bigendian=False,
                                                 point_step=OUSTER_DTYPE.itemsize,
                                                 row_step=OUSTER_DTYPE.itemsize * len(cloud),
                                                 is_dense=True,
                                                 data=cloud.tobytes()))
