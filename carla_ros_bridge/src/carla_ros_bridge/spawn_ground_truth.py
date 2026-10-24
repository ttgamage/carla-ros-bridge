#!/usr/bin/env python
#
# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
"""
One-shot helper that requests the ground-truth pseudo-sensors this project
needs (ego pose + track cone positions) from the running carla_ros_bridge,
via its /carla/spawn_object service.

These pseudo-sensors are not part of objects.json and are not spawned by
carla_spawn_objects, since the ego vehicle here is spawned directly against
the CARLA client (spawn_vehicles.py), not through the bridge's own
spawn-objects flow. Run this once after both the bridge and the vehicle are up.
"""

import sys

import carla
import ros_compatibility as roscomp
from ros_compatibility.node import CompatibleNode
from carla_msgs.srv import SpawnObject
from geometry_msgs.msg import Pose

CARLA_HOST = "localhost"
CARLA_PORT = 2000
CARLA_TIMEOUT_S = 5.0


def _find_ego_vehicle_id(node):
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(CARLA_TIMEOUT_S)
    world = client.get_world()
    # In synchronous mode a freshly-connected client's actor registry is only
    # populated after it has observed a tick - querying immediately returns empty.
    world.wait_for_tick(seconds=CARLA_TIMEOUT_S)
    vehicles = world.get_actors().filter("vehicle.*")
    if not vehicles:
        raise RuntimeError("No vehicle actor found in the CARLA world - spawn it first")
    if len(vehicles) > 1:
        node.logwarn(
            "Multiple vehicle actors found ({}), using the first one (id={})".format(
                len(vehicles), vehicles[0].id))
    return vehicles[0].id


def _identity_pose():
    pose = Pose()
    pose.orientation.w = 1.0
    return pose


class SpawnGroundTruth(CompatibleNode):

    def __init__(self):
        super(SpawnGroundTruth, self).__init__("spawn_ground_truth")
        self.spawn_object_client = self.new_client(SpawnObject, "/carla/spawn_object")

    def _spawn(self, type_id, object_id, attach_to):
        request = roscomp.get_service_request(SpawnObject)
        request.type = type_id
        request.id = object_id
        request.attach_to = attach_to
        request.transform = _identity_pose()
        request.random_pose = False
        response = self.call_service(self.spawn_object_client, request,
                                     spin_until_response_received=True)
        if response.id == -1:
            raise RuntimeError("Failed to spawn '{}': {}".format(type_id, response.error_string))
        self.loginfo("Spawned '{}' (id={})".format(type_id, response.id))

    def run(self):
        ego_vehicle_id = _find_ego_vehicle_id(self)
        self.loginfo("Attaching ground truth sensors to vehicle id={}".format(ego_vehicle_id))
        self._spawn("sensor.pseudo.ground_truth", "ground_truth", attach_to=ego_vehicle_id)
        # Legacy odometry topic, still read by lap_monitor.
        self._spawn("sensor.pseudo.odom", "ground_truth_odom", attach_to=ego_vehicle_id)
        self._spawn("sensor.pseudo.cone_ground_truth", "cone_ground_truth", attach_to=0)


def main(args=None):
    roscomp.init("spawn_ground_truth", args=args)
    node = None
    try:
        node = SpawnGroundTruth()
        node.run()
    except (RuntimeError, IndexError) as exc:
        if node:
            node.logerr(str(exc))
        else:
            print(exc, file=sys.stderr)
        sys.exit(1)
    finally:
        if node:
            node.destroy_node()
        roscomp.shutdown()


if __name__ == "__main__":
    main()
