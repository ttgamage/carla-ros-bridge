#!/usr/bin/env python
#
# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
"""
handle a cone ground truth sensor
"""

from carla_ros_bridge.actor import Actor
from carla_ros_bridge.pseudo_actor import PseudoActor

from perception_interfaces.msg import Cone, ConeArray

CONE_MAP_TOPIC = "/ground_truth/cone_map"

# CARLA blueprint id (lowercase, no separators) -> perception_interfaces color enum.
# Static cone props are never reclassified at runtime, so a fixed lookup is enough.
# The red cone prop stands in for the FS big orange cone.
_CONE_COLOR_BY_BLUEPRINT = {
    "staticpropbluecone": Cone.BLUE,
    "staticpropblue_cone": Cone.BLUE,
    "staticpropyellowcone": Cone.YELLOW,
    "staticpropyellow_cone": Cone.YELLOW,
    "staticproporangecone": Cone.ORANGE,
    "staticproporange_cone": Cone.ORANGE,
    "staticpropredcone": Cone.BIG_ORANGE,
    "staticpropred_cone": Cone.BIG_ORANGE,
}


def _cone_color(type_id):
    key = type_id.replace(".", "").replace("-", "").lower()
    return _CONE_COLOR_BY_BLUEPRINT.get(key, Cone.UNKNOWN)


class ConeGroundTruthSensor(PseudoActor):

    """
    Pseudo sensor exposing the exact world-frame position of every spawned
    track cone, bypassing perception entirely. Cones are static props (not
    Vehicle/Walker), so they are invisible to sensor.pseudo.objects.

    Publishes a perception_interfaces/ConeArray on /ground_truth/cone_map,
    expressed in the CARLA world frame (world_frame parameter).
    """

    def __init__(self, uid, name, parent, node, actor_list):
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
        :param actor_list: current list of actors
        :type actor_list: map(carla-actor-id -> python-actor-object)
        """
        super(ConeGroundTruthSensor, self).__init__(uid=uid,
                                                     name=name,
                                                     parent=parent,
                                                     node=node)
        self.actor_list = actor_list
        self.cone_publisher = node.new_publisher(ConeArray,
                                                 CONE_MAP_TOPIC,
                                                 qos_profile=10)

    def destroy(self):
        """
        Function to destroy this object.
        :return:
        """
        super(ConeGroundTruthSensor, self).destroy()
        self.actor_list = None
        self.node.destroy_publisher(self.cone_publisher)

    @staticmethod
    def get_blueprint_name():
        """
        Get the blueprint identifier for the pseudo sensor
        :return: name
        """
        return "sensor.pseudo.cone_ground_truth"

    def update(self, frame, timestamp):
        """
        Function (override) to update this object.
        Publishes the world-frame position of every known cone actor.
        :return:
        """
        ros_cones = ConeArray()
        ros_cones.header = self.get_msg_header(frame_id=self.node.parameters['world_frame'],
                                               timestamp=timestamp)

        for actor in self.actor_list.values():
            if not isinstance(actor, Actor):
                continue
            type_id = actor.carla_actor.type_id
            if "cone" not in type_id:
                continue

            cone = Cone()
            cone.id = actor.get_id()
            cone.position = actor.get_current_ros_pose().position
            cone.color = _cone_color(type_id)
            cone.confidence = 1.0
            ros_cones.cones.append(cone)

        self.cone_publisher.publish(ros_cones)
