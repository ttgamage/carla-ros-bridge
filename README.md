# ROS2 bridge for CARLA simulator v0.9.15

This ROS package is a modified fork of the [carla-simulator/ros-bridge](https://github.com/carla-simulator/ros-bridge) package that is adopted to work with ROS2 Humble running on Ubuntu 22.04 LTS with Scenario Runner v0.9.15. The ROS bridge enables two-way communication between ROS and CARLA. The information from the CARLA server is translated to ROS topics. In the same way, the messages sent between nodes in ROS get translated to commands to be applied in CARLA.

## eagletrt additions

On top of upstream, this fork adds what the perception stack needs for evaluation
(see the [perception workspace README](https://github.com/eagletrt/perception-ws-sw)).

**World frame.** The CARLA world TF frame is the `world_frame` parameter, default
`carla_world` (upstream hardcodes `map`). `map` is reserved for the perception
stack's own world, so the two never clash.

**Ground truth pseudo-sensors**, spawned with `ros2 run carla_ros_bridge spawn_ground_truth`
once the bridge and the ego vehicle are up:

| Blueprint | Publishes |
|---|---|
| `sensor.pseudo.ground_truth` (attached to the ego vehicle) | `/ground_truth/odometry` (`nav_msgs/Odometry`, `carla_world` → `gt/base_link`), TF `carla_world → gt/base_link`, static TF `gt/base_link → <role name>` |
| `sensor.pseudo.cone_ground_truth` | `/ground_truth/cone_map` (`perception_interfaces/ConeArray` in `carla_world`; red cones = `BIG_ORANGE`) |

`gt/base_link` is the CARLA vehicle origin.

**Anchoring the perception map.** The ground truth sensor listens to
`/perception/status`. When a perception stack reports that its map started at
time t (`map_origin_stamp`), the sensor looks up the vehicle's ground truth pose at
t (from a 120 s history, interpolated) and publishes it as static TF
`carla_world → map`. A new `map_id` re-anchors. The perception stack itself never
sees ground truth.

**RGB LiDAR pseudo-sensor** `sensor.pseudo.rgb_lidar`, spawned with
`ros2 run carla_ros_bridge spawn_rgb_lidar` once the bridge and the ego vehicle are up.
It reads a rig mounted on the vehicle (carla-sim-sw setup `microlino/rgb_lidar`): a
`sensor.lidar.ray_cast` with role name `rgb_lidar` and `sensor.camera.rgb` cameras with role
names `rgb_lidar_cam_*`, all with `sensor_tick: 0` and `1 / (rotation_frequency * fixed_delta_seconds)`
an integer. The bridge does not register the rig sensors itself (no image topics). Every LiDAR
slice is colored with the camera images of the same frame (pinhole projection, the camera
whose optical axis is closest), and a full turn is published as one message:

| Topic | Content |
|---|---|
| `/carla/hero/rgb_lidar` | `sensor_msgs/PointCloud2`, frame `hero/rgb_lidar`, stamp of the last slice, fields `x y z intensity rgb` (float32, 20 bytes, `rgb` packed as in `pcl::PointXYZRGB`); points no camera sees are magenta |
| TF `carla_world → hero/rgb_lidar` | LiDAR pose at the last slice |

- Each slice stays in the LiDAR frame of its own instant (no motion compensation), and there
  are no `ring` or per-point `time` fields.
- About 1.8 MB per turn: subscribe with reliable QoS, best effort (`SensorDataQoS`) loses
  around half of the turns.
- The cameras render at every tick, so the simulation runs at about 0.25x real time:
  consumers must use `use_sim_time:=true`.

**Notes**
- Do not spawn `sensor.pseudo.tf` next to the ground truth sensor: both would give
  the vehicle frame a parent.
- In passive mode the bridge publishes sensor TF directly from `carla_world`
  instead of from the vehicle frame; check that `<role name> → <role name>/<camera>`
  resolves for the perception stack's camera mount lookup.
- `lap_monitor`, `record_evaluation` and `evaluate_run` still use the previous
  topics (`/carla/autopilot/...`, `/slam/*`) and are due to be rewritten.

## Main Requirements

- OS: Ubuntu 22.04 LTS
- CARLA Version: 0.9.15
- Scenario Runner Version: 0.9.15
- ROS Version: Humble
- **NOTE**: All testing were peformed using Python 3.10. The default CARLA PythonAPI only supports Python 2.7 and 3.7 (and 3.8 by extension). Updated `.whl` and `.egg` files for Python 3.10 can be found at [https://github.com/gezp/carla_ros/releases/](https://github.com/gezp/carla_ros/releases/).
- > **⚠️ UPDATE (07/26)**: Starting with v0.9.16, CARLA now supports a natively integrated ros bridge. I've moved on to the native ros bridge, but if your workflow still involves a 3rd party ros bridge, try using newer `.whl` files for Python 3.10, 3.12, 3.14 found at [https://github.com/ttgamage/carla-whl-builder/releases](https://github.com/ttgamage/carla-whl-builder/releases). There are a few minor differences in more recent ROS2 versions but it's possible to run carla ros bridge on newer OS releases/python versions with a bit of an effort. `podman/docker` is your friend. :)
### Screenshot of Carla AD Demo in Action 

![rviz setup](./docs/images/ad_demo.png "AD Demo")

## Instructions (adapted from [ROS Bridge Documentation](https://carla.readthedocs.io/projects/ros-bridge/en/latest/ros_installation_ros2/))
1. Set up a project directory and clone the ROS bridge repository and submodules:
```
mkdir -p ~/Workspace/ros-bridge && cd ~/Workspace/ros-bridge
git clone --recurse-submodules https://github.com/ttgamage/carla-ros-bridge.git
mv carla-ros-bridge src
```
2. Set up ROS environment and install dependencies:
```
source /opt/ros/humble/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src -r
```
3. Build the ROS bridge workspace using colcon:
```
colcon build --symlink-install
```


