# Multi-robot DPVO over ROS 2

This frontend runs one DPVO instance per robot and performs distributed classic
loop-closure detection with a bandwidth-aware two-stage exchange:

1. A keyframe becomes stable after it leaves DPVO's removal window.
2. The robot publishes only its sparse, normalized DBoW2 vector.
3. Peers score that vector against their stable local keyframes. The standard
   three-consecutive-match check and loop NMS are applied.
4. The deterministic owner of the candidate calls the matched robot's
   `GetKeyframe` service. No image, point, or descriptor data is sent before
   this step.
5. The service returns the requested keyframe's locally triangulated 3D points,
   DISK keypoints/descriptors, pose, and image dimensions.
6. LightGlue forms putative cross-robot correspondences. TEASER++ estimates a
   robust Sim(3), which must pass the configured residual inlier count and ratio.
7. The accepted query-to-match Sim(3) and both local keyframe poses are
   published as an `InterRobotLoopClosure` message.
8. The included centralized solver estimates one Sim(3) local-map-to-world
   alignment per robot, publishes the alignments, and republishes each local
   trajectory in the common `world` frame.

Classic intra-robot loop closure and DPV-SLAM's patch-based loop closure remain
enabled. Local visual odometry remains independent on every robot; only the
map-level alignments are optimized centrally.

## Packages

- `dpvo_multi_robot_interfaces`: typed BoW, detailed keyframe, service, and
  inter-robot Sim(3) definitions.
- `dpvo_multi_robot`: live image/CameraInfo DPVO node and ROS 2 transport.

## Build

ROS 2 Humble and DPVO's Pixi environment must be available. All robots must use
the same ORB vocabulary.

```bash
cd /path/to/DPVO
pixi run verify-multi-robot

source /opt/ros/humble/setup.bash
colcon build --base-paths ros2 --symlink-install
source install/setup.bash
export DPVO_ROOT=$PWD
```

`verify-multi-robot` rebuilds the modified DBoW2 binding, installs the pinned
TEASER++ Python binding, and runs the transport-neutral tests. The ROS executable
automatically re-enters this Pixi environment; `DPVO_ROOT` is only needed for a
non-symlink installation where the source tree cannot be discovered. Set
`DPVO_PIXI_MANIFEST` when deployment uses a machine-specific manifest such as
`deploy/blackwell_ros2/pixi.toml`.

For a parent workspace, run colcon from its root instead:

```bash
colcon build --base-paths src/DPVO/ros2 --symlink-install
source install/setup.bash
export DPVO_ROOT=$PWD/src/DPVO
```

## Run a robot

```bash
ros2 launch dpvo_multi_robot multi_robot.launch.py \
  robot_id:=robot0 \
  image_topic:=/robot0/cam0/image_raw \
  camera_info_topic:=/robot0/cam0/camera_info \
  network:=/path/to/DPVO/dpvo.pth \
  config:=/path/to/DPVO/config/fast.yaml \
  orb_vocab:=/data3/mikexyl/datasets/orb_vocab/ORBvoc.txt
```

Run the same command on each robot with a unique `robot_id` and its camera
topics. Use the same `ROS_DOMAIN_ID` and a DDS configuration that allows the
robots to discover one another. Robot IDs may contain letters, numbers, and
underscores.

For remote Rerun visualization, append:

```bash
rerun_connect:=rerun+http://VIEWER_HOST:9876/proxy \
rerun_recording_id:=my_experiment \
rerun_entity_prefix:=world/robots/robot0
```

Use the same `rerun_recording_id` and a distinct `rerun_entity_prefix` on every
robot to merge their streams into one recording.

## Three-robot EuRoC experiment

The launch below assigns MH01, MH02, and MH03 to `robot0`, `robot1`, and
`robot2`, respectively. Each bag player waits for an acknowledgement from its
DPVO process, so inference load cannot silently drop input images.

```bash
ros2 launch dpvo_multi_robot three_robot_euroc.launch.py \
  bag_root:=/data3/mikexyl/datasets/euroc \
  network:=$DPVO_ROOT/dpvo.pth \
  config:=$DPVO_ROOT/config/fast.yaml \
  calib:=$DPVO_ROOT/calib/euroc.txt \
  orb_vocab:=/data3/mikexyl/datasets/orb_vocab/ORBvoc.txt \
  rerun_connect:=rerun+http://VIEWER_HOST:9876/proxy \
  rerun_recording_id:=mh_01_02_03_run_1 \
  pgo_output:=/data3/mikexyl/results/mh_01_02_03_centralized.json
```

This launch puts all three live camera/map streams and the centralized solver
in the same Rerun recording. The solver aligns the robot subtrees in `world`,
and logs optimized trajectories, accepted inter-robot constraints, PGO cost,
and residual norm alongside them.

On the Blackwell/Jazzy deployment host, the equivalent reproducible runner is
`deploy/blackwell_ros2/run_three_robot_euroc.sh`. It uses `/data3` for the
EuRoC bags and results and honors `DPVO_RERUN_CONNECT`,
`DPVO_RERUN_RECORDING_ID`, `DPVO_STRIDE`, and `DPVO_MAX_FRAMES` environment
overrides. By default the output tag is also used as the shared recording ID.

## ROS interfaces

| Interface | Default name | Purpose |
|---|---|---|
| `BowVector` topic | `/dpvo_multi_robot/bow` | Compact sparse BoW exchange |
| `GetKeyframe` service | `/dpvo_multi_robot/<robot_id>/get_keyframe` | Detailed data on demand |
| `InterRobotLoopClosure` topic | `/dpvo_multi_robot/loop_closure` | Verified query-to-match Sim(3) |
| `RobotMapTransform` topic | `/dpvo_multi_robot/map_transforms` | Central PGO local-map-to-world Sim(3) |
| `PoseStamped` topic | `dpvo/pose` | Latest local DPVO keyframe pose |
| `Path` topic | `dpvo/path` | Local DPVO keyframe path |
| `Path` topic | `dpvo/global_path` | Central-PGO-aligned path in `world` |

BoW publishers use reliable, transient-local QoS. Every message carries a robot
session ID and a SHA-256 vocabulary ID; self-matches and vocabulary mismatches
are rejected. Lexicographic robot/session ownership prevents symmetric duplicate
keyframe requests and duplicate constraints.

The keyframe service can carry roughly a megabyte of descriptor data. It is
called only after a repeated BoW match, but the selected DDS middleware must
still permit fragmented reliable service messages.

## Verification parameters

The important ROS parameters are:

- `bow_threshold` (default `0.04`)
- `bow_repetitions` (default `3`)
- `teaser_noise_bound` (default `0.10` in DPVO map units)
- `min_inliers` (default `30`)
- `min_inlier_ratio` (default `0.20`)
- `teaser_required` (default `false`)

With `teaser_required=false`, deterministic Umeyama-RANSAC is used only if the
TEASER++ binding cannot be loaded or throws an error. Set it to `true` during
deployment validation if falling back is undesirable.

The published transform maps points from the query robot keyframe's camera
coordinates into the matched robot keyframe's camera coordinates:

```text
p_match = scale * R_query_to_match * p_query + t_query_to_match
```
