# Multi-robot DPVO over ROS 2

This frontend runs one DPVO instance per robot and performs distributed learned
loop-closure detection with a bandwidth-aware two-stage exchange:

1. A keyframe becomes stable after it leaves DPVO's removal window.
2. MegaLoc extracts and publishes one L2-normalized global descriptor.
3. Peers use cosine similarity to retrieve only the top-1 local keyframe. The
   configured consecutive-match check and loop NMS are applied.
4. The deterministic owner of the candidate calls the matched robot's
   `GetKeyframe` service. No image, point, or descriptor data is sent before
   this step.
5. The service returns the requested keyframe's locally triangulated 3D points,
   XFeat keypoints/descriptors, pose, and image dimensions.
6. XFeat's officially trained LighterGlue (the XFeat-compatible LightGlue
   variant) forms putative cross-robot correspondences. TEASER++ estimates a
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

- `dpvo_multi_robot_interfaces`: typed global descriptor, legacy BoW, detailed
  keyframe, service, and inter-robot Sim(3) definitions.
- `dpvo_multi_robot`: live image/CameraInfo DPVO node and ROS 2 transport.

## Build

ROS 2 Humble and DPVO's Pixi environment must be available. All robots must use
the same learned-model IDs and the same ORB vocabulary (the latter remains
necessary for classic intra-robot loop closure).

```bash
cd /path/to/DPVO
pixi run verify-multi-robot

source /opt/ros/humble/setup.bash
colcon build --base-paths ros2 --symlink-install
source install/setup.bash
export DPVO_ROOT=$PWD
```

On the Blackwell workstation, cache the pinned official MegaLoc and XFeat model
repositories and weights under `/data3` before the first run:

```bash
./deploy/blackwell_ros2/prepare_learned_loop_frontend.sh
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

## Staged tracking, verification, and DPGO

Long multi-robot experiments can be separated so DPVO is run only once. This
also keeps local tracking memory isolated from the learned geometric-verification
models:

1. `stage1` runs the three DPVO trackers with classic intra-robot loop closure,
   but disables all inter-robot retrieval, keyframe exchange, verification, and
   global optimization. Each robot saves its final original-scale poses, DPVO
   depth state, intrinsics, timestamps, and exact preprocessed input images in a
   checksummed tracking artifact.
2. `stage2` reloads those immutable artifacts, shares/compares sparse DBoW2
   vectors, and requests detailed keyframe data only for the top-1 match after
   repetition and NMS. DISK/LightGlue reconstructs correspondences from saved
   images and DPVO depth, then TEASER++ verifies the Sim(3). It writes a raw
   keyframe graph containing odometry and accepted inter-robot factors, without
   running any optimizer.
3. `stage3` rejects graphs containing optimized poses or disconnected robots,
   then passes one copied input graph to CBS, global-pose centralized PGO, and
   explicit-anchor centralized PGO in the same executable. Thus no centralized
   result can initialize or otherwise influence CBS.

For covariance-transport studies, the offline Stage 3 runner accepts
`--sim3-covariance-transport bernoulli|adjoint|none`. The `none` baseline keeps
the covariance matrix unchanged across mean-chart changes. It also accepts
`--[no-]hellinger-quadratic-term`; the default includes the full
mean-covariance term in the Hellinger step-size approximation. A positive
`--target-hellinger` selects the fixed-target path, and `--d-reset` above one
effectively disables reset because Hellinger distance is bounded by one. Legacy raw graph
exports can be admitted only with `--allow-legacy-unoptimized-input`; this
explicit gate still rejects optimized vertices and disconnected robot graphs.

The Blackwell iPhone runner exposes the stages independently:

```bash
DPVO_OUTPUT_TAG=iphone_302_staged_run1 \
deploy/blackwell_ros2/run_three_robot_iphone_staged.sh stage1

DPVO_OUTPUT_TAG=iphone_302_staged_run1 \
DPVO_BOW_THRESHOLD=0.01 \
deploy/blackwell_ros2/run_three_robot_iphone_staged.sh stage2

DPVO_OUTPUT_TAG=iphone_302_staged_run1 \
DPVO_CBS_STAGE_MODE=random \
DPVO_CBS_TARGET_HELLINGER=0.1 \
deploy/blackwell_ros2/run_three_robot_iphone_staged.sh stage3
```

Use `all` to run the three commands in order. Stage 2 caches BoW vectors and
detailed payloads, so its thresholds can be tuned without repeating tracking;
stage 3 can likewise be rerun with different CBS or robust-loss settings. The
result directory contains `tracking/`, `geometric_verification/`, and `dpgo/`.
The first two directories include combined and per-robot lossless JSON/g2o
graphs. `offline_dpgo_provenance.json` records input checksums and the exact CBS
command.

## Newer College Quad staged experiment

`dpvo_newer_college_player` reads the ROS1
`/alphasense_driver_ros/cam0/compressed` stream, preserves each camera-header
timestamp, decodes grayscale JPEG, and rectifies the Collection 1 Kalibr
pinhole/equidistant model before publishing distortion-free `mono8` plus
updated `CameraInfo`. Playback retains the one-frame DPVO acknowledgement
barrier.

The reproducible runner keeps robot tracking artifacts shared while isolating
all scenario-dependent products:

```bash
deploy/blackwell_ros2/run_newer_college_quad_staged.sh all two

# After replacing the incomplete Quad-Medium bag with a complete camera bag:
deploy/blackwell_ros2/run_newer_college_quad_staged.sh all three
```

The default input root is
`/data3/mikexyl/datasets/newer_college/collection1`; unique Quad Easy/Hard (and
Medium for `three`) bags, ground-truth CSVs, and a Kalibr camchain are resolved
below it. Set `DPVO_NCD_EASY_BAG`, `DPVO_NCD_HARD_BAG`,
`DPVO_NCD_MEDIUM_BAG`, their corresponding `*_GROUNDTRUTH` variables, and
`DPVO_NCD_CALIB` when a layout is ambiguous. The fixed output root is
`/data3/mikexyl/results/dpvo_multi_robot/newer_college_quad_staged_20260902`
unless `DPVO_NCD_RESULT_ROOT` is set.

The Blackwell deployment machine referred to as host `148` is
`192.168.0.148` (for example, `ssh 192.168.0.148`).

`stage1` performs a 300-frame Easy preflight and then tracks missing robots
sequentially, saving a per-robot RRD and immutable artifact. `stage2` and
`stage3` write beneath `geometric_verification/{two,three}` and
`dpgo/{two,three}`. `evaluate` converts Base ground truth to cam0 using the
HALO Base-to-Alphasense-IMU joint and Collection 1 `T_cam_imu`, then runs evo
with Sim(3) alignment and a 55 ms association tolerance. `plot` writes PNG/PDF
trajectory/loop and sparse-map figures. The experiment manifest records source
checksums and fixed parameters; gate JSON files preserve frame-count, graph,
solver-provenance, scale, ATE-ratio, and required-output checks.

## GrAco Aerial 5--8 staged experiment

Host `148` keeps the complete GrAco copy under `/data3/graco` (the older
`/mnt/data/graco` directory is missing Aerial 6). The four-session experiment
maps Aerial 5, 6, 7, and 8 to `robot0` through `robot3`, uses the left 20 Hz
`mono8` camera, and rectifies its pinhole/radial-tangential `CameraInfo` at
1600x1100. DPVO then applies `image_scale=0.5`, producing 800x550 pixels before
the required multiple-of-16 crop to 800x544. Run the fixed pipeline with:

```bash
deploy/blackwell_ros2/run_graco_aerial_5_8_staged.sh all
```

The default output root is
`/data3/mikexyl/results/dpvo_multi_robot/graco_aerial_5_8_staged_20260902`.
Stage 1 begins with a 300-frame stride-4 preflight, feeding the 20 Hz source to
DPVO at 5 images/s, before tracking all four bags sequentially. The gate records
DPVO's separately retained keyframe rate without confusing it with the input
rate. Stage 2 uses CUDA MPS,
saves one-second GPU telemetry, and never retries or tunes a failed
verification. Evaluation transforms RTK `T_Base_Imu` ground truth into cam0
with `T_Imu_cam0`, applies Sim(3) alignment with a 55 ms association tolerance,
and reports explicit-anchor centralized PGO as the baseline. Tracking and DPGO
RRDs are saved to disk without a live Rerun stream.

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
  pgo_output:=/data3/mikexyl/results/mh_01_02_03_centralized.json \
  pose_graph_output:=/data3/mikexyl/results/mh_01_02_03_pose_graph
```

This launch puts all three live camera/map streams and the centralized solver
in the same Rerun recording. The solver aligns the robot subtrees in `world`,
and logs optimized trajectories, accepted inter-robot constraints, PGO cost,
and residual norm alongside them.

## Two-robot TUM RGB-D experiment

The Freiburg-1 `desk` and `desk2` sequences are two recordings of the same
four-desk scene.  The launch below treats them as independent monocular robots,
using only their RGB streams:

```bash
ros2 launch dpvo_multi_robot two_robot_tum.launch.py \
  dataset_root:=/data3/mikexyl/datasets/tum_rgbd \
  network:=$DPVO_ROOT/dpvo.pth \
  config:=$DPVO_ROOT/config/default.yaml \
  orb_vocab:=/data3/mikexyl/datasets/orb_vocab/ORBvoc.txt \
  rerun_connect:=rerun+http://VIEWER_HOST:9876/proxy \
  rerun_recording_id:=tum_fr1_desk_desk2 \
  pgo_output:=/data3/mikexyl/results/tum_fr1_desk_desk2_centralized.json \
  pose_graph_output:=/data3/mikexyl/results/tum_fr1_desk_desk2_pose_graph \
  cbs_output_dir:=/data3/mikexyl/results/tum_fr1_desk_desk2_cbs
```

`dpvo_tum_player` reproduces the preprocessing used by DPVO's TUM evaluation:
Freiburg-1 calibration, image undistortion, then an 8-pixel vertical and
16-pixel horizontal border crop.  It preserves source timestamps on every
keyframe vertex in the JSON pose-graph exports for ground-truth association.
The deployment equivalent is
`deploy/blackwell_ros2/run_two_robot_tum.sh`.

## Three-robot KITTI 00 experiment

`three_robot_kitti.launch.py` divides KITTI odometry sequence 00 into three
disjoint, contiguous half-open raw-frame windows: `[0,1450)`, `[1450,3000)`,
and `[3000,4541)`. The windows cover the full sequence exactly once. Ground
truth was used to select windows with pairwise revisitation, but is never loaded
by the ROS 2 player, DPVO, BoW retrieval, TEASER++, centralized PGO, or CBS.

```bash
DPVO_KITTI_ROOT=/data1/mikexyl/datasets/kitti_odometry/dataset \
DPVO_RERUN_CONNECT=rerun+http://VIEWER_HOST:9876/proxy \
DPVO_OUTPUT_TAG=kitti00_three_robot_full_20260827_run1 \
deploy/blackwell_ros2/run_three_robot_kitti.sh
```

The player uses the left grayscale camera (`image_0`, `P0`) and original KITTI
timestamps, with acknowledgement backpressure on every input. The default
stride is two, matching DPVO's official KITTI evaluator. Split-selection
provenance is recorded in
`deploy/blackwell_ros2/kitti00_three_robot_split.json`; post-run evo export and
evaluation are provided by `deploy/blackwell_ros2/evaluate_kitti_evo.sh`.

On the Blackwell/Jazzy deployment host, the equivalent reproducible runner is
`deploy/blackwell_ros2/run_three_robot_euroc.sh`. It uses `/data3` for the
EuRoC bags and results and honors `DPVO_RERUN_CONNECT`,
`DPVO_RERUN_RECORDING_ID`, `DPVO_STRIDE`, and `DPVO_MAX_FRAMES` environment
overrides. By default the output tag is also used as the shared recording ID.

## CBS distributed Sim(3) PGO

CBS is pinned as the `cbs/` Git submodule on the
`codex/joint-anchor-local-scale` branch. Initialize it and build the DPVO Sim(3)
executable with:

```bash
git submodule update --init --recursive cbs
./deploy/blackwell_ros2/build_cbs.sh
```

The build script defaults to the dependency prefix
`/home/mikexyl/workspaces/sb_slam_ros2/install`; override it with
`DPVO_CBS_DEPENDENCY_PREFIX`. It builds the scale-aware CBS example and runs
the CBS Sim(3) tests. Legacy GSL-dependent CBS examples are disabled because
they are not part of this integration.

The three-robot launch runs CBS alongside the existing centralized map-level
solver by default. Once all bag players finish and the loop-verification queue
has settled, the CBS node snapshots the original-scale keyframe graph, invokes
`cbs_dpvo_sim3_offline`, and publishes CBS plus two full-keyframe centralized
trajectories. All three paths consume exactly the same visual-odometry and
verified inter-robot Sim(3) factors. One centralized path directly optimizes
global keyframe poses; the other jointly optimizes robot-local poses and
explicit robot-anchor variables. These are the like-for-like comparisons. The
pre-existing online centralized solver optimizes one map transform per robot
and is also reported, but its objective value is not directly comparable.

Important launch arguments are:

- `enable_cbs` (default `true`)
- `cbs_executable`
- `cbs_output_dir`
- `cbs_iterations` (default `200`)
- `cbs_stage_mode` (`alternating`, `random`, or `fixed`; default `alternating`)
- `cbs_pose_warmup_iterations` (default `0`)
- `cbs_pose_block_iterations` (default `20`)
- `cbs_anchor_block_iterations` (default `20`)
- `cbs_target_hellinger` (fixed contraction target; default `0.1`)
- `cbs_settle_seconds` (default `30.0`)
- `cbs_run_centralized_baseline` (default `true`)
- `cbs_run_explicit_anchor_centralized_baseline` (default `true`)

The deployment runner exposes the corresponding `DPVO_ENABLE_CBS`,
`DPVO_CBS_EXECUTABLE`, `DPVO_CBS_OUTPUT_DIR`, `DPVO_CBS_ITERATIONS`,
`DPVO_CBS_STAGE_MODE`, `DPVO_CBS_POSE_WARMUP_ITERATIONS`,
`DPVO_CBS_POSE_BLOCK_ITERATIONS`, `DPVO_CBS_ANCHOR_BLOCK_ITERATIONS`,
`DPVO_CBS_SETTLE_SECONDS`, and
`DPVO_CBS_RUN_CENTRALIZED_BASELINE` environment variables. The explicit-anchor
path is controlled by `DPVO_CBS_RUN_EXPLICIT_ANCHOR_CENTRALIZED_BASELINE`. It also supplies
the CBS/GTSAM runtime library path. The CBS node joins the same Rerun recording
as the three robots and the online centralized solver.

CBS writes `input_keyframes_unoptimized.json/.g2o`, one distributed `cbs.csv`
trajectory in robot0's frame, `centralized.csv`,
`centralized_explicit_anchors.csv`, optimized explicit
anchor variables, the centralized-parameterization comparison, `summary.csv`,
`cbs.log`, and `comparison.json` under `cbs_output_dir`. The comparison JSON
contains graph sizes, costs and residual metrics, both full-graph centralized
map transforms, CBS transforms, and transform deltas from the online map-level
centralized result. CBS can also be rerun after new loop
constraints arrive with:

```bash
ros2 service call /dpvo_multi_robot/cbs/run std_srvs/srv/Trigger '{}'
```

## Offline pose-graph export

The centralized node writes two graph levels once during clean shutdown, so
graph serialization does not compete with live odometry and loop verification.
Set `pose_graph_export_period` (or `DPVO_POSE_GRAPH_EXPORT_PERIOD` in the
Blackwell runner) to a positive number of seconds only when periodic recovery
snapshots are needed. Each level is written atomically in both lossless JSON
and g2o Sim(3) Expmap form:

```text
mh_01_02_03_pose_graph_map.json
mh_01_02_03_pose_graph_map.g2o
mh_01_02_03_pose_graph_keyframes.json
mh_01_02_03_pose_graph_keyframes.g2o
mh_01_02_03_pose_graph_keyframes_unoptimized.json
mh_01_02_03_pose_graph_keyframes_unoptimized.g2o
mh_01_02_03_pose_graph_robot0.json
mh_01_02_03_pose_graph_robot0.g2o
mh_01_02_03_pose_graph_robot1.json
mh_01_02_03_pose_graph_robot1.g2o
mh_01_02_03_pose_graph_robot2.json
mh_01_02_03_pose_graph_robot2.g2o
```

The map graph is the exact problem solved by the live centralized optimizer:
one 7-DoF Sim(3) vertex per robot session and one edge per verified inter-robot
loop. Its JSON vertices contain both the deterministic initial estimate and the
live optimized estimate.

The keyframe graph contains every pose received in each robot's path, adjacent
visual-odometry edges, and the verified inter-robot keyframe edges. Its initial
vertices are placed using the current centralized map alignment. Local paths
already include DPVO's local BA and classic loop-closure corrections; the
exported adjacent edges preserve that corrected local trajectory. The default
visual-odometry information weight is `100.0` and can be changed with
`pose_graph_odometry_weight` or `DPVO_POSE_GRAPH_ODOMETRY_WEIGHT` in the
Blackwell runner.

The `keyframes_unoptimized` graph contains the same odometry and inter-robot
edges, but every vertex remains in its independent robot-local map with the
original DPVO scale (vertex scale `1.0`). The per-robot files are extracted
from this raw graph, contain only that robot's trajectory and odometry edges,
and fix keyframe zero. They do not apply centralized alignment or offline PGO.

Every JSON edge records the complete measurement, 7-vector information
diagonal, endpoint robot/session/keyframe identities, BoW score, TEASER++
inliers and ratio, verification method, raw keyframe poses, and raw
query-to-match transform. The convention is included in the file itself:

```text
T_world_frame: p_world = scale * R * p_frame + t
edge source->target: p_target = scale * R * p_source + t
```

The bundled sparse SciPy optimizer makes it easy to try losses, gates, and edge
subsets without ROS:

```bash
pixi run python -m dpvo.loop_closure.pose_graph \
  /data3/mikexyl/results/mh_01_02_03_pose_graph_map.json \
  --loss cauchy \
  --f-scale 0.5 \
  --min-information 0.5 \
  --output /data3/mikexyl/results/mh_01_02_03_map_cauchy.json
```

The command writes optimized estimates to JSON and also emits a sibling g2o
file initialized with that solution. Use repeated `--edge-type` arguments to
select `visual_odometry`, `inter_robot_loop_closure`, or
`inter_robot_map_alignment` edges. If `pose_graph_output` is omitted while
`pgo_output` is set, the graph base is derived from the PGO result filename.

## ROS interfaces

| Interface | Default name | Purpose |
|---|---|---|
| `GlobalDescriptor` topic | `/dpvo_multi_robot/global_descriptor` | MegaLoc descriptor exchange |
| `BowVector` topic | `/dpvo_multi_robot/bow` | Compact sparse BoW exchange |
| `GetKeyframe` service | `/dpvo_multi_robot/<robot_id>/get_keyframe` | Detailed data on demand |
| `InterRobotLoopClosure` topic | `/dpvo_multi_robot/loop_closure` | Verified query-to-match Sim(3) |
| `RobotMapTransform` topic | `/dpvo_multi_robot/map_transforms` | Central PGO local-map-to-world Sim(3) |
| `RobotMapTransform` topic | `/dpvo_multi_robot/cbs/map_transforms` | CBS local-map-to-world Sim(3) |
| `String` topic | `/dpvo_multi_robot/cbs/comparison` | CBS versus centralized JSON summary |
| `PoseStamped` topic | `dpvo/pose` | Latest local DPVO keyframe pose |
| `Path` topic | `dpvo/path` | Local DPVO keyframe path |
| `Path` topic | `dpvo/global_path` | Central-PGO-aligned path in `world` |
| `Path` topic | `dpvo/cbs_global_path` | CBS-optimized keyframe path in `world` |
| `Path` topic | `dpvo/cbs_centralized_global_path` | Like-for-like centralized keyframe path |
| `Path` topic | `dpvo/cbs_explicit_anchor_centralized_global_path` | Explicit-anchor centralized keyframe path |

Global-descriptor and BoW publishers use reliable, transient-local QoS. Every
MegaLoc message carries a robot session ID, model ID, and descriptor dimension;
self-matches and incompatible models are rejected. Lexicographic robot/session
ownership prevents symmetric duplicate keyframe requests and constraints.

The keyframe service can carry roughly a megabyte of descriptor data. It is
called only after a repeated top-1 retrieval match, but the selected DDS middleware must
still permit fragmented reliable service messages.

## Verification parameters

The important ROS parameters are:

- `retrieval_backend` (`megaloc` by default, or legacy `dbow2`)
- `megaloc_threshold` (default `0.20` cosine similarity)
- `local_feature_backend` (`xfeat` by default, or legacy `disk`)
- `xfeat_top_k` (default `2048`)
- `xfeat_detection_threshold` (default `0.05`)
- `lightglue_min_confidence` (default `0.10`)
- `bow_repetitions` (default `3`)
- `teaser_noise_bound` (default `0.10` in DPVO map units)
- `min_inliers` (default `30`)
- `min_inlier_ratio` (default `0.20`)
- `teaser_required` (default `true` in the ROS node)

With `teaser_required=false`, deterministic Umeyama-RANSAC is used only if the
TEASER++ binding cannot be loaded or throws an error. The learned ROS pipeline
defaults to `true`, so a missing TEASER++ binding rejects the candidate instead
of silently changing the verifier.

The published transform maps points from the query robot keyframe's camera
coordinates into the matched robot keyframe's camera coordinates:

```text
p_match = scale * R_query_to_match * p_query + t_query_to_match
```
