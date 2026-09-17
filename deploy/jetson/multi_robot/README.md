# Live DPVO fleet on Jetson

One RealSense D455F or ZED UVC camera, DPVO frontend, and CPU CBS agent per Jetson.
Robot0 schedules distributed CBS rounds and hosts the Viser fleet panel; the laptop
is only a browser client. See [distributed CBS](DISTRIBUTED_CBS.md) for the protocol,
native builds, validation, and scheduler availability limitations. `fleet.yaml`
supports two, three, or four robots. Robot IDs, addresses and JetPack profiles are
configuration, not code. No ROS installation is needed on either Jetson host.

The control services start idle. Each robot's **Start** launches a new process
owning its camera, DPVO, TensorRT encoders and learned loop frontend. **Stop**
terminates that process group and releases its camera/GPU resources. The last map
remains visible. The next Start creates a fresh session; old paths, loop closures
and alignments cannot be reused for it. The lightweight controls remain running. Controller containers use Docker restart
policy `always` so a previously stopped container returns after a host reboot;
this starts only the idle controller, never the camera. Override with
`DPVO_CONTROLLER_RESTART_POLICY` when creating a controller container.

## Offline operation and Wi-Fi reconnection

Today's demo selects direct Cyclone DDS on the `192.168.0.0` LAN, with static
peer addresses and the Zenoh bridges disabled. Wi-Fi profiles autoconnect, but
end-to-end DDS recovery after leaving coverage has not been verified in this
mode. Offline boot requires the configured LAN interface before ROS can start.
See [OFFLINE_NETWORK.md](OFFLINE_NETWORK.md) for these limitations and the
previously tested Zenoh bridge alternative. Do not mix the two transports.

The demo uses 15 minimum inliers for every robot pair (previously 30), with a
0.20 minimum inlier ratio and the other geometric checks unchanged. CBS displays
only completed cycles. See [DEMO_2026-09-17.md](DEMO_2026-09-17.md) for the checkpoint
and rollback instructions.

## Shared local and fleet panels

Run `bash deploy/jetson/multi_robot/run_robot_panel.sh robot0` on the robot
with the current online image. Its local panel uses port 9090 and restarts with
Docker after boot. `DPVO_LOCAL_WEB_PORT` overrides the port. This UI-only container
uses the same robot controller as the fleet panel: Start/Stop and the idle-only
dense-mapping selection are shared, with buttons reflecting the controller heartbeat.
The local panel works without the laptop. Camera and GPU ownership stays in the
single online worker; stopping from either panel releases both.

The separate camera area shows that robot's processed image and patch trails.
Trails follow patches from one source keyframe at a time, retaining it while it
remains in the tracking graph, then selecting the newest available source.
Each panel shows **Merged with …**, **Waiting for alignment — verified overlap
found**, or **Unmerged — waiting for verified overlap** per robot. A merged map
can remain visible when its robot goes offline; heartbeat status independently
shows connectivity. Restarting a robot invalidates only groups that depended on
its previous session. Disconnected merged groups remain visually separated until
a verified bridge joins them. If the coordinator disappears, the panel marks the
last alignment as previously merged rather than claiming it is updating.

Display sizes remain independent per viewer. Stop clears the camera preview;
Start creates a fresh tracking session.

Before enabling this panel, stop and disable the previous standalone DPVO viewer
service, which owns a separate camera worker and also uses port 9090. On robot0,
`dpvo-viewer.service` has been replaced by `dpvo-local-panel-robot0`. Robot1's
old `dpvo-jp62-viewer` is stopped with automatic restart disabled. All three
robots now use `dpvo-local-panel-robotX`. Both online controller and local panel start idle after boot.

## Prepare the fleet on the laptop

Edit `deploy/jetson/multi_robot/fleet.yaml`: keep the coordinator address on robot0
and confirm each Jetson's LAN address. The laptop is only a browser client. Use
DHCP reservations so peer addresses remain stable. The DPVO ROS domain is 73;
The current direct DDS profile binds discovery to the fleet LAN. The optional
Zenoh profile uses loopback DDS and TCP 17473 bridges instead.

```bash
python3 deploy/jetson/multi_robot/fleet.py \
  --output deploy/jetson/multi_robot/generated
```

This needs PyYAML; the repository Pixi Python also works. Copy the **same generated
folder** to every participant. `python3 deploy/jetson/multi_robot/stage_sources.py`
creates `output/dpvo-online-sources.tar.gz` containing the deployable source and
generated configurations, without models or compiled artifacts. Extract it into
a dedicated workspace on each board. Add `robot2` / `robot3` entries, then regenerate
and recreate the fleet containers to expand. CBS optimizes every group of at least two maps connected by verified overlap.
Offline robots and maps without verified connections do not block other groups;
the configured anchor is used when present, otherwise the group uses its first robot.

## Robot images and assets

The robot Dockerfile extends the existing, working single-camera DPVO images:

| Profile | Parent image | Online image | Native color / stride |
| --- | --- | --- | --- |
| `jetpack62` | `dpvo:jetson-jp62-viser` | `dpvo:online-jp62` | YUYV 1280×800, SDK 8 FPS, stride 1 |
| `jetpack72` | `dpvo:jetson-viser` | `dpvo:online-jp72` | BGR8 1280×800, 30 FPS, stride 3 |

Both rectify the full field of view to 384×240. The JP6 camera was previously
observed delivering faster than its advertised 8 FPS profile; measure actual
throughput on deployment. These are starting settings, not a multi-robot FPS claim.

Copy the repository source (including initialized DBoW2 and DPRetrieval/pybind11)
to the dedicated DPVO workspace on each board. Exclude `.pixi`, datasets, results,
build trees and the coordinator `bundle/`. Keep the colleague's workspaces,
containers, host packages, network configuration and other sensors untouched.
Run on the matching board, from the repository root:

```bash
bash deploy/jetson/multi_robot/build.sh jetpack62  # robot1
# On robot0 instead:
# bash deploy/jetson/multi_robot/build.sh jetpack72
```

Images use ROS Jazzy/Python 3.12 inside Ubuntu 24.04 containers, including on the
JP6 Ubuntu 22.04 host. Build guards reject an incompatible parent OS or NumPy major.
NVIDIA's Torch/CUDA/TensorRT remain in the matching parent image. The image adds
DBoW2, DPRetrieval and pinned TEASER++; its private DPRetrieval build uses newer
pybind11 for Python 3.12/NumPy 2. No host upgrades or clock changes are performed.

Place `dpvo.pth` and the existing `ORBvoc.txt` in
`deploy/jetson/multi_robot/models/` on each board, or set `DPVO_MODEL_DIR` to an
absolute directory containing them. Model files are not committed. Then:

```bash
bash deploy/jetson/multi_robot/prepare_robot.sh robot1  # robot0 on the other board
```

This downloads and exercises pinned MegaLoc and XFeat/LighterGlue models once,
then builds 240×384 TensorRT encoders on that board. It does not open the camera.
Do not copy TensorRT engine files between JetPack versions. The source commits
are fixed in `prepare_models.py`; weights/cache are reused read-only at runtime.
Internet access is required for image/model preparation, not normal operation.
Preparation defaults to HTTP downloads after Xet stalled on the third board;
set `DPVO_HF_HUB_DISABLE_XET=0` to opt into Xet transfers.
Preparation may take time and needs free storage for images and model weights.

## Start the idle robot controllers

First stop the **DPVO** single-camera worker via its existing panel so the D455F
is free. Do not stop unrelated containers or sensor services. Then on each board:

```bash
bash deploy/jetson/multi_robot/run_robot.sh robot1  # robot0 on the other board
```

The script creates `dpvo-online-robot1` (or robot0), with `restart: unless-stopped`.
It does not replace existing containers. Only the selected D455F USB device and
its video/hid nodes are passed through; no privileged mode or whole `/dev` mount.
If several D455F cameras are attached, configure `camera_serial` and regenerate.
On hardware that omits USB serial metadata, attach just one D455F for selection.

USB device numbers may change after unplug/replug or reboot. If the old container
cannot reopen its camera, recreate **that online DPVO container** with this script
once the camera is attached. Fully automatic hotplug remapping is not installed.
Restart policies apply only after Docker starts and the mapped devices exist.

Useful overrides: `DPVO_FLEET_DIR`, `DPVO_MODEL_DIR`, `DPVO_OUTPUT_DIR`,
`DPVO_JETSON_IMAGE`, `DPVO_CONTAINER_NAME`, `DPVO_BASE_IMAGE`, `DPVO_BUILD_JOBS`.
Use absolute paths for directory overrides. Default per-robot output directories
keep engines and loop diagnostics separate.

## Coordinator image and web panel

The production fleet panel runs on robot0 at **http://192.168.0.156:9091**.
Local panels remain on each Jetson's port 9090. No laptop service is required.
Build the native ARM64 agent images as described in [DISTRIBUTED_CBS.md](DISTRIBUTED_CBS.md),
then start one agent on each corresponding board:

```bash
bash deploy/jetson/multi_robot/run_cbs_agent.sh robot0  # robot1 / robot2 on those boards
```

On robot0, start the CPU scheduler and fleet panel:

```bash
bash deploy/jetson/multi_robot/run_coordinator.sh
```

These containers restart automatically after boot. The scheduler image is selected
by `coordinator.image` in `fleet.yaml` (or `DPVO_COORDINATOR_IMAGE`). Each agent
performs only its own local optimizer updates and exchanges beliefs with peers.
Robot0 is required to schedule new epochs; connected subsets can merge while
other robots are offline. Start each robot from its folder in the panel.
Move slowly to initialize, then observe overlapping textured scenes so learned
retrieval and geometric verification can connect the maps. Before alignment,
local maps are separated and explicitly labelled **unaligned**. CBS runs at most
one snapshot at a time, on a configurable 10-second cadence after graph changes.

The viewer displays local point clouds and paths transformed by CBS map anchors.
The distributed runtime publishes map transforms; the displayed clouds are not
a globally deformable dense reconstruction. The map
point-size slider is shared across clients. This panel is intended for your LAN;
it has no login or public Internet exposure configured.

Local camera transport keeps only the latest selected frame. The default sends
up to 3,000 map points per robot at 1 Hz; raw video is not streamed to the phone.
DPVO sessions are bounded by 8,192 processed frames or the 512-keyframe buffer.
Capacity or detected tracking failure stops the worker; Start begins a fresh run.
Multi-robot retrieval/verification adds GPU work, so camera stride may need tuning
once measured on both boards.

## Validation and troubleshooting

```bash
pixi run python -m unittest tests.test_online_fleet -v
pixi run verify-multi-robot
# With ROS, built interfaces and Viser on PYTHONPATH:
python -m unittest tests.test_online_ros -v
```

`smoke_test.py` additionally exercises real ROS services, fresh worker processes,
Viser initialization and HTTP serving without a camera or GPU. It is intended for
an isolated local ROS domain, as documented in the script.

Inspect only our logs:

```bash
docker logs --tail 100 dpvo-online-robot1
docker logs --tail 100 dpvo-online-coordinator
```

Check that all generated peer IPs match the active LAN and allow DDS UDP traffic
plus coordinator TCP 9091 through any existing firewall. An unavailable robot is
reported in its panel. A worker exit is reported separately from a disconnected
controller. Do not change the colleague's firewall or network services automatically.

Live overlap and sustained fleet throughput must be checked with the participating
robots and scene. Single-camera performance does not establish fleet performance.

Validated locally on 2026-09-14: 18 existing multi-robot tests; 11 fleet/session
regressions; both ROS packages built; the Jazzy CPU coordinator image built; and
the real ROS + HTTP + worker lifecycle + CBS synthetic integration test passed.
The smoke test used a six-vertex, two-robot graph and never opened a camera.

To repeat the container integration check:

```bash
docker run --rm --init -e ROS_DOMAIN_ID=173 \
  -e ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  dpvo:online-coordinator python deploy/jetson/multi_robot/smoke_test.py
```

## Deployed fleet (2026-09-14)

The fleet uses dedicated `~/workspaces/dpvo_online` workspaces:

| Robot | Address | Camera / platform |
| --- | --- | --- |
| robot0 | 192.168.0.156 | D455F, JP7.2 |
| robot1 | 192.168.0.189 | D455F, JP6.2 |
| robot2 | 192.168.0.225 | ZED 2 left image, R36.4.4 / JP6 |

The fleet and phone panel now use http://192.168.0.156:9091 on robot0.
Robot2's fresh ARM image, learned-model GPU inference, TensorRT encoders and
tracker configuration passed validation. Moving its camera produced live paths
and point clouds at the laptop (29 poses and 1,792 points in one sample). Two
camera-only capture cycles delivered 7.49 FPS; this is the input rate, not a claim
of sustained tracker throughput. Fleet Stop removed its worker and child process;
Start created a new worker PID and session ID. The Docker controller restarts on
boot and starts idle. No host sensor services, BSP packages or clock settings were
changed. Robot0 reconnected during this deployment; robot1 was unreachable, so
real three-robot loop closure and CBS alignment remain unverified.

Deployment fixes: generate ROS YAML without anchors/aliases; pin installed NVIDIA
package versions instead of obsolete local wheel URLs; prefer NVIDIA's bundled
UCX libraries in the robot entrypoint; check TensorRT with the NVIDIA runtime.
DDS now selects only configured fleet LAN addresses using optional
[network interfaces](https://cyclonedds.io/docs/cyclonedds/latest/config/config_file_reference.html),
so USB gadget and Docker bridges cannot become the advertised interface.

## ZED camera / third Jetson

`robot2` is `mikexyl@192.168.0.225` (falconeye, L4T R36.4.4). Its camera
identifies as **ZED 2**, serial **29882942**, rather than first-generation ZED.
The connected USB 2 UVC mode is YUYV, 1344×376 side-by-side stereo at 15 FPS.
`camera_type: zed` uses the **left 672×376 image only**, with factory distortion
correction and the entire view resized to 384×240. Both intrinsic axes are scaled
for this output aspect ratio. This is monocular DPVO; stereo depth and the ZED IMU
are not consumed. No ZED SDK installation or camera firmware changes are needed.

Store the camera-specific factory calibration in the ignored model directory:

```bash
curl -fL 'https://calib.stereolabs.com/?SN=29882942' \
  -o deploy/jetson/multi_robot/models/SN29882942.conf
```

Set `camera_type: zed` and its numeric `camera_serial` on that robot in
`fleet.yaml`. The default ZED settings are 15 FPS, stride 2 (7.5 FPS input),
with a two-second exposure/white-balance warm-up on every Start. Per-robot
`camera_fps` and `camera_stride` overrides are supported. Only the selected UVC
capture node is passed into Docker as `/dev/dpvo_camera`; other cameras and HID
sensors remain outside the container. Factory calibration must match the serial.
Only the VGA stereo UVC mode is currently supported; other negotiated image sizes
fail explicitly instead of silently using incorrect calibration.

On a fresh JetPack 6 board, build the native and runtime parent images first:

```bash
docker build --network host -f deploy/jetson/Dockerfile.jp62 \
  -t dpvo:jetson-jp62-native .
docker build --network host -f deploy/jetson/Dockerfile.jp62.runtime \
  -t dpvo:jetson-jp62-viser .
docker build --network host -f deploy/jetson/Dockerfile.jp62.dla \
  -t dpvo:jetson-jp62-viser-dla3644 .
DPVO_BASE_IMAGE=dpvo:jetson-jp62-viser-dla3644 \
  bash deploy/jetson/multi_robot/build.sh jetpack62
bash deploy/jetson/multi_robot/prepare_robot.sh robot2
bash deploy/jetson/multi_robot/run_robot.sh robot2
```

Host networking avoids requiring a Docker bridge/iptables `raw` table on this
Jetson kernel. Python packages, ROS, CUDA extensions and models stay in the
container or dedicated `~/workspaces/dpvo_online` directory. Docker starts the idle
controller on boot; the camera opens only after Start. Recreate the controller
with `run_robot.sh` if Linux renumbers the video device after changing USB devices.

Camera-only validation on this board passed two open/capture/release cycles at
7.49 FPS with stride 2, using the actual rectification code. Full tracker validation
is separate from that camera input rate.

This R36.4.4 CTI board omits `nvidia-l4t-dla-compiler`, although TensorRT imports
require `libnvdla_compiler.so` even when inference uses the GPU. The optional
`Dockerfile.jp62.dla` layer extracts only that library from NVIDIA's matching
36.4.4 package, checks its SHA-256, and registers it inside the image. It does not
install or upgrade host BSP packages. Skip this layer on hosts already providing
the library; do not substitute a different BSP version without matching its
package and checksum. `robot2.base_image` records the selected parent, and the
build command passes it explicitly through `DPVO_BASE_IMAGE`.

Additional checks: 10 camera/fleet tests and 18 existing multi-robot tests passed;
the updated coordinator ROS/service/session/CBS integration smoke test passed.
The browser default view now frames all configured robot origins.

## Optional DA3 dense mapping

See [DENSE_MAPPING.md](DENSE_MAPPING.md) for TensorRT conversion, default-off fleet controls, geometry, and per-Jetson deployment.

## Camera previews and robot health

The fleet panel groups all camera previews in a separate, resizable 480-pixel camera panel (left on desktop), with robot buttons and status in the right sidebar.

It displays each updated robot's processed RGB image with colored
DPVO patch boxes and short reprojection trails. During initialization it labels
newly extracted patches as candidates (no tracking trails). The overlay is drawn
on the corresponding processed image, not on a newer raw-camera frame. Stable
input-frame/patch IDs preserve trail colors across keyframe culling. Preview
history resets with the disposable tracker process. Patch identities are retained and observed on every processed frame; only JPEG publication is limited to 2 Hz. Trails use a dark outline and thicker colored strokes for visibility.

`online_preview_fps` controls compressed JPEG previews (default 2 Hz in fleet
worker files; zero disables previews and preview telemetry). Images use
`/robotX/dpvo/preview/compressed`, session-tagged headers and best-effort depth-one
QoS. The viewer rejects old sessions, out-of-order frames and malformed images.
Stopped robots show a blank camera view; a stale preview is explicitly labeled.

Controller heartbeats run at 2 Hz even when tracking is stopped. The panel shows
receive age, camera activity, tracker activity, initialization/tracking state,
processing FPS, keyframe count and visible patch count. Green indicates a recent
heartbeat; red means no heartbeat for over three seconds. Camera/tracker activity
is measured independently from camera-info and processed-frame acknowledgements,
so a live control process cannot mask a stalled worker. Legacy robots still show
basic online/idle status without requiring an update.

The shared panels and preview/telemetry update are deployed to all three robots.
Robot0 uses `dpvo:online-jp72-shared`; robot1 and robot2 each use their locally
built `dpvo:online-jp62-shared`, preserving their board-specific parent images.
On 2026-09-16, live images and shared controls were verified on all three robots.
Robot2's ZED reappeared after a reboot as `/dev/video0` (USB 2, 480 Mbps).
Its controller was recreated to map that node to `/dev/dpvo_camera`; local and
fleet preview reception passed, followed by a clean stop. The previous
camera-less controller is retained stopped with automatic restart disabled.
Dense mapping remains off by default; robot1/robot2
require their own validated DA3 TensorRT engines before enabling it.

The current robot0 scheduler/viewer image is `dpvo:cbs-agent-jp72`; each board
runs its own `dpvo-cbs-agent-robotX` container. Alignment state is
published at 1 Hz on `/dpvo_multi_robot/cbs/alignment_status` with transient-local
QoS so newly opened local panels recover the current group transforms.
