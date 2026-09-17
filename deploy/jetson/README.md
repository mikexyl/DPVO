# DPVO on Jetson Orin NX

Target stack: Orin NX 16 GB, JetPack 7.2 / L4T R39.2, Ubuntu 24.04,
CUDA 13.2. The NVIDIA `26.05-py3` ARM64 PyTorch image is the base;
DPVO and torch-scatter are compiled for `sm_87` on the board.
This image runs headless monocular DPVO. ROS, CBS, retrieval, and viewers
are separate deployment components.

## Measured results on the connected Orin NX

EuRoC MH_01 camera 0, 20 Hz source, no frame skipping, checkpoint `dpvo.pth`.
Warmup excludes the first 30 frames and all pre-initialization frames.

| Configuration | Frames | Pipeline FPS | p95 cycle |
| --- | ---: | ---: | ---: |
| Default PyTorch, 752x480, 96 patches, serial input | 300 | 2.74 | — |
| Default + TensorRT encoders, serial input | 300 | 2.90 | — |
| Fast preset, PyTorch, max clocks, prefetch | 300 | 20.32 | 50.13 ms |
| Fast preset, TensorRT, max clocks, prefetch | 300 | 23.21 | 44.04 ms |
| Fast preset, TensorRT, max clocks, prefetch | 1000 | 22.58 | 44.93 ms |

The fast preset uses 368x240 input and 32 patches; these gains combine graph-budget,
resolution, input-pipeline, clock, and encoder changes. TensorRT alone does not make
the default configuration real time. The 1000-frame run had zero 50 ms consumer-cycle
budget misses in 970 measured frames, no bundle-adjustment errors, and finite poses.
Its Sim(3)-aligned camera-position ATE RMSE was 0.00632 m over 978 matched frames
and an 8.77 m ground-truth path. This is a recorded monocular VO test, not validation
on live cameras, ROS, distributed CBS, or diverse long sequences.

The built image is `dpvo:jetson-jp7`; on the tested board the checkout is
`~/workspaces/dpvo_jetson`, with engines/reports under `results/jetson/`.
The offline Dockerfile was used because the board's USB network had no working
internet access. The base PyTorch emits an Orin architecture warning; the encoder,
correlation, bundle-adjustment, and Lie-group paths exercised here ran successfully.

From the repository root on the Jetson:

```bash
bash deploy/jetson/build.sh
DPVO_DATA_DIR=/path/to/data DPVO_WEIGHTS=/path/to/dpvo.pth \
  bash deploy/jetson/run.sh python deploy/jetson/benchmark.py \
  --images /data/euroc_mh01 --calib calib/euroc.txt \
  --network /models/dpvo.pth --frames 300 --warmup 30 --target-fps 20 \
  --output /output/pytorch.json
```

Add `--profile` in a separate run for CUDA-event timing of both encoders
and the update network. The benchmark synchronizes each frame, checks
initialization, fails on bundle-adjustment errors, and checks the trajectory
for nonfinite values. It reports warmup-excluded processing and end-to-end
throughput, median/p95 latency, deadline misses, and finalization time.
Image decoding, undistortion, and host-to-device transfer count toward
end-to-end timing. Model construction and finalization are outside steady-state
timing, as are frames before tracking initializes. Trajectory timestamps are input frame indices; this is a timing tool,
not a ground-truth trajectory evaluator. Finite poses alone do not validate
odometry accuracy.

Use `--scale 0.5` to evaluate a smaller input, or `--opts PATCHES_PER_FRAME 48`
to evaluate a different tracking budget. Such changes require separate accuracy
evaluation. Keep the input frame rate and `--stride` explicit when interpreting
real-time throughput.

## TensorRT and reduced tracking budget

Export and build FP16 engines on the target Jetson, then run the same DPVO
pipeline with its feature and context encoders replaced:

```bash
bash deploy/jetson/run.sh python deploy/jetson/tensorrt_encoder.py \
  --network /models/dpvo.pth --output /output/engines-480x752 \
  --validation-images /data/euroc_mh01 --calib calib/euroc.txt
bash deploy/jetson/run.sh python deploy/jetson/benchmark.py \
  --images /data/euroc_mh01 --calib calib/euroc.txt --network /models/dpvo.pth \
  --trt-encoders /output/engines-480x752 --output /output/tensorrt-default.json
```

Engines have fixed batch/image dimensions and are checked against the checkpoint
SHA-256 and TensorRT version. Inference uses a nondefault CUDA stream with explicit
dependencies on the caller's PyTorch stream. Patch sampling, correlation, dynamic
graph aggregation, recurrent updates, and bundle adjustment retain their existing
implementation. Engine export compares outputs against PyTorch mixed precision
on ten real frames and one synthetic input; a relative RMS discrepancy above 2%
fails the build. This check complements, but does not replace, trajectory testing.

The experimental `config/jetson_fast.yaml` uses 32 patches, an 8-keyframe removal
window, a 5-keyframe optimization window, and a patch lifetime of 4. Use half-size
images and a halved keyframe threshold with it. `KEYFRAME_INDEX: 2` keeps the
keyframe comparisons inside the shorter patch lifetime:

```bash
bash deploy/jetson/run.sh python deploy/jetson/tensorrt_encoder.py \
  --network /models/dpvo.pth --height 240 --width 368 \
  --output /output/engines-240x368 \
  --validation-images /data/euroc_mh01 --calib calib/euroc.txt
bash deploy/jetson/run.sh python deploy/jetson/benchmark.py \
  --images /data/euroc_mh01 --calib calib/euroc.txt --network /models/dpvo.pth \
  --config config/jetson_fast.yaml --scale 0.5 --remap --prefetch 8 \
  --trt-encoders /output/engines-240x368 --output /output/tensorrt-fast.json
```

`--remap` caches undistortion maps rather than rebuilding them each frame.
`--prefetch 8` overlaps decoding and undistortion on one reader thread with GPU
tracking, with at most eight prepared images in flight. No frames are dropped.
With prefetch, reported end-to-end times are consumer-loop cycle times including
queue waits and GPU transfers; they are not camera-capture-to-pose latency.
The 752x480 EuRoC image becomes 376x240 after resizing, then is cropped to 368x240
to satisfy DPVO's multiple-of-16 requirement. This preset changes the accuracy
budget; validate on your camera and motion before using it in an application.

For the measured performance configuration, run `sudo jetson_clocks` on the board
before inference to fix CPU/GPU/EMC clocks at the current power mode's maximum.
To preserve the prior settings, first use
`sudo jetson_clocks --store /tmp/dpvo-clocks.conf`, and afterward use
`sudo jetson_clocks --restore /tmp/dpvo-clocks.conf`. The tested board was already
in `MAXN_SUPER`; the deployment scripts do not change the power mode or clocks.
On this R39.2 image, NVIDIA's restore utility emits errors for an unavailable GPU
persistence-state file; the saved frequency and idle-state values were checked
separately after the tests. Inspect `sudo jetson_clocks --show` after restoration.

For a EuRoC ground-truth check, run `compare_euroc.py` on the workstation with the
full sequence available. It interpolates camera-center ground truth using `T_BS`,
performs Sim(3) alignment for monocular scale ambiguity, and reports position ATE:

```bash
python deploy/jetson/compare_euroc.py --sequence /data/euroc/MH_01_easy \
  --output results/jetson/accuracy.json results/jetson/*.tum
```

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DPVO_JETSON_IMAGE` | `dpvo:jetson-jp7` | Build/run image tag |
| `DPVO_BASE_IMAGE` | `nvcr.io/nvidia/pytorch:26.05-py3` | Compatible ARM64 base |
| `DPVO_BUILD_JOBS` | `2` | Limit native compiler memory usage |
| `DPVO_OFFLINE` | `0` | Build with staged dependencies when `1` |
| `DPVO_DATA_DIR` | Repository `data/` | Read-only dataset mount |
| `DPVO_WEIGHTS` | Repository `dpvo.pth` | Read-only checkpoint mount |
| `DPVO_OUTPUT_DIR` | Repository `results/jetson/` | Writable result mount |

## Offline build

The USB-connected board may have no internet route. On an internet-connected
Linux workstation with pip and Eigen 3 headers, prepare `deps/` outside version
control and transfer it into the Jetson build context:

```bash
mkdir -p /tmp/dpvo-jetson-deps/{include,wheels}
cp -a /usr/include/eigen3 /tmp/dpvo-jetson-deps/include/
python -m pip download --dest /tmp/dpvo-jetson-deps/wheels \
  --platform manylinux2014_aarch64 --platform manylinux_2_28_aarch64 \
  --python-version 312 --implementation cp --abi cp312 --only-binary=:all: \
  yacs evo plyfile opencv-python-headless
python -m pip download --no-deps -d /tmp/dpvo-jetson-deps/wheels pypose==0.9.5
curl -fL https://files.pythonhosted.org/packages/f5/ab/2a44ecac0f891dd0d765fc59ac8d277c6283a31907626560e72685df2ed6/torch_scatter-2.1.2.tar.gz \
  -o /tmp/dpvo-jetson-deps/wheels/torch_scatter-2.1.2.tar.gz
rsync -az /tmp/dpvo-jetson-deps/ USER@JETSON:PATH/TO/DPVO/deps/
```

On the board, use `DPVO_OFFLINE=1 bash deploy/jetson/build.sh`.
The base image must already be cached. Keep weights, wheels, engines, recordings,
and benchmark outputs out of Git. Dockerfile-specific ignore files exclude
datasets, existing host binaries, and unrelated submodules from builds.

NVIDIA references: [JetPack 7.2 stack](https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2),
[TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html).

## Live RealSense and browser Rerun

The live image reuses `dpvo/rerun_viewer.py` with **Rerun 0.37.2**, the latest
stable release checked when building it, and RealSense SDK 2.58.4.10922.
Prepare ARM64 dependencies on the internet-connected workstation:

```bash
python -m pip download --dest /tmp/dpvo-jetson-deps/live-wheels \
  --platform manylinux2014_aarch64 --platform manylinux_2_28_aarch64 \
  --python-version 312 --implementation cp --abi cp312 --only-binary=:all: \
  rerun-sdk==0.37.2 pyrealsense2==2.58.4.10922
curl -fL https://ports.ubuntu.com/pool/main/libu/libusb-1.0/libusb-1.0-0_1.0.27-1_arm64.deb \
  -o /tmp/dpvo-jetson-deps/live-wheels/libusb-1.0-0_1.0.27-1_arm64.deb
rsync -a /tmp/dpvo-jetson-deps/live-wheels/ USER@JETSON:PATH/TO/DPVO/deps/live-wheels/
```

On the Jetson, after building the base DPVO image:

```bash
docker build -f deploy/jetson/Dockerfile.live -t dpvo:jetson-live .
deploy/jetson/run.sh python deploy/jetson/tensorrt_encoder.py \
  --network /models/dpvo.pth --output /output/engines-240x384 --height 240 --width 384
DPVO_WEB_ORIGIN=http://192.168.55.1:9090 deploy/jetson/run_live.sh
```

Open [the hosted viewer](http://192.168.55.1:9090/?url=rerun%2Bhttp%3A%2F%2F192.168.55.1%3A9876%2Fproxy)
on the laptop. Refresh it after restarting the container. HTTP access was tested
in Chrome using WebGL; no local Rerun installation or SSH tunnel is required.
Use the matching host address in both the browser URL and `DPVO_WEB_ORIGIN`
when deploying elsewhere (the latter permits that origin in Rerun's gRPC CORS policy).

The D455F color stream is **1280×800 at 30 Hz**, with calibrated nominal FOV
89.7° horizontal × 63.8° vertical. SDK projection maps rectify and resample the
full view to **384×240**, preserving aspect ratio without a center crop.
Depth is not used; reconstructed scale is monocular.

The default `config/jetson_live.yaml` uses 64 gradient-biased patches and an
eight-frame patch lifetime. The previous 32-patch/four-frame fast preset
diverged during live use. The conservative preset ran around 9 FPS at maximum
clocks and tracked the subsequent live movement without depth-collapse resets.
This is an observed test result, not a general accuracy guarantee.

`DPVO_CAMERA_STRIDE=3` selects raw frames 1, 4, 7, … before image copies,
rectification, and inference: 30 Hz capture supplies at most 10 Hz to DPVO.
Original camera timestamps are retained. A single latest-frame slot prevents
backlogs; additional skipped frames under overload are counted separately from
intentional stride skips in `results/jetson/live/status.json`.
Use `DPVO_CAMERA_STRIDE=4` for a 7.5 Hz input if sustained processing is slower.
Other controls are `DPVO_VIEWER_FPS` (5), `DPVO_CAMERA_FPS` (30),
`DPVO_WEB_PORT` (9090), `DPVO_GRPC_PORT` (9876), `DPVO_LIVE_IMAGE`,
`DPVO_CONTAINER_NAME`, `DPVO_WEIGHTS`, and `DPVO_OUTPUT_DIR`.
Pass additional runner arguments after `run_live.sh`; for example `--serial SERIAL`.

Invalid poses, failed bundle adjustment, or recent median inverse depth hitting
the solver's minimum trigger an explicit tracking-loss message and fresh
initialization. This detects catastrophic failures, not all drift. Sessions also
restart at the graph capacity limit or 8192 processed frames to bound history.
Rerun retains at most 256 MB of buffered log data; raw images are not recorded
unless `--record-frames N` is explicitly requested for bounded diagnostics.

Stop with `docker stop dpvo-realsense`; remove the stopped container with
`docker rm dpvo-realsense` before launching again. Maximum clocks were enabled
for this live deployment; the prior settings are saved on the board in
`results/jetson/live/clocks.before-live-boost`. To restore them after stopping:
`sudo jetson_clocks --restore "$PWD/results/jetson/live/clocks.before-live-boost"`.
The R39.2 vendor script can emit GPU persistence-mode warnings during restore;
verify the CPU/GPU/EMC limits with `sudo jetson_clocks --show`.

Focused checks live in `tests/test_jetson_live.py` and run in the live image.
They cover full-FOV calibration round trips, raw stride/timestamp preservation,
finite-but-collapsed tracking detection, and existing scene logging with Rerun.
`diagnose_stationary.py` provides a synthetic planar move-then-stop comparison;
its drift ratios are diagnostic results, not real-world trajectory accuracy.

### Viser phone viewer

The optional Viser 1.1.0 adapter provides a live RGB preview, camera frustum,
trajectory, sparse map, tracking status, and a **Center view** button. It keeps
at most 6000 displayed points and 2048 trajectory vertices and updates existing
scene objects instead of accumulating recording history.

Stage ARM64 wheels as above, using `viser==1.1.0` and `deps/viser-wheels`, then:

```bash
docker build -f deploy/jetson/Dockerfile.viser -t dpvo:jetson-viser .
DPVO_VIEWER=viser DPVO_LIVE_IMAGE=dpvo:jetson-viser deploy/jetson/run_live.sh
```

Open `http://JETSON_WIFI_IP:9090/` on a phone connected to the same Wi-Fi.
No Rerun connection query or separate gRPC port is needed. The conservative
tracking preset, full-FOV input, TensorRT encoders, and stride 3 are retained.
Drag to orbit, pinch to zoom, and use **Center view** to frame the trajectory.
The mobile layout and live image were checked with an iPhone user agent and
viewport in Chrome; actual iPhone Safari compatibility remains a device test.
The user subsequently confirmed that this Viser viewer works on their iPhone.
**Stop DPVO** stops RealSense streaming and exits the camera/inference worker,
releasing its device handles, model, TensorRT engines, and CUDA context. The
website stays available with a blank preview and cleared map. **Start DPVO**
launches a new worker with fresh tracking state; loading takes several seconds.
A failed camera/model startup leaves the panel available for another attempt.
Button state is shared across connected browsers. Scene snapshots are bounded
and copied to CPU so the web server never holds worker GPU allocations.
`tests/test_viser_viewer.py` checks rendering and reset behavior;
`tests/test_jetson_control.py` checks worker shutdown and bounded snapshots.

### Automatic control panel at boot

The deployed Jetson enables `dpvo-viewer.service`. It launches Viser with
`--start-paused`: only the web panel starts at boot. The camera and inference
worker remain off until Start is pressed. Every service restart returns to this
idle state. Camera startup errors do not take down the web panel. Device nodes
are mapped at container creation; restart the service if adding a new camera.

The unit source is `deploy/jetson/dpvo-viewer.service`; its launcher is
`deploy/jetson/boot_viewer.sh`, installed as `/usr/local/bin/dpvo-viewer-launch`.
Deployment settings live in `/etc/dpvo-viewer.env`:

```ini
DPVO_ROOT=/home/mikexyl/workspaces/dpvo_jetson
DPVO_LIVE_IMAGE=dpvo:jetson-viser
DPVO_CONTAINER_NAME=dpvo-realsense
DPVO_CAMERA_STRIDE=3
DPVO_WEB_PORT=9090
DPVO_PERFORMANCE_CLOCKS=1
```

The host helper `viewer_clocks.py` enables maximum clocks while the worker is
starting or tracking. Original settings are saved once per boot under
`/run/dpvo-viewer.clocks` and restored after browser Stop or service shutdown. Setting
`DPVO_PERFORMANCE_CLOCKS=0` uses dynamic clock scaling instead.

With the boot service enabled, use `sudo systemctl stop dpvo-viewer` to stop
the website and camera, or `sudo systemctl disable --now dpvo-viewer` to turn
off automatic startup. Stopping only the Docker container causes systemd to
restart it. Browser Stop releases the camera and GPU worker and leaves the website available.
Use `sudo journalctl -u dpvo-viewer -f` for logs.

Avahi advertises the hostname `jetson-dpvo.local`; the intended phone URL is
`http://jetson-dpvo.local:9090/` on the same Wi-Fi. The numeric Wi-Fi address
also works. V4RL remains a NetworkManager autoconnect profile.

## Separate JetPack 6.2 target

For the isolated CTI Boson / Orin NX deployment, see
[README.jp62.md](README.jp62.md). It uses `Dockerfile.jp62`, `jetpack62.env`,
`build_jp62.sh`, and `run_jp62.sh`, with separate image/container/output names.
Do not use the JetPack 7.2 image or engine files on that target.
