# Isolated JetPack 6.2 deployment

Target inspected: `mikexyl@192.168.0.189` (`eagleeye`), Orin NX 16 GB,
CTI Boson carrier, JetPack 6.2.1 / L4T R36.4.7, CUDA 12.6.
This deployment is separate from the validated JetPack 7.2 image and service.

Validated on 2026-09-12: image build, CUDA execution, FP16 TensorRT encoder
conversion, live full-resolution camera capture, and two browser Start/Stop
cycles. The panel is deployed at **http://192.168.0.189:9090/** and left idle.
The camera was stationary during validation; moving-camera initialization,
trajectory accuracy, and sustained tracking FPS still need a physical test.

The final image is `dpvo:jetson-jp62-viser`, container `dpvo-jp62-viewer`.
Its native build is retained as `dpvo:jetson-jp62-native`; the separate
`Dockerfile.jp62.runtime` lets runtime fixes reuse the compiled extensions.
The runtime pins NumPy 1.26.4 because NVIDIA's Torch wheel uses the NumPy 1.x
ABI. Viser 1.1.0 is included; Rerun is omitted here because version 0.37 requires
NumPy 2. The JetPack 7.2 Rerun setup is unaffected.

The base is `nvcr.io/nvidia/pytorch:24.12-py3-igpu` (ARM64 tag verified).
NVIDIA lists container 24.12 for the JetPack 6.1 generation:
https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html
CUDA 12.6, PyTorch 2.6.0a0+df5bbc09d1.nv24.12, and TensorRT 10.7.0 were
verified on this target.
No host packages, drivers, Docker daemon configuration, user groups, clocks,
network settings, systemd units, or existing containers are modified by these
scripts. All new application files live in `~/workspaces/dpvo_jetpack62`.

## Build and validate

Run commands with an existing Docker-authorized account. Do not install packages
on the host or change Docker permissions as part of these scripts.

```bash
cd ~/workspaces/dpvo_jetpack62
set -a
source deploy/jetson/jetpack62.env
set +a
bash deploy/jetson/build_jp62.sh
docker run --rm --runtime=nvidia "$DPVO_JETSON_IMAGE" python -c \
  'import torch, tensorrt, pyrealsense2, viser, cuda_corr, cuda_ba, lietorch_backends; print(torch.__version__, tensorrt.__version__); print(torch.ones(1, device="cuda") + 1)'
mkdir -p results/jetpack62
docker run --rm --runtime=nvidia \
  -v "$PWD/dpvo.pth:/models/dpvo.pth:ro" \
  -v "$PWD/results/jetpack62:/output" \
  "$DPVO_JETSON_IMAGE" python deploy/jetson/tensorrt_encoder.py \
  --network /models/dpvo.pth --height 240 --width 384 \
  --output /output/engines-240x384
```

Build engines on this board; do not copy the JetPack 7.2 TensorRT plans.
The exporter compares encoder outputs with PyTorch. Its default validation uses
synthetic input; supply `--validation-images` and `--calib` for real images.

## Camera and panel

First confirm the D455F is not in use by a colleague's process. Never stop
another application to acquire it. The initial inspection found this camera on
a **480 Mbps USB 2 link**. Read-only V4L2 enumeration on the RealSense color
node `/dev/video8` reports 1280x800 at **8 Hz only**. The JP6.2 profile therefore
selects the advertised 8 Hz profile / stride 1 to preserve the full field of
view (approximately 89.7 by 63.8 degrees). Actual frame timestamps during the
stationary test advanced at about 15 Hz despite that profile; the configured
rate is not a measurement of delivered frames or moving-camera tracking FPS.
Use `DPVO_CAMERA_FORMAT=yuyv` (default here): the SDK's BGR conversion failed
on this stack, while raw YUYV capture followed by OpenCV CPU conversion worked.
The SDK exposes packed uint16 pixels, which the runner reinterprets as YUYV
bytes before rectification and full-frame resize to 384x240. With a verified USB 3 link, override
`DPVO_CAMERA_FPS=30 DPVO_CAMERA_STRIDE=3` to match the first board. No camera
firmware or host driver changes are needed for these profiles.

```bash
bash deploy/jetson/run_jp62.sh
# Open http://192.168.0.189:9090/
docker logs --tail 50 dpvo-jp62-viewer
```

`run_jp62.sh` exposes only the chosen D455F's USB/video/HID device nodes and
port 9090, using Docker bridge networking. Its sysfs ancestry check excludes
the carrier's four CSI cameras. Udev metadata is mounted read-only. The launcher
requires one unambiguous D455F; `DPVO_CAMERA_SERIAL` can also select a USB serial
when exposed and is passed to the SDK. This camera exposes no USB serial file. It fails rather than replace an existing container. The panel starts
idle; camera/model allocation begins only after Start. Stop exits the worker
and releases its camera/GPU resources. Host clocks remain under existing policy.

The dedicated container's `unless-stopped` restart policy brings back only the
idle panel when the existing Docker service restarts. It installs no host boot
service. USB device numbers can change after reboot/replug; if access fails,
recreate only this DPVO container with `run_jp62.sh` after checking camera usage.

```bash
docker stop dpvo-jp62-viewer
# Only when deliberately recreating this DPVO container:
docker rm dpvo-jp62-viewer
bash deploy/jetson/run_jp62.sh
```

JetPack 7.2 continues using its existing Dockerfiles, `run_live.sh`, and
`dpvo-viewer.service`; none of those defaults are changed by this configuration.

## Validation record

- 14 discovered tests: 11 passed, 3 Rerun-only tests skipped intentionally.
- TensorRT synthetic-input relative RMS differences: fnet 0.2496%, inet 0.1231%.
- Two fresh workers each processed 38 live frames in the browser check.
- After Stop, no camera or GPU device handles remained open in the container.
- Existing `ubuntu-jammy-humble-jetson` ROS container remained running.
- No host restart or host boot configuration change was performed. The dedicated
  Docker restart policy is configured but has not been verified by rebooting
  this shared board.

Logs are in the isolated workspace: `build-jp62.log`, `build-jp62-runtime.log`,
`engines-jp62.log`, and `tests-jp62.log`. Models, engines and logs are not committed.
