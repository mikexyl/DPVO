# Optional online DA3 dense mapping

Dense mapping is **disabled at boot and in generated worker YAML**. It is available
through each robot's `Dense mapping (TensorRT)` checkbox while stopped; select it,
then Start. Stop releases the camera, tracker, and spawned DA3 CUDA process. The
next Start creates an empty map in a fresh session. The selection persists across
Start/Stop in the current controller process; reboot resets it to disabled.

The online path executes **DA3-Small depth and confidence through TensorRT only**.
There is no PyTorch-model fallback. PyTorch is used for conversion/reference
validation and CUDA buffer allocation, not DA3 network inference. No DA3 package
or weights need to be installed on the robot. Missing engines prevent enabling
the checkbox through the control service. Engine load/runtime errors disable the
mapper and are reported in robot logs; DPVO can continue.

## Geometry and resource limits

* Full-FOV RGB resize from DPVO's rectified image to 378 x 238, with matching
  pixel-center intrinsics and ImageNet normalization. Neither camera uses a
  DA3-predicted calibration or pose.
* Every two seconds at most, select a keyframe four slots behind the current
  end, after DPVO initialization. Cache images by input ID, not mutable keyframe
  slot. The bounded queue holds at most one inference job.
* Bilinearly sample DA3 depth at DPVO patch centers. Fit multiplicative scale
  from landmark **camera Z** using confident samples and robust log-ratio/MAD
  rejection. Require at least 16 inliers, 60% agreement, and a median log residual
  at most 0.2. Reject unsupported depth maps instead of fusing them.
* DPVO's session Sim(3) scale is applied to landmark depths; its rigid camera
  transform places the dense points in the same stable session frame as sparse
  points. Scale is arbitrary monocular map scale, **not meters**.
* Retain at most 20 camera-local clouds; refresh their poses from DPVO BA and
  remove clouds whose keyframes were culled. Confidence and depth-edge filtering,
  pixel stride 3, voxel selection and a 50,000-point cap limit memory/bandwidth.
  This is a rolling colored cloud, not a globally optimized TSDF or mesh.
* `/robotX/dpvo/dense_points` uses packed XYZ/RGB PointCloud2. Viser applies the
  same current-session CBS Sim(3) as the sparse cloud and clears both on restart.

Worker parameters (in `/robotX/dpvo_multi_robot/ros__parameters`):

| Parameter | Default |
| --- | --- |
| `enable_dense_mapping` | `false` |
| `dense_engine` | `/output/da3-small-238x378/da3.engine` |
| `dense_fps` | `0.5` |
| `dense_max_points` | `50000` |
| `dense_voxel_size` | `0.02` (session-map units) |

Control service: `SetBool /robotX/dpvo/set_dense_mapping`. Changes are rejected
while tracking. An engine and validation manifest must exist before enabling.
The fleet viewer also has an independent dense-point-size slider.

## Reproduce conversion

Export ONNX on a workstation using a separate environment with compatible Torch
and OpenCV. Install the pinned DA3 source **without its broad dependency list**
so it cannot replace a Jetson's NVIDIA Torch, TensorRT, or NumPy:

```bash
python -m pip install onnx==1.17.0 onnxruntime==1.20.1 omegaconf addict einops safetensors
python -m pip install --no-deps 'depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@3d835ec1a5802d64a8b8b15f817a1ab54809bfe4'
curl -fL --retry 3 \
  https://huggingface.co/depth-anything/DA3-SMALL/resolve/e08cab65ca0ec38e7826075418411ab90cab4da3/model.safetensors \
  -o /tmp/da3-small.safetensors
python -m deploy.jetson.da3_export export \
  --weights /tmp/da3-small.safetensors \
  --images /path/to/camera-image-1.jpg /path/to/camera-image-2.jpg \
  --output deploy/jetson/multi_robot/output/da3-small-238x378
```

The exporter checks source/weight identity, preserves shared safetensors
parameters, precomputes static UV embeddings, and substitutes equivalent meshgrid
coordinates for ONNX's unsupported cartesian-product operation. Validation uses
the unmodified eager operations as reference. It saves ONNX, preprocessed image
inputs, reference depth/confidence, and checksums. ONNX relative RMS error must
be below 0.001.

Transfer that directory into the target's `output/robotX/da3-small-238x378`.
Build the runtime overlay on the target (JP7.2 shown; use `dpvo:online-jp62` for
JP6.2). This leaves the parent image and all host packages untouched:

```bash
docker build --network host -f deploy/jetson/multi_robot/Dockerfile.dense \
  --build-arg BASE_IMAGE=dpvo:online-jp72 -t dpvo:online-jp72-dense .
docker run --rm --runtime=nvidia --network host \
  -v "$PWD/deploy/jetson/multi_robot/output/robot0:/output" \
  dpvo:online-jp72-dense python -m deploy.jetson.da3_export build \
  --output /output/da3-small-238x378 --fp16
```

Build separately on **each GPU/TensorRT version**; do not copy serialized engines
between JP6.2 and JP7.2. The engine is published only after positive, finite
outputs and relative RMS error below 0.02 against saved PyTorch references.
`manifest.json` records GPU, TensorRT version, engine checksum, errors, and
standalone inference latency. If FP16 validation fails, omit `--fp16` to build
FP32. Runtime verifies the engine checksum, TensorRT version and GPU capability.

Use `image: dpvo:online-jp72-dense` in the appropriate fleet robot entry, regenerate,
and recreate that robot container with `run_robot.sh`. Its restart policy keeps
the idle control endpoint available after boot. For an existing coordinator,
build the same overlay with `BASE_IMAGE=dpvo:online-coordinator`, and run it using
`DPVO_COORDINATOR_IMAGE=dpvo:online-coordinator-dense`.

## Verification

```bash
pixi run python -m unittest tests.test_online_dense tests.test_online_fleet -v
pixi run verify-multi-robot
# In a coordinator/ROS environment:
python -m unittest tests.test_online_ros -v
python deploy/jetson/multi_robot/smoke_test.py
```

Live validation needs camera motion to initialize DPVO; stationary frames alone
cannot demonstrate landmark-aligned mapping. Inspect `DA3 dense:` records in
`docker logs dpvo-online-robot0` for accepted point count, scale, support,
residual and total inference/alignment milliseconds.

### Robot0 conversion validation (2026-09-14)

DA3-Small 378x238 FP16 built on Orin / TensorRT 10.16.1.11:
median standalone inference including input/output copies **13.18 ms** (10 runs).
Two reference images passed with depth relative RMS error 0.078–0.104% and
confidence error 0.343–0.787%. An additional robot0 D455F live image passed at
0.067% depth and 0.387% confidence error. This is DA3 inference timing, not an
end-to-end tracking or mapping FPS claim. Reports are in the ignored robot
output directory under `da3-small-238x378/`.

`dense_smoke.py` can also exercise the complete real DPVO-to-DA3 alignment and
fusion path on a calibrated replay, without opening a camera or publishing test
data into the live fleet. Its NPZ input contains BGR `uint8` images shaped
`[N,240,384,3]` and matching `[fx,fy,cx,cy]` intrinsics. It writes an alignment
report and colored cloud, and fails if DPVO does not initialize or fewer than
1,000 dense points are produced.

On the same robot, a 160-frame calibrated KITTI-09 replay exercised DPVO,
TensorRT DA3 and fusion together: 40 retained keyframes, nine accepted depth
updates, 29,865 fused points. Post-warm-up inference/alignment took 19.1–25.9 ms
per update at the default 0.5 Hz cadence; median tracker-frame processing was
122.2 ms. This isolated test had inter-robot retrieval disabled and did not
publish into the fleet. The RealSense live stream was also checked, but its
stationary view did not initialize sufficient landmarks for a live dense-map
quality assessment. Slow camera translation remains necessary for that check.
