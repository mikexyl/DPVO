# Experimental TensorRT update blocks — laptop validation

No Jetsons were accessed or changed. The online fleet configuration still uses
only the existing encoder engines. The new backend is opt-in through
`install_update()` or `benchmark.py --trt-update DIRECTORY`.

## What runs in TensorRT

In addition to the existing `fnet`/`inet` encoders, the new engines cover:

- Correlation projection MLP, context addition, and layer normalization.
- Both gated recurrent residual blocks and the delta/confidence output heads.

Neighbor lookup, the sequential neighbor MLPs, grouped soft aggregation,
correlation lookup, and bundle adjustment remain PyTorch/custom CUDA. The
implementation preserves the order of both graph aggregations. Engines support
1–16,384 active edges by default. Larger graphs explicitly fall back to the
original update operator; benchmark reports include the fallback count.
Engines execute on private CUDA streams with dependencies on the caller's stream.

## Laptop result (2026-09-17)

RTX 4070 Laptop GPU, PyTorch 2.3.1, TensorRT 10.13.2.6. EuRoC MH_01_easy,
first 300 selected frames, stride 2, 240×368 pixels, 64 patches/frame,
`jetson_online.yaml`, classic loop closure disabled. Forty warmup frames were
excluded from throughput. Camera capture and the multi-robot frontend were not
part of this offline VO benchmark.

| Backend | End-to-end FPS | Sim(3)-aligned position RMSE |
|---|---:|---:|
| PyTorch (one initial run) | 40.9 | 0.85 cm |
| TensorRT encoders, PyTorch update (two repeat runs) | 40.5 mean (40.0–41.0) | 0.85 cm |
| TensorRT encoders + FP16 update blocks (two repeat runs) | 43.7 mean (42.1–45.4) | 0.85 cm |

The paired repeat means show about **8% higher end-to-end throughput** over the
encoder-only TensorRT baseline. GPU/CPU clocks were not locked; the run spread
is material, and this short sequence does not establish Jetson speed or general
accuracy. No engine-profile fallbacks occurred in the sequence runs.

Accuracy evaluation excludes the first 11 images, which precede available ground
truth, and interpolates ground-truth positions at the remaining 289 image times.
It fits one Sim(3) alignment per trajectory. An earlier exploratory calculation
clamped out-of-range ground-truth timestamps and gave 2.30 cm; the final result
above uses only valid timestamp overlap. This is position error, not an
orientation metric or a long-sequence accuracy qualification.

FP32 engine parity against FP32 PyTorch passed with `atol=2e-4, rtol=1e-4`.
FP16 was checked on varying synthetic graphs including profile boundaries and
six captured real update inputs (6,400–13,376 edges). On the real inputs, maximum
delta difference from AMP PyTorch was 0.2422 feature-grid pixels and mean absolute
difference was at most 0.0039. The validation script rejects nonfinite outputs,
max delta error >0.5, or mean delta error >0.05 for mixed-precision comparisons.

Artifacts: `results/trt-update-laptop/` contains the engines, manifests,
per-frame timing JSON, trajectories, `accuracy.json`, and parity reports. These
and the experiment environment are ignored by Git.

## Reproduce on this laptop

The isolated environment inherits the existing Pixi Torch/CUDA extensions and
adds ONNX 1.17.0. TensorRT is the laptop's system installation:

```bash
export PYTHONPATH=.:/usr/lib/python3.10/dist-packages
PY=.runtime/trt-update-env/bin/python
$PY -m deploy.jetson.tensorrt_update --network dpvo.pth \
  --output results/trt-update-laptop/fp16 --fp16
$PY -m deploy.jetson.validate_tensorrt_update --network dpvo.pth \
  --engines results/trt-update-laptop/fp16 \
  --output results/trt-update-laptop/parity-fp16.json
$PY -m deploy.jetson.benchmark \
  --images /data/euroc/MH_01_easy/mav0/cam0/data --calib calib/euroc.txt \
  --network dpvo.pth --config config/jetson_online.yaml \
  --scale .5 --stride 2 --frames 300 --warmup 40 --remap --prefetch 2 \
  --trt-encoders results/trt-update-laptop/encoders \
  --trt-update results/trt-update-laptop/fp16 \
  --output results/trt-update-laptop/reproduction.json \
  --opts CLASSIC_LOOP_CLOSURE False
```

Omit `--trt-update` for the encoder-only baseline. `--capture-update-inputs FILE`
saves up to six real update inputs; do not use capture runs as clean timing
comparisons. Evaluate saved trajectories with
`evaluate_tensorrt_benchmark.py --images ... --groundtruth ... --results ... --output ...`.

The backend validates checkpoint checksum, TensorRT version, and GPU model.
Rebuild and validate engines on each target platform; the laptop engines must
not be copied onto Orin. The runtime follows NVIDIA's
[dynamic-shape execution interface](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dynamic-shapes-basics.html).
