"""Export pinned DA3-Small depth/confidence, then build and validate on target GPU.

Conversion uses PyTorch; online inference exclusively executes TensorRT.
Run --help for the two-stage workstation/Jetson workflow.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

CODE_REVISION = '3d835ec1a5802d64a8b8b15f817a1ab54809bfe4'
WEIGHTS_REVISION = 'e08cab65ca0ec38e7826075418411ab90cab4da3'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preprocess(bgr, height, width):
    import cv2
    rgb = cv2.resize(bgr[..., ::-1], (width, height), interpolation=cv2.INTER_LINEAR)
    rgb = (rgb.astype(np.float32) / 255 - np.array([.485, .456, .406], np.float32))
    rgb /= np.array([.229, .224, .225], np.float32)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, None])


def export(args):
    import cv2
    import torch
    import onnx
    import onnxruntime as ort
    from safetensors.torch import load_file
    from depth_anything_3.cfg import create_object, load_config

    if args.height % 14 or args.width % 14 or min(args.height, args.width) < 28:
        raise ValueError('DA3 dimensions must be positive multiples of 14')
    from importlib.metadata import distribution
    provenance = json.loads(distribution('depth-anything-3').read_text('direct_url.json') or '{}')
    revision = provenance.get('vcs_info', {}).get('commit_id')
    if revision != CODE_REVISION:
        raise ValueError(f'Install DA3 from pinned source revision {CODE_REVISION}; got {revision}')
    if digest(args.weights) != '364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf':
        raise ValueError('Weights do not match the pinned official DA3-Small checkpoint')
    torch.set_num_threads(4)
    model = create_object(load_config('depth_anything_3.configs.da3-small')).eval()
    weights = load_file(args.weights)
    weights = {k.removeprefix('model.'): v for k, v in weights.items()}
    # Safetensors omits aliased parameters (the auxiliary LayerNorm is shared).
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    for names in aliases.values():
        available = next((name for name in names if name in weights), None)
        if available is not None:
            for name in names:
                weights.setdefault(name, weights[available])
    model.load_state_dict(weights, strict=True)

    class DepthHead(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.backbone, self.head = network.backbone, network.head

        def forward(self, images):
            features, _ = self.backbone(images, export_feat_layers=[], ref_view_strategy='first')
            result = self.head(features, args.height, args.width, patch_start_idx=0)
            return result.depth, result.depth_conf

    wrapper = DepthHead(model).eval()
    # Static export: ONNX has no cartesian_prod. meshgrid preserves y-major order.
    from depth_anything_3.model.dinov2.layers.rope import PositionGetter
    def positions(self, batch_size, height, width, device):
        y, x = torch.meshgrid(torch.arange(height, device=device),
                              torch.arange(width, device=device), indexing='ij')
        return torch.stack((y, x), dim=-1).reshape(1, height * width, 2).expand(batch_size, -1, -1).clone()
    original_positions = PositionGetter.__call__
    PositionGetter.__call__ = positions
    # UV embeddings are constant at the chosen resolution. Precompute in eager
    # mode to avoid exporter mixed-double Einsum in upstream grid generation.
    import types
    original_embed = wrapper.head._add_pos_embed
    embeddings = {}
    def constant_embed(self, x, width, height, ratio=.1):
        key = tuple(int(v) for v in x.shape[1:])
        if key not in embeddings:
            embeddings[key] = original_embed(torch.zeros_like(x[:1]), width, height, ratio).detach()
        return x + embeddings[key]
    wrapper.head._add_pos_embed = types.MethodType(constant_embed, wrapper.head)
    inputs = []
    for path in args.images:
        bgr = cv2.imread(path)
        if bgr is None:
            raise ValueError(f'Cannot read validation image: {path}')
        inputs.append(preprocess(bgr, args.height, args.width))
    if not inputs:
        raise ValueError('Supply representative camera images for validation')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    onnx_path = output / 'da3.onnx'
    with torch.inference_mode():
        wrapper(torch.from_numpy(inputs[0]))
        torch.onnx.export(wrapper, torch.from_numpy(inputs[0]), str(onnx_path),
                          input_names=['images'], output_names=['depth', 'confidence'],
                          opset_version=17, dynamo=False)
        PositionGetter.__call__ = original_positions
        wrapper.head._add_pos_embed = original_embed
        onnx.checker.check_model(str(onnx_path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        session = ort.InferenceSession(str(onnx_path), options, providers=['CPUExecutionProvider'])
        errors = []
        for index, inp in enumerate(inputs):
            expected = [v.numpy() for v in wrapper(torch.from_numpy(inp))]
            actual = session.run(None, {'images': inp})
            errors.append(validate(expected, actual, .001))
            np.savez_compressed(output / f'validation-{index:02d}.npz', images=inp,
                                depth=expected[0], confidence=expected[1])
    metadata = dict(model='depth-anything/DA3-SMALL', code_revision=CODE_REVISION,
                    weights_revision=WEIGHTS_REVISION, weights_sha256=digest(args.weights),
                    onnx_sha256=digest(onnx_path), height=args.height, width=args.width,
                    preprocessing='RGB ImageNet normalization; full-FOV bilinear resize',
                    outputs=['depth', 'confidence'], onnx_relative_rmse=errors)
    (output / 'export.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


def validate(expected, actual, threshold):
    errors = []
    for reference, value in zip(expected, actual):
        if reference.shape != value.shape or not np.isfinite(value).all() or (value <= 0).any():
            raise ValueError('Invalid depth/confidence output')
        error = float(np.sqrt(np.mean((value.astype(np.float64) - reference) ** 2)) /
                      max(float(np.sqrt(np.mean(reference.astype(np.float64) ** 2))), 1e-8))
        if error > threshold:
            raise ValueError(f'Output relative RMSE {error:.5f} exceeds {threshold}')
        errors.append(error)
    return errors


def build(args):
    import tensorrt as trt
    import torch
    from deploy.jetson.da3_runtime import Da3TensorRT
    output = Path(args.output)
    metadata = json.loads((output / 'export.json').read_text())
    if digest(output / 'da3.onnx') != metadata['onnx_sha256']:
        raise ValueError('ONNX checksum mismatch')
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(output / 'da3.onnx')):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mb * 1024**2)
    # Keep sensitive normalization, exponentials and attention in FP32. FP16 is opt-in
    # and must pass the same reference validation before the manifest is published.
    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        for index in range(network.num_layers):
            layer = network.get_layer(index)
            if layer.type in (trt.LayerType.NORMALIZATION, trt.LayerType.SOFTMAX,
                              trt.LayerType.REDUCE, trt.LayerType.UNARY):
                layer.precision = trt.float32
    else:
        config.clear_flag(trt.BuilderFlag.TF32)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError('TensorRT build failed')
    candidate = output / 'da3.candidate.engine'
    candidate.write_bytes(bytes(plan))
    engine = Da3TensorRT(candidate, verify_manifest=False)
    errors, timings = [], []
    with torch.inference_mode():
        samples = sorted(output.glob('validation-*.npz'))
        if not samples:
            raise ValueError('No PyTorch validation samples')
        for path in samples:
            sample = np.load(path)
            actual = engine(sample['images'])
            errors.append(validate([sample['depth'], sample['confidence']], actual, .02))
        for _ in range(10):
            start = time.perf_counter()
            engine(sample['images'])
            timings.append((time.perf_counter() - start) * 1000)
    candidate.replace(output / 'da3.engine')
    metadata.update(tensorrt=trt.__version__, gpu=torch.cuda.get_device_name(),
                    capability=list(torch.cuda.get_device_capability()),
                    engine_sha256=digest(output / 'da3.engine'), fp16=args.fp16,
                    trt_relative_rmse=errors, inference_ms=float(np.median(timings)))
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=['export', 'build'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--weights')
    parser.add_argument('--images', nargs='*', default=[])
    parser.add_argument('--height', type=int, default=238)
    parser.add_argument('--width', type=int, default=378)
    parser.add_argument('--workspace-mb', type=int, default=1024)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()
    (export if args.action == 'export' else build)(args)


if __name__ == '__main__':
    main()
