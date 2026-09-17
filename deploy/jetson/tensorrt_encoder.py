"""Export DPVO encoders and execute fixed-shape TensorRT 10 engines."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
from torch import nn

_LOGGER = None


def trt_logger():
    import tensorrt as trt

    global _LOGGER
    if _LOGGER is None:
        _LOGGER = trt.Logger(trt.Logger.WARNING)
    return _LOGGER


class TensorRTEncoder(nn.Module):
    """Single-stream, inference-only replacement for BasicEncoder4."""

    def __init__(self, path):
        super().__init__()
        import tensorrt as trt

        self.logger = trt_logger()
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f'Cannot deserialize {path}')
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.input_shape = tuple(self.engine.get_tensor_shape('images'))
        self.output_shape = tuple(self.engine.get_tensor_shape('features'))
        types = {trt.float32: torch.float32, trt.float16: torch.float16}
        self.input_dtype = types[self.engine.get_tensor_dtype('images')]
        self.output_dtype = types[self.engine.get_tensor_dtype('features')]

    def forward(self, images):
        if not images.is_cuda or tuple(images.shape) != self.input_shape:
            raise ValueError(f'Engine requires CUDA input shaped {self.input_shape}')
        if torch.is_grad_enabled():
            raise RuntimeError('TensorRT encoders support inference only')
        images = images.to(dtype=self.input_dtype).contiguous()
        output = torch.empty(self.output_shape, device=images.device,
                             dtype=self.output_dtype)
        stream = torch.cuda.current_stream(images.device)
        self.stream.wait_stream(stream)
        self.context.set_tensor_address('images', images.data_ptr())
        self.context.set_tensor_address('features', output.data_ptr())
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError('TensorRT execution failed')
        stream.wait_stream(self.stream)
        images.record_stream(self.stream)
        output.record_stream(self.stream)
        return output


def install_encoders(network, directory, checkpoint):
    import tensorrt as trt

    directory = Path(directory)
    metadata = json.loads((directory / 'manifest.json').read_text())
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    if digest != metadata['checkpoint_sha256']:
        raise ValueError('TensorRT engines were built from a different checkpoint')
    if metadata['tensorrt'] != trt.__version__:
        raise ValueError('Rebuild engines with this TensorRT version')
    for name in ('fnet', 'inet'):
        setattr(network.patchify, name, TensorRTEncoder(directory / f'{name}.engine'))


@torch.no_grad()
def main():
    import tensorrt as trt
    from dpvo.extractor import BasicEncoder4

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--network', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=752)
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--validation-images')
    parser.add_argument('--calib')
    args = parser.parse_args()
    if args.height < 16 or args.width < 16 or args.height % 16 or args.width % 16:
        parser.error('Image dimensions must be positive multiples of 16')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    weights = torch.load(args.network, map_location='cpu', weights_only=True)
    weights = {k.replace('module.', ''): v for k, v in weights.items()}
    torch.manual_seed(1234)
    sample = torch.rand(1, 1, 3, args.height, args.width, device='cuda') * 2 - 0.5
    samples = [sample]
    if args.validation_images:
        import cv2
        import numpy as np

        if not args.calib:
            parser.error('--validation-images requires --calib')
        calibration = np.loadtxt(args.calib)
        fx, fy, cx, cy = calibration[:4]
        camera = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        files = sorted(p for p in Path(args.validation_images).iterdir()
                       if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
        if not files:
            parser.error('No validation images found')
        for index in np.linspace(0, len(files)-1, min(10, len(files)), dtype=int):
            image = cv2.imread(str(files[index]))
            if len(calibration) > 4:
                image = cv2.undistort(image, camera, calibration[4:])
            scale = args.height / image.shape[0]
            if scale != 1:
                image = cv2.resize(image, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
            image = image[:args.height, :args.width]
            if image.shape[:2] != (args.height, args.width):
                raise ValueError('Validation image cannot match engine dimensions')
            tensor = torch.from_numpy(image).permute(2, 0, 1).cuda()
            samples.append(2 * (tensor[None, None] / 255.0) - 0.5)
    metadata = dict(checkpoint_sha256=hashlib.sha256(Path(args.network).read_bytes()).hexdigest(),
                    tensorrt=trt.__version__, torch=torch.__version__,
                    gpu=torch.cuda.get_device_name(),
                    input_shape=list(sample.shape), fp16=not args.fp32, validation={})
    for name, channels, norm in [('fnet', 128, 'instance'), ('inet', 384, 'none')]:
        module = BasicEncoder4(output_dim=channels, norm_fn=norm).cuda().eval()
        prefix = f'patchify.{name}.'
        module.load_state_dict({k[len(prefix):]: v for k, v in weights.items()
                                if k.startswith(prefix)})
        onnx_path = output / f'{name}.onnx'
        torch.onnx.export(module, sample, str(onnx_path), opset_version=17,
                          input_names=['images'], output_names=['features'],
                          **({'dynamo': False} if 'dynamo' in inspect.signature(torch.onnx.export).parameters else {}))
        logger = trt_logger()
        builder = trt.Builder(logger)
        net = builder.create_network(0)
        onnx_parser = trt.OnnxParser(net, logger)
        if not onnx_parser.parse(onnx_path.read_bytes()):
            raise RuntimeError('\n'.join(str(onnx_parser.get_error(i))
                                         for i in range(onnx_parser.num_errors)))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        if not args.fp32:
            config.set_flag(trt.BuilderFlag.FP16)
        print(f'Building {name}', flush=True)
        serialized = builder.build_serialized_network(net, config)
        if serialized is None:
            raise RuntimeError(f'Failed to build {name}')
        path = output / f'{name}.engine'
        path.write_bytes(bytes(serialized))
        engine = TensorRTEncoder(path)
        errors = []
        for tensor in samples:
            with torch.autocast('cuda', dtype=torch.float16, enabled=not args.fp32):
                expected = module(tensor).float()
            actual = engine(tensor).float()
            torch.cuda.synchronize()
            difference = (actual - expected).abs()
            relative_rmse = ((actual - expected).square().mean().sqrt()
                             / expected.square().mean().sqrt().clamp_min(1e-8)).item()
            if not torch.isfinite(actual).all() or relative_rmse > 0.02:
                raise RuntimeError(f'{name} validation failed: relative RMSE {relative_rmse}')
            errors.append(dict(relative_rmse=relative_rmse,
                               max_abs_error=difference.max().item()))
        metadata['validation'][name] = errors
        print(name, metadata['validation'][name], flush=True)
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
