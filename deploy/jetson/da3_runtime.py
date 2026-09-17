"""Fixed-shape TensorRT-only DA3 inference. No DA3/PyTorch model is loaded."""
import hashlib
import json
from pathlib import Path


class Da3TensorRT:
    def __init__(self, path, verify_manifest=True):
        import tensorrt as trt
        import torch
        path = Path(path)
        data = path.read_bytes()
        if verify_manifest:
            metadata = json.loads(path.with_name('manifest.json').read_text())
            if (metadata['tensorrt'] != trt.__version__
                    or metadata['capability'] != list(torch.cuda.get_device_capability())
                    or metadata['engine_sha256'] != hashlib.sha256(data).hexdigest()):
                raise ValueError('DA3 TensorRT engine mismatch; rebuild and validate on this Jetson')
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(data)
        if self.engine is None:
            raise RuntimeError(f'Cannot load DA3 TensorRT engine: {path}')
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(priority=0)
        types = {trt.float32: torch.float32, trt.float16: torch.float16}
        self.buffers = {}
        for name in ('images', 'depth', 'confidence'):
            shape = tuple(self.engine.get_tensor_shape(name))
            if min(shape) < 1:
                raise ValueError('DA3 requires a static-shape engine')
            self.buffers[name] = torch.empty(shape, dtype=types[self.engine.get_tensor_dtype(name)], device='cuda')
            self.context.set_tensor_address(name, self.buffers[name].data_ptr())
        self.shape = tuple(self.buffers['images'].shape)

    def __call__(self, images):
        import torch
        if tuple(images.shape) != self.shape:
            raise ValueError(f'DA3 expects {self.shape}, got {images.shape}')
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            self.buffers['images'].copy_(torch.from_numpy(images))
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError('DA3 TensorRT execution failed')
        self.stream.synchronize()
        return tuple(self.buffers[name].float().cpu().numpy().copy() for name in ('depth', 'confidence'))
