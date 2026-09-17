"""Opt-in TensorRT dense update blocks; graph aggregation remains native PyTorch.

Build engines on the target GPU. This module does not alter deployment defaults.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
from torch import nn

from deploy.jetson.tensorrt_encoder import trt_logger


class UpdateHead(nn.Module):
    def __init__(self, update):
        super().__init__()
        self.corr, self.norm = update.corr, update.norm

    def forward(self, net, inp, corr):
        return self.norm(net + inp + self.corr(corr))


class UpdateTail(nn.Module):
    def __init__(self, update):
        super().__init__()
        self.gru, self.d, self.w = update.gru, update.d, update.w

    def forward(self, net):
        net = self.gru(net)
        return net, self.d(net), self.w(net)


class DynamicEngine(nn.Module):
    """Single-caller engine, enqueued on the caller's current CUDA stream."""
    def __init__(self, path):
        super().__init__()
        import tensorrt as trt
        self.device = torch.cuda.current_device()
        self.runtime = trt.Runtime(trt_logger())
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f'Cannot deserialize {path}')
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if n not in self.inputs]
        types = {trt.float32: torch.float32, trt.float16: torch.float16}
        self.dtypes = {n: types[self.engine.get_tensor_dtype(n)] for n in names}

    def forward(self, *args):
        if torch.is_grad_enabled():
            raise RuntimeError('TensorRT update supports inference only')
        if len(args) != len(self.inputs) or not all(x.is_cuda for x in args):
            raise ValueError('Expected CUDA tensors for every engine input')
        if args[0].device.index != self.device or any(x.device != args[0].device for x in args):
            raise ValueError('Inputs must share a CUDA device')
        tensors = []
        for name, value in zip(self.inputs, args):
            value = value.to(dtype=self.dtypes[name]).contiguous()
            if not self.context.set_input_shape(name, tuple(value.shape)):
                raise ValueError(f'Input outside engine profile: {name} {tuple(value.shape)}')
            self.context.set_tensor_address(name, value.data_ptr())
            tensors.append(value)
        outputs = []
        for name in self.outputs:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise RuntimeError(f'Unresolved output shape: {name}')
            value = torch.empty(shape, device=args[0].device, dtype=self.dtypes[name])
            self.context.set_tensor_address(name, value.data_ptr())
            outputs.append(value)
        stream = torch.cuda.current_stream(args[0].device)
        self.stream.wait_stream(stream)
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError('TensorRT update execution failed')
        stream.wait_stream(self.stream)
        for value in tensors + outputs:
            value.record_stream(self.stream)
        return tuple(outputs)


class TensorRTUpdate(nn.Module):
    def __init__(self, update, directory):
        super().__init__()
        directory = Path(directory)
        self.original = update
        self.head = DynamicEngine(directory / 'update_head.engine')
        self.tail = DynamicEngine(directory / 'update_tail.engine')
        metadata = json.loads((directory / 'manifest.json').read_text())
        self.max_edges = metadata['max_edges']
        self.fallback_calls = 0

    def forward(self, net, inp, corr, flow, ii, jj, kk):
        from dpvo import fastba
        if torch.is_grad_enabled():
            raise RuntimeError('TensorRT update supports inference only')
        if net.shape[0] != 1 or not 1 <= net.shape[1] <= self.max_edges:
            self.fallback_calls += 1
            return self.original(net, inp, corr, flow, ii, jj, kk)
        output_dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else net.dtype
        net, = self.head(net, inp, corr)
        ix, jx = fastba.neighbors(kk, jj)
        mask_ix = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx = (jx >= 0).float().reshape(1, -1, 1)
        net = net + self.original.c1(mask_ix * net[:, ix])
        net = net + self.original.c2(mask_jx * net[:, jx])
        net = net + self.original.agg_kk(net, kk)
        net = net + self.original.agg_ij(net, ii * 12345 + jj)
        net, delta, weight = self.tail(net)
        return net, (delta.to(output_dtype), weight.to(output_dtype), None)


def install_update(network, directory, checkpoint):
    import tensorrt as trt
    metadata = json.loads((Path(directory) / 'manifest.json').read_text())
    if metadata['checkpoint_sha256'] != hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest():
        raise ValueError('Update engines were built from a different checkpoint')
    if metadata['tensorrt'] != trt.__version__ or metadata['gpu'] != torch.cuda.get_device_name():
        raise ValueError('Rebuild update engines for this GPU and TensorRT version')
    network.update = TensorRTUpdate(network.update, directory)


@torch.no_grad()
def main():
    import tensorrt as trt
    from dpvo.net import VONet
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--network', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-edges', type=int, default=16384)
    parser.add_argument('--opt-edges', type=int, default=2048)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.opt_edges <= args.max_edges:
        parser.error('Require 1 <= opt-edges <= max-edges')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    weights = torch.load(args.network, map_location='cpu', weights_only=True)
    model = VONet().cuda().eval()
    incompatible = model.load_state_dict({k.replace('module.', ''): v for k, v in weights.items()}, strict=False)
    if incompatible.missing_keys:
        raise ValueError(f'Checkpoint is missing weights: {incompatible.missing_keys}')
    n = args.opt_edges
    samples = (torch.randn(1, n, 384, device='cuda'), torch.randn(1, n, 384, device='cuda'),
               torch.randn(1, n, 882, device='cuda'))
    for name, module, values, inputs, outputs in (
        ('update_head', UpdateHead(model.update), samples, ['net', 'inp', 'corr'], ['features']),
        ('update_tail', UpdateTail(model.update), samples[:1], ['features'], ['net_out', 'delta', 'weight']),
    ):
        onnx_path = output / f'{name}.onnx'
        torch.onnx.export(module.eval(), values, str(onnx_path), opset_version=17,
                          input_names=inputs, output_names=outputs,
                          dynamic_axes={key: {1: 'edges'} for key in inputs + outputs},
                          **({'dynamo': False} if 'dynamo' in inspect.signature(torch.onnx.export).parameters else {}))
        builder = trt.Builder(trt_logger())
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        onnx_parser = trt.OnnxParser(network, trt_logger())
        if not onnx_parser.parse(onnx_path.read_bytes()):
            raise RuntimeError('\n'.join(str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors)))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 << 20)
        config.clear_flag(trt.BuilderFlag.TF32)
        if args.fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        profile = builder.create_optimization_profile()
        for key, value in zip(inputs, values):
            profile.set_shape(key, (1, 1, value.shape[-1]), tuple(value.shape),
                              (1, args.max_edges, value.shape[-1]))
        config.add_optimization_profile(profile)
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError(f'Failed to build {name}')
        (output / f'{name}.engine').write_bytes(bytes(plan))
        print(f'Built {name}', flush=True)
    metadata = dict(checkpoint_sha256=hashlib.sha256(Path(args.network).read_bytes()).hexdigest(),
                    tensorrt=trt.__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                    max_edges=args.max_edges, opt_edges=args.opt_edges, fp16=args.fp16)
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
