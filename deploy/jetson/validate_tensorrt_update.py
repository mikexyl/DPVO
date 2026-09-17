"""Laptop/target GPU parity and timing checks for opt-in TensorRT update engines."""
import argparse
import json
from pathlib import Path
import time

import torch
from dpvo.net import VONet
from deploy.jetson.tensorrt_update import install_update


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--network', required=True)
    parser.add_argument('--engines', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--inputs', help='Real inputs captured by benchmark.py')
    args = parser.parse_args()
    torch.manual_seed(1234)
    model = VONet().cuda().eval()
    weights = torch.load(args.network, map_location='cpu', weights_only=True)
    incompatible = model.load_state_dict({k.replace('module.', ''): v for k,v in weights.items()}, strict=False)
    if incompatible.missing_keys:
        raise ValueError(f'Checkpoint is missing weights: {incompatible.missing_keys}')
    original = model.update
    install_update(model, args.engines, args.network)
    results = []
    captured = torch.load(args.inputs, weights_only=True) if args.inputs else None
    fp16 = json.loads((Path(args.engines) / 'manifest.json').read_text())['fp16']
    def outputs(result):
        return (result[0], result[1][0], result[1][1])
    def timing(module, inputs):
        for _ in range(5): module(*inputs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20): module(*inputs)
        torch.cuda.synchronize()
        return (time.perf_counter() - start)*1000/20
    for amp in (False, True):
        for case in (captured if captured is not None else (1, 37, 64, 512, 2048, 8192, model.update.max_edges, model.update.max_edges + 1)):
            count = case[0].shape[1] if captured is not None else case
            kk = torch.arange(count, device='cuda') // 4
            jj = torch.arange(count, device='cuda') % 4
            ii = kk // 64
            inputs = (torch.randn(1,count,384,device='cuda'), torch.randn(1,count,384,device='cuda'),
                      torch.randn(1,count,882,device='cuda'), None, ii, jj, kk)
            if captured is not None:
                inputs = tuple(x.cuda() if torch.is_tensor(x) else x for x in case)
                if not amp:
                    inputs = tuple(x.float() if torch.is_tensor(x) and x.is_floating_point() else x for x in inputs)
            with torch.autocast('cuda', enabled=amp):
                expected = outputs(original(*inputs))
                actual = outputs(model.update(*inputs))
                errors = {}
                for name,a,b in zip(('net','delta','weight'),actual,expected):
                    if not torch.isfinite(a).all(): raise AssertionError(f'Nonfinite {name}')
                    errors[name] = dict(max_abs=(a.float()-b.float()).abs().max().item(), mean_abs=(a.float()-b.float()).abs().mean().item(),
                                        reference_dtype=str(b.dtype), actual_dtype=str(a.dtype))
                if not fp16 and not amp:
                    for a,b in zip(actual, expected):
                        torch.testing.assert_close(a,b,rtol=1e-4,atol=2e-4)
                else:
                    if errors['delta']['max_abs'] > .5 or errors['delta']['mean_abs'] > .05:
                        raise AssertionError(f'Update delta parity exceeded validation limits: {errors}')
                results.append(dict(edges=count, amp=amp, errors=errors,
                    pytorch_ms=timing(original,inputs), tensorrt_ms=timing(model.update,inputs)))
                print(json.dumps(results[-1]),flush=True)
    Path(args.output).write_text(json.dumps(results,indent=2))


if __name__ == '__main__': main()
