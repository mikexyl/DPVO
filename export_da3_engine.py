"""Export the official Depth Anything 3 any-view model for a fixed two-view input."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file


class TwoViewDepth(torch.nn.Module):
    """Keep only DA3's depth and confidence outputs for TensorRT."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        height, width = images.shape[-2:]
        features, _ = self.model.backbone(
            images,
            cam_token=None,
            export_feat_layers=[],
            ref_view_strategy="first",
        )
        output = self.model._process_depth_head(features, height, width)
        return output.depth, output.depth_conf


class FixedPositionGetter:
    """ONNX-friendly replacement for DA3's cached torch.cartesian_prod grid."""

    def __init__(self, height, width, device):
        y, x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        self.positions = torch.stack((y, x), dim=-1).reshape(1, -1, 2)

    def __call__(self, batch_size, height, width, device):
        return self.positions.expand(batch_size, -1, -1).clone()


def load_da3_model(source_dir, checkpoint_dir):
    source_dir = Path(source_dir)
    checkpoint_dir = Path(checkpoint_dir)
    package_dir = source_dir / "src"
    if not package_dir.is_dir():
        raise FileNotFoundError(
            f"DA3 source not found at {source_dir}; clone the official repository first"
        )

    sys.path.insert(0, str(package_dir.resolve()))
    from depth_anything_3.cfg import create_object

    config_path = checkpoint_dir / "config.json"
    weights_path = checkpoint_dir / "model.safetensors"
    config = json.loads(config_path.read_text())["config"]
    model = create_object(OmegaConf.create(config))
    state = {
        key.removeprefix("model."): value
        for key, value in load_file(weights_path).items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Released DA3-SMALL weights omit default LayerNorm parameters in unused
    # auxiliary pyramid levels. All depth-path parameters must be present.
    invalid_missing = [
        key for key in missing if not key.startswith("head.scratch.output_conv2_aux.")
    ]
    if invalid_missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={invalid_missing}, unexpected={unexpected}"
        )
    return model.eval()


def main():
    parser = argparse.ArgumentParser(
        description="Export fixed-shape, two-view DA3 depth to TensorRT"
    )
    parser.add_argument("--source-dir", default="thirdparty/depth-anything-3")
    parser.add_argument("--checkpoint-dir", default="models/DA3-SMALL")
    parser.add_argument("--output", default="da3-small-2view-378x504.engine")
    parser.add_argument("--height", type=int, default=378)
    parser.add_argument("--width", type=int, default=504)
    parser.add_argument("--workspace", type=int, default=2048, help="workspace in MiB")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="use FP16 (faster, but validate depth accuracy on the target GPU)",
    )
    parser.add_argument("--keep-onnx", action="store_true")
    parser.add_argument("--onnx-only", action="store_true")
    parser.add_argument(
        "--reuse-onnx",
        action="store_true",
        help="skip ONNX export and compile the existing adjacent .onnx file",
    )
    args = parser.parse_args()

    if args.height % 14 or args.width % 14:
        parser.error("DA3 input height and width must be divisible by 14")
    if not args.reuse_onnx and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DA3 TensorRT export")

    output_path = Path(args.output).resolve()
    onnx_path = output_path.with_suffix(".onnx")
    if not args.reuse_onnx:
        model = TwoViewDepth(
            load_da3_model(args.source_dir, args.checkpoint_dir)
        ).cuda()
        device = torch.device("cuda")
        model.model.backbone.pretrained.position_getter = FixedPositionGetter(
            args.height // 14,
            args.width // 14,
            device,
        )
        images = torch.randn(1, 2, 3, args.height, args.width, device=device)

        with torch.inference_mode():
            # Initialize einops' Torch backend before entering the legacy Torch 2.3
            # ONNX tracer; lazy initialization from inside tracing trips Dynamo.
            model(images)
            torch.onnx.export(
                model,
                images,
                onnx_path,
                input_names=["images"],
                output_names=["depth", "confidence"],
                opset_version=17,
                do_constant_folding=True,
            )

    if args.onnx_only:
        print(onnx_path)
        return
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)

    trtexec = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        f"--memPoolSize=workspace:{args.workspace}M",
        "--builderOptimizationLevel=3",
        "--skipInference",
    ]
    if args.fp16:
        command.append("--fp16")
    subprocess.run(command, check=True)

    if not args.keep_onnx:
        onnx_path.unlink()
    print(output_path)


if __name__ == "__main__":
    main()
