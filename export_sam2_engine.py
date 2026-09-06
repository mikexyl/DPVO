"""Export SAM 2.1 Hiera Tiny's image encoder and batched automatic mask decoder.

Static shapes, no tracking/memory modules. Uses the exact same checkpoint and
prompt semantics as the reference Sam2Segmenter, including multimask output.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch


class SamImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        output = self.model.forward_image(images)
        features = output["backbone_fpn"]
        embedding = features[-1] + self.model.no_mem_embed.reshape(1, 256, 1, 1)
        return embedding, features[0], features[1]


class SamMaskDecoder(torch.nn.Module):
    def __init__(self, model, batch_size):
        super().__init__()
        self.prompt_encoder = model.sam_prompt_encoder
        self.mask_decoder = model.sam_mask_decoder
        self.batch_size = batch_size
        self.register_buffer("labels", torch.ones((batch_size, 1), dtype=torch.int32))
        self.register_buffer("image_pe", self.prompt_encoder.get_dense_pe().detach())

    def forward(self, image_embed, high_res_0, high_res_1, points):
        # Automatic generation always supplies one positive point plus SAM's
        # not-a-point padding token. Specialize that path to avoid ONNX's broken
        # advanced-index ScatterND export for the general label implementation.
        point_embedding = self.prompt_encoder.pe_layer._pe_encoding((points + 0.5) / 1024)
        point_embedding = point_embedding + self.prompt_encoder.point_embeddings[1].weight
        padding = self.prompt_encoder.not_a_point_embed.weight.reshape(1, 1, 256).expand(self.batch_size, 1, 256)
        sparse = torch.cat([point_embedding, padding], dim=1)
        dense = self.prompt_encoder.no_mask_embed.weight.reshape(1, 256, 1, 1).expand(self.batch_size, 256, 64, 64)
        masks, scores, _, _ = self.mask_decoder(
            image_embeddings=image_embed, image_pe=self.image_pe,
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=True, repeat_image=True,
            high_res_features=[high_res_0, high_res_1],
        )
        return masks, scores


def load_wrappers(checkpoint, batch_size):
    from ultralytics.models.sam.build import build_sam2_t

    model = build_sam2_t(str(checkpoint)).cuda().eval()
    return SamImageEncoder(model).eval(), SamMaskDecoder(model, batch_size).cuda().eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/sam2.1_hiera_tiny.pt")
    parser.add_argument("--output-dir", default="models/sam21-tiny-trt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--fp16-io", action="store_true", help="use half-precision features/masks between engines")
    parser.add_argument("--workspace", type=int, default=2048, help="TensorRT workspace MiB")
    parser.add_argument("--optimization-level", type=int, default=4)
    parser.add_argument("--onnx-only", action="store_true")
    parser.add_argument("--reuse-onnx", action="store_true")
    parser.add_argument("--only-decoder", action="store_true", help="reuse an already built encoder engine")
    parser.add_argument("--manifest-name", default="sam2.json")
    parser.add_argument("--trtexec", default="/usr/src/tensorrt/bin/trtexec")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    checkpoint = Path(args.checkpoint)
    if checkpoint.name not in {"sam2.1_hiera_tiny.pt", "sam2.1_t.pt"} or not checkpoint.is_file():
        parser.error("a local SAM 2.1 Hiera Tiny checkpoint is required")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    encoder_path = output / "encoder.onnx"
    decoder_path = output / f"decoder_b{args.batch_size}.onnx"
    torch.set_num_threads(2)

    if not args.reuse_onnx:
        encoder, decoder = load_wrappers(checkpoint, args.batch_size)
        images = torch.randn(1, 3, 1024, 1024, device="cuda")
        points = torch.rand(args.batch_size, 1, 2, device="cuda") * 1024
        with torch.inference_mode():
            features = encoder(images)
            decoder(*features, points)
            torch.onnx.export(
                encoder, images, encoder_path, opset_version=17,
                input_names=["images"], output_names=["image_embed", "high_res_0", "high_res_1"],
                do_constant_folding=True,
            )
            # Torch 2.3's legacy tracer creates a CPU repeat_interleave index for
            # the prompt batch. Trace this small module on CPU to avoid mixing
            # that index with CUDA tensors; the resulting graph is device-free.
            torch.onnx.export(
                decoder.cpu(), (*(feature.cpu() for feature in features), points.cpu()),
                decoder_path, opset_version=17,
                input_names=["image_embed", "high_res_0", "high_res_1", "points"],
                output_names=["masks", "scores"], do_constant_folding=True,
            )
        del encoder, decoder, images, features, points
        torch.cuda.empty_cache()
        # Check and simplify the static ONNX graphs before building engines.
        import onnx
        from onnxslim import slim
        for path in (encoder_path, decoder_path):
            graph = slim(onnx.load(path))
            onnx.checker.check_model(graph)
            onnx.save(graph, path)
            print(f"Exported {path}", flush=True)

    if args.onnx_only:
        return

    for path in (encoder_path, decoder_path):
        if args.only_decoder and path == encoder_path:
            if not path.with_suffix(".engine").is_file():
                raise FileNotFoundError(path.with_suffix(".engine"))
            continue
        engine_path = path.with_suffix(".engine")
        command = [
            args.trtexec, f"--onnx={path}", f"--saveEngine={engine_path}",
            f"--memPoolSize=workspace:{args.workspace}M",
            f"--builderOptimizationLevel={args.optimization_level}",
            f"--timingCacheFile={output / 'timing.cache'}", "--skipInference",
        ]
        if args.precision == "fp16":
            command.append("--fp16")
        if args.fp16_io:
            if args.precision != "fp16":
                raise ValueError("--fp16-io requires --precision=fp16")
            if path == encoder_path:
                command += ["--inputIOFormats=fp32:chw", "--outputIOFormats=fp16:chw"]
            else:
                command += ["--inputIOFormats=fp16:chw,fp16:chw,fp16:chw,fp32:chw",
                            "--outputIOFormats=fp16:chw,fp32:chw"]
        command.append("--profilingVerbosity=detailed")
        log_path = path.with_suffix(".build.log")
        print(f"Building {engine_path}; log: {log_path}", flush=True)
        with log_path.open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

    import tensorrt_bindings as trt
    manifest = dict(
        format="dpvo_sam2_tensorrt_v1", checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        encoder=encoder_path.with_suffix(".engine").name,
        decoder=decoder_path.with_suffix(".engine").name,
        image_size=1024, batch_size=args.batch_size, precision=args.precision,
        fp16_io=args.fp16_io,
        tensorrt_version=trt.__version__, gpu=torch.cuda.get_device_name(),
        torch_version=torch.__version__,
    )
    (output / args.manifest_name).write_text(json.dumps(manifest, indent=2))
    print(f"TensorRT bundle: {output / args.manifest_name}", flush=True)


if __name__ == "__main__":
    main()
