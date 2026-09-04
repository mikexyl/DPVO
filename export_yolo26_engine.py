from argparse import ArgumentParser
from pathlib import Path

from dpvo.yolo_detector import _load_tensorrt_bindings


def main():
    parser = ArgumentParser(description="Export an Ultralytics YOLO26 TensorRT engine")
    parser.add_argument("--model", default="yolo26n-seg.pt")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--workspace", type=int, default=2, help="TensorRT workspace in GiB")
    parser.add_argument("--fp32", action="store_true", help="export FP32 instead of FP16")
    args = parser.parse_args()

    _load_tensorrt_bindings()
    from ultralytics import YOLO

    engine = YOLO(args.model).export(
        format="engine",
        imgsz=args.image_size,
        quantize=32 if args.fp32 else 16,
        device=0,
        workspace=args.workspace,
        simplify=True,
    )
    print(Path(engine).resolve())


if __name__ == "__main__":
    main()
