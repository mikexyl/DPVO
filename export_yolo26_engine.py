from argparse import ArgumentParser
from pathlib import Path

from dpvo.yolo_detector import _load_tensorrt_bindings


def main():
    parser = ArgumentParser(
        description="Export an Ultralytics YOLO26 or YOLOE-26 TensorRT engine"
    )
    parser.add_argument("--model", default="yolo26n-seg.pt")
    parser.add_argument(
        "--classes",
        nargs="+",
        help="open-vocabulary text prompts to bake into a YOLOE engine",
    )
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--workspace", type=int, default=2, help="TensorRT workspace in GiB")
    parser.add_argument("--fp32", action="store_true", help="export FP32 instead of FP16")
    args = parser.parse_args()

    _load_tensorrt_bindings()
    from ultralytics import YOLO, YOLOE

    is_yoloe = Path(args.model).stem.startswith("yoloe-")
    if args.classes and not is_yoloe:
        parser.error("--classes requires a YOLOE model")
    if is_yoloe:
        model = YOLOE(args.model)
        if args.classes:
            model.set_classes(args.classes)
        elif "-pf" not in Path(args.model).stem:
            parser.error("text-prompt YOLOE models require --classes before export")
    else:
        model = YOLO(args.model)

    engine = model.export(
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
