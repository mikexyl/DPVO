"""Validate TensorRT SAM against FP32 and measure end-to-end automatic masks."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from dpvo.sam_segmenter import Sam2Segmenter


@torch.inference_mode()
def validate_raw(reference, trt, image):
    actual_masks, actual_scores = trt.infer_raw(image)
    actual_masks, actual_scores = actual_masks.clone(), actual_scores.clone()
    predictor = reference.predictor
    square = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    rgb = torch.from_numpy(np.ascontiguousarray(square[..., ::-1])).cuda().permute(2, 0, 1).float()
    features = predictor.get_im_features(((rgb - predictor.mean) / predictor.std)[None])
    masks, scores = [], []
    for points in trt.grid[:trt.num_prompts].split(trt.batch_size):
        mask, score = predictor._inference_features(
            features, points=points,
            labels=torch.ones(points.shape[:2], dtype=torch.int32, device="cuda"),
            multimask_output=True,
        )
        masks.append(mask)
        scores.append(score)
    masks, scores = torch.cat(masks), torch.cat(scores)
    if not torch.isfinite(actual_masks).all() or not torch.isfinite(actual_scores).all():
        raise AssertionError("nonfinite TensorRT output")
    # Compare corresponding prompt/multimask hypotheses before filtering or NMS.
    good = (scores > reference.config["pred_iou_thresh"]) & ((masks > 0).sum((1, 2)) > 4)
    truth = masks[good] > 0
    estimate = actual_masks[good] > 0
    iou = (truth & estimate).sum((1, 2)) / (truth | estimate).sum((1, 2)).clamp_min(1)
    metrics = dict(
        raw_mask_iou_mean=float(iou.mean()), raw_mask_iou_median=float(iou.median()),
        score_mae=float((scores - actual_scores).abs().mean()),
        score_max_error=float((scores - actual_scores).abs().max()),
        compared_masks=int(good.sum()),
    )
    if metrics["raw_mask_iou_mean"] < 0.97 or metrics["score_mae"] > 0.01:
        raise AssertionError(f"TensorRT accuracy check failed: {metrics}")
    return metrics


def summary(times):
    values = np.asarray(times) * 1000
    return dict(mean_ms=float(values.mean()), median_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/sam21-tiny-trt/sam2.json")
    parser.add_argument("--checkpoint", default="models/sam2.1_hiera_tiny.pt")
    parser.add_argument("--video", default="/tmp/office_01_30s.mp4")
    parser.add_argument("--points-per-side", type=int, default=16)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--output", default="saved_segmentations/sam21_tensorrt_benchmark.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    cap = cv2.VideoCapture(args.video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or frame_count < 5 or args.frames < 1:
        parser.error("a readable video and positive frame count are required")
    sampled = np.linspace(4, frame_count - 5, args.frames, dtype=int)
    images = []
    for index in sampled:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"could not read video frame {index}")
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        height, width = frame.shape[:2]
        images.append(frame[:height - height % 16, :width - width % 16])
    cap.release()
    candidate = Sam2Segmenter(args.model, points_per_side=args.points_per_side)
    reference = Sam2Segmenter(args.checkpoint, points_per_side=args.points_per_side)
    # Warm up CUDA kernels, allocators and both full postprocessing paths.
    for _ in range(2):
        candidate(images[0])
    reference(images[0])
    results, trt_times, ref_times, network_times = [], [], [], []
    for index, image in zip(sampled, images):
        metrics = validate_raw(reference, candidate.trt, image)
        start = perf_counter()
        trt_result = candidate(image)
        trt_times.append(perf_counter() - start)
        start = perf_counter()
        ref_result = reference(image)
        ref_times.append(perf_counter() - start)
        with torch.inference_mode():
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            candidate.trt.graph.replay()
            end.record()
            end.synchronize()
            network_times.append(begin.elapsed_time(end) / 1000)
        entry = dict(video_frame=int(index), **metrics,
                     trt_masks=0 if trt_result is None else len(trt_result.scores),
                     reference_masks=0 if ref_result is None else len(ref_result.scores),
                     trt_coverage=candidate.last_stats["coverage"],
                     reference_coverage=reference.last_stats["coverage"],
                     trt_ms=trt_times[-1] * 1000, reference_ms=ref_times[-1] * 1000)
        results.append(entry)
        print(json.dumps(entry), flush=True)
    report = dict(
        config=candidate.config, video=args.video, samples=len(images),
        tensorrt_end_to_end=summary(trt_times), pytorch_end_to_end=summary(ref_times),
        tensorrt_encoder_and_all_prompt_batches=summary(network_times),
        speedup=float(np.mean(ref_times) / np.mean(trt_times)), frames=results,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    candidate.save_report(output.with_name(output.stem + "_trt.json"), images[-1], trt_result)
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
