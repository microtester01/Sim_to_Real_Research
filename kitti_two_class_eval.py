from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


CLASSES = ("Pedestrian", "Cyclist")
IOU_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.50, 0.951, 0.05))
IGNORE_IOA_THRESHOLD = 0.5
PREDICTION_CONFIDENCE_FLOOR = 0.001
NMS_IOU_THRESHOLD = 0.7
MAX_DETECTIONS = 300
AGNOSTIC_NMS = False


def parse_kitti_annotation(path: Path) -> tuple[dict[str, list[list[float]]], list[list[float]], list[list[float]]]:
    ground_truth: dict[str, list[list[float]]] = {name: [] for name in CLASSES}
    dontcare: list[list[float]] = []
    ignored_pedestrian: list[list[float]] = []
    if not path.is_file():
        raise FileNotFoundError("KITTI annotation not found: {}".format(path))
    for line_number, raw in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        fields = raw.split()
        if len(fields) < 15:
            raise ValueError("{}:{}: expected at least 15 fields".format(path.name, line_number))
        try:
            box = [float(value) for value in fields[4:8]]
        except ValueError as exc:
            raise ValueError("{}:{}: invalid 2D box".format(path.name, line_number)) from exc
        if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("{}:{}: invalid 2D box {}".format(path.name, line_number, box))
        class_name = fields[0]
        if class_name in ground_truth:
            ground_truth[class_name].append(box)
        elif class_name == "DontCare":
            dontcare.append(box)
        elif class_name == "Person_sitting":
            ignored_pedestrian.append(box)
    return ground_truth, dontcare, ignored_pedestrian


def box_iou(box: list[float], boxes: list[list[float]]) -> np.ndarray:
    if not boxes:
        return np.zeros(0, dtype=np.float64)
    candidates = np.asarray(boxes, dtype=np.float64)
    left = np.maximum(box[0], candidates[:, 0])
    top = np.maximum(box[1], candidates[:, 1])
    right = np.minimum(box[2], candidates[:, 2])
    bottom = np.minimum(box[3], candidates[:, 3])
    intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    area_box = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_candidates = np.maximum(0.0, candidates[:, 2] - candidates[:, 0]) * np.maximum(
        0.0, candidates[:, 3] - candidates[:, 1]
    )
    union = area_box + area_candidates - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def box_ioa_detection(box: list[float], regions: list[list[float]]) -> np.ndarray:
    if not regions:
        return np.zeros(0, dtype=np.float64)
    candidates = np.asarray(regions, dtype=np.float64)
    left = np.maximum(box[0], candidates[:, 0])
    top = np.maximum(box[1], candidates[:, 1])
    right = np.minimum(box[2], candidates[:, 2])
    bottom = np.minimum(box[3], candidates[:, 3])
    intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    detection_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if detection_area == 0:
        return np.zeros(len(regions), dtype=np.float64)
    return intersection / detection_area


def average_precision(
    predictions: list[tuple[str, float, list[float]]],
    ground_truth: dict[str, list[list[float]]],
    ignored_regions: dict[str, list[list[float]]],
    iou_threshold: float,
) -> tuple[float, int, int, int]:
    total_ground_truth = sum(len(boxes) for boxes in ground_truth.values())
    if total_ground_truth == 0:
        return float("nan"), 0, 0, len(predictions)

    ordered = sorted(enumerate(predictions), key=lambda item: (-item[1][1], item[0]))
    matched = {image_id: set() for image_id in ground_truth}
    true_positives: list[int] = []
    false_positives: list[int] = []
    ignored_count = 0

    for _index, (image_id, _score, box) in ordered:
        gt_boxes = ground_truth.get(image_id, [])
        overlaps = box_iou(box, gt_boxes)
        unmatched = [
            index for index in range(len(gt_boxes)) if index not in matched[image_id]
        ]
        if unmatched:
            best_unmatched = max(unmatched, key=lambda index: overlaps[index])
            if float(overlaps[best_unmatched]) >= iou_threshold:
                matched[image_id].add(best_unmatched)
                true_positives.append(1)
                false_positives.append(0)
                continue
        if overlaps.size and float(overlaps.max()) >= iou_threshold:
            true_positives.append(0)
            false_positives.append(1)
            continue

        if np.max(box_ioa_detection(box, ignored_regions.get(image_id, [])), initial=0.0) >= IGNORE_IOA_THRESHOLD:
            ignored_count += 1
            continue
        true_positives.append(0)
        false_positives.append(1)

    if not true_positives:
        return 0.0, 0, 0, ignored_count
    cumulative_tp = np.cumsum(true_positives, dtype=np.float64)
    cumulative_fp = np.cumsum(false_positives, dtype=np.float64)
    recall = cumulative_tp / total_ground_truth
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1.0)
    sampled_precision = [
        float(precision[recall >= level].max()) if np.any(recall >= level) else 0.0
        for level in np.linspace(0.0, 1.0, 101)
    ]
    return float(np.mean(sampled_precision)), int(cumulative_tp[-1]), int(cumulative_fp[-1]), ignored_count


def calculate_metrics(
    predictions: dict[str, list[tuple[int, float, list[float]]]],
    annotations: dict[str, tuple[dict[str, list[list[float]]], list[list[float]], list[list[float]]]],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    all_class_aps: list[float] = []
    for class_id, class_name in enumerate(CLASSES):
        gt_by_image: dict[str, list[list[float]]] = {}
        ignore_by_image: dict[str, list[list[float]]] = {}
        class_predictions: list[tuple[str, float, list[float]]] = []
        for image_id, (gt, dontcare, ignored_pedestrian) in annotations.items():
            gt_by_image[image_id] = gt[class_name]
            ignores = list(dontcare)
            if class_name == "Pedestrian":
                ignores.extend(ignored_pedestrian)
            ignore_by_image[image_id] = ignores
            class_predictions.extend(
                (image_id, score, box)
                for predicted_class, score, box in predictions.get(image_id, [])
                if predicted_class == class_id
            )

        threshold_aps: dict[str, float] = {}
        totals = {"true_positives": 0, "false_positives": 0, "ignored_detections": 0}
        for threshold in IOU_THRESHOLDS:
            ap, tp, fp, ignored = average_precision(
                class_predictions, gt_by_image, ignore_by_image, threshold
            )
            threshold_aps["{:.2f}".format(threshold)] = (
                ap if sum(len(boxes) for boxes in gt_by_image.values()) else None
            )
            if threshold == 0.50:
                totals["true_positives"] = tp
                totals["false_positives"] = fp
                totals["ignored_detections"] = ignored
        class_ap_values = [
            value for value in threshold_aps.values() if value is not None and np.isfinite(value)
        ]
        mean_ap = float(np.mean(class_ap_values)) if class_ap_values else None
        results[class_name] = {
            "AP_by_IoU": threshold_aps,
            "mAP_50_95": mean_ap,
            "ground_truth_count": sum(len(boxes[class_name]) for boxes, _, _ in annotations.values()),
            **totals,
        }
        if mean_ap is not None:
            all_class_aps.append(mean_ap)
    return {
        "protocol": "internal KITTI two-class AP; 101-point interpolated precision; IoU 0.50:0.05:0.95",
        "ignore_policy": (
            "DontCare detections ignored when intersection/detection-area >= 0.5; "
            "Person_sitting treated likewise for Pedestrian only; target-class GT matched first"
        ),
        "ignore_ioa_threshold": IGNORE_IOA_THRESHOLD,
        "mAP_50_95": (
            float(np.mean(all_class_aps))
            if len(all_class_aps) == len(CLASSES)
            else None
        ),
        "classes": results,
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    image_paths: list[Path],
    labels_dir: Path,
    image_size: int = 640,
    batch_size: int = 8,
    device: str = "0",
    predictions_output: Path | None = None,
) -> dict[str, Any]:
    if not image_paths:
        raise ValueError("No validation images were supplied")
    annotations = {
        path.stem: parse_kitti_annotation(labels_dir / "{}.txt".format(path.stem))
        for path in image_paths
    }
    missing_classes = [
        class_name
        for class_name in CLASSES
        if not any(gt[class_name] for gt, _, _ in annotations.values())
    ]
    if missing_classes:
        raise ValueError(
            "Validation split has no ground truth for required class(es): {}".format(
                ", ".join(missing_classes)
            )
        )
    from ultralytics import YOLO

    previous_value = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        model = YOLO(str(checkpoint_path))
        names = model.names
        if any(names.get(index) != name for index, name in enumerate(CLASSES)):
            raise ValueError(
                "Checkpoint class mapping must be {0: Pedestrian, 1: Cyclist}; got {}".format(
                    names
                )
            )
        results = model.predict(
            source=[str(path) for path in image_paths],
            imgsz=image_size,
            batch=batch_size,
            device=device,
            conf=PREDICTION_CONFIDENCE_FLOOR,
            iou=NMS_IOU_THRESHOLD,
            max_det=MAX_DETECTIONS,
            agnostic_nms=AGNOSTIC_NMS,
            half=device != "cpu",
            stream=True,
            verbose=False,
        )
        predictions: dict[str, list[tuple[int, float, list[float]]]] = {}
        for result in results:
            image_id = Path(result.path).stem
            if result.boxes is None:
                predictions[image_id] = []
                continue
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
            predictions[image_id] = [
                (int(class_id), float(score), box.astype(float).tolist())
                for class_id, score, box in zip(classes, scores, boxes)
                if int(class_id) in (0, 1)
            ]
    finally:
        if previous_value is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous_value
    report = calculate_metrics(predictions, annotations)
    if predictions_output is not None:
        predictions_output.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(predictions_output, "wt", encoding="utf-8") as stream:
            json.dump(predictions, stream, separators=(",", ":"))
    report.update(
        {
            "checkpoint": str(checkpoint_path.resolve()),
            "image_count": len(image_paths),
            "image_size": image_size,
            "batch_size": batch_size,
            "device": device,
            "prediction_confidence_floor": PREDICTION_CONFIDENCE_FLOOR,
            "nms_iou": NMS_IOU_THRESHOLD,
            "max_detections_per_image": MAX_DETECTIONS,
            "agnostic_nms": AGNOSTIC_NMS,
            "selected_image_ids": [path.stem for path in image_paths],
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a two-class detector on internal KITTI labels")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    image_ids = manifest["split_image_ids"][args.split]
    image_paths = [args.images / "{}.png".format(image_id) for image_id in image_ids]
    missing = [path for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing selected images, first: {}".format(missing[0]))
    report = evaluate_checkpoint(
        args.checkpoint,
        image_paths,
        args.labels,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
        predictions_output=args.predictions_output,
    )
    report["evaluated_split"] = args.split
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "selected_image_ids"}, indent=2))


if __name__ == "__main__":
    main()
