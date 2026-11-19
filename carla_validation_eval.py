from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from kitti_two_class_eval import (
    AGNOSTIC_NMS,
    CLASSES,
    IGNORE_IOA_THRESHOLD,
    MAX_DETECTIONS,
    NMS_IOU_THRESHOLD,
    PREDICTION_CONFIDENCE_FLOOR,
    calculate_metrics,
)


VARIANTS = ("Baseline", "DR", "AFA", "Combo")
SEEDS = (40, 41, 42)
IMAGE_SIZE = 640
CARLA_IMAGE_SIZE = (1280, 720)
FORMAT_VERSION = 1


def parse_yolo_label(
    path: Path, image_width: int, image_height: int
) -> tuple[dict[str, list[list[float]]], list[list[float]], list[list[float]]]:
    ground_truth: dict[str, list[list[float]]] = {name: [] for name in CLASSES}
    ignore_dontcare: list[list[float]] = []
    ignore_person_sitting: list[list[float]] = []
    if not path.is_file():
        raise FileNotFoundError("CARLA label not found: {}".format(path))

    for line_number, raw in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        fields = raw.split()
        if len(fields) != 5:
            raise ValueError(
                "{}:{}: expected class_id and four normalized YOLO coordinates".format(
                    path.name, line_number
                )
            )
        try:
            class_id = int(fields[0])
            center_x, center_y, width, height = map(float, fields[1:])
        except ValueError as exc:
            raise ValueError(
                "{}:{}: invalid YOLO label".format(path.name, line_number)
            ) from exc
        values = (center_x, center_y, width, height)
        if (
            class_id not in (0, 1, 2, 3)
            or not all(math.isfinite(value) for value in values)
            or not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0)
            or not (0.0 < width <= 1.0 and 0.0 < height <= 1.0)
        ):
            raise ValueError(
                "{}:{}: invalid class or normalized box {}".format(
                    path.name, line_number, fields
                )
            )

        box = [
            (center_x - width / 2.0) * image_width,
            (center_y - height / 2.0) * image_height,
            (center_x + width / 2.0) * image_width,
            (center_y + height / 2.0) * image_height,
        ]
        if class_id == 0:
            ground_truth["Pedestrian"].append(box)
        elif class_id == 1:
            ground_truth["Cyclist"].append(box)
        elif class_id == 2:
            ignore_dontcare.append(box)
        else:
            ignore_person_sitting.append(box)

    return ground_truth, ignore_dontcare, ignore_person_sitting


def collect_validation_images(images_dir: Path, labels_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError("CARLA validation images not found: {}".format(images_dir))
    if not labels_dir.is_dir():
        raise FileNotFoundError("CARLA validation labels not found: {}".format(labels_dir))
    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not image_paths:
        raise ValueError("No validation images found in {}".format(images_dir))
    image_ids = [path.stem for path in image_paths]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Validation image IDs are not unique")

    label_ids = {path.stem for path in labels_dir.glob("*.txt")}
    if set(image_ids) != label_ids:
        missing_labels = sorted(set(image_ids) - label_ids)
        missing_images = sorted(label_ids - set(image_ids))
        raise ValueError(
            "Image/label IDs differ (missing labels: {}; missing images: {})".format(
                missing_labels[:5], missing_images[:5]
            )
        )
    return image_paths


def load_ground_truth(
    image_ids: list[str], labels_dir: Path
) -> dict[str, tuple[dict[str, list[list[float]]], list[list[float]], list[list[float]]]]:
    annotations = {}
    for image_id in image_ids:
        annotations[image_id] = parse_yolo_label(
            labels_dir / "{}.txt".format(image_id),
            CARLA_IMAGE_SIZE[0],
            CARLA_IMAGE_SIZE[1],
        )
    if not any(record[0]["Pedestrian"] for record in annotations.values()):
        raise ValueError("CARLA validation has no Pedestrian ground truth")
    if not any(record[0]["Cyclist"] for record in annotations.values()):
        raise ValueError("CARLA validation has no Cyclist ground truth")
    return annotations


def ground_truth_payload(
    image_ids: list[str],
    annotations: dict[
        str, tuple[dict[str, list[list[float]]], list[list[float]], list[list[float]]]
    ],
) -> dict[str, Any]:
    records = {}
    for image_id in image_ids:
        ground_truth, ignore_dontcare, ignore_person_sitting = annotations[image_id]
        records[image_id] = {
            "ground_truth": ground_truth,
            "ignore_dontcare": ignore_dontcare,
            "ignore_person_sitting": ignore_person_sitting,
        }
    return {
        "format_version": FORMAT_VERSION,
        "label_policy_version": "semantic-ignore-v1",
        "source_format": "YOLO class_id center_x center_y width height, normalized to image size",
        "image_size": list(CARLA_IMAGE_SIZE),
        "source_class_ids": {
            "0": "Pedestrian",
            "1": "Cyclist",
            "2": "IgnoreDontCare",
            "3": "IgnorePersonSitting",
        },
        "coordinate_conversion": (
            "pixel_xyxy = ((center_x - width/2)*image_width, "
            "(center_y - height/2)*image_height, "
            "(center_x + width/2)*image_width, "
            "(center_y + height/2)*image_height)"
        ),
        "coordinate_format": "pixel_xyxy_half_open",
        "class_ids": {"0": "Pedestrian", "1": "Cyclist"},
        "ignore_class_ids": {"2": "IgnoreDontCare", "3": "IgnorePersonSitting"},
        "records": records,
    }


def save_json_gzip(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(value, stream, separators=(",", ":"), ensure_ascii=False)


def load_json_gzip(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def annotations_from_payload(
    payload: dict[str, Any], image_ids: list[str]
) -> dict[str, tuple[dict[str, list[list[float]]], list[list[float]], list[list[float]]]]:
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported CARLA ground-truth format version")
    if payload.get("coordinate_format") != "pixel_xyxy_half_open":
        raise ValueError("Unsupported CARLA ground-truth coordinate format")
    records = payload.get("records")
    if not isinstance(records, dict) or set(records) != set(image_ids):
        raise ValueError("Ground-truth image IDs do not match the evaluation ID list")
    annotations = {}
    for image_id in image_ids:
        record = records[image_id]
        gt = record.get("ground_truth")
        if not isinstance(gt, dict) or set(gt) != set(CLASSES):
            raise ValueError("Invalid ground truth for image {}".format(image_id))
        annotations[image_id] = (
            gt,
            record.get("ignore_dontcare", []),
            record.get("ignore_person_sitting", []),
        )
    return annotations


def checkpoint_path_for(root: Path, variant: str, seed: int) -> Path:
    return root / "{}_seed{}".format(variant, seed) / "best.pt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_record(path: Path, repo_root: Path) -> dict[str, Any]:
    try:
        checkpoint_relative_path = path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        checkpoint_relative_path = path.name
    result: dict[str, Any] = {
        "checkpoint_id": path.parent.name,
        "checkpoint_path": checkpoint_relative_path,
        "checkpoint_sha256": sha256_file(path),
    }
    selection_path = path.parent / "best_selection.json"
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        result["selection"] = {
            key: selection[key]
            for key in ("selected_epoch", "selection_metric", "selection_value")
            if key in selection
        }
    return result


def run_inference(
    checkpoint_path: Path,
    image_paths: list[Path],
    output_path: Path,
    image_size: int,
    batch_size: int,
    device: str,
    checkpoint_metadata: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))

    import torch
    import ultralytics
    from ultralytics import YOLO

    previous_value = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        model = YOLO(str(checkpoint_path))
        names = model.names
        normalized_names = (
            {int(key): value for key, value in names.items()}
            if isinstance(names, dict)
            else dict(enumerate(names))
        )
        if normalized_names != {0: "Pedestrian", 1: "Cyclist"}:
            raise ValueError(
                "Checkpoint class mapping must be {0: Pedestrian, 1: Cyclist}; "
                "got {}".format(normalized_names)
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
        expected_ids = {path.stem for path in image_paths}
        predictions: dict[str, list[dict[str, Any]]] = {}
        for result in results:
            image_id = Path(result.path).stem
            if image_id not in expected_ids or image_id in predictions:
                raise ValueError("Unexpected or duplicate inference result: {}".format(image_id))
            detections = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                scores = result.boxes.conf.detach().cpu().numpy()
                classes = result.boxes.cls.detach().cpu().numpy().astype("int64")
                for class_id, score, box in zip(classes, scores, boxes):
                    if int(class_id) not in (0, 1):
                        continue
                    detections.append(
                        {
                            "class_id": int(class_id),
                            "confidence": float(score),
                            "bbox_xyxy": [float(value) for value in box],
                        }
                    )
            predictions[image_id] = detections
        if set(predictions) != expected_ids:
            missing = sorted(expected_ids - set(predictions))
            raise RuntimeError(
                "Inference did not return every requested image; missing: {}".format(
                    missing[:5]
                )
            )
    finally:
        if previous_value is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous_value

    save_json_gzip(
        output_path,
        {
            "format_version": FORMAT_VERSION,
            "coordinate_format": "pixel_xyxy",
            "confidence_floor": PREDICTION_CONFIDENCE_FLOOR,
            "image_size": image_size,
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "image_ids": [path.stem for path in image_paths],
            "checkpoint": checkpoint_metadata,
            "predictions": predictions,
        },
    )
    return predictions


def predictions_from_payload(
    payload: dict[str, Any], image_ids: list[str]
) -> dict[str, list[tuple[int, float, list[float]]]]:
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported CARLA prediction format version")
    if payload.get("coordinate_format") != "pixel_xyxy":
        raise ValueError("Unsupported CARLA prediction coordinate format")
    if payload.get("image_ids") != image_ids:
        raise ValueError("Prediction image IDs do not match the evaluation ID list")
    source = payload.get("predictions")
    if not isinstance(source, dict) or set(source) != set(image_ids):
        raise ValueError("Predictions do not contain exactly the evaluation image IDs")
    predictions: dict[str, list[tuple[int, float, list[float]]]] = {}
    for image_id in image_ids:
        detections = []
        for detection in source[image_id]:
            class_id = int(detection["class_id"])
            confidence = float(detection["confidence"])
            box = [float(value) for value in detection["bbox_xyxy"]]
            if (
                class_id not in (0, 1)
                or not math.isfinite(confidence)
                or not 0.0 <= confidence <= 1.0
                or len(box) != 4
                or not all(math.isfinite(value) for value in box)
                or box[2] < box[0]
                or box[3] < box[1]
            ):
                raise ValueError("Invalid prediction for image {}".format(image_id))
            detections.append((class_id, confidence, box))
        predictions[image_id] = detections
    return predictions


def score_cached_predictions(
    output_dir: Path, image_ids: list[str]
) -> dict[str, Any]:
    gt_path = output_dir / "carla_validation_ground_truth.json.gz"
    annotations = annotations_from_payload(load_json_gzip(gt_path), image_ids)
    runs = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            run_id = "{}_seed{}".format(variant, seed)
            prediction_path = output_dir / "predictions" / (run_id + "_predictions.json.gz")
            payload = load_json_gzip(prediction_path)
            predictions = predictions_from_payload(payload, image_ids)
            metrics = calculate_metrics(predictions, annotations)
            metrics.pop("protocol", None)
            metrics.pop("ignore_policy", None)
            checkpoint_metadata = payload.get("checkpoint")
            if (
                not isinstance(checkpoint_metadata, dict)
                or checkpoint_metadata.get("checkpoint_id") != run_id
                or not checkpoint_metadata.get("checkpoint_sha256")
            ):
                raise ValueError(
                    "Prediction file has no checkpoint identity/checksum: {}".format(
                        prediction_path
                    )
                )
            metrics["checkpoint"] = checkpoint_metadata
            metrics["image_count"] = len(image_ids)
            runs[run_id] = metrics

    return {
        "protocol": (
            "CARLA validation internal two-class mAP; YOLO normalized xywh labels "
            "decoded to pixel xyxy half-open boxes; 101-point interpolated AP; "
            "IoU 0.50:0.05:0.95"
        ),
        "ignore_policy": (
            "IgnoreDontCare detections ignored at IoA >= 0.5 for both classes; "
            "IgnorePersonSitting likewise for Pedestrian only; target-class ground "
            "truth matching is attempted first"
        ),
        "ground_truth_conversion": {
            "label_policy_version": "semantic-ignore-v1",
            "source_format": "YOLO class_id center_x center_y width height, normalized to image size",
            "class_ids": {
                "0": "Pedestrian",
                "1": "Cyclist",
                "2": "IgnoreDontCare",
                "3": "IgnorePersonSitting",
            },
            "coordinate_conversion": (
                "pixel_xyxy = ((center_x - width/2)*image_width, "
                "(center_y - height/2)*image_height, "
                "(center_x + width/2)*image_width, "
                "(center_y + height/2)*image_height)"
            ),
            "ignore_mapping": {
                "IgnoreDontCare": "ignored for Pedestrian and Cyclist",
                "IgnorePersonSitting": "ignored for Pedestrian only",
            },
        },
        "ignore_ioa_threshold": IGNORE_IOA_THRESHOLD,
        "prediction_confidence_floor": PREDICTION_CONFIDENCE_FLOOR,
        "nms_iou": NMS_IOU_THRESHOLD,
        "max_detections_per_image": MAX_DETECTIONS,
        "agnostic_nms": AGNOSTIC_NMS,
        "image_size": list(CARLA_IMAGE_SIZE),
        "image_count": len(image_ids),
        "image_ids_file": "carla_validation_image_ids.txt",
        "ground_truth_file": "carla_validation_ground_truth.json.gz",
        "runs": runs,
    }


def write_report(output_dir: Path, report: dict[str, Any]) -> Path:
    path = output_dir / "carla_validation_recomputed.json"
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Save CARLA validation predictions and score them from cached outputs."
    )
    parser.add_argument(
        "--mode", choices=("all", "prepare", "infer", "score"), default="all"
    )
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=repo_root / "data" / "kitti_experiment" / "student_15run_10k_retry",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=repo_root / "data" / "kitti_experiment" / "carla_yolo_regions_v2" / "images" / "val",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=repo_root / "data" / "kitti_experiment" / "carla_yolo_regions_v2" / "labels" / "val",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "data"
        / "kitti_experiment"
        / "carla_validation_eval"
        / "reproducible_package",
    )
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("all", "prepare", "infer"):
        image_paths = collect_validation_images(args.images, args.labels)
        image_ids = [path.stem for path in image_paths]
        (output_dir / "carla_validation_image_ids.txt").write_text(
            "\n".join(image_ids) + "\n", encoding="utf-8"
        )
        annotations = load_ground_truth(image_ids, args.labels)
        save_json_gzip(
            output_dir / "carla_validation_ground_truth.json.gz",
            ground_truth_payload(image_ids, annotations),
        )
    if args.mode in ("all", "infer"):
        for variant in VARIANTS:
            for seed in SEEDS:
                run_id = "{}_seed{}".format(variant, seed)
                checkpoint = checkpoint_path_for(args.checkpoints_root, variant, seed)
                destination = output_dir / "predictions" / (
                    run_id + "_predictions.json.gz"
                )
                print("Inferring {} on {} frames".format(run_id, len(image_ids)), flush=True)
                if not checkpoint.is_file():
                    raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint))
                run_inference(
                    checkpoint,
                    image_paths,
                    destination,
                    args.image_size,
                    args.batch_size,
                    args.device,
                    checkpoint_record(checkpoint, repo_root),
                )

    if args.mode in ("all", "score"):
        image_ids_path = output_dir / "carla_validation_image_ids.txt"
        if not image_ids_path.is_file():
            raise FileNotFoundError("Image ID list not found: {}".format(image_ids_path))
        image_ids = image_ids_path.read_text(encoding="utf-8").splitlines()
        if not image_ids or len(image_ids) != len(set(image_ids)):
            raise ValueError("Image ID list is empty or contains duplicates")
        report = score_cached_predictions(output_dir, image_ids)
        report_path = write_report(output_dir, report)
        print(
            json.dumps(
                {
                    "report": report_path.as_posix(),
                    "image_count": report["image_count"],
                    "runs": {
                        run_id: {
                            "mAP_50_95": result["mAP_50_95"],
                            "Pedestrian": result["classes"]["Pedestrian"]["mAP_50_95"],
                            "Cyclist": result["classes"]["Cyclist"]["mAP_50_95"],
                        }
                        for run_id, result in report["runs"].items()
                    },
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
