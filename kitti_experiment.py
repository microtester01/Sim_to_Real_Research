from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "kitti_raw" / "training"
EXPERIMENT_DIR = DATA_DIR / "kitti_experiment"
IMAGE_DIR = RAW_DIR / "image_2"
LABEL_DIR = RAW_DIR / "label_2"

KITTI_CLASSES = (
    "Car",
    "Van",
    "Truck",
    "Pedestrian",
    "Person_sitting",
    "Cyclist",
    "Tram",
    "Misc",
)
CLASS_TO_ID = {name: index for index, name in enumerate(KITTI_CLASSES)}
TARGET_CLASSES = {"Pedestrian", "Cyclist"}


def dct_basis(size: int) -> np.ndarray:
    coords = np.arange(size, dtype=np.float64)
    frequencies = coords[:, None]
    basis = np.cos(np.pi * (2 * coords[None, :] + 1) * frequencies / (2 * size))
    basis[0] *= np.sqrt(1 / size)
    basis[1:] *= np.sqrt(2 / size)
    return basis


_DCT_HEIGHT = dct_basis(32)
_DCT_WIDTH = dct_basis(64)


def perceptual_hash(path: Path) -> int:
    with Image.open(path) as image:
        image = image.convert("L").resize((64, 32), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float64)
    low = (_DCT_HEIGHT @ pixels @ _DCT_WIDTH.T)[:16, :16].reshape(-1)[1:]
    median = float(np.median(low))
    result = 0
    for bit in low >= median:
        result = (result << 1) | int(bit)
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def image_ssim(left_path: Path, right_path: Path) -> float:
    images = []
    for path in (left_path, right_path):
        with Image.open(path) as image:
            resized = image.convert("L").resize((160, 48), Image.Resampling.BILINEAR)
            images.append(np.asarray(resized, dtype=np.float32) / 255)
    left, right = images
    c1, c2 = 0.01**2, 0.03**2
    mean_left = cv2.GaussianBlur(left, (11, 11), 1.5)
    mean_right = cv2.GaussianBlur(right, (11, 11), 1.5)
    variance_left = cv2.GaussianBlur(left * left, (11, 11), 1.5) - mean_left * mean_left
    variance_right = cv2.GaussianBlur(right * right, (11, 11), 1.5) - mean_right * mean_right
    covariance = cv2.GaussianBlur(left * right, (11, 11), 1.5) - mean_left * mean_right
    score = ((2 * mean_left * mean_right + c1) * (2 * covariance + c2)) / (
        (mean_left * mean_left + mean_right * mean_right + c1)
        * (variance_left + variance_right + c2)
    )
    return float(np.mean(score))


class BKTree:
    def __init__(self) -> None:
        self.root: tuple[int, int, dict[int, tuple]] | None = None

    def add(self, value: int, item_index: int) -> None:
        if self.root is None:
            self.root = (value, item_index, {})
            return
        current = self.root
        while True:
            node_value, _node_index, children = current
            distance = hamming_distance(value, node_value)
            child = children.get(distance)
            if child is None:
                children[distance] = (value, item_index, {})
                return
            current = child

    def query(self, value: int, radius: int) -> list[tuple[int, int]]:
        if self.root is None:
            return []
        found: list[tuple[int, int]] = []
        stack = [self.root]
        while stack:
            node_value, node_index, children = stack.pop()
            distance = hamming_distance(value, node_value)
            if distance <= radius:
                found.append((node_index, distance))
            for edge, child in children.items():
                if distance - radius <= edge <= distance + radius:
                    stack.append(child)
        return found


def read_labels(path: Path) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for line_number, raw in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        fields = raw.split()
        if len(fields) < 15:
            raise ValueError(f"{path.name}:{line_number}: expected 15 fields, got {len(fields)}")
        kind = fields[0]
        try:
            values = [float(item) for item in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"{path.name}:{line_number}: non-numeric KITTI field") from exc
        box = values[3:7]
        if not all(np.isfinite(box)) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"{path.name}:{line_number}: invalid 2D box {box}")
        objects.append({"class": kind, "box": box})
    return objects


def make_contact_sheet(
    image_paths: list[Path], hashes: list[int], pairs: list[tuple[int, int, int]], out: Path
) -> None:
    chosen = sorted(pairs, key=lambda pair: (pair[2], pair[0], pair[1]))[:60]
    if not chosen:
        return
    thumb_w, thumb_h, caption_h = 240, 150, 28
    canvas = Image.new("RGB", (thumb_w * 4, (thumb_h + caption_h) * len(chosen)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, (left, right, distance) in enumerate(chosen):
        for column, index in enumerate((left, right)):
            with Image.open(image_paths[index]) as image:
                thumb = image.convert("RGB")
                thumb.thumbnail((thumb_w, thumb_h))
            x = column * thumb_w
            y = row * (thumb_h + caption_h)
            canvas.paste(thumb, (x, y))
            draw.text((x + 3, y + thumb_h + 2), image_paths[index].stem, fill="black")
        y = row * (thumb_h + caption_h)
        draw.text((thumb_w * 2 + 5, y + 5), f"Hamming distance={distance}", fill="darkblue")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def scan_dataset() -> dict[str, object]:
    if not IMAGE_DIR.is_dir() or not LABEL_DIR.is_dir():
        raise FileNotFoundError(f"Expected extracted KITTI directories under {RAW_DIR}")
    images = sorted(IMAGE_DIR.glob("*.png"))
    labels = sorted(LABEL_DIR.glob("*.txt"))
    image_ids = {path.stem for path in images}
    label_ids = {path.stem for path in labels}
    if image_ids != label_ids:
        raise ValueError(
            f"Image/label ID mismatch: missing labels={len(image_ids-label_ids)}, "
            f"missing images={len(label_ids-image_ids)}"
        )

    class_counts: Counter[str] = Counter()
    empty_label_files = 0
    dimensions: Counter[tuple[int, int]] = Counter()
    hashes: list[int] = []
    started = time.monotonic()
    for index, image_path in enumerate(images, 1):
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            dimensions[image.size] += 1
        objects = read_labels(LABEL_DIR / f"{image_path.stem}.txt")
        if not objects:
            empty_label_files += 1
        class_counts.update(str(obj["class"]) for obj in objects)
        hashes.append(perceptual_hash(image_path))
        if index % 500 == 0 or index == len(images):
            elapsed = time.monotonic() - started
            print(f"Scanned {index}/{len(images)} images ({elapsed:.1f}s)", flush=True)

    tree = BKTree()
    candidates: list[tuple[int, int, int]] = []
    for index, value in enumerate(hashes):
        candidates.extend((prior, index, distance) for prior, distance in tree.query(value, 16))
        tree.add(value, index)
    candidates.sort(key=lambda pair: (pair[2], pair[0], pair[1]))
    near_frame_candidates = [pair for pair in candidates if pair[2] <= 4]
    ssim_pairs = [
        {
            "id_a": images[left].stem,
            "id_b": images[right].stem,
            "hamming": distance,
            "ssim": image_ssim(images[left], images[right]),
        }
        for left, right, distance in near_frame_candidates
    ]
    make_contact_sheet(images, hashes, candidates, EXPERIMENT_DIR / "phash_candidates.png")

    report: dict[str, object] = {
        "image_count": len(images),
        "label_file_count": len(labels),
        "id_sets_match": image_ids == label_ids,
        "empty_label_files": empty_label_files,
        "image_dimensions": {f"{w}x{h}": n for (w, h), n in dimensions.items()},
        "object_counts_by_class": dict(sorted(class_counts.items())),
        "perceptual_hash_bits": 255,
        "perceptual_hash_candidate_pairs_hamming_le_16": len(candidates),
        "near_frame_ssim_audit": {
            "hash_hamming_threshold": 4,
            "pairs_audited": len(ssim_pairs),
            "resize": "grayscale 160x48",
            "ssim": "11x11 Gaussian window, sigma=1.5, standard C1/C2 constants",
            "minimum": min((item["ssim"] for item in ssim_pairs), default=None),
            "median": float(np.median([item["ssim"] for item in ssim_pairs])) if ssim_pairs else None,
            "maximum": max((item["ssim"] for item in ssim_pairs), default=None),
            "pairs": ssim_pairs,
        },
        "perceptual_hash_candidate_pairs": [
            {"id_a": images[left].stem, "id_b": images[right].stem, "hamming": distance}
            for left, right, distance in candidates
        ],
        "candidate_contact_sheet": str(EXPERIMENT_DIR / "phash_candidates.png"),
        "note": "Near-frame candidates require visual review before final scene grouping.",
    }
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_DIR / "integrity_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (EXPERIMENT_DIR / "phash_candidates.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["id_a", "id_b", "hamming"])
        writer.writerows((images[left].stem, images[right].stem, distance) for left, right, distance in candidates)
    return report


def union_find_components(
    image_paths: list[Path], hashes: list[int], duplicate_radius: int = 4
) -> list[list[int]]:
    parent = list(range(len(image_paths)))
    size = [1] * len(image_paths)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            return
        if size[root_a] < size[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        size[root_a] += size[root_b]

    tree = BKTree()
    for index, value in enumerate(hashes):
        for prior, _distance in tree.query(value, duplicate_radius):
            union(prior, index)
        tree.add(value, index)
    groups: dict[int, list[int]] = {}
    for index in range(len(image_paths)):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def assign_group_splits(
    groups: list[list[int]], seed: int = 2026
) -> dict[int, str]:
    targets = {"train": 0.60, "val": 0.20, "test": 0.20}
    total = sum(len(group) for group in groups)
    target_counts = {key: value * total for key, value in targets.items()}
    assigned: dict[str, list[int]] = {key: [] for key in targets}
    rng = random.Random(seed)
    ordered = list(groups)
    rng.shuffle(ordered)
    ordered.sort(key=len, reverse=True)
    for group in ordered:
        split = min(
            targets,
            key=lambda key: (len(assigned[key]) / target_counts[key], len(assigned[key])),
        )
        assigned[split].extend(group)
    split_by_index = {index: split for split, indices in assigned.items() for index in indices}
    return split_by_index


def convert_box(box: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Box collapsed after clipping: {box}")
    return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)


def prepare_yolo_dataset() -> dict[str, object]:
    images = sorted(IMAGE_DIR.glob("*.png"))
    if len(images) != 7481:
        raise ValueError(f"Expected 7481 KITTI training images, found {len(images)}")
    hashes = [perceptual_hash(path) for path in images]
    groups = union_find_components(images, hashes, duplicate_radius=4)
    split_by_index = assign_group_splits(groups, seed=2026)
    destination = EXPERIMENT_DIR / "yolo8"
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty dataset: {destination}")
    for split in ("train", "val", "test"):
        (destination / "images" / split).mkdir(parents=True, exist_ok=True)
        (destination / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_ids: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    split_class_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }
    for index, source_image in enumerate(images):
        split = split_by_index[index]
        sample_id = source_image.stem
        target_image = destination / "images" / split / source_image.name
        target_label = destination / "labels" / split / f"{sample_id}.txt"
        target_image.hardlink_to(source_image)
        with Image.open(source_image) as image:
            width, height = image.size
        records = read_labels(LABEL_DIR / f"{sample_id}.txt")
        rows: list[str] = []
        for record in records:
            kind = str(record["class"])
            if kind == "DontCare":
                continue
            if kind not in CLASS_TO_ID:
                raise ValueError(f"Unexpected KITTI class {kind!r} in {sample_id}")
            coords = convert_box(record["box"], width, height)  # type: ignore[arg-type]
            rows.append(f"{CLASS_TO_ID[kind]} " + " ".join(f"{value:.8f}" for value in coords))
            split_class_counts[split][kind] += 1
        target_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="ascii")
        split_ids[split].append(sample_id)
        if (index + 1) % 1000 == 0 or index + 1 == len(images):
            print(f"Prepared {index + 1}/{len(images)}", flush=True)

    manifest = {
        "seed": 2026,
        "split_method": (
            "connected components of 255-bit grayscale DCT perceptual hashes "
            "at Hamming distance <= 4; groups assigned by greedy 60/20/20 balancing"
        ),
        "hash_resize": "64x32 grayscale; low-frequency 16x16 DCT coefficients; median threshold",
        "duplicate_group_radius": 4,
        "group_count": len(groups),
        "group_size_max": max(map(len, groups)),
        "split_sizes": {key: len(value) for key, value in split_ids.items()},
        "class_object_counts_by_split": {
            split: dict(sorted(counts.items())) for split, counts in split_class_counts.items()
        },
        "class_names": list(KITTI_CLASSES),
        "ignored_label_class": "DontCare (excluded from training targets; evaluator must ignore matching detections explicitly)",
        "split_image_ids": split_ids,
    }
    (destination / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    yaml_lines = [
        f"path: {destination.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: {name}" for index, name in enumerate(KITTI_CLASSES))
    (destination / "kitti.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return manifest


def run_smoke_training(epochs: int = 3) -> None:
    from ultralytics import YOLO

    dataset_yaml = EXPERIMENT_DIR / "yolo8" / "kitti.yaml"
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Prepare the dataset first: {dataset_yaml}")
    model = YOLO("yolov8s.yaml")
    # Ultralytics finalizes only checkpoints created by this from-scratch run.
    force_no_weights_only = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            imgsz=640,
            batch=8,
            device=0,
            workers=4,
            seed=40,
            deterministic=True,
            project=str(EXPERIMENT_DIR / "runs"),
            name="kitti_diagnostic_8class_smoke",
            exist_ok=False,
            pretrained=False,
            optimizer="AdamW",
            lr0=0.001,
            weight_decay=0.0005,
            cache=False,
            amp=False,
            plots=True,
            verbose=True,
            mosaic=1.0,
            close_mosaic=1,
            mixup=0.0,
            copy_paste=0.0,
        )
    finally:
        if force_no_weights_only is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = force_no_weights_only


def evaluate_smoke_test() -> dict[str, object]:
    from ultralytics import YOLO

    checkpoint = EXPERIMENT_DIR / "runs" / "kitti_diagnostic_8class_smoke2" / "weights" / "best.pt"
    dataset_yaml = EXPERIMENT_DIR / "yolo8" / "kitti.yaml"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Diagnostic checkpoint not found: {checkpoint}")
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Prepared dataset not found: {dataset_yaml}")

    previous_value = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        model = YOLO(str(checkpoint))
        metrics = model.val(
            data=str(dataset_yaml),
            split="test",
            imgsz=640,
            batch=8,
            device=0,
            workers=0,
            project=str(EXPERIMENT_DIR / "runs"),
            name="kitti_diagnostic_internal_test",
            exist_ok=False,
            plots=True,
            verbose=True,
        )
    finally:
        if previous_value is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous_value

    box = metrics.box
    per_class_ap = {
        str(metrics.names[int(class_index)]): {
            "mAP50": float(box.ap50[row_index]),
            "mAP50_95": float(box.ap[row_index]),
        }
        for row_index, class_index in enumerate(box.ap_class_index)
    }
    report: dict[str, object] = {
        "model": "YOLOv8s 8-class diagnostic, trained from scratch",
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_yaml),
        "evaluated_split": "internal KITTI test, split manifest seed=2026",
        "n_images": len(json.loads((EXPERIMENT_DIR / "yolo8" / "split_manifest.json").read_text(
            encoding="utf-8"
        ))["split_image_ids"]["test"]),
        "image_size": 640,
        "batch_size": 8,
        "device": "cuda:0",
        "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
        "box_mAP50": float(box.map50),
        "box_mAP50_95": float(box.map),
        "per_class_AP50_95": per_class_ap,
        "run_dir": str(metrics.save_dir),
    }
    report_path = EXPERIMENT_DIR / "internal_test_metrics.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def measure_smoke_latency(series_count: int = 5, iterations: int = 1000) -> dict[str, object]:
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.ops import non_max_suppression

    checkpoint = EXPERIMENT_DIR / "runs" / "kitti_diagnostic_8class_smoke2" / "weights" / "best.pt"
    sample = IMAGE_DIR / "000000.png"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Diagnostic checkpoint not found: {checkpoint}")
    if not sample.is_file():
        raise FileNotFoundError(f"Latency sample not found: {sample}")

    with Image.open(sample) as image:
        rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    scale = min(640 / width, 640 / height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    resized = np.asarray(
        Image.fromarray(rgb).resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    )
    letterboxed = np.full((640, 640, 3), 114, dtype=np.uint8)
    left = (640 - resized_width) // 2
    top = (640 - resized_height) // 2
    letterboxed[top : top + resized_height, left : left + resized_width] = resized
    tensor = (
        torch.from_numpy(letterboxed.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device="cuda", dtype=torch.float16)
        .div_(255)
    )

    previous_value = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        model = YOLO(str(checkpoint))
    finally:
        if previous_value is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous_value
    model.model = model.model.cuda().half().eval()

    def run_inference() -> None:
        with torch.inference_mode():
            predictions = model.model(tensor)[0]
            non_max_suppression(predictions, conf_thres=0.001, iou_thres=0.7, max_det=300)

    series: list[dict[str, float]] = []
    all_times: list[float] = []
    for series_index in range(series_count):
        for _ in range(100):
            run_inference()
        torch.cuda.synchronize()
        starts = []
        ends = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_inference()
            end.record()
            starts.append(start)
            ends.append(end)
        torch.cuda.synchronize()
        elapsed_ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
        all_times.extend(elapsed_ms)
        series.append(
            {
                "series": series_index + 1,
                "p50_ms": float(np.percentile(elapsed_ms, 50)),
                "p95_ms": float(np.percentile(elapsed_ms, 95)),
            }
        )

    report: dict[str, object] = {
        "model": "YOLOv8s 8-class diagnostic checkpoint from scratch",
        "checkpoint": str(checkpoint),
        "sample_id": sample.stem,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "input_shape": [1, 3, 640, 640],
        "batch": 1,
        "precision": "FP16",
        "preprocessing": "KITTI RGB image, letterboxed to 640x640 with pad value 114; excluded from timing",
        "warmup_iterations_per_series": 100,
        "measured_iterations_per_series": iterations,
        "measured_series": series,
        "pooled_p50_ms": float(np.percentile(all_times, 50)),
        "pooled_p95_ms": float(np.percentile(all_times, 95)),
        "measured_region": "model forward plus Ultralytics NMS",
        "nms_conf": 0.001,
        "nms_iou": 0.7,
        "max_detections": 300,
    }
    report_path = EXPERIMENT_DIR / "diagnostic_latency.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="KITTI integrity and YOLO pilot pipeline")
    parser.add_argument(
        "command",
        choices=("scan", "prepare", "train-smoke", "eval-smoke-test", "measure-smoke-latency"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    if args.command == "scan":
        print(json.dumps(scan_dataset(), indent=2, ensure_ascii=False))
    elif args.command == "prepare":
        print(json.dumps(prepare_yolo_dataset(), indent=2, ensure_ascii=False))
    elif args.command == "train-smoke":
        run_smoke_training(args.epochs)
    elif args.command == "eval-smoke-test":
        print(json.dumps(evaluate_smoke_test(), indent=2, ensure_ascii=False))
    elif args.command == "measure-smoke-latency":
        print(json.dumps(measure_smoke_latency(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
