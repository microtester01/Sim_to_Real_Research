import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image

from kitti_experiment import perceptual_hash, union_find_components


CLASS_IDS = {"Pedestrian": 0, "Cyclist": 1}
IGNORE_CLASS_IDS = {"DontCare": 2, "Person_sitting": 3}
SUBSET_FRACTIONS = (0.10, 0.25, 0.50, 1.0)


def convert_kitti_annotations(label_path, width, height):
    rows = []
    counts = Counter()
    for line_number, raw in enumerate(
        Path(label_path).read_text(encoding="ascii").splitlines(), 1
    ):
        fields = raw.split()
        if not fields:
            continue
        if len(fields) < 8:
            raise ValueError(
                "{}:{}: expected at least 8 KITTI fields".format(
                    label_path, line_number
                )
            )
        class_name = fields[0]
        if class_name in CLASS_IDS:
            class_id = CLASS_IDS[class_name]
        elif class_name in IGNORE_CLASS_IDS:
            class_id = IGNORE_CLASS_IDS[class_name]
        else:
            continue
        try:
            x1, y1, x2, y2 = (float(value) for value in fields[4:8])
        except ValueError as exc:
            raise ValueError(
                "{}:{}: invalid 2D bounding box".format(label_path, line_number)
            ) from exc
        x1 = min(max(x1, 0.0), float(width))
        x2 = min(max(x2, 0.0), float(width))
        y1 = min(max(y1, 0.0), float(height))
        y2 = min(max(y2, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                "{}:{}: bounding box collapses after clipping".format(
                    label_path, line_number
                )
            )
        center_x = (x1 + x2) / (2 * width)
        center_y = (y1 + y2) / (2 * height)
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        rows.append(
            "{} {:.8f} {:.8f} {:.8f} {:.8f}".format(
                class_id, center_x, center_y, box_width, box_height
            )
        )
        counts[class_name] += 1
    return rows, counts


def nested_group_subsets(groups, seed=2026, fractions=SUBSET_FRACTIONS):
    ordered = [list(group) for group in groups]
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=len, reverse=True)
    total = sum(len(group) for group in ordered)
    if total == 0:
        raise ValueError("Cannot create subsets from no image IDs")
    selected_groups = []
    selected_ids = []
    result = {}
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("Subset fractions must be in (0, 1]")
        target_count = total if fraction == 1 else max(1, round(total * fraction))
        while len(selected_ids) < target_count and ordered:
            group = ordered.pop(0)
            selected_groups.append(group)
            selected_ids.extend(group)
        group_records = sorted(
            (
                {
                    "group_id": hashlib.sha256(
                        "\n".join(sorted(group)).encode("ascii")
                    ).hexdigest()[:16],
                    "image_ids": sorted(group),
                }
                for group in selected_groups
            ),
            key=lambda record: record["group_id"],
        )
        result[fraction] = {
            "image_ids": sorted(selected_ids),
            "group_count": len(selected_groups),
            "groups": group_records,
            "requested_images": target_count,
            "actual_images": len(selected_ids),
        }
    if ordered:
        raise AssertionError("The 100% subset did not include every image group")
    return result


def _write_split(image_ids, image_dir, label_dir, raw_images, raw_labels):
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    class_counts = Counter()
    for image_id in image_ids:
        source_image = raw_images / "{}.png".format(image_id)
        source_label = raw_labels / "{}.txt".format(image_id)
        if not source_image.is_file() or not source_label.is_file():
            raise FileNotFoundError(
                "Missing KITTI source for image {}".format(image_id)
            )
        target_image = image_dir / source_image.name
        target_label = label_dir / source_label.name
        if not target_image.exists():
            target_image.hardlink_to(source_image)
        if not target_label.exists():
            with Image.open(str(source_image)) as image:
                rows, counts = convert_kitti_annotations(
                    source_label, image.width, image.height
                )
            target_label.write_text(
                "\n".join(rows) + ("\n" if rows else ""), encoding="ascii"
            )
            class_counts.update(counts)
    return class_counts


def prepare_training_data(manifest_path, raw_images, raw_labels, output):
    manifest_path = Path(manifest_path)
    raw_images = Path(raw_images)
    raw_labels = Path(raw_labels)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite non-empty directory: {}".format(output))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_ids = manifest["split_image_ids"]
    train_ids = split_ids["train"]
    train_images = [raw_images / "{}.png".format(image_id) for image_id in train_ids]
    missing = [path for path in train_images if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing KITTI image: {}".format(missing[0]))

    hashes = [perceptual_hash(path) for path in train_images]
    groups = union_find_components(train_images, hashes, duplicate_radius=4)
    id_groups = [[train_ids[index] for index in group] for group in groups]
    subsets = nested_group_subsets(id_groups)
    output.mkdir(parents=True, exist_ok=True)

    validation_counts = _write_split(
        split_ids["val"],
        output / "images" / "val",
        output / "labels" / "val",
        raw_images,
        raw_labels,
    )
    subset_manifest = {
        "seed": 2026,
        "fraction_selection": "nested; whole perceptual-hash connected components",
        "perceptual_hash_radius": 4,
        "group_count": len(groups),
        "group_size_max": max(map(len, groups)),
        "validation_image_ids": split_ids["val"],
        "validation_class_counts": dict(sorted(validation_counts.items())),
        "subsets": {},
        "ignored_annotation_policy": (
            "Pedestrian=0, Cyclist=1, DontCare=2 (ignore both classes), "
            "Person_sitting=3 (ignore Pedestrian only); "
            "all other KITTI classes are excluded from target labels"
        ),
    }

    for fraction, subset in subsets.items():
        percent = int(round(fraction * 100))
        subset_name = "train_{}".format(percent)
        counts = _write_split(
            subset["image_ids"],
            output / subset_name / "images",
            output / subset_name / "labels",
            raw_images,
            raw_labels,
        )
        yaml_path = output / "kitti_{}.yaml".format(subset_name)
        yaml_text = "\n".join(
            (
                "path: {}".format(output.resolve().as_posix()),
                "train: {}/images".format(subset_name),
                "val: images/val",
                "",
                "names:",
                "  0: Pedestrian",
                "  1: Cyclist",
                "  2: IgnoreDontCare",
                "  3: IgnorePersonSitting",
                "",
            )
        )
        yaml_path.write_text(yaml_text, encoding="utf-8")
        subset_manifest["subsets"][str(percent)] = {
            **subset,
            "directory": subset_name,
            "yaml": yaml_path.name,
            "class_counts": dict(sorted(counts.items())),
        }

    (output / "subset_manifest.json").write_text(
        json.dumps(subset_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return subset_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_training_data(
        args.split_manifest, args.images, args.labels, args.output
    )
    for fraction, subset in report["subsets"].items():
        print(
            "{}% subset: {} images / {} hash groups".format(
                fraction, subset["actual_images"], subset["group_count"]
            ),
            flush=True,
        )
    print("Manifest: {}".format(args.output / "subset_manifest.json"), flush=True)


if __name__ == "__main__":
    main()
