from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from kitti_two_class_eval import (
    CLASSES,
    IGNORE_IOA_THRESHOLD,
    IOU_THRESHOLDS,
    box_ioa_detection,
    box_iou,
    parse_kitti_annotation,
)


VARIANTS = ("Baseline", "DR", "AFA", "Combo", "UpperBound")
SEEDS = (40, 41, 42)
BOOTSTRAP_STATISTICS = {
    "positive_interaction": ("interaction", 0.0),
    "combo_practical_gain": ("combo_minus_baseline", 0.05),
    "upper_bound_gap_within_15_percent": ("combo_minus_85pct_upper_bound", 0.0),
}


class RunMetricCache:
    def __init__(
        self,
        image_ids: list[str],
        predictions: dict[str, list[list[Any]]],
        annotations: dict[
            str, tuple[dict[str, list[list[float]]], list[list[float]], list[list[float]]]
        ],
    ) -> None:
        self.gt_counts: dict[str, np.ndarray] = {}
        self.events: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        for class_id, class_name in enumerate(CLASSES):
            gt_by_image = [annotations[image_id][0][class_name] for image_id in image_ids]
            self.gt_counts[class_name] = np.asarray(
                [len(boxes) for boxes in gt_by_image], dtype=np.float64
            )
            ignore_by_image = []
            for image_id in image_ids:
                _gt, dontcare, ignored_pedestrian = annotations[image_id]
                ignores = list(dontcare)
                if class_name == "Pedestrian":
                    ignores.extend(ignored_pedestrian)
                ignore_by_image.append(ignores)

            per_image_detections = []
            for image_id in image_ids:
                per_image_detections.append(
                    [
                        (index, float(item[1]), item[2])
                        for index, item in enumerate(predictions.get(image_id, []))
                        if int(item[0]) == class_id
                    ]
                )

            class_events = []
            for threshold in IOU_THRESHOLDS:
                records: list[tuple[float, int, int, bool, bool]] = []
                for image_index, detections in enumerate(per_image_detections):
                    gt_boxes = gt_by_image[image_index]
                    ignored_regions = ignore_by_image[image_index]
                    matched: set[int] = set()
                    for prediction_index, score, box in sorted(
                        detections, key=lambda item: (-item[1], item[0])
                    ):
                        overlaps = box_iou(box, gt_boxes)
                        unmatched = [
                            index
                            for index in range(len(gt_boxes))
                            if index not in matched
                        ]
                        if unmatched:
                            best = max(unmatched, key=lambda index: overlaps[index])
                            if float(overlaps[best]) >= threshold:
                                matched.add(best)
                                records.append(
                                    (score, image_index, prediction_index, True, False)
                                )
                                continue
                        if overlaps.size and float(overlaps.max()) >= threshold:
                            records.append(
                                (score, image_index, prediction_index, False, True)
                            )
                            continue
                        if (
                            np.max(
                                box_ioa_detection(box, ignored_regions), initial=0.0
                            )
                            >= IGNORE_IOA_THRESHOLD
                        ):
                            continue
                        records.append(
                            (score, image_index, prediction_index, False, True)
                        )

                records.sort(key=lambda item: (-item[0], item[1], item[2]))
                last_true_positive = max(
                    (index for index, item in enumerate(records) if item[3]),
                    default=-1,
                )
                if last_true_positive >= 0:
                    records = records[: last_true_positive + 1]
                else:
                    records = []
                class_events.append(
                    (
                        np.asarray([item[1] for item in records], dtype=np.int16),
                        np.asarray([item[3] for item in records], dtype=np.float64),
                        np.asarray([item[4] for item in records], dtype=np.float64),
                    )
                )
            self.events[class_name] = class_events

    @staticmethod
    def _weighted_ap(
        image_counts: np.ndarray,
        gt_counts: np.ndarray,
        events: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        total_gt = image_counts @ gt_counts
        image_indices, is_tp, is_fp = events
        if image_indices.size == 0:
            return np.zeros(image_counts.shape[0], dtype=np.float64)
        weights = image_counts[:, image_indices]
        true_positives = np.cumsum(weights * is_tp, axis=1)
        false_positives = np.cumsum(weights * is_fp, axis=1)
        precision = true_positives / np.maximum(
            true_positives + false_positives, 1.0
        )
        recall = np.divide(
            true_positives,
            total_gt[:, None],
            out=np.zeros_like(true_positives),
            where=total_gt[:, None] > 0,
        )
        precision_envelope = np.maximum.accumulate(precision[:, ::-1], axis=1)[
            :, ::-1
        ]
        recall_levels = np.linspace(0.0, 1.0, 101)
        ap = np.zeros(image_counts.shape[0], dtype=np.float64)
        for row in range(image_counts.shape[0]):
            if total_gt[row] > 0:
                positions = np.searchsorted(
                    recall[row], recall_levels, side="left"
                )
                valid = positions < image_indices.size
                ap[row] = np.mean(
                    np.where(
                        valid,
                        precision_envelope[row, np.minimum(positions, image_indices.size - 1)],
                        0.0,
                    )
                )
        ap[total_gt == 0] = np.nan
        return ap

    def map_for_image_counts(self, image_counts: np.ndarray) -> np.ndarray:
        class_aps = []
        for class_name in CLASSES:
            threshold_aps = [
                self._weighted_ap(
                    image_counts, self.gt_counts[class_name], events
                )
                for events in self.events[class_name]
            ]
            class_aps.append(np.nanmean(np.stack(threshold_aps), axis=0))
        return np.nanmean(np.stack(class_aps), axis=0)


def load_annotations(
    image_ids: list[str], labels_dir: Path
) -> dict[str, tuple[dict[str, list[list[float]]], list[list[float]], list[list[float]]]]:
    return {
        image_id: parse_kitti_annotation(labels_dir / "{}.txt".format(image_id))
        for image_id in image_ids
    }


def load_predictions(path: Path) -> dict[str, list[list[Any]]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        predictions = json.load(stream)
    if not isinstance(predictions, dict):
        raise ValueError("Prediction sidecar must contain an image-to-detections mapping")
    return predictions


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    names = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    count = len(names)
    for rank, name in enumerate(names):
        running_max = max(running_max, (count - rank) * p_values[name])
        adjusted[name] = min(1.0, running_max)
    return adjusted


def run_bootstrap(
    caches: dict[str, list[RunMetricCache]],
    image_count: int,
    draws: int,
    random_seed: int,
    batch_size: int,
    observed_effects: dict[str, float],
) -> dict[str, Any]:
    rng = np.random.default_rng(random_seed)
    distributions = {variant: np.empty(draws, dtype=np.float64) for variant in VARIANTS}
    for offset in range(0, draws, batch_size):
        size = min(batch_size, draws - offset)
        selected_seeds = rng.integers(0, len(SEEDS), size=(size, len(SEEDS)))
        counts = np.empty((size, len(SEEDS), image_count), dtype=np.uint16)
        for row in range(size):
            for slot in range(len(SEEDS)):
                sampled_images = rng.integers(0, image_count, size=image_count)
                counts[row, slot] = np.bincount(
                    sampled_images, minlength=image_count
                )

        per_variant = {
            variant: np.empty((size, len(SEEDS)), dtype=np.float64)
            for variant in VARIANTS
        }
        for slot in range(len(SEEDS)):
            for selected_seed in range(len(SEEDS)):
                rows = np.flatnonzero(selected_seeds[:, slot] == selected_seed)
                if rows.size == 0:
                    continue
                image_weights = counts[rows, slot].astype(np.float64)
                for variant in VARIANTS:
                    per_variant[variant][rows, slot] = caches[variant][
                        selected_seed
                    ].map_for_image_counts(image_weights)
        for variant in VARIANTS:
            distributions[variant][offset : offset + size] = np.mean(
                per_variant[variant], axis=1
            )
        if offset == 0 or (offset + size) % 1000 < batch_size:
            print(
                "Bootstrap progress: {}/{} replicates".format(offset + size, draws),
                file=sys.stderr,
                flush=True,
            )

    effects = {
        "interaction": (
            distributions["Combo"]
            - distributions["DR"]
            - distributions["AFA"]
            + distributions["Baseline"]
        ),
        "combo_minus_baseline": distributions["Combo"] - distributions["Baseline"],
        "combo_minus_85pct_upper_bound": (
            distributions["Combo"] - 0.85 * distributions["UpperBound"]
        ),
    }
    return {
        "variant_mAP": {
            variant: {
                "mean": float(np.mean(distributions[variant])),
                "percentile_95_ci": percentile_interval(distributions[variant]),
            }
            for variant in VARIANTS
        },
        "effects": {
            name: {
                "mean": float(np.mean(values)),
                "percentile_95_ci": percentile_interval(values),
            }
            for name, values in effects.items()
        },
        "p_values": {
            name: float(
                (1 + np.count_nonzero(
                    values - observed_effects[effect_name]
                    >= observed_effects[effect_name] - null
                ))
                / (draws + 1)
            )
            for name, (effect_name, null) in BOOTSTRAP_STATISTICS.items()
            for values in [effects[effect_name]]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hierarchical image-and-seed bootstrap for the CARLA-to-KITTI factorial experiment."
    )
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.draws < 1 or args.batch_size < 1:
        parser.error("--draws and --batch-size must be positive")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    image_ids = manifest["split_image_ids"]["test"]
    annotations = load_annotations(image_ids, args.labels)
    caches: dict[str, list[RunMetricCache]] = {}
    point_estimates: dict[str, list[float]] = {}
    for variant in VARIANTS:
        caches[variant] = []
        point_estimates[variant] = []
        for seed in SEEDS:
            run_dir = args.runs / "{}_seed{}".format(variant, seed)
            selection_path = run_dir / "best_selection.json"
            if not selection_path.is_file():
                raise FileNotFoundError(
                    "Missing validation checkpoint selection: {}".format(selection_path)
                )
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            report_path = args.predictions_dir / "{}_seed{}_test.json".format(
                variant, seed
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("evaluated_split") != "test" or report.get("image_count") != len(
                image_ids
            ):
                raise ValueError("Unexpected test evaluation report: {}".format(report_path))
            if Path(report["checkpoint"]).resolve() != (run_dir / "best.pt").resolve():
                raise ValueError("Test report does not use the validation-selected checkpoint")
            point_estimates[variant].append(float(report["mAP_50_95"]))
            prediction_path = args.predictions_dir / "{}_seed{}_predictions.json.gz".format(
                variant, seed
            )
            predictions = load_predictions(prediction_path)
            if list(predictions) != image_ids:
                raise ValueError(
                    "Prediction sidecar image order does not match test manifest: {}".format(
                        prediction_path
                    )
                )
            cache = RunMetricCache(image_ids, predictions, annotations)
            original_score = float(np.mean(cache.map_for_image_counts(np.ones((1, len(image_ids))))[0:1]))
            if not np.isclose(original_score, point_estimates[variant][-1], atol=1e-12):
                raise ValueError(
                    "Cached weighted AP does not reproduce test mAP for {} seed {}: {} vs {}".format(
                        variant, seed, original_score, point_estimates[variant][-1]
                    )
                )
            caches[variant].append(cache)

    point_estimate_effects = {
        "interaction": float(
            np.mean(point_estimates["Combo"])
            - np.mean(point_estimates["DR"])
            - np.mean(point_estimates["AFA"])
            + np.mean(point_estimates["Baseline"])
        ),
        "combo_minus_baseline": float(
            np.mean(point_estimates["Combo"]) - np.mean(point_estimates["Baseline"])
        ),
        "combo_minus_85pct_upper_bound": float(
            np.mean(point_estimates["Combo"])
            - 0.85 * np.mean(point_estimates["UpperBound"])
        ),
    }
    result = run_bootstrap(
        caches,
        len(image_ids),
        args.draws,
        args.seed,
        args.batch_size,
        point_estimate_effects,
    )
    result["protocol"] = {
        "bootstrap": "paired hierarchical cluster bootstrap: resample the three seed pairs with replacement; within each selected seed, resample test image IDs with replacement and apply identical image weights to all variants",
        "draws": args.draws,
        "random_seed": args.seed,
        "test_image_count": len(image_ids),
        "training_seeds": list(SEEDS),
        "checkpoint_selection": "best.pt selected exclusively by validation mAP",
        "primary_metric": "internal two-class mAP@0.50:0.95",
        "one_sided_nulls": {
            "positive_interaction": "I <= 0",
            "combo_practical_gain": "Combo - Baseline <= 0.05",
            "upper_bound_gap_within_15_percent": "Combo - 0.85 * UpperBound <= 0",
        },
        "p_value_method": "centered one-sided bootstrap tail probability with plus-one correction",
        "multiple_testing": "Holm step-down adjustment across the three preregistered criteria",
    }
    result["point_estimates_by_seed"] = point_estimates
    result["point_estimate_effects"] = point_estimate_effects
    adjusted = holm_adjust(result["p_values"])
    result["holm_adjusted_p_values"] = adjusted
    result["criteria"] = {
        name: {
            "raw_p": result["p_values"][name],
            "holm_adjusted_p": adjusted[name],
            "confirmed_at_0_05": adjusted[name] < 0.05,
        }
        for name in BOOTSTRAP_STATISTICS
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
