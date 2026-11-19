import json
import tempfile
import unittest
from pathlib import Path

from carla_validation_eval import (
    SEEDS,
    VARIANTS,
    annotations_from_payload,
    ground_truth_payload,
    load_ground_truth,
    parse_yolo_label,
    predictions_from_payload,
    save_json_gzip,
    score_cached_predictions,
)


class CarlaValidationEvaluationTests(unittest.TestCase):
    def test_yolo_rows_convert_classes_and_ignore_regions_to_pixel_xyxy(self):
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "frame.txt"
            label.write_text(
                "\n".join(
                    (
                        "0 0.5 0.5 0.2 0.4",
                        "1 0.25 0.25 0.1 0.2",
                        "2 0.75 0.75 0.1 0.2",
                        "3 0.5 0.75 0.2 0.1",
                    )
                )
                + "\n",
                encoding="ascii",
            )
            ground_truth, dontcare, person_sitting = parse_yolo_label(
                label, image_width=1280, image_height=720
            )

        self.assertEqual(
            [[round(value, 6) for value in box] for box in ground_truth["Pedestrian"]],
            [[512.0, 216.0, 768.0, 504.0]],
        )
        self.assertEqual(
            [[round(value, 6) for value in box] for box in ground_truth["Cyclist"]],
            [[256.0, 108.0, 384.0, 252.0]],
        )
        self.assertEqual(
            [[round(value, 6) for value in box] for box in dontcare],
            [[896.0, 468.0, 1024.0, 612.0]],
        )
        self.assertEqual(
            [[round(value, 6) for value in box] for box in person_sitting],
            [[512.0, 504.0, 768.0, 576.0]],
        )

    def test_ground_truth_payload_preserves_frame_ids_and_ignore_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory)
            (labels / "frame-a.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n1 0.25 0.25 0.1 0.1\n",
                encoding="ascii",
            )
            annotations = load_ground_truth(["frame-a"], labels)
            payload = ground_truth_payload(["frame-a"], annotations)
            serialized = json.loads(json.dumps(payload))
            restored = annotations_from_payload(serialized, ["frame-a"])

        self.assertIn("frame-a", restored)
        self.assertEqual(
            [
                [round(value, 6) for value in box]
                for box in restored["frame-a"][0]["Pedestrian"]
            ],
            [[512.0, 288.0, 768.0, 432.0]],
        )
        self.assertEqual(
            [
                [round(value, 6) for value in box]
                for box in restored["frame-a"][0]["Cyclist"]
            ],
            [[256.0, 144.0, 384.0, 216.0]],
        )

    def test_prediction_loader_requires_exact_image_ids_and_valid_boxes(self):
        payload = {
            "format_version": 1,
            "coordinate_format": "pixel_xyxy",
            "image_ids": ["frame-a"],
            "predictions": {
                "frame-a": [
                    {
                        "class_id": 1,
                        "confidence": 0.8,
                        "bbox_xyxy": [1.0, 2.0, 11.0, 22.0],
                    },
                    {
                        "class_id": 0,
                        "confidence": 0.1,
                        "bbox_xyxy": [12.0, 22.0, 12.0, 22.0],
                    },
                ]
            },
        }
        loaded = predictions_from_payload(payload, ["frame-a"])
        self.assertEqual(
            loaded["frame-a"],
            [
                (1, 0.8, [1.0, 2.0, 11.0, 22.0]),
                (0, 0.1, [12.0, 22.0, 12.0, 22.0]),
            ],
        )
        with self.assertRaises(ValueError):
            predictions_from_payload(payload, ["frame-b"])

    def test_parser_rejects_unknown_class_and_out_of_range_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "invalid.txt"
            for row in ("4 0.5 0.5 0.2 0.2\n", "0 0.5 0.5 1.2 0.2\n"):
                label.write_text(row, encoding="ascii")
                with self.assertRaises(ValueError):
                    parse_yolo_label(label, image_width=1280, image_height=720)

    def test_cached_predictions_recompute_metrics_without_checkpoints(self):
        image_ids = ["frame-a"]
        annotations = {
            "frame-a": (
                {
                    "Pedestrian": [[10.0, 10.0, 30.0, 50.0]],
                    "Cyclist": [[60.0, 10.0, 90.0, 50.0]],
                },
                [],
                [],
            )
        }
        prediction_rows = [
            {"class_id": 0, "confidence": 0.9, "bbox_xyxy": [10, 10, 30, 50]},
            {"class_id": 1, "confidence": 0.8, "bbox_xyxy": [60, 10, 90, 50]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_json_gzip(
                output_dir / "carla_validation_ground_truth.json.gz",
                ground_truth_payload(image_ids, annotations),
            )
            for variant in VARIANTS:
                for seed in SEEDS:
                    run_id = "{}_seed{}".format(variant, seed)
                    save_json_gzip(
                        output_dir / "predictions" / (run_id + "_predictions.json.gz"),
                        {
                            "format_version": 1,
                            "coordinate_format": "pixel_xyxy",
                            "image_ids": image_ids,
                            "checkpoint": {
                                "checkpoint_id": run_id,
                                "checkpoint_sha256": "a" * 64,
                            },
                            "predictions": {"frame-a": prediction_rows},
                        },
                    )

            report = score_cached_predictions(output_dir, image_ids)

        self.assertEqual(len(report["runs"]), 12)
        self.assertAlmostEqual(report["runs"]["Baseline_seed40"]["mAP_50_95"], 1.0)


if __name__ == "__main__":
    unittest.main()
