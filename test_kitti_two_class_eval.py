import tempfile
import unittest
from pathlib import Path

import torch
from ultralytics.utils.ops import non_max_suppression

from kitti_two_class_eval import (
    AGNOSTIC_NMS,
    MAX_DETECTIONS,
    NMS_IOU_THRESHOLD,
    PREDICTION_CONFIDENCE_FLOOR,
    calculate_metrics,
    parse_kitti_annotation,
)


def annotation_line(class_name, x1, y1, x2, y2):
    return "{} 0.00 0 -1.57 {} {} {} {} 1.7 0.5 0.4 1.0 2.0 10.0 0.0".format(
        class_name, x1, y1, x2, y2
    )


class KittiTwoClassEvaluationTests(unittest.TestCase):
    def test_confidence_nms_and_detection_cap_are_fixed(self):
        self.assertEqual(PREDICTION_CONFIDENCE_FLOOR, 0.001)
        self.assertEqual(NMS_IOU_THRESHOLD, 0.7)
        self.assertEqual(MAX_DETECTIONS, 300)
        self.assertFalse(AGNOSTIC_NMS)

    def test_ultralytics_nms_confidence_class_awareness_and_max_det(self):
        prediction = torch.tensor(
            [
                [
                    [50.0, 50.0, 50.0, 50.0],
                    [50.0, 50.0, 50.0, 50.0],
                    [20.0, 20.0, 20.0, 20.0],
                    [20.0, 20.0, 20.0, 20.0],
                    [0.90, 0.80, 0.01, 0.0005],
                    [0.01, 0.02, 0.85, 0.0004],
                ]
            ]
        )
        kept = non_max_suppression(
            prediction,
            conf_thres=PREDICTION_CONFIDENCE_FLOOR,
            iou_thres=NMS_IOU_THRESHOLD,
            agnostic=AGNOSTIC_NMS,
            max_det=MAX_DETECTIONS,
        )[0]
        self.assertEqual(len(kept), 2)
        self.assertEqual(set(kept[:, 5].tolist()), {0.0, 1.0})
        self.assertTrue(torch.all(kept[:, 4] >= PREDICTION_CONFIDENCE_FLOOR))

        count = MAX_DETECTIONS + 1
        boxes = torch.zeros((1, 6, count))
        boxes[0, 0] = torch.arange(count, dtype=torch.float32) * 20 + 10
        boxes[0, 1] = 10
        boxes[0, 2:4] = 10
        boxes[0, 4] = torch.linspace(0.99, 0.5, count)
        limited = non_max_suppression(
            boxes,
            conf_thres=PREDICTION_CONFIDENCE_FLOOR,
            iou_thres=NMS_IOU_THRESHOLD,
            agnostic=AGNOSTIC_NMS,
            max_det=MAX_DETECTIONS,
        )[0]
        self.assertEqual(len(limited), MAX_DETECTIONS)

    def test_parser_separates_targets_and_ignored_regions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "000001.txt"
            path.write_text(
                "\n".join(
                    (
                        annotation_line("Pedestrian", 10, 10, 30, 50),
                        annotation_line("Person_sitting", 40, 10, 60, 50),
                        annotation_line("DontCare", 70, 10, 90, 50),
                    )
                ),
                encoding="ascii",
            )
            gt, dontcare, ignored_pedestrian = parse_kitti_annotation(path)
        self.assertEqual(len(gt["Pedestrian"]), 1)
        self.assertEqual(len(gt["Cyclist"]), 0)
        self.assertEqual(len(dontcare), 1)
        self.assertEqual(len(ignored_pedestrian), 1)

    def test_perfect_detection_has_one_point_map(self):
        annotations = {
            "a": (
                {"Pedestrian": [[10, 10, 30, 50]], "Cyclist": []},
                [],
                [],
            )
        }
        predictions = {"a": [(0, 0.9, [10, 10, 30, 50])]}
        metrics = calculate_metrics(predictions, annotations)
        self.assertAlmostEqual(metrics["classes"]["Pedestrian"]["mAP_50_95"], 1.0)
        self.assertEqual(metrics["classes"]["Pedestrian"]["ground_truth_count"], 1)
        self.assertIsNone(metrics["mAP_50_95"])

    def test_dontcare_and_person_sitting_predictions_are_ignored_for_pedestrian(self):
        annotations = {
            "a": (
                {"Pedestrian": [[10, 10, 30, 50]], "Cyclist": []},
                [[70, 10, 90, 50]],
                [[40, 10, 60, 50]],
            )
        }
        predictions = {
            "a": [
                (0, 0.9, [10, 10, 30, 50]),
                (0, 0.8, [40, 10, 60, 50]),
                (0, 0.7, [70, 10, 90, 50]),
            ]
        }
        metrics = calculate_metrics(predictions, annotations)
        pedestrian = metrics["classes"]["Pedestrian"]
        self.assertAlmostEqual(pedestrian["mAP_50_95"], 1.0)
        self.assertEqual(pedestrian["ignored_detections"], 2)

    def test_duplicate_detection_is_false_positive_not_ignored(self):
        annotations = {
            "a": (
                {"Pedestrian": [[10, 10, 30, 50]], "Cyclist": []},
                [[10, 10, 30, 50]],
                [],
            )
        }
        predictions = {
            "a": [
                (0, 0.9, [10, 10, 30, 50]),
                (0, 0.8, [10, 10, 30, 50]),
            ]
        }
        metrics = calculate_metrics(predictions, annotations)
        pedestrian = metrics["classes"]["Pedestrian"]
        self.assertEqual(pedestrian["true_positives"], 1)
        self.assertEqual(pedestrian["false_positives"], 1)
        self.assertEqual(pedestrian["ignored_detections"], 0)

    def test_match_uses_next_best_unmatched_ground_truth(self):
        annotations = {
            "a": (
                {
                    "Pedestrian": [
                        [10, 10, 30, 50],
                        [15, 10, 35, 50],
                    ],
                    "Cyclist": [],
                },
                [],
                [],
            )
        }
        predictions = {
            "a": [
                (0, 0.9, [10, 10, 30, 50]),
                (0, 0.8, [10, 10, 35, 50]),
            ]
        }
        metrics = calculate_metrics(predictions, annotations)
        pedestrian = metrics["classes"]["Pedestrian"]
        self.assertEqual(pedestrian["true_positives"], 2)
        self.assertEqual(pedestrian["false_positives"], 0)

    def test_person_sitting_is_not_ignored_for_cyclist(self):
        annotations = {
            "a": (
                {"Pedestrian": [], "Cyclist": []},
                [],
                [[10, 10, 30, 50]],
            ),
            "b": (
                {"Pedestrian": [], "Cyclist": [[10, 10, 30, 50]]},
                [],
                [],
            ),
        }
        predictions = {
            "a": [(1, 0.9, [10, 10, 30, 50])],
            "b": [(1, 0.8, [10, 10, 30, 50])],
        }
        metrics = calculate_metrics(predictions, annotations)
        cyclist = metrics["classes"]["Cyclist"]
        self.assertEqual(cyclist["true_positives"], 1)
        self.assertEqual(cyclist["false_positives"], 1)
        self.assertEqual(cyclist["ignored_detections"], 0)


if __name__ == "__main__":
    unittest.main()
