import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from carla_dataset import (
    bbox_iou,
    find_actor_mask,
    prepare_output,
    write_dataset_yaml,
    yolo_row,
)


class CarlaDatasetTests(unittest.TestCase):
    def test_bbox_iou_is_one_for_equal_boxes(self):
        self.assertEqual(bbox_iou([2, 3, 10, 12], [2, 3, 10, 12]), 1.0)

    def test_semantic_mask_is_clipped_to_projected_actor_box(self):
        width = 100
        height = 100
        raw = np.zeros((height, width, 4), dtype=np.uint8)
        raw[15:45, 20:71, 2] = 13
        raw[45:80, 20:71, 2] = 19
        match = find_actor_mask(
            raw.tobytes(), width, height, [10, 10, 89, 89], {13, 19}
        )
        self.assertIsNotNone(match)
        self.assertEqual(match["bbox"], [20, 15, 70, 79])
        self.assertTrue(match["accepted"])

    def test_semantic_mask_ignores_non_target_classes(self):
        raw = np.zeros((100, 100, 4), dtype=np.uint8)
        raw[15:80, 20:71, 2] = 1
        self.assertIsNone(
            find_actor_mask(raw.tobytes(), 100, 100, [10, 10, 89, 89], {12})
        )

    def test_rejected_visible_fragment_is_retained_as_ignore_region(self):
        raw = np.zeros((100, 100, 4), dtype=np.uint8)
        raw[15:35, 20:30, 2] = 13
        match = find_actor_mask(
            raw.tobytes(), 100, 100, [10, 10, 89, 89], {13, 19}
        )
        self.assertIsNotNone(match)
        self.assertFalse(match["accepted"])
        ignore_row = yolo_row(2, match["bbox"], 100, 100)
        self.assertEqual(ignore_row.split()[0], "2")

    def test_dataset_yaml_reserves_ignore_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_dataset_yaml(output)
            content = (output / "dataset.yaml").read_text(encoding="utf-8")
        self.assertIn("2: IgnoreDontCare", content)
        self.assertIn("3: IgnorePersonSitting", content)

    def test_legacy_dataset_cannot_be_extended_with_mixed_label_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "manifest.json").write_text(
                json.dumps({"episodes": []}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "predate ignored-visible-object"):
                prepare_output(output)

    def test_yolo_row_is_normalized_and_uses_class_id(self):
        row = yolo_row(1, [10, 20, 29, 59], 100, 100)
        values = row.split()
        self.assertEqual(values[0], "1")
        self.assertEqual(
            [float(value) for value in values[1:]],
            [0.2, 0.4, 0.2, 0.4],
        )


if __name__ == "__main__":
    unittest.main()
