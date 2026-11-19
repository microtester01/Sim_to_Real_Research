import tempfile
import unittest
from pathlib import Path

from prepare_kitti_training_data import convert_kitti_annotations, nested_group_subsets


class PrepareKittiTrainingDataTests(unittest.TestCase):
    def test_module_imports_and_annotation_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            labels_path = Path(directory) / "000001.txt"
            labels_path.write_text(
                "Pedestrian 0 0 0 10 20 30 40\n"
                "DontCare 0 0 0 50 60 70 80\n",
                encoding="ascii",
            )

            rows, counts = convert_kitti_annotations(labels_path, 100, 100)

        self.assertEqual(
            rows,
            [
                "0 0.20000000 0.30000000 0.20000000 0.20000000",
                "2 0.60000000 0.70000000 0.20000000 0.20000000",
            ],
        )
        self.assertEqual(counts, {"Pedestrian": 1, "DontCare": 1})

    def test_nested_subsets_keep_groups_whole(self):
        subsets = nested_group_subsets([["a", "b"], ["c"], ["d", "e"]])
        for subset in subsets.values():
            image_ids = set(subset["image_ids"])
            for group in ({"a", "b"}, {"c"}, {"d", "e"}):
                self.assertTrue(not (image_ids & group) or group <= image_ids)


if __name__ == "__main__":
    unittest.main()
