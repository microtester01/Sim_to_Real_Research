import unittest

import numpy as np

from kitti_factorial_bootstrap import RunMetricCache, holm_adjust
from kitti_two_class_eval import calculate_metrics


class KittiFactorialBootstrapTests(unittest.TestCase):
    def test_weighted_ap_matches_explicitly_resampled_images(self):
        image_ids = ["a", "b"]
        annotations = {
            "a": ({"Pedestrian": [[0, 0, 10, 10]], "Cyclist": []}, [], []),
            "b": ({"Pedestrian": [], "Cyclist": [[20, 20, 30, 30]]}, [], []),
        }
        predictions = {
            "a": [(0, 0.9, [0, 0, 10, 10])],
            "b": [(1, 0.8, [20, 20, 30, 30])],
        }
        cache = RunMetricCache(image_ids, predictions, annotations)
        weighted_map = cache.map_for_image_counts(np.asarray([[2, 1]]))[0]

        expanded_annotations = {
            "a_copy0": annotations["a"],
            "a_copy1": annotations["a"],
            "b_copy0": annotations["b"],
        }
        expanded_predictions = {
            "a_copy0": predictions["a"],
            "a_copy1": predictions["a"],
            "b_copy0": predictions["b"],
        }
        expanded_map = calculate_metrics(
            expanded_predictions, expanded_annotations
        )["mAP_50_95"]

        self.assertAlmostEqual(weighted_map, expanded_map, places=12)

    def test_holm_adjustment_matches_step_down_example(self):
        adjusted = holm_adjust(
            {
                "first": 0.01,
                "second": 0.03,
                "third": 0.04,
            }
        )
        self.assertEqual(adjusted, {"first": 0.03, "second": 0.06, "third": 0.06})


if __name__ == "__main__":
    unittest.main()
