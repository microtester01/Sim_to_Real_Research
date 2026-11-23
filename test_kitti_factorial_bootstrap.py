import unittest

import numpy as np

from kitti_factorial_bootstrap import (
    RunMetricCache,
    holm_adjust,
    image_group_indices,
    sample_crossed_bootstrap_draws,
)
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

    def test_equal_confidence_detections_are_accumulated_as_one_score_block(self):
        image_ids = ["positive", "negative"]
        annotations = {
            "positive": (
                {
                    "Pedestrian": [[0, 0, 10, 10]],
                    "Cyclist": [[30, 30, 40, 40]],
                },
                [],
                [],
            ),
            "negative": (
                {"Pedestrian": [], "Cyclist": []},
                [],
                [],
            ),
        }
        predictions = {
            "positive": [
                (0, 0.5, [0, 0, 10, 10]),
                (1, 0.9, [30, 30, 40, 40]),
            ],
            "negative": [(0, 0.5, [20, 20, 30, 30])],
        }
        cache = RunMetricCache(image_ids, predictions, annotations)
        counts = np.asarray([[2, 1]], dtype=np.float64)

        tied_map = cache.map_for_image_counts(counts, group_ties=True)[0]
        stable_order_map = cache.map_for_image_counts(counts, group_ties=False)[0]

        self.assertAlmostEqual(tied_map, 5.0 / 6.0, places=12)
        self.assertAlmostEqual(stable_order_map, 1.0, places=12)

    def test_primary_point_estimate_matches_bootstrap_kernel_at_unit_weights(self):
        image_ids = ["positive", "negative"]
        annotations = {
            "positive": (
                {
                    "Pedestrian": [[0, 0, 10, 10]],
                    "Cyclist": [[30, 30, 40, 40]],
                },
                [],
                [],
            ),
            "negative": (
                {"Pedestrian": [], "Cyclist": []},
                [],
                [],
            ),
        }
        predictions = {
            "positive": [
                (0, 0.5, [0, 0, 10, 10]),
                (1, 0.9, [30, 30, 40, 40]),
            ],
            "negative": [(0, 0.5, [20, 20, 30, 30])],
        }
        cache = RunMetricCache(image_ids, predictions, annotations)
        unit_weights = np.ones((1, len(image_ids)), dtype=np.float64)

        primary_point_estimate = cache.map_for_image_counts(
            unit_weights, group_ties=True
        )[0]
        bootstrap_kernel_estimate = cache.map_for_image_counts(unit_weights)[0]
        class_ap = cache.class_ap_for_image_counts(unit_weights, group_ties=True)

        self.assertAlmostEqual(primary_point_estimate, 0.75, places=12)
        self.assertAlmostEqual(primary_point_estimate, bootstrap_kernel_estimate, places=12)
        self.assertAlmostEqual(primary_point_estimate, np.mean(class_ap[:, 0]), places=12)

    def test_group_map_is_a_complete_partition_of_test_images(self):
        groups = {
            "image_count": 3,
            "group_count": 2,
            "groups": [
                {"group_id": "g0", "image_ids": ["a", "b"]},
                {"group_id": "g1", "image_ids": ["c"]},
            ],
        }

        indices = image_group_indices(["a", "b", "c"], groups)

        self.assertEqual(indices.tolist(), [0, 0, 1])
        groups["groups"][1]["image_ids"] = ["b", "c"]
        with self.assertRaises(ValueError):
            image_group_indices(["a", "b", "c"], groups)

    def test_crossed_draws_use_one_group_weight_vector_per_replicate(self):
        rng = np.random.default_rng(42)
        group_index_by_image = np.asarray([0, 0, 1], dtype=np.int32)

        selected_seeds, image_weights = sample_crossed_bootstrap_draws(
            rng, draws=5, seed_count=3, image_group_indices=group_index_by_image,
            group_count=2,
        )

        self.assertEqual(selected_seeds.shape, (5, 3))
        self.assertEqual(image_weights.shape, (5, 3))
        np.testing.assert_array_equal(image_weights[:, 0], image_weights[:, 1])

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
