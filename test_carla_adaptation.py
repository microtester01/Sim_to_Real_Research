import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from carla_adaptation import (
    DomainDiscriminator,
    IgnoreAwareDetectionLoss,
    P4_CHANNELS,
    P4_INDEX,
    P4_STRIDE,
    UnlabeledImageDataset,
    apply_domain_randomization,
    batchnorm_running_stats_frozen,
    export_detector_state_dict,
    gradient_direction_check,
    lambda_schedule,
    source_batch_image_ids,
    training_schedule,
)
from prepare_kitti_training_data import (
    convert_kitti_annotations,
    nested_group_subsets,
)


class _FakeDetectHead(nn.Module):
    def __init__(self, nc=2):
        super().__init__()
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        self.nc = nc
        self.reg_max = 16
        self.no = self.nc + 4 * self.reg_max


class _FakeDetectionModel(nn.Module):
    def __init__(self, nc=2):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.model = nn.ModuleList([_FakeDetectHead(nc)])
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)


class CarlaAdaptationTests(unittest.TestCase):
    def test_grl_reverses_only_detector_feature_gradient(self):
        result = gradient_direction_check(0.5)
        self.assertAlmostEqual(result["feature_gradient_scale"], 0.5, places=6)
        self.assertLess(result["feature_gradient_dot"], 0.0)
        self.assertTrue(result["discriminator_gradient_equal"])
        zero = gradient_direction_check(0.0)
        self.assertEqual(zero["feature_gradient_scale"], 0.0)
        self.assertEqual(zero["feature_gradient_dot"], 0.0)

    def test_p4_layer_matches_yolov8s_p4_geometry(self):
        self.assertEqual(P4_INDEX, 18)
        self.assertEqual(P4_CHANNELS, 256)
        self.assertEqual(P4_STRIDE, 16)
        self.assertEqual(640 // P4_STRIDE, 40)

    def test_domain_discriminator_normalizes_large_pooled_features(self):
        torch.manual_seed(19)
        discriminator = DomainDiscriminator()
        discriminator.eval()
        feature_maps = torch.randn(4, P4_CHANNELS, 4, 4)
        ordinary_logits = discriminator(feature_maps, 1.0)
        large_logits = discriminator(feature_maps * 1_000_000.0, 1.0)
        self.assertTrue(torch.isfinite(large_logits).all().item())
        self.assertTrue(torch.allclose(ordinary_logits, large_logits, atol=1e-4))
        large_features = (feature_maps * 1_000_000.0).requires_grad_()
        discriminator(large_features, 1.0).sum().backward()
        self.assertIsNotNone(large_features.grad)
        self.assertTrue(torch.isfinite(large_features.grad).all().item())

    def test_lambda_schedule_reaches_zero_and_maximum(self):
        self.assertEqual(lambda_schedule(0.0, 1.0), 0.0)
        self.assertGreater(lambda_schedule(1.0, 1.0), 0.999)
        self.assertLess(lambda_schedule(1.0, 1.0), 1.0)

    def test_fixed_update_schedule_handles_different_dataset_sizes(self):
        source_size_schedule = training_schedule(940, 100, 0, 94000)
        larger_dataset_schedule = training_schedule(1122, 100, 0, 94000)
        self.assertEqual(source_size_schedule, (100, 940, 94000))
        self.assertEqual(larger_dataset_schedule, (84, 1122, 94000))

    def test_smoke_batch_cap_produces_exact_update_budget(self):
        self.assertEqual(training_schedule(940, 1, 2, 0), (1, 2, 2))

    def test_unlabeled_dataset_returns_only_image_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            image_dir = Path(directory) / "images"
            label_dir = Path(directory) / "labels"
            image_dir.mkdir()
            label_dir.mkdir()
            Image.fromarray(np.full((20, 40, 3), 127, dtype=np.uint8)).save(
                image_dir / "frame.jpg"
            )
            (label_dir / "frame.txt").write_text("0 0.5 0.5 0.2 0.4\n")
            dataset = UnlabeledImageDataset(image_dir, 64)
            sample = dataset[0]
            self.assertIsInstance(sample, torch.Tensor)
            self.assertEqual(tuple(sample.shape), (3, 64, 64))
            self.assertEqual(sample.dtype, torch.uint8)

    def test_batchnorm_target_forward_freezes_running_stats_and_restores_mode(self):
        module = nn.BatchNorm2d(3)
        module.train()
        running_mean = module.running_mean.clone()
        values = torch.randn(4, 3, 8, 8, requires_grad=True)
        with batchnorm_running_stats_frozen(module):
            self.assertFalse(module.training)
            output = module(values)
            output.sum().backward()
        self.assertTrue(module.training)
        self.assertTrue(torch.equal(module.running_mean, running_mean))
        self.assertIsNotNone(module.weight.grad)
        self.assertIsNotNone(values.grad)

    def test_domain_randomization_preserves_batch_shape_and_range(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        images = torch.full((2, 3, 32, 32), 128, dtype=torch.uint8)
        randomized = apply_domain_randomization(images)
        self.assertEqual(tuple(randomized.shape), (2, 3, 32, 32))
        self.assertTrue(torch.isfinite(randomized).all().item())
        self.assertGreaterEqual(float(randomized.min()), 0.0)
        self.assertLessEqual(float(randomized.max()), 1.0)

    def test_kitti_conversion_preserves_target_and_ignore_boxes(self):
        with tempfile.TemporaryDirectory() as directory:
            label_path = Path(directory) / "sample.txt"
            label_path.write_text(
                "\n".join(
                    (
                        "Pedestrian 0 0 0 10 20 30 60 0 0 0 0 0 0 0",
                        "Cyclist 0 0 0 20 10 60 50 0 0 0 0 0 0 0",
                        "Person_sitting 0 0 0 40 20 80 80 0 0 0 0 0 0 0",
                        "DontCare -1 -1 -1 0 0 20 20 -1 -1 -1 -1000 -1000 -1000 -10",
                        "Car 0 0 0 10 10 90 90 0 0 0 0 0 0 0",
                    )
                ),
                encoding="ascii",
            )
            rows, counts = convert_kitti_annotations(label_path, 100, 100)
        self.assertEqual([row.split()[0] for row in rows], ["0", "1", "3", "2"])
        self.assertEqual(counts["Pedestrian"], 1)
        self.assertEqual(counts["Cyclist"], 1)
        self.assertEqual(counts["Person_sitting"], 1)
        self.assertEqual(counts["DontCare"], 1)

    def test_training_subsets_are_nested_and_group_preserving(self):
        groups = [["a", "b"], ["c"], ["d", "e", "f"], ["g"]]
        subsets = nested_group_subsets(groups, seed=2026, fractions=(0.2, 0.5, 1.0))
        small = set(subsets[0.2]["image_ids"])
        medium = set(subsets[0.5]["image_ids"])
        full = set(subsets[1.0]["image_ids"])
        self.assertTrue(small.issubset(medium))
        self.assertTrue(medium.issubset(full))
        self.assertEqual(full, {"a", "b", "c", "d", "e", "f", "g"})
        for selected_ids in (small, medium, full):
            for group in groups:
                self.assertTrue(not selected_ids.intersection(group) or set(group).issubset(selected_ids))
        for subset in subsets.values():
            saved_ids = {
                image_id
                for group in subset["groups"]
                for image_id in group["image_ids"]
            }
            self.assertEqual(saved_ids, set(subset["image_ids"]))

    def test_detector_state_dict_export_is_strictly_reloadable(self):
        model = _FakeDetectionModel()
        model.half()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detector.pt"
            export_detector_state_dict(model, path)
            saved = torch.load(str(path), map_location="cpu", weights_only=True)
        self.assertEqual(set(saved), set(model.state_dict()))
        _FakeDetectionModel().load_state_dict(saved, strict=True)

    def test_source_batch_ids_are_required_and_keep_sampler_order(self):
        batch = {
            "im_file": [
                r"C:\data\image_b.png",
                r"C:\data\image_a.png",
            ]
        }
        self.assertEqual(source_batch_image_ids(batch), ["image_b", "image_a"])
        with self.assertRaises(ValueError):
            source_batch_image_ids({"img": torch.zeros((1, 3, 32, 32))})

    def test_ignore_aware_loss_does_not_train_background_inside_ignore_box(self):
        criterion = IgnoreAwareDetectionLoss(_FakeDetectionModel())
        features = [
            torch.zeros((1, 66, 8, 8), requires_grad=True),
            torch.zeros((1, 66, 4, 4), requires_grad=True),
            torch.zeros((1, 66, 2, 2), requires_grad=True),
        ]
        batch = {
            "batch_idx": torch.tensor([0.0]),
            "cls": torch.tensor([[2.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
        }
        total_loss, components = criterion(features, batch)
        self.assertTrue(torch.isfinite(total_loss))
        self.assertTrue(torch.isfinite(components).all())
        total_loss.backward()
        self.assertTrue(
            torch.equal(
                features[0].grad[0, 64:66, 3, 3],
                torch.zeros_like(features[0].grad[0, 64:66, 3, 3]),
            )
        )
        self.assertGreater(
            float(features[0].grad[0, 64:66, 0, 0].abs().sum()), 0.0
        )

    def test_person_sitting_masks_pedestrian_but_not_cyclist_background(self):
        criterion = IgnoreAwareDetectionLoss(_FakeDetectionModel())
        features = [
            torch.zeros((1, 66, 8, 8), requires_grad=True),
            torch.zeros((1, 66, 4, 4), requires_grad=True),
            torch.zeros((1, 66, 2, 2), requires_grad=True),
        ]
        batch = {
            "batch_idx": torch.tensor([0.0]),
            "cls": torch.tensor([[3.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
        }
        total_loss, _components = criterion(features, batch)
        total_loss.backward()
        self.assertEqual(float(features[0].grad[0, 64, 3, 3]), 0.0)
        self.assertGreater(float(features[0].grad[0, 65, 3, 3].abs()), 0.0)

    def test_ignore_aware_loss_handles_images_without_ground_truth_boxes(self):
        criterion = IgnoreAwareDetectionLoss(_FakeDetectionModel(nc=4))
        features = [
            torch.zeros((1, 68, 8, 8), requires_grad=True),
            torch.zeros((1, 68, 4, 4), requires_grad=True),
            torch.zeros((1, 68, 2, 2), requires_grad=True),
        ]
        batch = {
            "batch_idx": torch.empty((0,)),
            "cls": torch.empty((0, 1)),
            "bboxes": torch.empty((0, 4)),
        }

        total_loss, components = criterion(features, batch)

        self.assertTrue(torch.isfinite(total_loss))
        self.assertTrue(torch.isfinite(components).all())
        total_loss.backward()
        self.assertTrue(torch.isfinite(features[0].grad).all())


if __name__ == "__main__":
    unittest.main()
