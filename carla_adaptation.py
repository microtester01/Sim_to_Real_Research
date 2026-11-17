import argparse
import copy
import hashlib
import json
import math
import random
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ColorJitter

import ultralytics
from ultralytics.cfg import DEFAULT_CFG, get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


P4_INDEX = 18
P4_CHANNELS = 256
P4_STRIDE = 16
DOMAIN_DISCRIMINATOR_MAX_GRAD_NORM = 10.0
DEFAULT_DOMAIN_LEARNING_RATE = 0.0001
CLASS_NAMES = {0: "Pedestrian", 1: "Cyclist"}
SOURCE_LABEL_NAMES = {
    **CLASS_NAMES,
    2: "IgnoreDontCare",
    3: "IgnorePersonSitting",
}


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, features, coefficient):
        ctx.coefficient = coefficient
        return features.view_as(features)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


def reverse_gradient(features, coefficient):
    return GradientReversal.apply(features, coefficient)


def gradient_direction_check(coefficient=0.5):
    discriminator = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        discriminator.weight.copy_(
            torch.tensor([[0.5, -0.25, 0.75], [-0.3, 0.4, -0.6]])
        )
    values = torch.tensor(
        [[0.2, 0.7, -0.3], [0.9, -0.1, 0.4]], dtype=torch.float32
    )
    labels = torch.tensor([0, 1])

    ordinary_features = values.clone().requires_grad_(True)
    ordinary_logits = discriminator(ordinary_features)
    ordinary_loss = F.cross_entropy(ordinary_logits, labels)
    ordinary_feature_grad, ordinary_head_grad = torch.autograd.grad(
        ordinary_loss,
        (ordinary_features, discriminator.weight),
        retain_graph=False,
    )

    reversed_features = values.clone().requires_grad_(True)
    reversed_logits = discriminator(reverse_gradient(reversed_features, coefficient))
    reversed_loss = F.cross_entropy(reversed_logits, labels)
    reversed_feature_grad, reversed_head_grad = torch.autograd.grad(
        reversed_loss,
        (reversed_features, discriminator.weight),
        retain_graph=False,
    )

    expected = -coefficient * ordinary_feature_grad
    if not torch.allclose(reversed_feature_grad, expected, rtol=1e-5, atol=1e-7):
        raise AssertionError("GRL feature gradient is not -lambda times the CE gradient")
    if not torch.allclose(reversed_head_grad, ordinary_head_grad, rtol=1e-5, atol=1e-7):
        raise AssertionError("GRL changed the discriminator parameter gradient")
    gradient_dot = torch.sum(reversed_feature_grad * ordinary_feature_grad).item()
    if coefficient == 0.0 and not torch.allclose(
        reversed_feature_grad, torch.zeros_like(reversed_feature_grad)
    ):
        raise AssertionError("A zero GRL coefficient must zero the feature gradient")
    if coefficient > 0.0 and gradient_dot >= 0:
        raise AssertionError("GRL feature-gradient direction is not reversed")
    return {
        "coefficient": coefficient,
        "feature_gradient_scale": float(
            reversed_feature_grad.norm() / ordinary_feature_grad.norm()
        ),
        "feature_gradient_dot": float(
            gradient_dot
        ),
        "discriminator_gradient_equal": True,
    }


class DomainDiscriminator(nn.Module):
    def __init__(self, channels=P4_CHANNELS):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_norm = nn.LayerNorm(channels, elementwise_affine=False)
        self.classifier = nn.Sequential(
            nn.Linear(channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 2),
        )

    def forward(self, feature_map, coefficient):
        pooled = self.pool(feature_map).flatten(1)
        pooled = self.feature_norm(pooled)
        return self.classifier(reverse_gradient(pooled, coefficient))


def anchor_centers_in_boxes(anchor_centers, boxes, box_mask):
    inside = (
        (anchor_centers[None, :, None, 0] >= boxes[:, None, :, 0])
        & (anchor_centers[None, :, None, 1] >= boxes[:, None, :, 1])
        & (anchor_centers[None, :, None, 0] <= boxes[:, None, :, 2])
        & (anchor_centers[None, :, None, 1] <= boxes[:, None, :, 3])
        & box_mask[:, None, :]
    )
    return inside.any(dim=2)


def export_detector_state_dict(model, output_path):
    torch.save(
        {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        str(output_path),
    )


def source_batch_image_ids(source_batch):
    image_files = source_batch.get("im_file")
    if not isinstance(image_files, (list, tuple)) or not image_files:
        raise ValueError("Source batch must include image paths for pairing audit")
    return [Path(image_file).stem for image_file in image_files]


def parameter_gradient_norm(modules):
    gradients = [
        parameter.grad.detach().float().norm()
        for module in modules
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return (
        float(torch.stack(gradients).norm().item())
        if gradients
        else 0.0
    )


@contextmanager
def batchnorm_running_stats_frozen(model):
    batch_norms = [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    training_states = [module.training for module in batch_norms]
    try:
        for module in batch_norms:
            module.eval()
        yield
    finally:
        for module, training in zip(batch_norms, training_states):
            module.train(training)


class IgnoreAwareDetectionLoss(v8DetectionLoss):
    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [feature.view(feats[0].shape[0], self.no, -1) for feature in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        image_size = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
            1,
        )
        targets = self.preprocess(
            targets.to(self.device),
            batch_size,
            scale_tensor=image_size[[1, 0, 1, 0]],
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        valid_boxes = gt_bboxes.sum(2).gt(0)
        gt_classes = gt_labels.squeeze(-1)
        ignored_boxes = valid_boxes & gt_classes.ge(self.nc)
        ignored_dontcare = ignored_boxes & gt_classes.eq(self.nc)
        ignored_person_sitting = ignored_boxes & gt_classes.eq(self.nc + 1)
        target_mask = valid_boxes & gt_labels.squeeze(-1).lt(self.nc)
        assigner_labels = gt_labels.clone()
        assigner_labels[ignored_boxes.unsqueeze(-1)] = 0

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            assigner_labels,
            gt_bboxes,
            target_mask.unsqueeze(-1),
        )
        target_scores_sum = max(target_scores.sum(), 1)
        ignore_dontcare_anchors = anchor_centers_in_boxes(
            anchor_points * stride_tensor, gt_bboxes, ignored_dontcare
        )
        ignore_person_sitting_anchors = anchor_centers_in_boxes(
            anchor_points * stride_tensor, gt_bboxes, ignored_person_sitting
        )
        valid_classification = torch.ones_like(pred_scores, dtype=torch.bool)
        valid_classification[:, :, 0] = (
            ~ignore_dontcare_anchors & ~ignore_person_sitting_anchors
        )
        valid_classification[:, :, 1] = ~ignore_dontcare_anchors
        valid_classification |= fg_mask.bool().unsqueeze(-1)
        classification_loss = self.bce(pred_scores, target_scores.to(dtype))
        loss[1] = (classification_loss * valid_classification).sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss.sum() * batch_size, loss.detach()


class UnlabeledImageDataset(Dataset):
    def __init__(self, image_dir, image_size, return_paths=False):
        self.image_paths = sorted(
            path
            for path in Path(image_dir).iterdir()
            if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        )
        if not self.image_paths:
            raise FileNotFoundError("No target images found in {}".format(image_dir))
        self.image_size = image_size
        self.return_paths = return_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        with Image.open(str(self.image_paths[index])) as image:
            image = image.convert("RGB")
            width, height = image.size
            scale = min(self.image_size / width, self.image_size / height)
            resized = (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            image = image.resize(resized, Image.BILINEAR)
            canvas = Image.new("RGB", (self.image_size, self.image_size), (114, 114, 114))
            offset = (
                (self.image_size - resized[0]) // 2,
                (self.image_size - resized[1]) // 2,
            )
            canvas.paste(image, offset)
            array = np.asarray(canvas, dtype=np.uint8).transpose(2, 0, 1).copy()
        tensor = torch.from_numpy(array)
        return (tensor, self.image_paths[index].stem) if self.return_paths else tensor


def apply_domain_randomization(images):
    color_jitter = ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.02,
    )
    randomized = images.float() / 255.0
    for index in range(randomized.shape[0]):
        image = randomized[index]
        if random.random() < 0.5:
            brightness = random.uniform(-0.2, 0.2)
            contrast = random.uniform(0.8, 1.2)
            image = (image * contrast + brightness).clamp(0.0, 1.0)
        if random.random() < 0.5:
            image = color_jitter(image)
        if random.random() < 0.3:
            variance = random.uniform(10.0, 50.0)
            image = (
                image + torch.randn_like(image) * math.sqrt(variance) / 255.0
            ).clamp(0.0, 1.0)
        randomized[index] = image
    return randomized


def move_source_batch(batch, device, use_dr):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    if use_dr:
        moved["img"] = apply_domain_randomization(moved["img"])
    else:
        moved["img"] = moved["img"].float() / 255.0
    return moved


def lambda_schedule(progress, maximum):
    return maximum * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


def training_schedule(loader_batches, epochs, max_batches, max_updates):
    if loader_batches <= 0:
        raise ValueError("Source loader must have at least one full batch")
    if max_updates:
        return math.ceil(max_updates / loader_batches), loader_batches, max_updates
    steps_per_epoch = min(loader_batches, max_batches) if max_batches else loader_batches
    return epochs, steps_per_epoch, epochs * steps_per_epoch


def build_target_loader(image_dir, batch_size, image_size, workers, seed):
    dataset = UnlabeledImageDataset(image_dir, image_size, return_paths=True)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        generator=generator,
    )


def train(args):
    if args.configuration not in ("Baseline", "DR", "AFA", "Combo", "UpperBound"):
        raise ValueError("Unknown configuration: {}".format(args.configuration))
    use_dr = args.configuration in ("DR", "Combo")
    use_domain = args.configuration in ("AFA", "Combo")
    if use_domain and args.target_images is None:
        raise ValueError("AFA and Combo require unlabeled target images")
    gradient_test = gradient_direction_check(args.lambda_max)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    source_data = check_det_dataset(str(args.source_yaml))
    use_ignore_aware_loss = source_data.get("nc") == 4
    if use_ignore_aware_loss:
        names = source_data.get("names", {})
        if (
            not isinstance(names, dict)
            or names.get(2) != "IgnoreDontCare"
            or names.get(3) != "IgnorePersonSitting"
        ):
            raise ValueError(
                "Four-class source data must reserve IDs 2/3 for KITTI ignore regions"
            )
    elif source_data.get("nc") != 2:
        raise ValueError(
            "Source data must contain two target classes, with optional KITTI ignore IDs"
        )
    validation_args = (
        args.validation_manifest,
        args.validation_images,
        args.validation_labels,
    )
    if any(value is not None for value in validation_args) and not all(
        value is not None for value in validation_args
    ):
        raise ValueError(
            "validation-manifest, validation-images, and validation-labels must be supplied together"
        )
    validation_image_paths = []
    if args.validation_manifest is not None:
        manifest = json.loads(args.validation_manifest.read_text(encoding="utf-8"))
        validation_ids = manifest["split_image_ids"]["val"]
        validation_image_paths = [
            args.validation_images / "{}.png".format(image_id)
            for image_id in validation_ids
        ]
        missing_validation_images = [
            path for path in validation_image_paths if not path.is_file()
        ]
        if missing_validation_images:
            raise FileNotFoundError(
                "Missing validation image: {}".format(missing_validation_images[0])
            )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = get_cfg(
        DEFAULT_CFG,
        overrides={
            "task": "detect",
            "imgsz": args.image_size,
            "batch": args.batch_size,
            "workers": args.workers,
            "rect": False,
            "cache": False,
            "single_cls": False,
            "fraction": 1.0,
            "mosaic": 1.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "seed": args.seed,
        },
    )
    source_dataset = build_yolo_dataset(
        cfg,
        source_data["train"],
        args.batch_size,
        source_data,
        mode="train",
        rect=False,
        stride=32,
    )
    source_loader = build_dataloader(
        source_dataset,
        args.batch_size,
        args.workers,
        shuffle=True,
        rank=-1,
    )
    source_loader.generator.manual_seed(args.seed)
    target_loader = (
        build_target_loader(
            args.target_images,
            args.batch_size,
            args.image_size,
            args.target_workers,
            args.seed + 1,
        )
        if use_domain
        else None
    )
    planned_epochs, steps_per_epoch, total_steps = training_schedule(
        len(source_loader), args.epochs, args.max_batches, args.max_updates
    )

    model = DetectionModel("yolov8s.yaml", nc=2, verbose=False).to(device)
    if args.initialization_weights is not None:
        initial_state = torch.load(
            str(args.initialization_weights), map_location=device, weights_only=True
        )
        model.load_state_dict(initial_state, strict=True)
    elif args.pretrained_state_dict is not None:
        pretrained_state = torch.load(
            str(args.pretrained_state_dict), map_location=device, weights_only=True
        )
        model.load_state_dict(pretrained_state, strict=True)
    model.names = CLASS_NAMES
    model.args = cfg
    if use_ignore_aware_loss:
        model.criterion = IgnoreAwareDetectionLoss(model)
    discriminator = DomainDiscriminator().to(device) if use_domain else None
    initial_state_path = (
        args.initialization_weights
        if args.initialization_weights is not None
        else args.pretrained_state_dict
    )
    initial_state_hash = (
        hashlib.sha256(initial_state_path.read_bytes()).hexdigest()
        if initial_state_path is not None
        else None
    )
    parameter_groups = [{"params": list(model.parameters()), "lr": args.learning_rate}]
    if discriminator is not None:
        parameter_groups.append(
            {
                "params": list(discriminator.parameters()),
                "lr": args.domain_learning_rate,
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)

    captured = {}

    def capture_p4(_module, _inputs, output):
        captured["p4"] = output

    p4_module = model.model[P4_INDEX]
    hook = p4_module.register_forward_hook(capture_p4) if use_domain else None
    run_dir = args.output / "{}_seed{}".format(args.configuration, args.seed)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "configuration": args.configuration,
                "seed": args.seed,
                "initialization_weights": (
                    str(args.initialization_weights.resolve())
                    if args.initialization_weights is not None
                    else None
                ),
                "pretrained_state_dict": (
                    str(args.pretrained_state_dict.resolve())
                    if args.pretrained_state_dict is not None
                    else None
                ),
                "initialization_sha256": initial_state_hash,
                "initialization_mode": (
                    "shared seed-specific random initialization"
                    if args.initialization_weights is not None
                    else "fine-tuning from supplied detector state dict"
                    if args.pretrained_state_dict is not None
                    else "seeded random initialization generated in this run"
                ),
                "export_best_state_dict": args.export_best_state_dict,
                "source_yaml": str(args.source_yaml.resolve()),
                "target_images": (
                    str(args.target_images.resolve())
                    if args.target_images is not None
                    else None
                ),
                "source_batch": args.batch_size,
                "target_batch": args.batch_size if use_domain else 0,
                "image_size": args.image_size,
                "epochs": planned_epochs,
                "requested_epochs": args.epochs,
                "max_batches": args.max_batches,
                "max_updates": args.max_updates,
                "optimizer_updates": total_steps,
                "close_mosaic_epochs": cfg.close_mosaic,
                "lambda_max": args.lambda_max if use_domain else 0.0,
                "learning_rate": args.learning_rate,
                "domain_learning_rate": (
                    args.domain_learning_rate if use_domain else None
                ),
                "domain_discriminator_max_grad_norm": (
                    DOMAIN_DISCRIMINATOR_MAX_GRAD_NORM if use_domain else None
                ),
                "weight_decay": args.weight_decay,
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "ultralytics": ultralytics.__version__,
                "device": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else "cpu"
                ),
                "p4_module_index": P4_INDEX,
                "p4_channels": P4_CHANNELS,
                "p4_spatial_size": args.image_size // P4_STRIDE,
                "grl_gradient_check": gradient_test if use_domain else None,
                "target_detection_loss": False,
                "ignore_aware_detection_loss": use_ignore_aware_loss,
                "validation_manifest": (
                    str(args.validation_manifest.resolve())
                    if args.validation_manifest is not None
                    else None
                ),
                "checkpoint_selection": (
                    "maximum internal KITTI validation mAP@0.50:0.95; earliest epoch on ties"
                    if args.validation_manifest is not None
                    else "disabled: no validation set supplied"
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    log_path = run_dir / "train.jsonl"
    model.train()
    log_stream = log_path.open("a", encoding="utf-8")
    validation_log_path = run_dir / "validation.jsonl"
    best_validation_map = -math.inf
    best_epoch = None
    (run_dir / "training_schedule.json").write_text(
        json.dumps(
            {
                "source_loader_batches": len(source_loader),
                "planned_epochs": planned_epochs,
                "optimizer_updates": total_steps,
                "updates_per_epoch_limit": steps_per_epoch,
                "max_batches_per_epoch": args.max_batches,
                "max_updates": args.max_updates,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    close_mosaic_epoch = planned_epochs - int(cfg.close_mosaic)
    completed_updates = 0
    source_order_digest = hashlib.sha256()
    try:
        for epoch in range(planned_epochs):
            epoch_steps = min(steps_per_epoch, total_steps - completed_updates)
            if epoch_steps <= 0:
                break
            if epoch == close_mosaic_epoch:
                source_dataset.close_mosaic(cfg)
                print(
                    "Mosaic disabled before epoch {}".format(epoch + 1),
                    flush=True,
                )
            target_iterator = iter(target_loader) if target_loader is not None else None
            for batch_index, source_batch in enumerate(source_loader):
                if batch_index >= epoch_steps:
                    break
                source_ids = source_batch_image_ids(source_batch)
                source_order_digest.update(
                    (json.dumps(source_ids, separators=(",", ":")) + "\n").encode(
                        "utf-8"
                    )
                )
                source_image_labels = []
                source_labels = source_batch["cls"].detach().cpu().reshape(-1).tolist()
                source_boxes = (
                    source_batch["bboxes"].detach().cpu().tolist()
                    if "bboxes" in source_batch
                    else []
                )
                for class_id, box in zip(source_labels[:12], source_boxes[:12]):
                    source_image_labels.append(
                        {
                            "class": SOURCE_LABEL_NAMES[int(class_id)],
                            "normalized_xywh": [float(value) for value in box],
                        }
                    )
                optimizer.zero_grad(set_to_none=True)
                source_batch = move_source_batch(source_batch, device, use_dr)
                detection_loss, detection_items = model.loss(source_batch)
                domain_loss = torch.zeros((), device=device)
                reversal = 0.0
                domain_accuracy: Optional[float] = None
                target_ids = []
                source_features = None
                target_features = None
                target_bn_running_stats_max_delta = 0.0

                if use_domain:
                    try:
                        target_batch = next(target_iterator)
                    except StopIteration:
                        target_iterator = iter(target_loader)
                        target_batch = next(target_iterator)
                    target_images, target_ids = target_batch
                    target_images = target_images.to(
                        device, non_blocking=True
                    ).float() / 255.0
                    source_features = captured["p4"]
                    source_features.retain_grad()
                    if (
                        source_features.ndim != 4
                        or source_features.shape[1] != P4_CHANNELS
                        or source_features.shape[2:] != (
                            args.image_size // P4_STRIDE,
                            args.image_size // P4_STRIDE,
                        )
                    ):
                        raise RuntimeError(
                            "Unexpected P4 tensor shape: {}".format(
                                tuple(source_features.shape)
                            )
                        )
                    bn_layers = [
                        module
                        for module in model.modules()
                        if isinstance(module, nn.modules.batchnorm._BatchNorm)
                    ]
                    running_stats_before = [
                        (
                            module.running_mean.detach().clone()
                            if module.running_mean is not None
                            else None,
                            module.running_var.detach().clone()
                            if module.running_var is not None
                            else None,
                        )
                        for module in bn_layers
                    ]
                    with batchnorm_running_stats_frozen(model):
                        model(target_images)
                    target_features = captured["p4"]
                    target_features.retain_grad()
                    deltas = []
                    for module, (mean_before, variance_before) in zip(
                        bn_layers, running_stats_before
                    ):
                        if mean_before is not None:
                            deltas.append(
                                (module.running_mean - mean_before).abs().max()
                            )
                        if variance_before is not None:
                            deltas.append(
                                (module.running_var - variance_before).abs().max()
                            )
                    if deltas:
                        target_bn_running_stats_max_delta = float(
                            torch.stack(deltas).max().item()
                        )
                    if target_bn_running_stats_max_delta != 0.0:
                        raise RuntimeError(
                            "Target forward changed BatchNorm running statistics"
                        )
                    progress = completed_updates / float(max(total_steps - 1, 1))
                    reversal = lambda_schedule(progress, args.lambda_max)
                    source_logits = discriminator(source_features, reversal)
                    target_logits = discriminator(target_features, reversal)
                    logits = torch.cat((source_logits, target_logits), dim=0)
                    domain_labels = torch.cat(
                        (
                            torch.zeros(
                                source_logits.shape[0],
                                dtype=torch.long,
                                device=device,
                            ),
                            torch.ones(
                                target_logits.shape[0],
                                dtype=torch.long,
                                device=device,
                            ),
                        ),
                        dim=0,
                    )
                    domain_loss = F.cross_entropy(logits, domain_labels) * args.batch_size
                    with torch.no_grad():
                        domain_accuracy = float(
                            (logits.argmax(dim=1) == domain_labels)
                            .float()
                            .mean()
                            .item()
                        )

                total_loss = detection_loss + domain_loss
                if not torch.isfinite(total_loss):
                    raise FloatingPointError(
                        "Non-finite loss at epoch {}, batch {}".format(
                            epoch + 1, batch_index
                        )
                    )
                total_loss.backward()
                source_p4_gradient_norm = (
                    float(source_features.grad.detach().float().norm().item())
                    if source_features is not None and source_features.grad is not None
                    else 0.0
                )
                target_p4_gradient_norm = (
                    float(target_features.grad.detach().float().norm().item())
                    if target_features is not None and target_features.grad is not None
                    else 0.0
                )
                detector_gradient_norm = parameter_gradient_norm((model,))
                discriminator_gradient_norm = (
                    parameter_gradient_norm((discriminator,))
                    if discriminator is not None
                    else 0.0
                )
                if use_domain and discriminator_gradient_norm <= 0.0:
                    raise RuntimeError("Domain discriminator received no gradients")
                if use_domain and reversal > 0.0 and target_p4_gradient_norm <= 0.0:
                    raise RuntimeError("Target P4 received no GRL-reversed gradient")
                if discriminator is not None:
                    torch.nn.utils.clip_grad_norm_(
                        discriminator.parameters(),
                        DOMAIN_DISCRIMINATOR_MAX_GRAD_NORM,
                    )
                optimizer.step()
                completed_updates += 1

                record = {
                    "epoch": epoch + 1,
                    "batch": batch_index + 1,
                    "source_detection_loss": float(detection_loss.detach().item()),
                    "source_box_loss": float(detection_items[0].item()),
                    "source_cls_loss": float(detection_items[1].item()),
                    "source_dfl_loss": float(detection_items[2].item()),
                    "domain_loss": float(domain_loss.detach().item()),
                    "grl_lambda": reversal,
                    "domain_accuracy": domain_accuracy,
                    "source_image_ids": source_ids,
                    "source_label_examples": source_image_labels,
                    "target_image_ids": list(target_ids),
                    "source_batch_size": len(source_ids),
                    "target_batch_size": len(target_ids),
                    "p4_shape": (
                        list(source_features.shape)
                        if source_features is not None
                        else None
                    ),
                    "source_p4_gradient_norm": source_p4_gradient_norm,
                    "target_p4_gradient_norm": target_p4_gradient_norm,
                    "detector_parameter_gradient_norm": detector_gradient_norm,
                    "discriminator_parameter_gradient_norm": discriminator_gradient_norm,
                    "target_bn_running_stats_max_delta": target_bn_running_stats_max_delta,
                }
                log_stream.write(json.dumps(record) + "\n")
                if (batch_index + 1) % args.log_every == 0:
                    log_stream.flush()
                    print(json.dumps(record), flush=True)

            checkpoint_model = copy.deepcopy(model).cpu()
            for module in checkpoint_model.modules():
                module._forward_hooks.clear()
            if hasattr(checkpoint_model, "criterion"):
                del checkpoint_model.criterion
            checkpoint_model.args = {
                "task": "detect",
                "imgsz": args.image_size,
            }
            checkpoint = {
                "model": checkpoint_model.half(),
                "ema": None,
                "updates": completed_updates,
                "train_args": {"task": "detect", "imgsz": args.image_size},
            }
            checkpoint_path = run_dir / "last.pt"
            torch.save(checkpoint, str(checkpoint_path))
            if args.validation_manifest is not None:
                from kitti_two_class_eval import evaluate_checkpoint

                validation_report = evaluate_checkpoint(
                    checkpoint_path,
                    validation_image_paths,
                    args.validation_labels,
                    image_size=args.image_size,
                    batch_size=args.validation_batch_size,
                    device="0" if device.type == "cuda" else "cpu",
                )
                validation_map = validation_report["mAP_50_95"]
                if not math.isfinite(validation_map):
                    raise ValueError(
                        "Validation mAP is undefined; verify Pedestrian/Cyclist GT coverage"
                    )
                validation_report["epoch"] = epoch + 1
                with validation_log_path.open("a", encoding="utf-8") as validation_stream:
                    validation_stream.write(json.dumps(validation_report) + "\n")
                if validation_map > best_validation_map:
                    best_validation_map = validation_map
                    best_epoch = epoch + 1
                    shutil.copyfile(checkpoint_path, run_dir / "best.pt")
                    if args.export_best_state_dict:
                        export_detector_state_dict(
                            checkpoint_model, run_dir / "best_state_dict.pt"
                        )
                    (run_dir / "best_selection.json").write_text(
                        json.dumps(
                            {
                                "selected_epoch": best_epoch,
                                "selection_metric": "internal KITTI mAP@0.50:0.95",
                                "selection_value": best_validation_map,
                                "validation_manifest": str(
                                    args.validation_manifest.resolve()
                                ),
                            },
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                print(
                    "Validation epoch {}/{}: mAP@0.50:0.95={:.6f}; best epoch={}".format(
                        epoch + 1, args.epochs, validation_map, best_epoch
                    ),
                    flush=True,
                )
            print(
                "Epoch {}/{} complete; checkpoint={}".format(
                epoch + 1, planned_epochs, checkpoint_path
                ),
                flush=True,
            )
        (run_dir / "source_sampling.json").write_text(
            json.dumps(
                {
                    "sampler_seed": args.seed,
                    "batch_size": args.batch_size,
                    "optimizer_updates": completed_updates,
                    "ordered_batch_ids_sha256": source_order_digest.hexdigest(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        log_stream.close()
        if hook is not None:
            hook.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configuration",
        choices=("Baseline", "DR", "AFA", "Combo", "UpperBound"),
        required=True,
    )
    parser.add_argument("--source-yaml", type=Path, required=True)
    parser.add_argument("--target-images", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--validation-images", type=Path)
    parser.add_argument("--validation-labels", type=Path)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=40)
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--initialization-weights", type=Path)
    initialization.add_argument("--pretrained-state-dict", type=Path)
    parser.add_argument("--export-best-state-dict", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--target-workers", type=int, default=0)
    parser.add_argument("--lambda-max", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--domain-learning-rate", type=float, default=DEFAULT_DOMAIN_LEARNING_RATE
    )
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch-size must be positive")
    if args.image_size <= 0 or args.image_size % P4_STRIDE != 0:
        parser.error("image-size must be positive and divisible by 16")
    if args.log_every <= 0 or args.workers < 0 or args.target_workers < 0:
        parser.error("log-every must be positive and worker counts cannot be negative")
    if args.max_batches < 0:
        parser.error("max-batches cannot be negative")
    if args.max_updates < 0:
        parser.error("max-updates cannot be negative")
    if args.max_updates and args.max_batches:
        parser.error("max-updates and max-batches cannot be used together")
    if args.validation_batch_size <= 0:
        parser.error("validation-batch-size must be positive")
    validation_args = (
        args.validation_manifest,
        args.validation_images,
        args.validation_labels,
    )
    if any(value is not None for value in validation_args) and not all(
        value is not None for value in validation_args
    ):
        parser.error(
            "validation-manifest, validation-images, and validation-labels must be supplied together"
        )
    if args.lambda_max < 0:
        parser.error("lambda-max cannot be negative")
    if args.domain_learning_rate <= 0:
        parser.error("domain-learning-rate must be positive")
    if args.configuration in ("AFA", "Combo") and args.batch_size < 1:
        parser.error("adaptation requires a positive source and target batch")
    args.source_yaml = args.source_yaml.resolve()
    if args.initialization_weights is not None:
        args.initialization_weights = args.initialization_weights.resolve()
        if not args.initialization_weights.is_file():
            raise FileNotFoundError(args.initialization_weights)
        initialization_manifest = (
            args.initialization_weights.parent / "initialization_manifest.json"
        )
        if not initialization_manifest.is_file():
            raise FileNotFoundError(initialization_manifest)
        initialization_metadata = json.loads(
            initialization_manifest.read_text(encoding="utf-8")
        )
        seed_metadata = initialization_metadata.get("seeds", {}).get(str(args.seed))
        actual_hash = hashlib.sha256(
            args.initialization_weights.read_bytes()
        ).hexdigest()
        if (
            seed_metadata is None
            or seed_metadata.get("file") != args.initialization_weights.name
            or seed_metadata.get("sha256") != actual_hash
        ):
            parser.error(
                "initialization weights do not match seed {} and manifest".format(
                    args.seed
                )
            )
    if args.pretrained_state_dict is not None:
        args.pretrained_state_dict = args.pretrained_state_dict.resolve()
        if not args.pretrained_state_dict.is_file():
            raise FileNotFoundError(args.pretrained_state_dict)
    if args.export_best_state_dict and args.validation_manifest is None:
        parser.error("--export-best-state-dict requires KITTI validation arguments")
    if args.target_images is not None:
        args.target_images = args.target_images.resolve()
    args.output = args.output.resolve()
    if not args.source_yaml.is_file():
        raise FileNotFoundError(args.source_yaml)
    if args.target_images is not None and not args.target_images.is_dir():
        raise FileNotFoundError(args.target_images)
    if args.configuration in ("AFA", "Combo") and args.target_images is None:
        parser.error("AFA and Combo require --target-images")
    for path in (
        args.validation_manifest,
        args.validation_images,
        args.validation_labels,
    ):
        if path is not None:
            path = path.resolve()
            if not path.exists():
                raise FileNotFoundError(path)
    if args.validation_images is not None and not args.validation_images.is_dir():
        raise NotADirectoryError(args.validation_images)
    if args.validation_labels is not None and not args.validation_labels.is_dir():
        raise NotADirectoryError(args.validation_labels)
    if args.validation_manifest is not None:
        args.validation_manifest = args.validation_manifest.resolve()
        args.validation_images = args.validation_images.resolve()
        args.validation_labels = args.validation_labels.resolve()
    train(args)


if __name__ == "__main__":
    main()
