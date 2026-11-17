import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from ultralytics.nn.tasks import DetectionModel


def prepare_initializations(output, seeds):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Refusing to overwrite non-empty directory: {}".format(output)
        )
    output.mkdir(parents=True, exist_ok=True)
    entries = {}
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = DetectionModel("yolov8s.yaml", nc=2, verbose=False).cpu()
        path = output / "yolov8s_nc2_seed{}.pt".format(seed)
        torch.save(model.state_dict(), str(path))
        entries[str(seed)] = {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "initialization": "randomly initialized DetectionModel(yolov8s.yaml, nc=2)",
        }
        del model
    manifest = {
        "architecture": "YOLOv8s",
        "class_count": 2,
        "pretrained_weights": False,
        "seeds": entries,
        "pairing_rule": (
            "All configurations in a seed block load the exact same state_dict "
            "identified by its SHA-256; source loader order is seeded separately."
        ),
    }
    manifest_path = output / "initialization_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="40,41,42,43,44,45,46,47,48,49")
    args = parser.parse_args()
    try:
        seeds = [int(value) for value in args.seeds.split(",")]
    except ValueError as exc:
        parser.error("seeds must be a comma-separated list of integers")
        raise exc
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("seeds must be non-empty and unique")
    manifest = prepare_initializations(args.output, seeds)
    print(
        "Prepared {} shared initializations in {}".format(
            len(manifest["seeds"]), args.output
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
