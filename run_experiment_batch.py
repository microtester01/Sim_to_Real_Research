import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_CARLA_YAML = (
    ROOT / "data" / "kitti_experiment" / "carla_yolo_regions_v2" / "dataset.yaml"
)
DEFAULT_UPPER_BOUND_YAML = (
    ROOT / "data" / "kitti_experiment" / "upper_bound_data_v3" / "kitti_train_100.yaml"
)
DEFAULT_TARGET_IMAGES = ROOT / "data" / "kitti_experiment" / "yolo8" / "images" / "train"
DEFAULT_VALIDATION_MANIFEST = (
    ROOT / "data" / "kitti_experiment" / "yolo8" / "split_manifest.json"
)
DEFAULT_VALIDATION_IMAGES = ROOT / "data" / "kitti_experiment" / "yolo8" / "images" / "val"
DEFAULT_VALIDATION_LABELS = ROOT / "data" / "kitti_raw" / "training" / "label_2"
DEFAULT_INITIALIZATIONS = (
    ROOT / "data" / "kitti_experiment" / "initialization_states_v1"
)
DEFAULT_OUTPUT = ROOT / "data" / "kitti_experiment" / "student_15run_10k"
SEEDS = (40, 41, 42)
UPDATES = 10000
DOMAIN_LEARNING_RATE = 0.0001
CONFIGURATIONS = ("Baseline", "DR", "AFA", "Combo", "UpperBound")


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path):
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("Expected a YAML mapping in {}".format(path))
    return value


def class_name(names, class_id):
    if isinstance(names, dict):
        return names.get(class_id, names.get(str(class_id)))
    if isinstance(names, list) and class_id < len(names):
        return names[class_id]
    return None


def validate_carla_dataset(source_yaml):
    dataset = load_yaml(source_yaml)
    names = dataset.get("names")
    expected_names = {
        0: "Pedestrian",
        1: "Cyclist",
        2: "IgnoreDontCare",
        3: "IgnorePersonSitting",
    }
    if any(class_name(names, index) != value for index, value in expected_names.items()):
        raise ValueError(
            "CARLA YAML must define Pedestrian, Cyclist, IgnoreDontCare and "
            "IgnorePersonSitting as class IDs 0-3. Regenerate labels before training."
        )
    manifest_path = source_yaml.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "CARLA dataset manifest is missing: {}".format(manifest_path)
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("label_policy_version") != "semantic-ignore-v1":
        raise ValueError(
            "CARLA labels do not declare semantic-ignore-v1. Regenerate the dataset "
            "with the updated carla_dataset.py before training."
        )


def validate_inputs(args, source_yaml):
    for path in (
        source_yaml,
        args.validation_manifest,
        args.initializations / "initialization_manifest.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.validation_images, args.validation_labels):
        if not path.is_dir():
            raise NotADirectoryError(path)
    if args.configuration in ("AFA", "Combo") and not args.target_images.is_dir():
        raise NotADirectoryError(args.target_images)
    if args.configuration != "UpperBound":
        validate_carla_dataset(source_yaml)

    initialization_manifest = json.loads(
        (args.initializations / "initialization_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for seed in SEEDS:
        metadata = initialization_manifest.get("seeds", {}).get(str(seed))
        weight_path = args.initializations / "yolov8s_nc2_seed{}.pt".format(seed)
        if metadata is None or metadata.get("file") != weight_path.name:
            raise ValueError("Missing shared initialization metadata for seed {}".format(seed))
        if not weight_path.is_file():
            raise FileNotFoundError(weight_path)


def write_status(path, status):
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary_path), str(path))


def make_command(args, source_yaml, seed, output):
    command = [
        sys.executable,
        str(ROOT / "carla_adaptation.py"),
        "--configuration",
        args.configuration,
        "--source-yaml",
        str(source_yaml),
        "--output",
        str(output),
        "--seed",
        str(seed),
        "--initialization-weights",
        str(args.initializations / "yolov8s_nc2_seed{}.pt".format(seed)),
        "--max-updates",
        str(UPDATES),
        "--batch-size",
        "4",
        "--image-size",
        "640",
        "--workers",
        str(args.workers),
        "--target-workers",
        "0",
        "--validation-manifest",
        str(args.validation_manifest),
        "--validation-images",
        str(args.validation_images),
        "--validation-labels",
        str(args.validation_labels),
        "--log-every",
        "100",
    ]
    if args.configuration in ("AFA", "Combo"):
        command.extend(["--domain-learning-rate", str(DOMAIN_LEARNING_RATE)])
    if args.configuration in ("AFA", "Combo"):
        command.extend(["--target-images", str(args.target_images)])
    return command


def main():
    parser = argparse.ArgumentParser(
        description="Run three paired seeds for one experiment configuration."
    )
    parser.add_argument("--configuration", choices=CONFIGURATIONS, required=True)
    parser.add_argument("--carla-yaml", type=Path, default=DEFAULT_CARLA_YAML)
    parser.add_argument(
        "--upper-bound-yaml", type=Path, default=DEFAULT_UPPER_BOUND_YAML
    )
    parser.add_argument("--target-images", type=Path, default=DEFAULT_TARGET_IMAGES)
    parser.add_argument(
        "--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST
    )
    parser.add_argument(
        "--validation-images", type=Path, default=DEFAULT_VALIDATION_IMAGES
    )
    parser.add_argument(
        "--validation-labels", type=Path, default=DEFAULT_VALIDATION_LABELS
    )
    parser.add_argument(
        "--initializations", type=Path, default=DEFAULT_INITIALIZATIONS
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the three commands without starting training.",
    )
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("--workers cannot be negative")

    for attribute in (
        "carla_yaml",
        "upper_bound_yaml",
        "target_images",
        "validation_manifest",
        "validation_images",
        "validation_labels",
        "initializations",
        "output",
    ):
        setattr(args, attribute, getattr(args, attribute).resolve())
    source_yaml = (
        args.upper_bound_yaml
        if args.configuration == "UpperBound"
        else args.carla_yaml
    )
    validate_inputs(args, source_yaml)

    output = args.output
    status_path = output / "{}_status.json".format(args.configuration)
    run_dirs = [
        output / "{}_seed{}".format(args.configuration, seed) for seed in SEEDS
    ]
    existing_runs = [path for path in run_dirs if path.exists()]
    if existing_runs:
        raise FileExistsError(
            "Refusing to overwrite an existing run: {}".format(existing_runs[0])
        )

    if args.dry_run:
        for seed in SEEDS:
            print(subprocess.list2cmdline(make_command(args, source_yaml, seed, output)))
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to run the training batch.")
    device_name = torch.cuda.get_device_name(0)
    if device_name != "NVIDIA GeForce RTX 3060":
        raise RuntimeError(
            "Expected NVIDIA GeForce RTX 3060, found {}".format(device_name)
        )

    output.mkdir(parents=True, exist_ok=True)
    status = {
        "configuration": args.configuration,
        "seeds": list(SEEDS),
        "updates_per_seed": UPDATES,
        "device": device_name,
        "started_at": timestamp(),
        "runs": {},
    }
    write_status(status_path, status)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"

    for seed, run_dir in zip(SEEDS, run_dirs):
        command = make_command(args, source_yaml, seed, output)
        console_log = output / "{}_seed{}_console.log".format(
            args.configuration, seed
        )
        run = {
            "status": "running",
            "started_at": timestamp(),
            "run_dir": str(run_dir),
            "console_log": str(console_log),
            "command": command,
        }
        status["runs"][str(seed)] = run
        write_status(status_path, status)
        print(
            "Starting {} seed {} ({}/{}). Log: {}".format(
                args.configuration, seed, len(status["runs"]), len(SEEDS), console_log
            ),
            flush=True,
        )
        try:
            with console_log.open("w", encoding="utf-8") as log_stream:
                result = subprocess.run(
                    command,
                    cwd=str(ROOT),
                    env=environment,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
        except KeyboardInterrupt:
            run["status"] = "interrupted"
            run["finished_at"] = timestamp()
            write_status(status_path, status)
            raise

        run["exit_code"] = result.returncode
        run["finished_at"] = timestamp()
        if result.returncode == 0:
            run["status"] = "completed"
            run["training_log"] = str(run_dir / "train.jsonl")
            run["validation_log"] = str(run_dir / "validation.jsonl")
            run["checkpoint"] = str(run_dir / "best.pt")
            selection_path = run_dir / "best_selection.json"
            if selection_path.is_file():
                run["best_validation"] = json.loads(
                    selection_path.read_text(encoding="utf-8")
                )
        else:
            run["status"] = "failed"
        write_status(status_path, status)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, command)

    status["finished_at"] = timestamp()
    write_status(status_path, status)
    print("All three seeds completed. Summary: {}".format(status_path))


if __name__ == "__main__":
    main()
