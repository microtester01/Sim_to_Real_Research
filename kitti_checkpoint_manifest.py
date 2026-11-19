from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


VARIANTS = ("Baseline", "DR", "AFA", "Combo", "UpperBound")
SEEDS = (40, 41, 42)
RUN_IDS = tuple(
    "{}_seed{}".format(variant, seed)
    for variant in VARIANTS
    for seed in SEEDS
)
SELECTION_FIELDS = ("selected_epoch", "selection_metric", "selection_value")
MANIFEST_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_record(runs_root: Path, run_id: str) -> dict[str, Any]:
    run_dir = runs_root / run_id
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Selected checkpoint not found: {}".format(checkpoint_path))

    selection_path = run_dir / "best_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError("Checkpoint selection not found: {}".format(selection_path))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    missing_fields = [field for field in SELECTION_FIELDS if field not in selection]
    if missing_fields:
        raise ValueError(
            "{} is missing selection field(s): {}".format(
                selection_path, ", ".join(missing_fields)
            )
        )

    return {
        "checkpoint_id": run_id,
        "checkpoint_path": PurePosixPath(run_id, "best.pt").as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "selection": {field: selection[field] for field in SELECTION_FIELDS},
    }


def build_checkpoint_manifest(runs_root: Path) -> dict[str, Any]:
    runs = {}
    for run_id in RUN_IDS:
        runs[run_id] = checkpoint_record(runs_root, run_id)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "runs": runs,
    }


def verify_checkpoint_provenance(
    run_id: str,
    runs_root: Path,
    manifest: dict[str, Any],
    selection: dict[str, Any],
    report_checkpoint: str,
) -> Path:
    if run_id not in RUN_IDS:
        raise ValueError("Unexpected checkpoint run ID: {}".format(run_id))
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint manifest schema")
    runs = manifest.get("runs")
    if not isinstance(runs, dict) or run_id not in runs:
        raise ValueError("Checkpoint manifest has no run ID: {}".format(run_id))
    record = runs[run_id]
    if not isinstance(record, dict) or record.get("checkpoint_id") != run_id:
        raise ValueError("Checkpoint manifest run ID mismatch: {}".format(run_id))

    expected_relative_path = PurePosixPath(run_id, "best.pt")
    recorded_relative_path = PurePosixPath(
        str(record.get("checkpoint_path", "")).replace("\\", "/")
    )
    if (
        recorded_relative_path.is_absolute()
        or ".." in recorded_relative_path.parts
        or recorded_relative_path != expected_relative_path
    ):
        raise ValueError("Invalid relative checkpoint path for run {}".format(run_id))

    local_checkpoint = runs_root.joinpath(*expected_relative_path.parts)
    if not local_checkpoint.is_file():
        raise FileNotFoundError("Selected checkpoint not found: {}".format(local_checkpoint))
    expected_hash = record.get("checkpoint_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("Invalid checkpoint SHA-256 for run {}".format(run_id))
    if sha256_file(local_checkpoint) != expected_hash:
        raise ValueError("Checkpoint SHA-256 mismatch for run {}".format(run_id))

    expected_selection = record.get("selection")
    actual_selection = {field: selection.get(field) for field in SELECTION_FIELDS}
    if expected_selection != actual_selection:
        raise ValueError("Checkpoint selection metadata mismatch for run {}".format(run_id))

    reported_path = PurePosixPath(str(report_checkpoint).replace("\\", "/"))
    if len(reported_path.parts) < 2 or reported_path.parts[-2:] != expected_relative_path.parts:
        raise ValueError("Test report checkpoint run ID mismatch for {}".format(run_id))
    return local_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a portable SHA-256 manifest for the selected KITTI checkpoints."
    )
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_checkpoint_manifest(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("Checkpoint manifest: {}".format(args.output))
    print("Verified selected checkpoints: {}".format(len(manifest["runs"])))


if __name__ == "__main__":
    main()
