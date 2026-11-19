import json
import tempfile
import unittest
from pathlib import Path

from kitti_checkpoint_manifest import (
    build_checkpoint_manifest,
    checkpoint_record,
    verify_checkpoint_provenance,
)


class KittiCheckpointManifestTests(unittest.TestCase):
    def _create_run(self, root):
        run_id = "Baseline_seed40"
        run_dir = root / run_id
        run_dir.mkdir()
        checkpoint = run_dir / "best.pt"
        checkpoint.write_bytes(b"selected checkpoint")
        selection = {
            "selected_epoch": 6,
            "selection_metric": "internal KITTI mAP@0.50:0.95",
            "selection_value": 0.0058,
        }
        (run_dir / "best_selection.json").write_text(
            json.dumps(selection), encoding="utf-8"
        )
        return run_id, selection, checkpoint

    def test_manifest_uses_run_id_relative_path_and_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_id, selection, checkpoint = self._create_run(root)
            record = checkpoint_record(root, run_id)
            manifest = {
                "schema_version": 1,
                "runs": {run_id: record},
            }

            verified = verify_checkpoint_provenance(
                run_id,
                root,
                manifest,
                selection,
                r"C:\original\workspace\Baseline_seed40\best.pt",
            )

            self.assertEqual(verified, checkpoint)
            self.assertEqual(record["checkpoint_id"], run_id)
            self.assertEqual(record["checkpoint_path"], "Baseline_seed40/best.pt")
            self.assertEqual(len(record["checkpoint_sha256"]), 64)

    def test_manifest_rejects_checkpoint_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_id, selection, checkpoint = self._create_run(root)
            manifest = {
                "schema_version": 1,
                "runs": {run_id: checkpoint_record(root, run_id)},
            }
            checkpoint.write_bytes(b"replacement checkpoint")

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_checkpoint_provenance(
                    run_id,
                    root,
                    manifest,
                    selection,
                    r"C:\original\workspace\Baseline_seed40\best.pt",
                )

    def test_manifest_builder_records_all_run_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_run_ids = []
            for variant in ("Baseline", "DR", "AFA", "Combo", "UpperBound"):
                for seed in (40, 41, 42):
                    run_id = "{}_seed{}".format(variant, seed)
                    expected_run_ids.append(run_id)
                    self._create_run_for_id(root, run_id)

            manifest = build_checkpoint_manifest(root)

            self.assertEqual(set(manifest["runs"]), set(expected_run_ids))

    @staticmethod
    def _create_run_for_id(root, run_id):
        run_dir = root / run_id
        run_dir.mkdir()
        (run_dir / "best.pt").write_bytes(run_id.encode("ascii"))
        (run_dir / "best_selection.json").write_text(
            json.dumps(
                {
                    "selected_epoch": 1,
                    "selection_metric": "internal KITTI mAP@0.50:0.95",
                    "selection_value": 0.1,
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
