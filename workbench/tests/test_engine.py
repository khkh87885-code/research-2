"""State/provenance/safety tests with temporary software fixtures only.

No clinical images or model weights are loaded. Model construction is mocked
where needed, and all filesystem mutations remain in TemporaryDirectory.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from research_platform.adaptive import calibrate, decide, lock_candidate
from research_platform.common import load_config, read_json, sha256, write_json
from research_platform.compression import compress, recipes
from research_platform.engine import create_experiment, experiment_dir, run_experiment
from research_platform.features import VERSION
from research_platform.store import Store


class EngineSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="platform-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {
            "data_root": str(self.root / "original_data"),
            "pilot_root": str(self.root / "original_code"),
            "sam_source": str(self.root / "original_sam"),
            "checkpoint": str(self.root / "original_data" / "fixture_weights.pth"),
            "checkpoint_sha256": "unused_fixture_hash", "sam_commit": "fixture_commit",
            "output_root": str(self.root / "output"), "formal_protocol_locked": False,
            "patient_ids_verified": False, "random_seed": 7,
            "bandwidth_mbps": 10, "rtt_ms": 0, "boundary_tolerance_pixels": 2,
            "dice_noninferiority_margin": None, "hd95_margin": None,
        }
        for key in ("data_root", "pilot_root", "sam_source", "output_root"):
            Path(self.config[key]).mkdir()
        (Path(self.config["pilot_root"]) / "run_single_case.py").write_text("# software fixture, never executed\n", encoding="utf-8")
        Path(self.config["checkpoint"]).write_bytes(b"software fixture, not model weights")
        self.record = {
            "dataset": "BUS-BRA", "split": "development", "patient_id": "fixture_patient",
            "image_id": "fixture_image", "selected_in_frozen_manifest": True,
            "image_path": str(Path(self.config["data_root"]) / "image.png"),
            "mask_path": str(Path(self.config["data_root"]) / "mask.png"),
            "image_sha256": "fixture_image_hash", "mask_sha256": "fixture_mask_hash",
            "data_status": "ok", "annotation_status": "complete",
            "width": 8, "height": 8,
        }
        self.inventory = {"dataset_version": "software-fixture", "dataset_checksum": "fixture_checksum",
                          "source_checksums": {}, "records": [self.record]}
        write_json(Path(self.config["output_root"]) / "dataset_report" / "inventory.json", self.inventory)
        self.snapshot = {"fixture_source.py": "frozen_hash"}

    def make_experiment(self, options=None):
        with patch("research_platform.engine.source_snapshot", return_value=self.snapshot), \
                patch("research_platform.engine.hardware", return_value={"gpu": None}), \
                patch("research_platform.engine.runtime", return_value={"fixture": True}):
            return create_experiment(self.config, options or {"variants": ["baseline"], "device": "cpu"})

    def test_store_schedule_idempotence_and_append_events(self):
        store = Store(self.root / "store")
        store.schedule([self.record], ["baseline"])
        key = "BUS-BRA:fixture_image"
        store.update(key, "baseline", "running")
        store.update(key, "baseline", "failed", {"error": "fixture failure"})
        store.update(key, "baseline", "running")
        store.update(key, "baseline", "completed", {"dice": .8})
        store.schedule([self.record], ["baseline"])
        self.assertEqual(store.state(key, "baseline")["status"], "completed")
        self.assertEqual(store.state(key, "baseline")["attempts"], 2)
        with store.connect() as db:
            events = db.execute("SELECT status FROM events ORDER BY id").fetchall()
        self.assertEqual([event["status"] for event in events], ["running", "failed", "running", "completed"])
        self.assertEqual(store.summary()["progress"], 1)

    def test_store_completed_results_cannot_be_overwritten(self):
        store = Store(self.root / "store")
        store.schedule([self.record], ["baseline"])
        key = "BUS-BRA:fixture_image"
        store.update(key, "baseline", "running")
        store.update(key, "baseline", "completed", {"dice": .8})
        saved = store.state(key, "baseline")
        for status in ("running", "failed", "completed"):
            try:
                store.update(key, "baseline", status, {"dice": .99})
            except (ValueError, RuntimeError):
                pass
            self.assertEqual(store.state(key, "baseline"), saved)

    def test_config_content_locked_before_model_load(self):
        experiment = self.make_experiment()
        directory = experiment_dir(self.config, experiment)
        frozen = read_json(directory / "config.json")
        frozen["name"] = "changed after creation"
        write_json(directory / "config.json", frozen)
        with patch("research_platform.models.SamModel") as model:
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                run_experiment(self.config, experiment)
            model.assert_not_called()

    def test_source_snapshot_locked_before_model_load(self):
        experiment = self.make_experiment()
        with patch("research_platform.engine.source_snapshot", return_value={"fixture_source.py": "changed"}), \
                patch("research_platform.models.SamModel") as model:
            with self.assertRaisesRegex(ValueError, "Code changed"):
                run_experiment(self.config, experiment)
            model.assert_not_called()

    def test_legacy_prompt_source_locked_before_model_load(self):
        experiment = self.make_experiment()
        (Path(self.config["pilot_root"]) / "run_single_case.py").write_text("# changed software fixture\n", encoding="utf-8")
        with patch("research_platform.engine.source_snapshot", return_value=self.snapshot), \
                patch("research_platform.models.SamModel") as model:
            with self.assertRaises(ValueError):
                run_experiment(self.config, experiment)
            model.assert_not_called()

    def test_resume_uses_frozen_platform_settings(self):
        experiment = self.make_experiment()
        changed_global = dict(self.config, random_seed=9999)
        observed = []

        def model_constructor(config, device):
            observed.append(config["random_seed"])
            raise RuntimeError("fixture stops before model loading")

        with patch("research_platform.engine.source_snapshot", return_value=self.snapshot), \
                patch("research_platform.models.SamModel", side_effect=model_constructor):
            with self.assertRaisesRegex(RuntimeError, "fixture stops"):
                run_experiment(changed_global, experiment)
        self.assertEqual(observed, [7])
        store = Store(experiment_dir(self.config, experiment))
        self.assertEqual(store.state("BUS-BRA:fixture_image", "baseline")["status"], "failed")

    def test_completed_resume_skips_model_and_preserves_payload(self):
        experiment = self.make_experiment()
        store = Store(experiment_dir(self.config, experiment))
        store.update("BUS-BRA:fixture_image", "baseline", "completed", {"dice": .8})
        saved = store.state("BUS-BRA:fixture_image", "baseline")
        with patch("research_platform.engine.source_snapshot", return_value=self.snapshot), \
                patch("research_platform.models.SamModel") as model:
            summary = run_experiment(self.config, experiment)
            model.assert_not_called()
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(store.state("BUS-BRA:fixture_image", "baseline"), saved)

    def test_output_config_cannot_overlap_original_roots(self):
        for source_key in ("data_root", "pilot_root", "sam_source"):
            for output in (Path(self.config[source_key]), Path(self.config[source_key]) / "processed", self.root):
                config = dict(self.config, output_root=str(output))
                path = self.root / "fixture_config.json"
                write_json(path, config)
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_output_config_protects_external_checkpoint_directory(self):
        config = dict(self.config, checkpoint=str(self.root / "external_weights" / "fixture.pth"),
                      output_root=str(self.root / "external_weights" / "processed"))
        path = self.root / "fixture_config.json"
        write_json(path, config)
        with self.assertRaises(ValueError):
            load_config(path)

    def test_compression_rejects_nested_original_directory(self):
        source = Path(self.config["data_root"]) / "image.png"
        Image.fromarray(np.full((8, 8), 50, dtype=np.uint8)).save(source)
        before = sha256(source)
        with self.assertRaises(ValueError):
            compress(source, source.parent / "derived", "fixture_image", recipes()["baseline"])
        self.assertEqual(sha256(source), before)

    def test_iacs_external_and_holdout_calibration_prohibited(self):
        feature = {"version": VERSION, "uses_reference_mask": False,
                   "values": {key: .1 for key in ("noise", "sharpness", "local_contrast", "edge_complexity")}}
        for record in (dict(self.record, dataset="BrEaST"), dict(self.record, split="internal_holdout")):
            with self.assertRaises(ValueError):
                calibrate([record], [feature], self.root / "prohibited_candidate.json")
        with self.assertRaises(ValueError):
            calibrate([self.record], [dict(feature, uses_reference_mask=True)], self.root / "prohibited_candidate.json")
        self.assertFalse((self.root / "prohibited_candidate.json").exists())

    def test_iacs_lock_blocked_without_validation_and_margins(self):
        feature = {"version": VERSION, "uses_reference_mask": False,
                   "values": {key: .1 for key in ("noise", "sharpness", "local_contrast", "edge_complexity")}}
        path = self.root / "candidate.json"
        candidate = calibrate([self.record], [feature], path)
        self.assertFalse(candidate["quality_constraints_verified"])
        self.assertEqual(candidate["state"], "candidate_unvalidated")
        for protocol in ({"formal_protocol_locked": False}, {"formal_protocol_locked": True}):
            with self.assertRaises(ValueError):
                lock_candidate(path, protocol)
        candidate["quality_constraints_verified"] = True
        write_json(path, candidate)
        with self.assertRaises(ValueError):
            lock_candidate(path, {"formal_protocol_locked": True})
        self.assertEqual(list(self.root.glob("iacs_lock_*.json")), [])

    def test_iacs_nonfinite_decision_features_rejected(self):
        feature = {"version": VERSION, "uses_reference_mask": False,
                   "values": {key: .1 for key in ("noise", "sharpness", "local_contrast", "edge_complexity")}}
        candidate = calibrate([self.record], [feature], self.root / "candidate.json")
        feature["values"]["noise"] = float("nan")
        with self.assertRaises(ValueError):
            decide(feature, candidate)

    def test_iacs_nonpositive_lock_margins_rejected(self):
        feature = {"version": VERSION, "uses_reference_mask": False,
                   "values": {key: .1 for key in ("noise", "sharpness", "local_contrast", "edge_complexity")}}
        path = self.root / "candidate.json"
        candidate = calibrate([self.record], [feature], path)
        candidate.update(quality_constraints_verified=True, dice_margin=-.02, hd95_margin=2)
        write_json(path, candidate)
        with self.assertRaises(ValueError):
            lock_candidate(path, {"formal_protocol_locked": True})

    def test_formal_validation_blocked_with_unconfirmed_protocol(self):
        for dataset, split in (("BUS-BRA", "internal_holdout"), ("BrEaST", "external_test")):
            with self.assertRaisesRegex(ValueError, "blocked"):
                self.make_experiment({"dataset": dataset, "split": split, "limit": 0})


if __name__ == "__main__":
    unittest.main()
