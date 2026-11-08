"""Privacy and number-preservation checks on synthetic, temporary fixtures."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("export_legacy_numeric.py")
spec = importlib.util.spec_from_file_location("legacy_publication_export", str(SCRIPT))
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class PublicationExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sources = self.root / "experiments"
        self.sources.mkdir()
        self.key = self.root / "outside_release.key"
        self.secret = b"SYNTHETIC_TEST_SECRET_32_BYTES____"
        self.key.write_bytes(self.secret)
        self.before = {}
        for index in (1, 2):
            directory = self.sources / ("EXP_" + str(index) * 32)
            directory.mkdir()
            row = {"dataset": "BUS-BRA", "patient_id": "CASE_PRIVATE_0001", "image_id": "IMAGE_PRIVATE_0001", "image_key": "BUS-BRA:IMAGE_PRIVATE_0001",
                   "pathology": "PRIVATE_PATHOLOGY", "original_image": "D:/PRIVATE_SOURCE/image.png", "prediction_sha256": "private_pixel_hash",
                   "variant": "jpeg_q85", "status": "completed" if index == 1 else "failed", "dice": .875 if index == 1 else None,
                   "original_bytes": 1000, "compressed_bytes": 400, "attempts": index,
                   "features": {"values": {"noise": .0123, "pixels": 1024}}, "original_box_xyxy": [1., 2., 10., 12.]}
            if index == 2:
                row.update(error_type="RuntimeError", error="CUDA cumsum deterministic error D:/PRIVATE_SOURCE/CASE_PRIVATE_0001/image.png")
            config = {"experiment_id": directory.name, "dataset": "BUS-BRA", "split": "development", "device": "cuda", "records": [dict(row, patient_id_verified=False)],
                      "platform_config": {"data_root": "D:/PRIVATE_SOURCE", "random_seed": 4, "patient_ids_verified": False},
                      "iacs": {"calibration_patient_ids": ["CASE_PRIVATE_0001"], "calibration_images": 1, "quantiles": {"noise": [.01, .02]}}}
            for name, value in [("config.json", config), ("summary.json", {"total": 1, "progress": 1., "counts": {row["status"]: 1}})]:
                (directory / name).write_text(json.dumps(value), encoding="utf-8")
            (directory / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        for path in self.sources.rglob("*"):
            if path.is_file(): self.before[path] = path.read_bytes()

    def tearDown(self):
        self.temp.cleanup()

    def test_privacy_numbers_failures_hmac_and_sources(self):
        destination = self.root / "export"
        result = exporter.export_bundle(self.sources, destination, self.key)
        self.assertEqual(result["records"], 2)
        rows = [json.loads(line) for line in (destination / "all_results.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[0]["case_key"], rows[1]["case_key"])
        self.assertEqual(rows[0]["image_key"], rows[1]["image_key"])
        self.assertEqual(rows[0]["case_key"], exporter.pseudonym("BUS-BRA", "CASE_PRIVATE_0001", self.secret))
        self.assertEqual(len(rows[0]["case_key"]), 24)
        self.assertEqual(rows[0]["dice"], .875)
        self.assertEqual(rows[0]["features"]["values"]["pixels"], 1024)
        self.assertEqual(rows[1]["status"], "failed")
        self.assertEqual(rows[1]["error_type"], "RuntimeError")
        self.assertNotIn("error", rows[1])
        for path in destination.rglob("*"):
            if path.is_file() and path.suffix != ".py":
                text = path.read_text(encoding="utf-8-sig")
                for marker in ("CASE_PRIVATE_0001", "IMAGE_PRIVATE_0001", "PRIVATE_PATHOLOGY", "PRIVATE_SOURCE", self.secret.decode()):
                    self.assertNotIn(marker, text)
        for path, content in self.before.items(): self.assertEqual(path.read_bytes(), content)
        self.assertTrue(json.loads((destination / "EXPORT_AUDIT.json").read_text(encoding="utf-8"))["source_hashes_unchanged"])

    def test_new_export_uses_same_external_key_and_no_overwrite(self):
        first, second = self.root / "first", self.root / "second"
        exporter.export_bundle(self.sources, first, self.key)
        exporter.export_bundle(self.sources, second, self.key)
        self.assertEqual((first / "all_results.jsonl").read_bytes(), (second / "all_results.jsonl").read_bytes())
        with self.assertRaises(FileExistsError): exporter.export_bundle(self.sources, first, self.key)

    def test_overlapping_source_output_rejected(self):
        with self.assertRaises(ValueError): exporter.export_bundle(self.sources, self.sources / "unsafe", self.key)


if __name__ == "__main__":
    unittest.main()
