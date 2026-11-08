"""Constructed software fixtures only; these images are never research results."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from research_platform.datasets import (
    UnsupportedImageError, _parse_pixel_size_cm, import_manifest, load_mask, load_rgb, scan_datasets,
)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_alpha_black_and_grayscale_and_depth(self):
        path = self.root / "rgba.png"
        Image.fromarray(np.array([[[200, 100, 50, 0], [20, 40, 60, 255]]], dtype=np.uint8), "RGBA").save(path)
        np.testing.assert_array_equal(load_rgb(path), [[[0, 0, 0], [20, 40, 60]]])
        path = self.root / "gray.png"
        Image.fromarray(np.array([[0, 128, 255]], dtype=np.uint8)).save(path)
        self.assertEqual(load_rgb(path).shape, (1, 3, 3))
        self.assertEqual(load_rgb(path)[0, 1].tolist(), [128, 128, 128])
        Image.fromarray(np.array([[0, 4095]], dtype=np.uint16)).save(path)
        with self.assertRaises(UnsupportedImageError):
            load_rgb(path)

    def test_binary_mask_does_not_dither_or_use_alpha(self):
        path = self.root / "mask.png"
        Image.fromarray(np.array([[[0, 0, 0, 255], [255, 255, 255, 1]]], dtype=np.uint8), "RGBA").save(path)
        self.assertEqual(load_mask(path).tolist(), [[False, True]])
        Image.fromarray(np.array([[0, 100, 200]], dtype=np.uint8)).save(path)
        with self.assertRaises(ValueError):
            load_mask(path)

    def bus_fixture(self):
        bus = self.root / "BUSBRA"
        (bus / "Images").mkdir(parents=True)
        (bus / "Masks").mkdir()
        image = np.arange(12, dtype=np.uint8).reshape(3, 4)
        mask = np.zeros((3, 4), dtype=np.uint8); mask[1, 1] = 255
        for name in ("bus_a", "bus_b"):
            Image.fromarray(image).save(bus / "Images" / (name + ".png"))
            Image.fromarray(mask).save(bus / "Masks" / ("mask_" + name[4:] + ".png"))
        write_csv(bus / "bus_data.csv", [
            {"ID": "bus_a", "Case": "7", "Width": "4", "Height": "99", "Pathology": "benign"},
            {"ID": "bus_b", "Case": "7", "Width": "4", "Height": "3", "Pathology": "benign"}])
        write_csv(self.root / "busbra_splits" / "busbra_development.csv", [
            {"patient_id": "7", "image_id": "bus_a"}])

    def test_scanner_keeps_all_images_and_patient_split_and_issues(self):
        self.bus_fixture()
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob("*") if p.is_file()}
        report = scan_datasets({"data_root": str(self.root), "pilot_root": str(self.root / "no_pilot")})
        self.assertEqual(len(report["records"]), 2)
        self.assertEqual([r["split"] for r in report["records"]], ["development", "development"])
        self.assertEqual(sum(r["selected_in_frozen_manifest"] for r in report["records"]), 1)
        self.assertTrue(all(r["lesion_id"] is None and not r["patient_id_verified"] for r in report["records"]))
        self.assertEqual(report["records"][0]["height"], 3)
        codes = [i["code"] for i in report["issues"]]
        self.assertIn("metadata_dimension_mismatch", codes)
        self.assertIn("duplicate_file", codes)
        self.assertIn("duplicate_decoded_pixels", codes)
        after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_patient_leakage_rejected(self):
        self.bus_fixture()
        write_csv(self.root / "busbra_splits" / "busbra_internal_holdout.csv", [
            {"patient_id": "7", "image_id": "bus_b"}])
        with self.assertRaisesRegex(ValueError, "patient leakage"):
            scan_datasets({"data_root": str(self.root), "pilot_root": str(self.root / "none")})

    def test_import_requires_explicit_mapping_and_rejects_escape(self):
        path = self.root / "custom.csv"
        write_csv(path, [{"image_path": "anything.png"}])
        with self.assertRaises(ValueError):
            import_manifest(path, self.root)
        write_csv(path, [{"dataset": "Custom", "patient_id": "", "image_id": "x",
                          "image_path": "../escape.png", "mask_path": ""}])
        with self.assertRaisesRegex(ValueError, "leaves declared"):
            import_manifest(path, self.root)

    def test_inventory_preserves_high_depth_and_corrupt_records(self):
        Image.fromarray(np.array([[0, 4095]], dtype=np.uint16)).save(self.root / "high.png")
        (self.root / "bad.png").write_bytes(b"invalid image fixture")
        path = self.root / "custom.csv"
        write_csv(path, [{"dataset": "Custom", "patient_id": "", "image_id": iid,
                          "image_path": name, "mask_path": ""}
                         for iid, name in (("high", "high.png"), ("bad", "bad.png"))])
        report = import_manifest(path, self.root)
        self.assertEqual(len(report["records"]), 2)
        self.assertEqual(report["records"][0]["bit_depth"], 16)
        self.assertEqual(report["records"][0]["gray_max"], 4095)
        self.assertIsNone(report["records"][0]["patient_id"])
        self.assertEqual(report["records"][1]["data_status"], "error")

    def test_frozen_mapping_conflict_rejected(self):
        self.bus_fixture()
        write_csv(self.root / "busbra_splits" / "busbra_development.csv", [
            {"patient_id": "999", "image_id": "bus_a"}])
        with self.assertRaisesRegex(ValueError, "mapping conflicts"):
            scan_datasets({"data_root": str(self.root), "pilot_root": str(self.root / "none")})

    def test_official_spacing_cm_to_mm_axis_order_and_missingness(self):
        self.assertEqual(_parse_pixel_size_cm("7.8125E-3"), [0.078125, 0.078125])
        self.assertEqual(_parse_pixel_size_cm("[0.01, 0.02]"), [0.2, 0.1])
        self.assertIsNone(_parse_pixel_size_cm(""))
        self.assertIsNone(_parse_pixel_size_cm(None))
        for value in ("0", "-1", "NaN", "Infinity", "[1]", "[1,0]", "garbage"):
            with self.assertRaises((ValueError, TypeError)):
                _parse_pixel_size_cm(value)


if __name__ == "__main__":
    unittest.main()
