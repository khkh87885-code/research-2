"""Export unit-test fixtures only; these are not scientific experiment data."""

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_platform.exports import export_experiment


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="export_unit_fixture_")
        self.root = Path(self.temp.name)
        self.exp = self.root / "fixture_experiment"
        self.exp.mkdir()
        (self.exp / "config.json").write_text(json.dumps({"evidence_type": "unit_test_fixture"}), encoding="utf-8")
        (self.exp / "summary.json").write_text(json.dumps({"planned": 6, "completed": 4, "failed": 1}), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def _write_rows(self, rows):
        path = self.exp / "results.jsonl"
        path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
        return path

    def _base(self, **changes):
        row = dict(dataset="fixture_A", split="development", model="fixture_model", prompt_strategy="mechanism", patient_id="A", image_id="a1", variant="baseline", format="PNG", scale=1.0, quality=None, status="completed", dice=0.5, hd95=3.0, compressed_bytes=1024, original_bytes=2048, compression_saving=50)
        row.update(changes)
        return row

    def test_patient_means_prevent_image_weighting_and_context_mixing(self):
        rows = [self._base(dice=0.0), self._base(image_id="a2", dice=1.0), self._base(patient_id="B", image_id="b1", dice=1.0), self._base(dataset="fixture_B", patient_id="A", image_id="a1", dice=0.2), self._base(image_id="failed", status="failed", dice=None, error="fixture failure")]
        path = self._write_rows(rows)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        result = export_experiment(self.exp)
        self.assertEqual(result["successful_rows"], 4)
        self.assertEqual(result["failed_rows"], 1)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
        directory = Path(result["export_dir"])
        with (directory / "group_descriptive.csv").open(encoding="utf-8-sig", newline="") as stream:
            groups = list(csv.DictReader(stream))
        self.assertEqual(len(groups), 2)
        first = next(group for group in groups if group["dataset"] == "fixture_A")
        self.assertAlmostEqual(float(first["dice_patient_mean"]), 0.75)
        self.assertEqual(int(first["recorded_patient_or_case_count"]), 2)
        self.assertEqual(result["tables"]["Table_6_external_dataset"], "pending")
        self.assertEqual(result["tables"]["Table_8_model_comparison"], "pending")
        provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["source_sha256"]["results.jsonl"], before)
        self.assertIn("fixture failure", (directory / "experiment_report.md").read_text(encoding="utf-8"))

    def test_missing_metrics_stay_missing_and_no_fake_observations(self):
        self._write_rows([self._base(dice=None, hd95=None, status="completed", metric_status="invalid_reference")])
        result = export_experiment(self.exp)
        directory = Path(result["export_dir"])
        self.assertEqual(result["figure_status"]["Figure_4"]["status"], "pending")
        self.assertFalse((directory / "Figure_4_size_vs_Dice.png").exists())
        with (directory / "per_image.csv").open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["dice"], "")
        provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["invalid_dice_rows"], 1)

    def test_export_snapshots_are_distinct_and_empty_status_is_explicit(self):
        first = export_experiment(self.exp)
        second = export_experiment(self.exp)
        self.assertNotEqual(first["export_dir"], second["export_dir"])
        self.assertTrue(Path(first["export_dir"]).is_dir())
        self.assertEqual(first["row_count"], 0)
        self.assertEqual(first["tables"]["Table_1_dataset_characteristics"], "pending")

    def test_spreadsheet_formula_text_is_escaped(self):
        self._write_rows([self._base(image_id="=SUM(1,2)")])
        result = export_experiment(self.exp)
        with (Path(result["export_dir"]) / "per_image.csv").open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["image_id"], "'=SUM(1,2)")
        workbook = Path(result["export_dir"]) / "experiment_results.xlsx"
        if workbook.exists():
            from openpyxl import load_workbook
            sheet = load_workbook(workbook, data_only=False)["Per_image"]
            headers = [cell.value for cell in sheet[1]]
            cell = sheet.cell(2, headers.index("image_id") + 1)
            self.assertNotEqual(cell.data_type, "f")

    def test_observed_plot_exports_all_requested_plot_formats(self):
        self._write_rows([self._base()])
        result = export_experiment(self.exp)
        directory = Path(result["export_dir"])
        self.assertEqual(result["figure_status"]["Figure_4"]["observed_pairs"], 1)
        for extension in ("png", "svg", "pdf"):
            self.assertGreater((directory / ("Figure_4_size_vs_Dice." + extension)).stat().st_size, 100)

    def test_pixel_and_mm_boundaries_are_never_averaged_together(self):
        self._write_rows([
            self._base(boundary_unit="pixel", boundary_f1_unit="pixel", boundary_tolerance=2.0, hd95=10.0),
            self._base(image_id="a2", boundary_unit="mm", boundary_f1_unit="pixel", boundary_tolerance=2.0, hd95=1.0),
        ])
        result = export_experiment(self.exp)
        with (Path(result["export_dir"]) / "group_descriptive.csv").open(encoding="utf-8-sig", newline="") as stream:
            groups = list(csv.DictReader(stream))
        self.assertEqual(len(groups), 2)
        observed = {group["boundary_unit"]: float(group["hd95_patient_mean"]) for group in groups}
        self.assertEqual(observed, {"pixel": 10.0, "mm": 1.0})

    def test_IACS_parameter_lock_does_not_claim_validated_performance(self):
        (self.exp / "config.json").write_text(json.dumps({"iacs": {"state": "locked"}}), encoding="utf-8")
        self._write_rows([self._base(variant="iacs")])
        result = export_experiment(self.exp)
        self.assertEqual(result["tables"]["Table_5_IACS_comparison"], "observed_unvalidated_IACS")
        with (Path(result["export_dir"]) / "Table_5_IACS_comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["iacs_parameter_state"], "locked")
        self.assertIn("not established", row["iacs_evidence_status"])


if __name__ == "__main__":
    unittest.main()
