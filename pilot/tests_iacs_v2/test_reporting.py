"""Synthetic numeric fixtures only; outputs stay in temporary test directories."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

from iacs_v2.reporting import build_report


def fixture():
    features, compression, decisions = [], [], []
    for index in range(3):
        case, image = "PRIVATE_CASE_%d" % index, "PRIVATE_IMAGE_%d" % index
        features.append({"case_key": case, "image_key": image, "patient_id": "PRIVATE_PATIENT_ID",
                         "image_path": "D:/PRIVATE_IMAGE_PATH.png", "features": {"noise_mad": .01 + index / 100, "laplacian_variance": .1 + index / 30, "local_contrast": .2 + index / 20}})
        for condition, dice, size in [("baseline", .9, 100000), ("jpeg_q95", .9, 60000), ("jpeg_q85", .89, 40000),
                                       ("jpeg_q75", .88, 30000), ("jpeg_q60", .87, 25000), ("jpeg_q40", .84, 15000), ("png_l6", .9, 110000)]:
            compression.append(dict(case_key=case, image_key=image, condition=condition, status="ok", dice=dice-index*.01,
                                    hd95=5+index, assd=2+index/2, output_bytes=size, saving_pct=(1-size/100000)*100,
                                    delta_dice=dice-.9, delta_hd95=0))
        selected = next(row for row in compression if row["image_key"] == image and row["condition"] == "jpeg_q85")
        decisions.append(dict(case_key=case, image_key=image, algorithm="IACS_v2", fold=index+1, selected_condition="jpeg_q85", fallback=index == 2,
                              reason="synthetic_unit_test", evaluation_only=selected, subgroups={"noise_mad": "low" if index < 2 else "high"},
                              candidate_evaluation={"heavy_fixture": "HEAVY_CANDIDATE_EVALUATION_SENTINEL" + "x" * 10000},
                              predictions={"jpeg_q85": {"predicted_delta_dice": -.01, "predicted_delta_hd95": .2, "p_risk": .1, "estimated_feasible": True, "hd95_status": "not_specified"}}))
    compression.append(dict(case_key="PRIVATE_CASE_0", image_key="PRIVATE_IMAGE_0", condition="failed_fixture", status="error", dice=None))
    return {"diagnostics": {"status": "synthetic_unit_test", "raw_case_ids": ["PRIVATE_RAW_IDS"], "message": "</script><img src=x onerror=alert(1)>"},
            "feature_rows": features, "compression_rows": compression,
            "validation": {"outer_decisions": decisions, "full_policy": {"heavy_fixture": "HEAVY_RAW_VALIDATION_SENTINEL" + "z" * 10000},
                           "folds": [{"fold": 1, "train_count": 2, "validation_count": 1, "train_fit_descriptive": {"scope": "in_sample_not_validation"}}]},
            "statistics": {"feature_associations": [{"feature": "noise_mad", "spearman_rho": .1, "holm_p": .9, "n_cases": 3}],
                           "paired_comparisons": [{"condition": "jpeg_q85", "n_cases": 3, "ci95": [-.02, 0]}],
                           "group_summaries": [{"condition": "jpeg_q85", "n_cases": 3, "mean_dice": .88}]}}


class ReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.pilot = Path(cls.temp.name) / "pilot"
        cls.destination = cls.pilot / "research_output" / "IACS_v2" / "dev_fixture" / "reports" / "report_fixture"
        cls.paths = build_report(fixture(), cls.destination)
        cls.data = json.loads(Path(cls.paths["report"]).read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_no_overwrite(self):
        before = Path(self.paths["report"]).read_bytes()
        with self.assertRaises(FileExistsError):
            build_report(fixture(), self.destination)
        self.assertEqual(before, Path(self.paths["report"]).read_bytes())

    def test_anonymous_and_unset_margin_and_failures(self):
        text = Path(self.paths["report"]).read_text(encoding="utf-8")
        html = Path(self.paths["dashboard"]).read_text(encoding="utf-8")
        self.assertNotIn("PRIVATE_", text)
        self.assertNotIn("PRIVATE_", html)
        self.assertIsNone(self.data["hd95_margin"])
        self.assertIn("HD95 容忍界值未设", html)
        self.assertIn("不能判断双约束达标", html)
        self.assertEqual(self.data["summary"]["compression_records"], 22)
        self.assertEqual(self.data["summary"]["failed_or_missing_records"], 1)
        self.assertEqual(self.data["summary"]["cases"], 3)
        self.assertEqual(self.data["summary"]["outer_cases"], 3)
        self.assertEqual(self.data["summary"]["fallback_decisions"], 1)
        self.assertNotIn("</script><img", html)
        self.assertIn("connect-src 'none'", html)
        self.assertEqual(html.count('<section class="panel'), 5)

    def test_xlsx_roundtrip_and_eight_figures(self):
        from openpyxl import load_workbook
        book = load_workbook(self.paths["xlsx"], read_only=True)
        self.assertEqual(book["compression"].max_row, 23)
        self.assertEqual(book["paired_comparisons"].max_row, 2)
        self.assertIn("feature_associations", book.sheetnames)
        book.close()
        self.assertEqual(len(self.paths["figures"]), 8)
        self.assertEqual(len(list(self.destination.glob("figure_*.png"))), 8)
        self.assertEqual(len(list(self.destination.glob("figure_*.svg"))), 8)
        self.assertEqual(len(list(self.destination.glob("figure_*.pdf"))), 8)
        self.assertEqual(self.data["tables"]["feature_associations"][0]["holm_p"], .9)

    def test_html_is_light_interface_snapshot_and_full_json_keeps_provenance(self):
        html = Path(self.paths["dashboard"]).read_text(encoding="utf-8")
        full = Path(self.paths["report"]).read_text(encoding="utf-8")
        for sentinel in ("HEAVY_CANDIDATE_EVALUATION_SENTINEL", "HEAVY_RAW_VALIDATION_SENTINEL"):
            self.assertIn(sentinel, full)
            self.assertNotIn(sentinel, html)
        embedded = html.split('id="report-data">', 1)[1].split("</script>", 1)[0]
        snapshot = json.loads(embedded)
        self.assertEqual(snapshot["snapshot_type"], "interface_data_snapshot_not_full_report")
        self.assertEqual(snapshot["summary"], self.data["summary"])
        self.assertEqual(snapshot["tables"], self.data["tables"])
        for key in ("validation", "compression_rows", "feature_rows", "statistics"):
            self.assertNotIn(key, snapshot)
        self.assertIn("下载界面数据 JSON", html)
        self.assertIn("IACS_v2_interface_snapshot.json", html)

    def test_non_development_scope_rejected(self):
        payload = fixture()
        payload["compression_rows"][0]["split"] = "internal_holdout"
        destination = self.pilot / "rejected_report"
        with self.assertRaises(ValueError):
            build_report(payload, destination)
        self.assertFalse(destination.exists())

    def test_fixed_server_boundary_and_manifest_integrity(self):
        platform = Path("D:/codex学术论文2")
        if str(platform) not in sys.path:
            sys.path.insert(0, str(platform))
        from research_platform.server import _iacs_file
        config = {"pilot_root": str(self.pilot)}
        target = _iacs_file(config, "dev_fixture", "report_fixture", "report.json")
        self.assertEqual(target, (self.destination / "report.json").resolve())
        for name in ["../config.json", "run.json", "prediction.png", "../../secret.json"]:
            with self.assertRaises(ValueError):
                _iacs_file(config, "dev_fixture", "report_fixture", name)
        with self.assertRaises(ValueError):
            _iacs_file(config, "..", "report_fixture", "report.json")
        original = target.read_bytes()
        try:
            target.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                _iacs_file(config, "dev_fixture", "report_fixture", "report.json")
        finally:
            target.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
