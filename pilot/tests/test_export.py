import csv
import json
from pathlib import Path

from compression_study.export import clean_rows, export_results, summarize_local
from compression_study.dashboard import build_dashboard


def records():
    common = {"dataset": "fixture", "split": "test", "condition": "jpeg85", "prompt_strategy": "mechanism", "status": "ok", "original_bytes": 100, "output_bytes": 40}
    return [dict(common, patient_key="p1", image_key="i1", dice=.6),
            dict(common, patient_key="p1", image_key="i2", dice=1),
            dict(common, patient_key="p2", image_key="i3", dice=.2),
            dict(common, patient_key=None, image_key="i4", dice=.9),
            dict(common, patient_key="p2", image_key="i5", status="failed", dice=None)]


def test_patient_equal_weight_and_failed_preserved():
    patients, groups = summarize_local(clean_rows(records()))
    assert len(patients) == 2
    assert abs(groups[0]["dice_patient_mean"] - .5) < 1e-12
    assert groups[0]["n_records"] == 5
    assert groups[0]["n_failed"] == 1
    assert groups[0]["n_missing_patient"] == 1


def test_real_export_and_private_fields_omitted(tmp_path):
    rows = records()
    rows[0]["patient_id"] = "PRIVATE_RAW_ID"
    rows[0]["image_path"] = "PRIVATE_RAW_PATH"
    paths = export_results(rows, tmp_path, metadata={"run_id": "fixture", "data_path": "PRIVATE_DATA_PATH"})
    for path in paths.values():
        assert Path(path).is_file()
    with Path(paths["image_results_csv"]).open(encoding="utf-8-sig", newline="") as handle:
        exported = list(csv.DictReader(handle))
    assert len(exported) == 5
    assert exported[-1]["status"] == "failed"
    assert "PRIVATE_RAW_ID" not in Path(paths["image_results_csv"]).read_text(encoding="utf-8-sig")
    assert "PRIVATE_RAW_PATH" not in Path(paths["image_results_csv"]).read_text(encoding="utf-8-sig")
    assert "PRIVATE_DATA_PATH" not in Path(paths["report"]).read_text(encoding="utf-8")
    from openpyxl import load_workbook
    book = load_workbook(paths["xlsx"], read_only=True)
    assert book["image_results"].max_row == 6
    assert book["patient_results"].max_row == 3
    book.close()


def test_dashboard_escapes_html_and_keeps_failures(tmp_path):
    rows = records()
    rows[0]["condition"] = "</script><img src=x onerror=alert(1)>"
    path = build_dashboard(rows, tmp_path / "dashboard.html")
    html = Path(path).read_text(encoding="utf-8")
    assert "</script><img" not in html
    data = html.split('id="data">', 1)[1].split("</script>", 1)[0]
    parsed = json.loads(data)
    assert len(parsed["rows"]) == 5
    assert parsed["rows"][-1]["status"] == "failed"
    assert "connect-src 'none'" in html
    assert "https://" not in html


def test_nonfinite_metrics_are_missing_and_spreadsheet_safe(tmp_path):
    rows = records()
    rows[0]["dice"] = float("nan")
    rows[0]["image_key"] = "=1+1"
    paths = export_results(rows, tmp_path)
    with Path(paths["image_results_csv"]).open(encoding="utf-8-sig", newline="") as handle:
        exported = list(csv.DictReader(handle))
    assert exported[0]["dice"] == ""
    assert exported[0]["image_key"] == "'=1+1"
    dashboard = build_dashboard(rows, tmp_path / "dashboard.html")
    assert 'NaN' not in Path(dashboard).read_text(encoding="utf-8")
