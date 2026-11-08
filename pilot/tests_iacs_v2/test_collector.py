"""Synthetic fixtures only: validate append-only collection and safety invariants."""
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from compression_study.datasets import sha256
from iacs_v2.collector import collect, latest_rows


@pytest.fixture
def config(tmp_path):
    project = tmp_path / "project"
    data = tmp_path / "data"
    models = tmp_path / "models"
    for path in (project, data, models):
        path.mkdir()
    rgb = np.random.default_rng(7).integers(0, 255, (20, 30, 3), dtype=np.uint8)
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[4:14, 6:18] = 255
    Image.fromarray(rgb).save(data / "image.png")
    Image.fromarray(mask).save(data / "mask.png")
    (models / "weights").write_bytes(b"synthetic fixture; never loaded")
    with (data / "manifest.csv").open("w", newline="") as stream:
        fields = ["dataset", "split", "patient_id", "image_id", "image_path", "mask_path", "width", "height"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(dataset="BUS-BRA", split="development", patient_id="fixture-case", image_id="fixture-image",
                             image_path="image.png", mask_path="mask.png", width=30, height=20))
    return dict(project_root=str(project), run_root=str(project / "research_output" / "IACS_v2" / "test"),
                datasets=[dict(dataset="BUS-BRA", split="development", root=str(data), manifest=str(data / "manifest.csv"))],
                model=dict(checkpoint=str(models / "weights"), sha256=sha256(models / "weights"), commit="fixture"),
                conditions=["baseline", "jpeg_q85", "png_l6"], max_attempts=2)


class FakeSAM:
    calls = 0
    def __init__(self, config):
        self.info = {"synthetic": True}

    def predict(self, rgb, box):
        FakeSAM.calls += 1
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        mask[3:15, 5:19] = True
        return mask, dict(inference_ms=1.0, sam_predicted_iou=.8, gpu_peak_bytes=0)


def test_collection_png_reuse_resume_protection(config, monkeypatch):
    FakeSAM.calls = 0
    monkeypatch.setattr("iacs_v2.collector.ExistingSAM", FakeSAM)
    original = sha256(Path(config["datasets"][0]["root"]) / "image.png")
    rows, summary = collect(config)
    assert summary["statuses"] == {"ok": 3}
    assert FakeSAM.calls == 2  # lossless pixels reuse actual baseline prediction
    png = next(r for r in rows if r["condition"] == "png_l6")
    assert png["origin"] == "lossless_pixel_identity_reuse" and png["pixel_equal"]
    collect(config, resume=True)
    assert FakeSAM.calls == 2
    assert sha256(Path(config["datasets"][0]["root"]) / "image.png") == original
    with pytest.raises(ValueError, match="identical"):
        collect(config)
    prediction = Path(config["run_root"]) / png["prediction_mask"]
    prediction.write_bytes(b"corrupt synthetic artifact")
    with pytest.raises(ValueError, match="artifact"):
        collect(config, resume=True)


def test_external_and_invalid_attempts_are_rejected(config):
    config["datasets"][0]["split"] = "external_test"
    with pytest.raises(ValueError, match="development only"):
        collect(config)
    config["datasets"][0]["split"] = "development"
    config["max_attempts"] = 0
    with pytest.raises(ValueError, match="positive"):
        collect(config)


def test_failures_remain_and_retry_new_attempt(config, monkeypatch):
    class Broken(FakeSAM):
        def predict(self, rgb, box):
            raise RuntimeError("synthetic failure")
    monkeypatch.setattr("iacs_v2.collector.ExistingSAM", Broken)
    rows, summary = collect(config)
    assert summary["statuses"] == {"error": 3}
    monkeypatch.setattr("iacs_v2.collector.ExistingSAM", FakeSAM)
    rows, summary = collect(config, resume=True, retry_failed=True)
    assert summary["statuses"] == {"ok": 3}
    assert all(r["attempt"] == 3 for r in rows)
    assert len(list(Path(config["run_root"]).glob("compression/tasks/*/*/attempt_*/result.json"))) == 9


def test_sensitivity_keeps_not_run_conditions_without_faking_values(config, monkeypatch):
    from iacs_v2.sensitivity import prepare
    monkeypatch.setattr("iacs_v2.collector.ExistingSAM", FakeSAM)
    collect(config)
    samples = prepare(config)
    assert len(samples) == 1
    candidate = samples[0]["candidates"]["jpeg_q40"]
    assert candidate["status"] == "not_run" and candidate["dice"] is None
    assert candidate["joint_constraint_status"] == "not_defined_no_HD95_margin"
    assert samples[0]["features"]
    assert "lesion_area_fraction" not in samples[0]["features"]
    assert "lesion_area_fraction" in samples[0]["evaluation"]
    with pytest.raises(FileExistsError):
        prepare(config)
