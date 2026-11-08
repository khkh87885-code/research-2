"""Synthetic fixtures test invariants; they are never scientific results."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from compression_study.datasets import check_output, read_records, sha256
from compression_study.engine import read_results, run_experiment


@pytest.fixture
def local_config(tmp_path):
    source = tmp_path / "sources"
    models = tmp_path / "models"
    source.mkdir()
    models.mkdir()
    pixels = np.random.default_rng(42).integers(0, 255, (20, 30, 3), dtype=np.uint8)
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[4:16, 8:22] = 255
    Image.fromarray(pixels).save(source / "image.png")
    Image.fromarray(mask).save(source / "mask.png")
    (models / "weights.bin").write_bytes(b"test fixture; never loaded")
    manifest = source / "manifest.csv"
    fields = ["dataset", "split", "patient_id", "image_id", "image_path", "mask_path", "width", "height"]
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(dataset="BUS-BRA", split="development", patient_id="private-patient",
                             image_id="private-image", image_path="image.png", mask_path="mask.png", width=30, height=20))
    return dict(output=str(tmp_path / "run"), purpose="development_technical_smoke", seed=1,
        datasets=[dict(dataset="BUS-BRA", split="development", root=str(source), manifest=str(manifest))],
        model=dict(checkpoint=str(models / "weights.bin"), sha256=sha256(models / "weights.bin"), commit="fixture"),
        conditions=[dict(id="baseline", format="baseline"), dict(id="jpeg_q75", format="jpeg", quality=75, scale=.5)],
        prompt_strategies=["application_full_frame"], max_attempts=2)


class FixtureSAM:
    def __init__(self, config):
        pass

    def predict(self, rgb, box):
        h, w = rgb.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 4:3 * h // 4, w // 4:3 * w // 4] = True
        return mask, dict(inference_ms=1.0, sam_predicted_iou=.8, gpu_peak_bytes=None)


def test_append_only_resume_and_source_protection(local_config, tmp_path, monkeypatch):
    monkeypatch.setattr("compression_study.engine.ExistingSAM", FixtureSAM)
    original = sha256(Path(local_config["datasets"][0]["root"]) / "image.png")
    rows = run_experiment(local_config, tmp_path)
    assert len(rows) == 2 and all(r["status"] == "ok" for r in rows)
    assert "private-patient" not in json.dumps(rows)
    assert "image_path" not in json.dumps(rows)
    assert sha256(Path(local_config["datasets"][0]["root"]) / "image.png") == original
    before = list(Path(local_config["output"]).glob("tasks/*/attempt_*"))
    assert len(run_experiment(local_config, tmp_path, resume=True)) == 2
    assert before == list(Path(local_config["output"]).glob("tasks/*/attempt_*"))
    with pytest.raises(FileExistsError):
        run_experiment(local_config, tmp_path)
    altered = dict(local_config, seed=99)
    with pytest.raises(ValueError, match="Resume refused"):
        run_experiment(altered, tmp_path, resume=True)
    prediction = Path(local_config["output"]) / rows[0]["prediction_mask"]
    prediction.write_bytes(b"corrupt artifact")
    with pytest.raises(ValueError, match="artifact missing or changed"):
        run_experiment(local_config, tmp_path, resume=True)


def test_failed_attempts_persist_and_can_retry(local_config, tmp_path, monkeypatch):
    class Broken(FixtureSAM):
        def predict(self, rgb, box):
            raise RuntimeError("fixture failure")
    monkeypatch.setattr("compression_study.engine.ExistingSAM", Broken)
    rows = run_experiment(local_config, tmp_path)
    assert all(r["status"] == "error" and r["attempt"] == 2 for r in rows)
    monkeypatch.setattr("compression_study.engine.ExistingSAM", FixtureSAM)
    rows = run_experiment(local_config, tmp_path, resume=True, retry_failed=True)
    assert all(r["status"] == "ok" and r["attempt"] == 3 for r in rows)
    assert len(list(Path(local_config["output"]).glob("tasks/*/attempt_*/result.json"))) == 6


def test_output_path_boundaries(local_config, tmp_path):
    with pytest.raises(ValueError):
        check_output(Path(local_config["datasets"][0]["root"]) / "new", local_config)
    with pytest.raises(ValueError):
        check_output(tmp_path, local_config)
    with pytest.raises(ValueError):
        check_output(Path(local_config["model"]["checkpoint"]).parent / "new", local_config)


def test_patient_cross_split_and_path_escape_rejected(local_config):
    first = local_config["datasets"][0]
    text = Path(first["manifest"]).read_text()
    second = Path(first["root"]) / "other.csv"
    second.write_text(text.replace("development", "internal_holdout").replace("private-image", "other-image"))
    local_config["datasets"].append(dict(first, split="internal_holdout", manifest=str(second)))
    with pytest.raises(ValueError, match="more than one split"):
        read_records(local_config, b"test salt")
    local_config["datasets"] = [first]
    Path(first["manifest"]).write_text(text.replace("image.png", "../models/weights.bin"))
    with pytest.raises(ValueError, match="escapes"):
        read_records(local_config, b"test salt")
