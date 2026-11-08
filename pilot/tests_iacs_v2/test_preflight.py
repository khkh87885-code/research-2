"""Tiny local fixtures check repeat-run locks without loading model weights."""

import copy
import json

import pytest

from compression_study.datasets import sha256
from compression_study.engine import code_fingerprint
from iacs_v2.preflight import validate_config


@pytest.fixture
def source_config(tmp_path):
    weights = tmp_path / "weights.fixture"
    weights.write_bytes(b"synthetic model bytes; must never be deserialized")
    source = tmp_path / "sam"
    source.mkdir()
    model = dict(checkpoint=str(weights), sha256=sha256(weights), commit="a" * 40,
                 type="vit_b", device="cuda", source=str(source))
    run = tmp_path / "old"
    run.mkdir()
    specs = [dict(id="baseline", format="baseline"), dict(id="jpeg_q85", format="jpeg", quality=85),
             dict(id="png_l6", format="png")]
    metadata = dict(provenance=dict(config=dict(model=model, conditions=specs),
                                   code=code_fingerprint(), checkpoint_sha256=model["sha256"]))
    (run / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
    for index, spec in enumerate(specs):
        directory = run / "tasks" / ("fixture_%d" % index) / "attempt_0001"
        directory.mkdir(parents=True)
        row = dict(status="ok", dataset="BUS-BRA", split="development", prompt_strategy="mechanism_gt_box",
                   condition=spec["id"], format=spec["format"], scale=1.0, quality=spec.get("quality"))
        (directory / "result.json").write_text(json.dumps(row), encoding="utf-8")
    return dict(model=copy.deepcopy(model), datasets=[dict(dataset="BUS-BRA", split="development")],
                conditions=[spec["id"] for spec in specs], reuse_runs=[str(run)])


def change_source(config, mutate):
    path = __import__("pathlib").Path(config["reuse_runs"][0]) / "run.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data["provenance"])
    path.write_text(json.dumps(data), encoding="utf-8")


def test_matching_defaults_are_accepted_and_all_files_unchanged(source_config):
    root = __import__("pathlib").Path(source_config["model"]["checkpoint"]).parent
    before = {str(path): sha256(path) for path in root.rglob("*") if path.is_file()}
    result = validate_config(source_config)
    assert result["status"] == "ok" and result["reuse_candidate_rows_checked"] == 3
    assert result["device"] == "cuda:0" and result["no_weights_loaded"]
    assert before == {str(path): sha256(path) for path in root.rglob("*") if path.is_file()}


def test_weights_mismatch_fails_before_any_loading(source_config):
    source_config["model"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Actual model weight"):
        validate_config(source_config)


@pytest.mark.parametrize("key,value", [("commit", "b" * 40), ("type", "vit_h"), ("sha256", "0" * 64)])
def test_reused_model_identity_mismatch(source_config, key, value):
    change_source(source_config, lambda p: p["config"]["model"].update({key: value}))
    with pytest.raises(ValueError, match="model configuration"):
        validate_config(source_config)


def test_reused_code_mismatch(source_config):
    change_source(source_config, lambda p: p["code"].update({"fake.py": "different"}))
    with pytest.raises(ValueError, match="code fingerprint"):
        validate_config(source_config)


@pytest.mark.parametrize("index,changes", [(1, {"quality": 60}), (1, {"scale": 0.5}),
    (1, {"order": "encode_resize"}), (1, {"subsampling": 2}), (2, {"compress_level": 9}), (2, {"lossless": False})])
def test_reused_recipe_mismatch(source_config, index, changes):
    change_source(source_config, lambda p: p["config"]["conditions"][index].update(changes))
    with pytest.raises(ValueError, match="[Rr]euse"):
        validate_config(source_config)


@pytest.mark.parametrize("device", ["auto", "gpu", "cuda:-1", "cpu:1"])
def test_unsupported_device_rejected(source_config, device):
    source_config["device"] = device
    with pytest.raises(ValueError, match="devices"):
        validate_config(source_config)


def test_development_only_and_explicit_device_match(source_config):
    source_config["datasets"][0]["split"] = "internal_holdout"
    with pytest.raises(ValueError, match="development only"):
        validate_config(source_config)
    source_config["datasets"][0]["split"] = "development"
    source_config["device"] = "cpu"
    with pytest.raises(ValueError, match="device differs"):
        validate_config(source_config)
