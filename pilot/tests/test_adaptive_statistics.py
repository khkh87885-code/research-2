"""Synthetic unit fixtures only; these are never research observations."""

import copy
import json

import pytest

from compression_study.adaptive import choose_condition, fit_policy, initial_policy, rule_policy
from compression_study.statistics import mixed_effects, summarize


def record(patient, image, condition="baseline", dice=0.8, **extra):
    result = dict(dataset="BUS-BRA", split="development", prompt_strategy="mechanism",
                  patient_key=patient, image_key=image, condition=condition, status="ok",
                  dice=dice, iou=dice / 2 if dice is not None else None, hd95=2.0, assd=1.0,
                  output_bytes=1000 if condition == "baseline" else 500, original_bytes=1000)
    result.update(extra)
    return result


def comparison(result, metric="dice", condition="jpeg_q95"):
    return next(r for r in result["paired_comparisons"]
                if r["metric"] == metric and r["condition"] == condition)


def test_patient_averaging_prevents_image_pseudoreplication():
    rows = []
    for image in range(10):
        rows += [record("a", "a%d" % image), record("a", "a%d" % image, "jpeg_q95", dice=0.6)]
    rows += [record("b", "b0"), record("b", "b0", "jpeg_q95", dice=1.0)]
    result = summarize(rows, bootstrap=100)
    paired = comparison(result)
    assert paired["paired_patients"] == 2
    assert paired["paired_images"] == 11
    assert paired["patient_mean_difference"] == pytest.approx(0)
    assert paired["ci_exceeds_negative_margin"] is None


def test_pairing_intersects_images_and_retains_failures():
    rows = [record("a", "x", dice=0.1), record("a", "z", dice=0.9),
            record("a", "z", "jpeg_q95", dice=0.8),
            record("b", "y"), record("b", "y", "jpeg_q95", status="error", dice=None)]
    result = summarize(rows, bootstrap=100)
    paired = comparison(result)
    assert paired["patient_mean_difference"] == pytest.approx(-0.1)
    assert paired["paired_patients"] == 1
    assert paired["scheduled_patients"] == 2
    assert paired["patients_without_valid_pair"] == 1
    assert paired["ci95"] is None and paired["wilcoxon_p"] is None
    group = next(g for g in result["groups"] if g["condition"] == "jpeg_q95")
    assert group["failed_images"] == 1


def test_bootstrap_reproducible_margins_and_holm():
    rows = []
    for patient, changed in [("a", 0.79), ("b", 0.80), ("c", 0.81), ("d", 0.82)]:
        rows += [record(patient, patient), record(patient, patient, "jpeg_q95", dice=changed),
                 record(patient, patient, "jpeg_q85", dice=changed - 0.1)]
    first = summarize(rows, bootstrap=200, seed=22, delta=0.05)
    assert first == summarize(rows, bootstrap=200, seed=22, delta=0.05)
    assert comparison(first)["ci_exceeds_negative_margin"] is True
    for item in first["paired_comparisons"]:
        assert item["holm_p"] >= item["wilcoxon_p"]
    assert next(f for f in first["friedman"] if f["metric"] == "dice")["complete_patients"] == 4


def test_duplicate_records_rejected():
    row = record("a", "x")
    with pytest.raises(ValueError, match="Duplicate"):
        summarize([row, row], bootstrap=100)


def test_separate_prompt_cohorts_do_not_pair():
    rows = [record("a", "x"), record("a", "x", "jpeg_q95", prompt_strategy="application")]
    assert comparison(summarize(rows, bootstrap=100))["paired_patients"] == 0


def test_rule_policy_has_no_reference_dependency_and_safe_fallback():
    policy = initial_policy()
    assert policy["status"] == "unvalidated"
    assert policy["uses_reference_at_inference"] is False
    assert choose_condition({"width": 1000}, policy) == "jpeg_q95"
    assert choose_condition({"noise": 0.05, "contrast": 0.1, "clarity": 0.03}, policy) == "jpeg_q95"
    assert choose_condition({"noise": 0.02, "contrast": 0.06, "clarity": 0.01}, policy) == "jpeg_q85"
    assert choose_condition({"noise": 0.005, "contrast": 0.10, "clarity": 0.03}, policy) == "jpeg_q75"
    policy = rule_policy(["baseline"])
    assert choose_condition({}, policy) == "baseline"
    policy["rules"] = [{"feature": "dice", "min": 0, "condition": "baseline"}]
    with pytest.raises(ValueError, match="forbidden"):
        choose_condition({"dice": 0.9}, policy)


def training_fixture():
    rows, features = [], {}
    for i in range(8):
        for j in range(2):
            image = "image%d_%d" % (i, j)
            rows += [record("patient%d" % i, image),
                     record("patient%d" % i, image, "jpeg_q95", dice=0.79),
                     record("patient%d" % i, image, "jpeg_q85", dice=0.6, output_bytes=100)]
            features[image] = {"width": 800 + i * 10, "entropy": 5 + j}
    return rows, features


def test_fitting_requires_prospective_margins_and_rejects_test_data():
    rows, features = training_fixture()
    with pytest.raises(ValueError, match="Prespecify"):
        fit_policy(rows, features, {})
    rows[0]["split"] = "internal_holdout"
    with pytest.raises(ValueError, match="external/holdout"):
        fit_policy(rows, features, {"delta": 0.02, "hd95_delta": 1})


def test_development_tree_portable_lock_and_feasible_labels():
    rows, features = training_fixture()
    policy = fit_policy(rows, features, {"delta": 0.02, "hd95_delta": 1, "min_samples_leaf": 1})
    policy = json.loads(json.dumps(policy, allow_nan=False))
    assert policy["status"] == "unvalidated_research_policy"
    assert policy["training_patients"] + policy["internal_validation_patients"] == 8
    assert policy["training_patient_hash"] != policy["internal_validation_patient_hash"]
    assert choose_condition(features["image0_0"], policy) == "jpeg_q95"
    assert choose_condition({}, policy) == "baseline"
    assert "patient0" not in json.dumps(policy)
    assert policy["policy_sha256"]


def test_fitting_rejects_unknown_patient_and_shared_image():
    rows, features = training_fixture()
    damaged = copy.deepcopy(rows)
    damaged[0]["patient_key"] = ""
    with pytest.raises(ValueError, match="Known patient"):
        fit_policy(damaged, features, {"delta": 0.02, "hd95_delta": 1})
    damaged = copy.deepcopy(rows)
    damaged[0]["patient_key"] = "another_patient"
    with pytest.raises(ValueError, match="multiple patients"):
        fit_policy(damaged, features, {"delta": 0.02, "hd95_delta": 1})


def test_mixed_model_reports_insufficient_data():
    result = mixed_effects([record("a", "x"), record("a", "x", "jpeg_q95")])
    assert result["status"] in ("completed", "unavailable")
    if result["status"] == "completed":
        assert result["cohorts"][0]["status"] == "insufficient_data"
        assert result["cohorts"][0]["converged"] is False


def test_mixed_model_fit_failure_is_explicit(monkeypatch):
    smf = pytest.importorskip("statsmodels.formula.api")
    def failing_model(*args, **kwargs):
        raise RuntimeError("synthetic singular fit")
    monkeypatch.setattr(smf, "mixedlm", failing_model)
    rows = []
    for patient in ("a", "b", "c"):
        rows += [record(patient, patient), record(patient, patient, "jpeg_q95")]
    result = mixed_effects(rows)["cohorts"][0]
    assert result["status"] == "fit_failed"
    assert result["converged"] is False
    assert result["error"] == "RuntimeError"
