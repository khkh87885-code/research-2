"""Synthetic algorithmic tests; never interpreted as scientific measurements."""
import copy
import json

import numpy as np
import pytest

from iacs_v2.features import FEATURE_NAMES, FEATURE_VERSION
from iacs_v2.validation import (FAMILIES, QUALITIES, fit_policy, predict_policy,
                                run_validation, verify_policy)


def samples_fixture(cases=10, images=1, all_acceptable=False):
    output = []
    for case in range(cases):
        for image in range(images):
            features = {name: float((i + 1) * 0.01 + case * 0.001 + image * 0.0001)
                        for i, name in enumerate(FEATURE_NAMES)}
            features.update(width=float(400 + case * 10), height=float(300 + image * 5),
                            laplacian_variance=0.002 + case * 0.001,
                            noise_mad=0.01 + case * 0.002,
                            local_contrast=0.03 + case * 0.004)
            baseline = {"status": "ok", "dice": 0.9, "hd95": 10.0, "assd": 2.0, "iou": 0.8,
                        "precision": 0.9, "recall": 0.9, "boundary_f1": 0.85,
                        "output_bytes": 1000, "original_bytes": 1000, "pixel_equal": True}
            candidates = {"baseline": baseline, "png_l6": dict(baseline, output_bytes=1100)}
            for j, quality in enumerate(QUALITIES):
                loss = 0.005 if all_acceptable else 0.005 + j * 0.007 + case * j * 0.001
                candidates[quality] = dict(baseline, dice=0.9 - loss,
                                            hd95=10.0 + case * j * 0.2,
                                            output_bytes=800 - j * 150, pixel_equal=False)
            output.append({"case_key": "case_%02d" % case, "image_key": "image_%02d_%d" % (case, image),
                           "dataset": "BUS-BRA", "split": "development", "prompt_strategy": "mechanism_gt_box",
                           "features": features, "evaluation": {"lesion_area_fraction": case / 100 + 0.01,
                                                                   "shape_irregularity": case + 1.0},
                           "candidates": candidates})
    return output


def test_scaler_and_feature_selection_use_training_only():
    samples = samples_fixture()
    train, heldout = samples[:8], copy.deepcopy(samples[8:])
    policy = fit_policy(train, "C_ridge")
    expected = np.mean([[s["features"][n] for n in FEATURE_NAMES] for s in train], axis=0)
    assert policy["scaler"]["mean"] == pytest.approx(expected)
    heldout[0]["features"]["width"] = 1e12
    heldout[0]["candidates"]["jpeg_q40"]["dice"] = -1e9
    assert fit_policy(train, "C_ridge") == policy
    assert policy["feature_version"] == FEATURE_VERSION
    assert "lesion_area_fraction" not in policy["feature_names"]


@pytest.mark.parametrize("family", FAMILIES)
def test_heldout_outcomes_cannot_change_decision(family):
    samples = samples_fixture()
    policy = fit_policy(samples[:8], family)
    heldout = copy.deepcopy(samples[8])
    first = predict_policy(policy, heldout["features"], heldout["candidates"])
    for candidate in heldout["candidates"].values():
        candidate.update(dice=-1000, hd95=1e12, delta_dice=-1e12,
                         delta_hd95=1e12, status="failed", iou=None)
    heldout["evaluation"] = {"lesion_area_fraction": 1e12, "shape_irregularity": -100}
    assert predict_policy(policy, heldout["features"], heldout["candidates"]) == first


def test_missing_feature_has_explicit_lossless_fallback():
    samples = samples_fixture()
    policy = fit_policy(samples, "C_ridge")
    features = dict(samples[0]["features"])
    features.pop("noise_mad")
    candidates = copy.deepcopy(samples[0]["candidates"])
    candidates["png_l6"].update(output_bytes=900, pixel_equal=True)
    result = predict_policy(policy, features, candidates)
    assert result["selected_condition"] == "png_l6"
    assert result["fallback"] is True
    assert result["reason"] == "missing_features_fallback"
    assert result["missing_features"] == ["noise_mad"]
    candidates["png_l6"]["pixel_equal"] = False
    assert predict_policy(policy, features, candidates)["selected_condition"] == "baseline"


def test_single_class_risk_is_explicit_constant():
    samples = samples_fixture()
    policy = fit_policy(samples, "B_logistic")
    assert policy["models"]["jpeg_q95"]["risk"]["single_class"] is True
    assert policy["models"]["jpeg_q95"]["risk"]["kind"] == "constant"
    assert policy["models"]["jpeg_q40"]["risk"]["value"] == 1.0
    result = predict_policy(policy, samples[0]["features"], samples[0]["candidates"])
    assert result["predictions"]["jpeg_q95"]["unsafe_probability"] == 0
    assert result["predictions"]["jpeg_q40"]["unsafe_probability"] == 1


@pytest.mark.parametrize("family", FAMILIES)
def test_policy_does_not_silently_choose_q85_and_has_no_hd95_gate(family):
    samples = samples_fixture(all_acceptable=True)
    for sample in samples:
        sample["candidates"]["jpeg_q40"]["hd95"] = 10000
    policy = fit_policy(samples, family)
    result = predict_policy(policy, samples[0]["features"], samples[0]["candidates"])
    assert result["selected_condition"] == "jpeg_q40"
    assert result["predictions"]["jpeg_q40"]["predicted_delta_hd95"] > 1000
    assert result["HD95_status"] == "not_specified"
    assert result["clinical_acceptability"] == "not_established"


def test_policy_freezehash_and_scope_rejection():
    samples = samples_fixture()
    policy = fit_policy(samples)
    assert verify_policy(json.loads(json.dumps(policy, allow_nan=False)))
    policy["parameters"]["depth"] = 2
    with pytest.raises(ValueError, match="hash mismatch"):
        predict_policy(policy, samples[0]["features"], samples[0]["candidates"])
    with pytest.raises(ValueError, match="HD95 margin"):
        fit_policy(samples, config={"hd95_margin": 2})
    with pytest.raises(ValueError, match="0.02"):
        fit_policy(samples, config={"dice_margin": 0.03})
    samples[0]["split"] = "internal_holdout"
    with pytest.raises(ValueError, match="development"):
        fit_policy(samples)


def test_no_training_features_still_preserves_cases_and_falls_back():
    samples = samples_fixture()
    for sample in samples:
        sample["features"] = {}
    policy = fit_policy(samples)
    assert policy["training_missing_feature_images"] == len(samples)
    assert policy["training_cases"] == 10
    result = predict_policy(policy, {}, samples[0]["candidates"])
    assert result["fallback"] and result["selected_condition"] == "baseline"


def test_nested_case_cv_and_scaler_isolation(tmp_path):
    samples = samples_fixture(cases=10, images=2)
    config = {"a_depths": [1], "b_risk_thresholds": [0.25], "c_alphas": [10],
              "min_samples_leaf": 2, "dice_margin": 0.02, "hd95_margin": None}
    result = run_validation(samples, tmp_path / "cv", config)
    assert result["sample_cases"] == 10
    assert len(result["outer_decisions"]) == 20 * 4
    assert result["clinical_noninferiority_conclusion"] is None
    assigned = {}
    for item in result["fold_assignments"]:
        assigned.setdefault(item["case_key"], set()).add(item["fold"])
    assert all(len(folds) == 1 for folds in assigned.values())
    for fold in result["folds"]:
        assert set(fold["train_case_keys"]).isdisjoint(fold["validation_case_keys"])
        assert len(fold["inner_selection"]["inner_folds"]) == 3
        train = [s for s in samples if s["case_key"] in fold["train_case_keys"]]
        expected = np.mean([[s["features"][n] for n in FEATURE_NAMES] for s in train], axis=0)
        policy = json.loads((tmp_path / "cv" / ("fold_%d_C_ridge_policy.json" % fold["fold"])).read_text())
        assert policy["scaler"]["mean"] == pytest.approx(expected)
        assert fold["subgroup_thresholds"]["lesion_area_fraction"] == pytest.approx(
            np.median([s["evaluation"]["lesion_area_fraction"] for s in train]))
        assert fold["selected_family"] in FAMILIES
    assert (tmp_path / "cv" / "candidate_predictions.csv").is_file()
    assert (tmp_path / "cv" / "oof_decisions.csv").is_file()
    assert verify_policy(result["full_policy"])
