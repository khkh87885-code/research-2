"""Synthetic OOF probabilities only; no clinical or research results."""
import copy
import json

import pytest

from iacs_v2.risk_evaluation import ALGORITHMS, QUALITIES, evaluate_risk


def fixture():
    samples, decisions = [], []
    for i in range(8):
        case, image = "case%d" % i, "image%d" % i
        unsafe = i % 2 == 1
        candidates = {"baseline": {"status": "ok", "dice": 0.9}}
        for quality in QUALITIES:
            candidates[quality] = {"status": "ok", "dice": 0.86 if unsafe else 0.895,
                                   "delta_dice": -0.04 if unsafe else -0.005}
        samples.append(dict(case_key=case, image_key=image, dataset="BUS-BRA", split="development",
                            candidates=candidates))
        for algorithm in ALGORITHMS:
            probability = None if algorithm == "C_ridge" else 0.9 if unsafe else 0.1
            decisions.append(dict(case_key=case, image_key=image, fold=i // 4,
                                  algorithm=algorithm, underlying_family=algorithm,
                                  predictions={q: {"unsafe_probability": probability} for q in QUALITIES}))
    folds = [{"fold": 0, "train_case_keys": ["case%d" % i for i in range(4, 8)]},
             {"fold": 1, "train_case_keys": ["case%d" % i for i in range(4)]}]
    return samples, dict(folds=folds, outer_decisions=decisions)


def select(rows, algorithm="B_logistic", condition="jpeg_q85"):
    return next(r for r in rows if r["algorithm"] == algorithm and r["condition"] == condition)


def test_primary_logistic_probability_is_evaluated_separately_from_auxiliary_ridge():
    samples, validation = fixture()
    rows = evaluate_risk(samples, validation, {"auc_bootstrap": 0})
    row = select(rows)
    assert len(rows) == 20
    assert row["scheduled_cases"] == row["valid_cases"] == 8
    assert row["prevalence"] == 0.5
    assert row["roc_auc"] == row["average_precision"] == 1.0
    assert row["brier_score"] == pytest.approx(0.01)
    assert row["training_prevalence_brier"] == 0.25
    assert row["brier_difference"] == pytest.approx(-0.24)
    assert row["brier_bootstrap_valid_repetitions"] == 2000
    assert row["brier_difference_ci95"] == pytest.approx([-0.24, -0.24])
    assert "auxiliary Ridge" in row["continuous_delta_prediction_role"]
    assert row["tuning_performed"] is False
    assert json.loads(json.dumps(rows, allow_nan=False))


def test_regression_only_family_is_unavailable_instead_of_fake_probability():
    samples, validation = fixture()
    row = select(evaluate_risk(samples, validation, {"auc_bootstrap": 0}), "C_ridge")
    assert row["scheduled_cases"] == row["missing_cases"] == 8
    assert row["valid_cases"] == 0
    assert row["status"] == "unavailable_no_valid_OOF_risk_probability"
    assert row["roc_auc"] is None and row["brier_score"] is None
    assert row["all_observed_case_prevalence"] == 0.5


def test_outer_training_prevalence_ignores_heldout_labels():
    samples, validation = fixture()
    first = select(evaluate_risk(samples, validation, {"auc_bootstrap": 0}))
    changed = copy.deepcopy(samples)
    for sample in changed[:4]:
        for quality in QUALITIES:
            sample["candidates"][quality]["dice"] = 0.1
            sample["candidates"][quality]["delta_dice"] = -0.8
    second = select(evaluate_risk(changed, validation, {"auc_bootstrap": 0}))
    assert first["training_prevalence_by_fold"][0] == second["training_prevalence_by_fold"][0]
    assert first["training_prevalence_by_fold"][1] != second["training_prevalence_by_fold"][1]


def test_single_class_has_no_auc_and_missing_probability_denominator_is_explicit():
    samples, validation = fixture()
    for sample in samples:
        for quality in QUALITIES:
            sample["candidates"][quality].update(dice=0.895, delta_dice=-0.005)
    for d in validation["outer_decisions"]:
        if d["algorithm"] == "B_logistic" and d["case_key"] == "case0":
            d["predictions"]["jpeg_q85"]["unsafe_probability"] = None
    row = select(evaluate_risk(samples, validation, {"auc_bootstrap": 0}))
    assert row["valid_cases"] == 7 and row["missing_cases"] == 1
    assert row["roc_auc"] is None and row["average_precision"] is None
    assert row["prevalence"] == 0


def test_case_aggregation_uses_mean_delta_event_and_never_image_pseudoreplication():
    samples, validation = fixture()
    extra = copy.deepcopy(samples[0])
    extra["image_key"] = "image0_second"
    for quality in QUALITIES:
        extra["candidates"][quality].update(dice=0.87, delta_dice=-0.03)
    samples.append(extra)
    for algorithm in ALGORITHMS:
        decision = copy.deepcopy(next(d for d in validation["outer_decisions"]
                                      if d["case_key"] == "case0" and d["algorithm"] == algorithm))
        decision["image_key"] = extra["image_key"]
        validation["outer_decisions"].append(decision)
    row = select(evaluate_risk(samples, validation, {"auc_bootstrap": 0}))
    # (-.005-.03)/2=-.0175, so Case0 remains a non-event despite one unsafe image.
    assert row["scheduled_cases"] == row["valid_cases"] == 8
    assert row["unsafe_cases"] == 4
    assert row["maximum_images_per_case"] == 2


def test_auc_bootstrap_is_reproducible_and_input_is_unchanged():
    samples, validation = fixture()
    before = copy.deepcopy((samples, validation))
    first = evaluate_risk(samples, validation, {"auc_bootstrap": 30})
    assert first == evaluate_risk(samples, validation, {"auc_bootstrap": 30})
    assert (samples, validation) == before
    row = select(first)
    assert row["roc_auc_ci95"] == [1.0, 1.0]
    assert row["auc_bootstrap_valid_repetitions"] + row["auc_bootstrap_single_class_repetitions"] == 30


def test_invalid_probability_and_leakage_are_rejected_or_counted():
    samples, validation = fixture()
    for d in validation["outer_decisions"]:
        if d["algorithm"] == "B_logistic" and d["case_key"] == "case0":
            d["predictions"]["jpeg_q85"]["unsafe_probability"] = 1.1
    row = select(evaluate_risk(samples, validation, {"auc_bootstrap": 0}))
    assert row["invalid_probability_cases"] == 1
    validation["folds"][0]["train_case_keys"].append("case0")
    with pytest.raises(ValueError, match="leak"):
        evaluate_risk(samples, validation, {"auc_bootstrap": 0})


def test_external_data_and_underpowered_bootstrap_are_not_silently_accepted():
    samples, validation = fixture()
    with pytest.raises(ValueError, match="2000"):
        evaluate_risk(samples, validation, {"bootstrap": 100})
    samples[0]["split"] = "internal_holdout"
    with pytest.raises(ValueError, match="development"):
        evaluate_risk(samples, validation, {"auc_bootstrap": 0})
