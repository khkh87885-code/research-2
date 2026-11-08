"""Describe outer-OOF risk discrimination and calibration without any tuning.

For logistic policies, auxiliary continuous Ridge predictions and primary
Dice-loss risk probabilities are different model outputs. Negative continuous
regression R-squared does not by itself establish poor risk discrimination.
"""
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ALGORITHMS = ("A_rule_tree", "B_logistic", "C_ridge", "IACS_v2")
QUALITIES = ("jpeg_q95", "jpeg_q85", "jpeg_q75", "jpeg_q60", "jpeg_q40")


def _number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _delta(sample, condition):
    candidates = sample.get("candidates", {})
    candidate, baseline = candidates.get(condition, {}), candidates.get("baseline", {})
    if (candidate.get("status") not in ("ok", "completed")
            or baseline.get("status") not in ("ok", "completed")):
        return None
    current, original = _number(candidate.get("dice")), _number(baseline.get("dice"))
    if current is not None and original is not None:
        return current - original
    return _number(candidate.get("delta_dice"))


def _interval(values):
    return [float(v) for v in np.percentile(values, [2.5, 97.5])] if len(values) >= 2 else None


def _brier_bootstrap(y, probability, prior, repetitions, seed):
    """Jointly resample paired Case losses; no classifier is refitted."""
    if len(y) < 2:
        return {"brier_ci95": None, "training_prevalence_brier_ci95": None,
                "brier_difference_ci95": None, "valid_repetitions": 0}
    model_losses = (probability - y) ** 2
    prior_losses = (prior - y) ** 2
    rng = np.random.default_rng(seed)
    model_means, prior_means, differences = [], [], []
    for start in range(0, repetitions, 128):
        count = min(128, repetitions - start)
        indices = rng.integers(0, len(y), size=(count, len(y)))
        model = model_losses[indices].mean(axis=1)
        comparator = prior_losses[indices].mean(axis=1)
        model_means.extend(model.tolist())
        prior_means.extend(comparator.tolist())
        differences.extend((model - comparator).tolist())
    return {"brier_ci95": _interval(model_means),
            "training_prevalence_brier_ci95": _interval(prior_means),
            "brier_difference_ci95": _interval(differences), "valid_repetitions": repetitions}


def _auc_bootstrap(y, probability, repetitions, seed):
    if repetitions == 0 or len(y) < 2 or len(set(y)) < 2:
        return {"roc_auc_ci95": None, "auc_bootstrap_valid_repetitions": 0,
                "auc_bootstrap_single_class_repetitions": 0}
    rng = np.random.default_rng(seed)
    estimates, single_class = [], 0
    for _ in range(repetitions):
        indices = rng.integers(0, len(y), size=len(y))
        observed = y[indices]
        if len(set(observed)) < 2:
            single_class += 1
            continue
        estimates.append(float(roc_auc_score(observed, probability[indices])))
    return {"roc_auc_ci95": _interval(estimates),
            "auc_bootstrap_valid_repetitions": len(estimates),
            "auc_bootstrap_single_class_repetitions": single_class}


def evaluate_risk(samples, validation, config=None):
    """Return JSON-serializable per-algorithm/per-quality OOF risk descriptions.

    Event = mean within-Case delta Dice < -0.02. All scheduled Cases remain in
    denominators. With multiple images, complete within-Case outcome/probability
    coverage is required; predictions are averaged before evaluating the event.
    The comparator is each Case's OUTER-TRAINING event prevalence, never a
    prevalence estimated from that Case's held-out outcomes.

    Defaults: 2000 paired-Case Brier bootstrap draws and 500 AUC draws. AUC draws
    containing one outcome class are omitted and counted explicitly. Intervals
    condition on fixed OOF models and omit refitting/selection uncertainty.
    """
    config = dict(config or {})
    brier_reps = int(config.get("brier_bootstrap", config.get("bootstrap", 2000)))
    auc_reps = int(config.get("auc_bootstrap", 500))
    seed = int(config.get("seed", 20261108))
    if brier_reps < 2000:
        raise ValueError("Risk Brier-difference CIs require at least 2000 Case draws.")
    if auc_reps < 0:
        raise ValueError("auc_bootstrap must be nonnegative; zero disables its CI.")
    samples = list(samples)
    case_samples = defaultdict(list)
    image_owner = {}
    for sample in samples:
        if sample.get("dataset") != "BUS-BRA" or sample.get("split") != "development":
            raise ValueError("Risk description accepts only BUS-BRA development samples.")
        case, image = sample.get("case_key"), sample.get("image_key")
        if not case or not image or image in image_owner:
            raise ValueError("Unique pseudonymous Case/image samples are required.")
        image_owner[image] = case
        case_samples[case].append(sample)
    cases = sorted(case_samples)
    fold_info = {f["fold"]: f for f in validation.get("folds", [])}
    decisions = defaultdict(dict)
    fold_cases = defaultdict(set)
    case_fold = {}
    for decision in validation.get("outer_decisions", []):
        algorithm = decision.get("algorithm")
        if algorithm not in ALGORITHMS:
            continue
        image, case, fold = decision.get("image_key"), decision.get("case_key"), decision.get("fold")
        if image not in image_owner or image_owner[image] != case or fold not in fold_info:
            raise ValueError("OOF decision identity/fold does not match the supplied samples.")
        if image in decisions[algorithm]:
            raise ValueError("Duplicate OOF probability record for an algorithm/image.")
        decisions[algorithm][image] = decision
        if case in case_fold and case_fold[case] != fold:
            raise ValueError("Images of one Case cross outer folds.")
        case_fold[case] = fold
        fold_cases[fold].add(case)
    for fold, metadata in fold_info.items():
        training = set(metadata.get("train_case_keys", []))
        if training & fold_cases[fold]:
            raise ValueError("Outer-training prevalence would leak a held-out Case.")
        if training - set(cases):
            raise ValueError("Training prevalence refers to Cases outside the supplied development sample.")
    outcome = {}
    for quality in QUALITIES:
        outcome[quality] = {}
        for case in cases:
            values = [_delta(sample, quality) for sample in case_samples[case]]
            outcome[quality][case] = (int(float(np.mean(values)) < -0.02)
                                      if values and all(v is not None for v in values) else None)
    prior_by_quality, training_details = {}, {}
    for quality in QUALITIES:
        prior_by_quality[quality], training_details[quality] = {}, []
        for fold, metadata in sorted(fold_info.items()):
            training = sorted(set(metadata.get("train_case_keys", [])))
            labels = [outcome[quality][case] for case in training if outcome[quality][case] is not None]
            prevalence = float(np.mean(labels)) if labels else None
            prior_by_quality[quality][fold] = prevalence
            training_details[quality].append({"fold": fold, "scheduled_training_cases": len(training),
                "valid_training_cases": len(labels), "missing_training_outcomes": len(training) - len(labels),
                "training_case_prevalence": prevalence})
    rows = []
    for algorithm in ALGORITHMS:
        for quality_index, quality in enumerate(QUALITIES):
            observed = [outcome[quality][case] for case in cases if outcome[quality][case] is not None]
            triples, paired, missing_prob, missing_outcome, missing_prior = [], [], 0, 0, 0
            underlying, invalid_prob = Counter(), 0
            for case in cases:
                image_records = [decisions[algorithm].get(s["image_key"]) for s in case_samples[case]]
                probabilities = [(_number(d.get("predictions", {}).get(quality, {}).get("unsafe_probability"))
                                  if d is not None else None) for d in image_records]
                has_invalid = any(p is not None and not 0 <= p <= 1 for p in probabilities)
                has_missing = any(p is None for p in probabilities)
                if has_invalid:
                    invalid_prob += 1
                if has_missing or has_invalid:
                    missing_prob += 1
                for d in image_records:
                    if d is not None:
                        underlying[d.get("underlying_family", algorithm)] += 1
                label = outcome[quality][case]
                if label is None:
                    missing_outcome += 1
                if has_missing or has_invalid or label is None:
                    continue
                probability = float(np.mean(probabilities))
                triples.append((case, label, probability))
                prior = prior_by_quality[quality].get(case_fold.get(case))
                if prior is None:
                    missing_prior += 1
                else:
                    paired.append((case, label, probability, prior))
            y = np.array([v[1] for v in triples], dtype=int)
            probability = np.array([v[2] for v in triples], dtype=float)
            has_two_classes = len(set(y)) == 2
            row = {"algorithm": algorithm, "condition": quality,
                "event": "Case mean delta_dice < -0.02", "analysis_unit": "Case",
                "scheduled_cases": len(cases), "valid_cases": len(triples),
                "missing_cases": len(cases) - len(triples),
                "missing_or_invalid_probability_cases": missing_prob,
                "invalid_probability_cases": invalid_prob, "missing_outcome_cases": missing_outcome,
                "comparison_valid_cases": len(paired), "missing_training_prevalence_cases": missing_prior,
                "observed_outcome_cases": len(observed),
                "prevalence": float(np.mean(y)) if len(y) else None,
                "all_observed_case_prevalence": float(np.mean(observed)) if observed else None,
                "unsafe_cases": int(np.sum(y)) if len(y) else 0,
                "roc_auc": float(roc_auc_score(y, probability)) if has_two_classes else None,
                "average_precision": float(average_precision_score(y, probability)) if len(y) and np.any(y == 1) else None,
                "roc_auc_status": "estimated" if has_two_classes else "unavailable_single_class_or_missing_probability",
                "brier_score": float(np.mean((probability - y) ** 2)) if len(y) else None,
                "training_prevalence_brier": None, "paired_model_brier": None,
                "brier_difference": None, "brier_difference_direction": "model_minus_outer_training_prevalence; negative favors model",
                "brier_ci95": None, "training_prevalence_brier_ci95": None,
                "brier_ci95_scope": "comparison_valid_cases; corresponds to paired_model_brier",
                "brier_difference_ci95": None, "brier_bootstrap_repetitions": brier_reps,
                "brier_bootstrap_valid_repetitions": 0, "auc_bootstrap_repetitions": auc_reps,
                "roc_auc_ci95": None, "auc_bootstrap_valid_repetitions": 0,
                "auc_bootstrap_single_class_repetitions": 0,
                "status": "evaluated" if triples else "unavailable_no_valid_OOF_risk_probability",
                "underlying_family_image_records": dict(underlying),
                "training_prevalence_by_fold": training_details[quality],
                "probability_role": {"A_rule_tree": "training-leaf empirical event frequency",
                    "B_logistic": "primary LogisticRegression Dice-loss risk probability",
                    "C_ridge": "no risk-probability model is defined for continuous Ridge",
                    "IACS_v2": "inner-selected family risk probability; coverage may vary across outer folds"}[algorithm],
                "continuous_delta_prediction_role": "For B_logistic and B-selected IACS folds, continuous delta_dice is auxiliary Ridge, distinct from logistic risk discrimination.",
                "tuning_performed": False, "model_selection_use": "none; posthoc description only",
                "evidence_type": "outer_case_OOF_risk_description", "seed": seed,
                "bootstrap_conditioning": "Case resampling of fixed OOF predictions; no refitting, model-selection or fold-training-dependence uncertainty",
                "multi_image_policy": "complete images required; mean probability and event(mean delta_dice) within Case",
                "maximum_images_per_case": max((len(v) for v in case_samples.values()), default=0),
                "clinical_acceptability": "not_established; 0.02 is exploratory"}
            if paired:
                pair_y = np.array([v[1] for v in paired], dtype=int)
                pair_probability = np.array([v[2] for v in paired], dtype=float)
                prior = np.array([v[3] for v in paired], dtype=float)
                model_loss, baseline_loss = (pair_probability - pair_y) ** 2, (prior - pair_y) ** 2
                row.update(paired_model_brier=float(model_loss.mean()),
                           training_prevalence_brier=float(baseline_loss.mean()),
                           brier_difference=float((model_loss - baseline_loss).mean()))
                intervals = _brier_bootstrap(pair_y, pair_probability, prior, brier_reps, seed + quality_index)
                row.update({k: v for k, v in intervals.items() if k != "valid_repetitions"})
                row["brier_bootstrap_valid_repetitions"] = intervals["valid_repetitions"]
            if has_two_classes:
                row.update(_auc_bootstrap(y, probability, auc_reps, seed + quality_index + 100))
            rows.append(row)
    return rows
