"""Development-only nested case CV with portable, image-only decisions.

Clinical margins are unavailable. The sole exploratory Dice-loss candidate is
0.02; HD95 predictions are reported, never used as an acceptance gate.
"""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeRegressor

from .features import FEATURE_NAMES, FEATURE_VERSION


VERSION = "iacs_v2_nested_case_cv_1"
QUALITIES = ("jpeg_q95", "jpeg_q85", "jpeg_q75", "jpeg_q60", "jpeg_q40")
FAMILIES = ("A_rule_tree", "B_logistic", "C_ridge")
METRICS = ("dice", "hd95", "assd", "iou", "precision", "recall", "boundary_f1")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _settings(config=None):
    config = dict(config or {})
    margin = config.get("dice_loss_margin", config.get("dice_margin", config.get("delta", 0.02)))
    if _number(margin) != 0.02:
        raise ValueError("Only the exploratory Dice loss 0.02 candidate is authorized.")
    if config.get("hd95_margin") is not None or config.get("hd95_delta") is not None:
        raise ValueError("HD95 margin is not specified; do not invent an HD95 acceptance gate.")
    if int(config.get("outer_folds", 5)) != 5 or int(config.get("inner_folds", 3)) != 3:
        raise ValueError("The prespecified design requires outer5 and inner3 folds.")
    return dict(dice_loss_margin=0.02, hd95_margin=None, seed=int(config.get("seed", 20261108)),
                outer_folds=5, inner_folds=3, feature_names=list(FEATURE_NAMES),
                a_depths=list(config.get("a_depths", [1, 2])),
                b_risk_thresholds=list(config.get("b_risk_thresholds", [0.1, 0.25, 0.5])),
                c_alphas=list(config.get("c_alphas", [1.0, 10.0, 100.0])),
                min_samples_leaf=int(config.get("min_samples_leaf", 5)))


def _validate_samples(samples):
    seen = set()
    owners = defaultdict(set)
    for sample in samples:
        if sample.get("dataset") != "BUS-BRA" or sample.get("split") != "development":
            raise ValueError("Only BUS-BRA development samples may enter IACS_v2 development.")
        case, image = sample.get("case_key"), sample.get("image_key")
        if not case or not image:
            raise ValueError("Known pseudonymous case_key/image_key are required.")
        identity = (str(case), str(image))
        if identity in seen:
            raise ValueError("Duplicate case/image sample.")
        seen.add(identity)
        owners[str(image)].add(str(case))
        if len(owners[str(image)]) > 1:
            raise ValueError("An image cannot belong to multiple cases.")
        prompt = sample.get("prompt_strategy", "mechanism_gt_box")
        if prompt not in ("mechanism", "mechanism_gt_box"):
            raise ValueError("Only the reference-localized mechanism experiment is allowed here.")


def _feature_vector(features):
    features = features if isinstance(features, dict) else {}
    values = [_number(features.get(name)) for name in FEATURE_NAMES]
    return None if any(v is None for v in values) else values


def _delta(sample, condition, metric="dice"):
    candidates = sample.get("candidates", {})
    baseline, candidate = candidates.get("baseline", {}), candidates.get(condition, {})
    if baseline.get("status") not in ("ok", "completed") or candidate.get("status") not in ("ok", "completed"):
        return None
    original, changed = _number(baseline.get(metric)), _number(candidate.get(metric))
    if original is None or changed is None:
        return None
    return changed - original


def _linear(X, values, alpha):
    keep = [i for i, value in enumerate(values) if value is not None]
    if not keep:
        return {"kind": "unavailable", "n": 0}
    target = np.array([values[i] for i in keep], dtype=float)
    if len(keep) < 2 or np.ptp(target) <= 1e-14:
        return {"kind": "constant", "value": float(target.mean()), "n": len(keep)}
    model = Ridge(alpha=float(alpha))
    model.fit(X[keep], target)
    return {"kind": "linear", "coefficients": model.coef_.tolist(),
            "intercept": float(model.intercept_), "alpha": float(alpha), "n": len(keep)}


def _linear_predict(model, vector):
    if model["kind"] == "unavailable":
        return None
    if model["kind"] == "constant":
        return float(model["value"])
    return float(np.dot(model["coefficients"], vector) + model["intercept"])


def fit_policy(samples, family="A_rule_tree", parameters=None, config=None):
    """Fit exclusively on the samples passed here; return JSON, never pickle.

    Scaling and removal of constant features use this training set only.
    Missing-feature cases stay documented and are not assigned synthetic values.
    """
    samples = list(samples)
    _validate_samples(samples)
    settings = _settings(config)
    if family not in FAMILIES:
        raise ValueError("Unknown policy family.")
    defaults = {"A_rule_tree": {"depth": 1}, "B_logistic": {"risk_threshold": 0.25},
                "C_ridge": {"alpha": 10.0}}
    parameters = dict(defaults[family], **dict(parameters or {}))
    if family == "A_rule_tree" and int(parameters["depth"]) not in (1, 2):
        raise ValueError("Only shallow tree depths 1 and 2 are supported.")
    if family == "B_logistic" and not 0 < float(parameters["risk_threshold"]) < 1:
        raise ValueError("Risk threshold must be strictly between zero and one.")
    if family == "C_ridge" and float(parameters["alpha"]) <= 0:
        raise ValueError("Ridge regularization must be positive.")
    valid = [(sample, _feature_vector(sample.get("features"))) for sample in samples]
    valid = [(sample, vector) for sample, vector in valid if vector is not None]
    policy = {"schema_version": VERSION, "family": family, "parameters": parameters,
              "status": "development_only_unvalidated", "feature_version": FEATURE_VERSION,
              "feature_names": list(FEATURE_NAMES), "dice_loss_margin": 0.02,
              "HD95_status": "not_specified", "hd95_margin": None,
              "uses_reference_at_inference": False,
              "training_cases": len({str(s["case_key"]) for s in samples}),
              "training_images": len(samples), "training_valid_feature_images": len(valid),
              "training_missing_feature_images": len(samples) - len(valid),
              "training_case_hash": digest(sorted({str(s["case_key"]) for s in samples})),
              "training_image_hash": digest(sorted(str(s["image_key"]) for s in samples)),
              "training_sample_hash": digest(samples),
              "source_model_fingerprints": sorted({
                  str(c.get("weight_sha256", c.get("model_weight_hash", "not_recorded")))
                  for sample in samples for c in sample.get("candidates", {}).values()}),
              "feature_selection": "remove training-constant features only", "models": {}}
    if not valid:
        policy["scaler"] = {"mean": [0.0] * len(FEATURE_NAMES), "scale": [1.0] * len(FEATURE_NAMES),
                            "active_indices": [0], "no_training_features": True}
        policy["policy_sha256"] = digest(policy)
        return policy
    train = [sample for sample, _ in valid]
    raw = np.array([vector for _, vector in valid], dtype=float)
    mean, std = raw.mean(axis=0), raw.std(axis=0)
    active = np.flatnonzero(std > 1e-12).tolist()
    # One explicitly constant column lets intercept-only fits remain portable.
    if not active:
        active = [0]
    std = np.where(std > 1e-12, std, 1.0)
    X = ((raw - mean) / std)[:, active]
    policy["scaler"] = {"mean": mean.tolist(), "scale": std.tolist(), "active_indices": active,
                        "no_training_features": False}
    targets = {q: [_delta(s, q) for s in train] for q in QUALITIES}
    for q in QUALITIES:
        delta = targets[q]
        hd = [_delta(s, q, "hd95") for s in train]
        entry = {"dice_auxiliary": _linear(X, delta, float(parameters.get("alpha", 10))),
                 "hd95_auxiliary": _linear(X, hd, float(parameters.get("alpha", 10)))}
        if family == "B_logistic":
            keep = [i for i, value in enumerate(delta) if value is not None]
            labels = np.array([int(delta[i] < -0.02) for i in keep], dtype=int)
            if not keep:
                entry["risk"] = {"kind": "unavailable", "n": 0, "single_class": False}
            elif len(set(labels)) == 1:
                entry["risk"] = {"kind": "constant", "value": float(labels[0]),
                                 "n": len(keep), "single_class": True}
            else:
                classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500,
                                                random_state=settings["seed"])
                classifier.fit(X[keep], labels)
                entry["risk"] = {"kind": "logistic", "coefficients": classifier.coef_[0].tolist(),
                                 "intercept": float(classifier.intercept_[0]), "n": len(keep),
                                 "single_class": False, "classes": [0, 1],
                                 "iterations": int(classifier.n_iter_[0]),
                                 "converged_before_iteration_limit": bool(classifier.n_iter_[0] < 500)}
        policy["models"][q] = entry
    if family == "A_rule_tree":
        worst = [min([targets[q][i] for q in QUALITIES if targets[q][i] is not None], default=None)
                 for i in range(len(train))]
        keep = [i for i, value in enumerate(worst) if value is not None]
        if keep:
            estimator = DecisionTreeRegressor(max_depth=int(parameters["depth"]),
                min_samples_leaf=max(1, settings["min_samples_leaf"]), random_state=settings["seed"])
            estimator.fit(X[keep], [worst[i] for i in keep])
            assignments = estimator.apply(X)
            nodes = []
            for index in range(estimator.tree_.node_count):
                tree = estimator.tree_
                node = {"feature_index": int(tree.feature[index]), "threshold": float(tree.threshold[index]),
                        "left": int(tree.children_left[index]), "right": int(tree.children_right[index])}
                if node["left"] == node["right"]:
                    node["leaf_estimates"] = {}
                    for q in QUALITIES:
                        values = [targets[q][i] for i in range(len(train))
                                  if assignments[i] == index and targets[q][i] is not None]
                        node["leaf_estimates"][q] = {"n": len(values),
                            "predicted_delta_dice": float(np.mean(values)) if values else None,
                            "unsafe_probability": float(np.mean(np.array(values) < -0.02)) if values else None}
                nodes.append(node)
            policy["tree"] = nodes
        else:
            policy["tree"] = []
    policy["policy_sha256"] = digest(policy)
    return policy


def verify_policy(policy):
    payload = dict(policy)
    expected = payload.pop("policy_sha256", None)
    if not expected or digest(payload) != expected:
        raise ValueError("Frozen policy hash mismatch.")
    if payload.get("hd95_margin") is not None or payload.get("dice_loss_margin") != 0.02:
        raise ValueError("Policy margins are outside the authorized exploratory scope.")
    return True


def predict_policy(policy, features, candidates):
    """Pure deployment decision: features and encoding metadata only.

    Candidate status, Dice, HD95, lesion evaluation and all other observed model
    outcomes are deliberately unread. Those are attached later for evaluation.
    """
    verify_policy(policy)
    raw = _feature_vector(features)
    missing = [name for name in FEATURE_NAMES if _number((features or {}).get(name)) is None]
    usable = raw is not None and not policy["scaler"].get("no_training_features")
    predictions = {}
    vector, leaf = None, None
    if usable:
        scaler = policy["scaler"]
        vector = ((np.array(raw) - scaler["mean"]) / scaler["scale"])[scaler["active_indices"]]
        if policy["family"] == "A_rule_tree" and policy.get("tree"):
            leaf = policy["tree"][0]
            while "leaf_estimates" not in leaf:
                index = leaf["left"] if vector[leaf["feature_index"]] <= leaf["threshold"] else leaf["right"]
                leaf = policy["tree"][index]
    lossy_available = []
    for q in QUALITIES:
        pred_dice, risk, pred_hd = None, None, None
        entry = policy["models"].get(q)
        if usable and entry:
            pred_hd = _linear_predict(entry["hd95_auxiliary"], vector)
            if policy["family"] == "A_rule_tree" and leaf is not None:
                estimate = leaf["leaf_estimates"][q]
                pred_dice, risk = estimate["predicted_delta_dice"], estimate["unsafe_probability"]
            else:
                pred_dice = _linear_predict(entry["dice_auxiliary"], vector)
                if policy["family"] == "B_logistic":
                    model = entry["risk"]
                    if model["kind"] == "constant":
                        risk = float(model["value"])
                    elif model["kind"] == "logistic":
                        logit = float(np.dot(model["coefficients"], vector) + model["intercept"])
                        risk = 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, logit))))
        feasible = (risk is not None and risk <= policy["parameters"]["risk_threshold"]
                    if policy["family"] == "B_logistic"
                    else pred_dice is not None and pred_dice >= -0.02)
        size = _number(candidates.get(q, {}).get("output_bytes"))
        predictions[q] = {"predicted_delta_dice": pred_dice, "unsafe_probability": risk,
                          "p_risk": risk, "predicted_delta_hd95": pred_hd,
                          "estimated_feasible": bool(feasible), "HD95_status": "not_specified",
                          "file_bytes": size, "estimated_feasibility_basis": "Dice_only_exploratory"}
        if feasible and size is not None and size > 0:
            lossy_available.append((size, q))
    lossless = []
    base = candidates.get("baseline", {})
    base_size = _number(base.get("output_bytes", base.get("original_bytes")))
    if base_size is not None and base_size > 0:
        lossless.append((base_size, "baseline"))
    png = candidates.get("png_l6", {})
    png_size = _number(png.get("output_bytes"))
    if png.get("pixel_equal") is True and png_size is not None and png_size > 0:
        lossless.append((png_size, "png_l6"))
    for name in ("baseline", "png_l6"):
        item = candidates.get(name, {})
        equal = name == "baseline" or item.get("pixel_equal") is True
        predictions[name] = {"predicted_delta_dice": 0.0 if equal else None,
            "unsafe_probability": 0.0 if equal else None, "p_risk": 0.0 if equal else None,
            "predicted_delta_hd95": 0.0 if equal else None, "estimated_feasible": equal,
            "HD95_status": "not_specified", "file_bytes": _number(item.get("output_bytes")),
            "estimated_feasibility_basis": "unchanged_decoded_pixels" if equal else "not_verified"}
    choices = lossy_available + lossless
    selected = min(choices, key=lambda pair: (pair[0], pair[1]))[1] if choices else "baseline"
    fallback = not bool(lossy_available)
    reason = ("missing_features_fallback" if raw is None else
              "no_training_features_fallback" if not usable else
              "no_estimated_acceptable_lossy_candidate_fallback" if fallback else
              "minimum_encoded_bytes_among_estimated_Dice_acceptable_candidates")
    return {"selected_condition": selected, "selected": selected, "reason": reason,
            "fallback": fallback, "missing_features": missing, "predictions": predictions,
            "HD95_status": "not_specified", "clinical_acceptability": "not_established"}


def _evaluation(sample, selected):
    value = dict(sample.get("candidates", {}).get(selected, {}))
    value["delta_dice"] = _delta(sample, selected)
    value["delta_hd95"] = _delta(sample, selected, "hd95")
    original = _number(sample.get("candidates", {}).get("baseline", {}).get("output_bytes"))
    size = _number(value.get("output_bytes"))
    value["saving_pct"] = 100.0 * (1.0 - size / original) if size is not None and original and original > 0 else None
    return value


def _decision(sample, policy):
    decision = predict_policy(policy, sample.get("features"), sample.get("candidates", {}))
    decision.update(case_key=str(sample["case_key"]), image_key=str(sample["image_key"]),
                    evaluation_only=_evaluation(sample, decision["selected_condition"]))
    return decision


def _score(decisions):
    by_case = defaultdict(list)
    for decision in decisions:
        by_case[decision["case_key"]].append(decision["evaluation_only"])
    case_deltas, case_savings, observed, failed = [], [], 0, 0
    for evaluations in by_case.values():
        deltas = [e.get("delta_dice") for e in evaluations]
        valid = [v for v in deltas if v is not None]
        if len(valid) != len(evaluations):
            failed += 1
            case_deltas.append(None)
        else:
            observed += 1
            case_deltas.append(float(np.mean(valid)))
        savings = [_number(e.get("saving_pct")) for e in evaluations]
        case_savings.append(float(np.mean([s for s in savings if s is not None]))
                            if any(s is not None for s in savings) else 0.0)
    violations = sum(v is not None and v < -0.02 for v in case_deltas)
    n = len(by_case)
    return {"cases": n, "observed_cases": observed, "unobserved_or_failed_cases": failed,
            "dice_loss_violations": violations,
            "violation_or_unobserved_rate": (violations + failed) / n if n else 1.0,
            "observed_violation_rate": violations / observed if observed else None,
            "mean_saving_pct": float(np.mean(case_savings)) if case_savings else 0.0,
            "mean_delta_dice_observed": float(np.mean([v for v in case_deltas if v is not None])) if observed else None,
            "objective_missing_policy": "unobserved outcome penalized as violation; missing saving scored zero"}


def _fixed_score(samples, condition):
    return _score([dict(case_key=str(s["case_key"]), evaluation_only=_evaluation(s, condition)) for s in samples])


def _choose(scored, reference_rate):
    eligible = [row for row in scored if row["score"]["violation_or_unobserved_rate"] <= reference_rate + 1e-12]
    if eligible:
        selected = min(eligible, key=lambda r: (-r["score"]["mean_saving_pct"], r["complexity_order"]))
        reason = "within_inner_Q85_violation_reference_then_maximum_saving"
    else:
        selected = min(scored, key=lambda r: (r["score"]["violation_or_unobserved_rate"],
                                               -r["score"]["mean_saving_pct"], r["complexity_order"]))
        reason = "no_candidate_within_inner_Q85_reference_then_minimum_violation_then_saving"
    return selected, reason


def _grid(settings):
    grid = []
    for depth in settings["a_depths"]:
        grid.append(("A_rule_tree", {"depth": int(depth)}))
    for threshold in settings["b_risk_thresholds"]:
        grid.append(("B_logistic", {"risk_threshold": float(threshold)}))
    for alpha in settings["c_alphas"]:
        grid.append(("C_ridge", {"alpha": float(alpha)}))
    if not all(any(f == family for f, _ in grid) for family in FAMILIES):
        raise ValueError("Each of the three prespecified families needs at least one candidate.")
    return grid


def _nested_selection(samples, settings):
    groups = [str(s["case_key"]) for s in samples]
    if len(set(groups)) < 3:
        raise ValueError("Inner GroupKFold requires at least three training cases.")
    splits = list(GroupKFold(n_splits=3).split(np.zeros(len(samples)), groups=groups))
    assignment = []
    for fold, (train_ids, valid_ids) in enumerate(splits):
        train_cases = {groups[i] for i in train_ids}
        valid_cases = {groups[i] for i in valid_ids}
        if train_cases & valid_cases:
            raise RuntimeError("Inner case leakage.")
        assignment.append({"inner_fold": fold, "train_case_hash": digest(sorted(train_cases)),
                           "validation_case_hash": digest(sorted(valid_cases)),
                           "train_cases": len(train_cases), "validation_cases": len(valid_cases)})
    reference = _fixed_score(samples, "jpeg_q85")
    candidates = []
    for order, (family, parameters) in enumerate(_grid(settings)):
        decisions = []
        for train_ids, valid_ids in splits:
            policy = fit_policy([samples[i] for i in train_ids], family, parameters, settings)
            decisions.extend(_decision(samples[i], policy) for i in valid_ids)
        candidates.append({"family": family, "parameters": parameters, "complexity_order": order,
                           "score": _score(decisions)})
    best, reason = _choose(candidates, reference["violation_or_unobserved_rate"])
    families = {}
    for family in FAMILIES:
        choice, family_reason = _choose([c for c in candidates if c["family"] == family],
                                        reference["violation_or_unobserved_rate"])
        families[family] = dict(choice, selection_reason=family_reason)
    return {"inner_folds": assignment, "reference_Q85": reference, "candidates": candidates,
            "selected": dict(best, selection_reason=reason), "selected_by_family": families,
            "selection_scope": "training-side inner OOF only; no outer outcomes"}


def _best_fixed(samples):
    reference = _fixed_score(samples, "jpeg_q85")
    candidates = [{"condition": q, "score": _fixed_score(samples, q), "complexity_order": i}
                  for i, q in enumerate(QUALITIES)]
    selected, reason = _choose(candidates, reference["violation_or_unobserved_rate"])
    return dict(selected, selection_reason=reason, selection_scope="outer_training_cases_only",
                candidates=candidates, reference_Q85=reference)


def _subgroup_thresholds(samples):
    thresholds = {}
    for name in ("laplacian_variance", "noise_mad", "local_contrast", "edge_density"):
        values = [_number(s.get("features", {}).get(name)) for s in samples]
        values = [v for v in values if v is not None]
        thresholds[name] = float(np.median(values)) if values else None
    for name in ("lesion_area_fraction", "shape_irregularity"):
        values = [_number(s.get("evaluation", {}).get(name)) for s in samples]
        values = [v for v in values if v is not None]
        thresholds[name] = float(np.median(values)) if values else None
    return thresholds


def _subgroups(sample, thresholds):
    output = {}
    for name, threshold in thresholds.items():
        source = sample.get("evaluation", {}) if name in ("lesion_area_fraction", "shape_irregularity") else sample.get("features", {})
        value = _number(source.get(name))
        output[name] = ("missing" if value is None or threshold is None else
                        "below_or_equal_train_median" if value <= threshold else "above_train_median")
    return output


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False))


def _csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: canonical(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def run_validation(samples, output_dir, config=None):
    """Run fixed outer5/inner3 grouped CV; retain all three families and winner OOF.

    No family is selected using outer outcomes. Full-development fitting comes
    last and remains explicitly unvalidated outside development.
    """
    samples = sorted(list(samples), key=lambda s: (str(s.get("case_key")), str(s.get("image_key"))))
    _validate_samples(samples)
    settings = _settings(config)
    groups = [str(s["case_key"]) for s in samples]
    if len(set(groups)) < 5:
        raise ValueError("Outer GroupKFold requires at least five development cases.")
    output = Path(output_dir)
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError("CV output must be new or an existing empty directory.")
    else:
        output.mkdir(parents=True)
    folds, all_decisions, assignments = [], [], []
    for fold, (train_ids, valid_ids) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(samples)), groups=groups)):
        train, validation = [samples[i] for i in train_ids], [samples[i] for i in valid_ids]
        train_cases, valid_cases = {str(s["case_key"]) for s in train}, {str(s["case_key"]) for s in validation}
        if train_cases & valid_cases:
            raise RuntimeError("Outer case leakage.")
        selection = _nested_selection(train, settings)
        best_fixed = _best_fixed(train)
        thresholds = _subgroup_thresholds(train)
        policies = {}
        train_descriptive = {}
        for family in FAMILIES:
            parameters = selection["selected_by_family"][family]["parameters"]
            policy = fit_policy(train, family, parameters, settings)
            policies[family] = policy
            _write_json(output / ("fold_%d_%s_policy.json" % (fold, family)), policy)
            train_descriptive[family] = dict(_score([_decision(s, policy) for s in train]),
                                             evidence_type="in_sample_training_description_not_OOF")
        winner = selection["selected"]["family"]
        for sample in validation:
            assignments.append({"fold": fold, "case_key": sample["case_key"], "image_key": sample["image_key"]})
            for algorithm in FAMILIES + ("IACS_v2",):
                family = winner if algorithm == "IACS_v2" else algorithm
                decision = _decision(sample, policies[family])
                decision.update(fold=fold, algorithm=algorithm, underlying_family=family,
                    train_best_fixed=best_fixed["condition"], subgroups=_subgroups(sample, thresholds),
                    subgroup_thresholds=thresholds,
                    candidate_evaluation={q: _evaluation(sample, q) for q in ("baseline",) + QUALITIES + ("png_l6",)},
                    evidence_type="outer_case_OOF", policy_sha256=policies[family]["policy_sha256"])
                all_decisions.append(decision)
        fold_decisions = [d for d in all_decisions if d["fold"] == fold]
        fold_meta = {"fold": fold, "train_cases": len(train_cases), "validation_cases": len(valid_cases),
            "train_case_keys": sorted(train_cases), "validation_case_keys": sorted(valid_cases),
            "train_case_hash": digest(sorted(train_cases)), "validation_case_hash": digest(sorted(valid_cases)),
            "inner_selection": selection, "selected_family": winner,
            "selected_config": selection["selected"], "train_best_fixed": best_fixed,
            "subgroup_thresholds": thresholds, "subgroup_scope": "outer-training medians; GT area/shape evaluation only",
            "train_fit_descriptive": train_descriptive,
            "outer_scores": {a: _score([d for d in fold_decisions if d["algorithm"] == a])
                             for a in FAMILIES + ("IACS_v2",)}}
        folds.append(fold_meta)
        _write_json(output / ("fold_%d_metadata.json" % fold), fold_meta)
    full_selection = _nested_selection(samples, settings)
    full_config = full_selection["selected"]
    full_policy = fit_policy(samples, full_config["family"], full_config["parameters"], settings)
    _write_json(output / "full_development_policy.json", full_policy)
    _write_json(output / "full_development_inner_selection.json", full_selection)
    payload = {"schema_version": VERSION, "scope": "BUS-BRA development mechanism only",
        "status": "nested_development_validation_complete_external_unvalidated", "config": settings,
        "sample_images": len(samples), "sample_cases": len(set(groups)), "folds": folds,
        "outer_decisions": all_decisions, "full_policy": full_policy,
        "full_selection": full_selection, "fold_assignments": assignments,
        "outer_scores": {a: _score([d for d in all_decisions if d["algorithm"] == a])
                         for a in FAMILIES + ("IACS_v2",)},
        "HD95_status": "not_specified", "clinical_noninferiority_conclusion": None,
        "limitations": ["Dice loss 0.02 is an exploratory candidate, not a clinical margin",
                        "No HD95 gate or joint acceptability conclusion",
                        "Outer outcomes do not choose a winning family",
                        "Full-development policy has no independent external validation",
                        "GT area/shape are evaluation-only subgroup variables"]}
    _write_json(output / "validation.json", payload)
    _csv(output / "fold_assignments.csv", assignments)
    flat, predictions = [], []
    for decision in all_decisions:
        identity = {k: decision[k] for k in ("case_key", "image_key", "fold", "algorithm", "selected_condition")}
        flat.append(dict(identity, underlying_family=decision["underlying_family"],
                         reason=decision["reason"], fallback=decision["fallback"],
                         train_best_fixed=decision["train_best_fixed"],
                         **{"actual_" + k: v for k, v in decision["evaluation_only"].items()
                            if isinstance(v, (str, int, float, bool)) or v is None}))
        for condition, prediction in decision["predictions"].items():
            predictions.append(dict(identity, candidate=condition, **prediction))
    _csv(output / "oof_decisions.csv", flat)
    _csv(output / "candidate_predictions.csv", predictions)
    return payload
