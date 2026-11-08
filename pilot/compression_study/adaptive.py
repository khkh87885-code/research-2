"""Unvalidated image-only rules and development-only portable policy fitting."""

import hashlib
import json
import math
from collections import defaultdict


VERSION = "1.0"
DEFAULT_CONDITIONS = ("baseline", "jpeg_q95", "jpeg_q85", "jpeg_q75", "jpeg_q60", "jpeg_q40")
IMAGE_FEATURES = {"width", "height", "clarity", "noise", "contrast", "global_contrast", "edge_density",
                  "edge_strength", "entropy", "gray_mean", "dynamic_range"}
FORBIDDEN_FEATURES = {"dice", "iou", "hd95", "assd", "mask", "reference", "pathology", "label",
                      "patient_key", "image_key", "patient_id", "image_id", "split", "condition"}


def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def _condition_ids(conditions):
    result = []
    for condition in conditions:
        if isinstance(condition, str):
            result.append(condition)
        else:
            value = condition.get("id", condition.get("condition", condition.get("name")))
            if not value:
                raise ValueError("Each condition needs an id.")
            result.append(str(value))
    if len(result) != len(set(result)) or "baseline" not in result:
        raise ValueError("Conditions must be unique and include baseline.")
    return result


def initial_policy():
    return rule_policy(DEFAULT_CONDITIONS)


def rule_policy(conditions):
    """Conservative fixed rule, deliberately labelled unvalidated.

    No reference mask or segmentation quality is read at application time.
    Thresholds are engineering defaults and are not claimed to retain Dice.
    """
    ids = _condition_ids(conditions)
    conservative = "jpeg_q95" if "jpeg_q95" in ids else "baseline"
    moderate = "jpeg_q85" if "jpeg_q85" in ids else conservative
    efficient = "jpeg_q75" if "jpeg_q75" in ids else moderate
    return {"schema_version": VERSION, "kind": "image_only_rule", "status": "unvalidated",
            "conditions": ids, "default_condition": conservative,
            "feature_schema": ["clarity", "noise", "contrast"],
            "feature_version": "full_image_v1", "uses_reference_at_inference": False,
            "description": "Prospective engineering pilot thresholds on normalized [0,1] intensities; unvalidated, not safety standards.",
            "rules": [
                {"feature": "noise", "min": 0.025, "condition": conservative},
                {"feature": "contrast", "max": 0.04, "condition": conservative},
                {"feature": "clarity", "max": 0.005, "condition": conservative},
                {"all": [{"feature": "noise", "max": 0.01},
                         {"feature": "contrast", "min": 0.08},
                         {"feature": "clarity", "min": 0.02}], "condition": efficient},
                {"all": [{"feature": "noise", "min": 0},
                         {"feature": "contrast", "min": 0},
                         {"feature": "clarity", "min": 0}], "condition": moderate},
            ]}


def _matches(features, predicate):
    if "all" in predicate:
        return all(_matches(features, p) for p in predicate["all"])
    feature = predicate["feature"]
    if feature.lower() not in IMAGE_FEATURES:
        raise ValueError("Reference/outcome features are forbidden at inference.")
    value = _finite(features.get(feature))
    return (value is not None and predicate.get("min", -math.inf) <= value
            <= predicate.get("max", math.inf))


def choose_condition(features, policy):
    """Choose an available condition from image features, never from a mask."""
    if not isinstance(features, dict):
        raise ValueError("Image features must be a dictionary.")
    ids = _condition_ids(policy["conditions"])
    fallback = policy.get("default_condition", "baseline")
    if fallback not in ids:
        raise ValueError("Policy default condition is unavailable.")
    if policy.get("kind") == "image_only_rule":
        # Rules may be supplied prospectively, but defaults contain no tuned thresholds.
        for rule in policy.get("rules", []):
            if _matches(features, rule):
                selected = rule["condition"]
                if selected not in ids:
                    raise ValueError("Rule selected an unavailable condition.")
                return selected
        return fallback
    if policy.get("kind") != "development_tree":
        raise ValueError("Unsupported policy kind.")
    names = policy["feature_schema"]
    values = [_finite(features.get(name)) for name in names]
    if any(v is None for v in values):
        return fallback
    node = policy["tree"]
    while "condition" not in node:
        index = int(node["feature_index"])
        node = node["left"] if values[index] <= node["threshold"] else node["right"]
    if node["condition"] not in ids:
        raise ValueError("Tree selected an unavailable condition.")
    return node["condition"]


def _feature_lookup(feature_records, rows):
    if feature_records is None:
        return {str(row["image_key"]): row.get("features", {}) for row in rows}
    if isinstance(feature_records, dict):
        return {str(k): v.get("features", v) for k, v in feature_records.items()}
    if any(("split" in r and str(r["split"]).lower() != "development")
           or ("dataset" in r and str(r["dataset"]).upper() not in ("BUS-BRA", "BUSBRA"))
           for r in feature_records):
        raise ValueError("Feature records must also belong to BUS-BRA development only.")
    return {str(r["image_key"]): r.get("features", {k: v for k, v in r.items()
                                                   if k not in FORBIDDEN_FEATURES})
            for r in feature_records}


def fit_policy(training_rows, feature_records, config):
    """Fit a shallow tree on BUS-BRA development patients only.

    Label each image with the smallest observed file satisfying BOTH prespecified
    Dice loss and HD95 increase bounds versus its baseline. Training and internal
    validation patients are disjoint; external/holdout rows cause a hard error.
    The resulting JSON is an unvalidated research artifact, never a clinical claim.
    """
    rows = list(training_rows)
    if not rows:
        raise ValueError("No development rows supplied.")
    if any(str(r.get("dataset", "")).upper() not in ("BUS-BRA", "BUSBRA")
           or str(r.get("split", "")).lower() != "development" for r in rows):
        raise ValueError("Policy fitting accepts only BUS-BRA development rows; external/holdout input rejected.")
    if any(r.get("patient_key") is None or r.get("image_key") is None
           or not str(r.get("patient_key", "")).strip() or not str(r.get("image_key", "")).strip() for r in rows):
        raise ValueError("Known patient and image keys are required for patient-disjoint fitting.")
    strategies = {str(r.get("prompt_strategy", "")).strip() for r in rows}
    if len(strategies) != 1 or not next(iter(strategies)):
        raise ValueError("Fit one explicit prompt strategy at a time.")
    margins = {}
    for name in ("delta", "hd95_delta"):
        value = _finite(config.get(name))
        if value is None or value < 0:
            raise ValueError("Prespecify nonnegative delta and hd95_delta before policy fitting.")
        margins[name] = value
    conditions = _condition_ids(config.get("conditions", sorted({r["condition"] for r in rows})))
    image_owners = defaultdict(set)
    for row in rows:
        image_owners[str(row["image_key"])].add(str(row["patient_key"]))
    if any(len(owners) > 1 for owners in image_owners.values()):
        raise ValueError("An image key cannot belong to multiple patients.")
    feature_map = {key: value for key, value in _feature_lookup(feature_records, rows).items()
                   if key in image_owners}
    names = list(config.get("feature_names", []))
    if not names:
        candidates = set.intersection(*(set(f) for f in feature_map.values())) if feature_map else set()
        names = sorted(n for n in candidates if n.lower() in IMAGE_FEATURES
                       and all(_finite(f[n]) is not None for f in feature_map.values()))
    if not names or any(n.lower() not in IMAGE_FEATURES for n in names):
        raise ValueError("A nonempty schema of image-only numeric features is required.")
    by_image = defaultdict(dict)
    for row in rows:
        identity = (str(row["patient_key"]), str(row["image_key"]))
        condition = row["condition"]
        if condition in by_image[identity]:
            raise ValueError("Duplicate image-condition rows; resolve retries before fitting.")
        by_image[identity][condition] = row
    X, y, groups = [], [], []
    excluded = defaultdict(int)
    for (patient, image), candidates in sorted(by_image.items()):
        baseline = candidates.get("baseline", {})
        baseline_dice, baseline_hd = _finite(baseline.get("dice")), _finite(baseline.get("hd95"))
        if (baseline.get("status") != "ok" or baseline_dice is None or baseline_hd is None
                or not 0 <= baseline_dice <= 1 or baseline_hd < 0):
            excluded["invalid_baseline"] += 1
            continue
        feature = feature_map.get(image, {})
        vector = [_finite(feature.get(n)) for n in names]
        if any(value is None for value in vector):
            excluded["missing_features"] += 1
            continue
        eligible = []
        for name, row in candidates.items():
            dice_value, hd_value = _finite(row.get("dice")), _finite(row.get("hd95"))
            size = _finite(row.get("output_bytes"))
            if (name in conditions and row.get("status") == "ok" and dice_value is not None
                    and hd_value is not None and size is not None and size > 0
                    and 0 <= dice_value <= 1 and hd_value >= 0
                    and dice_value - baseline_dice >= -margins["delta"]
                    and hd_value - baseline_hd <= margins["hd95_delta"]):
                eligible.append((size, name))
        if not eligible:
            excluded["no_feasible_condition"] += 1
            continue
        X.append(vector)
        y.append(min(eligible)[1])
        groups.append(patient)
    if len(set(groups)) < 2:
        raise ValueError("At least two eligible development patients are required for disjoint fitting.")
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.tree import DecisionTreeClassifier
    import numpy as np

    seed = int(config.get("seed", 20261108))
    fraction = float(config.get("validation_fraction", 0.25))
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between zero and one.")
    depth = int(config.get("max_depth", 3))
    if depth < 1 or depth > 5:
        raise ValueError("Use a prespecified shallow tree max_depth between 1 and 5.")
    train, validation = next(GroupShuffleSplit(n_splits=1, test_size=fraction,
                                              random_state=seed).split(X, y, groups=groups))
    train_patients = {groups[i] for i in train}
    validation_patients = {groups[i] for i in validation}
    if train_patients & validation_patients:
        raise RuntimeError("Patient leakage detected.")
    estimator = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=int(config.get("min_samples_leaf", 2)),
                                       random_state=seed)
    features = np.asarray(X, dtype=float)
    labels = np.asarray(y)
    estimator.fit(features[train], labels[train])
    def export_node(index):
        tree = estimator.tree_
        if tree.children_left[index] == tree.children_right[index]:
            return {"condition": str(estimator.classes_[int(np.argmax(tree.value[index]))])}
        return {"feature_index": int(tree.feature[index]), "threshold": float(tree.threshold[index]),
                "left": export_node(int(tree.children_left[index])),
                "right": export_node(int(tree.children_right[index]))}
    def patient_hash(patients):
        return hashlib.sha256(json.dumps(sorted(patients), separators=(",", ":")).encode()).hexdigest()
    locked = {"schema_version": VERSION, "feature_version": str(config.get("feature_version", "full_image_v1")),
              "kind": "development_tree", "status": "unvalidated_research_policy",
              "prompt_strategy": next(iter(strategies)),
              "conditions": conditions, "default_condition": "baseline", "feature_schema": names,
              "uses_reference_at_inference": False, "training_dataset": "BUS-BRA",
              "training_split": "development", "training_patient_hash": patient_hash(train_patients),
              "internal_validation_patient_hash": patient_hash(validation_patients),
              "training_patients": len(train_patients), "internal_validation_patients": len(validation_patients),
              "training_images": len(train), "internal_validation_images": len(validation),
              "excluded_images": dict(excluded), "prespecified_margins": margins,
              "parameters": {"seed": seed, "max_depth": depth, "validation_fraction": fraction,
                             "min_samples_leaf": int(config.get("min_samples_leaf", 2))},
              "tree": export_node(0),
              "internal_validation_label_accuracy": float(estimator.score(features[validation], labels[validation])),
              "validation_interpretation": "Development label accuracy only; segmentation reliability and external validation pending."}
    locked["policy_sha256"] = hashlib.sha256(json.dumps(locked, sort_keys=True, separators=(",", ":"),
                                                        allow_nan=False).encode()).hexdigest()
    return locked
