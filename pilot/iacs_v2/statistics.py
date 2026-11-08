"""Exploratory Case-clustered estimates from fixed outer OOF predictions.

Bootstrap intervals condition on the fitted cross-validation records. They do
not include re-fitting/model-selection uncertainty, and are not clinical NI CIs.
"""
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon, spearmanr

from .common import write_json, write_csv
from .features import FEATURE_NAMES

METRICS = ("dice", "hd95", "assd", "saving_pct", "iou", "boundary_f1", "prediction_consistency_dice", "delta_area")
QUALITIES = ("jpeg_q95", "jpeg_q85", "jpeg_q75", "jpeg_q60", "jpeg_q40")


def number(value):
    if value is None:
        return None
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def _valid(row, metric):
    if row.get("status") not in ("ok", "completed"):
        return None
    return number(row.get(metric))


def bootstrap_ci(values, repetitions=2000, seed=20261108):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, repetitions, 100):
        indices = rng.integers(0, len(values), (min(100, repetitions - start), len(values)))
        means.extend(values[indices].mean(axis=1).tolist())
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def holm(rows, raw="p_value", corrected="holm_p"):
    available = sorted((r for r in rows if number(r.get(raw)) is not None), key=lambda r: r[raw])
    previous = 0.0
    for i, row in enumerate(available):
        previous = max(previous, min(1.0, (len(available) - i) * row[raw]))
        row[corrected] = float(previous)
        row["holm_family_size"] = len(available)
    for row in rows:
        row.setdefault(corrected, None)
    return rows


def _case_means(rows, metric):
    grouped = defaultdict(list)
    for row in rows:
        if not row.get("case_key"):
            raise ValueError("Known Case identity required")
        value = _valid(row, metric)
        if value is not None:
            grouped[row["case_key"]].append(value)
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def paired_difference(first, second, metric, repetitions=2000, seed=20261108):
    """Match identical images first, then average image differences per Case."""
    def index(rows):
        result = {}
        for row in rows:
            identity = row.get("case_key"), row.get("image_key")
            if not all(identity) or identity in result:
                raise ValueError("Unique Case/image records required for pairing")
            result[identity] = row
        return result
    a, b = index(first), index(second)
    grouped = defaultdict(list)
    for identity in a.keys() & b.keys():
        left, right = _valid(a[identity], metric), _valid(b[identity], metric)
        if left is not None and right is not None:
            grouped[identity[0]].append(left - right)
    values = [float(np.mean(grouped[key])) for key in sorted(grouped)]
    scheduled = {key[0] for key in a} | {key[0] for key in b}
    p_value, emitted = None, []
    if len(values) >= 2:
        if np.all(np.array(values) == 0):
            p_value = 1.0
        else:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                p_value = float(wilcoxon(values, method="auto", alternative="two-sided").pvalue)
            emitted = [str(w.message) for w in caught]
    return dict(metric=metric, direction="first_minus_reference", scheduled_cases=len(scheduled),
                valid_paired_cases=len(values), paired_images=sum(len(v) for v in grouped.values()),
                missing_or_failed_cases=len(scheduled) - len(values),
                mean_difference=float(np.mean(values)) if values else None,
                median_difference=float(np.median(values)) if values else None,
                ci95=bootstrap_ci(values, repetitions, seed), p_value=p_value, warnings=emitted,
                ci_method="Case percentile bootstrap of fixed outer-OOF pairs; no refitting",
                clinical_noninferiority=None)


def _records(samples, validation):
    groups = defaultdict(list)
    for sample in samples:
        identity = dict(case_key=sample["case_key"], image_key=sample["image_key"])
        for condition, candidate in sample["candidates"].items():
            groups[condition].append(dict(candidate, **identity))
    for decision in validation["outer_decisions"]:
        identity = dict(case_key=decision["case_key"], image_key=decision["image_key"], fold=decision["fold"])
        groups[decision["algorithm"]].append(dict(decision["evaluation_only"], **identity))
        if decision["algorithm"] == "IACS_v2":
            groups["train_best_fixed"].append(dict(decision["candidate_evaluation"][decision["train_best_fixed"]], **identity))
    return groups


def feature_associations(samples):
    rows = []
    for condition in QUALITIES:
        for outcome in ("delta_dice", "delta_hd95"):
            for feature in FEATURE_NAMES:
                pairs = defaultdict(list)
                for sample in samples:
                    x = number(sample["features"].get(feature))
                    y = _valid(sample["candidates"].get(condition, {}), outcome)
                    if x is not None and y is not None:
                        pairs[sample["case_key"]].append((x, y))
                values = [np.mean(value, axis=0) for value in pairs.values()]
                rho, p = None, None
                if len(values) >= 3:
                    array = np.asarray(values)
                    if np.ptp(array[:, 0]) > 0 and np.ptp(array[:, 1]) > 0:
                        result = spearmanr(array[:, 0], array[:, 1])
                        rho, p = number(result.statistic), number(result.pvalue)
                rows.append(dict(condition=condition, feature=feature, outcome=outcome, n_cases=len(values),
                                 spearman_rho=rho, p_value=p, interpretation="full_development_exploratory; never model selection"))
    return holm(rows)


def prediction_performance(samples, validation):
    """Evaluate all quality predictions OOF, including an outer-training mean comparator."""
    index = {s["image_key"]: s for s in samples}
    fold_meta = {f["fold"]: f for f in validation["folds"]}
    rows = []
    for algorithm in ("A_rule_tree", "B_logistic", "C_ridge", "IACS_v2"):
        decisions = [d for d in validation["outer_decisions"] if d["algorithm"] == algorithm]
        for condition in QUALITIES:
            for metric, predicted_key in (("delta_dice", "predicted_delta_dice"), ("delta_hd95", "predicted_delta_hd95")):
                pairs = defaultdict(list)
                for decision in decisions:
                    sample = index[decision["image_key"]]
                    actual = _valid(sample["candidates"][condition], metric)
                    predicted = number(decision["predictions"][condition].get(predicted_key))
                    train_keys = set(fold_meta[decision["fold"]]["train_case_keys"])
                    train_values = [_valid(s["candidates"][condition], metric) for s in samples if s["case_key"] in train_keys]
                    train_values = [v for v in train_values if v is not None]
                    if actual is not None and predicted is not None and train_values:
                        pairs[decision["case_key"]].append((actual, predicted, float(np.mean(train_values))))
                values = np.array([np.mean(v, axis=0) for v in pairs.values()])
                item = dict(algorithm=algorithm, condition=condition, endpoint=metric, valid_cases=len(values),
                            scheduled_cases=len({d["case_key"] for d in decisions}),
                            missing_or_failed_cases=len({d["case_key"] for d in decisions}) - len(values),
                            mae=None, rmse=None, training_mean_rmse=None, r2=None, spearman_rho=None)
                if len(values):
                    actual, predicted, comparator = values.T
                    item.update(mae=float(np.mean(np.abs(predicted - actual))), rmse=float(np.sqrt(np.mean((predicted - actual) ** 2))),
                                training_mean_rmse=float(np.sqrt(np.mean((comparator - actual) ** 2))))
                    if np.var(actual) > 0:
                        item["r2"] = float(1 - np.sum((actual - predicted) ** 2) / np.sum((actual - actual.mean()) ** 2))
                    if len(values) >= 3 and np.ptp(actual) > 0 and np.ptp(predicted) > 0:
                        item["spearman_rho"] = number(spearmanr(actual, predicted).statistic)
                rows.append(item)
    return rows


def mixed_effects(groups):
    import pandas as pd
    import statsmodels.formula.api as smf
    names = ["baseline", *QUALITIES, "IACS_v2"]
    records = [dict(case=case, condition=name, dice=value) for name in names
               for case, value in _case_means(groups[name], "dice").items()]
    result = dict(status="insufficient_data", formula="dice ~ C(condition); Case random intercept; REML",
                  records=len(records), cases=len({r["case"] for r in records}), converged=False, warnings=[])
    if result["cases"] < 3:
        return result
    caught = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = smf.mixedlm("dice ~ C(condition)", pd.DataFrame(records), groups=pd.DataFrame(records)["case"])
            fit = model.fit(reml=True, method="lbfgs", maxiter=200, disp=False)
        result.update(status="fitted" if fit.converged else "not_converged", converged=bool(fit.converged),
                      warnings=[str(w.message) for w in caught], coefficients={k: number(v) for k, v in fit.fe_params.items()},
                      p_values={k: number(fit.pvalues[k]) for k in fit.fe_params.index},
                      interpretation="exploratory condition effect; cross-fitted policy training dependence not modeled")
    except Exception as exc:
        result.update(status="fit_failed", error_type=type(exc).__name__,
                      warnings=[str(w.message) for w in caught] + [str(exc)])
    return result


def analyze(samples, validation, output_dir, config=None):
    config = config or {}
    repetitions, seed = int(config.get("bootstrap", 2000)), int(config.get("seed", 20261108))
    if repetitions < 100:
        raise ValueError("At least 100 Case-bootstrap repetitions required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Statistical output directory is not empty")
    groups = _records(samples, validation)
    summaries = []
    for name, rows in groups.items():
        scheduled = {r["case_key"] for r in rows}
        for metric in METRICS:
            values = list(_case_means(rows, metric).values())
            summaries.append(dict(algorithm=name, metric=metric, scheduled_cases=len(scheduled),
                valid_cases=len(values), missing_or_failed_cases=len(scheduled) - len(values),
                mean=float(np.mean(values)) if values else None, median=float(np.median(values)) if values else None,
                ci95=bootstrap_ci(values, repetitions, seed), analysis_unit="Case"))
    comparisons = []
    for reference in ("baseline", "jpeg_q85", "jpeg_q75", "jpeg_q60", "train_best_fixed"):
        for metric in METRICS:
            comparisons.append(dict(algorithm="IACS_v2", reference=reference,
                                    **paired_difference(groups["IACS_v2"], groups[reference], metric, repetitions, seed)))
    # One declared exploratory family: all 5 references x 8 endpoints.
    holm(comparisons)
    subgroup_rows = []
    primary = [d for d in validation["outer_decisions"] if d["algorithm"] == "IACS_v2"]
    subgroup_keys = sorted({k for d in primary for k in d["subgroups"]})
    for variable in subgroup_keys:
        for level in sorted({d["subgroups"][variable] for d in primary}):
            members = [d for d in primary if d["subgroups"][variable] == level]
            chosen = [dict(d["evaluation_only"], case_key=d["case_key"], image_key=d["image_key"]) for d in members]
            comparator = [dict(d["candidate_evaluation"]["jpeg_q85"], case_key=d["case_key"], image_key=d["image_key"]) for d in members]
            for metric in ("dice", "hd95", "assd", "saving_pct"):
                subgroup_rows.append(dict(variable=variable, level=level, reference="jpeg_q85", threshold="outer-training median, fold-specific",
                                          **paired_difference(chosen, comparator, metric, repetitions, seed)))
    holm(subgroup_rows)
    result = dict(analysis_unit="Case", patient_semantics_verified=False, exploration_only=True,
        hd95_margin=None, dice_loss_candidate=0.02, clinical_noninferiority_conclusion=None,
        bootstrap_repetitions=repetitions, seed=seed,
        ci_limitations="Case bootstrap of fixed outer OOF records; no model refitting, selection or fold-training dependence uncertainty",
        holm_family="all five IACS references x eight endpoints; separate family for all subgroup tests and all feature associations",
        group_summaries=summaries, paired_comparisons=comparisons, subgroups=subgroup_rows,
        feature_associations=feature_associations(samples), prediction_performance=prediction_performance(samples, validation),
        mixed_effects=mixed_effects(groups),
        missing_policy="All Cases retained; per-metric finite paired image intersection averaged within Case; missing counts explicit")
    write_json(output / "statistics.json", result)
    for key in ("group_summaries", "paired_comparisons", "subgroups", "feature_associations", "prediction_performance"):
        write_csv(output / (key + ".csv"), result[key])
    return result
