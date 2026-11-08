"""Patient-level paired analyses; failures remain in the reported denominators."""

import math
import warnings
from collections import defaultdict

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon


METRICS = ("dice", "iou", "hd95", "assd", "output_bytes")
COHORT = ("dataset", "prompt_strategy", "split")


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def _cohort(row):
    return tuple(str(row.get(k, "")) for k in COHORT)


def _index(rows):
    result = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if any(row.get(k) is None or not str(row[k]).strip() for k in ("patient_key", "image_key", "condition")):
            raise ValueError("Statistics requires known patient_key, image_key and condition.")
        patient = str(row.get("patient_key", ""))
        image = str(row.get("image_key", ""))
        condition = str(row.get("condition", ""))
        if not patient or not image or not condition:
            raise ValueError("Statistics requires known patient_key, image_key and condition.")
        target = result[_cohort(row)][condition]
        identity = (patient, image)
        if identity in target:
            raise ValueError("Duplicate patient/image/condition records; resolve retries before analysis.")
        target[identity] = row
    return result


def _metric(row, metric):
    if row.get("status") != "ok":
        return None
    return _number(row.get(metric))


def _mean(values):
    return float(np.mean(values)) if values else None


def _bootstrap(values, repetitions, rng):
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return None
    samples = []
    for start in range(0, repetitions, 128):
        count = min(128, repetitions - start)
        draws = rng.integers(0, len(array), size=(count, len(array)))
        samples.extend(array[draws].mean(axis=1).tolist())
    return [float(v) for v in np.percentile(samples, [2.5, 97.5])]


def _patient_means(records, metric):
    values = defaultdict(list)
    for (patient, _), row in records.items():
        value = _metric(row, metric)
        if value is not None:
            values[patient].append(value)
    return {patient: _mean(items) for patient, items in values.items()}


def _paired(baseline, changed, metric):
    values = defaultdict(list)
    for identity in baseline.keys() & changed.keys():
        first = _metric(baseline[identity], metric)
        second = _metric(changed[identity], metric)
        if first is not None and second is not None:
            values[identity[0]].append(second - first)
    # Average within patient before treating patients as independent units.
    return [float(np.mean(values[p])) for p in sorted(values)], sum(map(len, values.values()))


def summarize(rows, bootstrap=1000, seed=20261108, delta=None):
    """Return descriptive, paired-bootstrap, Wilcoxon/Holm and Friedman results.

    The paired effect is condition minus baseline. Complete image pairs are first
    averaged within each patient. A supplied delta applies only to Dice and must
    be prespecified; passing it does not establish a clinical noninferiority claim.
    """
    if not isinstance(bootstrap, int) or bootstrap < 100:
        raise ValueError("At least 100 bootstrap repetitions are required.")
    if delta is not None and (_number(delta) is None or float(delta) < 0):
        raise ValueError("delta must be a finite nonnegative prespecified Dice margin.")
    rows = list(rows)
    indexed = _index(rows)
    rng = np.random.default_rng(seed)
    groups, comparisons, friedman = [], [], []
    for cohort, conditions in sorted(indexed.items()):
        cohort_info = dict(zip(COHORT, cohort))
        baseline = conditions.get("baseline", {})
        for condition, records in sorted(conditions.items()):
            patients = {identity[0] for identity in records}
            metrics = {}
            for metric in METRICS:
                patient_values = _patient_means(records, metric)
                metrics[metric] = {
                    "mean_of_patient_means": _mean(list(patient_values.values())),
                    "valid_patients": len(patient_values),
                    "missing_or_failed_patients": len(patients - patient_values.keys()),
                    "valid_images": sum(_metric(r, metric) is not None for r in records.values()),
                    "missing_or_failed_images": sum(_metric(r, metric) is None for r in records.values()),
                }
            groups.append(dict(cohort_info, condition=condition, scheduled_images=len(records),
                               scheduled_patients=len(patients),
                               successful_images=sum(r.get("status") == "ok" for r in records.values()),
                               failed_images=sum(r.get("status") != "ok" for r in records.values()),
                               patients_with_any_failed_image=len({p for (p, _), r in records.items()
                                                                  if r.get("status") != "ok"}),
                               metrics=metrics))
            if condition == "baseline":
                continue
            for metric in METRICS:
                effects, paired_images = _paired(baseline, records, metric)
                all_patients = {p for p, _ in baseline} | patients
                interval = _bootstrap(effects, bootstrap, rng) if len(effects) >= 2 else None
                p_value = None
                captured = []
                if len(effects) >= 2:
                    if np.all(np.asarray(effects) == 0):
                        p_value = 1.0
                    else:
                        with warnings.catch_warnings(record=True) as emitted:
                            warnings.simplefilter("always")
                            p_value = float(wilcoxon(effects, alternative="two-sided",
                                                     zero_method="wilcox", method="auto").pvalue)
                        captured = [str(w.message) for w in emitted]
                item = dict(cohort_info, condition=condition, reference="baseline", metric=metric,
                            effect_direction="condition_minus_baseline", paired_patients=len(effects),
                            paired_images=paired_images, scheduled_patients=len(all_patients),
                            patients_without_valid_pair=len(all_patients) - len(effects),
                            patient_mean_difference=_mean(effects), ci95=interval,
                            wilcoxon_p=p_value, holm_p=None,
                            inference_status="estimated" if len(effects) >= 2 else "insufficient_patients",
                            warnings=captured)
                if metric == "dice":
                    item["prespecified_margin"] = float(delta) if delta is not None else None
                    item["ci_exceeds_negative_margin"] = (bool(interval[0] > -float(delta))
                                                          if interval is not None and delta is not None
                                                          else None)
                comparisons.append(item)
        # Holm families contain all non-baseline conditions for one metric/cohort.
        for metric in METRICS:
            family = [r for r in comparisons if _cohort(r) == cohort and r["metric"] == metric
                      and r["wilcoxon_p"] is not None]
            ordered = sorted(family, key=lambda r: r["wilcoxon_p"])
            previous = 0.0
            for rank, item in enumerate(ordered):
                previous = max(previous, min(1.0, (len(ordered) - rank) * item["wilcoxon_p"]))
                item["holm_p"] = float(previous)
                item["holm_family_size"] = len(ordered)
        names = sorted(conditions)
        for metric in METRICS:
            result = dict(cohort_info, metric=metric, conditions=names,
                          complete_patients=0, statistic=None, p_value=None,
                          scheduled_patients=len({p for records in conditions.values() for p, _ in records}),
                          complete_images=0, status="insufficient_conditions", warnings=[])
            if len(names) >= 3:
                identities = set.intersection(*(set(conditions[n]) for n in names))
                complete = defaultdict(lambda: defaultdict(list))
                for identity in identities:
                    values = [_metric(conditions[n][identity], metric) for n in names]
                    if all(v is not None for v in values):
                        for name, value in zip(names, values):
                            complete[identity[0]][name].append(value)
                patient_ids = sorted(complete)
                result["complete_patients"] = len(patient_ids)
                result["complete_images"] = sum(len(complete[p][names[0]]) for p in patient_ids)
                if len(patient_ids) >= 2:
                    matrix = np.array([[np.mean(complete[p][n]) for n in names] for p in patient_ids])
                    if np.all(matrix == matrix[:, :1]):
                        result.update(status="all_conditions_tied", statistic=0.0, p_value=1.0)
                    else:
                        with warnings.catch_warnings(record=True) as emitted:
                            warnings.simplefilter("always")
                            test = friedmanchisquare(*(matrix[:, i] for i in range(len(names))))
                        result.update(status="estimated", statistic=_number(test.statistic),
                                      p_value=_number(test.pvalue), warnings=[str(w.message) for w in emitted])
                        if len(patient_ids) <= 10 or len(names) <= 6:
                            result["warnings"].append("Chi-square approximation may be inaccurate with few patients/conditions.")
                else:
                    result["status"] = "insufficient_patients"
            result["patients_without_complete_data"] = result["scheduled_patients"] - result["complete_patients"]
            friedman.append(result)
    return {"analysis_unit": "patient", "paired_image_policy": "intersect images then average within patient",
            "bootstrap_repetitions": bootstrap, "seed": seed, "ci_method": "paired patient percentile bootstrap",
            "holm_family": "dataset/prompt_strategy/split/metric across non-baseline conditions",
            "margin_interpretation": "Prespecified threshold check only; no automatic clinical noninferiority conclusion.",
            "total_scheduled_records": len(rows), "groups": groups,
            "paired_comparisons": comparisons, "friedman": friedman}


def mixed_effects(rows):
    """Optional patient random-intercept analysis of patient-condition Dice means."""
    indexed = _index(list(rows))
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError as exc:
        return {"status": "unavailable", "reason": str(exc), "cohorts": []}
    results = []
    for cohort, conditions in sorted(indexed.items()):
        records = []
        for name, entries in conditions.items():
            for patient, value in _patient_means(entries, "dice").items():
                records.append({"patient": patient, "condition": name, "dice": value})
        output = dict(zip(COHORT, cohort))
        output.update(status="insufficient_data", converged=False, warnings=[],
                      valid_patient_condition_records=len(records),
                      patients=len({r["patient"] for r in records}))
        if output["patients"] < 3 or len({r["condition"] for r in records}) < 2:
            results.append(output)
            continue
        emitted = []
        try:
            with warnings.catch_warnings(record=True) as emitted:
                warnings.simplefilter("always")
                frame = pd.DataFrame(records)
                model = smf.mixedlm("dice ~ C(condition)", frame, groups=frame["patient"])
                fitted = model.fit(reml=True, method="lbfgs", maxiter=200, disp=False)
            output.update(status="fitted" if fitted.converged else "not_converged",
                          converged=bool(fitted.converged),
                          warnings=[str(w.message) for w in emitted],
                          coefficients={str(k): _number(v) for k, v in fitted.fe_params.items()},
                          fixed_effect_pvalues={str(k): _number(fitted.pvalues[k]) for k in fitted.fe_params.index})
        except Exception as exc:
            output.update(status="fit_failed", error=type(exc).__name__,
                          warnings=[str(w.message) for w in emitted] + [str(exc)])
        results.append(output)
    return {"status": "completed", "model": "patient-condition means; patient random intercept; REML",
            "caution": "Exploratory model; inspect convergence and boundary/singularity warnings.", "cohorts": results}
