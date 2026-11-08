"""Patient-clustered exploratory analyses with explicit missing-data accounting.

No patient-level inference is enabled until the caller verifies patient identity.
Each contrast first pairs baseline and processed values on the SAME image,
then averages image differences within patient, and weights patients equally.
Patient bootstrap samples with replacement; repeated draws retain their weights.
The tests are sensitivity analyses, not an automatic normality-driven selector.
Noninferiority requires an explicit, prospectively accepted margin from the caller.
"""

from collections import defaultdict
import math
import warnings

import numpy as np
from scipy import stats


STRATA = ("dataset", "model", "prompt_strategy", "split")


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _summary(values):
    data = np.asarray(values, dtype=float)
    if not len(data):
        return {"n": 0, "mean": None, "sd": None, "median": None, "q1": None, "q3": None}
    return {
        "n": len(data), "mean": float(np.mean(data)),
        "sd": float(np.std(data, ddof=1)) if len(data) > 1 else None,
        "median": float(np.median(data)), "q1": float(np.quantile(data, .25)),
        "q3": float(np.quantile(data, .75)),
    }


def holm_adjust(pvalues):
    """Holm over the supplied contrast family; missing tests retain a family slot.

    Missing p-values remain None in the output, and behave as non-rejections
    at the end of the ordered family rather than shrinking multiplicity.
    """
    result = [None] * len(pvalues)
    valid = [(i, _number(p)) for i, p in enumerate(pvalues) if _number(p) is not None]
    if any(not 0 <= p <= 1 for _, p in valid):
        raise ValueError("p values must lie between zero and one")
    ordered = sorted(valid, key=lambda pair: pair[1])
    previous = 0.0
    for rank, (index, pvalue) in enumerate(ordered):
        previous = max(previous, min(1.0, (len(pvalues) - rank) * pvalue))
        result[index] = previous
    return result


def patient_bootstrap(values, seed=20261108, n_boot=2000):
    """Percentile CI for an equal-patient mean; input has one value per patient."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or not np.isfinite(data).all():
        raise ValueError("Bootstrap requires a finite one-dimensional patient array")
    if int(n_boot) != n_boot or n_boot < 100:
        raise ValueError("n_boot must be an integer of at least 100")
    n_boot = int(n_boot)
    if len(data) < 2:
        return {"status": "insufficient_patients", "ci95": None, "n_patients": len(data)}
    rng = np.random.RandomState(seed)
    estimates = []
    for start in range(0, n_boot, 256):
        indexes = rng.randint(0, len(data), size=(min(256, n_boot - start), len(data)))
        estimates.extend(np.mean(data[indexes], axis=1).tolist())
    return {
        "status": "degenerate_zero_variance" if np.ptp(data) == 0 else "estimated",
        "ci95": np.percentile(estimates, [2.5, 97.5]).tolist(),
        "n_patients": len(data), "n_boot": n_boot, "seed": seed,
        "method": "patient percentile bootstrap with replacement; duplicate draws retained",
    }


def _test_result(function, *args, **kwargs):
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = function(*args, **kwargs)
        statistic, pvalue = _number(value.statistic), _number(value.pvalue)
        return {
            "status": "estimated" if statistic is not None and pvalue is not None else "not_estimable",
            "statistic": statistic, "p_value": pvalue,
            "warnings": [str(item.message) for item in caught],
        }
    except (ValueError, RuntimeError, ZeroDivisionError) as error:
        return {"status": "failed", "statistic": None, "p_value": None, "reason": str(error)}


def _unavailable(reason):
    return {"status": "not_estimable", "statistic": None, "p_value": None, "reason": reason}


def _contrast(differences, baseline_values, processed_values, metric, delta, seed, n_boot):
    values = np.asarray(differences, dtype=float)
    n = len(values)
    bootstrap = patient_bootstrap(values, seed, n_boot)
    variable = n >= 2 and np.ptp(values) > 0
    paired_t = _test_result(stats.ttest_1samp, values, 0.0) if variable else _unavailable("fewer_than_two_patients_or_zero_difference_variance")
    wilcoxon = _test_result(stats.wilcoxon, values, zero_method="wilcox", alternative="two-sided") if n >= 2 and np.any(values != 0) else _unavailable("insufficient_patients_or_all_zero_differences")
    normality = _test_result(stats.shapiro, values) if 3 <= n <= 5000 and np.ptp(values) > 0 else _unavailable("requires_3_to_5000_patients_and_nonconstant_differences")
    sd = float(np.std(values, ddof=1)) if n >= 2 else None
    mean_difference = float(np.mean(values)) if n else None
    noninferiority = {"status": "not_configured", "margin": delta, "meets_prespecified_criterion": None}
    if delta is not None:
        noninferiority.update(status="not_estimable", reason="requires_nondegenerate_patient_bootstrap")
        if bootstrap["status"] == "estimated":
            low, high = bootstrap["ci95"]
            if metric == "dice":
                noninferiority.update(status="estimated", criterion="CI95 lower > -margin", meets_prespecified_criterion=bool(low > -delta))
            elif metric == "hd95":
                noninferiority.update(status="estimated", criterion="CI95 upper < margin", meets_prespecified_criterion=bool(high < delta))
            else:
                noninferiority.update(status="unsupported_metric", reason="margin_direction_only_defined_for_dice_and_hd95")
            if noninferiority["status"] == "estimated":
                noninferiority.pop("reason", None)
        noninferiority["margin_source"] = "caller supplied; must be prospectively justified outside this analysis"
    return {
        "n_paired_patients": n, "patient_difference": _summary(values),
        "baseline_patient_means": _summary(baseline_values),
        "processed_patient_means": _summary(processed_values),
        "difference_direction": "processed minus baseline", "bootstrap": bootstrap,
        "paired_t": paired_t, "wilcoxon": wilcoxon, "normality_of_patient_differences": normality,
        "cohen_dz": mean_difference / sd if sd is not None and sd > 0 else None,
        "noninferiority": noninferiority,
    }


def _strata(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(field, "unknown")) for field in STRATA)].append(row)
    return sorted(groups.items())


def analyze(rows, metric="dice", baseline="baseline", delta=None, seed=20261108,
            n_boot=2000, patient_ids_verified=False):
    """Return JSON-safe descriptive and patient-paired results, separated by cohort/model/prompt/split."""
    if delta is not None and (_number(delta) is None or float(delta) <= 0):
        raise ValueError("An explicitly confirmed noninferiority margin must be positive and finite")
    delta = None if delta is None else float(delta)
    if int(n_boot) != n_boot or n_boot < 100:
        raise ValueError("n_boot must be an integer of at least 100")
    n_boot = int(n_boot)
    rows = list(rows)
    output = {
        "status": "no_data" if not rows else "descriptive_only" if not patient_ids_verified else "analyzed",
        "metric": metric, "baseline": baseline, "groups": [],
        "patient_ids_verified": bool(patient_ids_verified),
        "sampling_unit": "dataset + verified patient; one equal weight per patient",
        "pairing": "same image within patient, then equal-patient mean of image differences",
        "analysis_role": "exploratory unless covered by an independently frozen protocol",
        "multiplicity": "Holm within each stratum across all variant-baseline contrasts, separately for t and Wilcoxon; unavailable tests retain family slots",
        "alpha": 0.05, "confidence_level": 0.95,
    }
    for key, group in _strata(rows):
        result = dict(zip(STRATA, key))
        by_variant = defaultdict(list)
        for row in group:
            by_variant[str(row.get("variant", "unknown"))].append(row)
        result["descriptive"] = {}
        for variant, items in sorted(by_variant.items()):
            valid = [_number(row.get(metric)) for row in items]
            result["descriptive"][variant] = dict(
                _summary([x for x in valid if x is not None]),
                n_rows=len(items), n_missing_or_invalid=sum(x is None for x in valid),
                unit="image-version (descriptive only)",
            )
        result.update(contrasts=[], friedman=_unavailable("not_run"))
        if not patient_ids_verified:
            result.update(inference_status="blocked_unverified_patient_ids")
            output["groups"].append(result)
            continue
        if any(row.get("patient_id") in (None, "") or row.get("image_id") in (None, "") for row in group):
            result.update(inference_status="blocked_missing_patient_or_image_id")
            output["groups"].append(result)
            continue
        identities = [(str(row["patient_id"]), str(row["image_id"]), str(row.get("variant", "unknown"))) for row in group]
        if len(set(identities)) != len(identities):
            result.update(inference_status="blocked_duplicate_patient_image_variant_rows")
            output["groups"].append(result)
            continue
        result["inference_status"] = "computed_with_available_pairs"
        lookup = {(str(row["patient_id"]), str(row["image_id"]), str(row.get("variant", "unknown"))): _number(row.get(metric)) for row in group}
        variants = sorted(by_variant)
        for variant in (v for v in variants if v != baseline):
            patient_pairs = defaultdict(list)
            for (patient, image, condition), value in lookup.items():
                if condition != variant:
                    continue
                base = lookup.get((patient, image, baseline))
                if base is not None and value is not None:
                    patient_pairs[patient].append((base, value))
            patient_arrays = [np.mean(patient_pairs[p], axis=0) for p in sorted(patient_pairs)]
            differences = [float(pair[1] - pair[0]) for pair in patient_arrays]
            comparison = _contrast(differences, [float(p[0]) for p in patient_arrays],
                                   [float(p[1]) for p in patient_arrays], metric, delta, seed, n_boot)
            comparison.update(
                variant=variant, baseline=baseline,
                n_paired_images=sum(len(p) for p in patient_pairs.values()),
                n_variant_rows=len(by_variant[variant]),
                n_variant_rows_without_valid_baseline_pair=len(by_variant[variant]) - sum(len(p) for p in patient_pairs.values()),
            )
            result["contrasts"].append(comparison)
        for test in ("paired_t", "wilcoxon"):
            adjusted = holm_adjust([c[test]["p_value"] for c in result["contrasts"]])
            for contrast, adjusted_p in zip(result["contrasts"], adjusted):
                contrast[test]["p_holm"] = adjusted_p
        # Friedman uses common images across ALL variants before patient averaging.
        patient_images = defaultdict(set)
        for patient, image, _ in lookup:
            patient_images[patient].add(image)
        blocks = []
        for patient in sorted(patient_images):
            common = [image for image in sorted(patient_images[patient])
                      if all(lookup.get((patient, image, variant)) is not None for variant in variants)]
            if common:
                blocks.append([float(np.mean([lookup[(patient, image, variant)] for image in common])) for variant in variants])
        if len(variants) >= 3 and len(blocks) >= 3:
            array = np.asarray(blocks)
            result["friedman"] = _test_result(stats.friedmanchisquare, *[array[:, i] for i in range(len(variants))])
            result["friedman"].update(n_complete_patients=len(blocks), variants=variants, approximation_caution="chi-square approximation may be unreliable with small patient/group counts")
        else:
            result["friedman"] = _unavailable("requires_at_least_three_variants_and_three_complete_patients")
        output["groups"].append(result)
    return output


def mixed_effects(rows, metric="dice", patient_ids_verified=False):
    """Fit metric ~ C(format)*scale, patient intercept, optional nested image intercept.

    Formats use ``compression_format`` (or ``format``), scales use
    ``resolution_scale`` (or ``scale``). Every cohort/model/prompt/split is fit
    separately. Rank deficiency, fitting errors and nonconvergence are reported.
    No p-values are interpreted as confirmatory findings by this function.
    """
    if not patient_ids_verified:
        return {"status": "blocked_unverified_patient_ids", "groups": []}
    import pandas as pd
    import statsmodels.formula.api as smf
    result = {"status": "no_data", "groups": [], "formula": "value ~ C(compression_format) * resolution_scale"}
    for key, items in _strata(list(rows)):
        group_result = dict(zip(STRATA, key))
        group_result["n_input_rows"] = len(items)
        records = []
        for row in items:
            value = _number(row.get(metric))
            scale = _number(row.get("resolution_scale", row.get("scale")))
            encoding = row.get("compression_format", row.get("format"))
            patient, image = row.get("patient_id"), row.get("image_id")
            if value is not None and scale is not None and scale > 0 and encoding and patient not in (None, "") and image not in (None, ""):
                condition = row.get("variant") or repr((str(encoding), scale, row.get("compression_quality", row.get("quality"))))
                records.append({"value": value, "resolution_scale": scale,
                                "compression_format": str(encoding), "patient": str(patient),
                                "image": str(image), "condition": str(condition)})
        group_result.update(n_valid_rows=len(records), n_excluded_invalid_rows=len(items) - len(records))
        if len({r["patient"] for r in records}) < 3:
            group_result.update(status="insufficient_patients", reason="requires_at_least_three_verified_patients")
            result["groups"].append(group_result)
            continue
        identities = [(r["patient"], r["image"], r["condition"]) for r in records]
        if len(identities) != len(set(identities)):
            group_result.update(status="blocked_duplicate_patient_image_variant_rows")
            result["groups"].append(group_result)
            continue
        frame = pd.DataFrame.from_records(records)
        nested = frame.groupby("patient")["image"].nunique().max() > 1
        vc_formula = {"image": "0 + C(image)"} if nested else None
        try:
            model = smf.mixedlm(result["formula"], frame, groups=frame["patient"], re_formula="1", vc_formula=vc_formula)
            rank = int(np.linalg.matrix_rank(model.exog))
            group_result.update(fixed_effect_rank=rank, n_fixed_columns=model.exog.shape[1], n_patients=int(frame["patient"].nunique()),
                                random_effects="patient intercept + nested image intercept" if nested else "patient intercept (one image per patient)")
            if rank < model.exog.shape[1]:
                group_result.update(status="rank_deficient", reason="format_by_scale_design_does_not_identify_all_fixed_effects")
            elif len(frame) <= model.exog.shape[1]:
                group_result.update(status="insufficient_observations")
            else:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fitted = model.fit(reml=True, method="lbfgs", maxiter=300, disp=False)
                confidence = fitted.conf_int()
                parameters = {}
                for name in fitted.fe_params.index:
                    parameters[name] = {"estimate": _number(fitted.fe_params[name]), "p_value": _number(fitted.pvalues[name]),
                                        "ci95": [_number(x) for x in confidence.loc[name].tolist()]}
                group_result.update(status="converged" if fitted.converged else "not_converged", converged=bool(fitted.converged),
                                    fixed_effects=parameters, warnings=[str(x.message) for x in caught],
                                    residual_variance=_number(fitted.scale), image_variance_components=[_number(x) for x in fitted.vcomp],
                                    patient_intercept_variance=_number(fitted.cov_re.iloc[0, 0]))
        except (ValueError, RuntimeError, np.linalg.LinAlgError, ZeroDivisionError) as error:
            group_result.update(status="failed", reason="%s: %s" % (type(error).__name__, error))
        result["groups"].append(group_result)
    if result["groups"]:
        result["status"] = "completed_with_diagnostics"
    return result
