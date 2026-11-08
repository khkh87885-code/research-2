"""Compute the planned H1–H3 contrasts from the locked per-patient results."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys
from tempfile import NamedTemporaryFile


STUDY = Path(__file__).resolve().parent
OUTPUT = STUDY / "formal_validation_statistics.json"
INPUTS = {
    "protocol": ("formal_experiment_protocol.md", "9f8741e988cd5d894ddcc3a77ee3878db1db78a4820ffea050cd7f073aa0950d"),
    "threshold_lock": ("formal_validation_threshold_lock.json", "67395cc2c1c882ed4aabdd80ac9e614a3e100b42eaae0d0e55e8a24ac256f655"),
    "case_results": ("formal_validation/validation_cases.csv", "6ac44d3d3557caceba1e47dab4f87bc3b180568195c0d0ecb6506fbb04d07961"),
    "scoring_summary": ("formal_validation/validation_summary.json", "417a989469b5b2858eec5f7e16e5462b23558e592733367b0dc75e58b180813b"),
}
INDEPENDENT_SOURCE = "https://pubmed.ncbi.nlm.nih.gov/9595617/"
PAIRED_SOURCE = "https://www.site.uottawa.ca/~nat/Courses/csi5388/Newcombe.1998.pdf#page=5"
Z95 = NormalDist().inv_cdf(0.975)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def wilson(successes, total):
    """Wilson score limits without continuity correction."""
    p = successes / total
    z2 = Z95 ** 2
    scale = 1.0 + z2 / total
    center = (p + z2 / (2 * total)) / scale
    half_width = Z95 * math.sqrt(p * (1 - p) / total + z2 / (4 * total ** 2)) / scale
    return max(0.0, center - half_width), min(1.0, center + half_width)


def paired_phi(both, first_only, second_only, neither):
    """Newcombe's paired method 10 adjusts positive association only."""
    total = both + first_only + second_only + neither
    margin_product = (
        (both + first_only) * (second_only + neither)
        * (both + second_only) * (first_only + neither)
    )
    if margin_product == 0:
        return 0.0
    numerator = both * neither - first_only * second_only
    if numerator > 0:
        numerator = max(numerator - total / 2, 0.0)
    return numerator / math.sqrt(margin_product)


def newcombe_difference(x_first, n_first, x_second, n_second, phi=0.0):
    """First minus second, using the independent or paired method 10."""
    require(0 <= x_first <= n_first and 0 <= x_second <= n_second, "Invalid proportions.")
    if not n_first or not n_second:
        return None, None

    first, second = x_first / n_first, x_second / n_second
    lower_first, upper_first = wilson(x_first, n_first)
    lower_second, upper_second = wilson(x_second, n_second)
    estimate = first - second

    def radius(left, right):
        variance = left ** 2 + right ** 2 - 2 * phi * left * right
        return math.sqrt(max(variance, 0.0))

    lower = estimate - radius(first - lower_first, upper_second - second)
    upper = estimate + radius(upper_first - first, second - lower_second)
    return estimate, (lower, upper)


def describe_cohort(rows, cutoff, thresholds):
    valid = []
    failed = []
    for row in rows:
        if row["status"] == "failed":
            failed.append(row["image_id"])
            continue
        require(row["status"] == "scored", f"Unrecognized status for {row['image_id']}.")
        dice = float(row["dice_vs_reference"])
        risks = [float(row[f"risk_{method}"]) for method in ("A", "B")]
        require(0 <= dice <= 1 and all(math.isfinite(value) for value in risks), f"Invalid score for {row['image_id']}.")
        quality_flag = int(dice < cutoff)
        alerts = tuple(int(risk >= thresholds[method]) for method, risk in zip(("A", "B"), risks))
        require(
            int(row["low_quality"]) == quality_flag and tuple(int(row[f"alert_{method}"]) for method in ("A", "B")) == alerts,
            f"Saved flags differ from the locked rules for {row['image_id']}.",
        )
        valid.append((quality_flag, *alerts))

    n, f = len(valid), sum(entry[0] for entry in valid)
    methods = {}
    for index, method in enumerate(("A", "B"), 1):
        reviewed = sum(entry[index] for entry in valid)
        detected = sum(entry[index] for entry in valid if entry[0])
        missed = f - detected
        methods[method] = {
            "R": reviewed, "TP": detected, "FN": missed,
            "low_quality_detection_fraction": ratio(detected, f),
            "review_fraction": ratio(reviewed, n),
            "missed_low_quality_per_100": ratio(100 * missed, n),
            "acceptable_cases_alerted": reviewed - detected,
            "acceptable_cases": n - f,
            "review_fraction_among_acceptable": ratio(reviewed - detected, n - f),
        }
    pairs = Counter((a, b) for low, a, b in valid if low)
    return {
        "scheduled_patients": len(rows),
        "technical_failures": failed,
        "N": n,
        "F": f,
        "low_quality_fraction": ratio(f, n),
        "methods": methods,
        "paired_low_quality": {
            "both": pairs[1, 1], "A_only": pairs[1, 0],
            "B_only": pairs[0, 1], "neither": pairs[0, 0],
        },
    }


def contrast(label, role, first, second, source, phi=0.0):
    estimate, interval = newcombe_difference(*first, *second, phi=phi)
    return {
        "contrast": label,
        "role": role,
        "method": "Newcombe method 10",
        "reference": source,
        "first": {"events": first[0], "denominator": first[1]},
        "second": {"events": second[0], "denominator": second[1]},
        "difference_pp": None if estimate is None else 100 * estimate,
        "ci95_pp": None if interval is None else [100 * bound for bound in interval],
        "interval_excludes_zero": None if interval is None else interval[0] > 0 or interval[1] < 0,
    }


def analyze():
    inputs = {}
    for label, (relative, expected) in INPUTS.items():
        path = STUDY / relative
        require(file_hash(path) == expected, f"Locked source changed: {path}")
        inputs[label] = path

    lock = json.loads(inputs["threshold_lock"].read_text(encoding="utf-8"))
    summary = json.loads(inputs["scoring_summary"].read_text(encoding="utf-8"))
    thresholds = {method: lock["methods"][method]["threshold"] for method in ("A", "B")}
    quality = lock["quality_reference"]
    verification = summary["input_verification"]
    require(
        lock["status"] == "locked"
        and lock["alert_operator"] == verification["alert_operator"] == ">="
        and thresholds == verification["thresholds"]
        and quality == verification["quality_reference"]
        and quality["low_quality_operator"] == "<" and quality["cutoff"] == 0.7
        and summary["validation_cases_sha256"] == INPUTS["case_results"][1]
        and verification["protocol_sha256"] == INPUTS["protocol"][1]
        and verification["threshold_lock_sha256"] == INPUTS["threshold_lock"][1],
        "Scoring rules or result provenance differ from the locked inputs.",
    )

    with inputs["case_results"].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    splits = {"internal_holdout": "BUS-BRA", "external_test": "BrEaST"}
    groups = defaultdict(list)
    identities = set()
    for row in rows:
        split = row["split"]
        require(split in splits and row["dataset"] == splits[split], "Unexpected test cohort.")
        patient = (row["dataset"], int(row["patient_id"]))
        require(patient not in identities, f"Repeated patient within a cohort: {patient}")
        identities.add(patient)
        groups[split].append(row)
    require(set(groups) == set(splits), "Expected both test cohorts.")

    cohorts = {}
    for split, group in groups.items():
        result = describe_cohort(group, quality["cutoff"], thresholds)
        expected = summary["cohorts"][split]
        for field in ("scheduled_patients", "N", "F", "technical_failures", "paired_low_quality"):
            require(result[field] == expected[field], f"Scoring summary differs for {split}: {field}")
        for method, counts in result["methods"].items():
            require(
                all(counts[field] == value for field, value in expected["methods"][method].items()),
                f"Scoring summary differs for {split}, method {method}.",
            )
        cohorts[split] = result

    internal, external = cohorts["internal_holdout"], cohorts["external_test"]
    internal_a, external_a = internal["methods"]["A"], external["methods"]["A"]
    external_b = external["methods"]["B"]
    dev_budget = {method: lock["methods"][method]["development_review_count"] for method in thresholds}
    for method, count in dev_budget.items():
        require(
            count == verification["development_results_replayed"]["methods"][method]["R"],
            f"Development budget differs from the replay for method {method}.",
        )
    matched_budget = dev_budget["A"] == dev_budget["B"]

    h1 = contrast(
        "A low-quality detection, BrEaST minus BUS-BRA holdout", "primary",
        (external_a["TP"], external["F"]), (internal_a["TP"], internal["F"]), INDEPENDENT_SOURCE,
    )
    h2 = contrast(
        "A simulated review, BrEaST minus BUS-BRA holdout", "secondary_nominal",
        (external_a["R"], external["N"]), (internal_a["R"], internal["N"]), INDEPENDENT_SOURCE,
    )
    pair = external["paired_low_quality"]
    phi = paired_phi(pair["both"], pair["B_only"], pair["A_only"], pair["neither"])
    h3 = contrast(
        "Low-quality detection in BrEaST, B minus A",
        "secondary_nominal" if matched_budget else "descriptive_unmatched_development_budget",
        (external_b["TP"], external["F"]), (external_a["TP"], external["F"]), PAIRED_SOURCE, phi=phi,
    )
    h3.update(
        paired_low_quality=pair,
        phi_with_positive_association_continuity_correction=phi,
        development_review_counts=dev_budget,
        development_review_budget_matched=matched_budget,
        external_review_difference_B_minus_A_pp=ratio(100 * (external_b["R"] - external_a["R"]), external["N"]),
    )

    return {
        "status": "planned_h1_h3_analysis_complete",
        "unit": "patient",
        "scored_patients": internal["N"] + external["N"],
        "confidence_level": 0.95,
        "secondary_intervals": "nominal, without multiplicity adjustment",
        "low_quality_reference": quality,
        "alert_operator": lock["alert_operator"],
        "thresholds": thresholds,
        "cohorts": cohorts,
        "contrasts": {"H1": h1, "H2": h2, "H3": h3},
        "sources": {
            label: {"path": relative, "sha256": checksum}
            for label, (relative, checksum) in INPUTS.items()
        },
        "analysis_script_sha256": file_hash(Path(__file__)),
        "python": sys.version.split()[0],
    }


def main():
    report = analyze()
    content = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if OUTPUT.exists():
        require(OUTPUT.read_text(encoding="utf-8") == content, "An existing statistical record differs.")
    else:
        with NamedTemporaryFile(mode="w", dir=STUDY, prefix=".validation-stats-", encoding="utf-8", delete=False) as file:
            staged = Path(file.name)
            file.write(content)
        try:
            staged.rename(OUTPUT)
        finally:
            staged.unlink(missing_ok=True)
    print(json.dumps({
        "scored_patients": report["scored_patients"],
        "contrasts": report["contrasts"],
        "python": report["python"],
    }, allow_nan=False))


if __name__ == "__main__":
    main()
