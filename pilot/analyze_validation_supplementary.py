"""Run the prespecified supplementary checks on the locked case results."""

from collections import Counter
import csv
from fractions import Fraction
import json
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_formal_validation import (
    INDEPENDENT_SOURCE, PAIRED_SOURCE, STUDY, analyze, contrast,
    describe_cohort, file_hash, paired_phi, ratio, require, wilson,
)


PRIMARY_PATH = STUDY / "formal_validation_statistics.json"
PRIMARY_SHA256 = "ac02d8eccdca6983b34700256c55492548bab98e9193b4f6bad302ebda5b24df"
OUTPUT = STUDY / "supplementary_validation_statistics.json"
METHODS = ("A", "B")


def detection_record(detected, failures, total):
    return {
        "TP": detected,
        "FN": failures - detected,
        "low_quality_detection_fraction": ratio(detected, failures),
        "low_quality_detection_ci95": wilson(detected, failures) if failures else None,
        "missed_low_quality_per_100": ratio(100 * (failures - detected), total),
    }


def paired_detection(methods, failures, pairs, label, role):
    phi = paired_phi(pairs["both"], pairs["B_only"], pairs["A_only"], pairs["neither"])
    return contrast(
        label, role, (methods["B"]["TP"], failures), (methods["A"]["TP"], failures),
        PAIRED_SOURCE, phi=phi,
    )


def dice_sensitivity(groups, primary, cutoff):
    cohorts = {}
    for split, rows in groups.items():
        recoded = [
            dict(row, low_quality=int(float(row["dice_vs_reference"]) < cutoff))
            if row["status"] == "scored" else row
            for row in rows
        ]
        cohort = describe_cohort(recoded, cutoff, primary["thresholds"])
        previous = primary["cohorts"][split]
        require(cohort["N"] == previous["N"], f"Scored denominator changed in {split}.")
        for method in METHODS:
            counts = cohort["methods"][method]
            require(counts["R"] == previous["methods"][method]["R"], f"Alert count changed in {split}, {method}.")
            counts.update(detection_record(counts["TP"], cohort["F"], cohort["N"]))
        cohorts[split] = cohort

    internal, external = cohorts["internal_holdout"], cohorts["external_test"]
    a_across_cohorts = contrast(
        "A low-quality detection, BrEaST minus BUS-BRA holdout",
        "supplementary_nominal",
        (external["methods"]["A"]["TP"], external["F"]),
        (internal["methods"]["A"]["TP"], internal["F"]),
        INDEPENDENT_SOURCE,
    )
    b_minus_a = paired_detection(
        external["methods"], external["F"], external["paired_low_quality"],
        "BrEaST low-quality detection, B minus A", "supplementary_nominal",
    )
    return {
        "low_quality_dice_cutoff": cutoff,
        "cohorts": cohorts,
        "A_detection_external_minus_internal": a_across_cohorts,
        "external_detection_B_minus_A": b_minus_a,
    }


def equal_review(rows, cutoff, target):
    scored = [row for row in rows if row["status"] == "scored"]
    n = len(scored)
    budget = min(range(n + 1), key=lambda count: (abs(count - n * target), count))
    require(0 < budget <= n, "Equal-review budget cannot be evaluated.")
    failures = {row["patient_id"] for row in scored if float(row["dice_vs_reference"]) < cutoff}
    methods = {}
    selected_ids = {}
    for method in METHODS:
        score_column = f"risk_{method}"
        ordered = sorted(scored, key=lambda row: (-float(row[score_column]), row["patient_id"]))
        selected = ordered[:budget]
        selected_ids[method] = {row["patient_id"] for row in selected}
        require(len(selected_ids[method]) == budget, f"Repeated patient in ranking for {method}.")
        boundary = float(selected[-1][score_column])
        methods[method] = {
            "R": budget,
            "review_fraction": budget / n,
            **detection_record(len(selected_ids[method] & failures), len(failures), n),
            "last_selected_score": boundary,
            "patients_at_last_selected_score": sum(float(row[score_column]) == boundary for row in scored),
            "selected_at_last_selected_score": sum(float(row[score_column]) == boundary for row in selected),
            "selected_case_ids_in_rank_order": [row["patient_id"] for row in selected],
        }
    pairs = Counter((patient in selected_ids["A"], patient in selected_ids["B"]) for patient in failures)
    paired = {
        "both": pairs[True, True], "A_only": pairs[True, False],
        "B_only": pairs[False, True], "neither": pairs[False, False],
    }
    return {
        "dataset": "BrEaST",
        "N": n,
        "F": len(failures),
        "low_quality_dice_cutoff": cutoff,
        "target_review_fraction": float(target),
        "review_count_per_method": budget,
        "ranking_rule": "Descending risk score, then lexicographic official CaseID as stored in patient_id.",
        "methods": methods,
        "paired_low_quality": paired,
        "detection_B_minus_A": paired_detection(
            methods, len(failures), paired, "BrEaST equal-review detection, B minus A",
            "supplementary_descriptive_nominal",
        ),
        "interval_scope": "The nominal paired interval uses the observed ranking selections. It does not account for re-ranking in another sample.",
    }


def build_report():
    require(file_hash(PRIMARY_PATH) == PRIMARY_SHA256, "The verified primary analysis record changed.")
    primary = json.loads(PRIMARY_PATH.read_text(encoding="utf-8"))
    require(primary == analyze(), "The primary analysis could not be reproduced from its locked inputs.")
    sources = primary["sources"]
    csv_path = STUDY / sources["case_results"]["path"]
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = {
        split: [row for row in rows if row["split"] == split]
        for split in primary["cohorts"]
    }
    lock = json.loads((STUDY / sources["threshold_lock"]["path"]).read_text(encoding="utf-8"))
    target = Fraction(str(lock["calibration"]["review_target_fraction"]))
    return {
        "status": "prespecified_supplementary_analysis_complete",
        "scored_patients": primary["scored_patients"],
        "confidence_level": primary["confidence_level"],
        "intervals": "Nominal two-sided 95 percent. No multiplicity adjustment.",
        "fixed_thresholds": primary["thresholds"],
        "fixed_threshold_dice_sensitivity": {
            f"{cutoff:.2f}": dice_sensitivity(groups, primary, cutoff) for cutoff in (0.50, 0.80)
        },
        "external_equal_review": equal_review(
            groups["external_test"], primary["low_quality_reference"]["cutoff"], target,
        ),
        "sources": {
            **sources,
            "primary_analysis": {"path": PRIMARY_PATH.name, "sha256": PRIMARY_SHA256},
            "primary_script": {
                "path": "analyze_formal_validation.py", "sha256": primary["analysis_script_sha256"],
            },
        },
        "analysis_script_sha256": file_hash(Path(__file__)),
        "python": sys.version.split()[0],
    }


def main():
    report = build_report()
    content = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if OUTPUT.exists():
        require(OUTPUT.read_text(encoding="utf-8") == content, "An existing supplementary record differs.")
    else:
        with NamedTemporaryFile(mode="w", dir=STUDY, prefix=".supplementary-stats-", encoding="utf-8", delete=False) as file:
            staged = Path(file.name)
            file.write(content)
        try:
            staged.rename(OUTPUT)
        finally:
            staged.unlink(missing_ok=True)

    sensitivity = {}
    for cutoff, result in report["fixed_threshold_dice_sensitivity"].items():
        sensitivity[cutoff] = {
            split: {
                "N": group["N"], "F": group["F"],
                **{method: {
                    key: group["methods"][method][key] for key in ("R", "TP", "FN", "low_quality_detection_fraction")
                } for method in METHODS},
            }
            for split, group in result["cohorts"].items()
        }
        for key in ("A_detection_external_minus_internal", "external_detection_B_minus_A"):
            item = result[key]
            sensitivity[cutoff][key] = {"difference_pp": item["difference_pp"], "ci95_pp": item["ci95_pp"]}
    equal = report["external_equal_review"]
    print(json.dumps({
        "sensitivity": sensitivity,
        "equal_review": {
            "N": equal["N"], "F": equal["F"],
            "methods": {method: {
                key: value for key, value in equal["methods"][method].items() if key != "selected_case_ids_in_rank_order"
            } for method in METHODS},
            "paired_low_quality": equal["paired_low_quality"],
            "contrast": equal["detection_B_minus_A"],
        },
    }, allow_nan=False))


if __name__ == "__main__":
    main()
