
import csv
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_development_pilot import DEV_MANIFEST_SHA256, describe, measure_case, sha256, wilson_interval
from run_single_case import CHECKPOINT_SHA256, DATA_ROOT, LOW_QUALITY_DICE, MIB, PILOT_DIR, SAM_COMMIT, SAM_SOURCE


EXPECTED_PATIENTS = 213
PROTOCOL_SHA256 = "9f8741e988cd5d894ddcc3a77ee3878db1db78a4820ffea050cd7f073aa0950d"
PILOT_CASES_SHA256 = "22359d5e760a59f73ef180940a7a9e43452252b94706a38d3c14d700c75533cf"


def run_patient(row, predictor, device):
    errors = []
    for attempt in range(2):
        try:
            result = measure_case(row, predictor, device)
        except (OSError, ValueError, RuntimeError) as error:
            errors.append(f"{type(error).__name__}: {error}")
            predictor.reset_image()
            if isinstance(error, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
        else:
            result.update(status="scored", retries=attempt, error=" | ".join(errors))
            return result
    return {
        "patient_id": int(row["patient_id"]),
        "image_id": row["image_id"],
        "pathology": row["pathology"],
        "status": "failed",
        "retries": 1,
        "error": " | ".join(errors),
    }


def operating_points(cases, method):
    groups = defaultdict(list)
    for case in cases:
        groups[case[method]].append(case["low_quality"])
    above_max = float(np.nextafter(max(groups), np.inf))
    points = [(above_max, 0, 0, 0)]
    reviewed = detected = 0
    for score, labels in sorted(groups.items(), reverse=True):
        reviewed += len(labels)
        detected += sum(labels)
        points.append((score, reviewed, detected, len(labels)))
    return points


def summarize_method(cases, method, points, low_quality_count):
    total = len(cases)
    # Minimize distance to 20% using integer counts, keeping exact ties together.
    threshold, reviewed, detected, boundary_size = min(
        points, key=lambda point: (abs(5 * point[1] - total), point[1])
    )
    for case in cases:
        case[f"candidate_alert_{method[-1]}"] = int(case[method] >= threshold)
    values = [case[method] for case in cases]
    return {
        "threshold": threshold,
        "candidate_reviews": reviewed,
        "candidate_review_fraction": reviewed / total,
        "cases_equal_to_threshold": boundary_size,
        "true_alerts": detected,
        "missed_low_quality": low_quality_count - detected,
        "acceptable_cases_alerted": reviewed - detected,
        "acceptable_cases_not_alerted": total - low_quality_count - reviewed + detected,
        "low_quality_detection_fraction": detected / low_quality_count if low_quality_count else None,
        "descriptive_detection_wilson_95_ci": (
            wilson_interval(detected, low_quality_count) if low_quality_count else None
        ),
        "low_quality_per_review_fraction": detected / reviewed if reviewed else None,
        "missed_low_quality_per_100": 100 * (low_quality_count - detected) / total,
        "scores": {
            **describe(values),
            "distinct": len(set(values)),
            "zero": values.count(0.0),
            "negative": sum(value < 0 for value in values),
        },
    }


def check_repeated_pilot_cases(cases):
    pilot_file = PILOT_DIR / "pilot_40" / "pilot_40_cases.csv"
    if sha256(pilot_file) != PILOT_CASES_SHA256:
        raise RuntimeError("The prior 40-case results differ from the audited file.")
    with pilot_file.open(encoding="utf-8-sig", newline="") as handle:
        previous = {row["image_id"]: row for row in csv.DictReader(handle)}
    overlap = [row for row in cases if row["image_id"] in previous]
    keys = ("dice_vs_reference", "risk_A", "risk_B")
    return {
        "previous_pilot_count": len(previous),
        "previous_pilot_cases_repeated": len(overlap),
        "max_absolute_difference": {
            key: max(abs(row[key] - float(previous[row["image_id"]][key])) for row in overlap)
            if overlap else None
            for key in keys
        },
    }


def save_plot(points, summary, destination, config_dir):
    os.environ["MPLCONFIGDIR"] = str(config_dir)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    total = summary["scored_patients"]
    failures = summary["low_quality_count"]
    figure = Figure(figsize=(8.5, 5.2), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    for method, name, color, style, marker in (
        ("risk_A", "A  Model confidence", "#246c83", "-", "o"),
        ("risk_B", "B  Box perturbations", "#bf4f2a", "--", "s"),
    ):
        curve = points[method]
        axis.step(
            [100 * row[1] / total for row in curve],
            [100 * row[2] / failures for row in curve],
            where="post", color=color, linestyle=style, linewidth=2, label=name,
        )
        candidate = summary["methods"][method]
        axis.scatter(
            100 * candidate["candidate_review_fraction"],
            100 * candidate["low_quality_detection_fraction"],
            color=color, marker=marker, edgecolors="white", linewidths=0.6, s=65, zorder=5,
        )
    axis.axvline(20, color="#65717f", linestyle=":", linewidth=1.4, label="20% target")
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 103)
    axis.set_xlabel("Development cases sent to simulated review (%)")
    axis.set_ylabel("Low-quality cases captured (%)")
    axis.set_title(f"BUS-BRA development data | {total} patients, {failures} low-quality cases", loc="left", fontweight="bold")
    axis.grid(alpha=0.17)
    axis.legend(loc="lower right", frameon=False)
    figure.savefig(destination, dpi=160)


def main():
    output = PILOT_DIR / "development_213"
    if output.exists():
        raise FileExistsError(f"Development results already exist: {output}")

    manifest = DATA_ROOT / "busbra_splits" / "busbra_development.csv"
    protocol = PILOT_DIR / "formal_experiment_protocol.md"
    if sha256(manifest) != DEV_MANIFEST_SHA256 or sha256(protocol) != PROTOCOL_SHA256:
        raise RuntimeError("The development manifest or reviewed draft differs from the referenced file.")
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        development = sorted(csv.DictReader(stream), key=lambda row: int(row["patient_id"]))
    if (
        len(development) != EXPECTED_PATIENTS
        or len({row["patient_id"] for row in development}) != EXPECTED_PATIENTS
        or len({row["image_id"] for row in development}) != EXPECTED_PATIENTS
        or any(row["dataset"] != "BUS-BRA" or row["split"] != "development" for row in development)
    ):
        raise ValueError("The input must contain the 213 distinct BUS-BRA development patients.")
    print(f"Verified {len(development)} development patients.", flush=True)

    commit = subprocess.check_output(
        ["git.exe", "-C", str(SAM_SOURCE), "rev-parse", "--verify", "HEAD"], text=True
    ).strip()
    changes = subprocess.check_output(
        ["git.exe", "-C", str(SAM_SOURCE), "status", "--porcelain"], text=True
    ).strip()
    if commit != SAM_COMMIT or changes:
        raise RuntimeError("The local SAM source differs from the verified commit.")
    sys.path.insert(0, str(SAM_SOURCE))
    from segment_anything import SamPredictor, sam_model_registry

    checkpoint = DATA_ROOT / "models" / "sam_vit_b_01ec64.pth"
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("The SAM checkpoint SHA-256 did not match.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the pilot runtime.")

    device = torch.device("cuda:0")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    free_mib = torch.cuda.mem_get_info(device)[0] / MIB
    start = perf_counter()
    model = sam_model_registry["vit_b"]()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval()
    predictor = SamPredictor(model)
    torch.cuda.synchronize(device)
    load_ms = 1000 * (perf_counter() - start)
    print(f"SAM ready on {torch.cuda.get_device_name(device)}.", flush=True)

    results = []
    with torch.inference_mode():
        for index, row in enumerate(development, 1):
            results.append(run_patient(row, predictor, device))
            if index % 25 == 0 or index == EXPECTED_PATIENTS:
                print(f"Processed {index}/{EXPECTED_PATIENTS} development patients.", flush=True)

    scored = [row for row in results if row["status"] == "scored"]
    failures = [row for row in scored if row["low_quality"]]
    if not scored:
        raise RuntimeError(f"No patients produced valid scores. First error: {results[0]['error']}")
    points = {method: operating_points(scored, method) for method in ("risk_A", "risk_B")}
    methods = {
        method: summarize_method(scored, method, curve, len(failures))
        for method, curve in points.items()
    }
    pairs = Counter((case["candidate_alert_A"], case["candidate_alert_B"]) for case in failures)

    summary = {
        "status": "development_results_candidate_thresholds",
        "dataset": "BUS-BRA",
        "split": "development",
        "scheduled_patients": EXPECTED_PATIENTS,
        "scored_patients": len(scored),
        "technical_failures": [row for row in results if row["status"] == "failed"],
        "patients_with_a_retry": sum(bool(row["retries"]) for row in results),
        "pathology_counts_scored": dict(Counter(row["pathology"] for row in scored)),
        "dice_low_quality_cutoff": LOW_QUALITY_DICE,
        "low_quality_count": len(failures),
        "low_quality_fraction": len(failures) / len(scored),
        "low_quality_wilson_95_ci": wilson_interval(len(failures), len(scored)),
        "dice": describe([row["dice_vs_reference"] for row in scored]),
        "review_target_fraction": 0.20,
        "candidate_rule": "Alert when score >= threshold. Choose closest to 20%; exact ties stay together; fewer reviews breaks an equal distance.",
        "methods": methods,
        "same_candidate_review_count": methods["risk_A"]["candidate_reviews"] == methods["risk_B"]["candidate_reviews"],
        "paired_low_quality_at_candidates": {
            "both": pairs[(1, 1)],
            "A_only": pairs[(1, 0)],
            "B_only": pairs[(0, 1)],
            "neither": pairs[(0, 0)],
        },
        "repeated_pilot_case_check": check_repeated_pilot_cases(scored),
        "initial_empty_masks": sum(row["initial_mask_empty"] for row in scored),
        "inference_ms": describe([row["inference_ms"] for row in scored]),
        "total_inference_ms": sum(row["inference_ms"] for row in scored),
        "model_load_ms": load_ms,
        "gpu": torch.cuda.get_device_name(device),
        "free_gpu_before_model_mib": free_mib,
        "peak_pytorch_allocated_mib": torch.cuda.max_memory_allocated(device) / MIB,
        "peak_pytorch_reserved_mib": torch.cuda.max_memory_reserved(device) / MIB,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "weight_dtype": str(next(model.parameters()).dtype),
        "sam_commit": commit,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "development_manifest_sha256": DEV_MANIFEST_SHA256,
        "protocol_draft_sha256": PROTOCOL_SHA256,
    }

    with tempfile.TemporaryDirectory(prefix=".development-213-", dir=PILOT_DIR) as temp_name:
        staging = Path(temp_name).resolve()
        if staging.parent != PILOT_DIR or output.parent.resolve() != PILOT_DIR:
            raise RuntimeError("Development artifact paths could not be verified.")
        artifacts = staging / "artifacts"
        artifacts.mkdir()
        with (artifacts / "development_cases.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(scored[0]))
            writer.writeheader()
            writer.writerows(results)
        if failures:
            try:
                save_plot(points, summary, artifacts / "capture_vs_review.png", staging / "matplotlib")
            except (ImportError, OSError, RuntimeError, ValueError) as error:
                summary["figure_error"] = f"{type(error).__name__}: {error}"
        (artifacts / "development_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        artifacts.rename(output)
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

