"""Verify the locked study inputs, or score both test cohorts with --score."""

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile

import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION
import torch
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_development_pilot import sha256
from run_full_development import run_patient
from run_single_case import (
    BORDER_FRACTION, CHECKPOINT_SHA256, DATA_ROOT, LOW_QUALITY_DICE,
    PILOT_DIR, SAM_COMMIT, SAM_SOURCE, SHIFT_FRACTION,
)


THRESHOLD_LOCK_SHA256 = "67395cc2c1c882ed4aabdd80ac9e614a3e100b42eaae0d0e55e8a24ac256f655"
EXTERNAL_LOCK_SHA256 = "12c58755408f397f67f656b8a3ca89125ad4e0478590ad23bd8f3a03524f127c"
BUSBRA_ARCHIVE_MD5 = "1f8b2be6476d58fc97bfb5e5a1ea9bab"
MANIFEST_FIELDS = (
    "dataset", "split", "patient_id", "image_id", "pathology",
    "image_path", "mask_path", "width", "height",
)
RESULT_FIELDS = (
    "dataset", "split", "patient_id", "image_id", "pathology", "sam_commit",
    "checkpoint_sha256", "status", "retries", "error", "dice_vs_reference",
    "low_quality", "sam_predicted_iou", "risk_A", "risk_B", "alert_A", "alert_B",
    "dice_shift_left", "dice_shift_right", "dice_shift_up", "dice_shift_down",
    "initial_mask_empty", "distinct_perturbed_boxes", "inference_ms",
)
OUTPUT = PILOT_DIR / "formal_validation"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def checked_file(path, expected_sha256):
    require(sha256(path) == expected_sha256, f"SHA-256 mismatch: {path}")
    return path


def read_manifest(path, dataset, split, expected_count, fields=MANIFEST_FIELDS):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == list(fields), f"Unexpected manifest columns: {path}")
        rows = list(reader)

    patients = {int(row["patient_id"]) for row in rows}
    image_ids = {row["image_id"] for row in rows}
    require(
        len(rows) == len(patients) == len(image_ids) == expected_count
        and all(row["dataset"] == dataset and row["split"] == split for row in rows),
        f"Unexpected patient IDs, image IDs or study split: {path}",
    )
    return sorted(rows, key=lambda row: int(row["patient_id"]))


def rgb_over_black(image):
    rgba = image.convert("RGBA")
    black = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    return Image.alpha_composite(black, rgba).convert("RGB")


def verify_png_pairs(rows, data_root, archive=None):
    data_root = data_root.resolve(strict=True)
    changed_by_black = []
    for row in rows:
        image_id = row["image_id"]
        buffers = {}
        for kind in ("image", "mask"):
            key = f"{kind}_path"
            filename = (data_root / row[key]).resolve(strict=True)
            require(data_root in filename.parents, f"Path outside the data root: {filename}")
            content = filename.read_bytes()
            if archive is None:
                same_bytes = hashlib.sha256(content).hexdigest() == row[f"{kind}_sha256"]
            else:
                same_bytes = content == archive.read(row[key])
            require(same_bytes, f"PNG bytes differ from the verified source: {filename}")
            buffers[kind] = BytesIO(content)
            row[key] = str(filename)

        expected_size = (int(row["width"]), int(row["height"]))
        with Image.open(buffers["image"]) as image, Image.open(buffers["mask"]) as mask:
            require(image.format == mask.format == "PNG", f"Invalid PNG pair: {image_id}")
            require(image.size == mask.size == expected_size, f"Wrong PNG dimensions: {image_id}")
            image.load()
            require(mask.convert("1").getbbox() is not None, f"Empty reference mask: {image_id}")
            if archive is None and rgb_over_black(image).tobytes() != image.convert("RGB").tobytes():
                changed_by_black.append(int(row["patient_id"]))
    return changed_by_black


def apply_thresholds(result, thresholds):
    for method, threshold in thresholds.items():
        result[f"alert_{method}"] = int(result[f"risk_{method}"] >= threshold)


def summarize_cohort(results):
    scored = [case for case in results if case["status"] == "scored"]
    low_quality = [case for case in scored if case["low_quality"]]
    n, f = len(scored), len(low_quality)
    methods = {}
    for method in ("A", "B"):
        field = f"alert_{method}"
        reviewed = sum(case[field] for case in scored)
        detected = sum(case[field] for case in low_quality)
        missed = f - detected
        methods[method] = {
            "R": reviewed,
            "TP": detected,
            "FN": missed,
            "low_quality_detection_fraction": detected / f if f else None,
            "review_fraction": reviewed / n if n else None,
            "missed_low_quality_per_100": 100 * missed / n if n else None,
            "review_fraction_among_acceptable": (reviewed - detected) / (n - f) if n > f else None,
        }
    pairs = Counter((case["alert_A"], case["alert_B"]) for case in low_quality)
    return {
        "scheduled_patients": len(results),
        "N": n,
        "F": f,
        "technical_failures": [case["image_id"] for case in results if case["status"] != "scored"],
        "patients_with_a_retry": sum(bool(case.get("retries", 0)) for case in results),
        "methods": methods,
        "paired_low_quality": {
            "both": pairs[(1, 1)], "A_only": pairs[(1, 0)],
            "B_only": pairs[(0, 1)], "neither": pairs[(0, 0)],
        },
    }


def replay_development_results(path, development, lock, thresholds):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        saved = list(csv.DictReader(stream))
    expected = {(int(row["patient_id"]), row["image_id"]) for row in development}
    observed = {(int(row["patient_id"]), row["image_id"]) for row in saved}
    require(len(saved) == len(expected) and observed == expected, "Saved development cases differ from their manifest.")
    for case in saved:
        dice_value = float(case["dice_vs_reference"])
        case["risk_A"] = float(case["risk_A"])
        case["risk_B"] = float(case["risk_B"])
        case["low_quality"] = int(case["low_quality"])
        case["retries"] = int(case["retries"])
        require(
            case["status"] == "scored" and 0 <= dice_value <= 1
            and math.isfinite(case["risk_A"]) and math.isfinite(case["risk_B"])
            and case["low_quality"] == int(dice_value < LOW_QUALITY_DICE),
            f"Invalid saved development result: {case['image_id']}",
        )
        apply_thresholds(case, thresholds)

    replay = summarize_cohort(saved)
    require(replay["F"] == lock["calibration"]["low_quality_count"], "Development low-quality count changed.")
    for method, metrics in replay["methods"].items():
        expected_method = lock["methods"][method]
        require(
            metrics["R"] == expected_method["development_review_count"]
            and metrics["TP"] == expected_method["development_low_quality_detected"],
            f"Applying locked threshold {method} does not reproduce the saved development counts.",
        )
    return replay


def preflight():
    threshold_path = checked_file(PILOT_DIR / "formal_validation_threshold_lock.json", THRESHOLD_LOCK_SHA256)
    external_path = checked_file(PILOT_DIR / "breast_external_test" / "manifest_lock.json", EXTERNAL_LOCK_SHA256)
    lock = json.loads(threshold_path.read_text(encoding="utf-8"))
    external = json.loads(external_path.read_text(encoding="utf-8"))
    thresholds = {method: lock["methods"][method]["threshold"] for method in ("A", "B")}
    rules = external["scoring_rules"]
    require(
        lock["status"] == external["status"] == "locked"
        and thresholds == rules["thresholds"] and all(math.isfinite(value) for value in thresholds.values())
        and lock["alert_operator"] == rules["alert_operator"] == ">="
        and lock["quality_reference"] == rules["quality_reference"]
        and lock["quality_reference"]["low_quality_operator"] == "<"
        and lock["quality_reference"]["cutoff"] == LOW_QUALITY_DICE,
        "The formal and external records disagree on scoring rules.",
    )
    model = lock["model"]
    require(
        model == rules["model"] and model["source_commit"] == SAM_COMMIT
        and model["weight_sha256"] == CHECKPOINT_SHA256
        and model["prompt_box_expansion_per_side"] == BORDER_FRACTION
        and model["prompt_box_shift_fraction"] == SHIFT_FRACTION,
        "The SAM settings differ from the development lock.",
    )
    evidence = {
        name: checked_file(Path(source["path"]), source["sha256"])
        for name, source in lock["evidence"].items()
    }
    for source in external["sources"].values():
        checked_file(external_path.parent / source["path"], source["sha256"])

    development = read_manifest(
        evidence["development_manifest"], "BUS-BRA", "development", lock["calibration"]["patients"],
    )
    internal = read_manifest(
        evidence["internal_holdout_manifest"], "BUS-BRA", "internal_holdout",
        lock["planned_tests"]["internal"]["manifest_patients"],
    )
    dev_ids = {int(row["patient_id"]) for row in development}
    require(dev_ids.isdisjoint(int(row["patient_id"]) for row in internal), "A development patient is in the holdout.")
    replay = replay_development_results(evidence["development_case_results"], development, lock, thresholds)

    archive_path = DATA_ROOT / "BUSBRA.zip"
    md5 = hashlib.md5()
    with archive_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            md5.update(block)
    require(md5.hexdigest() == BUSBRA_ARCHIVE_MD5, "BUS-BRA archive MD5 mismatch.")
    with ZipFile(archive_path) as archive:
        verify_png_pairs(internal, DATA_ROOT, archive)

    manifest = external["manifest"]
    external_csv = checked_file(external_path.parent / manifest["path"], manifest["sha256"])
    breast = read_manifest(
        external_csv, "BrEaST", "external_test", manifest["eligible_patients"],
        MANIFEST_FIELDS + ("image_sha256", "mask_sha256"),
    )
    require(
        Counter(row["pathology"] for row in breast) == manifest["pathology_counts"]
        and external["dataset"]["doi"] == lock["planned_tests"]["external"]["dataset_doi"],
        "External cohort differs from the locked data identity.",
    )
    changed = verify_png_pairs(breast, external_path.parent / manifest["image_and_mask_path_base"])
    alpha_rule = external["preprocessing"]["original_image"]
    require(
        alpha_rule["background_rgba"] == [0, 0, 0, 255]
        and changed == alpha_rule["case_ids_with_changed_rgb_versus_ignoring_alpha"],
        "Black-background decoding differs from the external lock.",
    )

    git = ["git.exe", "--no-optional-locks", "-C", str(SAM_SOURCE)]
    commit = subprocess.check_output(git + ["rev-parse", "--verify", "HEAD"], text=True).strip()
    changes = subprocess.check_output(git + ["status", "--porcelain"], text=True).strip()
    require(commit == SAM_COMMIT and not changes, "Local SAM source differs from the verified commit.")
    require(
        sys.version.split()[0] == model["python"] and torch.__version__ == model["torch"]
        and torchvision.__version__ == "0.19.1+cu118" and torch.cuda.is_available(),
        "The development Python, PyTorch, torchvision and CUDA configuration is required.",
    )

    info = {
        "status": "preflight_verified",
        "patients": {"internal_holdout": len(internal), "external_test": len(breast)},
        "internal_holdout_pathology": dict(Counter(row["pathology"] for row in internal)),
        "external_test_pathology": dict(Counter(row["pathology"] for row in breast)),
        "development_results_replayed": replay,
        "thresholds": thresholds,
        "alert_operator": lock["alert_operator"],
        "quality_reference": lock["quality_reference"],
        "external_black_background_case_ids": changed,
        "threshold_lock_sha256": THRESHOLD_LOCK_SHA256,
        "external_manifest_lock_sha256": EXTERNAL_LOCK_SHA256,
        "internal_holdout_manifest_sha256": lock["evidence"]["internal_holdout_manifest"]["sha256"],
        "external_manifest_sha256": manifest["sha256"],
        "internal_archive_md5": BUSBRA_ARCHIVE_MD5,
        "protocol_sha256": lock["evidence"]["protocol"]["sha256"],
        "sam_commit": commit,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "runner_sha256": sha256(Path(__file__)),
        "runtime": {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "torchvision": torchvision.__version__, "pillow": PILLOW_VERSION,
            "numpy": np.__version__, "gpu": torch.cuda.get_device_name(0),
        },
    }
    cohorts = {"internal_holdout": internal, "external_test": breast}
    return cohorts, evidence["sam_weights"], info


def score(cohorts, checkpoint, preflight_info):
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    sys.path.insert(0, str(SAM_SOURCE))
    from segment_anything import SamPredictor, sam_model_registry

    with tempfile.TemporaryDirectory(prefix=".formal-validation-", dir=PILOT_DIR) as directory:
        staging = Path(directory).resolve(strict=True)
        require(staging.parent == OUTPUT.parent.resolve(strict=True) == PILOT_DIR, "Unsafe validation staging location.")
        normalized = staging / "external_rgb"
        normalized.mkdir()
        changed = set(preflight_info["external_black_background_case_ids"])
        for row in cohorts["external_test"]:
            if int(row["patient_id"]) in changed:
                rgb_path = normalized / f"case{int(row['patient_id']):03d}.png"
                with Image.open(row["image_path"]) as image:
                    rgb_over_black(image).save(rgb_path, format="PNG")
                row["image_path"] = str(rgb_path)

        device = torch.device("cuda:0")
        model = sam_model_registry["vit_b"]()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
        model.to(device).eval()
        require(next(model.parameters()).dtype == torch.float32, "SAM weights must be float32.")
        predictor = SamPredictor(model)
        results = {}
        with torch.inference_mode():
            for split, rows in cohorts.items():
                group = []
                for index, row in enumerate(rows, 1):
                    result = run_patient(row, predictor, device)
                    if result["status"] == "scored":
                        apply_thresholds(result, preflight_info["thresholds"])
                    result.update(
                        dataset=row["dataset"], split=split, sam_commit=SAM_COMMIT,
                        checkpoint_sha256=CHECKPOINT_SHA256,
                    )
                    group.append(result)
                    if index % 25 == 0 or index == len(rows):
                        print(f"{split}: processed {index}/{len(rows)} patients.", flush=True)
                results[split] = group

        package = staging / "results"
        package.mkdir()
        csv_path = package / "validation_cases.csv"
        with csv_path.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            for group in results.values():
                writer.writerows(group)
        summaries = {split: summarize_cohort(group) for split, group in results.items()}
        has_failures = any(summary["technical_failures"] for summary in summaries.values())
        summary = {
            "status": "scoring_completed_with_failures" if has_failures else "scoring_completed",
            "started_at": started,
            "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "input_verification": preflight_info,
            "cohorts": summaries,
            "validation_cases_sha256": sha256(csv_path),
        }
        (package / "validation_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        require(
            package.resolve(strict=True).parent == staging and not OUTPUT.exists(),
            "Validation output already exists or the results directory changed.",
        )
        package.rename(OUTPUT)
    print(json.dumps(summary, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", action="store_true", help="Run model inference after successful preflight.")
    args = parser.parse_args()
    if args.score and OUTPUT.exists():
        raise FileExistsError(f"Formal validation results already exist: {OUTPUT}")
    cohorts, checkpoint, info = preflight()
    print(json.dumps(info, allow_nan=False), flush=True)
    if args.score:
        score(cohorts, checkpoint, info)


if __name__ == "__main__":
    main()
