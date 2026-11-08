
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from statistics import NormalDist
from time import perf_counter

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_single_case import (
    CHECKPOINT_SHA256, DATA_ROOT, LOW_QUALITY_DICE, MIB, PILOT_DIR,
    SAM_COMMIT, SAM_SOURCE, dice, prompt_boxes,
)


PILOT_SEED = 20261107
PILOT_SIZE = 40
DEV_MANIFEST_SHA256 = "6db10ea5bae1095dba26ce70b3b376ad1390c94de1645c546320310de8d55274"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def measure_case(row, predictor, device):
    with Image.open(DATA_ROOT / row["image_path"]) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    with Image.open(DATA_ROOT / row["mask_path"]) as mask:
        reference = np.array(mask.convert("1"), dtype=bool)

    width, height = int(row["width"]), int(row["height"])
    if rgb.shape != (height, width, 3) or reference.shape != (height, width) or not reference.any():
        raise ValueError(f"Invalid image and reference pair for {row['image_id']}.")
    base, shifted = prompt_boxes(reference)
    start = perf_counter()
    predictor.set_image(rgb)
    masks, scores, _ = predictor.predict(box=base, multimask_output=False)
    if masks.shape != (1, height, width) or scores.shape != (1,):
        raise RuntimeError(f"Unexpected SAM output shape for {row['image_id']}.")
    prediction = masks[0].astype(bool)
    predicted_iou = float(scores[0])
    consistency = {}
    for direction, box in shifted.items():
        masks, _, _ = predictor.predict(box=box, multimask_output=False)
        if masks.shape != (1, height, width):
            raise RuntimeError(f"Unexpected perturbed mask shape for {row['image_id']}.")
        consistency[f"dice_shift_{direction}"] = dice(prediction, masks[0].astype(bool))
    torch.cuda.synchronize(device)
    elapsed_ms = 1000 * (perf_counter() - start)
    predictor.reset_image()

    reference_dice = dice(prediction, reference)
    risk_a = 1.0 - predicted_iou
    risk_b = 1.0 - sum(consistency.values()) / len(consistency)
    if not np.isfinite([reference_dice, risk_a, risk_b]).all():
        raise RuntimeError(f"A metric is non-finite for {row['image_id']}.")
    return {
        "patient_id": int(row["patient_id"]),
        "image_id": row["image_id"],
        "pathology": row["pathology"],
        "dice_vs_reference": reference_dice,
        "low_quality": int(reference_dice < LOW_QUALITY_DICE),
        "sam_predicted_iou": predicted_iou,
        "risk_A": risk_a,
        "risk_B": risk_b,
        **consistency,
        "initial_mask_empty": int(not prediction.any()),
        "distinct_perturbed_boxes": len({tuple(box) for box in shifted.values()}),
        "inference_ms": elapsed_ms,
    }


def describe(values):
    bounds = np.quantile(values, [0, 0.25, 0.5, 0.75, 1]).tolist()
    return dict(zip(("min", "q1", "median", "q3", "max"), bounds))


def wilson_interval(events, count):
    proportion = events / count
    z = NormalDist().inv_cdf(0.975)
    factor = 1.0 + z ** 2 / count
    center = (proportion + z ** 2 / (2 * count)) / factor
    half_width = z * math.sqrt((proportion * (1 - proportion) + z ** 2 / (4 * count)) / count) / factor
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def save_plot(cases, output, config_directory):
    os.environ["MPLCONFIGDIR"] = str(config_directory)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10.0, 5.2), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2, sharey=True)
    for axis, key, title in zip(axes, ("risk_A", "risk_B"), ("A  Model confidence", "B  Box perturbations")):
        axis.axhline(LOW_QUALITY_DICE, color="#5a6472", linestyle="--", linewidth=1.2)
        for flag, color, label in [
            (0, "#246c83", "Dice at or above 0.70"),
            (1, "#c65321", "Dice below 0.70"),
        ]:
            group = [case for case in cases if case["low_quality"] == flag]
            axis.scatter(
                [case[key] for case in group],
                [case["dice_vs_reference"] for case in group],
                s=35, color=color, alpha=0.85, edgecolors="white", linewidths=0.45,
                label=f"{label} (n={len(group)})",
            )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Risk score")
        axis.set_ylim(0, 1.025)
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Dice against manual contour")
    axes[0].legend(loc="center", bbox_to_anchor=(0.68, 0.24), frameon=False, fontsize=8.5)
    figure.suptitle("BUS-BRA development pilot  |  40 patients", fontsize=16, fontweight="bold")
    figure.savefig(output, dpi=160)


def main():
    output = PILOT_DIR / "pilot_40"
    if output.exists():
        raise FileExistsError(f"Pilot results already exist: {output}")

    manifest = DATA_ROOT / "busbra_splits" / "busbra_development.csv"
    if sha256(manifest) != DEV_MANIFEST_SHA256:
        raise RuntimeError("The BUS-BRA development manifest differs from the frozen split.")
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        development = sorted(csv.DictReader(stream), key=lambda row: int(row["patient_id"]))
    if (
        len(development) != 213
        or len({row["patient_id"] for row in development}) != 213
        or any(row["dataset"] != "BUS-BRA" or row["split"] != "development" for row in development)
    ):
        raise ValueError("Unexpected development patient manifest.")
    sample = sorted(
        random.Random(PILOT_SEED).sample(development, PILOT_SIZE),
        key=lambda row: int(row["patient_id"]),
    )
    print(f"Selected {len(sample)} development patients with seed {PILOT_SEED}.", flush=True)

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
        raise RuntimeError("Checkpoint SHA-256 verification failed.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA could not be accessed by the pilot runtime.")

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

    cases = []
    with torch.inference_mode():
        for row in sample:
            cases.append(measure_case(row, predictor, device))
            if len(cases) % 10 == 0:
                print(f"Completed {len(cases)}/{PILOT_SIZE} patients.", flush=True)

    failures = [case for case in cases if case["low_quality"]]
    review_n = PILOT_SIZE // 5
    risks = {}
    for key in ("risk_A", "risk_B"):
        ordered = sorted(cases, key=lambda case: (-case[key], case["patient_id"]))
        values = [case[key] for case in cases]
        risks[key] = {
            **describe(values),
            "distinct_scores": len(set(values)),
            "zero_scores": values.count(0.0),
            "negative_scores": sum(value < 0 for value in values),
            "tie_at_pilot_top_20pct_boundary": ordered[review_n - 1][key] == ordered[review_n][key],
            "low_quality_in_pilot_top_20pct": sum(case["low_quality"] for case in ordered[:review_n]),
        }
    summary = {
        "dataset": "BUS-BRA",
        "split": "development",
        "selection": "simple random sample without replacement from the sorted frozen development manifest",
        "development_patients": len(development),
        "sample_seed": PILOT_SEED,
        "sampled_patients": PILOT_SIZE,
        "completed_patients": len(cases),
        "pathology_counts": dict(Counter(case["pathology"] for case in cases)),
        # Publication copy: supply a local smoke ID explicitly; no clinical ID is embedded.
        "single_image_smoke_case_in_sample": (
            any(case["image_id"] == os.environ["SAM_SINGLE_IMAGE_SMOKE_ID"] for case in cases)
            if os.environ.get("SAM_SINGLE_IMAGE_SMOKE_ID") else None
        ),
        "dice_cutoff": LOW_QUALITY_DICE,
        "low_quality_count": len(failures),
        "low_quality_fraction": len(failures) / PILOT_SIZE,
        "low_quality_fraction_wilson_95_ci": wilson_interval(len(failures), PILOT_SIZE),
        "low_quality_image_ids": [case["image_id"] for case in failures],
        "dice": describe([case["dice_vs_reference"] for case in cases]),
        "risk_scores": risks,
        "pilot_rank_only_review_count": review_n,
        "initial_empty_masks": sum(case["initial_mask_empty"] for case in cases),
        "cases_with_fewer_than_four_distinct_perturbed_boxes": sum(
            case["distinct_perturbed_boxes"] < 4 for case in cases
        ),
        "inference_ms": describe([case["inference_ms"] for case in cases]),
        "total_inference_ms": sum(case["inference_ms"] for case in cases),
        "model_load_ms": load_ms,
        "gpu": torch.cuda.get_device_name(device),
        "free_gpu_before_model_mib": free_mib,
        "peak_pytorch_allocated_mib": torch.cuda.max_memory_allocated(device) / MIB,
        "peak_pytorch_reserved_mib": torch.cuda.max_memory_reserved(device) / MIB,
        "sam_commit": commit,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "development_manifest_sha256": DEV_MANIFEST_SHA256,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "weight_dtype": str(next(model.parameters()).dtype),
    }

    with tempfile.TemporaryDirectory(prefix=".pilot-40-", dir=PILOT_DIR) as temp_directory:
        staging = Path(temp_directory).resolve()
        if staging.parent != PILOT_DIR or output.parent.resolve() != PILOT_DIR:
            raise RuntimeError("The pilot output paths could not be verified.")
        artifacts = staging / "artifacts"
        artifacts.mkdir()
        with (artifacts / "pilot_40_cases.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
            writer.writeheader()
            writer.writerows(cases)
        (artifacts / "pilot_40_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        save_plot(cases, artifacts / "risk_vs_dice.png", staging / "matplotlib")
        artifacts.rename(output)
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

