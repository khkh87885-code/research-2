
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import torch
import torchvision


PILOT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path("D:/codex\u8bba\u6587")
SAM_SOURCE = PILOT_DIR.parent / "segment-anything"
SAM_COMMIT = "dca509fe793f601edb92606367a655c15ac00fdf"
CHECKPOINT_SHA256 = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
BORDER_FRACTION = 0.10
SHIFT_FRACTION = 0.05
LOW_QUALITY_DICE = 0.70
MIB = 1024 ** 2


def dice(first, second):
    total = int(first.sum()) + int(second.sum())
    if total == 0:
        return 1.0
    return 2.0 * np.count_nonzero(first & second) / total


def prompt_boxes(reference):
    height, width = reference.shape
    rows, columns = np.nonzero(reference)
    left, top = columns.min(), rows.min()
    right, bottom = columns.max() + 1, rows.max() + 1
    pad_x = BORDER_FRACTION * (right - left)
    pad_y = BORDER_FRACTION * (bottom - top)
    limits = np.array([width, height, width, height], dtype=np.float32)
    base = np.clip(
        np.array([left - pad_x, top - pad_y, right + pad_x, bottom + pad_y], dtype=np.float32),
        0,
        limits,
    )
    shift_x = SHIFT_FRACTION * float(base[2] - base[0])
    shift_y = SHIFT_FRACTION * float(base[3] - base[1])
    shifts = {
        "left": [-shift_x, 0, -shift_x, 0],
        "right": [shift_x, 0, shift_x, 0],
        "up": [0, -shift_y, 0, -shift_y],
        "down": [0, shift_y, 0, shift_y],
    }
    shifted = {
        name: np.clip(base + np.asarray(offset, dtype=np.float32), 0, limits)
        for name, offset in shifts.items()
    }
    return base, shifted


def contour(mask):
    layer = Image.fromarray(mask.astype(np.uint8) * 255)
    eroded = np.asarray(layer.filter(ImageFilter.MinFilter(3))) != 0
    return mask & ~eroded


def comparison_image(rgb, reference, prediction, base, case_id, result):
    colors = {
        "reference": (88, 233, 161),
        "sam": (255, 128, 92),
        "overlap": (255, 228, 122),
        "box": (99, 204, 255),
    }
    ref_edge = contour(reference)
    pred_edge = contour(prediction)
    marked = rgb.copy()
    marked[ref_edge] = colors["reference"]
    marked[pred_edge] = colors["sam"]
    marked[ref_edge & pred_edge] = colors["overlap"]

    height, width = reference.shape
    scale = max(1.0, 360 / width)
    panel_size = (round(width * scale), round(height * scale))
    first = Image.fromarray(rgb).resize(panel_size, Image.Resampling.BILINEAR)
    second = Image.fromarray(marked).resize(panel_size, Image.Resampling.NEAREST)
    box_pixels = tuple(
        min(round(value * scale), panel_size[index % 2] - 1)
        for index, value in enumerate(base)
    )
    ImageDraw.Draw(first).rectangle(box_pixels, outline=colors["box"], width=2)

    margin, gap, top_space, bottom_space = 18, 16, 92, 44
    canvas = Image.new(
        "RGB",
        (2 * margin + 2 * panel_size[0] + gap, top_space + panel_size[1] + bottom_space),
        (18, 26, 38),
    )
    second_left = margin + panel_size[0] + gap
    canvas.paste(first, (margin, top_space))
    canvas.paste(second, (second_left, top_space))
    draw = ImageDraw.Draw(canvas)
    fonts = Path("C:/Windows/Fonts")
    heading = ImageFont.truetype(str(fonts / "arialbd.ttf"), 22)
    body = ImageFont.truetype(str(fonts / "arial.ttf"), 15)
    draw.text((margin, 13), f"BUS-BRA single-image pilot  |  {case_id}", (245, 248, 252), font=heading)
    draw.text(
        (margin, 45),
        f"Dice {result['dice_vs_reference']:.4f}    Risk A {result['risk_A']:.4f}    Risk B {result['risk_B']:.4f}",
        (185, 200, 218),
        font=body,
    )
    draw.text((margin, 69), "Original and box prompt", colors["box"], font=body)
    draw.text((second_left, 69), "Manual contour and SAM prediction", (245, 248, 252), font=body)
    legend_y = top_space + panel_size[1] + 12
    for x, key, label in [
        (margin, "reference", "Manual contour"),
        (margin + 175, "sam", "SAM contour"),
        (margin + 350, "overlap", "Coincident edges"),
    ]:
        draw.rectangle((x, legend_y + 4, x + 12, legend_y + 15), fill=colors[key])
        draw.text((x + 19, legend_y), label, (225, 234, 244), font=body)
    return canvas


def main():
    output = PILOT_DIR / "results"
    if output.exists():
        raise FileExistsError(f"Pilot results already exist: {output}")

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
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 ** 2), b""):
            digest.update(chunk)
    if digest.hexdigest() != CHECKPOINT_SHA256:
        raise RuntimeError("Checkpoint SHA-256 verification failed.")

    manifest = DATA_ROOT / "busbra_splits" / "busbra_development.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    if row["dataset"] != "BUS-BRA" or row["split"] != "development":
        raise ValueError("The selected record must come from the BUS-BRA development set.")
    with Image.open(DATA_ROOT / row["image_path"]) as image, Image.open(DATA_ROOT / row["mask_path"]) as mask:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        reference = np.array(mask.convert("1"), dtype=bool)
    height, width = reference.shape
    if rgb.shape != (height, width, 3) or not reference.any():
        raise ValueError("The selected image and reference mask are incompatible.")
    if (width, height) != (int(row["width"]), int(row["height"])):
        raise ValueError("The selected image does not match the manifest dimensions.")
    base, shifted = prompt_boxes(reference)

    if not torch.cuda.is_available():
        raise RuntimeError("The verified PyTorch environment could not access CUDA.")
    device = torch.device("cuda:0")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    free_at_start, total_memory = torch.cuda.mem_get_info(device)

    start = perf_counter()
    model = sam_model_registry["vit_b"]()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval()
    predictor = SamPredictor(model)
    torch.cuda.synchronize(device)
    load_ms = 1000.0 * (perf_counter() - start)
    print(f"Loaded SAM ViT-B on {torch.cuda.get_device_name(device)}.", flush=True)

    with torch.inference_mode():
        start = perf_counter()
        predictor.set_image(rgb)
        torch.cuda.synchronize(device)
        encode_ms = 1000.0 * (perf_counter() - start)

        start = perf_counter()
        masks, scores, _ = predictor.predict(box=base, multimask_output=False)
        torch.cuda.synchronize(device)
        initial_ms = 1000.0 * (perf_counter() - start)
        if masks.shape != (1, height, width) or scores.shape != (1,):
            raise RuntimeError("Unexpected initial SAM mask or confidence shape.")
        prediction = masks[0].astype(bool)
        predicted_iou = float(scores[0])

        consistency = {}
        start = perf_counter()
        for direction, box in shifted.items():
            masks, _, _ = predictor.predict(box=box, multimask_output=False)
            if masks.shape != (1, height, width):
                raise RuntimeError(f"Unexpected mask shape for {direction} shift.")
            consistency[direction] = dice(prediction, masks[0].astype(bool))
        torch.cuda.synchronize(device)
        perturb_ms = 1000.0 * (perf_counter() - start)

    reference_dice = dice(reference, prediction)
    risk_a = 1.0 - predicted_iou
    risk_b = 1.0 - sum(consistency.values()) / len(consistency)
    if not np.isfinite([predicted_iou, reference_dice, risk_a, risk_b]).all():
        raise RuntimeError("A pilot metric is non-finite.")

    result = {
        "dataset": row["dataset"],
        "split": row["split"],
        "selection": "first row in frozen development manifest",
        "patient_id": int(row["patient_id"]),
        "image_id": row["image_id"],
        "source_image": row["image_path"],
        "reference_mask": row["mask_path"],
        "image_width": width,
        "image_height": height,
        "sam_commit": commit,
        "checkpoint_sha256": digest.hexdigest(),
        "base_box_xyxy": base.tolist(),
        "perturbed_boxes_xyxy": {name: box.tolist() for name, box in shifted.items()},
        "border_fraction_per_side": BORDER_FRACTION,
        "shift_fraction_of_initial_box": SHIFT_FRACTION,
        "reference_mask_pixels": int(reference.sum()),
        "predicted_mask_pixels": int(prediction.sum()),
        "dice_vs_reference": reference_dice,
        "low_quality_dice_cutoff": LOW_QUALITY_DICE,
        "low_quality_by_study_definition": bool(reference_dice < LOW_QUALITY_DICE),
        "sam_predicted_iou": predicted_iou,
        "risk_A": risk_a,
        "initial_vs_perturbed_dice": consistency,
        "risk_B": risk_b,
        "timing_ms": {
            "model_build_load_and_to_gpu": load_ms,
            "image_encoding": encode_ms,
            "initial_prompt": initial_ms,
            "four_perturbed_prompts": perturb_ms,
            "inference_total": encode_ms + initial_ms + perturb_ms,
        },
        "gpu": {
            "name": torch.cuda.get_device_name(device),
            "total_mib": total_memory / MIB,
            "free_before_model_mib": free_at_start / MIB,
            "peak_pytorch_allocated_mib": torch.cuda.max_memory_allocated(device) / MIB,
            "peak_pytorch_reserved_mib": torch.cuda.max_memory_reserved(device) / MIB,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda,
            "weight_dtype": str(next(model.parameters()).dtype),
        },
    }
    figure = comparison_image(rgb, reference, prediction, base, row["image_id"], result)
    output.mkdir()
    overlay = output / "contour_comparison.png"
    metrics = output / "single_case_result.json"
    try:
        figure.save(overlay)
        metrics.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except BaseException:
        overlay.unlink(missing_ok=True)
        metrics.unlink(missing_ok=True)
        output.rmdir()
        raise
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

