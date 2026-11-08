"""Plot measured development images and masks, selected before inspecting scores."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import ndimage
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from research_platform.common import load_config, read_json, write_json, sha256
from research_platform.datasets import load_rgb, load_mask


def main():
    config = load_config()
    root = Path(config["output_root"])
    experiment = root / "experiments" / "EXP_574059083cd74bbc9e57b8fc52988266"
    rows = [json.loads(line) for line in (experiment / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    image_id = sorted(set(r["image_id"] for r in rows))[0]
    variants = ("baseline", "jpeg_q40", "scale_25", "iacs")
    selected = [next(r for r in rows if r["image_id"] == image_id and r["variant"] == v) for v in variants]
    directory = root / "figures" / "observed_panels"
    directory.mkdir(exist_ok=False)
    figure = Figure(figsize=(12, 4.2), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 4)
    for axis, row in zip(axes, selected):
        axis.imshow(load_rgb(row["processed_path"]), interpolation="nearest")
        axis.set_title("%s\n%dx%d | %.1f KiB" % (row["variant"], row["width"], row["height"], row["compressed_bytes"]/1024), fontsize=10)
        axis.axis("off")
    figure.suptitle("Observed compression variants | %s | technical pilot" % image_id, fontsize=13)
    files = []
    for ext in ("png", "svg", "pdf"):
        path = directory / ("Figure_2_observed_compression." + ext)
        figure.savefig(path, dpi=300)
        files.append(path)
    inventory = read_json(root / "dataset_report" / "inventory.json")
    record = next(r for r in inventory["records"] if r["dataset"] == "BUS-BRA" and r["image_id"] == image_id)
    reference = load_mask(record["mask_path"])
    background = load_rgb(record["image_path"])
    figure = Figure(figsize=(12, 4.2), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 4)
    for axis, row in zip(axes, selected):
        prediction = load_mask(row["prediction_mask"])
        ref_edge = reference & ~ndimage.binary_erosion(reference)
        pred_edge = prediction & ~ndimage.binary_erosion(prediction)
        marked = background.copy()
        marked[ref_edge] = (30, 240, 140)
        marked[pred_edge] = (255, 100, 70)
        marked[ref_edge & pred_edge] = (255, 240, 80)
        axis.imshow(marked, interpolation="nearest")
        axis.set_title("%s | Dice %.4f\nHD95 %.2f pixel" % (row["variant"], row["dice"], row["hd95"]), fontsize=10)
        axis.axis("off")
    figure.suptitle("Original-coordinate masks | green: reference, red: SAM, yellow: overlap", fontsize=13)
    for ext in ("png", "svg", "pdf"):
        path = directory / ("Figure_3_observed_segmentation." + ext)
        figure.savefig(path, dpi=300)
        files.append(path)
    write_json(directory / "provenance.json", {"experiment_id": experiment.name, "source_results_sha256": sha256(experiment / "results.jsonl"),
        "selection": "first Image_ID in frozen technical pilot; fixed baseline/jpeg_q40/scale_25/iacs panels, no score-based selection",
        "script_sha256": sha256(__file__), "image_id": image_id,
        "inputs": [{"variant": r["variant"], "processed_sha256": r["processed_sha256"], "prediction_sha256": r["prediction_sha256"]} for r in selected],
        "figures": {p.name: sha256(p) for p in files}, "scientific_status": "illustrative observed technical pilot only"}, exclusive=True)
    print(str(directory))


if __name__ == "__main__":
    main()
