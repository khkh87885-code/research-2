"""Local, aggregate research exports. No medical images are embedded."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


FIELDS = ["dataset", "split", "patient_key", "image_key", "condition",
          "prompt_strategy", "status", "dice", "iou", "hd95", "assd",
          "precision", "recall", "sensitivity", "boundary_f1", "relative_area_error",
          "metric_status", "empty_prediction", "empty_reference", "distance_unit", "boundary_tolerance",
          "predicted_area_pixels", "reference_area_pixels", "prediction_consistency_dice", "predicted_area_change_pct",
          "output_bytes", "original_bytes", "encode_ms", "decode_ms", "resize_ms", "inference_ms",
          "simulated_upload_ms", "intermediate_bytes", "experiment_id", "selected_condition", "format", "quality",
          "scale", "interpolation", "processing_order", "pixel_equal", "model_commit", "weight_sha256",
          "gpu_peak_bytes", "attempt", "error_type", "sam_predicted_iou", "features",
          "prompt_box_original", "prompt_box_processed"]
METRICS = ["dice", "iou", "hd95", "assd", "precision", "recall", "boundary_f1", "relative_area_error",
           "prediction_consistency_dice", "saving_fraction", "encode_ms", "decode_ms", "inference_ms"]
SUCCESS = {"ok", "success", "completed"}


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def safe_metadata(metadata):
    allowed = {"run_id", "experiment_id", "created_at", "model_name", "model_version",
               "checkpoint_sha256", "prompt_strategy", "python", "torch", "device", "seed", "run_status",
               "purpose", "started_at", "fingerprint_sha256"}
    source = metadata or {}
    result = {k: v for k, v in source.items()
              if k in allowed and isinstance(v, (str, int, float, bool, type(None)))}
    if isinstance(source.get("runtime"), dict):
        result["runtime"] = {k: v for k, v in source["runtime"].items()
                             if k in {"python", "platform", "torch", "numpy", "pillow", "device"}
                             and isinstance(v, (str, int, float, bool, type(None)))}
    if isinstance(source.get("limitations"), list):
        result["limitations"] = [v for v in source["limitations"] if isinstance(v, str)]
    return result


def clean_rows(rows):
    result = []
    for source in rows:
        row = {key: source.get(key) for key in FIELDS}
        numeric_fields = set(METRICS) | {"sensitivity", "output_bytes", "original_bytes", "resize_ms", "simulated_upload_ms",
                                       "intermediate_bytes", "quality", "scale", "gpu_peak_bytes", "attempt", "boundary_tolerance",
                                       "predicted_area_pixels", "reference_area_pixels", "predicted_area_change_pct", "sam_predicted_iou"}
        for key in numeric_fields:
            if row.get(key) is not None and number(row[key]) is None:
                row[key] = None
        original, output = number(row.get("original_bytes")), number(row.get("output_bytes"))
        row["saving_fraction"] = 1 - output / original if original is not None and original > 0 and output is not None and output >= 0 else None
        result.append(row)
    return result


def summarize_local(rows):
    """Describe valid records, with one equal-weight observation per patient."""
    patient_groups = defaultdict(list)
    condition_groups = defaultdict(list)
    for row in rows:
        group = (row.get("dataset"), row.get("split"), row.get("condition"), row.get("prompt_strategy"))
        condition_groups[group].append(row)
        if row.get("patient_key") and str(row.get("status")).lower() in SUCCESS:
            patient_groups[group + (row["patient_key"],)].append(row)
    patients = []
    for key, members in sorted(patient_groups.items(), key=lambda x: str(x[0])):
        entry = dict(zip(["dataset", "split", "condition", "prompt_strategy", "patient_key"], key))
        entry["n_images"] = len(members)
        for metric in METRICS:
            values = [number(row.get(metric)) for row in members]
            values = [value for value in values if value is not None]
            entry[metric] = mean(values) if values else None
        patients.append(entry)
    groups = []
    for key, members in sorted(condition_groups.items(), key=lambda x: str(x[0])):
        entry = dict(zip(["dataset", "split", "condition", "prompt_strategy"], key))
        eligible = [p for p in patients if tuple(p.get(k) for k in ["dataset", "split", "condition", "prompt_strategy"]) == key]
        entry.update(n_records=len(members), n_success=sum(str(r.get("status")).lower() in SUCCESS for r in members),
                     n_failed=sum(str(r.get("status")).lower() not in SUCCESS for r in members),
                     n_patients=len(eligible), n_missing_patient=sum(not r.get("patient_key") for r in members))
        for metric in METRICS:
            values = [number(p.get(metric)) for p in eligible]
            values = [v for v in values if v is not None]
            entry[metric + "_patient_mean"] = mean(values) if values else None
        groups.append(entry)
    return patients, groups


def _cell(value):
    if isinstance(value, (list, dict, tuple)):
        value = json.dumps(value, ensure_ascii=False, allow_nan=False)
    # Prevent spreadsheet formula interpretation of untrusted string fields.
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in fields})


def export_results(rows, output_dir, metadata=None, statistics=None):
    """Write actual observations, patient summaries, tables and aggregate figures."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cleaned = clean_rows(list(rows))
    patients, groups = summarize_local(cleaned)
    patient_fields = ["dataset", "split", "condition", "prompt_strategy", "patient_key", "n_images"] + METRICS
    group_fields = ["dataset", "split", "condition", "prompt_strategy", "n_records", "n_success", "n_failed", "n_patients", "n_missing_patient"] + [m + "_patient_mean" for m in METRICS]
    tables = [("image_results", cleaned, FIELDS + ["saving_fraction"]),
              ("patient_results", patients, patient_fields), ("condition_summary", groups, group_fields)]
    if isinstance(statistics, dict):
        for name, key in [("paired_comparisons", "paired_comparisons"), ("friedman_tests", "friedman")]:
            statistical_rows = statistics.get(key)
            if isinstance(statistical_rows, list) and statistical_rows:
                fields = list(dict.fromkeys(k for row in statistical_rows for k in row))
                tables.append((name, statistical_rows, fields))
    paths = {}
    for name, records, fields in tables:
        path = output / (name + ".csv")
        write_csv(path, records, fields)
        paths[name + "_csv"] = str(path)
    try:
        from openpyxl import Workbook
    except ImportError as exc:
        raise RuntimeError("XLSX export requires openpyxl in the selected research environment.") from exc
    workbook = Workbook()
    workbook.remove(workbook.active)
    for name, records, fields in tables:
        sheet = workbook.create_sheet(name[:31])
        sheet.append(fields)
        for row in records:
            sheet.append([_cell(row.get(key)) for key in fields])
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
    path = output / "research_results.xlsx"
    workbook.save(path)
    paths["xlsx"] = str(path)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4.8, .28 * len(groups))))
    labels = [" / ".join(str(g.get(k) or "unknown") for k in ["dataset", "split", "condition", "prompt_strategy"]) for g in groups]
    for axis, metric, title in [(axes[0], "dice_patient_mean", "Patient mean Dice"),
                                (axes[1], "saving_fraction_patient_mean", "Patient mean file saving")]:
        available = [(i, number(g.get(metric))) for i, g in enumerate(groups) if number(g.get(metric)) is not None]
        if available:
            axis.barh([i for i, value in available], [value for i, value in available], color="#376a8c")
            axis.set_yticks([i for i, value in available], [labels[i] for i, value in available], fontsize=6)
        else:
            axis.text(.5, .5, "No eligible patient observations", ha="center", va="center", transform=axis.transAxes)
            axis.set_yticks([])
        axis.set_xlabel(title)
        axis.grid(axis="x", alpha=.2)
    fig.suptitle("Observed results; failures retained in exported tables")
    fig.tight_layout()
    for extension in ["png", "svg", "pdf"]:
        path = output / ("condition_summary." + extension)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths[extension] = str(path)
    plt.close(fig)
    successful = sum(str(row.get("status")).lower() in SUCCESS for row in cleaned)
    missing = sum(not row.get("patient_key") for row in cleaned)
    report = ["# Observed research export", "", "Records: %d; successful: %d; failed or incomplete: %d." % (len(cleaned), successful, len(cleaned) - successful),
              "Patient identifiers missing: %d records. Missing patient identities are not treated as separate patients." % missing,
              "", "All image records, including failures, are retained. Patient summaries average valid finite values within each dataset, split, condition and prompt strategy. Condition summaries give each eligible patient equal weight.",
              "", "No confidence intervals or hypothesis tests are inferred by this export module. Additional supplied statistical results are stored separately. Upload times, if present elsewhere, are simulations rather than network measurements.",
              "", "Figures contain aggregate results only. This export includes no original image paths or medical images.", "", "## Run metadata", "", "```json", json.dumps(safe_metadata(metadata), ensure_ascii=False, indent=2), "```", ""]
    report_path = output / "RESULTS_STATUS.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    paths["report"] = str(report_path)
    if statistics is not None:
        path = output / "statistics.json"
        path.write_text(json.dumps(statistics, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
        paths["statistics"] = str(path)
    return paths
