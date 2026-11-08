"""Export observed experiment records without inventing absent research results.

All aggregates are descriptive. Repeated images are averaged within the recorded
patient/case and variant first; no independent-patient inference is made here.
"""

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


METRICS = (
    "dice", "iou", "precision", "recall", "sensitivity", "hd95", "assd",
    "boundary_f1", "relative_area_error", "compression_saving",
    "original_bytes", "compressed_bytes", "encoding_ms", "decoding_ms",
    "inference_ms", "simulated_upload_ms",
)
# Boundary distances in millimetres and pixels are different estimands. Include
# units and F1 tolerance in every aggregate key; unknown units stay separate.
CONTEXT = (
    "dataset", "split", "model", "model_version", "model_weight_hash",
    "prompt_strategy", "boundary_unit", "boundary_f1_unit", "boundary_tolerance",
)
SUCCESS = {"completed", "complete", "succeeded", "success", "scored", "done"}


def _finite(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path, default):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else default


def _is_success(row):
    status = str(row.get("status", "")).lower()
    return status in SUCCESS or (not status and _finite(row.get("dice")) is not None)


def _cell(value):
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    # IDs and error text must remain data when opened in a spreadsheet.
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@"):
        return "'" + value
    return value


def _write_csv(path, rows, fallback_fields=("status", "reason")):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or list(fallback_fields)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in fields})


def _group_rows(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key, "")) for key in fields)].append(row)
    output = []
    for key, cases in sorted(groups.items()):
        record = dict(zip(fields, key))
        record["source_row_count"] = len(cases)
        record["image_count"] = len({str(row.get("image_id", "")) for row in cases})
        for metric in METRICS:
            numbers = [value for value in (_finite(row.get(metric)) for row in cases) if value is not None]
            record[metric + "_valid_n"] = len(numbers)
            record[metric + "_mean"] = sum(numbers) / len(numbers) if numbers else None
        output.append(record)
    return output


def _patient_aggregates(rows):
    available = [row for row in rows if row.get("patient_id") not in (None, "")]
    return _group_rows(available, CONTEXT + ("variant", "patient_id"))


def _patient_group_aggregates(patient_rows):
    groups = defaultdict(list)
    fields = CONTEXT + ("variant",)
    for row in patient_rows:
        groups[tuple(row.get(key, "") for key in fields)].append(row)
    result = []
    for key, patients in sorted(groups.items()):
        item = dict(zip(fields, key))
        item["recorded_patient_or_case_count"] = len(patients)
        item["unit_warning"] = "Recorded patient/case IDs; source identity semantics require separate verification"
        for metric in METRICS:
            values = [p[metric + "_mean"] for p in patients if p[metric + "_mean"] is not None]
            item[metric + "_valid_patient_n"] = len(values)
            item[metric + "_patient_mean"] = sum(values) / len(values) if values else None
        result.append(item)
    return result


def _select(rows, fields):
    return [{key: row.get(key) for key in fields} for row in rows]


def _pending(reason):
    return [{"status": "pending", "reason": reason}]


def _tables(rows, successful, patient_groups):
    datasets = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("dataset", ""), row.get("split", ""))].append(row)
    for (dataset, split), group in sorted(groups.items()):
        ids = {str(r["patient_id"]) for r in group if r.get("patient_id") not in (None, "")}
        datasets.append({
            "dataset": dataset, "split": split,
            "observed_images": len({str(r.get("image_id", "")) for r in group}),
            "recorded_patient_or_case_ids": len(ids),
            "identity_semantics": "must be verified against dataset source before patient inference",
            "task_rows": len(group), "successful_task_rows": sum(_is_success(r) for r in group),
        })
    common = CONTEXT + ("patient_id", "image_id", "variant", "format", "scale", "quality")
    adaptive = [r for r in successful if "iacs" in str(r.get("variant", "")).lower()]
    external = [r for r in successful if str(r.get("split", "")).lower() in ("external_test", "external", "external_validation")]
    subgroup = [r for r in successful if r.get("subgroup") or r.get("lesion_size_group")]
    models = {str(r.get("model")) for r in successful if r.get("model")}
    return {
        "Table_1_dataset_characteristics": datasets or _pending("No observed experiment records"),
        "Table_2_file_sizes": _select(successful, common + ("original_bytes", "compressed_bytes", "compression_saving", "encoding_ms", "decoding_ms")) or _pending("Compression experiments not run"),
        "Table_3_segmentation_metrics": patient_groups or _pending("No valid per-case segmentation results"),
        "Table_4_resolution_results": _select(successful, common + ("dice", "hd95", "assd")) or _pending("Resolution experiments not run"),
        "Table_5_IACS_comparison": _select(successful, common + ("dice", "hd95", "compressed_bytes", "iacs_parameter_state", "iacs_evidence_status")) if adaptive else _pending("IACS results unavailable; executable rules do not establish performance"),
        "Table_6_external_dataset": _select(external, common + ("dice", "hd95", "compressed_bytes")) if external else _pending("External dataset compression validation not run; independence not established"),
        "Table_7_subgroup_results": _select(subgroup, common + ("subgroup", "lesion_size_group", "dice", "hd95")) if subgroup else _pending("No prespecified subgroup results available"),
        "Table_8_model_comparison": _select(successful, common + ("dice", "hd95", "compressed_bytes")) if len(models) >= 2 else _pending("Fewer than two fixed segmentation models have been evaluated"),
    }


def _write_workbook(path, rows, patients, groups, tables):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    workbook = Workbook()
    workbook.remove(workbook.active)
    sheets = [("Per_image", rows), ("Per_patient_case", patients), ("Group_descriptive", groups)]
    sheets += [(name.split("_")[0] + "_" + name.split("_")[1], values) for name, values in tables.items()]
    for name, records in sheets:
        sheet = workbook.create_sheet(name[:31])
        fields = list(dict.fromkeys(key for row in records for key in row)) or ["status", "reason"]
        sheet.append(fields)
        for row in records:
            sheet.append([_cell(row.get(key)) for key in fields])
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="173E48")
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for index, field in enumerate(fields, 1):
            sheet.column_dimensions[get_column_letter(index)].width = min(max(len(field) + 2, 14), 42)
    workbook.save(path)


def _figures(directory, successful):
    valid = [row for row in successful if _finite(row.get("dice")) is not None and _finite(row.get("compressed_bytes")) is not None]
    result = {"Figure_4": {"status": "pending", "reason": "No observed size/Dice pairs"}}
    files = []
    if valid:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        figure = Figure(figsize=(8, 5), constrained_layout=True)
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        groups = defaultdict(list)
        for row in valid:
            label = " / ".join(str(row.get(key, "")) for key in ("dataset", "model", "prompt_strategy", "variant"))
            groups[label].append(row)
        for label, cases in sorted(groups.items()):
            axis.scatter([float(r["compressed_bytes"]) / 1024 for r in cases], [float(r["dice"]) for r in cases], s=25, alpha=0.75, label=label)
        axis.set_xlabel("Observed encoded size (KiB)")
        axis.set_ylabel("Dice against reference")
        axis.set_ylim(-0.025, 1.025)
        axis.set_title("Observed experiment records (descriptive)")
        axis.grid(alpha=0.18)
        axis.legend(fontsize=6, loc="best")
        for extension in ("png", "svg", "pdf"):
            filename = directory / ("Figure_4_size_vs_Dice." + extension)
            figure.savefig(filename, dpi=180)
            files.append(filename)
        result["Figure_4"] = {"status": "generated", "observed_pairs": len(valid), "excluded_invalid_pairs": len(successful) - len(valid)}
    # A single actual, traceable image is enough to illustrate an observed mask.
    candidates = [row for row in successful if row.get("prediction_mask") and Path(str(row["prediction_mask"])).is_file()]
    if candidates:
        from PIL import Image, ImageDraw
        row = candidates[0]
        mask_path = Path(str(row["prediction_mask"]))
        with Image.open(mask_path) as image:
            mask = image.convert("RGB")
        canvas = Image.new("RGB", (mask.width, mask.height + 52), "white")
        canvas.paste(mask, (0, 52))
        ImageDraw.Draw(canvas).text((10, 8), "Observed prediction mask", fill="#173E48")
        ImageDraw.Draw(canvas).text((10, 27), str(row.get("image_id", "")) + " | " + str(row.get("variant", "")), fill="#173E48")
        target = directory / "Figure_3_observed_prediction.png"
        canvas.save(target)
        files.append(target)
        result["Figure_3"] = {"status": "generated", "image_id": row.get("image_id"), "variant": row.get("variant"), "source_sha256": _hash(mask_path), "selection": "first successful row with an existing prediction mask; not selected by score"}
    else:
        result["Figure_3"] = {"status": "pending", "reason": "No saved prediction mask available"}
    for number, reason in ((1, "Protocol route figure not generated from experiment data"), (2, "Compression comparison panel pending"), (5, "IACS flowchart not part of an observed-results plot"), (6, "External validation plot requires corresponding observed records"), (7, "Prespecified small-lesion subgroup data unavailable"), (8, "Cross-model plot requires at least two evaluated models")):
        result["Figure_" + str(number)] = {"status": "pending", "reason": reason}
    return files, result


def export_experiment(experiment_dir, output_dir=None):
    """Create an immutable export snapshot of the supplied experiment directory.

    Input rows are never reinterpreted as completed formal validation. Absent
    tables/figures are explicit pending records, and invalid metrics stay missing.
    """
    experiment_dir = Path(experiment_dir).resolve(strict=True)
    result_path = experiment_dir / "results.jsonl"
    config_path = experiment_dir / "config.json"
    summary_path = experiment_dir / "summary.json"
    config = _read_json(config_path, {})
    summary = _read_json(summary_path, {})
    rows = []
    if result_path.exists():
        for line_number, line in enumerate(result_path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("results.jsonl line %d is not a record" % line_number)
            rows.append(row)
    for row in rows:
        if "iacs" in str(row.get("variant", "")).lower():
            row["iacs_parameter_state"] = (config.get("iacs") or {}).get("state", "not_recorded")
            row["iacs_evidence_status"] = "observed technical result; performance validation not established by this export"
    successful = [row for row in rows if _is_success(row)]
    failed = [row for row in rows if not _is_success(row)]
    base = Path(output_dir).resolve() if output_dir is not None else experiment_dir / "exports"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:10]
    directory = base / stamp
    directory.mkdir(parents=True, exist_ok=False)
    patients = _patient_aggregates(successful)
    groups = _patient_group_aggregates(patients)
    tables = _tables(rows, successful, groups)
    files = []
    for name, values in (("per_image", rows), ("per_patient_case", patients), ("group_descriptive", groups)):
        target = directory / (name + ".csv")
        _write_csv(target, values)
        files.append(target)
    for name, values in tables.items():
        target = directory / (name + ".csv")
        _write_csv(target, values)
        files.append(target)
    dependencies = {}
    try:
        target = directory / "experiment_results.xlsx"
        _write_workbook(target, rows, patients, groups, tables)
        files.append(target)
        dependencies["excel"] = "generated"
    except ImportError as error:
        dependencies["excel"] = "unavailable: " + str(error)
    try:
        figure_files, figure_status = _figures(directory, successful)
        files.extend(figure_files)
    except ImportError as error:
        figure_status = {"status": "unavailable", "reason": str(error)}
        dependencies["figures"] = "missing plotting dependency"
    provenance = {
        "source_directory": str(experiment_dir),
        "source_sha256": {path.name: _hash(path) for path in (result_path, config_path, summary_path) if path.exists()},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows), "successful_rows": len(successful), "failed_or_incomplete_rows": len(failed),
        "invalid_dice_rows": sum(_finite(row.get("dice")) is None for row in successful),
        "missing_patient_id_rows": sum(row.get("patient_id") in (None, "") for row in successful),
        "aggregation_method": "image means within dataset/split/model/version/weight/prompt/boundary units/tolerance/variant/recorded patient ID, then unweighted means across recorded patient IDs; descriptive only; pixel and millimetre distances are never pooled",
        "figure_status": figure_status, "dependencies": dependencies,
        "table_status": {key: "pending" if values and values[0].get("status") == "pending" else "observed" for key, values in tables.items()},
        "limitations": ["No independent external patient provenance has been established by this export.", "No inferential claim or clinical noninferiority is generated by descriptive exports.", "IACS candidate or parameter-locked observations do not establish validated performance; efficacy and cross-model generalizability require their own completed validation.", "Past experiments for other study questions are not included automatically."],
    }
    if any("iacs" in str(row.get("variant", "")).lower() for row in successful):
        provenance["table_status"]["Table_5_IACS_comparison"] = "observed_unvalidated_IACS"
    target = directory / "provenance.json"
    target.write_text(json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    files.append(target)
    report = [
        "# Experiment export report", "",
        "This report summarizes recorded observations only. It does not establish clinical validity or independent external validation.", "",
        "- Experiment: `" + experiment_dir.name + "`",
        "- Observed task rows: %d" % len(rows),
        "- Successful task rows: %d" % len(successful),
        "- Failed or incomplete task rows: %d" % len(failed),
        "- Successful rows without finite Dice: %d" % provenance["invalid_dice_rows"],
        "- Rows missing patient/case ID: %d" % provenance["missing_patient_id_rows"],
        "", "## Calculation and provenance", "", provenance["aggregation_method"], "",
        "CSV and workbook values are read directly from the result snapshot; missing or non-finite metrics remain missing. Source hashes are recorded in provenance.json. Compression units and boundary-distance conventions follow the experiment configuration and metrics module.",
        "", "## Table availability", "",
    ]
    report.extend("- %s: %s" % (name, status) for name, status in provenance["table_status"].items())
    report.extend(["", "## Figure availability", "", "```json", json.dumps(figure_status, ensure_ascii=False, indent=2), "```", "", "## Pending evidence and unverified hypotheses", ""])
    report.extend("- " + limitation for limitation in provenance["limitations"])
    report.extend(["", "## Recorded failures", ""])
    if failed:
        report.extend("- %s / %s / %s: %s" % (r.get("dataset", ""), r.get("image_id", ""), r.get("variant", ""), r.get("error", r.get("status", "unknown"))) for r in failed)
    else:
        report.append("No failure rows are present in this snapshot; planned tasks not yet represented in results must be checked in summary.json.")
    target = directory / "experiment_report.md"
    target.write_text("\n".join(report) + "\n", encoding="utf-8")
    files.append(target)
    return {"export_dir": str(directory), "files": [str(path) for path in files], "row_count": len(rows), "successful_rows": len(successful), "failed_rows": len(failed), "figure_status": figure_status, "tables": provenance["table_status"], "dependencies": dependencies}
