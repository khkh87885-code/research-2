"""Verify real local artifacts; unit-test fixtures are not read by this script."""
import csv
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from research_platform.common import ROOT, load_config, read_json, write_json, sha256, now
from research_platform.engine import inventory, run_experiment


def main():
    config = load_config()
    root = Path(config["output_root"])
    report = inventory(config)
    changed = []
    checked = 0
    for row in report["records"]:
        for path_key, hash_key in (("image_path", "image_sha256"), ("mask_path", "mask_sha256")):
            if row.get(path_key) and row.get(hash_key):
                checked += 1
                if sha256(row[path_key]) != row[hash_key]:
                    changed.append({"dataset": row["dataset"], "image_id": row["image_id"], "role": path_key})
    backups = read_json(ROOT / "backups" / "20261108_original" / "backup_manifest.json")
    old_changes = [r["source"] for r in backups if sha256(r["source"]) != r["sha256"] or sha256(r["backup"]) != r["sha256"]]
    experiments = []
    for directory in sorted((root / "experiments").glob("EXP_*")):
        frozen = read_json(directory / "config.json")
        rows = [json.loads(line) for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines() if line]
        completed = [r for r in rows if r["status"] == "completed"]
        mask_errors = []
        for row in completed:
            with Image.open(row["prediction_mask"]) as image:
                if image.size != (row["original_width"], row["original_height"]):
                    mask_errors.append(row["image_id"] + ":" + row["variant"])
            if sha256(row["prediction_mask"]) != row["prediction_sha256"]:
                mask_errors.append(row["image_id"] + ":" + row["variant"] + ":hash")
        resume = None
        if len(completed) == len(rows):
            before = sha256(directory / "results.jsonl")
            with sqlite3.connect(str(directory / "tasks.sqlite3")) as db:
                before_events = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            run_experiment(config, directory.name)
            with sqlite3.connect(str(directory / "tasks.sqlite3")) as db:
                after_events = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            resume = {"results_unchanged": before == sha256(directory / "results.jsonl"), "events_unchanged": before_events == after_events}
        groups = {}
        for row in completed:
            groups.setdefault(row["variant"], []).append(row)
        group_summary = {variant: {"n_images": len(group), "mean_dice": sum(r["dice"] for r in group if r["dice"] is not None) / len(group),
                        "mean_saving_pct": sum(r["compression_saving"] for r in group) / len(group),
                        "empty_predictions": sum(r["empty_prediction"] for r in group)} for variant, group in groups.items()}
        png_rows = [r for r in completed if r["variant"] == "png"]
        experiments.append({"experiment_id": directory.name, "device": frozen["device"], "prompt_strategy": frozen["prompt_strategy"],
            "dataset": frozen["dataset"], "split": frozen["split"], "scheduled_tasks": len(rows), "completed_tasks": len(completed),
            "failed_tasks": sum(r["status"] == "failed" for r in rows), "scheduled_images": len(frozen["records"]),
            "mask_geometry_or_hash_errors": mask_errors, "completed_resume": resume, "group_descriptive": group_summary,
            "png_pixels_all_equal": all(r["decoded_pixels_equal_original"] for r in png_rows) if png_rows else None,
            "error_types": sorted(set(r.get("error_type", "") for r in rows if r["status"] == "failed"))})
    result = {"checked_at": now(), "source_files_checked": checked, "changed_source_files": changed,
        "original_backups_verified": len(backups), "changed_original_scripts_or_backups": old_changes,
        "model_weight_sha256": sha256(config["checkpoint"]), "model_hash_matches": sha256(config["checkpoint"]) == config["checkpoint_sha256"],
        "dataset_summary": report["summary"], "experiments": experiments,
        "scientific_status": "technical pilots only; no formal noninferiority or external compression validation"}
    output = root / "logs" / "real_verification.json"
    write_json(output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("dataset_summary", "experiments")}, ensure_ascii=True, indent=2))
    print(json.dumps([{k:v for k,v in e.items() if k != "group_descriptive"} for e in experiments], indent=2))
    if changed or old_changes or not result["model_hash_matches"] or any(e["mask_geometry_or_hash_errors"] for e in experiments):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
