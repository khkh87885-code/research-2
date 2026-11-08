"""Reuse the audited compression/SAM/metrics; append-only complete 213x7 matrix."""
import importlib.metadata
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from compression_study.datasets import read_records, pseudonym_salt, load_reference, sha256, inside
from compression_study.imaging import load_rgb, transform_image, scaled_box, restore_mask
from compression_study.metrics import evaluate_masks
from compression_study.model import ExistingSAM, make_prompt
from compression_study.engine import read_results, code_fingerprint as legacy_fingerprint
from .common import read_json, write_json, digest, append_event, now, safe_run, require_development

CONDITIONS = [dict(id="baseline", format="baseline"),
              *[dict(id="jpeg_q%d" % q, format="jpeg", quality=q, scale=1.0,
                     interpolation="lanczos", order="resize_encode") for q in (95, 85, 75, 60, 40)],
              dict(id="png_l6", format="png", compress_level=6, lossless=True, scale=1.0)]


def collection_code():
    here = Path(__file__).resolve().parent
    return dict(new={p.name: sha256(p) for p in (here / "collector.py", here / "common.py")},
                reused=legacy_fingerprint())


def latest_rows(root):
    rows = []
    for task in sorted((Path(root) / "compression" / "tasks").glob("*/*")):
        files = sorted(task.glob("attempt_*/result.json"))
        if files:
            rows.append(read_json(files[-1]))
    return rows


def validate_artifacts(root, row):
    for artifact, checksum in (("prediction_mask", "prediction_sha256"), ("processed_image", "processed_sha256")):
        if artifact in row:
            path = (Path(root) / row[artifact]).resolve()
            if not inside(path, root) or not path.is_file() or sha256(path) != row[checksum]:
                raise ValueError("Saved task artifact is missing or changed")


def reusable_rows(config, sources):
    reusable = {}
    for old_root in config.get("reuse_runs", []):
        old_root = Path(old_root).resolve(strict=True)
        meta = read_json(old_root / "run.json")
        original = meta["provenance"]
        if original["code"] != legacy_fingerprint():
            raise ValueError("Reuse refused: original compression code snapshot differs")
        if original["checkpoint_sha256"] != config["model"]["sha256"]:
            raise ValueError("Reuse refused: weights differ")
        for row in read_results(old_root):
            if row.get("dataset") != "BUS-BRA" or row.get("split") != "development":
                continue
            if row.get("prompt_strategy") != "mechanism_gt_box" or row.get("condition") not in {s["id"] for s in CONDITIONS}:
                continue
            if row.get("status") != "ok" or row["image_key"] not in sources:
                continue
            if original["source_hashes"][row["image_key"]] != sources[row["image_key"]]:
                raise ValueError("Reuse refused: image or reference differs")
            validate_artifacts(old_root, row)
            reusable[(row["image_key"], row["condition"])] = (old_root, row, meta["fingerprint_sha256"])
    return reusable


def copy_artifact(source, destination):
    with Path(source).open("rb") as incoming, Path(destination).open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)


def collect(config, resume=False, retry_failed=False):
    require_development(config)
    root = safe_run(config)
    root.mkdir(parents=True, exist_ok=True)
    records, cohorts = read_records(config, pseudonym_salt(config["project_root"]))
    if config.get("technical_limit"):
        records = records[:int(config["technical_limit"])]
    if config.get("conditions"):
        wanted = config["conditions"]
        specs = [s for s in CONDITIONS if s["id"] in wanted]
        if len(specs) != len(wanted) or "baseline" not in wanted:
            raise ValueError("Technical conditions must be a subset containing baseline")
    else:
        specs = CONDITIONS
    if not isinstance(config.get("max_attempts", 2), int) or config.get("max_attempts", 2) < 1:
        raise ValueError("max_attempts must be positive")
    sources = {r["image_key"]: dict(image=sha256(r["image_path"]), mask=sha256(r["mask_path"])) for r in records}
    frozen = dict(config=config, code=collection_code(), sources=sources, conditions=specs,
                  manifest_sha256=[c["manifest_sha256"] for c in cohorts],
                  checkpoint_sha256=sha256(config["model"]["checkpoint"]))
    checksum = digest(frozen)
    lock = root / "configs" / "collection_lock.json"
    if lock.exists():
        if not resume or read_json(lock)["sha256"] != checksum:
            raise ValueError("Existing collection requires identical source/config/code and --resume")
    else:
        if resume:
            raise ValueError("No collection lock to resume")
        write_json(lock, dict(sha256=checksum, created_at=now(), provenance=frozen,
                   analysis_unit="Case", patient_semantics_verified=False))
        append_event(root, "collection_started", cases=len({r["patient_key"] for r in records}), images=len(records), tasks=len(records) * len(specs))
    cached = reusable_rows(config, sources)
    model, done = None, 0
    if config.get("device", config["model"].get("device", "cuda")) == "cpu":
        torch.set_num_threads(int(config.get("cpu_threads", 4)))
    for record in records:
        image_key = record["image_key"]
        original, reference, box, source_error = None, None, None, None
        baseline_row, baseline_mask = None, None
        try:
            original = load_rgb(record["image_path"])
            reference = load_reference(record)
            if original.shape[:2] != reference.shape:
                raise ValueError("Source geometry mismatch")
            box = make_prompt(original, reference, "mechanism_gt_box")  # once per original
        except Exception as exc:
            source_error = type(exc).__name__
        for spec in specs:
            condition = spec["id"]
            task = root / "compression" / "tasks" / image_key / condition
            task.mkdir(parents=True, exist_ok=True)
            saved = sorted(task.glob("attempt_*/result.json"))
            previous = read_json(saved[-1]) if saved else None
            if previous and previous["status"] == "ok":
                validate_artifacts(root, previous)
                if condition == "baseline":
                    baseline_row = previous
                    with Image.open(root / previous["prediction_mask"]) as image:
                        baseline_mask = np.asarray(image) != 0
                done += 1
                continue
            if previous and not retry_failed:
                done += 1
                continue
            count = len(list(task.glob("attempt_*")))
            for retry in range(config.get("max_attempts", 2)):
                attempt = task / ("attempt_%04d" % (count + retry + 1))
                attempt.mkdir(exist_ok=False)
                row = dict(dataset="BUS-BRA", split="development", case_key=record["patient_key"],
                    image_key=image_key, condition=condition, prompt_strategy="mechanism_gt_box",
                    status="error", error_type=None, stage="source_validation", attempt=count + retry + 1,
                    original_bytes=Path(record["image_path"]).stat().st_size,
                    format=spec["format"], quality=spec.get("quality"), scale=1.0,
                    distance_unit="pixel", weight_sha256=config["model"]["sha256"],
                    model_commit=config["model"]["commit"], input_sha256=sources[image_key]["image"],
                    reference_sha256=sources[image_key]["mask"], source_spacing=None,
                    inference_reused=False, origin="new_inference")
                try:
                    if source_error:
                        raise ValueError("Source validation failed")
                    if (image_key, condition) in cached:
                        old_root, old, old_fingerprint = cached[(image_key, condition)]
                        # Reuse only after checking source, code, weight, mask and payload hashes.
                        keep = {k: v for k, v in old.items() if k not in ("patient_key", "experiment_id", "features", "prediction_mask", "processed_image", "attempt")}
                        row.update(keep)
                        row.update(case_key=record["patient_key"], attempt=count + retry + 1,
                                   origin="reused_verified_smoke_v2", inference_reused=True,
                                   source_run_fingerprint=old_fingerprint)
                        for artifact in ("prediction_mask", "processed_image"):
                            if artifact in old:
                                src = old_root / old[artifact]
                                destination = attempt / src.name
                                copy_artifact(src, destination)
                                row[artifact] = str(destination.relative_to(root))
                    else:
                        row["stage"] = "compression"
                        if condition == "baseline":
                            pixels = original
                            transform = dict(output_bytes=row["original_bytes"], width=original.shape[1], height=original.shape[0],
                                             encode_ms=0.0, decode_ms=0.0, resize_ms=0.0, pixel_equal=True,
                                             codec="original", final_format="original", processing_order="original")
                        else:
                            destination = attempt / ("processed.jpg" if spec["format"] == "jpeg" else "processed.png")
                            pixels, transform = transform_image(original, spec, destination)
                            row.update(processed_image=str(destination.relative_to(root)), processed_sha256=sha256(destination))
                        row.update(transform)
                        mapped_box = scaled_box(box, (original.shape[1], original.shape[0]), (pixels.shape[1], pixels.shape[0]))
                        row.update(prompt_box_original=box.tolist(), prompt_box_processed=mapped_box.tolist())
                        if condition != "baseline" and baseline_mask is not None and np.array_equal(pixels, original):
                            prediction = baseline_mask.copy()
                            row.update(origin="lossless_pixel_identity_reuse", inference_reused=True,
                                       inference_ms=0.0, baseline_inference_ms=baseline_row.get("inference_ms"),
                                       sam_predicted_iou=baseline_row.get("sam_predicted_iou"), gpu_peak_bytes=None)
                        else:
                            row["stage"] = "model_inference"
                            if model is None:
                                torch.manual_seed(config.get("seed", 20261108))
                                torch.backends.cudnn.benchmark = False
                                model_config = dict(config["model"], device=config.get("device", config["model"].get("device", "cuda")))
                                model = ExistingSAM(model_config)
                                write_json(root / "logs" / ("model_session_%s.json" % len(list((root / "logs").glob("model_session_*.json")))), model.info)
                            prediction, timing = model.predict(pixels, mapped_box)
                            row.update(timing)
                            prediction = restore_mask(prediction, (original.shape[1], original.shape[0]))
                        row["stage"] = "evaluation"
                        metrics = evaluate_masks(prediction, reference)
                        row.update(metrics)
                        with (attempt / "prediction.png").open("xb") as stream:
                            Image.fromarray(prediction.astype(np.uint8) * 255).save(stream, format="PNG")
                        row.update(prediction_mask=str((attempt / "prediction.png").relative_to(root)),
                                   prediction_sha256=sha256(attempt / "prediction.png"))
                        if baseline_mask is not None:
                            from run_single_case import dice as legacy_consistency_dice
                            row["prediction_consistency_dice"] = legacy_consistency_dice(prediction, baseline_mask)
                            area = int(baseline_mask.sum())
                            row["delta_area"] = int(prediction.sum()) / area - 1 if area else None
                        row["status"] = "ok"
                    row.update(status="ok", error_type=None, stage="complete")
                    row["compression_ratio"] = row["output_bytes"] / row["original_bytes"]
                    row["saving_pct"] = 100 * (1 - row["compression_ratio"])
                    if condition == "baseline":
                        with Image.open(root / row["prediction_mask"]) as image:
                            baseline_mask = np.asarray(image) != 0
                        baseline_row = row
                except Exception as exc:
                    row.update(status="error", error_type=type(exc).__name__, input_error_type=source_error)
                write_json(attempt / "result.json", row)
                append_event(root, "task_attempt", image_key=image_key, condition=condition,
                             attempt=row["attempt"], status=row["status"], error_type=row["error_type"], origin=row["origin"])
                if row["status"] == "ok":
                    break
            done += 1
            if done % 35 == 0 or done == len(records) * len(specs):
                print("Collection tasks %d/%d" % (done, len(records) * len(specs)), flush=True)
    for record in records:
        if sources[record["image_key"]] != dict(image=sha256(record["image_path"]), mask=sha256(record["mask_path"])):
            raise RuntimeError("Source image/reference changed")
    if sha256(config["model"]["checkpoint"]) != frozen["checkpoint_sha256"]:
        raise RuntimeError("Model weights changed")
    rows = latest_rows(root)
    summary = dict(completed_at=now(), scheduled_cases=len({r["patient_key"] for r in records}),
        scheduled_images=len(records), scheduled_tasks=len(records) * len(specs), records=len(rows),
        statuses=dict(Counter(r["status"] for r in rows)), origins=dict(Counter(r["origin"] for r in rows)),
        source_hashes_unchanged=True, weight_hash_unchanged=True, collection_lock_sha256=checksum,
        analysis_unit="Case", distance_unit="pixel", holdout_or_external_used=False)
    write_json(root / "logs" / ("collection_summary_%03d.json" % len(list((root / "logs").glob("collection_summary_*.json")))), summary)
    append_event(root, "collection_verified", **summary)
    return rows, summary
