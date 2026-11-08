import contextlib
import json
import os
import uuid
from pathlib import Path

from PIL import Image
import numpy as np

from .common import contained, now, read_json, write_json, sha256, source_snapshot, runtime, hardware
from .compression import recipes, compress
from .datasets import load_rgb, load_mask
from .geometry import map_box, restore_mask, application_box
from .features import extract_features
from .store import Store


def inventory(config):
    path = Path(config["output_root"]) / "dataset_report" / "inventory.json"
    if not path.exists():
        raise ValueError("Run dataset scan before creating an experiment")
    return read_json(path)


def experiment_dir(config, experiment_id):
    if not experiment_id.startswith("EXP_") or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for c in experiment_id):
        raise ValueError("Invalid experiment ID")
    return contained(Path(config["output_root"]) / "experiments", experiment_id)


def create_experiment(config, options):
    report = inventory(config)
    dataset = options.get("dataset", "BUS-BRA")
    split = options.get("split", "development")
    prompt = options.get("prompt_strategy", options.get("prompt", "mechanism"))
    if prompt not in ("mechanism", "application"):
        raise ValueError("Invalid prompt strategy")
    if split != "development" and not config["formal_protocol_locked"]:
        raise ValueError("Formal internal/external compression validation is blocked until the new protocol and IACS are frozen")
    variants = options.get("variants", ["baseline", "jpeg_q85"])
    if isinstance(variants, str):
        variants = variants.split(",")
    variants = list(dict.fromkeys(["baseline"] + variants))
    all_recipes = recipes()
    if any(v not in all_recipes and v != "iacs" for v in variants):
        raise ValueError("Unknown compression variant")
    iacs_path = Path(config["output_root"]) / "experiment_configs" / "iacs_candidate.json"
    iacs = read_json(iacs_path) if "iacs" in variants else None
    if "iacs" in variants and config.get("iacs_lock_file"):
        locked = read_json(config["iacs_lock_file"])
        if locked["sha256"] != sha256(iacs_path):
            raise ValueError("IACS locked configuration hash mismatch")
        iacs = locked["configuration"]
    if iacs and dataset != "BUS-BRA" and iacs["state"] != "locked":
        raise ValueError("Unvalidated IACS cannot be tuned or evaluated on external data")
    records = [r for r in report["records"] if r["dataset"] == dataset and r["split"] == split
               and r["selected_in_frozen_manifest"]]
    # Do not silently filter corrupt/invalid cases from the frozen cohort: tasks will record their failures.
    records.sort(key=lambda r: r["image_id"])
    limit = int(options.get("limit", 2))
    if limit < 0:
        raise ValueError("limit must be >=0; zero requests all selected images")
    if limit:
        records = records[:limit]
    if not records:
        raise ValueError("No frozen image records match this dataset/split")
    experiment_id = "EXP_" + uuid.uuid4().hex
    directory = experiment_dir(config, experiment_id)
    directory.mkdir(parents=True, exist_ok=False)
    frozen = {"experiment_id": experiment_id, "name": str(options.get("name") or "Compression technical pilot")[:150],
        "created_at": now(), "status_at_creation": "planned", "dataset": dataset, "split": split,
        "study_status": "technical_pilot_unvalidated" if not config["formal_protocol_locked"] else "protocol_locked_validation",
        "dataset_version": report.get("dataset_version"), "dataset_checksum": report["dataset_checksum"],
        "source_checksums": report["source_checksums"], "records": records,
        "variant_recipes": {v: all_recipes[v] for v in variants if v != "iacs"}, "variants": variants, "iacs": iacs,
        "prompt_strategy": prompt, "prompt_description": "original-reference bbox+10%, mapped once to processed pixels" if prompt == "mechanism" else "fixed central 80% image field; unvalidated automatic candidate",
        "legacy_prompt_source_sha256": sha256(Path(config["pilot_root"]) / "run_single_case.py"),
        "device": options.get("device", "cuda"), "model_name": "SAM", "model_version": "v1-vit_b",
        "platform_config": dict(config), "code_snapshot": source_snapshot(), "runtime": runtime(), "hardware": hardware(),
        "selection_rule": "frozen one-image-per-case manifest; image_id ascending; first limit or all when limit=0",
        "patient_split_version": {k: v for k, v in report["source_checksums"].items() if "split" in k or "manifest" in k}}
    write_json(directory / "config.json", frozen, exclusive=True)
    write_json(directory / "config_lock.json", {"sha256": sha256(directory / "config.json")}, exclusive=True)
    store = Store(directory)
    store.schedule(records, variants)
    _publish(directory, store)
    return experiment_id


@contextlib.contextmanager
def execution_lock(root):
    path = Path(root) / ".execution.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0)
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("Another experiment is already running; wait before resuming")
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _publish(directory, store):
    results = store.results()
    temp = directory / "results.jsonl.tmp"
    with temp.open("w", encoding="utf-8") as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(str(temp), str(directory / "results.jsonl"))
    write_json(directory / "summary.json", dict(store.summary(), updated_at=now()))


def run_experiment(config, experiment_id, max_retries=1):
    directory = experiment_dir(config, experiment_id)
    lock = read_json(directory / "config_lock.json")
    if sha256(directory / "config.json") != lock["sha256"]:
        raise ValueError("Experiment configuration changed after creation")
    frozen = read_json(directory / "config.json")
    if frozen["code_snapshot"] != source_snapshot():
        raise ValueError("Code changed; create a new experiment version rather than mixing results")
    # Use the saved platform settings on resume, never newly edited global settings.
    config = frozen["platform_config"]
    if sha256(Path(config["pilot_root"]) / "run_single_case.py") != frozen["legacy_prompt_source_sha256"]:
        raise ValueError("Reused original prompt source changed since experiment creation")
    store = Store(directory)
    pending = [r for r in store.results() if r["status"] != "completed"]
    if not pending:
        return store.summary()
    from .models import SamModel, legacy_prompt_boxes
    from .metrics import evaluate, stability, compression_efficiency
    with execution_lock(config["output_root"]):
        try:
            model = SamModel(config, frozen["device"])
            write_json(directory / ("model_session_" + uuid.uuid4().hex + ".json"), model.info, exclusive=True)
        except Exception as error:
            for task in pending:
                store.update(task["image_key"], task["variant"], "failed", {"error_type": type(error).__name__, "error": str(error), "stage": "model_loading"})
            _publish(directory, store)
            raise
        for record in frozen["records"]:
            key = record["dataset"] + ":" + record["image_id"]
            original_size = (record["width"], record["height"])
            for variant in frozen["variants"]:
                task = store.state(key, variant)
                if task["status"] == "completed":
                    continue
                for retry in range(max_retries + 1):
                    store.update(key, variant, "running")
                    attempt = store.state(key, variant)["attempts"]
                    _publish(directory, store)
                    try:
                        if record["data_status"] != "ok" or record["annotation_status"] != "complete":
                            raise ValueError("Invalid frozen data/annotation status; retained as failed task")
                        if sha256(record["image_path"]) != record["image_sha256"] or sha256(record["mask_path"]) != record["mask_sha256"]:
                            raise ValueError("Original image/reference changed since inventory freeze")
                        reference = load_mask(record["mask_path"])
                        original = load_rgb(record["image_path"])
                        original_box = legacy_prompt_boxes(config, reference) if frozen["prompt_strategy"] == "mechanism" else application_box(original)
                        feat = extract_features(record["image_path"])
                        decision = None
                        selected = variant
                        if variant == "iacs":
                            from .adaptive import decide
                            decision = decide(feat, frozen["iacs"])
                            selected = decision["selected_variant"]
                        recipe = recipes()[selected] if variant == "iacs" else frozen["variant_recipes"][variant]
                        artifact_dir = directory / "artifacts" / record["image_id"] / variant / ("attempt_%d" % attempt)
                        encoded = compress(record["image_path"], artifact_dir, record["image_id"], recipe)
                        rgb = load_rgb(encoded["processed_path"])
                        box = map_box(original_box, original_size, (encoded["width"], encoded["height"]))
                        prediction, timing = model.predict(rgb, box)
                        Image.fromarray(prediction.astype(np.uint8) * 255).save(artifact_dir / "prediction_processed.png")
                        restored = restore_mask(prediction, original_size)
                        mask_path = artifact_dir / "prediction_original.png"
                        Image.fromarray(restored.astype(np.uint8) * 255).save(mask_path)
                        metric = evaluate(reference, restored, spacing=None, boundary_tolerance=config["boundary_tolerance_pixels"])
                        metric["boundary_f1_unit"] = "pixel"
                        if record.get("pixel_spacing"):
                            physical = evaluate(reference, restored, spacing=record["pixel_spacing"])
                            metric.update(hd95=physical["hd95"], assd=physical["assd"], boundary_unit="mm")
                        stability_result = {"metric_status": "baseline_unavailable", "reason": "Baseline did not complete; no stability estimate was imputed"}
                        base = store.state(key, "baseline")
                        if base["status"] == "completed":
                            base_result = json.loads(base["result"])
                            stability_result = stability(load_mask(base_result["prediction_mask"]), restored, spacing=record.get("pixel_spacing"))
                        efficiency = compression_efficiency(encoded["original_bytes"], encoded["compressed_bytes"], config["bandwidth_mbps"], config["rtt_ms"])
                        result = {"experiment_id": experiment_id, "dataset": record["dataset"], "patient_id": record["patient_id"],
                            "image_id": record["image_id"], "lesion_id": record["lesion_id"], "split": record["split"], "pathology": record["pathology"],
                            "variant": variant, "selected_variant": selected, "model": "SAM", "model_version": "v1-vit_b", "prompt_strategy": frozen["prompt_strategy"],
                            "original_box_xyxy": original_box.tolist(), "processed_box_xyxy": box.tolist(),
                            "prediction_mask": str(mask_path), "prediction_sha256": sha256(mask_path), "original_image": record["image_path"],
                            "features": feat, "iacs_decision": decision, "empty_prediction": not bool(restored.any()),
                            "reference_area_pixels": int(reference.sum()), "prediction_area_pixels": int(restored.sum()),
                            "model_weight_hash": model.info["weight_sha256"], "sam_commit": model.info["sam_commit"],
                            **encoded, **timing, **metric, "stability": stability_result, **efficiency,
                            "compression_saving": efficiency["compression_saving_pct"], "completed_at": now()}
                        write_json(artifact_dir / "result.json", result, exclusive=True)
                        store.update(key, variant, "completed", result)
                        break
                    except Exception as error:
                        store.update(key, variant, "failed", {"experiment_id": experiment_id, "dataset": record["dataset"], "patient_id": record["patient_id"],
                            "image_id": record["image_id"], "model": "SAM", "prompt_strategy": frozen["prompt_strategy"], "split": record["split"],
                            "error_type": type(error).__name__, "error": str(error), "failed_at": now(), "retry_in_this_run": retry})
                    finally:
                        _publish(directory, store)
        final_hash = sha256(config["checkpoint"])
        if final_hash != config["checkpoint_sha256"]:
            raise RuntimeError("Original model protection check failed")
        integrity = {"model_sha256": final_hash,
            "all_selected_original_images_unchanged": all(sha256(r["image_path"]) == r["image_sha256"] for r in frozen["records"]),
            "all_selected_reference_masks_unchanged": all(sha256(r["mask_path"]) == r["mask_sha256"] for r in frozen["records"]), "checked_at": now()}
        write_json(directory / ("integrity_" + uuid.uuid4().hex + ".json"), integrity, exclusive=True)
        if not integrity["all_selected_original_images_unchanged"] or not integrity["all_selected_reference_masks_unchanged"]:
            raise RuntimeError("Original input integrity check failed; experiment not accepted")
    return store.summary()


def details(config, experiment_id):
    directory = experiment_dir(config, experiment_id)
    return {"config": read_json(directory / "config.json"), "summary": Store(directory).summary(), "results": Store(directory).results()}


def export_results(config, experiment_id):
    from .exports import export_experiment
    return export_experiment(experiment_dir(config, experiment_id))


def analyze_results(config, experiment_id, metric="dice", delta=None):
    from .statistics import analyze, mixed_effects
    directory = experiment_dir(config, experiment_id)
    frozen = read_json(directory / "config.json")
    rows = Store(directory).results()
    # Formal margins come from the frozen protocol, not a post-hoc UI input.
    approved = frozen["platform_config"].get("dice_noninferiority_margin" if metric == "dice" else "hd95_margin")
    if delta is not None and delta != approved:
        raise ValueError("Margin must be specified in the protocol before experiment creation")
    verified = bool(frozen["platform_config"]["patient_ids_verified"])
    result = {"paired": analyze(rows, metric=metric, delta=approved, patient_ids_verified=verified),
              "mixed_effects": mixed_effects(rows, metric=metric, patient_ids_verified=verified), "source_experiment": experiment_id}
    output = directory / "analyses" / (uuid.uuid4().hex + ".json")
    write_json(output, result, exclusive=True)
    return result


def calibrate_candidate(config):
    from .adaptive import calibrate
    report = inventory(config)
    records = [r for r in report["records"] if r["dataset"] == "BUS-BRA" and r["split"] == "development" and r["selected_in_frozen_manifest"] and r["data_status"] == "ok"]
    path = Path(config["output_root"]) / "experiment_configs" / "iacs_candidate.json"
    return calibrate(records, [extract_features(r["image_path"]) for r in records], path)
