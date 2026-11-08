"""Append-only task attempts, validated resume, local SAM experiments."""
import hashlib
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version
import torch

from .datasets import check_output, load_reference, pseudonym_salt, read_records, sha256
from .model import ExistingSAM, make_prompt


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False))


def code_fingerprint():
    root = Path(__file__).resolve().parent
    result = {p.name: sha256(p) for p in sorted(root.glob("*.py"))}
    result["legacy_run_single_case.py"] = sha256(root.parent / "run_single_case.py")
    return result


def read_results(run):
    rows = []
    for task in sorted((Path(run) / "tasks").glob("*")):
        attempts = sorted(task.glob("attempt_*/result.json"))
        if attempts:
            rows.append(json.loads(attempts[-1].read_text(encoding="utf-8")))
    return rows


def selected_records(config, project):
    records, summaries = read_records(config, pseudonym_salt(project))
    allowed = config.get("select", {})
    records = [r for r in records if (not allowed.get("datasets") or r["dataset"] in allowed["datasets"])
               and (not allowed.get("splits") or r["split"] in allowed["splits"])]
    if allowed.get("limit"):
        # Deterministic sample of source manifest order: salt cannot affect selection.
        indexes = sorted(random.Random(config.get("seed", 20261108)).sample(
            range(len(records)), min(len(records), int(allowed["limit"]))))
        records = [records[i] for i in indexes]
    if not records:
        raise ValueError("No source records selected")
    return records, summaries


def run_experiment(config, project, resume=False, retry_failed=False):
    from .features import extract_features
    from .imaging import load_rgb, restore_mask, scaled_box, transform_image
    from .metrics import evaluate_masks
    from .adaptive import choose_condition, rule_policy
    run = check_output(config["output"], config)
    records, summaries = selected_records(config, project)
    conditions = config["conditions"]
    if not isinstance(config.get("max_attempts", 2), int) or config.get("max_attempts", 2) < 1:
        raise ValueError("max_attempts must be a positive integer")
    if config.get("upload_mbps", 10) <= 0:
        raise ValueError("upload_mbps must be positive")
    ids = [s["id"] for s in conditions]
    if len(set(ids)) != len(ids) or "baseline" not in ids or any(
            not s or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in s) for s in ids):
        raise ValueError("Conditions require unique safe IDs and baseline")
    if next(s for s in conditions if s["id"] == "baseline").get("format") != "baseline":
        raise ValueError("Baseline must reference the unchanged original image")
    if config.get("adaptive", {}).get("enabled"):
        if "iacs" in ids:
            raise ValueError("iacs is reserved for the adaptive strategy")
    # Baseline must always be first, irrespective of user configuration ordering.
    conditions = sorted(conditions, key=lambda s: s["id"] != "baseline")
    prompts = config.get("prompt_strategies", ["mechanism_gt_box"])
    if not prompts or len(set(prompts)) != len(prompts):
        raise ValueError("Prompt strategies must be unique and nonempty")
    if any(p not in ("mechanism_gt_box", "application_full_frame") for p in prompts):
        raise ValueError("Unsupported prompt strategy")
    if config.get("purpose") != "development_technical_smoke" and config.get("adaptive", {}).get("enabled"):
        if not config["adaptive"].get("policy_file"):
            raise ValueError("Formal adaptive evaluation requires an explicit frozen policy file")
    sources = {r["image_key"]: dict(image=sha256(r["image_path"]), mask=sha256(r["mask_path"])) for r in records}
    fingerprint = dict(config=config, code=code_fingerprint(), source_hashes=sources,
                       manifests=summaries, checkpoint_sha256=sha256(config["model"]["checkpoint"]))
    policy_file = config.get("adaptive", {}).get("policy_file")
    if policy_file:
        fingerprint["policy_file_sha256"] = sha256(policy_file)
    fingerprint_hash = hashlib.sha256(canonical(fingerprint).encode()).hexdigest()
    if run.exists():
        if not resume:
            raise FileExistsError("Run already exists; choose a new output or explicit resume")
        existing = json.loads((run / "run.json").read_text(encoding="utf-8"))
        if existing["fingerprint_sha256"] != fingerprint_hash:
            raise ValueError("Resume refused: code/config/inputs/model changed; create a new run")
    else:
        if resume:
            raise FileNotFoundError("Cannot resume a nonexistent run")
        run.mkdir(parents=True)
        write_json(run / "run.json", dict(fingerprint_sha256=fingerprint_hash, provenance=fingerprint,
                   started_at=datetime.now(timezone.utc).isoformat(), purpose=config.get("purpose"),
                   runtime=dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                                numpy=np.__version__, pillow=pillow_version),
                   limitations=["Technical pilot only unless separately prespecified", "Distances in pixels",
                                "Full-frame application prompt is unvalidated", "IACS performance unvalidated"]))
    tasks = run / "tasks"
    tasks.mkdir(exist_ok=True)
    policy = None
    if config.get("adaptive", {}).get("enabled"):
        policy_path = config["adaptive"].get("policy_file")
        policy = json.loads(Path(policy_path).read_text(encoding="utf-8")) if policy_path else rule_policy(conditions)
        if policy_path:
            locked_body = dict(policy)
            policy_hash = locked_body.pop("policy_sha256", None)
            if hashlib.sha256(canonical(locked_body).encode()).hexdigest() != policy_hash:
                raise ValueError("Adaptive policy content lock mismatch")
            if policy.get("kind") == "development_tree" and prompts != [policy.get("prompt_strategy")]:
                raise ValueError("Frozen policy prompt strategy differs from the evaluation prompt")
        policy_snapshot = run / "adaptive_policy.json"
        if not policy_snapshot.exists():
            write_json(policy_snapshot, policy)
        elif canonical(json.loads(policy_snapshot.read_text(encoding="utf-8"))) != canonical(policy):
            raise ValueError("Adaptive policy changed since run start")
    model = None
    completed, planned = 0, len(records) * len(prompts) * (len(conditions) + int(policy is not None))
    for record in records:
        # These are real image decodes; no networking is used anywhere in this engine.
        rgb, reference, features, input_error = None, None, None, None
        try:
            rgb = load_rgb(record["image_path"])
            if rgb.shape[:2] != (record["height"], record["width"]):
                raise ValueError("Image dimensions disagree with manifest")
            reference = load_reference(record)
            features = extract_features(rgb)
        except Exception as exc:
            input_error = type(exc).__name__
        for prompt in prompts:
            specs = list(conditions)
            if policy is not None:
                if features is not None:
                    chosen = choose_condition(features, policy)
                    selected = next((s for s in conditions if s["id"] == chosen), None)
                    if selected is None:
                        raise ValueError("Adaptive strategy selected a condition absent from the experiment")
                    specs.append(dict(selected, id="iacs", selected_condition=chosen))
                else:
                    specs.append(dict(id="iacs", selected_condition=None))
            for spec in specs:
                task_id = hashlib.sha256((record["image_key"] + ":" + prompt + ":" + spec["id"]).encode()).hexdigest()[:28]
                task_dir = tasks / task_id
                task_dir.mkdir(exist_ok=True)
                existing = sorted(task_dir.glob("attempt_*/result.json"))
                if existing:
                    previous = json.loads(existing[-1].read_text(encoding="utf-8"))
                    if previous["status"] == "ok":
                        for artifact, checksum in (("prediction_mask", "prediction_sha256"), ("processed_image", "processed_sha256")):
                            if artifact in previous:
                                path = run / previous[artifact]
                                if not path.is_file() or sha256(path) != previous.get(checksum):
                                    raise ValueError("Resume refused: successful task artifact missing or changed")
                    if previous["status"] == "ok" or not retry_failed:
                        completed += 1
                        continue
                # A crash can leave an unfinished directory; preserve it and use a new attempt.
                count = len(list(task_dir.glob("attempt_*")))
                for retry in range(int(config.get("max_attempts", 2))):
                    attempt = task_dir / ("attempt_%04d" % (count + retry + 1))
                    attempt.mkdir(exist_ok=False)
                    row = {k: record[k] for k in ("dataset", "split", "patient_key", "image_key")}
                    row.update(condition=spec["id"], selected_condition=spec.get("selected_condition"),
                        experiment_id=run.name, prompt_strategy=prompt, status="error", error_type=None,
                        attempt=count + retry + 1, format=spec.get("format", "baseline"),
                        quality=spec.get("quality"), scale=spec.get("scale", 1.0),
                        interpolation=spec.get("interpolation", "lanczos"), features=features,
                        original_bytes=Path(record["image_path"]).stat().st_size,
                        stage="source_validation", input_error_type=input_error,
                        model_commit=config["model"]["commit"], weight_sha256=config["model"]["sha256"])
                    try:
                        if input_error:
                            raise ValueError("Source validation failed: " + input_error)
                        start_box = make_prompt(rgb, reference, prompt)
                        row["stage"] = "compression"
                        if spec.get("format", "baseline") == "baseline":
                            processed = rgb
                            metadata = dict(output_bytes=row["original_bytes"], width=rgb.shape[1], height=rgb.shape[0],
                                            encode_ms=0.0, decode_ms=0.0, pixel_equal=True, processing_order="original")
                        else:
                            extensions = {"jpeg": ".jpg", "png": ".png", "webp": ".webp", "jpeg2000": ".jp2"}
                            suffix = ".png" if spec.get("order") == "encode_resize" else extensions[spec["format"]]
                            processed, metadata = transform_image(rgb, spec, attempt / ("processed" + suffix))
                            row["processed_image"] = str((attempt / ("processed" + suffix)).relative_to(run))
                            row["processed_sha256"] = sha256(run / row["processed_image"])
                        box = scaled_box(start_box, (rgb.shape[1], rgb.shape[0]), (processed.shape[1], processed.shape[0]))
                        row["stage"] = "model_inference"
                        if model is None:
                            torch.manual_seed(config.get("seed", 20261108))
                            np.random.seed(config.get("seed", 20261108))
                            torch.backends.cudnn.benchmark = False
                            model = ExistingSAM(config["model"])
                        prediction, model_meta = model.predict(processed, box)
                        restored = restore_mask(prediction, (rgb.shape[1], rgb.shape[0]))
                        row["stage"] = "evaluation_and_artifacts"
                        with (attempt / "prediction.png").open("xb") as stream:
                            Image.fromarray(restored.astype(np.uint8) * 255).save(stream, format="PNG")
                        evaluation = evaluate_masks(restored, reference)
                        row.update(metadata)
                        row.update(model_meta)
                        row.update(evaluation)
                        row.update(status="ok", prompt_box_original=start_box.tolist(), prompt_box_processed=box.tolist(),
                                   prediction_mask=str((attempt / "prediction.png").relative_to(run)))
                        row["prediction_sha256"] = sha256(attempt / "prediction.png")
                        ratio = row["output_bytes"] / row["original_bytes"]
                        row.update(compression_ratio=ratio, saving_pct=(1 - ratio) * 100,
                                   simulated_upload_ms=row["output_bytes"] * 8 / (config.get("upload_mbps", 10) * 1e6) * 1000)
                        baseline = next((x for x in read_results(run) if x["image_key"] == record["image_key"]
                                         and x["condition"] == "baseline" and x["prompt_strategy"] == prompt and x["status"] == "ok"), None)
                        if baseline:
                            with Image.open(run / baseline["prediction_mask"]) as image:
                                baseline_mask = np.asarray(image) != 0
                            from .metrics import evaluate_masks as compare
                            row["prediction_consistency_dice"] = compare(restored, baseline_mask)["dice"]
                            row["predicted_area_change_pct"] = (100 * (int(restored.sum()) - int(baseline_mask.sum())) / int(baseline_mask.sum())
                                                               if baseline_mask.any() else None)
                        row["stage"] = "complete"
                    except Exception as exc:
                        row["status"] = "error"
                        row["error_type"] = type(exc).__name__
                        # Do not put exception text, raw patient identifiers, or source paths in normal logs.
                    write_json(attempt / "result.json", row)
                    with (run / "events.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(canonical(dict(task=task_id, attempt=row["attempt"], status=row["status"], error_type=row["error_type"])) + "\n")
                    if row["status"] == "ok":
                        break
                completed += 1
                print("Tasks {}/{}".format(completed, planned), flush=True)
    for record in records:
        expected = sources[record["image_key"]]
        if sha256(record["image_path"]) != expected["image"] or sha256(record["mask_path"]) != expected["mask"]:
            raise RuntimeError("Source safety verification failed")
    if sha256(config["model"]["checkpoint"]) != fingerprint["checkpoint_sha256"]:
        raise RuntimeError("Weight safety verification failed")
    rows = read_results(run)
    with (run / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(canonical(dict(event="run_checked", tasks=len(rows), ok=sum(r["status"] == "ok" for r in rows),
                                    input_hashes_unchanged=True, weights_unchanged=True)) + "\n")
    return rows
