"""Export local legacy workbench numbers without images or raw identities.

Standard library only. An optional external key file gives stable HMAC keys.
Without --key-file an ephemeral key is generated and never written to output.
"""
import argparse
import csv
import hashlib
import hmac
import json
import math
import re
import secrets
import shutil
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ALIASES = {"patient_id_verified": "case_identity_verified", "patient_ids_verified": "case_identity_verified",
           "patient_identity_verified": "case_identity_verified", "patient_codes": "source_case_code_count"}
DROP = {"patient_id", "image_id", "image_key", "case_id", "lesion_id", "pathology", "original_image",
        "prediction_mask", "error", "error_text", "name", "executable", "source_checksums", "patient_split_version",
        "prediction_sha256", "original_sha256", "processed_sha256", "image_sha256", "mask_sha256", "pixel_sha256",
        "calibration_patient_ids", "calibration_image_hashes"}
PATH_PATTERN = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\|/Users/|/home/)")
ERROR_TYPE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,80}\Z")
METRICS = ("dice", "iou", "precision", "recall", "sensitivity", "hd95", "assd", "boundary_f1",
           "relative_area_error", "original_bytes", "compressed_bytes", "compression_ratio", "compression_saving_pct",
           "encoding_ms", "decoding_ms", "inference_ms", "gpu_peak_mib", "simulated_upload_time_s")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pseudonym(dataset, raw_id, key):
    """Match compression_study.datasets.key: dataset + ':' + raw ID, 24 hex."""
    return hmac.new(key, (str(dataset) + ":" + str(raw_id)).encode("utf-8"), hashlib.sha256).hexdigest()[:24]


def sensitive(name):
    lowered = str(name).lower()
    return lowered not in ALIASES and (lowered in DROP or lowered.endswith("_path") or lowered.endswith("_root")
                                       or lowered in {"checkpoint", "sam_source"})


def sanitize(value, raw_identifiers, name="", audit=None):
    if isinstance(value, dict):
        return {ALIASES.get(k, k): sanitize(v, raw_identifiers, str(k), audit) for k, v in value.items()
                if not sensitive(k) and not PATH_PATTERN.search(str(k))}
    if isinstance(value, list):
        return [sanitize(v, raw_identifiers, name, audit) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        if audit is not None:
            audit["nonfinite_numbers_as_null"] += 1
        return None
    if isinstance(value, (int, float, bool, type(None))):
        return value
    text = str(value)
    if PATH_PATTERN.search(text) or any(identifier in text for identifier in raw_identifiers if len(identifier) >= 5):
        return "[private text omitted]"
    if name == "error_type" and not ERROR_TYPE.fullmatch(text):
        return "UnspecifiedErrorType"
    return text


def identity_record(row, key, raw_identifiers, audit):
    output = sanitize(row, raw_identifiers, audit=audit)
    dataset = row.get("dataset", "unknown")
    output["case_key"] = pseudonym(dataset, row["patient_id"], key) if row.get("patient_id") is not None else None
    output["image_key"] = pseudonym(dataset, row["image_id"], key) if row.get("image_id") is not None else None
    if row.get("error"):
        output["error_text_omitted"] = True
    return output


def numeric_leaves(value, trail=()):
    if isinstance(value, dict):
        for name, item in value.items():
            if not sensitive(name):
                yield from numeric_leaves(item, trail + (ALIASES.get(name, name),))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from numeric_leaves(item, trail + (index,))
    elif isinstance(value, (int, float, bool)) and (not isinstance(value, float) or math.isfinite(value)):
        yield trail, value


def check_numeric_preservation(source, exported):
    count = 0
    for trail, original in numeric_leaves(source):
        current = exported
        for part in trail:
            current = current[part]
        if current != original or type(current) is not type(original):
            raise AssertionError("A non-private numeric value changed during export")
        count += 1
    return count


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def cell(value):
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def write_csv(path, rows):
    fields = list(dict.fromkeys(name for row in rows for name in row)) or ["status"]
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: cell(row.get(name)) for name in fields})


def metric_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("variant", "unknown")].append(row)
    output = []
    for variant, members in sorted(groups.items()):
        for metric in METRICS:
            values = [row.get(metric) for row in members if row.get("status") in {"completed", "ok", "success"}
                      and isinstance(row.get(metric), (int, float)) and not isinstance(row.get(metric), bool)
                      and math.isfinite(row[metric])]
            output.append({"variant": variant, "metric": metric, "scheduled_records": len(members),
                           "scheduled_cases": len({r["case_key"] for r in members if r.get("case_key")}),
                           "valid_records": len(values), "missing_or_failed_records": len(members) - len(values),
                           "mean": statistics.mean(values) if values else None,
                           "median": statistics.median(values) if values else None,
                           "min": min(values) if values else None, "max": max(values) if values else None})
    return output


def failure_summary(rows):
    types, contexts = Counter(), Counter()
    for row in rows:
        if row.get("status") not in {"failed", "error"}:
            continue
        error_type = str(row.get("error_type") or "UnspecifiedErrorType")
        types[error_type if ERROR_TYPE.fullmatch(error_type) else "UnspecifiedErrorType"] += 1
        text = str(row.get("error", "")).lower()
        if "deterministic" in text and ("cuda" in text or "cumsum" in text):
            contexts["deterministic_cuda_operation_unavailable"] += 1
        elif "out of memory" in text:
            contexts["out_of_memory"] += 1
        else:
            contexts["details_omitted_for_privacy"] += 1
    return {"error_type_counts": dict(types), "generic_context_counts": dict(contexts),
            "free_form_messages_exported": False,
            "interpretation": "Context labels use fixed phrase checks, not patient-specific error text"}


def export_bundle(source_root, output_dir, key_file=None, inventory_file=None):
    root, output = Path(source_root).resolve(strict=True), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Export destination exists; choose a new directory")
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("Export destination must be separate from the source experiments")
    if key_file:
        key_path = Path(key_file).resolve(strict=True)
        if output == key_path or output in key_path.parents:
            raise ValueError("The secret key must remain outside the export directory")
        key = key_path.read_bytes()
        if len(key) < 16:
            raise ValueError("Use an external key with at least 16 bytes")
        key_mode = "external_key_not_exported"
    else:
        key, key_mode = secrets.token_bytes(32), "ephemeral_key_not_exported"
    sources, batches, raw_ids = [], [], set()
    for directory in sorted(root.glob("EXP_*")):
        if not directory.is_dir() or directory.is_symlink() or root not in directory.resolve().parents:
            continue
        loaded = {}
        for name in ("results.jsonl", "summary.json", "config.json"):
            path = directory / name
            if not path.is_file() or path.is_symlink():
                raise ValueError("Expected regular legacy results, summary and configuration files")
            sources.append({"source": "experiments/" + directory.name + "/" + name,
                            "sha256": digest(path), "bytes": path.stat().st_size, "_path": path})
            text = path.read_text(encoding="utf-8-sig")
            loaded[name] = ([json.loads(line) for line in text.splitlines() if line.strip()] if name.endswith("jsonl") else json.loads(text))
        for row in loaded["results.jsonl"] + loaded["config.json"].get("records", []):
            for field in ("patient_id", "image_id", "image_key"):
                if row.get(field) is not None:
                    raw_ids.add(str(row[field]))
        for identity in (loaded["config.json"].get("iacs") or {}).get("calibration_patient_ids", []):
            raw_ids.add(str(identity))
        batches.append((directory.name, loaded))
    if not batches:
        raise ValueError("No legacy experiments were found")
    inventory = None
    if inventory_file:
        path = Path(inventory_file).resolve(strict=True)
        sources.append({"source": "dataset_inventory.json", "sha256": digest(path), "bytes": path.stat().st_size, "_path": path})
        inventory = json.loads(path.read_text(encoding="utf-8-sig"))
    output.mkdir(parents=True, exist_ok=False)
    audit = Counter()
    experiment_manifest, all_rows = [], []
    for experiment, loaded in batches:
        directory = output / experiment
        directory.mkdir()
        rows = [identity_record(row, key, raw_ids, audit) for row in loaded["results.jsonl"]]
        for original, exported in zip(loaded["results.jsonl"], rows):
            audit["result_numeric_values_verified"] += check_numeric_preservation(original, exported)
        with (directory / "results.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        write_csv(directory / "results.csv", rows)
        summary = sanitize(loaded["summary.json"], raw_ids, audit=audit)
        audit["summary_numeric_values_verified"] += check_numeric_preservation(loaded["summary.json"], summary)
        write_json(directory / "summary.json", summary)
        write_json(directory / "failure_summary.json", failure_summary(loaded["results.jsonl"]))
        config = loaded["config.json"]
        selected = {name: value for name, value in config.items() if name not in {"records", "source_checksums", "patient_split_version", "code_snapshot"}}
        public_config = sanitize(selected, raw_ids, audit=audit)
        public_config["records"] = [identity_record(row, key, raw_ids, audit) for row in config.get("records", [])]
        public_config["input_manifest_checksums"] = [{"source_file": Path(name).name, "sha256": value}
                                                     for name, value in config.get("source_checksums", {}).items()]
        public_config["code_checksums"] = [{"file": name.replace("\\", "/"), "sha256": value}
                                           for name, value in config.get("code_snapshot", {}).items()
                                           if not PATH_PATTERN.search(name) and ".." not in Path(name).parts]
        if config.get("iacs"):
            public_config["iacs"]["calibration_case_keys"] = [pseudonym(config["dataset"], raw, key) for raw in config["iacs"].get("calibration_patient_ids", [])]
        for original, exported in zip(config.get("records", []), public_config["records"]):
            audit["config_record_numeric_values_verified"] += check_numeric_preservation(original, exported)
        audit["config_numeric_values_verified"] += check_numeric_preservation(selected, public_config)
        public_config["publication_notes"] = {"analysis_unit": "Case/source identifier; patient semantics not inferred",
                                              "paths_raw_identifiers_pathology_and_per_image_hashes_removed": True,
                                              "failure_text_omitted": True, "clinical_conclusion": None}
        write_json(directory / "config_summary.json", public_config)
        aggregate = metric_summary(rows)
        write_json(directory / "metric_summary.json", aggregate)
        write_csv(directory / "metric_summary.csv", aggregate)
        experiment_manifest.append({"experiment_id": experiment, "dataset": config.get("dataset"), "split": config.get("split"),
                                    "prompt_strategy": config.get("prompt_strategy"), "device": config.get("device"),
                                    "records": len(rows), "cases": len({r["case_key"] for r in rows if r.get("case_key")}),
                                    "status_counts": dict(Counter(r.get("status", "unknown") for r in rows)),
                                    "error_type_counts": dict(Counter(r.get("error_type") for r in rows if r.get("error_type")))})
        all_rows.extend(rows)
    if inventory is not None:
        aggregate = {name: inventory.get(name) for name in ("schema_version", "dataset_version", "dataset_checksum", "summary")}
        write_json(output / "dataset_inventory_aggregate.json", sanitize(aggregate, raw_ids, audit=audit))
    write_csv(output / "all_results.csv", all_rows)
    with (output / "all_results.jsonl").open("x", encoding="utf-8") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    changed = [source["source"] for source in sources if digest(source["_path"]) != source["sha256"]]
    if changed:
        raise RuntimeError("A source file changed during export; do not publish this bundle")
    write_json(output / "SOURCE_SHA256.json", [{k: v for k, v in source.items() if k != "_path"} for source in sources])
    write_json(output / "EXPORT_AUDIT.json", {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
               "key_mode": key_mode, "secret_or_mapping_exported": False, "source_hashes_unchanged": True,
               "experiments": experiment_manifest, "total_records": len(all_rows),
               "unique_case_keys": len({row["case_key"] for row in all_rows if row.get("case_key")}),
               "unique_image_keys": len({row["image_key"] for row in all_rows if row.get("image_key")}),
               "numeric_verification": dict(audit),
               "scope": "Legacy workbench technical pilots; not new IACS_v2 efficacy or clinical results"})
    readme = """# Legacy workbench: deidentified numeric technical-pilot records

This bundle contains every final task row from the three legacy local workbench experiments, including the entire failed run. No new experiment or patient result was invented. These records are separate from the later IACS_v2 development study.

The JSONL files preserve nested numeric measurements; CSV stores nested objects as JSON cells. Source summaries and deidentified configuration summaries retain processing recipes, source-code/model checksums, feature definitions, numeric metadata, method descriptions and run settings. Descriptive metric tables retain planned/valid/missing-or-failed counts. They do not establish clinical acceptability or noninferiority. Upload times are simulated, not measured network performance. The source dataset inventory is aggregate-only.

Case and image keys are stable HMAC-SHA256 identifiers (24 hex characters), using the existing pilot's `dataset + ':' + raw identifier` convention. The external secret key and raw-to-pseudonym correspondence are not included. Case/source identifiers are not automatically confirmed independent patients.

Excluded: all original images, masks, derived pixel images, existing figures, SQLite databases, pathology, raw identities, source-image paths, per-image content hashes and free-form failure messages. Original error types and failure states are retained. Per-experiment failure summaries provide fixed, non-identifying context labels when a recognized phrase is present; omitting free-form text does not remove failure rows.

`SOURCE_SHA256.json` identifies the read-only input numeric/configuration files by relative experiment label and SHA-256. `EXPORT_AUDIT.json` records full row counts, numeric-value checks and before/after source hashes. `ARTIFACT_SHA256.json` verifies this export's files.

## Repeat locally

The included script uses only Python's standard library:

```powershell
python -I export_legacy_numeric.py --source-root '<local experiments directory>' --output '<new separate export directory>' --key-file '<external secret file>' --inventory '<optional local inventory.json>'
```

The key-file option is optional; omitting it creates a temporary key that is never exported and produces new pseudonyms on the next invocation. Supply the same external key to reproduce stable identities across exports. Never put the secret key or original correspondence data in a public repository. The destination must not exist; existing exports and input files are not overwritten.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    shutil.copyfile(Path(__file__), output / "export_legacy_numeric.py")
    artifacts = [{"file": str(path.relative_to(output)).replace("\\", "/"), "sha256": digest(path), "bytes": path.stat().st_size}
                 for path in sorted(output.rglob("*")) if path.is_file()]
    write_json(output / "ARTIFACT_SHA256.json", artifacts)
    return {"experiments": len(batches), "records": len(all_rows), "source_files_unchanged": not changed,
            "numeric_values_verified": sum(value for name, value in audit.items() if name.endswith("_verified")), "output_dir": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--key-file", default=None)
    parser.add_argument("--inventory", default=None)
    args = parser.parse_args()
    print(json.dumps(export_bundle(args.source_root, args.output, args.key_file, args.inventory), ensure_ascii=True))


if __name__ == "__main__":
    main()
