"""Original-image features and real, complete per-quality outcome tables."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from compression_study.datasets import read_records, pseudonym_salt, load_reference, sha256
from compression_study.imaging import load_rgb
from compression_study.features import extract_features as old_features
from compression_study.adaptive import rule_policy, _matches
from run_single_case import dice as consistency_dice
from .collector import latest_rows, validate_artifacts, CONDITIONS
from .features import extract_record, FEATURE_NAMES, FEATURE_VERSION, FEATURE_METHODS
from .common import require_development, safe_run, write_json, write_csv, append_event, read_json, digest


def _number(value):
    return float(value) if value is not None and np.isfinite(value) else None


def _change(current, baseline):
    first, second = _number(current), _number(baseline)
    return first - second if first is not None and second is not None else None


def evaluation_shape(reference):
    """GT-only evaluation strata. These values never enter the policy features."""
    reference = np.asarray(reference, dtype=bool)
    area = int(reference.sum())
    padded = np.pad(reference.astype(np.int8), 1)
    grid_perimeter = int(np.abs(np.diff(padded, axis=0)).sum() + np.abs(np.diff(padded, axis=1)).sum())
    return dict(lesion_area_fraction=area / reference.size if reference.size else None,
                shape_irregularity=grid_perimeter ** 2 / (4 * np.pi * area) if area else None,
                source="reference_mask_evaluation_only", perimeter_method="Manhattan grid-edge proxy")


def prepare(config):
    require_development(config)
    root = safe_run(config)
    if (root / "metrics" / "sensitivity_samples.json").exists():
        raise FileExistsError("Prepared database exists; never overwrite scientific records")
    lock = read_json(root / "configs" / "collection_lock.json")
    records, cohorts = read_records(config, pseudonym_salt(config["project_root"]))
    if config.get("technical_limit"):
        records = records[:int(config["technical_limit"])]
    rows = latest_rows(root)
    by_image = defaultdict(dict)
    for row in rows:
        key = row["image_key"]
        if row["condition"] in by_image[key]:
            raise ValueError("Unresolved duplicate task result")
        if row["status"] == "ok":
            validate_artifacts(root, row)
        by_image[key][row["condition"]] = row
    feature_records, samples, long, wide, legacy = [], [], [], [], []
    policy = rule_policy(CONDITIONS)
    for record in records:
        image_key = record["image_key"]
        if lock["provenance"]["sources"][image_key] != dict(image=sha256(record["image_path"]), mask=sha256(record["mask_path"])):
            raise ValueError("Prepared source differs from collection lock")
        feature_record = dict(case_key=record["patient_key"], image_key=image_key, dataset="BUS-BRA", split="development")
        original, reference = None, None
        try:
            original = load_rgb(record["image_path"])
            with Image.open(record["image_path"]) as source:
                image_format = source.format
            extraction = extract_record(original, format=image_format, bit_depth=8)
            feature_record.update(extraction)
            original_features = old_features(original)
            matched = next((i for i, rule in enumerate(policy["rules"]) if _matches(original_features, rule)), None)
            choice = policy["rules"][matched]["condition"] if matched is not None else policy["default_condition"]
            trace = dict(case_key=record["patient_key"], image_key=image_key,
                         original_rule_condition=choice, matched_rule_index=matched + 1 if matched is not None else None,
                         fallback=matched is None, fallback_reason="no_rule_matched" if matched is None else None,
                         original_risk_score=None, original_rule_features=original_features)
            feature_record.update(original_rule_condition=choice, original_rule_fallback=trace["fallback"])
            legacy.append(trace)
        except Exception as exc:
            feature_record.update(status="error", error_type=type(exc).__name__, features={k: None for k in FEATURE_NAMES},
                                  missing_features=list(FEATURE_NAMES), metadata={"feature_version": FEATURE_VERSION})
        feature_records.append(feature_record)
        sample = dict(case_key=record["patient_key"], image_key=image_key, dataset="BUS-BRA", split="development",
                      prompt_strategy="mechanism_gt_box", features=feature_record["features"],
                      feature_status=feature_record["status"], feature_version=FEATURE_VERSION,
                      evaluation={}, candidates={})
        try:
            reference = load_reference(record)
            sample["evaluation"] = evaluation_shape(reference)
        except Exception as exc:
            sample["evaluation"] = dict(status="invalid_reference", error_type=type(exc).__name__)
        actual = by_image[image_key]
        baseline = actual.get("baseline", {})
        baseline_mask = None
        if baseline.get("status") == "ok":
            with Image.open(root / baseline["prediction_mask"]) as image:
                baseline_mask = np.asarray(image) != 0
        table = dict(case_key=record["patient_key"], image_key=image_key, dataset="BUS-BRA", split="development")
        for spec in CONDITIONS:
            condition = spec["id"]
            result = actual.get(condition)
            if result is None:
                result = dict(condition=condition, status="not_run", error_type="MissingTask",
                              original_bytes=Path(record["image_path"]).stat().st_size)
            candidate = dict(result)
            if candidate.get("status") == "ok":
                candidate["delta_dice"] = _change(candidate.get("dice"), baseline.get("dice"))
                candidate["delta_hd95"] = _change(candidate.get("hd95"), baseline.get("hd95"))
                candidate["delta_assd"] = _change(candidate.get("assd"), baseline.get("assd"))
                if baseline_mask is not None:
                    with Image.open(root / candidate["prediction_mask"]) as image:
                        predicted = np.asarray(image) != 0
                    candidate["prediction_consistency_dice"] = consistency_dice(predicted, baseline_mask)
                    area = int(baseline_mask.sum())
                    candidate["delta_area"] = int(predicted.sum()) / area - 1 if area else None
                else:
                    candidate.update(prediction_consistency_dice=None, delta_area=None)
                change = candidate["delta_dice"]
                candidate["exploratory_dice_loss_gt_0_02"] = bool(change < -0.02) if change is not None else None
                candidate["boundary_status"] = candidate.get("metric_status")
            else:
                for key in ("dice", "hd95", "assd", "iou", "precision", "recall", "boundary_f1", "delta_dice", "delta_hd95", "delta_assd", "delta_area", "prediction_consistency_dice"):
                    candidate[key] = None
                candidate["exploratory_dice_loss_gt_0_02"] = None
            candidate["hd95_margin"] = None
            candidate["joint_constraint_status"] = "not_defined_no_HD95_margin"
            sample["candidates"][condition] = candidate
            flat = dict(case_key=record["patient_key"], image_key=image_key, condition=condition,
                        dataset="BUS-BRA", split="development", **{k: v for k, v in candidate.items()
                        if k not in ("case_key", "image_key", "condition", "dataset", "split", "prediction_mask", "processed_image")})
            long.append(flat)
            for metric in ("dice", "hd95", "assd", "boundary_f1", "output_bytes", "saving_pct", "delta_dice", "delta_hd95", "prediction_consistency_dice", "delta_area", "status"):
                table[condition + "_" + metric] = candidate.get(metric)
        samples.append(sample)
        wide.append(table)
    write_json(root / "features" / "feature_records.json", feature_records)
    write_json(root / "features" / "feature_methods.json", dict(version=FEATURE_VERSION, methods=FEATURE_METHODS,
               uses_reference_mask=False, bit_depth=8, software=dict(numpy=np.__version__), input="original_image_RGB_only"))
    feature_flat = [dict(case_key=r["case_key"], image_key=r["image_key"], dataset="BUS-BRA", split="development",
                    feature_status=r["status"], missing_features=r.get("missing_features", []),
                    feature_version=FEATURE_VERSION, **r["features"]) for r in feature_records]
    write_csv(root / "features" / "iacs_features.csv", feature_flat)
    write_json(root / "metrics" / "sensitivity_samples.json", samples)
    write_json(root / "metrics" / "sensitivity_long.json", long)
    write_csv(root / "metrics" / "compression_sensitivity_long.csv", long)
    write_csv(root / "metrics" / "Compression_Sensitivity_Dataset.csv", wide)
    write_csv(root / "diagnostics" / "original_rule_development_decisions.csv", legacy)
    distributions = {}
    for name in FEATURE_NAMES:
        values = [r["features"].get(name) for r in feature_records if r["features"].get(name) is not None]
        distributions[name] = dict(n=len(values), missing=len(feature_records) - len(values),
                                   quantiles=np.quantile(values, [0, .25, .5, .75, 1]).tolist() if values else None)
    diagnostics = dict(original_smoke_cases=3, original_smoke_iacs_tasks=6, original_smoke_Q85_rule_hits=6,
        original_smoke_fallbacks=0, original_continuous_risk_score="not_implemented",
        current_development_cases=len({s["case_key"] for s in samples}),
        current_original_rule_choices=dict(Counter(t["original_rule_condition"] for t in legacy)),
        current_original_rule_fallbacks=sum(t["fallback"] for t in legacy), feature_distributions=distributions,
        feature_version=FEATURE_VERSION, diagnostic_origin="original rule replay; no feature-outcome tuning",
        hd95_margin=None, dice_candidate=0.02, analysis_unit="Case")
    write_json(root / "diagnostics" / "diagnostic_payload.json", diagnostics)
    write_json(root / "metrics" / "prepared_lock.json", dict(samples_sha256=sha256(root / "metrics" / "sensitivity_samples.json"),
               features_sha256=sha256(root / "features" / "feature_records.json"), feature_code_sha256=sha256(Path(__file__).parent / "features.py"),
               sensitivity_code_sha256=sha256(__file__), collection_lock_sha256=lock["sha256"]))
    append_event(root, "database_prepared", images=len(samples), records=len(long),
                 feature_failures=sum(r["status"] != "ok" for r in feature_records))
    return samples
