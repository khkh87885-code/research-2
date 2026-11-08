"""Executable candidate IACS; performance is explicitly unvalidated."""
from pathlib import Path
import numpy as np
from .common import write_json, read_json, sha256, now
from .features import VERSION


def calibrate(records, feature_records, destination):
    if not records or len(records) != len(feature_records):
        raise ValueError("Calibration requires matched development features")
    if any(r["dataset"] != "BUS-BRA" or r["split"] != "development" for r in records):
        raise ValueError("IACS calibration is restricted to BUS-BRA development data")
    if any(f.get("uses_reference_mask") for f in feature_records):
        raise ValueError("Reference-derived decision features prohibited")
    if any(f.get("version") != VERSION for f in feature_records):
        raise ValueError("Feature extraction version mismatch")
    values = [f["values"] for f in feature_records]
    keys = ["noise", "sharpness", "local_contrast", "edge_complexity"]
    if any(not np.isfinite([f[k] for k in keys]).all() for f in values):
        raise ValueError("Non-finite calibration features")
    config = {"version": "IACS-candidate-0.1", "state": "candidate_unvalidated", "created_at": now(),
        "feature_version": VERSION, "calibration_dataset": "BUS-BRA", "calibration_split": "development",
        "calibration_images": len(records), "calibration_patient_ids": sorted(set(r["patient_id"] for r in records)),
        "calibration_image_hashes": [r["image_sha256"] for r in records],
        "quantiles": {k: np.quantile([f[k] for f in values], [.25, .75]).tolist() for k in keys},
        "rules": {"low": "cr2_q75_s75", "medium": "jpeg_q85", "high": "png"},
        "limitations": "Feature-quantile engineering heuristic. No sensitivity-outcome optimization or held-out strategy validation has been completed.",
        "quality_constraints_verified": False, "dice_margin": None, "hd95_margin": None}
    write_json(destination, config, exclusive=True)
    return config


def decide(features, config):
    if features["version"] != config["feature_version"] or features.get("uses_reference_mask"):
        raise ValueError("Invalid decision features")
    values, q = features["values"], config["quantiles"]
    if not np.isfinite([values[k] for k in ("noise", "sharpness", "local_contrast", "edge_complexity")]).all():
        raise ValueError("Non-finite decision features; aggressive compression prohibited")
    risk = int(values["noise"] >= q["noise"][1]) + int(values["sharpness"] <= q["sharpness"][0])
    risk += int(values["local_contrast"] <= q["local_contrast"][0]) + int(values["edge_complexity"] >= q["edge_complexity"][1])
    level = "high" if risk >= 3 else "medium" if risk >= 1 else "low"
    return {"risk_score": risk, "risk_level": level, "selected_variant": config["rules"][level],
            "algorithm_version": config["version"], "algorithm_state": config["state"], "validated": False}


def lock_candidate(path, protocol):
    config = read_json(path)
    # A user-entered reason is not a substitute for validation evidence or pre-specified margins.
    if not config.get("quality_constraints_verified") or not protocol.get("formal_protocol_locked"):
        raise ValueError("Cannot lock IACS: development optimization, strategy validation and pre-specified quality margins are still pending")
    if config.get("dice_margin") is None or config.get("hd95_margin") is None:
        raise ValueError("Both quality margins must be specified before locking")
    if any(not np.isfinite(float(config[k])) or float(config[k]) <= 0 for k in ("dice_margin", "hd95_margin")):
        raise ValueError("Quality margins must be positive finite values")
    lock_path = Path(path).with_name("iacs_lock_" + sha256(path)[:12] + ".json")
    locked = dict(config, state="locked")
    write_json(lock_path, {"configuration": locked, "sha256": sha256(path), "locked_at": now()}, exclusive=True)
    return str(lock_path)
