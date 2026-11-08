"""Read-only repeat-run checks kept separate from the frozen collector source.

Call validate_config from the CLI before collection. The collector remains an
internal implementation: this module does not load weights, create directories,
change salt keys or alter existing outputs.
"""

import re
from pathlib import Path

from compression_study.datasets import sha256
from compression_study.engine import code_fingerprint, read_results
from .collector import CONDITIONS
from .common import read_json, require_development


def _device(value):
    value = str(value)
    if value == "cpu":
        return value
    if value == "cuda":
        return "cuda:0"
    if re.fullmatch(r"cuda:[0-9]+", value):
        return "cuda:%d" % int(value.split(":")[1])
    raise ValueError("Only explicit cpu or cuda[:nonnegative-index] model devices are supported.")


def _recipe(spec):
    """Canonical actual defaults of compression_study.imaging for our grid."""
    fmt = str(spec.get("format", "baseline")).lower()
    if fmt not in ("baseline", "jpeg", "png"):
        raise ValueError("Reuse recipe format is outside the fixed IACS_v2 collection grid.")
    scale = float(spec.get("scale", 1.0))
    if scale != 1.0:
        raise ValueError("Reuse recipe must use original resolution scale=1.")
    baseline = fmt == "baseline"
    quality = spec.get("quality", 95)
    if quality is None:
        quality = 95
    level = spec.get("compress_level", 6)
    lossless = spec.get("lossless", fmt == "png")
    if fmt == "jpeg" and (lossless or spec.get("subsampling", 0) != 0):
        raise ValueError("Reuse JPEG must be lossy with the audited fixed subsampling=0.")
    if fmt == "png" and lossless is not True:
        raise ValueError("Reuse PNG must explicitly or by default be lossless.")
    return dict(id=str(spec["id"]), format=fmt, scale=scale,
                quality=int(quality) if fmt == "jpeg" else None,
                order="original" if baseline else spec.get("order", "resize_encode"),
                interpolation=None if baseline else spec.get("interpolation", "lanczos"),
                compress_level=int(level) if fmt == "png" else None,
                lossless=None if baseline else bool(lossless),
                subsampling=0 if fmt == "jpeg" else None)


def _model_identity(model):
    if not model.get("commit"):
        raise ValueError("An explicit locked SAM commit is required.")
    model_type = model.get("type", "vit_b")
    if model_type not in ("vit_b", "vit_l", "vit_h"):
        raise ValueError("Unsupported SAM v1 model type.")
    return dict(commit=str(model["commit"]), type=model_type,
                source=str(Path(model["source"]).resolve()) if model.get("source") else None)


def validate_config(config):
    """Fail early on source model/recipe disagreement; return aggregate checks.

    No patient/Case identifiers are included in the returned record. Reuse source
    image and artifact integrity are subsequently checked by the frozen collector.
    """
    require_development(config)
    model = config["model"]
    actual_hash = sha256(model["checkpoint"])
    expected_hash = str(model["sha256"]).lower()
    if actual_hash != expected_hash:
        raise ValueError("Actual model weight SHA256 differs from the locked configuration.")
    identity = _model_identity(model)
    device = _device(config.get("device", model.get("device", "cuda")))
    expected_code = code_fingerprint()
    recipes = {spec["id"]: _recipe(spec) for spec in CONDITIONS}
    wanted = config.get("conditions", list(recipes))
    if len(set(wanted)) != len(wanted) or not set(wanted).issubset(recipes) or "baseline" not in wanted:
        raise ValueError("Collection condition IDs must be a unique grid subset containing baseline.")
    runs = []
    for path in config.get("reuse_runs", []):
        provenance = read_json(Path(path) / "run.json")["provenance"]
        old_config = provenance["config"]
        if provenance["code"] != expected_code:
            raise ValueError("Reuse code fingerprint differs from the current audited legacy package.")
        if str(provenance["checkpoint_sha256"]).lower() != expected_hash:
            raise ValueError("Reuse provenance weight hash differs from the requested model.")
        old_model = old_config["model"]
        if str(old_model.get("sha256", "")).lower() != expected_hash or _model_identity(old_model) != identity:
            raise ValueError("Reuse model configuration commit/type/source/weight differs.")
        old_device = _device(old_config.get("device", old_model.get("device", "cuda")))
        if old_device != device:
            raise ValueError("Reuse inference device differs; use a separate device-specific collection.")
        old_specs = {spec["id"]: spec for spec in old_config["conditions"]}
        if len(old_specs) != len(old_config["conditions"]):
            raise ValueError("Reuse source contains duplicate recipe IDs.")
        checked = 0
        for row in read_results(path):
            condition = row.get("condition")
            if (row.get("status") != "ok" or row.get("dataset") != "BUS-BRA"
                    or row.get("split") != "development" or row.get("prompt_strategy") != "mechanism_gt_box"
                    or condition not in wanted):
                continue
            if condition not in old_specs or _recipe(old_specs[condition]) != recipes[condition]:
                raise ValueError("Reuse candidate recipe parameters differ from the collection grid.")
            expected = recipes[condition]
            if row.get("format", "baseline") != expected["format"] or float(row.get("scale", 1.0)) != expected["scale"]:
                raise ValueError("Reuse result format/scale disagrees with its recipe.")
            if expected["format"] == "jpeg" and row.get("quality") != expected["quality"]:
                raise ValueError("Reuse result quality disagrees with its recipe.")
            checked += 1
        runs.append(dict(eligible_candidate_rows_checked=checked, code_match=True, model_match=True,
                         recipe_match=True, inference_device=device))
    return dict(status="ok", development_only=True, weight_sha256=actual_hash,
                model_commit=identity["commit"], model_type=identity["type"], device=device,
                reuse_runs=runs, reuse_candidate_rows_checked=sum(run["eligible_candidate_rows_checked"] for run in runs),
                no_weights_loaded=True, no_files_modified=True)
