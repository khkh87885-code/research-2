"""Apply a frozen research policy using the original image and encodings only."""
from pathlib import Path

from compression_study.imaging import load_rgb, transform_image
from compression_study.datasets import sha256, inside, check_output
from .collector import CONDITIONS
from .features import extract_record, FEATURE_VERSION
from .validation import predict_policy, verify_policy
from .common import read_json, write_json, safe_run


def decide_image(image_path, policy_path, output_dir, config):
    """No mask, SAM prediction or observed segmentation metrics are inputs."""
    source = Path(image_path).resolve(strict=True)
    policy = read_json(policy_path)
    verify_policy(policy)
    if policy.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Feature extraction version differs from frozen policy")
    root = safe_run(config)
    output = check_output(output_dir, config)
    if not inside(output, root / "deployment") or output == root / "deployment":
        raise ValueError("Deployment output must be a new version inside run/deployment")
    if output.exists():
        raise FileExistsError("Deployment output already exists")
    original_hash = sha256(source)
    rgb = load_rgb(source)
    features = extract_record(rgb)
    output.mkdir(parents=True)
    candidates = {"baseline": dict(output_bytes=source.stat().st_size, original_bytes=source.stat().st_size, pixel_equal=True)}
    for spec in CONDITIONS:
        if spec["id"] == "baseline":
            continue
        destination = output / (spec["id"] + (".jpg" if spec["format"] == "jpeg" else ".png"))
        _, metadata = transform_image(rgb, spec, destination)
        candidates[spec["id"]] = dict(output_bytes=metadata["output_bytes"], original_bytes=source.stat().st_size,
                                      pixel_equal=metadata["pixel_equal"], encode_ms=metadata["encode_ms"], decode_ms=metadata["decode_ms"])
    decision = predict_policy(policy, features["features"], candidates)
    decision.update(source_sha256=original_hash, policy_sha256=policy["policy_sha256"],
                    feature_version=FEATURE_VERSION, features=features, encoding_metadata=candidates,
                    uses_reference_at_inference=False, uses_model_prediction_at_inference=False,
                    segmentation_performance="not_evaluated; image-only engineering decision")
    if sha256(source) != original_hash:
        raise RuntimeError("Original image changed")
    write_json(output / "decision.json", decision)
    return decision
