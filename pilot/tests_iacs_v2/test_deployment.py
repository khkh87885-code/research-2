import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from compression_study.datasets import sha256
from iacs_v2.features import FEATURE_NAMES
from iacs_v2.validation import fit_policy, QUALITIES
from iacs_v2.deployment import decide_image


def test_image_only_deployment_and_original_preservation(tmp_path):
    source_root, model_root, project = [tmp_path / name for name in ("source", "models", "project")]
    for root in (source_root, model_root, project):
        root.mkdir()
    source = source_root / "image.png"
    Image.fromarray(np.random.default_rng(4).integers(0, 255, (20, 30, 3), dtype=np.uint8)).save(source)
    samples = []
    for i in range(6):
        candidates = {"baseline": dict(status="ok", dice=.8, hd95=2, output_bytes=1000)}
        candidates.update({q: dict(status="ok", dice=.8, hd95=2, output_bytes=500) for q in QUALITIES})
        samples.append(dict(case_key=str(i), image_key=str(i), dataset="BUS-BRA", split="development",
                            features={name: float(i) for name in FEATURE_NAMES}, candidates=candidates))
    policy = fit_policy(samples, "B_logistic")
    policy_file = tmp_path / "policy.json"
    policy_file.write_text(json.dumps(policy))
    run = project / "research_output" / "IACS_v2" / "run"
    config = dict(project_root=str(project), run_root=str(run), datasets=[dict(root=str(source_root))],
                  model=dict(checkpoint=str(model_root / "weights")))
    digest = sha256(source)
    result = decide_image(source, policy_file, run / "deployment" / "new", config)
    assert sha256(source) == digest
    assert not result["uses_reference_at_inference"] and not result["uses_model_prediction_at_inference"]
    assert len(result["encoding_metadata"]) == 7
    assert len(result["predictions"]) == 7
    with pytest.raises(FileExistsError):
        decide_image(source, policy_file, run / "deployment" / "new", config)
