"""Use the existing SAM v1 source, weights and prompt implementation."""
import importlib
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .datasets import sha256


def legacy_module(name):
    project = str(Path(__file__).resolve().parent.parent)
    if project not in sys.path:
        sys.path.insert(0, project)
    return importlib.import_module(name)


def make_prompt(rgb, reference, strategy):
    if strategy == "mechanism_gt_box":
        return legacy_module("run_single_case").prompt_boxes(reference)[0]
    if strategy == "application_full_frame":
        # Explicit fixed, GT-free comparator; not a validated lesion detector.
        h, w = rgb.shape[:2]
        return np.array([0, 0, w, h], dtype=np.float32)
    raise ValueError("Unknown prompt strategy")


class ExistingSAM:
    def __init__(self, config):
        source = Path(config["source"]).resolve(strict=True)
        commit = subprocess.check_output(["git", "--no-optional-locks", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "--no-optional-locks", "-C", str(source), "status", "--porcelain"], text=True).strip()
        if commit != config["commit"] or dirty:
            raise ValueError("SAM source differs from locked clean commit")
        checkpoint = Path(config["checkpoint"]).resolve(strict=True)
        digest = sha256(checkpoint)
        if digest != config["sha256"]:
            raise ValueError("SAM weights differ from locked SHA256")
        sys.path.insert(0, str(source))
        from segment_anything import SamPredictor, sam_model_registry
        self.device = config.get("device", "cuda")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; explicitly choose cpu to run on CPU")
        model = sam_model_registry[config.get("type", "vit_b")]()
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
        model.eval().to(self.device)
        self.predictor = SamPredictor(model)
        self.info = dict(name="SAM-v1", type=config.get("type", "vit_b"), commit=commit,
                         checkpoint_sha256=digest, torch=torch.__version__, device=self.device,
                         multimask_output=False, postprocessing="none")

    def predict(self, rgb, box):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        start = perf_counter()
        try:
            with torch.inference_mode():
                self.predictor.set_image(rgb)
                masks, scores, _ = self.predictor.predict(box=box, multimask_output=False)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize(self.device)
            if masks.shape != (1, rgb.shape[0], rgb.shape[1]):
                raise RuntimeError("Unexpected SAM output dimensions")
            if not np.isfinite(scores).all():
                raise RuntimeError("Nonfinite model confidence output")
            return masks[0].astype(bool), dict(inference_ms=1000 * (perf_counter() - start),
                sam_predicted_iou=float(scores[0]), gpu_peak_bytes=(int(torch.cuda.max_memory_allocated(self.device))
                if self.device.startswith("cuda") else None))
        finally:
            self.predictor.reset_image()
