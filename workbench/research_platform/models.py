import importlib.util
import os
import random
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from .common import sha256


def legacy_prompt_boxes(config, reference):
    path = Path(config["pilot_root"]) / "run_single_case.py"
    spec = importlib.util.spec_from_file_location("audited_original_single_case", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not np.asarray(reference).any():
        raise ValueError("Cannot build mechanism prompt from empty reference")
    return module.prompt_boxes(reference)[0]


class SamModel:
    def __init__(self, config, device="cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        self.torch = torch
        source = Path(config["sam_source"])
        git = ["git", "--no-optional-locks", "-C", str(source)]
        commit = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(git + ["status", "--porcelain"], text=True).strip()
        if commit != config["sam_commit"] or dirty:
            raise RuntimeError("SAM source commit or clean-tree check failed")
        digest = sha256(config["checkpoint"])
        if digest != config["checkpoint_sha256"]:
            raise RuntimeError("Model weight SHA-256 mismatch")
        if device not in ("cuda", "cpu"):
            raise ValueError("device must be cuda or cpu")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; explicitly select CPU")
        random.seed(config["random_seed"])
        np.random.seed(config["random_seed"])
        torch.manual_seed(config["random_seed"])
        # SAM positional grid uses CUDA cumsum, unavailable under strict determinism in torch 2.4.1.
        # Request deterministic kernels when available and disclose this reproducibility limitation.
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
        self.device = torch.device(device)
        sys.path.insert(0, str(source))
        from segment_anything import SamPredictor, sam_model_registry
        started = perf_counter()
        model = sam_model_registry["vit_b"]()
        model.load_state_dict(torch.load(config["checkpoint"], map_location="cpu", weights_only=True), strict=True)
        model.to(self.device).eval()
        self.predictor = SamPredictor(model)
        self.sync()
        self.info = {"model": "SAM", "model_version": "v1-vit_b", "weight_sha256": digest,
                     "sam_commit": commit, "device": str(self.device), "dtype": "float32",
                     "multimask_output": False, "postprocessing": "SAM threshold; nearest-neighbor to original size",
                     "model_load_ms": 1000 * (perf_counter() - started), "deterministic_algorithms": "requested_warn_only",
                     "determinism_limitations": "CUDA cumsum in original SAM positional encoding has no strict deterministic implementation in torch 2.4.1; bitwise cross-device reproducibility is not guaranteed"}

    def sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def predict(self, rgb, box):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("SAM input must be uint8 RGB")
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)
        self.sync()
        started = perf_counter()
        try:
            with self.torch.inference_mode():
                self.predictor.set_image(rgb)
                masks, scores, _ = self.predictor.predict(box=box, multimask_output=False)
            self.sync()
            elapsed = 1000 * (perf_counter() - started)
            if masks.shape != (1, rgb.shape[0], rgb.shape[1]):
                raise RuntimeError("Unexpected SAM mask geometry")
            return masks[0].astype(bool), {"inference_ms": elapsed, "sam_predicted_iou": float(scores[0]),
                "gpu_peak_mib": self.torch.cuda.max_memory_allocated(self.device) / 1024**2 if self.device.type == "cuda" else None}
        finally:
            self.predictor.reset_image()
