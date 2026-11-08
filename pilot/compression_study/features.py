"""Fixed full-image intensity features; no masks, labels or patient information."""

import numpy as np
from scipy import ndimage


FEATURE_VERSION = "full_image_v1"


def extract_features(rgb):
    """Return finite scalars from uint8 RGB; intensities are normalized to [0,1].

    ``noise`` is a robust Laplacian residual estimate, not a calibrated physical
    noise measure. ``edge_density`` uses a fixed normalized gradient threshold
    of .08, and ``contrast`` is mean 9x9 local standard deviation.
    """
    array = np.asarray(rgb)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3 or not array.shape[0] or not array.shape[1]:
        raise ValueError("Feature extraction requires H x W x 3 uint8 RGB.")
    gray = np.dot(array.astype(np.float64) / 255.0, [0.299, 0.587, 0.114])
    laplacian = ndimage.laplace(gray, mode="reflect")
    local_mean = ndimage.uniform_filter(gray, size=9, mode="reflect")
    local_variance = np.maximum(0, ndimage.uniform_filter(gray * gray, size=9, mode="reflect") - local_mean * local_mean)
    gx = ndimage.sobel(gray, axis=1, mode="reflect") / 8.0
    gy = ndimage.sobel(gray, axis=0, mode="reflect") / 8.0
    gradient = np.hypot(gx, gy)
    histogram = np.bincount(np.clip(np.rint(gray * 255), 0, 255).astype(np.uint8).ravel(), minlength=256).astype(float)
    probabilities = histogram[histogram > 0] / gray.size
    return {
        "width": float(array.shape[1]), "height": float(array.shape[0]),
        "clarity": float(np.var(laplacian)),
        "noise": float(np.median(np.abs(laplacian - np.median(laplacian))) / (0.6744897501960817 * np.sqrt(20.0))),
        "contrast": float(np.mean(np.sqrt(local_variance))),
        "global_contrast": float(np.std(gray)),
        "edge_density": float(np.mean(gradient > 0.08)),
        "edge_strength": float(np.mean(gradient)),
        "entropy": float(-np.sum(probabilities * np.log2(probabilities))),
        "gray_mean": float(np.mean(gray)),
        "dynamic_range": float(np.max(gray) - np.min(gray)),
    }
