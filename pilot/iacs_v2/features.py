"""Versioned, reference-free whole-image descriptors for IACS_v2.

Only the original uint8 RGB array is accepted. No mask, outcome, Case identifier
or candidate reconstruction is an input. These are numerical image descriptors,
not calibrated estimates of ultrasound noise or clinical image quality.
"""

import math

import numpy as np
from scipy import ndimage


FEATURE_VERSION = "full_image_iacs_v2_1"
METHOD_VERSION = FEATURE_VERSION
GLCM_LEVELS = 32
GLCM_DISTANCES = (1, 2)
GLCM_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
EDGE_THRESHOLD = 0.05

FEATURE_METHODS = {
    "laplacian_variance": "Variance of 4-neighbor discrete Laplacian; reflected boundary; gray in [0,1].",
    "tenengrad": "Mean squared Sobel gradient magnitude; each Sobel derivative divided by 8.",
    "gradient_mean": "Mean normalized Sobel gradient magnitude.",
    "gradient_p90": "90th percentile of normalized Sobel gradient magnitude.",
    "gradient_direction_entropy": "8-bin [0,2pi) direction entropy in bits, weighted by nonzero gradient magnitude.",
    "noise_mad": "Laplacian MAD / (0.6744897501960817 * sqrt(20)); uncalibrated robust noise proxy.",
    "residual_mad": "MAD of gray minus 3x3 reflected median filter / 0.6744897501960817.",
    "high_frequency_ratio": "Variance of gray minus sigma=1 reflected Gaussian / gray variance; constant=0.",
    "global_contrast": "Population standard deviation of normalized grayscale.",
    "local_contrast": "Mean local standard deviation in reflected 9x9 windows.",
    "p01": "1st percentile of normalized grayscale.",
    "p99": "99th percentile of normalized grayscale.",
    "intensity_range": "p99 minus p01 of normalized grayscale.",
    "dynamic_range": "Maximum minus minimum normalized grayscale.",
    "gray_mean": "Mean normalized grayscale; ITU-R BT.601 weights 0.299,0.587,0.114.",
    "gray_std": "Population standard deviation of normalized grayscale.",
    "gray_skew": "Third standardized central moment; a constant image has skew=0 by explicit convention.",
    "edge_density": "Fraction of pixels with normalized Sobel gradient magnitude > 0.05.",
    "edge_direction_change": "Mean circular angular change / pi between horizontal/vertical neighboring edge pixels; no pairs=0.",
    "glcm_contrast": "32-bin symmetric GLCM sum(P*(i-j)^2); mean over valid distance1/2, four directions.",
    "glcm_homogeneity": "32-bin symmetric GLCM sum(P/(1+(i-j)^2)); mean over valid offsets; constant=1.",
    "glcm_entropy": "32-bin symmetric GLCM Shannon entropy in bits; mean over valid offsets.",
    "histogram_entropy": "256-bin normalized gray Shannon entropy in bits.",
    "width": "Original decoded image width in pixels.",
    "height": "Original decoded image height in pixels.",
    "aspect_ratio": "Original width divided by original height.",
}
FEATURE_NAMES = tuple(FEATURE_METHODS)


def _check_input(rgb, metadata):
    array = np.asarray(rgb)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("IACS_v2 features require an H x W x 3 uint8 original RGB array.")
    if not array.shape[0] or not array.shape[1]:
        raise ValueError("IACS_v2 features require positive image dimensions.")
    if metadata is not None:
        if not isinstance(metadata, dict) or set(metadata) - {"format", "bit_depth"}:
            raise ValueError("Feature metadata may contain only format and bit_depth; masks/outcomes are forbidden.")
        if "format" in metadata and metadata["format"] is not None and not isinstance(metadata["format"], str):
            raise ValueError("Source format must be a string or None.")
        if "bit_depth" in metadata and metadata["bit_depth"] != 8:
            raise ValueError("Only audited original 8-bit source images are supported.")
    return array


def _entropy(weights):
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if total == 0:
        return 0.0
    probabilities = weights[weights > 0] / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _glcm_features(gray):
    """Equal-weight directional metrics, not one pooled co-occurrence matrix.

    Missing offsets on small images are omitted. With no possible pixel pairs,
    all three texture measurements are genuinely missing and remain None.
    """
    quantized = np.minimum((gray * GLCM_LEVELS).astype(np.int64), GLCM_LEVELS - 1)
    height, width = quantized.shape
    levels = np.arange(GLCM_LEVELS, dtype=np.float64)
    difference_squared = (levels[:, None] - levels[None, :]) ** 2
    homogeneity_weights = 1.0 / (1.0 + difference_squared)
    measurements = []
    for distance in GLCM_DISTANCES:
        for unit_y, unit_x in GLCM_DIRECTIONS:
            offset_y, offset_x = distance * unit_y, distance * unit_x
            y0, y1 = max(0, -offset_y), min(height, height - offset_y)
            x0, x1 = max(0, -offset_x), min(width, width - offset_x)
            if y1 <= y0 or x1 <= x0:
                continue
            first = quantized[y0:y1, x0:x1]
            second = quantized[y0 + offset_y:y1 + offset_y, x0 + offset_x:x1 + offset_x]
            counts = np.bincount((first * GLCM_LEVELS + second).ravel(), minlength=GLCM_LEVELS ** 2)
            counts = counts.reshape(GLCM_LEVELS, GLCM_LEVELS).astype(np.float64)
            symmetric = counts + counts.T
            probabilities = symmetric / symmetric.sum()
            measurements.append((float(np.sum(probabilities * difference_squared)),
                                 float(np.sum(probabilities * homogeneity_weights)),
                                 _entropy(probabilities)))
    if not measurements:
        return dict(glcm_contrast=None, glcm_homogeneity=None, glcm_entropy=None)
    averaged = np.asarray(measurements, dtype=np.float64).mean(axis=0)
    return dict(zip(("glcm_contrast", "glcm_homogeneity", "glcm_entropy"), map(float, averaged)))


def _direction_change(angle, edge):
    changes = []
    for axis in (0, 1):
        if angle.shape[axis] < 2:
            continue
        first, second = (angle[:-1, :], angle[1:, :]) if axis == 0 else (angle[:, :-1], angle[:, 1:])
        eligible = (edge[:-1, :] & edge[1:, :]) if axis == 0 else (edge[:, :-1] & edge[:, 1:])
        if eligible.any():
            difference = np.abs(first[eligible] - second[eligible])
            changes.append(np.minimum(difference, 2 * math.pi - difference) / math.pi)
    return float(np.concatenate(changes).mean()) if changes else 0.0


def extract_features(rgb, metadata=None):
    """Return a flat FEATURE_NAMES dictionary; valid numbers are always finite.

    Unknown source precision and RGB conversion must be checked by the caller's
    audited image reader. Metadata are provenance only, never numeric predictors.
    None is reserved for GLCM measurements when a tiny image has no pixel pairs.
    Such rows require baseline fallback in a selector, not favorable imputation.
    """
    array = _check_input(rgb, metadata)
    gray = np.dot(array.astype(np.float64) / 255.0, [0.299, 0.587, 0.114])
    height, width = gray.shape
    laplacian = ndimage.laplace(gray, mode="reflect")
    gx = ndimage.sobel(gray, axis=1, mode="reflect") / 8.0
    gy = ndimage.sobel(gray, axis=0, mode="reflect") / 8.0
    gradient = np.hypot(gx, gy)
    angles = np.mod(np.arctan2(gy, gx), 2 * math.pi)
    directions = np.minimum((angles / (2 * math.pi) * 8).astype(np.int64), 7)
    direction_weights = np.bincount(directions.ravel(), weights=gradient.ravel(), minlength=8)
    edge = gradient > EDGE_THRESHOLD
    residual = gray - ndimage.median_filter(gray, size=3, mode="reflect")
    highpass = gray - ndimage.gaussian_filter(gray, sigma=1.0, mode="reflect")
    local_mean = ndimage.uniform_filter(gray, size=9, mode="reflect")
    local_variance = np.maximum(0.0, ndimage.uniform_filter(gray * gray, size=9, mode="reflect") - local_mean * local_mean)
    mean, std = float(gray.mean()), float(gray.std())
    p01, p99 = map(float, np.percentile(gray, [1, 99]))
    histogram = np.bincount(np.clip(np.rint(gray * 255), 0, 255).astype(np.uint8).ravel(), minlength=256)
    values = {
        "laplacian_variance": float(laplacian.var()),
        "tenengrad": float(np.mean(gradient ** 2)),
        "gradient_mean": float(gradient.mean()),
        "gradient_p90": float(np.percentile(gradient, 90)),
        "gradient_direction_entropy": _entropy(direction_weights),
        "noise_mad": float(np.median(np.abs(laplacian - np.median(laplacian))) / (0.6744897501960817 * math.sqrt(20.0))),
        "residual_mad": float(np.median(np.abs(residual - np.median(residual))) / 0.6744897501960817),
        "high_frequency_ratio": float(highpass.var() / (std * std)) if std > 1e-12 else 0.0,
        "global_contrast": std,
        "local_contrast": float(np.mean(np.sqrt(local_variance))),
        "p01": p01, "p99": p99, "intensity_range": p99 - p01,
        "dynamic_range": float(gray.max() - gray.min()),
        "gray_mean": mean, "gray_std": std,
        "gray_skew": float(np.mean(((gray - mean) / std) ** 3)) if std > 1e-12 else 0.0,
        "edge_density": float(edge.mean()),
        "edge_direction_change": _direction_change(angles, edge),
        "histogram_entropy": _entropy(histogram),
        "width": float(width), "height": float(height), "aspect_ratio": float(width) / height,
    }
    values.update(_glcm_features(gray))
    if set(values) != set(FEATURE_NAMES):
        raise RuntimeError("Feature schema mismatch.")
    result = {name: values[name] for name in FEATURE_NAMES}
    if any(value is not None and not math.isfinite(value) for value in result.values()):
        raise ValueError("Feature extraction produced a non-finite value; no imputation applied.")
    return result


def extract_record(rgb, format=None, bit_depth=8):
    """Return feature values and separate source/method metadata for safe storage."""
    values = extract_features(rgb, metadata={"format": format, "bit_depth": bit_depth})
    missing = [name for name in FEATURE_NAMES if values[name] is None]
    return {
        "features": values,
        "metadata": {"feature_version": FEATURE_VERSION, "method_version": METHOD_VERSION,
                     "source_format": format, "source_bit_depth": bit_depth,
                     "uses_reference_mask": False},
        "status": "missing_features" if missing else "ok",
        "missing_features": missing,
    }
