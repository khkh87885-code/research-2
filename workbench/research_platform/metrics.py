"""Auditable binary segmentation and file-size metrics.

Distances use 4-connected inner boundary pixels. HD95 is the 95th percentile
of the pooled two directed boundary-distance arrays; ASSD is their pooled
mean (surface-pixel weighted, not a mean of two means). ``spacing`` is
(row_mm, column_mm); without spacing all boundary distances and tolerance
are in pixels. Relative area error is signed: (prediction-reference)/reference.
Undefined quantities are None with explicit reasons, never filled with 0/1.
"""

import math

import numpy as np
from scipy import ndimage


METRIC_NAMES = (
    "dice", "iou", "precision", "recall", "sensitivity", "hd95", "assd",
    "boundary_f1", "relative_area_error",
)


def _mask(value, name):
    array = np.asarray(value)
    if array.ndim != 2 or not array.size:
        raise ValueError("%s must be a nonempty two-dimensional binary mask" % name)
    if array.dtype != np.bool_ and not np.isin(array, (0, 1)).all():
        raise ValueError("%s contains non-binary values" % name)
    return array.astype(bool, copy=False)


def _pair(reference, prediction):
    reference = _mask(reference, "reference")
    prediction = _mask(prediction, "prediction")
    if reference.shape != prediction.shape:
        raise ValueError("Masks must be mapped to the same original coordinates")
    return reference, prediction


def _sampling(spacing):
    if spacing is None:
        return (1.0, 1.0), "pixel"
    values = tuple(float(x) for x in spacing)
    if len(values) != 2 or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("spacing must contain two positive finite millimetre values")
    return values, "mm"


def _boundary(mask):
    return mask & ~ndimage.binary_erosion(
        mask, structure=ndimage.generate_binary_structure(2, 1), border_value=0,
    )


def evaluate(reference, prediction, spacing=None, boundary_tolerance=2.0):
    """Evaluate original-coordinate binary masks; see module-level conventions."""
    reference, prediction = _pair(reference, prediction)
    sampling, unit = _sampling(spacing)
    tolerance = float(boundary_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("boundary_tolerance must be finite and nonnegative")
    ref_area, pred_area = int(reference.sum()), int(prediction.sum())
    result = {name: None for name in METRIC_NAMES}
    result.update(
        metric_status="valid", metric_reasons={}, boundary_unit=unit,
        boundary_tolerance=tolerance, reference_area_pixels=ref_area,
        prediction_area_pixels=pred_area,
        area_error_definition="signed (prediction-reference)/reference",
        boundary_definition="4-connected inner pixel boundary",
        hd95_definition="95th percentile of pooled bidirectional distances",
        assd_definition="mean of pooled bidirectional distances",
    )
    if not ref_area:
        result["metric_status"] = "invalid_reference_empty"
        result["metric_reasons"] = {name: "empty_reference" for name in METRIC_NAMES}
        return result
    intersection = int(np.count_nonzero(reference & prediction))
    result.update(
        dice=2.0 * intersection / (ref_area + pred_area),
        iou=intersection / float(ref_area + pred_area - intersection),
        precision=intersection / float(pred_area) if pred_area else None,
        recall=intersection / float(ref_area),
        sensitivity=intersection / float(ref_area),
        relative_area_error=(pred_area - ref_area) / float(ref_area),
    )
    if not pred_area:
        result["metric_status"] = "partial_empty_prediction"
        result["metric_reasons"] = {
            name: "empty_prediction" for name in ("precision", "hd95", "assd", "boundary_f1")
        }
        return result
    reference_boundary, prediction_boundary = _boundary(reference), _boundary(prediction)
    to_reference = ndimage.distance_transform_edt(~reference_boundary, sampling=sampling)[prediction_boundary]
    to_prediction = ndimage.distance_transform_edt(~prediction_boundary, sampling=sampling)[reference_boundary]
    distances = np.concatenate((to_reference, to_prediction))
    boundary_precision = float(np.mean(to_reference <= tolerance))
    boundary_recall = float(np.mean(to_prediction <= tolerance))
    denominator = boundary_precision + boundary_recall
    result.update(
        hd95=float(np.percentile(distances, 95)), assd=float(np.mean(distances)),
        boundary_f1=2.0 * boundary_precision * boundary_recall / denominator if denominator else 0.0,
    )
    return result


def stability(base, prediction, spacing=None, boundary_tolerance=2.0):
    """Compare predictions without treating two empty outputs as perfect agreement."""
    base, prediction = _pair(base, prediction)
    evaluation = evaluate(base, prediction, spacing, boundary_tolerance)
    base_area, pred_area = int(base.sum()), int(prediction.sum())
    both_empty = not base_area and not pred_area
    dice_value = None if both_empty else 2.0 * np.count_nonzero(base & prediction) / (base_area + pred_area)
    union = np.count_nonzero(base | prediction)
    return {
        "consistency_dice": None if both_empty else float(dice_value),
        "consistency_iou": float(np.count_nonzero(base & prediction) / union) if union else None,
        "prediction_area_change_rate": (pred_area - base_area) / float(base_area) if base_area else None,
        "boundary_hd95": evaluation["hd95"], "boundary_assd": evaluation["assd"],
        "boundary_unit": evaluation["boundary_unit"],
        "baseline_empty": not bool(base_area), "prediction_empty": not bool(pred_area),
        "metric_status": "undefined_both_empty" if both_empty else evaluation["metric_status"],
        "metric_reasons": evaluation["metric_reasons"],
    }


def compression_efficiency(original_bytes, compressed_bytes, bandwidth_mbps=10, rtt_ms=0):
    """Size ratio and simulated transfer (one RTT + bytes*8/bandwidth), not measured upload."""
    original, compressed = float(original_bytes), float(compressed_bytes)
    bandwidth, rtt = float(bandwidth_mbps), float(rtt_ms)
    if any(not math.isfinite(x) or x <= 0 for x in (original, compressed, bandwidth)):
        raise ValueError("File sizes and bandwidth must be positive and finite")
    if not original.is_integer() or not compressed.is_integer():
        raise ValueError("File sizes must be whole bytes")
    if not math.isfinite(rtt) or rtt < 0:
        raise ValueError("rtt_ms must be finite and nonnegative")
    ratio = compressed / original
    original_seconds = original * 8 / (bandwidth * 1000000) + rtt / 1000
    compressed_seconds = compressed * 8 / (bandwidth * 1000000) + rtt / 1000
    return {
        "original_bytes": int(original), "compressed_bytes": int(compressed),
        "compression_ratio": ratio, "compression_saving_pct": (1.0 - ratio) * 100,
        "bandwidth_mbps": bandwidth, "rtt_ms": rtt,
        "simulated_upload_time_s": compressed_seconds,
        "simulated_original_upload_time_s": original_seconds,
        "simulated_upload_time_saving_s": original_seconds - compressed_seconds,
        "upload_time_method": "one RTT + file_bytes*8/(bandwidth_mbps*1000000); excludes protocol overhead",
    }
