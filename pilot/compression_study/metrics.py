"""Binary segmentation metrics with explicit undefined-value policies.

Surface distances use 4-connected pixel boundaries and Euclidean distance.
HD95 = max(p95(pred->ref), p95(ref->pred)); ASSD = half the sum of
the two directional mean distances. With spacing, (row, column) values
must be millimeters and the tolerance is also millimeters; otherwise pixels.
"""

import numpy as np
from scipy import ndimage


def _binary_mask(value):
    array = np.asarray(value)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError("Expected non-empty 2-D masks.")
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise ValueError("Masks must be numeric or boolean.")
    if not np.isfinite(array).all():
        raise ValueError("Masks must not contain NaN or infinity.")
    return array != 0


def evaluate_masks(pred, reference, spacing=None, tolerance=2.0):
    """Evaluate on a common grid; missing distances stay None, never good scores.

    Empty references (including double-empty) are invalid positive-lesion
    references: all accuracy and boundary metrics are None. An empty prediction
    against a non-empty reference has Dice/IoU/recall=0, undefined precision and
    surface metrics, and relative area error=1. Relative area error is absolute.
    """
    pred, reference = _binary_mask(pred), _binary_mask(reference)
    if pred.shape != reference.shape:
        raise ValueError("Prediction and reference must use the same coordinate grid.")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Boundary tolerance must be finite and nonnegative.")
    sampling = (1.0, 1.0)
    distance_unit = "pixel"
    if spacing is not None:
        sampling = tuple(float(value) for value in spacing)
        if len(sampling) != 2 or not all(np.isfinite(value) and value > 0 for value in sampling):
            raise ValueError("Spacing must be two positive finite millimeter values (row,column).")
        distance_unit = "mm"
    predicted_area, reference_area = int(pred.sum()), int(reference.sum())
    result = dict.fromkeys(("dice", "iou", "precision", "recall", "sensitivity", "hd95", "assd", "boundary_f1", "relative_area_error"))
    result.update(empty_prediction=not bool(predicted_area), empty_reference=not bool(reference_area), distance_unit=distance_unit, boundary_tolerance=tolerance, predicted_area_pixels=predicted_area, reference_area_pixels=reference_area)
    if not reference_area:
        result["metric_status"] = "both_empty" if not predicted_area else "empty_reference"
        return result
    intersection = int(np.count_nonzero(pred & reference))
    result.update(
        dice=2.0 * intersection / (predicted_area + reference_area),
        iou=float(intersection) / (predicted_area + reference_area - intersection),
        precision=float(intersection) / predicted_area if predicted_area else None,
        recall=float(intersection) / reference_area,
        sensitivity=float(intersection) / reference_area,
        relative_area_error=abs(predicted_area - reference_area) / float(reference_area),
    )
    if not predicted_area:
        result["metric_status"] = "empty_prediction_surface_undefined"
        return result
    connectivity = ndimage.generate_binary_structure(2, 1)
    pred_surface = pred & ~ndimage.binary_erosion(pred, structure=connectivity, border_value=0)
    reference_surface = reference & ~ndimage.binary_erosion(reference, structure=connectivity, border_value=0)
    pred_to_reference = ndimage.distance_transform_edt(~reference_surface, sampling=sampling)[pred_surface]
    reference_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=sampling)[reference_surface]
    boundary_precision = float(np.mean(pred_to_reference <= tolerance))
    boundary_recall = float(np.mean(reference_to_pred <= tolerance))
    boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall) if boundary_precision + boundary_recall else 0.0
    result.update(
        hd95=float(max(np.percentile(pred_to_reference, 95), np.percentile(reference_to_pred, 95))),
        assd=float(0.5 * (pred_to_reference.mean() + reference_to_pred.mean())),
        boundary_f1=boundary_f1, metric_status="ok",
    )
    return result
