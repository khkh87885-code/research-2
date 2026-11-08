import numpy as np
from PIL import Image


def map_box(box, original_size, processed_size):
    ow, oh = original_size
    nw, nh = processed_size
    if min(ow, oh, nw, nh) <= 0:
        raise ValueError("Invalid image dimensions")
    box = np.asarray(box, dtype=np.float32)
    if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Invalid xyxy box")
    return np.clip(box * [nw / ow, nh / oh, nw / ow, nh / oh], 0, [nw, nh, nw, nh]).astype(np.float32)


def restore_mask(mask, original_size):
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("Prediction mask must be 2D")
    return np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(tuple(original_size), Image.Resampling.NEAREST)) != 0


def application_box(rgb):
    """Fixed central 80% field. No mask input; NOT a validated lesion detector."""
    height, width = rgb.shape[:2]
    return np.asarray([.1 * width, .1 * height, .9 * width, .9 * height], dtype=np.float32)
