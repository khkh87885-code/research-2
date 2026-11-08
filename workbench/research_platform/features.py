"""Whole-image features; this API intentionally has no reference-mask input."""
import numpy as np
from scipy import ndimage
from .datasets import load_rgb


VERSION = "whole-image-v1.0"


def extract_features(image_path):
    rgb = load_rgb(image_path)
    gray = np.dot(rgb.astype(np.float64), [.299, .587, .114]) / 255.
    residual = gray - ndimage.median_filter(gray, size=3)
    local_mean = ndimage.uniform_filter(gray, size=9)
    local_var = np.maximum(0, ndimage.uniform_filter(gray**2, size=9) - local_mean**2)
    gradient = np.hypot(ndimage.sobel(gray, axis=0), ndimage.sobel(gray, axis=1)) / 8
    hist = np.histogram(gray, bins=32, range=(0, 1))[0].astype(float)
    probs = hist[hist > 0] / hist.sum()
    methods = {
        "sharpness": (float(ndimage.laplace(gray).var()), "variance of discrete Laplacian / normalized gray"),
        "noise": (float(np.median(np.abs(residual - np.median(residual))) / .67448975), "median residual MAD / 0.67448975; proxy in speckled ultrasound"),
        "local_contrast": (float(np.sqrt(local_var).mean()), "mean local standard deviation 9x9"),
        "edge_complexity": (float(np.mean(gradient > .05)), "Sobel gradient magnitude > 0.05 fraction"),
        "texture_entropy": (float(-np.sum(probs * np.log2(probs))), "32-bin gray histogram Shannon entropy"),
        "gray_mean": (float(gray.mean()), "mean normalized luminance"),
        "gray_std": (float(gray.std()), "standard deviation normalized luminance"),
        "dynamic_range": (float(gray.max() - gray.min()), "normalized luminance max-min"),
        "width": (int(gray.shape[1]), "decoded pixel width"),
        "height": (int(gray.shape[0]), "decoded pixel height"),
        "pixels": (int(gray.size), "width*height")}
    return {"version": VERSION, "uses_reference_mask": False, "values": {k: v[0] for k, v in methods.items()},
            "features": [{"Feature_Name": k, "Feature_Value": v, "Feature_Extraction_Method": method,
                          "Feature_Extraction_Version": VERSION} for k, (v, method) in methods.items()]}
