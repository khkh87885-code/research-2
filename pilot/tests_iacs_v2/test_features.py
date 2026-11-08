"""Analytic and reference-free IACS_v2 descriptors; fixtures are software tests."""

import hashlib
import inspect

import numpy as np
import pytest
from scipy import ndimage

from iacs_v2.features import FEATURE_METHODS, FEATURE_NAMES, FEATURE_VERSION, extract_features, extract_record


def rgb(gray):
    return np.repeat(np.asarray(gray, dtype=np.uint8)[..., None], 3, axis=2)


def test_manual_glcm_two_by_two_checkerboard():
    values = extract_features(rgb([[0, 255], [255, 0]]))
    # Horizontal/vertical pairs are opposite; both diagonals have equal pixels.
    # d=2 has no available pair. Each of four valid directions has equal weight.
    assert values["glcm_contrast"] == pytest.approx(31 ** 2 / 2)
    assert values["glcm_homogeneity"] == pytest.approx((1 + 1 / (1 + 31 ** 2)) / 2)
    assert values["glcm_entropy"] == pytest.approx(0.5)


def test_constant_is_valid_not_missing_or_division_error():
    record = extract_record(np.full((16, 17, 3), 128, dtype=np.uint8), format="PNG", bit_depth=8)
    assert record["status"] == "ok" and record["missing_features"] == []
    values = record["features"]
    for name in ("laplacian_variance", "tenengrad", "gradient_mean", "gradient_p90", "gradient_direction_entropy", "noise_mad", "residual_mad", "high_frequency_ratio", "global_contrast", "intensity_range", "dynamic_range", "gray_skew", "edge_density", "edge_direction_change", "glcm_contrast", "glcm_entropy", "histogram_entropy"):
        assert values[name] == pytest.approx(0, abs=1e-12)
    assert values["glcm_homogeneity"] == 1
    assert values["width"] == 17 and values["height"] == 16
    assert values["aspect_ratio"] == pytest.approx(17 / 16)
    assert record["metadata"]["feature_version"] == "full_image_iacs_v2_1"
    assert not record["metadata"]["uses_reference_mask"]


def test_single_pixel_retains_missing_glcm_without_favorable_fill():
    record = extract_record(np.zeros((1, 1, 3), dtype=np.uint8), format="PNG")
    assert record["status"] == "missing_features"
    assert set(record["missing_features"]) == {"glcm_contrast", "glcm_homogeneity", "glcm_entropy"}
    assert all(record["features"][name] is None for name in record["missing_features"])
    assert record["features"]["noise_mad"] == 0


def test_one_row_has_valid_horizontal_glcm():
    record = extract_record(rgb([[0, 255]]))
    assert record["status"] == "ok"
    assert record["features"]["glcm_contrast"] == 31 ** 2
    assert record["features"]["glcm_entropy"] == 1


def test_manual_glcm_includes_distance_two_with_equal_offset_weights():
    values = extract_features(rgb([[0, 0, 255]]))
    # d=1: P00=.5, P0,31=P31,0=.25. d=2: opposite levels .5 each.
    assert values["glcm_contrast"] == pytest.approx(31 ** 2 * 0.75)
    assert values["glcm_entropy"] == pytest.approx((1.5 + 1.0) / 2)
    assert values["glcm_homogeneity"] == pytest.approx((0.5 + 0.5 / 962 + 1 / 962) / 2)


def test_sharp_edges_exceed_blurred_sharpness_and_gradient():
    gray = np.zeros((64, 64), dtype=np.uint8)
    gray[:, 32:] = 255
    blurred = np.rint(ndimage.gaussian_filter(gray.astype(float), sigma=3)).astype(np.uint8)
    sharp_values, blurred_values = extract_features(rgb(gray)), extract_features(rgb(blurred))
    assert sharp_values["laplacian_variance"] > blurred_values["laplacian_variance"]
    assert sharp_values["tenengrad"] > blurred_values["tenengrad"]
    assert sharp_values["high_frequency_ratio"] > blurred_values["high_frequency_ratio"]


def test_noise_proxies_detect_noise_without_masks():
    constant = np.full((96, 96), 128, dtype=np.uint8)
    rng = np.random.RandomState(2718)
    noisy = np.clip(constant.astype(float) + rng.normal(0, 15, constant.shape), 0, 255).astype(np.uint8)
    base, observed = extract_features(rgb(constant)), extract_features(rgb(noisy))
    assert observed["noise_mad"] > base["noise_mad"]
    assert observed["residual_mad"] > base["residual_mad"]
    assert observed["histogram_entropy"] > base["histogram_entropy"]


def test_repeatability_schema_provenance_and_input_preservation():
    array = np.random.RandomState(73).randint(0, 256, (23, 31, 3), dtype=np.uint8)
    before = hashlib.sha256(array.tobytes()).hexdigest()
    first = extract_features(array, {"format": "PNG", "bit_depth": 8})
    second = extract_features(array, {"format": "JPEG", "bit_depth": 8})
    assert first == second
    assert tuple(first) == FEATURE_NAMES and set(FEATURE_METHODS) == set(FEATURE_NAMES)
    assert all(isinstance(value, float) and np.isfinite(value) for value in first.values())
    assert hashlib.sha256(array.tobytes()).hexdigest() == before
    assert FEATURE_VERSION == "full_image_iacs_v2_1"
    assert set(inspect.signature(extract_features).parameters) == {"rgb", "metadata"}
    assert "mask" not in inspect.signature(extract_record).parameters


@pytest.mark.parametrize("array", [np.ones((3, 3, 3), dtype=np.uint16), np.ones((3, 3, 3), dtype=np.float32), np.ones((3, 3), dtype=np.uint8), np.ones((3, 3, 4), dtype=np.uint8), np.ones((0, 3, 3), dtype=np.uint8)])
def test_unsupported_input_rejected(array):
    with pytest.raises(ValueError):
        extract_features(array)


@pytest.mark.parametrize("metadata", [{"format": "DICOM", "bit_depth": 16}, {"mask": "forbidden"}, {"dice": 1.0}, {"format": 42}, "not_a_dictionary"])
def test_metadata_never_becomes_outcome_or_numeric_feature(metadata):
    with pytest.raises(ValueError):
        extract_features(np.zeros((5, 5, 3), dtype=np.uint8), metadata)
