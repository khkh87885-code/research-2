"""Analytic geometry and codec checks; synthetic arrays are software tests only."""

import struct
import zlib

import numpy as np
import pytest
from PIL import Image, features

from compression_study.features import extract_features
from compression_study.imaging import load_rgb, restore_mask, scaled_box, transform_image
from compression_study.metrics import evaluate_masks


def test_identical_nonempty_and_disjoint():
    first = np.zeros((10, 10), dtype=bool)
    first[2:5, 2:5] = True
    same = evaluate_masks(first, first)
    assert same["dice"] == same["iou"] == same["boundary_f1"] == 1
    assert same["hd95"] == same["assd"] == same["relative_area_error"] == 0
    second = np.zeros_like(first)
    second[7, 7] = True
    assert evaluate_masks(first, second)["dice"] == 0


def test_analytic_surface_distance_and_spacing():
    pred = np.zeros((8, 8), dtype=bool)
    ref = pred.copy()
    pred[1, 1], ref[1, 4] = True, True
    result = evaluate_masks(pred, ref)
    assert result["hd95"] == result["assd"] == 3
    assert result["boundary_f1"] == 0
    assert evaluate_masks(pred, ref, tolerance=3)["boundary_f1"] == 1
    physical = evaluate_masks(pred, ref, spacing=(2, 0.5), tolerance=1.5)
    assert physical["hd95"] == physical["assd"] == 1.5
    assert physical["distance_unit"] == "mm"
    assert physical["boundary_f1"] == 1


def test_partial_overlap_and_empty_policies():
    pred = np.array([[1, 1], [0, 0]])
    ref = np.array([[1, 0], [1, 0]])
    result = evaluate_masks(pred, ref)
    assert result["dice"] == 0.5
    assert result["iou"] == pytest.approx(1 / 3)
    assert result["precision"] == result["recall"] == 0.5
    empty = np.zeros((2, 2), dtype=bool)
    result = evaluate_masks(empty, ref)
    assert result["dice"] == result["recall"] == 0
    assert result["precision"] is result["hd95"] is result["assd"] is None
    assert result["relative_area_error"] == 1
    for p in (empty, pred):
        invalid = evaluate_masks(p, empty)
        assert invalid["dice"] is invalid["hd95"] is None
        assert invalid["metric_status"] in ("both_empty", "empty_reference")


@pytest.mark.parametrize("spacing", [(0, 1), (-1, 1), (1,), (float("nan"), 1)])
def test_invalid_spacing_rejected(spacing):
    with pytest.raises(ValueError):
        evaluate_masks(np.ones((2, 2)), np.ones((2, 2)), spacing=spacing)


def test_invalid_masks_rejected():
    with pytest.raises(ValueError):
        evaluate_masks(np.zeros((2, 2)), np.zeros((3, 2)))
    with pytest.raises(ValueError):
        evaluate_masks(np.full((2, 2), np.nan), np.ones((2, 2)))


def test_png_roundtrip_is_lossless_and_exclusive(tmp_path):
    rgb = np.random.RandomState(3).randint(0, 256, (17, 19, 3), dtype=np.uint8)
    destination = tmp_path / "derivative.png"
    decoded, metadata = transform_image(rgb, {"format": "png", "scale": 1}, destination)
    assert np.array_equal(decoded, rgb)
    assert metadata["pixel_equal"]
    assert metadata["output_bytes"] == destination.stat().st_size
    assert metadata["quality"] is None
    with pytest.raises(FileExistsError):
        transform_image(rgb, {"format": "png"}, destination)


def test_processing_order_and_final_payload_are_recorded(tmp_path):
    rgb = np.random.RandomState(4).randint(0, 256, (24, 28, 3), dtype=np.uint8)
    decoded, metadata = transform_image(rgb, {"format": "jpeg", "quality": 40, "scale": 0.5, "order": "encode_resize"}, tmp_path / "cr3.png")
    assert decoded.shape == (12, 14, 3)
    assert metadata["codec"] == "JPEG" and metadata["final_format"] == "PNG"
    assert metadata["intermediate_bytes"] > 0
    assert metadata["output_bytes"] == (tmp_path / "cr3.png").stat().st_size
    with Image.open(tmp_path / "cr3.png") as image:
        assert image.format == "PNG"
    direct, direct_metadata = transform_image(rgb, {"format": "jpeg", "quality": 40, "scale": 0.5}, tmp_path / "cr2.jpg")
    assert direct.shape == decoded.shape
    assert direct_metadata["final_format"] == "JPEG"
    assert not np.array_equal(direct, decoded)


@pytest.mark.parametrize("codec,dependency", [("webp", "webp"), ("jpeg2000", "jpg_2000")])
def test_optional_lossless_codecs_or_explicit_dependency_error(tmp_path, codec, dependency):
    rgb = np.random.RandomState(42).randint(0, 256, (32, 36, 3), dtype=np.uint8)
    destination = tmp_path / (codec + ".img")
    if not features.check(dependency):
        with pytest.raises(RuntimeError, match="unavailable"):
            transform_image(rgb, {"format": codec, "lossless": True}, destination)
    else:
        decoded, metadata = transform_image(rgb, {"format": codec, "lossless": True}, destination)
        assert metadata["pixel_equal"] and np.array_equal(decoded, rgb)
        assert np.array_equal(load_rgb(destination), rgb)


@pytest.mark.parametrize("spec", [{"format": "jpeg", "scale": 0}, {"format": "jpeg", "quality": 101}, {"format": "png", "lossless": False}, {"format": "jpeg", "interpolation": "nearest"}])
def test_invalid_transform_fails_without_output(tmp_path, spec):
    destination = tmp_path / "invalid.png"
    with pytest.raises(ValueError):
        transform_image(np.zeros((10, 10, 3), dtype=np.uint8), spec, destination)
    assert not destination.exists()


def test_real_dimensions_drive_coordinates_and_nearest_labels():
    mapped = scaled_box([2, 1, 9, 6], (11, 7), (3, 2))
    assert np.allclose(mapped, [6 / 11, 2 / 7, 27 / 11, 12 / 7])
    mask = np.array([[0, 1], [1, 0]], dtype=bool)
    restored = restore_mask(mask, (4, 4))
    assert restored.dtype == np.bool_
    assert np.array_equal(restored, np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1))


def test_alpha_black_and_high_depth_rejection(tmp_path):
    rgba = np.array([[[255, 100, 0, 0], [200, 100, 50, 255]]], dtype=np.uint8)
    path = tmp_path / "alpha.png"
    Image.fromarray(rgba, "RGBA").save(path)
    original = path.read_bytes()
    assert np.array_equal(load_rgb(path), [[[0, 0, 0], [200, 100, 50]]])
    assert path.read_bytes() == original
    depth = tmp_path / "16bit.png"
    Image.fromarray(np.array([[0, 65535]], dtype=np.uint16)).save(depth)
    with pytest.raises(ValueError, match="mode"):
        load_rgb(depth)
    with pytest.raises(ValueError):
        transform_image(np.ones((2, 2, 3), dtype=np.uint16), {"format": "jpeg"}, tmp_path / "invalid.jpg")


def test_rgb16_png_rejected_even_when_pillow_reports_rgb(tmp_path):
    # Construct a standards-compliant 16-bit RGB PNG to catch Pillow's implicit
    # downconversion of multi-channel high-bit-depth sources.
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)
    path = tmp_path / "rgb16.png"
    raw = b"\x00" + struct.pack(">HHH", 65535, 32768, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 16, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    with Image.open(path) as image:
        assert image.mode == "RGB"
    with pytest.raises(ValueError, match="bit depth"):
        load_rgb(path)


def test_feature_values_are_finite_without_reference():
    constant = extract_features(np.zeros((10, 11, 3), dtype=np.uint8))
    assert constant["clarity"] == constant["noise"] == constant["entropy"] == 0
    assert constant["width"] == 11 and constant["height"] == 10
    noisy = extract_features(np.random.RandomState(5).randint(0, 256, (20, 20, 3), dtype=np.uint8))
    assert all(isinstance(value, float) and np.isfinite(value) for value in noisy.values())
    assert noisy["clarity"] > 0 and noisy["entropy"] > 0
