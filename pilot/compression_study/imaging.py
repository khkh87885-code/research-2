"""Local, non-destructive 8-bit image transforms for the compression study.

Input and output dimensions use (width, height). CR3 (encode_resize) measures
the JPEG/other encoded intermediate and stores the resized pixels in a final
lossless PNG container; its final byte count is the upload payload size.
"""

from io import BytesIO
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, features


def _check_rgb(rgb):
    array = np.asarray(rgb)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected an H x W x 3 uint8 RGB image; explicit conversion required.")
    if not array.shape[0] or not array.shape[1]:
        raise ValueError("Image dimensions must be positive.")
    return array


def load_rgb(path):
    """Read one 8-bit frame; composite alpha on black as in the old BrEaST code.

    High-bit-depth, floating-point, multi-frame and oriented inputs are rejected
    instead of silently changing the coordinate system or intensity scale.
    No EXIF or other source metadata are copied to experimental derivatives.
    """
    path = Path(path)
    if path.suffix.lower() in (".dcm", ".dicom"):
        raise ValueError("DICOM requires an explicit, audited display conversion.")
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Multi-frame images require an explicit frame policy.")
        if image.mode not in ("1", "L", "LA", "P", "RGB", "RGBA"):
            raise ValueError("Unsupported image mode %s; high-bit-depth conversion is forbidden." % image.mode)
        # Pillow can expose RGB PNG/TIFF/JPEG2000 as 8-bit RGB even when their
        # source samples have more bits. Inspect the original format declaration.
        declared_depth = 8
        if image.format == "PNG":
            with path.open("rb") as stream:
                header = stream.read(29)
            declared_depth = header[24] if len(header) >= 29 else 0
        elif image.format == "TIFF":
            depths = image.tag_v2.get(258, (8,))
            declared_depth = max(depths if isinstance(depths, (tuple, list)) else (depths,))
        elif image.format == "JPEG2000":
            with path.open("rb") as stream:
                header = stream.read(65536)
            marker = header.find(b"\xff\x51")
            if marker < 0 or marker + 40 > len(header):
                raise ValueError("Cannot verify JPEG2000 source precision; explicit conversion required.")
            channels = int.from_bytes(header[marker + 38:marker + 40], "big")
            if not channels or marker + 40 + 3 * channels > len(header):
                raise ValueError("Cannot verify JPEG2000 channel precision.")
            declared_depth = max((header[marker + 40 + 3 * index] & 127) + 1 for index in range(channels))
        if declared_depth <= 0 or declared_depth > 8:
            raise ValueError("Source bit depth exceeds 8 bits; explicit audited conversion required.")
        if image.getexif().get(274, 1) not in (None, 1):
            raise ValueError("Non-identity EXIF orientation requires an audited image/mask mapping.")
        # RGB over black is identical to run_formal_validation.rgb_over_black;
        # keeping this lightweight avoids importing the CUDA-only old runners.
        rgba = image.convert("RGBA")
        black = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        rgb = Image.alpha_composite(black, rgba).convert("RGB")
        return np.array(rgb, dtype=np.uint8)


def _encode(rgb, codec, quality, lossless, compress_level):
    target = BytesIO()
    options = {}
    if codec == "JPEG":
        if lossless:
            raise ValueError("JPEG does not support lossless mode in this platform.")
        if not features.check("jpg"):
            raise RuntimeError("Pillow JPEG encoder is unavailable; install a JPEG-enabled Pillow build.")
        options = {"quality": quality, "subsampling": 0, "optimize": False}
    elif codec == "PNG":
        options = {"compress_level": compress_level}
    elif codec == "WEBP":
        if not features.check("webp"):
            raise RuntimeError("Pillow WebP encoder is unavailable; install a WebP-enabled Pillow build.")
        options = {"quality": quality, "lossless": lossless, "method": 4}
    elif codec == "JPEG2000":
        if not features.check("jpg_2000"):
            raise RuntimeError("Pillow JPEG2000 encoder is unavailable; install an OpenJPEG-enabled Pillow build.")
        if lossless:
            options = {"irreversible": False}
        else:
            # Quality is explicitly a JPEG2000 target PSNR, not JPEG's scale.
            options = {"irreversible": True, "quality_mode": "dB", "quality_layers": [quality]}
    start = perf_counter()
    Image.fromarray(rgb, "RGB").save(target, format=codec, **options)
    elapsed = 1000.0 * (perf_counter() - start)
    return target.getvalue(), elapsed


def _decode(payload):
    start = perf_counter()
    with Image.open(BytesIO(payload)) as image:
        image.load()
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    return rgb, 1000.0 * (perf_counter() - start)


def transform_image(rgb, spec, destination):
    """Write exactly one new derivative and return decoded pixels and provenance.

    ``format`` selects jpeg/png/webp/jpeg2000. PNG is always lossless; PNG
    ``compress_level`` (0..9) is an encoding-effort setting. ``quality`` is 1..100
    for JPEG/WebP and target PSNR dB for lossy JPEG2000. ``scale`` is (0,1].
    """
    rgb = _check_rgb(rgb)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing image derivative.")
    formats = {"jpeg": "JPEG", "jpg": "JPEG", "png": "PNG", "webp": "WEBP", "jpeg2000": "JPEG2000", "jp2": "JPEG2000"}
    format_name = str(spec.get("format", "")).lower()
    if format_name not in formats:
        raise ValueError("Unsupported transform format %r; baseline must use the source directly." % format_name)
    codec = formats[format_name]
    scale = float(spec.get("scale", 1.0))
    if not np.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("Resolution scale must be finite and in (0, 1].")
    quality = float(spec.get("quality", 95) if spec.get("quality") is not None else 95)
    if not np.isfinite(quality) or not 1 <= quality <= 100:
        raise ValueError("Quality must be in [1,100]; JPEG2000 uses this as target PSNR dB.")
    quality = int(quality) if codec != "JPEG2000" else quality
    compress_level = spec.get("compress_level", 6)
    if not isinstance(compress_level, int) or isinstance(compress_level, bool) or not 0 <= compress_level <= 9:
        raise ValueError("PNG compress_level must be an integer in [0,9].")
    lossless = bool(spec.get("lossless", codec == "PNG"))
    if codec == "PNG" and not lossless:
        raise ValueError("PNG is lossless; its encoding level is not a lossy quality setting.")
    interpolation = str(spec.get("interpolation", "lanczos")).lower()
    methods = {"lanczos": Image.Resampling.LANCZOS, "bicubic": Image.Resampling.BICUBIC}
    if interpolation not in methods:
        raise ValueError("Interpolation must be lanczos or bicubic.")
    order = spec.get("order", "resize_encode")
    if order not in ("resize_encode", "encode_resize"):
        raise ValueError("Processing order must be resize_encode or encode_resize.")
    original_size = (rgb.shape[1], rgb.shape[0])
    target_size = tuple(max(1, int(round(value * scale))) for value in original_size)
    encode_ms = decode_ms = resize_ms = 0.0
    intermediate_bytes = None

    def resize(array):
        start = perf_counter()
        if (array.shape[1], array.shape[0]) == target_size:
            resized = array.copy()
        else:
            resized = np.array(Image.fromarray(array, "RGB").resize(target_size, methods[interpolation]), dtype=np.uint8)
        return resized, 1000.0 * (perf_counter() - start)

    if order == "resize_encode":
        transformed, resize_ms = resize(rgb)
        payload, encode_ms = _encode(transformed, codec, quality, lossless, compress_level)
        decoded, decode_ms = _decode(payload)
        final_codec = codec
        sequence = ["resize", codec.lower() + "_encode", "decode"]
    else:
        intermediate, first_encode = _encode(rgb, codec, quality, lossless, compress_level)
        intermediate_bytes = len(intermediate)
        transformed, first_decode = _decode(intermediate)
        transformed, resize_ms = resize(transformed)
        payload, final_encode = _encode(transformed, "PNG", quality, True, compress_level)
        decoded, final_decode = _decode(payload)
        encode_ms = first_encode + final_encode
        decode_ms = first_decode + final_decode
        final_codec = "PNG"
        sequence = [codec.lower() + "_encode", "decode", "resize", "png_encode", "decode"]
    # Exclusive creation protects original data even if another process creates
    # this path between the initial existence check and the final write.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(payload)
    metadata = {
        "codec": codec, "final_format": final_codec,
        "processing_order": order, "processing_steps": sequence,
        "quality": quality if codec != "PNG" else None,
        "quality_semantics": "target_psnr_db" if codec == "JPEG2000" and not lossless else "encoder_quality" if codec in ("JPEG", "WEBP") else None,
        "lossless": lossless, "compress_level": compress_level if codec == "PNG" or order == "encode_resize" else None,
        "codec_lossless": lossless,
        "scale": scale, "interpolation": interpolation,
        "original_width": original_size[0], "original_height": original_size[1],
        "width": decoded.shape[1], "height": decoded.shape[0],
        "output_bytes": len(payload), "intermediate_bytes": intermediate_bytes,
        "encode_ms": encode_ms, "decode_ms": decode_ms, "resize_ms": resize_ms,
        "pixel_equal": bool(decoded.shape == rgb.shape and np.array_equal(decoded, rgb)),
        "metadata_policy": "source_metadata_not_copied",
    }
    return decoded, metadata


def scaled_box(box, original_size, target_size):
    """Scale an original-image XYXY prompt once using actual rounded dimensions."""
    box = np.asarray(box, dtype=np.float32)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("Box must contain four finite XYXY coordinates.")
    if len(original_size) != 2 or len(target_size) != 2 or any(not np.isfinite(value) or int(value) != value or value <= 0 for value in tuple(original_size) + tuple(target_size)):
        raise ValueError("Image sizes must contain positive integer width and height.")
    if not box[0] < box[2] or not box[1] < box[3]:
        raise ValueError("Box must have positive width and height.")
    factor = np.asarray([target_size[0] / original_size[0], target_size[1] / original_size[1]] * 2, dtype=np.float32)
    limits = np.asarray([target_size[0], target_size[1]] * 2, dtype=np.float32)
    result = np.clip(box * factor, 0, limits)
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError("Clipping produces an empty box.")
    return result


def restore_mask(mask, original_size):
    """Map a 2-D predicted label mask to original dimensions with nearest neighbor."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or not mask.shape[0] or not mask.shape[1]:
        raise ValueError("Prediction mask must be a non-empty 2-D array.")
    if not np.isfinite(mask).all():
        raise ValueError("Prediction mask must contain finite values.")
    if len(original_size) != 2 or any(not np.isfinite(value) or int(value) != value or value <= 0 for value in original_size):
        raise ValueError("Original size must contain positive integer width and height.")
    labels = Image.fromarray((mask != 0).astype(np.uint8) * 255)
    return np.asarray(labels.resize(tuple(map(int, original_size)), Image.Resampling.NEAREST)) != 0
