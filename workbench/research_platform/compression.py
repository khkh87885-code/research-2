"""Encoding experiments. Originals are opened read-only; each output is exclusive."""
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, features, __version__ as PILLOW_VERSION

from .common import sha256


def recipes():
    result = {"baseline": {"format": "original", "scale": 1., "order": "copy_bytes"},
              "png": {"format": "PNG", "scale": 1., "level": 9, "order": "encode"}}
    for quality in (95, 85, 75, 60, 40):
        result["jpeg_q%d" % quality] = {"format": "JPEG", "quality": quality, "scale": 1., "order": "encode"}
    for interpolation in ("lanczos", "bicubic"):
        for percent in (100, 90, 75, 50, 25):
            key = "scale_%d" % percent + ("_bicubic" if interpolation == "bicubic" else "")
            result[key] = {"format": "PNG", "level": 9, "scale": percent / 100., "interpolation": interpolation, "order": "resize_encode"}
    for quality in (95, 85, 75, 60, 40):
        for percent in (90, 75, 50, 25):
            for kind, order, fmt in (("cr2", "resize_encode", "JPEG"), ("cr3", "jpeg_resize_png", "PNG")):
                result["%s_q%d_s%d" % (kind, quality, percent)] = {"format": fmt, "quality": quality, "scale": percent/100., "interpolation": "lanczos", "order": order, "level": 9}
    result["webp_lossless"] = {"format": "WEBP", "lossless": True, "scale": 1., "order": "encode"}
    result["webp_q85"] = {"format": "WEBP", "lossless": False, "quality": 85, "scale": 1., "order": "encode"}
    result["jp2_lossless"] = {"format": "JPEG2000", "lossless": True, "scale": 1., "order": "encode"}
    result["jp2_lossy"] = {"format": "JPEG2000", "lossless": False, "scale": 1., "rate": 10, "order": "encode"}
    for percent in (100, 90, 75, 50, 25):
        result["cr5_s%d" % percent] = {"format": "PNG", "level": 9, "scale": percent/100., "interpolation": "lanczos", "order": "resize_encode"}
    return result


def capabilities():
    return {"Pillow": PILLOW_VERSION, "jpeg": features.check("jpg"), "webp": features.check("webp"),
            "jpeg2000": features.check("jpg_2000"), "png": features.check("zlib")}


def canonical(image):
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", image.size, (0, 0, 0, 255)), rgba).convert("RGB")
    return image.copy() if image.mode == "L" else image.convert("RGB")


def _resize(image, recipe):
    scale = float(recipe.get("scale", 1))
    if not 0 < scale <= 1:
        raise ValueError("Resolution scale must be in (0,1]")
    method = recipe.get("interpolation", "lanczos")
    if method not in ("lanczos", "bicubic"):
        raise ValueError("Unknown interpolation")
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS if method == "lanczos" else Image.Resampling.BICUBIC)


def compress(source, output_dir, image_id, recipe):
    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    if source == output_dir or source.parent == output_dir or output_dir in source.parents or source.parent in output_dir.parents:
        raise ValueError("Compression output must be isolated from original image directory")
    if not image_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in image_id):
        raise ValueError("Unsafe image ID")
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = recipe["format"]
    extension = {"original": source.suffix, "JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "JPEG2000": ".jp2"}.get(fmt)
    if extension is None:
        raise ValueError("Unsupported codec")
    target = output_dir / (image_id + extension)
    if target.exists() or target == source:
        raise FileExistsError("Refusing to overwrite image output")
    original_hash = sha256(source)
    started = perf_counter()
    with Image.open(source) as original:
        original.load()
        if original.getexif().get(274, 1) != 1:
            raise ValueError("Nontrivial EXIF orientation requires an explicit image+mask transform; no silent rotation")
        arr = np.asarray(original)
        original_size = original.size
        if fmt == "original":
            with source.open("rb") as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            operations = ["byte-identical original copy"]
        else:
            if arr.dtype != np.uint8 or original.mode not in ("L", "RGB", "RGBA", "LA"):
                raise ValueError("Unsupported bit depth/mode: no implicit 12/16-bit to 8-bit conversion")
            image = original.copy()
            operations = []
            # Pure PNG retains original channels, including alpha. Losslessness is checked against these pixels.
            if fmt in ("JPEG", "WEBP", "JPEG2000") or recipe.get("order") == "jpeg_resize_png":
                image = canonical(image)
                operations.append("RGBA composite over black; L preserved; RGB unchanged")
            order = recipe.get("order", "encode")
            if order == "jpeg_resize_png":
                from io import BytesIO
                buffer = BytesIO()
                image.save(buffer, "JPEG", quality=int(recipe["quality"]), subsampling=0)
                buffer.seek(0)
                with Image.open(buffer) as intermediate:
                    image = intermediate.copy()
                operations.append("JPEG encode+decode q=%s" % recipe["quality"])
            if order in ("resize_encode", "jpeg_resize_png"):
                image = _resize(image, recipe)
                operations.append("resize %s to %dx%d" % (recipe.get("interpolation", "lanczos"), image.width, image.height))
            elif order != "encode":
                raise ValueError("Unknown processing order")
            kwargs = {}
            if fmt == "JPEG":
                quality = int(recipe["quality"])
                if not 1 <= quality <= 100:
                    raise ValueError("Invalid JPEG quality")
                kwargs = {"quality": quality, "subsampling": 0}
            elif fmt == "PNG":
                kwargs = {"compress_level": int(recipe.get("level", 9))}
            elif fmt == "WEBP":
                if not features.check("webp"):
                    raise RuntimeError("WebP codec unavailable: install Pillow with libwebp support")
                kwargs = {"lossless": bool(recipe["lossless"]), "quality": int(recipe.get("quality", 100)), "method": 6}
            elif fmt == "JPEG2000":
                if not features.check("jpg_2000"):
                    raise RuntimeError("JPEG2000 unavailable: install Pillow with OpenJPEG support")
                kwargs = {"irreversible": not recipe["lossless"]}
                if not recipe["lossless"]:
                    kwargs.update(quality_mode="rates", quality_layers=[recipe.get("rate", 10)])
            with target.open("xb") as stream:
                image.save(stream, format=fmt, **kwargs)
            operations.append("encode %s %s" % (fmt, kwargs))
        encoding_ms = 1000 * (perf_counter() - started)
        started = perf_counter()
        with Image.open(target) as decoded:
            decoded.load()
            processed_size = decoded.size
            equal = decoded.mode == original.mode and np.array_equal(np.asarray(decoded), arr)
        decoding_ms = 1000 * (perf_counter() - started)
    if sha256(source) != original_hash:
        raise RuntimeError("Original image changed during read-only compression")
    return {"processed_path": str(target), "original_sha256": original_hash, "processed_sha256": sha256(target),
            "original_bytes": source.stat().st_size, "compressed_bytes": target.stat().st_size,
            "width": processed_size[0], "height": processed_size[1], "original_width": original_size[0], "original_height": original_size[1],
            "encoding_ms": encoding_ms, "decoding_ms": decoding_ms, "decoded_pixels_equal_original": bool(equal),
            "operations": operations, "encoder": "Pillow " + PILLOW_VERSION, **recipe}
