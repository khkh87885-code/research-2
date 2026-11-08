"""Read the existing frozen manifests; never write to source datasets."""
import csv
import hashlib
import hmac
import json
import secrets
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents


def check_output(path, config):
    """Reject both descendants and ancestors of protected source locations."""
    target = Path(path).resolve()
    roots = [Path(d["root"]).resolve() for d in config["datasets"]]
    roots += [Path(config["model"]["checkpoint"]).resolve().parent]
    for root in roots:
        if inside(target, root) or inside(root, target):
            raise ValueError("Output overlaps a protected data/model location")
    return target


def pseudonym_salt(project):
    directory = Path(project) / ".compression_private"
    directory.mkdir(exist_ok=True)
    path = directory / "pseudonym.key"
    if not path.exists():
        try:
            with path.open("xb") as stream:
                stream.write(secrets.token_bytes(32))
        except FileExistsError:
            pass
    return path.read_bytes()


def key(value, salt):
    return hmac.new(salt, value.encode("utf-8"), hashlib.sha256).hexdigest()[:24]


def read_records(config, salt):
    records, summaries = [], []
    patient_roles = defaultdict(set)
    keys = set()
    for source in config["datasets"]:
        root = Path(source["root"]).resolve(strict=True)
        manifest = Path(source["manifest"]).resolve(strict=True)
        digest = sha256(manifest)
        if source.get("sha256") and digest.lower() != source["sha256"].lower():
            raise ValueError("Frozen manifest checksum mismatch")
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"dataset", "split", "patient_id", "image_id", "image_path", "mask_path", "width", "height"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("Manifest missing required columns")
            rows = list(reader)
        patients = set()
        for row in rows:
            if row["dataset"] != source["dataset"] or row["split"] != source["split"]:
                raise ValueError("Manifest dataset/split disagrees with configuration")
            if not row["patient_id"].strip():
                raise ValueError("Patient identifier unavailable: patient analysis disabled")
            if not row["image_id"].strip():
                raise ValueError("Image identifier unavailable")
            patient = key(source["dataset"] + ":" + row["patient_id"].strip(), salt)
            image = key(source["dataset"] + ":" + row["image_id"].strip(), salt)
            if (source["dataset"], image) in keys:
                raise ValueError("Duplicate image identifier in configured manifests")
            keys.add((source["dataset"], image))
            patient_roles[(source["dataset"], patient)].add(source["split"])
            if len(patient_roles[(source["dataset"], patient)]) > 1:
                raise ValueError("Patient appears in more than one split")
            paths = {}
            for kind in ("image", "mask"):
                path = (root / row[kind + "_path"]).resolve(strict=True)
                if not inside(path, root):
                    raise ValueError("Source path escapes configured dataset root")
                paths[kind + "_path"] = str(path)
                expected = row.get(kind + "_sha256")
                if expected and sha256(path) != expected:
                    raise ValueError("Source file checksum mismatch")
            records.append(dict(dataset=source["dataset"], split=source["split"],
                                patient_key=patient, image_key=image, width=int(row["width"]),
                                height=int(row["height"]), lesion_key=None, spacing=None, **paths))
            patients.add(patient)
        summaries.append(dict(dataset=source["dataset"], split=source["split"], images=len(rows),
                              patients=len(patients), manifest_sha256=digest))
    return records, summaries


def load_reference(record):
    with Image.open(record["mask_path"]) as image:
        # Reuse the existing project's binary-mask interpretation.
        mask = np.asarray(image.convert("1"), dtype=bool)
    if mask.shape != (record["height"], record["width"]):
        raise ValueError("Reference dimensions disagree with manifest")
    if not mask.any():
        raise ValueError("Empty reference mask")
    return mask


def inventory(config, project, output):
    output = check_output(output, config)
    output.mkdir(parents=True, exist_ok=False)
    records, summary = read_records(config, pseudonym_salt(project))
    report, hashes, failures = [], defaultdict(list), 0
    for row in records:
        public = {k: row[k] for k in ("dataset", "split", "patient_key", "image_key", "lesion_key", "spacing")}
        public.update(data_status="ok", annotation_status="present", error_type=None)
        try:
            with Image.open(row["image_path"]) as image:
                image.load()
                array = np.asarray(image)
                if image.size != (row["width"], row["height"]):
                    raise ValueError("Image dimensions disagree with manifest")
                public.update(width=image.width, height=image.height, image_format=image.format,
                              image_mode=image.mode, bit_depth=array.dtype.itemsize * 8,
                              gray_min=float(array.min()), gray_max=float(array.max()),
                              original_bytes=Path(row["image_path"]).stat().st_size)
            load_reference(row)
            digest = sha256(row["image_path"])
            hashes[digest].append(row["image_key"])
            public["image_sha256"] = digest
            public["mask_sha256"] = sha256(row["mask_path"])
        except Exception as exc:
            failures += 1
            public.update(data_status="error", error_type=type(exc).__name__)
        report.append(public)
    result = dict(status="complete", datasets=summary, records=report, failures=failures,
                  duplicate_byte_groups=[v for v in hashes.values() if len(v) > 1],
                  limitations=["Byte equality does not establish patient independence or exclude near duplicates",
                               "Pixel spacing and lesion identifiers unavailable; distances use pixels"])
    (output / "inventory.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    (output / "DATASET_REPORT.md").write_text(
        "# Local dataset inventory\n\n" + "\n".join("- {dataset} / {split}: {images} images, {patients} patients".format(**x) for x in summary)
        + "\n\nFailures: {}. Exact-byte duplicate groups: {}.\n\nNo patient-level source independence claim. Distances: pixels.\n".format(failures, len(result["duplicate_byte_groups"])), encoding="utf-8")
    return result
