"""Read-only inventory of existing research assets; never infers patient identity.

Test images in tests/ are software fixtures, not research observations.
"""
import csv
import hashlib
import json
import math
import posixpath
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


BREAST_PAPER_URL = "https://www.nature.com/articles/s41597-024-02984-z"


class UnsupportedImageError(ValueError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rgb(path):
    """Decode stored pixel coordinates, compositing alpha over opaque black.

    No EXIF rotation or high-depth windowing is applied implicitly.
    """
    with Image.open(path) as image:
        image.load()
        array = np.asarray(image)
        if array.dtype != np.uint8 and image.mode != "1":
            raise UnsupportedImageError(
                "Explicit high-bit-depth/display conversion required: mode=%s dtype=%s"
                % (image.mode, array.dtype)
            )
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            rgba = image.convert("RGBA")
            rgb = Image.alpha_composite(
                Image.new("RGBA", image.size, (0, 0, 0, 255)), rgba
            ).convert("RGB")
        else:
            rgb = image.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8).copy()


def load_mask(path):
    """Read a binary reference without Pillow's convert('1') dithering.

    Alpha is not a lesion label. White RGB content in official RGBA masks is
    the reference, including partially transparent white foreground pixels.
    Multiclass/color/nonbinary masks require a separately declared mapping.
    """
    with Image.open(path) as image:
        image.load()
        if image.mode == "P":
            array = np.asarray(image.convert("RGB"))
        else:
            array = np.asarray(image)
    if array.ndim == 3:
        channels = array[:, :, :3]
        if channels.shape[2] == 2:  # LA: luminance and alpha
            array = channels[:, :, 0]
        elif channels.shape[2] >= 3:
            if not np.all(channels == channels[:, :, :1]):
                raise ValueError("Color reference mask requires an explicit label mapping")
            array = channels[:, :, 0]
        else:
            array = channels[:, :, 0]
    if array.ndim != 2:
        raise ValueError("Reference mask must be a two-dimensional binary label image")
    values = np.unique(array)
    if len(values) > 2 or (len(values) == 2 and values[0] != 0):
        raise ValueError("Nonbinary reference mask requires an explicit label mapping")
    return (array > 0).copy()


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _issue(issues, code, dataset=None, image_id=None, **detail):
    row = {"code": code, "dataset": dataset, "image_id": image_id}
    row.update(detail)
    issues.append(row)


def _safe_relative(root, relative):
    root = Path(root).resolve()
    result = (root / str(relative)).resolve()
    try:
        result.relative_to(root)
    except ValueError:
        raise ValueError("Manifest path leaves declared dataset root")
    return result


def _xlsx_rows(path, sheet_name):
    """Read only the required worksheet with stdlib; no Excel execution."""
    main = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    relns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            for item in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("m:si", main):
                shared.append("".join(item.itertext()))
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook.findall("m:sheets/m:sheet", main)
        selected = next((s for s in sheets if s.get("name") == sheet_name), None)
        if selected is None:
            raise ValueError("Declared BrEaST clinical worksheet not found")
        rel_id = selected.get("{%s}id" % relns)
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(r.get("Target") for r in rels if r.get("Id") == rel_id)
        member = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
        tree = ET.fromstring(archive.read(member))
        grid = []
        for row in tree.findall("m:sheetData/m:row", main):
            cells = {}
            for cell in row.findall("m:c", main):
                label = "".join(ch for ch in cell.get("r", "") if ch.isalpha())
                index = 0
                for ch in label:
                    index = index * 26 + ord(ch) - ord("A") + 1
                value = cell.find("m:v", main)
                if cell.get("t") == "inlineStr":
                    value = "".join(cell.find("m:is", main).itertext())
                else:
                    value = value.text if value is not None else ""
                    if cell.get("t") == "s":
                        value = shared[int(value)]
                cells[index - 1] = value
            grid.append(cells)
    if not grid:
        return []
    headers = grid[0]
    required = {"CaseID", "Image_filename", "Mask_tumor_filename", "Classification"}
    if not required.issubset(set(headers.values())):
        raise ValueError("BrEaST workbook missing declared pairing columns")
    return [{name: row.get(i, "") for i, name in headers.items() if name in required | {"Pixel_size"}}
            for row in grid[1:] if any(row.values())]


def _parse_pixel_size_cm(value):
    """Official Pixel_size is pixel width/height in cm; return (row_mm,col_mm).

    Released v1 workbook contains one scalar per case (the common width and
    height). An explicit [width,height] pair is reordered for NumPy axes.
    Empty, nonpositive and nonfinite values are unavailable, never imputed.
    """
    text = str(value).strip() if value is not None else ""
    if not text:
        return None
    parsed = json.loads(text) if text.startswith("[") else float(text)
    if isinstance(parsed, (list, tuple)):
        if len(parsed) != 2:
            raise ValueError("Pixel_size pair must be [width_cm,height_cm]")
        width, height = (float(parsed[0]), float(parsed[1]))
    else:
        width = height = float(parsed)
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError("Pixel_size must be positive finite centimeters")
    return [height * 10.0, width * 10.0]


def _apply_breast_clinical(record, clinical, issues, workbook):
    """Attach only source-linked IDs and spacing; clinical names are not read."""
    expected_image = Path(record["image_path"]).name
    expected_mask = Path(record["mask_path"]).name if record["mask_path"] else ""
    if (clinical["Image_filename"] != expected_image or
            str(clinical.get("Mask_tumor_filename", "")) != expected_mask):
        record["data_status"] = "error"
        _issue(issues, "official_case_filename_mapping_conflict", "BrEaST", record["image_id"])
        return
    record["patient_id_verified"] = True
    record["patient_id_source_url"] = BREAST_PAPER_URL
    record["patient_id_verification_basis"] = (
        "Official paper reports 256 scans from 256 patients; Data Records defines "
        "one worksheet row per Case_ID linked to Image_filename; actual CaseID and filenames matched."
    )
    raw = clinical.get("Pixel_size", "")
    record["pixel_spacing_raw_cm"] = raw
    record["pixel_spacing_source"] = str(workbook) + ":Pixel_size"
    record["pixel_spacing_source_url"] = BREAST_PAPER_URL
    try:
        record["pixel_spacing"] = _parse_pixel_size_cm(raw)
        if record["pixel_spacing"] is not None:
            record["pixel_spacing_unit"] = "mm"
            record["pixel_spacing_axis_order"] = "row,column"
            record["pixel_spacing_conversion"] = "official_cm_times_10; scalar_applies_to_both_axes"
        else:
            _issue(issues, "pixel_spacing_missing", "BrEaST", record["image_id"])
    except (ValueError, TypeError, OverflowError):
        record["pixel_spacing"] = None
        _issue(issues, "pixel_spacing_invalid", "BrEaST", record["image_id"])


def _record(dataset, patient_id, image_id, image_path, mask_path, split,
            patient_id_source, pathology=None, selected=False):
    return {
        "dataset": dataset, "patient_id": str(patient_id) if patient_id not in (None, "") else None,
        "image_id": str(image_id), "lesion_id": None,
        "image_path": str(image_path), "mask_path": str(mask_path) if mask_path else None,
        "split": split, "width": None, "height": None, "original_bytes": None,
        "format": None, "bit_depth": None, "gray_min": None, "gray_max": None,
        "pixel_spacing": None, "annotation_status": "unverified", "data_status": "unverified",
        "image_sha256": None, "mask_sha256": None, "pixel_sha256": None,
        "patient_id_source": patient_id_source, "patient_id_verified": False,
        "pathology": pathology, "selected_in_frozen_manifest": bool(selected),
    }


def _inspect(record, issues, expected_size=None, expected_hashes=None):
    dataset, image_id = record["dataset"], record["image_id"]
    path = Path(record["image_path"])
    try:
        record["original_bytes"] = path.stat().st_size
        record["image_sha256"] = sha256_file(path)
        with Image.open(path) as image:
            image.load()
            array = np.asarray(image)
            stored_bit_depth = 1 if image.mode == "1" else int(array.dtype.itemsize * 8)
            if image.format == "PNG":
                with path.open("rb") as stream:
                    header = stream.read(25)
                if header[:8] == b"\x89PNG\r\n\x1a\n" and len(header) >= 25:
                    stored_bit_depth = int(header[24])
            elif image.format == "TIFF":
                tag_depth = image.tag_v2.get(258)
                if tag_depth is not None:
                    stored_bit_depth = int(max(tag_depth) if isinstance(tag_depth, tuple) else tag_depth)
            intensity = array
            if array.ndim == 3:
                intensity = array[:, :, 0] if image.mode == "LA" else array[:, :, :3]
            record.update(width=image.width, height=image.height, format=image.format,
                          image_mode=image.mode, dtype=str(array.dtype),
                          bit_depth=stored_bit_depth,
                          gray_min=float(intensity.min()), gray_max=float(intensity.max()),
                          exif_orientation=int(image.getexif().get(274, 1)))
            # PNG pHYs/dpi is display resolution, not proven physical medical spacing.
            if "dpi" in image.info:
                record["display_dpi"] = list(image.info["dpi"])
            if record["exif_orientation"] != 1:
                _issue(issues, "stored_exif_orientation", dataset, image_id,
                       orientation=record["exif_orientation"], handling="stored_pixels_no_rotation")
            if record["bit_depth"] <= 8:
                canonical = load_rgb(path)
            else:
                canonical = array
                _issue(issues, "high_bit_depth_requires_explicit_conversion", dataset, image_id,
                       dtype=str(array.dtype))
            digest = hashlib.sha256()
            digest.update(str(canonical.shape).encode("ascii"))
            digest.update(str(canonical.dtype).encode("ascii"))
            digest.update(canonical.tobytes())
            record["pixel_sha256"] = digest.hexdigest()
        record["data_status"] = "ok"
        if expected_size and tuple(expected_size) != (record["width"], record["height"]):
            _issue(issues, "metadata_dimension_mismatch", dataset, image_id,
                   declared_size=list(expected_size), actual_size=[record["width"], record["height"]])
        if expected_hashes and expected_hashes.get("image_sha256") != record["image_sha256"]:
            record["data_status"] = "error"
            _issue(issues, "locked_image_checksum_mismatch", dataset, image_id)
    except Exception as exc:
        record["data_status"] = "error"
        _issue(issues, "image_unreadable", dataset, image_id, error_type=type(exc).__name__)
    if record["mask_path"]:
        try:
            record["mask_sha256"] = sha256_file(record["mask_path"])
            mask = load_mask(record["mask_path"])
            if mask.shape != (record["height"], record["width"]):
                record["annotation_status"] = "dimension_mismatch"
                _issue(issues, "reference_dimension_mismatch", dataset, image_id)
            elif not mask.any():
                record["annotation_status"] = "empty_reference"
                _issue(issues, "empty_reference", dataset, image_id)
            else:
                record["annotation_status"] = "complete"
            if expected_hashes and expected_hashes.get("mask_sha256") != record["mask_sha256"]:
                record["data_status"] = "error"
                _issue(issues, "locked_mask_checksum_mismatch", dataset, image_id)
        except Exception as exc:
            record["annotation_status"] = "unreadable"
            _issue(issues, "reference_unreadable", dataset, image_id, error_type=type(exc).__name__)
    elif record["pathology"] == "normal":
        record["annotation_status"] = "not_applicable_normal"
    else:
        record["annotation_status"] = "missing"
        _issue(issues, "reference_missing", dataset, image_id)
    return record


def scan_datasets(config):
    """Return inventory and QC without modifying source files or writing reports."""
    root = Path(config.get("data_root", "D:/codex论文")).resolve()
    pilot = Path(config.get("pilot_root", "D:/Codex工作区/busbra_sam_pilot")).resolve()
    records, issues, sources = [], [], {}

    def source(path):
        path = Path(path)
        if path.is_file():
            sources[str(path)] = sha256_file(path)
            return True
        _issue(issues, "source_file_missing", path=str(path))
        return False

    bus_root = root / "BUSBRA"
    metadata = bus_root / "bus_data.csv"
    split_by_patient, selected_images, frozen_identity = {}, set(), {}
    for name, split in (("busbra_development.csv", "development"),
                        ("busbra_internal_holdout.csv", "internal_holdout")):
        manifest = root / "busbra_splits" / name
        if not source(manifest):
            continue
        for row in _read_csv(manifest):
            pid = row["patient_id"]
            previous = split_by_patient.get(pid)
            if previous is not None and previous != split:
                raise ValueError("Frozen BUS-BRA manifests contain cross-split patient leakage")
            split_by_patient[pid] = split
            if row["image_id"] in frozen_identity:
                raise ValueError("Repeated image in frozen BUS-BRA manifests")
            frozen_identity[row["image_id"]] = (pid, split)
            selected_images.add(row["image_id"])
    if source(metadata):
        seen_ids = set()
        for row in _read_csv(metadata):
            if not {"ID", "Case", "Width", "Height"}.issubset(row):
                raise ValueError("BUS-BRA metadata missing actual ID/Case/Width/Height fields")
            iid, pid = row["ID"], row["Case"]
            if iid in seen_ids:
                raise ValueError("Duplicate Image_ID in BUS-BRA metadata")
            seen_ids.add(iid)
            if iid in frozen_identity and frozen_identity[iid][0] != pid:
                raise ValueError("Frozen BUS-BRA image/patient mapping conflicts with source metadata")
            # Naming verified against the existing released BUS-BRA tree.
            ip = _safe_relative(bus_root / "Images", iid + ".png")
            if not iid.startswith("bus_"):
                raise ValueError("Unexpected BUS-BRA ID; explicit mask mapping required")
            mask = _safe_relative(bus_root / "Masks", "mask_" + iid[4:] + ".png")
            rec = _record("BUS-BRA", pid, iid, ip, mask,
                          split_by_patient.get(pid, "unassigned"), "bus_data.csv:Case",
                          row.get("Pathology"), iid in selected_images)
            _inspect(rec, issues, (int(row["Width"]), int(row["Height"])))
            records.append(rec)
        listed = {Path(r["image_path"]).resolve() for r in records}
        for path in (bus_root / "Images").glob("*"):
            if path.is_file() and path.resolve() not in listed:
                _issue(issues, "unlisted_image_file", "BUS-BRA", file_name=path.name)

    lock_path = pilot / "breast_external_test" / "manifest_lock.json"
    if source(lock_path):
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        archive_info = lock.get("sources", {}).get("official_image_archive", {})
        if archive_info.get("path"):
            archive = (lock_path.parent / archive_info["path"]).resolve()
            if source(archive) and archive_info.get("sha256") != sources[str(archive)]:
                _issue(issues, "official_archive_checksum_mismatch", "BrEaST")
        manifest = _safe_relative(lock_path.parent, lock["manifest"]["path"])
        # This base intentionally references a verified sibling asset directory.
        base = (lock_path.parent / lock["manifest"]["image_and_mask_path_base"]).resolve()
        if source(manifest):
            if sources[str(manifest)] != lock["manifest"]["sha256"]:
                raise ValueError("BrEaST frozen manifest checksum mismatch")
            external_rows = _read_csv(manifest)
            seen_images, seen_cases, external_by_case = set(), set(), {}
            for row in external_rows:
                if row["image_id"] in seen_images or row["patient_id"] in seen_cases:
                    raise ValueError("Duplicate image/case in frozen BrEaST manifest")
                seen_images.add(row["image_id"]); seen_cases.add(row["patient_id"])
                rec = _record("BrEaST", row["patient_id"], row["image_id"],
                              _safe_relative(base, row["image_path"]),
                              _safe_relative(base, row["mask_path"]), "external_test",
                              "official_workbook:CaseID; frozen_manifest", row.get("pathology"), True)
                records.append(_inspect(rec, issues, (int(row["width"]), int(row["height"])), row))
                external_by_case[row["patient_id"]] = rec
            workbook_info = lock.get("sources", {}).get("official_workbook", {})
            workbook = (lock_path.parent / workbook_info.get("path", "MISSING")).resolve()
            if source(workbook):
                if workbook_info.get("sha256") and sources[str(workbook)] != workbook_info["sha256"]:
                    _issue(issues, "official_workbook_checksum_mismatch", "BrEaST")
                else:
                    try:
                        clinical = _xlsx_rows(workbook, workbook_info["sheet"])
                        for row in clinical:
                            pid = str(int(float(row["CaseID"])))
                            if pid in seen_cases:
                                _apply_breast_clinical(external_by_case[pid], row, issues, workbook)
                                continue
                            filename = row["Image_filename"]
                            # The official workbook filename is paired with the manifest directory.
                            image_folder = Path(external_rows[0]["image_path"]).parent if external_rows else Path(".")
                            ip = _safe_relative(base, image_folder / filename)
                            pathology = row["Classification"].strip().lower()
                            rec = _record("BrEaST", pid, Path(filename).stem, ip, None,
                                          "external_test", "official_workbook:CaseID", pathology, False)
                            records.append(_inspect(rec, issues))
                            _apply_breast_clinical(rec, row, issues, workbook)
                            _issue(issues, "not_in_lesion_experiment_manifest", "BrEaST", rec["image_id"],
                                   reason="normal_without_primary_mask" if pathology == "normal" else "review_required")
                    except Exception as exc:
                        _issue(issues, "normal_case_inventory_unavailable", "BrEaST", error_type=type(exc).__name__)
            else:
                _issue(issues, "normal_case_inventory_unavailable", "BrEaST",
                       declared_excluded_count=len(lock.get("eligibility", {}).get("excluded_normal_case_ids_without_main_mask", [])))

    for field, code in (("image_sha256", "duplicate_file"), ("pixel_sha256", "duplicate_decoded_pixels")):
        grouped = defaultdict(list)
        for rec in records:
            if rec[field]:
                grouped[rec[field]].append({"dataset": rec["dataset"], "image_id": rec["image_id"]})
        for members in grouped.values():
            if len(members) > 1:
                _issue(issues, code, members=members,
                       cross_dataset=len({m["dataset"] for m in members}) > 1)
    unverified_datasets = sorted({r["dataset"] for r in records if not r["patient_id_verified"]})
    if unverified_datasets:
        _issue(issues, "patient_identity_semantics_unverified", datasets=unverified_datasets,
               explanation="Source case codes without a verified patient mapping are retained; image hashes do not establish identity.")
    _issue(issues, "external_patient_independence_unverified", "BrEaST",
           validation_label="cross_dataset_validation")
    by_dataset = {}
    for dataset in sorted({r["dataset"] for r in records}):
        group = [r for r in records if r["dataset"] == dataset]
        by_dataset[dataset] = {
            "images": len(group), "patient_codes": len({r["patient_id"] for r in group if r["patient_id"]}),
            "patient_identity_verified": all(r["patient_id_verified"] for r in group),
            "complete_annotations": sum(r["annotation_status"] == "complete" for r in group),
            "readable_images": sum(r["data_status"] == "ok" for r in group),
            "frozen_manifest_selected": sum(r["selected_in_frozen_manifest"] for r in group),
        }
    digest_input = [{k: r[k] for k in ("dataset", "patient_id", "image_id", "image_sha256", "mask_sha256", "split")}
                    for r in records]
    checksum = hashlib.sha256(json.dumps({"records": digest_input, "source_checksums": sources},
                                        sort_keys=True).encode("utf-8")).hexdigest()
    return {"schema_version": 1, "records": records,
            "summary": {"images": len(records), "datasets": by_dataset,
                        "issue_counts": dict(Counter(i["code"] for i in issues)),
                        "external_validation_label": "cross_dataset_validation"},
            "issues": issues, "qc_issues": issues, "source_checksums": sources,
            "dataset_checksum": checksum, "dataset_version": checksum[:16]}


def import_manifest(path, root):
    """Import an explicit third-party mapping without filename/identity guesses."""
    rows = _read_csv(path)
    required = {"dataset", "patient_id", "image_id", "image_path", "mask_path"}
    issues, records = [], []
    for row in rows:
        if not required.issubset(row):
            raise ValueError("Explicit manifest requires: " + ", ".join(sorted(required)))
        rec = _record(row["dataset"], row["patient_id"], row["image_id"],
                      _safe_relative(root, row["image_path"]),
                      _safe_relative(root, row["mask_path"]) if row["mask_path"] else None,
                      row.get("split", "unassigned"), "user_supplied_manifest", row.get("pathology"))
        records.append(_inspect(rec, issues))
    return {"records": records, "issues": issues, "source_checksums": {str(Path(path).resolve()): sha256_file(path)}}
