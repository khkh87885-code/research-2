"""Verify published file hashes and frozen study counts using the standard library."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/iacs_v2/dev_20261108_123935"


def read_bytes(path):
    path = Path(path)
    return path.read_bytes() if path.exists() else gzip.decompress(path.with_name(path.name + ".gz").read_bytes())


def load(path):
    return json.loads(read_bytes(path).decode("utf-8-sig"))


def verify():
    manifest = load(ROOT / "PUBLICATION_MANIFEST.json")
    for record in manifest["files"]:
        path = ROOT / record["path"]
        data = path.read_bytes()
        assert len(data) == record["bytes"], record["path"]
        assert hashlib.sha256(data).hexdigest() == record["sha256"], record["path"]
    sources = load(ROOT / "PUBLICATION_SOURCE_RECORDS.json")
    for record in sources:
        if record["transformation"].startswith("publication-only redaction"):
            continue
        data = (ROOT / record["path"]).read_bytes()
        if record["path"].endswith(".gz"):
            data = gzip.decompress(data)
        assert hashlib.sha256(data).hexdigest() == record["source_sha256"], record["path"]
    samples = load(RUN / "metrics/sensitivity_samples.json")
    assert hashlib.sha256(read_bytes(RUN / "metrics/sensitivity_samples.json")).hexdigest() == "671c1d1eb7323d8480993c8c5b2e286836dc72b802a01be04582d0ee5d504e47"
    assert len(samples) == 213 and len({s["case_key"] for s in samples}) == 213
    assert sum(len(s["candidates"]) for s in samples) == 1491
    validation = load(RUN / "cross_validation/validation.json")
    assert len(validation["outer_decisions"]) == 852
    assert len(validation["fold_assignments"]) == 213
    assert len(list((RUN / "compression").rglob("result.json"))) == 1491
    for fold in validation["folds"]:
        train, valid = set(fold["train_case_keys"]), set(fold["validation_case_keys"])
        assert not (train & valid)
        assert len(train | valid) == 213
    assert validation["config"]["hd95_margin"] is None
    print(json.dumps({"hashes_verified": len(manifest["files"]), "cases": 213,
                      "candidates": 1491, "outer_decisions": 852,
                      "clinical_noninferiority_claim": False}, indent=2))


if __name__ == "__main__":
    verify()
