import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from compression_study.datasets import sha256, inside


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def append_event(root, event, **detail):
    path = Path(root) / "logs" / "stages.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical(dict(time=now(), event=event, **detail)) + "\n")


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    def cell(value):
        if isinstance(value, (dict, list, tuple)):
            value = canonical(value)
        if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
            return "'" + value
        return value
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({k: cell(row.get(k)) for k in fields} for row in rows)


def require_development(config):
    if not config.get("datasets") or any(s.get("dataset") != "BUS-BRA" or s.get("split") != "development"
                                          for s in config["datasets"]):
        raise ValueError("IACS_v2 accepts BUS-BRA development only; external/holdout rejected")
    if config.get("prompt_strategy", "mechanism_gt_box") != "mechanism_gt_box":
        raise ValueError("This versioned sensitivity study fixes mechanism_gt_box; application data stay separate")


def safe_run(config):
    from compression_study.datasets import check_output
    project = Path(config["project_root"]).resolve(strict=True)
    base = project / "research_output" / "IACS_v2"
    root = check_output(config["run_root"], config)
    if root == base or not inside(root, base):
        raise ValueError("Run must be a versioned child of project/research_output/IACS_v2")
    return root
