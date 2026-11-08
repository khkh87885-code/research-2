import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
        return
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(str(temp), str(path))


def load_config(path=None):
    config = read_json(path or ROOT / "config.json")
    output = Path(config["output_root"]).resolve()
    protected = [Path(config[key]).resolve() for key in ("data_root", "pilot_root", "sam_source")]
    protected.append(Path(config["checkpoint"]).resolve().parent)
    for source in protected:
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("Output must be outside original data, code and model roots")
    config["output_root"] = str(output)
    return config


def contained(root, relative):
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    if root not in candidate.parents:
        raise ValueError("Path is outside permitted output directory")
    return candidate


def source_snapshot():
    paths = list((ROOT / "research_platform").glob("*.py")) + [ROOT / "web" / "index.html", ROOT / "run.py", ROOT / "requirements-extra.txt"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths) if p.is_file()}


def runtime():
    import importlib.metadata
    versions = {}
    for name in ("numpy", "Pillow", "torch", "torchvision", "scipy", "pandas", "statsmodels", "openpyxl", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "platform": platform.platform(), "libraries": versions,
            "processor": platform.processor(), "executable": sys.executable}


def hardware():
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"], text=True, timeout=10)
        return {"gpu": output.strip(), "cpu": platform.processor()}
    except (OSError, subprocess.SubprocessError):
        return {"gpu": None, "cpu": platform.processor()}
