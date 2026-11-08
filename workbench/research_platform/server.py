"""Loopback-only workbench, no cloud image requests and no external assets."""
import json
import mimetypes
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from .common import ROOT, contained, read_json, write_json, hardware, sha256
from . import engine


def _iacs_report_directory(config, run, report):
    """Only the configured pilot's versioned IACS_v2 reports are readable."""
    if not all(re.fullmatch(r"[A-Za-z0-9_-]{1,90}", str(value)) for value in (run, report)):
        raise ValueError("Invalid IACS report identifier")
    if not config.get("pilot_root"):
        raise ValueError("IACS pilot root is not configured")
    pilot = Path(config["pilot_root"]).resolve()
    expected_root = pilot / "research_output" / "IACS_v2"
    root = expected_root.resolve()
    if root != expected_root:
        raise ValueError("Linked IACS output roots are not permitted")
    directory = (root / run / "reports" / report).resolve(strict=True)
    if root not in directory.parents:
        raise ValueError("IACS report escaped its fixed root")
    manifest_path = directory / "report_manifest.json"
    if manifest_path.resolve(strict=True).parent != directory:
        raise ValueError("Linked IACS manifest is not permitted")
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "IACS_v2_report_1":
        raise ValueError("Unknown IACS report schema")
    return directory, manifest


def _iacs_file(config, run, report, name):
    fixed = {"report.json", "report_manifest.json", "dashboard.html", "IACS_v2_results.xlsx", "REPORT_STATUS.md",
             "features.csv", "compression.csv", "candidate_scores.csv", "outer_decisions.csv", "folds.csv", "feature_associations.csv",
             "group_summaries.csv", "paired_comparisons.csv", "subgroups.csv", "mixed_effects.csv", "prediction_performance.csv"}
    if name not in fixed and not re.fullmatch(r"figure_0[1-8]\.(png|svg|pdf)", str(name)):
        raise ValueError("Not a permitted IACS report file")
    directory, manifest = _iacs_report_directory(config, run, report)
    target = (directory / name).resolve(strict=True)
    if target.parent != directory:
        raise ValueError("Linked IACS files are not permitted")
    if name != "report_manifest.json":
        entry = next((row for row in manifest.get("files", []) if row.get("name") == name), None)
        if entry is None or entry.get("sha256") != sha256(target):
            raise ValueError("IACS report file is absent from the manifest or changed")
    return target


def make_handler(config, port):
    jobs = {}
    mutex = threading.Lock()
    state = {"scanning": False}

    def inventory_or_empty():
        try:
            return engine.inventory(config)
        except ValueError:
            return {"summary": {"status": "not_scanned"}, "records": []}

    def start(experiment_id):
        def work():
            try:
                engine.run_experiment(config, experiment_id)
            except Exception as error:
                # No patient names/clinical metadata or raw pixels enter service logs.
                write_json(engine.experiment_dir(config, experiment_id) / "last_service_error.json", {"type": type(error).__name__, "error": str(error)})
        with mutex:
            if any(t.is_alive() for t in jobs.values()):
                raise ValueError("Another experiment is running. Wait for completion.")
            thread = threading.Thread(target=work, daemon=True)
            jobs[experiment_id] = thread
            thread.start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body, status=200, content_type="application/json; charset=utf-8"):
            content = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(content)

        def valid_host(self):
            if self.headers.get("Host") not in ("127.0.0.1:%d" % port, "localhost:%d" % port):
                raise ValueError("Invalid local Host")

        def do_GET(self):
            try:
                self.valid_host()
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                path = parsed.path
                if path == "/":
                    self.send((ROOT / "web" / "index.html").read_bytes(), content_type="text/html; charset=utf-8")
                elif path == "/iacs-v2":
                    self.send((ROOT / "web" / "iacs_v2.html").read_bytes(), content_type="text/html; charset=utf-8")
                elif path == "/api/iacs-v2/runs":
                    reports = []
                    if config.get("pilot_root"):
                        root = (Path(config["pilot_root"]) / "research_output" / "IACS_v2").resolve()
                        for manifest_path in sorted(root.glob("*/reports/*/report_manifest.json"), reverse=True):
                            run, report = manifest_path.parents[2].name, manifest_path.parent.name
                            try:
                                directory, manifest = _iacs_report_directory(config, run, report)
                                reports.append({"run": run, "report": report, "summary": manifest.get("summary", {}),
                                                "notice": manifest.get("notice"), "schema_version": manifest.get("schema_version")})
                            except (ValueError, FileNotFoundError):
                                continue
                    self.send({"reports": reports, "analysis_unit": "Case-level", "read_only": True})
                elif path in ("/api/iacs-v2/report", "/api/iacs-v2/file"):
                    target = _iacs_file(config, query.get("run", [""])[0], query.get("report", [""])[0],
                                        "report.json" if path.endswith("/report") else query.get("name", [""])[0])
                    self.send(target.read_bytes(), content_type=mimetypes.guess_type(str(target))[0] or "application/octet-stream")
                elif path == "/api/overview":
                    experiments = []
                    root = Path(config["output_root"]) / "experiments"
                    for p in sorted(root.glob("EXP_*"), reverse=True):
                        if not (p / "config.json").exists():
                            continue
                        frozen = read_json(p / "config.json")
                        summary = engine.Store(p).summary()
                        summary.update(completed=summary["counts"].get("completed", 0), failed=summary["counts"].get("failed", 0))
                        experiments.append({"experiment_id": p.name, "name": frozen["name"], "summary": summary, "status": summary["status"]})
                    self.send({"config": config, "dataset_summary": inventory_or_empty()["summary"], "experiments": experiments,
                               "hardware": hardware(), "scanning": state["scanning"]})
                elif path == "/api/datasets":
                    self.send(inventory_or_empty())
                elif path.startswith("/api/experiments/"):
                    self.send(engine.details(config, path.split("/")[3]))
                elif path == "/api/files":
                    target = contained(config["output_root"], query.get("path", [""])[0])
                    if not target.is_file() or target.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".jp2", ".csv", ".json", ".jsonl", ".xlsx", ".md", ".svg", ".pdf"):
                        raise ValueError("Not an exportable output file")
                    self.send(target.read_bytes(), content_type=mimetypes.guess_type(str(target))[0] or "application/octet-stream")
                elif path == "/api/features":
                    from .features import extract_features
                    record = next((r for r in inventory_or_empty()["records"] if r["dataset"] == query.get("dataset", [None])[0] and r["image_id"] == query.get("image_id", [None])[0]), None)
                    if record is None:
                        raise ValueError("Unknown dataset image")
                    values = extract_features(record["image_path"])
                    candidate = Path(config["output_root"]) / "experiment_configs" / "iacs_candidate.json"
                    if candidate.exists():
                        from .adaptive import decide
                        values["decision"] = decide(values, read_json(candidate))
                    self.send(values)
                elif path == "/api/source":
                    from io import BytesIO
                    from PIL import Image
                    from .datasets import load_rgb, load_mask
                    record = next((r for r in inventory_or_empty()["records"] if r["dataset"] == query.get("dataset", [None])[0] and r["image_id"] == query.get("image_id", [None])[0]), None)
                    if record is None:
                        raise ValueError("Unknown source image")
                    role = query.get("role", ["image"])[0]
                    if role not in ("image", "mask"):
                        raise ValueError("Unknown source role")
                    if role == "mask" and not record.get("mask_path"):
                        raise ValueError("This normal case has no primary reference mask")
                    pixels = load_rgb(record["image_path"]) if role == "image" else load_mask(record["mask_path"]).astype("uint8") * 255
                    buffer = BytesIO()
                    Image.fromarray(pixels).save(buffer, "PNG")
                    self.send(buffer.getvalue(), content_type="image/png")
                elif path == "/api/iacs":
                    candidate = Path(config["output_root"]) / "experiment_configs" / "iacs_candidate.json"
                    self.send(read_json(candidate) if candidate.exists() else {"state": "not_calibrated", "instruction": "Run the development-only feature calibration action"})
                else:
                    self.send({"error": "Not found"}, 404)
            except (ValueError, FileNotFoundError, StopIteration, KeyError) as error:
                self.send({"error": str(error)}, 400)
            except Exception as error:
                self.send({"error": type(error).__name__ + ": " + str(error)}, 500)

        def do_POST(self):
            try:
                self.valid_host()
                origin = self.headers.get("Origin")
                if origin and origin not in ("http://127.0.0.1:%d" % port, "http://localhost:%d" % port):
                    raise ValueError("Cross-origin requests denied")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("JSON requests required")
                length = int(self.headers.get("Content-Length", "0"))
                if length > 65536:
                    raise ValueError("Request too large")
                body = json.loads(self.rfile.read(length) or b"{}")
                path = urlparse(self.path).path
                if path == "/api/scan":
                    from .datasets import scan_datasets
                    with mutex:
                        if state["scanning"]:
                            raise ValueError("Dataset scan already running")
                        state["scanning"] = True
                    try:
                        report = scan_datasets(config)
                        write_json(Path(config["output_root"]) / "dataset_report" / "inventory.json", report)
                        self.send({"summary": report["summary"]})
                    finally:
                        state["scanning"] = False
                elif path == "/api/experiments":
                    if any(t.is_alive() for t in jobs.values()):
                        raise ValueError("Another experiment is running")
                    experiment_id = engine.create_experiment(config, body)
                    start(experiment_id)
                    self.send({"experiment_id": experiment_id, "status": "submitted"}, 202)
                elif path.startswith("/api/experiments/"):
                    parts = path.split("/")
                    experiment_id, action = parts[3], parts[4]
                    if action == "resume":
                        start(experiment_id)
                        self.send({"status": "submitted", "message": "Completed tasks preserved; unfinished/failed tasks scheduled"}, 202)
                    elif action in ("export", "analyze"):
                        if jobs.get(experiment_id) and jobs[experiment_id].is_alive():
                            raise ValueError("Wait for this experiment to finish before creating a snapshot")
                        self.send(engine.export_results(config, experiment_id) if action == "export" else engine.analyze_results(config, experiment_id, delta=body.get("delta")))
                    else:
                        raise ValueError("Unknown action")
                elif path == "/api/iacs/calibrate":
                    self.send(engine.calibrate_candidate(config))
                elif path == "/api/iacs/lock":
                    from .adaptive import lock_candidate
                    candidate = Path(config["output_root"]) / "experiment_configs" / "iacs_candidate.json"
                    self.send({"lock_file": lock_candidate(candidate, config)})
                else:
                    self.send({"error": "Not found"}, 404)
            except (ValueError, FileNotFoundError, FileExistsError, KeyError, IndexError) as error:
                self.send({"error": str(error)}, 400)
            except Exception as error:
                self.send({"error": type(error).__name__ + ": " + str(error)}, 500)

    return Handler


def serve(config, port=8765):
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(config, port))
    print("Local workbench: http://127.0.0.1:%d" % port, flush=True)
    server.serve_forever()
