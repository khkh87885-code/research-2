import argparse
import json
from pathlib import Path

from .common import read_json, safe_run


def main(argv=None):
    parser = argparse.ArgumentParser(description="IACS_v2 development-only exploratory validation")
    parser.add_argument("step", choices=("collect", "prepare", "validate", "statistics", "report", "decide"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--output", help="New versioned statistics or report output directory")
    parser.add_argument("--statistics", help="Specific immutable statistics.json for report generation")
    parser.add_argument("--image", help="Original image only for a deployment decision; no reference mask")
    parser.add_argument("--policy", help="Portable frozen JSON policy")
    args = parser.parse_args(argv)
    config = read_json(args.config)
    root = safe_run(config)
    if args.step == "collect":
        from .preflight import validate_config
        from .common import append_event
        checks = validate_config(config)
        append_event(root, "repeat_run_preflight_verified", checks=checks)
        from .collector import collect
        rows, summary = collect(config, args.resume, args.retry_failed)
        print(json.dumps(summary, ensure_ascii=True))
        if summary["statuses"].get("error", 0):
            raise SystemExit(2)
    elif args.step == "prepare":
        from .sensitivity import prepare
        samples = prepare(config)
        print(json.dumps(dict(images=len(samples))))
    elif args.step == "validate":
        from .validation import run_validation
        from .statistics import analyze
        samples = read_json(root / "metrics" / "sensitivity_samples.json")
        validation = run_validation(samples, root / "cross_validation", config.get("validation", {}))
        analyze(samples, validation, root / "statistics", config.get("validation", {}))
        print("Nested Case-level exploratory validation completed")
    elif args.step == "statistics":
        from .statistics import analyze
        from compression_study.datasets import inside
        samples = read_json(root / "metrics" / "sensitivity_samples.json")
        validation = read_json(root / "cross_validation" / "validation.json")
        destination = Path(args.output).resolve() if args.output else root / "statistics" / "review_v2"
        if destination == root / "statistics" or not inside(destination, root / "statistics"):
            raise ValueError("Statistics revision must be a new child inside this run/statistics")
        analyze(samples, validation, destination, config.get("validation", {}))
        print("Versioned statistical review completed")
    elif args.step == "decide":
        from .deployment import decide_image
        if not args.image or not args.policy or not args.output:
            parser.error("decide needs --image, --policy and a new --output directory")
        decision = decide_image(args.image, args.policy, args.output, config)
        print(json.dumps({k: decision[k] for k in ("selected_condition", "reason", "fallback")}, ensure_ascii=True))
    else:
        from .reporting import build_report
        payload = dict(diagnostics=read_json(root / "diagnostics" / "diagnostic_payload.json"),
                       feature_rows=read_json(root / "features" / "feature_records.json"),
                       compression_rows=read_json(root / "metrics" / "sensitivity_long.json"),
                       validation=read_json(root / "cross_validation" / "validation.json"),
                       statistics=read_json(args.statistics if args.statistics else root / "statistics" / "statistics.json"))
        from compression_study.datasets import inside
        destination = Path(args.output).resolve() if args.output else root / "reports" / "publication"
        if destination == root / "reports" or not inside(destination, root / "reports"):
            raise ValueError("Report revision must be a new child inside this run/reports")
        build_report(payload, destination)
        print("Local reports generated")
