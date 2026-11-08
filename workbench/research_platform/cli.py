import argparse
import json
from pathlib import Path
from .common import load_config, write_json


def main():
    parser = argparse.ArgumentParser(description="Local breast ultrasound compression research extension")
    parser.add_argument("--config", default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("scan")
    commands.add_parser("capabilities")
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, default=8765)
    run = commands.add_parser("run")
    run.add_argument("--dataset", choices=["BUS-BRA", "BrEaST"], default="BUS-BRA")
    run.add_argument("--split", default="development")
    run.add_argument("--limit", type=int, default=2)
    run.add_argument("--variants", default="baseline,jpeg_q85,scale_50,cr2_q75_s50,png")
    run.add_argument("--prompt", choices=["mechanism", "application"], default="mechanism")
    run.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    resume = commands.add_parser("resume")
    resume.add_argument("experiment_id")
    export = commands.add_parser("export")
    export.add_argument("experiment_id")
    commands.add_parser("calibrate")
    config = load_config(parser.parse_args().config)
    args = parser.parse_args()
    root = Path(config["output_root"])
    if args.command == "scan":
        from .datasets import scan_datasets
        report = scan_datasets(config)
        write_json(root / "dataset_report" / "inventory.json", report)
        print(json.dumps(report["summary"], ensure_ascii=True, indent=2))
    elif args.command == "capabilities":
        from .compression import capabilities
        print(json.dumps(capabilities(), indent=2))
    elif args.command == "serve":
        from .server import serve
        serve(config, args.port)
    elif args.command in ("run", "resume"):
        from .engine import create_experiment, run_experiment
        exp = args.experiment_id if args.command == "resume" else create_experiment(config, vars(args))
        run_experiment(config, exp)
        print(exp)
    elif args.command == "export":
        from .engine import export_results
        print(json.dumps(export_results(config, args.experiment_id), ensure_ascii=True, indent=2))
    elif args.command == "calibrate":
        from .engine import calibrate_candidate
        print(json.dumps(calibrate_candidate(config), ensure_ascii=True, indent=2))
