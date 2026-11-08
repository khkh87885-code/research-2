"""PowerShell-friendly command interface, using only local paths."""
import argparse
import json
from pathlib import Path

from .datasets import check_output, inventory
from .engine import read_results, run_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local SAM compression research extension")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inventory", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True)
        if name == "inventory":
            sub.add_argument("--output", required=True)
        else:
            sub.add_argument("--resume", action="store_true")
            sub.add_argument("--retry-failed", action="store_true")
    for name in ("export", "dashboard"):
        sub = commands.add_parser(name)
        sub.add_argument("--run", required=True)
        sub.add_argument("--output", required=True)
        sub.add_argument("--mixed-effects", action="store_true")
    fit = commands.add_parser("fit-policy")
    fit.add_argument("--run", required=True)
    fit.add_argument("--settings", required=True, help="Prospectively specified margins, feature names and split parameters")
    fit.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parent.parent
    if args.command in ("inventory", "run"):
        config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
        if args.command == "inventory":
            result = inventory(config, project, args.output)
            print(json.dumps(dict(failures=result["failures"], datasets=result["datasets"])))
        else:
            rows = run_experiment(config, project, args.resume, args.retry_failed)
            print(json.dumps(dict(tasks=len(rows), ok=sum(r["status"] == "ok" for r in rows))))
        return
    run = Path(args.run).resolve(strict=True)
    metadata = json.loads((run / "run.json").read_text(encoding="utf-8"))
    config = metadata["provenance"]["config"]
    output = check_output(args.output, config)
    # Also reject any export location that would overwrite task records or provenance.
    if output.exists():
        raise FileExistsError("Export destination exists; choose a new output path")
    if output == run or (run / "tasks") == output or (run / "tasks") in output.parents:
        raise ValueError("Export cannot overwrite the run or task records")
    rows = read_results(run)
    if args.command == "fit-policy":
        from .adaptive import fit_policy
        from .engine import write_json
        settings = json.loads(Path(args.settings).read_text(encoding="utf-8-sig"))
        policy = fit_policy(rows, None, settings)
        policy["source_run_fingerprint_sha256"] = metadata["fingerprint_sha256"]
        # Recompute the portable content lock after adding source provenance.
        import hashlib
        from .engine import canonical
        policy.pop("policy_sha256", None)
        policy["policy_sha256"] = hashlib.sha256(canonical(policy).encode()).hexdigest()
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, policy)
        print("Development-only research policy frozen; external performance pending")
        return
    from .statistics import summarize
    stats = summarize(rows, bootstrap=config.get("bootstrap", 1000), seed=config.get("seed", 20261108), delta=config.get("noninferiority_delta"))
    if args.mixed_effects:
        from .statistics import mixed_effects
        stats["mixed_effects"] = mixed_effects(rows)
    public_meta = {k: metadata.get(k) for k in ("purpose", "started_at", "runtime", "limitations", "fingerprint_sha256")}
    if args.command == "export":
        from .export import export_results
        result = export_results(rows, output, metadata=public_meta, statistics=stats)
        print(json.dumps(result, default=str))
    else:
        from .dashboard import build_dashboard
        result = build_dashboard(rows, output, metadata=public_meta, statistics=stats)
        print(str(result))
