"""Replay all OOF choices; optionally refit nested CV using published anonymous data.

No images, masks, weights, original identifiers or pseudonym keys are required.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

from verify_publication import ROOT, RUN, load

sys.path.insert(0, str(ROOT / "pilot"))
from iacs_v2.validation import predict_policy, run_validation, verify_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refit-output", type=Path, help="New, empty local output directory for a complete numerical nested CV refit")
    args = parser.parse_args()
    samples = load(RUN / "metrics/sensitivity_samples.json")
    validation = load(RUN / "cross_validation/validation.json")
    index = {s["image_key"]: s for s in samples}
    policies = {}
    for decision in validation["outer_decisions"]:
        key = (decision["fold"], decision["underlying_family"])
        if key not in policies:
            policies[key] = load(RUN / "cross_validation" / ("fold_%d_%s_policy.json" % key))
            verify_policy(policies[key])
        sample = index[decision["image_key"]]
        replay = predict_policy(policies[key], sample["features"], sample["candidates"])
        assert replay["selected_condition"] == decision["selected_condition"]
        corrupted = copy.deepcopy(sample["candidates"])
        for candidate in corrupted.values():
            for metric in ("dice", "hd95", "assd", "delta_dice", "delta_hd95"):
                candidate[metric] = -999999
            candidate["status"] = "intentional_evaluation_corruption"
        blind = predict_policy(policies[key], sample["features"], corrupted)
        assert blind["selected_condition"] == decision["selected_condition"]
    result = {"replayed_oof_decisions": 852, "decisions_unchanged_after_outcome_corruption": 852}
    if args.refit_output:
        fresh = run_validation(samples, args.refit_output, validation["config"])
        keys = ("case_key", "image_key", "fold", "algorithm", "selected_condition", "underlying_family")
        signature = lambda rows: sorted(tuple(r[k] for k in keys) for r in rows)
        assert signature(fresh["outer_decisions"]) == signature(validation["outer_decisions"])
        assert fresh["full_policy"]["policy_sha256"] == validation["full_policy"]["policy_sha256"]
        result["refit_oof_choices_exact"] = True
        result["full_policy_sha256_exact"] = True
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
