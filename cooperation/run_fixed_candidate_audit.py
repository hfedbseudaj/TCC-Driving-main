"""Freeze the crossing joint action family to isolate discount selection.

The adaptive decoder ordinarily contributes one physical action per edge.
Different discount modes can therefore change both the edge reference and
the available joint actions. This single-decision audit takes the union of
all modes' candidates at the same state, then reruns every mode on that
identical finite family. It makes no closed-loop or collision-rate claim.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import uuid

from .scene_closed_loop import (
    DEFAULT_POLICY, MODEL, _choose, _trial_initial, normalize_policy,
)


ROOT = Path(__file__).resolve().parent.parent
MODES = ("ttc", "constant", "equal_share")


def _source_hashes():
    files = ("cooperation/physics.py", "cooperation/scene_closed_loop.py",
             "cooperation/run_fixed_candidate_audit.py")
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in files}


def _result(controls, info):
    return {
        "selected_actions": {name: action.name
                             for name, action in controls.items()},
        "joint_candidate_actions": info["joint_candidate_actions"],
        "contract_references": {
            "--".join(record["edge"]): {
                "reference_share_first": record["reference_share_first"],
                "realized_share_first": record["realized_share_first"],
                "share_error": record["share_error"],
                "reference_actions": record["K_e"]["reference_actions"],
            } for record in info["contract_records"]},
        "safe_candidate_count": info["safe_candidate_count"],
        "individual_rational": info["individual_rational"],
        "fallback": info["fallback"],
        "selected_welfare_m": info["selected_welfare_m"],
        "selected_allocation_error": info["selected_allocation_error"],
        "predicted_min_clearance_m": info["min_clearance"],
        "solver_ms": info["solver_ms"],
    }


def run_audit(output_root, counts, seeds, position_std_m=0.3,
              speed_std_mps=0.2, policy=None):
    if not counts or set(counts) - {2, 3, 4}:
        raise ValueError("Vehicle counts must be 2, 3 or 4")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Trial seeds must be nonempty and unique")
    if position_std_m < 0 or speed_std_mps < 0:
        raise ValueError("Perturbation scales must be nonnegative")
    settings = dict(policy or {})
    settings["decoder_mode"] = "adaptive_crossing"
    policy = normalize_policy(settings)
    output_root = Path(output_root).resolve()
    if ROOT.as_posix().startswith("/data/hhr/"):
        designated = Path("/data/hhr").resolve()
        if designated != output_root and designated not in output_root.parents:
            raise ValueError("Server output must stay under /data/hhr")
    trials = []
    summary = {}
    for count in counts:
        group = []
        for seed in seeds:
            cars, edges, _ = _trial_initial(
                "crossing", count, seed, position_std_m, speed_std_mps)
            mode_specific = {}
            for mode in MODES:
                mode_policy = normalize_policy(dict(policy, discount_mode=mode))
                controls, info = _choose("crossing", cars, edges, "global",
                                         mode_policy)
                mode_specific[mode] = _result(controls, info)
            shared_names = {name: sorted(set().union(*(
                set(mode_specific[mode]["joint_candidate_actions"][name])
                for mode in MODES))) for name in cars}
            frozen = {}
            for mode in MODES:
                mode_policy = normalize_policy(dict(policy, discount_mode=mode))
                controls, info = _choose("crossing", cars, edges, "global",
                                         mode_policy, shared_names)
                frozen[mode] = _result(controls, info)
                if frozen[mode]["joint_candidate_actions"] != shared_names:
                    raise AssertionError("Joint candidate family differs")
            row = {
                "scene": "crossing", "n_cars": count, "trial_seed": seed,
                "edges": [list(edge) for edge in edges],
                "shared_joint_action_family": shared_names,
                "mode_specific": mode_specific,
                "fixed_family": frozen,
            }
            trials.append(row)
            group.append(row)
        summary[str(count)] = {}
        for reference in ("constant", "equal_share"):
            summary[str(count)]["ttc_vs_" + reference] = {
                "trial_count": len(group),
                "mode_specific_action_changed_trials": sum(
                    row["mode_specific"]["ttc"]["selected_actions"] !=
                    row["mode_specific"][reference]["selected_actions"]
                    for row in group),
                "fixed_family_action_changed_trials": sum(
                    row["fixed_family"]["ttc"]["selected_actions"] !=
                    row["fixed_family"][reference]["selected_actions"]
                    for row in group),
                "fixed_family_equal_for_all_modes": all(
                    len({json.dumps(row["fixed_family"][mode][
                        "joint_candidate_actions"], sort_keys=True)
                         for mode in MODES}) == 1 for row in group),
            }
    run_id = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") +
              "_" + uuid.uuid4().hex[:8])
    directory = output_root / run_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "trials.json").write_text(
        json.dumps(trials, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    (directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    manifest = {
        "experiment": "fixed_joint_candidate_snapshot_ablation",
        "status": "constructed_single_decision_no_execution",
        "vehicle_counts": list(counts), "trial_seeds": list(seeds),
        "position_perturbation_std_m": position_std_m,
        "speed_perturbation_std_mps": speed_std_mps,
        "training_seed": None, "checkpoint_path": None,
        "policy": policy, "model": MODEL,
        "modes": list(MODES),
        "shared_action_family_definition": "Union of TTC, constant and equal-share pair-decoder candidates at one frozen state; all modes then use exactly this family.",
        "source_sha256": _source_hashes(),
        "python": sys.version, "platform": platform.platform(),
        "limitations": [
            "The union is constructed using outputs of all compared policies at the same state; this is an analysis control, not a deployable controller.",
            "Only one decision is compared; no closed-loop collision or efficiency conclusion follows.",
            "The safety and welfare claims are limited to finite sampled kinematics.",
        ],
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    return directory, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "Result" / "fixed_candidate_audit")
    parser.add_argument("--n-cars", nargs="+", type=int, choices=(2, 3, 4),
                        default=(2, 3, 4))
    parser.add_argument("--trial-seeds", nargs="+", type=int,
                        default=tuple(range(301, 311)))
    parser.add_argument("--position-std", type=float, default=0.3)
    parser.add_argument("--speed-std", type=float, default=0.2)
    parser.add_argument("--alpha", type=float,
                        default=DEFAULT_POLICY["alpha_per_s"])
    parser.add_argument("--eta-m", type=float,
                        default=DEFAULT_POLICY["surplus_tolerance_m"])
    parser.add_argument("--welfare-epsilon-m", type=float,
                        default=DEFAULT_POLICY["welfare_epsilon_m"])
    parser.add_argument("--joint-objective",
                        choices=("contract_fidelity", "contract_then_share",
                                 "share_then_contract"),
                        default=DEFAULT_POLICY["joint_objective"])
    args = parser.parse_args(argv)
    policy = {
        "alpha_per_s": args.alpha,
        "surplus_tolerance_m": args.eta_m,
        "welfare_epsilon_m": args.welfare_epsilon_m,
        "joint_objective": args.joint_objective,
    }
    directory, summary = run_audit(args.output_root, args.n_cars,
                                    args.trial_seeds, args.position_std,
                                    args.speed_std, policy)
    print(json.dumps({"result_dir": str(directory), "summary": summary},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
