"""E1 audit for two-car TTC/Rubinstein decoders in four scene families.

The crossing decoder uses reachable entry-time trajectory pairs.  The other
three scene-specific decoders use their explicit finite control families.
Every trial records the zero-distance set of the selected K_e reference and
whether the two-player cooperative selection regresses to that reference.
All scenes and parameters are constructed kinematic examples, not empirical
calibration or CARLA results.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import uuid

from .crossing import PairProposal, decode_crossing_pair
from .crossing_closed_loop import _candidate_times
from .physics import canonical_approach_plan, pair_safe
from .run_paired_closed_loop import _trial_initial_conditions
from .run_snapshot import _vehicle_from_row, _write_json
from .scene_extensions import (
    EXTENSION_FIXED_MODEL, _distance, _pair_decode, _rollout, _safe,
    _scene_data, evaluate_scene,
)
from .compose import compose_crossing_snapshot


ROOT = Path(__file__).resolve().parent.parent
SCENES = ("intersection", "chase", "merge", "head_on")


def _source_hash():
    digest = hashlib.sha256()
    for relative in (
            "cooperation/physics.py", "cooperation/crossing.py",
            "cooperation/compose.py", "cooperation/crossing_closed_loop.py",
            "cooperation/scene_extensions.py", "cooperation/run_decoder_audit.py"):
        digest.update(relative.encode("utf-8"))
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def _crossing_trial(config, seed, position_std_m, speed_std_mps):
    vehicles, perturbations, _ = _trial_initial_conditions(
        config, seed, position_std_m, speed_std_mps, 0.0)
    row = config["pair_proposals"][0]
    first_id, second_id = row["edge"]
    first, second = vehicles[first_id], vehicles[second_id]
    horizon = float(config["physics"]["horizon_s"])
    candidates = {
        first_id: _candidate_times(first, horizon, 0.2, 4.0),
        second_id: _candidate_times(second, horizon, 0.2, 4.0),
    }
    clearance = float(config["physics"]["clearance_s"])
    safe_pairs = []
    for first_time, second_time in itertools.product(
            candidates[first_id], candidates[second_id]):
        if pair_safe(canonical_approach_plan(first, first_time),
                     canonical_approach_plan(second, second_time), clearance):
            safe_pairs.append((first_time, second_time))
    if not safe_pairs:
        raise ValueError("No pair-safe reachable entry-time pair")
    disagreement = max(safe_pairs, key=lambda pair: (sum(pair), pair))
    proposals = [PairProposal("e1_{:.3f}_{:.3f}".format(*pair), *pair)
                 for pair in safe_pairs]
    bargaining = config["bargaining"]
    result = decode_crossing_pair(
        first, second, disagreement, proposals,
        float(bargaining["discount_alpha_s_inv"][first_id]),
        float(bargaining["discount_alpha_s_inv"][second_id]),
        float(bargaining["eta_s"]), clearance, (0.1, 0.1))
    reference = (result.contract.first_reference_s,
                 result.contract.second_reference_s)
    zero_ids = [proposal.proposal_id for proposal in proposals
                if (abs(proposal.first_entry_s - reference[0]) <= 1e-10 and
                    abs(proposal.second_entry_s - reference[1]) <= 1e-10)]
    joint = compose_crossing_snapshot(
        {first_id: first, second_id: second}, [result.contract], candidates,
        float(bargaining["welfare_epsilon_s"]),
        float(bargaining["t_ref_s"]))
    selected = joint.global_plan.entry_dict()
    regression_error_s = (abs(selected[first_id] - reference[0]) +
                          abs(selected[second_id] - reference[1]))
    selected_safe = pair_safe(
        canonical_approach_plan(first, selected[first_id]),
        canonical_approach_plan(second, selected[second_id]), clearance)
    selected_score = next(score for score in result.candidate_scores
                          if score.proposal_id == result.contract.reference_id)
    return {
        "scene": "intersection",
        "status": "ok",
        "trial_seed": int(seed),
        "training_seed": None,
        "perturbations": perturbations,
        "player_count": 2,
        "edge": [first_id, second_id],
        "candidate_pair_count": len(proposals),
        "K_e": asdict(result.contract),
        "qhat_entry_times_s": {first_id: reference[0], second_id: reference[1]},
        "d_c_zero_set_count": len(zero_ids),
        "d_c_zero_set_ids": zero_ids,
        "unique_zero_set": len(zero_ids) == 1,
        "SPE_share_error": selected_score.share_error,
        "individual_gains_s": {
            first_id: selected_score.first_gain_s,
            second_id: selected_score.second_gain_s,
        },
        "surplus_s": selected_score.surplus_s,
        "N2_safe": selected_safe,
        "N2_regression_error_s": regression_error_s,
        "N2_exact_reference_regression": regression_error_s <= 1e-9,
        "N2_selected_entry_times_s": selected,
        "N2_global_broken_contracts": joint.global_plan.broken_contracts,
        "finite_search_product": joint.enumerated_plan_count,
    }


def _extension_trial(scene, seed, alpha_per_s, eta_m,
                     welfare_epsilon_m):
    cars, action_sets, edges = _scene_data(scene, n_cars=2, trial_seed=seed)
    first_id, second_id = edges[0]
    first, second = cars[first_id], cars[second_id]
    plans = {
        first_id: {action.name: _rollout(first, action) for action in action_sets[first_id]},
        second_id: {action.name: _rollout(second, action) for action in action_sets[second_id]},
    }
    contract = _pair_decode(
        scene, first, second, action_sets[first_id], action_sets[second_id],
        plans, alpha_per_s, eta_m)
    ref_first = next(a for a in action_sets[first_id]
                     if a.name == contract.first_control)
    ref_second = next(a for a in action_sets[second_id]
                      if a.name == contract.second_control)
    zero_pairs = []
    safe_candidate_count = 0
    for action_first, action_second in itertools.product(
            action_sets[first_id], action_sets[second_id]):
        if not _safe(scene, first, plans[first_id][action_first.name],
                     second, plans[second_id][action_second.name]):
            continue
        safe_candidate_count += 1
        distance = _distance(scene, action_first, ref_first,
                             action_second, ref_second)
        if distance <= 1e-12:
            zero_pairs.append([action_first.name, action_second.name])
    joint = evaluate_scene(scene, alpha_per_s=alpha_per_s, eta_m=eta_m,
                           welfare_epsilon_m=welfare_epsilon_m,
                           n_cars=2, trial_seed=seed)
    global_controls = dict(joint.global_controls)
    selected_first = next(a for a in action_sets[first_id]
                          if a.name == global_controls[first_id])
    selected_second = next(a for a in action_sets[second_id]
                           if a.name == global_controls[second_id])
    regression_error = _distance(scene, selected_first, ref_first,
                                 selected_second, ref_second)
    return {
        "scene": scene,
        "status": "ok",
        "trial_seed": int(seed),
        "training_seed": None,
        "player_count": 2,
        "edge": [first_id, second_id],
        "safe_pair_candidate_count": safe_candidate_count,
        "K_e": asdict(contract),
        "qhat_control_ids": {
            first_id: contract.first_control,
            second_id: contract.second_control,
        },
        "d_c_zero_set_count": len(zero_pairs),
        "d_c_zero_set_ids": zero_pairs,
        "unique_zero_set": len(zero_pairs) == 1,
        "SPE_share_error": contract.share_error,
        "SPE_reference_share": contract.reference_share,
        "surplus_m": contract.surplus_m,
        "N2_safe": True,
        "N2_regression_error": regression_error,
        "N2_exact_reference_regression": regression_error <= 1e-12,
        "N2_selected_control_ids": global_controls,
        "N2_global_broken_contracts": joint.global_broken_contracts,
        "N2_welfare_loss_m": max(0.0, joint.welfare_optimum_m -
                                  joint.global_welfare_m),
        "finite_search_product": joint.enumerated_plan_count,
    }


def run_decoder_audit(config_path, output_root, trial_seeds,
                      position_std_m=0.3, speed_std_mps=0.2,
                      alpha_per_s=0.2, eta_m=4.0,
                      welfare_epsilon_m=5.0):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("mode") != "constructed_example":
        raise ValueError("E1 requires the explicit constructed crossing config")
    if not trial_seeds or len(set(trial_seeds)) != len(trial_seeds):
        raise ValueError("Provide nonempty unique scenario trial seeds")
    output_root = Path(output_root).resolve()
    if ROOT.as_posix().startswith("/data/hhr/"):
        allowed = Path("/data/hhr").resolve()
        if output_root != allowed and allowed not in output_root.parents:
            raise ValueError("Server output must remain under /data/hhr")
    rows = []
    for seed in trial_seeds:
        for scene in SCENES:
            try:
                if scene == "intersection":
                    row = _crossing_trial(config, seed, position_std_m,
                                          speed_std_mps)
                else:
                    row = _extension_trial(scene, seed, alpha_per_s, eta_m,
                                           welfare_epsilon_m)
                rows.append(row)
            except (ValueError, KeyError, StopIteration) as error:
                rows.append({"scene": scene, "status": "failed",
                             "trial_seed": int(seed),
                             "failure_reason": str(error)})
    summary = {}
    for scene in SCENES:
        group = [row for row in rows if row["scene"] == scene]
        valid = [row for row in group if row["status"] == "ok"]
        summary[scene] = {
            "trial_count": len(group),
            "successful_trials": len(valid),
            "failed_trials": len(group) - len(valid),
            "unique_zero_set_count": sum(row["unique_zero_set"] for row in valid),
            "exact_N2_regression_count": sum(
                row["N2_exact_reference_regression"] for row in valid),
            "safe_count": sum(row["N2_safe"] for row in valid),
            "mean_share_error": (
                sum(row["SPE_share_error"] for row in valid) / len(valid)
                if valid else None),
            "mean_N2_regression_error": (
                sum(row["N2_regression_error_s" if scene == "intersection"
                     else "N2_regression_error"] for row in valid) / len(valid)
                if valid else None),
            "mean_N2_welfare_loss_m": (
                sum(row.get("N2_welfare_loss_m", 0.0) for row in valid) / len(valid)
                if valid else None),
        }
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "trials.json", rows)
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "manifest.json", {
        "experiment": "E1_two_car_decoder_zero_set_and_N2_regression",
        "status": "constructed_finite_kinematic_audit_not_dataset_or_CARLA",
        "scenes": list(SCENES),
        "trial_seeds": list(trial_seeds),
        "trial_count_per_scene": len(trial_seeds),
        "training_seed": None,
        "checkpoint_path": None,
        "position_perturbation_std_m": position_std_m,
        "speed_perturbation_std_mps": speed_std_mps,
        "intersection_parameters": {
            "alpha_per_s": config["bargaining"]["discount_alpha_s_inv"],
            "eta_s": config["bargaining"]["eta_s"],
            "clearance_s": config["physics"]["clearance_s"],
            "candidate_grid_s": 0.2,
        },
        "other_scene_parameters": {
            "alpha_per_s": alpha_per_s,
            "eta_m": eta_m,
            "welfare_epsilon_m": welfare_epsilon_m,
            "physical_model": EXTENSION_FIXED_MODEL,
        },
        "source_sha256": _source_hash(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "unique_zero_test": "count candidate trajectories/actions with scene-specific distance d_c <= 1e-12 from the selected qhat; multiple zeros are reported as a feature-equivalence set",
        "N2_regression_test": "compare two-player joint selector output against the pair decoder reference under the same finite candidates and safety filter",
        "limitations": [
            "Other three scene decoders are provisional action-family definitions, not original learned-offer policies.",
            "Exactness and zero-set checks apply only to the recorded finite candidate families.",
            "No human behavior, signal control, public data or CARLA is used.",
        ],
    })
    return output_dir, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "config" / "multicar_cooperation_example.json")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "Result" / "multicar_decoder_audit")
    parser.add_argument("--trial-seeds", type=int, nargs="+",
                        default=tuple(range(101, 120)))
    parser.add_argument("--position-std", type=float, default=0.3)
    parser.add_argument("--speed-std", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--eta-m", type=float, default=4.0)
    parser.add_argument("--welfare-epsilon-m", type=float, default=5.0)
    args = parser.parse_args(argv)
    output_dir, summary = run_decoder_audit(
        args.config, args.output_root, args.trial_seeds,
        args.position_std, args.speed_std, args.alpha,
        args.eta_m, args.welfare_epsilon_m)
    print(json.dumps({"result_dir": str(output_dir), "summary": summary},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
