"""Run paired E5 constructed kinematic closed-loop trials."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import uuid

from .scene_closed_loop import (
    DEFAULT_POLICY, MODEL, VALID_METHODS, normalize_policy, run_sweep,
)


ROOT = Path(__file__).resolve().parent.parent


def _write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding="utf-8")


def _source_sha():
    digest = hashlib.sha256()
    for relative in ("cooperation/physics.py", "cooperation/scene_closed_loop.py",
                     "cooperation/run_scene_closed_loop.py"):
        digest.update(relative.encode("utf-8"))
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def run_closed_loop_sweep(output_root, scenes, n_car_counts, trial_seeds,
                          position_std_m=0.3, speed_std_mps=0.2,
                          methods=None, policy=None):
    if not scenes or set(scenes) - {"chase", "merge", "head_on", "crossing"}:
        raise ValueError("scenes must be selected from chase, merge, head_on, crossing")
    if not n_car_counts or set(n_car_counts) - {2, 3, 4}:
        raise ValueError("n_car_counts must be selected from 2, 3, 4")
    if not trial_seeds or len(set(trial_seeds)) != len(trial_seeds):
        raise ValueError("Provide nonempty, unique scenario trial seeds")
    if position_std_m < 0 or speed_std_mps < 0:
        raise ValueError("Perturbation standard deviations must be nonnegative")
    methods = tuple(methods or VALID_METHODS[:2])
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(VALID_METHODS):
        raise ValueError("Methods must be a nonempty unique subset of VALID_METHODS")
    policy = normalize_policy(policy)
    output_root = Path(output_root).resolve()
    if ROOT.as_posix().startswith("/data/hhr/"):
        designated = Path("/data/hhr").resolve()
        if designated != output_root and designated not in output_root.parents:
            raise ValueError("Server output must stay under /data/hhr")
    episodes, summary = run_sweep(
        scenes, n_car_counts, trial_seeds, methods, position_std_m,
        speed_std_mps, policy)
    effective_groups = [(scene, n_cars) for scene in scenes for n_cars in n_car_counts]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "model_snapshot.json", MODEL)
    _write_json(output_dir / "policy_snapshot.json", policy)
    _write_json(output_dir / "episodes.json", episodes)
    _write_json(output_dir / "summary.json", summary)
    manifest = {
        "experiment": "paired_scene_specific_kinematic_closed_loop",
        "status": "constructed_kinematic_example_not_CARLA",
        "scenes": list(scenes),
        "vehicle_counts": list(n_car_counts),
        "effective_scene_vehicle_groups": [
            {"scene": scene, "vehicle_count": count}
            for scene, count in effective_groups],
        "trial_seeds": list(trial_seeds),
        "trial_count_per_scene_vehicle_count_method": len(trial_seeds),
        "methods": list(methods),
        "policy": policy,
        "paired_initial_conditions_and_disturbances": True,
        "training_seed": None,
        "checkpoint_path": None,
        "position_perturbation_std_m": position_std_m,
        "speed_perturbation_std_mps": speed_std_mps,
        "acceleration_noise_std_mps2": MODEL["accel_noise_std_mps2"],
        "control_period_s": MODEL["dt_s"],
        "planning_horizon_s": MODEL["horizon_s"],
        "episode_horizon_s": MODEL["episode_horizon_s"],
        "vehicle_envelope": {
            "length_m": MODEL["vehicle_length_m"],
            "width_m": MODEL["vehicle_width_m"],
            "longitudinal_margin_m": MODEL["longitudinal_margin_m"],
            "lateral_margin_m": MODEL["lateral_margin_m"],
            "prediction_extra_margin_m": MODEL["robust_prediction_margin_m"],
        },
        "comparison": "All methods use the same finite action family, predicted vehicle-envelope safety filter and fleet-wide disagreement/individual-rationality rule. Global, sequential, welfare-only and clearance-first also share the welfare floor; safe_disagreement is a static conservative reference outside that floor.",
        "initial_braking_slack_diagnostic": "For same-lane closing chase pairs, base envelope clearance minus relative stopping distance assuming maximum rear braking and maximum lead acceleration; diagnostic only, not a formal feasibility or safety certificate.",
        "dynamic_braking_filter": {
            "enabled": policy["dynamic_braking_filter"],
            "rule": "For chase candidates, require sampled predicted robust braking slack to retain the 0.25 m buffer when the current state meets it; otherwise reject candidates that further reduce the current slack by more than 0.05 m. If the filter leaves no candidate in fallback, retain the best available action and mark infeasibility.",
            "noise_model": "Relative acceleration disturbance is tightened by 3*sqrt(2)*sigma using independent Gaussian per-vehicle acceleration noise. This is a heuristic screen, not a hard guarantee.",
        },
        "source_sha256": _source_sha(),
        "python": sys.version,
        "platform": platform.platform(),
        "limitations": [
            "All cases use newly constructed deterministic kinematic scenes, not author trajectories, a public dataset or CARLA.",
            "Safety uses inflated axis-aligned vehicle rectangles and piecewise-linear 2-second rollouts; swept clearance is evaluated exactly for each linear segment, but this is not a formal safety certificate.",
            "Longitudinal and lateral acceleration models are simplified; no steering-angle, tire, road-friction or actuator-delay model is used.",
            "TTC-calibrated pair contracts use the local constructed utility and disagreement baselines defined in scene_closed_loop.py; they are not the archived learned four-dimensional offers.",
            "The sequential baseline is a transparent reconstruction on the new contract interface, not the original paper's unavailable multi-car script.",
            "Individual rationality compares predicted 2-second utilities to one conservative safe joint candidate; it is not a human acceptance model or an infinite-horizon guarantee.",
            "If the finite action family has no safe joint candidate, the emergency overlap-minimizing action is explicitly marked uncertified.",
            "A non-urgent or non-closing edge uses equal shares; TTC activation is controlled by policy.ttc_start_s. The default TTC is first nominal predicted envelope conflict, while the original radial TTC remains selectable as an ablation.",
            "Crossing has 2/3/4-car groups with at least one vehicle on each axis; all active vehicle envelopes enter the common joint safety filter.",
            "The adaptive_crossing decoder searches a half-m/s^2 pair grid, then admits only each edge's selected reference actions into the joint family. It does not optimize the continuous action space.",
            "Optional contract_then_share and share_then_contract objectives compare normalized edge gain-allocation errors inside the same finite safety, individual-rationality and welfare family. They are experimental; a larger welfare epsilon permits lower-progress plans and has no validated driving benefit by itself.",
            "When no horizon-safe constant-action joint candidate exists in chase, a staged-braking family is searched only if the current state meets the full prediction margin. If the current state already violates that margin, or no staged plan is feasible, the fallback checks full-margin next-step candidates, then collision-free physical-envelope candidates, then the least-overlap candidate. Each source and tier is logged; model disturbance still prevents a formal execution guarantee.",
            "Executed swept-clearance checks analytically use the pre-step and post-step center positions for each piecewise-linear segment, avoiding coarse temporal sampling; the result still depends on the axis-aligned envelope and kinematic model.",
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "Result" / "multicar_scene_closed_loop")
    parser.add_argument("--scenes", nargs="+",
                        choices=("chase", "merge", "head_on", "crossing"),
                        default=("chase", "merge", "head_on", "crossing"))
    parser.add_argument("--n-cars", type=int, nargs="+", choices=(2, 3, 4),
                        default=(2, 3, 4))
    parser.add_argument("--trial-seeds", type=int, nargs="+",
                        default=tuple(range(301, 311)))
    parser.add_argument("--position-std", type=float, default=0.3)
    parser.add_argument("--speed-std", type=float, default=0.2)
    parser.add_argument("--methods", nargs="+", choices=VALID_METHODS,
                        default=VALID_METHODS[:2])
    parser.add_argument("--discount-mode", choices=("ttc", "constant", "equal_share"),
                        default=DEFAULT_POLICY["discount_mode"])
    parser.add_argument("--decoder-mode", choices=("coarse", "adaptive_crossing"),
                        default=DEFAULT_POLICY["decoder_mode"])
    parser.add_argument("--joint-objective",
                        choices=("contract_fidelity", "contract_then_share",
                                 "share_then_contract"),
                        default=DEFAULT_POLICY["joint_objective"])
    parser.add_argument("--ttc-source", choices=("nominal_conflict", "legacy_radial"),
                        default=DEFAULT_POLICY["ttc_source"])
    parser.add_argument("--alpha", type=float, default=DEFAULT_POLICY["alpha_per_s"])
    parser.add_argument("--ttc-start", type=float,
                        default=DEFAULT_POLICY["ttc_start_s"])
    parser.add_argument("--constant-ttc", type=float,
                        default=DEFAULT_POLICY["constant_ttc_s"])
    parser.add_argument("--eta-m", type=float,
                        default=DEFAULT_POLICY["surplus_tolerance_m"])
    parser.add_argument("--welfare-epsilon-m", type=float,
                        default=DEFAULT_POLICY["welfare_epsilon_m"])
    parser.add_argument("--audit-ir-only", action="store_true",
                        help="Measure but do not enforce fleet-wide individual rationality")
    parser.add_argument("--dynamic-braking-filter", action="store_true",
                        help="Screen chase-scene joint actions using a sampled 3-sigma braking-slack heuristic")
    args = parser.parse_args(argv)
    policy = {
        "discount_mode": args.discount_mode,
        "decoder_mode": args.decoder_mode,
        "joint_objective": args.joint_objective,
        "ttc_source": args.ttc_source,
        "alpha_per_s": args.alpha,
        "ttc_start_s": args.ttc_start,
        "constant_ttc_s": args.constant_ttc,
        "surplus_tolerance_m": args.eta_m,
        "welfare_epsilon_m": args.welfare_epsilon_m,
        "enforce_individual_rationality": not args.audit_ir_only,
        "dynamic_braking_filter": args.dynamic_braking_filter,
    }
    output_dir, summary = run_closed_loop_sweep(
        args.output_root, args.scenes, args.n_cars, args.trial_seeds,
        args.position_std, args.speed_std, args.methods, policy)
    print(json.dumps({"result_dir": str(output_dir), "summary": summary},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

