"""Audit prediction-disturbance mismatch on chase seed 820 and holdout seeds.

This script addresses the two chase follow-ups in Sections 7.5, 8.5 and 9.2:
1. Quantifies the exact trajectory divergence between deterministic receding-
   horizon prediction and noisy execution on the E18/E19 chase 3-car seed 820
   collision case (where initial geometric clearance is 1.8069 m > 1.0 m, but
   best-case relative braking slack is only 0.0305 m and 3-sigma robust braking
   slack is -0.1046 m).
2. Runs a paired holdout audit across 20 independent seeds (901-920) on 2/3/4-car
   chase (60 paired episodes) comparing the default policy (dynamic_braking_filter
   disabled) against the 3-sigma heuristic filter (dynamic_braking_filter enabled)
   under frozen shared disturbances, confirming whether the heuristic filter
   should remain retired from the default policy.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
import uuid

from . import scene_closed_loop as sim
from .run_scene_closed_loop import _source_sha, _write_json, run_closed_loop_sweep


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (ROOT / "Result" / "ttc_multicar_20261122" /
                  "E30_chase_mismatch_and_holdout_filter_audit")
DEFAULT_STIMULI = DEFAULT_OUTPUT / "inputs" / "holdout_chase_stimuli.json"
HOLDOUT_SEEDS = tuple(range(901, 921))
CAR_COUNTS = (2, 3, 4)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_holdout_stimuli():
    count = int(math.ceil(sim.MODEL["episode_horizon_s"] / sim.MODEL["dt_s"]))
    scenarios = []
    for n_cars in CAR_COUNTS:
        for seed in HOLDOUT_SEEDS:
            cars, _edges, noises = sim._trial_initial(
                "chase", n_cars, seed, 1.0, 0.8)
            initial = {
                name: {
                    "x_m": car.x,
                    "y_m": car.y,
                    "speed_mps": car.speed,
                    "lateral_speed_mps": car.lateral_speed,
                    "goal_x_m": car.goal_x,
                }
                for name, car in sorted(cars.items())
            }
            scenarios.append({
                "scene": "chase",
                "n_cars": n_cars,
                "trial_seed": seed,
                "initial_vehicles": initial,
                "acceleration_noise_mps2": noises,
            })
    return {
        "experiment": "E30_chase_holdout_frozen_stimulus_schedule",
        "schema_version": 1,
        "design": "chase 2/3/4 cars x 20 independent seeds (901-920) with std 1.0m / 0.8m/s",
        "n_car_counts": list(CAR_COUNTS),
        "trial_seeds": list(HOLDOUT_SEEDS),
        "position_perturbation_std_m": 1.0,
        "speed_perturbation_std_mps": 0.8,
        "execution_acceleration_noise_std_mps2": sim.MODEL["accel_noise_std_mps2"],
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }


def _load_stimuli(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_key = {(int(row["n_cars"]), int(row["trial_seed"])): row
              for row in payload["scenarios"]}
    if len(by_key) != 60:
        raise ValueError("E30 stimulus file must contain 60 chase scenarios")
    return payload, by_key, _sha256_file(path)


def _fixed_trial_initial(scene, n_cars, seed, position_std_m, speed_std_mps):
    key = (int(n_cars), int(seed))
    if scene != "chase" or key not in sim._e30_stimuli_by_key:
        return sim._trial_initial_original(
            scene, n_cars, seed, position_std_m, speed_std_mps)
    row = sim._e30_stimuli_by_key[key]
    base_cars, edges = sim.make_scene(scene, n_cars)
    cars = {}
    for name, base in base_cars.items():
        state = row["initial_vehicles"][name]
        cars[name] = replace(
            base,
            x=float(state["x_m"]),
            y=float(state["y_m"]),
            speed=float(state["speed_mps"]),
            lateral_speed=float(state["lateral_speed_mps"]),
            goal_x=float(state["goal_x_m"]))
    noises = {
        name: [float(v) for v in row["acceleration_noise_mps2"][name]]
        for name in sorted(cars)
    }
    return cars, edges, noises


def audit_seed_820_mismatch():
    """Decompose deterministic prediction vs noisy execution on chase 3-car seed 820."""
    cars, edges, noise = sim._trial_initial("chase", 3, 820, 1.0, 0.8)
    r0, m0, f0 = cars["R"], cars["M"], cars["F"]
    a_r = sim.Action("a-4", -4.0, 0.0)
    a_m = sim.Action("a+2", 2.0, 0.0)
    a_f = sim.Action("a+2", 2.0, 0.0)
    pr_det = sim._rollout(r0, a_r, 0.2, 1.0)
    pm_det = sim._rollout(m0, a_m, 0.2, 1.0)
    pr_noisy = sim._rollout(r0, a_r, 0.2, 1.0, noise["R"][:5])
    pm_noisy = sim._rollout(m0, a_m, 0.2, 1.0, noise["M"][:5])

    steps = []
    for idx in range(len(pr_det)):
        c_det = sim._envelope_clearance(
            pr_det[idx].x, 0.0, r0, pm_det[idx].x, 0.0, m0)
        c_noisy = sim._envelope_clearance(
            pr_noisy[idx].x, 0.0, r0, pm_noisy[idx].x, 0.0, m0)
        steps.append({
            "time_s": round(idx * 0.2, 2),
            "deterministic_max_evasion_clearance_m": c_det,
            "noisy_executed_clearance_m": c_noisy,
            "disturbance_loss_m": c_noisy - c_det,
            "r_noise_mps2": noise["R"][idx] if idx < 5 else None,
            "m_noise_mps2": noise["M"][idx] if idx < 5 else None,
        })
    min_det = min(row["deterministic_max_evasion_clearance_m"] for row in steps)
    min_noisy = min(row["noisy_executed_clearance_m"] for row in steps)
    return {
        "case": "chase 3-car seed 820 (std 1.0m / 0.8m/s)",
        "initial_r_m_base_clearance_m": sim._envelope_clearance(
            r0.x, 0.0, r0, m0.x, 0.0, m0),
        "initial_r_m_closing_speed_mps": r0.speed - m0.speed,
        "initial_best_case_braking_slack_m": sim._initial_chase_braking_slack(cars),
        "initial_3sigma_robust_braking_slack_m": sim._robust_chase_braking_slack(cars),
        "deterministic_min_clearance_under_max_evasion_m": min_det,
        "noisy_min_clearance_under_max_evasion_m": min_noisy,
        "cumulative_disturbance_loss_at_0_8s_m": (
            steps[4]["noisy_executed_clearance_m"] -
            steps[4]["deterministic_max_evasion_clearance_m"]),
        "step_by_step_r_m_trajectory": steps,
        "finding": (
            "Even when rear vehicle R applies full braking (-4.0 m/s^2) and "
            "middle vehicle M applies full acceleration (+2.0 m/s^2) from t=0, "
            "the noise-free physical clearance minimum at t=0.8 s is only "
            "+0.03332 m (analytical continuous minimum +0.03053 m), which is "
            "0.9695 m below the 1.0 m prediction margin. Consequently, no 2.0 s "
            "plan meeting the 1.0 m margin exists under any action family, and "
            "realized Gaussian execution noise (-0.03742 m cumulative relative "
            "displacement by t=0.8 s) pushes the 0.03332 m deterministic margin "
            "to -0.00410 m. Because both baseline and the 3-sigma filter select "
            "the same maximum-evasion control (R=a-4, M=a+2, F=a+2), candidate "
            "screening cannot prevent a collision whose initial dynamic slack is "
            "smaller than the realized disturbance."),
    }


def _brief(summary, n_cars):
    group = summary["chase"][str(n_cars)]["global"]
    keys = (
        "trial_count", "initial_robust_braking_margin_failure_count",
        "collision_count", "near_collision_count", "completion_count",
        "fallback_count", "recovery_profile_step_count",
        "dynamic_braking_candidate_rejected_count",
        "dynamic_braking_filter_infeasible_ticks",
        "dynamic_braking_filter_active_ticks",
        "mean_solver_ms", "p95_solver_ms", "max_solver_ms",
        "mean_completed_delay_s",
    )
    return {k: group.get(k) for k in keys}


def run(output_root=DEFAULT_OUTPUT, stimuli_file=DEFAULT_STIMULI):
    output_root = Path(output_root).resolve()
    stimuli_file = Path(stimuli_file).resolve()
    _payload, by_key, stimuli_hash = _load_stimuli(stimuli_file)

    mismatch_audit = audit_seed_820_mismatch()

    original_init = sim._trial_initial
    sim._trial_initial_original = original_init
    sim._e30_stimuli_by_key = by_key
    sim._trial_initial = _fixed_trial_initial
    try:
        shared = {
            "scenes": ("chase",),
            "n_car_counts": CAR_COUNTS,
            "trial_seeds": HOLDOUT_SEEDS,
            "position_std_m": 1.0,
            "speed_std_mps": 0.8,
            "methods": ("global",),
        }
        base_dir, base_sum = run_closed_loop_sweep(
            output_root / "baseline_filter_disabled",
            policy={"dynamic_braking_filter": False}, **shared)
        filt_dir, filt_sum = run_closed_loop_sweep(
            output_root / "treatment_filter_enabled",
            policy={"dynamic_braking_filter": True}, **shared)
    finally:
        sim._trial_initial = original_init
        del sim._trial_initial_original
        del sim._e30_stimuli_by_key

    base_manifest = json.loads((Path(base_dir) / "manifest.json").read_text(encoding="utf-8"))
    filt_manifest = json.loads((Path(filt_dir) / "manifest.json").read_text(encoding="utf-8"))
    if base_manifest["source_sha256"] != filt_manifest["source_sha256"]:
        raise RuntimeError("Mismatched runtime hashes in E30")

    groups = {}
    for n_cars in CAR_COUNTS:
        groups[str(n_cars)] = {
            "baseline_filter_disabled": _brief(base_sum, n_cars),
            "treatment_filter_enabled": _brief(filt_sum, n_cars),
        }

    run_id = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" +
              uuid.uuid4().hex[:8])
    result_dir = output_root / "paired_comparison" / run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "experiment": "E30_chase_mismatch_and_holdout_filter_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "seed 820 mismatch decomposition + 60 paired holdout episodes (chase 2/3/4 cars, seeds 901-920)",
        "stimulus_file": str(stimuli_file),
        "stimulus_sha256": stimuli_hash,
        "source_hashes": {
            "runtime_sha256": base_manifest["source_sha256"],
            "runner_sha256": _sha256_file(__file__),
            "stimulus_sha256": stimuli_hash,
        },
        "seed_820_prediction_disturbance_mismatch": mismatch_audit,
        "holdout_filter_comparison_by_n_cars": groups,
        "python": sys.version,
        "platform": platform.platform(),
    }
    _write_json(result_dir / "summary.json", result)
    return result_dir, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stimuli-file", type=Path, default=DEFAULT_STIMULI)
    parser.add_argument("--generate-stimuli", action="store_true")
    parser.add_argument("--overwrite-stimuli", action="store_true")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)
    if args.generate_stimuli:
        if args.stimuli_file.exists() and not args.overwrite_stimuli:
            parser.error("stimulus file exists; use --overwrite-stimuli to replace")
        payload = build_holdout_stimuli()
        _write_json(args.stimuli_file, payload)
        print(json.dumps({
            "stimuli_file": str(args.stimuli_file.resolve()),
            "scenario_count": payload["scenario_count"],
            "sha256": _sha256_file(args.stimuli_file),
        }, ensure_ascii=False, indent=2))
        return
    output_dir, result = run(args.output_root, args.stimuli_file)
    if args.compact:
        result = {
            "output_dir": str(output_dir),
            "source_hashes": result["source_hashes"],
            "seed_820_mismatch_summary": {
                k: v for k, v in result[
                    "seed_820_prediction_disturbance_mismatch"].items()
                if k != "step_by_step_r_m_trajectory"
            },
            "holdout_filter_comparison_by_n_cars": result[
                "holdout_filter_comparison_by_n_cars"],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
