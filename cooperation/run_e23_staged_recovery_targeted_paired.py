"""Paired closed-loop stress test around the E20 staged-braking boundary state.

This is a targeted constructed kinematic stress test, not a population-level
random benchmark. Both arms receive identical small initial perturbations and
the same per-seed execution disturbance sequence. The control arm removes only
the multi-step scheduled braking candidates.
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

from . import scene_closed_loop
from .run_scene_closed_loop import (
    _source_sha, _write_json, run_closed_loop_sweep,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (ROOT / "Result" / "ttc_multicar_20261121" /
                  "E23_staged_recovery_targeted_paired")
SEEDS = tuple(range(861, 881))
POSITION_STD_M = 0.005
SPEED_STD_MPS = 0.005
METHODS = ("global",)
BASE_POSITIONS_M = {"R": 0.0, "M": 6.480418060080996,
                    "F": 13.746349185372345}
BASE_SPEEDS_MPS = {"R": 13.816695470887003,
                   "M": 11.992656303349882,
                   "F": 9.712356443982612}


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_pair_noise(seed, position_std_m=POSITION_STD_M,
                       speed_std_mps=SPEED_STD_MPS):
    rng = random.Random(int(seed))
    perturbations = {}
    for name in ("R", "M", "F"):
        perturbations[name] = (
            rng.gauss(0.0, position_std_m),
            rng.gauss(0.0, speed_std_mps))
    count = int(math.ceil(scene_closed_loop.MODEL["episode_horizon_s"] /
                          scene_closed_loop.MODEL["dt_s"]))
    noises = {
        name: [rng.gauss(0.0,
                         scene_closed_loop.MODEL["accel_noise_std_mps2"])
               for _ in range(count)]
        for name in ("F", "M", "R")
    }
    return perturbations, noises


def _write_pair_initial(scene, n_cars, seed, position_std_m, speed_std_mps):
    if scene != "chase" or n_cars != 3:
        return scene_closed_loop._trial_initial_original(
            scene, n_cars, seed, position_std_m, speed_std_mps)
    base_cars, edges = scene_closed_loop.make_scene(scene, n_cars)
    perturbations, noises = _sample_pair_noise(
        seed, position_std_m, speed_std_mps)
    cars = {}
    for name, base in base_cars.items():
        dx, dv = perturbations[name]
        cars[name] = replace(
            base,
            x=BASE_POSITIONS_M[name] + base.direction * dx,
            y=0.0,
            speed=max(0.5, min(base.max_speed,
                               BASE_SPEEDS_MPS[name] + dv)),
            goal_x=30.0)
    return cars, edges, noises


def _episode_map(run_dir):
    with (Path(run_dir) / "episodes.json").open("r", encoding="utf-8") as f:
        return {row["trial_seed"]: row for row in json.load(f)}


def _run_arm(output_root, seeds, staged_enabled):
    original = scene_closed_loop._recovery_actions
    try:
        if not staged_enabled:
            scene_closed_loop._recovery_actions = (
                lambda scene, car, base_actions, other_cars=(): tuple(
                    action for action in base_actions
                    if not action.accel_schedule))
        return run_closed_loop_sweep(
            output_root=output_root,
            scenes=("chase",), n_car_counts=(3,),
            trial_seeds=tuple(seeds),
            position_std_m=POSITION_STD_M,
            speed_std_mps=SPEED_STD_MPS,
            methods=METHODS,
            policy={"dynamic_braking_filter": False})
    finally:
        scene_closed_loop._recovery_actions = original


def _aggregate(episodes):
    count = len(episodes)
    sum_fields = (
        "collision", "near_collision", "completed", "fallback_count",
        "recovery_step_count", "recovery_profile_step_count",
        "recovery_candidate_count", "next_step_uncertified_fallback_ticks",
        "next_step_collision_uncertified_fallback_ticks")
    result = {key: sum(int(row[key]) for row in episodes)
              for key in sum_fields}
    solver = [float(row["mean_solver_ms"]) for row in episodes]
    delay = [float(row["mean_completed_delay_s"])
             for row in episodes if row["mean_completed_delay_s"] is not None]
    result.update({
        "trial_count": count,
        "collision_rate": result["collision"] / float(count),
        "completion_rate": result["completed"] / float(count),
        "mean_solver_ms_per_episode": (
            sum(solver) / len(solver) if solver else None),
        "mean_completed_delay_s": (sum(delay) / len(delay) if delay else None),
        "staged_profile_trial_count": sum(
            int(row["recovery_profile_step_count"] > 0) for row in episodes),
        "near_collision_trial_count": sum(
            int(row["near_collision"]) for row in episodes),
    })
    return result


def run(output_root=DEFAULT_OUTPUT, seeds=SEEDS):
    output_root = Path(output_root).resolve()
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide nonempty unique scenario trial seeds")
    if set(seeds) & (set(range(801, 841))):
        raise ValueError("E23 seeds must be separate from E19/E22 seeds")

    original_init = scene_closed_loop._trial_initial
    scene_closed_loop._trial_initial_original = original_init
    scene_closed_loop._trial_initial = _write_pair_initial
    try:
        disabled_dir, disabled_summary = _run_arm(
            output_root / "staged_recovery_disabled", seeds, False)
        enabled_dir, enabled_summary = _run_arm(
            output_root / "staged_recovery_enabled", seeds, True)
    finally:
        scene_closed_loop._trial_initial = original_init
        del scene_closed_loop._trial_initial_original

    disabled = _episode_map(disabled_dir)
    enabled = _episode_map(enabled_dir)
    if set(disabled) != set(enabled):
        raise RuntimeError("Paired arms produced different scenario seeds")
    pairs = []
    for seed in sorted(disabled):
        a, b = disabled[seed], enabled[seed]
        initial_equal = a["initial_vehicles"] == b["initial_vehicles"]
        noise = _sample_pair_noise(seed)[1]
        noise_hash = _sha256_bytes(json.dumps(
            noise, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("utf-8"))
        pairs.append({
            "scenario_trial_seed": seed,
            "same_initial_vehicles": initial_equal,
            "precomputed_execution_noise_sha256": noise_hash,
            "control_collision": a["collision"],
            "staged_collision": b["collision"],
            "control_near_collision": a["near_collision"],
            "staged_near_collision": b["near_collision"],
            "control_completed": a["completed"],
            "staged_completed": b["completed"],
            "control_fallback_count": a["fallback_count"],
            "staged_fallback_count": b["fallback_count"],
            "control_profile_steps": a["recovery_profile_step_count"],
            "staged_profile_steps": b["recovery_profile_step_count"],
            "control_min_clearance_m": a["min_clearance_m"],
            "staged_min_clearance_m": b["min_clearance_m"],
        })
    if not all(row["same_initial_vehicles"] for row in pairs):
        raise RuntimeError("Paired arms used different initial conditions")

    disabled_manifest = json.loads(
        (Path(disabled_dir) / "manifest.json").read_text(encoding="utf-8"))
    enabled_manifest = json.loads(
        (Path(enabled_dir) / "manifest.json").read_text(encoding="utf-8"))
    if disabled_manifest["source_sha256"] != enabled_manifest["source_sha256"]:
        raise RuntimeError("Paired arms used different runtime source hashes")
    run_id = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" +
              uuid.uuid4().hex[:8])
    result_dir = output_root / "paired_comparison" / run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "experiment": "E23_targeted_staged_recovery_paired_closed_loop",
        "status": "targeted constructed kinematic stress test",
        "scope": "three-car same-lane chase; global controller; paired arms",
        "scenario_trial_seeds": list(seeds),
        "trial_count": len(seeds),
        "training_seed": None,
        "checkpoint_path": None,
        "initial_state_design": {
            "base_state_source": "E20 constructed staged-braking boundary case",
            "positions_m": BASE_POSITIONS_M,
            "speeds_mps": BASE_SPEEDS_MPS,
            "position_perturbation_std_m": POSITION_STD_M,
            "speed_perturbation_std_mps": SPEED_STD_MPS,
            "goal_x_m": 30.0,
            "interpretation": "Mechanism-activation stress test; not a population sample.",
        },
        "execution_acceleration_noise_std_mps2":
            scene_closed_loop.MODEL["accel_noise_std_mps2"],
        "paired_initial_conditions_and_execution_noise": True,
        "source_hashes": {
            "runtime_sha256": disabled_manifest["source_sha256"],
            "paired_runner_sha256": _sha256_file(__file__),
        },
        "arms": {
            "staged_recovery_disabled": {
                "run_dir": str(disabled_dir),
                "summary": _aggregate(list(disabled.values())),
            },
            "staged_recovery_enabled": {
                "run_dir": str(enabled_dir),
                "summary": _aggregate(list(enabled.values())),
            },
        },
        "paired_outcomes": pairs,
        "interpretation": (
            "The initial states are deliberately concentrated around the E20 "
            "constructed candidate-family boundary. Results test whether the "
            "staged action family changes outcomes under this stress design; "
            "they do not estimate general-road safety or establish a formal "
            "safety guarantee, CARLA validation, or real-vehicle performance."),
        "python": sys.version,
        "platform": platform.platform(),
    }
    _write_json(result_dir / "paired_comparison.json", summary)
    _write_json(result_dir / "manifest.json", {
        "experiment": summary["experiment"], "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_trial_seeds": list(seeds),
        "initial_state_design": summary["initial_state_design"],
        "source_hashes": summary["source_hashes"],
        "paired_initial_conditions_and_execution_noise": True,
        "arm_run_dirs": summary["arms"],
        "python": sys.version, "platform": platform.platform(),
    })
    return result_dir, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trial-seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)
    output_dir, result = run(args.output_root, args.trial_seeds)
    if args.compact:
        result = {
            "output_dir": str(output_dir), "trial_count": result["trial_count"],
            "source_hashes": result["source_hashes"],
            "arms": {name: arm["summary"]
                     for name, arm in result["arms"].items()},
            "profile_trial_counts": {
                name: arm["summary"]["staged_profile_trial_count"]
                for name, arm in result["arms"].items()},
            "collision_pairs": [
                row["scenario_trial_seed"] for row in result["paired_outcomes"]
                if row["control_collision"] or row["staged_collision"]],
            "near_collision_pairs": [
                row["scenario_trial_seed"] for row in result["paired_outcomes"]
                if row["control_near_collision"] or row["staged_near_collision"]],
            "all_initial_states_match": all(
                row["same_initial_vehicles"]
                for row in result["paired_outcomes"]),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
