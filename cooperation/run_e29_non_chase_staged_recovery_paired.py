"""Run paired closed-loop tests for non-chase scenes from a frozen shared input schedule.

Extending the E23/E24 frozen-disturbance protocol to crossing, merge and head_on:
for each scene, 20 predeclared perturbed initial states around the E27 boundary
counterexample and 80 per-vehicle execution acceleration-noise samples are
serialized once to a shared JSON file and replayed identically across both arms
(staged_recovery_disabled vs staged_recovery_enabled) and across local/server hosts.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
import uuid

from . import scene_closed_loop
from .run_e27_non_chase_staged_counterexamples import build_counterexample_states
from .run_scene_closed_loop import (
    _source_sha, _write_json, run_closed_loop_sweep,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (ROOT / "Result" / "ttc_multicar_20261122" /
                  "E29_non_chase_staged_recovery_paired")
DEFAULT_STIMULI = DEFAULT_OUTPUT / "inputs" / "paired_stimuli.json"
SCENES = ("crossing", "merge", "head_on")
SEEDS_BY_SCENE = {
    "crossing": tuple(range(2901, 2921)),
    "merge": tuple(range(2921, 2941)),
    "head_on": tuple(range(2941, 2961)),
}
POSITION_STD_M = 0.005
SPEED_STD_MPS = 0.005
METHODS = ("global",)


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_stimuli():
    base_states = build_counterexample_states()
    count = int(math.ceil(scene_closed_loop.MODEL["episode_horizon_s"] /
                          scene_closed_loop.MODEL["dt_s"]))
    scenarios = []
    for scene in SCENES:
        base_cars, _edges = base_states[scene]
        names = tuple(sorted(base_cars))
        for idx, trial_seed in enumerate(SEEDS_BY_SCENE[scene]):
            noise_seed = 950000 + trial_seed
            rng = random.Random(noise_seed)
            initial = {}
            offsets = {}
            for name in names:
                car = base_cars[name]
                d_pos = rng.gauss(0.0, POSITION_STD_M)
                d_spd = rng.gauss(0.0, SPEED_STD_MPS)
                offsets[name] = {"d_longitudinal_m": d_pos, "d_speed_mps": d_spd}
                x_val = (car.x + car.direction * d_pos
                         if car.axis == "x" else car.x)
                y_val = (car.y + car.direction * d_pos
                         if car.axis == "y" else car.y)
                spd_val = max(0.5, min(car.max_speed, car.speed + d_spd))
                initial[name] = {
                    "x_m": x_val,
                    "y_m": y_val,
                    "speed_mps": spd_val,
                    "lateral_speed_mps": car.lateral_speed,
                    "goal_x_m": car.goal_x,
                }
            noises = {
                name: [rng.gauss(0.0,
                                 scene_closed_loop.MODEL["accel_noise_std_mps2"])
                       for _ in range(count)]
                for name in names
            }
            scenarios.append({
                "scene": scene,
                "n_cars": 3,
                "trial_seed": trial_seed,
                "state_id": "{}_paired_{:02d}".format(scene, idx),
                "noise_seed_used_for_generation_only": noise_seed,
                "offsets": offsets,
                "initial_vehicles": initial,
                "acceleration_noise_mps2": noises,
            })
    return {
        "experiment": "E29_non_chase_frozen_paired_stimulus_schedule",
        "schema_version": 1,
        "design": "3 non-chase scenes x 20 targeted boundary perturbations around E27 counterexamples",
        "scenes": list(SCENES),
        "position_perturbation_std_m": POSITION_STD_M,
        "speed_perturbation_std_mps": SPEED_STD_MPS,
        "execution_acceleration_noise_std_mps2": (
            scene_closed_loop.MODEL["accel_noise_std_mps2"]),
        "acceleration_noise_samples_per_vehicle": count,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }


def _load_stimuli(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("scenario_count") != 60 or len(payload.get("scenarios", [])) != 60:
        raise ValueError("E29 fixed stimulus schedule must contain 60 states")
    by_key = {(row["scene"], int(row["trial_seed"])): row
              for row in payload["scenarios"]}
    if len(by_key) != 60:
        raise ValueError("E29 stimulus keys must be unique")
    return payload, by_key, _sha256_file(path)


def _fixed_trial_initial(scene, n_cars, seed, position_std_m, speed_std_mps):
    key = (scene, int(seed))
    if n_cars != 3 or key not in scene_closed_loop._e29_stimuli_by_key:
        return scene_closed_loop._trial_initial_original(
            scene, n_cars, seed, position_std_m, speed_std_mps)
    row = scene_closed_loop._e29_stimuli_by_key[key]
    base_cars, edges = scene_closed_loop.make_scene(scene, n_cars)
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
        name: [float(value) for value in row["acceleration_noise_mps2"][name]]
        for name in sorted(cars)
    }
    return cars, edges, noises


def _run_scene_arm(output_root, scene, seeds, staged_enabled):
    original = scene_closed_loop._recovery_actions
    try:
        if not staged_enabled:
            scene_closed_loop._recovery_actions = (
                lambda sc, car, base_actions, other_cars=(): tuple(
                    action for action in base_actions
                    if not action.accel_schedule))
        run_dir, summary = run_closed_loop_sweep(
            output_root=output_root,
            scenes=(scene,),
            n_car_counts=(3,),
            trial_seeds=tuple(seeds),
            position_std_m=0.0,
            speed_std_mps=0.0,
            methods=METHODS,
            policy={"dynamic_braking_filter": False})
    finally:
        scene_closed_loop._recovery_actions = original
    return run_dir, summary


def _episode_map(run_dir):
    with (Path(run_dir) / "episodes.json").open("r", encoding="utf-8") as handle:
        return {int(row["trial_seed"]): row for row in json.load(handle)}


def _aggregate(episodes):
    count = len(episodes)
    completed_delays = [
        float(row["mean_completed_delay_s"]) for row in episodes
        if row["mean_completed_delay_s"] is not None
    ]
    return {
        "trial_count": count,
        "collision_count": sum(int(row["collision"]) for row in episodes),
        "near_collision_count": sum(int(row["near_collision"]) for row in episodes),
        "completion_count": sum(int(row["completed"]) for row in episodes),
        "fallback_count": sum(int(row["fallback_count"]) for row in episodes),
        "profile_step_count": sum(
            int(row["recovery_profile_step_count"]) for row in episodes),
        "recovery_candidate_count": sum(
            int(row["recovery_candidate_count"]) for row in episodes),
        "uncertified_fallback_ticks": sum(
            int(row["next_step_uncertified_fallback_ticks"]) for row in episodes),
        "collision_uncertified_fallback_ticks": sum(
            int(row["next_step_collision_uncertified_fallback_ticks"])
            for row in episodes),
        "staged_profile_trial_count": sum(
            int(row["recovery_profile_step_count"] > 0) for row in episodes),
        "mean_completed_delay_s": (
            sum(completed_delays) / len(completed_delays)
            if completed_delays else None),
        "mean_solver_ms_per_episode": (
            sum(float(row["mean_solver_ms"]) for row in episodes) /
            max(1, count)),
        "max_solver_ms": max(float(row["max_solver_ms"]) for row in episodes),
        "mean_episode_min_clearance_m": (
            sum(float(row["min_clearance_m"]) for row in episodes) /
            max(1, count)),
        "collision_rate": sum(int(row["collision"]) for row in episodes) /
                          float(max(1, count)),
        "completion_rate": sum(int(row["completed"]) for row in episodes) /
                           float(max(1, count)),
    }


def _compress_episode_file(run_dir):
    source = Path(run_dir) / "episodes.json"
    target = Path(run_dir) / "episodes.json.gz"
    uncompressed_bytes = source.stat().st_size
    with source.open("rb") as fin, gzip.open(str(target), "wb", compresslevel=6) as fout:
        while True:
            block = fin.read(1024 * 1024)
            if not block:
                break
            fout.write(block)
    digest = _sha256_file(target)
    source.unlink()
    manifest_path = Path(run_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["episode_record"] = {
        "path": "episodes.json.gz",
        "compression": "gzip",
        "sha256": digest,
        "uncompressed_bytes": uncompressed_bytes,
    }
    _write_json(manifest_path, manifest)
    return digest


def run(output_root=DEFAULT_OUTPUT, stimuli_file=DEFAULT_STIMULI):
    output_root = Path(output_root).resolve()
    stimuli_file = Path(stimuli_file).resolve()
    stimuli, by_key, stimuli_hash = _load_stimuli(stimuli_file)

    original_init = scene_closed_loop._trial_initial
    scene_closed_loop._trial_initial_original = original_init
    scene_closed_loop._e29_stimuli_by_key = by_key
    scene_closed_loop._trial_initial = _fixed_trial_initial

    scene_results = {}
    all_pairs = []
    compressed_shas = {}
    runtime_sha = None
    try:
        for scene in SCENES:
            seeds = SEEDS_BY_SCENE[scene]
            disabled_dir, _ = _run_scene_arm(
                output_root / scene / "staged_recovery_disabled",
                scene, seeds, False)
            enabled_dir, _ = _run_scene_arm(
                output_root / scene / "staged_recovery_enabled",
                scene, seeds, True)
            disabled = _episode_map(disabled_dir)
            enabled = _episode_map(enabled_dir)
            if set(disabled) != set(enabled) or set(disabled) != set(seeds):
                raise RuntimeError("Paired arms did not retain all seeds for {}".format(scene))
            disabled_manifest = json.loads(
                (Path(disabled_dir) / "manifest.json").read_text(encoding="utf-8"))
            enabled_manifest = json.loads(
                (Path(enabled_dir) / "manifest.json").read_text(encoding="utf-8"))
            if disabled_manifest["source_sha256"] != enabled_manifest["source_sha256"]:
                raise RuntimeError("Mismatched runtime hashes in {}".format(scene))
            runtime_sha = disabled_manifest["source_sha256"]

            scene_pairs = []
            for seed in sorted(disabled):
                a, b = disabled[seed], enabled[seed]
                input_row = by_key[(scene, seed)]
                noise_hash = _sha256_bytes(json.dumps(
                    input_row["acceleration_noise_mps2"], sort_keys=True,
                    separators=(",", ":"), allow_nan=False).encode("utf-8"))
                initial_json = json.dumps(
                    input_row["initial_vehicles"], sort_keys=True,
                    separators=(",", ":"), allow_nan=False)
                pair_row = {
                    "scene": scene,
                    "scenario_trial_seed": seed,
                    "state_id": input_row["state_id"],
                    "same_initial_vehicles_between_arms": (
                        a["initial_vehicles"] == b["initial_vehicles"]),
                    "frozen_initial_state_sha256": _sha256_bytes(
                        initial_json.encode("utf-8")),
                    "frozen_execution_noise_sha256": noise_hash,
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
                    "control_delay_s": a["mean_completed_delay_s"],
                    "staged_delay_s": b["mean_completed_delay_s"],
                }
                scene_pairs.append(pair_row)
                all_pairs.append(pair_row)
            if not all(r["same_initial_vehicles_between_arms"] for r in scene_pairs):
                raise RuntimeError("Initial states differed between arms in {}".format(scene))
            dis_sha = _compress_episode_file(disabled_dir)
            ena_sha = _compress_episode_file(enabled_dir)
            compressed_shas[scene] = {
                "staged_recovery_disabled": dis_sha,
                "staged_recovery_enabled": ena_sha,
            }
            scene_results[scene] = {
                "trial_seeds": list(seeds),
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
                "paired_outcomes": scene_pairs,
            }
    finally:
        scene_closed_loop._trial_initial = original_init
        del scene_closed_loop._trial_initial_original
        del scene_closed_loop._e29_stimuli_by_key

    run_id = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" +
              uuid.uuid4().hex[:8])
    result_dir = output_root / "paired_comparison" / run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "experiment": "E29_non_chase_staged_recovery_paired_closed_loop",
        "status": "60 fixed boundary-perturbed states across crossing, merge and head_on with frozen shared disturbances",
        "scope": "three-car crossing, merge and head_on (20 paired seeds each); global method",
        "total_paired_trials": len(all_pairs),
        "training_seed": None,
        "checkpoint_path": None,
        "stimulus_file": str(stimuli_file),
        "stimulus_sha256": stimuli_hash,
        "source_hashes": {
            "runtime_sha256": runtime_sha,
            "paired_runner_sha256": _sha256_file(__file__),
            "stimulus_sha256": stimuli_hash,
        },
        "scenes": scene_results,
        "interpretation": (
            "All 60 predeclared boundary-perturbed episodes (20 per non-chase "
            "scene) use frozen initial states and execution acceleration noise "
            "shared across arms and hosts. The experiment verifies closed-loop "
            "activation and paired outcomes of scene-specific staged recovery "
            "in crossing, merge and head_on."),
        "python": sys.version,
        "platform": platform.platform(),
    }
    _write_json(result_dir / "paired_comparison.json", summary)
    _write_json(result_dir / "manifest.json", {
        "experiment": summary["experiment"],
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stimulus_file": str(stimuli_file),
        "stimulus_sha256": stimuli_hash,
        "source_hashes": summary["source_hashes"],
        "compressed_episode_sha256": compressed_shas,
        "python": sys.version,
        "platform": platform.platform(),
    })
    return result_dir, summary


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
        payload = build_stimuli()
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
            "total_paired_trials": result["total_paired_trials"],
            "stimulus_sha256": result["stimulus_sha256"],
            "source_hashes": result["source_hashes"],
            "scenes": {
                scene: {
                    arm_name: arm_data["summary"]
                    for arm_name, arm_data in scene_data["arms"].items()
                }
                for scene, scene_data in result["scenes"].items()
            },
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
