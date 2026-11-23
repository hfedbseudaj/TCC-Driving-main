"""Run a paired broad-grid chase test from a frozen shared input schedule.

The 3^5 grid is centered on the E20 constructed state with wider +/-0.5 m
gap and +/-0.5 m/s speed offsets. Initial states and all acceleration-noise
samples are serialized once and replayed identically on local and server.
This remains a constructed kinematic stress test, not a traffic distribution.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
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
                  "E24_broad_staged_recovery_paired")
DEFAULT_STIMULI = DEFAULT_OUTPUT / "inputs" / "paired_stimuli.json"
OFFSET_LEVELS = (-0.5, 0.0, 0.5)
BASE_POSITIONS_M = {"R": 0.0, "M": 6.480418060080996,
                    "F": 13.746349185372345}
BASE_SPEEDS_MPS = {"R": 13.816695470887003,
                   "M": 11.992656303349882,
                   "F": 9.712356443982612}
SEEDS = tuple(range(1001, 1244))
METHODS = ("global",)


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding="utf-8")


def build_stimuli():
    base_gap_rm = BASE_POSITIONS_M["M"] - BASE_POSITIONS_M["R"]
    base_gap_mf = BASE_POSITIONS_M["F"] - BASE_POSITIONS_M["M"]
    count = int(math.ceil(scene_closed_loop.MODEL["episode_horizon_s"] /
                          scene_closed_loop.MODEL["dt_s"]))
    scenarios = []
    grid = itertools.product(OFFSET_LEVELS, repeat=5)
    for index, offsets in enumerate(grid):
        gap_rm, gap_mf, dv_r, dv_m, dv_f = offsets
        x_r = BASE_POSITIONS_M["R"]
        x_m = x_r + base_gap_rm + gap_rm
        x_f = x_m + base_gap_mf + gap_mf
        initial = {
            "R": {"x_m": x_r, "y_m": 0.0,
                  "speed_mps": BASE_SPEEDS_MPS["R"] + dv_r,
                  "goal_x_m": 30.0},
            "M": {"x_m": x_m, "y_m": 0.0,
                  "speed_mps": BASE_SPEEDS_MPS["M"] + dv_m,
                  "goal_x_m": 30.0},
            "F": {"x_m": x_f, "y_m": 0.0,
                  "speed_mps": BASE_SPEEDS_MPS["F"] + dv_f,
                  "goal_x_m": 30.0},
        }
        # Generate disturbances once. The stored decimal JSON values, rather
        # than another platform's Gaussian implementation, define both arms.
        noise_seed = 900001 + index
        rng = random.Random(noise_seed)
        noises = {
            name: [rng.gauss(0.0,
                            scene_closed_loop.MODEL["accel_noise_std_mps2"])
                   for _ in range(count)]
            for name in ("F", "M", "R")
        }
        trial_seed = 1001 + index
        scenarios.append({
            "trial_seed": trial_seed,
            "grid_state_id": "wide_{:03d}".format(index),
            "noise_seed_used_for_generation_only": noise_seed,
            "offsets": {
                "gap_rm_m": gap_rm, "gap_mf_m": gap_mf,
                "dv_r_mps": dv_r, "dv_m_mps": dv_m, "dv_f_mps": dv_f,
            },
            "initial_vehicles": initial,
            "acceleration_noise_mps2": noises,
        })
    return {
        "experiment": "E24_frozen_paired_stimulus_schedule",
        "schema_version": 1,
        "design": "3^5 deterministic broad local grid around E20 state",
        "offset_levels": list(OFFSET_LEVELS),
        "dimensions": [
            "rear-middle center gap offset (m)",
            "middle-front center gap offset (m)",
            "rear speed offset (m/s)",
            "middle speed offset (m/s)",
            "front speed offset (m/s)",
        ],
        "base_positions_m": BASE_POSITIONS_M,
        "base_speeds_mps": BASE_SPEEDS_MPS,
        "execution_acceleration_noise_std_mps2": (
            scene_closed_loop.MODEL["accel_noise_std_mps2"]),
        "acceleration_noise_samples_per_vehicle": count,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }


def _load_stimuli(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("scenario_count") != 243 or len(payload.get("scenarios", [])) != 243:
        raise ValueError("E24 fixed stimulus schedule must contain 243 states")
    by_seed = {int(row["trial_seed"]): row for row in payload["scenarios"]}
    if len(by_seed) != 243:
        raise ValueError("E24 stimulus seeds must be unique")
    expected = set(SEEDS)
    if set(by_seed) != expected:
        raise ValueError("E24 stimulus seeds do not match the declared 1001-1243 grid")
    expected_count = int(math.ceil(
        scene_closed_loop.MODEL["episode_horizon_s"] /
        scene_closed_loop.MODEL["dt_s"]))
    for row in payload["scenarios"]:
        if set(row["initial_vehicles"]) != {"R", "M", "F"}:
            raise ValueError("Each E24 stimulus must specify R, M and F")
        if set(row["acceleration_noise_mps2"]) != {"R", "M", "F"}:
            raise ValueError("Each E24 stimulus needs one noise sequence per car")
        if any(len(row["acceleration_noise_mps2"][name]) != expected_count
               for name in ("R", "M", "F")):
            raise ValueError("E24 noise sequence length does not match episode horizon")
    return payload, by_seed, _sha256_file(path)


def _fixed_trial_initial(scene, n_cars, seed, position_std_m, speed_std_mps):
    if scene != "chase" or n_cars != 3:
        return scene_closed_loop._trial_initial_original(
            scene, n_cars, seed, position_std_m, speed_std_mps)
    row = scene_closed_loop._e24_stimuli_by_seed[int(seed)]
    base_cars, edges = scene_closed_loop.make_scene(scene, n_cars)
    cars = {}
    for name, base in base_cars.items():
        state = row["initial_vehicles"][name]
        cars[name] = replace(
            base, x=float(state["x_m"]), y=float(state["y_m"]),
            speed=float(state["speed_mps"]), goal_x=float(state["goal_x_m"]))
    noises = {
        name: [float(value) for value in row["acceleration_noise_mps2"][name]]
        for name in ("F", "M", "R")
    }
    return cars, edges, noises


def _episode_map(run_dir):
    with (Path(run_dir) / "episodes.json").open("r", encoding="utf-8") as handle:
        return {int(row["trial_seed"]): row for row in json.load(handle)}


def _run_arm(output_root, seeds, staged_enabled):
    original = scene_closed_loop._recovery_actions
    try:
        if not staged_enabled:
            scene_closed_loop._recovery_actions = (
                lambda scene, car, base_actions, other_cars=(): tuple(
                    action for action in base_actions
                    if not action.accel_schedule))
        run_dir, summary = run_closed_loop_sweep(
            output_root=output_root, scenes=("chase",), n_car_counts=(3,),
            trial_seeds=tuple(seeds), position_std_m=0.0, speed_std_mps=0.0,
            methods=METHODS, policy={"dynamic_braking_filter": False})
    finally:
        scene_closed_loop._recovery_actions = original
    _write_json(Path(run_dir) / "experiment_arm_override.json", {
        "staged_recovery_action_family_enabled": staged_enabled,
        "control_arm_override": (
            "ordinary base actions only; scheduled multi-step braking removed"
            if not staged_enabled else "none"),
        "initial_conditions_and_execution_noise": "frozen external stimulus file",
    })
    return run_dir, summary


def _aggregate(episodes):
    count = len(episodes)
    result = {
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
            sum(float(row["mean_completed_delay_s"]) for row in episodes
                if row["mean_completed_delay_s"] is not None) /
            max(1, sum(int(row["mean_completed_delay_s"] is not None)
                       for row in episodes))),
        "mean_solver_ms_per_episode": (
            sum(float(row["mean_solver_ms"]) for row in episodes) /
            max(1, count)),
        "mean_episode_min_clearance_m": (
            sum(float(row["min_clearance_m"]) for row in episodes) /
            max(1, count)),
        "min_initial_robust_braking_slack_m": min(
            float(row["initial_robust_braking_slack_m"])
            for row in episodes),
        "initial_margin_failure_count": sum(
            int(not row["initial_dynamic_braking_margin_satisfied"])
            for row in episodes),
        "collision_rate": sum(int(row["collision"]) for row in episodes) /
                          float(max(1, count)),
        "completion_rate": sum(int(row["completed"]) for row in episodes) /
                           float(max(1, count)),
    }
    return result


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


def run(output_root=DEFAULT_OUTPUT, stimuli_file=DEFAULT_STIMULI,
        trial_seeds=SEEDS):
    output_root = Path(output_root).resolve()
    stimuli_file = Path(stimuli_file).resolve()
    stimuli, by_seed, stimuli_hash = _load_stimuli(stimuli_file)
    seeds = tuple(int(seed) for seed in trial_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide nonempty unique E24 trial seeds")
    if set(seeds) - set(by_seed):
        raise ValueError("Requested trial seed absent from frozen stimulus file")

    original_init = scene_closed_loop._trial_initial
    scene_closed_loop._trial_initial_original = original_init
    scene_closed_loop._e24_stimuli_by_seed = by_seed
    scene_closed_loop._trial_initial = _fixed_trial_initial
    try:
        disabled_dir, disabled_summary = _run_arm(
            output_root / "staged_recovery_disabled", seeds, False)
        enabled_dir, enabled_summary = _run_arm(
            output_root / "staged_recovery_enabled", seeds, True)
    finally:
        scene_closed_loop._trial_initial = original_init
        del scene_closed_loop._trial_initial_original
        del scene_closed_loop._e24_stimuli_by_seed

    disabled = _episode_map(disabled_dir)
    enabled = _episode_map(enabled_dir)
    if set(disabled) != set(enabled) or set(disabled) != set(seeds):
        raise RuntimeError("Paired arms did not retain every requested grid state")
    pairs = []
    for seed in sorted(disabled):
        a, b = disabled[seed], enabled[seed]
        input_row = by_seed[seed]
        noise_hash = _sha256_bytes(json.dumps(
            input_row["acceleration_noise_mps2"], sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode("utf-8"))
        initial_json = json.dumps(input_row["initial_vehicles"], sort_keys=True,
                                  separators=(",", ":"), allow_nan=False)
        pairs.append({
            "scenario_trial_seed": seed,
            "grid_state_id": input_row["grid_state_id"],
            "same_initial_vehicles_between_arms": (
                a["initial_vehicles"] == b["initial_vehicles"]),
            "frozen_initial_state_sha256": _sha256_bytes(
                initial_json.encode("utf-8")),
            "frozen_execution_noise_sha256": noise_hash,
            "offsets": input_row["offsets"],
            "initial_robust_braking_slack_m":
                a["initial_robust_braking_slack_m"],
            "initial_dynamic_margin_satisfied":
                a["initial_dynamic_braking_margin_satisfied"],
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
        })
    if not all(row["same_initial_vehicles_between_arms"] for row in pairs):
        raise RuntimeError("Paired E24 arms used different initial states")

    disabled_manifest = json.loads(
        (Path(disabled_dir) / "manifest.json").read_text(encoding="utf-8"))
    enabled_manifest = json.loads(
        (Path(enabled_dir) / "manifest.json").read_text(encoding="utf-8"))
    if disabled_manifest["source_sha256"] != enabled_manifest["source_sha256"]:
        raise RuntimeError("E24 paired arms used different runtime hashes")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    result_dir = output_root / "paired_comparison" / run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "experiment": "E24_broad_staged_recovery_paired_closed_loop",
        "status": "243 fixed broad-grid states with frozen shared disturbances",
        "scope": "three-car same-lane chase; global method; paired arms",
        "scenario_trial_seeds": list(seeds),
        "trial_count": len(seeds),
        "training_seed": None,
        "checkpoint_path": None,
        "stimulus_file": str(stimuli_file),
        "stimulus_sha256": stimuli_hash,
        "grid_definition": {
            "state_count_full_grid": 243,
            "offset_levels": list(OFFSET_LEVELS),
            "dimensions": stimuli["dimensions"],
            "base_positions_m": BASE_POSITIONS_M,
            "base_speeds_mps": BASE_SPEEDS_MPS,
            "execution_acceleration_noise_std_mps2": (
                stimuli["execution_acceleration_noise_std_mps2"]),
            "failure_states_retained": True,
        },
        "source_hashes": {
            "runtime_sha256": disabled_manifest["source_sha256"],
            "paired_runner_sha256": _sha256_file(__file__),
            "stimulus_sha256": stimuli_hash,
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
            "All 243 predeclared states, including low-margin and failed "
            "cases, are retained. Frozen initial states and acceleration "
            "noise are shared across arms and hosts by the input file. This "
            "is a broad local constructed grid, not a general traffic sample, "
            "formal safety proof, CARLA result, or real-vehicle evidence."),
        "python": sys.version,
        "platform": platform.platform(),
    }
    _write_json(result_dir / "paired_comparison.json", summary)
    disabled_episode_sha = _compress_episode_file(disabled_dir)
    enabled_episode_sha = _compress_episode_file(enabled_dir)
    _write_json(result_dir / "manifest.json", {
        "experiment": summary["experiment"],
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_trial_seeds": list(seeds),
        "stimulus_file": str(stimuli_file),
        "stimulus_sha256": stimuli_hash,
        "source_hashes": summary["source_hashes"],
        "compressed_episode_sha256": {
            "staged_recovery_disabled": disabled_episode_sha,
            "staged_recovery_enabled": enabled_episode_sha,
        },
        "arm_run_dirs": summary["arms"],
        "python": sys.version,
        "platform": platform.platform(),
    })
    return result_dir, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stimuli-file", type=Path, default=DEFAULT_STIMULI)
    parser.add_argument("--trial-seeds", type=int, nargs="+", default=list(SEEDS))
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
    output_dir, result = run(
        args.output_root, args.stimuli_file, args.trial_seeds)
    if args.compact:
        result = {
            "output_dir": str(output_dir),
            "trial_count": result["trial_count"],
            "stimulus_sha256": result["stimulus_sha256"],
            "source_hashes": result["source_hashes"],
            "arms": {key: value["summary"]
                     for key, value in result["arms"].items()},
            "collision_seed_ids": [
                row["grid_state_id"] for row in result["paired_outcomes"]
                if row["control_collision"] or row["staged_collision"]],
            "near_collision_seed_ids": [
                row["grid_state_id"] for row in result["paired_outcomes"]
                if row["control_near_collision"] or row["staged_near_collision"]],
            "all_paired_initial_states_match": all(
                row["same_initial_vehicles_between_arms"]
                for row in result["paired_outcomes"]),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
