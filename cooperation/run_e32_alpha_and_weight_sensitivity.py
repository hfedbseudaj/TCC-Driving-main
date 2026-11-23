"""Sensitivity scan for discount alpha and conflict-graph edge degree weights w_e.

This script fulfills Sections 4.4 and 5.1 of the research plan:
1. Scans discount sensitivity alpha in {0.1, 0.2, 0.3, 0.5} s^-1 under paired
   closed-loop trials (crossing 2/3/4 cars, seeds 921-930 with frozen shared
   disturbances), reporting decoded share error, broken contracts, completion,
   collisions, delay and solver time.
2. Scans conflict-graph edge weight w_e degree normalization modes:
   - uniform: w_e = 1.0
   - inverse_max_degree: w_e = 1 / max(deg(i), deg(j))
   - inverse_mean_degree: w_e = 2 / (deg(i) + deg(j))
   across asymmetric multi-edge topologies (3-car crossing where S_to_N has
   degree 2 while horizontal approaches have degree 1, and 3/4-car merge and
   head_on), under the same frozen disturbances.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import uuid

from . import scene_closed_loop as sim
from .run_scene_closed_loop import _source_sha, _write_json, run_closed_loop_sweep


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (ROOT / "Result" / "ttc_multicar_20261122" /
                  "E32_alpha_and_edge_weight_sensitivity")
DEFAULT_STIMULI = DEFAULT_OUTPUT / "inputs" / "sensitivity_stimuli.json"
SEEDS = tuple(range(921, 931))
SCENES = ("crossing", "chase", "merge", "head_on")
CAR_COUNTS = (2, 3, 4)
ALPHAS = (0.1, 0.2, 0.3, 0.5)
WEIGHT_MODES = ("uniform", "inverse_max_degree", "inverse_mean_degree")


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_stimuli():
    count = int(math.ceil(sim.MODEL["episode_horizon_s"] / sim.MODEL["dt_s"]))
    scenarios = []
    for scene in SCENES:
        for n_cars in CAR_COUNTS:
            for seed in SEEDS:
                cars, _edges, noises = sim._trial_initial(
                    scene, n_cars, seed, 1.0, 0.8)
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
                    "scene": scene,
                    "n_cars": n_cars,
                    "trial_seed": seed,
                    "initial_vehicles": initial,
                    "acceleration_noise_mps2": noises,
                })
    return {
        "experiment": "E32_E33_shared_frozen_stimulus_schedule",
        "schema_version": 1,
        "design": "4 scenes x 2/3/4 cars x 10 independent seeds (921-930) with std 1.0m / 0.8m/s",
        "scenes": list(SCENES),
        "n_car_counts": list(CAR_COUNTS),
        "trial_seeds": list(SEEDS),
        "position_perturbation_std_m": 1.0,
        "speed_perturbation_std_mps": 0.8,
        "execution_acceleration_noise_std_mps2": sim.MODEL["accel_noise_std_mps2"],
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }


def _load_stimuli(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_key = {(row["scene"], int(row["n_cars"]), int(row["trial_seed"])): row
              for row in payload["scenarios"]}
    if len(by_key) != 120:
        raise ValueError("E32 stimulus file must contain 120 scenarios")
    return payload, by_key, _sha256_file(path)


def _fixed_trial_initial(scene, n_cars, seed, position_std_m, speed_std_mps):
    key = (scene, int(n_cars), int(seed))
    if key not in sim._e32_stimuli_by_key:
        return sim._trial_initial_original(
            scene, n_cars, seed, position_std_m, speed_std_mps)
    row = sim._e32_stimuli_by_key[key]
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


def _brief(summary, scene, n_cars):
    g = summary[scene][str(n_cars)]["global"]
    keys = (
        "trial_count", "completion_count", "collision_count",
        "near_collision_count", "fallback_count",
        "recovery_profile_step_count", "mean_broken_contracts_per_decision",
        "mean_decoded_share_error", "mean_selected_allocation_error",
        "mean_completed_delay_s", "mean_solver_ms", "p95_solver_ms",
        "max_solver_ms",
    )
    return {k: g.get(k) for k in keys}


def run(output_root=DEFAULT_OUTPUT, stimuli_file=DEFAULT_STIMULI):
    output_root = Path(output_root).resolve()
    stimuli_file = Path(stimuli_file).resolve()
    _payload, by_key, stimuli_hash = _load_stimuli(stimuli_file)

    original_init = sim._trial_initial
    sim._trial_initial_original = original_init
    sim._e32_stimuli_by_key = by_key
    sim._trial_initial = _fixed_trial_initial

    alpha_results = {}
    weight_results = {}
    runtime_sha = None
    try:
        # 1. Alpha scan on crossing (2/3/4 cars) under coarse and adaptive_crossing
        for decoder_mode in ("coarse", "adaptive_crossing"):
            alpha_results[decoder_mode] = {}
            for alpha in ALPHAS:
                tag = "alpha_{:.1f}".format(alpha).replace(".", "p")
                r_dir, r_sum = run_closed_loop_sweep(
                    output_root / "alpha_scan" / decoder_mode / tag,
                    scenes=("crossing",),
                    n_car_counts=CAR_COUNTS,
                    trial_seeds=SEEDS,
                    position_std_m=1.0,
                    speed_std_mps=0.8,
                    methods=("global",),
                    policy={
                        "alpha_per_s": alpha,
                        "decoder_mode": decoder_mode,
                        "joint_objective": (
                            "share_then_contract"
                            if decoder_mode == "adaptive_crossing"
                            else "contract_fidelity"),
                        "welfare_epsilon_m": (
                            2.0 if decoder_mode == "adaptive_crossing" else 0.5),
                    },
                )
                manifest = json.loads((Path(r_dir) / "manifest.json").read_text(encoding="utf-8"))
                runtime_sha = manifest["source_sha256"]
                alpha_results[decoder_mode][str(alpha)] = {
                    str(n): _brief(r_sum, "crossing", n) for n in CAR_COUNTS
                }

        # 2. Edge degree-weight w_e scan across 3/4-car topologies with asymmetric degrees
        for mode in WEIGHT_MODES:
            r_dir, r_sum = run_closed_loop_sweep(
                output_root / "edge_weight_scan" / mode,
                scenes=SCENES,
                n_car_counts=(3, 4),
                trial_seeds=SEEDS,
                position_std_m=1.0,
                speed_std_mps=0.8,
                methods=("global",),
                policy={"edge_weight_mode": mode},
            )
            weight_results[mode] = {
                scene: {
                    str(n): _brief(r_sum, scene, n) for n in (3, 4)
                }
                for scene in SCENES
            }
    finally:
        sim._trial_initial = original_init
        del sim._trial_initial_original
        del sim._e32_stimuli_by_key

    run_id = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" +
              uuid.uuid4().hex[:8])
    result_dir = output_root / "summary" / run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "experiment": "E32_alpha_and_edge_weight_sensitivity",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "alpha in {0.1, 0.2, 0.3, 0.5} s^-1 on crossing 2/3/4 cars; w_e in {uniform, inverse_max_degree, inverse_mean_degree} on 4 scenes x 3/4 cars",
        "trial_seeds": list(SEEDS),
        "stimulus_file": str(stimuli_file),
        "stimulus_sha256": stimuli_hash,
        "source_hashes": {
            "runtime_sha256": runtime_sha,
            "runner_sha256": _sha256_file(__file__),
            "stimulus_sha256": stimuli_hash,
        },
        "alpha_sensitivity": alpha_results,
        "edge_weight_degree_sensitivity": weight_results,
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
            "source_hashes": result["source_hashes"],
            "alpha_sensitivity": result["alpha_sensitivity"],
            "edge_weight_degree_sensitivity": result[
                "edge_weight_degree_sensitivity"],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
