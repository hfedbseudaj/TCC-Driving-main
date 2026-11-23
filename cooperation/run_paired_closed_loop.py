"""Run paired constructed crossing E3b kinematic trials with complete logs."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import uuid

from .crossing_closed_loop import run_crossing_episode
from .run_snapshot import _vehicle_from_row, _write_json


_ROOT = Path(__file__).resolve().parent.parent


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_sha():
    paths = ("cooperation/physics.py", "cooperation/crossing.py",
             "cooperation/compose.py", "cooperation/crossing_closed_loop.py",
             "cooperation/run_paired_closed_loop.py")
    digest = hashlib.sha256()
    for relative in paths:
        digest.update(relative.encode("utf-8"))
        digest.update((_ROOT / relative).read_bytes())
    return digest.hexdigest()


def _trial_initial_conditions(config, trial_seed, position_std_m,
                              speed_std_mps, noise_accel_std_mps2):
    rng = random.Random(int(trial_seed))
    vehicles = {}
    perturbations = {}
    for row in config["vehicles"]:
        original = _vehicle_from_row(row)
        longitudinal_delta = rng.gauss(0.0, position_std_m)
        speed_delta = rng.gauss(0.0, speed_std_mps)
        new_distance = max(1.0, original.distance_to_entry_m + longitudinal_delta)
        new_speed = min(original.max_speed_mps,
                        max(0.3, original.speed_mps + speed_delta))
        # Keep the physical zone entry coordinate fixed while perturbing the
        # car longitudinally along its original lane centerline.
        entry = original.entry_position_xy
        new_start = tuple(entry[index] - new_distance * original.direction_xy[index]
                          for index in (0, 1))
        vehicles[original.id] = replace(
            original, start_xy=new_start, distance_to_entry_m=new_distance,
            speed_mps=new_speed)
        perturbations[original.id] = {
            "longitudinal_distance_delta_m": new_distance - original.distance_to_entry_m,
            "speed_delta_mps": new_speed - original.speed_mps,
        }
    steps = int(math.ceil(config["physics"]["horizon_s"] / config["physics"]["dt_s"]))
    names = sorted(vehicles)
    noise = [[rng.gauss(0.0, noise_accel_std_mps2) for _ in names]
             for _ in range(steps)]
    return vehicles, perturbations, noise


def run_paired(config_path, output_root, trial_seeds,
               position_std_m=0.0, speed_std_mps=0.0,
               noise_accel_std_mps2=0.0, allow_constructed_example=False,
               commitment_weight=0.0):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("mode") != "constructed_example":
        raise ValueError("This runner requires the explicit constructed-example schema")
    if not allow_constructed_example:
        raise ValueError("Pass --allow-constructed-example to label these synthetic trials")
    if not trial_seeds or len(set(trial_seeds)) != len(trial_seeds):
        raise ValueError("Provide nonempty, unique scenario trial seeds")
    if any(x < 0 or not math.isfinite(x)
           for x in (position_std_m, speed_std_mps, noise_accel_std_mps2,
                     commitment_weight)):
        raise ValueError("Perturbation scales must be finite and nonnegative")
    output_root = Path(output_root).resolve()
    if _ROOT.as_posix().startswith("/data/hhr/"):
        designated = Path("/data/hhr").resolve()
        if designated != output_root and designated not in output_root.parents:
            raise ValueError("Server output must stay under /data/hhr")

    edges = [tuple(row["edge"]) for row in config["pair_proposals"]]
    if len(edges) != 2 or len(config["vehicles"]) != 3:
        raise ValueError("The current E3b implementation is the three-CAV two-edge crossing case")
    methods = ("global", "sequential_pairwise_reconstruction")
    episodes = []
    for seed in trial_seeds:
        vehicles, perturbations, noise = _trial_initial_conditions(
            config, seed, position_std_m, speed_std_mps,
            noise_accel_std_mps2)
        for method in methods:
            result = run_crossing_episode(
                vehicles, edges, method, noise,
                dt_s=float(config["physics"]["dt_s"]),
                horizon_s=float(config["physics"]["horizon_s"]),
                alpha_per_s={name: float(value) for name, value in
                             config["bargaining"]["discount_alpha_s_inv"].items()},
                eta_s=float(config["bargaining"]["eta_s"]),
                clearance_s=float(config["physics"]["clearance_s"]),
                welfare_epsilon_s=float(config["bargaining"]["welfare_epsilon_s"]),
                commitment_weight=commitment_weight,
            )
            episodes.append({
                "trial_seed": int(seed), "method": method,
                "initial_perturbations": perturbations,
                "initial_vehicles": {name: asdict(vehicle)
                                     for name, vehicle in vehicles.items()},
                "result": result,
            })

    summary = {}
    for method in methods:
        group = [row["result"] for row in episodes if row["method"] == method]
        completed_delays = [row["total_entry_delay_s"] for row in group
                            if row["completed"] and row["total_entry_delay_s"] is not None]
        summary[method] = {
            "trial_count": len(group),
            "collision_count": sum(bool(row["collision"]) for row in group),
            "collision_rate": sum(bool(row["collision"]) for row in group) / len(group),
            "near_collision_count": sum(bool(row["near_collision"]) for row in group),
            "completion_count": sum(bool(row["completed"]) for row in group),
            "completion_rate": sum(bool(row["completed"]) for row in group) / len(group),
            "planner_abort_count": sum(bool(row["planner_aborted"]) for row in group),
            "fallback_count": sum(row["fallback_count"] for row in group),
            "recovery_step_count": sum(row["recovery_step_count"] for row in group),
            "raw_pair_unsafe_ticks": sum(row["raw_pair_unsafe_ticks"] for row in group),
            "mean_completed_delay_s": (
                sum(completed_delays) / len(completed_delays) if completed_delays else None),
            "mean_solver_ms": sum(row["mean_solver_ms"] for row in group) / len(group),
        }
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "config_snapshot.json", config)
    _write_json(output_dir / "episodes.json", episodes)
    _write_json(output_dir / "summary.json", summary)
    manifest = {
        "experiment": "E3b_paired_kinematic_crossing_closed_loop",
        "status": "constructed_example_not_CARLA",
        "trial_seeds": list(trial_seeds),
        "trial_count_per_method": len(trial_seeds),
        "training_seed": None,
        "checkpoint_path": None,
        "paired_initial_conditions_and_disturbance": True,
        "commitment_weight": commitment_weight,
        "methods": list(methods),
        "aggregator_definition": {
            "global": "joint lexicographic finite-candidate contract repair",
            "sequential_pairwise_reconstruction": "sort K_e by pre-decision TTC, commit each local pair proposal, overwrite shared-vehicle targets in order, then project onto the same safe welfare-qualified joint candidate family",
        },
        "filter_layers": [
            "pre_filter_TTC_weighted_raw_pair_target",
            "finite_candidate_hard_zone_occupation_filter",
            "post_execution_swept_center_distance",
        ],
        "config_sha256": _sha(config_path),
        "source_sha256": _source_sha(),
        "python": platform.python_version(),
        "limitations": [
            "Only deterministic crossing uses a complete K_e decoder.",
            "Safety filter uses conflict-zone occupation and center distance, not vehicle envelopes.",
            "When the joint planner has no candidate, a bounded one-step safety shield enumerates brake/coast/accelerate commands; it is a kinematic recovery layer, not a formal safety certificate.",
            "The learned legacy offer adapter is evaluated separately.",
            "The comparison baseline is a transparent sequential reconstruction over K_e, not the exact archived multi-car code.",
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=_ROOT / "config" / "multicar_cooperation_example.json")
    parser.add_argument("--output-root", type=Path,
                        default=_ROOT / "Result" / "multicar_cooperation_closed_loop")
    parser.add_argument("--trial-seeds", type=int, nargs="+", default=[101, 102, 103])
    parser.add_argument("--position-std", type=float, default=0.3)
    parser.add_argument("--speed-std", type=float, default=0.2)
    parser.add_argument("--accel-noise-std", type=float, default=0.1)
    parser.add_argument("--commitment-weight", type=float, default=0.0)
    parser.add_argument("--allow-constructed-example", action="store_true")
    args = parser.parse_args(argv)
    output_dir, summary = run_paired(
        args.config, args.output_root, args.trial_seeds,
        args.position_std, args.speed_std, args.accel_noise_std,
        args.allow_constructed_example, args.commitment_weight)
    print(json.dumps({"result_dir": str(output_dir), "summary": summary},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
