"""Run the final 4-scene x 2/3/4-car x 4-method comparison on the updated controller.

This script updates the E5-E9 multi-method baseline comparison on the final
controller (with all four scenes' staged recovery and lossless DFS pruning
enabled), using the shared frozen stimulus schedule from E32 (seeds 921-930,
position std 1.0 m, speed std 0.8 m/s).
Methods compared:
- global (TTC-calibrated contract repair)
- sequential_pairwise_reconstruction (TTC-ordered sequential overwrite + safety filter)
- welfare_only (maximum total utility on the common IR/safe candidate set)
- clearance_first (maximum separation above the common welfare floor)
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import uuid

from . import scene_closed_loop as sim
from .run_e32_alpha_and_weight_sensitivity import (
    CAR_COUNTS, DEFAULT_STIMULI, SCENES, SEEDS, _fixed_trial_initial,
    _load_stimuli,
)
from .run_scene_closed_loop import _write_json, run_closed_loop_sweep


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (ROOT / "Result" / "ttc_multicar_20261122" /
                  "E33_final_four_method_closed_loop_comparison")
METHODS = (
    "global",
    "sequential_pairwise_reconstruction",
    "welfare_only",
    "clearance_first",
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _brief(group):
    keys = (
        "trial_count", "completion_count", "collision_count",
        "near_collision_count", "fallback_count",
        "recovery_profile_step_count", "mean_broken_contracts_per_decision",
        "raw_pair_unsafe_ticks", "mean_completed_delay_s",
        "mean_solver_ms", "p95_solver_ms", "max_solver_ms",
    )
    return {k: group.get(k) for k in keys}


def run(output_root=DEFAULT_OUTPUT, stimuli_file=DEFAULT_STIMULI):
    output_root = Path(output_root).resolve()
    stimuli_file = Path(stimuli_file).resolve()
    _payload, by_key, stimuli_hash = _load_stimuli(stimuli_file)

    original_init = sim._trial_initial
    sim._trial_initial_original = original_init
    sim._e32_stimuli_by_key = by_key
    sim._trial_initial = _fixed_trial_initial
    try:
        run_dir, summary = run_closed_loop_sweep(
            output_root=output_root / "runs",
            scenes=SCENES,
            n_car_counts=CAR_COUNTS,
            trial_seeds=SEEDS,
            position_std_m=1.0,
            speed_std_mps=0.8,
            methods=METHODS,
            policy={"dynamic_braking_filter": False},
        )
    finally:
        sim._trial_initial = original_init
        del sim._trial_initial_original
        del sim._e32_stimuli_by_key

    manifest = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
    table = {}
    overall_by_method = {
        m: {
            "total_episodes": 0,
            "completion_count": 0,
            "collision_count": 0,
            "near_collision_count": 0,
            "fallback_count": 0,
            "mean_broken_contracts_per_decision": 0.0,
            "mean_completed_delay_s": 0.0,
            "mean_solver_ms": 0.0,
            "worst_p95_solver_ms": 0.0,
        }
        for m in METHODS
    }
    group_count = len(SCENES) * len(CAR_COUNTS)
    for scene in SCENES:
        table[scene] = {}
        for n_cars in CAR_COUNTS:
            table[scene][str(n_cars)] = {}
            for m in METHODS:
                g = summary[scene][str(n_cars)][m]
                table[scene][str(n_cars)][m] = _brief(g)
                agg = overall_by_method[m]
                agg["total_episodes"] += int(g["trial_count"])
                agg["completion_count"] += int(g["completion_count"])
                agg["collision_count"] += int(g["collision_count"])
                agg["near_collision_count"] += int(g["near_collision_count"])
                agg["fallback_count"] += int(g["fallback_count"])
                agg["mean_broken_contracts_per_decision"] += (
                    float(g["mean_broken_contracts_per_decision"] or 0.0) / group_count)
                agg["mean_completed_delay_s"] += (
                    float(g["mean_completed_delay_s"] or 0.0) / group_count)
                agg["mean_solver_ms"] += (
                    float(g["mean_solver_ms"]) / group_count)
                agg["worst_p95_solver_ms"] = max(
                    agg["worst_p95_solver_ms"], float(g["p95_solver_ms"]))

    result = {
        "experiment": "E33_final_four_method_closed_loop_comparison",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "4 scenes x 2/3/4 cars x 10 seeds (921-930) x 4 methods = 480 episodes",
        "run_dir": str(run_dir),
        "stimulus_file": str(stimuli_file),
        "stimulus_sha256": stimuli_hash,
        "source_hashes": {
            "runtime_sha256": manifest["source_sha256"],
            "runner_sha256": _sha256_file(__file__),
            "stimulus_sha256": stimuli_hash,
        },
        "overall_by_method": overall_by_method,
        "groups": table,
        "python": sys.version,
        "platform": platform.platform(),
    }
    _write_json(Path(run_dir) / "four_method_comparison.json", result)
    return run_dir, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stimuli-file", type=Path, default=DEFAULT_STIMULI)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)
    run_dir, result = run(args.output_root, args.stimuli_file)
    if args.compact:
        result = {
            "run_dir": str(run_dir),
            "source_hashes": result["source_hashes"],
            "overall_by_method": result["overall_by_method"],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
