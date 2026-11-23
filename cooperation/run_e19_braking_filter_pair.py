"""Run the paired E19 chase braking-filter comparison.

This is a constructed kinematic stress experiment, not a CARLA or dataset run.
Both arms use the same scenario seeds and disturbance-generation procedure.
"""

import argparse
import hashlib
import json
from pathlib import Path

from .run_scene_closed_loop import _write_json, run_closed_loop_sweep


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (ROOT / "Result" / "ttc_multicar_20261121" /
                  "E19_braking_filter_pair")
SEEDS = tuple(range(801, 821))
SUMMARY_KEYS = (
    "trial_count", "initial_robust_braking_margin_failure_count",
    "collision_count", "near_collision_count", "completion_count",
    "fallback_count", "dynamic_braking_candidate_rejected_count",
    "dynamic_braking_filter_infeasible_ticks",
    "dynamic_braking_filter_active_ticks", "mean_solver_ms",
    "p95_solver_ms", "max_solver_ms", "mean_completed_delay_s",
)


def _episode_map(path):
    with (Path(path) / "episodes.json").open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    return {row["trial_seed"]: row for row in rows}


def _brief(summary):
    group = summary["chase"]["3"]["global"]
    return {key: group.get(key) for key in SUMMARY_KEYS}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(output_root, seeds=SEEDS):
    output_root = Path(output_root).resolve()
    shared = {
        "scenes": ["chase"],
        "n_car_counts": [3],
        "trial_seeds": list(seeds),
        "position_std_m": 1.0,
        "speed_std_mps": 0.8,
        "methods": ["global"],
    }
    baseline_dir, baseline_summary = run_closed_loop_sweep(
        output_root / "baseline", policy={}, **shared)
    filter_dir, filter_summary = run_closed_loop_sweep(
        output_root / "dynamic_filter",
        policy={"dynamic_braking_filter": True}, **shared)
    baseline_episodes = _episode_map(baseline_dir)
    filter_episodes = _episode_map(filter_dir)
    if set(baseline_episodes) != set(filter_episodes):
        raise RuntimeError("Paired runs did not produce matching seed sets")
    paired = []
    for seed in sorted(baseline_episodes):
        base = baseline_episodes[seed]
        screened = filter_episodes[seed]
        paired.append({
            "trial_seed": seed,
            "same_initial_vehicles": (
                base["initial_vehicles"] == screened["initial_vehicles"]),
            "baseline_collision": base["collision"],
            "filter_collision": screened["collision"],
            "baseline_completed": base["completed"],
            "filter_completed": screened["completed"],
            "baseline_collision_time_s": base["collision_time_s"],
            "filter_collision_time_s": screened["collision_time_s"],
            "initial_robust_braking_slack_m": (
                base["initial_robust_braking_slack_m"]),
        })
    baseline_manifest = json.loads(
        (baseline_dir / "manifest.json").read_text(encoding="utf-8"))
    filter_manifest = json.loads(
        (filter_dir / "manifest.json").read_text(encoding="utf-8"))
    if baseline_manifest["source_sha256"] != filter_manifest["source_sha256"]:
        raise RuntimeError("Paired arms used different runtime source hashes")
    comparison = {
        "experiment": "E19_paired_dynamic_braking_slack_filter",
        "scope": "constructed chase scene, 3 vehicles, global controller",
        "status": "exploratory; heuristic screen; not a safety guarantee",
        "scenario_trial_seeds": list(seeds),
        "training_seed": None,
        "paired_initial_conditions_and_disturbances": True,
        "source_hashes": {
            "runtime_sha256": baseline_manifest["source_sha256"],
            "paired_runner_sha256": _sha256(__file__),
        },
        "position_perturbation_std_m": shared["position_std_m"],
        "speed_perturbation_std_mps": shared["speed_std_mps"],
        "arms": {
            "baseline": {"policy": {"dynamic_braking_filter": False},
                         "result_dir": str(baseline_dir),
                         "summary": _brief(baseline_summary)},
            "dynamic_filter": {"policy": {"dynamic_braking_filter": True},
                               "result_dir": str(filter_dir),
                               "summary": _brief(filter_summary)},
        },
        "paired_outcomes": paired,
        "collision_seeds": {
            "baseline": [seed for seed, row in baseline_episodes.items()
                         if row["collision"]],
            "dynamic_filter": [seed for seed, row in filter_episodes.items()
                               if row["collision"]],
        },
        "interpretation": (
            "Report all trials, including starts below the configured dynamic "
            "braking-slack buffer. A margin failure is a diagnostic label, "
            "not proof that every control is physically unrecoverable."),
    }
    _write_json(output_root / "paired_comparison.json", comparison)
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--compact", action="store_true",
                        help="Print aggregate outcomes without per-seed rows")
    args = parser.parse_args(argv)
    comparison = run(args.output_root, tuple(args.seeds))
    if args.compact:
        comparison = {
            "experiment": comparison["experiment"],
            "output_root": str(args.output_root.resolve()),
            "paired_seed_count": len(comparison["paired_outcomes"]),
            "all_initial_conditions_match": all(
                row["same_initial_vehicles"]
                for row in comparison["paired_outcomes"]),
            "collision_seeds": comparison["collision_seeds"],
            "baseline": comparison["arms"]["baseline"]["summary"],
            "dynamic_filter": comparison["arms"]["dynamic_filter"]["summary"],
        }
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
