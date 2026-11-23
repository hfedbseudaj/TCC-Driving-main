"""Run a paired closed-loop ablation of the staged chase recovery family.

The control arm disables only the staged-recovery action family; its existing
one-step fallback, safety checks, utility and all other controller settings
remain active. Seeds are held out from E19's 801-820 range.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import uuid

from . import scene_closed_loop
from .run_scene_closed_loop import (
    _source_sha, _write_json, run_closed_loop_sweep,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (ROOT / "Result" / "ttc_multicar_20261121" /
                  "E22_staged_recovery_paired")
SEEDS = tuple(range(821, 841))
POSITION_STD_M = 1.0
SPEED_STD_MPS = 0.8
METHODS = ("global",)
SUMMARY_KEYS = (
    "trial_count", "initial_robust_braking_margin_failure_count",
    "collision_count", "near_collision_count", "completion_count",
    "fallback_count", "recovery_step_count",
    "recovery_profile_step_count", "recovery_candidate_count",
    "next_step_uncertified_fallback_ticks",
    "next_step_collision_uncertified_fallback_ticks",
    "mean_solver_ms", "p95_solver_ms", "max_solver_ms",
    "mean_completed_delay_s",
)


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_map(path):
    with (Path(path) / "episodes.json").open("r", encoding="utf-8") as handle:
        episodes = json.load(handle)
    return {row["trial_seed"]: row for row in episodes}


def _brief(summary):
    group = summary["chase"]["3"]["global"]
    return {key: group.get(key) for key in SUMMARY_KEYS}


def _noise_sequence_hash(seed):
    _, _, noises = scene_closed_loop._trial_initial(
        "chase", 3, seed, POSITION_STD_M, SPEED_STD_MPS)
    serialized = json.dumps(
        noises, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256_bytes(serialized.encode("utf-8"))


def _run_arm(output_dir, seeds, staged_recovery_enabled):
    original = scene_closed_loop._recovery_actions
    try:
        if not staged_recovery_enabled:
            # This ablation removes only multi-step recovery candidates. The
            # ordinary constant-action candidates and one-step fallback remain
            # available to the existing recovery/fallback logic.
            scene_closed_loop._recovery_actions = (
                lambda scene, car, base_actions, other_cars=(): tuple(
                    action for action in base_actions
                    if not action.accel_schedule))
        run_dir, summary = run_closed_loop_sweep(
            output_root=output_dir,
            scenes=("chase",),
            n_car_counts=(3,),
            trial_seeds=tuple(seeds),
            position_std_m=POSITION_STD_M,
            speed_std_mps=SPEED_STD_MPS,
            methods=METHODS,
            policy={"dynamic_braking_filter": False},
        )
    finally:
        scene_closed_loop._recovery_actions = original
    _write_json(run_dir / "experiment_arm_override.json", {
        "staged_recovery_action_family_enabled": staged_recovery_enabled,
        "control_arm_override": (
            "_recovery_actions returns ordinary base actions only; all "
            "multi-step scheduled braking candidates are removed; one-step "
            "fallback and other policy/safety logic remain enabled"
            if not staged_recovery_enabled else "none"),
        "selection_method": "global",
        "dynamic_braking_filter": False,
    })
    return run_dir, summary


def run(output_root=DEFAULT_OUTPUT, seeds=SEEDS):
    output_root = Path(output_root).resolve()
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide nonempty unique scenario trial seeds")
    if set(seeds) & set(range(801, 821)):
        raise ValueError("E22 seeds must be separate from E19 seeds 801-820")

    disabled_dir, disabled_summary = _run_arm(
        output_root / "staged_recovery_disabled", seeds, False)
    enabled_dir, enabled_summary = _run_arm(
        output_root / "staged_recovery_enabled", seeds, True)
    disabled_episodes = _episode_map(disabled_dir)
    enabled_episodes = _episode_map(enabled_dir)
    if set(disabled_episodes) != set(enabled_episodes):
        raise RuntimeError("Paired arms did not produce matching seed sets")

    paired = []
    for seed in sorted(disabled_episodes):
        control = disabled_episodes[seed]
        treatment = enabled_episodes[seed]
        noise_sha = _noise_sequence_hash(seed)
        paired.append({
            "scenario_trial_seed": seed,
            "same_initial_vehicles": (
                control["initial_vehicles"] == treatment["initial_vehicles"]),
            "precomputed_execution_noise_sha256": noise_sha,
            "control_collision": control["collision"],
            "staged_collision": treatment["collision"],
            "control_near_collision": control["near_collision"],
            "staged_near_collision": treatment["near_collision"],
            "control_completed": control["completed"],
            "staged_completed": treatment["completed"],
            "control_collision_time_s": control["collision_time_s"],
            "staged_collision_time_s": treatment["collision_time_s"],
            "control_fallback_count": control["fallback_count"],
            "staged_fallback_count": treatment["fallback_count"],
            "control_recovery_profile_steps": control[
                "recovery_profile_step_count"],
            "staged_recovery_profile_steps": treatment[
                "recovery_profile_step_count"],
        })
    if not all(row["same_initial_vehicles"] for row in paired):
        raise RuntimeError("Paired arms used different initial vehicle states")

    disabled_manifest = json.loads(
        (disabled_dir / "manifest.json").read_text(encoding="utf-8"))
    enabled_manifest = json.loads(
        (enabled_dir / "manifest.json").read_text(encoding="utf-8"))
    if disabled_manifest["source_sha256"] != enabled_manifest["source_sha256"]:
        raise RuntimeError("Paired arms used different runtime source hashes")
    run_id = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" +
              uuid.uuid4().hex[:8])
    result_dir = output_root / "paired_comparison" / run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "experiment": "E22_paired_staged_braking_recovery_ablation",
        "status": "paired constructed kinematic closed-loop comparison",
        "scope": "chase scene; three vehicles; global method",
        "scenario_trial_seeds": list(seeds),
        "trial_count": len(seeds),
        "training_seed": None,
        "checkpoint_path": None,
        "paired_initial_conditions_and_execution_noise": True,
        "position_perturbation_std_m": POSITION_STD_M,
        "speed_perturbation_std_mps": SPEED_STD_MPS,
        "execution_acceleration_noise_std_mps2": (
            scene_closed_loop.MODEL["accel_noise_std_mps2"]),
        "source_hashes": {
            "runtime_sha256": disabled_manifest["source_sha256"],
            "paired_runner_sha256": _sha256_file(__file__),
        },
        "arms": {
            "staged_recovery_disabled": {
                "run_dir": str(disabled_dir),
                "summary": _brief(disabled_summary),
            },
            "staged_recovery_enabled": {
                "run_dir": str(enabled_dir),
                "summary": _brief(enabled_summary),
            },
        },
        "paired_outcomes": paired,
        "interpretation": (
            "This paired ablation measures the effect of adding the staged "
            "recovery action family under one constructed chase generator and "
            "the listed held-out seeds. It does not establish safety beyond "
            "this model or distribution, and is not CARLA or real-vehicle "
            "evidence."),
        "python": sys.version,
        "platform": platform.platform(),
    }
    _write_json(result_dir / "paired_comparison.json", summary)
    _write_json(result_dir / "manifest.json", {
        "experiment": summary["experiment"],
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_trial_seeds": list(seeds),
        "training_seed": None,
        "position_perturbation_std_m": POSITION_STD_M,
        "speed_perturbation_std_mps": SPEED_STD_MPS,
        "execution_acceleration_noise_std_mps2": (
            scene_closed_loop.MODEL["accel_noise_std_mps2"]),
        "arms": {
            "staged_recovery_disabled": str(disabled_dir),
            "staged_recovery_enabled": str(enabled_dir),
        },
        "source_hashes": summary["source_hashes"],
        "python": sys.version,
        "platform": platform.platform(),
        "paired_initial_conditions_and_execution_noise": True,
    })
    return result_dir, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trial-seeds", nargs="+", type=int,
                        default=list(SEEDS))
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)
    output_dir, result = run(args.output_root, args.trial_seeds)
    if args.compact:
        result = {
            "output_dir": str(output_dir),
            "trial_count": result["trial_count"],
            "source_hashes": result["source_hashes"],
            "arms": {name: arm["summary"]
                     for name, arm in result["arms"].items()},
            "collision_pairs": [
                row["scenario_trial_seed"] for row in result["paired_outcomes"]
                if row["control_collision"] or row["staged_collision"]],
            "near_collision_counts": {
                "staged_recovery_disabled": sum(
                    row["control_near_collision"]
                    for row in result["paired_outcomes"]),
                "staged_recovery_enabled": sum(
                    row["staged_near_collision"]
                    for row in result["paired_outcomes"]),
            },
            "all_initial_states_match": all(
                row["same_initial_vehicles"]
                for row in result["paired_outcomes"]),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
