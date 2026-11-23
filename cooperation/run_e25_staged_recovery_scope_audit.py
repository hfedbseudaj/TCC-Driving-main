"""Audit whether the current staged-recovery implementation covers all scenes.

This is a code-scope and baseline behavior audit, not evidence that staged
braking is physically ineffective in the three non-chase scenes.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

from . import scene_closed_loop


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (ROOT / "Result" / "ttc_multicar_20261121" /
                  "E25_staged_recovery_scope_audit")
SCENES = ("crossing", "merge", "head_on")
CAR_COUNTS = (2, 3, 4)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run_audit(output_root=DEFAULT_OUTPUT, trial_seed=2501):
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    episodes = []
    direct_action_checks = []
    for scene in SCENES:
        for count in CAR_COUNTS:
            cars, edges, _noise = scene_closed_loop._trial_initial(
                scene, count, trial_seed, 0.0, 0.0)
            for name, car in cars.items():
                base = scene_closed_loop._actions(scene, car)
                returned = scene_closed_loop._recovery_actions(
                    scene, car, base,
                    [other for other_name, other in cars.items()
                     if other_name != name])
                direct_action_checks.append({
                    "scene": scene,
                    "n_cars": count,
                    "vehicle": name,
                    "base_action_count": len(base),
                    "returned_action_count": len(returned),
                    "scheduled_profile_count": sum(
                        bool(action.accel_schedule) for action in returned),
                    "returned_base_action_names_unchanged": (
                        tuple(action.name for action in returned) ==
                        tuple(action.name for action in base)),
                })
            episode = scene_closed_loop.run_episode(
                scene, count, trial_seed, "global", 0.0, 0.0,
                {"dynamic_braking_filter": False})
            episodes.append({
                "scene": scene,
                "n_cars": count,
                "trial_seed": trial_seed,
                "collision": episode["collision"],
                "near_collision": episode["near_collision"],
                "completed": episode["completed"],
                "fallback_count": episode["fallback_count"],
                "recovery_candidate_count": episode[
                    "recovery_candidate_count"],
                "recovery_profile_step_count": episode[
                    "recovery_profile_step_count"],
                "initial_vehicles": episode["initial_vehicles"],
                "trace": episode["trace"],
            })

    source_files = ("cooperation/physics.py",
                    "cooperation/scene_closed_loop.py",
                    "cooperation/run_scene_closed_loop.py",
                    "cooperation/run_e25_staged_recovery_scope_audit.py")
    runtime_digest = hashlib.sha256()
    for relative in source_files:
        runtime_digest.update(relative.encode("utf-8"))
        runtime_digest.update((ROOT / relative).read_bytes())

    result = {
        "experiment": "E25_staged_recovery_scope_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "crossing, merge, head_on; 2/3/4 cars; constructed baseline states",
        "trial_seed": trial_seed,
        "position_perturbation_std_m": 0.0,
        "speed_perturbation_std_mps": 0.0,
        "code_findings": {
            "recovery_search_guard": "scene == 'chase' and current_full_margin_satisfied",
            "recovery_action_factory_guard": "scene != 'chase' returns base actions unchanged",
            "supported_by_current_staged_family": ["chase"],
            "other_scene_physical_applicability": "not established by this audit",
        },
        "scene_assessment": {
            "crossing": (
                "Longitudinal staged braking could alter conflict-zone arrival "
                "times, but needs crossing-zone timing and signal/route-aware "
                "triggers; current code does not add or search it."),
            "merge": (
                "Longitudinal staged braking could alter merge gaps, but needs "
                "joint longitudinal/lateral prediction and merge-completion "
                "checks; current code does not add or search it."),
            "head_on": (
                "Braking may help but is not a sufficient generic response to "
                "head-on conflict; lateral evasion and road-bound constraints "
                "must be coordinated. Current code does not add or search it."),
        },
        "direct_action_checks": direct_action_checks,
        "baseline_episodes": episodes,
        "source_files_sha256": {
            relative: _sha256(ROOT / relative) for relative in source_files
        },
        "runtime_composite_sha256": runtime_digest.hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "limitations": [
            "The audit confirms implementation scope and current baseline behavior only.",
            "It does not claim staged braking is ineffective or unsafe in other scenes.",
            "No staged-vs-control causal comparison is possible there until a "
            "scene-specific action family, trigger and safety filter are defined.",
            "Results are constructed kinematic simulations, not CARLA or real traffic.",
        ],
    }
    output_path = output_root / "scope_audit.json"
    output_path.write_text(json.dumps(
        result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return output_path, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trial-seed", type=int, default=2501)
    args = parser.parse_args(argv)
    path, result = _run_audit(args.output_root, args.trial_seed)
    print(json.dumps({
        "output_file": str(path),
        "runtime_composite_sha256": result["runtime_composite_sha256"],
        "direct_action_check_count": len(result["direct_action_checks"]),
        "baseline_episode_count": len(result["baseline_episodes"]),
        "episodes_with_recovery_profiles": sum(
            row["recovery_profile_step_count"] > 0
            for row in result["baseline_episodes"]),
        "collision_count": sum(
            int(row["collision"]) for row in result["baseline_episodes"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
