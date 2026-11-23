"""Probe staged-braking candidate transfer to the other three scenes.

The audit injects the four E20/E23 longitudinal acceleration schedules into
scene-aware action targets at one deterministic snapshot per scene/group. It
compares the existing joint safety filter's candidate counts. It does not run
closed loop and does not claim causal safety or efficiency gains.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

from . import scene_closed_loop as sim


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (ROOT / "Result" / "ttc_multicar_20261121" /
                  "E26_cross_scene_staged_candidate_audit")
SCENES = ("crossing", "merge", "head_on")
CAR_COUNTS = (2, 3, 4)


def _profiles(count):
    return (
        ("brake_release", (-4.0,) + (0.0,) * (count - 1)),
        ("soft_then_hard", (-2.0, -2.0) +
         (-4.0,) * max(0, count - 2)),
        ("coast_then_hard", (0.0,) + (-4.0,) * max(0, count - 1)),
        ("delayed_hard", (0.0,) * min(3, count) +
         (-4.0,) * max(0, count - 3)),
    )


def _target_roles(scene, cars, base_actions):
    if scene == "crossing":
        return {name: sorted(set(action.target_y for action in actions))
                for name, actions in base_actions.items()}
    if scene == "merge":
        return {
            name: (sorted(set(action.target_y for action in base_actions[name]))
                   if car.role == "merging" else [])
            for name, car in cars.items()
        }
    # For head-on, add braking schedules only to the opposing lead pair and
    # retain each vehicle's existing lane-hold / evasive targets.
    opposing = {"E1", "W"}
    return {
        name: (sorted(set(action.target_y for action in base_actions[name]))
               if name in opposing else [])
        for name in cars
    }


def _snapshot(scene, n_cars):
    cars, edges = sim.make_scene(scene, n_cars)
    base = {name: sim._actions(scene, car) for name, car in cars.items()}
    targets = _target_roles(scene, cars, base)
    count = int(round(sim.MODEL["horizon_s"] / sim.MODEL["dt_s"]))
    augmented = {}
    for name, actions in base.items():
        additions = []
        if targets[name]:
            existing = {(action.accel, action.target_y) for action in actions}
            for target_y in targets[name]:
                for label, schedule in _profiles(count):
                    if (all(value == schedule[0] for value in schedule) and
                            (schedule[0], target_y) in existing):
                        continue
                    additions.append(sim.Action(
                        "stage_{}_{}_{}".format(name, label, len(additions)),
                        schedule[0], target_y, tuple(schedule)))
        augmented[name] = tuple(actions) + tuple(additions)

    policy = {"dynamic_braking_filter": False}
    control_actions, control = sim._choose(
        scene, cars, edges, "global", policy)
    original_actions = sim._actions
    try:
        sim._actions = lambda _scene, car: augmented[car.id]
        treatment_actions, treatment = sim._choose(
            scene, cars, edges, "global", policy)
    finally:
        sim._actions = original_actions

    return {
        "scene": scene,
        "n_cars": n_cars,
        "initial_vehicles": {
            name: {"x_m": car.x, "y_m": car.y,
                   "speed_mps": car.speed, "direction": car.direction,
                   "axis": car.axis, "role": car.role}
            for name, car in cars.items()
        },
        "conflict_edges": [list(edge) for edge in edges],
        "baseline_action_counts": {
            name: len(actions) for name, actions in base.items()},
        "staged_candidate_counts": {
            name: sum(bool(action.accel_schedule)
                      for action in augmented[name])
            for name in cars},
        "baseline": {
            "safe_joint_candidate_count": control["safe_candidate_count"],
            "individual_rational_candidate_count": control[
                "ir_candidate_count"],
            "fallback": control["fallback"],
            "selected_actions": {
                name: action.name for name, action in control_actions.items()},
            "predicted_min_clearance_m": control["min_clearance"],
            "solver_ms": control["solver_ms"],
        },
        "staged_candidates_added": {
            "safe_joint_candidate_count": treatment["safe_candidate_count"],
            "individual_rational_candidate_count": treatment[
                "ir_candidate_count"],
            "fallback": treatment["fallback"],
            "selected_actions": {
                name: action.name for name, action in treatment_actions.items()},
            "selected_accel_schedules": treatment["selected_accel_schedules"],
            "predicted_min_clearance_m": treatment["min_clearance"],
            "solver_ms": treatment["solver_ms"],
        },
        "interpretation_scope": (
            "One deterministic planning snapshot; staged candidates are added "
            "to the joint candidate family for this comparison, rather than "
            "being enabled by a closed-loop recovery trigger."),
    }


def run(output_root=DEFAULT_OUTPUT):
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    snapshots = [_snapshot(scene, count)
                 for scene in SCENES for count in CAR_COUNTS]
    source_files = ("cooperation/physics.py",
                    "cooperation/scene_closed_loop.py",
                    "cooperation/run_e26_cross_scene_staged_candidate_audit.py")
    hashes = {relative: hashlib.sha256(
        (ROOT / relative).read_bytes()).hexdigest() for relative in source_files}
    digest = hashlib.sha256()
    for relative in source_files:
        digest.update(relative.encode("utf-8"))
        digest.update((ROOT / relative).read_bytes())
    result = {
        "experiment": "E26_cross_scene_staged_candidate_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "one deterministic snapshot for each of 3 scenes × 3 vehicle counts",
        "base_schedules": {
            name: list(schedule) for name, schedule in _profiles(
                int(round(sim.MODEL["horizon_s"] / sim.MODEL["dt_s"])))
        },
        "scene_specific_targeting": {
            "crossing": "all vehicles and their existing lane targets",
            "merge": "merging vehicle only, retaining its existing merge/abort targets",
            "head_on": "opposing lead pair E1/W and their existing hold/evasion targets",
        },
        "snapshots": snapshots,
        "source_sha256": hashes,
        "runtime_composite_sha256": digest.hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "limitations": [
            "This is an action-family candidate audit, not a paired closed-loop experiment.",
            "The staged schedules are considered in the joint candidate family at "
            "the snapshot; no recovery trigger is modeled here.",
            "A candidate count does not prove that a schedule is dynamically "
            "appropriate, selected in closed loop, or beneficial across initial states.",
            "All scenes are constructed kinematics, not CARLA or naturalistic traffic.",
        ],
    }
    path = output_root / "cross_scene_candidates.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")
    return path, result


def main():
    path, result = run()
    summary = []
    for row in result["snapshots"]:
        base = row["baseline"]["safe_joint_candidate_count"]
        extended = row["staged_candidates_added"]["safe_joint_candidate_count"]
        summary.append({
            "scene": row["scene"], "n_cars": row["n_cars"],
            "baseline_safe_candidates": base,
            "with_staged_safe_candidates": extended,
            "staged_added_candidates": extended - base,
            "baseline_selected_staged": bool(any(
                name.startswith("stage_")
                for name in row["staged_candidates_added"][
                    "selected_actions"].values())),
        })
    print(json.dumps({"output_file": str(path), "snapshots": summary},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
