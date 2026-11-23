"""Audit constructed states where staged recovery fills non-chase action-family gaps.

This is a deterministic kinematic counterexample audit for the three non-chase
scenes (crossing, merge, head_on), extending E20 beyond same-lane chase. At each
explicit three-car state, the ordinary constant-action joint family has zero
candidates meeting the configured 1.0 m extra predicted-clearance margin,
whereas the scene-specific staged recovery family has at least one feasible
continuation and is selected by the controller.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import uuid

from .run_scene_closed_loop import _source_sha, _write_json
from .scene_closed_loop import (
    MODEL, _actions, _choose, _envelope_clearance, _pair_clearance,
    _recovery_actions, _rollout, make_scene,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (ROOT / "Result" / "ttc_multicar_20261122" /
                  "E27_non_chase_staged_counterexamples")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_counterexample_states():
    # 1. Three-car crossing: S_to_N must briefly brake so W_to_E clears the
    # first crossing band, then release braking to clear the second band before
    # E_to_W arrives.
    c_cross, edges_cross = make_scene("crossing", 3)
    cars_cross = {
        "W_to_E": replace(c_cross["W_to_E"], x=3.87, speed=10.0, goal_x=20.0),
        "E_to_W": replace(c_cross["E_to_W"], x=21.45, speed=12.0, goal_x=-20.0),
        "S_to_N": replace(c_cross["S_to_N"], y=-8.20, speed=8.8, goal_x=20.0),
    }

    # 2. Three-car merge: partially merged vehicle M between rear mainline B1
    # and slower front mainline B2 cannot maintain constant slow_merge (-1.0)
    # without encroaching on B2, nor constant abort_brake (-3.0) without B1
    # encroaching from behind; short brake-then-release succeeds.
    c_merge, edges_merge = make_scene("merge", 3)
    cars_merge = {
        "B1": replace(c_merge["B1"], x=0.0, speed=11.95, goal_x=30.0),
        "M": replace(c_merge["M"], x=6.45, y=-2.2, speed=11.5,
                     lateral_speed=0.3, goal_x=30.0),
        "B2": replace(c_merge["B2"], x=13.00, speed=9.40, goal_x=30.0),
    }

    # 3. Three-car head-on: eastbound leader E1 is followed closely by E2 while
    # approaching westbound W. Constant braking on E1 causes rear conflict with
    # E2, while constant coasting while evading cannot clear W's longitudinal
    # overlap before lateral separation completes; delayed hard braking coupled
    # with lateral evasion succeeds.
    c_head, edges_head = make_scene("head_on", 3)
    cars_head = {
        "E2": replace(c_head["E2"], x=-6.75, speed=11.8, goal_x=30.0),
        "E1": replace(c_head["E1"], x=0.0, speed=10.2, goal_x=30.0),
        "W": replace(c_head["W"], x=31.8, speed=9.5, goal_x=-20.0),
    }

    return {
        "crossing": (cars_cross, edges_cross),
        "merge": (cars_merge, edges_merge),
        "head_on": (cars_head, edges_head),
    }


def _family_audit(scene, cars, action_family):
    names = tuple(sorted(cars))
    plans = {
        name: {
            action.name: _rollout(
                cars[name], action, MODEL["dt_s"], MODEL["horizon_s"])
            for action in action_family[name]
        }
        for name in names
    }
    best_worst_clearance = float("-inf")
    best_choice = None
    safe_joint_count = 0
    total_joint_count = 1
    for name in names:
        total_joint_count *= len(action_family[name])
    for selected in itertools.product(*(action_family[name] for name in names)):
        choice = dict(zip(names, selected))
        pair_values = []
        for index, first_name in enumerate(names):
            for second_name in names[index + 1:]:
                pair_values.append(_pair_clearance(
                    scene, plans[first_name][choice[first_name].name],
                    plans[second_name][choice[second_name].name],
                    cars[first_name], cars[second_name]))
        worst_clearance = min(pair_values) if pair_values else float("inf")
        if worst_clearance >= -1e-9:
            safe_joint_count += 1
        if worst_clearance > best_worst_clearance:
            best_worst_clearance = worst_clearance
            best_choice = choice
    return {
        "joint_candidate_count": total_joint_count,
        "joint_candidates_meeting_extra_margin": safe_joint_count,
        "best_min_predicted_margin_residual_m": best_worst_clearance,
        "best_action_names": {
            name: action.name for name, action in best_choice.items()},
        "best_action_targets_y_m": {
            name: action.target_y for name, action in best_choice.items()},
        "best_acceleration_schedules_mps2": {
            name: list(action.accel_schedule)
            for name, action in best_choice.items()
            if action.accel_schedule},
    }


def _audit_scene(scene, cars, edges):
    names = tuple(sorted(cars))
    base_actions = {name: _actions(scene, cars[name]) for name in names}
    recovery_actions = {
        name: _recovery_actions(
            scene, cars[name], base_actions[name],
            [cars[other] for other in names if other != name])
        for name in names
    }
    constant_result = _family_audit(scene, cars, base_actions)
    recovery_result = _family_audit(scene, cars, recovery_actions)
    current_pair_clearances = {}
    for index, first_name in enumerate(names):
        for second_name in names[index + 1:]:
            first, second = cars[first_name], cars[second_name]
            key = "{}-{}".format(first_name, second_name)
            env_clear = _envelope_clearance(
                first.x, first.y, first, second.x, second.y, second)
            current_pair_clearances[key] = {
                "envelope_clearance_m": env_clear,
                "residual_after_1m_extra_margin_m": (
                    env_clear - MODEL["robust_prediction_margin_m"]),
            }
    controls, plan = _choose(
        scene, cars, edges, "global", {"dynamic_braking_filter": False})
    selected_actions = {
        name: {
            "name": action.name,
            "first_control_accel_mps2": action.accel,
            "target_y_m": action.target_y,
            "predicted_schedule_mps2": list(action.accel_schedule),
        }
        for name, action in controls.items()
    }
    selected_min_clearance = plan.get("min_clearance")
    if (constant_result["joint_candidates_meeting_extra_margin"] != 0 or
            recovery_result["joint_candidates_meeting_extra_margin"] < 1 or
            plan.get("selected_source") != "piecewise_braking_recovery" or
            selected_min_clearance is None or selected_min_clearance < 0.0):
        raise RuntimeError(
            "Constructed counterexample failed for scene {}".format(scene))
    return {
        "scene": scene,
        "n_cars": len(cars),
        "vehicle_state": [
            {"id": name, "x_m": cars[name].x, "y_m": cars[name].y,
             "speed_mps": cars[name].speed,
             "lateral_speed_mps": cars[name].lateral_speed,
             "role": cars[name].role, "direction": cars[name].direction,
             "axis": cars[name].axis}
            for name in names
        ],
        "conflict_edges": [list(edge) for edge in edges],
        "initial_pair_clearances": current_pair_clearances,
        "constant_action_family": constant_result,
        "staged_recovery_family": recovery_result,
        "controller_selection": {
            "policy": "global; dynamic braking filter disabled",
            "fallback_entered": plan.get("fallback"),
            "selected_source": plan.get("selected_source"),
            "recovery_candidate_count": plan.get("recovery_candidate_count"),
            "recovery_profile_used": plan.get("recovery_profile_used"),
            "minimum_predicted_margin_residual_m": selected_min_clearance,
            "selected_actions": selected_actions,
            "solver_ms": plan.get("solver_ms"),
        },
    }


def run(output_root=DEFAULT_OUTPUT):
    states = build_counterexample_states()
    scenes_report = {
        scene: _audit_scene(scene, cars, edges)
        for scene, (cars, edges) in states.items()
    }
    result = {
        "experiment": "E27_non_chase_staged_recovery_counterexamples",
        "status": "three constructed deterministic kinematic counterexamples",
        "scope": "three-car crossing, merge and head_on; 2.0 s rollout at 0.2 s step",
        "training_seed": None,
        "scenario_trial_seed": None,
        "disturbances": "disabled for deterministic candidate-family counterexample audit",
        "model_parameters": {
            "vehicle_length_m": MODEL["vehicle_length_m"],
            "vehicle_width_m": MODEL["vehicle_width_m"],
            "longitudinal_margin_m": MODEL["longitudinal_margin_m"],
            "lateral_margin_m": MODEL["lateral_margin_m"],
            "extra_prediction_margin_m": MODEL["robust_prediction_margin_m"],
            "control_period_s": MODEL["dt_s"],
            "planning_horizon_s": MODEL["horizon_s"],
        },
        "scenes": scenes_report,
        "interpretation": (
            "In each of the three non-chase scenes (crossing, merge, head_on), "
            "an explicit three-car initial state exists where all initial "
            "pair clearances exceed the 1.0 m extra margin, yet the ordinary "
            "constant-action joint family has zero feasible 2.0 s plans. "
            "The scene-specific staged recovery family restores at least one "
            "feasible plan and is selected by the controller. These single-state "
            "counterexamples establish candidate-level empty-set repair only, "
            "not traffic-distribution prevalence or closed-loop safety gains."),
        "source_hashes": {
            "runtime_sha256": _source_sha(),
            "audit_runner_sha256": _sha256(__file__),
        },
        "python": sys.version,
        "platform": platform.platform(),
    }
    output_root = Path(output_root).resolve()
    run_id = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" +
              uuid.uuid4().hex[:8])
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "summary.json", result)
    _write_json(output_dir / "manifest.json", {
        "experiment": result["experiment"],
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": result["source_hashes"],
        "python": sys.version,
        "platform": platform.platform(),
        "result_files": ["summary.json"],
    })
    return output_dir, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)
    output_dir, result = run(args.output_root)
    if args.compact:
        result = {
            "output_dir": str(output_dir),
            "source_hashes": result["source_hashes"],
            "scenes": {
                scene: {
                    "constant_joint_candidates": row["constant_action_family"][
                        "joint_candidate_count"],
                    "constant_safe_candidates": row["constant_action_family"][
                        "joint_candidates_meeting_extra_margin"],
                    "constant_best_residual_m": row["constant_action_family"][
                        "best_min_predicted_margin_residual_m"],
                    "staged_joint_candidates": row["staged_recovery_family"][
                        "joint_candidate_count"],
                    "staged_safe_candidates": row["staged_recovery_family"][
                        "joint_candidates_meeting_extra_margin"],
                    "staged_best_residual_m": row["staged_recovery_family"][
                        "best_min_predicted_margin_residual_m"],
                    "selected_source": row["controller_selection"][
                        "selected_source"],
                    "selected_actions": {
                        k: v["name"] for k, v in row["controller_selection"][
                            "selected_actions"].items()
                    },
                }
                for scene, row in result["scenes"].items()
            },
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
