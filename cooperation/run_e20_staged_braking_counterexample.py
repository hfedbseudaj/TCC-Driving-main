"""Audit a constructed state where staged braking fills an action-family gap.

This is a deterministic kinematic counterexample, not a CARLA or real-vehicle
experiment. It checks whether the current constant-action family can meet the
configured extra-clearance margin and compares it with the staged recovery
family at one explicit three-car chase state.
"""

import argparse
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
    MODEL, _actions, _car, _choose, _envelope_clearance,
    _pair_clearance, _recovery_actions, _robust_chase_braking_slack,
    _rollout,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (ROOT / "Result" / "ttc_multicar_20261121" /
                  "E20_staged_braking_counterexample")

# Fixed, explicitly constructed state found by a bounded exploratory search.
POSITIONS_M = (0.0, 6.480418060080996, 13.746349185372345)
SPEEDS_MPS = (13.816695470887003, 11.992656303349882, 9.712356443982612)
CAR_NAMES = ("R", "M", "F")
ROLES = ("rear", "middle", "front")
EDGES = (("R", "M"), ("M", "F"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fixed_cars():
    return {
        name: _car(name, x, 0.0, 1, speed, role, 30.0)
        for name, x, speed, role in zip(
            CAR_NAMES, POSITIONS_M, SPEEDS_MPS, ROLES)
    }


def _family_audit(cars, action_family):
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
                    "chase", plans[first_name][choice[first_name].name],
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
        "best_acceleration_schedules_mps2": {
            name: list(action.accel_schedule)
            for name, action in best_choice.items()
            if action.accel_schedule},
    }


def run(output_root=DEFAULT_OUTPUT):
    cars = _fixed_cars()
    names = tuple(sorted(cars))
    base_actions = {name: _actions("chase", cars[name]) for name in names}
    recovery_actions = {
        name: _recovery_actions(
            "chase", cars[name], base_actions[name],
            [cars[other] for other in names if other != name])
        for name in names
    }
    constant_result = _family_audit(cars, base_actions)
    recovery_result = _family_audit(cars, recovery_actions)
    current_pair_clearances = {}
    for index, first_name in enumerate(names):
        for second_name in names[index + 1:]:
            first, second = cars[first_name], cars[second_name]
            key = "{}-{}".format(first_name, second_name)
            current_pair_clearances[key] = {
                "envelope_clearance_m": _envelope_clearance(
                    first.x, first.y, first, second.x, second.y, second),
                "residual_after_1m_extra_margin_m": _envelope_clearance(
                    first.x, first.y, first, second.x, second.y, second) -
                MODEL["robust_prediction_margin_m"],
            }

    controls, plan = _choose(
        "chase", cars, EDGES, "global",
        {"dynamic_braking_filter": False})
    selected_actions = {
        name: {
            "name": action.name,
            "first_control_accel_mps2": action.accel,
            "predicted_schedule_mps2": list(action.accel_schedule),
        }
        for name, action in controls.items()
    }
    selected_min_clearance = plan.get("min_clearance")
    result = {
        "experiment": "E20_staged_braking_constructed_counterexample",
        "status": "single constructed deterministic kinematic state",
        "scope": "three same-lane vehicles; 2-second, 0.2-second-grid rollout",
        "training_seed": None,
        "scenario_trial_seed": None,
        "disturbances": "disabled for this deterministic audit",
        "vehicle_state": [
            {"id": name, "x_m": cars[name].x, "y_m": cars[name].y,
             "speed_mps": cars[name].speed, "role": cars[name].role,
             "direction": cars[name].direction}
            for name in names
        ],
        "conflict_edges": [list(edge) for edge in EDGES],
        "model_parameters": {
            "vehicle_length_m": MODEL["vehicle_length_m"],
            "longitudinal_margin_m": MODEL["longitudinal_margin_m"],
            "extra_prediction_margin_m": MODEL["robust_prediction_margin_m"],
            "control_period_s": MODEL["dt_s"],
            "planning_horizon_s": MODEL["horizon_s"],
            "max_accel_mps2": MODEL["max_accel_mps2"],
            "max_decel_mps2": MODEL["max_decel_mps2"],
        },
        "initial_pair_clearances": current_pair_clearances,
        "initial_robust_braking_slack_m": _robust_chase_braking_slack(cars),
        "constant_action_family": constant_result,
        "staged_recovery_family": recovery_result,
        "controller_selection": {
            "policy": "global; dynamic braking filter disabled",
            "fallback_entered": plan.get("fallback"),
            "selected_source": plan.get("selected_source"),
            "recovery_candidate_count": plan.get("recovery_candidate_count"),
            "recovery_profile_used": plan.get("recovery_profile_used"),
            "minimum_predicted_margin_residual_m": selected_min_clearance,
            "selected_predicted_robust_braking_slack_m": plan.get(
                "selected_predicted_robust_braking_slack_m"),
            "selected_actions": selected_actions,
        },
        "interpretation": (
            "At this constructed state, no constant-action joint candidate "
            "meets the configured 1 m extra predicted-clearance margin; the "
            "staged family has a feasible predicted continuation, and the "
            "controller selects it. This demonstrates an action-family gap "
            "only under the current simplified model and finite rollout. The "
            "remaining margin is small; this is not a collision-rate result, "
            "formal safety guarantee, CARLA result, or real-vehicle evidence."),
        "source_hashes": {
            "runtime_sha256": _source_sha(),
            "audit_runner_sha256": _sha256(__file__),
        },
        "python": sys.version,
        "platform": platform.platform(),
    }
    if (constant_result["joint_candidates_meeting_extra_margin"] != 0 or
            recovery_result["joint_candidates_meeting_extra_margin"] < 1 or
            plan.get("selected_source") != "piecewise_braking_recovery" or
            selected_min_clearance is None or selected_min_clearance < 0.0):
        raise RuntimeError("Constructed staged-braking counterexample failed")

    output_root = Path(output_root).resolve()
    run_id = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" +
              uuid.uuid4().hex[:8])
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "summary.json", result)
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
            "constant_best_min_margin_residual_m": result[
                "constant_action_family"][
                    "best_min_predicted_margin_residual_m"],
            "constant_candidates_meeting_extra_margin": result[
                "constant_action_family"][
                    "joint_candidates_meeting_extra_margin"],
            "staged_best_min_margin_residual_m": result[
                "staged_recovery_family"][
                    "best_min_predicted_margin_residual_m"],
            "staged_candidates_meeting_extra_margin": result[
                "staged_recovery_family"][
                    "joint_candidates_meeting_extra_margin"],
            "selected_source": result["controller_selection"][
                "selected_source"],
            "selected_actions": result["controller_selection"][
                "selected_actions"],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
