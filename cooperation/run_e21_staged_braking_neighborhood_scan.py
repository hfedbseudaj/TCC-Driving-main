"""Scan a fixed local grid around the E20 constructed chase state.

This deterministic neighborhood audit tests whether the staged-action-family
gap repeats under small, predeclared position/speed offsets. It is not a random
traffic-distribution estimate or closed-loop safety experiment.
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

from .run_e20_staged_braking_counterexample import (
    CAR_NAMES, EDGES, POSITIONS_M, ROLES, SPEEDS_MPS, _family_audit,
)
from .run_scene_closed_loop import _source_sha, _write_json
from .scene_closed_loop import (
    MODEL, _actions, _car, _choose, _envelope_clearance, _recovery_actions,
    _robust_chase_braking_slack,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (ROOT / "Result" / "ttc_multicar_20261121" /
                  "E21_staged_braking_neighborhood_scan")
OFFSETS = (-0.2, 0.0, 0.2)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_grid():
    base_gap_rm = POSITIONS_M[1] - POSITIONS_M[0]
    base_gap_mf = POSITIONS_M[2] - POSITIONS_M[1]
    grid = []
    for values in itertools.product(OFFSETS, repeat=5):
        gap_rm_delta, gap_mf_delta, dv_r, dv_m, dv_f = values
        x_r = POSITIONS_M[0]
        x_m = x_r + base_gap_rm + gap_rm_delta
        x_f = x_m + base_gap_mf + gap_mf_delta
        speeds = (SPEEDS_MPS[0] + dv_r,
                  SPEEDS_MPS[1] + dv_m,
                  SPEEDS_MPS[2] + dv_f)
        cars = {
            name: _car(name, x, 0.0, 1, speed, role, 30.0)
            for name, x, speed, role in zip(
                CAR_NAMES, (x_r, x_m, x_f), speeds, ROLES)
        }
        state_id = "grid_{:03d}".format(len(grid))
        grid.append({
            "state_id": state_id,
            "offsets": {
                "gap_rm_m": gap_rm_delta,
                "gap_mf_m": gap_mf_delta,
                "dv_r_mps": dv_r,
                "dv_m_mps": dv_m,
                "dv_f_mps": dv_f,
            },
            "cars": cars,
        })
    return grid


def _clearance_summary(cars):
    values = []
    names = tuple(sorted(cars))
    for index, first_name in enumerate(names):
        for second_name in names[index + 1:]:
            first, second = cars[first_name], cars[second_name]
            clear = _envelope_clearance(
                first.x, first.y, first, second.x, second.y, second)
            values.append(clear - MODEL["robust_prediction_margin_m"])
    return min(values) if values else float("inf")


def _audit_one(state):
    cars = state["cars"]
    names = tuple(sorted(cars))
    base_actions = {
        name: _actions("chase", cars[name]) for name in names}
    recovery_actions = {
        name: _recovery_actions(
            "chase", cars[name], base_actions[name],
            [cars[other] for other in names if other != name])
        for name in names
    }
    constant = _family_audit(cars, base_actions)
    staged = _family_audit(cars, recovery_actions)
    controls, plan = _choose(
        "chase", cars, EDGES, "global",
        {"dynamic_braking_filter": False})
    return {
        "state_id": state["state_id"],
        "offsets": state["offsets"],
        "initial_min_clearance_residual_m": _clearance_summary(cars),
        "initial_robust_braking_slack_m": _robust_chase_braking_slack(cars),
        "constant_action_family": constant,
        "staged_recovery_family": staged,
        "controller_selection": {
            "fallback_entered": plan.get("fallback"),
            "selected_source": plan.get("selected_source"),
            "recovery_candidate_count": plan.get("recovery_candidate_count"),
            "minimum_predicted_margin_residual_m": plan.get("min_clearance"),
            "selected_actions": {
                name: {
                    "name": action.name,
                    "first_control_accel_mps2": action.accel,
                }
                for name, action in controls.items()
            },
        },
    }


def run(output_root=DEFAULT_OUTPUT):
    states = _state_grid()
    rows = [_audit_one(state) for state in states]
    target_margin_m = MODEL["robust_prediction_margin_m"]
    constant_pass = [
        row["constant_action_family"][
            "joint_candidates_meeting_extra_margin"] > 0
        for row in rows
    ]
    staged_pass = [
        row["staged_recovery_family"][
            "joint_candidates_meeting_extra_margin"] > 0
        for row in rows
    ]
    gap_rows = [
        row for row, c_ok, s_ok in zip(rows, constant_pass, staged_pass)
        if not c_ok and s_ok
    ]
    both_fail = [
        row["state_id"] for row, c_ok, s_ok in zip(rows, constant_pass, staged_pass)
        if not c_ok and not s_ok
    ]
    staged_best = [
        row["staged_recovery_family"][
            "best_min_predicted_margin_residual_m"] for row in rows
    ]
    constant_best = [
        row["constant_action_family"][
            "best_min_predicted_margin_residual_m"] for row in rows
    ]
    result = {
        "experiment": "E21_staged_braking_local_neighborhood_grid",
        "status": "deterministic local grid; not sampled traffic distribution",
        "scope": "three same-lane vehicles; 243 fixed states; 2-second rollout",
        "training_seed": None,
        "scenario_trial_seed": None,
        "state_count": len(rows),
        "disturbances": "disabled; fixed predeclared offsets only",
        "grid_definition": {
            "base_positions_m": dict(zip(CAR_NAMES, POSITIONS_M)),
            "base_speeds_mps": dict(zip(CAR_NAMES, SPEEDS_MPS)),
            "offset_levels": list(OFFSETS),
            "dimensions": [
                "rear-middle center gap offset (m)",
                "middle-front center gap offset (m)",
                "rear speed offset (m/s)",
                "middle speed offset (m/s)",
                "front speed offset (m/s)",
            ],
            "grid_size": "3^5",
        },
        "model_parameters": {
            "vehicle_length_m": MODEL["vehicle_length_m"],
            "longitudinal_margin_m": MODEL["longitudinal_margin_m"],
            "extra_prediction_margin_m": target_margin_m,
            "control_period_s": MODEL["dt_s"],
            "planning_horizon_s": MODEL["horizon_s"],
            "max_accel_mps2": MODEL["max_accel_mps2"],
            "max_decel_mps2": MODEL["max_decel_mps2"],
        },
        "aggregate": {
            "constant_family_feasible_states": sum(constant_pass),
            "staged_family_feasible_states": sum(staged_pass),
            "states_where_staged_adds_feasible_candidate": len(gap_rows),
            "states_where_both_families_fail": len(both_fail),
            "states_with_initial_robust_slack_below_0_25m": sum(
                row["initial_robust_braking_slack_m"] < 0.25 for row in rows),
            "constant_best_residual_range_m": [
                min(constant_best), max(constant_best)],
            "staged_best_residual_range_m": [
                min(staged_best), max(staged_best)],
            "gap_state_ids": [row["state_id"] for row in gap_rows],
            "both_fail_state_ids": both_fail,
            "controller_selected_staged_count": sum(
                row["controller_selection"]["selected_source"] ==
                "piecewise_braking_recovery" for row in rows),
        },
        "interpretation": (
            "The grid is a fixed local perturbation around the constructed "
            "E20 state. It measures whether the candidate-family gap repeats "
            "within this neighborhood only. It is not a prevalence estimate, "
            "closed-loop safety result, formal guarantee, CARLA result, or "
            "real-vehicle evidence."),
        "source_hashes": {
            "runtime_sha256": _source_sha(),
            "e20_helper_sha256": _sha256(
                Path(__file__).with_name(
                    "run_e20_staged_braking_counterexample.py")),
            "audit_runner_sha256": _sha256(__file__),
        },
        "python": sys.version,
        "platform": platform.platform(),
    }
    if len(rows) != 243:
        raise RuntimeError("The fixed E21 grid must contain 243 states")
    output_root = Path(output_root).resolve()
    run_id = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" +
              uuid.uuid4().hex[:8])
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "summary.json", result)
    _write_json(output_dir / "states.json", rows)
    manifest = {
        "experiment": result["experiment"],
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": result["grid_definition"],
        "model_parameters": result["model_parameters"],
        "source_hashes": result["source_hashes"],
        "python": sys.version,
        "platform": platform.platform(),
        "training_seed": None,
        "scenario_trial_seed": None,
        "result_files": ["summary.json", "states.json"],
    }
    _write_json(output_dir / "manifest.json", manifest)
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
            "state_count": result["state_count"],
            "aggregate": result["aggregate"],
            "runtime_sha256": result["source_hashes"]["runtime_sha256"],
            "audit_runner_sha256": result["source_hashes"][
                "audit_runner_sha256"],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
