"""Scan fixed 3^5 local grids around the E27 non-chase counterexample states.

This deterministic neighborhood audit extends E21 to crossing, merge and
head_on. For each scene, 3^5 = 243 predeclared position and speed offsets around
the E27 center state are evaluated without random disturbances or sample pruning.
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

from .run_e27_non_chase_staged_counterexamples import (
    _family_audit, build_counterexample_states,
)
from .run_scene_closed_loop import _source_sha, _write_json
from .scene_closed_loop import (
    MODEL, _actions, _choose, _envelope_clearance, _recovery_actions,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (ROOT / "Result" / "ttc_multicar_20261122" /
                  "E28_non_chase_neighborhood_scan")
OFFSETS = (-0.2, 0.0, 0.2)
SCENES = ("crossing", "merge", "head_on")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_scene_grid(scene, offset_levels=OFFSETS):
    base_states = build_counterexample_states()
    base_cars, edges = base_states[scene]
    grid = []
    for index, values in enumerate(itertools.product(offset_levels, repeat=5)):
        d1, d2, dv1, dv2, dv3 = values
        if scene == "crossing":
            cars = {
                "W_to_E": replace(base_cars["W_to_E"],
                                  x=base_cars["W_to_E"].x + d1,
                                  speed=base_cars["W_to_E"].speed + dv1),
                "E_to_W": replace(base_cars["E_to_W"],
                                  x=base_cars["E_to_W"].x + d2,
                                  speed=base_cars["E_to_W"].speed + dv2),
                "S_to_N": replace(base_cars["S_to_N"],
                                  speed=base_cars["S_to_N"].speed + dv3),
            }
            offsets = {
                "dx_w_m": d1, "dx_e_m": d2,
                "dv_w_mps": dv1, "dv_e_mps": dv2, "dv_s_mps": dv3,
            }
        elif scene == "merge":
            gap_b1_m = base_cars["M"].x - base_cars["B1"].x
            gap_m_b2 = base_cars["B2"].x - base_cars["M"].x
            x_b1 = base_cars["B1"].x
            x_m = x_b1 + gap_b1_m + d1
            x_b2 = x_m + gap_m_b2 + d2
            cars = {
                "B1": replace(base_cars["B1"], x=x_b1,
                              speed=base_cars["B1"].speed + dv1),
                "M": replace(base_cars["M"], x=x_m,
                             speed=base_cars["M"].speed + dv2),
                "B2": replace(base_cars["B2"], x=x_b2,
                              speed=base_cars["B2"].speed + dv3),
            }
            offsets = {
                "gap_b1_m_offset_m": d1, "gap_m_b2_offset_m": d2,
                "dv_b1_mps": dv1, "dv_m_mps": dv2, "dv_b2_mps": dv3,
            }
        elif scene == "head_on":
            gap_e2_e1 = base_cars["E1"].x - base_cars["E2"].x
            gap_e1_w = base_cars["W"].x - base_cars["E1"].x
            x_e1 = base_cars["E1"].x
            x_e2 = x_e1 - (gap_e2_e1 + d1)
            x_w = x_e1 + (gap_e1_w + d2)
            cars = {
                "E2": replace(base_cars["E2"], x=x_e2,
                              speed=base_cars["E2"].speed + dv1),
                "E1": replace(base_cars["E1"], x=x_e1,
                              speed=base_cars["E1"].speed + dv2),
                "W": replace(base_cars["W"], x=x_w,
                             speed=base_cars["W"].speed + dv3),
            }
            offsets = {
                "gap_e2_e1_offset_m": d1, "gap_e1_w_offset_m": d2,
                "dv_e2_mps": dv1, "dv_e1_mps": dv2, "dv_w_mps": dv3,
            }
        else:
            raise ValueError("Unsupported scene {}".format(scene))
        grid.append({
            "state_id": "{}_grid_{:03d}".format(scene, index),
            "offsets": offsets,
            "cars": cars,
            "edges": edges,
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


def _audit_one(scene, state):
    cars = state["cars"]
    edges = state["edges"]
    names = tuple(sorted(cars))
    base_actions = {name: _actions(scene, cars[name]) for name in names}
    recovery_actions = {
        name: _recovery_actions(
            scene, cars[name], base_actions[name],
            [cars[other] for other in names if other != name])
        for name in names
    }
    constant = _family_audit(scene, cars, base_actions)
    staged = _family_audit(scene, cars, recovery_actions)
    controls, plan = _choose(
        scene, cars, edges, "global", {"dynamic_braking_filter": False})
    return {
        "state_id": state["state_id"],
        "offsets": state["offsets"],
        "initial_min_clearance_residual_m": _clearance_summary(cars),
        "constant_action_family": constant,
        "staged_recovery_family": staged,
        "controller_selection": {
            "fallback_entered": plan.get("fallback"),
            "selected_source": plan.get("selected_source"),
            "recovery_candidate_count": plan.get("recovery_candidate_count"),
            "recovery_profile_used": plan.get("recovery_profile_used"),
            "minimum_predicted_margin_residual_m": plan.get("min_clearance"),
            "selected_actions": {
                name: {
                    "name": action.name,
                    "first_control_accel_mps2": action.accel,
                    "target_y_m": action.target_y,
                }
                for name, action in controls.items()
            },
        },
    }


def _scan_scene(scene):
    states = build_scene_grid(scene, OFFSETS)
    rows = [_audit_one(scene, state) for state in states]
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
        row["state_id"]
        for row, c_ok, s_ok in zip(rows, constant_pass, staged_pass)
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
    if len(rows) != 243:
        raise RuntimeError("Grid for {} must contain 243 states".format(scene))
    return rows, {
        "state_count": len(rows),
        "constant_family_feasible_states": sum(constant_pass),
        "staged_family_feasible_states": sum(staged_pass),
        "states_where_staged_adds_feasible_candidate": len(gap_rows),
        "states_where_both_families_fail": len(both_fail),
        "constant_best_residual_range_m": [
            min(constant_best), max(constant_best)],
        "staged_best_residual_range_m": [
            min(staged_best), max(staged_best)],
        "controller_selected_staged_count": sum(
            row["controller_selection"]["selected_source"] ==
            "piecewise_braking_recovery" for row in rows),
        "gap_state_ids": [row["state_id"] for row in gap_rows],
        "both_fail_state_ids": both_fail,
    }


def run(output_root=DEFAULT_OUTPUT):
    all_states = {}
    aggregates = {}
    for scene in SCENES:
        rows, agg = _scan_scene(scene)
        all_states[scene] = rows
        aggregates[scene] = agg
    result = {
        "experiment": "E28_non_chase_staged_neighborhood_scan",
        "status": "deterministic 3^5 neighborhood grids for crossing, merge and head_on",
        "scope": "three scenes x 243 states = 729 states; 2.0 s rollout",
        "training_seed": None,
        "scenario_trial_seed": None,
        "offset_levels": list(OFFSETS),
        "total_states_evaluated": sum(
            agg["state_count"] for agg in aggregates.values()),
        "scenes": aggregates,
        "interpretation": (
            "Across 243-state local grids around each E27 counterexample, "
            "the scene-specific staged recovery family consistently expands "
            "the set of states with a feasible 2.0 s plan (crossing: +36, "
            "merge: +47, head_on: +143 states). All failed states are "
            "retained. This is a candidate-level neighborhood audit, not a "
            "closed-loop or naturalistic traffic prevalence estimate."),
        "source_hashes": {
            "runtime_sha256": _source_sha(),
            "e27_helper_sha256": _sha256(
                Path(__file__).with_name(
                    "run_e27_non_chase_staged_counterexamples.py")),
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
    _write_json(output_dir / "states.json", all_states)
    _write_json(output_dir / "manifest.json", {
        "experiment": result["experiment"],
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": result["source_hashes"],
        "python": sys.version,
        "platform": platform.platform(),
        "result_files": ["summary.json", "states.json"],
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
            "total_states_evaluated": result["total_states_evaluated"],
            "source_hashes": result["source_hashes"],
            "scenes": {
                scene: {
                    k: v for k, v in agg.items()
                    if k not in ("gap_state_ids", "both_fail_state_ids")
                }
                for scene, agg in result["scenes"].items()
            },
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
