"""Run a reproducible, constructed crossing-cooperation snapshot.

This entry point does not train a policy, advance a vehicle, or estimate a
collision rate. It checks the report's D2 -> K_e -> finite-family cooperation
chain on explicitly supplied physical candidates.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

from .compose import compose_crossing_snapshot
from .crossing import CrossingContract, PairProposal, decode_crossing_pair
from .physics import (
    VehicleSpec, canonical_approach_plan, legacy_closing_ttc, pair_safe,
)


_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _ROOT / "config" / "multicar_cooperation_example.json"
_SOURCE_FILES = (
    "cooperation/physics.py",
    "cooperation/crossing.py",
    "cooperation/compose.py",
    "cooperation/run_snapshot.py",
)


def _safe_json(value):
    """Keep written JSON strict while preserving an infinite-TTC label."""
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "infinity" if value > 0 else "negative_infinity"
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_safe_json(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def _version_record(config_path):
    digest = hashlib.sha256()
    for relative in _SOURCE_FILES:
        path = _ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    config_bytes = config_path.read_bytes()
    commit = None
    try:
        command = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(_ROOT),
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 universal_newlines=True, timeout=5, check=False)
        if command.returncode == 0:
            commit = command.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "git_commit": commit,
        "source_sha256": digest.hexdigest(),
        "source_files": list(_SOURCE_FILES),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }


def _vehicle_from_row(row):
    return VehicleSpec(
        id=str(row["id"]),
        start_xy=tuple(row["start_xy"]),
        direction_xy=tuple(row["direction_xy"]),
        distance_to_entry_m=float(row["distance_to_entry_m"]),
        speed_mps=float(row["speed_mps"]),
        zone_length_m=float(row["zone_length_m"]),
        max_accel_mps2=float(row["max_accel_mps2"]),
        max_brake_mps2=float(row["max_brake_mps2"]),
        max_speed_mps=float(row["max_speed_mps"]),
        min_entry_s=float(row["min_entry_s"]),
        max_entry_s=float(row["max_entry_s"]),
    )


def _decode_edges(config, vehicles):
    physics = config["physics"]
    bargaining = config["bargaining"]
    decoded = []
    contracts = []
    for row in config["pair_proposals"]:
        first_id, second_id = row["edge"]
        first, second = vehicles[first_id], vehicles[second_id]
        source = row["source"]
        if "ttc_s" in row:
            predecision_ttc_s = float(row["ttc_s"])
            ttc_source = row.get("ttc_source", "explicit_config_value")
        else:
            first_velocity = tuple(direction * first.speed_mps
                                   for direction in first.direction_xy)
            second_velocity = tuple(direction * second.speed_mps
                                    for direction in second.direction_xy)
            predecision_ttc_s = legacy_closing_ttc(
                first.start_xy, first_velocity,
                second.start_xy, second_velocity)
            ttc_source = "computed_from_initial_state_with_legacy_closing_ttc"
        if source == "rubinstein_share_decoder":
            proposals = [
                PairProposal(str(candidate["id"]),
                             float(candidate["entry_s"][first_id]),
                             float(candidate["entry_s"][second_id]))
                for candidate in row["candidates"]
            ]
            result = decode_crossing_pair(
                first, second,
                (float(row["disagreement_entry_s"][first_id]),
                 float(row["disagreement_entry_s"][second_id])),
                proposals,
                float(bargaining["discount_alpha_s_inv"][first_id]),
                float(bargaining["discount_alpha_s_inv"][second_id]),
                float(bargaining["eta_s"]),
                float(physics["clearance_s"]),
                (float(row.get("tolerance_s", {}).get(first_id, 0.0)),
                 float(row.get("tolerance_s", {}).get(second_id, 0.0))),
                predecision_ttc_s,
                tuple(float(value) for value in
                      bargaining.get("delta_bounds", [0.05, 0.99])),
            )
            contracts.append(result.contract)
            decoded.append({"edge": [first_id, second_id], "source": source,
                            "ttc_source": ttc_source,
                            "result": asdict(result)})
        elif source == "manual_constructed_contract":
            reference = row["reference_entry_s"]
            tolerance = row["tolerance_s"]
            t_first = float(reference[first_id])
            t_second = float(reference[second_id])
            p_first = canonical_approach_plan(first, t_first)
            p_second = canonical_approach_plan(second, t_second)
            clearance = float(physics["clearance_s"])
            if not pair_safe(p_first, p_second, clearance):
                raise ValueError("Manual example contract is not physically safe")
            first_before = p_first.occupation_window_s[1] + clearance <= t_second + 1e-9
            contract = CrossingContract(
                first_id, second_id, t_first, t_second,
                float(tolerance[first_id]), float(tolerance[second_id]),
                first_before, clearance, source,
                str(row.get("id", "manual_" + first_id + "_" + second_id)),
                predecision_ttc_s,
            )
            contracts.append(contract)
            decoded.append({"edge": [first_id, second_id], "source": source,
                            "ttc_source": ttc_source,
                            "result": {"contract": asdict(contract),
                                       "candidates": "not generated by SPE decoder"}})
        else:
            raise ValueError("Unknown pair source: {}".format(source))
    return contracts, decoded


def _sample_selected_plan(vehicles, selected, dt_s, horizon_s):
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("physics.dt_s must be finite and positive")
    if not math.isfinite(horizon_s) or horizon_s <= 0:
        raise ValueError("physics.horizon_s must be finite and positive")
    entry_times = selected.entry_dict()
    plans = {name: canonical_approach_plan(vehicle, entry_times[name])
             for name, vehicle in vehicles.items()}
    if any(plan.occupation_window_s[1] > horizon_s + 1e-9
           for plan in plans.values()):
        raise ValueError("The recorded horizon ends before a selected zone exit")
    step_count = int(math.floor(horizon_s / dt_s))
    times = [step * dt_s for step in range(step_count + 1)]
    if times[-1] < horizon_s - 1e-9:
        times.append(horizon_s)
    return {
        "status": "kinematic_reference_only_not_collision_checked",
        "dt_s": dt_s,
        "horizon_s": horizon_s,
        "vehicles": {
            name: {
                "entry_time_s": plan.entry_time_s,
                "occupation_window_s": plan.occupation_window_s,
                "profile_kind": plan.profile_kind,
                "upstream_wait_position_xy": plan.upstream_wait_position_xy,
                "samples": [asdict(plan.sample(t)) for t in times],
            }
            for name, plan in plans.items()
        },
    }


def run_config(config_path, output_root, allow_constructed_example=False):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Expected schema_version=1")
    if config.get("mode") != "constructed_example" or not allow_constructed_example:
        raise ValueError("This runner accepts only explicit constructed examples; "
                         "pass --allow-constructed-example")
    parsed_vehicles = [_vehicle_from_row(row) for row in config["vehicles"]]
    vehicles = {vehicle.id: vehicle for vehicle in parsed_vehicles}
    if len(vehicles) != len(config["vehicles"]):
        raise ValueError("Vehicle IDs must be unique")
    if len(vehicles) < 3:
        raise ValueError("The example needs at least three cooperating CAVs")
    before = time.perf_counter()
    contracts, decoded = _decode_edges(config, vehicles)
    candidates = {name: [float(t) for t in times]
                  for name, times in config["three_car_candidates"].items()}
    bargaining = config["bargaining"]
    snapshot = compose_crossing_snapshot(
        vehicles, contracts, candidates,
        float(bargaining["welfare_epsilon_s"]),
        float(bargaining["t_ref_s"]),
        max_joint_combinations=int(bargaining.get("max_joint_combinations", 100000)),
    )
    trajectory = _sample_selected_plan(
        vehicles, snapshot.global_plan, float(config["physics"]["dt_s"]),
        float(config["physics"]["horizon_s"]))
    elapsed_ms = (time.perf_counter() - before) * 1000.0
    if not any(row["source"] == "rubinstein_share_decoder" for row in decoded):
        raise ValueError("At least one edge must come from the SPE-share decoder")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    resolved_output_root = Path(output_root).resolve()
    if _ROOT.as_posix().startswith("/data/hhr/"):
        authorized_root = Path("/data/hhr").resolve()
        if (resolved_output_root != authorized_root and
                authorized_root not in resolved_output_root.parents):
            raise ValueError("On this server, output must remain under /data/hhr")
    output_dir = resolved_output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "config_snapshot.json", config)
    _write_json(output_dir / "pair_contracts.json", decoded)
    _write_json(output_dir / "cooperation_snapshot.json", asdict(snapshot))
    _write_json(output_dir / "selected_trajectory_samples.json", trajectory)
    manifest = {
        "run_id": run_id,
        "status": "constructed_example_only",
        "scenario": "deterministic_crossing_three_CAV_two_conflict_edges",
        "scope": "single_decision_snapshot_exact_over_supplied_finite_candidates",
        "experimental_collision_rate": None,
        "closed_loop_trial_count": 0,
        "constructed_snapshot_count": 1,
        "train_seed": config.get("seeds", {}).get("train_seed"),
        "trial_seed": config.get("seeds", {}).get("trial_seed"),
        "checkpoint_path": None,
        "elapsed_ms_single_snapshot": elapsed_ms,
        "elapsed_ms_is_realtime_evidence": False,
        "version": _version_record(config_path),
        "limitations": [
            "The BC contract is manually constructed, not generated by the original code.",
            "Only the crossing time-window decoder is implemented.",
            "The paired closed-loop, trained offer adapter, and other scenes are pending.",
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir, snapshot, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path,
                        default=_ROOT / "Result" / "multicar_cooperation")
    parser.add_argument("--allow-constructed-example", action="store_true")
    args = parser.parse_args(argv)
    output_dir, snapshot, _ = run_config(
        args.config, args.output_root, args.allow_constructed_example)
    print(json.dumps({
        "result_dir": str(output_dir),
        "status": "constructed_example_only",
        "safe_candidate_count": snapshot.safe_plan_count,
        "exactly_composable_count": snapshot.exactly_composable_count,
        "global_broken_contracts": snapshot.global_plan.broken_contracts,
        "baseline_broken_contracts": snapshot.sequential_plan.broken_contracts,
        "empirical_collision_rate": None,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
