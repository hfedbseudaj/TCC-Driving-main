"""E4 factorial audit: archived learned offers through a fixed new adapter.

Four cells (old/new edge generator x adapted pairwise/global aggregation) are
reported on matched three-car crossing snapshots only when both old edges
adapt successfully.  Failed adapters are counted explicitly, never replaced
with hand-crafted contracts or silently discarded.
"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import uuid

import torch

from .archived_audit import _archive_module, _default_archive_source, _weights
from .compose import compose_crossing_snapshot
from .crossing_closed_loop import _candidate_times, _edge_contract
from .crossing import CrossingContract
from .legacy_adapter import adapt_learned_offers_to_crossing_contract
from .run_paired_closed_loop import _trial_initial_conditions
from .run_snapshot import _write_json
from .physics import canonical_approach_plan, pair_safe


_ROOT = Path(__file__).resolve().parent.parent


def _pair_observation(first, second, dt_s, device):
    vehicles = (first, second)
    positions = torch.tensor([vehicle.start_xy for vehicle in vehicles],
                             dtype=torch.float32, device=device)
    velocities = torch.tensor([
        [coordinate * vehicle.speed_mps for coordinate in vehicle.direction_xy]
        for vehicle in vehicles], dtype=torch.float32, device=device)
    headings = torch.tensor([
        math.atan2(vehicle.direction_xy[1], vehicle.direction_xy[0])
        for vehicle in vehicles], dtype=torch.float32, device=device)
    waypoints = torch.tensor([
        [[vehicle.direction_xy[0] * vehicle.speed_mps * dt_s * (index + 1),
          vehicle.direction_xy[1] * vehicle.speed_mps * dt_s * (index + 1)]
         for index in range(10)] for vehicle in vehicles],
        dtype=torch.float32, device=device)
    return {
        "positions": positions, "velocities": velocities,
        "headings": headings, "waypoints": waypoints,
        "record_len": torch.tensor([2], dtype=torch.long, device=device),
    }


def _learned_edge(model, first, second, trial_seed, edge_index,
                  dt_s, clearance_s, tolerance_s, feature_mode, device,
                  candidate_entry_times, allow_safety_projection):
    state = _pair_observation(first, second, dt_s, device)
    if feature_mode == "original_random_bev":
        generator = torch.Generator(device=device)
        generator.manual_seed(int(trial_seed) * 100 + edge_index)
        features = torch.randn((2, 256, 48, 48), device=device,
                               generator=generator)
    elif feature_mode == "deterministic_pair_bev":
        from common.deterministic_bev import deterministic_bev
        features = deterministic_bev(state, device=device)
    else:
        raise ValueError("Unknown feature mode")
    with torch.no_grad():
        output = model(features, state["waypoints"], state["positions"],
                       state["velocities"], state["headings"], state["record_len"])
    negotiated = output["bargaining_offers"]
    adaptation = adapt_learned_offers_to_crossing_contract(
        model, negotiated, first, second, dt_s, clearance_s, tolerance_s,
        candidate_entry_times=candidate_entry_times,
        allow_safety_projection=allow_safety_projection)
    return {
        "edge": [first.id, second.id],
        "negotiated_offers": negotiated.detach().cpu().tolist(),
        "accepted": output["acceptance_mask"].detach().cpu().tolist(),
        "adaptation": asdict(adaptation),
    }, adaptation.contract


def _safe_fallback_contract(first, second, candidate_entry_times,
                            clearance_s, edge_index):
    """Deterministic no-bargaining fallback used only after adapter failure.

    It picks the safe, reachable, non-overlapping pair with minimum total
    entry delay.  This is reported as a hybrid pipeline fallback, never as a
    successful conversion of the old learned offer.
    """
    rows = []
    for first_time, second_time in itertools.product(*candidate_entry_times):
        try:
            first_plan = canonical_approach_plan(first, first_time)
            second_plan = canonical_approach_plan(second, second_time)
        except ValueError:
            continue
        if not pair_safe(first_plan, second_plan, clearance_s):
            continue
        first_before = (first_plan.occupation_window_s[1] + clearance_s <=
                        second_time + 1e-9)
        second_before = (second_plan.occupation_window_s[1] + clearance_s <=
                         first_time + 1e-9)
        if not (first_before or second_before):
            continue
        free_first = first.distance_to_entry_m / first.speed_mps
        free_second = second.distance_to_entry_m / second.speed_mps
        total_delay = max(0.0, first_time - free_first) + max(
            0.0, second_time - free_second)
        rows.append((total_delay, first_time, second_time, first_before))
    if not rows:
        return None
    _, first_time, second_time, first_before = min(rows)
    return CrossingContract(
        first_id=first.id,
        second_id=second.id,
        first_reference_s=first_time,
        second_reference_s=second_time,
        first_tolerance_s=0.0,
        second_tolerance_s=0.0,
        first_before_second=first_before,
        clearance_s=clearance_s,
        source="adapter_failure_safe_fallback",
        reference_id="minimum_delay_safe_nonoverlap_pair_{}".format(edge_index),
    )


def run_adapter_audit(config_path, archive_source, output_root,
                      trial_seeds, model_seed=0, checkpoint_path=None,
                      feature_mode="original_random_bev", device="cpu",
                      allow_constructed_example=False,
                      allow_safety_projection=False):
    config_path = Path(config_path).resolve()
    archive_source = Path(archive_source).resolve()
    output_root = Path(output_root).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("mode") != "constructed_example":
        raise ValueError("E4 currently requires the constructed crossing schema")
    if not allow_constructed_example:
        raise ValueError("Pass --allow-constructed-example")
    if _ROOT.as_posix().startswith("/data/hhr/"):
        designated = Path("/data/hhr").resolve()
        if designated != output_root and designated not in output_root.parents:
            raise ValueError("Server output must stay under /data/hhr")
    module = _archive_module(archive_source)
    torch.manual_seed(int(model_seed))
    model = module.RubinsteinBargaining({
        "state_dim": 64, "offer_dim": 4, "hidden_dim": 128,
        "max_rounds": 3, "discount_alpha": 0.3,
        "delta_min": 0.05, "delta_max": 0.99,
        "waypoint_offset_scale": 2.0,
    })
    checkpoint_info = _weights(model, checkpoint_path)
    model.to(device).eval()
    dt_s = float(config["physics"]["dt_s"])
    clearance_s = float(config["physics"]["clearance_s"])
    horizon_s = float(config["physics"]["horizon_s"])
    eta_s = float(config["bargaining"]["eta_s"])
    epsilon_w = float(config["bargaining"]["welfare_epsilon_s"])
    edges = [tuple(row["edge"]) for row in config["pair_proposals"]]
    trials = []
    for seed in trial_seeds:
        vehicles, perturbations, _ = _trial_initial_conditions(
            config, seed, 0.3, 0.2, 0.0)
        candidate_times = {name: _candidate_times(vehicle, horizon_s, 0.4, 4.0)
                           for name, vehicle in vehicles.items()}
        new_contracts = []
        old_contracts = []
        adaptations = []
        for edge_index, (first_id, second_id) in enumerate(edges):
            first, second = vehicles[first_id], vehicles[second_id]
            new_contracts.append(_edge_contract(
                first, second, candidate_times[first_id],
                candidate_times[second_id],
                float(config["bargaining"]["discount_alpha_s_inv"][first_id]),
                float(config["bargaining"]["discount_alpha_s_inv"][second_id]),
                eta_s, clearance_s, 0.1))
            record, contract = _learned_edge(
                model, first, second, seed, edge_index,
                dt_s, clearance_s, 0.1, feature_mode, device,
                (candidate_times[first_id], candidate_times[second_id]),
                allow_safety_projection)
            adaptations.append(record)
            if contract is not None:
                old_contracts.append(contract)
                adaptations[-1]["fallback"] = None
            else:
                fallback_contract = _safe_fallback_contract(
                    first, second,
                    (candidate_times[first_id], candidate_times[second_id]),
                    clearance_s, edge_index)
                adaptations[-1]["fallback"] = {
                    "used": fallback_contract is not None,
                    "rule": "minimum_total_delay_safe_reachable_nonoverlap_pair",
                    "contract": (asdict(fallback_contract)
                                 if fallback_contract is not None else None),
                }
                if fallback_contract is not None:
                    old_contracts.append(fallback_contract)
        cells = {}
        for source, contracts in (("new_spe_decoder", new_contracts),
                                  ("old_learned_offer_adapter_with_safe_fallback",
                                   old_contracts)):
            if len(contracts) != len(edges):
                for aggregator in ("global", "sequential_pairwise_reconstruction"):
                    cells[source + "__" + aggregator] = {
                        "available": False, "reason": "at_least_one_adapter_failed"}
                continue
            try:
                result = compose_crossing_snapshot(
                    vehicles, contracts, candidate_times, epsilon_w, 1.0)
            except ValueError as error:
                for aggregator in ("global", "sequential_pairwise_reconstruction"):
                    cells[source + "__" + aggregator] = {
                        "available": False,
                        "reason": "no_joint_feasible_candidate: " + str(error)}
                continue
            for aggregator, plan in (("global", result.global_plan),
                                     ("sequential_pairwise_reconstruction",
                                      result.sequential_plan)):
                cells[source + "__" + aggregator] = {
                    "available": True,
                    "plan": asdict(plan),
                    "exactly_composable_count": result.exactly_composable_count,
                    "safe_candidate_count": result.safe_plan_count,
                }
        trials.append({
            "trial_seed": int(seed),
            "perturbations": perturbations,
            "candidate_entry_times_s": candidate_times,
            "new_edge_contracts": [asdict(contract) for contract in new_contracts],
            "old_edge_adaptations": adaptations,
            "cells": cells,
        })
    edge_attempts = len(trials) * len(edges)
    successes = sum(row["adaptation"]["success"]
                    for trial in trials for row in trial["old_edge_adaptations"])
    projected = [row["adaptation"] for trial in trials
                 for row in trial["old_edge_adaptations"]
                 if row["adaptation"]["reason"] == "safety_projected"]
    raw_failures = [row["adaptation"]["raw_failure_reason"]
                    for trial in trials for row in trial["old_edge_adaptations"]]
    fallback_count = sum(bool((row.get("fallback") or {}).get("used"))
                         for trial in trials for row in trial["old_edge_adaptations"])
    summary = {
        "trial_count": len(trials),
        "edge_adaptation_attempts": edge_attempts,
        "edge_adaptation_successes": successes,
        "edge_adaptation_failure_rate": 1.0 - successes / edge_attempts,
        "raw_edge_failure_count": sum(reason is not None for reason in raw_failures),
        "safety_projection_count": len(projected),
        "safe_fallback_count": fallback_count,
        "safe_fallback_rate": fallback_count / edge_attempts,
        "mean_safety_projection_l1_s": (
            sum(row["safety_projection_l1_s"] for row in projected) / len(projected)
            if projected else None),
        "matched_four_cell_trial_count": sum(
            all(cell["available"] for cell in trial["cells"].values())
            for trial in trials),
        "adapter_failure_reasons": {
            reason: raw_failures.count(reason)
            for reason in ("dynamically_unreachable_entry", "pair_zone_occupation_unsafe")
        },
    }
    cell_names = tuple(sorted({name for trial in trials for name in trial["cells"]}))
    summary["cells"] = {}
    for name in cell_names:
        available = [trial["cells"][name]["plan"] for trial in trials
                     if trial["cells"][name]["available"]]
        summary["cells"][name] = {
            "available_trials": len(available),
            "mean_broken_contracts": (
                sum(plan["broken_contracts"] for plan in available) / len(available)
                if available else None),
            "mean_edge_fidelity": (
                sum(plan["edge_fidelity"] for plan in available) / len(available)
                if available else None),
            "mean_welfare_s": (
                sum(plan["welfare_s"] for plan in available) / len(available)
                if available else None),
        }
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "trials.json", trials)
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "manifest.json", {
        "experiment": "E4_crossing_factorial_adapter_audit",
        "adapter_definition": "learned_4D_offer_to_waypoint_first_offset_to_archived_speed_command_to_reachable_entry_time",
        "adapter_failure_handling": "record_each_adapter_failure_then_use_a_deterministic_safe_non_bargaining_pair_contract_if_one_exists; fallback edges are never counted as adapter successes",
        "safe_fallback_rule": "minimum total delay among safe reachable pairs with nonoverlapping conflict-zone occupation, then fixed entry-time order",
        "fallback_edge_count": fallback_count,
        "allow_safety_projection": allow_safety_projection,
        "safety_projection_rule": "when_raw_offer_is_infeasible_choose_shared_safe_candidate_minimizing_L1_entry_time_change_then_total_entry_time_then_fixed_pair_order",
        "inference_boundary": "old_offer_pipeline_with_fixed_new_adapter_and_reported_safe_fallback_vs_new_decoder; fallback-heavy cells are hybrid pipeline results, not a pure old-offer comparison",
        "archive_source": str(archive_source),
        "archive_sha256": hashlib.sha256(archive_source.read_bytes()).hexdigest(),
        "checkpoint_path": str(Path(checkpoint_path).resolve()) if checkpoint_path else None,
        "checkpoint_info": checkpoint_info,
        "checkpoint_sha256": checkpoint_info["sha256"] if checkpoint_info else None,
        "training_seed": checkpoint_info["training_seed"] if checkpoint_info else None,
        "model_status": "untrained_random_initialization" if checkpoint_path is None else "strict_checkpoint_load",
        "model_seed": model_seed, "trial_seeds": list(trial_seeds),
        "feature_mode": feature_mode, "device": device,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(b"".join(
            (_ROOT / relative).read_bytes() for relative in (
                "cooperation/legacy_adapter.py", "cooperation/run_adapter_audit.py",
                "cooperation/physics.py", "cooperation/crossing.py",
                "cooperation/compose.py", "cooperation/crossing_closed_loop.py"
            ))).hexdigest(),
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "scope_note": "The archived model has no K_e interface; this adapter is new and may dominate the old-source cell outcomes.",
    })
    return output_dir, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=_ROOT / "config" / "multicar_cooperation_example.json")
    parser.add_argument("--archive-source", type=Path,
                        default=_default_archive_source())
    parser.add_argument("--output-root", type=Path,
                        default=_ROOT / "Result" / "multicar_adapter_audit")
    parser.add_argument("--trial-seeds", type=int, nargs="+", default=[101, 102, 103])
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--feature-mode", choices=("original_random_bev", "deterministic_pair_bev"),
                        default="original_random_bev")
    parser.add_argument("--allow-constructed-example", action="store_true")
    parser.add_argument("--allow-safety-projection", action="store_true")
    args = parser.parse_args(argv)
    if args.archive_source is None:
        parser.error("Archived source not found; pass --archive-source")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    output_dir, summary = run_adapter_audit(
        args.config, args.archive_source, args.output_root,
        args.trial_seeds, args.model_seed, args.checkpoint,
        args.feature_mode, args.device, args.allow_constructed_example,
        args.allow_safety_projection)
    print(json.dumps({"result_dir": str(output_dir), "summary": summary},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
