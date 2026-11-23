"""Compare E4 naive speed-jump adapter vs priority-separated bounded-kinematic bridge.

This script addresses the Section 5.2 (E4) and 2026-11-20 Report Section 7
follow-up ("design and validate a lower-distortion physical bridge for legacy
4D offers"):
1. Loads the archived RubinsteinBargaining model and best trained checkpoint
   (train_3r_ttc_learned_a0.3_seed608_20261031_144750/checkpoints/best_model.pth,
   training seed 608) in read-only mode on deterministic pair BEV inputs.
2. Evaluates 10 three-car crossing trials (seeds 101-110, 20 conflict edges)
   under three frozen adapter modes:
   - Mode A (E4 strict naive quotient): distance / jump_speed without priority
     separation or projection -> 0/20 feasible.
   - Mode B (E4 naive quotient + post-hoc safety projection): 20/20 projected
     with mean L1 modification 2.530 s.
   - Mode C (New priority-separated bounded-kinematic bridge): orders the pair
     by offer priority (speed_factor - yield_will, delta_v, -free_flow_time),
     computes bounded reachable entry time for the leader and conflict-clearing
     entry time for the yielder, and snaps to the shared candidate grid.
3. Compares four-cell factorial outcomes (edge generator x global/sequential
   aggregator) across all 10 matched trials.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import uuid

import torch

from .archived_audit import _archive_module, _default_archive_source, _weights
from .compose import compose_crossing_snapshot
from .crossing_closed_loop import _candidate_times, _edge_contract
from .legacy_adapter import (
    adapt_learned_offers_to_crossing_contract,
    adapt_offers_with_priority_separation,
)
from .run_adapter_audit import _pair_observation
from .run_paired_closed_loop import _trial_initial_conditions
from .run_scene_closed_loop import _write_json


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (ROOT / "Result" / "ttc_multicar_20261122" /
                  "E34_legacy_offer_priority_bridge_audit")
DEFAULT_CONFIG = ROOT / "config" / "multicar_cooperation_example.json"
DEFAULT_CKPT = (
    ROOT / "Result" / "train_3r_ttc_learned_a0.3_seed608_20261031_144750" /
    "checkpoints" / "best_model.pth")
SEEDS = tuple(range(101, 111))


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _adapt_priority_learned(model, offers, first, second, dt_s, clearance_s,
                            tolerance_s, candidate_entry_times):
    with torch.no_grad():
        offsets = model.offer_to_waypoint(offers) * model.waypoint_offset_scale
    return adapt_offers_with_priority_separation(
        first, second,
        offsets[0, :2].detach().cpu().tolist(),
        offsets[1, :2].detach().cpu().tolist(),
        offers.detach().cpu().tolist(),
        dt_s, clearance_s, tolerance_s,
        candidate_entry_times=candidate_entry_times)


def run(output_root=DEFAULT_OUTPUT, config_path=DEFAULT_CONFIG,
        archive_source=None, checkpoint_path=DEFAULT_CKPT):
    output_root = Path(output_root).resolve()
    config_path = Path(config_path).resolve()
    archive_source = Path(archive_source or _default_archive_source()).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    module = _archive_module(archive_source)
    torch.manual_seed(0)
    model = module.RubinsteinBargaining({
        "state_dim": 64, "offer_dim": 4, "hidden_dim": 128,
        "max_rounds": 3, "discount_alpha": 0.3,
        "delta_min": 0.05, "delta_max": 0.99,
        "waypoint_offset_scale": 2.0,
    })
    checkpoint_info = _weights(model, checkpoint_path)
    model.to("cpu").eval()

    dt_s = float(config["physics"]["dt_s"])
    clearance_s = float(config["physics"]["clearance_s"])
    horizon_s = float(config["physics"]["horizon_s"])
    eta_s = float(config["bargaining"]["eta_s"])
    epsilon_w = float(config["bargaining"]["welfare_epsilon_s"])
    edges = [tuple(row["edge"]) for row in config["pair_proposals"]]

    from common.deterministic_bev import deterministic_bev

    modes_summary = {
        "naive_strict": {"successes": 0, "failures": 0, "l1_shifts": [], "wp_errors": []},
        "naive_safety_projected": {"successes": 0, "failures": 0, "l1_shifts": [], "wp_errors": []},
        "priority_bounded_bridge": {"successes": 0, "failures": 0, "l1_shifts": [], "wp_errors": []},
    }
    factorial_cells = {
        "new_spe_decoder__global": [],
        "new_spe_decoder__sequential": [],
        "naive_projected__global": [],
        "naive_projected__sequential": [],
        "priority_bridge__global": [],
        "priority_bridge__sequential": [],
    }
    trial_records = []

    for seed in SEEDS:
        vehicles, perturbations, _ = _trial_initial_conditions(
            config, seed, 0.3, 0.2, 0.0)
        candidate_times = {
            name: _candidate_times(vehicle, horizon_s, 0.4, 4.0)
            for name, vehicle in vehicles.items()
        }
        new_contracts = []
        naive_proj_contracts = []
        priority_contracts = []
        edge_logs = []

        for first_id, second_id in edges:
            first, second = vehicles[first_id], vehicles[second_id]
            c_pair = (candidate_times[first_id], candidate_times[second_id])
            new_c = _edge_contract(
                first, second, c_pair[0], c_pair[1],
                float(config["bargaining"]["discount_alpha_s_inv"][first_id]),
                float(config["bargaining"]["discount_alpha_s_inv"][second_id]),
                eta_s, clearance_s, 0.1)
            new_contracts.append(new_c)

            state = _pair_observation(first, second, dt_s, "cpu")
            features = deterministic_bev(state, device="cpu")
            with torch.no_grad():
                out = model(features, state["waypoints"], state["positions"],
                            state["velocities"], state["headings"], state["record_len"])
            offers = out["bargaining_offers"]

            out_strict = adapt_learned_offers_to_crossing_contract(
                model, offers, first, second, dt_s, clearance_s, 0.1,
                candidate_entry_times=c_pair, allow_safety_projection=False)
            out_proj = adapt_learned_offers_to_crossing_contract(
                model, offers, first, second, dt_s, clearance_s, 0.1,
                candidate_entry_times=c_pair, allow_safety_projection=True)
            out_prio = _adapt_priority_learned(
                model, offers, first, second, dt_s, clearance_s, 0.1, c_pair)

            for key, outcome in (("naive_strict", out_strict),
                                 ("naive_safety_projected", out_proj),
                                 ("priority_bounded_bridge", out_prio)):
                if outcome.success and outcome.raw_failure_reason is None:
                    modes_summary[key]["successes"] += 1
                elif outcome.success:
                    modes_summary[key]["successes"] += 1
                else:
                    modes_summary[key]["failures"] += 1
                if outcome.contract is not None:
                    modes_summary[key]["l1_shifts"].append(outcome.safety_projection_l1_s)
                    modes_summary[key]["wp_errors"].extend(
                        e for e in outcome.first_waypoint_projection_error_m if e is not None)

            naive_proj_contracts.append(out_proj.contract)
            priority_contracts.append(out_prio.contract)
            edge_logs.append({
                "edge": [first_id, second_id],
                "naive_strict": asdict(out_strict),
                "naive_projected": asdict(out_proj),
                "priority_bridge": asdict(out_prio),
            })

        for prefix, c_list in (("new_spe_decoder", new_contracts),
                               ("naive_projected", naive_proj_contracts),
                               ("priority_bridge", priority_contracts)):
            res = compose_crossing_snapshot(
                vehicles, c_list, candidate_times, epsilon_w, 1.0)
            factorial_cells[prefix + "__global"].append(asdict(res.global_plan))
            factorial_cells[prefix + "__sequential"].append(asdict(res.sequential_plan))

        trial_records.append({
            "trial_seed": seed,
            "perturbations": perturbations,
            "edges": edge_logs,
        })

    adapter_comparison = {}
    for key, stats in modes_summary.items():
        shifts = stats["l1_shifts"]
        wpe = stats["wp_errors"]
        adapter_comparison[key] = {
            "edge_attempts": 20,
            "feasible_contracts_produced": stats["successes"],
            "failure_count": stats["failures"],
            "mean_l1_time_modification_s": (sum(shifts) / len(shifts) if shifts else None),
            "max_l1_time_modification_s": (max(shifts) if shifts else None),
            "mean_first_waypoint_error_m": (sum(wpe) / len(wpe) if wpe else None),
        }

    cell_summaries = {}
    for cell_name, plans in factorial_cells.items():
        cell_summaries[cell_name] = {
            "available_trials": len(plans),
            "mean_broken_contracts": sum(p["broken_contracts"] for p in plans) / len(plans),
            "mean_edge_fidelity": sum(p["edge_fidelity"] for p in plans) / len(plans),
            "mean_welfare_s": sum(p["welfare_s"] for p in plans) / len(plans),
        }

    run_id = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" +
              uuid.uuid4().hex[:8])
    result_dir = output_root / run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "experiment": "E34_legacy_offer_priority_bridge_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "10 three-car crossing trials (20 edges); archived SL best checkpoint seed 608; deterministic pair BEV",
        "trial_seeds": list(SEEDS),
        "checkpoint_info": checkpoint_info,
        "source_hashes": {
            "archive_sha256": _sha256_file(archive_source),
            "checkpoint_sha256": checkpoint_info["sha256"],
            "legacy_adapter_sha256": _sha256_file(
                Path(__file__).with_name("legacy_adapter.py")),
            "runner_sha256": _sha256_file(__file__),
        },
        "adapter_mode_comparison": adapter_comparison,
        "four_cell_factorial_summary": cell_summaries,
        "python": sys.version,
        "platform": platform.platform(),
    }
    _write_json(result_dir / "summary.json", result)
    _write_json(result_dir / "trials.json", trial_records)
    return result_dir, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)
    result_dir, result = run(args.output_root)
    if args.compact:
        result = {
            "output_dir": str(result_dir),
            "source_hashes": result["source_hashes"],
            "adapter_mode_comparison": result["adapter_mode_comparison"],
            "four_cell_factorial_summary": result["four_cell_factorial_summary"],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
