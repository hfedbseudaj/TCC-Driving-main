"""Paired TTC mechanism ablation in the constructed kinematic closed loop.

Each policy sees the same perturbed initial state and sampled acceleration
noise. A frozen initial-decision snapshot separates changes in Rubinstein
shares, decoded edge contracts, and selected fleet actions. Closed-loop trials
then measure the cumulative outcome. These are not author or CARLA results.
"""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import uuid

from .scene_closed_loop import (
    DEFAULT_POLICY, MODEL, _choose, _trial_initial, normalize_policy,
    run_episode,
)


ROOT = Path(__file__).resolve().parent.parent
SCENES = ("crossing", "chase", "merge", "head_on")
MODES = ("ttc", "constant", "equal_share")


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def _source_hashes():
    hashes = {}
    for relative in ("cooperation/physics.py", "cooperation/scene_closed_loop.py",
                     "cooperation/run_mechanism_ablation.py"):
        hashes[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    return hashes


def _snapshot(scene, count, seed, position_std_m, speed_std_mps, policy):
    cars, edges, _ = _trial_initial(scene, count, seed, position_std_m,
                                     speed_std_mps)
    decisions = {}
    for mode in MODES:
        mode_policy = normalize_policy(dict(policy, discount_mode=mode))
        controls, info = _choose(scene, cars, edges, "global", mode_policy)
        contracts = {}
        for record in info["contract_records"]:
            edge = "--".join(record["edge"])
            contracts[edge] = {
                "ttc_s": record["predecision_ttc_s"],
            "ttc_source": record["ttc_source"],
            "legacy_radial_ttc_s": record["legacy_radial_ttc_s"],
                "active": record["ttc_active"],
                "applied": record["ttc_applied"],
                "share_first": record["reference_share_first"],
                "realized_share_first": record["realized_share_first"],
                "share_error": record["share_error"],
                "actions": record["K_e"]["reference_actions"],
            }
        decisions[mode] = {
            "selected_actions": {name: action.name
                                 for name, action in controls.items()},
            "contracts": contracts,
            "contract_count": info["contract_count"],
            "ttc_active_contract_count": info["ttc_active_contract_count"],
            "ttc_applied_contract_count": info["ttc_applied_contract_count"],
            "safe_candidate_count": info["safe_candidate_count"],
            "ir_candidate_count": info["ir_candidate_count"],
            "safe_disagreement_available": info["safe_disagreement_available"],
            "individual_rational": info["individual_rational"],
            "fallback": info["fallback"],
            "broken_contracts": info["broken_contracts"],
            "selected_welfare_m": info["selected_welfare_m"],
        }
    return {
        "scene": scene, "n_cars": count, "trial_seed": seed,
        "initial_vehicles": {
            name: {"x_m": car.x, "y_m": car.y, "speed_mps": car.speed,
                   "axis": car.axis}
            for name, car in cars.items()},
        "edges": [list(edge) for edge in edges],
        "decisions": decisions,
    }


def _mean_or_none(values):
    return statistics.mean(values) if values else None


def _pair_summary(snapshots, episodes, reference):
    changed_shares = 0
    changed_contracts = 0
    comparable_contracts = 0
    changed_actions = 0
    changed_completed = 0
    changed_collisions = 0
    changed_near = 0
    delay_differences = []
    for snapshot in snapshots:
        first = snapshot["decisions"]["ttc"]
        second = snapshot["decisions"][reference]
        changed_actions += first["selected_actions"] != second["selected_actions"]
        for edge in sorted(set(first["contracts"]) | set(second["contracts"])):
            a = first["contracts"].get(edge)
            b = second["contracts"].get(edge)
            if a is not None and b is not None:
                comparable_contracts += 1
                changed_shares += abs(a["share_first"] - b["share_first"]) > 1e-10
                changed_contracts += a["actions"] != b["actions"]
            elif a is not b:
                changed_contracts += 1
    for by_mode in episodes:
        first = by_mode["ttc"]
        second = by_mode[reference]
        changed_completed += first["completed"] != second["completed"]
        changed_collisions += first["collision"] != second["collision"]
        changed_near += first["near_collision"] != second["near_collision"]
        if (first["mean_completed_delay_s"] is not None and
                second["mean_completed_delay_s"] is not None):
            delay_differences.append(first["mean_completed_delay_s"] -
                                     second["mean_completed_delay_s"])
    return {
        "paired_trial_count": len(snapshots),
        "comparable_initial_edge_contracts": comparable_contracts,
        "initial_share_changed_edges": changed_shares,
        "initial_decoded_contract_changed_edges": changed_contracts,
        "initial_selected_action_changed_trials": changed_actions,
        "closed_loop_completion_changed_trials": changed_completed,
        "closed_loop_collision_changed_trials": changed_collisions,
        "closed_loop_near_collision_changed_trials": changed_near,
        "mean_paired_completed_delay_ttc_minus_reference_s":
            _mean_or_none(delay_differences),
        "paired_completed_delay_count": len(delay_differences),
    }


def _mode_summary(episodes):
    delays = [episode["mean_completed_delay_s"] for episode in episodes
              if episode["mean_completed_delay_s"] is not None]
    times = [value for episode in episodes
             for value in episode["solver_times_ms"]]
    return {
        "trial_count": len(episodes),
        "completion_count": sum(row["completed"] for row in episodes),
        "collision_count": sum(row["collision"] for row in episodes),
        "near_collision_count": sum(row["near_collision"] for row in episodes),
        "mean_completed_delay_s": _mean_or_none(delays),
        "completed_delay_count": len(delays),
        "fallback_count": sum(row["fallback_count"] for row in episodes),
        "ir_violation_ticks": sum(row["ir_violation_ticks"] for row in episodes),
        "safe_disagreement_unavailable_ticks": sum(
            row["safe_disagreement_unavailable_ticks"] for row in episodes),
        "ttc_active_contracts": sum(row["ttc_active_contracts"]
                                    for row in episodes),
        "ttc_applied_contracts": sum(row["ttc_applied_contracts"]
                                     for row in episodes),
        "total_contracts": sum(row["total_contracts"] for row in episodes),
        "mean_decoded_share_error": (
            sum(row["mean_decoded_share_error"] * row["total_contracts"]
                for row in episodes) /
            max(1, sum(row["total_contracts"] for row in episodes))),
        "next_step_uncertified_fallback_ticks": sum(
            row["next_step_uncertified_fallback_ticks"] for row in episodes),
        "next_step_full_margin_shortfall_ticks": sum(
            row["next_step_full_margin_shortfall_ticks"] for row in episodes),
        "next_step_collision_uncertified_fallback_ticks": sum(
            row["next_step_collision_uncertified_fallback_ticks"]
            for row in episodes),
        "mean_solver_ms": _mean_or_none(times),
        "max_solver_ms": max(times) if times else None,
        "control_period_exceeded_ticks": sum(
            value > MODEL["dt_s"] * 1000.0 for value in times),
    }


def run_ablation(output_root, scenes, counts, seeds, position_std_m=0.3,
                 speed_std_mps=0.2, policy=None):
    if not scenes or set(scenes) - set(SCENES):
        raise ValueError("Unknown or empty scenes")
    if not counts or set(counts) - {2, 3, 4}:
        raise ValueError("Vehicle counts must be 2, 3 or 4")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Trial seeds must be nonempty and unique")
    if position_std_m < 0 or speed_std_mps < 0:
        raise ValueError("Perturbation scales must be nonnegative")
    policy = normalize_policy(policy)
    output_root = Path(output_root).resolve()
    if ROOT.as_posix().startswith("/data/hhr/"):
        designated = Path("/data/hhr").resolve()
        if designated != output_root and designated not in output_root.parents:
            raise ValueError("Server output must stay under /data/hhr")
    snapshots = []
    raw_episodes = []
    summary = {}
    for scene in scenes:
        summary[scene] = {}
        for count in counts:
            group_snapshots = []
            group_episodes = []
            for seed in seeds:
                frozen = _snapshot(scene, count, seed, position_std_m,
                                   speed_std_mps, policy)
                by_mode = {}
                for mode in MODES:
                    mode_policy = normalize_policy(dict(policy,
                                                        discount_mode=mode))
                    episode = run_episode(scene, count, seed, "global",
                                          position_std_m, speed_std_mps,
                                          mode_policy)
                    by_mode[mode] = episode
                    raw_episodes.append(episode)
                snapshots.append(frozen)
                group_snapshots.append(frozen)
                group_episodes.append(by_mode)
            summary[scene][str(count)] = {
                "modes": {mode: _mode_summary([
                    group[mode] for group in group_episodes])
                    for mode in MODES},
                "ttc_vs_constant": _pair_summary(
                    group_snapshots, group_episodes, "constant"),
                "ttc_vs_equal_share": _pair_summary(
                    group_snapshots, group_episodes, "equal_share"),
            }
    run_id = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") +
              "_" + uuid.uuid4().hex[:8])
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "initial_snapshots.json", snapshots)
    with gzip.open(str(output_dir / "episodes.json.gz"), "wt",
                   encoding="utf-8") as stream:
        json.dump(raw_episodes, stream, ensure_ascii=False, allow_nan=False)
    manifest = {
        "experiment": "TTC_discount_paired_mechanism_ablation",
        "status": "constructed_kinematic_not_CARLA_or_author_results",
        "scenes": list(scenes), "vehicle_counts": list(counts),
        "trial_seeds": list(seeds),
        "training_seed": None, "checkpoint_path": None,
        "compared_modes": list(MODES),
        "controller": "global",
        "base_policy": policy,
        "position_perturbation_std_m": position_std_m,
        "speed_perturbation_std_mps": speed_std_mps,
        "acceleration_noise_std_mps2": 0.1,
        "paired_initial_state_and_noise": True,
        "snapshot_definition": "One shared perturbed initial state; only discount mode changes, with the same safety filter and joint selector.",
        "closed_loop_definition": "Each policy recomputes contracts and actions every 0.2 s, starting from the same state and using the same pre-sampled noise per trial.",
        "model": MODEL,
        "source_sha256": _source_hashes(),
        "python": sys.version, "platform": platform.platform(),
        "limitations": [
            "Shares can change without changing a decoded finite action contract or executed action.",
            "The default TTC is the first predicted inflated-envelope overlap under fixed nominal actions; it is sampled and limited by ttc_start_s, not a continuous-time collision certificate.",
            "The legacy radial closing-time TTC remains selectable as an ablation; it can be infinite in a merge despite a later nominal conflict.",
            "The first proposer currently follows each scene edge orientation; this is not an identity-invariant priority rule.",
            "If decoder_mode is adaptive_crossing, pairwise reference search uses a bounded half-m/s^2 grid but joint enumeration only admits selected edge references and the shared base actions.",
            "All safety and individual-rationality claims refer only to sampled 2-second kinematic candidate trajectories.",
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "Result" / "ttc_mechanism_ablation")
    parser.add_argument("--scenes", nargs="+", choices=SCENES,
                        default=SCENES)
    parser.add_argument("--n-cars", nargs="+", type=int, choices=(2, 3, 4),
                        default=(2, 3, 4))
    parser.add_argument("--trial-seeds", nargs="+", type=int,
                        default=tuple(range(301, 311)))
    parser.add_argument("--position-std", type=float, default=0.3)
    parser.add_argument("--speed-std", type=float, default=0.2)
    parser.add_argument("--alpha", type=float,
                        default=DEFAULT_POLICY["alpha_per_s"])
    parser.add_argument("--decoder-mode", choices=("coarse", "adaptive_crossing"),
                        default=DEFAULT_POLICY["decoder_mode"])
    parser.add_argument("--joint-objective",
                        choices=("contract_fidelity", "contract_then_share",
                                 "share_then_contract"),
                        default=DEFAULT_POLICY["joint_objective"])
    parser.add_argument("--ttc-source", choices=("nominal_conflict", "legacy_radial"),
                        default=DEFAULT_POLICY["ttc_source"])
    parser.add_argument("--ttc-start", type=float,
                        default=DEFAULT_POLICY["ttc_start_s"])
    parser.add_argument("--constant-ttc", type=float,
                        default=DEFAULT_POLICY["constant_ttc_s"])
    parser.add_argument("--eta-m", type=float,
                        default=DEFAULT_POLICY["surplus_tolerance_m"])
    parser.add_argument("--welfare-epsilon-m", type=float,
                        default=DEFAULT_POLICY["welfare_epsilon_m"])
    args = parser.parse_args(argv)
    policy = normalize_policy({
        "decoder_mode": args.decoder_mode,
        "joint_objective": args.joint_objective,
        "ttc_source": args.ttc_source,
        "alpha_per_s": args.alpha, "ttc_start_s": args.ttc_start,
        "constant_ttc_s": args.constant_ttc,
        "surplus_tolerance_m": args.eta_m,
        "welfare_epsilon_m": args.welfare_epsilon_m,
    })
    directory, summary = run_ablation(
        args.output_root, args.scenes, args.n_cars, args.trial_seeds,
        args.position_std, args.speed_std, policy)
    compact = {
        scene: {count: {
            "ttc_vs_equal_share": result["ttc_vs_equal_share"],
            "ttc_vs_constant": result["ttc_vs_constant"],
        } for count, result in groups.items()}
        for scene, groups in summary.items()}
    print(json.dumps({"result_dir": str(directory), "paired_summary": compact},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
