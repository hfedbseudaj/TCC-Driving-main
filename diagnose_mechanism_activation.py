#!/usr/bin/env python3
"""Measure whether PPO table-4 mechanism settings affect the active model path."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from common.bargaining_evaluation import SCENARIOS, SimState
from common.deterministic_bev import deterministic_bev
from common.experiment_protocol import (
    derive_trial_seed,
    environment_metadata,
    perturb_scenario,
    project_fingerprint,
)
from opencood.models.comm_modules.rubinstein_bargaining import (
    RubinsteinBargaining,
    compute_pairwise_ttc,
)
from train_bargaining_ppo import StochasticPolicy


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINING_RUNS = {
    608: ROOT / "Result/ppo_train_20261031_154640_650394",
    609: ROOT / "Result/ppo_train_20261031_155526_651319",
    610: ROOT / "Result/ppo_train_20261031_160559_652249",
}
VARIANTS = [
    {"label": "baseline_ttc_a0.3_r3", "discount_type": "ttc", "discount_alpha": 0.3, "max_rounds": 3, "utility_type": "learned"},
    {"label": "constant_delta0.9_r3", "discount_type": "constant", "discount_alpha": 0.3, "max_rounds": 3, "utility_type": "learned"},
    {"label": "ttc_a0.1_r3", "discount_type": "ttc", "discount_alpha": 0.1, "max_rounds": 3, "utility_type": "learned"},
    {"label": "ttc_a0.5_r3", "discount_type": "ttc", "discount_alpha": 0.5, "max_rounds": 3, "utility_type": "learned"},
    {"label": "ttc_a1.0_r3", "discount_type": "ttc", "discount_alpha": 1.0, "max_rounds": 3, "utility_type": "learned"},
    {"label": "ttc_a0.3_r1", "discount_type": "ttc", "discount_alpha": 0.3, "max_rounds": 1, "utility_type": "learned"},
    {"label": "ttc_a0.3_r3_handcrafted_utility", "discount_type": "ttc", "discount_alpha": 0.3, "max_rounds": 3, "utility_type": "handcrafted"},
]


class ConstantDiscount(torch.nn.Module):
    def __init__(self, delta=0.9):
        super().__init__()
        self.register_buffer("delta", torch.tensor(float(delta)))

    def forward(self, ttc):
        return self.delta.expand_as(ttc)


class HandcraftedUtility(torch.nn.Module):
    def forward(self, state, offer):
        return torch.sigmoid(10.0 * (offer[:, 3] - 0.5 * offer[:, 0]) - 2.0)


def make_variant(checkpoint, variant):
    model = RubinsteinBargaining({
        "state_dim": 64,
        "offer_dim": 4,
        "hidden_dim": 128,
        "max_rounds": int(variant["max_rounds"]),
        "discount_alpha": float(variant["discount_alpha"]),
        "delta_min": 0.05,
        "delta_max": 0.99,
        "waypoint_offset_scale": 2.0,
        "n_player_protocol": "sequential_pairwise",
    }).to(DEVICE)
    model.load_state_dict(checkpoint["model"], strict=True)
    if variant["discount_type"] == "constant":
        model.discount = ConstantDiscount(0.9).to(DEVICE)
    if variant["utility_type"] == "handcrafted":
        model.utility_estimator = HandcraftedUtility().to(DEVICE)
    model.eval()

    policy = StochasticPolicy(state_dim=64, offer_dim=4, hidden_dim=128).to(DEVICE)
    policy.load_state_dict(checkpoint["policy"], strict=True)
    policy.eval()
    return model, policy


def checkpoint_path(seed, kind):
    run_dir = TRAINING_RUNS[int(seed)]
    filename = "best_checkpoint_full.pth" if kind == "best" else "final_checkpoint_full.pth"
    path = run_dir / "checkpoints" / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def run_diagnostic(seeds, checkpoint_kinds, trials, seed_base):
    rows = []
    checkpoint_hashes = {}
    if DEVICE.type != "cuda":
        raise RuntimeError("Mechanism activation diagnosis requires the authorized GPU host")

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    for training_seed in seeds:
        for checkpoint_kind in checkpoint_kinds:
            path = checkpoint_path(training_seed, checkpoint_kind)
            checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
            if "model" not in checkpoint or "policy" not in checkpoint:
                raise RuntimeError("PPO checkpoint must contain full model and policy state dictionaries")
            checkpoint_hashes["{}:{}".format(training_seed, checkpoint_kind)] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            variant_models = {}
            shared_policy = None
            for variant in VARIANTS:
                model, policy = make_variant(checkpoint, variant)
                variant_models[variant["label"]] = model
                shared_policy = policy

            for scenario in SCENARIOS:
                for trial_index in range(int(trials)):
                    trial_seed = derive_trial_seed(seed_base, scenario, trial_index, list(SCENARIOS))
                    sim = SimState(scenario)
                    perturbation = perturb_scenario(sim, trial_seed)
                    state = sim.get_state(device=DEVICE)
                    features = deterministic_bev(state, device=DEVICE)
                    with torch.no_grad():
                        encoded = variant_models[VARIANTS[0]["label"]].encode_states(features)[0]
                        initial_offers, _ = shared_policy(encoded, deterministic=True)

                    state_ttc = compute_pairwise_ttc(
                        state["positions"], state["velocities"], state["headings"]
                    )
                    finite_ttc = state_ttc[torch.isfinite(state_ttc)]
                    scene_min_ttc = float(finite_ttc.min().item()) if finite_ttc.numel() else float("inf")
                    base_output = None

                    for variant in VARIANTS:
                        model = variant_models[variant["label"]]
                        with torch.no_grad():
                            discount_matrix = model.discount(state_ttc)
                            effective_delta = float(discount_matrix[0, 1].item()) if torch.isfinite(state_ttc[0, 1]) else float("nan")
                            output = model(
                                spatial_features=encoded,
                                waypoints=state["waypoints"],
                                positions=state["positions"],
                                velocities=state["velocities"],
                                headings=state["headings"],
                                record_len=state["record_len"],
                                initial_offers=initial_offers,
                            )

                        metadata = output["metadata"][0]
                        schedule = metadata.get("pair_schedule", [])
                        if variant["label"] == VARIANTS[0]["label"]:
                            base_output = output
                        offer_diff = float("nan")
                        waypoint_diff = float("nan")
                        if base_output is not None:
                            offer_diff = float(torch.norm(
                                output["bargaining_offers"] - base_output["bargaining_offers"]
                            ).item())
                            waypoint_diff = float(torch.norm(
                                output["adjusted_waypoints"] - base_output["adjusted_waypoints"]
                            ).item())

                        rows.append({
                            "training_seed": int(training_seed),
                            "checkpoint_kind": checkpoint_kind,
                            "scenario": scenario,
                            "trial_index": int(trial_index),
                            "trial_seed": int(trial_seed),
                            "scene_min_ttc_s": scene_min_ttc,
                            "scene_perturbation": json.dumps(perturbation, separators=(",", ":")),
                            "variant": variant["label"],
                            "configured_max_rounds": int(model.max_rounds),
                            "configured_discount_type": variant["discount_type"],
                            "configured_discount_alpha": float(variant["discount_alpha"]),
                            "configured_utility_type": variant["utility_type"],
                            "utility_module": type(model.utility_estimator).__name__,
                            "active_pair_count": int(metadata.get("pair_count", 0)),
                            "effective_delta_01": effective_delta,
                            "metadata_n_rounds": int(metadata.get("n_rounds", 0)),
                            "schedule_rounds": ";".join(str(int(pair["rounds"])) for pair in schedule),
                            "schedule_acceptance": ";".join(
                                "".join("1" if value else "0" for value in pair.get("accepted", []))
                                for pair in schedule
                            ),
                            "schedule_convergence": ";".join(
                                "1" if pair.get("converged", False) else "0"
                                for pair in schedule
                            ),
                            "converged": bool(metadata.get("converged", False)),
                            "initial_offer_l2_between_agents": float(torch.norm(
                                initial_offers[0] - initial_offers[1]
                            ).item()),
                            "negotiated_offer_l2_from_initial": float(torch.norm(
                                output["bargaining_offers"] - initial_offers
                            ).item()),
                            "acceptance_mask": "".join(
                                "1" if value else "0"
                                for value in output["acceptance_mask"].detach().cpu().tolist()
                            ),
                            "mean_waypoint_adjustment_m": float((
                                output["adjusted_waypoints"] - state["waypoints"]
                            ).norm(dim=-1).mean().item()),
                            "offer_l2_diff_vs_baseline": offer_diff,
                            "waypoint_l2_diff_vs_baseline": waypoint_diff,
                        })

    return rows, checkpoint_hashes


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    result = []
    for variant in [item["label"] for item in VARIANTS]:
        group = grouped[variant]
        active = [row for row in group if row["active_pair_count"] > 0]
        finite_delta = [row["effective_delta_01"] for row in active if math.isfinite(row["effective_delta_01"])]
        compared = [row for row in group if math.isfinite(row["offer_l2_diff_vs_baseline"])]
        rounds_hist = Counter(row["metadata_n_rounds"] for row in active)
        result.append({
            "variant": variant,
            "samples": len(group),
            "active_pair_samples": len(active),
            "active_pair_rate_pct": 100.0 * len(active) / max(1, len(group)),
            "mean_scene_min_ttc_s_when_active": float(np.mean([row["scene_min_ttc_s"] for row in active])) if active else None,
            "mean_effective_delta_when_active": float(np.mean(finite_delta)) if finite_delta else None,
            "effective_delta_min": float(np.min(finite_delta)) if finite_delta else None,
            "effective_delta_max": float(np.max(finite_delta)) if finite_delta else None,
            "delta_near_lower_clamp_pct": 100.0 * sum(value <= 0.050001 for value in finite_delta) / max(1, len(finite_delta)),
            "delta_near_upper_clamp_pct": 100.0 * sum(value >= 0.989999 for value in finite_delta) / max(1, len(finite_delta)),
            "mean_observed_rounds_when_active": float(np.mean([row["metadata_n_rounds"] for row in active])) if active else None,
            "max_observed_rounds": max([row["metadata_n_rounds"] for row in active], default=0),
            "observed_rounds_histogram": {str(key): value for key, value in sorted(rounds_hist.items())},
            "convergence_pct_when_active": 100.0 * np.mean([row["converged"] for row in active]) if active else None,
            "mean_initial_offer_l2_between_agents": float(np.mean([row["initial_offer_l2_between_agents"] for row in group])) if group else None,
            "negotiation_changed_offer_pct": 100.0 * np.mean([row["negotiated_offer_l2_from_initial"] > 1e-7 for row in group]) if group else None,
            "mean_negotiated_offer_l2_from_initial": float(np.mean([row["negotiated_offer_l2_from_initial"] for row in group])) if group else None,
            "offer_changed_vs_baseline_pct": 100.0 * np.mean([row["offer_l2_diff_vs_baseline"] > 1e-7 for row in compared]) if compared else None,
            "mean_offer_l2_diff_vs_baseline": float(np.mean([row["offer_l2_diff_vs_baseline"] for row in compared])) if compared else None,
            "waypoint_changed_vs_baseline_pct": 100.0 * np.mean([row["waypoint_l2_diff_vs_baseline"] > 1e-7 for row in compared]) if compared else None,
            "mean_waypoint_l2_diff_vs_baseline": float(np.mean([row["waypoint_l2_diff_vs_baseline"] for row in compared])) if compared else None,
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[608, 609, 610])
    parser.add_argument("--checkpoint-kinds", nargs="+", choices=("best", "final"), default=["best", "final"])
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--test-seed-base", type=int, default=20261200)
    parser.add_argument(
        "--training-runs",
        nargs="+",
        default=None,
        metavar="SEED=RUN_DIR",
        help="Override checkpoint run directories, e.g. 608=Result/ppo_train_run",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Result" / "mechanism_activation_20261031")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    if args.training_runs:
        for item in args.training_runs:
            if "=" not in item:
                parser.error("--training-runs entries must use SEED=RUN_DIR")
            seed_text, run_dir = item.split("=", 1)
            try:
                seed = int(seed_text)
            except ValueError:
                parser.error("--training-runs seed must be an integer: {}".format(seed_text))
            TRAINING_RUNS[seed] = Path(run_dir)

    rows, checkpoint_hashes = run_diagnostic(
        args.training_seeds, args.checkpoint_kinds, args.trials, args.test_seed_base
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "mechanism_activation_episodes.csv"
    fields = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "training_seeds": args.training_seeds,
        "checkpoint_kinds": args.checkpoint_kinds,
        "trials_per_scenario": args.trials,
        "scenario_order": list(SCENARIOS),
        "test_seed_base": args.test_seed_base,
        "feature_mode": "common.deterministic_bev",
        "simulator": "common.kinematic_bargaining_env",
        "device": str(DEVICE),
        "environment": environment_metadata(torch, np),
        "checkpoint_sha256": checkpoint_hashes,
        "source_code": project_fingerprint(ROOT, [
            "diagnose_mechanism_activation.py",
            "train_bargaining_ppo.py",
            "common/bargaining_evaluation.py",
            "common/experiment_protocol.py",
            "common/deterministic_bev.py",
            "common/kinematic_bargaining_env.py",
            "common/nplayer_bargaining.py",
            "opencood/models/comm_modules/rubinstein_bargaining.py",
        ]),
        "variants": VARIANTS,
        "summary": summarize(rows),
        "episode_csv": str(csv_path.resolve()),
    }
    summary_path = args.output_dir / "mechanism_activation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))
    print("Episodes: {}".format(csv_path.resolve()))
    print("Summary: {}".format(summary_path.resolve()))


if __name__ == "__main__":
    main()
