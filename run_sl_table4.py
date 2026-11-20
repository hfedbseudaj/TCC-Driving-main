#!/usr/bin/env python3
"""Reproduce trained SL Table 4 and the best/final checkpoint comparison.

For every ablation and training seed, train a complete SL model, select its best
checkpoint on a fixed multi-seed validation cohort, then evaluate best and final
on the same independent test scenes. All artifacts are written under Result/.
"""

import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from common.experiment_protocol import (
    environment_metadata,
    project_fingerprint,
    selection_score,
    write_json,
)


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "Result"
TRAIN_SCRIPT = ROOT / "train_bargaining_supervised.py"
VARIANTS = [
    {"label": "baseline", "max_rounds": 3, "discount_type": "ttc", "discount_alpha": 0.3, "utility_type": "learned"},
    {"label": "rounds_1", "max_rounds": 1, "discount_type": "ttc", "discount_alpha": 0.3, "utility_type": "learned"},
    {"label": "const_delta", "max_rounds": 3, "discount_type": "constant", "discount_alpha": 0.3, "utility_type": "learned"},
    {"label": "handcraft_util", "max_rounds": 3, "discount_type": "ttc", "discount_alpha": 0.3, "utility_type": "handcrafted"},
    {"label": "alpha_0.1", "max_rounds": 3, "discount_type": "ttc", "discount_alpha": 0.1, "utility_type": "learned"},
    {"label": "alpha_0.5", "max_rounds": 3, "discount_type": "ttc", "discount_alpha": 0.5, "utility_type": "learned"},
    {"label": "alpha_1.0", "max_rounds": 3, "discount_type": "ttc", "discount_alpha": 1.0, "utility_type": "learned"},
]
SCENARIOS = ["intersection", "chase", "merge", "head_on"]


def run_training(variant, training_seed, args):
    prefix = "{max_rounds}r_{discount_type}_{utility_type}_a{discount_alpha}_seed{seed}_".format(
        seed=training_seed, **variant)
    before = {p.resolve() for p in RESULT_DIR.glob("train_" + prefix + "*")}
    command = [
        sys.executable, str(TRAIN_SCRIPT),
        "--config", str(ROOT / "runtime" / "unused_config.yaml"),
        "--epochs", str(args.epochs),
        "--ep-per-epoch", str(args.ep_per_epoch),
        "--eval-interval", str(args.eval_interval),
        "--validation-trials", str(args.validation_trials),
        "--validation-seed", str(args.validation_seed),
        "--test-trials", str(args.test_trials),
        "--test-seed", str(args.test_seed),
        "--latency-budget-ms", str(args.latency_budget_ms),
        "--max-rounds", str(variant["max_rounds"]),
        "--discount-type", variant["discount_type"],
        "--discount-alpha", str(variant["discount_alpha"]),
        "--utility-type", variant["utility_type"],
        "--scenarios", *SCENARIOS,
        "--seed", str(training_seed),
    ]
    print("\n[Train] {} seed={}".format(variant["label"], training_seed), flush=True)
    process = subprocess.Popen(
        command, cwd=str(ROOT), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError("Training failed: {} seed={} exit={}".format(
            variant["label"], training_seed, return_code))

    new_runs = [p for p in RESULT_DIR.glob("train_" + prefix + "*")
                if p.resolve() not in before]
    if not new_runs:
        raise FileNotFoundError("Training completed but its Result directory was not found")
    run_dir = max(new_runs, key=lambda p: p.stat().st_mtime)
    comparison_path = run_dir / "best_final_test_comparison.json"
    if not comparison_path.is_file():
        raise FileNotFoundError("Missing independent best/final test report: {}".format(comparison_path))
    return run_dir, json.loads(comparison_path.read_text(encoding="utf-8"))


def aggregate_runs(run_records, output_dir, args):
    episodes = []
    run_rows = []
    for record in run_records:
        variant = record["variant"]
        training_seed = record["training_seed"]
        run_dir = record["run_dir"]
        report = record["report"]
        run_rows.append({
            "variant": variant["label"], "training_seed": training_seed,
            "run_dir": str(run_dir.resolve()),
            "best_checkpoint": report["checkpoint_paths"]["best"],
            "final_checkpoint": report["checkpoint_paths"]["final"],
            "best_epoch": json.loads((run_dir / "eval_best.json").read_text(encoding="utf-8"))["epoch"],
            "test_seed_base": report["test_seed_base"],
            "trials_per_scenario": report["trials_per_scenario"],
            "source_sha256": report["code_version"]["source_sha256"],
        })
        for checkpoint_name, model_result in report["models"].items():
            for episode in model_result["episodes"]:
                episodes.append({
                    "variant": variant["label"],
                    "checkpoint_kind": checkpoint_name,
                    "checkpoint_path": report["checkpoint_paths"][checkpoint_name],
                    "training_seed": training_seed,
                    "trial": episode["trial"],
                    "trial_seed": episode["trial_seed"],
                    "scenario": episode["scenario"],
                    "steps": episode["steps"],
                    "collision": episode["collision"],
                    "near_collision": episode["near_collision"],
                    "min_distance": episode["min_distance"],
                    "min_ttc": episode["min_ttc"],
                    "ttc_risk": episode["ttc_risk"],
                    "conv": episode["conv"],
                    "latency_ms": episode["latency_ms"],
                })

    summary_rows = []
    labels = list(dict.fromkeys(row["variant"] for row in episodes))
    for label in labels:
        for checkpoint_kind in ("best", "final"):
            group = [row for row in episodes
                     if row["variant"] == label and row["checkpoint_kind"] == checkpoint_kind]
            score = selection_score(group, latency_budget_ms=args.latency_budget_ms)
            summary_rows.append({
                "variant": label,
                "checkpoint_kind": checkpoint_kind,
                "training_seeds": len(set(r["training_seed"] for r in group)),
                "trials_per_scenario_per_training_seed": args.test_trials,
                "scenario_count": len(SCENARIOS),
                "total_episodes": len(group),
                "collision_rate_pct": round(100 * score["collision_rate"], 3),
                "near_collision_rate_pct": round(100 * score["near_collision_rate"], 3),
                "mean_min_ttc_s": round(float(np.mean([r["min_ttc"] for r in group])), 4),
                "convergence_rate_pct": round(100 * score["convergence_rate"], 3),
                "mean_latency_ms": round(score["mean_latency_ms"], 5),
                "selection_score": round(score["selection_score"], 5),
            })

    with open(output_dir / "table4_sl_episodes.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episodes[0]))
        writer.writeheader(); writer.writerows(episodes)
    with open(output_dir / "table4_sl_summary.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader(); writer.writerows(summary_rows)
    with open(output_dir / "training_runs.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(run_rows[0]))
        writer.writeheader(); writer.writerows(run_rows)
    with open(output_dir / "table4_sl.md", "w", encoding="utf-8") as stream:
        stream.write("| Variant | Checkpoint | Train seeds | Trials / scenario / seed | Collision % | Near collision % | TTC (s) | Convergence % | Latency (ms) | Score |\n")
        stream.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            stream.write("| {variant} | {checkpoint_kind} | {training_seeds} | {trials_per_scenario_per_training_seed} | {collision_rate_pct} | {near_collision_rate_pct} | {mean_min_ttc_s} | {convergence_rate_pct} | {mean_latency_ms} | {selection_score} |\n".format(**row))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[608, 609, 610])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--ep-per-epoch", type=int, default=5)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--validation-trials", type=int, default=3)
    parser.add_argument("--validation-seed", type=int, default=20261100)
    parser.add_argument("--test-trials", type=int, default=25)
    parser.add_argument("--test-seed", type=int, default=20261200)
    parser.add_argument("--latency-budget-ms", type=float, default=50.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.epochs, args.ep_per_epoch, args.eval_interval,
           args.validation_trials, args.test_trials) < 1:
        raise SystemExit("epochs, episode counts, eval interval, and trial counts must be positive")
    if len(args.train_seeds) != len(set(args.train_seeds)):
        raise SystemExit("training seeds must be unique")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULT_DIR / "sl_table4_repro_" / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "training_seeds": args.train_seeds,
        "validation_seed_base": args.validation_seed,
        "validation_trials_per_scenario": args.validation_trials,
        "test_seed_base": args.test_seed,
        "test_trials_per_scenario_per_training_seed": args.test_trials,
        "scenarios": SCENARIOS,
        "variants": VARIANTS,
        "selection": {
            "direction": "lower_is_better",
            "weights": {"collision_rate": 100, "near_collision_rate": 25,
                        "ttc_risk": 20, "convergence_failure": 10,
                        "latency_fraction_of_50ms_budget": 5},
        },
        "latency_budget_ms": args.latency_budget_ms,
        "epochs": args.epochs,
        "episodes_per_epoch": args.ep_per_epoch,
        "eval_interval": args.eval_interval,
        "code_version": project_fingerprint(ROOT, [
            "run_sl_table4.py", "train_bargaining_supervised.py",
            "test_bargaining_ablation.py", "common/experiment_protocol.py",
            "common/sl_checkpoint.py", "common/deterministic_bev.py",
            "opencood/models/comm_modules/rubinstein_bargaining.py"]),
        "environment": environment_metadata(torch, np),
    }
    write_json(output_dir / "run_config.json", config)

    records = []
    for variant in VARIANTS:
        for training_seed in args.train_seeds:
            run_dir, report = run_training(variant, training_seed, args)
            records.append({"variant": variant, "training_seed": training_seed,
                            "run_dir": run_dir, "report": report})
            write_json(output_dir / "progress.json", {
                "completed_training_runs": len(records),
                "total_training_runs": len(VARIANTS) * len(args.train_seeds),
                "last_run": str(run_dir.resolve()),
            })
    aggregate_runs(records, output_dir, args)
    print("\nSL Table 4 artifacts: {}".format(output_dir.resolve()))


if __name__ == "__main__":
    main()
