# -*- coding: utf-8 -*-
"""test_bargaining_ablation_trained.py — Ablation study WITH trained weights.

Loads the best_model.pth checkpoint from the latest training run and
re-runs all ablation variants with the trained state_encoder, offer_decoder,
and utility_estimator. Incomplete legacy checkpoints are rejected.

Usage:
    python test_bargaining_ablation_trained.py                         # auto-detects latest checkpoint
    python test_bargaining_ablation_trained.py --checkpoint <path>     # specify checkpoint manually
    python test_bargaining_ablation_trained.py --compare               # run both random + trained side by side

Output: Result/ablation_trained_YYYYMMDD_HHMMSS/
"""

import os, sys, json, math, datetime, csv
from pathlib import Path
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from common.sl_checkpoint import SL_CHECKPOINT_SCHEMA_VERSION, read_sl_checkpoint
from common.experiment_protocol import (
    derive_trial_seed,
    environment_metadata,
    project_fingerprint,
    selection_score,
)

import torch
from opencood.models.comm_modules.rubinstein_bargaining import (
    RubinsteinBargaining,
)
# Reuse the ablation framework from the base script
# We import the functions but will create separate model instances
import importlib.util
spec = importlib.util.spec_from_file_location(
    "ablation_base",
    str(Path(PROJECT_ROOT) / "test_bargaining_ablation.py"),
)
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)
# Now we have access to: SCENARIOS, DEVICE, SimState, run_episode, ABLATIONS, save_results, etc.

# Override create_variant to optionally load trained weights
ORIGINAL_CREATE = ablation.create_variant

def create_variant_trained(label, checkpoint=None, **kwargs):
    """Create variant and load trained weights if checkpoint provided."""
    model = ORIGINAL_CREATE(label=label, **kwargs)
    if checkpoint is not None:
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError("SL checkpoint does not exist: " + str(checkpoint))
        ablation.load_ckpt(model, checkpoint)
    return model

ablation.create_variant = create_variant_trained

# Find latest checkpoint
def find_latest_checkpoint():
    result_dir = Path(PROJECT_ROOT) / "Result"
    train_dirs = sorted([d for d in result_dir.iterdir() if d.is_dir() and d.name.startswith("train_")])
    if not train_dirs:
        return None
    latest = train_dirs[-1]
    ckpt = latest / "checkpoints" / "best_model.pth"
    return str(ckpt) if ckpt.is_file() else None


def run_trained_ablation(checkpoint, scenarios=None, max_steps=60, trials=1,
                         scene_seed=20261100, training_seed=None):
    """Run ablation with trained weights."""
    if scenarios is None:
        scenarios = list(ablation.SCENARIOS)
    
    results = []
    n_total = len(ablation.ABLATIONS) * sum(len(a["configs"]) for a in ablation.ABLATIONS) * len(scenarios)
    print(f"Running {len(ablation.ABLATIONS)} ablations x {len(scenarios)} scenarios ({n_total} total, trained={checkpoint is not None})\n")
    
    for ab in ablation.ABLATIONS:
        print(f"[{ab['name']}] {ab['desc']}")
        for cfg in ab["configs"]:
            # Add trained_ prefix to distinguish from random-init runs
            label = cfg["label"]
            # Remove any existing prefix
            label = label.replace("trained_", "")
            cfg_with_label = {**cfg, "label": f"trained_{label}"}
            
            model = create_variant_trained(checkpoint=checkpoint, **cfg_with_label)
            for sc in scenarios:
                for t in range(trials):
                    trial_model = create_variant_trained(checkpoint=checkpoint, **cfg_with_label)
                    trial_seed = derive_trial_seed(scene_seed, sc, t, scenarios)
                    r = ablation.run_episode(trial_model, sc, max_steps, seed=trial_seed)
                    r["ablation"] = ab["name"]
                    r["trained"] = True
                    r["trial"] = t
                    r["training_seed"] = training_seed
                    results.append(r)
                    coll = "Y" if r["collision"] else "."
                    print(f"  {r['label']:<25} {sc:<15} trial={t:>2d}  coll={coll}  conv={r['conv']:>6.2f}%  ttc={r['min_ttc']:>7.3f}s  wpd={r['avg_wpd']:>8.4f}")
        print()
    
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Trained ablation study")
    ap.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "configs", "test_bargaining_ablation_trained.yaml"), help="YAML config file")
    pre = ap.parse_known_args()
    try:
        from common.io import load_config_from_yaml
        cfg = load_config_from_yaml(pre[0].config) or {}
    except Exception as e:
        cfg = {}
        print(f"[Config] Warning: failed to load config: {e}")
    ap.add_argument("--checkpoint", type=str, default=cfg.get("checkpoint", None))
    ap.add_argument("--steps", type=int, default=cfg.get("steps", 60))
    ap.add_argument("--scenarios", nargs="+", default=cfg.get("scenarios") or list(ablation.SCENARIOS),
                    choices=list(ablation.SCENARIOS))
    ap.add_argument("--trials", type=int, default=cfg.get("trials", 1))
    ap.add_argument("--scene-seed", type=int, default=cfg.get("scene_seed", 20261100),
                    help="Base seed for paired scene perturbations")
    ap.add_argument("--training-seed", type=int, default=cfg.get("training_seed", None),
                    help="Training seed associated with this checkpoint")
    ap.add_argument("--compare", action="store_true", default=cfg.get("compare", False),
                    help="Run both random init and trained, save combined table")
    args = ap.parse_args()
    
    if args.checkpoint is None:
        args.checkpoint = find_latest_checkpoint()
        if args.checkpoint is None:
            print("Error: No checkpoint found. Train first or use --checkpoint.")
            return
        print(f"Auto-detected checkpoint: {args.checkpoint}\n")

    if args.training_seed is None:
        payload = read_sl_checkpoint(args.checkpoint, map_location="cpu")
        if isinstance(payload, dict):
            args.training_seed = payload.get("training_seed", payload.get("seed"))
    
    print("=" * 60)
    print("  Ablation Study WITH Trained Weights")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Scenarios: {args.scenarios}, Steps: {args.steps}")
    print("=" * 60)
    print()
    
    # Run trained ablation
    trained_results = run_trained_ablation(trials=args.trials,
        checkpoint=args.checkpoint,
        scenarios=args.scenarios,
        max_steps=args.steps,
        scene_seed=args.scene_seed,
        training_seed=args.training_seed,
    )
    
    # Save results
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = Path(PROJECT_ROOT) / "Result" / f"ablation_trained_{ts}"
    res_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "checkpoint_path": os.path.abspath(args.checkpoint),
        "checkpoint_schema_version": SL_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_components": ["state_encoder", "offer_decoder", "utility_estimator"],
        "steps": args.steps,
        "trials_per_scenario": args.trials,
        "scene_seed_base": args.scene_seed,
        "training_seed": args.training_seed,
        "scenarios": args.scenarios,
        "code_version": project_fingerprint(PROJECT_ROOT, [
            "test_bargaining_ablation_trained.py", "test_bargaining_ablation.py",
            "common/experiment_protocol.py", "common/sl_checkpoint.py",
            "common/deterministic_bev.py"]),
        "env": environment_metadata(ablation.torch, np),
    }
    with open(str(res_dir / "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    
    # Full CSV
    with open(str(res_dir / "ablation_trained.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ablation","label","scenario","trial","trial_seed","training_seed","steps","collision","near_collision","min_distance","min_ttc","ttc_risk","conv","latency_ms","avg_wpd","scene_perturbation","trained"])
        w.writeheader()
        for row in trained_results:
            output = dict(row)
            output["scene_perturbation"] = json.dumps(output.get("scene_perturbation"), ensure_ascii=False)
            w.writerow(output)
    
    # Summary
    from collections import defaultdict
    g = defaultdict(list)
    for r in trained_results: g[(r["ablation"], r["label"])].append(r)
    rows = []
    for k, v in sorted(g.items()):
        score = selection_score(v)
        rows.append({"ablation":k[0],"label":k[1],"n":len(v),"trials":len(set(r.get("trial",0)for r in v)),
                     "coll%":round(sum(r["collision"]for r in v)/len(v)*100,1),
                     "near%":round(sum(r["near_collision"]for r in v)/len(v)*100,1),
                     "conv%":round(sum(r["conv"]for r in v)/len(v),1),
                     "ttc":round(sum(r["min_ttc"]for r in v)/len(v),3),
                     "latency_ms":round(sum(r["latency_ms"]for r in v)/len(v),5),
                     "selection_score":round(score["selection_score"],5),
                     "wpd":round(sum(r["avg_wpd"]for r in v)/len(v),4)})
    
    with open(str(res_dir / "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ablation","label","n","trials","coll%","near%","conv%","ttc","latency_ms","selection_score","wpd"])
        w.writeheader(); w.writerows(rows)
    
    # Markdown table
    with open(str(res_dir / "paper_table.md"), "w") as f:
        f.write("| Ablation | Variant (Trained) | Coll% | Near% | Conv% | MinTTC | Latency (ms) | Score | AvgWPD |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f'| {r["ablation"]} | {r["label"]} | {r["coll%"]} | {r["near%"]} | {r["conv%"]} | {r["ttc"]} | {r["latency_ms"]} | {r["selection_score"]} | {r["wpd"]} |\n')
    
    # Comparison with untrained (if --compare)
    if args.compare:
        untrained_results = ablation.run_all(
            scenarios=args.scenarios, max_steps=args.steps, trials=args.trials,
            scene_seed=args.scene_seed)
        # Save combined
        combined = untrained_results + trained_results
        with open(str(res_dir / "ablation_combined.csv"), "w", newline="") as f:
            if combined:
                fieldnames = list(dict.fromkeys(key for row in combined for key in row))
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for row in combined:
                    output = dict(row)
                    output["scene_perturbation"] = json.dumps(output.get("scene_perturbation"), ensure_ascii=False)
                    w.writerow(output)
        
        g2 = defaultdict(list)
        for r in combined: g2[(r["ablation"], r["label"])].append(r)
        rows2 = []
        for k, v in sorted(g2.items()):
            rows2.append({"ablation":k[0],"label":k[1],"n":len(v),
                         "coll%":round(sum(r["collision"]for r in v)/len(v)*100,1),
                         "conv%":round(sum(r["conv"]for r in v)/len(v),1),
                         "ttc":round(sum(r["min_ttc"]for r in v)/len(v),3),
                         "wpd":round(sum(r["avg_wpd"]for r in v)/len(v),4)})
        
        with open(str(res_dir / "comparison_table.md"), "w") as f:
            f.write("| Ablation | Variant | Init | Coll% | Conv% | MinTTC | AvgWPD |\n|---|---|---:|---:|---:|---:|\n")
            for r in rows2:
                init = "trained" if "trained_" in r["label"] else "random"
                lbl = r["label"].replace("trained_", "")
                f.write(f'| {r["ablation"]} | {lbl} | {init} | {r["coll%"]} | {r["conv%"]} | {r["ttc"]} | {r["wpd"]} |\n')
        
        # Combined plot
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            labels_rand = [r["label"] for r in rows if "trained_" not in r["label"]]
            coll_rand = [r["coll%"] for r in rows if "trained_" not in r["label"]]
            labels_train = [r["label"].replace("trained_","") for r in rows if "trained_" in r["label"]]
            coll_train = [r["coll%"] for r in rows if "trained_" in r["label"]]
            
            if labels_rand and labels_train:
                x = range(len(labels_rand))
                axes[0].bar([i-0.2 for i in x], coll_rand, 0.4, label="Random init", color="red", alpha=0.6)
                axes[0].bar([i+0.2 for i in x], coll_train, 0.4, label="Trained", color="green", alpha=0.8)
                axes[0].set_xticks(list(x)); axes[0].set_xticklabels(labels_rand, fontsize=7)
                axes[0].set_ylabel("Collision Rate (%)"); axes[0].legend(); axes[0].grid(alpha=0.3)
                axes[0].set_title("Random vs Trained — Collision Rate")
                
                ttc_r = [r["ttc"] for r in rows if "trained_" not in r["label"]]
                ttc_t = [r["ttc"] for r in rows if "trained_" in r["label"]]
                axes[1].bar([i-0.2 for i in x], ttc_r, 0.4, label="Random init", color="red", alpha=0.6)
                axes[1].bar([i+0.2 for i in x], ttc_t, 0.4, label="Trained", color="green", alpha=0.8)
                axes[1].set_xticks(list(x)); axes[1].set_xticklabels(labels_rand, fontsize=7)
                axes[1].set_ylabel("Min TTC (s)"); axes[1].legend(); axes[1].grid(alpha=0.3)
                axes[1].set_title("Random vs Trained — Min TTC")
            
            plt.suptitle("Ablation: Random Init vs Trained Weights", fontsize=14)
            plt.tight_layout()
            plt.savefig(str(res_dir / "comparison_plot.png"), dpi=150)
            plt.close()
        except Exception as e:
            print(f"[Plot] {e}")
    
    print(f"\nAll results -> {res_dir}")
    (Path(PROJECT_ROOT) / "Result" / "latest.txt").write_text(str(res_dir))


if __name__ == "__main__":
    main()
