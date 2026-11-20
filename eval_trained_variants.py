#!/usr/bin/env python3
"""eval_trained_variants.py — Evaluate all trained variants with 25 trials each.

Loads checkpoints from trained variants and runs evaluation on 4 scenarios
with 25 random seeds per (variant, scenario) pair.
"""

import os, sys, json, math, datetime, csv, glob
from pathlib import Path
import numpy as np
import test_bargaining_ablation as ablation_core
from common.experiment_protocol import (
    derive_trial_seed,
    environment_metadata,
    project_fingerprint,
    selection_score,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = Path(PROJECT_ROOT) / "Result"

# Model variant configs (same as batch_train_variants.py)
VARIANTS = [
    {"label":"baseline",      "prefix":"3r_ttc_learned_a0.3"},
    {"label":"rounds_1",      "prefix":"1r_ttc_learned_a0.3"},
    {"label":"const_delta",   "prefix":"3r_constant_learned_a0.3"},
    {"label":"handcraft_util","prefix":"3r_ttc_handcrafted_a0.3"},
    {"label":"alpha_0.1",     "prefix":"3r_ttc_learned_a0.1"},
    {"label":"alpha_0.5",     "prefix":"3r_ttc_learned_a0.5"},
    {"label":"alpha_1.0",     "prefix":"3r_ttc_learned_a1.0"},
]

SCENARIOS = {
    "intersection":{"a":{"start":[-30,-2],"target":[0,0],"speed":8},"b":{"start":[-2,-30],"target":[0,0],"speed":8}},
    "chase":{"a":{"start":[-25,0],"target":[15,0],"speed":6},"b":{"start":[-15,0],"target":[15,0],"speed":10}},
    "merge":{"a":{"start":[-25,-3],"target":[5,0],"speed":8},"b":{"start":[-30,0],"target":[10,0],"speed":7}},
    "head_on":{"a":{"start":[-20,0],"target":[10,0],"speed":8},"b":{"start":[20,0],"target":[-10,0],"speed":8}},
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") if "torch" in dir() else None

import torch
import numpy as np
import math
from opencood.models.comm_modules.rubinstein_bargaining import RubinsteinBargaining, compute_pairwise_ttc
from common.sl_checkpoint import load_sl_checkpoint, read_sl_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConstantDiscount(torch.nn.Module):
    def __init__(self, delta=0.9):
        super().__init__();self.delta=torch.nn.Parameter(torch.tensor(delta),requires_grad=False)
    def forward(self, ttc):return self.delta.expand_as(ttc)


class HandcraftedUtility(torch.nn.Module):
    def forward(self, state, offer):
        return torch.sigmoid(10.0 * (offer[:, 3] - 0.5 * offer[:, 0]) - 2.0)


class SimState:
    def __init__(self, scenario="intersection", dt=0.1):
        cfg = SCENARIOS[scenario];self.dt=dt;self.name=scenario;self.step=0
        self.collision=False;self.finished=False
        a,b=cfg["a"],cfg["b"]
        self.pos=[np.array(a["start"],np.float32),np.array(b["start"],np.float32)]
        ta=np.array(a["target"],np.float32)-self.pos[0]
        tb=np.array(b["target"],np.float32)-self.pos[1]
        self.spd=[a["speed"],b["speed"]]
        self.hdg=[math.atan2(ta[1],ta[0]),math.atan2(tb[1],tb[0])]
        self._update_wps()
    def _update_wps(self):
        self.wps=[np.zeros((10,2),np.float32) for _ in range(2)]
        for v in range(2):
            for i in range(10):
                t=(i+1)*self.dt
                self.wps[v][i]=[self.spd[v]*t*math.cos(self.hdg[v]),self.spd[v]*t*math.sin(self.hdg[v])]
    def adjust(self,vid,dx,dy):
        spdc=dx*math.cos(self.hdg[vid])+dy*math.sin(self.hdg[vid])
        self.spd[vid]=max(0.3,min(15.0,self.spd[vid]+spdc*2.0))
    def step_sim(self):
        if self.finished:return
        for v in range(2):
            self.pos[v]+=np.array([self.spd[v]*self.dt*math.cos(self.hdg[v]),self.spd[v]*self.dt*math.sin(self.hdg[v])])
        self._update_wps();self.step+=1
        d=np.linalg.norm(self.pos[0]-self.pos[1])
        if d<1.0:self.collision=True;self.finished=True
        if self.step>200 or any(abs(p[0])>40 for p in self.pos) or any(abs(p[1])>40 for p in self.pos):
            self.finished=True
    def get_state(self):
        pos=torch.from_numpy(np.array(self.pos)).float().to(DEVICE)
        vel=torch.tensor([[self.spd[0]*math.cos(self.hdg[0]),self.spd[0]*math.sin(self.hdg[0])],
                          [self.spd[1]*math.cos(self.hdg[1]),self.spd[1]*math.sin(self.hdg[1])]],
                         dtype=torch.float32,device=DEVICE)
        wps=torch.from_numpy(np.array(self.wps)).float().to(DEVICE)
        return {"positions":pos,"velocities":vel,"waypoints":wps,
                "headings":torch.tensor(self.hdg,device=DEVICE),
                "record_len":torch.tensor([2],device=DEVICE)}


def run_episode(model, scenario, max_steps=60, seed=None):
    return ablation_core.run_episode(model, scenario, max_steps=max_steps, seed=seed)


def load_model(variant_config, checkpoint_path, label="unknown"):
    cfg = {"state_dim":64,"offer_dim":4,"hidden_dim":128,
           "max_rounds":variant_config.get("max_rounds",3),
           "discount_alpha":variant_config.get("discount_alpha",0.3),
           "delta_min":0.05,"delta_max":0.99,"waypoint_offset_scale":2.0}
    model = RubinsteinBargaining(cfg).to(DEVICE)
    model.eval()
    
    # Apply variant modifications
    if variant_config.get("discount_type") == "constant":
        model.discount = ConstantDiscount(delta=0.9).to(DEVICE)
    if variant_config.get("utility_type") == "handcrafted":
        model.utility_estimator = HandcraftedUtility().to(DEVICE)
    
    # Load trained weights
    load_utility = not isinstance(model.utility_estimator, HandcraftedUtility)
    load_sl_checkpoint(
        model,
        checkpoint_path,
        map_location=DEVICE,
        load_utility=load_utility,
    )
    
    model.label = label
    return model


def find_checkpoint(prefix):
    """Find the latest checkpoint directory for a given prefix."""
    train_dirs = sorted(RESULT_DIR.glob(f"train_{prefix}_*"))
    if not train_dirs:
        return None
    latest = train_dirs[-1]
    # Check for saved model
    ckpt = latest / "best_model.pth"
    if ckpt.exists():
        return str(ckpt)
    ckpt = latest / "checkpoints" / "best_model.pth"
    if ckpt.exists():
        return str(ckpt)
    return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25, help="evaluation trials per scenario")
    ap.add_argument("--scene-seed", type=int, default=20261200,
                    help="Base seed for paired scenario perturbations")
    args = ap.parse_args()
    
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = RESULT_DIR / f"ablation_trained_25trials_{ts}"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    checkpoint_metadata = {}
    variant_configs = {
        "baseline":      {"max_rounds":3, "discount_type":"ttc",     "discount_alpha":0.3, "utility_type":"learned"},
        "rounds_1":      {"max_rounds":1, "discount_type":"ttc",     "discount_alpha":0.3, "utility_type":"learned"},
        "const_delta":   {"max_rounds":3, "discount_type":"constant", "discount_alpha":0.3, "utility_type":"learned"},
        "handcraft_util":{"max_rounds":3, "discount_type":"ttc",     "discount_alpha":0.3, "utility_type":"handcrafted"},
        "alpha_0.1":     {"max_rounds":3, "discount_type":"ttc",     "discount_alpha":0.1, "utility_type":"learned"},
        "alpha_0.5":     {"max_rounds":3, "discount_type":"ttc",     "discount_alpha":0.5, "utility_type":"learned"},
        "alpha_1.0":     {"max_rounds":3, "discount_type":"ttc",     "discount_alpha":1.0, "utility_type":"learned"},
    }
    
    print(f"Evaluating trained variants with {args.trials} trials each...")
    
    for v in VARIANTS:
        label = v["label"]
        prefix = v["prefix"]
        ckpt_path = find_checkpoint(prefix)
        
        if ckpt_path is None:
            print(f"  {label}: checkpoint NOT FOUND")
            continue
        
        config = variant_configs[label]
        model = load_model(config, ckpt_path, label)
        payload = read_sl_checkpoint(ckpt_path, map_location="cpu")
        training_seed = payload.get("training_seed", payload.get("seed")) if isinstance(payload, dict) else None
        checkpoint_metadata[label] = {
            "checkpoint_path": os.path.abspath(ckpt_path),
            "training_seed": training_seed,
            "checkpoint_schema_version": payload.get("schema_version") if isinstance(payload, dict) else None,
        }
        print(f"  {label}: loaded from {ckpt_path}")
        
        for sc_name in list(SCENARIOS.keys()):
            for t in range(args.trials):
                trial_seed = derive_trial_seed(args.scene_seed, sc_name, t, list(SCENARIOS))
                r = run_episode(model, sc_name, seed=trial_seed)
                r["variant"] = label
                r["scenario"] = sc_name
                r["trial"] = t
                r["trial_seed"] = trial_seed
                r["training_seed"] = training_seed
                r["checkpoint_path"] = os.path.abspath(ckpt_path)
                all_results.append(r)
            
            # Print per-scenario summary
            sc_results = [r for r in all_results if r["variant"]==label and r["scenario"]==sc_name]
            coll = sum(1 for r in sc_results if r["collision"])
            print(f"    {sc_name:<15}: {coll}/{args.trials} collisions ({coll/args.trials*100:.1f}%)")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY ({args.trials} trials per scenario)")
    print(f"{'='*60}")
    print(f"  {'Variant':<20} {'Coll%':>8} {'Near%':>8} {'AvgTTC':>10} {'Conv%':>8} {'Latency':>10}")
    print(f"  {'-'*48}")
    
    from collections import defaultdict
    g = defaultdict(list)
    for r in all_results: g[r["variant"]].append(r)
    
    summary_rows = []
    for label, results in sorted(g.items()):
        coll = sum(1 for r in results if r["collision"])
        total = len(results)
        coll_pct = coll/total*100
        avg_ttc = round(np.mean([r["min_ttc"] for r in results]), 3)
        avg_wpd = round(np.mean([r["avg_wpd"] for r in results]), 4)
        score = selection_score(results)
        near_pct = 100*sum(r["near_collision"] for r in results)/total
        conv_pct = np.mean([r["conv"] for r in results])
        latency = np.mean([r["latency_ms"] for r in results])
        summary_rows.append({"label":label,"coll_pct":f"{coll_pct:.1f}",
                             "near_collision_pct":round(near_pct,1),"avg_ttc":avg_ttc,
                             "convergence_pct":round(conv_pct,2),
                             "mean_latency_ms":round(latency,5),
                             "selection_score":round(score["selection_score"],5),"avg_wpd":avg_wpd})
        print(f"  {label:<20} {coll_pct:>7.1f}% {near_pct:>7.1f}% {avg_ttc:>9.3f}s {conv_pct:>7.1f}% {latency:>9.4f}ms")
    
    # Save
    with open(str(res_dir / "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label","coll_pct","near_collision_pct","avg_ttc","convergence_pct","mean_latency_ms","selection_score","avg_wpd"])
        w.writeheader(); w.writerows(summary_rows)
    
    with open(str(res_dir / "all_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant","label","scenario","trial","trial_seed","training_seed","checkpoint_path","steps","collision","near_collision","min_distance","conv","min_ttc","ttc_risk","latency_ms","avg_wpd","scene_perturbation"])
        w.writeheader()
        for row in all_results:
            output = dict(row)
            output["scene_perturbation"] = json.dumps(output.get("scene_perturbation"), ensure_ascii=False)
            w.writerow(output)
    
    run_config = {
        "scene_seed_base": args.scene_seed,
        "trials_per_scenario": args.trials,
        "scenarios": list(SCENARIOS),
        "checkpoints": checkpoint_metadata,
        "code_version": project_fingerprint(PROJECT_ROOT, [
            "eval_trained_variants.py", "test_bargaining_ablation.py",
            "common/experiment_protocol.py", "common/sl_checkpoint.py",
            "common/deterministic_bev.py",
            "opencood/models/comm_modules/rubinstein_bargaining.py"]),
        "environment": environment_metadata(torch, np),
    }
    with open(res_dir / "run_config.json", "w", encoding="utf-8") as stream:
        json.dump(run_config, stream, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {res_dir}")


if __name__ == "__main__":
    main()
