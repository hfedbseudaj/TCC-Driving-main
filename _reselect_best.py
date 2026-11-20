# -*- coding: utf-8 -*-
"""Defect-3 reselect for HISTORICAL runs: compare train-best vs final weights
by in-training-style evaluation (1 episode/scenario, 60 steps), no retraining.

- SL runs (pre/post fix): complete best_model.pth vs final_model.pth,
  including the trained state_encoder.
- RL / PPO runs: best_policy.pth vs final_policy.pth, random BEV (their native standard).

Score = sum over scenarios of (0 if collision else 1.0) + conv/100  (max 8).
Output: _reselect_result.json
"""
import json
import sys

import torch

ROOT = r"F:\My\Synthesis\Project\TCC-Driving-main"
sys.path.insert(0, ROOT)
from torch import nn

from common.deterministic_bev import deterministic_bev
from common.sl_checkpoint import load_sl_checkpoint
import train_bargaining_supervised as tsl  # SimState, no side effects

SCEN = list(tsl.SCENARIOS.keys())


def make_model():
    return tsl.RubinsteinBargaining({
        "state_dim": 64, "offer_dim": 4, "hidden_dim": 128, "max_rounds": 3,
        "discount_alpha": 0.3, "delta_min": 0.05, "delta_max": 0.99,
        "waypoint_offset_scale": 2.0}).eval()


class OfferDec(nn.Module):
    def __init__(self, mean_net):
        super().__init__()
        self.m = mean_net

    def forward(self, x):
        return torch.sigmoid(self.m(x))


def run_eval(model, feat_fn, scenario, max_steps=60):
    sim = tsl.SimState(scenario)
    conv = 0
    steps = 0
    ttcs = []
    wpds = []
    while not sim.finished and sim.step < max_steps:
        s = sim.get_state()
        pos = s["positions"].float().to("cpu")
        vel = s["velocities"].float().to("cpu")
        hdg = s["headings"].float().to("cpu")
        wps = s["waypoints"].float().to("cpu")
        s["positions"], s["velocities"], s["headings"] = pos, vel, hdg
        feat = feat_fn(s)
        with torch.no_grad():
            out = model(spatial_features=feat, waypoints=wps, positions=pos,
                        velocities=vel, headings=hdg,
                        record_len=torch.tensor([2], dtype=torch.long))
        ttcs.append(s["ttc"])
        wpds.append((out["adjusted_waypoints"] - wps).norm(dim=-1).mean().item())
        steps += 1
        if out["metadata"][0]["converged"]:
            conv += 1
        for v in range(2):
            if v < out["adjusted_waypoints"].shape[0]:
                dxy = (out["adjusted_waypoints"][v, 0] - wps[v, 0]).cpu().numpy()
            else:
                dxy = -(out["adjusted_waypoints"][0, 0] - wps[0, 0]).cpu().numpy() * 0.5
            sim.adjust(v, dxy[0], dxy[1])
        sim.step_sim()
    n = max(1, steps)
    return {"collision": sim.collision, "conv": round(conv / n * 100, 2),
            "minttc": round(min(ttcs) if ttcs else 0, 3),
            "avgwpd": round(float(sum(wpds) / len(wpds)) if wpds else 0, 4)}


def score(results):
    return sum((0.0 if r["collision"] else 1.0) + r["conv"] / 100.0 for r in results.values())


def eval_sl_ckpt(ckpt_path):
    m = make_model()
    load_sl_checkpoint(m, ckpt_path, map_location="cpu")
    res = {}
    for sc in SCEN:
        res[sc] = run_eval(m, lambda s: deterministic_bev(s, device="cpu"), sc)
    return score(res), res


def eval_policy_ckpt(ckpt_path):
    m = make_model()
    sd = torch.load(ckpt_path, map_location="cpu")
    if "policy" in sd:
        sd = sd["policy"]
    # Rebuild mean_net (Sequential Linear->LN->ReLU x2 -> Linear) from state dict
    mean = nn.Sequential(
        nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(),
        nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU(),
        nn.Linear(128, 4))
    mean.load_state_dict({k[len("mean_net."):]: v for k, v in sd.items()
                          if k.startswith("mean_net.")})
    m.offer_decoder = OfferDec(mean)
    res = {}
    for sc in SCEN:
        res[sc] = run_eval(m, lambda s: torch.randn(2, 256, 48, 48), sc)
    return score(res), res


D = ROOT + r"\Result"
SL_GROUPS = [
    ("sl_pre_608", D + r"\train_3r_ttc_learned_a0.3_20261027_213949"),
    ("sl_pre_609", D + r"\train_3r_ttc_learned_a0.3_20261027_214251"),
    ("sl_pre_610", D + r"\train_3r_ttc_learned_a0.3_20261027_214555"),
    ("sl_fix_608", D + r"\train_3r_ttc_learned_a0.3_20261028_141015"),
    ("sl_fix_609", D + r"\train_3r_ttc_learned_a0.3_20261028_141213"),
    ("sl_fix_610", D + r"\train_3r_ttc_learned_a0.3_20261028_141402"),
]
RL_GROUPS = [
    ("rl_608", D + r"\rl_train_20261027_213950"),
    ("rl_609", D + r"\rl_train_20261027_215703"),
    ("rl_610", D + r"\rl_train_20261027_221200"),
]
PPO_GROUPS = [
    ("ppo_fix_608", D + r"\ppo_train_20261027_213952"),
    ("ppo_fix_609", D + r"\ppo_train_20261027_232508_12132"),
    ("ppo_fix_610", D + r"\ppo_train_20261027_234834_31304"),
]

out = {}
lines = []
for name, d in SL_GROUPS:
    ck = d + r"\checkpoints"
    try:
        b, br = eval_sl_ckpt(ck + r"\best_model.pth")
        f, fr = eval_sl_ckpt(ck + r"\final_model.pth")
    except (ValueError, FileNotFoundError) as exc:
        out[name] = {"kind": "sl", "status": "invalid_incomplete_checkpoint", "reason": str(exc)}
        lines.append(f"{name:<12} INVALID: incomplete SL checkpoint")
        continue
    better = "best" if b >= f else "final"
    out[name] = {"kind": "sl", "best_score": round(b, 4), "final_score": round(f, 4),
                 "better": better, "best_detail": br, "final_detail": fr}
    lines.append(f"{name:<12} best={b:.3f} final={f:.3f} -> {better}")
for name, d in RL_GROUPS:
    ck = d + r"\checkpoints"
    b, br = eval_policy_ckpt(ck + r"\best_policy.pth")
    f, fr = eval_policy_ckpt(ck + r"\final_policy.pth")
    better = "best" if b >= f else "final"
    out[name] = {"kind": "rl", "best_score": round(b, 4), "final_score": round(f, 4),
                 "better": better, "best_detail": br, "final_detail": fr}
    lines.append(f"{name:<12} best={b:.3f} final={f:.3f} -> {better}")
for name, d in PPO_GROUPS:
    ck = d + r"\checkpoints"
    b, br = eval_policy_ckpt(ck + r"\best_policy.pth")
    f, fr = eval_policy_ckpt(ck + r"\final_policy.pth")
    better = "best" if b >= f else "final"
    out[name] = {"kind": "ppo", "best_score": round(b, 4), "final_score": round(f, 4),
                 "better": better, "best_detail": br, "final_detail": fr}
    lines.append(f"{name:<12} best={b:.3f} final={f:.3f} -> {better}")

json.dump(out, open(ROOT + r"\_reselect_result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
with open(ROOT + r"\_reselect_summary.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("RESELECT_DONE")
