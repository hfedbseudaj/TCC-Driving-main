# -*- coding: utf-8 -*-
"""
train_bargaining_supervised.py - Supervised training of the Rubinstein bargaining module.
Uses rule-based ground truth offers to train the state encoder, offer decoder,
and utility estimator.
The rules encode socially optimal behavior: faster vehicle yields in conflict.
"""

import os, sys, json, math, datetime, csv, argparse, time
from pathlib import Path
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from common.sl_checkpoint import (
    SL_CHECKPOINT_SCHEMA_VERSION,
    build_sl_checkpoint,
    load_sl_checkpoint,
)
from common.kinematic_bargaining_env import (
    OBSERVATION_SPEC,
    SCENARIOS as SHARED_SCENARIOS,
    SimState as SharedSimState,
    apply_adjusted_waypoints,
)
from common.experiment_protocol import (
    derive_trial_seed,
    environment_metadata,
    episode_metrics,
    perturb_scenario,
    project_fingerprint,
    selection_score,
    write_json,
)

class ConstantDiscount(nn.Module):
    def __init__(self, delta=0.9):
        super().__init__();self.delta=nn.Parameter(torch.tensor(delta),requires_grad=False)
    def forward(self, ttc):return self.delta.expand_as(ttc)


class HandcraftedUtility(nn.Module):
    def forward(self, state, offer):
        return torch.sigmoid(10.0 * (offer[:, 3] - 0.5 * offer[:, 0]) - 2.0)

from opencood.models.comm_modules.rubinstein_bargaining import (
    RubinsteinBargaining, compute_pairwise_ttc,
)

SCENARIOS = {
    "intersection": {"a":{"start":[-30,-2],"target":[0,0],"speed":8},"b":{"start":[-2,-30],"target":[0,0],"speed":8}},
    "chase": {"a":{"start":[-25,0],"target":[15,0],"speed":6},"b":{"start":[-15,0],"target":[15,0],"speed":10}},
    "merge": {"a":{"start":[-25,-3],"target":[5,0],"speed":8},"b":{"start":[-30,0],"target":[10,0],"speed":7}},
    "head_on": {"a":{"start":[-20,0],"target":[10,0],"speed":8},"b":{"start":[20,0],"target":[-10,0],"speed":8}},
}

class SimState:
    def __init__(self, scenario="intersection", dt=0.1):
        cfg = SCENARIOS[scenario]; self.dt=dt; self.name=scenario; self.step=0
        self.collision=False; self.finished=False
        a,b=cfg["a"],cfg["b"]
        self.pos=[np.array(a["start"],np.float32),np.array(b["start"],np.float32)]
        tgt_a=np.array(a["target"],np.float32)-self.pos[0]
        tgt_b=np.array(b["target"],np.float32)-self.pos[1]
        self.spd=[a["speed"],b["speed"]]
        self.hdg=[math.atan2(tgt_a[1],tgt_a[0]),math.atan2(tgt_b[1],tgt_b[0])]
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
        if self.finished: return
        for v in range(2):
            self.pos[v]+=np.array([self.spd[v]*self.dt*math.cos(self.hdg[v]),self.spd[v]*self.dt*math.sin(self.hdg[v])])
        self._update_wps();self.step+=1
        d=np.linalg.norm(self.pos[0]-self.pos[1])
        if d<1.0:self.collision=True;self.finished=True
        if self.step>200 or any(abs(p[0])>40 for p in self.pos) or any(abs(p[1])>40 for p in self.pos):
            self.finished=True
    def get_state(self):
        pos=torch.from_numpy(np.array(self.pos)).float()
        vel=torch.tensor([[self.spd[0]*math.cos(self.hdg[0]),self.spd[0]*math.sin(self.hdg[0])],
                          [self.spd[1]*math.cos(self.hdg[1]),self.spd[1]*math.sin(self.hdg[1])]],dtype=torch.float32)
        hdg=torch.tensor(self.hdg,dtype=torch.float32)
        wps=torch.stack([torch.from_numpy(self.wps[0]),torch.from_numpy(self.wps[1])],dim=0)
        ttc_m=compute_pairwise_ttc(pos,vel,hdg)
        mt=ttc_m[ttc_m.isfinite()].min().item() if ttc_m.isfinite().any() else 100.0
        return {"positions":pos,"velocities":vel,"headings":hdg,"waypoints":wps,
                "record_len":torch.tensor([2]),"ttc":mt,"speeds":[self.spd[0],self.spd[1]]}


# Use the common simulator in all active training/evaluation paths. The local
# definition above is retained temporarily for old scripts that inspect this
# file, but no longer supplies experimental states.
SCENARIOS = SHARED_SCENARIOS
SimState = SharedSimState


def compute_ground_truth(state):
    """
    Rule-based expert: generates socially optimal offers.
    
    Rule: When TTC < threshold, the faster vehicle yields.
    
    Offer space: [speed_factor, lateral_shift, time_headway, yield_willingness]
    Returns: (2,4) tensor, (2,) weights
    """
    ttc = state["ttc"]
    sa, sb = state["speeds"]
    dev = state["positions"].device if hasattr(state["positions"],"device") else "cpu"
    
    if ttc < 3.0:
        if sa > sb:
            gt = [[0.15, 0.35, 0.80, 0.85], [0.75, 0.10, 0.25, 0.15]]
        else:
            gt = [[0.75, 0.10, 0.25, 0.15], [0.15, 0.35, 0.80, 0.85]]
        w = [2.0, 2.0]
    elif ttc < 6.0:
        ra = sb / (sa + sb + 1e-6)
        rb = sa / (sa + sb + 1e-6)
        gt = [[0.3+0.4*ra, 0.25, 0.6, 0.5+0.3*ra],
              [0.3+0.4*rb, 0.25, 0.6, 0.5+0.3*rb]]
        w = [1.5, 1.5]
    elif ttc < 10.0:
        gt = [[0.45, 0.15, 0.55, 0.55], [0.45, 0.15, 0.55, 0.55]]
        w = [1.0, 1.0]
    else:
        gt = [[0.50, 0.10, 0.50, 0.50], [0.50, 0.10, 0.50, 0.50]]
        w = [0.5, 0.5]
    
    return torch.tensor(gt, dtype=torch.float32, device=dev), torch.tensor(w, device=dev)


def collect_episode(model, scenario, max_steps, device, collect_mode="closed_loop"):
    """Run episode and collect (state_feature -> gt_offer) pairs.

    closed_loop (defect-3/4 fix): the model's own adjusted waypoints drive the
    simulation, so the training distribution matches the evaluation distribution.
    open_loop: legacy behavior (sim.adjust(0,0,0)), kept for comparison.
    """
    sim = SimState(scenario)
    feats, gts = [], []
    model.eval()
    with torch.no_grad():
        while not sim.finished and sim.step < max_steps:
            state = sim.get_state()
            pos = state["positions"].to(device); vel = state["velocities"].to(device)
            hdg = state["headings"].to(device); wps = state["waypoints"].to(device)
            state["positions"], state["velocities"], state["headings"] = pos, vel, hdg
            state["waypoints"] = wps
            # Defect-4 fix: deterministic BEV rendered from the kinematic state
            # (replaces torch.randn(2,256,48,48) which was statistically
            # independent of the state and made the input-target pairing broken).
            from common.deterministic_bev import deterministic_bev
            feat = deterministic_bev(state, device=device)
            gt, _ = compute_ground_truth(state)
            if collect_mode == "closed_loop":
                out = model(spatial_features=feat, waypoints=wps, positions=pos,
                            velocities=vel, headings=hdg,
                            record_len=torch.tensor([2], device=device))
                apply_adjusted_waypoints(sim, state, out["adjusted_waypoints"])
            else:
                sim.adjust(0, 0, 0); sim.adjust(1, 0, 0)
            feats.append(feat.cpu()); gts.append(gt.cpu())
            sim.step_sim()
    model.train()
    return torch.stack(feats), torch.stack(gts)


class BargainingTrainer:
    def __init__(self, device=None, lr=1e-3, max_rounds=3, discount_type="ttc", discount_alpha=0.3, utility_type="learned",
                 validation_trials=5, validation_seed=20261100, test_trials=25,
                 test_seed=20261200, latency_budget_ms=50.0, seed=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.seed = seed
        self.max_rounds = max_rounds
        self.discount_type = discount_type
        self.discount_alpha = discount_alpha
        self.utility_type = utility_type
        self.checkpoint_path = None
        
        self.model = RubinsteinBargaining({
            "state_dim":64,"offer_dim":4,"hidden_dim":128,"max_rounds":max_rounds,
            "discount_alpha":discount_alpha,"delta_min":0.05,"delta_max":0.99,"waypoint_offset_scale":2.0,
        }).to(self.device)
        if discount_type == "constant":
            self.model.discount = ConstantDiscount(delta=0.9).to(self.device)
        if utility_type == "handcrafted":
            self.model.utility_estimator = HandcraftedUtility().to(self.device)
        self.model.train()
        
        # Defect-3(1) fix: unfreeze state_encoder so the model can actually
        # encode the (now deterministic) BEV spatial structure.
        # (was: for p in self.model.state_encoder.parameters(): p.requires_grad = False)
        self.freeze_encoder = False
        
        enc_params = [] if self.freeze_encoder else [{"params": self.model.state_encoder.parameters(), "lr": lr}]
        util_params = [] if utility_type == "handcrafted" else [{"params": self.model.utility_estimator.parameters(), "lr": lr}]
        self.opt = optim.Adam(enc_params + [
            {"params": self.model.offer_decoder.parameters(), "lr": lr},
        ] + util_params)
        self.mse = nn.MSELoss()
        self.margin = nn.MarginRankingLoss(margin=0.5)
        
        self.log = {"epoch":[],"offer_loss":[],"util_loss":[],"total":[]}
        self.best_loss = float("inf")
        # Defect-3 fix: best checkpoint selected by EVALUATION score, not train loss.
        self.best_eval_score = float("inf")
        self.best_eval_epoch = -1
        self.eval_interval = 10
        self.validation_trials = int(validation_trials)
        self.validation_seed = int(validation_seed)
        self.test_trials = int(test_trials)
        self.test_seed = int(test_seed)
        self.latency_budget_ms = float(latency_budget_ms)
        self.eval_history = []
        self.collect_mode = "closed_loop"  # closed_loop / open_loop
        
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        label = f"{max_rounds}r_{discount_type}_{utility_type}_a{discount_alpha}"
        seed_suffix = f"_seed{self.seed}" if self.seed is not None else ""
        self.res_dir = Path(PROJECT_ROOT) / "Result" / f"train_{label}{seed_suffix}_{ts}"
        (self.res_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.res_dir / "plots").mkdir(exist_ok=True)
        print(f"[Train] Results -> {self.res_dir}")
    
    def step(self, features, gt_offers):
        """Single training step."""
        features = features.to(self.device)
        gt_offers = gt_offers.to(self.device)
        
        # Flatten: (T,2,C,H,W) -> (T*2,C,H,W)
        B = features.shape[0] * features.shape[1]
        feat_flat = features.view(B, 256, 48, 48)
        gt_flat = gt_offers.view(B, 4)
        
        states = self.model.state_encoder(feat_flat)  # (B, 64)
        pred = self.model.offer_decoder(states)        # (B, 4)
        offer_loss = self.mse(pred, gt_flat)
        
        # Utility ranking loss: good offers > bad offers
        bad = 1.0 - gt_flat
        u_good = self.model.utility_estimator(states, gt_flat).squeeze(-1)
        u_bad = self.model.utility_estimator(states, bad).squeeze(-1)
        util_loss = self.margin(u_good, u_bad, torch.ones_like(u_good))
        
        total = offer_loss + 0.5 * util_loss
        
        self.opt.zero_grad(); total.backward()
        nn.utils.clip_grad_norm_(
            list(self.model.state_encoder.parameters()) +
            list(self.model.offer_decoder.parameters()) +
            list(self.model.utility_estimator.parameters()), 1.0)
        self.opt.step()
        
        return {"offer_loss":offer_loss.item(),"util_loss":util_loss.item(),"total":total.item()}

    def _checkpoint_model_config(self):
        return {
            "max_rounds": self.max_rounds,
            "discount_type": self.discount_type,
            "discount_alpha": self.discount_alpha,
            "utility_type": self.utility_type,
            "freeze_encoder": bool(self.freeze_encoder),
            "collect_mode": self.collect_mode,
        }

    def _checkpoint_metadata(self):
        return {
            "observation_spec": OBSERVATION_SPEC,
            "training_seed": self.seed,
            "source_fingerprint": project_fingerprint(PROJECT_ROOT, [
                "train_bargaining_supervised.py",
                "common/kinematic_bargaining_env.py",
                "common/deterministic_bev.py",
                "common/sl_checkpoint.py",
                "opencood/models/comm_modules/rubinstein_bargaining.py",
            ]),
            "env": environment_metadata(torch, np),
        }
    
    def train(self, n_epochs=50, ep_per_epoch=5, scenarios=None):
        if scenarios is None: scenarios = list(SCENARIOS)
        print(f"[Train] {n_epochs} epochs, {ep_per_epoch} ep/epoch, scenarios={scenarios}")
        
        for ep in range(n_epochs):
            losses = []
            for _ in range(ep_per_epoch):
                sc = np.random.choice(scenarios)
                feats, gts = collect_episode(self.model, sc, 40, self.device, self.collect_mode)
                losses.append(self.step(feats, gts))
            
            avg = {k: np.mean([l[k] for l in losses]) for k in losses[0]}
            self.log["epoch"].append(ep)
            [self.log[k].append(avg[k]) for k in ["offer_loss","util_loss","total"]]
            
            if ep % max(1, n_epochs//10) == 0 or ep == n_epochs-1:
                print(f"  Ep {ep:>3d} | Offer:{avg['offer_loss']:.6f} | Util:{avg['util_loss']:.6f} | Total:{avg['total']:.6f}")
            
            if avg["total"] < self.best_loss:
                self.best_loss = float(avg["total"])
                torch.save(build_sl_checkpoint(
                    self.model,
                    optimizer=self.opt,
                    best_loss=self.best_loss,
                    epoch=ep,
                    seed=self.seed,
                    model_config=self._checkpoint_model_config(),
                    **self._checkpoint_metadata(),
                ), str(self.res_dir / "checkpoints" / "train_best_model.pth"))

            # Select on a fixed, multi-seed validation cohort. Its scene seeds
            # are disjoint from the final independent test cohort.
            if ep % self.eval_interval == 0 or ep == n_epochs - 1:
                validation_rows = self.evaluate_cohort(
                    self.model, trial_count=self.validation_trials,
                    seed_base=self.validation_seed, cohort="validation", epoch=ep)
                validation_summary = selection_score(
                    validation_rows, latency_budget_ms=self.latency_budget_ms)
                eval_score = validation_summary["selection_score"]
                self.eval_history.append({
                    "epoch": ep, "summary": validation_summary,
                    "episodes": validation_rows,
                })
                self.model.train()
                if eval_score < self.best_eval_score:
                    self.best_eval_score = eval_score
                    self.best_eval_epoch = ep
                    torch.save(build_sl_checkpoint(
                        self.model,
                        optimizer=self.opt,
                        best_eval_score=self.best_eval_score,
                        epoch=ep,
                        seed=self.seed,
                        validation_summary=validation_summary,
                        model_config=self._checkpoint_model_config(),
                        **self._checkpoint_metadata(),
                    ), str(self.res_dir / "checkpoints" / "best_model.pth"))
                    write_json(self.res_dir / "eval_best.json", {
                        "epoch": ep, "summary": validation_summary,
                        "validation_seed_base": self.validation_seed,
                        "validation_trials_per_scenario": self.validation_trials,
                        "episodes": validation_rows,
                    })
                    print(f"  [EvalPick] ep={ep} score={eval_score:.4f} (lower is better) -> best_model.pth")
        
        final_checkpoint = build_sl_checkpoint(
            self.model,
            epoch=n_epochs - 1,
            seed=self.seed,
            best_loss=self.best_loss,
            best_eval_score=self.best_eval_score,
            model_config=self._checkpoint_model_config(),
            **self._checkpoint_metadata(),
        )
        checkpoint_dir = self.res_dir / "checkpoints"
        torch.save(final_checkpoint, str(checkpoint_dir / "final_model.pth"))
        torch.save(self.model.state_encoder.state_dict(),
                   str(checkpoint_dir / "state_encoder_final.pth"))
        torch.save(self.model.offer_decoder.state_dict(),
                   str(checkpoint_dir / "offer_decoder_final.pth"))
        torch.save(self.model.utility_estimator.state_dict(),
                   str(checkpoint_dir / "utility_estimator_final.pth"))
        torch.save(self.model.offer_to_waypoint.state_dict(),
                   str(checkpoint_dir / "offer_to_waypoint_final.pth"))

        self._save_validation_history()
        self._report_best_final(test_trials=self.test_trials, seed_base=self.test_seed)
        
        self._save_log_csv()
        self._save_config()
        print(f"[Train] Done. Best loss: {self.best_loss:.6f}")
        print(f"[Train] Done. Best validation score: {self.best_eval_score:.4f} @ epoch {self.best_eval_epoch} (lower is better)")
    
    def _save_log_csv(self):
        import csv
        p = str(self.res_dir / "training_log.csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.log.keys())
            w.writeheader()
            for i in range(len(self.log["epoch"])):
                w.writerow({k: self.log[k][i] for k in self.log})
        print(f"[Log] Training log -> {p}")

    def _save_config(self):
        cfg = {"learning_rate": self.lr, "device": str(self.device),
               "max_rounds": self.max_rounds,
               "discount_type": self.discount_type,
               "discount_alpha": self.discount_alpha,
               "utility_type": self.utility_type,
               "checkpoint_schema_version": SL_CHECKPOINT_SCHEMA_VERSION,
               "evaluation_checkpoint": self.checkpoint_path,
               "state_dim": 64, "offer_dim": 4, "loss_offer_weight": 1.0,
               "loss_utility_weight": 0.5,
               "seed": self.seed,
               "training_seed": self.seed,
               "observation_spec": OBSERVATION_SPEC,
               "collect_mode": self.collect_mode,
               "freeze_encoder": bool(self.freeze_encoder),
               "eval_interval": self.eval_interval,
               "validation": {"seed_base": self.validation_seed,
                              "trials_per_scenario": self.validation_trials,
                              "scenarios": list(SCENARIOS),
                              "selection": "weighted multi-metric; lower is better",
                              "latency_budget_ms": self.latency_budget_ms},
               "independent_test": {"seed_base": self.test_seed,
                                    "trials_per_scenario": self.test_trials,
                                    "scenarios": list(SCENARIOS)},
               "code_version": project_fingerprint(PROJECT_ROOT, [
                   "train_bargaining_supervised.py", "run_sl_table4.py",
                   "test_bargaining_ablation.py", "common/experiment_protocol.py",
                   "common/sl_checkpoint.py", "common/deterministic_bev.py",
                   "opencood/models/comm_modules/rubinstein_bargaining.py"]),
               "env": environment_metadata(torch, np),
               "rule": "Faster vehicle yields when TTC below 3 seconds."}
        import json
        json.dump(cfg, open(str(self.res_dir / "config.json"), "w"), indent=2)
        print(f"[Config] Saved -> {self.res_dir / 'config.json'}")

    def _plot_curve(self):
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, (a1,a2) = plt.subplots(1,2,figsize=(12,4))
            e = self.log["epoch"]
            a1.plot(e,self.log["offer_loss"],"b-",label="Offer")
            a1.plot(e,self.log["util_loss"],"r-",label="Utility")
            a1.set_xlabel("Epoch");a1.set_ylabel("Loss");a1.legend();a1.grid(alpha=0.3)
            a2.plot(e,self.log["total"],"g-",label="Total",lw=2)
            a2.set_xlabel("Epoch");a2.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(str(self.res_dir/"plots"/"training_curve.png"),dpi=150);plt.close()
        except Exception as ex: print(f"[Plot] {ex}")
    
    def _evaluate_episode(self, model, scenario, trial_seed, max_steps=60):
        """Evaluate a model on one reproducibly perturbed kinematic scene."""
        sim = SimState(scenario)
        perturbation = perturb_scenario(sim, trial_seed)
        model.eval()
        conv=0; wpds=[]; ttc_log=[]; distances=[]; latencies=[]
        while not sim.finished and sim.step < max_steps:
            s=sim.get_state()
            s["positions"]=s["positions"].to(self.device)
            s["velocities"]=s["velocities"].to(self.device)
            s["headings"]=s["headings"].to(self.device)
            s["waypoints"]=s["waypoints"].to(self.device)
            from common.deterministic_bev import deterministic_bev
            feat=deterministic_bev(s, device=self.device)
            ttc_log.append(float(s["ttc"]))
            distances.append(float(torch.norm(
                s["positions"][0] - s["positions"][1]).item()))
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            started=time.perf_counter()
            with torch.no_grad():
                out=model(spatial_features=feat,waypoints=s["waypoints"],
                          positions=s["positions"],velocities=s["velocities"],
                          headings=s["headings"],
                          record_len=torch.tensor([2],device=self.device))
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            latencies.append((time.perf_counter()-started)*1000.0)
            wpds.append((out["adjusted_waypoints"]-s["waypoints"]).norm(dim=-1).mean().item())
            if out["metadata"][0]["converged"]: conv+=1
            apply_adjusted_waypoints(sim, s, out["adjusted_waypoints"])
            sim.step_sim()
        n=max(1,sim.step)
        min_ttc=min(ttc_log) if ttc_log else 100.0
        min_distance=min(distances) if distances else float("inf")
        final_distance=float(np.linalg.norm(sim.pos[0]-sim.pos[1]))
        min_distance=min(min_distance,final_distance)
        result=episode_metrics(
            collision=sim.collision,
            near_collision=(not sim.collision and (min_distance<4.0 or min_ttc<1.0)),
            min_distance=min_distance,min_ttc=min_ttc,
            conv_rate=conv/n*100,
            latency_ms=float(np.mean(latencies)) if latencies else 0.0,
            steps=sim.step)
        result.update({"scenario":scenario,"trial_seed":int(trial_seed),
                       "scene_perturbation":perturbation,
                       "min_ttc":round(result["min_ttc"],4),
                       "min_distance":round(result["min_distance"],4),
                       "conv":round(result["conv"],4),
                       "latency_ms":round(result["latency_ms"],5),
                       "ttc_risk":round(result["ttc_risk"],5)})
        return result

    def evaluate_cohort(self, model, trial_count, seed_base, cohort, epoch=None):
        rows=[]
        scenarios=list(SCENARIOS)
        for scenario in scenarios:
            for trial in range(trial_count):
                trial_seed=derive_trial_seed(seed_base,scenario,trial,scenarios)
                row=self._evaluate_episode(model,scenario,trial_seed,max_steps=60)
                row.update({"cohort":cohort,"trial":trial,
                            "training_seed":self.seed,"epoch":epoch})
                rows.append(row)
        return rows

    def _save_validation_history(self):
        write_json(self.res_dir / "validation_history.json", self.eval_history)
        rows=[]
        for checkpoint_eval in self.eval_history:
            rows.append({"epoch":checkpoint_eval["epoch"], **checkpoint_eval["summary"]})
        if rows:
            with open(self.res_dir / "validation_history.csv", "w", newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
                writer.writeheader();writer.writerows(rows)

    def _report_best_final(self, test_trials, seed_base):
        checkpoint_dir=self.res_dir / "checkpoints"
        candidates={"best":checkpoint_dir / "best_model.pth",
                    "final":checkpoint_dir / "final_model.pth"}
        report={"training_seed":self.seed,"test_seed_base":seed_base,
                "trials_per_scenario":test_trials,"scenarios":list(SCENARIOS),
                "checkpoint_paths":{k:str(v.resolve()) for k,v in candidates.items()},
                "code_version":project_fingerprint(PROJECT_ROOT,[
                    "train_bargaining_supervised.py","run_sl_table4.py",
                    "common/experiment_protocol.py","common/sl_checkpoint.py",
                    "common/deterministic_bev.py",
                    "opencood/models/comm_modules/rubinstein_bargaining.py"]),
                "environment":environment_metadata(torch,np),"models":{}}
        for name,path in candidates.items():
            model=RubinsteinBargaining({
                "state_dim":64,"offer_dim":4,"hidden_dim":128,
                "max_rounds":self.max_rounds,"discount_alpha":self.discount_alpha,
                "delta_min":0.05,"delta_max":0.99,"waypoint_offset_scale":2.0,
            }).to(self.device)
            if self.discount_type=="constant": model.discount=ConstantDiscount(delta=0.9).to(self.device)
            if self.utility_type=="handcrafted": model.utility_estimator=HandcraftedUtility().to(self.device)
            load_sl_checkpoint(model,path,map_location=self.device,
                               load_utility=self.utility_type!="handcrafted")
            rows=self.evaluate_cohort(model,test_trials,seed_base,"independent_test")
            summary=selection_score(rows,latency_budget_ms=self.latency_budget_ms)
            report["models"][name]={"summary":summary,"episodes":rows}
        write_json(self.res_dir / "best_final_test_comparison.json",report)
        summary_rows=[]
        for name,item in report["models"].items():
            summary_rows.append({"checkpoint":name,"checkpoint_path":report["checkpoint_paths"][name],
                                 **item["summary"]})
        with open(self.res_dir / "best_final_test_comparison.csv","w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(summary_rows[0]))
            writer.writeheader();writer.writerows(summary_rows)

    def evaluate(self, scenario="intersection", max_steps=60, trial_seed=20261200):
        """Compare random initialization and trained weights on a paired scene."""
        base = RubinsteinBargaining({
            "state_dim":64,"offer_dim":4,"hidden_dim":128,"max_rounds":3,
            "discount_alpha":0.3,"delta_min":0.05,"delta_max":0.99,"waypoint_offset_scale":2.0,
        }).to(self.device).eval()
        bl=self._evaluate_episode(base,scenario,trial_seed,max_steps)
        tr=self._evaluate_episode(self.model,scenario,trial_seed,max_steps)
        bl["label"]="Baseline";tr["label"]="Trained"
        eval_res = {"eval_scenario": scenario,"trial_seed":trial_seed,
                    "baseline": {k: v for k,v in bl.items() if k != "label"},
                    "trained": {k: v for k,v in tr.items() if k != "label"}}
        write_json(self.res_dir / "evaluation.json",eval_res)
        return bl, tr


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",type=str,default=os.path.join(PROJECT_ROOT,"configs","train_supervised.yaml"),help="YAML config file")
    pre=ap.parse_known_args()
    if os.path.isfile(pre[0].config):
        try:
            from common.io import load_config_from_yaml
            cfg=load_config_from_yaml(pre[0].config) or {}
        except Exception as e:
            cfg={}
            print(f"[Config] Warning: failed to load config: {e}")
    else:
        cfg={}
    ap.add_argument("--epochs",type=int,default=cfg.get("epochs",50))
    ap.add_argument("--lr",type=float,default=cfg.get("lr",1e-3))
    ap.add_argument("--scenarios",nargs="+",default=cfg.get("scenarios",list(SCENARIOS)),choices=list(SCENARIOS))
    ap.add_argument("--max-rounds",type=int,default=cfg.get("max_rounds",3))
    ap.add_argument("--discount-type",type=str,default=cfg.get("discount_type","ttc"),choices=["ttc","constant"])
    ap.add_argument("--discount-alpha",type=float,default=cfg.get("discount_alpha",0.3))
    ap.add_argument("--utility-type",type=str,default=cfg.get("utility_type","learned"),choices=["learned","handcrafted"])
    ap.add_argument("--eval-scenario",default=cfg.get("eval_scenario","intersection"),choices=list(SCENARIOS))
    ap.add_argument("--eval",action="store_true",default=cfg.get("eval_only",False),help="Skip train, only eval from checkpoint")
    ap.add_argument("--checkpoint",type=str,default=cfg.get("checkpoint",None),help="Complete SL checkpoint to load when using --eval")
    ap.add_argument("--ep-per-epoch",type=int,default=cfg.get("ep_per_epoch",5))
    ap.add_argument("--eval-interval",type=int,default=cfg.get("eval_interval",10),help="Epochs between in-training evals; best picked by eval score")
    ap.add_argument("--validation-trials",type=int,default=cfg.get("validation_trials",5),
                    help="Perturbed trials per scenario used for periodic best-checkpoint selection")
    ap.add_argument("--validation-seed",type=int,default=cfg.get("validation_seed",20261100))
    ap.add_argument("--test-trials",type=int,default=cfg.get("test_trials",25),
                    help="Independent test trials per scenario for best/final reporting")
    ap.add_argument("--test-seed",type=int,default=cfg.get("test_seed",20261200))
    ap.add_argument("--latency-budget-ms",type=float,default=cfg.get("latency_budget_ms",50.0))
    ap.add_argument("--collect-mode",type=str,default=cfg.get("collect_mode","closed_loop"),choices=["closed_loop","open_loop"],help="closed_loop: model drives sim during data collection")
    ap.add_argument("--seed",type=int,default=cfg.get("seed",None),help="Fix random seed for reproducibility")
    args=ap.parse_args()
    if args.epochs < 1 or args.ep_per_epoch < 1:
        ap.error("--epochs and --ep-per-epoch must both be positive.")
    if args.eval_interval < 1 or args.validation_trials < 1 or args.test_trials < 1 or args.latency_budget_ms <= 0:
        ap.error("Evaluation interval, validation/test trial counts and latency budget must be positive.")
    if args.eval and not args.checkpoint:
        ap.error("--eval requires --checkpoint pointing to a complete SL checkpoint.")
    if not args.eval and args.checkpoint:
        ap.error("--checkpoint can only be used with --eval.")
    if args.seed is not None:
        import random
        random.seed(args.seed)           # 数据/环境侧
        np.random.seed(args.seed)        # 环境/奖励侧
        torch.manual_seed(args.seed + 1) # 网络/策略侧（与奖励基线解耦）
        print(f"[Seed] Fixed RNG seed = {args.seed} (np=env, torch=net+1)")
    
    t=BargainingTrainer(lr=args.lr,max_rounds=args.max_rounds,discount_type=args.discount_type,
                        discount_alpha=args.discount_alpha,utility_type=args.utility_type,
                        validation_trials=args.validation_trials,validation_seed=args.validation_seed,
                        test_trials=args.test_trials,test_seed=args.test_seed,
                        latency_budget_ms=args.latency_budget_ms,seed=args.seed)
    t.checkpoint_path = args.checkpoint if args.eval else None
    t.eval_interval = args.eval_interval
    t.collect_mode = args.collect_mode
    if args.eval:
        load_sl_checkpoint(t.model, args.checkpoint, map_location=t.device)
    else:
        t.train(n_epochs=args.epochs,ep_per_epoch=args.ep_per_epoch,scenarios=args.scenarios)
    all_evals=[]
    for sc in list(SCENARIOS.keys()):
        r=t.evaluate(scenario=sc,max_steps=60)
        all_evals.append(r)
    import json
    json.dump(all_evals,open(str(Path(t.res_dir)/"all_evals.json"),"w"),indent=2)
    t._save_config()
    print(f"\nDone. Results: {t.res_dir}")

if __name__=="__main__":
    main()
