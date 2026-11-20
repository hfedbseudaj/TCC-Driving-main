# -*- coding: utf-8 -*-
"""
train_bargaining_rl.py - Self-play REINFORCE training for the Rubinstein bargaining module.

The offer_decoder acts as a stochastic policy pi(a|s) with learnable log_std.
The alternating_offers_protocol + kinematic simulation form the environment.

Reward design (multi-objective):
  R = w_conv * conv + w_coll * coll + w_eff * efficiency + w_speed * speed_penalty

Usage:
    python train_bargaining_rl.py --epochs=50
    python train_bargaining_rl.py --resume <ckpt_path>

Output:
    Result/rl_train_YYYYMMDD_HHMMSS/  with logs and model checkpoints
"""

import os, sys, json, math, datetime, csv, hashlib
from pathlib import Path
import numpy as np
from collections import defaultdict
from itertools import accumulate
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from opencood.models.comm_modules.rubinstein_bargaining import (
    RubinsteinBargaining, compute_pairwise_ttc, alternating_offers_protocol,
)
from common.sl_checkpoint import load_sl_checkpoint_as_policy
from common.kinematic_bargaining_env import (
    OBSERVATION_SPEC,
    SCENARIOS as SHARED_SCENARIOS,
    SimState as SharedSimState,
    apply_adjusted_waypoints,
    bev_from_snapshots,
    snapshot_state,
)
from common.deterministic_bev import deterministic_bev
from common.bargaining_reward import (
    REWARD_MODES,
    bargaining_step_reward,
    terminal_reward,
)
from common.bargaining_evaluation import evaluate_policy
from common.experiment_protocol import environment_metadata, project_fingerprint, write_json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCENARIOS = {
    "intersection": {
        "a": {"start":[-30,-2],"target":[0,0],"speed":8},
        "b": {"start":[-2,-30],"target":[0,0],"speed":8},
        "desc": "Perpendicular crossing at an intersection.",
    },
    "chase": {
        "a": {"start":[-25,0],"target":[15,0],"speed":6},
        "b": {"start":[-15,0],"target":[15,0],"speed":10},
        "desc": "Faster vehicle catching up from behind.",
    },
    "merge": {
        "a": {"start":[-25,-3],"target":[5,0],"speed":8},
        "b": {"start":[-30,0],"target":[10,0],"speed":7},
        "desc": "Lane merging from a side lane.",
    },
    "head_on": {
        "a": {"start":[-20,0],"target":[10,0],"speed":8},
        "b": {"start":[20,0],"target":[-10,0],"speed":8},
        "desc": "Head-on approach.",
    },
}


class SimState:
    """Kinematic simulation of two vehicles approaching a conflict point."""
    def __init__(self, scenario="intersection", dt=0.1):
        cfg = SCENARIOS[scenario]; self.dt=dt; self.name=scenario; self.step=0
        self.collision=False; self.finished=False
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
        if self.finished: return
        for v in range(2):
            self.pos[v]+=np.array([self.spd[v]*self.dt*math.cos(self.hdg[v]),
                                   self.spd[v]*self.dt*math.sin(self.hdg[v])])
        self._update_wps(); self.step+=1
        d=np.linalg.norm(self.pos[0]-self.pos[1])
        if d<1.0: self.collision=True; self.finished=True
        if self.step>200 or any(abs(p[0])>40 for p in self.pos) or any(abs(p[1])>40 for p in self.pos):
            self.finished=True
    def get_state(self):
        pos=torch.from_numpy(np.array(self.pos)).float().to(DEVICE)
        vel=torch.tensor([[self.spd[0]*math.cos(self.hdg[0]),self.spd[0]*math.sin(self.hdg[0])],
                          [self.spd[1]*math.cos(self.hdg[1]),self.spd[1]*math.sin(self.hdg[1])]],
                         dtype=torch.float32,device=DEVICE)
        hdg=torch.tensor(self.hdg,dtype=torch.float32,device=DEVICE)
        wps=torch.stack([torch.from_numpy(self.wps[0]).to(DEVICE),
                         torch.from_numpy(self.wps[1]).to(DEVICE)],dim=0)
        ttc_m=compute_pairwise_ttc(pos,vel,hdg)
        mt=ttc_m[ttc_m.isfinite()].min().item() if ttc_m.isfinite().any() else 100.0
        return {"positions":pos,"velocities":vel,"headings":hdg,"waypoints":wps,
                "record_len":torch.tensor([2],device=DEVICE),"ttc":mt,"spd":list(self.spd)}


# All active training and evaluation paths now use the shared scenario and
# simulator definitions. This removes the prior PPO chase-speed mismatch.
SCENARIOS = SHARED_SCENARIOS
SimState = SharedSimState



# ── RL Policy Components ──

class StochasticPolicy(nn.Module):
    """
    Stochastic policy wrapping the offer_decoder with learnable exploration noise.
    
    The policy outputs a diagonal Gaussian over the 4D offer space (logit-space).
    Sigmoid is applied to the sampled action to produce bounded offers in [0,1].
    
    pi(a|s) = Normal(mean=decoder(s), std=exp(log_std))
    """
    def __init__(self, state_dim=64, offer_dim=4, hidden_dim=128, init_log_std=0.0):
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, offer_dim),
        )
        self.log_std = nn.Parameter(torch.full((offer_dim,), init_log_std))
    
    def forward(self, state, deterministic=False):
        """
        Args:
            state: (B, state_dim) bargaining state embeddings
            deterministic: if True, return mean action without noise
        Returns:
            action: (B, offer_dim) sigmoid-bounded offers in [0,1]
            log_prob: (B,) log probability of sampled action (or None if deterministic)
        """
        mean = self.mean_net(state)
        if deterministic:
            return torch.sigmoid(mean), None
        
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        action_logit = dist.rsample()  # reparameterized sample
        action = torch.sigmoid(action_logit)
        
        # Log probability with change of variables for sigmoid
        log_prob = dist.log_prob(action_logit).sum(dim=-1)
        # Jacobian correction for sigmoid: d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        log_prob -= torch.log(action * (1 - action) + 1e-8).sum(dim=-1)
        
        return action, log_prob
    
    def get_log_prob(self, state, action):
        """Compute log pi(action | state) for a given action."""
        mean = self.mean_net(state)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        # Inverse sigmoid to get logit
        action_logit = torch.log(action / (1 - action + 1e-8) + 1e-8)
        log_prob = dist.log_prob(action_logit).sum(dim=-1)
        log_prob -= torch.log(action * (1 - action) + 1e-8).sum(dim=-1)
        return log_prob

    def get_entropy(self, state):
        mean = self.mean_net(state)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std).entropy().sum(dim=-1).mean()


class ValueBaseline(nn.Module):
    """
    State-value estimator V(s) for advantage computation in REINFORCE.
    Trained with MSE loss against Monte Carlo returns.
    """
    def __init__(self, state_dim=64, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state):
        """(B, state_dim) -> (B, 1)"""
        return self.net(state)


def compute_return(rewards, gamma=0.95):
    """Compute discounted cumulative returns from a list of rewards."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def compute_step_reward(converged, collision, speed_a, speed_b, n_rounds, dist,
                        ttc=float("inf"), mode="reinforce_legacy"):
    """Backward-compatible wrapper around the shared reward definitions."""
    return bargaining_step_reward(
        converged=converged, collision=collision, speed_a=speed_a,
        speed_b=speed_b, n_rounds=n_rounds, dist=dist, ttc=ttc, mode=mode,
    )



class REINFORCETrainer:
    """
    Self-play REINFORCE training for the bargaining module.
    
    Algorithm:
      1. Sample initial offers from stochastic policy pi(a|s)
      2. Run bargaining protocol (deterministic, part of environment)
      3. Step kinematic simulation, get reward
      4. Store (state, action, reward) trajectory
      5. At episode end: compute returns, advantages, update policy
    
    Value baseline: separate V(s) network trained with MSE.
    """
    def __init__(self, lr=3e-4, gamma=0.95, entropy_coef=0.05,
                 reward_mode="ttc_calibrated"):
        self.lr = lr
        self.seed = None
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}")
        self.reward_mode = reward_mode
        self.base_checkpoint_path = None
        
        # Base bargaining model (provides state encoder and utility estimator)
        self.model = RubinsteinBargaining({
            "state_dim":64,"offer_dim":4,"hidden_dim":128,"max_rounds":3,
            "discount_alpha":0.3,"delta_min":0.05,"delta_max":0.99,"waypoint_offset_scale":2.0,
        }).to(DEVICE)
        self.model.train()
        
        # The shared deterministic BEV encoder is trained with the policy.
        # Freezing a random encoder made the old RL runs state-blind.
        for p in self.model.state_encoder.parameters():
            p.requires_grad = True
        
        # Stochastic policy (replaces the original offer_decoder)
        self.policy = StochasticPolicy(
            state_dim=64, offer_dim=4, hidden_dim=128, init_log_std=0.0
        ).to(DEVICE)
        
        # Value baseline
        self.value_net = ValueBaseline(state_dim=64, hidden_dim=128).to(DEVICE)
        
        # Optimizers
        self.policy_opt = optim.Adam(
            list(self.policy.parameters()) + list(self.model.state_encoder.parameters()),
            lr=lr,
        )
        self.value_opt = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Logging
        self.log = defaultdict(list)
        self.best_reward = -float('inf')
        # Defect-3 fix: best checkpoint selected by EVALUATION score, not train reward.
        self.best_eval_score = float("inf")
        self.best_eval_epoch = -1
        self.eval_interval = 10
        self.total_env_steps = 0
        self.total_training_episodes = 0
        self.training_budget_env_steps = None
        self.eval_step_interval = None
        self.validation_trials = 5
        self.validation_seed_base = 20261100
        self.latency_budget_ms = 50.0
        self.test_trials = 25
        self.test_seed_base = 20261200
        self.base_checkpoint_path = None
        self.validation_history = []
        self.validation_rows = []
        
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.res_dir = Path(PROJECT_ROOT) / "Result" / f"rl_train_{ts}"
        (self.res_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        print(f"[RL] Results -> {self.res_dir}")

    def _checkpoint_metadata(self):
        base_checkpoint_sha256 = None
        if self.base_checkpoint_path and Path(self.base_checkpoint_path).is_file():
            base_checkpoint_sha256 = hashlib.sha256(
                Path(self.base_checkpoint_path).read_bytes()
            ).hexdigest()
        return {
            "observation_spec": OBSERVATION_SPEC,
            "reward_mode": self.reward_mode,
            "base_checkpoint_path": self.base_checkpoint_path,
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "training_seed": self.seed,
            "training_env_steps": self.total_env_steps,
            "training_episodes": self.total_training_episodes,
            "training_budget_env_steps": self.training_budget_env_steps,
            "source_fingerprint": project_fingerprint(PROJECT_ROOT, [
                "train_bargaining_rl.py",
                "common/kinematic_bargaining_env.py",
                "common/deterministic_bev.py",
                "common/bargaining_reward.py",
                "common/bargaining_evaluation.py",
                "common/sl_checkpoint.py",
                "opencood/models/comm_modules/rubinstein_bargaining.py",
            ]),
            "env": environment_metadata(torch),
        }
    
    def collect_episode(self, scenario, max_steps=50):
        """Run one episode, return trajectory of (state, action, reward, info)."""
        sim = SimState(scenario)
        trajectory = []
        
        while not sim.finished and sim.step < max_steps:
            state = sim.get_state(device=DEVICE)
            features = deterministic_bev(state, device=DEVICE)
            with torch.no_grad():
                state_enc = self.model.state_encoder(features)
                init_offers, log_prob = self.policy(state_enc)
                out = self.model(
                    spatial_features=state_enc,
                    waypoints=state["waypoints"],
                    positions=state["positions"],
                    velocities=state["velocities"],
                    headings=state["headings"],
                    record_len=state["record_len"],
                    initial_offers=init_offers,
                )
            
            adj_wps = out["adjusted_waypoints"]
            apply_adjusted_waypoints(sim, state, adj_wps)
            
            # Score the exact negotiated policy offers used to produce the
            # applied waypoints, not a separate model-generated offer.
            dist = np.linalg.norm(sim.pos[0] - sim.pos[1])
            metadata = out["metadata"][0]
            converged = bool(metadata.get("converged", False))
            reward = bargaining_step_reward(
                converged=converged,
                collision=False,
                speed_a=sim.spd[0],
                speed_b=sim.spd[1],
                n_rounds=metadata.get("n_rounds", 0),
                dist=dist,
                ttc=state["ttc"],
                mode=self.reward_mode,
            )
            
            trajectory.append({
                "state_snapshot": snapshot_state(state),
                "action": init_offers.detach().cpu(),
                "log_prob": log_prob.detach().cpu(),
                "reward": reward,
                "converged": converged,
                "done": False,
            })
            
            sim.step_sim()
        
        if trajectory:
            trajectory[-1]["reward"] += terminal_reward(sim.collision)
            trajectory[-1]["done"] = True
        
        return trajectory, sim
    
    def update(self, trajectory):
        """REINFORCE update with baseline and entropy bonus."""
        if not trajectory:
            return {}
        
        device = DEVICE
        snapshots = [t["state_snapshot"] for t in trajectory]
        features = bev_from_snapshots(snapshots, device=device)
        states = self.model.state_encoder(features)
        states_detached = states.detach()
        actions = torch.stack([t["action"] for t in trajectory]).to(device).reshape(-1, 4)
        rewards = torch.tensor([t["reward"] for t in trajectory], device=device, dtype=torch.float32)

        returns = torch.tensor(compute_return(rewards.tolist(), self.gamma), device=device, dtype=torch.float32)
        raw_return_mean = float(returns.mean().item())
        returns = (returns - returns.mean()) / returns.std(unbiased=False).clamp_min(1e-8)
        returns = returns[:, None].expand(-1, 2).reshape(-1)

        # Both agents receive the shared episode return; the previous code
        # updated only vehicle 0 although both policy actions affected motion.
        values = self.value_net(states_detached).squeeze(-1)
        advantages = returns - values.detach()
        log_probs = self.policy.get_log_prob(states, actions)
        policy_loss = -(log_probs * advantages).mean()
        entropy = self.policy.get_entropy(states)
        total_loss = policy_loss - self.entropy_coef * entropy

        self.policy_opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.model.state_encoder.parameters()),
            max_norm=1.0,
        )
        self.policy_opt.step()

        value_loss = F.mse_loss(values, returns)
        self.value_opt.zero_grad()
        value_loss.backward()
        self.value_opt.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "mean_return": raw_return_mean,
            "mean_advantage": advantages.mean().item(),
        }
    
    def train(self, n_epochs=100, ep_per_epoch=5, scenarios=None,
              total_env_steps=None, eval_step_interval=None):
        """Main training loop."""
        if scenarios is None:
            scenarios = ["intersection", "merge"]

        self.total_env_steps = 0
        self.total_training_episodes = 0
        self.training_budget_env_steps = (
            int(total_env_steps) if total_env_steps is not None else None
        )
        if self.training_budget_env_steps is not None and self.training_budget_env_steps < 1:
            raise ValueError("total_env_steps must be positive")
        if eval_step_interval is not None and int(eval_step_interval) < 1:
            raise ValueError("eval_step_interval must be positive")
        self.eval_step_interval = (
            int(eval_step_interval)
            if eval_step_interval is not None
            else (
                max(1, self.training_budget_env_steps // 10)
                if self.training_budget_env_steps is not None
                else None
            )
        )
        next_validation_step = self.eval_step_interval
        
        print()
        print("  {:>5} | {:>8} | {:>5} | {:>7} | {:>8} | {:>8}".format("Ep", "Reward", "Coll", "Conv%", "Loss", "Entropy"))
        print("-" * 55)
        
        for ep in range(n_epochs):
            ep_rewards = []; ep_collisions = []; ep_losses = []; ep_conv_rates = []
            
            for _ in range(ep_per_epoch):
                if (
                    self.training_budget_env_steps is not None
                    and self.total_env_steps >= self.training_budget_env_steps
                ):
                    break
                sc = np.random.choice(scenarios)
                max_steps = 50
                if self.training_budget_env_steps is not None:
                    max_steps = min(
                        max_steps,
                        self.training_budget_env_steps - self.total_env_steps,
                    )
                trajectory, sim = self.collect_episode(sc, max_steps=max_steps)
                if not trajectory:
                    raise RuntimeError("REINFORCE collected an empty trajectory")
                stats = self.update(trajectory)
                self.total_env_steps += len(trajectory)
                self.total_training_episodes += 1
                
                total_r = sum(t["reward"] for t in trajectory)
                ep_rewards.append(total_r)
                ep_collisions.append(sim.collision)
                ep_losses.append(stats.get("policy_loss", 0))
                ep_conv_rates.append(100.0 * np.mean([t["converged"] for t in trajectory]) if trajectory else 0.0)

            if not ep_rewards:
                break
            
            avg_reward = np.mean(ep_rewards)
            avg_coll = np.mean(ep_collisions) * 100
            avg_loss = np.mean(ep_losses)
            entropy = stats.get("entropy", 0)
            conv_pct = float(np.mean(ep_conv_rates)) if ep_conv_rates else 0.0
            
            self.log["epoch"].append(ep)
            self.log["env_steps"].append(self.total_env_steps)
            self.log["episodes"].append(self.total_training_episodes)
            self.log["reward"].append(avg_reward)
            self.log["collision_pct"].append(avg_coll)
            self.log["convergence_pct"].append(conv_pct)
            self.log["policy_loss"].append(avg_loss)
            self.log["entropy"].append(entropy)
            
            if ep % max(1, n_epochs // 10) == 0 or ep == n_epochs - 1:
                print(f"  {ep:>5} | {avg_reward:>+8.3f} | {avg_coll:>5.1f}% | {conv_pct:>7.1f} | {avg_loss:>8.5f} | {entropy:>8.5f}")
            
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                torch.save(self.policy.state_dict(),
                           str(self.res_dir / "checkpoints" / "train_best_policy.pth"))
                torch.save({
                    "epoch": ep,
                    "best_reward": float(self.best_reward),
                    "policy": self.policy.state_dict(),
                    "value_net": self.value_net.state_dict(),
                    "model": self.model.state_dict(),
                    "observation_spec": OBSERVATION_SPEC,
                    "reward_mode": self.reward_mode,
                    **self._checkpoint_metadata(),
                }, str(self.res_dir / "checkpoints" / "train_best_checkpoint_full.pth"))

            # Defect-3 fix: best selected by evaluation score (kept as best_policy.pth).
            if self.training_budget_env_steps is None:
                should_validate = ep % self.eval_interval == 0 or ep == n_epochs - 1
            else:
                should_validate = (
                    self.total_env_steps >= next_validation_step
                    or self.total_env_steps >= self.training_budget_env_steps
                    or ep == n_epochs - 1
                )
            if should_validate:
                rows, summary = self.evaluate_for_checkpoint_selection(max_steps=60)
                eval_score = summary["selection_score"]
                self.validation_history.append({
                    "epoch": ep,
                    "training_env_steps": self.total_env_steps,
                    **summary,
                })
                self.validation_rows.extend(
                    {
                        "training_seed": self.seed,
                        "epoch": ep,
                        "training_env_steps": self.total_env_steps,
                        **row,
                    }
                    for row in rows
                )
                self._save_validation_history(rows)
                if eval_score < self.best_eval_score:
                    self.best_eval_score = float(eval_score)
                    self.best_eval_epoch = ep
                    torch.save(self.policy.state_dict(),
                               str(self.res_dir / "checkpoints" / "best_policy.pth"))
                    torch.save({
                        "epoch": ep,
                        "best_eval_score": self.best_eval_score,
                        "validation_metrics": summary,
                        "policy": self.policy.state_dict(),
                        "value_net": self.value_net.state_dict(),
                        "model": self.model.state_dict(),
                        "observation_spec": OBSERVATION_SPEC,
                        "reward_mode": self.reward_mode,
                        "base_checkpoint_path": self.base_checkpoint_path,
                        **self._checkpoint_metadata(),
                    }, str(self.res_dir / "checkpoints" / "best_checkpoint_full.pth"))
                    write_json(self.res_dir / "eval_best.json", {
                        "epoch": ep, "score": eval_score,
                        "validation_summary": summary,
                    })
                    print(f"  [EvalPick] ep={ep} validation_score={eval_score:.4f} -> best_policy.pth")

                if self.training_budget_env_steps is not None:
                    while next_validation_step <= self.total_env_steps:
                        next_validation_step += self.eval_step_interval

            if (
                self.training_budget_env_steps is not None
                and self.total_env_steps >= self.training_budget_env_steps
            ):
                break

        if (
            self.training_budget_env_steps is not None
            and self.total_env_steps != self.training_budget_env_steps
        ):
            raise RuntimeError(
                "REINFORCE exhausted max epochs at {} / {} environment steps".format(
                    self.total_env_steps, self.training_budget_env_steps
                )
            )
        
        torch.save(self.policy.state_dict(),
                   str(self.res_dir / "checkpoints" / "final_policy.pth"))
        torch.save({
            "epoch": self.log["epoch"][-1] if self.log["epoch"] else -1,
            "best_reward": float(self.best_reward),
            "best_eval_score": float(self.best_eval_score),
            "best_eval_epoch": self.best_eval_epoch,
            "policy": self.policy.state_dict(),
            "value_net": self.value_net.state_dict(),
            "model": self.model.state_dict(),
            "observation_spec": OBSERVATION_SPEC,
            "reward_mode": self.reward_mode,
            "base_checkpoint_path": self.base_checkpoint_path,
            **self._checkpoint_metadata(),
        }, str(self.res_dir / "checkpoints" / "final_checkpoint_full.pth"))
        self._save_log()
        self._plot_curve()
        print()
        print(f"[RL] Done. Best avg reward: {self.best_reward:.3f}")
        print(f"[RL] Done. Best eval score: {self.best_eval_score:.4f} @ epoch {self.best_eval_epoch} (eval-interval={self.eval_interval})")
    
    def _save_log(self):
        with open(str(self.res_dir / "rl_log.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.log.keys()))
            w.writeheader()
            for i in range(len(self.log["epoch"])):
                w.writerow({k: self.log[k][i] for k in self.log})
        
        json.dump({k: v for k, v in self.log.items()},
                  open(str(self.res_dir / "rl_log.json"), "w"), default=float)
        json.dump({"seed": self.seed, "training_seed": self.seed,
                   "lr": self.lr, "gamma": self.gamma,
                   "entropy_coef": self.entropy_coef,
                   "reward_mode": self.reward_mode,
                   "observation_spec": OBSERVATION_SPEC,
                   "base_checkpoint_path": self.base_checkpoint_path,
                   "base_checkpoint_sha256": self._checkpoint_metadata()["base_checkpoint_sha256"],
                   "training_env_steps": self.total_env_steps,
                   "training_episodes": self.total_training_episodes,
                   "training_budget_env_steps": self.training_budget_env_steps,
                   "eval_step_interval": self.eval_step_interval,
                   "checkpoint_selection_rule": "minimum shared multi-scenario collision/near-collision/TTC/convergence/latency score",
                   "validation_trials_per_scenario": self.validation_trials,
                   "validation_seed_base": self.validation_seed_base,
                   "latency_budget_ms": self.latency_budget_ms,
                   "validation_best_epoch": self.best_eval_epoch,
                   "validation_best_score": self.best_eval_score,
                   "test_trials_per_scenario": self.test_trials,
                   "test_seed_base": self.test_seed_base,
                   "source_fingerprint": project_fingerprint(PROJECT_ROOT, [
                       "train_bargaining_rl.py",
                       "common/kinematic_bargaining_env.py",
                       "common/deterministic_bev.py",
                       "common/bargaining_reward.py",
                       "common/bargaining_evaluation.py",
                       "opencood/models/comm_modules/rubinstein_bargaining.py",
                   ]),
                   "env": environment_metadata(torch)},
                  open(str(self.res_dir / "config.json"), "w", encoding="utf-8"), indent=2, default=str)
        print(f"[Log] Saved to {self.res_dir}")
    
    def _plot_curve(self):
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            e = self.log["epoch"]
            axes[0].plot(e, self.log["reward"], "b-")
            axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Avg Reward"); axes[0].grid(alpha=0.3)
            
            axes[1].plot(e, self.log["collision_pct"], "r-")
            axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Collision %"); axes[1].grid(alpha=0.3)
            
            axes[2].plot(e, self.log["entropy"], "g-")
            axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Policy Entropy"); axes[2].grid(alpha=0.3)
            
            plt.suptitle("REINFORCE Training Curves")
            plt.tight_layout()
            plt.savefig(str(self.res_dir / "rl_curves.png"), dpi=150); plt.close()
            print(f"[Plot] -> {self.res_dir / 'rl_curves.png'}")
        except Exception as ex:
            print(f"[Plot] {ex}")


    def evaluate_for_checkpoint_selection(self, max_steps=60):
        return evaluate_policy(
            self.model,
            self.policy,
            DEVICE,
            scenarios=list(SCENARIOS.keys()),
            trials_per_scenario=self.validation_trials,
            seed_base=self.validation_seed_base,
            max_steps=max_steps,
            latency_budget_ms=self.latency_budget_ms,
        )

    def _save_validation_history(self, latest_rows):
        write_json(self.res_dir / "validation_history.json", self.validation_history)
        summary_fields = [
            "epoch", "training_env_steps", "selection_score", "collision_rate", "near_collision_rate",
            "mean_ttc_risk", "convergence_rate", "mean_latency_ms",
            "validation_episodes", "latency_budget_ms",
        ]
        with open(self.res_dir / "validation_history.csv", "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=summary_fields)
            writer.writeheader()
            for row in self.validation_history:
                writer.writerow({key: row.get(key) for key in summary_fields})
        self.validation_rows = getattr(self, "validation_rows", [])
        episode_fields = [
            "training_seed", "epoch", "training_env_steps", "scenario", "trial_index", "trial_seed", "steps", "collision",
            "near_collision", "min_distance", "min_ttc", "ttc_risk", "conv",
            "latency_ms", "scene_perturbation",
        ]
        with open(self.res_dir / "validation_episodes.csv", "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=episode_fields)
            writer.writeheader()
            for row in self.validation_rows:
                output = dict(row)
                output["scene_perturbation"] = json.dumps(output.get("scene_perturbation"), ensure_ascii=False)
                writer.writerow({key: output.get(key) for key in episode_fields})

    def evaluate_test_checkpoints(self):
        """Report best and final checkpoints on a separate, fixed test cohort."""
        checkpoint_dir = self.res_dir / "checkpoints"
        candidates = {
            "best": checkpoint_dir / "best_checkpoint_full.pth",
            "final": checkpoint_dir / "final_checkpoint_full.pth",
        }
        final_model = {key: value.detach().clone() for key, value in self.model.state_dict().items()}
        final_policy = {key: value.detach().clone() for key, value in self.policy.state_dict().items()}
        reports = {}
        all_rows = []
        try:
            for label, checkpoint_path in candidates.items():
                if not checkpoint_path.is_file():
                    continue
                try:
                    checkpoint = torch.load(
                        checkpoint_path, map_location=DEVICE, weights_only=False
                    )
                except TypeError:
                    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                self.model.load_state_dict(checkpoint["model"], strict=True)
                self.policy.load_state_dict(checkpoint["policy"], strict=True)
                rows, summary = evaluate_policy(
                    self.model,
                    self.policy,
                    DEVICE,
                    scenarios=list(SCENARIOS.keys()),
                    trials_per_scenario=self.test_trials,
                    seed_base=self.test_seed_base,
                    max_steps=60,
                    latency_budget_ms=self.latency_budget_ms,
                )
                reports[label] = {
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    **summary,
                }
                all_rows.extend(
                    {"training_seed": self.seed, "checkpoint": label, **row}
                    for row in rows
                )
        finally:
            self.model.load_state_dict(final_model, strict=True)
            self.policy.load_state_dict(final_policy, strict=True)

        report = {
            "training_seed": self.seed,
            "training_env_steps": self.total_env_steps,
            "training_episodes": self.total_training_episodes,
            "training_budget_env_steps": self.training_budget_env_steps,
            "base_checkpoint_path": self.base_checkpoint_path,
            "test_seed_base": self.test_seed_base,
            "test_trials_per_scenario": self.test_trials,
            "observation_spec": OBSERVATION_SPEC,
            "reward_mode": self.reward_mode,
            "checkpoints": reports,
            "episodes": all_rows,
        }
        write_json(self.res_dir / "independent_test.json", report)
        episode_fields = [
            "training_seed", "checkpoint", "scenario", "scenario_index", "trial_index", "trial_seed",
            "steps", "collision", "near_collision", "min_distance", "min_ttc",
            "ttc_risk", "conv", "latency_ms", "mean_waypoint_delta",
            "scene_perturbation",
        ]
        with open(self.res_dir / "independent_test_episodes.csv", "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=episode_fields)
            writer.writeheader()
            for row in all_rows:
                output = dict(row)
                output["scene_perturbation"] = json.dumps(
                    output.get("scene_perturbation"), ensure_ascii=False
                )
                writer.writerow({key: output.get(key) for key in episode_fields})
        print(
            f"[Test] Independent cohort: {len(all_rows)} episodes "
            f"(seed base={self.test_seed_base}, {self.test_trials}/scenario)"
        )
        for label, summary in reports.items():
            print(
                f"[Test:{label}] score={summary['selection_score']:.4f} "
                f"collision={summary['collision_rate']:.3f} "
                f"near={summary['near_collision_rate']:.3f} "
                f"conv={summary['convergence_rate']:.3f} "
                f"latency={summary['mean_latency_ms']:.3f}ms"
            )

    def evaluate(self, scenarios=None, max_steps=60):
        """Evaluate trained policy on multiple scenarios; return per-scenario dicts."""
        if scenarios is None:
            scenarios = list(SCENARIOS.keys())
        rows, _ = evaluate_policy(
            self.model, self.policy, DEVICE,
            scenarios=scenarios, trials_per_scenario=1,
            seed_base=20261200, max_steps=max_steps,
            latency_budget_ms=self.latency_budget_ms,
        )
        print("\n[Eval] Trained policy results:")
        print("  {:>15} | {:>5} | {:>5} | {:>7} | {:>8} | {:>8}".format("Scenario","Steps","Coll","Conv%","MinTTC","AvgWPD"))
        print("  " + "-" * 55)
        results = {}
        for row in rows:
            sc = row["scenario"]
            coll = "Y" if row["collision"] else "."
            results[sc] = {
                "steps": row["steps"],
                "collision": row["collision"],
                "conv": round(row["conv"], 2),
                "minttc": round(row["min_ttc"], 3),
                "avgwpd": round(row.get("mean_waypoint_delta", 0.0), 4),
            }
            conv_pct = results[sc]["conv"]
            print(f"  {sc:<15} | {row['steps']:>5d} | {coll:>5} | {round(conv_pct,1):>7.1f} | {round(results[sc]['minttc'],3):>8.3f} | {results[sc]['avgwpd']:>8.4f}")
        return results

def main():
    ap = argparse.ArgumentParser(description="REINFORCE training for bargaining")
    ap.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "configs", "train_rl.yaml"), help="YAML config file")
    pre = ap.parse_known_args()
    try:
        from common.io import load_config_from_yaml
        cfg = load_config_from_yaml(pre[0].config) or {}
    except Exception as e:
        cfg = {}
        print(f"[Config] Warning: failed to load config: {e}")
    ap.add_argument("--epochs", type=int, default=cfg.get("epochs", 100))
    ap.add_argument("--lr", type=float, default=cfg.get("lr", 3e-4))
    ap.add_argument("--gamma", type=float, default=cfg.get("gamma", 0.95))
    ap.add_argument("--entropy-coef", type=float, default=cfg.get("entropy_coef", 0.05))
    ap.add_argument("--reward-mode", choices=REWARD_MODES, default=cfg.get("reward_mode", "ttc_calibrated"), help="Shared PPO/REINFORCE reward formula")
    ap.add_argument("--ep-per-epoch", type=int, default=cfg.get("ep_per_epoch", 5))
    ap.add_argument("--total-env-steps", type=int, default=cfg.get("total_env_steps"),
                    help="Stop after exactly this many simulator transitions")
    ap.add_argument("--eval-step-interval", type=int, default=cfg.get("eval_step_interval"),
                    help="Validation interval in transitions when --total-env-steps is set")
    ap.add_argument("--eval-interval", type=int, default=cfg.get("eval_interval", 10), help="Epochs between in-training evals; best picked by eval score")
    ap.add_argument("--validation-trials", type=int, default=cfg.get("validation_trials", 5))
    ap.add_argument("--validation-seed-base", type=int, default=cfg.get("validation_seed_base", 20261100))
    ap.add_argument("--test-trials", type=int, default=cfg.get("test_trials", 25))
    ap.add_argument("--test-seed-base", type=int, default=cfg.get("test_seed_base", 20261200))
    ap.add_argument("--latency-budget-ms", type=float, default=cfg.get("latency_budget_ms", 50.0))
    ap.add_argument("--scenarios", nargs="+", default=cfg.get("scenarios", ["intersection","chase","merge","head_on"]),
                    choices=list(SCENARIOS))
    ap.add_argument("--base-sl-checkpoint", type=str, default=cfg.get("base_sl_checkpoint", None), help="Complete SL checkpoint used to initialize the same bargaining model and policy mean")
    ap.add_argument("--load-policy", type=str, default=cfg.get("load_policy", None), help="Legacy option for loading an RL policy-only checkpoint")
    ap.add_argument("--seed", type=int, default=cfg.get("seed", None), help="Fix random seed for reproducibility")
    args = ap.parse_args()
    if args.epochs < 1 or args.ep_per_epoch < 1:
        ap.error("--epochs and --ep-per-epoch must both be positive.")
    if args.total_env_steps is not None and args.total_env_steps < 1:
        ap.error("--total-env-steps must be positive.")
    if args.eval_step_interval is not None and args.eval_step_interval < 1:
        ap.error("--eval-step-interval must be positive.")
    if args.eval_interval < 1 or args.validation_trials < 1 or args.test_trials < 1:
        ap.error("Evaluation interval and validation/test trial counts must be positive.")
    if args.seed is not None:
        import random
        random.seed(args.seed)           # 数据/环境侧
        np.random.seed(args.seed)        # 环境/奖励侧
        torch.manual_seed(args.seed + 1) # 网络/策略侧（与奖励基线解耦）
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + 1)
        print(f"[Seed] Fixed RNG seed = {args.seed} (np=env, torch=net+1)")
    if args.base_sl_checkpoint and args.load_policy:
        ap.error("Use only one of --base-sl-checkpoint or --load-policy.")
    
    print("=" * 55)
    print("  REINFORCE: Self-Play Bargaining Training")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Gamma: {args.gamma}")
    print(f"  Scenarios: {args.scenarios}")
    print("=" * 55)
    print()
    
    trainer = REINFORCETrainer(lr=args.lr, gamma=args.gamma, entropy_coef=args.entropy_coef,
                               reward_mode=args.reward_mode)
    trainer.seed = args.seed
    trainer.eval_interval = args.eval_interval
    trainer.validation_trials = args.validation_trials
    trainer.validation_seed_base = args.validation_seed_base
    trainer.test_trials = args.test_trials
    trainer.test_seed_base = args.test_seed_base
    trainer.latency_budget_ms = args.latency_budget_ms
    if args.base_sl_checkpoint:
        load_sl_checkpoint_as_policy(
            trainer.model, trainer.policy, args.base_sl_checkpoint, map_location=DEVICE
        )
        trainer.base_checkpoint_path = str(Path(args.base_sl_checkpoint).resolve())
        print(f"[Init] Loaded complete SL model and policy mean from {trainer.base_checkpoint_path}")
    elif args.load_policy:
        ckpt = torch.load(args.load_policy, map_location=DEVICE)
        if "offer_decoder" in ckpt:
            load_sl_checkpoint_as_policy(
                trainer.model, trainer.policy, ckpt, map_location=DEVICE
            )
            trainer.base_checkpoint_path = str(Path(args.load_policy).resolve())
            print(f"[Init] Loaded complete SL model and policy mean from {trainer.base_checkpoint_path}")
        else:
            trainer.policy.load_state_dict(ckpt)
            print(f"[Joint] Loaded policy from {args.load_policy}")
    else:
        print("[Init] No base SL checkpoint supplied; bargaining components start from random initialization.")
    trainer.train(
        n_epochs=args.epochs,
        ep_per_epoch=args.ep_per_epoch,
        scenarios=args.scenarios,
        total_env_steps=args.total_env_steps,
        eval_step_interval=args.eval_step_interval,
    )
    trainer.evaluate_test_checkpoints()
    print()
    print(f"Done. Results: {trainer.res_dir}")
if __name__ == "__main__":
    main()
