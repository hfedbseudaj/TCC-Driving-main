# -*- coding: utf-8 -*-
"""
train_bargaining_ppo.py - PPO training with curriculum learning and entropy scheduling.

Improvements over REINFORCE training (train_bargaining_rl.py):
  1. PPO (clipped surrogate objective) - much more stable updates
  2. GAE (Generalized Advantage Estimation) - better advantage estimates
  3. Curriculum learning - progressive scenario difficulty
  4. Adaptive entropy scheduling - decays exploration over time

Usage:
    python train_bargaining_ppo.py --epochs=200
    python train_bargaining_ppo.py --curriculum
    python train_bargaining_ppo.py --resume <policy.pth>

Output:
    Result/ppo_train_YYYYMMDD_HHMMSS/
"""

import os, sys, json, math, datetime, csv, copy, random, time, hashlib
from pathlib import Path
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from common.kinematic_bargaining_env import (
    OBSERVATION_SPEC,
    SCENARIOS as SHARED_SCENARIOS,
    SimState as SharedSimState,
    apply_adjusted_waypoints,
    bev_from_snapshots,
    snapshot_state,
)
from common.deterministic_bev import deterministic_bev
from common.bargaining_reward import REWARD_MODES, bargaining_step_reward, terminal_reward
from common.bargaining_evaluation import evaluate_policy
from common.experiment_protocol import environment_metadata, project_fingerprint, write_json
from common.sl_checkpoint import load_sl_checkpoint_as_policy

from opencood.models.comm_modules.rubinstein_bargaining import (
    RubinsteinBargaining, compute_pairwise_ttc,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Curriculum: (scenario, epochs) or None = all
CURRICULUM = [
    ("intersection", 40),   # easy, longer for stable base
    ("chase", 20),          # medium
    ("merge", 20),          # medium
    ("head_on", 40),        # hard, shorter with smoother transition
    (None, 80),            # all mixed, same total
]

SCENARIOS = {
    "intersection":{"a":{"start":[-30,-2],"target":[0,0],"speed":8},"b":{"start":[-2,-30],"target":[0,0],"speed":8}},
    "chase":{"a":{"start":[-25,0],"target":[15,0],"speed":10},"b":{"start":[-15,0],"target":[15,0],"speed":6}},
    "merge":{"a":{"start":[-25,-3],"target":[5,0],"speed":8},"b":{"start":[-30,0],"target":[10,0],"speed":7}},
    "head_on":{"a":{"start":[-20,0],"target":[10,0],"speed":8},"b":{"start":[20,0],"target":[-10,0],"speed":8}},
}

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
        hdg=torch.tensor(self.hdg,dtype=torch.float32,device=DEVICE)
        wps=torch.stack([torch.from_numpy(self.wps[0]).to(DEVICE),torch.from_numpy(self.wps[1]).to(DEVICE)],dim=0)
        ttc_m=compute_pairwise_ttc(pos,vel,hdg)
        mt=ttc_m[ttc_m.isfinite()].min().item() if ttc_m.isfinite().any() else 100.0
        return {"positions":pos,"velocities":vel,"headings":hdg,"waypoints":wps,
                "record_len":torch.tensor([2],device=DEVICE),"ttc":mt,"spd":list(self.spd)}


# Replace the legacy local simulator above with the shared state contract.
SCENARIOS = SHARED_SCENARIOS
SimState = SharedSimState


class StochasticPolicy(nn.Module):
    def __init__(self, state_dim=64, offer_dim=4, hidden_dim=128, init_log_std=-1.0):
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, offer_dim),
        )
        self.log_std = nn.Parameter(torch.full((offer_dim,), init_log_std))
    def forward(self, state, deterministic=False):
        mean = self.mean_net(state)
        if deterministic:
            return torch.sigmoid(mean), None
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        action_logit = dist.rsample()
        action = torch.sigmoid(action_logit)
        log_prob = dist.log_prob(action_logit).sum(dim=-1)
        log_prob -= torch.log(action * (1 - action) + 1e-8).sum(dim=-1)
        return action, log_prob
    def get_log_prob(self, state, action):
        mean = self.mean_net(state)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        action_logit = torch.log(action / (1 - action + 1e-8) + 1e-8)
        log_prob = dist.log_prob(action_logit).sum(dim=-1)
        log_prob -= torch.log(action * (1 - action) + 1e-8).sum(dim=-1)
        return log_prob
    def get_entropy(self, state):
        mean = self.mean_net(state)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        return dist.entropy().sum(dim=-1).mean()


def compute_step_reward(converged, speed_a, speed_b, n_rounds, dist, ttc):
    return bargaining_step_reward(
        converged, speed_a, speed_b, n_rounds, dist, ttc,
        mode="ppo_legacy",
    )


def compute_step_reward_fixed(converged, speed_a, speed_b, n_rounds, dist, ttc):
    """
    Reward-fix variant: aligns the convergence signal with protocol behavior
    (paper 4.4 wording: "aligning the convergence signal with protocol behavior").

    Mismatches in the original reward that this variant addresses:
      1) converged (+1.0, dominant) is a *process* signal of the bargaining
         protocol, not the driving objective.  The protocol is designed so that
         low TTC -> small delta -> easier acceptance -> more likely to converge;
         tying a big reward/punishment to it forces the policy to "fake
         convergence" instead of learning safe behavior.  Demoted to +0.5.
      2) Safety is scored continuously via TTC (large TTC = safe gap = reward)
         instead of hard -2/-3 thresholds on "dangerous AND not converged".
      3) Removed the anti-safety distance-attraction term (+0.02*(1-dist/50)),
         which rewarded the two vehicles for being close together.
      4) Kept speed efficiency and round efficiency, scaled down.

    NOTE: reconstructed from the paper's description; the exact formula is not
          published.  Validation = post-training Table-4 style kinematic rerun.
    """
    return bargaining_step_reward(
        converged, speed_a, speed_b, n_rounds, dist, ttc,
        mode="ttc_calibrated",
    )



class PPOTrainer:
    def __init__(self, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, ppo_epochs=10,
                 entropy_init=0.1, entropy_min=0.001,
                 use_curriculum=True, reward_fix=False,
                 transition_window=15, curriculum=None,
                 reward_mode=None, minibatch_size=64):
        self.gamma = gamma
        self.lr = lr
        self.reward_mode = reward_mode or "ttc_calibrated"
        if self.reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {self.reward_mode!r}")
        self.reward_fn = compute_step_reward_fixed if self.reward_mode == "ttc_calibrated" else compute_step_reward
        self.reward_fix = self.reward_mode == "ttc_calibrated"
        self.minibatch_size = max(1, int(minibatch_size))
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.entropy_init = entropy_init
        self.entropy_min = entropy_min
        self.use_curriculum = use_curriculum
        self.transition_window = transition_window  # smooth transition epochs between stages
        self.curriculum = curriculum if curriculum is not None else CURRICULUM
        self.epoch = 0
        self.total_epochs = 1
        self.seed = None
        self.base_checkpoint_path = None

        # Model
        self.model = RubinsteinBargaining({
            "state_dim":64,"offer_dim":4,"hidden_dim":128,"max_rounds":3,
            "discount_alpha":0.3,"delta_min":0.05,"delta_max":0.99,"waypoint_offset_scale":2.0,
        }).to(DEVICE)
        self.model.train()
        for p in self.model.state_encoder.parameters():
            p.requires_grad = True

        # Policy + Value
        self.policy = StochasticPolicy(
            state_dim=64, offer_dim=4, hidden_dim=128, init_log_std=0.0
        ).to(DEVICE)
        self.value_net = nn.Sequential(
            nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 1),
        ).to(DEVICE)

        # Optimizer
        self.optimizer = optim.Adam([
            {"params": self.model.state_encoder.parameters(), "lr": lr},
            {"params": self.policy.parameters(), "lr": lr},
            {"params": self.value_net.parameters(), "lr": lr},
        ])

        # Logging
        self.log = {"epoch":[],"env_steps":[],"episodes":[],"reward":[],"collision_pct":[],"convergence_pct":[],"policy_loss":[],"value_loss":[],"entropy":[],"kl":[]}
        self.best_reward = -float("inf")
        # Best is selected only from a fixed, perturbed, multi-scenario
        # validation set. The independent test seeds are never used here.
        self.best_eval_score = float("inf")
        self.best_eval_epoch = -1
        self.eval_interval = 10
        self.total_env_steps = 0
        self.total_training_episodes = 0
        self.training_budget_env_steps = None
        self.eval_step_interval = None
        self.validation_trials = 5
        self.validation_seed_base = 20261100
        self.test_trials = 25
        self.test_seed_base = 20261200
        self.latency_budget_ms = 50.0
        self.validation_history = []
        self.validation_rows = []

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.res_dir = Path(PROJECT_ROOT) / "Result" / f"ppo_train_{ts}_{os.getpid()}"
        (self.res_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        print(f"[PPO] Results -> {self.res_dir}")

    def collect_episode(self, scenario, max_steps=50):
        sim = SimState(scenario)
        trajectory = []

        while not sim.finished and sim.step < max_steps:
            state = sim.get_state(device=DEVICE)
            feat = deterministic_bev(state, device=DEVICE)
            with torch.no_grad():
                state_enc = self.model.state_encoder(feat)
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
                values = self.value_net(state_enc).squeeze(-1)

            adj_wps = out["adjusted_waypoints"]
            apply_adjusted_waypoints(sim, state, adj_wps)

            dist = np.linalg.norm(sim.pos[0] - sim.pos[1])
            conv = out["metadata"][0]["converged"] if "metadata" in out else False
            reward = bargaining_step_reward(
                converged=conv,
                collision=False,
                speed_a=sim.spd[0],
                speed_b=sim.spd[1],
                n_rounds=out["metadata"][0].get("n_rounds", 0),
                dist=dist,
                ttc=state["ttc"],
                mode=self.reward_mode,
            )

            trajectory.append({
                "state_snapshot": snapshot_state(state),
                "action": init_offers.detach().cpu(),
                "log_prob": log_prob.detach().cpu(),
                "reward": reward,
                "value": values.detach().cpu(),
                "converged": bool(conv),
                "done": False,
            })
            sim.step_sim()

        if trajectory:
            trajectory[-1]["reward"] += terminal_reward(sim.collision)
            # The final transition must not bootstrap from the next episode.
            trajectory[-1]["done"] = True
        return trajectory, sim

    def process_trajectories(self, trajectories):
        snapshots, all_a, all_lp, all_ret, all_adv = [], [], [], [], []
        for traj in trajectories:
            if not traj:
                continue
            values = torch.stack([t["value"] for t in traj]).to(DEVICE)  # (T, 2)
            rewards = torch.tensor([t["reward"] for t in traj], dtype=torch.float32, device=DEVICE)
            dones = torch.tensor([t["done"] for t in traj], dtype=torch.float32, device=DEVICE)
            advantages = torch.zeros_like(values)
            gae = torch.zeros(2, dtype=torch.float32, device=DEVICE)
            for step in range(len(traj) - 1, -1, -1):
                if step == len(traj) - 1 or bool(dones[step].item()):
                    next_value = torch.zeros(2, dtype=torch.float32, device=DEVICE)
                else:
                    next_value = values[step + 1]
                mask = 1.0 - dones[step]
                delta = rewards[step] + self.gamma * next_value * mask - values[step]
                gae = delta + self.gamma * self.gae_lambda * mask * gae
                advantages[step] = gae
            returns = advantages + values
            snapshots.extend(t["state_snapshot"] for t in traj)
            all_a.extend(t["action"] for t in traj)
            all_lp.extend(t["log_prob"] for t in traj)
            all_ret.extend(returns.detach().cpu())
            all_adv.extend(advantages.detach().cpu())
        if not snapshots:
            return None, None, None, None, None
        actions = torch.stack(all_a).to(DEVICE)  # (T, 2, 4)
        old_log_probs = torch.stack(all_lp).to(DEVICE)  # (T, 2)
        returns = torch.stack(all_ret).to(DEVICE)  # (T, 2)
        advantages = torch.stack(all_adv).to(DEVICE)  # (T, 2)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        return snapshots, actions, old_log_probs, returns, advantages

    def compute_gae(self, rewards, values, dones):
        adv = torch.zeros_like(rewards)
        gae = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            if t == len(rewards) - 1:
                delta = rewards[t] - values[t]
            else:
                delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            adv[t] = gae
        return adv

    def ppo_update(self, snapshots, actions, old_log_probs, returns, advantages):
        if snapshots is None:
            return {}
        total_policy = 0.0; total_value = 0.0; total_entropy = 0.0; total_kl = 0.0
        update_count = 0
        if self.training_budget_env_steps:
            progress = min(1.0, float(self.total_env_steps) / self.training_budget_env_steps)
        else:
            progress = float(self.epoch) / max(1, self.total_epochs)
        ec = max(self.entropy_min, self.entropy_init * (1 - progress))
        n_steps = len(snapshots)
        stop_updates = False
        for _ in range(self.ppo_epochs):
            permutation = torch.randperm(n_steps)
            for start in range(0, n_steps, self.minibatch_size):
                indices = permutation[start:start + self.minibatch_size].tolist()
                features = bev_from_snapshots([snapshots[i] for i in indices], device=DEVICE)
                states = self.model.state_encoder(features).view(len(indices), 2, -1).reshape(-1, 64)
                batch_actions = actions[indices].reshape(-1, 4)
                batch_old_log_probs = old_log_probs[indices].reshape(-1)
                batch_returns = returns[indices].reshape(-1)
                batch_advantages = advantages[indices].reshape(-1)

                lp = self.policy.get_log_prob(states, batch_actions)
                entropy = self.policy.get_entropy(states)
                ratio = torch.exp(lp - batch_old_log_probs.detach())
                obj1 = ratio * batch_advantages
                obj2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                p_loss = -torch.min(obj1, obj2).mean()
                values = self.value_net(states).squeeze(-1)
                v_loss = F.mse_loss(values, batch_returns)
                loss = p_loss + 0.5 * v_loss - ec * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.model.state_encoder.parameters())
                    + list(self.policy.parameters())
                    + list(self.value_net.parameters()), 0.5
                )
                self.optimizer.step()

                with torch.no_grad():
                    kl = float((batch_old_log_probs - lp).mean().item())
                total_policy += p_loss.item(); total_value += v_loss.item()
                total_entropy += entropy.item(); total_kl += kl; update_count += 1
                if kl > 0.02:
                    stop_updates = True
                    break
            if stop_updates:
                break
        n = max(update_count, 1)
        return {"policy_loss": total_policy / n, "value_loss": total_value / n,
                "entropy": total_entropy / n, "kl": total_kl / n}

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
                "train_bargaining_ppo.py",
                "common/kinematic_bargaining_env.py",
                "common/deterministic_bev.py",
                "common/bargaining_reward.py",
                "common/bargaining_evaluation.py",
                "common/sl_checkpoint.py",
                "opencood/models/comm_modules/rubinstein_bargaining.py",
            ]),
            "env": environment_metadata(torch, np),
        }

    def train(self, n_epochs=200, ep_per_epoch=5, total_env_steps=None,
              eval_step_interval=None):
        self.total_epochs = n_epochs
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
        print(f"[PPO] Training: {n_epochs} epochs, {ep_per_epoch} ep/epoch")
        if self.training_budget_env_steps is not None:
            print("  Exact training transition budget: {}".format(
                self.training_budget_env_steps
            ))
        print(f"  Entropy: {self.entropy_init} -> {self.entropy_min}, Curriculum: {self.use_curriculum}")
        hdr = "  {:>5} | {:>8} | {:>6} | {:>6} | {:>9} | {:>8} | {:>6}"
        print(hdr.format("Ep", "Reward", "Coll%", "Conv%", "P_Loss", "Entropy", "KL"))
        print("  " + "-" * 60)

        curriculum_stage = 0
        stage_epoch = 0

        for ep in range(n_epochs):
            self.epoch = ep

            # Curriculum learning
            if self.use_curriculum:
                stage_epoch += 1
                if curriculum_stage < len(self.curriculum) and stage_epoch > self.curriculum[curriculum_stage][1]:
                    curriculum_stage += 1
                    stage_epoch = 0
                curr_scenario = self.curriculum[min(curriculum_stage, len(self.curriculum)-1)][0]
                scenarios = [curr_scenario] if curr_scenario else list(SCENARIOS)
                weights = None
                if curriculum_stage > 0 and stage_epoch < self.transition_window:
                    prev_scenario = self.curriculum[curriculum_stage - 1][0]
                    mix = stage_epoch / self.transition_window
                    if prev_scenario and curr_scenario:
                        scenarios = [prev_scenario, curr_scenario]
                        weights = [1.0 - mix, mix]
            else:
                scenarios = list(SCENARIOS)
                weights = None

            trajectories = []
            ep_collisions = []
            for _ in range(ep_per_epoch):
                if (
                    self.training_budget_env_steps is not None
                    and self.total_env_steps >= self.training_budget_env_steps
                ):
                    break
                sc = np.random.choice(scenarios, p=weights) if weights else np.random.choice(scenarios)
                max_steps = 50
                if self.training_budget_env_steps is not None:
                    max_steps = min(
                        max_steps,
                        self.training_budget_env_steps - self.total_env_steps,
                    )
                traj, sim = self.collect_episode(sc, max_steps)
                if not traj:
                    raise RuntimeError("PPO collected an empty trajectory")
                trajectories.append(traj)
                ep_collisions.append(sim.collision)
                self.total_env_steps += len(traj)
                self.total_training_episodes += 1

            if not trajectories:
                break

            snapshots, actions, log_probs, returns, adv = self.process_trajectories(trajectories)
            stats = self.ppo_update(snapshots, actions, log_probs, returns, adv)
            total_r = sum(sum(t["reward"] for t in traj) for traj in trajectories) / len(trajectories)
            avg_coll = np.mean(ep_collisions) * 100
            episode_convergence = [
                100.0 * np.mean([step["converged"] for step in traj])
                for traj in trajectories if traj
            ]
            avg_conv = float(np.mean(episode_convergence)) if episode_convergence else 0.0

            self.log["epoch"].append(ep); self.log["env_steps"].append(self.total_env_steps)
            self.log["episodes"].append(self.total_training_episodes)
            self.log["reward"].append(total_r)
            self.log["collision_pct"].append(avg_coll)
            self.log["convergence_pct"].append(avg_conv)
            self.log["policy_loss"].append(stats.get("policy_loss",0))
            self.log["value_loss"].append(stats.get("value_loss",0))
            self.log["entropy"].append(stats.get("entropy",0))
            self.log["kl"].append(stats.get("kl",0))

            if ep % max(1, n_epochs // 10) == 0 or ep == n_epochs - 1:
                print(hdr.format(ep, f"{total_r:.3f}", f"{avg_coll:.1f}", f"{avg_conv:.1f}",
                      f"{stats.get('policy_loss',0):.5f}", f"{stats.get('entropy',0):.5f}",
                      f"{stats.get('kl',0):.5f}"))

            if total_r > self.best_reward:
                self.best_reward = total_r
                torch.save(self.policy.state_dict(),
                           str(self.res_dir / "checkpoints" / "train_best_policy.pth"))
                torch.save({
                    "epoch": ep,
                    "best_reward": self.best_reward,
                    "policy": self.policy.state_dict(),
                    "value_net": self.value_net.state_dict(),
                    "model": self.model.state_dict(),
                    "scenarios": SCENARIOS,
                    "curriculum": self.curriculum,
                    **self._checkpoint_metadata(),
                }, str(self.res_dir / "checkpoints" / "train_best_checkpoint_full.pth"))

            # Select checkpoints by the shared safety/convergence/latency score.
            if self.training_budget_env_steps is None:
                should_validate = ep % self.eval_interval == 0 or ep == n_epochs - 1
            else:
                should_validate = (
                    self.total_env_steps >= next_validation_step
                    or self.total_env_steps >= self.training_budget_env_steps
                    or ep == n_epochs - 1
                )
            if should_validate:
                validation_rows, validation_summary = self.evaluate_for_checkpoint_selection()
                eval_score = validation_summary["selection_score"]
                history_row = {
                    "epoch": ep,
                    "training_env_steps": self.total_env_steps,
                    **validation_summary,
                }
                self.validation_history.append(history_row)
                self.validation_rows.extend(
                    {
                        "training_seed": self.seed,
                        "epoch": ep,
                        "training_env_steps": self.total_env_steps,
                        **row,
                    }
                    for row in validation_rows
                )
                self._save_validation_history()
                if eval_score < self.best_eval_score:
                    self.best_eval_score = eval_score
                    self.best_eval_epoch = ep
                    torch.save(self.policy.state_dict(),
                               str(self.res_dir / "checkpoints" / "best_policy.pth"))
                    torch.save({
                        "epoch": ep,
                        "best_eval_score": self.best_eval_score,
                        "validation_metrics": validation_summary,
                        "policy": self.policy.state_dict(),
                        "value_net": self.value_net.state_dict(),
                        "model": self.model.state_dict(),
                        "scenarios": SCENARIOS,
                        "curriculum": self.curriculum,
                        **self._checkpoint_metadata(),
                    }, str(self.res_dir / "checkpoints" / "best_checkpoint_full.pth"))
                    write_json(self.res_dir / "eval_best.json", {
                        "epoch": ep,
                        "score": eval_score,
                        "validation_summary": validation_summary,
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
                "PPO exhausted max epochs at {} / {} environment steps".format(
                    self.total_env_steps, self.training_budget_env_steps
                )
            )

        torch.save(self.policy.state_dict(), str(self.res_dir / "checkpoints" / "final_policy.pth"))
        torch.save({
            "epoch": self.log["epoch"][-1] if self.log["epoch"] else -1,
            "best_reward": self.best_reward,
            "best_eval_score": self.best_eval_score,
            "best_eval_epoch": self.best_eval_epoch,
            "policy": self.policy.state_dict(),
            "value_net": self.value_net.state_dict(),
            "model": self.model.state_dict(),
            "scenarios": SCENARIOS,
            "curriculum": self.curriculum,
            **self._checkpoint_metadata(),
        }, str(self.res_dir / "checkpoints" / "final_checkpoint_full.pth"))
        self._save_log()
        self._plot_curves()
        print(f"[PPO] Done. Best avg reward: {self.best_reward:.3f}")
        print(f"[PPO] Done. Best validation score: {self.best_eval_score:.4f} @ epoch {self.best_eval_epoch} (eval-interval={self.eval_interval})")

    def _save_log(self):
        with open(str(self.res_dir / "ppo_log.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.log.keys()))
            w.writeheader()
            for i in range(len(self.log["epoch"])):
                w.writerow({k: self.log[k][i] for k in self.log})
        json.dump({k: v for k, v in self.log.items()},
                  open(str(self.res_dir / "ppo_log.json"), "w"), default=float)
        source_sha256 = {}
        for relative in (
            "train_bargaining_ppo.py",
            "opencood/models/comm_modules/rubinstein_bargaining.py",
            "common/experiment_protocol.py",
            "common/kinematic_bargaining_env.py",
            "common/deterministic_bev.py",
            "common/bargaining_reward.py",
            "common/bargaining_evaluation.py",
            "common/sl_checkpoint.py",
        ):
            source_path = Path(PROJECT_ROOT) / relative
            if source_path.is_file():
                source_sha256[relative] = hashlib.sha256(source_path.read_bytes()).hexdigest()
        json.dump({"seed": self.seed,
                   "training_seed": self.seed,
                   "reward_fix": self.reward_fn is compute_step_reward_fixed,
                   "reward_mode": self.reward_mode,
                   "observation_spec": OBSERVATION_SPEC,
                   "base_checkpoint_path": self.base_checkpoint_path,
                   "base_checkpoint_sha256": self._checkpoint_metadata()["base_checkpoint_sha256"],
                   "lr": getattr(self, "lr", None),
                   "gamma": self.gamma, "gae_lambda": self.gae_lambda,
                   "clip_epsilon": self.clip_epsilon, "ppo_epochs": self.ppo_epochs,
                   "entropy_init": self.entropy_init, "entropy_min": self.entropy_min,
                   "use_curriculum": self.use_curriculum,
                   "transition_window": self.transition_window,
                   "curriculum": self.curriculum,
                   "eval_interval": self.eval_interval,
                   "eval_step_interval": self.eval_step_interval,
                   "training_env_steps": self.total_env_steps,
                   "training_episodes": self.total_training_episodes,
                   "training_budget_env_steps": self.training_budget_env_steps,
                   "validation_trials_per_scenario": self.validation_trials,
                   "validation_seed_base": self.validation_seed_base,
                   "latency_budget_ms": self.latency_budget_ms,
                   "validation_best_epoch": self.best_eval_epoch,
                   "validation_best_score": self.best_eval_score,
                   "test_seed_base": self.test_seed_base,
                   "test_trials_per_scenario": self.test_trials,
                   "checkpoint_selection_rule": "minimum five-component score on fixed paired multi-scenario validation episodes",
                   "validation_trial_count": len(SCENARIOS) * self.validation_trials,
                   "source_sha256": source_sha256,
                   "validation_score_weights": {
                       "collision_rate": 100.0,
                       "near_collision_rate": 25.0,
                       "ttc_risk": 20.0,
                       "convergence_failure": 10.0,
                       "latency_fraction_of_50ms_budget": 5.0,
                   },
                   "source_fingerprint": project_fingerprint(PROJECT_ROOT, list(source_sha256)),
                   "env": environment_metadata(torch, np)},
                  open(str(self.res_dir / "config.json"), "w"), indent=2, default=str)
        print(f"[Log] Saved to {self.res_dir}")

    def _plot_curves(self):
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            e = self.log["epoch"]
            d = [("reward","Reward","blue"),("collision_pct","Collision %","red"),
                 ("policy_loss","Policy Loss","green"),("value_loss","Value Loss","orange"),
                 ("entropy","Entropy","purple"),("kl","KL Div","brown")]
            for ax, (k, lbl, c) in zip(axes.flat, d):
                ax.plot(e, self.log[k], color=c); ax.set_xlabel("Epoch")
                ax.set_ylabel(lbl); ax.grid(alpha=0.3)
            plt.suptitle("PPO Training Curves"); plt.tight_layout()
            plt.savefig(str(self.res_dir / "ppo_curves.png"), dpi=150); plt.close()
            print(f"[Plot] -> {self.res_dir / 'ppo_curves.png'}")
        except Exception as ex: print(f"[Plot] {ex}")

    def evaluate(self, max_steps=60):
        """Run one paired diagnostic trial per scenario with the shared input."""
        rows, _ = evaluate_policy(
            self.model, self.policy, DEVICE,
            scenarios=list(SCENARIOS.keys()),
            trials_per_scenario=1,
            seed_base=20261200,
            max_steps=max_steps,
            latency_budget_ms=self.latency_budget_ms,
        )
        results = {}
        for row in rows:
            scenario = row["scenario"]
            results[scenario] = {
                "steps": row["steps"],
                "collision": row["collision"],
                "conv": round(row["conv"], 2),
                "minttc": round(row["min_ttc"], 3),
                "avgwpd": round(row["mean_waypoint_delta"], 4),
            }
            print(
                f"  {scenario:<15} steps={row['steps']:>3d} "
                f"coll={'Y' if row['collision'] else '.'} "
                f"conv={row['conv']:>6.1f}% ttc={row['min_ttc']:>6.3f}s "
                f"wpd={row['mean_waypoint_delta']:.4f}"
            )
        return results

    def evaluate_for_checkpoint_selection(self, max_steps=60):
        """Use the same perturbed multi-scenario score as REINFORCE."""
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

    def _save_validation_history(self):
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
                output["scene_perturbation"] = json.dumps(
                    output["scene_perturbation"], ensure_ascii=False
                )
                writer.writerow({key: output.get(key) for key in episode_fields})

    def evaluate_test_checkpoints(self):
        """Report best and final checkpoints on a separate, fixed test cohort."""
        checkpoint_dir = self.res_dir / "checkpoints"
        candidates = {
            "best": checkpoint_dir / "best_checkpoint_full.pth",
            "final": checkpoint_dir / "final_checkpoint_full.pth",
        }
        final_model = copy.deepcopy(self.model.state_dict())
        final_policy = copy.deepcopy(self.policy.state_dict())
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
        with open(self.res_dir / "independent_test.json", "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, default=str)
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


def main():
    import argparse
    ap = argparse.ArgumentParser(description="PPO training for bargaining")
    ap.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "configs", "train_ppo.yaml"), help="YAML config file")
    pre = ap.parse_known_args()
    try:
        from common.io import load_config_from_yaml
        cfg = load_config_from_yaml(pre[0].config) or {}
    except Exception as e:
        cfg = {}
        print(f"[Config] Warning: failed to load config: {e}")
    ap.add_argument("--epochs", type=int, default=cfg.get("epochs", 200))
    ap.add_argument("--lr", type=float, default=cfg.get("lr", 3e-4))
    ap.add_argument("--no-curriculum", action="store_true", default=None, help="Disable curriculum learning")
    ap.add_argument("--reward-fix", action="store_true", default=cfg.get("reward_fix", False), help="Use reward-fix variant (convergence-signal aligned)")
    ap.add_argument("--reward-mode", choices=REWARD_MODES, default=cfg.get("reward_mode"), help="Shared PPO/REINFORCE reward formula")
    ap.add_argument("--ep-per-epoch", type=int, default=cfg.get("ep_per_epoch", 5))
    ap.add_argument("--total-env-steps", type=int, default=cfg.get("total_env_steps"),
                    help="Stop after exactly this many simulator transitions")
    ap.add_argument("--eval-step-interval", type=int, default=cfg.get("eval_step_interval"),
                    help="Validation interval in transitions when --total-env-steps is set")
    ap.add_argument("--gamma", type=float, default=cfg.get("gamma", 0.99))
    ap.add_argument("--gae-lambda", type=float, default=cfg.get("gae_lambda", 0.95))
    ap.add_argument("--clip-epsilon", type=float, default=cfg.get("clip_epsilon", 0.2))
    ap.add_argument("--ppo-epochs", type=int, default=cfg.get("ppo_epochs", 10))
    ap.add_argument("--entropy-init", type=float, default=cfg.get("entropy_init", 0.1))
    ap.add_argument("--entropy-min", type=float, default=cfg.get("entropy_min", 0.001))
    ap.add_argument("--transition-window", type=int, default=cfg.get("transition_window", 15))
    ap.add_argument("--eval-interval", type=int, default=cfg.get("eval_interval", 10), help="Epochs between perturbed multi-scenario validation runs")
    ap.add_argument("--validation-trials", type=int, default=cfg.get("validation_trials", 5), help="Validation trials per scenario at each checkpoint-selection interval")
    ap.add_argument("--validation-seed-base", type=int, default=cfg.get("validation_seed_base", 20261100), help="Base seed reserved for checkpoint-selection validation")
    ap.add_argument("--test-trials", type=int, default=cfg.get("test_trials", 25), help="Independent test trials per scenario after training")
    ap.add_argument("--test-seed-base", type=int, default=cfg.get("test_seed_base", 20261200), help="Base seed reserved for independent final testing")
    ap.add_argument("--latency-budget-ms", type=float, default=cfg.get("latency_budget_ms", 50.0))
    ap.add_argument("--base-sl-checkpoint", type=str, default=cfg.get("base_sl_checkpoint"), help="Complete SL checkpoint used to initialize the same bargaining model and policy mean")
    ap.add_argument("--seed", type=int, default=cfg.get("seed", None), help="Fix random seed for reproducibility")
    args = ap.parse_args()
    if args.epochs < 1 or args.ep_per_epoch < 1:
        ap.error("--epochs and --ep-per-epoch must both be positive.")
    if args.total_env_steps is not None and args.total_env_steps < 1:
        ap.error("--total-env-steps must be positive.")
    if args.eval_step_interval is not None and args.eval_step_interval < 1:
        ap.error("--eval-step-interval must be positive.")
    if args.eval_interval < 1 or args.validation_trials < 1 or args.test_trials < 1 or args.latency_budget_ms <= 0:
        ap.error("Evaluation interval and validation/test trial counts must be positive; latency budget must exceed zero.")
    if args.seed is not None:
        import random
        random.seed(args.seed)           # 数据/环境侧
        np.random.seed(args.seed)        # 环境/奖励侧
        torch.manual_seed(args.seed + 1) # 网络/策略侧（与奖励基线解耦）
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + 1)
        print(f"[Seed] Fixed RNG seed = {args.seed} (np=env, torch=net+1)")
    use_curriculum = (not args.no_curriculum) if args.no_curriculum is not None else cfg.get("use_curriculum", True)
    curriculum = [tuple(x) if isinstance(x, list) else x for x in cfg.get("curriculum", CURRICULUM)]
    print("=" * 60); print("  PPO: Stable Self-Play Bargaining Training")
    reward_mode = (
        "ttc_calibrated"
        if args.reward_fix
        else (args.reward_mode or "ttc_calibrated")
    )
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Curriculum: {use_curriculum}, Reward: {reward_mode}")
    print("=" * 60); print()
    trainer = PPOTrainer(lr=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
                         clip_epsilon=args.clip_epsilon, ppo_epochs=args.ppo_epochs,
                         entropy_init=args.entropy_init, entropy_min=args.entropy_min,
                         use_curriculum=use_curriculum, reward_fix=args.reward_fix,
                         transition_window=args.transition_window, curriculum=curriculum,
                         reward_mode=reward_mode)
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
    else:
        print("[Init] No base SL checkpoint supplied; bargaining components start from random initialization.")
    trainer.train(
        n_epochs=args.epochs,
        ep_per_epoch=args.ep_per_epoch,
        total_env_steps=args.total_env_steps,
        eval_step_interval=args.eval_step_interval,
    )
    trainer.evaluate_test_checkpoints()
    print()
    print(f"Done. Results: {trainer.res_dir}")

if __name__ == "__main__":
    main()
