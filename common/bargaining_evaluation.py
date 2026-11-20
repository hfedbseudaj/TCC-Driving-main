"""Common paired validation for REINFORCE and PPO bargaining policies."""

from __future__ import annotations

import random
import time

import numpy as np
import torch

from common.deterministic_bev import deterministic_bev
from common.experiment_protocol import (
    derive_trial_seed,
    episode_metrics,
    perturb_scenario,
    selection_score,
)
from common.kinematic_bargaining_env import (
    SCENARIOS,
    SimState,
    apply_adjusted_waypoints,
)
from opencood.models.comm_modules.rubinstein_bargaining import compute_pairwise_ttc


def evaluate_policy(
    model,
    policy,
    device,
    scenarios=None,
    trials_per_scenario=5,
    seed_base=20261100,
    max_steps=60,
    latency_budget_ms=50.0,
):
    """Run deterministic policy actions on a paired, perturbed scene set.

    The same function and scene-seed derivation are used by PPO and
    REINFORCE. Only validation seeds should be passed here; final test seeds
    are reserved for post-training reporting.
    """
    scenario_names = list(scenarios or SCENARIOS.keys())
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    cpu_torch_rng = torch.get_rng_state()
    cuda_torch_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    model_training = model.training
    policy_training = policy.training
    model.eval()
    policy.eval()
    rows = []
    try:
        for scenario_index, scenario in enumerate(scenario_names):
            for trial_index in range(int(trials_per_scenario)):
                trial_seed = derive_trial_seed(
                    seed_base, scenario, trial_index, scenario_names
                )
                sim = SimState(scenario)
                perturbation = perturb_scenario(sim, trial_seed)
                min_distance = float("inf")
                min_ttc = float("inf")
                near_collision = False
                converged_steps = 0
                latency_samples = []
                waypoint_deltas = []

                while not sim.finished and sim.step < max_steps:
                    state = sim.get_state(device=device)
                    distance = float(torch.norm(
                        state["positions"][0] - state["positions"][1]
                    ).item())
                    pairwise_ttc = compute_pairwise_ttc(
                        state["positions"], state["velocities"], state["headings"]
                    )
                    finite_ttc = pairwise_ttc[torch.isfinite(pairwise_ttc)]
                    current_ttc = float(finite_ttc.min().item()) if finite_ttc.numel() else 99.0
                    min_distance = min(min_distance, distance)
                    min_ttc = min(min_ttc, current_ttc)
                    if distance < 4.0 or current_ttc < 1.0:
                        near_collision = True

                    features = deterministic_bev(state, device=device)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        encoded = model.encode_states(features)[0]
                        offers, _ = policy(encoded, deterministic=True)
                        output = model(
                            spatial_features=encoded,
                            waypoints=state["waypoints"],
                            positions=state["positions"],
                            velocities=state["velocities"],
                            headings=state["headings"],
                            record_len=state["record_len"],
                            initial_offers=offers,
                        )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(device)
                    latency_samples.append((time.perf_counter() - started) * 1000.0)
                    if output["metadata"][0].get("converged", False):
                        converged_steps += 1

                    adjusted = output["adjusted_waypoints"]
                    waypoint_deltas.append(float((adjusted - state["waypoints"]).norm(dim=-1).mean().item()))
                    apply_adjusted_waypoints(sim, state, adjusted)
                    sim.step_sim()

                final_distance = float(np.linalg.norm(sim.pos[0] - sim.pos[1]))
                min_distance = min(min_distance, final_distance)
                if final_distance < 4.0:
                    near_collision = True
                rows.append({
                    **episode_metrics(
                        collision=sim.collision,
                        near_collision=near_collision and not sim.collision,
                        min_distance=min_distance if np.isfinite(min_distance) else 0.0,
                        min_ttc=min_ttc if np.isfinite(min_ttc) else 99.0,
                        conv_rate=100.0 * converged_steps / max(sim.step, 1),
                        latency_ms=float(np.mean(latency_samples)) if latency_samples else 0.0,
                        steps=sim.step,
                    ),
                    "scenario": scenario,
                    "scenario_index": scenario_index,
                    "trial_index": trial_index,
                    "trial_seed": trial_seed,
                    "scene_perturbation": perturbation,
                    "mean_waypoint_delta": float(np.mean(waypoint_deltas)) if waypoint_deltas else 0.0,
                })
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_torch_rng)
        if cuda_torch_rng is not None:
            torch.cuda.set_rng_state_all(cuda_torch_rng)
        model.train(model_training)
        policy.train(policy_training)

    return rows, selection_score(rows, latency_budget_ms=latency_budget_ms)
