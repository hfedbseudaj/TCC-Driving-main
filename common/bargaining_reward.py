"""Reward functions shared by the policy-gradient training scripts."""

from __future__ import annotations


REWARD_MODES = ("ttc_calibrated", "ppo_legacy", "reinforce_legacy")


def bargaining_step_reward(
    converged: bool,
    speed_a: float,
    speed_b: float,
    n_rounds: int,
    dist: float,
    ttc: float,
    mode: str = "ttc_calibrated",
    collision: bool = False,
) -> float:
    """Return a per-step reward under an explicitly named shared formula.

    ``ttc_calibrated`` is the current common PPO/REINFORCE comparison reward.
    The two legacy modes remain available to reproduce their previous code
    paths; they should not be mixed in an algorithm comparison.
    """
    if mode not in REWARD_MODES:
        raise ValueError(f"Unknown reward mode {mode!r}; choose one of {REWARD_MODES}")

    if mode == "ttc_calibrated":
        reward = 0.5 if converged else 0.0
        reward += 0.3 * min(float(ttc) / 3.0, 1.0)
        if ttc < 1.5:
            reward -= 0.5
        avg_speed = (float(speed_a) + float(speed_b)) / 2.0
        reward += 0.02 * min(avg_speed / 10.0, 1.0)
        reward -= 0.05 * int(n_rounds) / 3.0
        return reward - (2.0 if collision else 0.0)

    if mode == "ppo_legacy":
        reward = 1.0 if converged else 0.0
        if ttc < 3.0 and not converged:
            reward -= 2.0
        if ttc < 1.5 and not converged:
            reward -= 3.0
        avg_speed = (float(speed_a) + float(speed_b)) / 2.0
        reward += 0.02 * min(avg_speed / 10.0, 1.0)
        reward += 0.02 * max(0.0, 1.0 - float(dist) / 50.0)
        reward -= 0.1 * int(n_rounds) / 3.0
        return reward - (2.0 if collision else 0.0)

    reward = 1.0 if converged else 0.0
    if collision:
        reward -= 2.0
    avg_speed = (float(speed_a) + float(speed_b)) / 2.0
    reward += 0.05 * min(avg_speed / 10.0, 1.0)
    reward += 0.01 * min(float(dist) / 50.0, 1.0)
    reward -= 0.05 * int(n_rounds) / 3.0
    return reward


def terminal_reward(collision: bool) -> float:
    """Common terminal outcome, applied once after each simulation episode."""
    return -10.0 if collision else 1.0
