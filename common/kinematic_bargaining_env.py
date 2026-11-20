"""Shared deterministic kinematic environment and observation contract.

SL, REINFORCE and PPO must see the same scenario state and BEV construction.
The simulation remains synthetic; this module does not claim to provide a
sensor-derived BEV.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List

import numpy as np
import torch


OBSERVATION_SPEC = {
    "name": "paired_kinematic_bev_v1",
    "shape": [2, 256, 48, 48],
    "dtype": "float32",
    "channels": "ego occupancy in 0:128; other-agent occupancy in 128:256 for each agent row",
    "source": "positions, velocities, headings from the shared kinematic simulator",
}

SCENARIOS = {
    "intersection": {
        "a": {"start": [-30, -2], "target": [0, 0], "speed": 8},
        "b": {"start": [-2, -30], "target": [0, 0], "speed": 8},
    },
    "chase": {
        "a": {"start": [-25, 0], "target": [15, 0], "speed": 6},
        "b": {"start": [-15, 0], "target": [15, 0], "speed": 10},
    },
    "merge": {
        "a": {"start": [-25, -3], "target": [5, 0], "speed": 8},
        "b": {"start": [-30, 0], "target": [10, 0], "speed": 7},
    },
    "head_on": {
        "a": {"start": [-20, 0], "target": [10, 0], "speed": 8},
        "b": {"start": [20, 0], "target": [-10, 0], "speed": 8},
    },
}


class SimState:
    """Two-vehicle simulator shared by all bargaining training paradigms."""

    def __init__(self, scenario: str = "intersection", dt: float = 0.1):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario {scenario!r}; expected one of {tuple(SCENARIOS)}")
        cfg = SCENARIOS[scenario]
        self.dt = float(dt)
        self.name = scenario
        self.step = 0
        self.collision = False
        self.finished = False
        a, b = cfg["a"], cfg["b"]
        self.pos = [np.asarray(a["start"], dtype=np.float32), np.asarray(b["start"], dtype=np.float32)]
        target_a = np.asarray(a["target"], dtype=np.float32) - self.pos[0]
        target_b = np.asarray(b["target"], dtype=np.float32) - self.pos[1]
        self.spd = [float(a["speed"]), float(b["speed"])]
        self.hdg = [math.atan2(target_a[1], target_a[0]), math.atan2(target_b[1], target_b[0])]
        self._update_wps()

    @property
    def speeds(self):
        return self.spd

    def _update_wps(self):
        self.wps = [np.zeros((10, 2), dtype=np.float32) for _ in range(2)]
        for vehicle in range(2):
            cos_h, sin_h = math.cos(self.hdg[vehicle]), math.sin(self.hdg[vehicle])
            for index in range(10):
                time_s = (index + 1) * self.dt
                self.wps[vehicle][index] = [
                    self.spd[vehicle] * time_s * cos_h,
                    self.spd[vehicle] * time_s * sin_h,
                ]

    def adjust(self, vehicle: int, dx: float, dy: float):
        speed_delta = float(dx) * math.cos(self.hdg[vehicle]) + float(dy) * math.sin(self.hdg[vehicle])
        self.spd[vehicle] = max(0.3, min(15.0, self.spd[vehicle] + speed_delta * 2.0))

    def step_sim(self):
        if self.finished:
            return
        for vehicle in range(2):
            self.pos[vehicle] += np.asarray([
                self.spd[vehicle] * self.dt * math.cos(self.hdg[vehicle]),
                self.spd[vehicle] * self.dt * math.sin(self.hdg[vehicle]),
            ], dtype=np.float32)
        self._update_wps()
        self.step += 1
        if float(np.linalg.norm(self.pos[0] - self.pos[1])) < 1.0:
            self.collision = True
            self.finished = True
        if self.step > 200 or any(abs(p[0]) > 40 for p in self.pos) or any(abs(p[1]) > 40 for p in self.pos):
            self.finished = True

    def get_state(self, device="cpu") -> Dict[str, torch.Tensor]:
        from opencood.models.comm_modules.rubinstein_bargaining import compute_pairwise_ttc

        positions = torch.as_tensor(np.asarray(self.pos), dtype=torch.float32, device=device)
        velocities = torch.as_tensor([
            [self.spd[v] * math.cos(self.hdg[v]), self.spd[v] * math.sin(self.hdg[v])]
            for v in range(2)
        ], dtype=torch.float32, device=device)
        headings = torch.as_tensor(self.hdg, dtype=torch.float32, device=device)
        waypoints = torch.as_tensor(np.stack(self.wps), dtype=torch.float32, device=device)
        ttc_matrix = compute_pairwise_ttc(positions, velocities, headings)
        finite = ttc_matrix[torch.isfinite(ttc_matrix)]
        min_ttc = float(finite.min().item()) if finite.numel() else 100.0
        speeds = [float(self.spd[0]), float(self.spd[1])]
        return {
            "positions": positions,
            "velocities": velocities,
            "headings": headings,
            "waypoints": waypoints,
            "record_len": torch.tensor([2], dtype=torch.long, device=device),
            "ttc": min_ttc,
            # Keep both names while old callers are migrated; they point to
            # the same values and are removed from new call sites.
            "speeds": speeds,
            "spd": speeds,
        }


def snapshot_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Store only the compact physical state needed to regenerate the BEV."""
    return {
        key: state[key].detach().to(device="cpu", dtype=torch.float32).clone()
        for key in ("positions", "velocities", "headings")
    }


def state_from_snapshot(snapshot: Dict[str, torch.Tensor], device="cpu") -> Dict[str, torch.Tensor]:
    return {key: snapshot[key].to(device=device, dtype=torch.float32) for key in ("positions", "velocities", "headings")}


def bev_from_snapshots(snapshots: Iterable[Dict[str, torch.Tensor]], device="cpu") -> torch.Tensor:
    from common.deterministic_bev import deterministic_bev

    feature_batches: List[torch.Tensor] = []
    for snapshot in snapshots:
        feature_batches.append(deterministic_bev(state_from_snapshot(snapshot), device=device))
    if not feature_batches:
        return torch.empty((0, 256, 48, 48), dtype=torch.float32, device=device)
    return torch.cat(feature_batches, dim=0)


def apply_adjusted_waypoints(sim: SimState, state: Dict[str, torch.Tensor], adjusted_waypoints: torch.Tensor):
    """Apply each vehicle's own first adjusted waypoint to the shared simulator.

    Synthetic training provides per-agent waypoint rows. Keeping this update
    in one place prevents SL, PPO and REINFORCE from silently applying
    different fallback offsets to the second vehicle.
    """
    base_waypoints = state["waypoints"]
    if adjusted_waypoints.shape != base_waypoints.shape:
        raise ValueError(
            "Per-agent waypoint update requires matching shapes; "
            f"got adjusted={tuple(adjusted_waypoints.shape)} and "
            f"base={tuple(base_waypoints.shape)}."
        )
    if adjusted_waypoints.shape[0] != len(sim.pos):
        raise ValueError(
            f"Expected one adjusted waypoint row per vehicle ({len(sim.pos)}), "
            f"got {adjusted_waypoints.shape[0]}."
        )
    for vehicle in range(len(sim.pos)):
        delta = (
            adjusted_waypoints[vehicle, 0] - base_waypoints[vehicle, 0]
        ).detach().to(device="cpu").numpy()
        sim.adjust(vehicle, float(delta[0]), float(delta[1]))
