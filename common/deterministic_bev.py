# -*- coding: utf-8 -*-
"""Deterministic BEV renderer for the kinematic bargaining simulators.

Defect-4 fix (SL input-target disconnection):
  The SL training / SL-eval pipelines previously fed torch.randn(2,256,48,48)
  as "spatial features", which carried no information about the actual vehicle
  state (positions / velocities / headings). Consequently the frozen
  state_encoder output could not encode the state and the decoder had nothing
  to learn from (util_loss collapsed to 0).

This module renders a deterministic (2,256,48,48) BEV map from the kinematic
state. Each agent row contains both vehicles: channels 0..127 are the ego
vehicle and channels 128..255 are the other vehicle. This fixes the previous
layout bug where each row contained only one vehicle and the shared encoder
never saw the pairwise geometry together.

Deterministic: identical state -> identical map (no RNG involved).
"""
import math

import torch


def _render_layer(grid_pos, heading_unit, speed_norm,
                  grid=48, extent=30.0, blob_sigma=1.75, tail_len=5):
    """Render a vectorized occupancy layer for one vehicle on CPU."""
    cell = 2.0 * extent / grid
    pos = torch.as_tensor(grid_pos, dtype=torch.float32, device="cpu")
    heading = torch.as_tensor(heading_unit, dtype=torch.float32, device="cpu")
    speed_norm = float(speed_norm)
    axis = torch.arange(grid, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    xc = (pos[0] + extent) / cell
    yc = (pos[1] + extent) / cell
    peak = 1.0 + speed_norm
    d2 = (xx - xc).square() + (yy - yc).square()
    layer = peak * torch.exp(-d2 / (2.0 * blob_sigma ** 2))
    in_bounds = (xc >= 0) & (xc < grid) & (yc >= 0) & (yc < grid)
    layer = layer * in_bounds.to(layer.dtype)

    # Preserve the original one-cell motion tail while keeping rendering
    # inexpensive enough to regenerate PPO minibatch observations.
    for k in range(1, tail_len + 1):
        tail = pos + heading * (float(k) * cell)
        xk = float(((tail[0] + extent) / cell).item())
        yk = float(((tail[1] + extent) / cell).item())
        if 0.0 <= xk < grid and 0.0 <= yk < grid:
            layer[int(yk), int(xk)] += peak * (0.45 ** k)
    return layer


def deterministic_bev(state, device="cpu"):
    """Build a (2, 256, 48, 48) BEV map from a kinematic state dict.

    state must contain (torch tensors):
      positions  (2, 2)
      velocities (2, 2)
      headings   (2,)
    """
    pos = state["positions"].float().to("cpu")
    vel = state["velocities"].float().to("cpu")
    speed = vel.norm(dim=-1)  # (2,)
    head_unit = torch.zeros_like(vel)
    for v in range(2):
        s = speed[v].item()
        if s > 1e-3:
            head_unit[v] = vel[v] / s
        else:
            h = float(state["headings"][v].item())
            head_unit[v] = torch.tensor([math.cos(h), math.sin(h)],
                                        dtype=vel.dtype)
    speed_norm = torch.clamp(speed / 15.0, 0.0, 1.0)
    layers = [_render_layer(pos[v], head_unit[v], speed_norm[v].item())
              for v in range(2)]
    bev = torch.zeros(2, 256, 48, 48, dtype=torch.float32, device=device)
    for ego in range(2):
        other = 1 - ego
        bev[ego, :128] = layers[ego].to(device).expand(128, 48, 48)
        bev[ego, 128:] = layers[other].to(device).expand(128, 48, 48)
    return bev
