"""Sequential pairwise N-player extension for TTC bargaining.

The paper's N-player section describes sequential 2-vehicle negotiations,
ordered by collision urgency, with resulting adjustments weighted by 1/TTC.
This module adds that protocol around the existing two-vehicle bargaining
components without changing the legacy ``RubinsteinBargaining.forward`` API.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor, nn


def sequential_pairwise_bargaining(
    model: nn.Module,
    spatial_features: Tensor,
    waypoints: Tensor,
    positions: Tensor,
    velocities: Tensor,
    headings: Tensor,
    initial_offers: Tensor | None = None,
) -> Dict[str, Any]:
    """Run the model's two-agent bargaining protocol sequentially over N agents.

    Args:
        model: A ``RubinsteinBargaining`` instance.
        spatial_features: ``(N, state_dim)`` encoded states or ``(N,C,H,W)`` BEV maps.
        waypoints: Per-agent local future waypoints with shape ``(N,H,2)``.
        positions: Per-agent positions with shape ``(N,2)``.
        velocities: Per-agent velocity vectors with shape ``(N,2)``.
        headings: Per-agent heading angles with shape ``(N,)``.

    Returns:
        A dictionary parallel to the legacy module output, plus the ordered
        ``pair_schedule`` in ``metadata``. Non-conflicting agents retain their
        decoded initial adjustment. For agents in multiple conflicts, each
        pair's waypoint adjustment contributes with weight ``1 / TTC``.

    Pair selection is discrete because it sorts the current TTC values. The
    bargaining and weighted adjustment operations remain differentiable.
    """
    from opencood.models.comm_modules.rubinstein_bargaining import (
        alternating_offers_protocol,
        compute_pairwise_ttc,
    )

    if positions.ndim != 2 or positions.shape[-1] != 2:
        raise ValueError(f"positions must have shape (N, 2), got {tuple(positions.shape)}")
    n_agents = int(positions.shape[0])
    if n_agents < 2:
        raise ValueError("Sequential pairwise bargaining requires at least two agents")
    if velocities.shape != positions.shape:
        raise ValueError("velocities must have the same (N, 2) shape as positions")
    if headings.shape != (n_agents,):
        raise ValueError(f"headings must have shape ({n_agents},), got {tuple(headings.shape)}")
    if waypoints.ndim != 3 or waypoints.shape[0] != n_agents or waypoints.shape[-1] != 2:
        raise ValueError(f"waypoints must have shape (N, H, 2), got {tuple(waypoints.shape)}")

    states, input_mode = model.encode_states(spatial_features)
    if states.ndim != 2 or states.shape[0] != n_agents:
        raise ValueError("encoded states must contain one row per agent")

    pairwise_ttc = compute_pairwise_ttc(positions, velocities, headings)
    pairs = []
    for first in range(n_agents):
        for second in range(first + 1, n_agents):
            ttc = pairwise_ttc[first, second]
            if torch.isfinite(ttc):
                pairs.append((float(ttc.detach().cpu()), first, second))
    pairs.sort(key=lambda row: (row[0], row[1], row[2]))

    if initial_offers is None:
        initial_offers = model.offer_decoder(states)
    else:
        if initial_offers.shape != (n_agents, model.offer_dim):
            raise ValueError(
                "initial_offers must have shape "
                f"({n_agents}, {model.offer_dim}), got {tuple(initial_offers.shape)}"
            )
        initial_offers = initial_offers.to(device=states.device, dtype=states.dtype)
    current_offers = [initial_offers[index] for index in range(n_agents)]
    adjustment_terms = [[] for _ in range(n_agents)]
    acceptance = torch.ones(n_agents, dtype=torch.bool, device=positions.device)
    pair_records = []

    for ttc_value, first, second in pairs:
        pair_indices = torch.tensor([first, second], dtype=torch.long, device=positions.device)
        pair_ttc = pairwise_ttc.index_select(0, pair_indices).index_select(1, pair_indices)
        pair_discount = model.discount(pair_ttc).min(dim=1)[0]
        pair_offers = torch.stack((current_offers[first], current_offers[second]), dim=0)
        negotiated_offers, pair_acceptance, pair_meta = alternating_offers_protocol(
            pair_offers,
            pair_discount,
            model.utility_estimator,
            states.index_select(0, pair_indices),
            max_rounds=model.max_rounds,
        )

        current_offers[first] = negotiated_offers[0]
        current_offers[second] = negotiated_offers[1]
        acceptance[first] &= pair_acceptance[0]
        acceptance[second] &= pair_acceptance[1]

        pair_adjustments = (
            model.offer_to_waypoint(negotiated_offers) * model.waypoint_offset_scale
        )
        # Keep the TTC weighting in the computation graph; only the ordering
        # key and human-readable metadata are detached.
        pair_weight = 1.0 / pair_ttc[0, 1].clamp_min(1e-3)
        adjustment_terms[first].append((pair_adjustments[0], pair_weight))
        adjustment_terms[second].append((pair_adjustments[1], pair_weight))
        pair_records.append({
            "agents": [first, second],
            "ttc_s": ttc_value,
            "priority_weight_1_over_ttc": float(pair_weight.detach().cpu()),
            "rounds": int(pair_meta["n_rounds"]),
            "converged": bool(pair_meta["converged"]),
            "accepted": [bool(pair_acceptance[0]), bool(pair_acceptance[1])],
        })

    final_offers = torch.stack(current_offers, dim=0)
    initial_adjustments = (
        model.offer_to_waypoint(initial_offers) * model.waypoint_offset_scale
    )
    final_adjustments = []
    for agent_index, terms in enumerate(adjustment_terms):
        if not terms:
            final_adjustments.append(initial_adjustments[agent_index])
            continue
        weight_sum = torch.stack([weight for _, weight in terms]).sum().clamp_min(1e-8)
        weighted = torch.stack([adjustment * weight for adjustment, weight in terms]).sum(dim=0)
        final_adjustments.append(weighted / weight_sum)
    final_adjustments = torch.stack(final_adjustments, dim=0)

    adjusted_waypoints = waypoints.clone()
    adjusted_waypoints[:, 0, :] = adjusted_waypoints[:, 0, :] + final_adjustments[:, :2]
    if waypoints.shape[1] > 1:
        adjusted_waypoints[:, 1, :] = adjusted_waypoints[:, 1, :] + final_adjustments[:, 2:4]

    finite_ttc = pairwise_ttc[torch.isfinite(pairwise_ttc)]
    min_ttc = float(finite_ttc.min().detach().cpu()) if finite_ttc.numel() else float("inf")
    global_converged = all(record["converged"] for record in pair_records)
    metadata = {
        "input_mode": input_mode,
        "n_agents": n_agents,
        "n_rounds": sum(record["rounds"] for record in pair_records),
        "converged": global_converged,
        "min_ttc": min_ttc,
        "pair_count": len(pair_records),
        "pair_schedule": pair_records,
    }
    with torch.no_grad():
        weights = model.utility_estimator(states, final_offers)

    return {
        "adjusted_waypoints": adjusted_waypoints,
        "bargaining_offers": final_offers,
        "acceptance_mask": acceptance,
        "bargaining_weights": weights,
        "weighted_waypoint_adjustments": final_adjustments,
        "pairwise_ttc": pairwise_ttc,
        "metadata": metadata,
    }
