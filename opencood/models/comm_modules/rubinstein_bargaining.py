# -*- coding: utf-8 -*-
# Rubinstein Bargaining Module for Cooperative Driving
# 
# Implements a differentiable alternating-offers bargaining protocol
# between multiple autonomous vehicles, grounded in Rubinstein's (1982)
# strategic bargaining theory.
#
# Key innovations:
#   1. TTC-based discount factor: δ = 1 - exp(-α * TTC)
#      - Binds impatience to physical collision risk
#      - Creates a natural bridge between game theory and physics
#   2. Differentiable bargaining rounds in a learned embedding space
#   3. Utility estimation via neural network that learns social preferences
#
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List


def compute_pairwise_ttc(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    headings: torch.Tensor,
) -> torch.Tensor:
    """
    Compute pairwise Time-To-Collision between agents.
    
    Args:
        positions: (N, 2) agent positions in BEV coordinates
        velocities: (N, 2) agent velocity vectors (m/s)
        headings: (N,) agent heading angles (radians)
    
    Returns:
        ttc_matrix: (N, N) pairwise TTC, INF if no collision course
    """
    N = positions.shape[0]
    device = positions.device
    
    rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)
    rel_vel = velocities.unsqueeze(1) - velocities.unsqueeze(0)
    dist = torch.norm(rel_pos, dim=-1)
    rel_pos_unit = F.normalize(rel_pos, dim=-1, eps=1e-8)
    # With rel_pos = p_i - p_j and rel_vel = v_i - v_j, agents are closing
    # when their relative velocity points opposite to the relative position.
    closing_speed = -torch.sum(rel_vel * rel_pos_unit, dim=-1)
    
    mask = closing_speed > 0.1
    ttc = torch.full_like(dist, float('inf'))
    ttc[mask] = dist[mask] / closing_speed[mask]
    ttc.fill_diagonal_(float('inf'))
    return ttc


class TTCBasedDiscount(nn.Module):
    """
    Compute the discount factor δ from Time-To-Collision.
    δ = 1 - exp(-α * TTC)
    Large TTC → δ → 1 (patient). Small TTC → δ → 0 (impatient).
    """
    def __init__(self, alpha: float = 0.3, delta_min: float = 0.05, delta_max: float = 0.99):
        super().__init__()
        self.alpha = alpha
        self.delta_min = delta_min
        self.delta_max = delta_max
    
    def forward(self, ttc: torch.Tensor) -> torch.Tensor:
        valid_ttc = torch.where(
            torch.isfinite(ttc), ttc, torch.full_like(ttc, 1e6))
        delta = 1.0 - torch.exp(-self.alpha * valid_ttc)
        delta = torch.clamp(delta, self.delta_min, self.delta_max)
        if delta.dim() == 2:
            delta.fill_diagonal_(1.0)
        return delta


class UtilityEstimator(nn.Module):
    """Estimates utility u(agent_state, offer) → ℝ."""
    def __init__(self, state_dim: int = 64, offer_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        )
        self.offer_encoder = nn.Sequential(
            nn.Linear(offer_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        )
        self.joint_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, agent_state: torch.Tensor, offer: torch.Tensor) -> torch.Tensor:
        state_emb = self.state_encoder(agent_state)
        offer_emb = self.offer_encoder(offer)
        joint = torch.cat([state_emb, offer_emb], dim=-1)
        return self.joint_encoder(joint)


class OfferDecoder(nn.Module):
    """Decodes agent state → bargaining offer in (0,1) via sigmoid.
    Offer dims: [speed_factor, lateral_shift, time_headway, yield_willingness]"""
    def __init__(self, state_dim: int = 64, offer_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, offer_dim),
        )
    
    def forward(self, agent_state: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decoder(agent_state))


def alternating_offers_protocol(
    offers, discount_factors, utility_estimator, agent_states,
    max_rounds=3, return_all_offers=False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Alternating-offers bargaining. Each round a different proposer updates
    offer toward consensus. Acceptance threshold = δ^remaining · u(status_quo).
    Converges when non-proposers accept.
    """
    N = offers.shape[0]
    if N < 2:
        raise ValueError("Alternating-offers bargaining requires at least two agents")
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least one")
    device = offers.device
    delta = discount_factors.diag() if discount_factors.dim() == 2 else discount_factors
    
    status_quo_utility = utility_estimator(agent_states, offers).squeeze(-1)
    current_offers = offers.clone()
    all_offers = [offers.clone()] if return_all_offers else []
    agreement = torch.zeros(N, dtype=torch.bool, device=device)
    converged = False
    n_updates = 0
    
    for round_idx in range(max_rounds):
        proposer_idx = round_idx % N
        remaining = max_rounds - round_idx
        threshold = (delta ** remaining) * status_quo_utility

        # The paper's differentiable protocol first moves the active
        # proposer's offer toward the mean of offers accepted at this round,
        # then checks whether the non-proposers accept. The previous ordering
        # returned on acceptance before applying this update, which made the
        # common one-round consensus path a no-op.
        utilities = utility_estimator(agent_states, current_offers).squeeze(-1)
        accepted = utilities >= threshold
        accepted_offers = current_offers[accepted]
        if accepted_offers.shape[0] > 0:
            target = accepted_offers.mean(dim=0)
            # Keep the discount factor in the computation graph. This existing
            # interpolation scale is preserved; δ and the accepted mean now
            # both influence the negotiated offer.
            step = 0.3 * (1.0 - delta[proposer_idx])
            updated_proposal = (
                current_offers[proposer_idx] * (1.0 - step) + target * step
            )
            current_offers = torch.cat(
                (
                    current_offers[:proposer_idx],
                    updated_proposal.unsqueeze(0),
                    current_offers[proposer_idx + 1 :],
                ),
                dim=0,
            )
            n_updates += 1

        # Re-evaluate after the proposal update so convergence corresponds to
        # the offers that the caller will actually apply to the waypoints.
        utilities = utility_estimator(agent_states, current_offers).squeeze(-1)
        accepted = utilities >= threshold
        
        non_proposers = torch.ones(N, dtype=torch.bool, device=device)
        non_proposers[proposer_idx] = False
        
        if accepted[non_proposers].all():
            agreement = accepted
            converged = True
        
        if return_all_offers:
            all_offers.append(current_offers.clone())
        if converged:
            break
    
    if not converged:
        agreement = accepted
    
    metadata = {
        'n_rounds': round_idx + 1,
        'converged': converged,
        'delta_values': delta.detach().cpu(),
        'n_offer_updates': n_updates,
    }
    if return_all_offers:
        metadata['all_offers'] = torch.stack(all_offers)
    return current_offers, agreement, metadata


class RubinsteinBargaining(nn.Module):
    """
    Main differentiable Rubinstein bargaining module.
    
    Input: BEV features, waypoints, positions, velocities, headings, record_len
    Process: Encode → TTC → δ → Generate offers → Alternating-offers → Waypoint adj.
    Output: Adjusted waypoints + consensus weights + bargaining diagnostics
    
    Integrates into CoLMDriver pipeline between perception fusion and planning.
    """
    def __init__(self, args: dict):
        super().__init__()
        self.state_dim = args.get('state_dim', 64)
        self.offer_dim = args.get('offer_dim', 4)
        self.max_rounds = args.get('max_rounds', 3)
        # The improved project follows the paper's sequential N-player
        # protocol by default; select ``global`` only for the legacy ablation.
        self.n_player_protocol = args.get('n_player_protocol', 'sequential_pairwise')
        if self.n_player_protocol not in {'global', 'sequential_pairwise'}:
            raise ValueError(
                "n_player_protocol must be 'global' or 'sequential_pairwise', "
                f"got {self.n_player_protocol!r}.")
        
        self.discount = TTCBasedDiscount(
            alpha=args.get('discount_alpha', 0.3),
            delta_min=args.get('delta_min', 0.05),
            delta_max=args.get('delta_max', 0.99),
        )
        
        self.state_encoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, self.state_dim),
            nn.LayerNorm(self.state_dim),
            nn.ReLU(),
        )
        
        self.offer_decoder = OfferDecoder(
            state_dim=self.state_dim, offer_dim=self.offer_dim,
            hidden_dim=args.get('hidden_dim', 128))
        self.utility_estimator = UtilityEstimator(
            state_dim=self.state_dim, offer_dim=self.offer_dim,
            hidden_dim=args.get('hidden_dim', 128))
        
        self.offer_to_waypoint = nn.Sequential(
            nn.Linear(self.offer_dim, 32), nn.ReLU(), nn.Linear(32, 4))
        self.waypoint_offset_scale = args.get('waypoint_offset_scale', 2.0)

    def encode_states(self, spatial_features: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """Accept either BEV feature maps or pre-encoded agent states."""
        if spatial_features.dim() == 2:
            if spatial_features.shape[-1] != self.state_dim:
                raise ValueError(
                    f"Expected encoded bargaining states with dim {self.state_dim}, "
                    f"got {spatial_features.shape[-1]}.")
            return spatial_features, 'encoded_state'
        if spatial_features.dim() == 4:
            return self.state_encoder(spatial_features), 'bev_feature'
        raise ValueError(
            "RubinsteinBargaining expects spatial_features as either "
            f"(N, {self.state_dim}) states or (N, C, H, W) BEV maps, "
            f"got shape {tuple(spatial_features.shape)}.")
    
    def forward(self, spatial_features, waypoints, positions, velocities, headings, record_len,
                initial_offers=None, record_pair_trace=False):
        B = record_len.shape[0]
        device = spatial_features.device
        bargaining_states, input_mode = self.encode_states(spatial_features)

        if waypoints.dim() != 3 or waypoints.shape[-1] != 2:
            raise ValueError(
                "waypoints must have shape (batch_or_agents, horizon, 2), "
                f"got {tuple(waypoints.shape)}."
            )
        # OpenCOOD supplies one reference trajectory per scene; the shared
        # kinematic experiments supply an individual trajectory per agent.
        # Infer the convention from the existing batch dimensions.
        waypoints_per_agent = waypoints.shape[0] == bargaining_states.shape[0]
        if not waypoints_per_agent and waypoints.shape[0] != B:
            raise ValueError(
                "waypoints must have one row per scene or per agent; "
                f"got {waypoints.shape[0]} rows for {B} scenes and "
                f"{bargaining_states.shape[0]} agents."
            )

        if initial_offers is not None:
            if initial_offers.shape != (bargaining_states.shape[0], self.offer_dim):
                raise ValueError(
                    "initial_offers must have one offer row per encoded agent: "
                    f"expected {(bargaining_states.shape[0], self.offer_dim)}, "
                    f"got {tuple(initial_offers.shape)}."
                )
            initial_offers = initial_offers.to(device=device, dtype=bargaining_states.dtype)
        
        cumsum = torch.cumsum(record_len, dim=0)
        batch_splits = torch.tensor_split(
            torch.arange(spatial_features.shape[0], device=device), cumsum[:-1].cpu())
        
        adj_wps, all_off, all_acc, all_w, all_meta = [], [], [], [], []
        for b in range(B):
            n_agents = record_len[b].item()
            idx = batch_splits[b]
            st = bargaining_states[idx]
            pos = positions[idx] if positions.dim() == 2 else positions[b]
            vel = velocities[idx] if velocities.dim() == 2 else velocities[b]
            hdg = headings[idx] if headings.dim() == 1 else headings[b]
            
            ttc_mat = compute_pairwise_ttc(pos, vel, hdg)
            delta = self.discount(ttc_mat).min(dim=1)[0]
            init_off = self.offer_decoder(st) if initial_offers is None else initial_offers[idx]
            
            if n_agents >= 2 and self.n_player_protocol == 'sequential_pairwise':
                from common.nplayer_bargaining import sequential_pairwise_bargaining

                # Keep one trajectory per CAV in the synthetic experiments;
                # retain the scene-level ego trajectory convention used by
                # the OpenCOOD integration.
                agent_waypoints = (
                    waypoints[idx].clone()
                    if waypoints_per_agent
                    else waypoints[b:b + 1].expand(n_agents, -1, -1).clone()
                )
                pairwise_output = sequential_pairwise_bargaining(
                    self,
                    st,
                    agent_waypoints,
                    pos,
                    vel,
                    hdg,
                    initial_offers=init_off,
                    record_pair_trace=record_pair_trace,
                )
                agreed_off = pairwise_output['bargaining_offers']
                accept = pairwise_output['acceptance_mask']
                meta = pairwise_output['metadata']
                negotiated_waypoints = pairwise_output['adjusted_waypoints']
                scene_waypoints = (
                    negotiated_waypoints
                    if waypoints_per_agent
                    else negotiated_waypoints[0:1]
                )
            elif n_agents >= 2:
                agreed_off, accept, meta = alternating_offers_protocol(
                    init_off, delta, self.utility_estimator, st, self.max_rounds)
                if waypoints_per_agent:
                    scene_waypoints = waypoints[idx].clone()
                    offsets = self.offer_to_waypoint(agreed_off) * self.waypoint_offset_scale
                    scene_waypoints[:, 0, :] += offsets[:, :2]
                    if scene_waypoints.shape[1] > 1:
                        scene_waypoints[:, 1, :] += offsets[:, 2:4]
                else:
                    scene_waypoints = waypoints[b:b + 1].clone()
                    offset = self.offer_to_waypoint(agreed_off[0:1]) * self.waypoint_offset_scale
                    scene_waypoints[0, 0, 0] += offset[0, 0]
                    scene_waypoints[0, 0, 1] += offset[0, 1]
                    if scene_waypoints.shape[1] > 1:
                        scene_waypoints[0, 1, 0] += offset[0, 2]
                        scene_waypoints[0, 1, 1] += offset[0, 3]
            else:
                agreed_off, accept = init_off, torch.ones(1, dtype=torch.bool, device=device)
                meta = {'n_rounds': 0, 'converged': True}
                scene_waypoints = (
                    waypoints[idx].clone()
                    if waypoints_per_agent
                    else waypoints[b:b + 1].clone()
                )
                offset = self.offer_to_waypoint(agreed_off[0:1]) * self.waypoint_offset_scale
                scene_waypoints[0, 0, 0] += offset[0, 0]
                scene_waypoints[0, 0, 1] += offset[0, 1]
                if scene_waypoints.shape[1] > 1:
                    scene_waypoints[0, 1, 0] += offset[0, 2]
                    scene_waypoints[0, 1, 1] += offset[0, 3]
            
            adj_wps.append(scene_waypoints)
            all_off.append(agreed_off)
            all_acc.append(accept)
            with torch.no_grad():
                w = self.utility_estimator(st, agreed_off).mean().item()
            all_w.append(torch.tensor([[w]], device=device))
            meta['input_mode'] = input_mode
            meta['min_ttc'] = (
                torch.min(ttc_mat[torch.isfinite(ttc_mat)]).detach().cpu()
                if torch.isfinite(ttc_mat).any()
                else torch.tensor(float('inf'))
            )
            all_meta.append(meta)
        
        return {
            'adjusted_waypoints': torch.cat(adj_wps, dim=0),
            'bargaining_offers': torch.cat(all_off, dim=0),
            'acceptance_mask': torch.cat(all_acc, dim=0),
            'bargaining_weights': torch.cat(all_w, dim=0),
            'metadata': all_meta,
        }
