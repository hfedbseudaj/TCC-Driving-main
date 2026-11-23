"""Explicit bridge from archived four-dimensional offers to crossing K_e.

The archived offer is not a time contract.  Its learned waypoint head produces
two local waypoint offsets; the shared synthetic controller uses the first
offset's forward projection to change speed by a factor of two.  This bridge
keeps that calculation and derives a tentative entry time. The strict mode
requires a reachable, pair-safe trajectory; an explicitly enabled alternate
projects an unsafe tentative time onto the same finite safe pair family used
by the new decoder. Both raw failures and projection magnitudes are reported.
This is a *new* adapter, not part of the original learned model.
"""

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

from .crossing import CrossingContract
from .physics import (
    VehicleSpec, canonical_approach_plan, legacy_closing_ttc, pair_safe,
)


@dataclass(frozen=True)
class AdapterOutcome:
    success: bool
    reason: str
    contract: Optional[CrossingContract]
    tentative_entry_s: Tuple[float, float]
    old_controller_speed_mps: Tuple[float, float]
    first_waypoint_projection_error_m: Tuple[Optional[float], Optional[float]]
    raw_failure_reason: Optional[str] = None
    selected_entry_s: Optional[Tuple[float, float]] = None
    safety_projection_l1_s: float = 0.0


def adapt_waypoint_offsets_to_crossing_contract(
        first: VehicleSpec, second: VehicleSpec,
        first_offset_xy, second_offset_xy, dt_s: float,
        clearance_s: float, tolerance_s: float = 0.1,
        old_speed_min_mps: float = 0.3,
        old_speed_max_mps: float = 15.0,
        candidate_entry_times: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        allow_safety_projection: bool = False) -> AdapterOutcome:
    """Adapt two already decoded first-waypoint offsets without hiding failures.

    The old synthetic controller's speed command is
    ``clip(v_i + 2 * dot(offset_i, heading_i), .3, 15)``.  Dividing the
    remaining distance by that speed yields a tentative time only; the new
    canonical trajectory generator decides whether it is dynamically
    reachable from the *original* speed. No abrupt speed jump is accepted as
    a physical trajectory. Optional safety projection uses a supplied finite
    pair family and minimizes total absolute change in tentative entry time.
    The difference between the selected first sampled point and the old
    adjusted first waypoint is reported per vehicle.
    """
    scalars = (dt_s, clearance_s, tolerance_s,
               old_speed_min_mps, old_speed_max_mps)
    if any(not math.isfinite(float(value)) for value in scalars):
        raise ValueError("Adapter parameters must be finite")
    if (dt_s <= 0 or clearance_s < 0 or tolerance_s < 0 or
            old_speed_min_mps <= 0 or old_speed_max_mps < old_speed_min_mps):
        raise ValueError("Invalid adapter time, clearance, tolerance or speed bounds")
    if first.id == second.id:
        raise ValueError("Adapter needs two distinct vehicles")
    offsets = []
    for raw in (first_offset_xy, second_offset_xy):
        try:
            xy = (float(raw[0]), float(raw[1]))
        except (TypeError, IndexError, ValueError):
            raise ValueError("Each first-waypoint offset must be an (x, y) pair")
        if not all(math.isfinite(value) for value in xy):
            raise ValueError("Waypoint offsets must be finite")
        offsets.append(xy)

    vehicles = (first, second)
    speeds = tuple(max(old_speed_min_mps, min(old_speed_max_mps,
        vehicle.speed_mps + 2.0 * sum(offset * direction for offset, direction
                                     in zip(xy, vehicle.direction_xy))))
        for vehicle, xy in zip(vehicles, offsets))
    tentative = tuple(vehicle.distance_to_entry_m / speed
                      for vehicle, speed in zip(vehicles, speeds))
    empty_errors = (None, None)
    raw_failure = None
    try:
        plans = tuple(canonical_approach_plan(vehicle, time_s)
                      for vehicle, time_s in zip(vehicles, tentative))
    except ValueError:
        raw_failure = "dynamically_unreachable_entry"
        plans = None
    if plans is not None and not pair_safe(plans[0], plans[1], clearance_s):
        raw_failure = "pair_zone_occupation_unsafe"
    selected = tentative
    if raw_failure is not None:
        if not allow_safety_projection:
            return AdapterOutcome(False, raw_failure, None,
                                  tentative, speeds, empty_errors,
                                  raw_failure_reason=raw_failure)
        if candidate_entry_times is None or len(candidate_entry_times) != 2:
            raise ValueError("Safety projection needs the shared two-car candidate family")
        safe = []
        for first_time in candidate_entry_times[0]:
            for second_time in candidate_entry_times[1]:
                try:
                    proposal = (canonical_approach_plan(first, first_time),
                                canonical_approach_plan(second, second_time))
                except ValueError:
                    continue
                if pair_safe(proposal[0], proposal[1], clearance_s):
                    pair = (float(first_time), float(second_time))
                    change = sum(abs(now - old) for now, old in zip(pair, tentative))
                    safe.append((change, sum(pair), pair, proposal))
        if not safe:
            return AdapterOutcome(False, "no_safe_projection_candidate", None,
                                  tentative, speeds, empty_errors,
                                  raw_failure_reason=raw_failure)
        _, _, selected, plans = min(safe, key=lambda row: row[:3])

    errors = []
    for vehicle, plan, xy in zip(vehicles, plans, offsets):
        old_first_waypoint_xy = (
            vehicle.start_xy[0] + vehicle.direction_xy[0] * vehicle.speed_mps * dt_s + xy[0],
            vehicle.start_xy[1] + vehicle.direction_xy[1] * vehicle.speed_mps * dt_s + xy[1],
        )
        projected = plan.sample(dt_s).position_xy
        errors.append(math.hypot(projected[0] - old_first_waypoint_xy[0],
                                 projected[1] - old_first_waypoint_xy[1]))

    first_before = (
        plans[0].occupation_window_s[1] + clearance_s <= selected[1] + 1e-9)
    ttc_s = legacy_closing_ttc(
        first.start_xy,
        tuple(d * first.speed_mps for d in first.direction_xy),
        second.start_xy,
        tuple(d * second.speed_mps for d in second.direction_xy),
    )
    contract = CrossingContract(
        first.id, second.id, selected[0], selected[1],
        tolerance_s, tolerance_s, first_before, clearance_s,
        "legacy_learned_offer_adapter", "adapted_first_waypoint_speed" if raw_failure is None else "safety_projected_on_shared_family",
        ttc_s=ttc_s,
    )
    return AdapterOutcome(True, "ok" if raw_failure is None else "safety_projected",
                          contract, tentative, speeds, tuple(errors), raw_failure,
                          selected,
                          sum(abs(now - old) for now, old in zip(selected, tentative)))


def adapt_offers_with_priority_separation(
        first: VehicleSpec, second: VehicleSpec,
        first_offset_xy, second_offset_xy,
        negotiated_offers, dt_s: float,
        clearance_s: float, tolerance_s: float = 0.1,
        old_speed_min_mps: float = 0.3,
        old_speed_max_mps: float = 15.0,
        candidate_entry_times: Optional[Tuple[Sequence[float], Sequence[float]]] = None) -> AdapterOutcome:
    """Bounded-kinematic priority-separated bridge from 4D offers to K_e.

    Unlike the naive direct speed-jump quotient (which yields nearly identical
    entry times for symmetric offers and fails occupation checks 20/20 times),
    this bridge uses both the decoded first-waypoint speed command and the
    4D offer's semantic priority fields (speed_factor = offer[:, 0],
    yield_will = offer[:, 3]):
    1. Compute each vehicle's bounded-kinematic earliest reachable entry time
       toward its target speed, clamped to [t_min_reachable, t_max_reachable].
    2. If the tentative occupation windows overlap, order the pair by
       priority_i = (speed_factor_i - yield_will_i, longitudinal_delta_v_i, -free_flow_time_i)
       so the higher-priority vehicle enters at its bounded target time t_lead
       and the yielding vehicle enters right after the leader's occupation
       window plus clearance_s.
    3. Snap to the nearest safe reachable pair in candidate_entry_times (if
       provided) and report the resulting L1 time adjustment from the priority-
       separated physical target.
    """
    vehicles = (first, second)
    offsets = [(float(first_offset_xy[0]), float(first_offset_xy[1])),
               (float(second_offset_xy[0]), float(second_offset_xy[1]))]
    delta_vs = tuple(
        2.0 * sum(offset * direction for offset, direction
                  in zip(xy, vehicle.direction_xy))
        for vehicle, xy in zip(vehicles, offsets))
    speeds = tuple(
        max(old_speed_min_mps, min(old_speed_max_mps,
                                   vehicle.speed_mps + dv))
        for vehicle, dv in zip(vehicles, delta_vs))
    raw_quotient = tuple(vehicle.distance_to_entry_m / spd
                         for vehicle, spd in zip(vehicles, speeds))

    if candidate_entry_times is None or len(candidate_entry_times) != 2:
        raise ValueError("Priority-separated bridge requires candidate_entry_times")

    reach_times = []
    for idx, vehicle in enumerate(vehicles):
        valid_t = []
        for t_cand in candidate_entry_times[idx]:
            try:
                canonical_approach_plan(vehicle, float(t_cand))
                valid_t.append(float(t_cand))
            except ValueError:
                continue
        if not valid_t:
            return AdapterOutcome(False, "no_reachable_candidate", None,
                                  raw_quotient, speeds, (None, None),
                                  raw_failure_reason="dynamically_unreachable_entry")
        reach_times.append(sorted(valid_t))

    # Clamp raw target time to each vehicle's dynamically reachable range
    clamped = tuple(
        max(reach_times[i][0], min(reach_times[i][-1], raw_quotient[i]))
        for i in range(2))

    # Extract priority score from 4D offers: [speed_factor, lateral_shift, time_headway, yield_will]
    scores = []
    for i, vehicle in enumerate(vehicles):
        sf = float(negotiated_offers[i][0])
        yw = float(negotiated_offers[i][3])
        free_t = vehicle.distance_to_entry_m / vehicle.speed_mps
        scores.append((sf - yw, delta_vs[i], -free_t, -i))

    first_leads = scores[0] >= scores[1]
    lead_idx = 0 if first_leads else 1
    yield_idx = 1 - lead_idx

    # Leader takes its fastest reachable candidate near clamped[lead_idx]
    t_lead = min(reach_times[lead_idx],
                 key=lambda t: (abs(t - clamped[lead_idx]), t))
    plan_lead = canonical_approach_plan(vehicles[lead_idx], t_lead)
    required_yield_entry = plan_lead.occupation_window_s[1] + clearance_s

    # Yielding vehicle targets max(clamped[yield_idx], required_yield_entry)
    physical_target_yield = max(clamped[yield_idx], required_yield_entry)
    t_yield_candidates = [
        t for t in reach_times[yield_idx]
        if t >= required_yield_entry - 1e-9 and
        pair_safe(
            plan_lead if lead_idx == 0 else canonical_approach_plan(first, t),
            canonical_approach_plan(second, t) if lead_idx == 0 else plan_lead,
            clearance_s)
    ]
    if t_yield_candidates:
        t_yield = min(t_yield_candidates,
                      key=lambda t: (abs(t - physical_target_yield), t))
        selected = (t_lead, t_yield) if lead_idx == 0 else (t_yield, t_lead)
        physical_tentative = (
            (clamped[0], physical_target_yield) if lead_idx == 0
            else (physical_target_yield, clamped[1]))
        reason = "priority_separated_feasible"
        raw_fail = None
    else:
        # Fall back to closest safe pair on the grid if priority order cannot fit
        safe_pairs = []
        for t0 in reach_times[0]:
            p0 = canonical_approach_plan(first, t0)
            for t1 in reach_times[1]:
                p1 = canonical_approach_plan(second, t1)
                if pair_safe(p0, p1, clearance_s):
                    safe_pairs.append((
                        abs(t0 - clamped[0]) + abs(t1 - clamped[1]),
                        t0 + t1, (t0, t1)))
        if not safe_pairs:
            return AdapterOutcome(False, "no_safe_projection_candidate", None,
                                  clamped, speeds, (None, None),
                                  raw_failure_reason="pair_zone_occupation_unsafe")
        _, _, selected = min(safe_pairs, key=lambda r: r[:3])
        physical_tentative = clamped
        reason = "safety_projected"
        raw_fail = "priority_order_exceeded_horizon"

    plans = (canonical_approach_plan(first, selected[0]),
             canonical_approach_plan(second, selected[1]))
    errors = []
    for vehicle, plan, xy in zip(vehicles, plans, offsets):
        old_first_waypoint_xy = (
            vehicle.start_xy[0] + vehicle.direction_xy[0] * vehicle.speed_mps * dt_s + xy[0],
            vehicle.start_xy[1] + vehicle.direction_xy[1] * vehicle.speed_mps * dt_s + xy[1],
        )
        projected = plan.sample(dt_s).position_xy
        errors.append(math.hypot(projected[0] - old_first_waypoint_xy[0],
                                 projected[1] - old_first_waypoint_xy[1]))
    first_before = (
        plans[0].occupation_window_s[1] + clearance_s <= selected[1] + 1e-9)
    ttc_s = legacy_closing_ttc(
        first.start_xy,
        tuple(d * first.speed_mps for d in first.direction_xy),
        second.start_xy,
        tuple(d * second.speed_mps for d in second.direction_xy),
    )
    contract = CrossingContract(
        first.id, second.id, selected[0], selected[1],
        tolerance_s, tolerance_s, first_before, clearance_s,
        "legacy_learned_offer_priority_bridge",
        "priority_separated_bounded_kinematic",
        ttc_s=ttc_s,
    )
    l1_grid_snap = sum(abs(now - old)
                       for now, old in zip(selected, physical_tentative))
    return AdapterOutcome(True, reason, contract, physical_tentative,
                          speeds, tuple(errors), raw_fail, selected,
                          l1_grid_snap)


def adapt_learned_offers_to_crossing_contract(
        model, offers, first: VehicleSpec, second: VehicleSpec,
        dt_s: float, clearance_s: float,
        tolerance_s: float = 0.1,
        candidate_entry_times: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        allow_safety_projection: bool = False) -> AdapterOutcome:
    """Pass archived offers through the model's own trained waypoint head.

    ``offers`` must be the negotiated (2, 4) tensor.  Any checkpoint used to
    initialize ``model`` belongs in the experiment manifest; this function
    never loads weights and never asserts the old model was trained.
    """
    import torch

    if offers.shape != (2, 4):
        raise ValueError("Negotiated offers must have shape (2, 4)")
    if not bool(torch.isfinite(offers).all()):
        raise ValueError("Negotiated offers must be finite")
    with torch.no_grad():
        offsets = model.offer_to_waypoint(offers) * model.waypoint_offset_scale
    return adapt_waypoint_offsets_to_crossing_contract(
        first, second, offsets[0, :2].detach().cpu().tolist(),
        offsets[1, :2].detach().cpu().tolist(),
        dt_s, clearance_s, tolerance_s,
        candidate_entry_times=candidate_entry_times,
        allow_safety_projection=allow_safety_projection)
