"""Finite-candidate joint cooperation for deterministic crossing conflicts.

The solver is exact over the *supplied finite trajectory family*. It is not a
continuous-time LP solver and does not establish a general complexity bound.
E3a snapshot comparisons may call both aggregators below with the same frozen
contracts, candidates and hard safety rule. Neither aggregator executes a
vehicle or reports a collision rate.
"""

from dataclasses import dataclass
import itertools
import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .crossing import CrossingContract
from .physics import VehicleSpec, canonical_approach_plan, pair_safe


_EPS = 1e-9


@dataclass(frozen=True)
class JointScore:
    entries_s: Tuple[Tuple[str, float], ...]
    welfare_s: float
    broken_contracts: int
    edge_fidelity: float
    vehicle_change: float

    def entry_dict(self):
        return dict(self.entries_s)


@dataclass(frozen=True)
class SnapshotResult:
    global_plan: JointScore
    sequential_plan: JointScore
    safe_plan_count: int
    exactly_composable_count: int
    welfare_eligible_count: int
    enumerated_plan_count: int
    welfare_optimum_s: float
    welfare_floor_s: float
    sequential_raw_targets_s: Tuple[Tuple[str, float], ...]
    sequential_raw_target_safe: bool
    sequential_edge_order: Tuple[str, ...] = tuple()
    sequential_edge_history: Tuple[dict, ...] = tuple()
    sequential_overwrite_count: int = 0
    search_scope: str = "exact_over_supplied_finite_candidates"


def _safe_global(entries, precomputed_plans, contracts):
    for edge in contracts:
        if not pair_safe(precomputed_plans[edge.first_id][entries[edge.first_id]],
                         precomputed_plans[edge.second_id][entries[edge.second_id]],
                         edge.clearance_s):
            return False
    return True


def _score(entries, vehicles, contracts, edge_weights, vehicle_weights,
           vehicle_reference_s, time_reference_s):
    welfare = -sum(entries[name] - vehicles[name].distance_to_entry_m /
                   vehicles[name].speed_mps for name in vehicles)
    broken = sum(not edge.contains(entries, vehicles) for edge in contracts)
    fidelity = 0.0
    for edge in contracts:
        weight = edge_weights[edge.edge_id]
        fidelity += weight * (
            abs(entries[edge.first_id] - edge.first_reference_s) +
            abs(entries[edge.second_id] - edge.second_reference_s)) / time_reference_s
    vehicle_change = sum(
        vehicle_weights[name] * abs(entries[name] - vehicle_reference_s[name]) /
        time_reference_s for name in vehicles)
    return JointScore(tuple(sorted(entries.items())), welfare, int(broken),
                      fidelity, vehicle_change)


def _sequential_targets(vehicles, contracts, candidates, precomputed):
    """Commit local contracts in TTC order and overwrite shared-vehicle times.

    This transparent algorithm is a reconstruction over the new K_e interface,
    not the unavailable original author multi-car implementation. Each edge
    chooses a safe reachable pair nearest to its contract reference; later
    proposals overwrite earlier targets for a shared vehicle.
    """
    targets = {}
    ordered = sorted(contracts, key=lambda c: (
        c.ttc_s if c.ttc_s is not None else math.inf, c.edge_id))
    history = []
    overwrite_count = 0
    for index, edge in enumerate(ordered):
        first, second = edge.first_id, edge.second_id
        pairs = []
        for first_time, second_time in itertools.product(candidates[first],
                                                         candidates[second]):
            if not pair_safe(precomputed[first][first_time],
                             precomputed[second][second_time], edge.clearance_s):
                continue
            entry = {first: first_time, second: second_time}
            satisfies_contract = edge.contains(entry, vehicles)
            reference_distance = (abs(first_time - edge.first_reference_s) +
                                  abs(second_time - edge.second_reference_s))
            pair_welfare = -(
                first_time - vehicles[first].distance_to_entry_m /
                vehicles[first].speed_mps +
                second_time - vehicles[second].distance_to_entry_m /
                vehicles[second].speed_mps)
            pairs.append((not satisfies_contract, reference_distance,
                          -pair_welfare, first_time, second_time))
        if pairs:
            best = min(pairs)
            proposed = {first: best[3], second: best[4]}
            local_safe_pair_found = True
        else:
            proposed = {first: edge.first_reference_s,
                        second: edge.second_reference_s}
            local_safe_pair_found = False
        overwritten = {}
        for name, value in proposed.items():
            if name in targets and abs(targets[name] - value) > _EPS:
                overwrite_count += 1
                overwritten[name] = {"previous_time_s": targets[name],
                                     "new_time_s": value}
            targets[name] = value
        history.append({
            "order": index,
            "edge_id": edge.edge_id,
            "predecision_ttc_s": edge.ttc_s,
            "proposed_entry_times_s": proposed,
            "local_safe_pair_found": local_safe_pair_found,
            "overwritten_vehicle_targets": overwritten,
        })
    for name, vehicle in vehicles.items():
        targets.setdefault(name, vehicle.distance_to_entry_m / vehicle.speed_mps)
    return targets, ordered, history, overwrite_count


def compose_crossing_snapshot(
    vehicles: Mapping[str, VehicleSpec],
    contracts: Sequence[CrossingContract],
    candidate_entry_times_s: Mapping[str, Sequence[float]],
    welfare_tolerance_s: float,
    time_reference_s: float,
    edge_weights: Optional[Mapping[str, float]] = None,
    vehicle_weights: Optional[Mapping[str, float]] = None,
    vehicle_reference_s: Optional[Mapping[str, float]] = None,
    max_joint_combinations: int = 100000,
    commitment_weight: float = 0.0,
) -> SnapshotResult:
    """Test exact synthesis and perform lexicographic finite-family repair.

    First compute F_N and W* over the supplied candidate product. Then retain
    W >= W* - epsilon_W and minimize (broken edges, edge fidelity plus weighted
    change from the prior committed schedule, -welfare). The sequential
    comparator applies each local contract in TTC order, overwrites shared
    vehicle targets with later edge proposals, then projects onto the same
    feasible candidate family.
    """
    if len(vehicles) < 2 or not contracts:
        raise ValueError("Need at least two vehicles and one conflict edge")
    if not math.isfinite(welfare_tolerance_s) or welfare_tolerance_s < 0:
        raise ValueError("welfare_tolerance_s must be finite and nonnegative")
    if not math.isfinite(time_reference_s) or time_reference_s <= 0:
        raise ValueError("time_reference_s must be finite and positive")
    if not math.isfinite(commitment_weight) or commitment_weight < 0:
        raise ValueError("commitment_weight must be finite and nonnegative")
    if max_joint_combinations < 1:
        raise ValueError("max_joint_combinations must be positive")
    names = sorted(vehicles)
    if set(candidate_entry_times_s) != set(vehicles):
        raise ValueError("Every vehicle needs its own candidate entry-time family")
    unordered_edges = set()
    for edge in contracts:
        if edge.first_id not in vehicles or edge.second_id not in vehicles:
            raise ValueError("Edge {} refers to an unknown vehicle".format(edge.edge_id))
        key = frozenset((edge.first_id, edge.second_id))
        if key in unordered_edges:
            raise ValueError("Duplicate conflict edge {}".format(edge.edge_id))
        unordered_edges.add(key)

    edge_weights = dict(edge_weights or {})
    vehicle_weights = dict(vehicle_weights or {})
    vehicle_reference_s = dict(vehicle_reference_s or {})
    for edge in contracts:
        edge_weights.setdefault(edge.edge_id, 1.0)
        if not math.isfinite(edge_weights[edge.edge_id]) or edge_weights[edge.edge_id] <= 0:
            raise ValueError("Every edge weight must be finite and positive")
    for name, vehicle in vehicles.items():
        vehicle_weights.setdefault(name, 1.0)
        vehicle_reference_s.setdefault(name, vehicle.distance_to_entry_m /
                                       vehicle.speed_mps)
        if not math.isfinite(vehicle_weights[name]) or vehicle_weights[name] < 0:
            raise ValueError("Vehicle weights must be finite and nonnegative")
        if not math.isfinite(vehicle_reference_s[name]):
            raise ValueError("Vehicle reference times must be finite")

    candidates = {}
    precomputed = {}
    total = 1
    for name in names:
        times = tuple(sorted(set(float(t) for t in candidate_entry_times_s[name])))
        if not times:
            raise ValueError("Vehicle {} has no candidate entry time".format(name))
        plans = {}
        for t in times:
            plans[t] = canonical_approach_plan(vehicles[name], t)
        candidates[name] = times
        precomputed[name] = plans
        total *= len(times)
        if total > max_joint_combinations:
            raise ValueError("Candidate product {} exceeds explicit cap {}".format(
                total, max_joint_combinations))

    safe_scores = []
    for combination in itertools.product(*(candidates[name] for name in names)):
        entries = dict(zip(names, combination))
        if not _safe_global(entries, precomputed, contracts):
            continue
        safe_scores.append(_score(entries, vehicles, contracts, edge_weights,
                                  vehicle_weights, vehicle_reference_s,
                                  time_reference_s))
    if not safe_scores:
        raise ValueError("F_N is empty for the supplied finite candidate family")
    welfare_star = max(row.welfare_s for row in safe_scores)
    welfare_floor = welfare_star - welfare_tolerance_s
    eligible = [row for row in safe_scores if row.welfare_s >= welfare_floor - _EPS]
    exact_count = sum(row.broken_contracts == 0 for row in safe_scores)
    global_plan = min(eligible, key=lambda row: (
        row.broken_contracts,
        row.edge_fidelity + commitment_weight * row.vehicle_change,
        -row.welfare_s, row.entries_s))

    raw_targets, sequential_order, sequential_history, overwrite_count = (
        _sequential_targets(vehicles, contracts, candidates, precomputed))
    raw_target_safe = False
    try:
        raw_plans = {name: canonical_approach_plan(vehicles[name], raw_targets[name])
                     for name in names}
        raw_target_safe = all(pair_safe(raw_plans[e.first_id], raw_plans[e.second_id],
                                        e.clearance_s) for e in contracts)
    except ValueError:
        pass
    sequential_plan = min(eligible, key=lambda row: (
        sum(abs(value - raw_targets[name]) for name, value in row.entries_s) +
        commitment_weight * row.vehicle_change,
        -row.welfare_s, row.entries_s))
    return SnapshotResult(global_plan, sequential_plan, len(safe_scores),
                          exact_count, len(eligible), total, welfare_star,
                          welfare_floor, tuple(sorted(raw_targets.items())),
                          raw_target_safe,
                          tuple(edge.edge_id for edge in sequential_order),
                          tuple(sequential_history), overwrite_count)
