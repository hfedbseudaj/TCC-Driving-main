"""Paired receding-horizon kinematic E3b for the crossing prototype.

Every control tick rebuilds the current VehicleSpec objects, reachable entry
families, edge contracts, and joint feasibility set.  A car already inside the
shared conflict zone is committed: its remaining occupation is a fixed
boundary condition, not a new Rubinstein bargain.  Both policies use the same
candidate family and hard pair-zone filter.  This is a kinematic closed loop,
not CARLA, vehicle-body collision certification, or an author N-player test.
"""

from dataclasses import asdict, dataclass
import itertools
import math
import time
from typing import Dict, Mapping, Sequence, Tuple

from .compose import compose_crossing_snapshot
from .crossing import CrossingContract, PairProposal, decode_crossing_pair
from .physics import (
    VehicleSpec, canonical_approach_plan, legacy_closing_ttc, pair_safe,
)


@dataclass
class MovingCar:
    spec: VehicleSpec
    progress_m: float = 0.0
    speed_mps: float = 0.0
    entry_clock_s: float = -1.0

    def __post_init__(self):
        if self.speed_mps == 0.0:
            self.speed_mps = self.spec.speed_mps

    @property
    def cleared(self):
        return self.progress_m >= self.spec.distance_to_entry_m + self.spec.zone_length_m - 1e-8

    @property
    def in_zone(self):
        return self.progress_m >= self.spec.distance_to_entry_m - 1e-8 and not self.cleared

    @property
    def position_xy(self):
        return tuple(self.spec.start_xy[index] + self.spec.direction_xy[index] * self.progress_m
                     for index in (0, 1))

    def current_spec(self, remaining_horizon_s):
        distance = max(0.0, self.spec.distance_to_entry_m - self.progress_m)
        zone_length = (self.spec.zone_length_m if distance > 0.0 else
                       max(1e-6, self.spec.distance_to_entry_m +
                           self.spec.zone_length_m - self.progress_m))
        return VehicleSpec(
            self.spec.id, self.position_xy, self.spec.direction_xy,
            distance, max(0.3, self.speed_mps), zone_length,
            self.spec.max_accel_mps2, self.spec.max_brake_mps2,
            self.spec.max_speed_mps, 0.0, max(0.0, remaining_horizon_s))


def _candidate_times(vehicle, horizon_s, grid_step_s, max_delay_s):
    if vehicle.distance_to_entry_m == 0.0:
        return (0.0,)
    free = vehicle.distance_to_entry_m / vehicle.speed_mps
    count = int(math.floor(max_delay_s / grid_step_s))
    times = sorted(set(round(free + index * grid_step_s, 6)
                       for index in range(count + 1)))
    valid = []
    for entry in times:
        if entry + vehicle.zone_length_m / vehicle.speed_mps > horizon_s + 1e-8:
            continue
        try:
            canonical_approach_plan(vehicle, entry)
        except ValueError:
            continue
        valid.append(entry)
    return tuple(valid)


def _current_ttc(first, second):
    v_first = tuple(x * first.speed_mps for x in first.direction_xy)
    v_second = tuple(x * second.speed_mps for x in second.direction_xy)
    return legacy_closing_ttc(first.start_xy, v_first,
                              second.start_xy, v_second)


def _edge_contract(first, second, first_times, second_times,
                   alpha_first_per_s, alpha_second_per_s,
                   eta_s, clearance_s, tolerance_s):
    safe = []
    for first_time, second_time in itertools.product(first_times, second_times):
        first_plan = canonical_approach_plan(first, first_time)
        second_plan = canonical_approach_plan(second, second_time)
        if pair_safe(first_plan, second_plan, clearance_s):
            safe.append((first_time, second_time))
    if not safe:
        raise ValueError("No pair-safe entry times for {}--{}".format(first.id, second.id))
    ttc_s = _current_ttc(first, second)
    # An already entered car cannot renegotiate its occupation.  The earliest
    # compatible pair is an explicit commitment boundary, not an SPE output.
    if first.distance_to_entry_m == 0.0 or second.distance_to_entry_m == 0.0:
        chosen = min(safe, key=lambda pair: (sum(pair), pair))
        source = "committed_zone_boundary"
    else:
        # A conservative safe disagreement point is the latest pair in the
        # supplied finite family.  It is not a learned continuation value.
        disagreement = max(safe, key=lambda pair: (sum(pair), pair))
        proposals = [PairProposal("p_{:.6f}_{:.6f}".format(*pair), *pair)
                     for pair in safe]
        try:
            result = decode_crossing_pair(
                first, second, disagreement, proposals,
                alpha_first_per_s, alpha_second_per_s,
                eta_s, clearance_s,
                (tolerance_s, tolerance_s), ttc_s=ttc_s)
            return result.contract
        except ValueError:
            chosen = min(safe, key=lambda pair: (sum(pair), pair))
            source = "no_positive_surplus_fallback"
    first_plan = canonical_approach_plan(first, chosen[0])
    first_before = first_plan.occupation_window_s[1] + clearance_s <= chosen[1] + 1e-9
    return CrossingContract(
        first.id, second.id, chosen[0], chosen[1], tolerance_s,
        tolerance_s, first_before, clearance_s, source,
        "current_safe_pair", ttc_s=ttc_s)


def _swept_min_distance(first_before, first_after, second_before, second_after):
    rx = first_before[0] - second_before[0]
    ry = first_before[1] - second_before[1]
    vx = (first_after[0] - first_before[0]) - (second_after[0] - second_before[0])
    vy = (first_after[1] - first_before[1]) - (second_after[1] - second_before[1])
    denominator = vx * vx + vy * vy
    fraction = max(0.0, min(1.0, -(rx * vx + ry * vy) / denominator)) if denominator > 1e-12 else 0.0
    return math.hypot(rx + fraction * vx, ry + fraction * vy)


def _one_step_safety_shield(fleet, names, edges, disturbance, dt_s,
                            collision_distance_m, safety_buffer_m):
    """Choose a safe one-step acceleration tuple when joint plans fail.

    This is a recovery controller, not a substitute for the cooperation
    planner. It enumerates bounded brake/coast/accelerate commands, accounts
    for the already sampled disturbance, and rejects commands whose swept
    center distance violates the same collision threshold used for scoring.
    """
    active = [name for name in names if not fleet[name].cleared]
    options = {
        name: tuple(sorted(set((
            -fleet[name].spec.max_brake_mps2,
            -0.5 * fleet[name].spec.max_brake_mps2,
            0.0,
            0.5 * fleet[name].spec.max_accel_mps2,
            fleet[name].spec.max_accel_mps2,
        ))))
        for name in active
    }
    safe_rows = []
    least_unsafe = None
    for accelerations in itertools.product(*(options[name] for name in active)):
        command = dict(zip(active, accelerations))
        after = {}
        desired_speeds = {}
        predicted_speeds = {}
        for index, name in enumerate(names):
            car = fleet[name]
            if car.cleared:
                desired_speeds[name] = car.speed_mps
                predicted_speeds[name] = car.speed_mps
                after[name] = (car.position_xy[0] + car.spec.direction_xy[0] *
                               car.speed_mps * dt_s,
                               car.position_xy[1] + car.spec.direction_xy[1] *
                               car.speed_mps * dt_s)
                continue
            acceleration = command[name]
            desired = max(0.0, min(car.spec.max_speed_mps,
                                   car.speed_mps + acceleration * dt_s))
            speed = max(0.0, min(car.spec.max_speed_mps,
                                 desired + disturbance[index] * dt_s))
            desired_speeds[name] = desired
            predicted_speeds[name] = speed
            distance = 0.5 * (car.speed_mps + speed) * dt_s
            after[name] = (car.position_xy[0] + car.spec.direction_xy[0] * distance,
                           car.position_xy[1] + car.spec.direction_xy[1] * distance)
        distances = [
            _swept_min_distance(fleet[first].position_xy, after[first],
                                fleet[second].position_xy, after[second])
            for first, second in edges
        ]
        minimum_distance = min(distances, default=math.inf)
        # Clear committed zone occupants first, then maximize safe progress.
        progress = 0.0
        effort = 0.0
        for name in active:
            car = fleet[name]
            distance = 0.5 * (car.speed_mps + predicted_speeds[name]) * dt_s
            weight = 3.0 if car.in_zone else 1.0
            progress += weight * distance
            effort += command[name] * command[name]
        tie = tuple(command[name] for name in active)
        safety_margin = min(minimum_distance, safety_buffer_m)
        score = (safety_margin, progress - 0.001 * effort, -effort, tie)
        row = (score, command, desired_speeds, minimum_distance)
        if least_unsafe is None or (minimum_distance, progress - 0.001 * effort) > (
                least_unsafe[3], least_unsafe[0][0]):
            least_unsafe = row
        if minimum_distance + 1e-9 >= collision_distance_m:
            safe_rows.append(row)
    if safe_rows:
        _, command, speeds, minimum_distance = max(safe_rows, key=lambda row: row[0])
        return {
            "safe": True,
            "desired_speeds": speeds,
            "accelerations": command,
            "min_swept_distance_m": minimum_distance,
        }
    if least_unsafe is None:
        return None
    _, command, speeds, minimum_distance = least_unsafe
    return {
        "safe": False,
        "desired_speeds": speeds,
        "accelerations": command,
        "min_swept_distance_m": minimum_distance,
    }


def run_crossing_episode(
        vehicles: Mapping[str, VehicleSpec], edges: Sequence[Tuple[str, str]],
        method: str, disturbance_accel_by_step,
        dt_s=0.1, horizon_s=15.0, alpha_per_s=0.2, eta_s=1.0,
        clearance_s=0.2, tolerance_s=0.1, welfare_epsilon_s=1.0,
        grid_step_s=0.4, max_delay_s=4.0,
        collision_distance_m=1.0, near_distance_m=2.2,
        commitment_weight=0.0):
    """Run one policy under a pre-generated, method-independent disturbance.

    ``method`` selects global lexicographic repair or the explicit sequential
    TTC-ordered edge-contract reconstruction. Per-step log separates pre-filter
    raw target safety, candidate/repair, controller speed-floor intervention,
    and post-step swept geometric distance.  No episode is silently dropped.
    """
    if method not in ("global", "sequential_pairwise_reconstruction",
                      "adapted_pairwise"):
        raise ValueError("Unknown crossing policy")
    if len(vehicles) < 2 or not edges or dt_s <= 0 or horizon_s <= 0:
        raise ValueError("Need vehicles, conflict edges, positive dt and horizon")
    names = sorted(vehicles)
    fleet = {name: MovingCar(vehicles[name]) for name in names}
    committed_entry_abs = {}
    if any(len(row) != len(names) for row in disturbance_accel_by_step):
        raise ValueError("Each disturbance step needs one value per vehicle")
    logs = []
    min_swept_distance = math.inf
    collision = False
    near = False
    fallback_count = 0
    recovery_step_count = 0
    speed_floor_count = 0
    aborted = False
    abort_reason = None
    max_steps = min(len(disturbance_accel_by_step), int(math.ceil(horizon_s / dt_s)))
    for step in range(max_steps):
        clock = step * dt_s
        if all(car.cleared for car in fleet.values()):
            break
        active = {name: car for name, car in fleet.items() if not car.cleared}
        current = {name: car.current_spec(horizon_s - clock)
                   for name, car in active.items()}
        candidates = {name: _candidate_times(spec, horizon_s - clock,
                                             grid_step_s, max_delay_s)
                      for name, spec in current.items()}
        commitment_refs = {}
        for name, spec in current.items():
            free_time = spec.distance_to_entry_m / spec.speed_mps
            prior_time = committed_entry_abs.get(name, clock + free_time) - clock
            commitment_refs[name] = max(free_time, prior_time, 0.0)
        active_edges = [(first, second) for first, second in edges
                        if first in current and second in current]
        contracts = []
        selected = None
        result = None
        failure = None
        recovery = None
        solve_start = time.perf_counter()
        try:
            if any(not values for values in candidates.values()):
                raise ValueError("No dynamically reachable candidate within horizon")
            for first_id, second_id in active_edges:
                first_alpha = (alpha_per_s[first_id] if isinstance(alpha_per_s, Mapping)
                               else alpha_per_s)
                second_alpha = (alpha_per_s[second_id] if isinstance(alpha_per_s, Mapping)
                                else alpha_per_s)
                contracts.append(_edge_contract(
                    current[first_id], current[second_id],
                    candidates[first_id], candidates[second_id],
                    first_alpha, second_alpha,
                    eta_s, clearance_s, tolerance_s))
            if len(current) >= 2 and contracts:
                result = compose_crossing_snapshot(
                    current, contracts, candidates, welfare_epsilon_s, 1.0,
                    max_joint_combinations=100000,
                    vehicle_reference_s=commitment_refs,
                    commitment_weight=commitment_weight)
                selected = (result.global_plan if method == "global" else
                            result.sequential_plan).entry_dict()
            else:
                selected = {name: values[0] for name, values in candidates.items()}
        except ValueError as error:
            failure = str(error)
            fallback_count += 1
            recovery = _one_step_safety_shield(
                fleet, names, edges,
                disturbance_accel_by_step[step], dt_s, collision_distance_m,
                near_distance_m)
            if recovery is None or not recovery["safe"]:
                aborted = True
                abort_reason = ("safety_shield_infeasible: {}".format(failure)
                                if recovery is None else
                                "safety_shield_collision_unavoidable: {}".format(failure))
            else:
                recovery_step_count += 1
        if selected is not None:
            for name, entry_s in selected.items():
                committed_entry_abs[name] = clock + entry_s
        solve_ms = (time.perf_counter() - solve_start) * 1000.0

        before = {name: car.position_xy for name, car in fleet.items()}
        controls = {}
        for index, name in enumerate(names):
            car = fleet[name]
            if car.cleared:
                car.progress_m += car.speed_mps * dt_s
                controls[name] = "already_cleared"
                continue
            old_speed = car.speed_mps
            if selected is None:
                if recovery is not None and recovery["safe"]:
                    desired_speed = recovery["desired_speeds"][name]
                    controls[name] = "one_step_safety_shield_{:+.1f}".format(
                        recovery["accelerations"].get(name, 0.0))
                elif car.in_zone:
                    desired_speed = old_speed
                    controls[name] = "committed_zone_clearance_on_abort"
                else:
                    desired_speed = max(0.0, old_speed - car.spec.max_brake_mps2 * dt_s)
                    controls[name] = "upstream_brake_on_abort"
            else:
                plan = canonical_approach_plan(current[name], selected[name])
                desired_speed = plan.sample(dt_s).speed_mps
                controls[name] = "committed_zone_cruise" if car.in_zone else "planned"
            # The current VehicleSpec requires v>0.  The synthetic controller
            # retains the archived 0.3 m/s floor and records its use.
            new_speed = max(0.3, desired_speed +
                            float(disturbance_accel_by_step[step][index]) * dt_s)
            new_speed = min(new_speed, car.spec.max_speed_mps)
            if new_speed == 0.3 and desired_speed < 0.3:
                speed_floor_count += 1
                controls[name] += "+speed_floor"
            car.progress_m += 0.5 * (old_speed + new_speed) * dt_s
            car.speed_mps = new_speed
            if car.entry_clock_s < 0 and car.progress_m >= car.spec.distance_to_entry_m:
                car.entry_clock_s = clock + dt_s

        after = {name: car.position_xy for name, car in fleet.items()}
        pair_distances = {}
        for first_id, second_id in edges:
            distance = _swept_min_distance(before[first_id], after[first_id],
                                           before[second_id], after[second_id])
            pair_distances[first_id + "--" + second_id] = distance
            min_swept_distance = min(min_swept_distance, distance)
            collision = collision or distance < collision_distance_m
            near = near or distance < near_distance_m
        logs.append({
            "step": step, "clock_s": clock,
            "positions_before_m": before, "positions_after_m": after,
            "entry_s": selected, "controls": controls,
            "commitment_reference_s": commitment_refs,
            "contracts": [asdict(contract) for contract in contracts],
            "pre_filter_raw_target_safe": (
                result.sequential_raw_target_safe if result is not None else None),
            "sequential_edge_order": (
                result.sequential_edge_order if result is not None else None),
            "sequential_edge_history": (
                result.sequential_edge_history if result is not None else None),
            "sequential_overwrite_count": (
                result.sequential_overwrite_count if result is not None else None),
            "raw_pair_target_change_s": (
                sum(abs(selected[name] - dict(result.sequential_raw_targets_s)[name])
                    for name in selected) if result is not None else None),
            "safe_candidate_count": result.safe_plan_count if result is not None else 0,
            "selected_broken_contracts": (
                (result.global_plan if method == "global" else result.sequential_plan).broken_contracts
                if result is not None else None),
            "fallback_reason": failure, "solve_ms": solve_ms,
            "recovery_safe": (recovery["safe"] if recovery is not None else None),
            "recovery_min_swept_distance_m": (
                recovery["min_swept_distance_m"] if recovery is not None else None),
            "swept_pair_distances_m": pair_distances,
            "collision_so_far": collision,
        })
        if collision or aborted:
            break
    completed = all(car.cleared for car in fleet.values())
    entry_times = {name: (None if car.entry_clock_s < 0 else car.entry_clock_s)
                   for name, car in fleet.items()}
    free_times = {name: car.spec.distance_to_entry_m / car.spec.speed_mps
                  for name, car in fleet.items()}
    total_delay = (sum(max(0.0, entry_times[name] - free_times[name])
                       for name in names) if all(value is not None for value in entry_times.values())
                   else None)
    return {
        "method": method,
        "steps": len(logs),
        "completed": completed,
        "planner_aborted": aborted,
        "planner_abort_reason": abort_reason,
        "commitment_weight": commitment_weight,
        "collision": collision,
        "near_collision": near and not collision,
        "min_swept_distance_m": min_swept_distance,
        "fallback_count": fallback_count,
        "recovery_step_count": recovery_step_count,
        "speed_floor_count": speed_floor_count,
        "raw_pair_unsafe_ticks": sum(row["pre_filter_raw_target_safe"] is False
                                     for row in logs),
        "mean_raw_pair_target_change_s": (
            sum(row["raw_pair_target_change_s"] for row in logs
                if row["raw_pair_target_change_s"] is not None) /
            max(1, sum(row["raw_pair_target_change_s"] is not None for row in logs))),
        "entry_times_s": entry_times,
        "total_entry_delay_s": total_delay,
        "mean_solver_ms": (sum(row["solve_ms"] for row in logs) / len(logs)) if logs else 0.0,
        "max_solver_ms": max((row["solve_ms"] for row in logs), default=0.0),
        "logs": logs,
    }
