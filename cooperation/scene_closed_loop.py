"""Provisional receding-horizon closed loop for four deterministic scenes.

These are constructed deterministic kinematic stress cases, not CARLA or
dataset-based validation. Each controller re-decodes local TTC/Rubinstein
contracts and chooses a safe joint action every control tick.
"""

from dataclasses import dataclass
import gc
import itertools
import math
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

from .physics import legacy_closing_ttc, rubinstein_shares, ttc_discount


MODEL = {
    "dt_s": 0.2,
    "horizon_s": 2.0,
    "episode_horizon_s": 16.0,
    "vehicle_length_m": 4.5,
    "vehicle_width_m": 1.8,
    "longitudinal_margin_m": 0.5,
    "lateral_margin_m": 0.2,
    # Exploratory crossing stress trials showed that 0.75 m left the next
    # state without a 2 s constant-action plan. Reserve 1.0 m in prediction;
    # independent seeds must still verify this choice.
    "robust_prediction_margin_m": 1.0,
    "max_accel_mps2": 2.0,
    "max_decel_mps2": 4.0,
    # The simulator samples independent Gaussian acceleration disturbances.
    # A 3-sigma relative bound is used only as a candidate-screening heuristic;
    # it is not a hard safety guarantee for an unbounded Gaussian model.
    "accel_noise_std_mps2": 0.1,
    "dynamic_braking_sigma": 3.0,
    "dynamic_braking_slack_buffer_m": 0.25,
    "max_lateral_accel_mps2": 1.5,
    "max_lateral_speed_mps": 1.5,
    "lateral_kp": 2.0,
    "lateral_kd": 1.6,
    "lane_centers_y_m": (-7.0, -3.5, 0.0, 3.5, 7.0),
    "welfare_epsilon_m": 0.5,
    "contract_surplus_tolerance_m": 4.0,
    "ttc_alpha_per_s": 0.2,
    "near_clearance_m": 0.5,
}


DEFAULT_POLICY = {
    "discount_mode": "ttc",
    "decoder_mode": "coarse",
    "joint_objective": "contract_fidelity",
    "ttc_source": "nominal_conflict",
    "alpha_per_s": MODEL["ttc_alpha_per_s"],
    "ttc_start_s": 6.0,
    "constant_ttc_s": 3.0,
    "surplus_tolerance_m": MODEL["contract_surplus_tolerance_m"],
    "welfare_epsilon_m": MODEL["welfare_epsilon_m"],
    "enforce_individual_rationality": True,
    "dynamic_braking_filter": False,
    "edge_weight_mode": "uniform",
}

VALID_METHODS = ("global", "sequential_pairwise_reconstruction",
                 "welfare_only", "clearance_first", "safe_disagreement")


def normalize_policy(settings=None):
    policy = dict(DEFAULT_POLICY)
    if settings is not None:
        unknown = set(settings) - set(policy)
        if unknown:
            raise ValueError("Unknown policy fields: {}".format(sorted(unknown)))
        policy.update(settings)
    if policy["discount_mode"] not in ("ttc", "constant", "equal_share"):
        raise ValueError("discount_mode must be ttc, constant or equal_share")
    if policy["decoder_mode"] not in ("coarse", "adaptive_crossing"):
        raise ValueError("decoder_mode must be coarse or adaptive_crossing")
    if policy["joint_objective"] not in ("contract_fidelity",
                                         "contract_then_share",
                                         "share_then_contract"):
        raise ValueError("Unknown joint_objective")
    if policy["ttc_source"] not in ("nominal_conflict", "legacy_radial"):
        raise ValueError("ttc_source must be nominal_conflict or legacy_radial")
    for key in ("alpha_per_s", "surplus_tolerance_m", "welfare_epsilon_m"):
        value = float(policy[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError("{} must be finite and nonnegative".format(key))
        policy[key] = value
    for key in ("ttc_start_s", "constant_ttc_s"):
        value = float(policy[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("{} must be finite and positive".format(key))
        policy[key] = value
    if not isinstance(policy["enforce_individual_rationality"], bool):
        raise ValueError("enforce_individual_rationality must be boolean")
    if not isinstance(policy["dynamic_braking_filter"], bool):
        raise ValueError("dynamic_braking_filter must be boolean")
    if policy["edge_weight_mode"] not in (
            "uniform", "inverse_max_degree", "inverse_mean_degree"):
        raise ValueError(
            "edge_weight_mode must be uniform, inverse_max_degree or inverse_mean_degree")
    return policy


@dataclass(frozen=True)
class Car:
    id: str
    x: float
    y: float
    direction: int
    speed: float
    lateral_speed: float
    role: str
    goal_x: float
    max_speed: float = 15.0
    finished: bool = False
    arrival_s: Optional[float] = None
    axis: str = "x"


@dataclass(frozen=True)
class Action:
    name: str
    accel: float
    target_y: float
    # Optional per-step acceleration plan used by the safety-recovery search.
    # The first entry is the control applied now; later entries only describe
    # the predicted receding-horizon continuation.
    accel_schedule: Tuple[float, ...] = ()


@dataclass(frozen=True)
class Contract:
    first: str
    second: str
    first_action: Action
    second_action: Action
    ttc_s: float
    reference_share: float
    surplus_m: float
    delta_first: float = 0.05
    delta_second: float = 0.05
    first_gain_m: float = 0.0
    second_gain_m: float = 0.0
    scene_type: str = ""
    tolerance: float = 0.0
    discount_mode: str = "ttc"
    ttc_active: bool = False
    ttc_applied: bool = False
    ttc_source: str = "legacy_radial"
    legacy_radial_ttc_s: float = math.inf
    realized_share_first: float = 0.0
    share_error: float = 0.0
    eligible_pair_count: int = 0


def _car(car_id, x, y, direction, speed, role, goal_x, max_speed=15.0,
         axis="x"):
    if axis not in ("x", "y"):
        raise ValueError("axis must be x or y")
    return Car(car_id, float(x), float(y), int(direction), float(speed), 0.0,
               role, float(goal_x), float(max_speed), axis=axis)


def _longitudinal_position(car):
    return car.x if car.axis == "x" else car.y


def _lateral_position(car):
    return car.y if car.axis == "x" else car.x


def _world_velocity(car):
    if car.axis == "x":
        return (car.direction * car.speed, car.lateral_speed)
    return (car.lateral_speed, car.direction * car.speed)


def make_scene(scene, n_cars):
    """Return a fixed, explicit synthetic starting geometry and conflict graph."""
    if n_cars not in (2, 3, 4):
        raise ValueError("n_cars must be 2, 3 or 4")
    if scene == "chase":
        all_cars = [
            _car("R2", -40, 0, 1, 11, "rear", 20),
            _car("R", -30, 0, 1, 10, "rear", 20),
            _car("M", -20, 0, 1, 7, "middle", 20),
            _car("F", -10, 0, 1, 6, "front", 20),
        ]
        selected = all_cars[1:] if n_cars == 3 else (
            all_cars[:] if n_cars == 4 else all_cars[1:3])
        edges = tuple((selected[i].id, selected[i + 1].id)
                      for i in range(len(selected) - 1))
    elif scene == "merge":
        all_cars = [
            _car("M", -25, -3.5, 1, 8, "merging", 20),
            _car("B1", -30, 0, 1, 7, "mainline", 20),
            _car("B2", -10, 0, 1, 6, "mainline", 20),
            _car("B3", 5, 0, 1, 6, "mainline", 20),
        ]
        # Preserve one merging vehicle and add mainline vehicles by group size.
        selected = [all_cars[0], all_cars[1]]
        if n_cars >= 3:
            selected.append(all_cars[2])
        if n_cars >= 4:
            selected.append(all_cars[3])
        mainline = [car.id for car in selected if car.role == "mainline"]
        edges = tuple([("M", name) for name in mainline] +
                      [(mainline[i], mainline[i + 1])
                       for i in range(len(mainline) - 1)])
    elif scene == "head_on":
        all_cars = [
            _car("E1", -25, 0, 1, 8, "eastbound", 20),
            _car("W", 25, 0, -1, 8, "westbound", -20),
            _car("E2", -35, 0, 1, 8, "eastbound", 20),
            _car("E3", -45, 0, 1, 8, "eastbound", 20),
        ]
        selected = [all_cars[0], all_cars[1]]
        if n_cars >= 3:
            selected.append(all_cars[2])
        if n_cars >= 4:
            selected.append(all_cars[3])
        eastbound = [car.id for car in selected if car.direction == 1]
        edges = tuple([("E1", "W")] + [("W", name) for name in eastbound[1:]] +
                      [(eastbound[i], eastbound[i + 1])
                       for i in range(len(eastbound) - 1)])
    elif scene == "crossing":
        all_cars = [
            _car("W_to_E", -16, -1.75, 1, 8, "horizontal", 20, axis="x"),
            _car("E_to_W", 16, 1.75, -1, 8, "horizontal", -20, axis="x"),
            _car("S_to_N", 1.75, -16, 1, 8, "vertical", 20, axis="y"),
            _car("N_to_S", -1.75, 16, -1, 8, "vertical", -20, axis="y"),
        ]
        # A crossing group must contain both axes. N=3 puts two local
        # contracts on S_to_N; N=4 adds the second vertical approach.
        indices = {2: (0, 2), 3: (0, 1, 2), 4: (0, 1, 2, 3)}
        selected = [all_cars[index] for index in indices[n_cars]]
        horizontal = [car.id for car in selected if car.axis == "x"]
        vertical = [car.id for car in selected if car.axis == "y"]
        edges = tuple((horizontal_car, vertical_car)
                      for horizontal_car in horizontal
                      for vertical_car in vertical)
    else:
        raise ValueError("scene must be chase, merge, head_on or crossing")
    return {car.id: car for car in selected}, edges


def _lane_target(car, side=0):
    lateral = _lateral_position(car)
    if side == 0:
        # The crossing geometry uses lane centres at +/-1.75 m. Snapping a
        # hold-lane action to the global 3.5 m grid silently changes lanes.
        return lateral
    # Positive side is the vehicle's left, so it reverses in westbound traffic.
    return max(-7.0, min(7.0, lateral + side * 3.5 * car.direction))


def _actions(scene, car):
    hold_y = _lane_target(car, 0)
    if scene == "crossing":
        return tuple(Action("cross_{:+.0f}".format(accel), accel, hold_y)
                     for accel in (-4.0, -2.0, 0.0, 1.0, 2.0))
    if scene == "chase":
        accelerations = (-4.0, -2.0, 0.0, 1.0, 2.0)
        if car.role == "front":
            accelerations = (-2.0, 0.0, 1.0, 2.0)
        return tuple(Action("a{:+.0f}".format(accel), accel, hold_y)
                     for accel in accelerations)
    if scene == "merge" and car.role == "merging":
        return (Action("abort_brake", -3.0, -3.5),
                Action("slow_merge", -1.0, 0.0),
                Action("coast_merge", 0.0, 0.0),
                Action("accelerate_merge", 2.0, 0.0))
    if scene == "merge":
        return tuple(Action("mainline_{:+.0f}".format(accel), accel, hold_y)
                     for accel in (-2.0, 0.0, 2.0))
    return (Action("keep_lane", 0.0, hold_y),
            Action("accelerate_keep_lane", 1.0, hold_y),
            Action("brake_keep_lane", -3.0, hold_y),
            Action("coast_evade_left", 0.0, _lane_target(car, 1)),
            Action("coast_evade_right", 0.0, _lane_target(car, -1)),
            Action("brake_evade_left", -3.0, _lane_target(car, 1)),
            Action("brake_evade_right", -3.0, _lane_target(car, -1)))


def _recovery_actions(scene, car, base_actions, other_cars=()):
    """Add a small set of scene-specific piecewise braking plans for empty families.

    These plans are considered only after the ordinary constant-action joint
    family has no feasible 2 s candidate. The selected action still applies
    only its first 0.2 s acceleration; the remaining schedule is a safety
    continuation used by the next receding-horizon prediction.

    Scene-specific targeting prevents resident combinatorial explosion:
    - chase: same-lane longitudinal schedules + state-dependent leader braking onset
    - crossing: arrival-timing longitudinal schedules on each approach's hold lane
    - merge: merging vehicle only, coupled with both merge (0.0 m) and abort (-3.5 m)
      lateral targets
    - head_on: opposing lead pair (E1, W) only, coupled with lane-hold and
      left/right lateral evasion targets
    """
    if scene not in ("chase", "crossing", "merge", "head_on"):
        return base_actions
    if scene == "merge" and car.role != "merging":
        return base_actions
    if scene == "head_on" and car.id not in ("E1", "W"):
        return base_actions
    count = int(round(MODEL["horizon_s"] / MODEL["dt_s"]))
    profiles = (
        ("recovery_brake_release", (-4.0,) + (0.0,) * (count - 1)),
        ("recovery_soft_then_hard", (-2.0, -2.0) +
         (-4.0,) * max(0, count - 2)),
        ("recovery_coast_then_hard", (0.0,) +
         (-4.0,) * max(0, count - 1)),
        ("recovery_delayed_hard", (0.0,) * min(3, count) +
         (-4.0,) * max(0, count - 3)),
    )
    candidates = list(profiles)
    if scene == "chase":
        # Add a state-dependent braking onset for the closest same-lane leader.
        # The lead-time estimate uses current clearance after the robust prediction
        # margin and positive closing speed; the schedule starts braking one
        # control step before the nominal margin crossing.
        closest_lead = None
        for lead in other_cars:
            if lead.id == car.id or lead.axis != car.axis or lead.direction != car.direction:
                continue
            lateral_gap = abs(_lateral_position(lead) - _lateral_position(car))
            half_width = MODEL["vehicle_width_m"] + MODEL["lateral_margin_m"]
            if lateral_gap >= half_width:
                continue
            forward_gap = car.direction * (
                _longitudinal_position(lead) - _longitudinal_position(car))
            closing_speed = car.speed - lead.speed
            if forward_gap <= 0.0 or closing_speed <= 1e-6:
                continue
            base_clearance = _envelope_clearance(
                car.x, car.y, car, lead.x, lead.y, lead)
            time_to_margin = max(
                0.0, (base_clearance -
                      MODEL["robust_prediction_margin_m"]) / closing_speed)
            if closest_lead is None or forward_gap < closest_lead[0]:
                closest_lead = (forward_gap, time_to_margin)
        if closest_lead is not None and closest_lead[1] <= MODEL["horizon_s"]:
            time_to_margin = closest_lead[1]
            onset = max(0, min(count - 1, int(math.floor(
                time_to_margin / MODEL["dt_s"])) - 1))
            schedule = ((0.0,) * onset + (-4.0,) * (count - onset))
            candidates.append(("state_brake_at_{:02d}".format(onset), schedule))
    target_ys = (
        (_lane_target(car, 0),) if scene in ("chase", "crossing") else
        tuple(sorted(set(action.target_y for action in base_actions))))
    existing_pairs = {(action.accel, action.target_y) for action in base_actions}
    additions = []
    seen = set()
    for target_y in target_ys:
        for name, schedule in candidates:
            if not schedule:
                continue
            key = (target_y, schedule)
            if key in seen:
                continue
            seen.add(key)
            if (all(value == schedule[0] for value in schedule) and
                    (schedule[0], target_y) in existing_pairs):
                continue
            action_name = (
                name if len(target_ys) == 1 else
                "{}_y{:+.1f}".format(name, target_y))
            additions.append(Action(action_name, schedule[0], target_y,
                                    tuple(schedule)))
    return tuple(base_actions) + tuple(additions)


def _adaptive_crossing_actions(car):
    """Bounded half-m/s² pair-decoder grid, never used as a full joint grid."""
    return tuple(Action("cross_{:+g}".format(0.5 * index),
                        0.5 * index, _lane_target(car, 0))
                 for index in range(-8, 5))


def _step_car(car, action, dt_s, noise_accel=0.0):
    accel = action.accel + noise_accel
    min_a = -MODEL["max_decel_mps2"]
    max_a = MODEL["max_accel_mps2"]
    if accel < min_a:
        accel = min_a
    elif accel > max_a:
        accel = max_a
    speed = car.speed
    speed_next = speed + accel * dt_s
    if speed_next < 0.0:
        speed_next = 0.0
    elif speed_next > car.max_speed:
        speed_next = car.max_speed
    axis = car.axis
    if axis == "x":
        long_pos = car.x
        lateral_position = car.y
    else:
        long_pos = car.y
        lateral_position = car.x
    longitudinal_next = long_pos + car.direction * 0.5 * (speed + speed_next) * dt_s
    lat_speed = car.lateral_speed
    lateral_accel = (MODEL["lateral_kp"] * (action.target_y - lateral_position) -
                     MODEL["lateral_kd"] * lat_speed)
    max_la = MODEL["max_lateral_accel_mps2"]
    if lateral_accel < -max_la:
        lateral_accel = -max_la
    elif lateral_accel > max_la:
        lateral_accel = max_la
    lateral_speed_next = lat_speed + lateral_accel * dt_s
    max_ls = MODEL["max_lateral_speed_mps"]
    if lateral_speed_next < -max_ls:
        lateral_speed_next = -max_ls
    elif lateral_speed_next > max_ls:
        lateral_speed_next = max_ls
    lateral_next = lateral_position + 0.5 * (lat_speed + lateral_speed_next) * dt_s
    if lateral_next <= -7.0:
        lateral_next = -7.0
        lateral_speed_next = 0.0
    elif lateral_next >= 7.0:
        lateral_next = 7.0
        lateral_speed_next = 0.0
    if axis == "x":
        x_next, y_next = longitudinal_next, lateral_next
    else:
        x_next, y_next = lateral_next, longitudinal_next
    return Car(car.id, x_next, y_next, car.direction, speed_next,
               lateral_speed_next, car.role, car.goal_x, car.max_speed,
               car.finished, car.arrival_s, axis)


def _rollout(car, action, dt_s, horizon_s, noise=()):
    count = int(round(horizon_s / dt_s))
    path = [car]
    current = car
    for index in range(count):
        noise_value = noise[index] if index < len(noise) else 0.0
        accel = (action.accel_schedule[index]
                 if index < len(action.accel_schedule) else action.accel)
        step_action = (action if accel == action.accel else
                       Action(action.name, accel, action.target_y))
        current = _step_car(current, step_action, dt_s, noise_value)
        path.append(current)
    return tuple(path)


def _pair_envelope_radii(first, second):
    half_length = (MODEL["vehicle_length_m"] +
                   MODEL["longitudinal_margin_m"]) / 2.0
    half_width = (MODEL["vehicle_width_m"] +
                  MODEL["lateral_margin_m"]) / 2.0
    first_x = half_length if first.axis == "x" else half_width
    first_y = half_width if first.axis == "x" else half_length
    second_x = half_length if second.axis == "x" else half_width
    second_y = half_width if second.axis == "x" else half_length
    return first_x + second_x, first_y + second_y


def _segment_min_clearance(first0, first1, second0, second1, first, second,
                           radii=None):
    """Exact minimum rectangle-envelope clearance during linear motion.

    For one path segment, relative x/y positions are affine in phase t. The
    axis-aligned envelope clearance is max(|dx|-rx, |dy|-ry), a piecewise
    linear convex function. Its minimum can only occur at a segment endpoint,
    an absolute-value kink, or an intersection of the active x/y branches.
    Checking those points avoids the old six-sample approximation and cannot
    tunnel through a collision between samples under the linear-segment model.
    """
    radius_x, radius_y = radii or _pair_envelope_radii(first, second)

    dx0 = first0.x - second0.x
    dy0 = first0.y - second0.y
    dx1 = first1.x - second1.x
    dy1 = first1.y - second1.y
    dxv = dx1 - dx0
    dyv = dy1 - dy0

    # Fast allocation-free exact path when neither axis crosses zero inside (0, 1).
    if (dx0 * dx1 >= 0.0) and (dy0 * dy1 >= 0.0):
        c0 = max((dx0 if dx0 >= 0.0 else -dx0) - radius_x,
                 (dy0 if dy0 >= 0.0 else -dy0) - radius_y)
        c1 = max((dx1 if dx1 >= 0.0 else -dx1) - radius_x,
                 (dy1 if dy1 >= 0.0 else -dy1) - radius_y)
        minimum = c0 if c0 < c1 else c1
        mid_x = dx0 + 0.5 * dxv
        mid_y = dy0 + 0.5 * dyv
        sign_x = 1.0 if mid_x >= 0.0 else -1.0
        sign_y = 1.0 if mid_y >= 0.0 else -1.0
        x_slope = sign_x * dxv
        y_slope = sign_y * dyv
        slope_delta = x_slope - y_slope
        if slope_delta > 1e-12 or slope_delta < -1e-12:
            crossing = ((sign_y * dy0 - radius_y) -
                        (sign_x * dx0 - radius_x)) / slope_delta
            if 0.0 < crossing < 1.0:
                cx = dx0 + dxv * crossing
                cy = dy0 + dyv * crossing
                cc = max((cx if cx >= 0.0 else -cx) - radius_x,
                         (cy if cy >= 0.0 else -cy) - radius_y)
                if cc < minimum:
                    minimum = cc
        return minimum

    breaks = [0.0, 1.0]
    if abs(dxv) > 1e-12:
        root = -dx0 / dxv
        if 0.0 < root < 1.0:
            breaks.append(root)
    if abs(dyv) > 1e-12:
        root = -dy0 / dyv
        if 0.0 < root < 1.0:
            breaks.append(root)
    breaks.sort()
    minimum = float("inf")
    for phase in breaks:
        clearance = max(abs(dx0 + dxv * phase) - radius_x,
                        abs(dy0 + dyv * phase) - radius_y)
        if clearance < minimum:
            minimum = clearance

    for lo, hi in zip(breaks[:-1], breaks[1:]):
        mid = 0.5 * (lo + hi)
        sign_x = 1.0 if dx0 + dxv * mid >= 0.0 else -1.0
        sign_y = 1.0 if dy0 + dyv * mid >= 0.0 else -1.0
        x_intercept = sign_x * dx0 - radius_x
        x_slope = sign_x * dxv
        y_intercept = sign_y * dy0 - radius_y
        y_slope = sign_y * dyv
        slope_delta = x_slope - y_slope
        if abs(slope_delta) > 1e-12:
            crossing = (y_intercept - x_intercept) / slope_delta
            if lo < crossing < hi:
                clearance = max(abs(dx0 + dxv * crossing) - radius_x,
                                abs(dy0 + dyv * crossing) - radius_y)
                if clearance < minimum:
                    minimum = clearance

    return minimum


def _pair_clearance_and_next(scene, first_path, second_path, first, second):
    min_clearance = float("inf")
    next_clearance = float("inf")
    radius_x, radius_y = _pair_envelope_radii(first, second)
    radii = (radius_x, radius_y)
    margin = MODEL["robust_prediction_margin_m"]
    n_steps = len(first_path) if len(first_path) < len(second_path) else len(second_path)
    p1_prev = first_path[0]
    p2_prev = second_path[0]
    dx0 = p1_prev.x - p2_prev.x
    dy0 = p1_prev.y - p2_prev.y
    for index in range(1, n_steps):
        p1_curr = first_path[index]
        p2_curr = second_path[index]
        dx1 = p1_curr.x - p2_curr.x
        dy1 = p1_curr.y - p2_curr.y
        if (dx0 * dx1 >= 0.0) and (dy0 * dy1 >= 0.0):
            c0x = (dx0 if dx0 >= 0.0 else -dx0) - radius_x
            c0y = (dy0 if dy0 >= 0.0 else -dy0) - radius_y
            c0 = c0x if c0x >= c0y else c0y
            c1x = (dx1 if dx1 >= 0.0 else -dx1) - radius_x
            c1y = (dy1 if dy1 >= 0.0 else -dy1) - radius_y
            c1 = c1x if c1x >= c1y else c1y
            seg_min = c0 if c0 < c1 else c1
            dxv = dx1 - dx0
            dyv = dy1 - dy0
            sign_x = 1.0 if (dx0 + 0.5 * dxv) >= 0.0 else -1.0
            sign_y = 1.0 if (dy0 + 0.5 * dyv) >= 0.0 else -1.0
            slope_delta = sign_x * dxv - sign_y * dyv
            if slope_delta > 1e-12 or slope_delta < -1e-12:
                crossing = ((sign_y * dy0 - radius_y) -
                            (sign_x * dx0 - radius_x)) / slope_delta
                if 0.0 < crossing < 1.0:
                    cx = dx0 + dxv * crossing
                    cy = dy0 + dyv * crossing
                    ccx = (cx if cx >= 0.0 else -cx) - radius_x
                    ccy = (cy if cy >= 0.0 else -cy) - radius_y
                    cc = ccx if ccx >= ccy else ccy
                    if cc < seg_min:
                        seg_min = cc
        else:
            seg_min = _segment_min_clearance(
                p1_prev, p1_curr, p2_prev, p2_curr, first, second, radii)
        clearance = seg_min - margin
        if index == 1:
            next_clearance = clearance
        if clearance < min_clearance:
            min_clearance = clearance
        p1_prev = p1_curr
        p2_prev = p2_curr
        dx0 = dx1
        dy0 = dy1
    return min_clearance, next_clearance


def _pair_clearance(scene, first_path, second_path, first, second):
    return _pair_clearance_and_next(
        scene, first_path, second_path, first, second)[0]


def _envelope_clearance(ax, ay, first, bx, by, second):
    """Signed separating-axis clearance for axis-aligned vehicle rectangles."""
    half_length = (MODEL["vehicle_length_m"] + MODEL["longitudinal_margin_m"]) / 2.0
    half_width = (MODEL["vehicle_width_m"] + MODEL["lateral_margin_m"]) / 2.0
    first_x = half_length if first.axis == "x" else half_width
    first_y = half_width if first.axis == "x" else half_length
    second_x = half_length if second.axis == "x" else half_width
    second_y = half_width if second.axis == "x" else half_length
    return max(abs(ax - bx) - first_x - second_x,
               abs(ay - by) - first_y - second_y)


def _safe_pair(scene, first_path, second_path, first, second):
    return _pair_clearance(scene, first_path, second_path, first, second) >= -1e-9


def _utility(scene, car, action, path, horizon_s):
    progress = abs(_longitudinal_position(path[-1]) -
                   _longitudinal_position(car))
    count = int(round(horizon_s / MODEL["dt_s"]))
    effort = 0.06 * sum(
        (action.accel_schedule[index]
         if index < len(action.accel_schedule) else action.accel) ** 2 *
        MODEL["dt_s"] for index in range(count))
    lateral_effort = 0.25 * (_lateral_position(path[-1]) -
                             _lateral_position(car)) ** 2
    merge_progress = 0.0
    if scene == "merge" and car.role == "merging":
        merge_progress = 0.7 * (abs(car.y) - abs(path[-1].y))
    return progress - effort - lateral_effort + merge_progress


def _ttc(first, second):
    return legacy_closing_ttc(
        (first.x, first.y),
        _world_velocity(first),
        (second.x, second.y),
        _world_velocity(second))


def _nominal_action(scene, car):
    """A fixed no-negotiation motion used only to assess imminent conflict."""
    name = ("cross_+0" if scene == "crossing" else
            "a+0" if scene == "chase" else
            ("coast_merge" if car.role == "merging" else "mainline_+0")
            if scene == "merge" else "keep_lane")
    return next(action for action in _actions(scene, car) if action.name == name)


def _nominal_conflict_ttc(scene, first, second, horizon_s,
                          first_path=None, second_path=None):
    """First predicted inflated-envelope overlap under nominal controls.

    This is a scene-aware, sampled conflict TTC. It is separate from the
    paper's radial closing TTC, which is retained as an ablation source.
    """
    dt_s = MODEL["dt_s"]
    if first_path is None:
        first_path = _rollout(first, _nominal_action(scene, first), dt_s, horizon_s)
    if second_path is None:
        second_path = _rollout(second, _nominal_action(scene, second), dt_s,
                               horizon_s)
    radius_x, radius_y = _pair_envelope_radii(first, second)
    margin = MODEL["robust_prediction_margin_m"]
    n_steps = len(first_path) if len(first_path) < len(second_path) else len(second_path)
    for index in range(1, n_steps):
        a0, a1 = first_path[index - 1], first_path[index]
        b0, b1 = second_path[index - 1], second_path[index]
        a0x, a0y = a0.x, a0.y
        b0x, b0y = b0.x, b0.y
        adx, ady = a1.x - a0x, a1.y - a0y
        bdx, bdy = b1.x - b0x, b1.y - b0y
        for substep in range(6):
            phase = substep / 5.0
            dx = (a0x + adx * phase) - (b0x + bdx * phase)
            dy = (a0y + ady * phase) - (b0y + bdy * phase)
            cx = (dx if dx >= 0.0 else -dx) - radius_x
            cy = (dy if dy >= 0.0 else -dy) - radius_y
            clearance = (cx if cx >= cy else cy) - margin
            if clearance <= 0.0:
                return (index - 1 + phase) * dt_s
    return math.inf


def _pair_contract(scene, first, second, actions, plans, policy,
                   pair_clearances=None, action_utilities=None,
                   nominal_plans=None):
    if scene == "chase" or (scene == "merge" and first.role == second.role):
        rear, front = ((first, second) if first.x < second.x else (second, first))
        disagreement_names = {
            rear.id: ("a-4", -4.0, _lane_target(rear, 0)),
            front.id: ("a+0" if scene == "chase" else "mainline_+0",
                       0.0, _lane_target(front, 0)),
        }
    elif scene == "crossing":
        disagreement_names = {
            first.id: ("cross_-4", -4.0, _lane_target(first, 0)),
            second.id: ("cross_-4", -4.0, _lane_target(second, 0)),
        }
    elif scene == "merge":
        merging = first if first.role == "merging" else second
        mainline = second if merging.id == first.id else first
        disagreement_names = {
            merging.id: ("abort_brake", -3.0, -3.5),
            mainline.id: ("mainline_+0", 0.0, _lane_target(mainline, 0)),
        }
    else:
        disagreement_names = {
            first.id: ("brake_keep_lane", -3.0, _lane_target(first, 0)),
            second.id: ("brake_keep_lane", -3.0, _lane_target(second, 0)),
        }
    name_a, accel_a, ty_a = disagreement_names[first.id]
    name_b, accel_b, ty_b = disagreement_names[second.id]
    first_id, second_id = first.id, second.id
    swap_pair = first_id > second_id
    if (name_a in plans[first_id] and name_b in plans[second_id]):
        base_a = plans[first_id][name_a]
        base_b = plans[second_id][name_b]
        if pair_clearances is not None:
            dis_key = ((second_id, name_b, first_id, name_a)
                       if swap_pair else (first_id, name_a, second_id, name_b))
            dis_clear = pair_clearances.get(dis_key)
            if dis_clear is None:
                dis_clear = _pair_clearance(scene, base_a, base_b, first, second)
        else:
            dis_clear = _pair_clearance(scene, base_a, base_b, first, second)
        if dis_clear < -1e-9:
            return None
        utility_a0 = (action_utilities[(first_id, name_a)]
                      if action_utilities is not None and
                      (first_id, name_a) in action_utilities else
                      _utility(scene, first, Action(name_a, accel_a, ty_a),
                               base_a, MODEL["horizon_s"]))
        utility_b0 = (action_utilities[(second_id, name_b)]
                      if action_utilities is not None and
                      (second_id, name_b) in action_utilities else
                      _utility(scene, second, Action(name_b, accel_b, ty_b),
                               base_b, MODEL["horizon_s"]))
    else:
        act_a0 = Action(name_a, accel_a, ty_a)
        act_b0 = Action(name_b, accel_b, ty_b)
        base_a = _rollout(first, act_a0, MODEL["dt_s"], MODEL["horizon_s"])
        base_b = _rollout(second, act_b0, MODEL["dt_s"], MODEL["horizon_s"])
        if not _safe_pair(scene, base_a, base_b, first, second):
            return None
        utility_a0 = _utility(scene, first, act_a0, base_a, MODEL["horizon_s"])
        utility_b0 = _utility(scene, second, act_b0, base_b, MODEL["horizon_s"])
    safe = []
    for action_a, action_b in itertools.product(actions[first_id], actions[second_id]):
        if pair_clearances is not None:
            key = ((second_id, action_b.name, first_id, action_a.name)
                   if swap_pair else
                   (first_id, action_a.name, second_id, action_b.name))
            clearance = pair_clearances.get(key)
            if clearance is None:
                clearance = _pair_clearance(
                    scene, plans[first_id][action_a.name],
                    plans[second_id][action_b.name], first, second)
        else:
            clearance = _pair_clearance(
                scene, plans[first_id][action_a.name],
                plans[second_id][action_b.name], first, second)
        if clearance < -1e-9:
            continue
        if action_utilities is not None and (first_id, action_a.name) in action_utilities:
            util_a = action_utilities[(first_id, action_a.name)]
        else:
            util_a = _utility(scene, first, action_a,
                              plans[first_id][action_a.name], MODEL["horizon_s"])
        if action_utilities is not None and (second_id, action_b.name) in action_utilities:
            util_b = action_utilities[(second_id, action_b.name)]
        else:
            util_b = _utility(scene, second, action_b,
                              plans[second_id][action_b.name], MODEL["horizon_s"])
        gain_a = util_a - utility_a0
        gain_b = util_b - utility_b0
        if gain_a >= -1e-9 and gain_b >= -1e-9 and gain_a + gain_b > 1e-9:
            safe.append((action_a, action_b, gain_a, gain_b))
    if not safe:
        return None
    maximum = max(row[2] + row[3] for row in safe)
    legacy_ttc = _ttc(first, second)
    ttc = (legacy_ttc if policy["ttc_source"] == "legacy_radial" else
           _nominal_conflict_ttc(
               scene, first, second, policy["ttc_start_s"],
               first_path=(nominal_plans.get(first.id)
                           if nominal_plans is not None else None),
               second_path=(nominal_plans.get(second.id)
                            if nominal_plans is not None else None)))
    active = math.isfinite(ttc) and ttc < policy["ttc_start_s"]
    if policy["discount_mode"] == "equal_share" or not active:
        # Geometric edges without imminent closing risk still coordinate, but
        # they do not borrow TTC impatience from a non-conflict situation.
        delta_a = delta_b = 0.0
        share = 0.5
        used_mode = ("equal_share" if policy["discount_mode"] == "equal_share"
                     else "nonurgent_equal_share")
    else:
        reference_ttc = (ttc if policy["discount_mode"] == "ttc" else
                         policy["constant_ttc_s"])
        delta_a = ttc_discount(policy["alpha_per_s"], reference_ttc)
        delta_b = ttc_discount(policy["alpha_per_s"], reference_ttc)
        share, _ = rubinstein_shares(delta_a, delta_b)
        used_mode = policy["discount_mode"]
    eligible = [row for row in safe
                if row[2] + row[3] >= maximum - policy["surplus_tolerance_m"]]
    selected = min(eligible, key=lambda row: (
        abs(row[2] / max(1e-9, row[2] + row[3]) - share),
        -(row[2] + row[3]), row[0].name, row[1].name))
    realized_share = selected[2] / (selected[2] + selected[3])
    return Contract(first.id, second.id, selected[0], selected[1], ttc,
                    share, selected[2] + selected[3], delta_a, delta_b,
                    selected[2], selected[3], scene, 0.0, used_mode, active,
                    used_mode == "ttc", policy["ttc_source"], legacy_ttc,
                    realized_share, abs(realized_share - share), len(eligible))


def _action_distance(actual, reference):
    return (abs(actual.accel - reference.accel) / 4.0 +
            abs(actual.target_y - reference.target_y) / 3.5)


def _safe_joint_rows(scene, names, cars, actions, plans, edges,
                     contracts, welfare_floor, pair_clearances,
                     action_utilities, compute_chase_slack=False,
                     edge_weight_mode="uniform"):
    rows = []
    n_cars = len(names)
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    degrees = {name: 0 for name in names}
    for edge in contracts:
        if edge is not None:
            degrees[edge.first] = degrees.get(edge.first, 0) + 1
            degrees[edge.second] = degrees.get(edge.second, 0) + 1
    edge_specs = []
    for edge in contracts:
        if edge is None:
            continue
        i_first = name_to_idx[edge.first]
        i_second = name_to_idx[edge.second]
        deg_f = max(1, degrees.get(edge.first, 1))
        deg_s = max(1, degrees.get(edge.second, 1))
        if edge_weight_mode == "inverse_max_degree":
            w_e = 1.0 / float(max(deg_f, deg_s))
        elif edge_weight_mode == "inverse_mean_degree":
            w_e = 2.0 / float(deg_f + deg_s)
        else:
            w_e = 1.0
        first_ref_name = edge.first_action.name
        second_ref_name = edge.second_action.name
        first_baseline = (action_utilities[(edge.first, first_ref_name)] -
                          edge.first_gain_m)
        second_baseline = (action_utilities[(edge.second, second_ref_name)] -
                           edge.second_gain_m)
        first_dists = {
            act.name: w_e * _action_distance(act, edge.first_action)
            for act in actions[edge.first]
        }
        second_dists = {
            act.name: w_e * _action_distance(act, edge.second_action)
            for act in actions[edge.second]
        }
        first_gains = {
            act.name: action_utilities[(edge.first, act.name)] - first_baseline
            for act in actions[edge.first]
        }
        second_gains = {
            act.name: action_utilities[(edge.second, act.name)] - second_baseline
            for act in actions[edge.second]
        }
        edge_specs.append((
            i_first, i_second, first_ref_name, second_ref_name,
            first_dists, second_dists, first_gains, second_gains,
            edge.reference_share, w_e / max(edge.surplus_m, 1.0)))

    util_by_idx = [
        {act.name: action_utilities[(name, act.name)] for act in actions[name]}
        for name in names
    ]
    pair_tables = {}
    for i in range(n_cars):
        first_name = names[i]
        for j in range(i + 1, n_cars):
            second_name = names[j]
            table = {}
            for act_a in actions[first_name]:
                row_map = {}
                for act_b in actions[second_name]:
                    row_map[act_b.name] = pair_clearances[
                        (first_name, act_a.name, second_name, act_b.name)]
                table[act_a.name] = row_map
            pair_tables[(i, j)] = table

    # Incremental DFS pruning unsafe partial assignments before deeper levels.
    chosen_actions = [None] * n_cars
    chosen_names = [None] * n_cars

    def _dfs(depth, current_min_clearance, current_welfare):
        if depth == n_cars:
            if current_welfare < welfare_floor - 1e-9:
                return
            choice = dict(zip(names, chosen_actions))
            broken = 0
            fidelity = 0.0
            allocation_error = 0.0
            for (i_f, i_s, f_ref, s_ref, f_dists, s_dists,
                 f_gains, s_gains, ref_share, inv_scale) in edge_specs:
                fn = chosen_names[i_f]
                sn = chosen_names[i_s]
                if fn != f_ref or sn != s_ref:
                    broken += 1
                fidelity += f_dists[fn] + s_dists[sn]
                fg = f_gains[fn]
                sg = s_gains[sn]
                allocation_error += (
                    abs(fg - ref_share * (fg + sg)) +
                    (0.0 if fg >= 0.0 else -fg) +
                    (0.0 if sg >= 0.0 else -sg)) * inv_scale
            rows.append((
                choice, current_welfare, broken, fidelity,
                current_min_clearance, allocation_error,
                (_predicted_min_robust_chase_slack(
                    cars, names, plans, choice)
                 if compute_chase_slack else None),
                tuple((names[i], chosen_names[i]) for i in range(n_cars))))
            return

        name_d = names[depth]
        utils_d = util_by_idx[depth]
        prior_tables = [pair_tables[(i, depth)] for i in range(depth)]
        for act in actions[name_d]:
            act_name = act.name
            min_c = current_min_clearance
            feasible = True
            for i in range(depth):
                c = prior_tables[i][chosen_names[i]][act_name]
                if c < -1e-9:
                    feasible = False
                    break
                if c < min_c:
                    min_c = c
            if not feasible:
                continue
            chosen_actions[depth] = act
            chosen_names[depth] = act_name
            _dfs(depth + 1, min_c, current_welfare + utils_d[act_name])

    _dfs(0, float("inf"), 0.0)
    return rows


def _select_joint_fast(scene, names, cars, actions, contracts,
                       pair_clearances, action_utilities, method,
                       policy, sequential_targets):
    """Exact integer-indexed joint selection with deferred edge/dict allocation.

    Produces identical candidate counts, disagreement baseline, welfare optimum,
    and selected joint choice as _safe_joint_rows + _choose selection, while
    avoiding per-candidate dict/tuple/edge-metric construction until after
    the welfare floor filters candidates down to a tiny survivor set.
    """
    n_cars = len(names)
    acts_by_d = [actions[name] for name in names]
    util_by_d = [
        [action_utilities[(name, act.name)] for act in acts_by_d[d]]
        for d, name in enumerate(names)
    ]
    pair_mats = {}
    for i in range(n_cars):
        first_name = names[i]
        acts_i = acts_by_d[i]
        for j in range(i + 1, n_cars):
            second_name = names[j]
            acts_j = acts_by_d[j]
            pair_mats[(i, j)] = [
                [pair_clearances[(first_name, act_a.name,
                                  second_name, act_b.name)]
                 for act_b in acts_j]
                for act_a in acts_i
            ]

    raw_safe = []
    chosen_idx = [0] * n_cars

    def _dfs(depth, current_min_c, current_welfare):
        if depth == n_cars:
            raw_safe.append((tuple(chosen_idx), current_welfare, current_min_c))
            return
        mats = [pair_mats[(i, depth)] for i in range(depth)]
        utils_d = util_by_d[depth]
        for a_idx in range(len(acts_by_d[depth])):
            min_c = current_min_c
            ok = True
            for i in range(depth):
                c = mats[i][chosen_idx[i]][a_idx]
                if c < -1e-9:
                    ok = False
                    break
                if c < min_c:
                    min_c = c
            if ok:
                chosen_idx[depth] = a_idx
                _dfs(depth + 1, min_c, current_welfare + utils_d[a_idx])

    _dfs(0, float("inf"), 0.0)
    if not raw_safe:
        return None

    # Disagreement baseline selection over all safe candidates.
    nom_dists = []
    for d, name in enumerate(names):
        car = cars[name]
        if scene == "crossing":
            nom_name = "cross_-4"
        elif scene == "chase":
            nom_name = "a+0" if car.role == "front" else "a-4"
        elif scene == "merge":
            nom_name = "abort_brake" if car.role == "merging" else "mainline_+0"
        else:
            nom_name = "brake_keep_lane"
        nom_act = next(act for act in acts_by_d[d] if act.name == nom_name)
        nom_dists.append([_action_distance(act, nom_act) for act in acts_by_d[d]])

    dis_row = None
    best_dis_primary = None
    best_dis_tie = None
    for r in raw_safe:
        idx_t = r[0]
        nd = 0.0
        for d in range(n_cars):
            nd += nom_dists[d][idx_t[d]]
        primary = (nd, -r[2], -r[1])
        if best_dis_primary is None or primary < best_dis_primary:
            best_dis_primary = primary
            best_dis_tie = None
            dis_row = r
        elif primary == best_dis_primary:
            if best_dis_tie is None:
                b_idx = dis_row[0]
                best_dis_tie = tuple(
                    (names[d], acts_by_d[d][b_idx[d]].name)
                    for d in range(n_cars))
            tie = tuple((names[d], acts_by_d[d][idx_t[d]].name)
                        for d in range(n_cars))
            if tie < best_dis_tie:
                best_dis_tie = tie
                dis_row = r
    dis_idx = dis_row[0]
    dis_utils = [util_by_d[d][dis_idx[d]] for d in range(n_cars)]
    dis_choice = {names[d]: acts_by_d[d][dis_idx[d]] for d in range(n_cars)}
    dis_utilities_map = {names[d]: dis_utils[d] for d in range(n_cars)}

    du_thresh = [du - 1e-9 for du in dis_utils]
    ir_rows = []
    for r in raw_safe:
        idx_t = r[0]
        ok = True
        for d in range(n_cars):
            if util_by_d[d][idx_t[d]] < du_thresh[d]:
                ok = False
                break
        if ok:
            ir_rows.append(r)
    family = ir_rows if policy["enforce_individual_rationality"] else raw_safe
    welfare_star = max(r[1] for r in family)
    floor = welfare_star - policy["welfare_epsilon_m"]
    floor_thresh = floor - 1e-9
    eligible = [r for r in family if r[1] >= floor_thresh]

    if method == "safe_disagreement":
        sel_idx = dis_idx
        sel_welfare = dis_row[1]
        sel_min_c = dis_row[2]
        sel_broken, sel_alloc = _eval_single_edge_metrics(
            names, acts_by_d, util_by_d, action_utilities, contracts,
            policy["edge_weight_mode"], sel_idx)
    elif not eligible:
        return {
            "safe_count": len(raw_safe),
            "ir_count": len(ir_rows),
            "eligible_count": 0,
            "disagreement_choice": dis_choice,
            "disagreement_utilities": dis_utilities_map,
            "welfare_star": welfare_star,
            "selected": None,
        }
    else:
        edge_specs = _build_edge_specs(
            names, acts_by_d, util_by_d, action_utilities, contracts,
            policy["edge_weight_mode"])
        if method == "global":
            obj = policy["joint_objective"]
            best_primary = None
            best_tie = None
            best_row = None
            best_metrics = None
            for r in eligible:
                idx_t, wel, min_c = r
                broken = 0
                fidelity = 0.0
                alloc_err = 0.0
                for (i_f, i_s, f_ref_idx, s_ref_idx, f_dists, s_dists,
                     f_gains, s_gains, ref_share, inv_scale) in edge_specs:
                    fi = idx_t[i_f]
                    si = idx_t[i_s]
                    if fi != f_ref_idx or si != s_ref_idx:
                        broken += 1
                    fidelity += f_dists[fi] + s_dists[si]
                    fg = f_gains[fi]
                    sg = s_gains[si]
                    alloc_err += (
                        abs(fg - ref_share * (fg + sg)) +
                        (0.0 if fg >= 0.0 else -fg) +
                        (0.0 if sg >= 0.0 else -sg)) * inv_scale
                if obj == "share_then_contract":
                    primary = (alloc_err, broken, fidelity, -wel)
                elif obj == "contract_then_share":
                    primary = (broken, alloc_err, fidelity, -wel)
                else:
                    primary = (broken, fidelity, -wel)
                if best_primary is None or primary < best_primary:
                    best_primary = primary
                    best_tie = None
                    best_row = r
                    best_metrics = (broken, fidelity, alloc_err)
                elif primary == best_primary:
                    if best_tie is None:
                        b_idx = best_row[0]
                        best_tie = tuple(
                            (names[d], acts_by_d[d][b_idx[d]].name)
                            for d in range(n_cars))
                    tie = tuple((names[d], acts_by_d[d][idx_t[d]].name)
                                for d in range(n_cars))
                    if tie < best_tie:
                        best_tie = tie
                        best_row = r
                        best_metrics = (broken, fidelity, alloc_err)
            sel_idx, sel_welfare, sel_min_c = best_row
            sel_broken, _, sel_alloc = best_metrics
        elif method == "sequential_pairwise_reconstruction":
            seq_dists = [
                [_action_distance(act, sequential_targets[names[d]])
                 for act in acts_by_d[d]]
                for d in range(n_cars)
            ]
            best_primary = None
            best_tie = None
            best_row = None
            for r in eligible:
                idx_t = r[0]
                sd = 0.0
                for d in range(n_cars):
                    sd += seq_dists[d][idx_t[d]]
                primary = (sd, -r[1])
                if best_primary is None or primary < best_primary:
                    best_primary = primary
                    best_tie = None
                    best_row = r
                elif primary == best_primary:
                    if best_tie is None:
                        b_idx = best_row[0]
                        best_tie = tuple(
                            (names[d], acts_by_d[d][b_idx[d]].name)
                            for d in range(n_cars))
                    tie = tuple((names[d], acts_by_d[d][idx_t[d]].name)
                                for d in range(n_cars))
                    if tie < best_tie:
                        best_tie = tie
                        best_row = r
            sel_idx, sel_welfare, sel_min_c = best_row
            sel_broken, sel_alloc = _eval_single_edge_metrics_from_specs(
                edge_specs, sel_idx)
        elif method == "welfare_only":
            best_primary = None
            best_tie = None
            best_row = None
            for r in eligible:
                primary = -r[1]
                if best_primary is None or primary < best_primary:
                    best_primary = primary
                    best_tie = None
                    best_row = r
                elif primary == best_primary:
                    if best_tie is None:
                        b_idx = best_row[0]
                        best_tie = tuple(
                            (names[d], acts_by_d[d][b_idx[d]].name)
                            for d in range(n_cars))
                    idx_t = r[0]
                    tie = tuple((names[d], acts_by_d[d][idx_t[d]].name)
                                for d in range(n_cars))
                    if tie < best_tie:
                        best_tie = tie
                        best_row = r
            sel_idx, sel_welfare, sel_min_c = best_row
            sel_broken, sel_alloc = _eval_single_edge_metrics_from_specs(
                edge_specs, sel_idx)
        elif method == "clearance_first":
            best_primary = None
            best_tie = None
            best_row = None
            for r in eligible:
                primary = (-r[2], -r[1])
                if best_primary is None or primary < best_primary:
                    best_primary = primary
                    best_tie = None
                    best_row = r
                elif primary == best_primary:
                    if best_tie is None:
                        b_idx = best_row[0]
                        best_tie = tuple(
                            (names[d], acts_by_d[d][b_idx[d]].name)
                            for d in range(n_cars))
                    idx_t = r[0]
                    tie = tuple((names[d], acts_by_d[d][idx_t[d]].name)
                                for d in range(n_cars))
                    if tie < best_tie:
                        best_tie = tie
                        best_row = r
            sel_idx, sel_welfare, sel_min_c = best_row
            sel_broken, sel_alloc = _eval_single_edge_metrics_from_specs(
                edge_specs, sel_idx)
        else:
            raise ValueError("Unknown joint selection method")

    sel_choice = {names[d]: acts_by_d[d][sel_idx[d]] for d in range(n_cars)}
    return {
        "safe_count": len(raw_safe),
        "ir_count": len(ir_rows),
        "eligible_count": len(eligible),
        "disagreement_choice": dis_choice,
        "disagreement_utilities": dis_utilities_map,
        "welfare_star": welfare_star,
        "selected": (sel_choice, sel_welfare, sel_broken, 0.0, sel_min_c,
                     sel_alloc, None),
    }


def _build_edge_specs(names, acts_by_d, util_by_d, action_utilities,
                      contracts, edge_weight_mode):
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    act_idx_by_name = [
        {act.name: idx for idx, act in enumerate(acts)} for acts in acts_by_d
    ]
    degrees = {name: 0 for name in names}
    for edge in contracts:
        if edge is not None:
            degrees[edge.first] = degrees.get(edge.first, 0) + 1
            degrees[edge.second] = degrees.get(edge.second, 0) + 1
    specs = []
    for edge in contracts:
        if edge is None:
            continue
        i_f = name_to_idx[edge.first]
        i_s = name_to_idx[edge.second]
        deg_f = max(1, degrees.get(edge.first, 1))
        deg_s = max(1, degrees.get(edge.second, 1))
        if edge_weight_mode == "inverse_max_degree":
            w_e = 1.0 / float(max(deg_f, deg_s))
        elif edge_weight_mode == "inverse_mean_degree":
            w_e = 2.0 / float(deg_f + deg_s)
        else:
            w_e = 1.0
        f_ref_idx = act_idx_by_name[i_f][edge.first_action.name]
        s_ref_idx = act_idx_by_name[i_s][edge.second_action.name]
        f_base = action_utilities[(edge.first, edge.first_action.name)] - edge.first_gain_m
        s_base = action_utilities[(edge.second, edge.second_action.name)] - edge.second_gain_m
        f_dists = [w_e * _action_distance(a, edge.first_action) for a in acts_by_d[i_f]]
        s_dists = [w_e * _action_distance(a, edge.second_action) for a in acts_by_d[i_s]]
        f_gains = [u - f_base for u in util_by_d[i_f]]
        s_gains = [u - s_base for u in util_by_d[i_s]]
        specs.append((
            i_f, i_s, f_ref_idx, s_ref_idx, f_dists, s_dists, f_gains, s_gains,
            edge.reference_share, w_e / max(edge.surplus_m, 1.0)))
    return specs


def _eval_single_edge_metrics_from_specs(edge_specs, idx_t):
    broken = 0
    alloc_err = 0.0
    for (i_f, i_s, f_ref_idx, s_ref_idx, _fd, _sd,
         f_gains, s_gains, ref_share, inv_scale) in edge_specs:
        fi = idx_t[i_f]
        si = idx_t[i_s]
        if fi != f_ref_idx or si != s_ref_idx:
            broken += 1
        fg = f_gains[fi]
        sg = s_gains[si]
        alloc_err += (
            abs(fg - ref_share * (fg + sg)) +
            (0.0 if fg >= 0.0 else -fg) +
            (0.0 if sg >= 0.0 else -sg)) * inv_scale
    return broken, alloc_err


def _eval_single_edge_metrics(names, acts_by_d, util_by_d, action_utilities,
                              contracts, edge_weight_mode, idx_t):
    specs = _build_edge_specs(
        names, acts_by_d, util_by_d, action_utilities, contracts,
        edge_weight_mode)
    return _eval_single_edge_metrics_from_specs(specs, idx_t)


def _safe_disagreement_row(scene, names, cars, actions, safe_rows):
    """Select one executable fleet-wide conservative reference plan.

    Pairwise disagreement actions need not compose. The reference is the safe
    joint candidate closest to explicit scene-specific conservative commands.
    This gives every AV the same, jointly executable utility baseline.
    """
    nominal_names = {}
    for name in names:
        car = cars[name]
        if scene == "crossing":
            nominal_names[name] = "cross_-4"
        elif scene == "chase":
            nominal_names[name] = "a+0" if car.role == "front" else "a-4"
        elif scene == "merge":
            nominal_names[name] = ("abort_brake" if car.role == "merging"
                                   else "mainline_+0")
        else:
            nominal_names[name] = "brake_keep_lane"
    nominal = {name: next(action for action in actions[name]
                          if action.name == nominal_names[name])
               for name in names}
    return min(safe_rows, key=lambda row: (
        sum(_action_distance(row[0][name], nominal[name]) for name in names),
        -row[4], -row[1],
        tuple((name, row[0][name].name) for name in names)))


def _sequential_pairwise_targets(names, cars, actions, contracts):
    """Apply local K_e proposals in fixed TTC order, then overwrite conflicts.

    This is a transparent reconstruction baseline for the new contract
    interface.  It is not the archived author's unavailable multi-car code.
    The last edge in the deterministic order controls a vehicle's raw target;
    the common hard-safety/welfare filter projects the resulting targets onto
    the same joint candidate set used by the global method.
    """
    order = sorted((edge for edge in contracts if edge is not None),
                   key=lambda edge: (edge.ttc_s if math.isfinite(edge.ttc_s)
                                     else math.inf,
                                     edge.first, edge.second))
    targets = {}
    history = []
    incompatible = 0
    for edge_index, edge in enumerate(order):
        proposed = {
            edge.first: edge.first_action,
            edge.second: edge.second_action,
        }
        overwritten = {}
        for name, action in proposed.items():
            previous = targets.get(name)
            if previous is not None and previous.name != action.name:
                incompatible += 1
                overwritten[name] = {
                    "previous_action": previous.name,
                    "new_action": action.name,
                }
            targets[name] = action
        history.append({
            "order": edge_index,
            "edge": [edge.first, edge.second],
            "ttc_s": edge.ttc_s if math.isfinite(edge.ttc_s) else "infinity",
            "proposed_actions": {name: action.name
                                 for name, action in proposed.items()},
            "overwritten_vehicle_targets": overwritten,
        })
    for name in names:
        if name not in targets:
            targets[name] = min(actions[name], key=lambda action: (
                abs(action.accel), abs(action.target_y - _lane_target(cars[name], 0)),
                action.name))
    return targets, history, incompatible


def _contract_record(scene, edge, plans):
    def path_record(path):
        return [{
            "time_s": round(index * MODEL["dt_s"], 3),
            "position_xy_m": [round(car.x, 5), round(car.y, 5)],
            "speed_mps": round(car.speed, 5),
            "velocity_xy_mps": [round(value, 5)
                                for value in _world_velocity(car)],
        } for index, car in enumerate(path)]
    first_path = plans[edge.first][edge.first_action.name]
    second_path = plans[edge.second][edge.second_action.name]
    return {
        "edge": [edge.first, edge.second],
        "scene_type": scene,
        "protocol": "TTC_calibrated_pairwise_Rubinstein_inspired_contract",
        "predecision_ttc_s": (edge.ttc_s if math.isfinite(edge.ttc_s)
                              else "infinity"),
        "ttc_source": edge.ttc_source,
        "legacy_radial_ttc_s": (
            edge.legacy_radial_ttc_s if math.isfinite(
                edge.legacy_radial_ttc_s) else "infinity"),
        "discount_factors": {
            edge.first: edge.delta_first,
            edge.second: edge.delta_second,
        },
        "reference_share_first": edge.reference_share,
        "realized_share_first": edge.realized_share_first,
        "share_error": edge.share_error,
        "eligible_pair_count": edge.eligible_pair_count,
        "discount_mode": edge.discount_mode,
        "ttc_active": edge.ttc_active,
        "ttc_applied": edge.ttc_applied,
        "individual_gains_m": {
            edge.first: edge.first_gain_m,
            edge.second: edge.second_gain_m,
        },
        "total_surplus_m": edge.surplus_m,
        "K_e": {
            "tolerance": edge.tolerance,
            "C_e_definition": "singleton action-pair reference, filtered by the common inflated-rectangle safety rule",
            "reference_actions": {
                edge.first: edge.first_action.name,
                edge.second: edge.second_action.name,
            },
            "reference_trajectory_pair": {
                edge.first: path_record(first_path),
                edge.second: path_record(second_path),
            },
        },
    }


def _choose(scene, cars, edges, method, policy=None,
            shared_action_names=None):
    policy = normalize_policy(policy)
    started = time.perf_counter()
    names = tuple(sorted(name for name, car in cars.items() if not car.finished))
    if not names:
        return {}, {"fallback": False, "raw_unsafe": 0, "min_clearance": None,
                    "solver_ms": 0.0, "contract_count": 0, "broken_contracts": 0,
                    "safe_disagreement_available": True,
                    "individual_rational": True,
                    "per_vehicle_gain_m": {}, "ir_candidate_count": 0,
                    "safe_candidate_count": 0}
    if len(names) == 1:
        name = names[0]
        car = cars[name]
        action = Action("free_progress", 1.0, _lane_target(car, 0))
        return {name: action}, {"fallback": False, "raw_unsafe": 0,
                                "min_clearance": None,
                                "solver_ms": (time.perf_counter() - started) * 1000.0,
                                "contract_count": 0, "broken_contracts": 0,
                                "safe_disagreement_available": True,
                                "individual_rational": True,
                                "per_vehicle_gain_m": {name: 0.0},
                                "ir_candidate_count": 1,
                                "safe_candidate_count": 1}
    live_cars = {name: cars[name] for name in names}
    base_actions = {name: _actions(scene, live_cars[name]) for name in names}
    base_plans = {name: {action.name: _rollout(
        live_cars[name], action, MODEL["dt_s"], MODEL["horizon_s"])
        for action in base_actions[name]} for name in names}
    adaptive = (scene == "crossing" and
                policy["decoder_mode"] == "adaptive_crossing")
    if shared_action_names is not None and not adaptive:
        raise ValueError("shared_action_names requires adaptive_crossing")
    if adaptive:
        decoder_actions = {name: _adaptive_crossing_actions(live_cars[name])
                           for name in names}
        decoder_plans = {name: {action.name: _rollout(
            live_cars[name], action, MODEL["dt_s"], MODEL["horizon_s"])
            for action in decoder_actions[name]} for name in names}
    else:
        decoder_actions, decoder_plans = base_actions, base_plans
    base_pair_clearances = {}
    base_pair_next_clearances = {}
    base_action_utilities = {}
    if not adaptive:
        for name in names:
            for action in base_actions[name]:
                base_action_utilities[(name, action.name)] = _utility(
                    scene, live_cars[name], action,
                    base_plans[name][action.name], MODEL["horizon_s"])
        for index, first_name in enumerate(names):
            for second_name in names[index + 1:]:
                for first_action in base_actions[first_name]:
                    for second_action in base_actions[second_name]:
                        key = (first_name, first_action.name,
                               second_name, second_action.name)
                        min_clear, next_clear = _pair_clearance_and_next(
                            scene, base_plans[first_name][first_action.name],
                            base_plans[second_name][second_action.name],
                            live_cars[first_name], live_cars[second_name])
                        base_pair_clearances[key] = min_clear
                        base_pair_next_clearances[key] = next_clear
    nominal_plans = (
        {name: _rollout(live_cars[name], _nominal_action(scene, live_cars[name]),
                        MODEL["dt_s"], policy["ttc_start_s"])
         for name in names}
        if policy["ttc_source"] != "legacy_radial" else None)
    contracts = []
    contract_attempts = []
    for first_name, second_name in edges:
        if first_name in live_cars and second_name in live_cars:
            contract = _pair_contract(
                scene, live_cars[first_name], live_cars[second_name],
                decoder_actions, decoder_plans, policy,
                pair_clearances=(None if adaptive else base_pair_clearances),
                action_utilities=(None if adaptive else base_action_utilities),
                nominal_plans=nominal_plans)
            contracts.append(contract)
            contract_attempts.append({
                "edge": [first_name, second_name],
                "status": "decoded" if contract is not None else "no_contract",
                "failure_reason": (None if contract is not None else
                                   "no_safe_individually_rational_pair_or_unsafe_disagreement"),
            })
    # The pair grid is never enumerated jointly. Each edge contributes at
    # most one reference action per endpoint to the common finite family.
    if adaptive:
        action_maps = {name: {action.name: action for action in base_actions[name]}
                       for name in names}
        for contract in contracts:
            if contract is not None:
                action_maps[contract.first][contract.first_action.name] = (
                    contract.first_action)
                action_maps[contract.second][contract.second_action.name] = (
                    contract.second_action)
        if shared_action_names is not None:
            if set(shared_action_names) != set(names):
                raise ValueError("shared_action_names must cover live cars")
            for name in names:
                dense_by_name = {action.name: action
                                 for action in decoder_actions[name]}
                for action_name in shared_action_names[name]:
                    if action_name not in dense_by_name:
                        raise ValueError("Unknown shared crossing action")
                    action_maps[name][action_name] = dense_by_name[action_name]
        actions = {name: tuple(action_maps[name].values()) for name in names}
        plans = {name: {action.name: decoder_plans[name][action.name]
                        for action in actions[name]} for name in names}
    else:
        actions, plans = base_actions, base_plans
    current_min_clearance = min(
        _envelope_clearance(live_cars[first_name].x,
                            live_cars[first_name].y, live_cars[first_name],
                            live_cars[second_name].x,
                            live_cars[second_name].y, live_cars[second_name])
        for index, first_name in enumerate(names)
        for second_name in names[index + 1:])
    current_full_margin_satisfied = (
        current_min_clearance >=
        MODEL["robust_prediction_margin_m"] - 1e-9)
    if adaptive:
        pair_clearances = {}
        pair_next_clearances = {}
        action_utilities = {}
        for name in names:
            for action in actions[name]:
                action_utilities[(name, action.name)] = _utility(
                    scene, live_cars[name], action, plans[name][action.name],
                    MODEL["horizon_s"])
        for index, first_name in enumerate(names):
            for second_name in names[index + 1:]:
                for first_action in actions[first_name]:
                    for second_action in actions[second_name]:
                        key = (first_name, first_action.name,
                               second_name, second_action.name)
                        min_clear, next_clear = _pair_clearance_and_next(
                            scene, plans[first_name][first_action.name],
                            plans[second_name][second_action.name],
                            live_cars[first_name], live_cars[second_name])
                        pair_clearances[key] = min_clear
                        pair_next_clearances[key] = next_clear
    else:
        pair_clearances = base_pair_clearances
        pair_next_clearances = base_pair_next_clearances
        action_utilities = base_action_utilities
    sequential_targets, edge_history, incompatible_targets = (
        _sequential_pairwise_targets(names, live_cars, actions, contracts))
    raw_choice = {name: sequential_targets[name] for name in names}
    raw_unsafe = 0
    for first_index, first_name in enumerate(names):
        for second_name in names[first_index + 1:]:
            if pair_clearances[(first_name, raw_choice[first_name].name,
                                second_name, raw_choice[second_name].name)] < -1e-9:
                raw_unsafe += 1
    # Enumerate the safe joint family once, then apply the welfare floor.
    # The previous implementation repeated the full Cartesian-product scan
    # solely to filter by this floor, inflating the worst-case control-cycle
    # time without changing the selected candidate.
    dynamic_filter_active = (scene == "chase" and
                             policy["dynamic_braking_filter"])
    current_dynamic_slack = (_robust_chase_braking_slack(live_cars)
                             if scene == "chase" else None)
    dynamic_braking_threshold = None
    dynamic_braking_rejected_count = 0
    dynamic_braking_infeasible = False
    disagreement_choice = None
    disagreement_utilities = {}
    welfare_star = -float("inf")
    selected = None
    if dynamic_filter_active:
        safe_rows = _safe_joint_rows(
            scene, names, live_cars, actions, plans, edges, contracts,
            -float("inf"), pair_clearances, action_utilities,
            compute_chase_slack=True,
            edge_weight_mode=policy["edge_weight_mode"])
        buffer = MODEL["dynamic_braking_slack_buffer_m"]
        dynamic_braking_threshold = (
            buffer if current_dynamic_slack is None or
            current_dynamic_slack >= buffer else current_dynamic_slack - 0.05)
        screened = [row for row in safe_rows
                    if row[6] is None or
                    row[6] >= dynamic_braking_threshold - 1e-9]
        dynamic_braking_rejected_count = len(safe_rows) - len(screened)
        dynamic_braking_infeasible = bool(safe_rows and not screened)
        safe_rows = screened
        disagreement = None
        ir_rows = []
        if safe_rows:
            disagreement = _safe_disagreement_row(
                scene, names, live_cars, actions, safe_rows)
            disagreement_choice = disagreement[0]
            disagreement_utilities = {
                name: action_utilities[(name, disagreement[0][name].name)]
                for name in names}
            ir_rows = [row for row in safe_rows if all(
                action_utilities[(name, row[0][name].name)] >=
                disagreement_utilities[name] - 1e-9 for name in names)]
            family = (ir_rows if policy["enforce_individual_rationality"]
                      else safe_rows)
            welfare_star = max(row[1] for row in family)
            floor = welfare_star - policy["welfare_epsilon_m"]
            rows = [row for row in family if row[1] >= floor - 1e-9]
        else:
            rows = []
        safe_candidate_count = len(safe_rows)
        ir_candidate_count = len(ir_rows)
        welfare_eligible_count = len(rows)
        if method == "safe_disagreement" and disagreement is not None:
            selected = disagreement
        elif rows:
            if method == "global":
                if policy["joint_objective"] == "share_then_contract":
                    selected = min(rows, key=lambda row: (
                        row[5], row[2], row[3], -row[1], row[7]))
                elif policy["joint_objective"] == "contract_then_share":
                    selected = min(rows, key=lambda row: (
                        row[2], row[5], row[3], -row[1], row[7]))
                else:
                    selected = min(rows, key=lambda row: (
                        row[2], row[3], -row[1], row[7]))
            elif method == "sequential_pairwise_reconstruction":
                selected = min(rows, key=lambda row: (
                    sum(_action_distance(row[0][name], sequential_targets[name])
                        for name in names), -row[1], row[7]))
            elif method == "welfare_only":
                selected = min(rows, key=lambda row: (-row[1], row[7]))
            elif method == "clearance_first":
                # A non-bargaining, conservative yet progressing comparator.
                # The common welfare floor prevents all-stop deadlock from being
                # selected merely to maximize separation.
                selected = min(rows, key=lambda row: (-row[4], -row[1], row[7]))
            else:
                raise ValueError("Unknown joint selection method")
    else:
        fast_result = _select_joint_fast(
            scene, names, live_cars, actions, contracts,
            pair_clearances, action_utilities, method, policy,
            sequential_targets)
        if fast_result is not None:
            safe_candidate_count = fast_result["safe_count"]
            ir_candidate_count = fast_result["ir_count"]
            welfare_eligible_count = fast_result["eligible_count"]
            disagreement_choice = fast_result["disagreement_choice"]
            disagreement_utilities = fast_result["disagreement_utilities"]
            welfare_star = fast_result["welfare_star"]
            selected = fast_result["selected"]
        else:
            safe_candidate_count = 0
            ir_candidate_count = 0
            welfare_eligible_count = 0
    fallback = False
    recovery_profile_used = False
    recovery_candidate_count = 0
    selected_source = "standard_joint_candidate"
    next_step_safe_candidate_count = None
    next_step_collision_free_candidate_count = None
    fallback_safety_tier = None
    if selected is None:
        # A 2 s constant-action plan can be empty even when a safe trajectory
        # exists after an early-braking switch. Search a small staged-braking
        # family for chase scenes before dropping to a one-step fallback.
        fallback = True
        # A full-horizon robust plan cannot repair a margin violation that is
        # already present at t=0, because every rollout includes that state.
        # Skip the staged search there and go directly to the separately
        # logged one-step physical fallback.
        recovery_search_applicable = (
            scene in ("chase", "crossing", "merge", "head_on") and
            current_full_margin_satisfied)
        recovery_actions = ({name: _recovery_actions(
            scene, live_cars[name], actions[name],
            [live_cars[other] for other in names if other != name])
            for name in names}
            if recovery_search_applicable else {name: () for name in names})
        recovery_plans = {
            name: {action.name: (
                plans[name][action.name] if action.name in plans[name] else
                _rollout(live_cars[name], action, MODEL["dt_s"],
                         MODEL["horizon_s"]))
                   for action in recovery_actions[name]}
            for name in names}
        recovery_utilities = {
            (name, action.name): (
                action_utilities[(name, action.name)]
                if (name, action.name) in action_utilities else
                _utility(scene, live_cars[name], action,
                         recovery_plans[name][action.name],
                         MODEL["horizon_s"]))
            for name in names for action in recovery_actions[name]}
        recovery_pair_clearances = {}
        recovery_pair_next_clearances = {}
        for index, first_name in enumerate(names):
            for second_name in names[index + 1:]:
                for first_action in recovery_actions[first_name]:
                    for second_action in recovery_actions[second_name]:
                        key = (first_name, first_action.name,
                               second_name, second_action.name)
                        if key in pair_clearances:
                            recovery_pair_clearances[key] = pair_clearances[key]
                            recovery_pair_next_clearances[key] = (
                                pair_next_clearances[key])
                        else:
                            first_path = recovery_plans[first_name][first_action.name]
                            second_path = recovery_plans[second_name][second_action.name]
                            min_c, next_c = _pair_clearance_and_next(
                                scene, first_path, second_path,
                                live_cars[first_name], live_cars[second_name])
                            recovery_pair_clearances[key] = min_c
                            recovery_pair_next_clearances[key] = next_c
        recovery_rows = (_safe_joint_rows(
            scene, names, live_cars, recovery_actions, recovery_plans,
            edges, contracts, -float("inf"), recovery_pair_clearances,
            recovery_utilities, compute_chase_slack=dynamic_filter_active)
            if recovery_search_applicable else [])
        if dynamic_filter_active and recovery_rows:
            screened_recovery = [row for row in recovery_rows
                                 if row[6] is None or
                                 row[6] >= dynamic_braking_threshold - 1e-9]
            dynamic_braking_rejected_count += (
                len(recovery_rows) - len(screened_recovery))
            if not screened_recovery:
                dynamic_braking_infeasible = True
            recovery_rows = screened_recovery
        recovery_candidate_count = len(recovery_rows)
        if recovery_rows:
            # Safety is primary in this recovery-only branch; welfare breaks
            # ties between plans with the same minimum clearance.
            selected = min(recovery_rows, key=lambda row: (
                -row[4], -row[1], tuple(sorted(
                    (name, action.name) for name, action in row[0].items()))))
            recovery_profile_used = any(
                action.accel_schedule for action in selected[0].values())
            selected_source = "piecewise_braking_recovery"
            fallback_safety_tier = (
                "full_horizon_piecewise_braking" if selected[4] >= -1e-9
                else "piecewise_braking_best_effort")
            actions, plans = recovery_actions, recovery_plans
            action_utilities = recovery_utilities
            pair_clearances = recovery_pair_clearances
            pair_next_clearances = recovery_pair_next_clearances
        else:
            selected_source = "one_step_fallback"
            # A safe 0.2 s control may remain even when no 2 s continuation
            # is certified. Preserve the best available next-step tier.
            emergency = []
            for selected_actions in itertools.product(*(actions[name] for name in names)):
                choice = dict(zip(names, selected_actions))
                min_clearance = float("inf")
                next_clearance = float("inf")
                for i, first_name in enumerate(names):
                    for second_name in names[i + 1:]:
                        key = (first_name, choice[first_name].name,
                               second_name, choice[second_name].name)
                        min_clearance = min(min_clearance,
                                            pair_clearances[key])
                        next_clearance = min(next_clearance,
                                             pair_next_clearances[key])
                welfare = sum(action_utilities[(name, choice[name].name)]
                              for name in names)
                next_dynamic_slack = (
                    _predicted_min_robust_chase_slack(
                        live_cars, names, plans, choice)
                    if dynamic_filter_active else None)
                emergency.append((choice, welfare, min_clearance,
                                  next_clearance, next_dynamic_slack))
            full_margin = [row for row in emergency if row[3] >= -1e-9]
            physical = [row for row in emergency if row[3] >=
                        -MODEL["robust_prediction_margin_m"] - 1e-9]
            next_step_safe_candidate_count = len(full_margin)
            next_step_collision_free_candidate_count = len(physical)
            if full_margin:
                emergency_family = full_margin
                fallback_safety_tier = "full_prediction_margin"
            elif physical:
                emergency_family = physical
                fallback_safety_tier = "physical_envelope_only"
            else:
                emergency_family = emergency
                fallback_safety_tier = "uncertified_next_step"
            if dynamic_filter_active:
                dynamically_admissible = [
                    row for row in emergency_family
                    if row[4] is None or
                    row[4] >= dynamic_braking_threshold - 1e-9]
                dynamic_braking_rejected_count += (
                    len(emergency_family) - len(dynamically_admissible))
                if dynamically_admissible:
                    emergency_family = dynamically_admissible
                else:
                    dynamic_braking_infeasible = True
            selected_emergency = max(
                emergency_family,
                # In fallback mode immediate physical clearance is the first
                # priority. A rollout can be uncertified later in its horizon;
                # sacrificing the next 0.2 s for a slightly better distant
                # minimum can turn a recoverable state into a collision.
                key=lambda row: ((row[4] if dynamic_filter_active and
                                   row[4] is not None else float("-inf")),
                                 row[3], row[2], row[1],
                                 tuple(sorted((name, action.name)
                                              for name, action in row[0].items()))))
            selected = (selected_emergency[0], selected_emergency[1],
                        sum(edge is None for edge in contracts), 0.0,
                        selected_emergency[2], None,
                        selected_emergency[4])
    selected_next_clearance = min(
        pair_next_clearances[(first_name, selected[0][first_name].name,
                              second_name, selected[0][second_name].name)]
        for index, first_name in enumerate(names)
        for second_name in names[index + 1:])
    per_vehicle_gain = {
        name: action_utilities[(name, selected[0][name].name)] -
              disagreement_utilities[name]
        for name in names if name in disagreement_utilities}
    result = {
        "fallback": fallback,
        "fallback_reason": (
            "piecewise_braking_recovery_after_constant_action_failure"
            if recovery_profile_used else
            "current_state_below_full_prediction_margin"
            if fallback and not current_full_margin_satisfied else
            "no_horizon_safe_joint_candidate" if fallback else None),
        "selected_source": selected_source,
        "recovery_profile_used": recovery_profile_used,
        "recovery_candidate_count": recovery_candidate_count,
        "current_min_base_clearance_m": current_min_clearance,
        "current_full_margin_satisfied": current_full_margin_satisfied,
        "dynamic_braking_filter_active": dynamic_filter_active,
        "current_robust_braking_slack_m": current_dynamic_slack,
        "dynamic_braking_threshold_m": dynamic_braking_threshold,
        "dynamic_braking_candidate_rejected_count": (
            dynamic_braking_rejected_count),
        "dynamic_braking_filter_infeasible": dynamic_braking_infeasible,
        "selected_predicted_robust_braking_slack_m": (
            selected[6] if len(selected) > 6 else None),
        "selected_accel_schedules": {
            name: list(action.accel_schedule)
            for name, action in selected[0].items()
            if action.accel_schedule},
        "next_step_safe_candidate_count": next_step_safe_candidate_count,
        "next_step_collision_free_candidate_count": (
            next_step_collision_free_candidate_count),
        "fallback_safety_tier": fallback_safety_tier,
        "next_step_predicted_clearance_m": selected_next_clearance,
        "next_step_predicted_safe": selected_next_clearance >= -1e-9,
        "next_step_predicted_collision_free": (
            selected_next_clearance >=
            -MODEL["robust_prediction_margin_m"] - 1e-9),
        "raw_unsafe": raw_unsafe,
        "min_clearance": selected[4],
        "solver_ms": (time.perf_counter() - started) * 1000.0,
        "contract_count": sum(edge is not None for edge in contracts),
        "decoder_mode": policy["decoder_mode"],
        "action_count_per_vehicle": {name: len(actions[name]) for name in names},
        "joint_candidate_actions": {
            name: sorted(action.name for action in actions[name])
            for name in names},
        "mean_decoded_share_error": (
            sum(edge.share_error for edge in contracts if edge is not None) /
            max(1, sum(edge is not None for edge in contracts))),
        "broken_contracts": selected[2],
        "safe_candidate_count": safe_candidate_count,
        "ir_candidate_count": ir_candidate_count,
        "welfare_eligible_count": welfare_eligible_count,
        "safe_disagreement_available": disagreement_choice is not None,
        "safe_disagreement_actions": (
            {name: disagreement_choice[name].name for name in names}
            if disagreement_choice is not None else {}),
        "individual_rational": (disagreement_choice is not None and all(
            gain >= -1e-9 for gain in per_vehicle_gain.values())),
        "per_vehicle_gain_m": per_vehicle_gain,
        "selected_welfare_m": selected[1],
        "selected_allocation_error": (selected[5] if len(selected) > 5
                                      else None),
        "welfare_optimum_m": (welfare_star if safe_candidate_count else None),
        "ttc_active_contract_count": sum(
            edge is not None and edge.ttc_active for edge in contracts),
        "ttc_applied_contract_count": sum(
            edge is not None and edge.ttc_applied for edge in contracts),
        "incompatible_edge_target_count": incompatible_targets,
        "sequential_edge_order": edge_history,
        "contract_records": [_contract_record(scene, edge, plans)
                             for edge in contracts if edge is not None],
        "contract_attempts": contract_attempts,
    }
    return selected[0], result


def _trial_initial(scene, n_cars, seed, position_std_m, speed_std_mps):
    cars, edges = make_scene(scene, n_cars)
    rng = random.Random(int(seed))
    perturbed = {}
    for name, car in cars.items():
        longitudinal_delta = rng.gauss(0.0, position_std_m)
        speed_delta = rng.gauss(0.0, speed_std_mps)
        perturbed[name] = Car(
            car.id,
            car.x + car.direction * longitudinal_delta if car.axis == "x" else car.x,
            car.y + car.direction * longitudinal_delta if car.axis == "y" else car.y,
            car.direction, max(0.5, min(car.max_speed, car.speed + speed_delta)),
            car.lateral_speed, car.role, car.goal_x, car.max_speed, axis=car.axis)
    count = int(math.ceil(MODEL["episode_horizon_s"] / MODEL["dt_s"]))
    noises = {name: [rng.gauss(0.0, MODEL["accel_noise_std_mps2"])
                     for _ in range(count)]
              for name in sorted(perturbed)}
    return perturbed, edges, noises


def _actual_clearance(scene, cars):
    names = sorted(name for name, car in cars.items() if not car.finished)
    minimum = float("inf")
    for index, first_name in enumerate(names):
        for second_name in names[index + 1:]:
            a, b = cars[first_name], cars[second_name]
            clearance = _envelope_clearance(a.x, a.y, a, b.x, b.y, b)
            minimum = min(minimum, clearance)
    return minimum


def _initial_chase_braking_slack(cars):
    """Best-case relative-braking slack for initially closing same-lane pairs.

    This diagnostic assumes the rear vehicle brakes at the model limit while
    the lead vehicle accelerates at its model limit. It is not a safety
    certificate; it helps distinguish a geometrically separated start from
    one that has almost no dynamic recovery room.
    """
    live = [car for car in cars.values() if not car.finished]
    best_slack = None
    max_relative_decel = (MODEL["max_decel_mps2"] +
                           MODEL["max_accel_mps2"])
    for index, first in enumerate(live):
        for second in live[index + 1:]:
            if (first.axis != second.axis or
                    first.direction != second.direction):
                continue
            if abs(_lateral_position(first) -
                   _lateral_position(second)) >= (
                    MODEL["vehicle_width_m"] +
                    MODEL["lateral_margin_m"]):
                continue
            first_pos = _longitudinal_position(first)
            second_pos = _longitudinal_position(second)
            first_ahead = first.direction * (first_pos - second_pos) > 0.0
            rear, lead = ((second, first) if first_ahead else (first, second))
            closing_speed = rear.speed - lead.speed
            if closing_speed <= 1e-6:
                continue
            clearance = _envelope_clearance(
                rear.x, rear.y, rear, lead.x, lead.y, lead)
            stopping_distance = (closing_speed * closing_speed /
                                 (2.0 * max_relative_decel))
            slack = clearance - stopping_distance
            best_slack = slack if best_slack is None else min(best_slack, slack)
    return best_slack


def _robust_chase_braking_slack(cars):
    """Estimate braking slack using a 3-sigma acceleration-noise tightening.

    This is a screening diagnostic for same-lane closing pairs, not a safety
    certificate: the simulator uses unbounded Gaussian disturbances and the
    calculation assumes the model's maximum rear braking and lead acceleration.
    """
    live = [car for car in cars.values() if not car.finished]
    noise_tightening = (MODEL["dynamic_braking_sigma"] * math.sqrt(2.0) *
                        MODEL["accel_noise_std_mps2"])
    relative_decel = (MODEL["max_decel_mps2"] + MODEL["max_accel_mps2"] -
                      noise_tightening)
    if relative_decel <= 1e-9:
        return float("-inf")
    worst_slack = None
    for index, first in enumerate(live):
        for second in live[index + 1:]:
            if (first.axis != second.axis or
                    first.direction != second.direction):
                continue
            if abs(_lateral_position(first) -
                   _lateral_position(second)) >= (
                    MODEL["vehicle_width_m"] +
                    MODEL["lateral_margin_m"]):
                continue
            first_pos = _longitudinal_position(first)
            second_pos = _longitudinal_position(second)
            first_ahead = first.direction * (first_pos - second_pos) > 0.0
            rear, lead = ((second, first) if first_ahead else (first, second))
            closing_speed = rear.speed - lead.speed
            if closing_speed <= 1e-6:
                continue
            clearance = _envelope_clearance(
                rear.x, rear.y, rear, lead.x, lead.y, lead)
            stopping_distance = closing_speed * closing_speed / (
                2.0 * relative_decel)
            slack = clearance - stopping_distance
            worst_slack = (slack if worst_slack is None else
                           min(worst_slack, slack))
    return worst_slack


def _predicted_min_robust_chase_slack(cars, names, plans, choice,
                                      include_current=False):
    """Minimum sampled robust braking slack over a joint rollout."""
    if not names:
        return None
    paths = [plans[name][choice[name].name] for name in names]
    path_length = min(len(path) for path in paths)
    first_index = 0 if include_current else 1
    values = []
    for index in range(first_index, path_length):
        state = {name: plans[name][choice[name].name][index]
                 for name in names}
        slack = _robust_chase_braking_slack(state)
        if slack is not None:
            values.append(slack)
    return min(values) if values else None


def _swept_actual_clearance(previous, current):
    """Check exact swept base-envelope clearance for each executed segment."""
    names = sorted(name for name, car in previous.items() if not car.finished)
    minimum = float("inf")
    for index, first_name in enumerate(names):
        for second_name in names[index + 1:]:
            first0, first1 = previous[first_name], current[first_name]
            second0, second1 = previous[second_name], current[second_name]
            minimum = min(minimum, _segment_min_clearance(
                first0, first1, second0, second1, first0, second0,
                _pair_envelope_radii(first0, second0)))
    return minimum


def _goals_met(scene, cars):
    for car in cars.values():
        if car.finished:
            continue
        if car.direction * (_longitudinal_position(car) - car.goal_x) < 0.0:
            return False
        if scene == "merge" and car.role == "merging":
            if abs(car.y) > 0.4 or abs(car.lateral_speed) > 0.25:
                return False
    return True


def run_episode(scene, n_cars, trial_seed, method,
                position_std_m=0.3, speed_std_mps=0.2, policy=None):
    if method not in VALID_METHODS:
        raise ValueError("unknown controller method")
    policy = normalize_policy(policy)
    cars, edges, noise = _trial_initial(
        scene, n_cars, trial_seed, position_std_m, speed_std_mps)
    initial = {name: car for name, car in cars.items()}
    initial_physical_clearance = _actual_clearance(scene, cars)
    initial_braking_slack = (_initial_chase_braking_slack(cars)
                             if scene == "chase" else None)
    initial_robust_braking_slack = (
        _robust_chase_braking_slack(cars) if scene == "chase" else None)
    initial_dynamic_margin_satisfied = (
        None if initial_robust_braking_slack is None else
        initial_robust_braking_slack >=
        MODEL["dynamic_braking_slack_buffer_m"] - 1e-9)
    dt_s = MODEL["dt_s"]
    count = int(math.ceil(MODEL["episode_horizon_s"] / dt_s))
    trace = []
    elapsed_by_car = {}
    summaries = {"fallback_count": 0, "recovery_step_count": 0,
                 "recovery_profile_step_count": 0,
                 "recovery_candidate_count": 0,
                 "current_margin_deficit_ticks": 0,
                 "next_step_uncertified_fallback_ticks": 0,
                 "next_step_collision_uncertified_fallback_ticks": 0,
                 "raw_pair_unsafe_ticks": 0, "broken_contracts": 0,
                 "min_clearance": float("inf"), "mean_solver_ms": 0.0,
                 "solver_times_ms": [], "ir_violation_ticks": 0,
                 "safe_disagreement_unavailable_ticks": 0,
                 "ttc_active_contracts": 0, "ttc_applied_contracts": 0,
                 "total_contracts": 0, "min_predicted_gain_m": float("inf"),
                 "weighted_share_error_sum": 0.0,
                 "dynamic_braking_candidate_rejected_count": 0,
                 "dynamic_braking_filter_infeasible_ticks": 0,
                 "dynamic_braking_filter_active_ticks": 0}
    collision = False
    near_collision = False
    completed = False
    collision_time = None
    decision_trace = []
    for step in range(count):
        now = step * dt_s
        controls, plan_info = _choose(scene, cars, edges, method, policy)
        summaries["fallback_count"] += int(plan_info["fallback"])
        summaries["recovery_step_count"] += int(plan_info["fallback"])
        summaries["recovery_profile_step_count"] += int(
            plan_info.get("recovery_profile_used", False))
        summaries["recovery_candidate_count"] += plan_info.get(
            "recovery_candidate_count", 0)
        summaries["current_margin_deficit_ticks"] += int(
            not plan_info.get("current_full_margin_satisfied", True))
        summaries["dynamic_braking_candidate_rejected_count"] += plan_info.get(
            "dynamic_braking_candidate_rejected_count", 0)
        summaries["dynamic_braking_filter_infeasible_ticks"] += int(
            plan_info.get("dynamic_braking_filter_infeasible", False))
        summaries["dynamic_braking_filter_active_ticks"] += int(
            plan_info.get("dynamic_braking_filter_active", False))
        summaries["next_step_uncertified_fallback_ticks"] += int(
            plan_info["fallback"] and not plan_info.get(
                "next_step_predicted_safe", True))
        summaries["next_step_collision_uncertified_fallback_ticks"] += int(
            plan_info["fallback"] and not plan_info.get(
                "next_step_predicted_collision_free", True))
        summaries["raw_pair_unsafe_ticks"] += int(plan_info["raw_unsafe"])
        summaries["broken_contracts"] += int(plan_info["broken_contracts"])
        summaries["ir_violation_ticks"] += int(
            not plan_info["individual_rational"] and
            plan_info["safe_disagreement_available"])
        summaries["safe_disagreement_unavailable_ticks"] += int(
            not plan_info["safe_disagreement_available"])
        summaries["ttc_active_contracts"] += plan_info.get(
            "ttc_active_contract_count", 0)
        summaries["ttc_applied_contracts"] += plan_info.get(
            "ttc_applied_contract_count", 0)
        summaries["total_contracts"] += plan_info["contract_count"]
        summaries["weighted_share_error_sum"] += (
            plan_info.get("mean_decoded_share_error", 0.0) *
            plan_info["contract_count"])
        if plan_info["per_vehicle_gain_m"]:
            summaries["min_predicted_gain_m"] = min(
                summaries["min_predicted_gain_m"],
                min(plan_info["per_vehicle_gain_m"].values()))
        summaries["mean_solver_ms"] += plan_info["solver_ms"]
        summaries["solver_times_ms"].append(plan_info["solver_ms"])
        step_noise = noise
        previous_cars = dict(cars)
        next_cars = dict(cars)
        for name, car in cars.items():
            if car.finished:
                continue
            action = controls.get(name)
            if action is None:
                action = Action("emergency_brake", -4.0, _lane_target(car, 0))
            value = step_noise[name][step]
            updated = _step_car(car, action, dt_s, value)
            arrival = None
            reached_x = car.direction * (_longitudinal_position(updated) -
                                          car.goal_x) >= 0.0
            merge_done = (scene != "merge" or car.role != "merging" or
                          (abs(updated.y) <= 0.4 and abs(updated.lateral_speed) <= 0.25))
            finished = reached_x and merge_done
            if finished:
                arrival = now + dt_s
                elapsed_by_car[name] = arrival
            next_cars[name] = Car(
                updated.id, updated.x, updated.y, updated.direction,
                updated.speed, updated.lateral_speed, updated.role,
                updated.goal_x, updated.max_speed, finished, arrival,
                updated.axis)
            trace.append({
                "time_s": round(now + dt_s, 3), "vehicle": name,
                "x_m": updated.x, "y_m": updated.y,
                "speed_mps": updated.speed,
                "lateral_speed_mps": updated.lateral_speed,
                "action": action.name,
                "selected_source": plan_info.get(
                    "selected_source", "standard_joint_candidate"),
                "accel_schedule": list(action.accel_schedule),
                "fallback": plan_info["fallback"],
                "contract_count": plan_info["contract_count"],
                "broken_contracts": plan_info["broken_contracts"],
                "predicted_min_clearance_m": plan_info["min_clearance"],
                "solver_ms": plan_info["solver_ms"],
            })
        decision_trace.append({
            "time_s": round(now, 3),
            "method": method,
            "proposed_action_pair_conflicts": plan_info.get(
                "incompatible_edge_target_count", 0),
            "sequential_edge_order": plan_info.get("sequential_edge_order", []),
            "contract_records": plan_info.get("contract_records", []),
            "contract_attempts": plan_info.get("contract_attempts", []),
            "selected_actions": {name: action.name
                                 for name, action in controls.items()},
            "predicted_min_clearance_m": plan_info["min_clearance"],
            "fallback": plan_info["fallback"],
            "fallback_reason": plan_info.get("fallback_reason"),
            "selected_source": plan_info.get(
                "selected_source", "standard_joint_candidate"),
            "recovery_profile_used": plan_info.get(
                "recovery_profile_used", False),
            "recovery_candidate_count": plan_info.get(
                "recovery_candidate_count", 0),
            "selected_accel_schedules": plan_info.get(
                "selected_accel_schedules", {}),
            "current_min_base_clearance_m": plan_info.get(
                "current_min_base_clearance_m"),
            "current_full_margin_satisfied": plan_info.get(
                "current_full_margin_satisfied", True),
            "dynamic_braking_filter_active": plan_info.get(
                "dynamic_braking_filter_active", False),
            "current_robust_braking_slack_m": plan_info.get(
                "current_robust_braking_slack_m"),
            "dynamic_braking_threshold_m": plan_info.get(
                "dynamic_braking_threshold_m"),
            "dynamic_braking_candidate_rejected_count": plan_info.get(
                "dynamic_braking_candidate_rejected_count", 0),
            "dynamic_braking_filter_infeasible": plan_info.get(
                "dynamic_braking_filter_infeasible", False),
            "selected_predicted_robust_braking_slack_m": plan_info.get(
                "selected_predicted_robust_braking_slack_m"),
            "next_step_predicted_safe": plan_info.get(
                "next_step_predicted_safe", True),
            "next_step_predicted_collision_free": plan_info.get(
                "next_step_predicted_collision_free", True),
            "fallback_safety_tier": plan_info.get("fallback_safety_tier"),
            "next_step_predicted_clearance_m": plan_info.get(
                "next_step_predicted_clearance_m"),
            "next_step_safe_candidate_count": plan_info.get(
                "next_step_safe_candidate_count"),
            "next_step_collision_free_candidate_count": plan_info.get(
                "next_step_collision_free_candidate_count"),
            "decoder_mode": policy["decoder_mode"],
            "action_count_per_vehicle": plan_info.get(
                "action_count_per_vehicle", {}),
            "mean_decoded_share_error": plan_info.get(
                "mean_decoded_share_error", 0.0),
            "selected_allocation_error": plan_info.get(
                "selected_allocation_error"),
            "safe_candidate_count": plan_info.get("safe_candidate_count", 0),
            "ir_candidate_count": plan_info.get("ir_candidate_count", 0),
            "safe_disagreement_available": plan_info["safe_disagreement_available"],
            "safe_disagreement_actions": plan_info.get(
                "safe_disagreement_actions", {}),
            "individual_rational": plan_info["individual_rational"],
            "per_vehicle_gain_m": plan_info["per_vehicle_gain_m"],
            "ttc_active_contract_count": plan_info.get(
                "ttc_active_contract_count", 0),
            "ttc_applied_contract_count": plan_info.get(
                "ttc_applied_contract_count", 0),
        })
        cars = next_cars
        clearance = _swept_actual_clearance(previous_cars, next_cars)
        summaries["min_clearance"] = min(summaries["min_clearance"], clearance)
        if clearance < -1e-9:
            collision = True
            collision_time = now + dt_s
            break
        if clearance < MODEL["near_clearance_m"]:
            near_collision = True
        if _goals_met(scene, cars):
            completed = True
            break
    summaries["mean_solver_ms"] /= max(1, step + 1)
    sorted_solver_times = sorted(summaries["solver_times_ms"])
    p95_index = max(0, int(math.ceil(0.95 * len(sorted_solver_times))) - 1)
    delays = {}
    for name, car in initial.items():
        distance = abs(car.goal_x - _longitudinal_position(car))
        free_time = distance / max(car.speed, 1e-6)
        arrival = elapsed_by_car.get(name)
        delays[name] = max(0.0, arrival - free_time) if arrival is not None else None
    valid_delays = [value for value in delays.values() if value is not None]
    return {
        "scene": scene, "n_cars": n_cars, "trial_seed": int(trial_seed),
        "method": method, "policy": policy,
        "initial_physical_clearance_m": initial_physical_clearance,
        "initial_best_case_braking_slack_m": initial_braking_slack,
        "initial_robust_braking_slack_m": initial_robust_braking_slack,
        "initial_dynamic_braking_margin_satisfied": (
            initial_dynamic_margin_satisfied),
        "initial_full_margin_satisfied": (
            initial_physical_clearance >=
            MODEL["robust_prediction_margin_m"] - 1e-9),
        "completed": completed, "collision": collision,
        "near_collision": near_collision, "collision_time_s": collision_time,
        "elapsed_s": (step + 1) * dt_s,
        "vehicle_arrival_s": elapsed_by_car,
        "per_vehicle_delay_s": delays,
        "mean_completed_delay_s": (
            sum(valid_delays) / len(valid_delays)
            if completed and valid_delays else None),
        "fallback_count": summaries["fallback_count"],
        "next_step_uncertified_fallback_ticks": summaries[
            "next_step_uncertified_fallback_ticks"],
        "next_step_full_margin_shortfall_ticks": summaries[
            "next_step_uncertified_fallback_ticks"],
        "next_step_collision_uncertified_fallback_ticks": summaries[
            "next_step_collision_uncertified_fallback_ticks"],
        "recovery_step_count": summaries["recovery_step_count"],
        "recovery_profile_step_count": summaries[
            "recovery_profile_step_count"],
        "recovery_candidate_count": summaries["recovery_candidate_count"],
        "current_margin_deficit_ticks": summaries[
            "current_margin_deficit_ticks"],
        "dynamic_braking_candidate_rejected_count": summaries[
            "dynamic_braking_candidate_rejected_count"],
        "dynamic_braking_filter_infeasible_ticks": summaries[
            "dynamic_braking_filter_infeasible_ticks"],
        "dynamic_braking_filter_active_ticks": summaries[
            "dynamic_braking_filter_active_ticks"],
        "raw_pair_unsafe_ticks": summaries["raw_pair_unsafe_ticks"],
        "broken_contracts": summaries["broken_contracts"],
        "ir_violation_ticks": summaries["ir_violation_ticks"],
        "safe_disagreement_unavailable_ticks": summaries[
            "safe_disagreement_unavailable_ticks"],
        "ttc_active_contracts": summaries["ttc_active_contracts"],
        "ttc_applied_contracts": summaries["ttc_applied_contracts"],
        "total_contracts": summaries["total_contracts"],
        "mean_decoded_share_error": (
            summaries["weighted_share_error_sum"] /
            max(1, summaries["total_contracts"])),
        "min_predicted_gain_m": (
            summaries["min_predicted_gain_m"]
            if math.isfinite(summaries["min_predicted_gain_m"]) else None),
        "min_clearance_m": (summaries["min_clearance"]
                            if math.isfinite(summaries["min_clearance"])
                            else None),
        "mean_solver_ms": summaries["mean_solver_ms"],
        "p95_solver_ms": sorted_solver_times[p95_index] if sorted_solver_times else 0.0,
        "max_solver_ms": max(sorted_solver_times) if sorted_solver_times else 0.0,
        "solver_times_ms": summaries["solver_times_ms"],
        "decision_trace": decision_trace,
        "initial_vehicles": {name: {
            "x_m": car.x, "y_m": car.y, "direction": car.direction,
            "axis": car.axis, "longitudinal_position_m": _longitudinal_position(car),
            "speed_mps": car.speed, "role": car.role,
        } for name, car in initial.items()},
        "edges": [list(edge) for edge in edges], "trace": trace,
    }


def run_sweep(scenes, n_car_counts, trial_seeds, methods=None,
              position_std_m=0.3, speed_std_mps=0.2, policy=None):
    methods = tuple(methods or VALID_METHODS[:2])
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(VALID_METHODS):
        raise ValueError("methods must be a nonempty unique subset of VALID_METHODS")
    policy = normalize_policy(policy)
    episodes = []
    effective_groups = [(scene, n_cars) for scene in scenes for n_cars in n_car_counts]
    for scene, n_cars in effective_groups:
        for seed in trial_seeds:
            for method in methods:
                if gc.isenabled() and gc.get_count()[2] >= max(
                        1, gc.get_threshold()[2] - 6):
                    gc.collect()
                episodes.append(run_episode(
                    scene, n_cars, seed, method, position_std_m,
                    speed_std_mps, policy))
    summary = {}
    for scene in scenes:
        summary[scene] = {}
        for n_cars in [count for s, count in effective_groups if s == scene]:
            summary[scene][str(n_cars)] = {}
            for method in methods:
                group = [row for row in episodes if row["scene"] == scene and
                         row["n_cars"] == n_cars and row["method"] == method]
                completed_delay = [row["mean_completed_delay_s"] for row in group
                                   if row["mean_completed_delay_s"] is not None]
                max_delays = [max(value for value in row["per_vehicle_delay_s"].values()
                                  if value is not None)
                              for row in group if row["completed"]]
                decision_count = sum(len(row["decision_trace"]) for row in group)
                initial_braking_slacks = [
                    row["initial_best_case_braking_slack_m"]
                    for row in group
                    if row.get("initial_best_case_braking_slack_m") is not None]
                allocation_values = [
                    step["selected_allocation_error"]
                    for row in group for step in row["decision_trace"]
                    if step.get("selected_allocation_error") is not None]
                summary[scene][str(n_cars)][method] = {
                    "trial_count": len(group),
                    "initial_full_margin_violation_count": sum(
                        not row["initial_full_margin_satisfied"]
                        for row in group),
                    "initial_negative_braking_slack_count": sum(
                        value < 0.0 for value in initial_braking_slacks),
                    "initial_near_braking_boundary_count": sum(
                        value < 0.25 for value in initial_braking_slacks),
                    "initial_robust_braking_margin_failure_count": sum(
                        row.get("initial_dynamic_braking_margin_satisfied") is False
                        for row in group),
                    "mean_initial_robust_braking_slack_m": (
                        sum(row["initial_robust_braking_slack_m"]
                            for row in group
                            if row.get("initial_robust_braking_slack_m") is not None) /
                        max(1, sum(row.get("initial_robust_braking_slack_m") is not None
                                   for row in group))),
                    "mean_initial_best_case_braking_slack_m": (
                        sum(initial_braking_slacks) /
                        len(initial_braking_slacks)
                        if initial_braking_slacks else None),
                    "collision_count": sum(row["collision"] for row in group),
                    "collision_rate": sum(row["collision"] for row in group) / len(group),
                    "near_collision_count": sum(row["near_collision"] for row in group),
                    "completion_count": sum(row["completed"] for row in group),
                    "completion_rate": sum(row["completed"] for row in group) / len(group),
                    "fallback_count": sum(row["fallback_count"] for row in group),
                    "recovery_profile_step_count": sum(
                        row.get("recovery_profile_step_count", 0)
                        for row in group),
                    "recovery_candidate_count": sum(
                        row.get("recovery_candidate_count", 0)
                        for row in group),
                    "current_margin_deficit_ticks": sum(
                        row.get("current_margin_deficit_ticks", 0)
                        for row in group),
                    "dynamic_braking_candidate_rejected_count": sum(
                        row.get("dynamic_braking_candidate_rejected_count", 0)
                        for row in group),
                    "dynamic_braking_filter_infeasible_ticks": sum(
                        row.get("dynamic_braking_filter_infeasible_ticks", 0)
                        for row in group),
                    "dynamic_braking_filter_active_ticks": sum(
                        row.get("dynamic_braking_filter_active_ticks", 0)
                        for row in group),
                    "next_step_uncertified_fallback_ticks": sum(
                        row["next_step_uncertified_fallback_ticks"]
                        for row in group),
                    "next_step_full_margin_shortfall_ticks": sum(
                        row["next_step_full_margin_shortfall_ticks"]
                        for row in group),
                    "next_step_collision_uncertified_fallback_ticks": sum(
                        row["next_step_collision_uncertified_fallback_ticks"]
                        for row in group),
                    "ir_violation_ticks": sum(row["ir_violation_ticks"] for row in group),
                    "safe_disagreement_unavailable_ticks": sum(
                        row["safe_disagreement_unavailable_ticks"] for row in group),
                    "ttc_active_contracts": sum(row["ttc_active_contracts"] for row in group),
                    "ttc_applied_contracts": sum(row["ttc_applied_contracts"] for row in group),
                    "total_contracts": sum(row["total_contracts"] for row in group),
                    "mean_decoded_share_error": (
                        sum(row["mean_decoded_share_error"] *
                            row["total_contracts"] for row in group) /
                        max(1, sum(row["total_contracts"] for row in group))),
                    "mean_selected_allocation_error": (
                        sum(allocation_values) / len(allocation_values)
                        if allocation_values else None),
                    "mean_broken_contracts_per_decision": (
                        sum(row["broken_contracts"] for row in group) /
                        max(1, decision_count)),
                    "raw_pair_unsafe_ticks": sum(row["raw_pair_unsafe_ticks"] for row in group),
                    "incompatible_edge_target_count": sum(
                        sum(step["proposed_action_pair_conflicts"]
                            for step in row["decision_trace"])
                        for row in group),
                    "mean_completed_delay_s": (
                        sum(completed_delay) / len(completed_delay) if completed_delay else None),
                    "mean_worst_vehicle_delay_s": (
                        sum(max_delays) / len(max_delays) if max_delays else None),
                    "mean_solver_ms": sum(row["mean_solver_ms"] for row in group) / len(group),
                    "p95_solver_ms": sorted(
                        value for row in group for value in row["solver_times_ms"])[
                            max(0, int(math.ceil(0.95 * sum(
                                len(row["solver_times_ms"]) for row in group))) - 1)],
                    "max_solver_ms": max(row["max_solver_ms"] for row in group),
                }
    return episodes, summary

