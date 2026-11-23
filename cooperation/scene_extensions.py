"""Provisional scene-specific D2/K_e definitions for the other three scenes.

These deterministic finite-control models are newly defined for E5; they are
not part of the archived paper's learned four-dimensional offer controller.
Following uses bumper headway; merging uses a completed lateral lane change
and swept center clearance; head-on uses swept center clearance with lateral
avoidance.  The three scene distances and welfare values are reported only
within their own scene, never summed across scenes.
"""

from dataclasses import dataclass
import itertools
import math
import random
import time
from typing import Dict, Mapping, Sequence, Tuple

from .physics import legacy_closing_ttc, rubinstein_shares, ttc_discount


EXTENSION_FIXED_MODEL = {
    "horizon_s": 6.0,
    "dt_s": 0.1,
    "following_min_bumper_gap_m": 5.0,
    "merge_and_head_on_min_swept_center_distance_m": 2.2,
    "lateral_shift_duration_s": 3.0,
}


@dataclass(frozen=True)
class SceneCar:
    id: str
    start_xy: Tuple[float, float]
    direction_xy: Tuple[float, float]
    speed_mps: float
    role: str
    max_speed_mps: float = 15.0


@dataclass(frozen=True)
class SceneControl:
    name: str
    accel_mps2: float
    lateral_m: float = 0.0


@dataclass(frozen=True)
class SceneContract:
    scene: str
    first_id: str
    second_id: str
    first_control: str
    second_control: str
    ttc_s: float
    delta_first: float
    delta_second: float
    rub_share_first: float
    surplus_m: float
    reference_share: float
    share_error: float

    @property
    def edge_id(self):
        return self.first_id + "--" + self.second_id


@dataclass(frozen=True)
class SceneResult:
    scene: str
    n_cars: int
    trial_seed: int
    vehicles: Tuple[str, ...]
    edges: Tuple[Tuple[str, str], ...]
    contracts: Tuple[SceneContract, ...]
    safe_plan_count: int
    exactly_composable_count: int
    global_controls: Tuple[Tuple[str, str], ...]
    adapted_pairwise_controls: Tuple[Tuple[str, str], ...]
    global_broken_contracts: int
    adapted_pairwise_broken_contracts: int
    global_welfare_m: float
    adapted_pairwise_welfare_m: float
    global_edge_distance: float
    adapted_pairwise_edge_distance: float
    welfare_optimum_m: float
    welfare_floor_m: float
    enumerated_plan_count: int
    solve_ms: float
    status: str = "provisional_finite_kinematic_scene_definition"


def _control(name, accel, lateral=0.0):
    return SceneControl(name, accel, lateral)


def _scene_data(scene, n_cars=3, trial_seed=0):
    """Two-, three- and four-CAV extensions of the archived scene families."""
    if n_cars not in (2, 3, 4):
        raise ValueError("Extension scale must contain 2, 3 or 4 CAVs")
    east = (1.0, 0.0)
    west = (-1.0, 0.0)
    if scene == "chase":
        cars = {
            "R": SceneCar("R", (-30.0, 0.0), east, 10.0, "rear"),
            "M": SceneCar("M", (-20.0, 0.0), east, 7.0, "middle"),
            "F": SceneCar("F", (-10.0, 0.0), east, 6.0, "front"),
        }
        controls = {
            "R": (_control("brake4", -4.0), _control("brake3", -3.0),
                  _control("brake2", -2.0), _control("coast", 0.0)),
            "M": (_control("brake4", -4.0), _control("brake3", -3.0),
                  _control("brake2", -2.0), _control("coast", 0.0),
                  _control("accelerate", 1.0)),
            "F": (_control("coast", 0.0), _control("accelerate", 1.0)),
        }
        edges = (("R", "M"), ("M", "F"))
        if n_cars == 2:
            cars.pop("F")
            controls.pop("F")
            edges = (("R", "M"),)
        elif n_cars == 4:
            cars["R2"] = SceneCar("R2", (-40.0, 0.0), east, 11.0, "rear")
            controls["R2"] = controls["R"]
            edges = (("R2", "R"),) + edges
    elif scene == "merge":
        cars = {
            "M": SceneCar("M", (-25.0, -3.0), east, 8.0, "merging"),
            "B1": SceneCar("B1", (-30.0, 0.0), east, 7.0, "mainline"),
            "B2": SceneCar("B2", (-10.0, 0.0), east, 6.0, "mainline"),
        }
        controls = {
            "M": (_control("yield_merge", -3.0, 3.0),
                  _control("slow_merge", -1.0, 3.0),
                  _control("maintain_merge", 0.0, 3.0)),
            "B1": (_control("brake2", -2.0), _control("coast", 0.0),
                   _control("accelerate", 1.0)),
            "B2": (_control("brake1", -1.0), _control("coast", 0.0),
                   _control("accelerate", 1.0)),
        }
        edges = (("M", "B1"), ("M", "B2"), ("B1", "B2"))
        if n_cars == 2:
            cars.pop("B2")
            controls.pop("B2")
            edges = (("M", "B1"),)
        elif n_cars == 4:
            cars["B3"] = SceneCar("B3", (5.0, 0.0), east, 6.0, "mainline")
            controls["B3"] = controls["B2"]
            edges = edges + (("M", "B3"), ("B2", "B3"))
    elif scene == "head_on":
        cars = {
            "E1": SceneCar("E1", (-25.0, -1.0), east, 8.0, "eastbound"),
            "W": SceneCar("W", (25.0, 0.0), west, 8.0, "westbound"),
            "E2": SceneCar("E2", (-35.0, 1.0), east, 8.0, "eastbound"),
        }
        options = tuple(_control("{}_{}".format(a, l), a, l)
                        for a in (-3.0, 0.0) for l in (-3.0, 0.0, 3.0))
        controls = {name: options for name in cars}
        edges = (("E1", "W"), ("W", "E2"))
        if n_cars == 2:
            cars.pop("E2")
            controls.pop("E2")
            edges = (("E1", "W"),)
        elif n_cars == 4:
            cars["E3"] = SceneCar("E3", (-45.0, -3.0), east, 8.0, "eastbound")
            controls["E3"] = options
            edges = edges + (("W", "E3"),)
    else:
        raise ValueError("Only chase, merge and head_on are defined here")
    if trial_seed:
        rng = random.Random(int(trial_seed))
        shifted = {}
        for name, car in cars.items():
            longitudinal = rng.uniform(-0.3, 0.3)
            speed = max(0.3, min(car.max_speed_mps,
                                 car.speed_mps + rng.uniform(-0.2, 0.2)))
            shifted[name] = SceneCar(
                name,
                (car.start_xy[0] + car.direction_xy[0] * longitudinal,
                 car.start_xy[1] + car.direction_xy[1] * longitudinal),
                car.direction_xy, speed, car.role, car.max_speed_mps)
        cars = shifted
    return cars, controls, edges


def _rollout(car, control, horizon_s=EXTENSION_FIXED_MODEL["horizon_s"],
             dt_s=EXTENSION_FIXED_MODEL["dt_s"]):
    """Straight progress plus quintic lateral transition (new model)."""
    points = [car.start_xy]
    speed = car.speed_mps
    progress = 0.0
    count = int(round(horizon_s / dt_s))
    direction = car.direction_xy
    normal = (-direction[1], direction[0])
    for index in range(1, count + 1):
        next_speed = min(car.max_speed_mps, max(0.0, speed + control.accel_mps2 * dt_s))
        progress += 0.5 * (speed + next_speed) * dt_s
        speed = next_speed
        phase = min(1.0, (index * dt_s) /
                    EXTENSION_FIXED_MODEL["lateral_shift_duration_s"])
        smooth = 10.0 * phase ** 3 - 15.0 * phase ** 4 + 6.0 * phase ** 5
        lateral = control.lateral_m * smooth
        points.append((car.start_xy[0] + direction[0] * progress + normal[0] * lateral,
                       car.start_xy[1] + direction[1] * progress + normal[1] * lateral))
    return tuple(points), progress, speed


def _segment_min_distance(a0, a1, b0, b1):
    rx, ry = a0[0] - b0[0], a0[1] - b0[1]
    dx = (a1[0] - a0[0]) - (b1[0] - b0[0])
    dy = (a1[1] - a0[1]) - (b1[1] - b0[1])
    norm_sq = dx * dx + dy * dy
    fraction = max(0.0, min(1.0, -(rx * dx + ry * dy) / norm_sq)) if norm_sq > 1e-12 else 0.0
    return math.hypot(rx + fraction * dx, ry + fraction * dy)


def _safe(scene, first, first_plan, second, second_plan):
    a_points, _, _ = first_plan
    b_points, _, _ = second_plan
    if scene == "chase":
        rear, front = ((first, second) if first.start_xy[0] < second.start_xy[0]
                       else (second, first))
        rear_points = a_points if rear.id == first.id else b_points
        front_points = b_points if front.id == second.id else a_points
        return all(front_point[0] - rear_point[0] >=
                   EXTENSION_FIXED_MODEL["following_min_bumper_gap_m"] - 1e-9
                   for rear_point, front_point in zip(rear_points, front_points))
    minimum = min(_segment_min_distance(a_points[index - 1], a_points[index],
                                        b_points[index - 1], b_points[index])
                  for index in range(1, len(a_points)))
    return minimum >= EXTENSION_FIXED_MODEL["merge_and_head_on_min_swept_center_distance_m"] - 1e-9


def _utility(scene, car, control, plan):
    _, progress, _ = plan
    effort_m = 0.08 * control.accel_mps2 ** 2 * 6.0
    lateral_m = 0.04 * abs(control.lateral_m)
    completion_bonus_m = 2.0 if scene == "merge" and car.role == "merging" and control.lateral_m >= 3.0 else 0.0
    return progress - effort_m - lateral_m + completion_bonus_m


def _disagreement(scene, first, second):
    if scene == "chase" or (scene == "merge" and first.role == second.role):
        rear_first = first.start_xy[0] < second.start_xy[0]
        return ((_control("safe_rear_brake", -4.0), _control("front_coast", 0.0))
                if rear_first else
                (_control("front_coast", 0.0), _control("safe_rear_brake", -4.0)))
    if scene == "merge":
        merging_first = first.role == "merging"
        return ((_control("abort_merge_and_brake", -3.0, 0.0), _control("mainline_coast", 0.0))
                if merging_first else
                (_control("mainline_coast", 0.0), _control("abort_merge_and_brake", -3.0, 0.0)))
    return (_control("stop_short", -3.0), _control("stop_short", -3.0))


def _pair_decode(scene, first, second, first_controls, second_controls,
                 plans, alpha_per_s, eta_m):
    p0_first, p0_second = _disagreement(scene, first, second)
    p0a = _rollout(first, p0_first)
    p0b = _rollout(second, p0_second)
    if not _safe(scene, first, p0a, second, p0b):
        raise ValueError("Constructed disagreement is unsafe for {}--{}".format(first.id, second.id))
    u0_first = _utility(scene, first, p0_first, p0a)
    u0_second = _utility(scene, second, p0_second, p0b)
    candidates = []
    for control_a, control_b in itertools.product(first_controls, second_controls):
        plan_a = plans[first.id][control_a.name]
        plan_b = plans[second.id][control_b.name]
        if not _safe(scene, first, plan_a, second, plan_b):
            continue
        gain_a = _utility(scene, first, control_a, plan_a) - u0_first
        gain_b = _utility(scene, second, control_b, plan_b) - u0_second
        surplus = gain_a + gain_b
        if gain_a >= -1e-9 and gain_b >= -1e-9 and surplus > 1e-9:
            candidates.append((control_a, control_b, gain_a, gain_b, surplus))
    if not candidates:
        raise ValueError("No safe, individually rational {} pair plan for {}--{}".format(
            scene, first.id, second.id))
    maximum = max(row[4] for row in candidates)
    velocities = (tuple(x * first.speed_mps for x in first.direction_xy),
                  tuple(x * second.speed_mps for x in second.direction_xy))
    ttc = legacy_closing_ttc(first.start_xy, velocities[0],
                             second.start_xy, velocities[1])
    delta_a = ttc_discount(alpha_per_s, ttc)
    delta_b = ttc_discount(alpha_per_s, ttc)
    share, _ = rubinstein_shares(delta_a, delta_b)
    eligible = [row for row in candidates if row[4] >= maximum - eta_m - 1e-9]
    selected = min(eligible, key=lambda row: (
        abs(row[2] / row[4] - share), -row[4],
        row[0].accel_mps2 ** 2 + row[1].accel_mps2 ** 2,
        row[0].name, row[1].name))
    return SceneContract(
        scene, first.id, second.id, selected[0].name, selected[1].name,
        ttc, delta_a, delta_b, share, selected[4],
        selected[2] / selected[4], abs(selected[2] / selected[4] - share))


def _distance(scene, selected_first, reference_first,
              selected_second, reference_second):
    accel_scale = 4.0 if scene == "chase" else 3.0
    result = (abs(selected_first.accel_mps2 - reference_first.accel_mps2) +
              abs(selected_second.accel_mps2 - reference_second.accel_mps2)) / accel_scale
    if scene == "merge":
        result += (abs(selected_first.lateral_m - reference_first.lateral_m) +
                   abs(selected_second.lateral_m - reference_second.lateral_m)) / 3.0
    elif scene == "head_on":
        result += (abs(selected_first.lateral_m - reference_first.lateral_m) +
                   abs(selected_second.lateral_m - reference_second.lateral_m)) / 3.0
    return result


def evaluate_scene(scene, alpha_per_s=0.2, eta_m=4.0, welfare_epsilon_m=5.0,
                   n_cars=3, trial_seed=0):
    """Exact finite-family pair decoding and three-CAV cooperative selection."""
    started = time.perf_counter()
    cars, action_sets, edges = _scene_data(scene, n_cars, trial_seed)
    actions = {name: {action.name: action for action in candidates}
               for name, candidates in action_sets.items()}
    plans = {name: {action.name: _rollout(cars[name], action)
                    for action in candidates}
             for name, candidates in action_sets.items()}
    contracts = tuple(_pair_decode(
        scene, cars[first], cars[second], action_sets[first], action_sets[second],
        plans, alpha_per_s, eta_m) for first, second in edges)
    names = tuple(sorted(cars))
    safe_rows = []
    for chosen_names in itertools.product(*(actions[name] for name in names)):
        choice = dict(zip(names, chosen_names))
        if not all(_safe(scene, cars[first], plans[first][choice[first]],
                         cars[second], plans[second][choice[second]])
                   for first, second in edges):
            continue
        welfare = sum(_utility(scene, cars[name], actions[name][choice[name]],
                               plans[name][choice[name]]) for name in names)
        broken = sum(choice[edge.first_id] != edge.first_control or
                     choice[edge.second_id] != edge.second_control
                     for edge in contracts)
        fidelity = sum(_distance(
            scene, actions[edge.first_id][choice[edge.first_id]],
            actions[edge.first_id][edge.first_control],
            actions[edge.second_id][choice[edge.second_id]],
            actions[edge.second_id][edge.second_control]) for edge in contracts)
        safe_rows.append((choice, welfare, broken, fidelity))
    if not safe_rows:
        raise ValueError("No safe joint {} plan".format(scene))
    welfare_star = max(row[1] for row in safe_rows)
    floor = welfare_star - welfare_epsilon_m
    eligible = [row for row in safe_rows if row[1] >= floor - 1e-9]
    global_row = min(eligible, key=lambda row: (
        row[2], row[3], -row[1], tuple(sorted(row[0].items()))))
    # Reconstruct the old 1/TTC edge-reference aggregation in this action
    # space, then project onto the identical safe and welfare-eligible family.
    targets = {}
    for name in names:
        incident = [edge for edge in contracts if name in (edge.first_id, edge.second_id)]
        weighted = []
        for edge in incident:
            reference = (edge.first_control if name == edge.first_id else edge.second_control)
            weight = 1.0 / max(edge.ttc_s, 1e-3) if math.isfinite(edge.ttc_s) else 0.0
            weighted.append((actions[name][reference], weight))
        if not weighted or sum(weight for _, weight in weighted) <= 0:
            targets[name] = (0.0, 0.0)
        else:
            denominator = sum(weight for _, weight in weighted)
            targets[name] = (
                sum(action.accel_mps2 * weight for action, weight in weighted) / denominator,
                sum(action.lateral_m * weight for action, weight in weighted) / denominator,
            )
    adapted_row = min(eligible, key=lambda row: (
        sum((actions[name][row[0][name]].accel_mps2 - targets[name][0]) ** 2 +
            (actions[name][row[0][name]].lateral_m - targets[name][1]) ** 2
            for name in names), -row[1], tuple(sorted(row[0].items()))))
    enumerated = 1
    for name in names:
        enumerated *= len(actions[name])
    return SceneResult(
        scene, n_cars, int(trial_seed), names, edges, contracts, len(safe_rows),
        sum(row[2] == 0 for row in safe_rows),
        tuple(sorted(global_row[0].items())),
        tuple(sorted(adapted_row[0].items())),
        global_row[2], adapted_row[2], global_row[1], adapted_row[1],
        global_row[3], adapted_row[3], welfare_star, floor,
        enumerated, (time.perf_counter() - started) * 1000.0)
