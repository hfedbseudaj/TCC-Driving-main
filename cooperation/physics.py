"""Standard-library kinematics for the deterministic crossing model.

This module deliberately separates a physically reachable *approach* from
the high-level agreement about conflict-zone occupation.  A vehicle travels
along one straight ray, enters the zone at a requested time with its initial
speed, and then continues through the zone at that speed.  It does not model
vehicle envelopes, other lanes, or non-crossing conflict geometry.

The canonical generator chooses a unique profile for a given entry time:

* constant speed when the requested time is the free-flow time;
* an equal-duration accelerate/brake or brake/accelerate pulse when feasible;
* a bounded peak/valley speed profile otherwise;
* for a long delay, cruise, brake to a stop *upstream* of the entry, wait,
  and accelerate back to the initial speed.

No arbitrary "teleport" or unbounded braking is used to fill a time slot.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple


_EPS = 1e-9
_DIST_TOL = 1e-7


def _finite_number(name, value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be a finite number".format(name))
    if not math.isfinite(number):
        raise ValueError("{} must be a finite number".format(name))
    return number


def _xy(name, value):
    try:
        if len(value) != 2:
            raise ValueError
        x = _finite_number(name + "[0]", value[0])
        y = _finite_number(name + "[1]", value[1])
    except (TypeError, IndexError, ValueError):
        raise ValueError("{} must be a finite (x, y) pair".format(name))
    return (x, y)


@dataclass(frozen=True)
class VehicleSpec:
    """One straight-line approach and its entry-time bounds.

    ``zone_length_m`` is the *effective occupied travel distance*: it must
    include any vehicle-body clearance desired by the caller.  The supplied
    direction is normalized once so that progress and distance use metres.
    A strictly positive initial speed is required because this restricted
    model restores the same speed at the zone entry and uses it to calculate
    the occupation duration.
    """

    id: str
    start_xy: Tuple[float, float]
    direction_xy: Tuple[float, float]
    distance_to_entry_m: float
    speed_mps: float
    zone_length_m: float
    max_accel_mps2: float
    max_brake_mps2: float
    max_speed_mps: float
    min_entry_s: float = 0.0
    max_entry_s: float = math.inf

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("vehicle id must be a nonempty string")
        start = _xy("start_xy", self.start_xy)
        direction = _xy("direction_xy", self.direction_xy)
        norm = math.hypot(*direction)
        if norm <= _EPS:
            raise ValueError("direction_xy must have nonzero length")
        object.__setattr__(self, "start_xy", start)
        object.__setattr__(self, "direction_xy",
                           (direction[0] / norm, direction[1] / norm))

        fields = ("distance_to_entry_m", "speed_mps", "zone_length_m",
                  "max_accel_mps2", "max_brake_mps2", "max_speed_mps",
                  "min_entry_s")
        for name in fields:
            object.__setattr__(self, name, _finite_number(name, getattr(self, name)))
        if self.max_entry_s == math.inf:
            maximum = math.inf
        else:
            maximum = _finite_number("max_entry_s", self.max_entry_s)
        object.__setattr__(self, "max_entry_s", maximum)

        if self.distance_to_entry_m < 0:
            raise ValueError("distance_to_entry_m must be nonnegative")
        for name in ("speed_mps", "zone_length_m", "max_accel_mps2",
                     "max_brake_mps2", "max_speed_mps"):
            if getattr(self, name) <= 0:
                raise ValueError("{} must be positive".format(name))
        if self.max_speed_mps + _EPS < self.speed_mps:
            raise ValueError("max_speed_mps must be at least speed_mps")
        if self.min_entry_s < 0 or self.max_entry_s < self.min_entry_s:
            raise ValueError("entry bounds must satisfy 0 <= min <= max")

    @property
    def entry_position_xy(self):
        return (self.start_xy[0] + self.distance_to_entry_m * self.direction_xy[0],
                self.start_xy[1] + self.distance_to_entry_m * self.direction_xy[1])


@dataclass(frozen=True)
class MotionSegment:
    """A constant-acceleration approach segment, including zero-speed waits."""

    start_s: float
    duration_s: float
    start_progress_m: float
    start_speed_mps: float
    acceleration_mps2: float

    @property
    def end_s(self):
        return self.start_s + self.duration_s

    @property
    def end_speed_mps(self):
        return self.start_speed_mps + self.acceleration_mps2 * self.duration_s

    @property
    def end_progress_m(self):
        duration = self.duration_s
        return (self.start_progress_m + self.start_speed_mps * duration
                + 0.5 * self.acceleration_mps2 * duration * duration)


@dataclass(frozen=True)
class TrajectorySample:
    time_s: float
    position_xy: Tuple[float, float]
    velocity_xy: Tuple[float, float]
    speed_mps: float
    progress_m: float


@dataclass(frozen=True)
class ApproachPlan:
    """A unique entry-time trajectory with a fixed entry/zone speed."""

    vehicle: VehicleSpec
    entry_time_s: float
    segments: Tuple[MotionSegment, ...]
    profile_kind: str

    @property
    def occupation_window_s(self):
        return (self.entry_time_s,
                self.entry_time_s + self.vehicle.zone_length_m / self.vehicle.speed_mps)

    @property
    def upstream_wait_position_xy(self):
        """Stop position, if any; guaranteed before the point-zone entry.

        Global geometric safety, body length, and possible lane blockage are
        intentionally left for the caller's full safety filter.
        """
        for segment in self.segments:
            if segment.start_speed_mps == 0.0 and segment.acceleration_mps2 == 0.0:
                return self._position_at_progress(segment.start_progress_m)
        return None

    def _position_at_progress(self, progress_m):
        direction = self.vehicle.direction_xy
        start = self.vehicle.start_xy
        return (start[0] + direction[0] * progress_m,
                start[1] + direction[1] * progress_m)

    def sample(self, t_s):
        """Position and velocity at any nonnegative time (seconds).

        After entry the vehicle holds its initial speed, including through
        the conflict zone.  This is a kinematic reference, not a controller.
        """
        t_s = _finite_number("t_s", t_s)
        if t_s < 0:
            raise ValueError("sample time must be nonnegative")
        vehicle = self.vehicle
        if t_s >= self.entry_time_s:
            progress = (vehicle.distance_to_entry_m
                        + vehicle.speed_mps * (t_s - self.entry_time_s))
            speed = vehicle.speed_mps
        else:
            segment = next((part for part in self.segments
                            if t_s <= part.end_s + _EPS), None)
            if segment is None:
                raise RuntimeError("approach plan has an uncovered time interval")
            dt = min(max(0.0, t_s - segment.start_s), segment.duration_s)
            speed = segment.start_speed_mps + segment.acceleration_mps2 * dt
            progress = (segment.start_progress_m + segment.start_speed_mps * dt
                        + 0.5 * segment.acceleration_mps2 * dt * dt)
            if speed < 0 and speed > -_DIST_TOL:
                speed = 0.0
        direction = vehicle.direction_xy
        return TrajectorySample(
            t_s, self._position_at_progress(progress),
            (direction[0] * speed, direction[1] * speed), speed, progress)


def _assemble(vehicle, entry_time_s, phases, profile_kind):
    """Integrate phases and reject numerical or physical inconsistencies."""
    segments = []
    elapsed = 0.0
    progress = 0.0
    speed = vehicle.speed_mps
    for duration, accel in phases:
        if duration < -_EPS:
            raise ValueError("negative motion phase duration")
        if duration <= _EPS:
            continue
        if accel > vehicle.max_accel_mps2 + _DIST_TOL:
            raise ValueError("required acceleration exceeds limit")
        if -accel > vehicle.max_brake_mps2 + _DIST_TOL:
            raise ValueError("required braking exceeds limit")
        end_speed = speed + accel * duration
        if min(speed, end_speed) < -_DIST_TOL:
            raise ValueError("motion profile reverses direction")
        if max(speed, end_speed) > vehicle.max_speed_mps + _DIST_TOL:
            raise ValueError("motion profile exceeds speed limit")
        part = MotionSegment(elapsed, duration, progress, speed, accel)
        segments.append(part)
        elapsed = part.end_s
        progress = part.end_progress_m
        speed = max(0.0, end_speed)
        if progress > vehicle.distance_to_entry_m + _DIST_TOL * max(1.0, vehicle.distance_to_entry_m):
            raise ValueError("motion profile enters the conflict zone early")
    if abs(elapsed - entry_time_s) > _DIST_TOL * max(1.0, entry_time_s):
        raise ValueError("motion profile misses the requested entry time")
    if abs(progress - vehicle.distance_to_entry_m) > _DIST_TOL * max(1.0, vehicle.distance_to_entry_m):
        raise ValueError("motion profile misses the zone entry position")
    if abs(speed - vehicle.speed_mps) > _DIST_TOL * max(1.0, vehicle.speed_mps):
        raise ValueError("motion profile does not restore the entry speed")
    return ApproachPlan(vehicle, entry_time_s, tuple(segments), profile_kind)


def canonical_approach_plan(vehicle, entry_time_s):
    """Construct one reachable straight-line plan for a requested entry time.

    An infeasible time raises ``ValueError``.  The symmetric profile uses
    equal-duration opposite accelerations, with the magnitude selected to
    cover exactly the approach distance.  If it violates a dynamic limit, a
    deterministic max-rate peak/valley profile is considered.  A long delay
    uses an upstream stop-and-wait when enough approach distance is available.
    """
    if not isinstance(vehicle, VehicleSpec):
        raise TypeError("vehicle must be VehicleSpec")
    entry_time_s = _finite_number("entry_time_s", entry_time_s)
    if entry_time_s < vehicle.min_entry_s - _EPS or entry_time_s > vehicle.max_entry_s + _EPS:
        raise ValueError("requested entry time is outside vehicle bounds")
    if entry_time_s < 0:
        raise ValueError("requested entry time must be nonnegative")
    distance = vehicle.distance_to_entry_m
    speed = vehicle.speed_mps
    accel = vehicle.max_accel_mps2
    brake = vehicle.max_brake_mps2

    if distance == 0.0:
        if entry_time_s != 0.0:
            raise ValueError("a vehicle already at the entry cannot wait there")
        return ApproachPlan(vehicle, 0.0, tuple(), "at_entry")
    if entry_time_s <= 0:
        raise ValueError("positive approach distance requires positive time")

    free_time = distance / speed
    if abs(entry_time_s - free_time) <= _EPS * max(1.0, free_time):
        return _assemble(vehicle, entry_time_s,
                         ((entry_time_s, 0.0),), "cruise")

    if entry_time_s < free_time:
        required = 4.0 * (distance - speed * entry_time_s) / (entry_time_s ** 2)
        peak = speed + required * entry_time_s / 2.0
        if (required <= min(accel, brake) + _EPS
                and peak <= vehicle.max_speed_mps + _EPS):
            return _assemble(vehicle, entry_time_s,
                             ((entry_time_s / 2.0, required),
                              (entry_time_s / 2.0, -required)), "symmetric_fast")

        # Max-rate accelerate/cruise/brake, with the lowest peak that covers
        # exactly the required distance in the requested time.
        inv_rates = 1.0 / accel + 1.0 / brake
        peak_gain_limit = min(vehicle.max_speed_mps - speed,
                              entry_time_s / inv_rates)
        excess = distance - speed * entry_time_s
        max_excess = (peak_gain_limit * entry_time_s
                      - 0.5 * inv_rates * peak_gain_limit ** 2)
        if excess > max_excess + _DIST_TOL * max(1.0, distance):
            raise ValueError("entry time is too early for acceleration/speed limits")
        discriminant = max(0.0, entry_time_s ** 2 - 2.0 * inv_rates * excess)
        gain = 2.0 * excess / (entry_time_s + math.sqrt(discriminant))
        accelerate_s = gain / accel
        brake_s = gain / brake
        cruise_s = entry_time_s - accelerate_s - brake_s
        return _assemble(vehicle, entry_time_s,
                         ((accelerate_s, accel), (cruise_s, 0.0),
                          (brake_s, -brake)), "fast_trapezoid")

    # Delaying entry: try the unique full-horizon symmetric dip first.
    required = 4.0 * (speed * entry_time_s - distance) / (entry_time_s ** 2)
    valley = speed - required * entry_time_s / 2.0
    if required <= min(accel, brake) + _EPS and valley >= -_EPS:
        return _assemble(vehicle, entry_time_s,
                         ((entry_time_s / 2.0, -required),
                          (entry_time_s / 2.0, required)), "symmetric_slow")

    # A complete upstream stop is possible only when the vehicle has enough
    # distance to brake to zero and later regain its entry speed.  Cruise
    # first, then brake, wait outside the zone, and accelerate to the entry.
    brake_distance = speed ** 2 / (2.0 * brake)
    accel_distance = speed ** 2 / (2.0 * accel)
    stop_resume_distance = brake_distance + accel_distance
    if distance >= stop_resume_distance - _DIST_TOL:
        cruise_distance = max(0.0, distance - stop_resume_distance)
        cruise_s = cruise_distance / speed
        brake_s = speed / brake
        accelerate_s = speed / accel
        travel_s = cruise_s + brake_s + accelerate_s
        if entry_time_s >= travel_s - _EPS:
            wait_s = max(0.0, entry_time_s - travel_s)
            return _assemble(vehicle, entry_time_s,
                             ((cruise_s, 0.0), (brake_s, -brake),
                              (wait_s, 0.0), (accelerate_s, accel)),
                             "upstream_wait")

    # If no full stop is possible, seek the lowest reachable valley speed.
    inv_rates = 1.0 / accel + 1.0 / brake
    deficit = speed * entry_time_s - distance
    drop_limit = min(speed, entry_time_s / inv_rates)
    max_deficit = drop_limit * entry_time_s - 0.5 * inv_rates * drop_limit ** 2
    if deficit > max_deficit + _DIST_TOL * max(1.0, distance):
        raise ValueError("entry time is too late without reversing or unsafe waiting")
    discriminant = max(0.0, entry_time_s ** 2 - 2.0 * inv_rates * deficit)
    drop = 2.0 * deficit / (entry_time_s + math.sqrt(discriminant))
    brake_s = drop / brake
    accelerate_s = drop / accel
    valley_s = entry_time_s - brake_s - accelerate_s
    return _assemble(vehicle, entry_time_s,
                     ((brake_s, -brake), (valley_s, 0.0),
                      (accelerate_s, accel)), "slow_trapezoid")


def pair_safe(plan_i, plan_j, clearance_s):
    """Temporal separation of two vehicles occupying the *same* zone.

    This does not establish full vehicle-body or route safety.  Such checks
    belong to the global hard-safety filter of the cooperation method.
    """
    if not isinstance(plan_i, ApproachPlan) or not isinstance(plan_j, ApproachPlan):
        raise TypeError("pair_safe requires two ApproachPlan instances")
    if plan_i.vehicle.id == plan_j.vehicle.id:
        raise ValueError("pair_safe requires two distinct vehicles")
    clearance_s = _finite_number("clearance_s", clearance_s)
    if clearance_s < 0:
        raise ValueError("clearance_s must be nonnegative")
    start_i, end_i = plan_i.occupation_window_s
    start_j, end_j = plan_j.occupation_window_s
    return (end_i + clearance_s <= start_j + _EPS
            or end_j + clearance_s <= start_i + _EPS)


def legacy_closing_ttc(position_i, velocity_i, position_j, velocity_j,
                       threshold_mps=0.1):
    """Mirror the old code's distance/line-of-sight closing-speed TTC.

    As in the archived tensor implementation, a non-closing pair (including
    coincident positions with undefined line-of-sight) returns infinity.
    This TTC alone is not a geometric collision or occupancy test.
    """
    position_i = _xy("position_i", position_i)
    velocity_i = _xy("velocity_i", velocity_i)
    position_j = _xy("position_j", position_j)
    velocity_j = _xy("velocity_j", velocity_j)
    threshold_mps = _finite_number("threshold_mps", threshold_mps)
    if threshold_mps < 0:
        raise ValueError("threshold_mps must be nonnegative")
    dx = position_i[0] - position_j[0]
    dy = position_i[1] - position_j[1]
    distance = math.hypot(dx, dy)
    if distance == 0.0:
        return math.inf
    dvx = velocity_i[0] - velocity_j[0]
    dvy = velocity_i[1] - velocity_j[1]
    closing_speed = -(dvx * dx + dvy * dy) / distance
    if closing_speed <= threshold_mps:
        return math.inf
    return distance / closing_speed


def ttc_discount(alpha_per_s, ttc_s, min_delta=0.05, max_delta=0.99):
    """Old code's clipped delta = 1 - exp(-alpha * TTC).

    An infinite TTC is replaced by 1e6 seconds before the exponential, as
    in the archived implementation.  The clip keeps pairwise SPE finite.
    """
    alpha_per_s = _finite_number("alpha_per_s", alpha_per_s)
    min_delta = _finite_number("min_delta", min_delta)
    max_delta = _finite_number("max_delta", max_delta)
    if alpha_per_s < 0:
        raise ValueError("alpha_per_s must be nonnegative")
    if not 0.0 <= min_delta <= max_delta < 1.0:
        raise ValueError("discount bounds must satisfy 0 <= min <= max < 1")
    try:
        ttc_s = float(ttc_s)
    except (TypeError, ValueError):
        raise ValueError("ttc_s must be nonnegative or infinity")
    if math.isnan(ttc_s) or ttc_s < 0:
        raise ValueError("ttc_s must be nonnegative or infinity")
    effective_ttc = 1e6 if math.isinf(ttc_s) else ttc_s
    delta = 1.0 - math.exp(-alpha_per_s * effective_ttc)
    return min(max(delta, min_delta), max_delta)


def rubinstein_shares(delta_first, delta_second):
    """Infinite-horizon unit-surplus SPE; the first car proposes first.

    This theoretical pair share is not the archived learned four-dimensional
    offer and does not specify a physical trajectory on its own.
    """
    delta_first = _finite_number("delta_first", delta_first)
    delta_second = _finite_number("delta_second", delta_second)
    if not 0.0 <= delta_first < 1.0 or not 0.0 <= delta_second < 1.0:
        raise ValueError("Rubinstein discounts must lie in [0, 1)")
    denominator = 1.0 - delta_first * delta_second
    first = (1.0 - delta_second) / denominator
    second = delta_second * (1.0 - delta_first) / denominator
    return (first, second)
