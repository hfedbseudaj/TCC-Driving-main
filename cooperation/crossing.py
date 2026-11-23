"""Two-vehicle SPE-share to physical crossing-contract decoder.

Only the deterministic crossing scene is defined here. Other scenes need their
own trajectory generator, utility, and distance before they can use this API.
All optimization is exact only over the caller's finite candidate set.
"""

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .physics import (
    VehicleSpec,
    canonical_approach_plan,
    legacy_closing_ttc,
    pair_safe,
    rubinstein_shares,
    ttc_discount,
)


_EPS = 1e-9


@dataclass(frozen=True)
class PairProposal:
    """One fully specified pair of canonical entry-time trajectories."""

    proposal_id: str
    first_entry_s: float
    second_entry_s: float


@dataclass(frozen=True)
class CandidateScore:
    proposal_id: str
    first_entry_s: float
    second_entry_s: float
    first_gain_s: float
    second_gain_s: float
    surplus_s: float
    first_share: float
    share_error: float
    control_effort: float


@dataclass(frozen=True)
class CrossingContract:
    """K_e: type, protocol parameters, and a canonical trajectory-pair set.

    ``contains`` defines C_e over H_i(t_i), H_j(t_j): both entry times must
    lie in the stated windows, preserve the selected ordering, and satisfy
    the same hard pair safety rule. Window endpoints alone are never taken as
    proof of safety.
    """

    first_id: str
    second_id: str
    first_reference_s: float
    second_reference_s: float
    first_tolerance_s: float
    second_tolerance_s: float
    first_before_second: bool
    clearance_s: float
    source: str
    reference_id: str
    ttc_s: Optional[float] = None
    delta_first: Optional[float] = None
    delta_second: Optional[float] = None
    rub_share_first: Optional[float] = None
    rub_share_second: Optional[float] = None
    disagreement_first_s: Optional[float] = None
    disagreement_second_s: Optional[float] = None

    def __post_init__(self):
        if not self.first_id or not self.second_id or self.first_id == self.second_id:
            raise ValueError("A crossing contract needs two distinct vehicle IDs")
        for name in ("first_reference_s", "second_reference_s", "first_tolerance_s",
                     "second_tolerance_s", "clearance_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError("{} must be finite and nonnegative".format(name))

    @property
    def edge_id(self):
        return "{}--{}".format(self.first_id, self.second_id)

    def reference_for(self, vehicle_id):
        if vehicle_id == self.first_id:
            return self.first_reference_s
        if vehicle_id == self.second_id:
            return self.second_reference_s
        raise KeyError(vehicle_id)

    def contains(self, entry_times_s, vehicles):
        """Whether a global plan satisfies this edge's physical contract."""
        if self.first_id not in entry_times_s or self.second_id not in entry_times_s:
            return False
        t_first = float(entry_times_s[self.first_id])
        t_second = float(entry_times_s[self.second_id])
        if abs(t_first - self.first_reference_s) > self.first_tolerance_s + _EPS:
            return False
        if abs(t_second - self.second_reference_s) > self.second_tolerance_s + _EPS:
            return False
        try:
            p_first = canonical_approach_plan(vehicles[self.first_id], t_first)
            p_second = canonical_approach_plan(vehicles[self.second_id], t_second)
        except ValueError:
            return False
        if not pair_safe(p_first, p_second, self.clearance_s):
            return False
        f_end = p_first.occupation_window_s[1]
        s_end = p_second.occupation_window_s[1]
        if self.first_before_second:
            return f_end + self.clearance_s <= t_second + _EPS
        return s_end + self.clearance_s <= t_first + _EPS


@dataclass(frozen=True)
class PairDecodeResult:
    contract: CrossingContract
    candidate_scores: Tuple[CandidateScore, ...]
    admissible_ids: Tuple[str, ...]
    maximal_surplus_s: float
    eta_s: float


def _finite_nonnegative(name, value):
    if not math.isfinite(value) or value < 0:
        raise ValueError("{} must be finite and nonnegative".format(name))


def _control_effort(plan):
    """Integral of squared longitudinal acceleration for the approach.

    This is a deterministic third-level tie-break, not an extra welfare term.
    ``ApproachPlan.segments`` contain (duration_s, acceleration_mps2).
    """
    return sum(float(segment.duration_s) * float(segment.acceleration_mps2) ** 2
               for segment in plan.segments)


def decode_crossing_pair(
    first: VehicleSpec,
    second: VehicleSpec,
    disagreement_entry_s: Tuple[float, float],
    proposals: Sequence[PairProposal],
    alpha_first_per_s: float,
    alpha_second_per_s: float,
    eta_s: float,
    clearance_s: float,
    tolerance_s: Tuple[float, float] = (0.0, 0.0),
    ttc_s: Optional[float] = None,
    delta_bounds: Tuple[float, float] = (0.05, 0.99),
) -> PairDecodeResult:
    """Decode the original pairwise SPE share into a unique physical plan.

    The crossing utility is negative entry delay, so gains and surplus have
    seconds as units. ``Smax`` is computed over safe, individually rational
    proposals. The selection key is (share error, -surplus, control effort,
    stable proposal ID, entry times). The last items guarantee a unique result.
    """
    if first.id == second.id:
        raise ValueError("Pair vehicle IDs must differ")
    for name, value in (("eta_s", eta_s), ("clearance_s", clearance_s),
                        ("first_tolerance_s", tolerance_s[0]),
                        ("second_tolerance_s", tolerance_s[1])):
        _finite_nonnegative(name, float(value))
    if not proposals:
        raise ValueError("At least one named physical pair proposal is required")
    names = [p.proposal_id for p in proposals]
    if len(names) != len(set(names)) or any(not name for name in names):
        raise ValueError("Pair proposal IDs must be nonempty and unique")

    disagreement_first = canonical_approach_plan(first, disagreement_entry_s[0])
    disagreement_second = canonical_approach_plan(second, disagreement_entry_s[1])
    if not pair_safe(disagreement_first, disagreement_second, clearance_s):
        raise ValueError("The disagreement plan must itself be safe")

    if ttc_s is None:
        first_velocity = (first.direction_xy[0] * first.speed_mps,
                          first.direction_xy[1] * first.speed_mps)
        second_velocity = (second.direction_xy[0] * second.speed_mps,
                           second.direction_xy[1] * second.speed_mps)
        ttc_s = legacy_closing_ttc(first.start_xy, first_velocity,
                                   second.start_xy, second_velocity)
    if math.isnan(ttc_s) or ttc_s < 0:
        raise ValueError("Pre-decision TTC must be nonnegative")
    delta_first = ttc_discount(alpha_first_per_s, ttc_s, *delta_bounds)
    delta_second = ttc_discount(alpha_second_per_s, ttc_s, *delta_bounds)
    share_first, share_second = rubinstein_shares(delta_first, delta_second)

    rational = []
    for proposal in proposals:
        try:
            p_first = canonical_approach_plan(first, proposal.first_entry_s)
            p_second = canonical_approach_plan(second, proposal.second_entry_s)
        except ValueError:
            continue
        if not pair_safe(p_first, p_second, clearance_s):
            continue
        gain_first = float(disagreement_entry_s[0]) - proposal.first_entry_s
        gain_second = float(disagreement_entry_s[1]) - proposal.second_entry_s
        surplus = gain_first + gain_second
        if gain_first < -_EPS or gain_second < -_EPS or surplus <= _EPS:
            continue
        rational.append((proposal, p_first, p_second, gain_first, gain_second, surplus))
    if not rational:
        raise ValueError("No safe, individually rational pair proposal")
    smax = max(row[5] for row in rational)
    scores = []
    admissible_ids = []
    for proposal, p_first, p_second, gain_first, gain_second, surplus in rational:
        first_share = gain_first / surplus
        score = CandidateScore(
            proposal.proposal_id, proposal.first_entry_s, proposal.second_entry_s,
            gain_first, gain_second, surplus, first_share,
            abs(first_share - share_first), _control_effort(p_first) + _control_effort(p_second))
        scores.append(score)
        if surplus >= smax - eta_s - _EPS:
            admissible_ids.append(proposal.proposal_id)
    eligible = [row for row in scores if row.proposal_id in set(admissible_ids)]
    selected = min(eligible, key=lambda row: (
        row.share_error, -row.surplus_s, row.control_effort,
        row.proposal_id, row.first_entry_s, row.second_entry_s))
    selected_first = canonical_approach_plan(first, selected.first_entry_s)
    selected_second = canonical_approach_plan(second, selected.second_entry_s)
    first_before_second = (
        selected_first.occupation_window_s[1] + clearance_s
        <= selected.second_entry_s + _EPS)
    contract = CrossingContract(
        first.id, second.id, selected.first_entry_s, selected.second_entry_s,
        float(tolerance_s[0]), float(tolerance_s[1]),
        first_before_second, clearance_s, "spe_decoder", selected.proposal_id,
        float(ttc_s), delta_first, delta_second, share_first, share_second,
        float(disagreement_entry_s[0]), float(disagreement_entry_s[1]))
    return PairDecodeResult(contract, tuple(scores), tuple(admissible_ids), smax, eta_s)
