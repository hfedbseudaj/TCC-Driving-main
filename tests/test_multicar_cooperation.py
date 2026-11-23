"""Behavioral checks for the opt-in deterministic crossing cooperation path.

These examples test the finite candidate model, not the archived learned-offer
controller or a continuous-time optimum. The sequential comparator commits
pair contracts in TTC order, overwrites shared targets, and projects onto the
same joint feasible family as the global selector.
"""

import math
import unittest
from dataclasses import replace

from cooperation.compose import compose_crossing_snapshot
from cooperation.crossing import CrossingContract, PairProposal, decode_crossing_pair
from cooperation.physics import (
    VehicleSpec,
    canonical_approach_plan,
    legacy_closing_ttc,
    pair_safe,
    rubinstein_shares,
    ttc_discount,
)


def crossing_vehicles():
    """Two perpendicular approaches with a 3 s free-flow entry time."""
    first = VehicleSpec("A", (-15.0, 0.0), (1.0, 0.0), 15.0, 5.0,
                        13.0, 3.0, 3.0, 12.0)
    second = VehicleSpec("B", (0.0, -15.0), (0.0, 1.0), 15.0, 5.0,
                         10.0, 3.0, 3.0, 12.0)
    return first, second


def crossing_contracts():
    """Two individually safe but mutually incompatible references for B."""
    a, b = crossing_vehicles()
    c = VehicleSpec("C", (-15.0, 2.0), (1.0, 0.0), 15.0, 5.0,
                    10.0, 3.0, 3.0, 12.0)
    ab = CrossingContract("A", "B", 3.0, 5.8, 0.0, 0.1, True,
                          0.2, "constructed_test", "AB", ttc_s=3.0)
    bc = CrossingContract("B", "C", 7.0, 3.0, 0.1, 0.0, False,
                          0.2, "constructed_test", "BC", ttc_s=1.0)
    return {v.id: v for v in (a, b, c)}, (ab, bc)


class TestOriginalPairFormula(unittest.TestCase):
    def test_legacy_closing_ttc_discount_and_spe_numbers(self):
        a, b = crossing_vehicles()
        ttc = legacy_closing_ttc(a.start_xy, (5.0, 0.0),
                                 b.start_xy, (0.0, 5.0))
        self.assertAlmostEqual(ttc, 3.0, places=10)
        delta = ttc_discount(0.2, ttc)
        self.assertAlmostEqual(delta, 1.0 - math.exp(-0.6), places=10)
        first_share, second_share = rubinstein_shares(delta, delta)
        self.assertAlmostEqual(first_share, 1.0 / (1.0 + delta), places=10)
        self.assertAlmostEqual(first_share + second_share, 1.0, places=10)
        self.assertGreater(first_share, second_share)

    def test_nonclosing_ttc_and_discount_are_finite_by_clipping(self):
        ttc = legacy_closing_ttc((0.0, 0.0), (0.0, 0.0),
                                 (10.0, 0.0), (1.0, 0.0))
        self.assertTrue(math.isinf(ttc))
        self.assertEqual(ttc_discount(0.2, ttc), 0.99)
        self.assertEqual(ttc_discount(0.2, 0.0), 0.05)


class TestPhysicalPairDecoder(unittest.TestCase):
    def test_canonical_delay_stays_upstream_and_respects_dynamics(self):
        a, _ = crossing_vehicles()
        plan = canonical_approach_plan(a, 10.2)
        self.assertEqual(plan.profile_kind, "upstream_wait")
        self.assertIsNotNone(plan.upstream_wait_position_xy)
        self.assertLess(plan.upstream_wait_position_xy[0],
                        a.entry_position_xy[0])
        self.assertAlmostEqual(plan.sample(10.2).progress_m, 15.0)
        self.assertAlmostEqual(plan.sample(10.2).speed_mps, 5.0)
        for segment in plan.segments:
            self.assertLessEqual(segment.acceleration_mps2,
                                 a.max_accel_mps2 + 1e-9)
            self.assertGreaterEqual(segment.acceleration_mps2,
                                    -a.max_brake_mps2 - 1e-9)
        with self.assertRaises(ValueError):
            canonical_approach_plan(a, 1.0)

    def test_occupation_clearance_boundary_is_inclusive(self):
        a, b = crossing_vehicles()
        first = canonical_approach_plan(a, 3.0)
        self.assertEqual(first.occupation_window_s, (3.0, 5.6))
        self.assertTrue(pair_safe(first, canonical_approach_plan(b, 5.8), 0.2))
        self.assertFalse(pair_safe(first, canonical_approach_plan(b, 5.79), 0.2))

    def test_eta_and_ttc_coefficient_change_selected_physical_plan(self):
        a, b = crossing_vehicles()
        proposals = (PairProposal("p1", 3.0, 5.8),
                     PairProposal("p2", 5.2, 3.0),
                     PairProposal("unsafe", 3.0, 5.7),
                     PairProposal("irrational", 3.0, 11.0))

        def decode(alpha, eta):
            return decode_crossing_pair(a, b, (10.2, 8.0), proposals,
                                        alpha, alpha, eta, 0.2, ttc_s=3.0)

        low_alpha = decode(0.2, 1.0)
        self.assertEqual(low_alpha.contract.reference_id, "p1")
        self.assertEqual(set(low_alpha.admissible_ids), {"p1", "p2"})
        self.assertEqual({x.proposal_id for x in low_alpha.candidate_scores},
                         {"p1", "p2"})
        self.assertAlmostEqual(low_alpha.maximal_surplus_s, 10.0)
        scores = {x.proposal_id: x for x in low_alpha.candidate_scores}
        self.assertAlmostEqual(scores["p1"].first_gain_s, 7.2)
        self.assertAlmostEqual(scores["p1"].second_gain_s, 2.2)
        self.assertAlmostEqual(scores["p1"].first_share, 7.2 / 9.4)
        self.assertAlmostEqual(low_alpha.contract.rub_share_first,
                               1.0 / (1.0 + (1.0 - math.exp(-0.6))))
        self.assertEqual(decode(0.3, 1.0).contract.reference_id, "p2")
        self.assertEqual(decode(0.2, 0.0).contract.reference_id, "p2")

    def test_contract_checks_direction_reachability_and_occupancy(self):
        vehicles, contracts = crossing_contracts()
        ab = contracts[0]
        self.assertTrue(ab.contains({"A": 3.0, "B": 5.8}, vehicles))
        self.assertFalse(ab.contains({"A": 3.0, "B": 5.79}, vehicles))
        self.assertFalse(replace(ab, first_before_second=False).contains(
            {"A": 3.0, "B": 5.8}, vehicles))
        self.assertFalse(ab.contains({"A": 1.0, "B": 5.8}, vehicles))


class TestFiniteCrossingCooperation(unittest.TestCase):
    def test_three_car_conflicting_contracts_have_exact_minimum_repair(self):
        vehicles, contracts = crossing_contracts()
        result = compose_crossing_snapshot(
            vehicles, contracts, {"A": (3.0,), "B": (5.8, 7.0), "C": (3.0,)},
            welfare_tolerance_s=2.0, time_reference_s=1.0,
            edge_weights={"A--B": 2.0, "B--C": 1.0})
        self.assertEqual(result.search_scope, "exact_over_supplied_finite_candidates")
        self.assertEqual(result.enumerated_plan_count, 2)
        self.assertEqual(result.safe_plan_count, 2)
        self.assertEqual(result.exactly_composable_count, 0)
        self.assertEqual(result.welfare_eligible_count, 2)
        self.assertEqual(result.global_plan.broken_contracts, 1)
        self.assertEqual(result.global_plan.entry_dict(),
                         {"A": 3.0, "B": 5.8, "C": 3.0})
        self.assertAlmostEqual(result.global_plan.edge_fidelity, 1.2)
        self.assertAlmostEqual(result.global_plan.vehicle_change, 2.8)
        self.assertAlmostEqual(result.welfare_optimum_s, -2.8)
        self.assertTrue(result.sequential_raw_target_safe)
        self.assertEqual(result.sequential_edge_order, ("B--C", "A--B"))
        self.assertEqual(result.sequential_overwrite_count, 1)
        self.assertEqual(
            result.sequential_edge_history[1]["overwritten_vehicle_targets"]["B"],
            {"previous_time_s": 7.0, "new_time_s": 5.8})
        self.assertAlmostEqual(dict(result.sequential_raw_targets_s)["B"], 5.8)
        self.assertEqual(result.sequential_plan.entry_dict()["B"], 5.8)

    def test_two_car_result_returns_its_own_contract_reference(self):
        vehicles, contracts = crossing_contracts()
        result = compose_crossing_snapshot(
            {name: vehicles[name] for name in ("A", "B")}, contracts[:1],
            {"A": (3.0,), "B": (5.8, 7.0)},
            welfare_tolerance_s=2.0, time_reference_s=1.0)
        self.assertEqual(result.exactly_composable_count, 1)
        self.assertEqual(result.global_plan.broken_contracts, 0)
        self.assertEqual(result.global_plan.entry_dict(),
                         {"A": 3.0, "B": 5.8})

    def test_welfare_floor_and_edge_weights_change_selection(self):
        vehicles, contracts = crossing_contracts()
        candidates = {"A": (3.0,), "B": (5.8, 7.0), "C": (3.0,)}
        weighted = compose_crossing_snapshot(
            vehicles, contracts, candidates, 2.0, 1.0,
            edge_weights={"A--B": 1.0, "B--C": 2.0})
        self.assertEqual(weighted.global_plan.entry_dict()["B"], 7.0)
        constrained = compose_crossing_snapshot(
            vehicles, contracts, candidates, 0.0, 1.0,
            edge_weights={"A--B": 1.0, "B--C": 2.0})
        self.assertEqual(constrained.welfare_eligible_count, 1)
        self.assertEqual(constrained.global_plan.entry_dict()["B"], 5.8)

    def test_search_cap_rejects_unadvertised_partial_enumeration(self):
        vehicles, contracts = crossing_contracts()
        with self.assertRaisesRegex(ValueError, "exceeds explicit cap"):
            compose_crossing_snapshot(
                vehicles, contracts,
                {"A": (3.0,), "B": (5.8, 7.0), "C": (3.0,)},
                welfare_tolerance_s=2.0, time_reference_s=1.0,
                max_joint_combinations=1)


if __name__ == "__main__":
    unittest.main()
