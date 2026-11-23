"""The new adapter must expose failures and preserve the old speed mapping."""

import unittest

from cooperation.legacy_adapter import adapt_waypoint_offsets_to_crossing_contract
from cooperation.physics import VehicleSpec


class LegacyOfferAdapterTests(unittest.TestCase):
    def setUp(self):
        self.a = VehicleSpec("A", (-15.0, 0.0), (1.0, 0.0),
                             15.0, 5.0, 13.0, 2.0, 2.0, 7.0,
                             0.0, 12.0)
        self.b = VehicleSpec("B", (0.0, -15.0), (0.0, 1.0),
                             15.0, 5.0, 10.0, 2.0, 2.0, 7.0,
                             0.0, 12.0)

    def test_safe_offer_creates_contract_and_reports_projection_error(self):
        # The archived synthetic controller maps a forward offset of -1.25 m
        # to a 2.5 m/s speed reduction; this delays B enough to follow A.
        result = adapt_waypoint_offsets_to_crossing_contract(
            self.a, self.b, (0.0, 0.0), (0.0, -1.25), 0.1, 0.2)
        self.assertTrue(result.success)
        self.assertEqual(result.reason, "ok")
        self.assertAlmostEqual(result.old_controller_speed_mps[1], 2.5)
        self.assertGreater(result.tentative_entry_s[1], 5.8)
        self.assertTrue(result.contract.first_before_second)
        self.assertGreater(result.first_waypoint_projection_error_m[1], 0.0)

    def test_unsafe_and_dynamically_unreachable_are_separate_failures(self):
        unsafe = adapt_waypoint_offsets_to_crossing_contract(
            self.a, self.b, (0.0, 0.0), (0.0, 0.0), 0.1, 0.2)
        self.assertFalse(unsafe.success)
        self.assertEqual(unsafe.reason, "pair_zone_occupation_unsafe")
        self.assertIsNone(unsafe.contract)
        unreachable = adapt_waypoint_offsets_to_crossing_contract(
            self.a, self.b, (10.0, 0.0), (0.0, 0.0), 0.1, 0.2)
        self.assertFalse(unreachable.success)
        self.assertEqual(unreachable.reason, "dynamically_unreachable_entry")
        self.assertIsNone(unreachable.contract)

    def test_optional_shared_family_projection_is_explicit(self):
        result = adapt_waypoint_offsets_to_crossing_contract(
            self.a, self.b, (0.0, 0.0), (0.0, 0.0), 0.1, 0.2,
            candidate_entry_times=((3.0, 6.0), (3.0, 6.0)),
            allow_safety_projection=True)
        self.assertTrue(result.success)
        self.assertEqual(result.reason, "safety_projected")
        self.assertEqual(result.raw_failure_reason, "pair_zone_occupation_unsafe")
        self.assertEqual(result.selected_entry_s, (3.0, 6.0))
        self.assertAlmostEqual(result.safety_projection_l1_s, 3.0)


if __name__ == "__main__":
    unittest.main()
