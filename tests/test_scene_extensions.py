"""Provisional E5 definitions retain two-car regression and finite safety."""

import unittest

from cooperation.scene_extensions import evaluate_scene


class SceneExtensionsTests(unittest.TestCase):
    def test_two_car_reference_is_jointly_executable_in_each_scene(self):
        for scene in ("chase", "merge", "head_on"):
            with self.subTest(scene=scene):
                result = evaluate_scene(scene, n_cars=2, trial_seed=101)
                self.assertGreater(result.safe_plan_count, 0)
                self.assertGreater(result.exactly_composable_count, 0)
                self.assertEqual(result.global_broken_contracts, 0)
                self.assertEqual(dict(result.global_controls),
                                 {result.contracts[0].first_id: result.contracts[0].first_control,
                                  result.contracts[0].second_id: result.contracts[0].second_control})

    def test_three_to_four_cars_expand_the_explicit_search_family(self):
        for scene in ("chase", "merge", "head_on"):
            with self.subTest(scene=scene):
                three = evaluate_scene(scene, n_cars=3, trial_seed=101)
                four = evaluate_scene(scene, n_cars=4, trial_seed=101)
                self.assertGreater(four.enumerated_plan_count,
                                   three.enumerated_plan_count)
                self.assertGreater(three.safe_plan_count, 0)
                self.assertGreater(four.safe_plan_count, 0)
                self.assertGreaterEqual(three.global_welfare_m,
                                        three.welfare_floor_m - 1e-9)
                self.assertGreaterEqual(four.global_welfare_m,
                                        four.welfare_floor_m - 1e-9)


if __name__ == "__main__":
    unittest.main()
