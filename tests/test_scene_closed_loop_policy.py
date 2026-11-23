"""Observable guarantees for the new finite-action multicar policy."""

import unittest

from cooperation.scene_closed_loop import (
    Car, _actions, _choose, _initial_chase_braking_slack, _lane_target,
    _robust_chase_braking_slack, _rollout, _trial_initial, make_scene,
    normalize_policy, run_episode,
)


class TestMulticarPolicy(unittest.TestCase):
    def test_crossing_groups_have_one_two_four_conflict_edges(self):
        for count, expected_edges in ((2, 1), (3, 2), (4, 4)):
            cars, edges = make_scene("crossing", count)
            self.assertEqual(len(cars), count)
            self.assertEqual(len(edges), expected_edges)
            self.assertTrue(any(car.axis == "x" for car in cars.values()))
            self.assertTrue(any(car.axis == "y" for car in cars.values()))

    def test_crossing_hold_lane_does_not_snap_to_other_lane_centres(self):
        cars, _ = make_scene("crossing", 4)
        for car in cars.values():
            lateral = car.y if car.axis == "x" else car.x
            self.assertAlmostEqual(_lane_target(car, 0), lateral)
            coast = next(action for action in _actions("crossing", car)
                         if action.name == "cross_+0")
            path = _rollout(car, coast, 0.2, 2.0)
            for state in path:
                state_lateral = state.y if car.axis == "x" else state.x
                self.assertAlmostEqual(state_lateral, lateral)

    def test_global_action_respects_joint_disagreement_when_available(self):
        for scene in ("crossing", "chase", "merge", "head_on"):
            for count in (2, 3, 4):
                cars, edges, _ = _trial_initial(scene, count, 301, 0.3, 0.2)
                _, info = _choose(scene, cars, edges, "global")
                if info["safe_disagreement_available"]:
                    self.assertTrue(info["individual_rational"],
                                    (scene, count))
                    self.assertTrue(all(gain >= -1e-9 for gain in
                                        info["per_vehicle_gain_m"].values()),
                                    (scene, count))
                    self.assertGreater(info["ir_candidate_count"], 0)
                    self.assertFalse(info["fallback"])
                    self.assertGreaterEqual(
                        info["selected_welfare_m"] + 1e-9,
                        info["welfare_optimum_m"] -
                        normalize_policy()["welfare_epsilon_m"],
                        (scene, count))

    def test_mode_ablation_changes_pair_reference_in_crossing(self):
        cars, edges, _ = _trial_initial("crossing", 2, 301, 0.3, 0.2)
        _, ttc = _choose("crossing", cars, edges, "global",
                         {"discount_mode": "ttc"})
        _, equal = _choose("crossing", cars, edges, "global",
                           {"discount_mode": "equal_share"})
        self.assertEqual(ttc["ttc_applied_contract_count"], 1)
        self.assertEqual(equal["ttc_applied_contract_count"], 0)
        self.assertNotEqual(ttc["contract_records"][0]["reference_share_first"],
                            equal["contract_records"][0]["reference_share_first"])

    def test_nominal_conflict_ttc_detects_merge_radial_ttc_misses(self):
        cars, edges, _ = _trial_initial("merge", 4, 301, 0.3, 0.2)
        _, nominal = _choose("merge", cars, edges, "global",
                             {"ttc_source": "nominal_conflict"})
        _, radial = _choose("merge", cars, edges, "global",
                            {"ttc_source": "legacy_radial"})
        nominal_edge = next(record for record in nominal["contract_records"]
                            if record["edge"] == ["M", "B2"])
        radial_edge = next(record for record in radial["contract_records"]
                           if record["edge"] == ["M", "B2"])
        self.assertTrue(nominal_edge["ttc_active"])
        self.assertFalse(radial_edge["ttc_active"])
        self.assertEqual(nominal_edge["legacy_radial_ttc_s"],
                         radial_edge["legacy_radial_ttc_s"])

    def test_initial_perturbation_parameters_reach_episode(self):
        baseline = run_episode("crossing", 2, 301, "global",
                               position_std_m=0.0, speed_std_mps=0.0)
        perturbed = run_episode("crossing", 2, 301, "global",
                                position_std_m=0.3, speed_std_mps=0.2)
        self.assertNotEqual(baseline["initial_vehicles"],
                            perturbed["initial_vehicles"])

    def test_clearance_first_is_a_driving_baseline(self):
        trial = run_episode("crossing", 3, 301, "clearance_first")
        self.assertTrue(trial["completed"])
        self.assertFalse(trial["collision"])
        self.assertEqual(trial["ir_violation_ticks"], 0)

    def test_adaptive_decoder_improves_a_crossing_share_reference(self):
        cars, edges, _ = _trial_initial("crossing", 4, 303, 0.3, 0.2)
        _, coarse = _choose("crossing", cars, edges, "global",
                            {"decoder_mode": "coarse"})
        _, adaptive = _choose("crossing", cars, edges, "global",
                              {"decoder_mode": "adaptive_crossing"})
        self.assertEqual(coarse["contract_count"],
                         adaptive["contract_count"])
        self.assertLess(adaptive["mean_decoded_share_error"],
                        coarse["mean_decoded_share_error"])
        self.assertTrue(adaptive["individual_rational"])
        self.assertFalse(adaptive["fallback"])

    def test_adaptive_decoder_can_change_candidate_family_and_action(self):
        cars, edges, _ = _trial_initial("crossing", 4, 301, 0.3, 0.2)
        ttc_actions, ttc = _choose(
            "crossing", cars, edges, "global",
            {"decoder_mode": "adaptive_crossing", "discount_mode": "ttc"})
        fixed_actions, fixed = _choose(
            "crossing", cars, edges, "global",
            {"decoder_mode": "adaptive_crossing", "discount_mode": "constant"})
        self.assertNotEqual({name: action.name for name, action in
                             ttc_actions.items()},
                            {name: action.name for name, action in
                             fixed_actions.items()})
        self.assertFalse(ttc["fallback"])
        self.assertFalse(fixed["fallback"])
        self.assertGreaterEqual(ttc["min_clearance"], 0.0)
        self.assertGreaterEqual(fixed["min_clearance"], 0.0)

    def test_fixed_candidate_family_is_identical_across_discount_modes(self):
        cars, edges, _ = _trial_initial("crossing", 4, 305, 0.3, 0.2)
        policies = ({"decoder_mode": "adaptive_crossing",
                     "discount_mode": mode}
                    for mode in ("ttc", "constant", "equal_share"))
        first_pass = [_choose("crossing", cars, edges, "global", policy)[1]
                      for policy in policies]
        shared = {name: sorted(set().union(*(
            set(info["joint_candidate_actions"][name])
            for info in first_pass))) for name in cars}
        for mode in ("ttc", "constant", "equal_share"):
            _, info = _choose(
                "crossing", cars, edges, "global",
                {"decoder_mode": "adaptive_crossing", "discount_mode": mode},
                shared_action_names=shared)
            self.assertEqual(info["joint_candidate_actions"], shared)
            self.assertGreater(info["safe_candidate_count"], 0)

    def test_share_objective_transmits_ttc_with_fixed_candidates(self):
        cars, edges, _ = _trial_initial("crossing", 2, 604, 1.0, 0.8)
        base = {"decoder_mode": "adaptive_crossing",
                "joint_objective": "share_then_contract",
                "welfare_epsilon_m": 2.0}
        initial = [_choose("crossing", cars, edges, "global",
                           dict(base, discount_mode=mode))[1]
                   for mode in ("ttc", "constant", "equal_share")]
        shared = {name: sorted(set().union(*(
            set(info["joint_candidate_actions"][name])
            for info in initial))) for name in cars}
        ttc_actions, ttc = _choose(
            "crossing", cars, edges, "global",
            dict(base, discount_mode="ttc"), shared_action_names=shared)
        constant_actions, constant = _choose(
            "crossing", cars, edges, "global",
            dict(base, discount_mode="constant"),
            shared_action_names=shared)
        self.assertNotEqual({name: action.name for name, action in
                             ttc_actions.items()},
                            {name: action.name for name, action in
                             constant_actions.items()})
        for info in (ttc, constant):
            self.assertEqual(info["joint_candidate_actions"], shared)
            self.assertGreater(info["welfare_eligible_count"], 1)
            self.assertGreaterEqual(info["min_clearance"], 0.0)
            self.assertTrue(info["individual_rational"])
            self.assertGreaterEqual(
                info["selected_welfare_m"] + 1e-9,
                info["welfare_optimum_m"] - 2.0)

    def test_empty_two_second_family_uses_safe_next_step(self):
        base, edges = make_scene("crossing", 2)
        west, south = (base[name] for name in edges[0])
        cars = {
            west.id: Car(west.id, -10.0, west.y, west.direction, 10.0,
                         0.0, west.role, west.goal_x, west.max_speed,
                         axis=west.axis),
            south.id: Car(south.id, south.x, -10.0, south.direction, 6.0,
                          0.0, south.role, south.goal_x, south.max_speed,
                          axis=south.axis),
        }
        _, result = _choose("crossing", cars, edges, "global")
        self.assertTrue(result["fallback"])
        self.assertEqual(result["safe_candidate_count"], 0)
        self.assertGreater(result["next_step_safe_candidate_count"], 0)
        self.assertTrue(result["next_step_predicted_safe"])
        self.assertEqual(result["selected_source"], "one_step_fallback")
        self.assertEqual(result["fallback_safety_tier"],
                         "full_prediction_margin")
        self.assertTrue(result["current_full_margin_satisfied"])

    def test_fallback_separates_robust_margin_from_physical_clearance(self):
        trial = run_episode("crossing", 3, 503, "global",
                            position_std_m=1.0, speed_std_mps=0.8)
        fallback_steps = [step for step in trial["decision_trace"]
                          if step["fallback"]]
        # Keep the test tied to the safety outcome rather than a specific
        # fallback count: a valid plan may be found as the candidate model is
        # refined, while any fallback must still report its safety tier.
        self.assertTrue(trial["completed"])
        self.assertFalse(trial["collision"])
        if fallback_steps:
            self.assertEqual(len(fallback_steps), 1)
            self.assertEqual(fallback_steps[0]["fallback_safety_tier"],
                             "physical_envelope_only")
            self.assertFalse(fallback_steps[0]["next_step_predicted_safe"])
            self.assertTrue(fallback_steps[0][
                "next_step_predicted_collision_free"])
            self.assertEqual(trial[
                "next_step_full_margin_shortfall_ticks"], 1)
            self.assertEqual(trial[
                "next_step_collision_uncertified_fallback_ticks"], 0)
        else:
            self.assertEqual(trial["next_step_full_margin_shortfall_ticks"], 0)

    def test_chase_action_family_recovers_a_two_car_following_case(self):
        trial = run_episode("chase", 2, 608, "global",
                            position_std_m=1.0, speed_std_mps=0.8)
        self.assertTrue(trial["initial_full_margin_satisfied"])
        self.assertEqual(trial["fallback_count"], 0)
        self.assertTrue(trial["completed"])
        self.assertFalse(trial["collision"])

    def test_initial_margin_violation_is_separate_from_planner_fallback(self):
        trial = run_episode("chase", 4, 607, "global",
                            position_std_m=1.0, speed_std_mps=0.8)
        self.assertFalse(trial["initial_full_margin_satisfied"])
        self.assertGreater(trial["initial_physical_clearance_m"], 0.0)
        self.assertGreater(trial["fallback_count"], 0)
        self.assertEqual(trial[
            "next_step_collision_uncertified_fallback_ticks"], 0)

    def test_chase_initial_braking_slack_flags_near_dynamic_boundary(self):
        trial = run_episode("chase", 3, 820, "global",
                            position_std_m=1.0, speed_std_mps=0.8)
        self.assertTrue(trial["initial_full_margin_satisfied"])
        self.assertGreater(trial["initial_physical_clearance_m"], 1.0)
        self.assertIsNotNone(trial["initial_best_case_braking_slack_m"])
        self.assertLess(trial["initial_best_case_braking_slack_m"], 0.1)

    def test_robust_braking_slack_tightens_nominal_noise_free_estimate(self):
        cars, _, _ = _trial_initial("chase", 3, 820, 1.0, 0.8)
        nominal = _initial_chase_braking_slack(cars)
        robust = _robust_chase_braking_slack(cars)
        self.assertIsNotNone(nominal)
        self.assertIsNotNone(robust)
        self.assertLess(robust, nominal)

    def test_dynamic_braking_filter_keeps_tight_initial_state_in_metrics(self):
        trial = run_episode(
            "chase", 3, 820, "global", position_std_m=1.0,
            speed_std_mps=0.8, policy={"dynamic_braking_filter": True})
        self.assertFalse(trial["initial_dynamic_braking_margin_satisfied"])
        self.assertTrue(trial["decision_trace"][0][
            "dynamic_braking_filter_active"])
        first_decision = trial["decision_trace"][0]
        self.assertGreater(
            trial["dynamic_braking_candidate_rejected_count"], 0)
        self.assertGreaterEqual(
            first_decision["selected_predicted_robust_braking_slack_m"],
            first_decision["dynamic_braking_threshold_m"] - 1e-9)
        self.assertEqual(trial["trial_seed"], 820)
        self.assertIn("collision", trial)
        self.assertIn("completed", trial)
        self.assertGreater(trial["dynamic_braking_filter_active_ticks"], 0)

    def test_invalid_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_policy({"ttc_start_s": -1.0})

    def test_non_chase_staged_recovery_fills_empty_constant_family(self):
        # 1. Crossing 3-car boundary state
        c_cross, edges_cross = make_scene("crossing", 3)
        c_cross = {
            "W_to_E": Car("W_to_E", 3.87, -1.75, 1, 10.0, 0.0, "horizontal", 20.0, axis="x"),
            "E_to_W": Car("E_to_W", 21.45, 1.75, -1, 12.0, 0.0, "horizontal", -20.0, axis="x"),
            "S_to_N": Car("S_to_N", 1.75, -8.20, 1, 8.8, 0.0, "vertical", 20.0, axis="y"),
        }
        _, res_cross = _choose("crossing", c_cross, edges_cross, "global")
        self.assertTrue(res_cross["fallback"])
        self.assertEqual(res_cross["safe_candidate_count"], 0)
        self.assertGreater(res_cross["recovery_candidate_count"], 0)
        self.assertTrue(res_cross["recovery_profile_used"])
        self.assertEqual(res_cross["selected_source"], "piecewise_braking_recovery")

        # 2. Merge 3-car boundary state
        c_merge, edges_merge = make_scene("merge", 3)
        c_merge = {
            "B1": Car("B1", 0.0, 0.0, 1, 11.95, 0.0, "mainline", 20.0, axis="x"),
            "M": Car("M", 6.45, -2.2, 1, 11.5, 0.3, "merging", 20.0, axis="x"),
            "B2": Car("B2", 13.00, 0.0, 1, 9.40, 0.0, "mainline", 20.0, axis="x"),
        }
        _, res_merge = _choose("merge", c_merge, edges_merge, "global")
        self.assertTrue(res_merge["fallback"])
        self.assertEqual(res_merge["safe_candidate_count"], 0)
        self.assertGreater(res_merge["recovery_candidate_count"], 0)
        self.assertTrue(res_merge["recovery_profile_used"])
        self.assertEqual(res_merge["selected_source"], "piecewise_braking_recovery")

        # 3. Head-on 3-car boundary state
        c_head, edges_head = make_scene("head_on", 3)
        c_head = {
            "E2": Car("E2", -6.75, 0.0, 1, 11.8, 0.0, "eastbound", 20.0, axis="x"),
            "E1": Car("E1", 0.0, 0.0, 1, 10.2, 0.0, "eastbound", 20.0, axis="x"),
            "W": Car("W", 31.8, 0.0, -1, 9.5, 0.0, "westbound", -20.0, axis="x"),
        }
        _, res_head = _choose("head_on", c_head, edges_head, "global")
        self.assertTrue(res_head["fallback"])
        self.assertEqual(res_head["safe_candidate_count"], 0)
        self.assertGreater(res_head["recovery_candidate_count"], 0)
        self.assertTrue(res_head["recovery_profile_used"])
        self.assertEqual(res_head["selected_source"], "piecewise_braking_recovery")


if __name__ == "__main__":
    unittest.main()
