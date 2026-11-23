"""Paired crossing episodes must replan and preserve physical input pairing."""

import json
from pathlib import Path
import unittest

from cooperation.crossing_closed_loop import run_crossing_episode
from cooperation.run_paired_closed_loop import _trial_initial_conditions
from cooperation.run_snapshot import _vehicle_from_row


_CONFIG = (Path(__file__).resolve().parent.parent / "config" /
           "multicar_cooperation_example.json")


class CrossingClosedLoopTests(unittest.TestCase):
    def test_each_tick_redecodes_without_initial_conflict(self):
        config = json.loads(_CONFIG.read_text(encoding="utf-8"))
        vehicles = {row["id"]: _vehicle_from_row(row)
                    for row in config["vehicles"]}
        edges = [tuple(row["edge"]) for row in config["pair_proposals"]]
        noise = [[0.0, 0.0, 0.0] for _ in range(8)]
        global_result = run_crossing_episode(vehicles, edges, "global", noise)
        pair_result = run_crossing_episode(vehicles, edges, "adapted_pairwise", noise)
        self.assertEqual(global_result["steps"], 8)
        self.assertEqual(pair_result["steps"], 8)
        self.assertEqual(global_result["fallback_count"], 0)
        self.assertEqual(global_result["logs"][0]["contracts"],
                         pair_result["logs"][0]["contracts"])
        self.assertNotEqual(global_result["logs"][0]["contracts"][0]["ttc_s"],
                            global_result["logs"][1]["contracts"][0]["ttc_s"])
        self.assertFalse(global_result["collision"])

    def test_paired_trials_reuse_same_perturbation_and_disturbance(self):
        config = json.loads(_CONFIG.read_text(encoding="utf-8"))
        a = _trial_initial_conditions(config, 123, 0.3, 0.2, 0.1)
        b = _trial_initial_conditions(config, 123, 0.3, 0.2, 0.1)
        self.assertEqual(a, b)
        c = _trial_initial_conditions(config, 124, 0.3, 0.2, 0.1)
        self.assertNotEqual(a[1], c[1])

    def test_infeasible_replan_uses_shield_and_recovers(self):
        config = json.loads(_CONFIG.read_text(encoding="utf-8"))
        vehicles, _, noise = _trial_initial_conditions(config, 105, 0.3, 0.2, 0.1)
        edges = [tuple(row["edge"]) for row in config["pair_proposals"]]
        result = run_crossing_episode(vehicles, edges, "global", noise)
        self.assertFalse(result["planner_aborted"])
        self.assertTrue(result["completed"])
        self.assertFalse(result["collision"])
        self.assertGreater(result["recovery_step_count"], 0)
        self.assertEqual(result["fallback_count"], result["recovery_step_count"])
        self.assertTrue(all(row["recovery_safe"] for row in result["logs"]
                            if row["recovery_safe"] is not None))


if __name__ == "__main__":
    unittest.main()
