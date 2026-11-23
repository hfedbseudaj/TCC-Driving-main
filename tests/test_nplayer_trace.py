"""Trace opt-in must preserve the archived multi-car policy outputs."""

import unittest

import torch
from torch import nn

from common.nplayer_bargaining import sequential_pairwise_bargaining


class _Utility(nn.Module):
    def forward(self, states, offers):
        return offers[:, :1]


class _DeterministicBargainer(nn.Module):
    offer_dim = 4
    max_rounds = 3
    waypoint_offset_scale = 1.0

    def __init__(self):
        super().__init__()
        self.utility_estimator = _Utility()

    def encode_states(self, features):
        return features, "test_features"

    def offer_decoder(self, states):
        return torch.sigmoid(states[:, :4])

    def discount(self, ttc):
        return (1.0 - torch.exp(-0.3 * ttc)).clamp(0.05, 0.99)

    def offer_to_waypoint(self, offers):
        return offers


class NPlayerTraceTests(unittest.TestCase):
    def test_enabling_trace_preserves_actions_and_records_each_edge(self):
        model = _DeterministicBargainer()
        features = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                 [0.5, 0.1, 0.0, 0.0],
                                 [0.2, 0.3, 0.0, 0.0]])
        positions = torch.tensor([[-10.0, 0.0], [0.0, -10.0], [10.0, 0.0]])
        velocities = torch.tensor([[5.0, 0.0], [0.0, 5.0], [-5.0, 0.0]])
        headings = torch.tensor([0.0, 1.5707963268, 3.1415926536])
        waypoints = torch.zeros((3, 2, 2))
        with torch.no_grad():
            plain = sequential_pairwise_bargaining(
                model, features, waypoints, positions, velocities, headings)
            traced = sequential_pairwise_bargaining(
                model, features, waypoints, positions, velocities, headings,
                record_pair_trace=True)
        self.assertTrue(torch.equal(plain["adjusted_waypoints"],
                                    traced["adjusted_waypoints"]))
        self.assertTrue(torch.equal(plain["bargaining_offers"],
                                    traced["bargaining_offers"]))
        self.assertNotIn("trace", plain["metadata"])
        self.assertEqual(len(traced["metadata"]["pair_schedule"]), 3)
        for edge in traced["metadata"]["pair_schedule"]:
            self.assertEqual(len(edge["input_offers"]), 2)
            self.assertEqual(len(edge["negotiated_offers"]), 2)
            self.assertEqual(len(edge["discount_factors"]), 2)
            self.assertEqual(len(edge["waypoint_adjustments"]), 2)
        self.assertEqual(len(traced["metadata"]["trace"]["adjusted_waypoints"]), 3)


if __name__ == "__main__":
    unittest.main()
