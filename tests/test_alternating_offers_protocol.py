import unittest

import torch
from torch import nn

from opencood.models.comm_modules.rubinstein_bargaining import (
    alternating_offers_protocol,
)


class OfferValue(nn.Module):
    """A deterministic utility used to isolate the bargaining loop."""

    def forward(self, states, offers):
        return offers[:, :1]


class NegativeOfferValue(nn.Module):
    def forward(self, states, offers):
        return offers[:, :1] - 1.2


class ShiftedOfferValue(nn.Module):
    def forward(self, states, offers):
        return offers[:, :1] - 0.3


class AlternatingOffersProtocolTests(unittest.TestCase):
    def setUp(self):
        self.states = torch.zeros(2, 1)
        self.offers = torch.tensor(
            [[0.8, 0.1, 0.2, 0.3], [0.2, 0.5, 0.4, 0.6]],
            dtype=torch.float32,
        )

    def test_accepted_mean_updates_proposer_before_convergence(self):
        low_discount, accepted, metadata = alternating_offers_protocol(
            self.offers,
            torch.tensor([0.5, 0.5]),
            OfferValue(),
            self.states,
            max_rounds=3,
            return_all_offers=True,
        )

        self.assertTrue(metadata["converged"])
        self.assertEqual(metadata["n_rounds"], 1)
        self.assertEqual(metadata["n_offer_updates"], 1)
        self.assertTrue(bool(accepted[1]))
        # The proposer moves 15% toward the accepted-offer mean (0.5).
        self.assertAlmostEqual(float(low_discount[0, 0]), 0.755, places=6)
        self.assertAlmostEqual(float(low_discount[1, 0]), 0.2, places=6)
        self.assertEqual(metadata["all_offers"].shape[0], 2)

    def test_ttc_discount_changes_offer_and_keeps_gradient(self):
        low_delta = torch.tensor([0.5, 0.5], requires_grad=True)
        high_delta = torch.tensor([0.9, 0.9])
        low_offer, _, _ = alternating_offers_protocol(
            self.offers,
            low_delta,
            OfferValue(),
            self.states,
            max_rounds=3,
        )
        high_offer, _, _ = alternating_offers_protocol(
            self.offers,
            high_delta,
            OfferValue(),
            self.states,
            max_rounds=3,
        )

        self.assertLess(float(low_offer[0, 0]), float(high_offer[0, 0]))
        self.assertGreater(float((low_offer - high_offer).norm()), 1e-4)
        low_offer[0, 0].backward()
        self.assertIsNotNone(low_delta.grad)
        self.assertGreater(abs(float(low_delta.grad[0])), 0.0)

    def test_no_accepted_offer_keeps_status_quo_and_uses_round_cap(self):
        final_offers, accepted, metadata = alternating_offers_protocol(
            self.offers,
            torch.tensor([0.5, 0.5]),
            NegativeOfferValue(),
            self.states,
            max_rounds=3,
        )

        self.assertFalse(metadata["converged"])
        self.assertEqual(metadata["n_rounds"], 3)
        self.assertEqual(metadata["n_offer_updates"], 0)
        self.assertFalse(bool(accepted.any()))
        self.assertTrue(torch.equal(final_offers, self.offers))

    def test_round_horizon_and_discount_change_counteroffer_outcome(self):
        short_horizon, _, short_metadata = alternating_offers_protocol(
            self.offers,
            torch.tensor([0.5, 0.5]),
            ShiftedOfferValue(),
            self.states,
            max_rounds=1,
        )
        longer_horizon, _, longer_metadata = alternating_offers_protocol(
            self.offers,
            torch.tensor([0.5, 0.5]),
            ShiftedOfferValue(),
            self.states,
            max_rounds=3,
        )
        impatient_delta, _, impatient_metadata = alternating_offers_protocol(
            self.offers,
            torch.tensor([0.9, 0.9]),
            ShiftedOfferValue(),
            self.states,
            max_rounds=3,
        )

        self.assertFalse(short_metadata["converged"])
        self.assertTrue(longer_metadata["converged"])
        self.assertLess(short_metadata["n_rounds"], longer_metadata["n_rounds"])
        self.assertNotAlmostEqual(
            float(longer_horizon[1, 0]), float(impatient_delta[1, 0]), places=4
        )


if __name__ == "__main__":
    unittest.main()
