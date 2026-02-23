import itertools
import unittest

import numpy as np

from experiments.legacy.exp47_expected_visibility_scaling import (
    expected_rhs_from_roi,
    monte_carlo_lhs_for_random_holdouts,
)
from iqp_generative import core as hv


class TestExp47ExpectedVisibilityScaling(unittest.TestCase):
    def test_expected_rhs_matches_exact_enumeration_small_case(self) -> None:
        n = 4
        h = 3
        K = 7

        bits_table = hv.make_bits_table(n)
        p_star, support, _scores = hv.build_target_distribution_paper(n, beta=0.8)
        roi_mask = support.astype(bool)
        roi_idx = np.where(roi_mask)[0]

        alphas = hv.sample_alphas(n=n, sigma=1.0, K=K, seed=42)
        P = hv.build_parity_matrix(alphas, bits_table)
        z = P @ p_star
        q_lin = hv.linear_band_reconstruction(P, z, n=n)

        rhs = expected_rhs_from_roi(P=P, z=z, roi_mask=roi_mask, h=h, n=n)

        combos = list(itertools.combinations(roi_idx.tolist(), h))
        N = 2 ** n
        acc_hat1 = np.zeros(K, dtype=np.float64)
        acc_vis = 0.0
        acc_q = 0.0

        for comb in combos:
            pick = np.asarray(comb, dtype=int)
            hat1_h = np.sum(P[:, pick], axis=1) / float(N)
            acc_hat1 += hat1_h
            acc_vis += float(np.dot(z, hat1_h))
            acc_q += float(np.sum(q_lin[pick]))

        denom = float(len(combos))
        lhs_hat1 = acc_hat1 / denom
        lhs_vis = acc_vis / denom
        lhs_q = acc_q / denom

        self.assertTrue(np.allclose(lhs_hat1, np.asarray(rhs["rhs_hat1"]), atol=1e-12))
        self.assertAlmostEqual(lhs_vis, float(rhs["rhs_vis"]), places=12)
        self.assertAlmostEqual(lhs_q, float(rhs["rhs_q_lin_h"]), places=12)

    def test_monte_carlo_lhs_tracks_rhs(self) -> None:
        n = 4
        h = 3
        K = 9
        trials = 3000

        bits_table = hv.make_bits_table(n)
        p_star, support, _scores = hv.build_target_distribution_paper(n, beta=1.0)
        roi_mask = support.astype(bool)
        roi_idx = np.where(roi_mask)[0]

        alphas = hv.sample_alphas(n=n, sigma=1.0, K=K, seed=123)
        P = hv.build_parity_matrix(alphas, bits_table)
        z = P @ p_star
        q_lin = hv.linear_band_reconstruction(P, z, n=n)

        rhs = expected_rhs_from_roi(P=P, z=z, roi_mask=roi_mask, h=h, n=n)
        lhs = monte_carlo_lhs_for_random_holdouts(
            P=P,
            z=z,
            q_lin=q_lin,
            roi_indices=roi_idx,
            h=h,
            n=n,
            trials=trials,
            rng=np.random.default_rng(7),
        )

        self.assertAlmostEqual(float(lhs["lhs_vis"]), float(rhs["rhs_vis"]), delta=3e-3)
        self.assertAlmostEqual(float(lhs["lhs_q_lin_h"]), float(rhs["rhs_q_lin_h"]), delta=3e-3)


if __name__ == "__main__":
    unittest.main()
