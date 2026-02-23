import unittest

import numpy as np

from experiments.legacy.exp48_global_visibility_predicts_discovery import (
    predicted_q_lin_holdout_from_roi,
    sample_uniform_holdout_mask,
)
from iqp_generative import core as hv


class TestExp48GlobalVisibilityPredictsDiscovery(unittest.TestCase):
    def test_predicted_q_lin_formula_matches_manual(self) -> None:
        n = 5
        N = 2 ** n
        holdout_k = 4

        bits_table = hv.make_bits_table(n)
        alphas = hv.sample_alphas(n=n, sigma=1.0, K=10, seed=13)
        P = hv.build_parity_matrix(alphas, bits_table)

        rng = np.random.default_rng(9)
        p = rng.random(N)
        p = p / float(np.sum(p))
        z = P @ p

        roi_mask = np.zeros(N, dtype=bool)
        roi_mask[np.arange(0, N, 2)] = True
        roi_size = int(np.sum(roi_mask))

        got = predicted_q_lin_holdout_from_roi(
            P=P,
            z=z,
            roi_mask=roi_mask,
            holdout_k=holdout_k,
            n=n,
        )

        hat1_g = hv.indicator_walsh_coeffs(P=P, holdout_mask=roi_mask, n=n)
        vis_g = float(np.dot(z, hat1_g))
        want = float(holdout_k / N) + float(holdout_k / roi_size) * vis_g
        self.assertAlmostEqual(got, want, places=12)

    def test_uniform_holdout_mask_has_valid_subset_and_size(self) -> None:
        N = 16
        roi_indices = np.array([1, 4, 7, 10, 13], dtype=int)
        holdout_k = 3

        mask = sample_uniform_holdout_mask(
            rng=np.random.default_rng(5),
            roi_indices=roi_indices,
            holdout_k=holdout_k,
            N=N,
        )
        picked = np.where(mask)[0]
        self.assertEqual(int(np.sum(mask)), holdout_k)
        self.assertTrue(set(picked.tolist()).issubset(set(roi_indices.tolist())))

    def test_uniform_holdout_mask_invalid_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            _ = sample_uniform_holdout_mask(
                rng=np.random.default_rng(0),
                roi_indices=np.array([0, 1, 2], dtype=int),
                holdout_k=4,
                N=8,
            )


if __name__ == "__main__":
    unittest.main()
