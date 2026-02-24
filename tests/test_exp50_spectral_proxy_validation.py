import unittest

import numpy as np

from experiments.legacy.exp50_validate_spectral_proxy_multibeta import (
    _bootstrap_ci,
    _pick_best_worst_by_q80_spec,
    _spearman_with_permutation,
)


class TestExp50SpectralProxyValidation(unittest.TestCase):
    def test_pick_best_worst_ignores_non_finite_q80(self) -> None:
        rows = [
            {"sigma": 0.5, "K": 128, "Q80_spec": float("inf"), "qH_ratio_spec": 1.0},
            {"sigma": 1.0, "K": 128, "Q80_spec": 3000.0, "qH_ratio_spec": 2.0},
            {"sigma": 1.5, "K": 256, "Q80_spec": 9000.0, "qH_ratio_spec": 1.5},
            {"sigma": 2.0, "K": 256, "Q80_spec": float("nan"), "qH_ratio_spec": 1.1},
        ]
        best, worst = _pick_best_worst_by_q80_spec(rows)
        self.assertEqual((float(best["sigma"]), int(best["K"])), (1.0, 128))
        self.assertEqual((float(worst["sigma"]), int(worst["K"])), (1.5, 256))

    def test_spearman_monotone_and_antimonotone(self) -> None:
        x = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
        y_pos = np.array([10, 11, 12, 13, 14, 15], dtype=np.float64)
        y_neg = y_pos[::-1]

        pos = _spearman_with_permutation(x, y_pos, perm_repeats=300, seed=1)
        neg = _spearman_with_permutation(x, y_neg, perm_repeats=300, seed=2)

        self.assertAlmostEqual(float(pos["rho"]), 1.0, places=12)
        self.assertAlmostEqual(float(neg["rho"]), -1.0, places=12)
        self.assertTrue(float(pos["p_perm"]) < 0.05)
        self.assertTrue(float(neg["p_perm"]) < 0.05)
        self.assertEqual(int(pos["n_pairs"]), 6)

    def test_bootstrap_ci_single_value(self) -> None:
        ci = _bootstrap_ci([0.42], alpha=0.05, n_boot=1000, seed=7)
        self.assertAlmostEqual(float(ci["center"]), 0.42, places=12)
        self.assertAlmostEqual(float(ci["ci_lo"]), 0.42, places=12)
        self.assertAlmostEqual(float(ci["ci_hi"]), 0.42, places=12)


if __name__ == "__main__":
    unittest.main()

