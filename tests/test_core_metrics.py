import math
import unittest

import numpy as np

from iqp_generative import core as hv


class TestCoreMetrics(unittest.TestCase):
    def test_expected_unique_fraction_toy(self) -> None:
        probs = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float64)
        holdout_mask = np.array([True, True, False, False], dtype=bool)
        q_vals = np.array([1, 2, 3], dtype=int)
        got = hv.expected_unique_fraction(probs, holdout_mask, q_vals)
        want = np.array([0.5, 0.75, 0.875], dtype=np.float64)
        self.assertTrue(np.allclose(got, want, atol=1e-12))

    def test_compute_metrics_q80_toy(self) -> None:
        q = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float64)
        holdout_mask = np.array([True, True, False, False], dtype=bool)
        metrics = hv.compute_metrics_for_q(
            q=q,
            holdout_mask=holdout_mask,
            qH_unif=0.5,
            H_size=2,
            Q80_thr=0.8,
            Q80_search_max=1000,
        )
        self.assertEqual(int(metrics["Q80"]), 3)
        self.assertAlmostEqual(metrics["qH"], 1.0, places=12)
        self.assertAlmostEqual(metrics["qH_ratio"], 2.0, places=12)
        self.assertGreaterEqual(metrics["R_Q1000"], 0.999)

    def test_compute_metrics_inf_when_zero_holdout_mass(self) -> None:
        q = np.array([0.0, 0.0, 0.5, 0.5], dtype=np.float64)
        holdout_mask = np.array([True, True, False, False], dtype=bool)
        metrics = hv.compute_metrics_for_q(
            q=q,
            holdout_mask=holdout_mask,
            qH_unif=0.5,
            H_size=2,
            Q80_thr=0.8,
            Q80_search_max=1000,
        )
        self.assertTrue(math.isinf(metrics["Q80"]))
        self.assertAlmostEqual(metrics["qH"], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
