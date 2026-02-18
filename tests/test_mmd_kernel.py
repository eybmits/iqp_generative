import unittest

import numpy as np

from iqp_generative import core as hv


class TestMMDKernel(unittest.TestCase):
    def test_hamming_rbf_kernel_matrix_properties(self) -> None:
        k_mat = hv._build_mmd_kernel_matrix(n=4, kernel="hamming_rbf", tau=2.0)
        self.assertEqual(k_mat.shape, (16, 16))
        self.assertTrue(np.allclose(k_mat, k_mat.T, atol=1e-12))
        self.assertTrue(np.allclose(np.diag(k_mat), 1.0, atol=1e-12))
        self.assertGreaterEqual(float(np.min(k_mat)), 0.0)
        self.assertLessEqual(float(np.max(k_mat)), 1.0 + 1e-12)

        eigvals = np.linalg.eigvalsh(k_mat)
        self.assertGreater(float(np.min(eigvals)), -1e-8)


if __name__ == "__main__":
    unittest.main()
