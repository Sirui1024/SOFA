import unittest
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import MinMaxScaler

import Kimchi
import SWFAGA


class SelectionTests(unittest.TestCase):
    def test_kimchi_keeps_utility_improvement_with_zero_carbon(self):
        scaler = MinMaxScaler((0, 100)).fit([[0] * 4, [100] * 4])
        original = np.array([[0, 0.25, 0.25, 0.25, 0.25]])
        candidate = np.array([[0, 0.2, 0.3, 0.25, 0.25]])
        with patch.object(Kimchi, 'perturb', return_value=candidate), \
                patch.object(Kimchi, 'simulation', return_value=([2, 0, 0, 0], None)):
            _, value, design = Kimchi.perturb_and_simulate_10time(
                [5, 0, 0, 0], original, 0.05, None, None, None,
                None, 95, scaler, 20, 100,
            )
        self.assertEqual(value, [2, 0, 0, 0])
        np.testing.assert_array_equal(design, candidate)

    def test_swfaga_constant_objectives_are_finite(self):
        with np.errstate(divide='raise', invalid='raise'):
            for values in ([[5, 0, 0, 0]], [[5, 0, 0, 0], [2, 0, 0, 0]]):
                score = SWFAGA.CV_calculation(4, values, [50, 100, 100, 100])
                self.assertTrue(np.isfinite(score))


if __name__ == '__main__':
    unittest.main()
