import importlib
import unittest

import numpy as np

calculate_final_phv = importlib.import_module('15min_simulator').calculate_final_phv


class FinalPhvTests(unittest.TestCase):
    def test_single_policy_matches_minimization_box_volume(self):
        self.assertAlmostEqual(calculate_final_phv([[10, 20, 30, 40]], [100]*4),
                               90*80*70*60)

    def test_improving_each_objective_increases_phv(self):
        original = np.array([[20., 20, 20, 20]])
        baseline = calculate_final_phv(original, [100]*4)
        for column in range(4):
            improved = original.copy()
            improved[0, column] = 10
            self.assertGreater(calculate_final_phv(improved, [100]*4), baseline)
        np.testing.assert_array_equal(original, [[20, 20, 20, 20]])

    def test_population_and_single_policy_share_scale(self):
        best = [10, 20, 30, 40]
        self.assertEqual(calculate_final_phv([best], [100]*4),
                         calculate_final_phv([best, [20, 30, 40, 50]], [100]*4))

    def test_infeasible_and_outside_reference_contribute_zero(self):
        self.assertEqual(calculate_final_phv([[100, np.inf, np.inf, np.inf],
                                            [101, 0, 0, 0]], [100]*4), 0)
        self.assertEqual(calculate_final_phv([], [100]*4), 0)


if __name__ == '__main__':
    unittest.main()
