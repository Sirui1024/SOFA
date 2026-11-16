import unittest
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import SOFA
import NSGA3
import SWFAGA


class PopulationLimitTests(unittest.TestCase):
    def test_limits_and_policy_value_pairing(self):
        scaler = MinMaxScaler((0, 100)).fit([[0]*4, [100]*4])
        values = [[float(i+1), float(90-i), float(i+2), float(80-i)] for i in range(12)]
        designs = list(range(len(values)))
        for limit in (1, 4, 7, 12, 20):
            for name, prune in (
                ('SOFA', lambda: SOFA.phv_based_pruning(values, designs, scaler, [100]*4, limit)),
                ('NSGA3', lambda: NSGA3.create_sparse_population(values, designs, scaler, limit)),
                ('SWFAGA', lambda: SWFAGA.create_sparse_population(values, designs, scaler, limit)),
            ):
                with self.subTest(framework=name, limit=limit):
                    kept_values, kept_designs = prune()
                    self.assertGreater(len(kept_values), 0)
                    self.assertLessEqual(len(kept_values), limit)
                    self.assertEqual(len(kept_values), len(kept_designs))
                    self.assertEqual(len(set(kept_designs)), len(kept_designs))
                    for value, design in zip(kept_values, kept_designs):
                        np.testing.assert_array_equal(value, values[design])

    def test_singleton_population_limit(self):
        scaler = MinMaxScaler((0, 100)).fit([[0]*4, [100]*4])
        for module in (NSGA3, SWFAGA, SOFA):
            values, designs = module.create_sparse_population([[1]*4], ['policy'], scaler, 1)
            self.assertEqual(designs, ['policy'])


if __name__ == '__main__':
    unittest.main()
