import importlib
import unittest
import numpy as np
from unittest.mock import patch
import ENV_V3 as env

scale = importlib.import_module('15min_simulator').scale_execution_durations


class DurationTests(unittest.TestCase):
    def test_nested_list_scales_only_execution_time(self):
        original = [[[2, .5, 10, 20, 30], [3, 1, 40, 50, 60]], [[4, 2, 70, 80, 90]]]
        result = scale(original, 3)
        self.assertEqual(result, [[[6, .5, 10, 20, 30], [9, 1, 40, 50, 60]], [[12, 2, 70, 80, 90]]])
        self.assertEqual(original[0][0][0], 2)
        self.assertEqual(len(result), len(original))

    def test_numpy_rows_produce_integer_slice_indices(self):
        result = scale(np.array([[[2., .5, 10., 20., 30.]]]), 2)
        self.assertIsInstance(result[0][0][0], int)
        self.assertEqual(result[0][0], [4, .5, 10, 20, 30])

    def test_default_preserves_values_and_limits_are_explicit(self):
        original = [[[900, 1, 2, 3, 4]]]
        self.assertEqual(scale(original, 1), original)
        for multiplier in (0, -1, 1.5, 2):
            with self.assertRaises(ValueError):
                scale(original, multiplier)

    def test_doubled_duration_doubles_resource_seconds(self):
        lut = [[[2, 1, 1e9, 0, 1e9]]]
        before = env.calculate_resource_time([0]*4, [0]*4, [0]*4, lut, 0, 1, 0, 0)[0]
        after = env.calculate_resource_time([0]*4, [0]*4, [0]*4, scale(lut, 2), 0, 1, 0, 0)[0]
        np.testing.assert_allclose(np.asarray(after).sum(axis=(1, 2)),
                                   2*np.asarray(before).sum(axis=(1, 2)))


if __name__ == '__main__':
    unittest.main()
