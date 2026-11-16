import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import CASA
import ENV_V3 as env


class CasaTests(unittest.TestCase):
    def test_switch_and_acceptance_use_separate_objectives(self):
        self.assertEqual(CASA.search_mode([3, 1, 1, 1], 2), 'utility')
        self.assertEqual(CASA.search_mode([2, 1, 1, 1], 2), 'carbon')
        self.assertTrue(CASA.improves([1, 9, 9, 9], [3, 1, 1, 1], 'utility'))
        self.assertTrue(CASA.improves([9, 9, .5, 9], [2, 1, 1, 1], 'carbon'))
        self.assertFalse(CASA.improves([1, 0, 2, 0], [2, 1, 1, 1], 'carbon'))

    def test_shared_moves_preserve_counts_and_move_all_functions(self):
        from local_search import local_search
        policy = np.array([[7, .25, .25, .25, .25], [8, .25, .25, .25, .25]])
        candidate = CASA.local_search(policy, .05, random.Random(0))
        self.assertIs(CASA.local_search, local_search)
        np.testing.assert_allclose(candidate[:, 1:].sum(axis=1), 1)
        np.testing.assert_allclose(candidate[0, 1:], candidate[1, 1:])
        np.testing.assert_array_equal(candidate[:, 0], policy[:, 0])
        self.assertFalse(np.array_equal(policy, candidate))
        np.testing.assert_array_equal(policy[:, 1:], np.full((2, 4), .25))

    def test_evaluation_is_repeatable_and_restores_random_state(self):
        lut = [[[1, 1, 0, 0, 0], [2, 2, 0, 0, 0]]]
        counts = np.array([[20]])
        args = (lut, counts, np.zeros((3, 4, 900)), np.ones(4)/4,
                np.array([[0, .25, .25, .25, .25]]), 0, 20, 100, 3)
        random.seed(11)
        np.random.seed(12)
        before = random.getstate(), np.random.get_state()
        first, losses = CASA.evaluate_policy(*args)
        second, _ = CASA.evaluate_policy(*args)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(before[0], random.getstate())
        np.testing.assert_array_equal(before[1][1], np.random.get_state()[1])
        self.assertAlmostEqual(losses.sum()/20*100, first[0])

    def test_stagnation_continues_until_deadline_and_saved_pair_matches_return(self):
        clock = [0.]
        def evaluate(*args):
            clock[0] += 1
            return np.array([1., 2, 3, 4]), np.array([1.])
        with tempfile.TemporaryDirectory() as directory, patch.object(
                CASA, 'evaluate_policy', side_effect=evaluate), patch.object(
                CASA.time, 'monotonic', side_effect=lambda: clock[0]):
            result = CASA.casa(directory, [], np.array([[10]]), np.zeros((3, 4, 900)),
                np.empty((0, 5)), 100, 0, np.ones(4)/4, 20, decision_time=20, failure_limit=2)
            meta = json.loads((Path(directory)/'metadata.json').read_text())
            self.assertGreater(meta['iterations'], 2)
            self.assertEqual(meta['elapsed_seconds'], 20)
            self.assertEqual(meta['evaluations'], 20)
            self.assertEqual(meta['termination_reason'], 'time_budget')
            last = meta['iterations']
            stored = json.loads((Path(directory)/f'design_e{last}.txt').read_text())
            np.testing.assert_array_equal(stored, result)
            np.testing.assert_array_equal(np.loadtxt(Path(directory)/f'value_e{last}.txt', delimiter=','), [1, 2, 3, 4])

    def test_feasible_incumbent_survives_carbon_step_across_target(self):
        baseline = np.array([[0, .25, .25, .25, .25]])
        shifted = np.array([[0, .2, .3, .25, .25]])
        clock = [0.]
        def evaluate(*args):
            clock[0] += 1
            return (np.array([1., 1, 2, 1]) if np.array_equal(args[4], baseline)
                    else np.array([3., 1, 1, 1])), np.array([1.])
        with tempfile.TemporaryDirectory() as directory, patch.object(CASA, 'evaluate_policy', side_effect=evaluate), \
                patch.object(CASA, 'local_search', return_value=shifted), \
                patch.object(CASA.time, 'monotonic', side_effect=lambda: clock[0]):
            result = CASA.casa(directory, [], np.array([[10]]), np.zeros((3, 4, 900)),
                baseline, 100, 0, np.ones(4)/4, 20, decision_time=20, failure_limit=1)
            np.testing.assert_array_equal(result, baseline)
            records = [json.loads(line) for line in (Path(directory)/'search.jsonl').read_text().splitlines()]
            self.assertEqual(records[0]['mode'], 'carbon')
            self.assertTrue(all(r['mode'] == 'utility' for r in records[1:]))
            self.assertGreater(len(records), 2)
            self.assertTrue(records[-1]['stagnant'])


if __name__ == '__main__':
    unittest.main()
