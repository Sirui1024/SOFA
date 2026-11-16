"""End-to-end CLI checks for fixed-routing policies."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import pickle
import runpy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import ENV_V3 as env


class FixedRoutingTests(unittest.TestCase):
    def test_each_fixed_policy_is_evaluated_by_current_environment(self):
        script = Path(__file__).resolve().with_name('15min_simulator.py')
        policies = {
            'even': [.25, .25, .25, .25],
            'coast': [.4, .1, .1, .4],
            'centre': [.1, .4, .4, .1],
            'change': [.7, .1, .1, .1],
        }
        original_cwd = Path.cwd()
        for framework, expected in policies.items():
            with self.subTest(framework=framework), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                arrivals = np.zeros((1, 845, 2), dtype=int)
                arrivals[0, 0] = [1, 1]
                with (root / 'synthetic_workload_arrival_pattern_arr.pkl').open('wb') as stream:
                    pickle.dump(arrivals, stream)
                with (root / 'synthetic_workload_feature_lut_list.pkl').open('wb') as stream:
                    pickle.dump([[[1, 1, 1, 1, 1]]] * 845, stream)
                argv = [str(script), '-w', 'synthetic', '-f', framework,
                        '-e', '1', '-r', '6', '--seed', '2',
                        '--output-root', str(root / 'results')]
                try:
                    os.chdir(root)
                    with patch('sys.argv', argv), redirect_stdout(io.StringIO()), \
                            patch.object(env, 'simulation', return_value=(
                                [1., 2., 3., 4.], np.zeros((3, 4, 900)))) as evaluate:
                        runpy.run_path(str(script), run_name='__main__')
                    evaluate.assert_called_once()
                    policy = evaluate.call_args.args[4]
                    np.testing.assert_allclose(policy, [[0, *expected]])
                    summaries = list((root / 'results').rglob('run_summary.json'))
                    self.assertEqual(len(summaries), 1)
                    summary = json.loads(summaries[0].read_text())
                    self.assertEqual(summary['status'], 'completed')
                    self.assertEqual(summary['objectives'], [[1., 2., 3., 4.]])
                finally:
                    os.chdir(original_cwd)


if __name__ == '__main__':
    unittest.main()
