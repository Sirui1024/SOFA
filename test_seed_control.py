import importlib
import json
import random
import tempfile
import unittest
import itertools
import io
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np
from randomness import SeededPool, seed_all
from summarize_seed_runs import FRAMEWORKS, statistics, summarize


def random_task(index):
    return index, random.random(), float(np.random.random())


class SeedTests(unittest.TestCase):
    def test_results_are_separated_by_framework_seed_and_invocation(self):
        create = importlib.import_module('15min_simulator').create_run_directory
        with tempfile.TemporaryDirectory() as directory:
            first = create(directory, 'SOFA', 1)
            (first/'keep.txt').write_text('existing result')
            repeated = create(directory, 'SOFA', 1)
            other_seed = create(directory, 'SOFA', 2)
            other_framework = create(directory, 'CASA', 1)
            root = Path(directory).resolve()
            self.assertEqual(first.parent, root/'SOFA'/'seed1')
            self.assertEqual(repeated.parent, first.parent)
            self.assertEqual(other_seed.parent, root/'SOFA'/'seed2')
            self.assertEqual(other_framework.parent, root/'CASA'/'seed1')
            self.assertEqual(len({first, repeated, other_seed, other_framework}), 4)
            self.assertEqual((first/'keep.txt').read_text(), 'existing result')

    def test_task_streams_do_not_depend_on_worker_count(self):
        with SeededPool(processes=1, seed=42) as pool:
            first = pool.map(random_task, range(8))
            subsequent = pool.map(random_task, range(8))
        with SeededPool(processes=2, seed=42) as pool:
            self.assertEqual(first, pool.map(random_task, range(8), chunksize=1))
            self.assertEqual(subsequent, pool.map(random_task, range(8), chunksize=3))
        with SeededPool(processes=1, seed=43) as pool:
            self.assertNotEqual(first, pool.map(random_task, range(8)))
        self.assertNotEqual(first, subsequent)

    def test_frameworks_accept_shared_seed(self):
        import inspect
        for module, function in [('SOFA','sofa_parallel'), ('NSGA3','nsga'),
                                 ('SWFAGA','swfaga'), ('Kimchi','kimchi'),
                                 ('Greencourier','greencourier'), ('CASA','casa')]:
            self.assertIn('seed', inspect.signature(getattr(importlib.import_module(module), function)).parameters)

    def test_main_rngs_repeat(self):
        seed_all(15)
        first = random_task(0)
        seed_all(15)
        self.assertEqual(first, random_task(0))

    def test_frameworks_return_when_setup_exhausts_budget(self):
        entries = [('SOFA','sofa_parallel'), ('NSGA3','nsga'), ('SWFAGA','swfaga'),
                   ('Kimchi','kimchi'), ('Greencourier','greencourier')]
        for name, function in entries:
            module = importlib.import_module(name)
            with self.subTest(framework=name), tempfile.TemporaryDirectory() as directory, \
                    patch.object(module, 'simulation', return_value=([1., 2., 3., 4.], np.zeros((3,4,900)))), \
                    patch.object(module, 'Pool', return_value=MagicMock()) as pool, \
                    patch.object(module.time, 'time', side_effect=itertools.count(0, 1000)), \
                    redirect_stdout(io.StringIO()):
                counts = np.ones((1,16)); counts[0,0] = 15
                result = getattr(module, function)(
                    dir_name=directory+'/', lut_list=[[[1,1,0,0,0]]],
                    func_distribution_arr=counts, leftover_resource_time_arr=np.zeros((3,4,900)),
                    req_distribution_arr=np.array([[0,.25,.25,.25,.25]]),
                    epoch=0, original_req_distribution_arr=np.full(4,.25), delay_factor=20,
                    slo_constraint=100, decision_time=.01, seed=1)
                self.assertIsNotNone(result)
                pool.return_value.__enter__.return_value.map.assert_not_called()

    def test_confidence_interval_is_not_min_max(self):
        result = statistics([1, 2, 3, 4, 5])
        self.assertEqual(result['mean'], 3)
        self.assertEqual((result['min'], result['max']), (1, 5))
        self.assertAlmostEqual(result['ci_low'], 1.0367568385)
        self.assertAlmostEqual(result['ci_high'], 4.9632431615)
        self.assertIsNone(statistics([1])['ci_low'])

    def test_requires_ten_distinct_seeds_and_consistent_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in FRAMEWORKS:
                for seed in range(1,11):
                    folder = root/f'{name}_{seed}'
                    folder.mkdir()
                    record = dict(status='completed', framework=name, seed=seed,
                                  configuration=dict(framework=name, seed=seed, request=3),
                                  objectives=[[1+seed*.01, 2, 3, 4], [2, 1, 4, 3]])
                    (folder/'run_summary.json').write_text(json.dumps(record))
            report = summarize(root, list(range(1,11)), [20, 10, 10, 10])
            self.assertEqual(len(report['per_run']), 60)
            self.assertTrue(all(row['n'] == 10 for row in report['statistics']))
            target = root/'SOFA_1'/'run_summary.json'
            data = target.read_text()
            target.unlink()
            with self.assertRaisesRegex(ValueError, 'Missing runs'):
                summarize(root, list(range(1,11)))
            target.write_text(data)
            duplicate = root/'duplicate'
            duplicate.mkdir()
            (duplicate/'run_summary.json').write_text(data)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                summarize(root, list(range(1,11)))

    def test_same_framework_seed_in_different_experiments_is_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for request in (3,4):
                folder = root/str(request)
                folder.mkdir()
                record = dict(status='completed', framework='SOFA', seed=1,
                              configuration=dict(framework='SOFA', seed=1, request=request),
                              objectives=[[1, request, 3, 4]])
                (folder/'run_summary.json').write_text(json.dumps(record))
            for request in (3,4):
                report = summarize(root, list(range(1,11)), [20,10,10,10],
                                   allow_partial=True, configuration_filter={'request':request})
                self.assertEqual(len(report['per_run']), 1)
                self.assertEqual(report['per_run'][0]['cost_USD'], request)


if __name__ == '__main__':
    unittest.main()
