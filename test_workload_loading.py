"""Plain and bundled compressed workload loading use the same inputs."""

import gzip
import importlib
from pathlib import Path
import pickle
import tempfile
import unittest


load = importlib.import_module('15min_simulator').load_workload_pickle


class WorkloadLoadingTests(unittest.TestCase):
    def test_gzip_fallback_and_plain_file_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'workload.pkl'
            with gzip.open(str(path) + '.gz', 'wb') as stream:
                pickle.dump([[1, 2, 3]], stream)
            self.assertEqual(load(path), [[1, 2, 3]])
            with path.open('wb') as stream:
                pickle.dump([[4, 5, 6]], stream)
            self.assertEqual(load(path), [[4, 5, 6]])

    def test_missing_workload_has_setup_instructions(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, 'docs/WORKLOADS.md'):
                load(Path(directory) / 'missing.pkl')


if __name__ == '__main__':
    unittest.main()
