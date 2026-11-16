"""Reproducible Python/NumPy streams, independent of worker scheduling."""
import multiprocessing
import random
import numpy as np


def seed_all(seed):
    if seed is not None:
        if not 0 <= seed < 2**32:
            raise ValueError('seed must be in [0, 2**32-1]')
        random.seed(int(seed))
        np.random.seed(int(seed))


def task_seed(seed, index):
    return int(np.random.SeedSequence([int(seed), 1, index]).generate_state(1)[0])


def _seeded_call(payload):
    function, argument, seed = payload
    seed_all(seed)
    return function(argument)


class SeededPool:
    """Ordered map with a stable seed per task, not per operating-system PID."""
    def __init__(self, *args, seed=None, **kwargs):
        self.seed = seed
        self.next_index = 0
        self.pool = multiprocessing.Pool(*args, **kwargs)

    def map(self, function, iterable, chunksize=None):
        if self.seed is None:
            return self.pool.map(function, iterable, chunksize)
        items = list(iterable)
        payloads = [(function, item, task_seed(self.seed, self.next_index+i))
                    for i, item in enumerate(items)]
        self.next_index += len(items)
        return self.pool.map(_seeded_call, payloads, chunksize)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.pool.__exit__(*args)
