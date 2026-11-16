"""Common routing neighborhood used by all six optimization frameworks."""
import random

import numpy as np


def local_search(policy, step_size, rng=None):
    """Move a fixed share from one region to another for every function.

    Try at most 21 proposals using the supplied random-number stream.
    Feasibility and objective acceptance remain the caller's responsibility.
    """
    rng = random if rng is None else rng
    candidate = policy.copy()
    rows, columns = candidate.shape
    if not rows:
        return candidate
    available_rows = list(range(rows))
    for _ in range(21):
        source, destination = rng.sample(list(range(1, columns)), 2)
        rng.sample(available_rows, 1)  # Advance the row-selection random stream.
        if np.all(candidate[:, source] > step_size):
            candidate[:, source] -= step_size
            candidate[:, destination] += step_size
            break
    return candidate
