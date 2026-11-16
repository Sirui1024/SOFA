"""CASA utility- and carbon-aware search for four-region workload routing."""
import json
import random
import time
from pathlib import Path

import hvwfg
import numpy as np
import ENV_V3 as env
from local_search import local_search


def search_mode(values, utility_target):
    return 'utility' if values[0] > utility_target else 'carbon'


def improves(candidate, current, mode):
    column = 0 if mode == 'utility' else 2
    return np.isfinite(candidate).all() and candidate[column] < current[column] - 1e-12


def evaluate_policy(lut, counts, leftover, origin, policy, epoch, delay, relocation, seed):
    """Use a repeatable simulation seed without changing the search RNG state."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        values, _, details = env.simulation(
            lut, counts, leftover, origin, policy, epoch, delay, relocation,
            return_details=True)
        return np.asarray(values, dtype=float), details['utility_loss_by_function']
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def casa(dir_name, lut_list, func_distribution_arr, leftover_resource_time_arr,
         req_distribution_arr, slo_constraint, epoch, original_req_distribution_arr,
         delay_factor, step_size=0.05, decision_time=900, utility_target=2.0,
         failure_limit=5, seed=0):
    """Return one policy: lowest-carbon feasible visited policy, else lowest loss.

    Carbon moves may cross the utility target to enable optimizer switching.
    A feasible incumbent is retained separately for the final single solution.
    Unsuccessful rounds are diagnostic only: search continues to the deadline.
    failure_limit marks stagnation in the log; it does not terminate search.
    seed is supplied by the simulator's shared --seed option and controls both
    the private search RNG and repeatable policy evaluation.
    """
    if not 0 <= utility_target <= 100 or not np.isfinite(utility_target):
        raise ValueError('CASA utility target must be in [0, 100]')
    if not 0 < step_size <= 1 or decision_time <= 0 or failure_limit < 1:
        raise ValueError('positive search budget, step in (0, 1], and failure limit required')
    origin = np.asarray(original_req_distribution_arr, dtype=float)
    active = np.flatnonzero(func_distribution_arr[:, 0])
    policy = np.column_stack((active, np.tile(origin, (len(active), 1))))
    # Reuse a complete, feasible supplied policy when available.
    supplied = np.asarray(req_distribution_arr)
    if (supplied.shape == policy.shape and
            np.array_equal(np.sort(supplied[:, 0]), active) and
            env.check_slo_violation(origin, supplied, slo_constraint)):
        policy = supplied[np.argsort(supplied[:, 0])].copy()
    out = Path(dir_name)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    start = time.monotonic()
    deadline = start + decision_time

    def evaluate(design):
        return evaluate_policy(lut_list, func_distribution_arr, leftover_resource_time_arr,
                               origin, design, epoch, delay_factor, slo_constraint, seed)

    values, losses = evaluate(policy)
    if not np.isfinite(values).all():
        raise ValueError('CASA initial policy has nonfinite objectives')
    scale = np.maximum(2 * values, 1e-12)
    scale[0] = 100
    reference = np.array([50., 100., 100., 100.])
    best_policy, best_values = policy.copy(), values.copy()

    def rank(v):
        return (0, v[2], v[0]) if v[0] <= utility_target else (1, v[0], v[2])

    def save(index):
        np.savetxt(out / f'value_e{index}.txt', [best_values], delimiter=',')
        normalized = best_values / scale * 100
        np.savetxt(out / f'true_value_e{index}.txt', [normalized], delimiter=',')
        (out / f'design_e{index}.txt').write_text(repr(best_policy.tolist())+'\n')
        phv = float(hvwfg.wfg(np.array([normalized]), reference)) if np.all(normalized < reference) else 0.
        with (out / 'phv.txt').open('a') as stream:
            stream.write(f'{index},{phv},{time.monotonic()-start}\n')

    (out / 'phv.txt').write_text('')
    (out / 'search.jsonl').write_text('')
    save(0)
    failures = dict(utility=0, carbon=0)
    iteration = 0
    evaluations = 1
    while len(active) and time.monotonic() < deadline:
        mode = search_mode(values, utility_target)
        accepted = False
        for _ in range(5):
            if time.monotonic() >= deadline:
                break
            candidate = local_search(policy, step_size, rng)
            if not env.check_slo_violation(origin, candidate, slo_constraint):
                continue
            candidate_values, candidate_losses = evaluate(candidate)
            evaluations += 1
            if rank(candidate_values) < rank(best_values) and np.isfinite(candidate_values).all():
                best_policy, best_values = candidate.copy(), candidate_values.copy()
            if improves(candidate_values, values, mode):
                policy, values, losses = candidate, candidate_values, candidate_losses
                accepted = True
                failures[mode] = 0
        if not accepted:
            failures[mode] += 1
        iteration += 1
        with (out / 'search.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(iteration=iteration, mode=mode,
                accepted=accepted, consecutive_failures=int(failures[mode]),
                stagnant=bool(failures[mode] >= failure_limit), current=values.tolist()))+'\n')
        save(iteration)
    (out / 'metadata.json').write_text(json.dumps(dict(
        implementation='CASA regional-routing adaptation', local_search='shared all-function routing',
        trials_per_round=5, utility_target=utility_target,
        relocation_limit=slo_constraint, failure_limit=failure_limit, seed=seed,
        decision_time=decision_time, elapsed_seconds=time.monotonic()-start,
        termination_reason='time_budget' if len(active) else 'no_active_functions',
        iterations=iteration, evaluations=evaluations, normalization_scale=scale.tolist(),
        reference_point=reference.tolist(), final_objectives=best_values.tolist(),
        target_met=bool(best_values[0] <= utility_target)), indent=2))
    print('CASA final objectives:', best_values.tolist(), 'evaluations:', evaluations)
    return best_policy
