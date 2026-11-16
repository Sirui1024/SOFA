"""Component-only SOFA searches sharing its operators, initialization and pruning."""
import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import hvwfg
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
import SOFA as core
from randomness import SeededPool, seed_all


def search_step(mode, values, designs, step_size, pool, brain, history, gains):
    """DEA does not call local search; MLS/random-LS do not generate offspring."""
    if mode == 'dea':
        members = list(range(len(designs)))
        children = []
        for _ in range(min(12, len(members)//2)):
            first = members.pop(random.randrange(len(members)))
            second = members.pop(random.randrange(len(members)))
            # Native crossover edits its input arrays; protect evaluated parents.
            children.extend(core.GA(designs[first].copy(), designs[second].copy(), step_size))
        results = pool.map(core.parallel_simulation, children)
        for child, (objectives, _) in zip(children, results):
            if np.isfinite(objectives).all():
                designs.append(copy.deepcopy(child)); values.append(copy.deepcopy(objectives))
    else:
        if mode == 'mls' and hasattr(brain, 'estimators_'):
            selected, _ = core.smart_brain_decision(designs, brain)
        else:
            selected = random.sample(list(range(len(designs))), min(12, len(designs)))
        proposals = [(values[i], designs[i]) for i in selected]
        results = pool.map(core.perturb_and_simulate_10time_p, proposals)
        for i, result in zip(selected, results):
            if result[1]:
                if mode == 'mls':
                    history.append(designs[i].flatten().copy()); gains.append(result[3])
                for objectives, design in zip(result[1], result[2]):
                    if np.isfinite(objectives).all():
                        values.append(copy.deepcopy(objectives)); designs.append(copy.deepcopy(design))
    return values, designs


def run_ablation(dir_name, lut_list, func_distribution_arr, leftover_resource_time_arr,
                 req_distribution_arr, epoch, original_req_distribution_arr, delay_factor,
                 slo_constraint, step_size=.05, number_of_random_search=20, fit_frequency=2,
                 population_size=70, decision_time=900, seed=None, *, mode):
    if mode not in ('dea','mls','random_ls'):
        raise ValueError('Unknown SOFA ablation mode')
    if population_size < 1 or fit_frequency < 1 or decision_time <= 0:
        raise ValueError('Population, fit frequency and search budget must be positive')
    seed_all(seed)
    out = Path(dir_name); out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    counts = func_distribution_arr
    if counts.shape[1] < 901:
        counts = core.extend_func_distribution(counts, counts.shape[1]-1, 900)
    active = sorted(np.flatnonzero(counts[:,0]), key=lambda i: -counts[i,0])
    designs, values = [], []
    initial = [np.full(4,.25)] + [np.array([.7 if j==i else .1 for j in range(4)]) for i in range(4)]
    for shares in initial:
        design = np.column_stack((active, np.tile(shares, (len(active),1))))
        design = core.feasible_initial_policy(design, original_req_distribution_arr, slo_constraint)
        objectives, _ = core.simulation(lut_list, counts, leftover_resource_time_arr,
                                       original_req_distribution_arr, design, epoch, delay_factor, slo_constraint)
        if np.isfinite(objectives).all():
            designs.append(design); values.append(objectives)
    if not values:
        design = np.column_stack((active, np.tile(original_req_distribution_arr, (len(active),1))))
        objectives, _ = core.simulation(lut_list, counts, leftover_resource_time_arr,
                                       original_req_distribution_arr, design, epoch, delay_factor, slo_constraint)
        if not np.isfinite(objectives).all():
            raise ValueError('No finite initial ablation policy')
        designs.append(design); values.append(objectives)
    standard = copy.deepcopy(values[0])
    upper = np.asarray(standard)*2; upper[0]=100
    scaler=MinMaxScaler((0,100)).fit([np.zeros(4),upper])
    reference=np.array([50.,100.,100.,100.])
    values, designs=core.phv_based_pruning(values,designs,scaler,reference,threshold=population_size)
    config=dict(step_size=step_size,lut_list=lut_list,func_distribution_arr=counts,
                leftover_resource_time_arr=leftover_resource_time_arr,original_req_distribution_arr=original_req_distribution_arr,
                epoch=epoch,scaler=scaler,delay_factor=delay_factor,slo_constraint=slo_constraint)
    brain=GradientBoostingRegressor(); history=[]; gains=[]
    def save(index):
        np.savetxt(out/f'value_e{index}.txt',values,delimiter=',')
        normalized=np.asarray(scaler.transform(values),dtype=float)
        np.savetxt(out/f'true_value_e{index}.txt',normalized,delimiter=',')
        (out/f'design_e{index}.txt').write_text(''.join(repr(d.tolist())+'\n' for d in designs))
        inside=normalized[(normalized<reference).all(axis=1)]
        hv=float(hvwfg.wfg(np.ascontiguousarray(inside),reference)) if len(inside) else 0.
        with (out/'phv.txt').open('a') as stream:stream.write(f'{index},{hv},{time.monotonic()-start}\n')
    (out/'phv.txt').write_text(''); save(0)
    iteration=0
    # As in full SOFA, setup precedes the optimization budget.
    deadline=time.monotonic()+decision_time
    with SeededPool(processes=12,initializer=core.init_worker,initargs=(config,),seed=seed) as pool:
        while len(active) and time.monotonic()<deadline:
            if mode=='mls' and iteration>=number_of_random_search and (iteration-number_of_random_search)%fit_frequency==0:
                core.fit_search_model(brain,history,gains)
                if iteration>number_of_random_search:history=[];gains=[]
            values,designs=search_step(mode,values,designs,step_size,pool,brain,history,gains)
            values,designs=core.create_sparse_population(values,designs,scaler,population_threshold=population_size)
            values,designs=core.phv_based_pruning(values,designs,scaler,reference,threshold=population_size)
            iteration+=1; save(iteration)
    (out/'ablation_metadata.json').write_text(json.dumps(dict(mode=mode,seed=seed,iterations=iteration,
        dea_enabled=mode=='dea',mls_enabled=mode=='mls',local_search_enabled=mode!='dea',
        initialization='SOFA equal and 70%-regional policies',pruning='SOFA sparse and edge-aware PHV pruning'),indent=2))
    return designs,values,standard
