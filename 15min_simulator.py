"""Run one seeded, 15-minute routing experiment and save its raw objectives."""

import random
import copy
import numpy as np
import time
import pickle
import gzip
import math
import argparse
import os
import csv
import hvwfg
import json
import hashlib
import importlib
import tempfile
from pathlib import Path
import ENV_V3 as env
from randomness import seed_all


REQUEST_LEVEL_MULTIPLIERS = {1: 3330, 2: 6650, 3: 9980, 4: 13300, 5: 16630}


def load_workload_pickle(filename):
    """Load a trusted workload, accepting either plain or bundled gzip pickle."""
    path = Path(filename)
    if not path.is_file():
        path = Path(str(path) + '.gz')
    if not path.is_file():
        raise FileNotFoundError(
            f'Workload not found: {filename} or {filename}.gz. '
            'Run from the repository directory; see docs/WORKLOADS.md.')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as stream:
        return pickle.load(stream)


def create_run_directory(output_root, framework, seed):
    """Group by framework/seed; allocate a unique directory for every invocation."""
    parent = Path(output_root).resolve() / framework / f'seed{seed}'
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='run_', dir=parent))


def resolve_request_multiplier(request):
    """Map request levels 1-5 to scenario multipliers; larger values are literal."""
    if request < 1:
        raise ValueError('request level or multiplier must be positive')
    return REQUEST_LEVEL_MULTIPLIERS.get(request, request)


def scale_execution_durations(lut, multiplier):
    """Copy feature rows, scaling only integer execution seconds (field zero).

    ENV_V3 reserves at most 900 seconds per request. Reject longer durations
    instead of allowing its fixed resource buffer to truncate execution.
    """
    if not isinstance(multiplier, (int, np.integer)) or multiplier < 1:
        raise ValueError('duration multiplier must be a positive integer')
    scaled = []
    for func_id, variants in enumerate(lut):
        scaled_variants = []
        for features in variants:
            row = list(features)
            if len(row) != 5:
                raise ValueError(f'function {func_id}: expected five feature fields')
            seconds = row[0]
            if not np.isfinite(seconds) or seconds < 0 or seconds != int(seconds):
                raise ValueError(f'function {func_id}: execution duration must be nonnegative integer seconds')
            row[0] = int(seconds) * int(multiplier)
            if row[0] > 900:
                raise ValueError(f'function {func_id}: scaled duration {row[0]}s exceeds the 900s resource buffer')
            scaled_variants.append(row)
        scaled.append(scaled_variants)
    return scaled


def calculate_final_phv(objectives, reference_point):
    """Final-report hypervolume in raw units; all four objectives are minimized.

    Use one fixed reference for every framework. This is distinct from the
    frameworks' internally normalized search-history hypervolume.
    """
    points = np.asarray(objectives, dtype=float).reshape(-1, 4)
    reference = np.asarray(reference_point, dtype=float)
    # Infeasible policies and points outside the reference box contribute zero.
    points = points[np.isfinite(points).all(axis=1) & (points < reference).all(axis=1)]
    if not len(points):
        return 0.0
    return float(hvwfg.wfg(np.ascontiguousarray(points), reference))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--workload', type=str, help='workload type', default='compute', choices=['compute', 'io', 'synthetic', 'c2'])
    parser.add_argument('-l', '--laxity', type=int, help='deadline laxity', default=10)
    parser.add_argument('-dl', '--delay', type=int, help='delay factor', default=20)
    parser.add_argument('-s', '--slo', type=int, help='maximum percentage of each source region\'s requests per function that may be relocated (0-100)', default=100)
    parser.add_argument('-e', '--epoch', type=int, help='1-based interval to simulate (1-96); one 15-minute interval per process', default=96)
    parser.add_argument('-t', '--time', type=int, help='search budget in seconds; initialization and the final iteration may add overhead', default=180)
    parser.add_argument('-n', '--node', type=int, choices=[4], help='number of routing regions (fixed at 4)', default=4)
    parser.add_argument('-d', '--duration', type=int, help='positive execution-time multiplier; other request features are unchanged', default=1)
    parser.add_argument('-r', '--request', type=int,
                        help='request level: 1=3330, 2=6650, 3=9980, 4=13300, 5=16630; values above 5 are raw multipliers',
                        default=100)
    parser.add_argument('-o', '--origin', type=str, help='origin pattern of request', default='change', 
                        choices=['even', 'coast', 'centre', 'change'])
    parser.add_argument('-f', '--framework', type=str, help='framework', default='even', choices= 
                        ['even', 'coast', 'centre', 'change',
                         'SOFA', 'SOFA_DEA', 'SOFA_MLS', 'SOFA_RandomLS', 'CASA',
                         'kimchi', 'greenc', 'NSGA3', 'SWFAGA'])
    parser.add_argument('-st', '--step', type=int, help='step size', default=5)
    parser.add_argument('-p', '--population', type=int, help='maximum retained population for SOFA, NSGA3 and SWFAGA', default=70)
    parser.add_argument('-ff', '--fit_frequency', type=int, help='SOFA retraining interval in search iterations (default: 5)', default=5)
    parser.add_argument('-rs', '--random_search', type=int, help='number of random searches', default=15)
    parser.add_argument('--casa-utility-target', type=float, default=2.0,
                        help='CASA utility-loss target in percent; distinct from --slo relocation limit')
    parser.add_argument('--seed', '--casa-seed', dest='seed', type=int, default=0,
                        help='Python/NumPy and worker-task seed for every framework (default: 0)')
    parser.add_argument('--output-root', default='.',
                        help='root for <framework>/seed<N>/run_<unique>/ results (default: current directory)')

    args = parser.parse_args()
    if not 0 <= args.casa_utility_target <= 100:
        parser.error('--casa-utility-target must be between 0 and 100')
    if not 0 <= args.seed < 2**32:
        parser.error('--seed must be between 0 and 2**32-1')
    if not 1 <= args.epoch <= 96:
        parser.error('--epoch must be between 1 and 96 (1-based interval)')
    if args.time <= 0:
        parser.error('--time must be greater than zero')
    if args.fit_frequency <= 0:
        parser.error('--fit_frequency must be greater than zero')

    if args.population < 1:
        parser.error('--population must be positive')
    if args.duration < 1:
        parser.error('--duration must be positive')

    seed_all(args.seed)
    workload = args.workload
    ddl_laxity = args.laxity
    slo_constraint = args.slo
    number_of_epoch = args.epoch
    decision_time = args.time
    number_of_node = args.node
    time_of_duration = args.duration
    try:
        time_of_request = resolve_request_multiplier(args.request)
    except ValueError as error:
        parser.error(str(error))
    if args.request in REQUEST_LEVEL_MULTIPLIERS:
        print(f'Request level r{args.request}: workload multiplier {time_of_request}')
    framework = args.framework
    origin_pattern = args.origin
    delay_factor = args.delay
    step_size = args.step /100
    population_size = args.population
    fit_frequency = args.fit_frequency
    number_of_random_search = args.random_search
    print(number_of_node, time_of_duration, time_of_request, framework)

    a_day_workload_arrival_pattern_arr =[]
    if workload == 'compute':
        total_number_of_func_id = 487
        distribution_file="R5_compute_workload_arrival_pattern_arr.pkl"
        lut_file="R5_compute_workload_feature_lut_list.pkl"
    elif workload == 'io':
        total_number_of_func_id = 358
        distribution_file="io_workload_arrival_pattern_arr.pkl"
        lut_file="io_workload_feature_lut_list.pkl"
    elif workload == 'synthetic':
        total_number_of_func_id = 845
        distribution_file="synthetic_workload_arrival_pattern_arr.pkl"
        lut_file="synthetic_workload_feature_lut_list.pkl"
    else:
        exit()
    total_func_list=list(range(total_number_of_func_id))

    a_day_workload_arrival_pattern_arr = load_workload_pickle(distribution_file)
    if args.epoch > len(a_day_workload_arrival_pattern_arr):
        parser.error(f'--epoch {args.epoch} is unavailable: workload contains {len(a_day_workload_arrival_pattern_arr)} intervals')

    for epoch in range(number_of_epoch-1, number_of_epoch):
        request_intensity=0
        for func_id in range(total_number_of_func_id):
            request_intensity+=int(a_day_workload_arrival_pattern_arr[epoch][func_id,0])

    print("distribution load success")

    lut_list = []
    lut_list = load_workload_pickle(lut_file)
    try:
        lut_list = scale_execution_durations(lut_list, time_of_duration)
    except ValueError as error:
        parser.error(str(error))
    lut_arr = lut_list
    print("runtime load success")

    run_directory = create_run_directory(args.output_root, framework, args.seed)
    run_configuration = vars(args).copy()
    run_configuration.update(request_multiplier=time_of_request, nodes_per_region=env.number_of_node_per_region,
                             cores_per_node=env.CPU_CORES_PER_NODE, solar_factor=env.SOLAR_ENERGY_FACTOR,
                             power_floor_W=env.NODE_IDLE_POWER_W, power_ceiling_W=env.NODE_PEAK_POWER_W)
    source_directory = Path(__file__).resolve().parent
    run_configuration['source_sha256'] = {
        name: hashlib.sha256((source_directory / name).read_bytes()).hexdigest()
        for name in ('15min_simulator.py', 'ENV_V3.py', 'SOFA.py', 'NSGA3.py',
                     'SWFAGA.py', 'Kimchi.py', 'Greencourier.py', 'CASA.py',
                     'local_search.py', 'randomness.py', 'sofa_ablation.py',
                     'SOFA_DEA.py', 'SOFA_MLS.py', 'SOFA_RandomLS.py')}
    (run_directory / 'run_config.json').write_text(json.dumps(run_configuration, indent=2))
    print(f'Seed: {args.seed}; output directory: {run_directory}')
    os.chdir(run_directory)

    objective_arr = np.zeros((number_of_epoch,4))
    leftover_resource_time_arr = [np.zeros((number_of_node, 900))] * 3
    req_distribution_arr = np.zeros((total_number_of_func_id, number_of_node+1))


    if 'SOFA' not in framework:
        print('epoch', 'total workload', '[utility loss (%), cost ($), carbon (kg), water (L)]')
    else:
        print('epoch', 'phv', 'time')
    pareto_front_value_arr=np.zeros((20,4))
    very_beginning_time = time.time()
    for epoch in range(number_of_epoch-1,number_of_epoch): 
        the_start_time=time.time()
        if origin_pattern == 'even':
            original_req_distribution_arr = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
        if origin_pattern == 'coast':
            original_req_distribution_arr = np.array([0.4, 0.1, 0.1, 0.4], dtype=float)
        if origin_pattern == 'centre':
            original_req_distribution_arr = np.array([0.1, 0.4, 0.4, 0.1], dtype=float)
        if origin_pattern == 'change':
            original_req_distribution_arr = np.array([0.1, 0.1, 0.1, 0.1], dtype=float)
            a,b = divmod(epoch, 32)
            original_req_distribution_arr[a] = 0.7 - 0.01875*b
            original_req_distribution_arr[a+1] = 0.1 + 0.01875*b
        func_distribution_arr = time_of_request*a_day_workload_arrival_pattern_arr[epoch]
        total_number_of_request = np.sum(func_distribution_arr[:,0])
        print('Requests (millions):', total_number_of_request/1000000)
        reference_point = [100]*4

        s_time = time.time()
        previous_func_type = [] 
        if framework == 'change':
            from Change_Distribution import change_distribution
            new_req_distribution_arr = change_distribution(lut_arr, time_of_request*a_day_workload_arrival_pattern_arr[epoch], leftover_resource_time_arr, req_distribution_arr, original_req_distribution_arr)
        if framework == 'even':
            from Even_Distribution import even_distribution
            new_req_distribution_arr = even_distribution(lut_arr, time_of_request*a_day_workload_arrival_pattern_arr[epoch], leftover_resource_time_arr, req_distribution_arr)
        if framework == 'coast':
            from Coast_Distribution import coast_distribution
            new_req_distribution_arr = coast_distribution(lut_arr, time_of_request*a_day_workload_arrival_pattern_arr[epoch], leftover_resource_time_arr, req_distribution_arr)
        if framework == 'centre':
            from Centre_Distribution import centre_distribution
            new_req_distribution_arr = centre_distribution(lut_arr, time_of_request*a_day_workload_arrival_pattern_arr[epoch], leftover_resource_time_arr, req_distribution_arr)
        if framework == 'CASA':
            from CASA import casa, evaluate_policy
            dir_name = (f'CASA/{workload}_{origin_pattern}_t{decision_time}_st{int(step_size*100)}'
                        f'_n{number_of_node}_d{time_of_duration}_r{time_of_request}_e{number_of_epoch}'
                        f'_dl{ddl_laxity}_de{delay_factor}_slo{slo_constraint}'
                        f'_target{args.casa_utility_target}_seed{args.seed}_{time.time_ns()}/')
            new_req_distribution_arr = casa(
                dir_name, lut_list, func_distribution_arr, leftover_resource_time_arr,
                req_distribution_arr, slo_constraint, epoch, original_req_distribution_arr,
                delay_factor, step_size=step_size, decision_time=decision_time,
                utility_target=args.casa_utility_target, seed=args.seed)

        if framework == 'kimchi':
            dir_name = framework+'/'+workload+'_'+origin_pattern+'_t'+str(decision_time)+'_st'+str(int(step_size*100))+'_n'+str(number_of_node)+'_d'+str(time_of_duration)+'_r'+str(time_of_request)+'_e'+str(number_of_epoch)+'_dl'+str(ddl_laxity)+'_de'+str(delay_factor)+'_slo'+str(slo_constraint)+'/'
            if not os.path.isdir(framework+'/'):
                os.mkdir(framework+'/')
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)
            else:
                for filename in os.listdir(dir_name):
                    file_path = os.path.join(dir_name, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"Deleted file: {filename}")

            from Kimchi import *
            new_req_distribution_arr = kimchi(dir_name, lut_arr, time_of_request*a_day_workload_arrival_pattern_arr[epoch], leftover_resource_time_arr, req_distribution_arr, slo_constraint, epoch, original_req_distribution_arr, delay_factor, step_size, decision_time=decision_time, seed=args.seed)

        if framework == 'greenc':
            dir_name = framework+'/'+workload+'_'+origin_pattern+'_t'+str(decision_time)+'_st'+str(int(step_size*100))+'_n'+str(number_of_node)+'_d'+str(time_of_duration)+'_r'+str(time_of_request)+'_e'+str(number_of_epoch)+'_dl'+str(ddl_laxity)+'_de'+str(delay_factor)+'_slo'+str(slo_constraint)+'/'
            if not os.path.isdir(framework+'/'):
                os.mkdir(framework+'/')
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)
            else:
                for filename in os.listdir(dir_name):
                    file_path = os.path.join(dir_name, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"Deleted file: {filename}")

            from Greencourier import *
            new_req_distribution_arr = greencourier(dir_name, lut_arr, time_of_request*a_day_workload_arrival_pattern_arr[epoch], leftover_resource_time_arr, req_distribution_arr, slo_constraint, epoch, original_req_distribution_arr, delay_factor, step_size, decision_time=decision_time, seed=args.seed)

        if framework == 'SWFAGA':
            dir_name = framework+'/'+workload+'_'+origin_pattern+'_t'+str(decision_time)+'_st'+str(int(step_size*100))+'_n'+str(number_of_node)+'_d'+str(time_of_duration)+'_r'+str(time_of_request)+'_e'+str(number_of_epoch)+'_dl'+str(ddl_laxity)+'_de'+str(delay_factor)+'_slo'+str(slo_constraint)+'/'
            if not os.path.isdir(framework+'/'):
                os.mkdir(framework+'/')
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)
            else:
                for filename in os.listdir(dir_name):
                    file_path = os.path.join(dir_name, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"Deleted file: {filename}")

            from SWFAGA import *
            start_time=time.time()
            print('loading time', start_time-the_start_time)
            pareto_front_design, pareto_front_value, standard_objective = swfaga(dir_name, lut_list, func_distribution_arr, leftover_resource_time_arr, req_distribution_arr, epoch, original_req_distribution_arr, delay_factor, slo_constraint, step_size, number_of_random_search, decision_time=decision_time, population_size=population_size, seed=args.seed)
            pareto_front_value_arr=np.zeros((len(pareto_front_value),4))
            for j in range(len(pareto_front_value)):
                pareto_front_value_arr[j,0]+=pareto_front_value[j][0]
                pareto_front_value_arr[j,1]+=pareto_front_value[j][1]
                pareto_front_value_arr[j,2]+=pareto_front_value[j][2]
                pareto_front_value_arr[j,3]+=pareto_front_value[j][3]


        if framework == 'NSGA3':
            dir_name = framework+'/'+workload+'_'+origin_pattern+'_t'+str(decision_time)+'_st'+str(int(step_size*100))+'_n'+str(number_of_node)+'_d'+str(time_of_duration)+'_r'+str(time_of_request)+'_e'+str(number_of_epoch)+'_dl'+str(ddl_laxity)+'_de'+str(delay_factor)+'_slo'+str(slo_constraint)+'/'
            if not os.path.isdir(framework+'/'):
                os.mkdir(framework+'/')
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)
            else:
                for filename in os.listdir(dir_name):
                    file_path = os.path.join(dir_name, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"Deleted file: {filename}")

            from NSGA3 import *
            start_time=time.time()
            print('loading time', start_time-the_start_time)
            pareto_front_design, pareto_front_value, standard_objective = nsga(dir_name, lut_list, func_distribution_arr, leftover_resource_time_arr, req_distribution_arr, epoch, original_req_distribution_arr, delay_factor, slo_constraint, step_size, number_of_random_search, decision_time=decision_time, population_size=population_size, seed=args.seed)
            pareto_front_value_arr=np.zeros((len(pareto_front_value),4))
            for j in range(len(pareto_front_value)):
                pareto_front_value_arr[j,0]+=pareto_front_value[j][0]
                pareto_front_value_arr[j,1]+=pareto_front_value[j][1]
                pareto_front_value_arr[j,2]+=pareto_front_value[j][2]
                pareto_front_value_arr[j,3]+=pareto_front_value[j][3]

        if framework in ('SOFA', 'SOFA_DEA', 'SOFA_MLS', 'SOFA_RandomLS'):
            dir_name = framework+'/'+workload+'_'+origin_pattern+'_t'+str(decision_time)+'_st'+str(int(step_size*100))+'_n'+str(number_of_node)+'_d'+str(time_of_duration)+'_r'+str(time_of_request)+'_e'+str(number_of_epoch)+'_dl'+str(ddl_laxity)+'_de'+str(delay_factor)+'_slo'+str(slo_constraint)+'_random'+str(number_of_random_search)+'_fit'+str(fit_frequency)+'_pop'+str(population_size)+'/'
            if not os.path.isdir(framework+'/'):
                os.mkdir(framework+'/')
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)
            else:
                for filename in os.listdir(dir_name):
                    file_path = os.path.join(dir_name, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"Deleted file: {filename}")

            from SOFA import *
            if framework != 'SOFA':
                sofa_parallel = importlib.import_module(framework).sofa_parallel
            start_time=time.time()
            print('loading time', start_time-the_start_time)
            pareto_front_design, pareto_front_value, standard_objective = sofa_parallel(dir_name,lut_arr, time_of_request*a_day_workload_arrival_pattern_arr[epoch], leftover_resource_time_arr, req_distribution_arr, epoch, original_req_distribution_arr, delay_factor, slo_constraint, step_size, number_of_random_search, fit_frequency, population_size, decision_time=decision_time, seed=args.seed)
            pareto_front_value_arr=np.zeros((len(pareto_front_value),4))
            for j in range(len(pareto_front_value)):
                pareto_front_value_arr[j,0]+=pareto_front_value[j][0]
                pareto_front_value_arr[j,1]+=pareto_front_value[j][1]
                pareto_front_value_arr[j,2]+=pareto_front_value[j][2]
                pareto_front_value_arr[j,3]+=pareto_front_value[j][3]

        if 'SOFA' not in framework and 'NSGA3' not in framework and 'SWFAGA' not in framework:
            seed_all(args.seed)
            if framework == 'CASA':
                epoch_objectives, _ = evaluate_policy(
                    lut_list, func_distribution_arr, leftover_resource_time_arr,
                    original_req_distribution_arr, new_req_distribution_arr, epoch,
                    delay_factor, slo_constraint, args.seed)
            else:
                epoch_objectives, leftover_resource_time_arr = env.simulation(lut_arr, time_of_request*a_day_workload_arrival_pattern_arr[epoch], leftover_resource_time_arr,
                                                                        original_req_distribution_arr, new_req_distribution_arr, epoch, delay_factor, slo_constraint)


            objective_arr[epoch] = epoch_objectives
            phv_value = calculate_final_phv([epoch_objectives], reference_point)
            pass
        else:
            phv_value = calculate_final_phv(pareto_front_value, reference_point)
            pass
        print('end of epoch', epoch, time.time()-very_beginning_time)


    print("utility loss (%), cost ($), carbon (kg), water (L)")
    if 'SOFA' not in framework and 'NSGA3' not in framework and 'SWFAGA' not in framework:
        ave_violation_rate = np.sum(objective_arr[:,0])
        cumulative_cost = np.sum(objective_arr[:,1])
        cumulative_carbon = np.sum(objective_arr[:,2])
        cumulative_water = np.sum(objective_arr[:,3])
        print(ave_violation_rate, cumulative_cost, cumulative_carbon, cumulative_water)
    else:
        pass
        for j in range(len(pareto_front_value)):
            ave_violation_rate = pareto_front_value_arr[j,0]
            cumulative_cost = pareto_front_value_arr[j,1]
            cumulative_carbon = pareto_front_value_arr[j,2]
            cumulative_water = pareto_front_value_arr[j,3]
            print(ave_violation_rate, cumulative_cost, cumulative_carbon, cumulative_water)

    if not os.path.exists('step_simples'):
        os.makedirs('step_simples')
    if 'SOFA' not in framework and 'NSGA3' not in framework and 'SWFAGA' not in framework:
        with open('step_simples/s'+str(int(step_size*100))+'_'+workload+'_'+framework+'_n'+str(number_of_node)+'_d'+str(time_of_duration)+'_r'+str(time_of_request)+'_e'+str(number_of_epoch)+'_'+origin_pattern, 'w') as f:
            f.writelines(str(ave_violation_rate) + ',' + str(cumulative_cost) + ',' + str(cumulative_carbon) + ',' + str(cumulative_water))
    else:
        np.savetxt('step_simples/s'+str(int(step_size*100))+'_'+workload+'_'+framework+'_n'+str(number_of_node)+'_d'+str(time_of_duration)+'_r'+str(time_of_request)+'_e'+str(number_of_epoch)+'_'+origin_pattern, pareto_front_value, delimiter=',')
        pass


    final_values = (np.asarray(pareto_front_value, dtype=float).reshape(-1, 4)
                    if framework in ('SOFA', 'SOFA_DEA', 'SOFA_MLS', 'SOFA_RandomLS', 'NSGA3', 'SWFAGA')
                    else np.asarray([epoch_objectives], dtype=float))
    if not np.isfinite(final_values).all():
        raise ValueError('Final objectives contain nonfinite values; run is not complete')
    summary = dict(status='completed', framework=framework, seed=args.seed,
                   configuration=run_configuration, objectives=final_values.tolist(),
                   search_directory=dir_name if framework in ('SOFA', 'SOFA_DEA', 'SOFA_MLS', 'SOFA_RandomLS', 'NSGA3', 'SWFAGA', 'kimchi', 'greenc', 'CASA') else None,
                   note='Time-budgeted runs can complete different iteration counts on different hardware.')
    (run_directory / 'run_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
