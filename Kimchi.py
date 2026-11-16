from ENV_V3 import *
from local_search import local_search as perturb
from randomness import SeededPool as Pool, seed_all
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans 
from sklearn.preprocessing import MinMaxScaler

global_min_weight = 0.0005

SIM_CONFIG = None


def init_worker(config):
    global SIM_CONFIG
    SIM_CONFIG = config

def kimchi(dir_name, lut_list, func_distribution_arr, leftover_resource_time_arr, req_distribution_arr, slo_constraint, epoch, original_req_distribution_arr, delay_factor, step_size=0.05, decision_time=900, seed=None):
    seed_all(seed)
    start_time=time.time()
    total_number_of_func_type = func_distribution_arr.shape[0]

    number_of_worker = 12

    prewarm_func_list = []
    func_invocation_list = []
    for func_id in range(total_number_of_func_type):
        if func_distribution_arr[func_id,0]!=0:
            prewarm_func_list.append(func_id)
            func_invocation_list.append(func_distribution_arr[func_id,0])

    scored_prewarm_func_list = []
    scored_invocation_count_list = []
    while len(func_invocation_list):
        max_invocation = max(func_invocation_list)
        max_invocation_index = func_invocation_list.index(max_invocation)
        scored_invocation_count_list.append(func_invocation_list.pop(max_invocation_index))
        scored_prewarm_func_list.append(prewarm_func_list.pop(max_invocation_index))

    epoch_number_of_func_type = len(scored_prewarm_func_list)

    new_req_distribution_arr = np.ones((epoch_number_of_func_type, 4+1), dtype=float)
    new_req_distribution_arr[:,1]=0.25
    new_req_distribution_arr[:,2]=0.25
    new_req_distribution_arr[:,3]=0.25
    new_req_distribution_arr[:,4]=0.25
    for i in range(epoch_number_of_func_type):
        new_req_distribution_arr[i,0]=scored_prewarm_func_list[i]

    new_req_distribution_arr = feasible_initial_policy(new_req_distribution_arr, original_req_distribution_arr, slo_constraint)
    new_objectives, new_leftover_resource_time_arr = simulation(lut_list, func_distribution_arr, leftover_resource_time_arr, 
                                                                        original_req_distribution_arr, new_req_distribution_arr, epoch, delay_factor, slo_constraint)
    if not np.isfinite(new_objectives).all():
        raise ValueError('Initial feasible policy returned nonfinite objectives')

    standard_objective = copy.deepcopy(new_objectives)
    reference_point = [100]*4
    print(standard_objective)

    min_objs=[element * 0.1 for element in standard_objective]
    max_objs=[element * 2.0 for element in standard_objective]

    total_number_of_request = np.sum(func_distribution_arr[:,0])
    min_objs = [0] * 4
    max_objs[0] = 100
    print(min_objs)
    print(max_objs)
    scaler = MinMaxScaler((0, 100))
    reference_point=[100]*4
    reference_point[0] = 50
    print(scaler.fit([min_objs, max_objs]))

    value_arr = [copy.deepcopy(new_objectives)] * 1
    design_arr = [copy.deepcopy(new_req_distribution_arr)] * 1

    search_epoch_time = time.time()
    search_epoch = 0
    num_func_type = new_req_distribution_arr.shape[0]

    config = {"step_size": step_size, "lut_list": lut_list, "func_distribution_arr": func_distribution_arr, 
              "leftover_resource_time_arr": leftover_resource_time_arr, "original_req_distribution_arr": original_req_distribution_arr, "epoch": epoch, 
              "scaler": scaler, "delay_factor": delay_factor, "slo_constraint": slo_constraint}

    # A short budget may expire during setup before the first iteration.
    initial_points = np.asarray(scaler.transform(value_arr), dtype=float)
    initial_reference = np.asarray(reference_point, dtype=float)
    initial_points = initial_points[np.isfinite(initial_points).all(axis=1) &
                                    (initial_points < initial_reference).all(axis=1)]
    phv_value = (float(hvwfg.wfg(np.ascontiguousarray(initial_points), initial_reference))
                 if len(initial_points) else 0.0)
    with Pool(processes=number_of_worker, initializer=init_worker, initargs=(config,), seed=seed) as pool:
        while time.time()-search_epoch_time < decision_time:

            search_location_list = [0]*number_of_worker

            arg_list = []
            callback_location = 0
            for search_location in search_location_list:
                child_design = design_arr[search_location]
                child_value = value_arr[search_location]
                arg_list.append((child_value, child_design))
                callback_location+=1

            local_results = pool.map(perturb_and_simulate_10time_p, arg_list)


            for result in local_results:
                fit_old = local_fit(value_arr[search_location], scaler)
                fit_new = local_fit(result[1], scaler)
                fit_difference = fit_old - fit_new
                if fit_difference > 0:
                    new_value, new_design = result[1], result[2]
                    value_arr = [copy.deepcopy(new_value)]
                    design_arr = [copy.deepcopy(new_design)]


            pareto_front = copy.deepcopy(value_arr)
            normailized_pareto_front = np.array(scaler.transform(pareto_front)).astype('double')
            phv_value = hvwfg.wfg(normailized_pareto_front, np.array(reference_point).astype('double'))
            print(search_epoch, phv_value, value_arr[0], time.time()-start_time)
            np.savetxt(dir_name+'value_e'+str(search_epoch)+'.txt', value_arr, delimiter=',')
            np.savetxt(dir_name+'true_value_e'+str(search_epoch)+'.txt', scaler.transform(value_arr), delimiter=',')
            with open(dir_name+'design_e'+str(search_epoch)+'.txt', 'a') as f:
                for i in range(len(design_arr)):
                    f.write(str(design_arr[i].tolist())+'\n')
            with open(dir_name+'phv.txt', 'a') as f:
                f.write(str(search_epoch)+','+str(phv_value)+','+str(time.time()-start_time)+'\n')
            search_epoch+=1
    print('Best PHV:', phv_value)            
    return design_arr[0]

def perturb_and_simulate_10time_p(args):
    the_value, the_design = args
    step_size = SIM_CONFIG["step_size"]
    lut_list = SIM_CONFIG["lut_list"]
    func_distribution_arr = SIM_CONFIG["func_distribution_arr"]
    leftover_resource_time_arr = SIM_CONFIG["leftover_resource_time_arr"]
    original_req_distribution_arr = SIM_CONFIG["original_req_distribution_arr"]
    epoch = SIM_CONFIG["epoch"]
    scaler = SIM_CONFIG["scaler"]
    delay_factor = SIM_CONFIG["delay_factor"]
    slo_constraint = SIM_CONFIG["slo_constraint"]
    return perturb_and_simulate_10time(the_value, the_design, 
                                       step_size, lut_list, func_distribution_arr, leftover_resource_time_arr, 
                                       original_req_distribution_arr, epoch, scaler, delay_factor, slo_constraint)

def perturb_and_simulate_10time(the_value, the_design,
                                step_size, lut_list, func_distribution_arr, leftover_resource_time_arr,
                                original_req_distribution_arr, epoch, scaler, delay_factor, slo_constraint):

    for _ in range(5):
        search_req_distribution_arr = perturb(the_design, step_size)
        search_objectives, search_leftover_resource_time_arr = simulation(lut_list, func_distribution_arr, leftover_resource_time_arr, 
                                                                        original_req_distribution_arr, search_req_distribution_arr, epoch, delay_factor, slo_constraint)
        if not np.isfinite(search_objectives).all():
            continue
        if local_fit(search_objectives, scaler) < local_fit(the_value, scaler):
            the_value = copy.deepcopy(search_objectives)
            the_design = copy.deepcopy(search_req_distribution_arr)

    return 0, the_value, the_design

def local_fit(real_values, scaler):
    if not np.isfinite(real_values).all():
        return float('inf')
    normalized_values = scaler.transform([real_values])[0]
    utility = normalized_values[0]
    carbon_emissions = normalized_values[2]
    score = 100 / (carbon_emissions + 1) + utility
    score = carbon_emissions + utility

    return score

# Shared routing neighborhood; each framework retains its acceptance rule.
