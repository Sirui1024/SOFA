from ENV_V3 import *
from local_search import local_search as perturb
from randomness import SeededPool as Pool, seed_all
from multiprocessing import Process
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans 
from sklearn.preprocessing import MinMaxScaler

global_min_weight = 0.0005

SIM_CONFIG = None


def init_worker(config):
    global SIM_CONFIG
    SIM_CONFIG = config

def swfaga(dir_name, lut_list, func_distribution_arr, leftover_resource_time_arr, req_distribution_arr, epoch, original_req_distribution_arr, delay_factor, slo_constraint, step_size=0.05, number_of_random_search=20, decision_time=900, population_size=70, seed=None):
    seed_all(seed)
    if population_size < 1:
        raise ValueError("population_size must be positive")
    population_limit = population_size
    start_time=time.time()
    brain = GradientBoostingRegressor()
    total_number_of_func_type = func_distribution_arr.shape[0]
    fit_frequency = number_of_random_search
    number_of_objective = 4
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

    simulation_start_time = time.time()
    new_req_distribution_arr = feasible_initial_policy(new_req_distribution_arr, original_req_distribution_arr, slo_constraint)
    new_objectives, new_leftover_resource_time_arr = simulation(lut_list, func_distribution_arr, leftover_resource_time_arr, 
                                                                        original_req_distribution_arr, new_req_distribution_arr, epoch, delay_factor, slo_constraint)
    if not np.isfinite(new_objectives).all():
        raise ValueError('Initial feasible policy returned nonfinite objectives')
    simulation_time = time.time() - simulation_start_time
    print("Initial simulation time:", simulation_time)

    standard_objective = copy.deepcopy(new_objectives)

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

    for i in range(4):
        new_req_distribution_arr = np.ones((epoch_number_of_func_type, 4+1), dtype=float)
        new_req_distribution_arr[:,1]=0.1
        new_req_distribution_arr[:,2]=0.1
        new_req_distribution_arr[:,3]=0.1
        new_req_distribution_arr[:,4]=0.1
        new_req_distribution_arr[:,i+1]=0.7
        for j in range(epoch_number_of_func_type): 
            new_req_distribution_arr[j,0]=scored_prewarm_func_list[j]
        new_req_distribution_arr = feasible_initial_policy(new_req_distribution_arr, original_req_distribution_arr, slo_constraint)
        new_objectives, new_leftover_resource_time_arr = simulation(lut_list, func_distribution_arr, leftover_resource_time_arr, 
                                                                    original_req_distribution_arr, new_req_distribution_arr, epoch, delay_factor, slo_constraint)
        if not np.isfinite(new_objectives).all():
            raise ValueError('Initial feasible policy returned nonfinite objectives')
        value_arr.append(copy.deepcopy(new_objectives))
        design_arr.append(copy.deepcopy(new_req_distribution_arr))


    update_count=0

    design_trajectory = []
    best_fit_trajectory = []
    search_epoch_time = time.time()
    search_epoch = 0

    n_states = 4
    n_actions = 4
    goal_state = 0
    Q_table = np.zeros((n_states, n_actions))

    learning_rate = 0.5
    discount_factor = 0.6
    exploration_prob = 0.7
    epochs = 1000
    old_CV = 0
    old_DV = 0
    action = 0
    current_state = 0

    config = {"step_size": step_size, "lut_list": lut_list, "func_distribution_arr": func_distribution_arr, 
              "leftover_resource_time_arr": leftover_resource_time_arr, "original_req_distribution_arr": original_req_distribution_arr, "epoch": epoch, 
              "scaler": scaler, "delay_factor": delay_factor, "slo_constraint": slo_constraint}

    value_arr, design_arr = create_sparse_population(value_arr, design_arr, scaler, population_threshold=population_limit)
    # A short budget may expire during setup before the first iteration.
    initial_points = np.asarray(scaler.transform(value_arr), dtype=float)
    initial_reference = np.asarray(reference_point, dtype=float)
    initial_points = initial_points[np.isfinite(initial_points).all(axis=1) &
                                    (initial_points < initial_reference).all(axis=1)]
    phv_value = (float(hvwfg.wfg(np.ascontiguousarray(initial_points), initial_reference))
                 if len(initial_points) else 0.0)
    with Pool(processes=number_of_worker, initializer=init_worker, initargs=(config,), seed=seed) as pool:
        while time.time()-search_epoch_time < decision_time:

            if action == 0:
                crossover_rate = 4
            elif action == 1:
                crossover_rate = 3
            elif action == 2:
                crossover_rate = 2
            else:
                crossover_rate = 1

            member_list = list(range(len(design_arr)))
            member_list_size = len(design_arr)
            all_children_design_list = []
            all_children_value_list = []
            for i in range(min(4,member_list_size//2)):
                parent_1 = member_list.pop(random.randrange(len(member_list)))
                parent_2 = member_list.pop(random.randrange(len(member_list)))

                parent_design_1 = design_arr[parent_1].copy()
                parent_design_2 = design_arr[parent_2].copy()

                child_design_candidate_list = GA(parent_design_1, parent_design_2, crossover_rate, step_size)
                all_children_design_list.extend(child_design_candidate_list)

            local_results = []
            if all_children_design_list:
                arg_list = []
                callback_location = 0
                for child_design in all_children_design_list:
                    arg_list.append((child_design))
                    callback_location+=1
                local_results = pool.map(parallel_simulation, arg_list)

                for result in local_results:
                    child_value = result[0]
                    all_children_value_list.append(child_value)
                all_children_value_list, all_children_design_list = finite_population(all_children_value_list, all_children_design_list)


                arg_list = []
                callback_location = 0
                for search_location in range(len(all_children_design_list)):
                    child_design = all_children_design_list[search_location]
                    child_value = all_children_value_list[search_location]
                    arg_list.append((child_value, child_design))
                    callback_location+=1
                local_results = pool.map(perturb_and_simulate_10time_p, arg_list)
                for result in local_results:        
                    if result[1] != []:
                        new_value_list = result[1]
                        new_design_list = result[2]

                        value_arr.extend(copy.deepcopy(new_value_list))
                        design_arr.extend(copy.deepcopy(new_design_list))

                value_arr, design_arr = create_non_dominated_population(value_arr, design_arr)

            value_arr, design_arr = create_sparse_population(value_arr, design_arr, scaler, population_threshold=population_limit)
            pareto_front = copy.deepcopy(value_arr)
            normalized_pareto_front = np.array(scaler.transform(pareto_front)).astype('double')
            population_size = len(value_arr)
            phv_value = hvwfg.wfg(normalized_pareto_front, np.array(reference_point).astype('double'))
            print("Epoch:", search_epoch, "PHV:", phv_value, "Population Size:", population_size, "Time:", time.time()-start_time)

            np.savetxt(dir_name+'value_e'+str(search_epoch)+'.txt', value_arr, delimiter=',')
            np.savetxt(dir_name+'true_value_e'+str(search_epoch)+'.txt', scaler.transform(value_arr), delimiter=',')
            with open(dir_name+'design_e'+str(search_epoch)+'.txt', 'a') as f:
                for i in range(len(design_arr)):
                    f.write(str(design_arr[i].tolist())+'\n')
            if search_epoch == 0:
                with open(dir_name+'phv.txt', 'w') as f:
                    f.write(str(search_epoch)+','+str(phv_value)+','+str(time.time()-start_time)+'\n')
            else:
                with open(dir_name+'phv.txt', 'a') as f:
                    f.write(str(search_epoch)+','+str(phv_value)+','+str(time.time()-start_time)+'\n')
            search_epoch+=1

            new_DV = DV_calculation(value_arr, scaler)
            new_CV = CV_calculation(len(value_arr[0]), value_arr, reference_point)
            delta_DV = old_DV - new_DV
            delta_CV = old_CV - new_CV
            if delta_CV > 0 and delta_DV > 0:
                reward = 10
                next_state = 0
            elif delta_CV > 0 and delta_DV < 0:
                reward = 5
                next_state = 1
            elif delta_CV < 0 and delta_DV > 0:
                reward = 5
                next_state = 2
            else:
                reward = 0
                next_state = 3

            Q_table[current_state, action] += learning_rate * (reward + discount_factor * np.max(Q_table[next_state]) - Q_table[current_state, action])

            if np.random.rand() < exploration_prob:
                action = np.argmax(Q_table[current_state])
            else:
                action = np.random.randint(0, n_actions)
            current_state = copy.deepcopy(next_state)
            old_DV = copy.deepcopy(new_DV)
            old_CV = copy.deepcopy(new_CV)

    best_utility_index = np.argmin(np.array(value_arr)[:, 0])
    best_cost_index = np.argmin(np.array(value_arr)[:, 1])
    best_carbon_index = np.argmin(np.array(value_arr)[:, 2])
    best_water_index = np.argmin(np.array(value_arr)[:, 3])
    print('Best  utility index', best_utility_index, ':', value_arr[best_utility_index])
    print('Best  cost index', best_cost_index, ':', value_arr[best_cost_index])
    print('Best  carbon index', best_carbon_index, ':', value_arr[best_carbon_index])
    print('Best  water index', best_water_index, ':', value_arr[best_water_index])
    print('Best PHV:', phv_value)            
    return design_arr, value_arr, standard_objective

def DV_calculation(value_arr, scaler):
    crowding_distance = calculate_crowding_distance(value_arr, scaler)
    DV = sum(crowding_distance) / len(crowding_distance)
    return DV

def CV_calculation(number_of_objective, population_value, standard_objective):
    CV = 0
    global_min_value = [0]*number_of_objective
    global_max_value = [100]*number_of_objective
    for dimension in range(number_of_objective):
        min_value = min([population_value[i][dimension] for i in range(len(population_value))])
        max_value = max([population_value[i][dimension] for i in range(len(population_value))])
        global_min_value[dimension] = min_value
        global_max_value[dimension] = max_value
    for i in range(len(population_value)):
        span = np.array(global_max_value) - np.array(global_min_value)
        first_point = np.divide(
            population_value[i] - np.array(global_min_value), span,
            out=np.zeros(number_of_objective, dtype=float), where=span != 0,
        ) + 1
        ref_point = standard_objective
        CV += np.linalg.norm(first_point - ref_point)
    CV = CV / len(population_value)
    return CV


def calculate_crowding_distance(value_arr, scaler):
    population_value = scaler.transform(value_arr)
    number_of_objective = len(value_arr[0])

    crowding_distance = [0]*len(population_value)
    population_value_arr = np.array(population_value)

    for i in range(number_of_objective):
        sorted_population_value_arr = np.sort(population_value_arr[:,i])
        sorted_ranks = np.argsort(population_value_arr[:,i])
        for rank in range(len(population_value)):
            current_rank = rank
            current_rank_index = sorted_ranks[current_rank]
            previous_rank = rank - 1
            next_rank = rank + 1
            if previous_rank < 0:
                previous_rank_value = 0
            else: 
                previous_rank_value = sorted_population_value_arr[previous_rank]
            if next_rank > (len(population_value)-1):
                next_rank_value = 100
            else: 
                next_rank_value = sorted_population_value_arr[next_rank]
            crowding_distance[current_rank_index] += (next_rank_value - previous_rank_value)
    return crowding_distance

def create_sparse_population(value_arr, design_arr, scaler, population_threshold=100):
    filtered_value_arr = copy.deepcopy(value_arr)
    filtered_design_arr = copy.deepcopy(design_arr)
    population_size = len(filtered_value_arr)
    crowding_distance_list = [0] * population_size

    if population_size > population_threshold: 
        normalized_values = scaler.transform(filtered_value_arr)
        boundary_1 = np.array([0,0,0,0])
        boundary_2 = np.array([100,100,100,100])
        for i in range(1,5):
            index_list = np.lexsort(np.transpose(normalized_values)[::-i])
            the_first_index = index_list[0]
            the_second_index = index_list[1]
            the_last_index = index_list[-1]
            the_penultimate_index = index_list[-2]
            crowding_distance_list[the_first_index] += (np.linalg.norm(np.array(filtered_value_arr[the_first_index])-boundary_1) + np.linalg.norm(np.array(filtered_value_arr[the_first_index])-filtered_value_arr[the_second_index]))
            crowding_distance_list[the_last_index] += (np.linalg.norm(np.array(filtered_value_arr[the_last_index])-boundary_2) + np.linalg.norm(np.array(filtered_value_arr[the_last_index])-filtered_value_arr[the_penultimate_index]))
            for j in range(1, len(index_list)-1):
                previous_index = index_list[j-1]
                previous_value = np.array(filtered_value_arr[previous_index])
                current_index = index_list[j]
                current_value = np.array(filtered_value_arr[current_index])
                next_index = index_list[j-1]
                next_value = np.array(filtered_value_arr[next_index])
                crowding_distance_list[current_index] += (np.linalg.norm(previous_value - current_value) + np.linalg.norm(next_value - current_value))
        sorted_index_list = sorted(range(len(crowding_distance_list)), key=lambda i: crowding_distance_list[i], reverse = True)
        sparse_sorted_index_list = sorted_index_list[:population_threshold]
        sparse_sorted_index_list = sorted(sparse_sorted_index_list, reverse=True)

        for i in reversed(range(population_size)):
            if i not in sparse_sorted_index_list:
                del filtered_value_arr[i]
                del filtered_design_arr[i]

    return filtered_value_arr, filtered_design_arr

def create_non_dominated_population(value_arr, design_arr):
    filtered_value_arr = copy.deepcopy(value_arr)
    filtered_design_arr = copy.deepcopy(design_arr)
    population_size = len(filtered_value_arr)

    to_be_removed_list = []
    for i in range(1,population_size):
        non_dominated_flag = True
        index_count = 0
        while non_dominated_flag:
            if np.all(np.array(filtered_value_arr[i]) > np.array(filtered_value_arr[index_count])):
                non_dominated_flag = False
                to_be_removed_list.append(i)
            else:
                if i != index_count and np.all(np.array(filtered_value_arr[i]) == np.array(filtered_value_arr[index_count])):
                    non_dominated_flag = False
                    to_be_removed_list.append(i)
                else:
                    index_count+=1

            if index_count == population_size:
                non_dominated_flag = False

    for i in sorted(to_be_removed_list, reverse=True):
        del filtered_value_arr[i]
        del filtered_design_arr[i]

    return filtered_value_arr, filtered_design_arr

def smart_brain_decision(design_arr, brain):
    chosen_starting_points = []
    prediction = []

    pred = brain.predict(design_arr)
    pre = pred.tolist()
    sorted_pre = sorted(range(len(pre)), key=lambda i: pre[i], reverse = True)
    for i in range(min(4,len(sorted_pre))):
        index = sorted_pre[i]
        prediction.append(pre[index])
        chosen_starting_points.append(design_arr[index])
    return chosen_starting_points, prediction


def update_population(value_arr, design_arr, all_children_value_list, all_children_design_list, weight_arr, scaler):
    update_count = 0
    new_design_arr = copy.deepcopy(design_arr)
    new_value_arr = copy.deepcopy(value_arr)
    update_list = list(range(len(weight_arr)))
    random.shuffle(update_list)

    updated = False
    length_of_update_list = len(update_list)
    child_list = list(range(len(all_children_value_list)))
    random.shuffle(child_list)

    while child_list:
        index = child_list.pop()
        child_design = all_children_design_list[index]
        child_value = all_children_value_list[index]
        j = 0
        while j < length_of_update_list:
            the_index = update_list[j]
            candidate = value_arr[the_index]
            weight = weight_arr[the_index]
            replace = False
            fit_new = local_fit(child_value, weight, scaler)
            fit_old = local_fit(candidate, weight,scaler)
            if fit_new < fit_old:
                replace = True
            if replace:
                new_design_arr[the_index] = copy.deepcopy(child_design)
                new_value_arr[the_index] = copy.deepcopy(child_value)
                update_count += 1
                updated = True
                j = length_of_update_list
            else:
                j+=1

    return new_design_arr, new_value_arr

def GA(parent_design_1, parent_design_2, crossover_rate=1, step_size=0.1):
    mutation_rate = 0.5

    return crossover(parent_design_1, parent_design_2, crossover_rate)

def crossover(parent_design_1, parent_design_2, crossover_rate=1):
    child_1 = copy.deepcopy(parent_design_1)
    child_2 = copy.deepcopy(parent_design_2)
    child_3 = (child_1 + child_2)/2
    number_of_rows = child_1.shape[0]
    number_of_columns = child_1.shape[1]
    avail_rows = []
    for i in range(number_of_rows):
        if np.any(parent_design_1[i,:] != parent_design_2[i,:]):
            avail_rows.append(i)

    if avail_rows == []:
        return []

    number_of_rows_to_be_exchanged = int(min(len(avail_rows), crossover_rate))

    exchange_list = random.sample(avail_rows, k = number_of_rows_to_be_exchanged)

    for i in range(number_of_rows):
        if i in exchange_list:
            child_1[i,:] = copy.deepcopy(parent_design_2[i,:])
            child_2[i,:] = copy.deepcopy(parent_design_1[i,:])
        else:
            child_1[i,:] = copy.deepcopy(parent_design_1[i,:])
            child_2[i,:] = copy.deepcopy(parent_design_2[i,:])

    return [child_1, child_2, child_3]

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

    normalized_values = scaler.transform([the_value])[0]
    improvement = 0
    the_value_list = []
    the_design_list = []
    for _ in range(5):
        search_req_distribution_arr = perturb(the_design, step_size)
        search_objectives, search_leftover_resource_time_arr = simulation(lut_list, func_distribution_arr, leftover_resource_time_arr, 
                                                                        original_req_distribution_arr, search_req_distribution_arr, epoch, delay_factor, slo_constraint)
        if not np.isfinite(search_objectives).all():
            continue


        if np.all(np.array(search_objectives) >= np.array(the_value)):
            pass
        else:
            the_design_list.append(copy.deepcopy(search_req_distribution_arr))
            the_value_list.append(copy.deepcopy(search_objectives))
            the_normalized_values = scaler.transform([search_objectives])[0]
            improvement += np.sum(the_normalized_values - normalized_values)

    return 0, the_value_list, the_design_list, improvement

def local_fit(real_values, scaler):
    if not np.isfinite(real_values).all():
        return float('inf')
    normalized_values = scaler.transform([real_values])[0]
    fit_0 = 100 - normalized_values[0]
    fit_1 = normalized_values[1]
    fit_2 = normalized_values[2]*2
    fit_3 = normalized_values[3]
    fit = sum([fit_0, fit_1, fit_2, fit_3])
    return fit

# Shared routing neighborhood; each framework retains its acceptance rule.


def parallel_simulation(args):
    candidate_req_distribution = args
    lut_list = SIM_CONFIG["lut_list"]
    func_distribution_arr = SIM_CONFIG["func_distribution_arr"]
    leftover_resource_time_arr = SIM_CONFIG["leftover_resource_time_arr"]
    original_req_distribution_arr = SIM_CONFIG["original_req_distribution_arr"]
    epoch = SIM_CONFIG["epoch"]
    delay_factor = SIM_CONFIG["delay_factor"]
    slo_constraint = SIM_CONFIG["slo_constraint"]
    return simulation(lut_list, func_distribution_arr, leftover_resource_time_arr, original_req_distribution_arr, candidate_req_distribution, epoch, delay_factor, slo_constraint)
