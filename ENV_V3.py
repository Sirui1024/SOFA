"""Four-region workload simulation and environmental objective accounting.

Objectives are minimized: utility loss (%), electricity cost (USD), grid
electricity carbon (kg), and water consumption (L). Scenario constants below
are model inputs, not measurements inferred from the host machine.
"""

import numpy as np
import math
import random
import copy
import csv
import hvwfg
numbe_of_region = 4
epoch_length = 900
# Solar scenario: panel area and conversion efficiency per region.
# Solar profile values are used without an additional 0.25-hour factor.
SOLAR_ENABLED = True
SOLAR_PANEL_AREA_M2_PER_REGION = 1000.0
SOLAR_CONVERSION_EFFICIENCY = 0.35
SOLAR_ENERGY_FACTOR = SOLAR_PANEL_AREA_M2_PER_REGION * SOLAR_CONVERSION_EFFICIENCY / 1000.0
ddl_laxity = 10
resource_per_request = 2
number_of_node_per_region = 20000
# Scenario hardware: dual-socket EPYC 7713; power endpoints are model settings.
CPU_MODEL = 'AMD EPYC 7713'
SOCKETS_PER_NODE = 2
PHYSICAL_CORES_PER_SOCKET = 64
THREADS_PER_CORE = 2
CPU_CORES_PER_NODE = SOCKETS_PER_NODE * PHYSICAL_CORES_PER_SOCKET
CPU_THREADS_PER_NODE = CPU_CORES_PER_NODE * THREADS_PER_CORE
NODE_IDLE_POWER_W = 100.0
NODE_PEAK_POWER_W = 400.0
# Admission and power utilization use physical-core units, not SMT threads.
number_of_core_per_region = CPU_CORES_PER_NODE * number_of_node_per_region
mem_size_per_region = 512 * number_of_node_per_region
io_size_per_region = 20 * number_of_node_per_region

solar_list = [[22.16082,11.67315,4.416712,0.573425,0,0,0,0,0,0,0.018082,2.140274,6.836438,14.19973,24.52575,39.09096,52.99863,63.38521,69.21397,69.85178,65.76795,57.23918,46.41534,34.2674], 
              [9.672877,1.869589,0.033425,0,0,0,0,0,0,0,0.495342,3.067397,11.82493,25.25233,40.81151,54.19726,63.83836,69.89808,72.98767,70.13644,63.53644,52.02521,38.35425,23.56219],
              [9.222466,2.96,0.61726,0,0,0,0,0,0,0,0.288767,5.385753,14.74466,27.7926,42.04795,55.02685,65.4074,72.11945,72.72904,68.20356,60.09562,49.01781,34.80849,21.50027],
              [2.652055,0.219178,0,0,0,0,0,0,0,0.050685,2.525479,10.68712,24.34959,39.62767,53.75507,64.07753,72.00274,75.38877,73.76356,67.11096,55.09836,40.30274,24.91781,11.14767]]

cop_list = [5.45709417, 4.90560884, 4.1080754, 4.429865,
            5.7278023, 3.7502685, 5.45709417, 4.90560884,
            4.1080754, 4.429865, 5.7278023, 3.7502685,
            5.45709417, 4.90560884, 4.1080754, 4.429865]

# Cooling-energy / IT-energy ratios (not COP).
cooling_efficiency_list = [0.2814, 0.2152, 0.2179, 0.1935]

cop_factor_list = [0.85, 0.92, 1.02, 1.10, 1.14, 1.20, 1.21, 1.22, 
                    1.21, 1.20, 1.15, 1.11, 1.03, 0.96, 0.91, 0.816, 
                    0.83, 0.82, 0.82, 0.81, 0.81, 0.80, 0.80, 0.80]

# Hourly tariffs are specified in cents/kWh and converted to dollars/kWh below.
price_list = [[48,48,48,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39],
              [29,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,20,20,29,29,29],
              [15.45,15.45,15.45,15.45,15.45,15.45,15.45,15.45,15.45,15.45,15.45,15.45,15.45,15.45,15.45,15.45,15.45,15.45,15.45,21,21,21,21,15.45],
              [38.5,38.5,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,38.5,38.5,38.5,38.5]]
price_list = [[price / 100.0 for price in region_prices] for region_prices in price_list]

carbon_list = [0.229, 0.548, 0.477, 0.120] # kg/kWh; NYC uses 120 g/kWh

water_list = [0.19, 1.93, 1.89, 3.22] # L/kWh, in region order: Los Angeles, Denver, Detroit, New York.

# Evaporative cooling model: latent heat and concentration-cycle assumptions.
water_latent_heat = 0.66  # kWh/L
cycle_of_concentration = 5  # Blowdown = evaporation / (cycles - 1).

network_penalty_arr = np.array([[0.00,0.05,0.25,0.75],
                                [0.05,0.00,0.05,0.25],
                                [0.25,0.05,0.00,0.05],
                                [0.75,0.25,0.05,0.00]])

# With the default delay_factor=20, adjacent/2-hop/3-hop delays are 3/6/9 seconds.
round_trip_time_matrix = np.array([[0.00,0.15,0.30,0.45],
                                [0.15,0.00,0.15,0.30],
                                [0.30,0.15,0.00,0.15],
                                [0.45,0.30,0.15,0.00]])

utility_list = []

for func_id in range(424):
    utility_list.append([random.randint(1,9)*0.1,-random.randint(1,9)*0.1])

def simulation(lut_list, func_distribution_arr, leftover_resource_time_arr, original_req_distribution_arr, new_req_distribution_arr, epoch, delay_factor, slo, return_details=False):
    ideal_utility = np.sum(func_distribution_arr[:,0])
    total_number_of_func_type = func_distribution_arr.shape[0]
    cumulative_resource_time_arr = np.zeros((numbe_of_region, 2*epoch_length)) # Reserve two epochs to account for carryover work.
    cumulative_mem_time_arr = np.zeros((numbe_of_region, 2*epoch_length))
    cumulative_io_time_arr = np.zeros((numbe_of_region, 2*epoch_length))
    cumulative_resource_time_arr[:,:epoch_length] += leftover_resource_time_arr[0]
    cumulative_mem_time_arr[:,:epoch_length] += leftover_resource_time_arr[1]
    cumulative_io_time_arr[:,:epoch_length] += leftover_resource_time_arr[2]

    if check_slo_violation(original_req_distribution_arr, new_req_distribution_arr, slo):
        split_func_distribution_arr = split(func_distribution_arr, new_req_distribution_arr)
        holding_time_arr = np.zeros((total_number_of_func_type, 60), dtype=int)

        scored_prewarm_func_list = np.flatnonzero(func_distribution_arr[:, 0]).tolist()
        for region_id in range(numbe_of_region):
            queued_func_id_list=[]
            queued_func_req_list=[]
            queued_func_delay_list=[]
            cpu_load_list = []
            mem_load_list = []
            io_load_list = []

            for second_time in range(0, epoch_length, 60):
                # Process queued requests before new arrivals.
                updated_queued_func_id_list=[]
                updated_queued_func_req_list=[]
                updated_queued_func_delay_list=[]
                if sum(queued_func_req_list) != 0:
                    pass
                for i in range(len(queued_func_id_list)):
                    func_id=queued_func_id_list[i]
                    number_of_invocation=queued_func_req_list[i]
                    delay_time=queued_func_delay_list[i]
                    if second_time >= 0:
                        cpu_load_list=[]
                        mem_load_list=[]
                        io_load_list=[]
                        for node_location in range(numbe_of_region):
                            cpu_load_list.append(max(cumulative_resource_time_arr[node_location,second_time:second_time+15]))
                            mem_load_list.append(max(cumulative_mem_time_arr[node_location,second_time:second_time+15]))
                            io_load_list.append(max(cumulative_io_time_arr[node_location,second_time:second_time+15]))

                        resource_time_arr, holding_time_list, disused_cold_startup, delay_time = calculate_resource_time(cpu_load_list, mem_load_list, io_load_list, lut_list, func_id, number_of_invocation, region_id, delay_time)
                        cumulative_resource_time_arr[:,second_time:second_time+900]+=resource_time_arr[0]
                        cumulative_mem_time_arr[:,second_time:second_time+900]+=resource_time_arr[1]
                        cumulative_io_time_arr[:,second_time:second_time+900]+=resource_time_arr[2]
                        holding_time_arr[func_id]+=holding_time_list
                        if disused_cold_startup:
                            if delay_time > 0:
                                updated_queued_func_id_list.append(func_id)
                                updated_queued_func_req_list.append(disused_cold_startup)
                                updated_queued_func_delay_list.append(delay_time)
                            else:
                                print("Warning!!!")
                                exit()
                queued_func_id_list = copy.deepcopy(updated_queued_func_id_list)
                queued_func_req_list = copy.deepcopy(updated_queued_func_req_list)
                queued_func_delay_list = copy.deepcopy(updated_queued_func_delay_list)

                # Admit new arrivals after updating the queue.
                for i in range(len(scored_prewarm_func_list)):
                    func_id = scored_prewarm_func_list[i]
                    splitted_number_of_invocation = split_func_distribution_arr[region_id, func_id, second_time+1:second_time+61].sum()
                    container_location = region_id
                    if splitted_number_of_invocation:

                        cpu_load_list=[]
                        mem_load_list=[]
                        io_load_list=[]
                        for node_location in range(numbe_of_region):
                            cpu_load_list.append(max(cumulative_resource_time_arr[node_location,second_time:second_time+15]))
                            mem_load_list.append(max(cumulative_mem_time_arr[node_location,second_time:second_time+15]))
                            io_load_list.append(max(cumulative_io_time_arr[node_location,second_time:second_time+15]))
                        resource_time_arr, holding_time_list, disused_cold_startup, delay_time = calculate_resource_time(cpu_load_list, mem_load_list, io_load_list, lut_list, func_id, splitted_number_of_invocation, region_id, 0)
                        cumulative_resource_time_arr[:,second_time:second_time+900]+=resource_time_arr[0]
                        cumulative_mem_time_arr[:,second_time:second_time+900]+=resource_time_arr[1]
                        cumulative_io_time_arr[:,second_time:second_time+900]+=resource_time_arr[2]
                        holding_time_arr[func_id]+=holding_time_list
                        if disused_cold_startup:
                            if delay_time > 0:
                                queued_func_id_list.append(func_id)
                                queued_func_req_list.append(disused_cold_startup)
                                queued_func_delay_list.append(delay_time)
                            else:
                                print("Warning!!!")

            for i in range(len(queued_func_id_list)):
                the_func_id = queued_func_id_list[i]
                holding_time_arr[the_func_id,59] += queued_func_req_list[i]


        update_holding_time_arr = update_holding_time(holding_time_arr, original_req_distribution_arr, new_req_distribution_arr, func_distribution_arr, delay_factor)
        utility = calculate_utility(update_holding_time_arr, func_distribution_arr)
        energy_per_region = calculate_energy(cumulative_resource_time_arr, cumulative_io_time_arr, epoch,
                                             cumulative_mem_time_arr)
        cost_per_region = calculate_cost(energy_per_region, epoch)
        carbon_per_region = calculate_carbon(energy_per_region, epoch)
        water_per_region = calculate_water(energy_per_region, epoch)
        cost = sum(cost_per_region)
        carbon = sum(carbon_per_region)
        water = sum(water_per_region)
        utility_loss = ideal_utility - utility
        epoch_objectives = [utility_loss/ideal_utility*100 if ideal_utility else 0.0, cost, carbon, water]
    else:
        pass
        epoch_objectives = [100.0, math.inf, math.inf, math.inf]
    carryover = [arr[:, epoch_length:].copy() for arr in
                 (cumulative_resource_time_arr, cumulative_mem_time_arr, cumulative_io_time_arr)]
    if return_details:
        # Per-function lost utility drives CASA's performance-search ordering.
        losses = np.asarray(func_distribution_arr[:, 0], dtype=float).copy()
        if np.isfinite(epoch_objectives).all():
            weights = np.maximum(1 - np.arange(60)/60, 0)
            weights[59] = 0
            losses -= update_holding_time_arr @ weights
        return epoch_objectives, carryover, {'utility_loss_by_function': losses}
    return epoch_objectives, carryover

def simulation_empty(lut_list, func_distribution_arr, leftover_resource_time_arr, original_req_distribution_arr, new_req_distribution_arr, epoch, delay_factor, slo):
    total_number_of_func_type = func_distribution_arr.shape[0]
    cumulative_resource_time_arr = np.zeros((numbe_of_region, 2*epoch_length)) # Reserve two epochs to account for carryover work.
    cumulative_mem_time_arr = np.zeros((numbe_of_region, 2*epoch_length))
    cumulative_io_time_arr = np.zeros((numbe_of_region, 2*epoch_length))
    cumulative_resource_time_arr[:,:epoch_length] += leftover_resource_time_arr[0]
    cumulative_mem_time_arr[:,:epoch_length] += leftover_resource_time_arr[1]
    cumulative_io_time_arr[:,:epoch_length] += leftover_resource_time_arr[2]

    epoch_objectives = [50,50,50,50]
    return epoch_objectives, cumulative_resource_time_arr[:,900:]

def check_slo_violation_original(original_req_distribution_arr, new_req_distribution_arr, slo_constraint):
    total_number_of_func_type = new_req_distribution_arr.shape[0]

    for func_id in range(total_number_of_func_type):
        region_req_flow = np.zeros((numbe_of_region,numbe_of_region),dtype=float)
        the_index = np.where(new_req_distribution_arr[:,0] == func_id)[0][0]
        difference = new_req_distribution_arr[the_index,1:] - original_req_distribution_arr

        for region_id in range(numbe_of_region):
            if region_id == 0:
                search_list = [1,2,3]
            if region_id == 1:
                search_list = [0,2,3]
            if region_id == 2:
                search_list = [1,3,0]
            if region_id == 3:
                search_list = [2,1,0]

            if difference[region_id] >0 :
                for move_out_region in search_list:
                    if difference[move_out_region] < 0:
                        the_difference = difference[move_out_region] + difference[region_id]
                        if the_difference > 0:
                            region_req_flow[move_out_region,region_id]= -difference[move_out_region]
                            difference[region_id] = the_difference
                            difference[move_out_region] = 0
                        else:
                            region_req_flow[move_out_region,region_id]= difference[region_id]
                            difference[region_id] = 0
                            difference[move_out_region] = the_difference

        for region_id in range(numbe_of_region):
            if region_req_flow[region_id,:].sum() > (slo_constraint/100):
                print('SLO violation!!!')
                return False
    return True

def check_slo_violation(original_req_distribution_arr, new_req_distribution_arr, slo_constraint):
    """Limit relocation to slo_constraint percent of each source's own requests.

    This is a routing constraint, not a measured latency violation rate.
    """
    origin = np.asarray(original_req_distribution_arr, dtype=float)
    policy = np.asarray(new_req_distribution_arr, dtype=float)
    if (origin.shape != (numbe_of_region,) or policy.ndim != 2
            or policy.shape[1] != numbe_of_region + 1
            or not np.isfinite(origin).all() or not np.isfinite(policy).all()
            or np.any(origin < 0) or np.any(policy[:, 1:] < 0)
            or not np.isclose(origin.sum(), 1.0, atol=1e-9, rtol=0)
            or not np.allclose(policy[:, 1:].sum(axis=1), 1.0, atol=1e-9, rtol=0)
            or not np.isfinite(slo_constraint) or not 0 <= slo_constraint <= 100
            or np.any(policy[:, 0] < 0) or np.any(policy[:, 0] != np.floor(policy[:, 0]))
            or len(np.unique(policy[:, 0])) != len(policy)):
        return False
    total_number_of_policy_rows = new_req_distribution_arr.shape[0]

    for the_index in range(total_number_of_policy_rows):
        region_req_flow = np.zeros((numbe_of_region, numbe_of_region), dtype=float)
        difference = new_req_distribution_arr[the_index, 1:numbe_of_region+1] - original_req_distribution_arr

        for region_id in range(numbe_of_region):
            if region_id == 0:
                search_list = [1, 2, 3]
            if region_id == 1:
                search_list = [0, 2, 3]
            if region_id == 2:
                search_list = [1, 3, 0]
            if region_id == 3:
                search_list = [2, 1, 0]

            if difference[region_id] > 0:
                for move_out_region in search_list:
                    if difference[move_out_region] < 0:
                        the_difference = difference[move_out_region] + difference[region_id]
                        if the_difference > 0:
                            region_req_flow[move_out_region, region_id] = -difference[move_out_region]
                            difference[region_id] = the_difference
                            difference[move_out_region] = 0
                        else:
                            region_req_flow[move_out_region, region_id] = difference[region_id]
                            difference[region_id] = 0
                            difference[move_out_region] = the_difference

        for region_id in range(numbe_of_region):
            # Flows are fractions of the function total; scale the limit by
            # this source's original share to use its local request count.
            local_limit = origin[region_id] * (slo_constraint / 100)
            if region_req_flow[region_id, :].sum() > local_limit + 1e-12:
                print('SLO violation!!!')
                return False
    return True

def calculate_utility(holding_time_arr, func_distribution_arr):
    """Linear waiting-time utility; bucket 59 contains dropped/overflow requests."""
    utility = 0
    total_number_of_func_type = func_distribution_arr.shape[0]
    for func_id in range(total_number_of_func_type):
        if func_distribution_arr[func_id,0]:
            for time_count in range(60):
                weight = 0.0 if time_count == 59 else max(1 - time_count / 60, 0)
                utility += holding_time_arr[func_id, time_count] * weight
    return utility


def update_holding_time_copy(holding_time_arr, original_req_distribution_arr, new_req_distribution_arr, func_distribution_arr, delay_factor=1):
    total_number_of_func_type = func_distribution_arr.shape[0]
    new_holding_time_arr = holding_time_arr.copy()
    update_holding_time_arr = np.zeros((total_number_of_func_type, 60), dtype=int)
    region_search_order = {
        0: (1, 2, 3),
        1: (0, 2, 3),
        2: (1, 3, 0),
        3: (2, 1, 0),
    }

    for func_id in range(total_number_of_func_type):
        if func_distribution_arr[func_id,0]:

            region_req_flow = np.zeros((numbe_of_region,numbe_of_region),dtype=float)
            the_index = np.where(new_req_distribution_arr[:,0] == func_id)[0][0]
            difference = new_req_distribution_arr[the_index,1:] - original_req_distribution_arr
            for region_id in range(numbe_of_region):
                search_list = region_search_order[region_id]

                if difference[region_id] >0 :
                    for move_out_region in search_list:
                        if difference[move_out_region] < 0:
                            the_difference = difference[move_out_region] + difference[region_id]
                            if the_difference > 0:
                                region_req_flow[move_out_region,region_id]= -difference[move_out_region]
                                difference[region_id] = the_difference
                                difference[move_out_region] = 0
                            else:
                                region_req_flow[move_out_region,region_id]= difference[region_id]
                                difference[region_id] = 0
                                difference[move_out_region] = the_difference
            for move_out_region in range(4):
                for move_in_region in range(4):
                    difference = region_req_flow[move_out_region, move_in_region]
                    if difference > 0:
                        round_trip_time = round(round_trip_time_matrix[move_out_region, move_in_region] * delay_factor)
                        number_of_move_out_requests = difference * func_distribution_arr[func_id,0]
                        the_index = 0
                        while number_of_move_out_requests:
                            if new_holding_time_arr[func_id, the_index]:
                                if new_holding_time_arr[func_id, the_index] >= number_of_move_out_requests:
                                    number_of_requests = number_of_move_out_requests
                                else:
                                    number_of_requests = new_holding_time_arr[func_id, the_index]

                                new_holding_time_arr[func_id, the_index] -= number_of_requests
                                update_holding_time_arr[func_id, min(the_index+round_trip_time, 59)] += number_of_requests
                                number_of_move_out_requests = number_of_move_out_requests - number_of_requests

                            the_index+=1
                            if the_index > 59:
                                update_holding_time_arr[func_id,59] += number_of_move_out_requests
                                number_of_move_out_requests = 0

    update_holding_time_arr = update_holding_time_arr + new_holding_time_arr
    return update_holding_time_arr                            

def update_holding_time(holding_time_arr, original_req_distribution_arr, new_req_distribution_arr, func_distribution_arr, delay_factor=1):
    total_number_of_func_type = func_distribution_arr.shape[0]
    # Routing fractions produce expected (fractional) request counts.
    # Integer arrays silently discard those counts and inflate utility loss.
    new_holding_time_arr = np.asarray(holding_time_arr, dtype=float).copy()
    update_holding_time_arr = np.zeros((total_number_of_func_type, 60), dtype=float)
    region_search_order = {
        0: (1, 2, 3),
        1: (0, 2, 3),
        2: (1, 3, 0),
        3: (2, 1, 0),
    }

    # Map function IDs to policy rows.
    policy_func_ids = new_req_distribution_arr[:, 0].astype(int)
    policy_index_map = {fid: idx for idx, fid in enumerate(policy_func_ids)}

    for func_id in range(total_number_of_func_type):
        if func_distribution_arr[func_id, 0]:
            if func_id not in policy_index_map:
                continue  # no policy row for this function

            region_req_flow = np.zeros((numbe_of_region, numbe_of_region), dtype=float)
            row_idx = policy_index_map[func_id]
            difference = new_req_distribution_arr[row_idx, 1:numbe_of_region+1] - original_req_distribution_arr

            for region_id in range(numbe_of_region):
                search_list = region_search_order[region_id]
                if difference[region_id] > 0:
                    for move_out_region in search_list:
                        if difference[move_out_region] < 0:
                            the_difference = difference[move_out_region] + difference[region_id]
                            if the_difference > 0:
                                region_req_flow[move_out_region, region_id] = -difference[move_out_region]
                                difference[region_id] = the_difference
                                difference[move_out_region] = 0
                            else:
                                region_req_flow[move_out_region, region_id] = difference[region_id]
                                difference[region_id] = 0
                                difference[move_out_region] = the_difference

            for move_out_region in range(4):
                for move_in_region in range(4):
                    moved = region_req_flow[move_out_region, move_in_region]
                    if moved > 0:
                        round_trip_time = round(round_trip_time_matrix[move_out_region, move_in_region] * delay_factor)
                        number_of_move_out_requests = moved * func_distribution_arr[func_id, 0]
                        t = 0
                        while number_of_move_out_requests:
                            if new_holding_time_arr[func_id, t]:
                                if new_holding_time_arr[func_id, t] >= number_of_move_out_requests:
                                    number_of_requests = number_of_move_out_requests
                                else:
                                    number_of_requests = new_holding_time_arr[func_id, t]

                                new_holding_time_arr[func_id, t] -= number_of_requests
                                update_holding_time_arr[func_id, min(t + round_trip_time, 59)] += number_of_requests
                                number_of_move_out_requests -= number_of_requests

                            t += 1
                            if t > 59:
                                update_holding_time_arr[func_id, 59] += number_of_move_out_requests
                                number_of_move_out_requests = 0

    update_holding_time_arr = update_holding_time_arr + new_holding_time_arr
    return update_holding_time_arr

def update_slo_copy(slo_violation_arr, original_req_distribution_arr, new_req_distribution_arr):
    total_number_of_func_type = new_req_distribution_arr.shape[0]
    new_slo_violation_arr = copy.deepcopy(slo_violation_arr)

    for func_id in range(total_number_of_func_type):
        if slo_violation_arr[func_id,1]:
            region_req_flow = np.zeros((numbe_of_region,numbe_of_region),dtype=float)
            the_index = np.where(new_req_distribution_arr[:,0] == func_id)[0][0]
            difference = new_req_distribution_arr[the_index,1:] - original_req_distribution_arr
            for region_id in range(numbe_of_region):
                if region_id == 0:
                    search_list = [1,2,3]
                if region_id == 1:
                    search_list = [0,2,3]
                if region_id == 2:
                    search_list = [1,3,0]
                if region_id == 3:
                    search_list = [2,1,0]

                if difference[region_id] >0 :
                    for move_out_region in search_list:
                        if difference[move_out_region] < 0:
                            the_difference = difference[move_out_region] + difference[region_id]
                            if the_difference > 0:
                                region_req_flow[move_out_region,region_id]= -difference[move_out_region]
                                difference[region_id] = the_difference
                                difference[move_out_region] = 0
                            else:
                                region_req_flow[move_out_region,region_id]= difference[region_id]
                                difference[region_id] = 0
                                difference[move_out_region] = the_difference


            add_on_slo = np.sum(region_req_flow*network_penalty_arr)

            new_slo_violation_arr[func_id,0]+=(slo_violation_arr[func_id,1]-slo_violation_arr[func_id,0])*add_on_slo


    return new_slo_violation_arr

def update_slo(slo_violation_arr, original_req_distribution_arr, new_req_distribution_arr):
    total_number_of_func_type = slo_violation_arr.shape[0]
    new_slo_violation_arr = copy.deepcopy(slo_violation_arr)

    # Map function IDs to policy rows.
    policy_func_ids = new_req_distribution_arr[:, 0].astype(int)
    policy_index_map = {fid: idx for idx, fid in enumerate(policy_func_ids)}

    for func_id in range(total_number_of_func_type):
        if slo_violation_arr[func_id, 1]:
            if func_id not in policy_index_map:
                continue  # no policy row for this function

            region_req_flow = np.zeros((numbe_of_region, numbe_of_region), dtype=float)
            row_idx = policy_index_map[func_id]
            difference = new_req_distribution_arr[row_idx, 1:numbe_of_region+1] - original_req_distribution_arr

            for region_id in range(numbe_of_region):
                if region_id == 0:
                    search_list = [1, 2, 3]
                if region_id == 1:
                    search_list = [0, 2, 3]
                if region_id == 2:
                    search_list = [1, 3, 0]
                if region_id == 3:
                    search_list = [2, 1, 0]

                if difference[region_id] > 0:
                    for move_out_region in search_list:
                        if difference[move_out_region] < 0:
                            the_difference = difference[move_out_region] + difference[region_id]
                            if the_difference > 0:
                                region_req_flow[move_out_region, region_id] = -difference[move_out_region]
                                difference[region_id] = the_difference
                                difference[move_out_region] = 0
                            else:
                                region_req_flow[move_out_region, region_id] = difference[region_id]
                                difference[region_id] = 0
                                difference[move_out_region] = the_difference

            add_on_slo = np.sum(region_req_flow * network_penalty_arr)
            new_slo_violation_arr[func_id, 0] += (slo_violation_arr[func_id, 1] - slo_violation_arr[func_id, 0]) * add_on_slo

    return new_slo_violation_arr

def calculate_water(energy_per_region, epoch):
    """Total liters: grid-generation water plus onsite evaporation and blowdown.

    Following the reference model, total facility kWh is the heat-rejection
    basis; onsite water is independent of the electricity source.
    """
    water_per_region = []
    hour = epoch//4
    hour = hour%24
    solar_factor = SOLAR_ENERGY_FACTOR if SOLAR_ENABLED else 0
    for region_id in range(numbe_of_region):
        total_energy = energy_per_region[region_id]
        grid_energy = max(total_energy - solar_list[region_id][hour]*solar_factor, 0)
        source_water = grid_energy * water_list[region_id]
        evaporation_water = total_energy / water_latent_heat
        blowdown_water = evaporation_water / (cycle_of_concentration - 1)
        water_per_region.append(source_water + evaporation_water + blowdown_water)

    return water_per_region
def feasible_initial_policy(policy, origin, relocation_limit):
    """Keep a feasible anchor unchanged; otherwise blend it with local routing.

    Every region retains at least origin[r]*(1-relocation_limit/100), so the
    existing per-source relocation limit is respected without hiding penalties.
    """
    policy = np.asarray(policy, dtype=float).copy()
    origin = np.asarray(origin, dtype=float)
    if check_slo_violation(origin, policy, relocation_limit):
        return policy
    if (origin.shape != (numbe_of_region,) or not np.isfinite(origin).all()
            or (origin < 0).any() or not np.isclose(origin.sum(), 1.)
            or not np.isfinite(relocation_limit) or not 0 <= relocation_limit <= 100):
        raise ValueError('Invalid original routing or relocation limit')
    share = relocation_limit / 100.
    policy[:, 1:] = (1-share)*origin + share*policy[:, 1:]
    if not check_slo_violation(origin, policy, relocation_limit):
        raise ValueError('Invalid initial routing policy after feasibility repair')
    return policy


def finite_population(values, designs):
    """Reject nonfinite candidates while keeping objective/design pairs aligned."""
    if len(values) != len(designs):
        raise ValueError('Mismatched objective and design counts')
    pairs = [(v, d) for v, d in zip(values, designs) if np.isfinite(v).all()]
    return [v for v, _ in pairs], [d for _, d in pairs]


def calculate_carbon(energy_per_region, epoch):
    carbon_per_region=[]
    hour = epoch//4
    hour = hour%24
    solar_factor = SOLAR_ENERGY_FACTOR if SOLAR_ENABLED else 0
    for region_id in range(numbe_of_region):
        carbon_per_region.append(max(energy_per_region[region_id]-solar_list[region_id][hour]*solar_factor,0)*carbon_list[region_id])

    return carbon_per_region


def calculate_cost(energy_per_region, epoch):
    cost_per_region = []
    hour = epoch//4
    hour = hour%24
    solar_factor = SOLAR_ENERGY_FACTOR if SOLAR_ENABLED else 0
    for region_id in range(numbe_of_region):
        cost_per_region.append(max(energy_per_region[region_id]-solar_list[region_id][hour]*solar_factor,0)*price_list[region_id][hour])

    return cost_per_region

def calculate_power(core_usage):
    """Rescaled fourth-degree CPU curve; fraction 0..1, unused CPU off.

    Powered-on idle uses NODE_IDLE_POWER_W separately for memory/I/O nodes.
    Power endpoints are scenario assumptions rather than measured hardware limits.
    """
    u = np.clip(np.asarray(core_usage, dtype=float), 0.0, 1.0)
    dynamic = 0.000002*u**4 - 0.0003*u**3 - 0.0184*u**2 + 5.1778*u
    normalized = dynamic / 5.159102
    power = np.where(u > 0, NODE_IDLE_POWER_W +
                     (NODE_PEAK_POWER_W - NODE_IDLE_POWER_W)*normalized, 0.0)
    return float(power) if power.ndim == 0 else power

def calculate_energy(cumulative_resource_time_arr, cumulative_io_time_arr, epoch,
                     cumulative_mem_time_arr=None):
    """Sum independent packed-node blocks without allocating a full fleet tensor."""
    total = np.zeros(numbe_of_region)
    cores_per_node = number_of_core_per_region / number_of_node_per_region
    io_per_node = io_size_per_region / number_of_node_per_region
    memory_per_node = mem_size_per_region / number_of_node_per_region
    used_nodes = max(
        int(np.ceil(np.max(cumulative_resource_time_arr[:, :900]) / cores_per_node)),
        int(np.ceil(np.max(cumulative_io_time_arr[:, :900]) / io_per_node)),
        int(np.ceil(np.max(cumulative_mem_time_arr[:, :900]) / memory_per_node))
        if cumulative_mem_time_arr is not None else 0,
    )
    # Nodes beyond the maximum demand are never powered on in this epoch.
    used_nodes = min(used_nodes, number_of_node_per_region)
    for start in range(0, used_nodes, 250):
        total += _calculate_energy_block(
            cumulative_resource_time_arr, cumulative_io_time_arr, epoch,
            cumulative_mem_time_arr, start, min(250, used_nodes - start))
    return total.tolist()


def _calculate_energy_block(cumulative_resource_time_arr, cumulative_io_time_arr, epoch,
                            cumulative_mem_time_arr, node_start, node_count):

    node_indices = np.arange(node_start, node_start + node_count)[None, :, None]
    cores_per_node = number_of_core_per_region / number_of_node_per_region
    io_capacity_per_node = io_size_per_region / number_of_node_per_region
    # Pack cores onto nodes; the polynomial takes fractions, not percentages.
    load_per_region = np.clip(
        cumulative_resource_time_arr[:, None, :900] / cores_per_node - node_indices,
        0.0, 1.0,
    )
    io_load_per_region = np.clip(
        cumulative_io_time_arr[:, None, :900] / io_capacity_per_node - node_indices,
        0.0, 1.0,
    )
    # Once any resource activates a packed node, it stays on through this epoch.
    # This accounting does not change the 60-second admission/scheduling model.
    resource_active = (load_per_region > 0) | (io_load_per_region > 0)
    if cumulative_mem_time_arr is not None:
        memory_capacity_per_node = mem_size_per_region / number_of_node_per_region
        memory_active = (cumulative_mem_time_arr[:, None, :900] /
                         memory_capacity_per_node) > node_indices
        resource_active |= memory_active
    powered_on = np.logical_or.accumulate(resource_active, axis=2)
    idle_only = powered_on & (load_per_region == 0) & (io_load_per_region == 0)
    idle_power_values = np.where(idle_only, NODE_IDLE_POWER_W, 0.0)
    hour = epoch//4
    hour = hour%24
    power_values = calculate_power(load_per_region)

    # Empirical peak I/O power is 13 W per active node.
    peak_io_mask = (io_load_per_region > 0) & (load_per_region > 0)
    active_io_mask = (io_load_per_region > 0) & (load_per_region == 0)
    io_power_values = np.where(
        peak_io_mask,
        13.0,  # Peak I/O power when I/O load is at maximum
        np.where(
            active_io_mask,
            NODE_IDLE_POWER_W + 11.2,  # Active I/O power when I/O load is present but not at peak
            0.0,   # Idle I/O power when there is no I/O load
        )
    )


    for region_id in range(numbe_of_region):
        region_hour = (hour + region_id) % 24
        time_factor = cop_factor_list[region_hour]
        joule_in_15min = power_values[region_id, :, :900].sum() + io_power_values[region_id, :, :900].sum()

        # Power already includes every node; one-second samples sum to joules.
        it_energy = joule_in_15min / 3_600_000.0
        it_energy += idle_power_values[region_id].sum() / 3_600_000.0
        ipcs_energy = it_energy * 0.13
        cooling_energy = it_energy * cooling_efficiency_list[region_id] * time_factor
        total_energy = it_energy + ipcs_energy + cooling_energy
        if region_id == 0:
            energy_per_region = [total_energy]
        else:
            energy_per_region.append(total_energy)
    return energy_per_region

def feature_varaiation(lut_list, func_id):
    variation_list = lut_list[func_id]
    varaying_features = variation_list[random.randint(0, len(variation_list)-1)]
    return varaying_features

def calculate_resource_time(cpu_load_list, mem_load_list, io_load_list, lut_list, func_id, number_of_invocation, region_id, delay_time): 
    """Admit what fits now and drop the rest; never queue across batches.

    Admitted work reserves resources for its execution duration, including
    time beyond the current 60-second batch.
    """
    cpu_load = cpu_load_list[region_id]
    mem_load = mem_load_list[region_id]
    io_load = io_load_list[region_id]

    req_features = feature_varaiation(lut_list, func_id)
    base_execution_time, cpu_usage, memory_usage, req_size, io_size = req_features
    ddl_time =  base_execution_time * ddl_laxity
    holding_time_list = [0]*60 # At most 60 seconds of holding time, excluding execution.


    resource_time_arr = [
        np.zeros((numbe_of_region, 900)),
        np.zeros((numbe_of_region, 900)),
        np.zeros((numbe_of_region, 900)),
    ]
    slo_violation_count = 0

    prewarm_shortage = number_of_invocation
    disused_cold_startup = 0

    if delay_time > 59 and prewarm_shortage:
        delay_time=-1
        holding_time_list[59] = number_of_invocation
        disused_cold_startup = 0       
        prewarm_shortage = 0

    if prewarm_shortage:
        if (cpu_load + cpu_usage * prewarm_shortage) <= number_of_core_per_region and (mem_load + memory_usage/1e9 * prewarm_shortage) <= mem_size_per_region and (io_load + io_size/1e9 * prewarm_shortage) <= io_size_per_region:
            resource_time_arr[0][region_id, :base_execution_time] += cpu_usage * prewarm_shortage
            resource_time_arr[1][region_id, :base_execution_time] += memory_usage/1e9 * prewarm_shortage
            resource_time_arr[2][region_id, :base_execution_time] += io_size/1e9 * prewarm_shortage

            cpu_load += cpu_usage * prewarm_shortage
            mem_load += memory_usage/1e9 * prewarm_shortage
            io_load += io_size/1e9 * prewarm_shortage
            holding_time_list[delay_time] = int(copy.deepcopy(prewarm_shortage))
            prewarm_shortage = 0
            disused_cold_startup = 0
        else:
            if cpu_usage == 0:
                survived_cold_startup_1 = prewarm_shortage
            else:
                survived_cold_startup_1 = max(0, int((number_of_core_per_region - cpu_load) // cpu_usage))
            if memory_usage == 0:
                survived_cold_startup_2 = prewarm_shortage
            else:
                survived_cold_startup_2 = max(0, int((mem_size_per_region - mem_load) // (memory_usage/1e9)))
            if io_size == 0:
                survived_cold_startup_3 = prewarm_shortage
            else:
                survived_cold_startup_3 = max(0, int((io_size_per_region - io_load) // (io_size/1e9)))
            survived_cold_startup = int(min(survived_cold_startup_1, survived_cold_startup_2, survived_cold_startup_3, prewarm_shortage))

            disused_cold_startup = prewarm_shortage - survived_cold_startup
            resource_time_arr[0][region_id, :base_execution_time] += cpu_usage * survived_cold_startup
            resource_time_arr[1][region_id, :base_execution_time] += memory_usage/1e9 * survived_cold_startup
            resource_time_arr[2][region_id, :base_execution_time] += io_size/1e9 * survived_cold_startup
            cpu_load += cpu_usage * survived_cold_startup
            mem_load += memory_usage/1e9 * survived_cold_startup
            io_load += io_size/1e9 * survived_cold_startup
            cold_startup_rerouting_time = 0
            if disused_cold_startup == math.inf:
                print('inf warning!!!')
                print(survived_cold_startup_1, survived_cold_startup_2, survived_cold_startup_3)
            holding_time_list[delay_time] = int(copy.deepcopy(survived_cold_startup))
            prewarm_shortage = 0
            # Requests that cannot start are dropped, not deferred to a batch.
            holding_time_list[59] += int(disused_cold_startup)
            disused_cold_startup = 0
            delay_time = -1

    return (resource_time_arr, holding_time_list, disused_cold_startup, delay_time)

def split(func_distribution_arr, new_req_distribution_arr):
    total_number_of_func_type = func_distribution_arr.shape[0]
    split_func_distribution_arr = np.zeros((numbe_of_region, total_number_of_func_type, epoch_length+1), dtype=int)
    policy_func_ids = new_req_distribution_arr[:, 0].astype(int)
    policy_index_map = {func_id: idx for idx, func_id in enumerate(policy_func_ids)}
    active_func_ids = np.flatnonzero(func_distribution_arr[:, 0]).astype(int)
    second_level_prob = np.full(epoch_length, 1.0 / epoch_length)

    for func_id in active_func_ids:
        if func_id in policy_index_map:
            the_index = policy_index_map[func_id]
            total_req_num = int(func_distribution_arr[func_id, 0])

            region_probs = new_req_distribution_arr[the_index, 1:numbe_of_region+1]
            region_split = np.floor(total_req_num * region_probs).astype(int)

            difference = total_req_num - int(np.sum(region_split))
            if difference < 0:
                print(func_id, "warning")
                print(new_req_distribution_arr[the_index,:])
                for j in range(numbe_of_region):
                    print(math.floor(total_req_num * new_req_distribution_arr[the_index, j+1]))
                exit()
            region_split[0] += difference

            split_func_distribution_arr[:, func_id, 0] = region_split

            for region_id in range(numbe_of_region):
                list_sum = int(region_split[region_id])
                if list_sum > 0:
                    split_func_distribution_arr[region_id, func_id, 1:] = np.random.multinomial(list_sum, second_level_prob)
        else:
            print('func id', func_id, 'is not split by policy' )
            exit()

    return split_func_distribution_arr
