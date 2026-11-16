import unittest
from unittest.mock import patch

import numpy as np
import ENV_V3 as env


class SloAccountingTests(unittest.TestCase):
    def test_baseline_network_delays_are_three_six_nine_seconds(self):
        expected = np.array([[0, 3, 6, 9], [3, 0, 3, 6],
                             [6, 3, 0, 3], [9, 6, 3, 0]])
        np.testing.assert_allclose(env.round_trip_time_matrix * 20, expected)
        holding = np.zeros((1, 60), dtype=int)
        holding[0, 0] = 10
        result = env.update_holding_time(holding, np.array([1., 0, 0, 0]),
            np.array([[0, 0., 0, 0, 1]]), np.array([[10]]), 20)
        self.assertEqual(result[0, 9], 10)
        self.assertEqual(result.sum(), 10)

    def test_network_delay_preserves_fractional_counts(self):
        holding = np.zeros((1, 60), dtype=int)
        holding[0, 0] = 3
        origin = np.array([1., 0, 0, 0])
        policy = np.array([[0, .5, .5, 0, 0]])
        result = env.update_holding_time(holding, origin, policy, np.array([[3]]), 20)
        self.assertAlmostEqual(result.sum(), 3)
        self.assertAlmostEqual(result[0, 0], 1.5)
        self.assertAlmostEqual(result[0, 3], 1.5)
        self.assertAlmostEqual(env.calculate_utility(result, np.array([[3]])), 2.925)
        np.testing.assert_array_equal(holding[0, :4], [3, 0, 0, 0])

    def test_routing_cap_boundary_and_invalid_policies(self):
        origin = np.array([.4, .2, .2, .2])
        self.assertTrue(env.check_slo_violation(origin, np.array([[7, .36, .24, .2, .2]]), 10))
        self.assertFalse(env.check_slo_violation(origin, np.array([[7, .359, .241, .2, .2]]), 10))
        for row in ([0, -.1, .5, .3, .3], [0, .2, .2, .2, .2], [0, np.nan, 0, 0, 1]):
            self.assertFalse(env.check_slo_violation(origin, np.array([row]), 100))

    def test_local_200_request_limit_is_20_not_100(self):
        origin = np.array([.2, .4, .4, 0])
        for moved, allowed in ((19, True), (20, True), (21, False), (100, False)):
            policy = np.array([[0, .2-moved/1000, .4, .4, moved/1000]])
            self.assertEqual(env.check_slo_violation(origin, policy, 10), allowed)

    def test_local_cap_zero_full_and_multiple_sources(self):
        origin = np.array([.2, .4, .4, 0])
        self.assertTrue(env.check_slo_violation(origin, np.array([[0, .2, .4, .4, 0]]), 0))
        self.assertFalse(env.check_slo_violation(origin, np.array([[0, .19, .4, .4, .01]]), 0))
        # An initially empty region may receive requests; no division by zero.
        self.assertTrue(env.check_slo_violation(origin, np.array([[0, 0, 0, 0, 1]]), 100))
        self.assertTrue(env.check_slo_violation(origin, np.array([[0, .18, .36, .36, .10]]), 10))
        self.assertFalse(env.check_slo_violation(origin, np.array([[0, .18, .35, .36, .11]]), 10))

    def test_empty_workload_and_three_resource_carryover(self):
        leftover = np.zeros((3, 4, 900))
        leftover[0, 0, 0] = 256
        with patch.object(env, 'SOLAR_ENABLED', False):
            objective, carry = env.simulation([], np.zeros((1, 901)), leftover,
                                             np.ones(4)/4, np.empty((0, 5)), 0, 20, 100)
        self.assertEqual(objective[0], 0)
        self.assertTrue(np.isfinite(objective).all())
        self.assertGreater(objective[1], 0)
        self.assertEqual(np.asarray(carry).shape, (3, 4, 900))

    def test_carryover_retains_memory_and_io(self):
        arrivals = np.zeros((4, 1, 901), dtype=int)
        arrivals[0, 0, 841] = 1
        resources = np.zeros((3, 4, 900))
        resources[:, 0, :100] = np.array([2, 3, 4])[:, None]
        holding = [1] + [0]*59
        with patch.object(env, 'split', return_value=arrivals), patch.object(
                env, 'calculate_resource_time', return_value=(resources, holding, 0, 0)):
            _, carry = env.simulation([], np.array([[1]]), np.zeros((3, 4, 900)),
                                      np.array([1., 0, 0, 0]), np.array([[0, 1., 0, 0, 0]]), 0, 20, 100)
        np.testing.assert_array_equal(np.asarray(carry)[:, 0, :40],
                                      np.repeat(np.array([[2], [3], [4]]), 40, axis=1))

    def test_rejected_policy_uses_percentage_penalty(self):
        objective, _ = env.simulation([], np.array([[10000]]), np.zeros((3, 4, 900)),
                                     np.array([1., 0, 0, 0]), np.array([[0, 0., 1, 0, 0]]), 0, 20, 0)
        self.assertEqual(objective[0], 100)
        self.assertTrue(np.isinf(objective[1:]).all())

    def test_capacity_shortfall_drops_requests_without_queueing(self):
        with patch.object(env, 'feature_varaiation', return_value=(1, 1, 0, 0, 0)):
            _, holding, queued, delay = env.calculate_resource_time(
                [env.number_of_core_per_region]*4, [0]*4, [0]*4, [], 0, 10, 0, 0)
        self.assertEqual(queued, 0)
        self.assertEqual(delay, -1)
        self.assertEqual(holding[59], 10)
        self.assertEqual(env.calculate_utility(np.array([holding]), np.array([[10]])), 0)

    def test_running_work_crosses_batches_and_blocks_new_admissions(self):
        arrivals = np.zeros((4, 1, 901), dtype=int)
        arrivals[0, 0, [1, 61, 121]] = 1
        features = (90, env.number_of_core_per_region,
                    env.mem_size_per_region * 1e9, 0, env.io_size_per_region * 1e9)
        with patch.object(env, 'split', return_value=arrivals), \
                patch.object(env, 'feature_varaiation', return_value=features), \
                patch.object(env, 'calculate_energy', wraps=env.calculate_energy) as energy:
            objective, _ = env.simulation([], np.array([[3]]), np.zeros((3, 4, 900)),
                np.array([1., 0, 0, 0]), np.array([[0, 1., 0, 0, 0]]), 0, 20, 100)
        # First request runs through t=89; second is dropped at t=60.
        # Third starts at t=120 after capacity becomes available.
        cpu, io, _, memory = energy.call_args.args
        for load, capacity in ((cpu, env.number_of_core_per_region), (io, env.io_size_per_region),
                               (memory, env.mem_size_per_region)):
            np.testing.assert_array_equal(load[0, :90], np.full(90, capacity))
            np.testing.assert_array_equal(load[0, 90:120], np.zeros(30))
            np.testing.assert_array_equal(load[0, 120:210], np.full(90, capacity))
            np.testing.assert_array_equal(load[0, 210:], np.zeros(1590))
        self.assertAlmostEqual(objective[0], 100/3)

    def test_running_work_carries_all_resources_to_next_epoch(self):
        arrivals = np.zeros((4, 1, 901), dtype=int)
        arrivals[0, 0, 841] = 1
        with patch.object(env, 'split', return_value=arrivals), \
                patch.object(env, 'feature_varaiation', return_value=(90, 2, 3e9, 0, 4e9)):
            objective, carry = env.simulation([], np.array([[1]]), np.zeros((3, 4, 900)),
                np.array([1., 0, 0, 0]), np.array([[0, 1., 0, 0, 0]]), 0, 20, 100)
        self.assertEqual(objective[0], 0)
        for load, amount in zip(carry, (2, 3, 4)):
            np.testing.assert_array_equal(load[0, :30], np.full(30, amount))
            np.testing.assert_array_equal(load[0, 30:], np.zeros(870))


if __name__ == '__main__':
    unittest.main()
