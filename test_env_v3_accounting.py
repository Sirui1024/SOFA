import unittest
from unittest.mock import patch

import numpy as np

import ENV_V3 as env


class AccountingTests(unittest.TestCase):
    def test_energy_blocks_match_whole_fleet_with_idle_retention(self):
        cpu = np.zeros((4, 1800))
        io = cpu.copy()
        memory = cpu.copy()
        cpu[:, 10:12] = 128 * 260.5
        io[:, 20:23] = 20 * 280.5
        memory[:, 40:45] = 512 * 300.5
        expected = env._calculate_energy_block(cpu, io, 95, memory, 0, 301)
        np.testing.assert_allclose(env.calculate_energy(cpu, io, 95, memory), expected)

    def test_first_use_keeps_node_on_to_epoch_end_for_each_resource(self):
        for resource, load, busy_watts in (('cpu', 128, 400.),
                                           ('io', 20, 111.2),
                                           ('memory', 512, 100.)):
            with self.subTest(resource=resource), patch.object(env, 'cooling_efficiency_list', [0.]*4):
                arrays = {key: np.zeros((4, 1800)) for key in ('cpu', 'io', 'memory')}
                arrays[resource][:, 10] = load
                # Off for 10s, busy for 1s, then idle for 889s.
                expected = (busy_watts + 889*100.) / 3_600_000 * 1.13
                actual = env.calculate_energy(arrays['cpu'], arrays['io'], 0, arrays['memory'])
                np.testing.assert_allclose(actual, [expected]*4)
                # Repeated use does not add a second idle baseline.
                arrays[resource][:, 20] = load
                expected += (busy_watts - 100.) / 3_600_000 * 1.13
                np.testing.assert_allclose(env.calculate_energy(arrays['cpu'], arrays['io'], 0,
                                                               arrays['memory']), [expected]*4)
                # On-state is epoch-local; an empty next epoch starts powered off.
                zero = np.zeros((4, 1800))
                np.testing.assert_array_equal(env.calculate_energy(zero, zero, 1, zero), [0]*4)

    def test_memory_only_nodes_draw_idle_power_without_double_counting(self):
        zero = np.zeros((4, 1800))
        memory = zero.copy()
        memory[:, :900] = 513  # One full 512-GB node plus a partial node.
        with patch.object(env, 'cooling_efficiency_list', [0.] * 4):
            expected = 2 * 100.0 * 900 / 3_600_000 * 1.13
            np.testing.assert_allclose(env.calculate_energy(zero, zero, 0, memory), [expected]*4)
            cpu = zero.copy()
            cpu[:, :900] = 256  # Both memory-bearing nodes already have CPU power.
            np.testing.assert_allclose(env.calculate_energy(cpu, zero, 0, memory),
                                       env.calculate_energy(cpu, zero, 0))
            io = zero.copy()
            io[:, :900] = 40
            np.testing.assert_allclose(env.calculate_energy(zero, io, 0, memory),
                                       env.calculate_energy(zero, io, 0))
            memory[:, :900] = 0
            memory[:, 900:] = 513
            np.testing.assert_array_equal(env.calculate_energy(zero, zero, 0, memory), [0]*4)

    def energy(self, cpu=0.0, io=0.0, epoch=0):
        return np.array(env.calculate_energy(
            np.full((4, 1800), cpu), np.full((4, 1800), io), epoch
        ))

    def test_idle(self):
        np.testing.assert_array_equal(self.energy(), np.zeros(4))

    def test_cpu_polynomial_uses_utilization_fraction(self):
        # 64/128 physical cores is 0.5; 128/128 is 1.0.
        with patch.object(env, 'cooling_efficiency_list', [0.0] * 4):
            for cores, fraction, watts in ((64, 0.5, 100 + 300 * 2.584262625 / 5.159102),
                                           (128, 1.0, 400.0)):
                self.assertAlmostEqual(env.calculate_power(fraction), watts)
                expected_kwh = watts * 900 / 3_600_000 * 1.13
                np.testing.assert_allclose(self.energy(cpu=cores), [expected_kwh] * 4)

    def test_partial_cpu_load_is_counted(self):
        half = self.energy(cpu=64)
        full = self.energy(cpu=128)
        self.assertTrue(np.all(half > 0))
        self.assertTrue(np.all(full > half))
        # A full node plus a half node must retain both contributions.
        np.testing.assert_allclose(self.energy(cpu=192), full + half)

    def test_node_count_not_applied_twice(self):
        with patch.object(env, 'cooling_efficiency_list', [0.0] * 4):
            expected = env.calculate_power(1.0) * 900 / 3_600_000 * 1.13
            np.testing.assert_allclose(self.energy(cpu=128), [expected] * 4)
            np.testing.assert_allclose(self.energy(cpu=12800), [100 * expected] * 4)

    def test_partial_io_and_capacity(self):
        with patch.object(env, 'cooling_efficiency_list', [0.0] * 4):
            expected = (100.0 + 11.2) * 900 / 3_600_000 * 1.13
            np.testing.assert_allclose(self.energy(io=0.1), [expected] * 4)
            np.testing.assert_allclose(self.energy(io=20), [expected] * 4)
            np.testing.assert_allclose(self.energy(io=20.1), [2 * expected] * 4)

    def test_cpu_and_io_share_node_baseline(self):
        with patch.object(env, 'cooling_efficiency_list', [0.0] * 4):
            delta = self.energy(cpu=128, io=1) - self.energy(cpu=128)
            np.testing.assert_allclose(delta, [13 * 900 / 3_600_000 * 1.13] * 4)

    def test_only_current_interval_is_counted(self):
        cpu = np.zeros((4, 1800))
        io = cpu.copy()
        cpu[:, 900:] = 256
        io[:, 900:] = 20
        np.testing.assert_array_equal(env.calculate_energy(cpu, io, 0), np.zeros(4))

    def test_cooling_hours_do_not_accumulate_and_wrap(self):
        with patch.object(env, 'cooling_efficiency_list', [1.0] * 4), \
                patch.object(env, 'cop_factor_list', list(range(24))):
            it_kwh = env.calculate_power(1.0) * 900 / 3_600_000
            np.testing.assert_allclose(self.energy(cpu=128, epoch=92),
                                       it_kwh * np.array([24.13, 1.13, 2.13, 3.13]))

    def test_cost_carbon_water_units_and_solar_reduction(self):
        with patch.object(env, 'SOLAR_ENABLED', True), \
                patch.object(env, 'solar_list', [[1.0 / 0.35] * 24 for _ in range(4)]):
            # 2 kWh demand minus (1/0.35)*0.35 kWh solar leaves 1 kWh grid demand.
            np.testing.assert_allclose(env.calculate_cost([2] * 4, 0),
                                       [0.48, 0.29, 0.1545, 0.385])
            np.testing.assert_allclose(env.calculate_carbon([2] * 4, 0),
                                       [0.229, 0.548, 0.477, 0.120])
            np.testing.assert_allclose(env.calculate_water([2] * 4, 0),
                                       np.array([0.19, 1.93, 1.89, 3.22]) + 2 / 0.66 * 1.25)
            for calculate in (env.calculate_cost, env.calculate_carbon):
                np.testing.assert_array_equal(calculate([0.5] * 4, 0), np.zeros(4))
            for calculate in (env.calculate_cost, env.calculate_carbon, env.calculate_water):
                np.testing.assert_allclose(calculate([2] * 4, 96), calculate([2] * 4, 0))
            np.testing.assert_allclose(env.calculate_water([0.5] * 4, 0),
                                       np.full(4, 0.5 / 0.66 * 1.25))

    def test_solar_can_be_disabled(self):
        with patch.object(env, 'SOLAR_ENABLED', False), \
                patch.object(env, 'solar_list', [[100000.0] * 24 for _ in range(4)]):
            np.testing.assert_allclose(env.calculate_cost([2] * 4, 0),
                                       np.array([0.48, 0.29, 0.1545, 0.385]) * 2)
            np.testing.assert_allclose(env.calculate_carbon([2] * 4, 0),
                                       np.array(env.carbon_list) * 2)
            np.testing.assert_allclose(env.calculate_water([2] * 4, 0),
                                       np.array(env.water_list) * 2 + 2 / 0.66 * 1.25)

    def test_onsite_water_latent_heat_and_blowdown(self):
        # 0.66 kWh evaporates 1 L; five cycles add 0.25 L blowdown.
        with patch.object(env, 'water_list', [0.0] * 4):
            np.testing.assert_allclose(env.calculate_water([0.66] * 4, 95), [1.25] * 4)
            np.testing.assert_array_equal(env.calculate_water([0.0] * 4, 95), np.zeros(4))


if __name__ == '__main__':
    unittest.main()
