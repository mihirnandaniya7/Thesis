from __future__ import annotations

import unittest

import numpy as np

from surrogate_thesis.config import SimulatorConfig
from surrogate_thesis.simulation import ReferenceSimulator


class ReferenceSimulatorTests(unittest.TestCase):
    def test_reference_simulator_is_reproducible_for_same_seed(self) -> None:
        config = SimulatorConfig(days=2, internal_substeps=4)
        simulator = ReferenceSimulator()

        first = simulator.generate_series(config, seed=11)
        second = simulator.generate_series(config, seed=11)

        self.assertTrue(np.allclose(first["load_kw"], second["load_kw"]))
        self.assertTrue(np.allclose(first["net_load_kw"], second["net_load_kw"]))

    def test_reference_simulator_produces_non_negative_stage1_load(self) -> None:
        config = SimulatorConfig(days=3, internal_substeps=4, include_stage2_microgrid=False)
        simulator = ReferenceSimulator()

        frame = simulator.generate_series(config, seed=5)

        self.assertTrue((frame["load_kw"] >= 0.0).all())
        self.assertTrue((frame["runtime_ms"] > 0.0).all())
        self.assertTrue(frame["timestamp"].diff().dropna().dt.total_seconds().eq(15 * 60).all())

    def test_stage2_mode_emits_microgrid_columns(self) -> None:
        config = SimulatorConfig(days=1, internal_substeps=3, include_stage2_microgrid=True)
        simulator = ReferenceSimulator()

        frame = simulator.generate_series(config, seed=19)

        self.assertTrue(
            {"pv_kw", "battery_power_kw", "battery_soc_kwh", "net_load_kw"}.issubset(frame.columns)
        )
        self.assertTrue(frame["battery_soc_kwh"].between(0.0, config.battery_capacity_kwh).all())


if __name__ == "__main__":
    unittest.main()
