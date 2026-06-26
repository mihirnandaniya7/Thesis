from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from surrogate_thesis.config import DatasetConfig, SimulatorConfig
from surrogate_thesis.data import build_component_transitions, build_windows, prepare_dataset
from surrogate_thesis.simulation import ReferenceSimulator


def _generate_frame(days: int = 4):
    simulator = ReferenceSimulator()
    return simulator.generate_series(SimulatorConfig(days=days, internal_substeps=4), seed=13)


class DatasetPipelineTests(unittest.TestCase):
    def test_build_windows_uses_expected_shapes(self) -> None:
        frame = _generate_frame(days=4)
        prepared_frame = frame.assign(
            hour_sin=np.sin(2 * np.pi * frame["hour"] / 24),
            hour_cos=np.cos(2 * np.pi * frame["hour"] / 24),
            is_weekend=frame["is_weekend"].astype(float),
        )
        config = DatasetConfig(lookback=16, horizon=1)

        windowed = build_windows(prepared_frame, config)

        expected_samples = len(frame) - config.lookback - config.horizon + 1
        self.assertEqual(windowed.X.shape, (expected_samples, 16, 4))
        self.assertEqual(windowed.y.shape, (expected_samples, 1))

    def test_build_windows_aligns_targets_with_target_timestamps(self) -> None:
        lookback = 32
        horizon = 1
        timestamps = pd.date_range("2025-01-01", periods=40, freq="15min")
        target_values = np.arange(len(timestamps), dtype=np.float32) + 1000.0
        hours = timestamps.hour + timestamps.minute / 60
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "net_load_kw": target_values,
                "hour": hours,
                "hour_sin": np.sin(2 * np.pi * hours / 24),
                "hour_cos": np.cos(2 * np.pi * hours / 24),
                "is_weekend": timestamps.dayofweek >= 5,
                "runtime_ms": np.arange(len(timestamps), dtype=np.float32) + 10.0,
            }
        )
        config = DatasetConfig(
            lookback=lookback,
            horizon=horizon,
            feature_columns=["net_load_kw", "hour_sin", "hour_cos", "is_weekend"],
            target_column="net_load_kw",
        )

        windowed = build_windows(frame, config)

        self.assertEqual(windowed.X[0, 0, 0], 1000.0)
        self.assertEqual(windowed.X[0, -1, 0], 1031.0)
        self.assertEqual(windowed.y[0, 0], 1032.0)
        self.assertEqual(pd.Timestamp(windowed.target_timestamps[0]), timestamps[32])

        for sample_index in [0, 1, len(windowed.y) - 1]:
            target_index = sample_index + lookback + horizon - 1
            expected_window = target_values[sample_index : sample_index + lookback]
            np.testing.assert_array_equal(windowed.X[sample_index, :, 0], expected_window)
            self.assertEqual(windowed.y[sample_index, 0], target_values[target_index])
            self.assertEqual(
                pd.Timestamp(windowed.target_timestamps[sample_index]),
                timestamps[target_index],
            )
            self.assertEqual(
                windowed.reference_runtime_ms[sample_index],
                frame["runtime_ms"].iloc[target_index],
            )

    def test_prepare_dataset_respects_time_ordered_splits(self) -> None:
        frame = _generate_frame(days=5)
        prepared = prepare_dataset(frame, DatasetConfig(lookback=16, horizon=1))

        self.assertGreater(len(prepared.train.X), 0)
        self.assertGreater(len(prepared.val.X), 0)
        self.assertGreater(len(prepared.test.X), 0)
        self.assertLess(prepared.train.target_timestamps.max(), prepared.val.target_timestamps.min())
        self.assertLess(prepared.val.target_timestamps.max(), prepared.test.target_timestamps.min())

    def test_build_component_transitions_uses_state_parameter_action_form(self) -> None:
        simulator = ReferenceSimulator()
        frame = simulator.generate_series(
            SimulatorConfig(days=1, internal_substeps=4, include_stage2_microgrid=True),
            seed=17,
        )

        transitions = build_component_transitions(
            frame,
            state_columns=["battery_soc_kwh"],
            parameter_values={"battery_capacity_kwh": 5.0, "battery_power_kw": 1.8},
            action_columns=["load_kw", "pv_kw"],
            delta_t_hours=0.25,
            next_state_columns=["battery_soc_kwh", "net_load_kw"],
        )

        expected_samples = len(frame) - 1
        self.assertEqual(transitions.state.shape, (expected_samples, 1))
        self.assertEqual(transitions.parameters.shape, (expected_samples, 2))
        self.assertEqual(transitions.action.shape, (expected_samples, 2))
        self.assertEqual(transitions.delta_t.shape, (expected_samples, 1))
        self.assertEqual(transitions.next_state.shape, (expected_samples, 2))


if __name__ == "__main__":
    unittest.main()
