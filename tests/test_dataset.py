from __future__ import annotations

import unittest

import numpy as np

from surrogate_thesis.config import DatasetConfig, SimulatorConfig
from surrogate_thesis.data import build_windows, prepare_dataset
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

    def test_prepare_dataset_respects_time_ordered_splits(self) -> None:
        frame = _generate_frame(days=5)
        prepared = prepare_dataset(frame, DatasetConfig(lookback=16, horizon=1))

        self.assertGreater(len(prepared.train.X), 0)
        self.assertGreater(len(prepared.val.X), 0)
        self.assertGreater(len(prepared.test.X), 0)
        self.assertLess(prepared.train.target_timestamps.max(), prepared.val.target_timestamps.min())
        self.assertLess(prepared.val.target_timestamps.max(), prepared.test.target_timestamps.min())


if __name__ == "__main__":
    unittest.main()
