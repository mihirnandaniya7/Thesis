from __future__ import annotations

import unittest

import numpy as np

from surrogate_thesis.controller.hybrid_controller import HybridController
from surrogate_thesis.controller.surrogate_decorator import (
    HighFidelitySimulationAdapter,
    SurrogateDecorator,
    SurrogateModelAdapter,
)


class DecoratorTests(unittest.TestCase):
    def test_decorator_switches_to_simulation_when_rolling_error_exceeds_threshold(self) -> None:
        y_true = np.array([[1.0], [1.0], [1.0], [1.0], [1.0]], dtype=np.float32)
        y_pred = np.array([[1.05], [1.08], [1.5], [1.45], [1.02]], dtype=np.float32)
        simulator_runtime = np.full(5, 1.0, dtype=np.float32)
        surrogate_runtime = np.full(5, 0.2, dtype=np.float32)
        timestamps = np.arange(
            np.datetime64("2025-01-01T00"),
            np.datetime64("2025-01-01T05"),
            dtype="datetime64[h]",
        )
        hours = np.arange(5, dtype=np.float32)

        decorator = SurrogateDecorator(
            simulator=HighFidelitySimulationAdapter(
                y_true=y_true,
                runtime_ms=simulator_runtime,
                timestamps=timestamps,
                hours=hours,
            ),
            surrogate=SurrogateModelAdapter(y_pred=y_pred, runtime_ms=surrogate_runtime),
            controller=HybridController(threshold=0.12),
            rolling_window=2,
            warmup_steps=2,
        )

        result = decorator.run(model_name="unit_model", threshold_multiplier=1.0)

        modes = result.trace["mode"].tolist()
        self.assertEqual(modes[:2], ["simulation", "simulation"])
        self.assertIn("surrogate", modes)
        self.assertIn("simulation", modes[2:])
        self.assertGreater(result.metrics["simulation_usage_ratio"], 0.0)
        self.assertGreater(result.metrics["surrogate_usage_ratio"], 0.0)
        self.assertLessEqual(
            result.metrics["decorator_mae"],
            result.metrics["pure_surrogate_mae"],
        )

    def test_more_lenient_threshold_increases_surrogate_usage(self) -> None:
        y_true = np.array([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]], dtype=np.float32)
        y_pred = np.array([[1.1], [1.05], [1.15], [1.2], [1.08], [1.02]], dtype=np.float32)
        simulator_runtime = np.full(6, 1.0, dtype=np.float32)
        surrogate_runtime = np.full(6, 0.2, dtype=np.float32)
        timestamps = np.arange(
            np.datetime64("2025-01-01T00"),
            np.datetime64("2025-01-01T06"),
            dtype="datetime64[h]",
        )
        hours = np.arange(6, dtype=np.float32)

        strict = SurrogateDecorator(
            simulator=HighFidelitySimulationAdapter(y_true, simulator_runtime, timestamps, hours),
            surrogate=SurrogateModelAdapter(y_pred, surrogate_runtime),
            controller=HybridController(threshold=0.08),
            rolling_window=3,
            warmup_steps=1,
        ).run(model_name="strict", threshold_multiplier=0.8)

        lenient = SurrogateDecorator(
            simulator=HighFidelitySimulationAdapter(y_true, simulator_runtime, timestamps, hours),
            surrogate=SurrogateModelAdapter(y_pred, surrogate_runtime),
            controller=HybridController(threshold=0.20),
            rolling_window=3,
            warmup_steps=1,
        ).run(model_name="lenient", threshold_multiplier=1.5)

        self.assertGreater(
            lenient.metrics["surrogate_usage_ratio"],
            strict.metrics["surrogate_usage_ratio"],
        )


if __name__ == "__main__":
    unittest.main()
