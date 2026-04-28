from __future__ import annotations

import unittest

import numpy as np

from surrogate_thesis.controller.hybrid_controller import HybridController
from surrogate_thesis.controller.surrogate_decorator import (
    HighFidelitySimulationAdapter,
    SurrogateDecorator,
    SurrogateModelAdapter,
)


def _make_adapters(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    simulator_runtime: float = 1.0,
    surrogate_runtime: float = 0.2,
) -> tuple[HighFidelitySimulationAdapter, SurrogateModelAdapter]:
    step_count = len(y_true)
    timestamps = np.arange(
        np.datetime64("2025-01-01T00"),
        np.datetime64("2025-01-01T00") + np.timedelta64(step_count, "h"),
        dtype="datetime64[h]",
    )
    hours = np.arange(step_count, dtype=np.float32)
    return (
        HighFidelitySimulationAdapter(
            y_true=y_true,
            runtime_ms=np.full(step_count, simulator_runtime, dtype=np.float32),
            timestamps=timestamps,
            hours=hours,
        ),
        SurrogateModelAdapter(
            y_pred=y_pred,
            runtime_ms=np.full(step_count, surrogate_runtime, dtype=np.float32),
        ),
    )


class DecoratorTests(unittest.TestCase):
    def test_decorator_managed_recalibration_uses_trusted_labels_to_improve_surrogate(self) -> None:
        y_true = np.array(
            [[1.0], [1.4], [1.8], [2.2], [2.6], [3.0], [3.4], [3.8]],
            dtype=np.float32,
        )
        y_pred = 0.72 * y_true + 0.18
        simulator, surrogate = _make_adapters(y_true=y_true, y_pred=y_pred)

        without_recalibration = SurrogateDecorator(
            simulator=simulator,
            surrogate=surrogate,
            controller=HybridController(
                threshold=10.0,
                validation_interval_steps=2,
                simulation_cooldown_steps=1,
                reentry_probe_interval_steps=1,
            ),
            rolling_window=2,
            warmup_steps=2,
        ).run(model_name="no_recalibration", threshold_multiplier=1.0)

        with_recalibration = SurrogateDecorator(
            simulator=simulator,
            surrogate=surrogate,
            controller=HybridController(
                threshold=10.0,
                validation_interval_steps=2,
                simulation_cooldown_steps=1,
                reentry_probe_interval_steps=1,
            ),
            rolling_window=2,
            warmup_steps=2,
            enable_online_recalibration=True,
            recalibration_min_samples=2,
            recalibration_interval_steps=1,
            recalibration_max_samples=32,
            recalibration_ridge=1e-6,
        ).run(model_name="with_recalibration", threshold_multiplier=1.0)

        self.assertGreater(with_recalibration.metrics["recalibration_updates"], 0.0)
        self.assertGreater(with_recalibration.metrics["recalibration_sample_count"], 0.0)
        self.assertLess(
            with_recalibration.metrics["decorator_mae"],
            without_recalibration.metrics["decorator_mae"],
        )

    def test_decorator_switches_to_simulation_when_validation_probe_detects_large_error(self) -> None:
        y_true = np.array([[1.0], [1.0], [1.0], [1.0], [1.0]], dtype=np.float32)
        y_pred = np.array([[1.05], [1.08], [1.06], [1.45], [1.02]], dtype=np.float32)
        simulator, surrogate = _make_adapters(y_true=y_true, y_pred=y_pred)

        decorator = SurrogateDecorator(
            simulator=simulator,
            surrogate=surrogate,
            controller=HybridController(
                threshold=0.12,
                hysteresis_ratio=1.1,
                validation_interval_steps=2,
                simulation_cooldown_steps=1,
                reentry_probe_interval_steps=1,
            ),
            rolling_window=2,
            warmup_steps=2,
        )

        result = decorator.run(model_name="unit_model", threshold_multiplier=1.0)

        modes = result.trace["mode"].tolist()
        reasons = result.trace["reason"].tolist()
        self.assertEqual(modes[:2], ["simulation", "simulation"])
        self.assertEqual(modes[2], "surrogate")
        self.assertIn("validation_probe_fallback", reasons)
        self.assertGreater(result.metrics["simulation_usage_ratio"], 0.0)
        self.assertGreater(result.metrics["surrogate_usage_ratio"], 0.0)
        self.assertGreater(result.metrics["probe_step_ratio"], 0.0)
        self.assertLessEqual(
            result.metrics["decorator_mae"],
            result.metrics["pure_surrogate_mae"],
        )

    def test_more_lenient_threshold_increases_surrogate_usage(self) -> None:
        y_true = np.array([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]], dtype=np.float32)
        y_pred = np.array([[1.08], [1.05], [1.07], [1.12], [1.10], [1.02]], dtype=np.float32)
        simulator, surrogate = _make_adapters(y_true=y_true, y_pred=y_pred)

        strict = SurrogateDecorator(
            simulator=simulator,
            surrogate=surrogate,
            controller=HybridController(
                threshold=0.06,
                validation_interval_steps=2,
                simulation_cooldown_steps=1,
                reentry_probe_interval_steps=1,
            ),
            rolling_window=3,
            warmup_steps=1,
        ).run(model_name="strict", threshold_multiplier=0.8)

        lenient = SurrogateDecorator(
            simulator=simulator,
            surrogate=surrogate,
            controller=HybridController(
                threshold=0.20,
                validation_interval_steps=2,
                simulation_cooldown_steps=1,
                reentry_probe_interval_steps=1,
            ),
            rolling_window=3,
            warmup_steps=1,
        ).run(model_name="lenient", threshold_multiplier=1.5)

        self.assertGreater(
            lenient.metrics["surrogate_usage_ratio"],
            strict.metrics["surrogate_usage_ratio"],
        )

    def test_decorator_uses_periodic_probes_instead_of_observing_every_step(self) -> None:
        y_true = np.ones((10, 1), dtype=np.float32)
        y_pred = np.array(
            [[1.01], [1.02], [1.01], [1.02], [1.00], [1.03], [1.01], [1.02], [1.00], [1.01]],
            dtype=np.float32,
        )
        simulator, surrogate = _make_adapters(y_true=y_true, y_pred=y_pred)

        result = SurrogateDecorator(
            simulator=simulator,
            surrogate=surrogate,
            controller=HybridController(
                threshold=0.05,
                validation_interval_steps=3,
                simulation_cooldown_steps=1,
                reentry_probe_interval_steps=2,
            ),
            rolling_window=3,
            warmup_steps=2,
        ).run(model_name="periodic_probe", threshold_multiplier=1.0)

        self.assertLess(result.metrics["probe_step_ratio"], 1.0)
        self.assertLess(result.metrics["observation_step_ratio"], 1.0)
        self.assertIn("trusted_surrogate", result.trace["reason"].tolist())

    def test_cooldown_prevents_immediate_return_to_surrogate(self) -> None:
        y_true = np.ones((7, 1), dtype=np.float32)
        y_pred = np.array(
            [[1.01], [1.02], [1.01], [1.40], [1.00], [1.00], [1.00]],
            dtype=np.float32,
        )
        simulator, surrogate = _make_adapters(y_true=y_true, y_pred=y_pred)

        result = SurrogateDecorator(
            simulator=simulator,
            surrogate=surrogate,
            controller=HybridController(
                threshold=0.05,
                hysteresis_ratio=1.0,
                validation_interval_steps=2,
                simulation_cooldown_steps=2,
                reentry_probe_interval_steps=1,
            ),
            rolling_window=2,
            warmup_steps=2,
        ).run(model_name="cooldown", threshold_multiplier=1.0)

        reasons = result.trace["reason"].tolist()
        fallback_index = reasons.index("validation_probe_fallback")
        self.assertEqual(reasons[fallback_index + 1], "simulation_cooldown")
        self.assertEqual(reasons[fallback_index + 2], "simulation_cooldown")


if __name__ == "__main__":
    unittest.main()
