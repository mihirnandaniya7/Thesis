from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from surrogate_thesis.config import DecoratorConfig, ExperimentConfig
from surrogate_thesis.data.dataset import DatasetSplit, NormalizationStats
from surrogate_thesis.evaluation.metrics import mae, rmse
from surrogate_thesis.training import ModelArtifacts, predict_model

from .hybrid_controller import HybridController


@dataclass(slots=True)
class HighFidelitySimulationAdapter:
    y_true: np.ndarray
    runtime_ms: np.ndarray
    timestamps: np.ndarray
    hours: np.ndarray

    def forecast(self, index: int) -> tuple[np.ndarray, float]:
        return self.y_true[index], float(self.runtime_ms[index])


@dataclass(slots=True)
class SurrogateModelAdapter:
    y_pred: np.ndarray
    runtime_ms: np.ndarray

    def forecast(self, index: int) -> tuple[np.ndarray, float]:
        return self.y_pred[index], float(self.runtime_ms[index])


@dataclass(slots=True)
class DecoratorRunResult:
    model_name: str
    threshold: float
    threshold_multiplier: float
    metrics: dict[str, float]
    trace: pd.DataFrame

    def to_record(self) -> dict[str, float | str]:
        return {"model_name": self.model_name, "threshold": self.threshold, **self.metrics}


@dataclass(slots=True)
class DecoratorEvaluationArtifacts:
    model_name: str
    sensitivity_frame: pd.DataFrame
    preferred_result: DecoratorRunResult


class SurrogateDecorator:
    """Adaptive runtime wrapper that switches between surrogate and simulator."""

    def __init__(
        self,
        simulator: HighFidelitySimulationAdapter,
        surrogate: SurrogateModelAdapter,
        controller: HybridController,
        rolling_window: int,
        warmup_steps: int,
    ) -> None:
        self.simulator = simulator
        self.surrogate = surrogate
        self.controller = controller
        self.rolling_window = rolling_window
        self.warmup_steps = warmup_steps

    def run(
        self,
        model_name: str,
        threshold_multiplier: float,
    ) -> DecoratorRunResult:
        recent_errors: deque[float] = deque(maxlen=self.rolling_window)
        rows: list[dict[str, float | int | str]] = []
        previous_mode: str | None = None
        surrogate_steps = 0
        simulation_steps = 0
        switch_count = 0
        effective_runtime_ms = 0.0
        effective_step_errors = []
        surrogate_step_errors = []

        total_steps = len(self.surrogate.y_pred)
        for index in range(total_steps):
            surrogate_prediction, surrogate_runtime_ms = self.surrogate.forecast(index)
            true_value, simulation_runtime_ms = self.simulator.forecast(index)
            surrogate_error = float(np.mean(np.abs(surrogate_prediction - true_value)))
            rolling_error_before = float(np.mean(recent_errors)) if recent_errors else float("inf")

            if index < self.warmup_steps:
                mode = "simulation"
                reason = "warmup"
            elif not recent_errors:
                mode = "simulation"
                reason = "history_unavailable"
            else:
                mode = self.controller.decide(rolling_error_before)
                reason = "threshold"

            if mode == "surrogate":
                output = surrogate_prediction
                step_runtime_ms = surrogate_runtime_ms
                surrogate_steps += 1
            else:
                output = true_value
                step_runtime_ms = simulation_runtime_ms
                simulation_steps += 1

            if previous_mode is not None and previous_mode != mode:
                switch_count += 1
            previous_mode = mode

            effective_error = float(np.mean(np.abs(output - true_value)))
            effective_runtime_ms += step_runtime_ms
            surrogate_step_errors.append(surrogate_error)
            effective_step_errors.append(effective_error)
            recent_errors.append(surrogate_error)
            rolling_error_after = float(np.mean(recent_errors))

            rows.append(
                {
                    "index": index,
                    "timestamp": str(self.simulator.timestamps[index]),
                    "hour": float(self.simulator.hours[index]),
                    "mode": mode,
                    "reason": reason,
                    "threshold": float(self.controller.threshold),
                    "rolling_error_before": rolling_error_before,
                    "rolling_error_after": rolling_error_after,
                    "surrogate_step_error": surrogate_error,
                    "effective_step_error": effective_error,
                    "surrogate_runtime_ms": surrogate_runtime_ms,
                    "simulation_runtime_ms": simulation_runtime_ms,
                    "effective_runtime_ms": step_runtime_ms,
                }
            )

        simulator_total_runtime_ms = float(np.sum(self.simulator.runtime_ms))
        surrogate_total_runtime_ms = float(np.sum(self.surrogate.runtime_ms))
        metrics = {
            "threshold_multiplier": float(threshold_multiplier),
            "pure_surrogate_mae": float(np.mean(surrogate_step_errors)),
            "decorator_mae": float(np.mean(effective_step_errors)),
            "decorator_rmse": float(np.sqrt(np.mean(np.square(effective_step_errors)))),
            "fallback_improvement": float(np.mean(surrogate_step_errors) - np.mean(effective_step_errors)),
            "surrogate_usage_ratio": float(surrogate_steps / total_steps),
            "simulation_usage_ratio": float(simulation_steps / total_steps),
            "switch_count": float(switch_count),
            "avg_shadow_rolling_error": float(np.mean(surrogate_step_errors)),
            "decorator_runtime_ms": effective_runtime_ms,
            "simulator_runtime_ms": simulator_total_runtime_ms,
            "pure_surrogate_runtime_ms": surrogate_total_runtime_ms,
            "decorator_speedup": float(simulator_total_runtime_ms / max(effective_runtime_ms, 1e-9)),
            "pure_surrogate_speedup": float(
                simulator_total_runtime_ms / max(surrogate_total_runtime_ms, 1e-9)
            ),
        }
        return DecoratorRunResult(
            model_name=model_name,
            threshold=float(self.controller.threshold),
            threshold_multiplier=float(threshold_multiplier),
            metrics=metrics,
            trace=pd.DataFrame(rows),
        )


def evaluate_decorator_thresholds(
    artifacts: ModelArtifacts,
    test_split: DatasetSplit,
    normalization: NormalizationStats,
    config: ExperimentConfig,
) -> DecoratorEvaluationArtifacts:
    y_true = _denormalize(test_split.y, normalization)
    surrogate_predictions, surrogate_runtimes = _collect_single_step_surrogate_outputs(
        artifacts=artifacts,
        X=test_split.X,
        normalization=normalization,
        device=config.training.device,
    )

    pure_surrogate_mae = mae(y_true, surrogate_predictions)
    thresholds = _build_threshold_schedule(
        base_error=pure_surrogate_mae,
        multipliers=config.decorator.threshold_multipliers,
    )

    simulator_adapter = HighFidelitySimulationAdapter(
        y_true=y_true,
        runtime_ms=test_split.reference_runtime_ms,
        timestamps=test_split.target_timestamps,
        hours=test_split.target_hours,
    )
    surrogate_adapter = SurrogateModelAdapter(
        y_pred=surrogate_predictions,
        runtime_ms=surrogate_runtimes,
    )

    results: list[DecoratorRunResult] = []
    for multiplier, threshold in thresholds:
        decorator = SurrogateDecorator(
            simulator=simulator_adapter,
            surrogate=surrogate_adapter,
            controller=HybridController(threshold=threshold),
            rolling_window=config.decorator.rolling_window,
            warmup_steps=config.decorator.warmup_steps,
        )
        results.append(
            decorator.run(
                model_name=artifacts.name,
                threshold_multiplier=multiplier,
            )
        )

    preferred_result = min(
        results,
        key=lambda item: abs(
            item.threshold_multiplier - config.decorator.preferred_threshold_multiplier
        ),
    )
    sensitivity_frame = pd.DataFrame([item.to_record() for item in results])
    return DecoratorEvaluationArtifacts(
        model_name=artifacts.name,
        sensitivity_frame=sensitivity_frame,
        preferred_result=preferred_result,
    )


def _collect_single_step_surrogate_outputs(
    artifacts: ModelArtifacts,
    X: np.ndarray,
    normalization: NormalizationStats,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    runtimes_ms = []
    for sample in X:
        start = perf_counter()
        pred_norm = predict_model(
            artifacts=artifacts,
            X=sample[np.newaxis, ...],
            device=device,
            batch_size=1,
        )
        runtimes_ms.append((perf_counter() - start) * 1000)
        predictions.append(pred_norm[0])
    predictions_array = np.asarray(predictions, dtype=np.float32)
    return _denormalize(predictions_array, normalization), np.asarray(runtimes_ms, dtype=np.float32)


def _build_threshold_schedule(
    base_error: float,
    multipliers: list[float],
) -> list[tuple[float, float]]:
    thresholds = []
    for multiplier in multipliers:
        thresholds.append((float(multiplier), max(base_error * float(multiplier), 1e-6)))
    return thresholds


def _denormalize(values: np.ndarray, normalization: NormalizationStats) -> np.ndarray:
    return values * normalization.target_std + normalization.target_mean
