from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol
import warnings

import numpy as np
import pandas as pd
import torch

from surrogate_thesis.config import ExperimentConfig
from surrogate_thesis.data.dataset import DatasetSplit, NormalizationStats
from surrogate_thesis.evaluation.metrics import mae, mape, nmae, nrmse, rmse, smape
from surrogate_thesis.training import ModelArtifacts, predict_model

from .hybrid_controller import HybridController


@dataclass(slots=True)
class ForecastResult:
    value: np.ndarray
    runtime_ms: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ForecastProvider(Protocol):
    """Common runtime interface for raw and decorated forecast providers."""

    def forecast(self, index: int) -> ForecastResult:
        ...


@dataclass(slots=True)
class HighFidelitySimulationAdapter:
    y_true: np.ndarray
    runtime_ms: np.ndarray
    timestamps: np.ndarray
    hours: np.ndarray

    def forecast(self, index: int) -> ForecastResult:
        return ForecastResult(
            value=self.value_at(index),
            runtime_ms=float(self.runtime_ms[index]),
            source="simulation",
            metadata={
                "timestamp": self.timestamps[index],
                "hour": float(self.hours[index]),
                "simulation_executed": True,
                "surrogate_executed": False,
            },
        )

    def value_at(self, index: int) -> np.ndarray:
        return np.asarray(self.y_true[index], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.y_true)


@dataclass(slots=True)
class SurrogateModelAdapter:
    y_pred: np.ndarray
    runtime_ms: np.ndarray

    def forecast(self, index: int) -> ForecastResult:
        return ForecastResult(
            value=self.value_at(index),
            runtime_ms=float(self.runtime_ms[index]),
            source="surrogate",
            metadata={
                "simulation_executed": False,
                "surrogate_executed": True,
                "base_prediction": self.value_at(index).copy(),
            },
        )

    def value_at(self, index: int) -> np.ndarray:
        return np.asarray(self.y_pred[index], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.y_pred)


@dataclass(slots=True)
class OnlineRecalibrationManager:
    min_samples: int
    update_interval_steps: int
    max_samples: int
    ridge: float
    sample_predictions: deque[np.ndarray] = field(init=False, repr=False)
    sample_targets: deque[np.ndarray] = field(init=False, repr=False)
    slope: np.ndarray | None = None
    intercept: np.ndarray | None = None
    update_count: int = 0
    last_update_step: int = -1

    def __post_init__(self) -> None:
        self.sample_predictions = deque(maxlen=self.max_samples)
        self.sample_targets = deque(maxlen=self.max_samples)

    @property
    def sample_count(self) -> int:
        return len(self.sample_predictions)

    @property
    def is_active(self) -> bool:
        return self.slope is not None and self.intercept is not None

    def transform(self, prediction: np.ndarray) -> np.ndarray:
        if not self.is_active:
            return np.asarray(prediction, dtype=np.float32)
        prediction_array = np.asarray(prediction, dtype=np.float32)
        return (self.slope * prediction_array + self.intercept).astype(np.float32)

    def observe(
        self,
        *,
        step_index: int,
        base_prediction: np.ndarray,
        target: np.ndarray,
    ) -> bool:
        self.sample_predictions.append(np.asarray(base_prediction, dtype=np.float32).copy())
        self.sample_targets.append(np.asarray(target, dtype=np.float32).copy())
        if self.sample_count < self.min_samples:
            return False
        if self.last_update_step >= 0 and (
            step_index - self.last_update_step < self.update_interval_steps
        ):
            return False
        self._fit()
        self.last_update_step = step_index
        self.update_count += 1
        return True

    def _fit(self) -> None:
        predictions = np.asarray(self.sample_predictions, dtype=np.float32)
        targets = np.asarray(self.sample_targets, dtype=np.float32)
        output_dim = predictions.shape[-1]

        slopes = np.zeros(output_dim, dtype=np.float32)
        intercepts = np.zeros(output_dim, dtype=np.float32)
        identity = np.eye(2, dtype=np.float32)
        ones = np.ones((predictions.shape[0], 1), dtype=np.float32)

        for dim in range(output_dim):
            design = np.concatenate([predictions[:, dim : dim + 1], ones], axis=1)
            lhs = design.T @ design + self.ridge * identity
            rhs = design.T @ targets[:, dim]
            params = np.linalg.solve(lhs, rhs)
            slopes[dim] = float(params[0])
            intercepts[dim] = float(params[1])

        self.slope = slopes
        self.intercept = intercepts


class ForecastProviderDecorator:
    """Base decorator: same interface as the wrapped provider."""

    def __init__(self, wrapped: ForecastProvider) -> None:
        self.wrapped = wrapped

    def forecast(self, index: int) -> ForecastResult:
        return self.wrapped.forecast(index)


class RecalibratingForecastDecorator(ForecastProviderDecorator):
    """Applies a lightweight linear correction learned from trusted labels."""

    def __init__(
        self,
        wrapped: ForecastProvider,
        *,
        min_samples: int,
        update_interval_steps: int,
        max_samples: int,
        ridge: float,
    ) -> None:
        super().__init__(wrapped)
        self.manager = OnlineRecalibrationManager(
            min_samples=min_samples,
            update_interval_steps=update_interval_steps,
            max_samples=max_samples,
            ridge=ridge,
        )

    def forecast(self, index: int) -> ForecastResult:
        result = self.wrapped.forecast(index)
        base_prediction = np.asarray(
            result.metadata.get("base_prediction", result.value),
            dtype=np.float32,
        )
        adjusted_prediction = self.manager.transform(base_prediction)
        metadata = {
            **result.metadata,
            "base_prediction": base_prediction.copy(),
            "recalibration_active": self.manager.is_active,
            "recalibration_sample_count": float(self.manager.sample_count),
            "recalibration_updates": float(self.manager.update_count),
        }
        return ForecastResult(
            value=adjusted_prediction,
            runtime_ms=result.runtime_ms,
            source=result.source,
            metadata=metadata,
        )

    def observe_trusted_label(
        self,
        *,
        step_index: int,
        base_prediction: np.ndarray,
        target: np.ndarray,
    ) -> bool:
        return self.manager.observe(
            step_index=step_index,
            base_prediction=base_prediction,
            target=target,
        )

    @property
    def update_count(self) -> int:
        return self.manager.update_count

    @property
    def sample_count(self) -> int:
        return self.manager.sample_count

    @property
    def is_active(self) -> bool:
        return self.manager.is_active


@dataclass(slots=True)
class RollingErrorTracker:
    window_size: int
    recent_errors: deque[float] = field(init=False, repr=False)
    last_observed_step: int = -1

    def __post_init__(self) -> None:
        self.recent_errors = deque(maxlen=self.window_size)

    @property
    def has_history(self) -> bool:
        return bool(self.recent_errors)

    @property
    def mean_error(self) -> float:
        if not self.recent_errors:
            return float("nan")
        return float(np.mean(self.recent_errors))

    def observe(self, *, step_index: int, error: float) -> float:
        self.recent_errors.append(float(error))
        self.last_observed_step = step_index
        return self.mean_error

    def steps_since_last_observation(self, step_index: int) -> int:
        if self.last_observed_step < 0:
            return step_index + 1
        return step_index - self.last_observed_step


class SurrogateDecorator(ForecastProviderDecorator):
    """Trust-managed runtime decorator with lazy simulator execution."""

    def __init__(
        self,
        surrogate: ForecastProvider,
        simulator: ForecastProvider,
        controller: HybridController,
        rolling_window: int,
        warmup_steps: int,
    ) -> None:
        super().__init__(surrogate)
        self.surrogate = surrogate
        self.simulator = simulator
        self.controller = controller
        self.rolling_window = rolling_window
        self.warmup_steps = warmup_steps
        self.reset()

    def reset(self) -> None:
        self.tracker = RollingErrorTracker(window_size=self.rolling_window)
        self.selected_path_previous: str | None = None
        self.controller_mode = "simulation"
        self.cooldown_remaining = 0
        self.surrogate_validation_count = 0
        self.switch_count = 0

    def forecast(self, index: int) -> ForecastResult:
        rolling_error_before = self.tracker.mean_error
        steps_since_last_observation = self.tracker.steps_since_last_observation(index)
        cooldown_before = self.cooldown_remaining
        probe_executed = False
        observed_error = float("nan")
        surrogate_result: ForecastResult | None = None
        simulation_result: ForecastResult | None = None
        rolling_error_after = rolling_error_before
        reason = "simulation_hold"

        if index < self.warmup_steps:
            surrogate_result = self.surrogate.forecast(index)
            simulation_result = self.simulator.forecast(index)
            probe_executed = True
            observed_error = _mean_absolute_error(surrogate_result.value, simulation_result.value)
            rolling_error_after = self.tracker.observe(step_index=index, error=observed_error)
            selected_path = "simulation"
            reason = "warmup_probe"
            self.controller_mode = self.controller.decide(
                rolling_error_after,
                current_mode="simulation",
            )
        elif self.controller_mode == "surrogate":
            if self.controller.should_probe(
                current_mode=self.controller_mode,
                steps_since_last_observation=steps_since_last_observation,
                cooldown_remaining=self.cooldown_remaining,
                error_estimate=rolling_error_before,
                allow_relaxed_interval=self.surrogate_validation_count > 0,
            ):
                surrogate_result = self.surrogate.forecast(index)
                simulation_result = self.simulator.forecast(index)
                probe_executed = True
                observed_error = _mean_absolute_error(surrogate_result.value, simulation_result.value)
                rolling_error_after = self.tracker.observe(step_index=index, error=observed_error)
                if self.controller.should_fallback_to_simulation(rolling_error_after):
                    selected_path = "simulation"
                    reason = "validation_probe_fallback"
                    self.controller_mode = "simulation"
                    self.cooldown_remaining = self.controller.simulation_cooldown_steps
                else:
                    selected_path = "surrogate"
                    reason = "validation_probe_keep"
                self.surrogate_validation_count += 1
            else:
                surrogate_result = self.surrogate.forecast(index)
                selected_path = "surrogate"
                reason = "trusted_surrogate"
        elif self.cooldown_remaining > 0:
            simulation_result = self.simulator.forecast(index)
            selected_path = "simulation"
            reason = "simulation_cooldown"
            self.cooldown_remaining -= 1
        elif self.controller.should_probe(
            current_mode=self.controller_mode,
            steps_since_last_observation=steps_since_last_observation,
            cooldown_remaining=self.cooldown_remaining,
            error_estimate=rolling_error_before,
        ):
            surrogate_result = self.surrogate.forecast(index)
            simulation_result = self.simulator.forecast(index)
            probe_executed = True
            observed_error = _mean_absolute_error(surrogate_result.value, simulation_result.value)
            rolling_error_after = self.tracker.observe(step_index=index, error=observed_error)
            if self.controller.should_enter_surrogate(rolling_error_after):
                selected_path = "surrogate"
                reason = "reentry_probe_accept"
                self.controller_mode = "surrogate"
            else:
                selected_path = "simulation"
                reason = "reentry_probe_reject"
        else:
            simulation_result = self.simulator.forecast(index)
            selected_path = "simulation"
            reason = "simulation_hold"

        if surrogate_result is not None and simulation_result is not None:
            recalibration_updated = self._observe_trusted_label(
                step_index=index,
                surrogate_result=surrogate_result,
                simulation_result=simulation_result,
            )
        else:
            recalibration_updated = False

        if selected_path == "surrogate":
            if surrogate_result is None:
                surrogate_result = self.surrogate.forecast(index)
            selected_result = surrogate_result
        else:
            if simulation_result is None:
                simulation_result = self.simulator.forecast(index)
            selected_result = simulation_result

        if self.selected_path_previous is not None and self.selected_path_previous != selected_path:
            self.switch_count += 1
        self.selected_path_previous = selected_path

        metadata = {
            "index": index,
            "mode": selected_path,
            "controller_mode": self.controller_mode,
            "reason": reason,
            "probe_executed": probe_executed,
            "cooldown_remaining_before": float(cooldown_before),
            "cooldown_remaining_after": float(self.cooldown_remaining),
            "steps_since_last_observation": float(steps_since_last_observation),
            "entry_threshold": float(self.controller.entry_threshold),
            "exit_threshold": float(self.controller.exit_threshold),
            "threshold": float(self.controller.threshold),
            "rolling_error_before": rolling_error_before,
            "rolling_error_after": rolling_error_after,
            "observed_error": observed_error,
            "simulation_executed": simulation_result is not None,
            "surrogate_executed": surrogate_result is not None,
            "simulation_runtime_ms": (
                float(simulation_result.runtime_ms) if simulation_result is not None else 0.0
            ),
            "surrogate_runtime_ms": (
                float(surrogate_result.runtime_ms) if surrogate_result is not None else 0.0
            ),
            "surrogate_prediction": (
                surrogate_result.value.copy() if surrogate_result is not None else None
            ),
            "base_surrogate_prediction": (
                np.asarray(
                    surrogate_result.metadata.get("base_prediction", surrogate_result.value),
                    dtype=np.float32,
                ).copy()
                if surrogate_result is not None
                else None
            ),
            "recalibration_active": self._recalibration_active,
            "recalibration_updated": recalibration_updated,
            "recalibration_updates": float(self._recalibration_updates),
            "recalibration_sample_count": float(self._recalibration_sample_count),
            "switch_count": float(self.switch_count),
        }
        timestamp = (
            simulation_result.metadata.get("timestamp")
            if simulation_result is not None
            else selected_result.metadata.get("timestamp")
        )
        hour = (
            simulation_result.metadata.get("hour")
            if simulation_result is not None
            else selected_result.metadata.get("hour")
        )
        if timestamp is not None:
            metadata["timestamp"] = timestamp
        if hour is not None:
            metadata["hour"] = hour

        return ForecastResult(
            value=np.asarray(selected_result.value, dtype=np.float32),
            runtime_ms=float(
                (surrogate_result.runtime_ms if surrogate_result is not None else 0.0)
                + (simulation_result.runtime_ms if simulation_result is not None else 0.0)
            ),
            source=selected_path,
            metadata=metadata,
        )

    def _observe_trusted_label(
        self,
        *,
        step_index: int,
        surrogate_result: ForecastResult,
        simulation_result: ForecastResult,
    ) -> bool:
        observer = getattr(self.surrogate, "observe_trusted_label", None)
        if observer is None:
            return False
        base_prediction = np.asarray(
            surrogate_result.metadata.get("base_prediction", surrogate_result.value),
            dtype=np.float32,
        )
        return bool(
            observer(
                step_index=step_index,
                base_prediction=base_prediction,
                target=simulation_result.value,
            )
        )

    @property
    def _recalibration_updates(self) -> int:
        return int(getattr(self.surrogate, "update_count", 0))

    @property
    def _recalibration_sample_count(self) -> int:
        return int(getattr(self.surrogate, "sample_count", 0))

    @property
    def _recalibration_active(self) -> bool:
        return bool(getattr(self.surrogate, "is_active", False))


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


class DecoratorEvaluationRunner:
    """Experiment runner kept separate from the runtime decorator."""

    def __init__(
        self,
        *,
        provider: SurrogateDecorator,
        simulator: HighFidelitySimulationAdapter,
        surrogate: SurrogateModelAdapter,
        calibration_error: float,
    ) -> None:
        self.provider = provider
        self.simulator = simulator
        self.surrogate = surrogate
        self.calibration_error = calibration_error

    def run(
        self,
        *,
        model_name: str,
        threshold_multiplier: float,
    ) -> DecoratorRunResult:
        self.provider.reset()
        rows: list[dict[str, float | int | str | bool]] = []
        effective_outputs: list[np.ndarray] = []
        executed_surrogate_outputs: list[np.ndarray] = []
        executed_surrogate_targets: list[np.ndarray] = []
        effective_step_errors: list[float] = []
        observed_errors: list[float] = []
        surrogate_steps = 0
        simulation_steps = 0
        probe_steps = 0
        observed_steps = 0
        effective_runtime_ms = 0.0

        total_steps = len(self.simulator)
        for index in range(total_steps):
            result = self.provider.forecast(index)
            true_value = self.simulator.value_at(index)
            pure_surrogate_value = self.surrogate.value_at(index)
            metadata = result.metadata

            effective_error = _mean_absolute_error(result.value, true_value)
            base_surrogate_error = _mean_absolute_error(pure_surrogate_value, true_value)
            executed_surrogate_prediction = metadata.get("surrogate_prediction")
            if executed_surrogate_prediction is not None:
                surrogate_step_error = _mean_absolute_error(
                    np.asarray(executed_surrogate_prediction, dtype=np.float32),
                    true_value,
                )
                executed_surrogate_outputs.append(
                    np.asarray(executed_surrogate_prediction, dtype=np.float32).copy()
                )
                executed_surrogate_targets.append(true_value.copy())
            else:
                surrogate_step_error = float("nan")

            effective_runtime_ms += result.runtime_ms
            effective_step_errors.append(effective_error)
            effective_outputs.append(np.asarray(result.value, dtype=np.float32).copy())

            if result.source == "surrogate":
                surrogate_steps += 1
            else:
                simulation_steps += 1

            if bool(metadata["probe_executed"]):
                probe_steps += 1
            observed_error = float(metadata["observed_error"])
            if np.isfinite(observed_error):
                observed_steps += 1
                observed_errors.append(observed_error)

            rows.append(
                {
                    "index": index,
                    "timestamp": str(self.simulator.timestamps[index]),
                    "hour": float(self.simulator.hours[index]),
                    "mode": result.source,
                    "controller_mode": str(metadata["controller_mode"]),
                    "reason": str(metadata["reason"]),
                    "probe_executed": bool(metadata["probe_executed"]),
                    "simulation_executed": bool(metadata["simulation_executed"]),
                    "surrogate_executed": bool(metadata["surrogate_executed"]),
                    "cooldown_remaining_before": float(metadata["cooldown_remaining_before"]),
                    "cooldown_remaining_after": float(metadata["cooldown_remaining_after"]),
                    "steps_since_last_observation": float(
                        metadata["steps_since_last_observation"]
                    ),
                    "entry_threshold": float(metadata["entry_threshold"]),
                    "exit_threshold": float(metadata["exit_threshold"]),
                    "threshold": float(metadata["threshold"]),
                    "rolling_error_before": float(metadata["rolling_error_before"]),
                    "rolling_error_after": float(metadata["rolling_error_after"]),
                    "observed_error": observed_error,
                    "base_surrogate_step_error": base_surrogate_error,
                    "surrogate_step_error": surrogate_step_error,
                    "effective_step_error": effective_error,
                    "surrogate_runtime_ms": float(metadata["surrogate_runtime_ms"]),
                    "simulation_runtime_ms": float(metadata["simulation_runtime_ms"]),
                    "recalibration_active": bool(metadata["recalibration_active"]),
                    "recalibration_updated": bool(metadata["recalibration_updated"]),
                    "recalibration_sample_count": float(metadata["recalibration_sample_count"]),
                    "effective_runtime_ms": float(result.runtime_ms),
                }
            )

        y_true = np.asarray(self.simulator.y_true, dtype=np.float32)
        base_surrogate_array = np.asarray(self.surrogate.y_pred, dtype=np.float32)
        effective_output_array = np.asarray(effective_outputs, dtype=np.float32)
        simulator_total_runtime_ms = float(np.sum(self.simulator.runtime_ms))
        surrogate_total_runtime_ms = float(np.sum(self.surrogate.runtime_ms))
        pure_surrogate_mae = mae(y_true, base_surrogate_array)

        if executed_surrogate_outputs:
            managed_surrogate_array = np.asarray(executed_surrogate_outputs, dtype=np.float32)
            managed_surrogate_targets = np.asarray(executed_surrogate_targets, dtype=np.float32)
            managed_surrogate_mae = mae(managed_surrogate_targets, managed_surrogate_array)
            managed_surrogate_mape = mape(managed_surrogate_targets, managed_surrogate_array)
            managed_surrogate_smape = smape(managed_surrogate_targets, managed_surrogate_array)
            managed_surrogate_nmae = nmae(managed_surrogate_targets, managed_surrogate_array)
            managed_surrogate_nrmse = nrmse(managed_surrogate_targets, managed_surrogate_array)
        else:
            managed_surrogate_mae = float("nan")
            managed_surrogate_mape = float("nan")
            managed_surrogate_smape = float("nan")
            managed_surrogate_nmae = float("nan")
            managed_surrogate_nrmse = float("nan")

        decorator_mae = mae(y_true, effective_output_array)
        metrics = {
            "threshold_multiplier": float(threshold_multiplier),
            "calibration_mae": float(self.calibration_error),
            "pure_surrogate_mae": pure_surrogate_mae,
            "pure_surrogate_mape": mape(y_true, base_surrogate_array),
            "pure_surrogate_smape": smape(y_true, base_surrogate_array),
            "pure_surrogate_nmae": nmae(y_true, base_surrogate_array),
            "pure_surrogate_nrmse": nrmse(y_true, base_surrogate_array),
            "managed_surrogate_mae": managed_surrogate_mae,
            "managed_surrogate_mape": managed_surrogate_mape,
            "managed_surrogate_smape": managed_surrogate_smape,
            "managed_surrogate_nmae": managed_surrogate_nmae,
            "managed_surrogate_nrmse": managed_surrogate_nrmse,
            "managed_surrogate_evaluated_steps": float(len(executed_surrogate_outputs)),
            "decorator_mae": decorator_mae,
            "decorator_rmse": rmse(y_true, effective_output_array),
            "decorator_mape": mape(y_true, effective_output_array),
            "decorator_smape": smape(y_true, effective_output_array),
            "decorator_nmae": nmae(y_true, effective_output_array),
            "decorator_nrmse": nrmse(y_true, effective_output_array),
            "fallback_improvement": float(pure_surrogate_mae - decorator_mae),
            "surrogate_usage_ratio": float(surrogate_steps / total_steps),
            "simulation_usage_ratio": float(simulation_steps / total_steps),
            "observation_step_ratio": float(observed_steps / total_steps),
            "probe_step_ratio": float(probe_steps / total_steps),
            "switch_count": float(self.provider.switch_count),
            "avg_observed_error": float(np.mean(observed_errors)) if observed_errors else float("nan"),
            "avg_observed_rolling_error": self.provider.tracker.mean_error,
            "decorator_runtime_ms": float(effective_runtime_ms),
            "simulator_runtime_ms": simulator_total_runtime_ms,
            "pure_surrogate_runtime_ms": surrogate_total_runtime_ms,
            "decorator_speedup": float(simulator_total_runtime_ms / max(effective_runtime_ms, 1e-9)),
            "pure_surrogate_speedup": float(
                simulator_total_runtime_ms / max(surrogate_total_runtime_ms, 1e-9)
            ),
            "online_recalibration_enabled": float(
                isinstance(self.provider.surrogate, RecalibratingForecastDecorator)
            ),
            "recalibration_active_final": float(self.provider._recalibration_active),
            "recalibration_updates": float(self.provider._recalibration_updates),
            "recalibration_sample_count": float(self.provider._recalibration_sample_count),
        }
        return DecoratorRunResult(
            model_name=model_name,
            threshold=float(self.provider.controller.threshold),
            threshold_multiplier=float(threshold_multiplier),
            metrics=metrics,
            trace=pd.DataFrame(rows),
        )


def evaluate_decorator_thresholds(
    artifacts: ModelArtifacts,
    calibration_split: DatasetSplit,
    test_split: DatasetSplit,
    normalization: NormalizationStats,
    config: ExperimentConfig,
) -> DecoratorEvaluationArtifacts:
    y_calibration = _denormalize(calibration_split.y, normalization)
    calibration_predictions = _collect_surrogate_predictions(
        artifacts=artifacts,
        X=calibration_split.X,
        normalization=normalization,
        device=config.training.device,
        batch_size=config.training.batch_size,
    )
    calibration_mae = mae(y_calibration, calibration_predictions)
    thresholds = _build_threshold_schedule(
        base_error=calibration_mae,
        multipliers=config.decorator.threshold_multipliers,
    )

    y_true = _denormalize(test_split.y, normalization)
    surrogate_predictions, surrogate_runtimes = _collect_streaming_surrogate_outputs(
        artifacts=artifacts,
        X=test_split.X,
        normalization=normalization,
        device=config.training.device,
        batch_size=config.training.batch_size,
        config=config,
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
        runtime_surrogate: ForecastProvider = surrogate_adapter
        if config.decorator.enable_online_recalibration:
            runtime_surrogate = RecalibratingForecastDecorator(
                runtime_surrogate,
                min_samples=config.decorator.recalibration_min_samples,
                update_interval_steps=config.decorator.recalibration_interval_steps,
                max_samples=config.decorator.recalibration_max_samples,
                ridge=config.decorator.recalibration_ridge,
            )

        decorator = SurrogateDecorator(
            surrogate=runtime_surrogate,
            simulator=simulator_adapter,
            controller=HybridController(
                threshold=threshold,
                hysteresis_ratio=config.decorator.hysteresis_ratio,
                validation_interval_steps=config.decorator.validation_interval_steps,
                max_validation_interval_steps=config.decorator.max_validation_interval_steps,
                simulation_cooldown_steps=config.decorator.simulation_cooldown_steps,
                reentry_probe_interval_steps=config.decorator.reentry_probe_interval_steps,
            ),
            rolling_window=config.decorator.rolling_window,
            warmup_steps=config.decorator.warmup_steps,
        )
        runner = DecoratorEvaluationRunner(
            provider=decorator,
            simulator=simulator_adapter,
            surrogate=surrogate_adapter,
            calibration_error=calibration_mae,
        )
        results.append(
            runner.run(
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


def _collect_surrogate_predictions(
    artifacts: ModelArtifacts,
    X: np.ndarray,
    normalization: NormalizationStats,
    device: str,
    batch_size: int,
) -> np.ndarray:
    predictions_norm = predict_model(
        artifacts=artifacts,
        X=X,
        device=device,
        batch_size=batch_size,
    )
    return _denormalize(np.asarray(predictions_norm, dtype=np.float32), normalization)


def _collect_streaming_surrogate_outputs(
    artifacts: ModelArtifacts,
    X: np.ndarray,
    normalization: NormalizationStats,
    device: str,
    batch_size: int,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = _collect_surrogate_predictions(
        artifacts=artifacts,
        X=X,
        normalization=normalization,
        device=device,
        batch_size=batch_size,
    )
    streaming_latency_ms = _measure_streaming_step_latency_ms(
        artifacts=artifacts,
        X=X,
        device=device,
        thread_count=config.decorator.streaming_inference_threads,
        warmup_calls=config.decorator.streaming_warmup_calls,
        enable_jit_optimization=config.decorator.enable_jit_streaming_optimization,
    )
    runtimes_ms = np.full(len(predictions), streaming_latency_ms, dtype=np.float32)
    return predictions, runtimes_ms


def _measure_streaming_step_latency_ms(
    artifacts: ModelArtifacts,
    X: np.ndarray,
    device: str,
    thread_count: int,
    warmup_calls: int,
    enable_jit_optimization: bool,
    max_samples: int = 96,
) -> float:
    if len(X) == 0:
        return 0.0

    sample_count = min(len(X), max_samples)
    sample_batch = X[:sample_count]

    if hasattr(artifacts.model, "predict"):
        start = perf_counter()
        artifacts.model.predict(sample_batch)
        return float(((perf_counter() - start) * 1000) / sample_count)

    model = artifacts.model
    model.eval()
    torch_device = torch.device(device)
    model.to(torch_device)
    features = torch.tensor(sample_batch, dtype=torch.float32, device=torch_device)
    runtime_model = _prepare_streaming_torch_model(
        model=model,
        example_input=features[:1],
        enable_jit_optimization=enable_jit_optimization,
    )

    with _temporary_torch_num_threads(thread_count), torch.inference_mode():
        for sample in features[: min(sample_count, max(warmup_calls, 1))]:
            runtime_model(sample.unsqueeze(0))

        start = perf_counter()
        for sample in features:
            runtime_model(sample.unsqueeze(0))
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)

    return float(((perf_counter() - start) * 1000) / sample_count)


def _prepare_streaming_torch_model(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    enable_jit_optimization: bool,
) -> torch.nn.Module:
    runtime_model = model
    if not enable_jit_optimization:
        return runtime_model

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
            traced = torch.jit.trace(runtime_model, example_input, check_trace=False)
        return torch.jit.optimize_for_inference(traced)
    except Exception:
        return runtime_model


@contextmanager
def _temporary_torch_num_threads(thread_count: int):
    if thread_count <= 0:
        yield
        return

    original_threads = torch.get_num_threads()
    if original_threads == thread_count:
        yield
        return

    torch.set_num_threads(thread_count)
    try:
        yield
    finally:
        torch.set_num_threads(original_threads)


def _build_threshold_schedule(
    base_error: float,
    multipliers: list[float],
) -> list[tuple[float, float]]:
    thresholds = []
    for multiplier in multipliers:
        thresholds.append((float(multiplier), float(base_error * multiplier)))
    return thresholds


def _denormalize(values: np.ndarray, normalization: NormalizationStats) -> np.ndarray:
    return values * normalization.target_std + normalization.target_mean


def _mean_absolute_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32))))
