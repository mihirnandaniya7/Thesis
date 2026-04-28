from __future__ import annotations

from contextlib import contextmanager
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
import torch

from surrogate_thesis.config import ExperimentConfig
from surrogate_thesis.data.dataset import DatasetSplit, NormalizationStats
from surrogate_thesis.evaluation.metrics import mae
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
    """Adaptive runtime wrapper with warmup, probing, and cooldown-based fallback."""

    def __init__(
        self,
        simulator: HighFidelitySimulationAdapter,
        surrogate: SurrogateModelAdapter,
        controller: HybridController,
        rolling_window: int,
        warmup_steps: int,
        enable_online_recalibration: bool = False,
        recalibration_min_samples: int = 8,
        recalibration_interval_steps: int = 12,
        recalibration_max_samples: int = 256,
        recalibration_ridge: float = 1e-4,
    ) -> None:
        self.simulator = simulator
        self.surrogate = surrogate
        self.controller = controller
        self.rolling_window = rolling_window
        self.warmup_steps = warmup_steps
        self.enable_online_recalibration = enable_online_recalibration
        self.recalibration_min_samples = recalibration_min_samples
        self.recalibration_interval_steps = recalibration_interval_steps
        self.recalibration_max_samples = recalibration_max_samples
        self.recalibration_ridge = recalibration_ridge

    def run(
        self,
        model_name: str,
        threshold_multiplier: float,
    ) -> DecoratorRunResult:
        tracker = RollingErrorTracker(window_size=self.rolling_window)
        recalibration_manager = (
            OnlineRecalibrationManager(
                min_samples=self.recalibration_min_samples,
                update_interval_steps=self.recalibration_interval_steps,
                max_samples=self.recalibration_max_samples,
                ridge=self.recalibration_ridge,
            )
            if self.enable_online_recalibration
            else None
        )
        rows: list[dict[str, float | int | str | bool]] = []
        selected_path_previous: str | None = None
        controller_mode = "simulation"
        cooldown_remaining = 0
        surrogate_steps = 0
        simulation_steps = 0
        probe_steps = 0
        observed_steps = 0
        switch_count = 0
        effective_runtime_ms = 0.0
        effective_step_errors: list[float] = []
        base_surrogate_step_errors: list[float] = []
        surrogate_step_errors: list[float] = []
        observed_errors: list[float] = []

        total_steps = len(self.surrogate.y_pred)
        for index in range(total_steps):
            base_surrogate_prediction, surrogate_runtime_ms = self.surrogate.forecast(index)
            true_value, simulation_runtime_ms = self.simulator.forecast(index)
            surrogate_prediction = (
                recalibration_manager.transform(base_surrogate_prediction)
                if recalibration_manager is not None
                else base_surrogate_prediction
            )
            base_surrogate_error = float(np.mean(np.abs(base_surrogate_prediction - true_value)))
            surrogate_error = float(np.mean(np.abs(surrogate_prediction - true_value)))
            base_surrogate_step_errors.append(base_surrogate_error)
            surrogate_step_errors.append(surrogate_error)

            rolling_error_before = tracker.mean_error
            steps_since_last_observation = tracker.steps_since_last_observation(index)
            probe_executed = False
            observed_error = float("nan")
            cooldown_before = cooldown_remaining
            recalibration_updated = False

            if index < self.warmup_steps:
                selected_path = "simulation"
                reason = "warmup_probe"
                probe_executed = True
                observed_error = surrogate_error
                rolling_error_after = tracker.observe(step_index=index, error=surrogate_error)
                observed_steps += 1
                probe_steps += 1
                effective_runtime_ms += simulation_runtime_ms + surrogate_runtime_ms
                controller_mode = self.controller.decide(rolling_error_after, current_mode="simulation")
            elif controller_mode == "surrogate":
                if self.controller.should_probe(
                    current_mode=controller_mode,
                    steps_since_last_observation=steps_since_last_observation,
                    cooldown_remaining=cooldown_remaining,
                ):
                    probe_executed = True
                    observed_error = surrogate_error
                    rolling_error_after = tracker.observe(step_index=index, error=surrogate_error)
                    observed_steps += 1
                    probe_steps += 1
                    effective_runtime_ms += simulation_runtime_ms + surrogate_runtime_ms
                    if self.controller.should_fallback_to_simulation(rolling_error_after):
                        selected_path = "simulation"
                        reason = "validation_probe_fallback"
                        controller_mode = "simulation"
                        cooldown_remaining = self.controller.simulation_cooldown_steps
                    else:
                        selected_path = "surrogate"
                        reason = "validation_probe_keep"
                else:
                    selected_path = "surrogate"
                    reason = "trusted_surrogate"
                    rolling_error_after = rolling_error_before
                    effective_runtime_ms += surrogate_runtime_ms
            else:
                if cooldown_remaining > 0:
                    selected_path = "simulation"
                    reason = "simulation_cooldown"
                    rolling_error_after = rolling_error_before
                    effective_runtime_ms += simulation_runtime_ms
                    cooldown_remaining -= 1
                elif self.controller.should_probe(
                    current_mode=controller_mode,
                    steps_since_last_observation=steps_since_last_observation,
                    cooldown_remaining=cooldown_remaining,
                ):
                    probe_executed = True
                    observed_error = surrogate_error
                    rolling_error_after = tracker.observe(step_index=index, error=surrogate_error)
                    observed_steps += 1
                    probe_steps += 1
                    effective_runtime_ms += simulation_runtime_ms + surrogate_runtime_ms
                    if self.controller.should_enter_surrogate(rolling_error_after):
                        selected_path = "surrogate"
                        reason = "reentry_probe_accept"
                        controller_mode = "surrogate"
                    else:
                        selected_path = "simulation"
                        reason = "reentry_probe_reject"
                else:
                    selected_path = "simulation"
                    reason = "simulation_hold"
                    rolling_error_after = rolling_error_before
                    effective_runtime_ms += simulation_runtime_ms

            if selected_path == "surrogate":
                output = surrogate_prediction
                surrogate_steps += 1
            else:
                output = true_value
                simulation_steps += 1

            if recalibration_manager is not None and (probe_executed or selected_path == "simulation"):
                recalibration_updated = recalibration_manager.observe(
                    step_index=index,
                    base_prediction=base_surrogate_prediction,
                    target=true_value,
                )

            if selected_path_previous is not None and selected_path_previous != selected_path:
                switch_count += 1
            selected_path_previous = selected_path

            effective_error = float(np.mean(np.abs(output - true_value)))
            effective_step_errors.append(effective_error)
            if not np.isnan(observed_error):
                observed_errors.append(observed_error)

            rows.append(
                {
                    "index": index,
                    "timestamp": str(self.simulator.timestamps[index]),
                    "hour": float(self.simulator.hours[index]),
                    "mode": selected_path,
                    "controller_mode": controller_mode,
                    "reason": reason,
                    "probe_executed": probe_executed,
                    "cooldown_remaining_before": float(cooldown_before),
                    "cooldown_remaining_after": float(cooldown_remaining),
                    "steps_since_last_observation": float(steps_since_last_observation),
                    "entry_threshold": float(self.controller.entry_threshold),
                    "exit_threshold": float(self.controller.exit_threshold),
                    "threshold": float(self.controller.threshold),
                    "rolling_error_before": rolling_error_before,
                    "rolling_error_after": rolling_error_after,
                    "observed_error": observed_error,
                    "base_surrogate_step_error": base_surrogate_error,
                    "surrogate_step_error": surrogate_error,
                    "effective_step_error": effective_error,
                    "surrogate_runtime_ms": surrogate_runtime_ms,
                    "simulation_runtime_ms": simulation_runtime_ms,
                    "recalibration_active": bool(
                        recalibration_manager.is_active if recalibration_manager is not None else False
                    ),
                    "recalibration_updated": recalibration_updated,
                    "recalibration_sample_count": float(
                        recalibration_manager.sample_count if recalibration_manager is not None else 0
                    ),
                    "effective_runtime_ms": float(
                        simulation_runtime_ms + surrogate_runtime_ms
                        if probe_executed
                        else (surrogate_runtime_ms if selected_path == "surrogate" else simulation_runtime_ms)
                    ),
                }
            )

        simulator_total_runtime_ms = float(np.sum(self.simulator.runtime_ms))
        surrogate_total_runtime_ms = float(np.sum(self.surrogate.runtime_ms))
        pure_surrogate_mae = float(np.mean(base_surrogate_step_errors))
        managed_surrogate_mae = float(np.mean(surrogate_step_errors))
        metrics = {
            "threshold_multiplier": float(threshold_multiplier),
            "pure_surrogate_mae": pure_surrogate_mae,
            "managed_surrogate_mae": managed_surrogate_mae,
            "decorator_mae": float(np.mean(effective_step_errors)),
            "decorator_rmse": float(np.sqrt(np.mean(np.square(effective_step_errors)))),
            "recalibration_improvement": float(pure_surrogate_mae - managed_surrogate_mae),
            "fallback_improvement": float(managed_surrogate_mae - np.mean(effective_step_errors)),
            "surrogate_usage_ratio": float(surrogate_steps / total_steps),
            "simulation_usage_ratio": float(simulation_steps / total_steps),
            "observation_step_ratio": float(observed_steps / total_steps),
            "probe_step_ratio": float(probe_steps / total_steps),
            "switch_count": float(switch_count),
            "avg_observed_error": float(np.mean(observed_errors)) if observed_errors else float("nan"),
            "avg_observed_rolling_error": tracker.mean_error,
            "decorator_runtime_ms": float(effective_runtime_ms),
            "simulator_runtime_ms": simulator_total_runtime_ms,
            "pure_surrogate_runtime_ms": surrogate_total_runtime_ms,
            "decorator_speedup": float(simulator_total_runtime_ms / max(effective_runtime_ms, 1e-9)),
            "pure_surrogate_speedup": float(
                simulator_total_runtime_ms / max(surrogate_total_runtime_ms, 1e-9)
            ),
            "online_recalibration_enabled": float(bool(recalibration_manager is not None)),
            "recalibration_active_final": float(
                recalibration_manager.is_active if recalibration_manager is not None else False
            ),
            "recalibration_updates": float(
                recalibration_manager.update_count if recalibration_manager is not None else 0
            ),
            "recalibration_sample_count": float(
                recalibration_manager.sample_count if recalibration_manager is not None else 0
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
    surrogate_predictions, surrogate_runtimes = _collect_streaming_surrogate_outputs(
        artifacts=artifacts,
        X=test_split.X,
        normalization=normalization,
        device=config.training.device,
        batch_size=config.training.batch_size,
        config=config,
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
            controller=HybridController(
                threshold=threshold,
                hysteresis_ratio=config.decorator.hysteresis_ratio,
                validation_interval_steps=config.decorator.validation_interval_steps,
                simulation_cooldown_steps=config.decorator.simulation_cooldown_steps,
                reentry_probe_interval_steps=config.decorator.reentry_probe_interval_steps,
            ),
            rolling_window=config.decorator.rolling_window,
            warmup_steps=config.decorator.warmup_steps,
            enable_online_recalibration=config.decorator.enable_online_recalibration,
            recalibration_min_samples=config.decorator.recalibration_min_samples,
            recalibration_interval_steps=config.decorator.recalibration_interval_steps,
            recalibration_max_samples=config.decorator.recalibration_max_samples,
            recalibration_ridge=config.decorator.recalibration_ridge,
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


def _collect_streaming_surrogate_outputs(
    artifacts: ModelArtifacts,
    X: np.ndarray,
    normalization: NormalizationStats,
    device: str,
    batch_size: int,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    predictions_norm = predict_model(
        artifacts=artifacts,
        X=X,
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
    predictions_array = np.asarray(predictions_norm, dtype=np.float32)
    runtimes_ms = np.full(len(predictions_array), streaming_latency_ms, dtype=np.float32)
    return _denormalize(predictions_array, normalization), runtimes_ms


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
