from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from surrogate_thesis.config import ExperimentConfig
from surrogate_thesis.data.dataset import DatasetSplit, NormalizationStats
from surrogate_thesis.evaluation.metrics import mae, rmse, smape
from surrogate_thesis.training import ModelArtifacts, predict_model


@dataclass(slots=True)
class EvaluationResult:
    model_name: str
    metrics: dict[str, float]
    predictions: np.ndarray
    y_true: np.ndarray
    timestamps: np.ndarray
    hours: np.ndarray

    def to_record(self) -> dict[str, float | str]:
        return {"model_name": self.model_name, **self.metrics}


def evaluate_model(
    reference_simulator: object,
    artifacts: ModelArtifacts,
    test_split: DatasetSplit,
    normalization: NormalizationStats,
    config: ExperimentConfig,
) -> EvaluationResult:
    del reference_simulator  # kept for interface clarity and future extensions

    predictions_norm = predict_model(
        artifacts=artifacts,
        X=test_split.X,
        device=config.training.device,
        batch_size=config.training.batch_size,
    )
    y_true = _denormalize(test_split.y, normalization)
    y_pred = _denormalize(predictions_norm, normalization)

    single_sample_latency_ms = _measure_single_sample_latency_ms(
        artifacts=artifacts,
        X=test_split.X[: config.evaluation.latency_samples],
        device=config.training.device,
    )
    batch_metrics = _measure_batched_runtime_ms(
        artifacts=artifacts,
        X=test_split.X[: config.evaluation.latency_samples],
        device=config.training.device,
        batch_size=config.training.batch_size,
    )
    full_test_runtime_ms = _measure_full_test_runtime_ms(
        artifacts=artifacts,
        X=test_split.X,
        device=config.training.device,
        batch_size=config.training.batch_size,
    )
    simulator_latency_ms = float(np.mean(test_split.reference_runtime_ms))
    simulator_full_test_runtime_ms = float(np.sum(test_split.reference_runtime_ms))
    metrics = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "latency_ms": batch_metrics["per_sample_latency_ms"],
        "single_sample_latency_ms": single_sample_latency_ms,
        "batch_latency_ms": batch_metrics["per_sample_latency_ms"],
        "batch_wall_time_ms": batch_metrics["wall_time_ms"],
        "batch_size_used": float(batch_metrics["batch_size"]),
        "simulator_latency_ms": simulator_latency_ms,
        "full_test_runtime_ms": full_test_runtime_ms,
        "simulator_full_test_runtime_ms": simulator_full_test_runtime_ms,
        "speedup": float(simulator_full_test_runtime_ms / max(full_test_runtime_ms, 1e-9)),
        "single_step_speedup": float(simulator_latency_ms / max(single_sample_latency_ms, 1e-9)),
        "training_seconds": artifacts.training_seconds,
    }
    return EvaluationResult(
        model_name=artifacts.name,
        metrics=metrics,
        predictions=y_pred,
        y_true=y_true,
        timestamps=test_split.target_timestamps,
        hours=test_split.target_hours,
    )


def _denormalize(values: np.ndarray, normalization: NormalizationStats) -> np.ndarray:
    return values * normalization.target_std + normalization.target_mean


def _measure_single_sample_latency_ms(
    artifacts: ModelArtifacts,
    X: np.ndarray,
    device: str,
) -> float:
    if len(X) == 0:
        return 0.0

    latencies = []
    for sample in X:
        start = perf_counter()
        predict_model(
            artifacts=artifacts,
            X=sample[np.newaxis, ...],
            device=device,
            batch_size=1,
        )
        latencies.append((perf_counter() - start) * 1000)
    return float(np.mean(latencies))


def _measure_batched_runtime_ms(
    artifacts: ModelArtifacts,
    X: np.ndarray,
    device: str,
    batch_size: int,
) -> dict[str, float]:
    if len(X) == 0:
        return {"wall_time_ms": 0.0, "per_sample_latency_ms": 0.0, "batch_size": float(batch_size)}

    start = perf_counter()
    predict_model(
        artifacts=artifacts,
        X=X,
        device=device,
        batch_size=batch_size,
    )
    wall_time_ms = (perf_counter() - start) * 1000
    return {
        "wall_time_ms": float(wall_time_ms),
        "per_sample_latency_ms": float(wall_time_ms / len(X)),
        "batch_size": float(batch_size),
    }


def _measure_full_test_runtime_ms(
    artifacts: ModelArtifacts,
    X: np.ndarray,
    device: str,
    batch_size: int,
) -> float:
    if len(X) == 0:
        return 0.0

    start = perf_counter()
    predict_model(
        artifacts=artifacts,
        X=X,
        device=device,
        batch_size=batch_size,
    )
    return float((perf_counter() - start) * 1000)
