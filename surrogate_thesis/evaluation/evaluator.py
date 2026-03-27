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

    model_latency_ms = _measure_latency_ms(
        artifacts=artifacts,
        X=test_split.X[: config.evaluation.latency_samples],
        device=config.training.device,
    )
    simulator_latency_ms = float(np.mean(test_split.reference_runtime_ms))
    metrics = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "latency_ms": model_latency_ms,
        "simulator_latency_ms": simulator_latency_ms,
        "speedup": float(simulator_latency_ms / max(model_latency_ms, 1e-9)),
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


def _measure_latency_ms(
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

