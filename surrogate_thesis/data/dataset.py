from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from surrogate_thesis.config import DatasetConfig


@dataclass(slots=True)
class WindowedDataset:
    X: np.ndarray
    y: np.ndarray
    target_timestamps: np.ndarray
    target_hours: np.ndarray
    reference_runtime_ms: np.ndarray
    feature_columns: list[str]
    target_column: str


@dataclass(slots=True)
class DatasetSplit:
    X: np.ndarray
    y: np.ndarray
    target_timestamps: np.ndarray
    target_hours: np.ndarray
    reference_runtime_ms: np.ndarray


@dataclass(slots=True)
class NormalizationStats:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
        }


@dataclass(slots=True)
class PreparedDataset:
    raw_frame: pd.DataFrame
    feature_columns: list[str]
    target_column: str
    normalization: NormalizationStats
    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    timestamps = pd.to_datetime(enriched["timestamp"])
    hour_of_day = timestamps.dt.hour + timestamps.dt.minute / 60
    enriched["hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24)
    enriched["hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24)
    enriched["is_weekend"] = timestamps.dt.dayofweek.ge(5).astype(float)
    return enriched


def build_windows(frame: pd.DataFrame, config: DatasetConfig) -> WindowedDataset:
    feature_values = frame[config.feature_columns].to_numpy(dtype=np.float32)
    target_values = frame[config.target_column].to_numpy(dtype=np.float32)
    timestamps = pd.to_datetime(frame["timestamp"]).to_numpy()
    hours = frame["hour"].to_numpy(dtype=np.float32)
    runtimes = frame["runtime_ms"].to_numpy(dtype=np.float32)

    X, y = [], []
    target_timestamps, target_hours, reference_runtime_ms = [], [], []

    max_start = len(frame) - config.lookback - config.horizon + 1
    for start in range(max_start):
        window_end = start + config.lookback
        target_end = window_end + config.horizon
        X.append(feature_values[start:window_end])
        y.append(target_values[window_end:target_end])
        target_index = target_end - 1
        target_timestamps.append(timestamps[target_index])
        target_hours.append(hours[target_index])
        reference_runtime_ms.append(runtimes[target_index])

    return WindowedDataset(
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.float32),
        target_timestamps=np.asarray(target_timestamps),
        target_hours=np.asarray(target_hours, dtype=np.float32),
        reference_runtime_ms=np.asarray(reference_runtime_ms, dtype=np.float32),
        feature_columns=list(config.feature_columns),
        target_column=config.target_column,
    )


def prepare_dataset(frame: pd.DataFrame, config: DatasetConfig) -> PreparedDataset:
    if not np.isclose(config.train_ratio + config.val_ratio + config.test_ratio, 1.0):
        raise ValueError("Train/validation/test ratios must sum to 1.0.")

    enriched = add_time_features(frame)
    windowed = build_windows(enriched, config)

    n_samples = len(windowed.X)
    train_end = int(n_samples * config.train_ratio)
    val_end = train_end + int(n_samples * config.val_ratio)

    train = _make_split(windowed, 0, train_end)
    val = _make_split(windowed, train_end, val_end)
    test = _make_split(windowed, val_end, n_samples)

    normalization = _fit_normalization(train=train, feature_columns=windowed.feature_columns, target_column=windowed.target_column)
    train = _normalize_split(train, normalization)
    val = _normalize_split(val, normalization)
    test = _normalize_split(test, normalization)

    return PreparedDataset(
        raw_frame=enriched,
        feature_columns=windowed.feature_columns,
        target_column=windowed.target_column,
        normalization=normalization,
        train=train,
        val=val,
        test=test,
    )


def save_dataset_artifacts(dataset: PreparedDataset, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset.raw_frame.to_csv(output_path / "raw_simulation.csv", index=False)
    (output_path / "normalization.json").write_text(
        json.dumps(dataset.normalization.to_dict(), indent=2)
    )
    _save_split(dataset.train, output_path / "train.npz")
    _save_split(dataset.val, output_path / "val.npz")
    _save_split(dataset.test, output_path / "test.npz")


def _make_split(windowed: WindowedDataset, start: int, end: int) -> DatasetSplit:
    return DatasetSplit(
        X=windowed.X[start:end].copy(),
        y=windowed.y[start:end].copy(),
        target_timestamps=windowed.target_timestamps[start:end].copy(),
        target_hours=windowed.target_hours[start:end].copy(),
        reference_runtime_ms=windowed.reference_runtime_ms[start:end].copy(),
    )


def _fit_normalization(
    train: DatasetSplit, feature_columns: list[str], target_column: str
) -> NormalizationStats:
    feature_mean = train.X.mean(axis=(0, 1))
    feature_std = train.X.std(axis=(0, 1))
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

    if target_column in feature_columns:
        target_index = feature_columns.index(target_column)
        target_mean = np.asarray([feature_mean[target_index]], dtype=np.float32)
        target_std = np.asarray([feature_std[target_index]], dtype=np.float32)
    else:
        target_mean = train.y.mean(axis=0)
        target_std = train.y.std(axis=0)
        target_std = np.where(target_std < 1e-6, 1.0, target_std)

    return NormalizationStats(
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        target_mean=target_mean.astype(np.float32),
        target_std=np.asarray(target_std, dtype=np.float32),
    )


def _normalize_split(split: DatasetSplit, normalization: NormalizationStats) -> DatasetSplit:
    X = (split.X - normalization.feature_mean) / normalization.feature_std
    y = (split.y - normalization.target_mean) / normalization.target_std
    return DatasetSplit(
        X=X.astype(np.float32),
        y=y.astype(np.float32),
        target_timestamps=split.target_timestamps,
        target_hours=split.target_hours,
        reference_runtime_ms=split.reference_runtime_ms,
    )


def _save_split(split: DatasetSplit, path: Path) -> None:
    np.savez_compressed(
        path,
        X=split.X,
        y=split.y,
        target_timestamps=split.target_timestamps.astype("datetime64[ns]").astype(str),
        target_hours=split.target_hours,
        reference_runtime_ms=split.reference_runtime_ms,
    )

