"""Forecast error metrics used in experiment tables and decorator analysis."""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error in the target unit."""

    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error in the target unit."""

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error with zero-safe denominators."""

    safe_denominator = np.where(np.abs(y_true) < 1e-6, 1.0, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / safe_denominator)) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric mean absolute percentage error."""

    denominator = np.abs(y_true) + np.abs(y_pred)
    safe_denominator = np.where(denominator < 1e-6, 1.0, denominator)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / safe_denominator) * 100.0)


def nmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE normalized by the mean absolute target magnitude."""

    scale = _mean_absolute_scale(y_true)
    return float(np.mean(np.abs(y_true - y_pred)) / scale * 100.0)


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE normalized by the mean absolute target magnitude."""

    scale = _mean_absolute_scale(y_true)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / scale * 100.0)


def _mean_absolute_scale(y_true: np.ndarray) -> float:
    """Return a nonzero scale for normalized error metrics."""

    scale = float(np.mean(np.abs(y_true)))
    return max(scale, 1e-6)
