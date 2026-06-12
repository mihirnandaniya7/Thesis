"""Classical baseline models used for surrogate comparison."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


class PersistenceBaseline:
    """Predict the future target as the most recent observed target value."""

    def __init__(self, target_feature_index: int = 0, horizon: int = 1) -> None:
        self.target_feature_index = target_feature_index
        self.horizon = horizon

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PersistenceBaseline":
        """Keep a fit method so the baseline matches sklearn-style models."""

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Repeat the last target feature across the configured horizon."""

        last_value = X[:, -1, self.target_feature_index : self.target_feature_index + 1]
        return np.repeat(last_value, repeats=self.horizon, axis=1)


class LinearRegressionBaseline:
    """Linear regression baseline on flattened lookback windows."""

    def __init__(self) -> None:
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionBaseline":
        """Fit a linear map from the full window to the forecast horizon."""

        self.model.fit(X.reshape(len(X), -1), y.reshape(len(y), -1))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict and return float32 arrays compatible with neural outputs."""

        predictions = self.model.predict(X.reshape(len(X), -1))
        return np.asarray(predictions, dtype=np.float32)
