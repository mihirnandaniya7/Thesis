from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


class PersistenceBaseline:
    """Predicts the next value as the most recent observed load."""

    def __init__(self, target_feature_index: int = 0) -> None:
        self.target_feature_index = target_feature_index

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PersistenceBaseline":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, -1, self.target_feature_index : self.target_feature_index + 1]


class LinearRegressionBaseline:
    """Flattened linear regression baseline."""

    def __init__(self) -> None:
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionBaseline":
        self.model.fit(X.reshape(len(X), -1), y.reshape(len(y), -1))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = self.model.predict(X.reshape(len(X), -1))
        return np.asarray(predictions, dtype=np.float32)

