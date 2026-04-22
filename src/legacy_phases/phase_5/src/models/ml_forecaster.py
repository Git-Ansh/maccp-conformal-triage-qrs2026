"""
Phase 5: Machine Learning Forecasting Models
Random Forest and other ML-based approaches.
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from .base import BaseForecaster


def create_lag_features(y: np.ndarray, n_lags: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create lag features for supervised forecasting.

    Args:
        y: Time series values
        n_lags: Number of lag features

    Returns:
        Tuple of (X features, y targets)
    """
    n = len(y)
    if n <= n_lags:
        return np.array([]), np.array([])

    X = []
    targets = []

    for i in range(n_lags, n):
        X.append(y[i-n_lags:i])
        targets.append(y[i])

    return np.array(X), np.array(targets)


class RandomForestForecaster(BaseForecaster):
    """Random Forest regression for time series forecasting."""

    def __init__(
        self,
        n_lags: int = 10,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42
    ):
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model_ = None
        self.last_values_ = None

    @property
    def name(self) -> str:
        return f"RF_lag{self.n_lags}"

    def fit(self, y: np.ndarray) -> 'RandomForestForecaster':
        y = np.asarray(y).flatten()

        X, targets = create_lag_features(y, self.n_lags)

        if len(X) < 5:
            # Not enough data, use naive approach
            self.model_ = None
            self.last_values_ = y[-self.n_lags:] if len(y) >= self.n_lags else y
            return self

        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model_.fit(X, targets)
        self.last_values_ = y[-self.n_lags:]

        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self.last_values_ is None:
            raise ValueError("Model not fitted")

        if self.model_ is None:
            # Naive fallback
            return np.full(horizon, self.last_values_[-1])

        predictions = []
        current = self.last_values_.copy()

        for _ in range(horizon):
            X = current[-self.n_lags:].reshape(1, -1)
            pred = self.model_.predict(X)[0]
            predictions.append(pred)
            current = np.append(current, pred)

        return np.array(predictions)


class GradientBoostingForecaster(BaseForecaster):
    """Gradient Boosting regression for time series forecasting."""

    def __init__(
        self,
        n_lags: int = 10,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        random_state: int = 42
    ):
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model_ = None
        self.last_values_ = None

    @property
    def name(self) -> str:
        return f"GBR_lag{self.n_lags}"

    def fit(self, y: np.ndarray) -> 'GradientBoostingForecaster':
        y = np.asarray(y).flatten()

        X, targets = create_lag_features(y, self.n_lags)

        if len(X) < 5:
            self.model_ = None
            self.last_values_ = y[-self.n_lags:] if len(y) >= self.n_lags else y
            return self

        self.model_ = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state
        )
        self.model_.fit(X, targets)
        self.last_values_ = y[-self.n_lags:]

        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self.last_values_ is None:
            raise ValueError("Model not fitted")

        if self.model_ is None:
            return np.full(horizon, self.last_values_[-1])

        predictions = []
        current = self.last_values_.copy()

        for _ in range(horizon):
            X = current[-self.n_lags:].reshape(1, -1)
            pred = self.model_.predict(X)[0]
            predictions.append(pred)
            current = np.append(current, pred)

        return np.array(predictions)


if __name__ == "__main__":
    # Test ML forecasters
    np.random.seed(42)
    y = np.cumsum(np.random.normal(0, 1, 50)) + 100

    print("Testing ML forecasters...")

    for Model in [RandomForestForecaster, GradientBoostingForecaster]:
        model = Model()
        model.fit(y)
        pred = model.predict(5)
        print(f"  {model.name}: {pred}")
