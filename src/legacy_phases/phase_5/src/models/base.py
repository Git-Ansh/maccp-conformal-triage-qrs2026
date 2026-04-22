"""
Phase 5: Base Forecasting Model Interface
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class BaseForecaster(ABC):
    """Base class for all forecasting models."""

    @abstractmethod
    def fit(self, y: np.ndarray) -> 'BaseForecaster':
        """Fit the model to historical data."""
        pass

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Predict future values."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name."""
        pass

    def fit_predict(self, y: np.ndarray, horizon: int) -> np.ndarray:
        """Fit and predict in one step."""
        self.fit(y)
        return self.predict(horizon)

    def compute_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """Compute prediction residuals."""
        return y_true - y_pred

    def compute_anomaly_score(
        self,
        residuals: np.ndarray,
        method: str = 'zscore'
    ) -> np.ndarray:
        """
        Compute anomaly scores from residuals.

        Args:
            residuals: Prediction residuals
            method: 'zscore', 'abs', or 'squared'

        Returns:
            Anomaly scores
        """
        if method == 'zscore':
            mean = np.mean(residuals)
            std = np.std(residuals)
            if std > 0:
                return np.abs((residuals - mean) / std)
            return np.abs(residuals)
        elif method == 'abs':
            return np.abs(residuals)
        elif method == 'squared':
            return residuals ** 2
        return np.abs(residuals)
