"""
Phase 5: Statistical Forecasting Models
ARIMA and Holt-Winters implementations.
"""

import numpy as np
import warnings
from typing import Optional

from .base import BaseForecaster

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


class ARIMAForecaster(BaseForecaster):
    """ARIMA forecasting model."""

    def __init__(self, order: tuple = (1, 0, 1), seasonal_order: tuple = None):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_ = None
        self.fitted_ = None

    @property
    def name(self) -> str:
        return f"ARIMA{self.order}"

    def fit(self, y: np.ndarray) -> 'ARIMAForecaster':
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels not installed")

        y = np.asarray(y).flatten()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.model_ = ARIMA(y, order=self.order)
                self.fitted_ = self.model_.fit()
            except Exception:
                # Fallback to simpler model
                self.model_ = ARIMA(y, order=(1, 0, 0))
                self.fitted_ = self.model_.fit()

        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self.fitted_ is None:
            raise ValueError("Model not fitted")

        forecast = self.fitted_.forecast(steps=horizon)
        return np.asarray(forecast)


class HoltWintersForecaster(BaseForecaster):
    """Holt-Winters Exponential Smoothing."""

    def __init__(
        self,
        trend: str = 'add',
        seasonal: str = None,
        seasonal_periods: int = None
    ):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model_ = None
        self.fitted_ = None

    @property
    def name(self) -> str:
        return "HoltWinters"

    def fit(self, y: np.ndarray) -> 'HoltWintersForecaster':
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels not installed")

        y = np.asarray(y).flatten()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.model_ = ExponentialSmoothing(
                    y,
                    trend=self.trend,
                    seasonal=self.seasonal,
                    seasonal_periods=self.seasonal_periods
                )
                self.fitted_ = self.model_.fit(optimized=True)
            except Exception:
                # Fallback to simple exponential smoothing
                self.model_ = ExponentialSmoothing(y, trend=None, seasonal=None)
                self.fitted_ = self.model_.fit()

        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self.fitted_ is None:
            raise ValueError("Model not fitted")

        forecast = self.fitted_.forecast(horizon)
        return np.asarray(forecast)


class NaiveForecaster(BaseForecaster):
    """Simple naive forecaster (last value or mean)."""

    def __init__(self, method: str = 'last'):
        self.method = method
        self.value_ = None

    @property
    def name(self) -> str:
        return f"Naive_{self.method}"

    def fit(self, y: np.ndarray) -> 'NaiveForecaster':
        y = np.asarray(y).flatten()

        if self.method == 'last':
            self.value_ = y[-1]
        elif self.method == 'mean':
            self.value_ = np.mean(y)
        elif self.method == 'median':
            self.value_ = np.median(y)

        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self.value_ is None:
            raise ValueError("Model not fitted")
        return np.full(horizon, self.value_)


if __name__ == "__main__":
    # Test statistical models
    np.random.seed(42)
    y = np.cumsum(np.random.normal(0, 1, 50)) + 100

    print("Testing statistical forecasters...")

    for Model in [ARIMAForecaster, HoltWintersForecaster, NaiveForecaster]:
        try:
            model = Model()
            model.fit(y)
            pred = model.predict(5)
            print(f"  {model.name}: {pred}")
        except Exception as e:
            print(f"  {Model.__name__}: Error - {e}")
