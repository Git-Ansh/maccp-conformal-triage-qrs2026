"""
Phase 5: Neural Network Forecasting Models
LSTM and GRU implementations using TensorFlow/Keras.
"""

import numpy as np
import warnings
from typing import Optional, Tuple
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from .base import BaseForecaster

HAS_TF = False

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM as _LSTM, GRU as _GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except ImportError:
    Sequential = None
    _LSTM = None
    _GRU = None
    Dense = None
    Dropout = None
    EarlyStopping = None


def create_sequences(y: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM/GRU training."""
    X, targets = [], []
    for i in range(lookback, len(y)):
        X.append(y[i-lookback:i])
        targets.append(y[i])
    return np.array(X), np.array(targets)


class LSTMForecaster(BaseForecaster):
    """LSTM-based forecasting model."""

    def __init__(
        self,
        lookback: int = 10,
        units: int = 32,
        epochs: int = 50,
        batch_size: int = 16,
        verbose: int = 0
    ):
        self.lookback = lookback
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_ = None
        self.last_values_ = None
        self.scaler_mean_ = None
        self.scaler_std_ = None

    @property
    def name(self) -> str:
        return f"LSTM_{self.units}"

    def _build_model(self):
        if not HAS_TF:
            return None
        model = Sequential([
            _LSTM(self.units, input_shape=(self.lookback, 1), return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def fit(self, y: np.ndarray) -> 'LSTMForecaster':
        if not HAS_TF:
            raise ImportError("TensorFlow not installed")

        y = np.asarray(y).flatten()

        self.scaler_mean_ = np.mean(y)
        self.scaler_std_ = np.std(y)
        if self.scaler_std_ == 0:
            self.scaler_std_ = 1
        y_scaled = (y - self.scaler_mean_) / self.scaler_std_

        X, targets = create_sequences(y_scaled, self.lookback)

        if len(X) < 5:
            self.model_ = None
            self.last_values_ = y[-self.lookback:] if len(y) >= self.lookback else y
            return self

        X = X.reshape(-1, self.lookback, 1)

        self.model_ = self._build_model()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(
                X, targets,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
            )

        self.last_values_ = y[-self.lookback:]

        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self.last_values_ is None:
            raise ValueError("Model not fitted")

        if self.model_ is None:
            return np.full(horizon, self.last_values_[-1])

        predictions = []
        current = (self.last_values_ - self.scaler_mean_) / self.scaler_std_

        for _ in range(horizon):
            X = current[-self.lookback:].reshape(1, self.lookback, 1)
            pred_scaled = self.model_.predict(X, verbose=0)[0, 0]
            pred = pred_scaled * self.scaler_std_ + self.scaler_mean_
            predictions.append(pred)
            current = np.append(current, pred_scaled)

        return np.array(predictions)


class GRUForecaster(BaseForecaster):
    """GRU-based forecasting model."""

    def __init__(
        self,
        lookback: int = 10,
        units: int = 32,
        epochs: int = 50,
        batch_size: int = 16,
        verbose: int = 0
    ):
        self.lookback = lookback
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_ = None
        self.last_values_ = None
        self.scaler_mean_ = None
        self.scaler_std_ = None

    @property
    def name(self) -> str:
        return f"GRU_{self.units}"

    def _build_model(self):
        if not HAS_TF:
            return None
        model = Sequential([
            _GRU(self.units, input_shape=(self.lookback, 1), return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def fit(self, y: np.ndarray) -> 'GRUForecaster':
        if not HAS_TF:
            raise ImportError("TensorFlow not installed")

        y = np.asarray(y).flatten()

        self.scaler_mean_ = np.mean(y)
        self.scaler_std_ = np.std(y)
        if self.scaler_std_ == 0:
            self.scaler_std_ = 1
        y_scaled = (y - self.scaler_mean_) / self.scaler_std_

        X, targets = create_sequences(y_scaled, self.lookback)

        if len(X) < 5:
            self.model_ = None
            self.last_values_ = y[-self.lookback:] if len(y) >= self.lookback else y
            return self

        X = X.reshape(-1, self.lookback, 1)

        self.model_ = self._build_model()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(
                X, targets,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
            )

        self.last_values_ = y[-self.lookback:]

        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self.last_values_ is None:
            raise ValueError("Model not fitted")

        if self.model_ is None:
            return np.full(horizon, self.last_values_[-1])

        predictions = []
        current = (self.last_values_ - self.scaler_mean_) / self.scaler_std_

        for _ in range(horizon):
            X = current[-self.lookback:].reshape(1, self.lookback, 1)
            pred_scaled = self.model_.predict(X, verbose=0)[0, 0]
            pred = pred_scaled * self.scaler_std_ + self.scaler_mean_
            predictions.append(pred)
            current = np.append(current, pred_scaled)

        return np.array(predictions)


if __name__ == "__main__":
    if HAS_TF:
        np.random.seed(42)
        y = np.cumsum(np.random.normal(0, 1, 50)) + 100

        print("Testing neural forecasters...")

        for Model in [LSTMForecaster, GRUForecaster]:
            model = Model(epochs=10)
            model.fit(y)
            pred = model.predict(5)
            print(f"  {model.name}: {pred}")
    else:
        print("TensorFlow not installed")
