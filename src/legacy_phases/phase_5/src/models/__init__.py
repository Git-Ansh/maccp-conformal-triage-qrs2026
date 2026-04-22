"""
Phase 5: Forecasting Models
"""

from .base import BaseForecaster
from .statistical import ARIMAForecaster, HoltWintersForecaster, NaiveForecaster
from .ml_forecaster import RandomForestForecaster, GradientBoostingForecaster

try:
    from .neural import LSTMForecaster, GRUForecaster, HAS_TF
except ImportError:
    HAS_TF = False
    LSTMForecaster = None
    GRUForecaster = None


def get_all_forecasters(include_neural: bool = True):
    """Get all available forecasters."""
    forecasters = {
        'Naive_last': NaiveForecaster(method='last'),
        'Naive_mean': NaiveForecaster(method='mean'),
        'ARIMA': ARIMAForecaster(order=(1, 0, 1)),
        'HoltWinters': HoltWintersForecaster(),
        'RF_lag10': RandomForestForecaster(n_lags=10),
        'GBR_lag10': GradientBoostingForecaster(n_lags=10),
    }

    if include_neural and HAS_TF:
        forecasters['LSTM_32'] = LSTMForecaster(units=32, epochs=30)
        forecasters['GRU_32'] = GRUForecaster(units=32, epochs=30)

    return forecasters
