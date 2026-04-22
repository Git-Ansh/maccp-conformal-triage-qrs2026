"""
Phase 3: Time-Series Feature Engineering
Extract features from performance time series.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.signal import detrend


def compute_window_stats(
    values: np.ndarray,
    window_size: int = 20
) -> Dict[str, float]:
    """
    Compute basic statistics over a window.

    Args:
        values: Array of values (window before alert)
        window_size: Expected window size

    Returns:
        Dictionary of statistics
    """
    if len(values) < 3:
        return {
            'ts_mean': np.nan,
            'ts_median': np.nan,
            'ts_std': np.nan,
            'ts_cv': np.nan,
            'ts_min': np.nan,
            'ts_max': np.nan,
            'ts_range': np.nan,
            'ts_iqr': np.nan,
            'ts_skew': np.nan,
            'ts_kurtosis': np.nan
        }

    mean_val = np.mean(values)
    std_val = np.std(values)

    return {
        'ts_mean': mean_val,
        'ts_median': np.median(values),
        'ts_std': std_val,
        'ts_cv': std_val / mean_val if mean_val != 0 else np.nan,
        'ts_min': np.min(values),
        'ts_max': np.max(values),
        'ts_range': np.max(values) - np.min(values),
        'ts_iqr': np.percentile(values, 75) - np.percentile(values, 25),
        'ts_skew': stats.skew(values) if len(values) > 2 else 0,
        'ts_kurtosis': stats.kurtosis(values) if len(values) > 3 else 0
    }


def compute_change_metrics(
    pre_values: np.ndarray,
    alert_value: float
) -> Dict[str, float]:
    """
    Compute change metrics comparing pre-alert window to alert value.

    Args:
        pre_values: Values before the alert
        alert_value: The value at the alert point

    Returns:
        Dictionary of change metrics
    """
    if len(pre_values) < 2:
        return {
            'ts_change_abs': np.nan,
            'ts_change_pct': np.nan,
            'ts_normalized_change': np.nan,
            'ts_zscore': np.nan
        }

    pre_mean = np.mean(pre_values)
    pre_std = np.std(pre_values)

    change_abs = alert_value - pre_mean
    change_pct = (change_abs / pre_mean * 100) if pre_mean != 0 else np.nan
    normalized_change = change_abs / pre_std if pre_std > 0 else np.nan
    zscore = (alert_value - pre_mean) / pre_std if pre_std > 0 else np.nan

    return {
        'ts_change_abs': change_abs,
        'ts_change_pct': change_pct,
        'ts_normalized_change': normalized_change,
        'ts_zscore': zscore
    }


def compute_slope_features(
    values: np.ndarray,
    window_size: int = 20
) -> Dict[str, float]:
    """
    Compute slope and trend features.

    Args:
        values: Array of values
        window_size: Expected window size

    Returns:
        Dictionary of slope features
    """
    if len(values) < 3:
        return {
            'ts_local_slope': np.nan,
            'ts_slope_r2': np.nan,
            'ts_early_slope': np.nan,
            'ts_late_slope': np.nan,
            'ts_slope_diff': np.nan
        }

    n = len(values)
    x = np.arange(n)

    # Overall slope using linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)

    # Early vs late slope (detect trend changes)
    mid = n // 2
    if mid > 2:
        early_slope, _, _, _, _ = stats.linregress(x[:mid], values[:mid])
        late_slope, _, _, _, _ = stats.linregress(x[mid:], values[mid:])
        slope_diff = late_slope - early_slope
    else:
        early_slope = late_slope = slope_diff = np.nan

    return {
        'ts_local_slope': slope,
        'ts_slope_r2': r_value ** 2,
        'ts_early_slope': early_slope,
        'ts_late_slope': late_slope,
        'ts_slope_diff': slope_diff
    }


def compute_stability_features(
    values: np.ndarray
) -> Dict[str, float]:
    """
    Compute stability and noise indicators.

    Args:
        values: Array of values

    Returns:
        Dictionary of stability features
    """
    if len(values) < 5:
        return {
            'ts_direction_changes': np.nan,
            'ts_direction_change_rate': np.nan,
            'ts_variance_ratio': np.nan,
            'ts_autocorr_lag1': np.nan
        }

    # Direction changes
    diffs = np.diff(values)
    signs = np.sign(diffs)
    direction_changes = np.sum(signs[:-1] != signs[1:])
    direction_change_rate = direction_changes / len(diffs)

    # Variance ratio (recent vs overall)
    n = len(values)
    recent_n = max(n // 3, 3)
    recent_var = np.var(values[-recent_n:])
    overall_var = np.var(values)
    variance_ratio = recent_var / overall_var if overall_var > 0 else np.nan

    # Autocorrelation at lag 1
    if len(values) > 2:
        autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
    else:
        autocorr = np.nan

    return {
        'ts_direction_changes': direction_changes,
        'ts_direction_change_rate': direction_change_rate,
        'ts_variance_ratio': variance_ratio,
        'ts_autocorr_lag1': autocorr
    }


def compute_drift_features(
    values: np.ndarray
) -> Dict[str, float]:
    """
    Compute drift and trend detection features.

    Args:
        values: Array of values

    Returns:
        Dictionary of drift features
    """
    if len(values) < 5:
        return {
            'ts_cusum_max': np.nan,
            'ts_cusum_min': np.nan,
            'ts_ewma_deviation': np.nan,
            'ts_trend_strength': np.nan
        }

    mean_val = np.mean(values)

    # CUSUM (Cumulative Sum)
    cusum = np.cumsum(values - mean_val)
    cusum_max = np.max(cusum)
    cusum_min = np.min(cusum)

    # EWMA deviation
    alpha = 0.3
    ewma = pd.Series(values).ewm(alpha=alpha).mean().values
    ewma_deviation = np.std(values - ewma)

    # Trend strength (ratio of linear trend to total variation)
    x = np.arange(len(values))
    slope, intercept, _, _, _ = stats.linregress(x, values)
    trend_line = slope * x + intercept
    trend_var = np.var(trend_line)
    total_var = np.var(values)
    trend_strength = trend_var / total_var if total_var > 0 else 0

    return {
        'ts_cusum_max': cusum_max,
        'ts_cusum_min': cusum_min,
        'ts_ewma_deviation': ewma_deviation,
        'ts_trend_strength': trend_strength
    }


def extract_all_features(
    values: np.ndarray,
    alert_value: float,
    window_size: int = 20
) -> Dict[str, float]:
    """
    Extract all time-series features for a single alert.

    Args:
        values: Pre-alert values (window before alert)
        alert_value: Value at the alert point
        window_size: Window size used

    Returns:
        Dictionary of all features
    """
    features = {}

    # Window statistics
    features.update(compute_window_stats(values, window_size))

    # Change metrics
    features.update(compute_change_metrics(values, alert_value))

    # Slope features
    features.update(compute_slope_features(values, window_size))

    # Stability features
    features.update(compute_stability_features(values))

    # Drift features
    features.update(compute_drift_features(values))

    # Add window metadata
    features['ts_window_size'] = len(values)
    features['ts_alert_value'] = alert_value

    return features


def get_feature_groups() -> Dict[str, List[str]]:
    """
    Get feature group definitions for ablation studies.

    Returns:
        Dictionary of group_name -> list of feature prefixes
    """
    return {
        'local_stats': ['ts_mean', 'ts_median', 'ts_std', 'ts_cv', 'ts_min',
                       'ts_max', 'ts_range', 'ts_iqr', 'ts_skew', 'ts_kurtosis'],
        'change_metrics': ['ts_change_abs', 'ts_change_pct',
                          'ts_normalized_change', 'ts_zscore'],
        'slope_features': ['ts_local_slope', 'ts_slope_r2', 'ts_early_slope',
                          'ts_late_slope', 'ts_slope_diff'],
        'stability': ['ts_direction_changes', 'ts_direction_change_rate',
                     'ts_variance_ratio', 'ts_autocorr_lag1'],
        'drift': ['ts_cusum_max', 'ts_cusum_min', 'ts_ewma_deviation',
                 'ts_trend_strength']
    }


if __name__ == "__main__":
    # Test feature extraction
    np.random.seed(42)

    # Simulate stable series with sudden jump
    stable = np.random.normal(100, 5, 20)
    alert_value = 130  # Regression

    features = extract_all_features(stable, alert_value)
    print("Features for stable->regression:")
    for k, v in features.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n" + "="*50)

    # Simulate noisy series
    noisy = np.random.normal(100, 25, 20)
    alert_value_noisy = 115  # Might be noise

    features_noisy = extract_all_features(noisy, alert_value_noisy)
    print("Features for noisy series:")
    for k, v in features_noisy.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
