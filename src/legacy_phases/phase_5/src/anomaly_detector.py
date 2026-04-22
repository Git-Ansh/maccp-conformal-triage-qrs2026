"""
Phase 5: Anomaly Detection from Forecast Residuals
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def detect_anomaly_threshold(
    residuals: np.ndarray,
    threshold: float = 2.0,
    method: str = 'zscore'
) -> np.ndarray:
    """
    Detect anomalies using threshold on residuals.

    Args:
        residuals: Forecast residuals
        threshold: Detection threshold
        method: 'zscore', 'abs', or 'mad'

    Returns:
        Binary anomaly indicators
    """
    if method == 'zscore':
        mean = np.mean(residuals)
        std = np.std(residuals)
        if std > 0:
            scores = np.abs((residuals - mean) / std)
        else:
            scores = np.abs(residuals)
    elif method == 'abs':
        scores = np.abs(residuals)
    elif method == 'mad':
        median = np.median(residuals)
        mad = np.median(np.abs(residuals - median))
        if mad > 0:
            scores = np.abs(residuals - median) / (1.4826 * mad)
        else:
            scores = np.abs(residuals)
    else:
        scores = np.abs(residuals)

    return (scores > threshold).astype(int)


def compute_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray = None
) -> Dict[str, float]:
    """
    Compute detection metrics.

    Args:
        y_true: True labels (1=regression, 0=normal)
        y_pred: Predicted labels
        y_scores: Anomaly scores (for AUC)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }

    if y_scores is not None and len(np.unique(y_true)) > 1:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
        except:
            metrics['roc_auc'] = 0.5

    return metrics


def evaluate_forecast_detection(
    forecaster,
    series_list: List[np.ndarray],
    true_labels: List[int],
    window_size: int = 20,
    horizon: int = 3,
    threshold: float = 2.0
) -> Dict[str, float]:
    """
    Evaluate forecasting-based anomaly detection.

    Args:
        forecaster: Forecasting model
        series_list: List of time series (each ending at potential anomaly)
        true_labels: True regression labels (1=regression, 0=not)
        window_size: Training window size
        horizon: Forecast horizon
        threshold: Detection threshold

    Returns:
        Detection metrics
    """
    predictions = []
    scores = []

    for series in series_list:
        if len(series) < window_size + horizon:
            predictions.append(0)
            scores.append(0)
            continue

        # Split into train and test
        train = series[:window_size]
        actual = series[window_size:window_size + horizon]

        try:
            # Fit and predict
            forecaster.fit(train)
            forecast = forecaster.predict(horizon)

            # Compute residual
            residual = np.mean(np.abs(actual - forecast))
            train_std = np.std(train)

            if train_std > 0:
                score = residual / train_std
            else:
                score = residual

            scores.append(score)
            predictions.append(1 if score > threshold else 0)

        except Exception:
            predictions.append(0)
            scores.append(0)

    y_pred = np.array(predictions)
    y_scores = np.array(scores)
    y_true = np.array(true_labels)

    return compute_detection_metrics(y_true, y_pred, y_scores)


def find_optimal_threshold(
    scores: np.ndarray,
    true_labels: np.ndarray,
    thresholds: np.ndarray = None
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal detection threshold.

    Args:
        scores: Anomaly scores
        true_labels: True labels
        thresholds: Thresholds to try

    Returns:
        Tuple of (optimal_threshold, best_metrics)
    """
    if thresholds is None:
        thresholds = np.linspace(0.5, 5.0, 20)

    best_threshold = 1.0
    best_f1 = 0

    for thresh in thresholds:
        preds = (scores > thresh).astype(int)
        f1 = f1_score(true_labels, preds, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    final_preds = (scores > best_threshold).astype(int)
    metrics = compute_detection_metrics(true_labels, final_preds, scores)

    return best_threshold, metrics
