"""
Confidence calibration for cascade stages.
Provides Platt scaling, isotonic regression wrappers,
per-class confidence thresholds, and OOF threshold tuning.
"""

import numpy as np
from typing import Tuple, Optional, Union
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold


def calibrate_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = 'isotonic',
    cv: int = 5
) -> CalibratedClassifierCV:
    """
    Wrap a trained model with probability calibration.

    Args:
        model: Pre-trained classifier (must have predict_proba or decision_function)
        X_train: Training features (used for calibration fitting)
        y_train: Training labels
        method: 'sigmoid' (Platt scaling) or 'isotonic'
        cv: Number of CV folds for calibration

    Returns:
        CalibratedClassifierCV wrapping the model
    """
    calibrated = CalibratedClassifierCV(
        model,
        method=method,
        cv=cv
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def get_confidence(proba: np.ndarray) -> np.ndarray:
    """
    Extract confidence score from probability predictions.
    For binary: max(p, 1-p). For multiclass: max class probability.

    Args:
        proba: Probability array (n_samples,) or (n_samples, n_classes)

    Returns:
        Confidence scores (n_samples,)
    """
    if proba.ndim == 1:
        return np.maximum(proba, 1 - proba)
    return np.max(proba, axis=1)


def get_predicted_class(proba: np.ndarray) -> np.ndarray:
    """
    Get predicted class from probability array.

    Args:
        proba: Probability array (n_samples, n_classes)

    Returns:
        Predicted class indices (n_samples,)
    """
    if proba.ndim == 1:
        return (proba >= 0.5).astype(int)
    return np.argmax(proba, axis=1)


def apply_confidence_gate(
    proba: np.ndarray,
    threshold: Union[float, np.ndarray],
    classes: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply confidence gating: split predictions into confident and uncertain.

    Args:
        proba: Calibrated probability array (n_samples, n_classes)
        threshold: Confidence threshold -- scalar (global) or array of shape
                   (n_classes,) for per-class thresholds
        classes: Class labels to map indices back to original labels

    Returns:
        Tuple of:
            - predictions: predicted class for all samples (-1 for uncertain)
            - confidence: confidence score for each sample
            - is_confident: boolean mask (True = confident, False = deferred)
    """
    confidence = get_confidence(proba)
    predicted_idx = get_predicted_class(proba)

    if classes is not None:
        predictions = classes[predicted_idx]
    else:
        predictions = predicted_idx

    # Per-class or global threshold
    if isinstance(threshold, np.ndarray) and threshold.ndim == 1:
        per_sample_threshold = threshold[predicted_idx]
        is_confident = confidence >= per_sample_threshold
    else:
        is_confident = confidence >= float(threshold)

    # Mark uncertain predictions
    deferred_predictions = predictions.copy()
    if hasattr(deferred_predictions, 'dtype') and np.issubdtype(deferred_predictions.dtype, np.integer):
        deferred_predictions[~is_confident] = -1
    else:
        deferred_predictions = np.where(is_confident, predictions, -1)

    return deferred_predictions, confidence, is_confident


def find_threshold_for_target_precision(
    y_true: np.ndarray,
    proba: np.ndarray,
    target_precision: float = 0.90,
    target_class: Optional[int] = None
) -> float:
    """
    Find the confidence threshold that achieves target precision on the confident subset.

    Args:
        y_true: True labels
        proba: Calibrated probabilities
        target_precision: Desired precision on confident predictions
        target_class: For binary, which class to optimize precision for

    Returns:
        Optimal confidence threshold
    """
    confidence = get_confidence(proba)
    predicted = get_predicted_class(proba)

    thresholds = np.arange(0.50, 0.99, 0.01)
    best_threshold = 0.50
    best_coverage = 0.0

    for t in thresholds:
        mask = confidence >= t
        if mask.sum() < 10:
            continue

        if target_class is not None:
            # Precision for specific class
            class_mask = mask & (predicted == target_class)
            if class_mask.sum() == 0:
                continue
            precision = (y_true[class_mask] == target_class).mean()
        else:
            # Overall accuracy on confident subset
            precision = (y_true[mask] == predicted[mask]).mean()

        if precision >= target_precision and mask.mean() > best_coverage:
            best_threshold = t
            best_coverage = mask.mean()

    return best_threshold


def find_per_class_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    target_accuracy: float = 0.80,
    min_samples: int = 5
) -> np.ndarray:
    """
    Find per-class confidence thresholds that achieve target accuracy
    on each class's confident subset independently.

    Args:
        y_true: True labels (encoded 0..n_classes-1)
        proba: Calibrated probabilities (n_samples, n_classes)
        target_accuracy: Desired accuracy per class on confident subset
        min_samples: Minimum samples needed per class to set threshold

    Returns:
        Array of shape (n_classes,) with per-class thresholds
    """
    n_classes = proba.shape[1]
    thresholds = np.full(n_classes, 0.50)
    predicted = np.argmax(proba, axis=1)

    for c in range(n_classes):
        class_mask = predicted == c
        if class_mask.sum() < min_samples:
            continue

        class_true = y_true[class_mask]
        class_conf = np.max(proba[class_mask], axis=1)

        best_t = 0.50
        best_cov = 0.0
        for t in np.arange(0.40, 0.96, 0.01):
            t_mask = class_conf >= t
            if t_mask.sum() < min_samples:
                continue
            acc = (class_true[t_mask] == c).mean()
            cov = t_mask.mean()
            if acc >= target_accuracy and cov > best_cov:
                best_t = t
                best_cov = cov
        thresholds[c] = best_t

    return thresholds


def get_oof_predictions(
    base_model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42
) -> np.ndarray:
    """
    Generate out-of-fold probability predictions for threshold tuning.
    Each sample gets predicted by a model that never saw it during training.

    Args:
        base_model: Unfitted sklearn-compatible estimator (will be cloned)
        X: Feature matrix
        y: Labels
        n_folds: Number of CV folds
        random_state: Random seed for fold splits

    Returns:
        OOF probability array of shape (n_samples, n_classes)
    """
    n_classes = len(np.unique(y))
    oof_proba = np.zeros((len(y), n_classes))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for train_idx, val_idx in skf.split(X, y):
        fold_model = clone(base_model)
        fold_model.fit(X[train_idx], y[train_idx])
        oof_proba[val_idx] = fold_model.predict_proba(X[val_idx])

    return oof_proba
