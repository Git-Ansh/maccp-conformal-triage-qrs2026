"""
Phase 1: Models Module
Baseline classifiers for binary regression detection.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import BaseEstimator
import sys

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed")

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.data_paths import RANDOM_SEED
from common.model_utils import get_class_weights, get_scale_pos_weight


def get_logistic_regression(
    class_weight: str = 'balanced',
    max_iter: int = 1000,
    C: float = 1.0
) -> LogisticRegression:
    """
    Get Logistic Regression classifier.

    Args:
        class_weight: 'balanced' to handle imbalance
        max_iter: Maximum iterations
        C: Regularization strength (smaller = stronger)

    Returns:
        Configured LogisticRegression
    """
    return LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        C=C,
        random_state=RANDOM_SEED,
        solver='lbfgs',
        n_jobs=-1
    )


def get_random_forest(
    class_weight: str = 'balanced',
    n_estimators: int = 100,
    max_depth: Optional[int] = 10,
    min_samples_leaf: int = 5
) -> RandomForestClassifier:
    """
    Get Random Forest classifier.

    Args:
        class_weight: 'balanced' to handle imbalance
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        min_samples_leaf: Minimum samples in leaf

    Returns:
        Configured RandomForestClassifier
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )


def get_gradient_boosting(
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1
) -> GradientBoostingClassifier:
    """
    Get Gradient Boosting classifier (sklearn version).

    Args:
        n_estimators: Number of boosting stages
        max_depth: Maximum tree depth
        learning_rate: Learning rate

    Returns:
        Configured GradientBoostingClassifier
    """
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=RANDOM_SEED
    )


def get_xgboost_classifier(
    scale_pos_weight: Optional[float] = None,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    **kwargs
) -> 'XGBClassifier':
    """
    Get XGBoost classifier.

    Args:
        scale_pos_weight: Weight for positive class (for imbalance)
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        **kwargs: Additional XGBoost parameters

    Returns:
        Configured XGBClassifier
    """
    if not HAS_XGBOOST:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")

    params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'random_state': RANDOM_SEED,
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'n_jobs': -1
    }

    if scale_pos_weight is not None:
        params['scale_pos_weight'] = scale_pos_weight

    params.update(kwargs)

    return XGBClassifier(**params)


def get_all_baseline_models(
    y_train: Optional[np.ndarray] = None
) -> Dict[str, BaseEstimator]:
    """
    Get all baseline models for comparison.

    Args:
        y_train: Training labels (for computing class weights)

    Returns:
        Dictionary of model_name -> model
    """
    models = {
        'LogisticRegression': get_logistic_regression(),
        'RandomForest': get_random_forest(),
        'GradientBoosting': get_gradient_boosting()
    }

    if HAS_XGBOOST:
        scale_weight = None
        if y_train is not None:
            scale_weight = get_scale_pos_weight(y_train)

        models['XGBoost'] = get_xgboost_classifier(scale_pos_weight=scale_weight)

    return models


def train_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    early_stopping: bool = True
) -> Tuple[BaseEstimator, Dict]:
    """
    Train a model with optional early stopping.

    Args:
        model: Model to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (for early stopping)
        y_val: Validation labels (for early stopping)
        early_stopping: Whether to use early stopping (XGBoost only)

    Returns:
        Tuple of (fitted model, training info)
    """
    info = {
        'model_type': type(model).__name__,
        'n_train_samples': len(y_train)
    }

    # Handle XGBoost early stopping
    if HAS_XGBOOST and isinstance(model, XGBClassifier) and early_stopping and X_val is not None:
        model.set_params(early_stopping_rounds=50)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        try:
            info['best_iteration'] = model.best_iteration
            info['early_stopped'] = model.best_iteration < model.n_estimators
        except AttributeError:
            info['best_iteration'] = model.n_estimators
            info['early_stopped'] = False
    else:
        model.fit(X_train, y_train)

    return model, info


def predict_with_threshold(
    model: BaseEstimator,
    X: np.ndarray,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict with custom threshold.

    Args:
        model: Trained model
        X: Features
        threshold: Decision threshold

    Returns:
        Tuple of (predictions, probabilities)
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return y_pred, y_prob


def get_model_params(model: BaseEstimator) -> Dict:
    """
    Get model hyperparameters.

    Args:
        model: Model instance

    Returns:
        Dictionary of parameters
    """
    return model.get_params()


def compare_models(
    models: Dict[str, BaseEstimator],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Train and compare multiple models.

    Args:
        models: Dictionary of model_name -> model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)

    Returns:
        DataFrame with comparison results
    """
    from common.evaluation_utils import compute_binary_metrics

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Train
        model, train_info = train_model(
            model, X_train, y_train, X_val, y_val
        )

        # Predict
        y_pred, y_prob = predict_with_threshold(model, X_test)

        # Evaluate
        metrics = compute_binary_metrics(y_test, y_pred, y_prob)
        metrics['model'] = name
        results.append(metrics)

        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1_score']:.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('precision', ascending=False)

    return results_df


if __name__ == "__main__":
    # Test models
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2,
        weights=[0.7, 0.3], random_state=RANDOM_SEED
    )

    # Split
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    # Get models
    models = get_all_baseline_models(y_train)
    print(f"Available models: {list(models.keys())}")

    # Compare
    results = compare_models(models, X_train, y_train, X_test, y_test)
    print("\nModel Comparison:")
    print(results[['model', 'precision', 'recall', 'f1_score', 'roc_auc']])
