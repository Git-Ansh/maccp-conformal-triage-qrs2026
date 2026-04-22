"""
Phase 2: Models Module
Multi-class classifiers for status prediction.
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

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.data_paths import RANDOM_SEED


def get_logistic_regression_multiclass(
    class_weight: str = 'balanced',
    max_iter: int = 1000,
    C: float = 1.0
) -> LogisticRegression:
    """
    Get Logistic Regression with softmax for multi-class.

    Args:
        class_weight: 'balanced' to handle imbalance
        max_iter: Maximum iterations
        C: Regularization strength

    Returns:
        Configured LogisticRegression
    """
    return LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        C=C,
        solver='lbfgs',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )


def get_random_forest_multiclass(
    class_weight: str = 'balanced',
    n_estimators: int = 100,
    max_depth: Optional[int] = 10,
    min_samples_leaf: int = 5
) -> RandomForestClassifier:
    """
    Get Random Forest for multi-class classification.

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


def get_gradient_boosting_multiclass(
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1
) -> GradientBoostingClassifier:
    """
    Get Gradient Boosting for multi-class.

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


def get_xgboost_multiclass(
    n_classes: int,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    use_gpu: bool = True,
    **kwargs
) -> 'XGBClassifier':
    """
    Get XGBoost for multi-class classification.

    Args:
        n_classes: Number of classes
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        use_gpu: Whether to use GPU acceleration
        **kwargs: Additional XGBoost parameters

    Returns:
        Configured XGBClassifier
    """
    if not HAS_XGBOOST:
        raise ImportError("XGBoost not installed")

    params = {
        'objective': 'multi:softmax',
        'num_class': n_classes,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'random_state': RANDOM_SEED,
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
    }

    # Enable GPU if available
    if use_gpu:
        params['tree_method'] = 'hist'
        params['device'] = 'cuda'
    else:
        params['n_jobs'] = -1

    params.update(kwargs)

    return XGBClassifier(**params)


def get_lightgbm_multiclass(
    n_classes: int,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    class_weight: str = 'balanced',
    **kwargs
) -> 'LGBMClassifier':
    """
    Get LightGBM for multi-class classification.

    Args:
        n_classes: Number of classes
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        class_weight: Class weight strategy
        **kwargs: Additional parameters

    Returns:
        Configured LGBMClassifier
    """
    if not HAS_LIGHTGBM:
        raise ImportError("LightGBM not installed")

    return LGBMClassifier(
        objective='multiclass',
        num_class=n_classes,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        class_weight=class_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
        **kwargs
    )


def get_all_multiclass_models(
    n_classes: int,
    class_weights: Optional[Dict[int, float]] = None
) -> Dict[str, BaseEstimator]:
    """
    Get all baseline multi-class models.

    Args:
        n_classes: Number of classes
        class_weights: Optional class weight dictionary

    Returns:
        Dictionary of model_name -> model
    """
    models = {
        'LogisticRegression': get_logistic_regression_multiclass(),
        'RandomForest': get_random_forest_multiclass(),
        'GradientBoosting': get_gradient_boosting_multiclass()
    }

    if HAS_XGBOOST:
        models['XGBoost'] = get_xgboost_multiclass(n_classes)

    if HAS_LIGHTGBM:
        models['LightGBM'] = get_lightgbm_multiclass(n_classes)

    return models


def train_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    sample_weights: Optional[np.ndarray] = None,
    early_stopping: bool = True
) -> Tuple[BaseEstimator, Dict]:
    """
    Train a multi-class model.

    Args:
        model: Model to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        sample_weights: Optional sample weights
        early_stopping: Whether to use early stopping

    Returns:
        Tuple of (fitted model, training info)
    """
    info = {
        'model_type': type(model).__name__,
        'n_train_samples': len(y_train),
        'n_classes': len(np.unique(y_train))
    }

    # Handle XGBoost early stopping
    if HAS_XGBOOST and isinstance(model, XGBClassifier) and early_stopping and X_val is not None:
        model.set_params(early_stopping_rounds=50)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            sample_weight=sample_weights
        )
        try:
            info['best_iteration'] = model.best_iteration
            info['early_stopped'] = model.best_iteration < model.n_estimators
        except AttributeError:
            info['best_iteration'] = model.n_estimators
            info['early_stopped'] = False
    elif HAS_LIGHTGBM and isinstance(model, LGBMClassifier) and early_stopping and X_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[],
            sample_weight=sample_weights
        )
    else:
        if sample_weights is not None and hasattr(model, 'fit'):
            try:
                model.fit(X_train, y_train, sample_weight=sample_weights)
            except TypeError:
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

    return model, info


def predict_proba_multiclass(
    model: BaseEstimator,
    X: np.ndarray
) -> np.ndarray:
    """
    Get probability predictions for all classes.

    Args:
        model: Trained model
        X: Features

    Returns:
        Probability matrix (n_samples, n_classes)
    """
    return model.predict_proba(X)


if __name__ == "__main__":
    # Test multi-class models
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_classes=5, n_clusters_per_class=1,
        random_state=RANDOM_SEED
    )

    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    n_classes = len(np.unique(y))
    models = get_all_multiclass_models(n_classes)

    print(f"Available models: {list(models.keys())}")

    for name, model in models.items():
        model, info = train_model(model, X_train, y_train)
        y_pred = model.predict(X_test)
        acc = (y_pred == y_test).mean()
        print(f"{name}: accuracy = {acc:.4f}")
