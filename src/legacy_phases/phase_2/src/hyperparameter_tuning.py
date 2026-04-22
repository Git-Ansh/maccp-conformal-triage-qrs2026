"""
Phase 2: Hyperparameter Tuning Module
Bayesian optimization for multi-class models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys

try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.data_paths import RANDOM_SEED


def objective_logistic_regression(
    trial: 'optuna.Trial',
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: str = 'f1_macro'
) -> float:
    """Optuna objective for Logistic Regression."""
    C = trial.suggest_float('C', 1e-4, 10.0, log=True)
    solver = trial.suggest_categorical('solver', ['lbfgs', 'saga'])

    model = LogisticRegression(
        C=C,
        solver=solver,
        class_weight='balanced',
        max_iter=1000,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    if metric == 'f1_macro':
        return f1_score(y_val, y_pred, average='macro', zero_division=0)
    elif metric == 'f1_weighted':
        return f1_score(y_val, y_pred, average='weighted', zero_division=0)


def objective_random_forest(
    trial: 'optuna.Trial',
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: str = 'f1_macro'
) -> float:
    """Optuna objective for Random Forest."""
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    if metric == 'f1_macro':
        return f1_score(y_val, y_pred, average='macro', zero_division=0)
    elif metric == 'f1_weighted':
        return f1_score(y_val, y_pred, average='weighted', zero_division=0)


def objective_xgboost(
    trial: 'optuna.Trial',
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    metric: str = 'f1_macro'
) -> float:
    """Optuna objective for XGBoost multi-class."""
    if not HAS_XGBOOST:
        raise ImportError("XGBoost not installed")

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    model = XGBClassifier(
        **params,
        objective='multi:softmax',
        num_class=n_classes,
        random_state=RANDOM_SEED,
        eval_metric='mlogloss',
        use_label_encoder=False,
        tree_method='hist',
        device='cuda'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    y_pred = model.predict(X_val)

    if metric == 'f1_macro':
        return f1_score(y_val, y_pred, average='macro', zero_division=0)
    elif metric == 'f1_weighted':
        return f1_score(y_val, y_pred, average='weighted', zero_division=0)


def run_bayesian_optimization(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    metric: str = 'f1_macro',
    n_classes: Optional[int] = None,
    verbose: bool = True
) -> Tuple[Dict, float]:
    """
    Run Bayesian optimization for multi-class models.

    Args:
        model_type: One of 'logistic_regression', 'random_forest', 'xgboost'
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of optimization trials
        metric: Metric to optimize
        n_classes: Number of classes (required for XGBoost)
        verbose: Whether to print progress

    Returns:
        Tuple of (best_params, best_score)
    """
    if not HAS_OPTUNA:
        raise ImportError("Optuna not installed")

    if n_classes is None:
        n_classes = len(np.unique(y_train))

    sampler = TPESampler(seed=RANDOM_SEED, n_startup_trials=10)
    study = optuna.create_study(
        direction='maximize',
        study_name=f'{model_type}_multiclass_tuning',
        sampler=sampler
    )

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        if model_type == 'logistic_regression':
            return objective_logistic_regression(trial, X_train, y_train, X_val, y_val, metric)
        elif model_type == 'random_forest':
            return objective_random_forest(trial, X_train, y_train, X_val, y_val, metric)
        elif model_type == 'xgboost':
            return objective_xgboost(trial, X_train, y_train, X_val, y_val, n_classes, metric)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    print(f"\nOptimization complete for {model_type}")
    print(f"Best {metric}: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    return study.best_params, study.best_value


def get_tuned_model(
    model_type: str,
    best_params: Dict,
    n_classes: int
):
    """
    Get a model with tuned hyperparameters.

    Args:
        model_type: Model type
        best_params: Best parameters from tuning
        n_classes: Number of classes

    Returns:
        Configured model
    """
    if model_type == 'logistic_regression':
        return LogisticRegression(
            **best_params,
            class_weight='balanced',
            max_iter=1000,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
    elif model_type == 'random_forest':
        return RandomForestClassifier(
            **best_params,
            class_weight='balanced',
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
    elif model_type == 'xgboost':
        return XGBClassifier(
            **best_params,
            objective='multi:softmax',
            num_class=n_classes,
            random_state=RANDOM_SEED,
            eval_metric='mlogloss',
            use_label_encoder=False,
            tree_method='hist',
            device='cuda'
        )


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=5,
        n_clusters_per_class=1, random_state=RANDOM_SEED
    )

    X_train, X_val = X[:700], X[700:]
    y_train, y_val = y[:700], y[700:]

    if HAS_OPTUNA and HAS_XGBOOST:
        best_params, best_score = run_bayesian_optimization(
            'xgboost', X_train, y_train, X_val, y_val,
            n_trials=20, metric='f1_macro', n_classes=5
        )
        print(f"\nBest macro F1: {best_score:.4f}")
