"""
Phase 1: Hyperparameter Tuning Module
Bayesian optimization using Optuna for model tuning.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable
import sys

try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Warning: Optuna not installed. Run: pip install optuna")

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score, f1_score, make_scorer
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
from common.model_utils import get_scale_pos_weight


def create_optuna_study(
    direction: str = 'maximize',
    study_name: str = 'phase1_tuning',
    n_startup_trials: int = 10
) -> 'optuna.Study':
    """
    Create an Optuna study for hyperparameter optimization.

    Args:
        direction: 'maximize' or 'minimize'
        study_name: Name for the study
        n_startup_trials: Number of random trials before TPE

    Returns:
        Optuna study object
    """
    if not HAS_OPTUNA:
        raise ImportError("Optuna not installed. Run: pip install optuna")

    sampler = TPESampler(seed=RANDOM_SEED, n_startup_trials=n_startup_trials)

    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        sampler=sampler
    )

    return study


def objective_logistic_regression(
    trial: 'optuna.Trial',
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: str = 'precision'
) -> float:
    """
    Optuna objective for Logistic Regression.

    Args:
        trial: Optuna trial
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        metric: Metric to optimize ('precision', 'f1', 'roc_auc')

    Returns:
        Metric score
    """
    # Hyperparameters
    C = trial.suggest_float('C', 1e-4, 10.0, log=True)
    solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga'])

    model = LogisticRegression(
        C=C,
        solver=solver,
        class_weight='balanced',
        max_iter=1000,
        random_state=RANDOM_SEED
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    if metric == 'precision':
        return precision_score(y_val, y_pred, zero_division=0)
    elif metric == 'f1':
        return f1_score(y_val, y_pred, zero_division=0)
    elif metric == 'roc_auc':
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_val, y_prob)


def objective_random_forest(
    trial: 'optuna.Trial',
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: str = 'precision'
) -> float:
    """
    Optuna objective for Random Forest.

    Args:
        trial: Optuna trial
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        metric: Metric to optimize

    Returns:
        Metric score
    """
    # Hyperparameters
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
    y_prob = model.predict_proba(X_val)[:, 1]

    if metric == 'precision':
        return precision_score(y_val, y_pred, zero_division=0)
    elif metric == 'f1':
        return f1_score(y_val, y_pred, zero_division=0)
    elif metric == 'roc_auc':
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_val, y_prob)


def objective_xgboost(
    trial: 'optuna.Trial',
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: str = 'precision'
) -> float:
    """
    Optuna objective for XGBoost.

    Args:
        trial: Optuna trial
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        metric: Metric to optimize

    Returns:
        Metric score
    """
    if not HAS_XGBOOST:
        raise ImportError("XGBoost not installed")

    # Hyperparameters
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

    # Add class imbalance handling
    scale_pos_weight = get_scale_pos_weight(y_train)

    model = XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1
    )

    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    if metric == 'precision':
        return precision_score(y_val, y_pred, zero_division=0)
    elif metric == 'f1':
        return f1_score(y_val, y_pred, zero_division=0)
    elif metric == 'roc_auc':
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_val, y_prob)


def run_bayesian_optimization(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 100,
    metric: str = 'precision',
    verbose: bool = True
) -> Tuple[Dict, float]:
    """
    Run Bayesian optimization for a model type.

    Args:
        model_type: One of 'logistic_regression', 'random_forest', 'xgboost'
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of optimization trials
        metric: Metric to optimize
        verbose: Whether to print progress

    Returns:
        Tuple of (best_params, best_score)
    """
    if not HAS_OPTUNA:
        raise ImportError("Optuna not installed")

    # Select objective function
    objective_funcs = {
        'logistic_regression': objective_logistic_regression,
        'random_forest': objective_random_forest,
        'xgboost': objective_xgboost
    }

    if model_type not in objective_funcs:
        raise ValueError(f"Unknown model type: {model_type}")

    objective_func = objective_funcs[model_type]

    # Create study
    study = create_optuna_study(
        direction='maximize',
        study_name=f'{model_type}_tuning'
    )

    # Suppress Optuna logging if not verbose
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Optimize
    def objective(trial):
        return objective_func(trial, X_train, y_train, X_val, y_val, metric)

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=verbose
    )

    print(f"\nOptimization complete for {model_type}")
    print(f"Best {metric}: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    return study.best_params, study.best_value


def get_tuned_model(
    model_type: str,
    best_params: Dict,
    y_train: Optional[np.ndarray] = None
):
    """
    Get a model with tuned hyperparameters.

    Args:
        model_type: Model type
        best_params: Best parameters from tuning
        y_train: Training labels (for scale_pos_weight)

    Returns:
        Configured model
    """
    if model_type == 'logistic_regression':
        return LogisticRegression(
            **best_params,
            class_weight='balanced',
            max_iter=1000,
            random_state=RANDOM_SEED
        )
    elif model_type == 'random_forest':
        return RandomForestClassifier(
            **best_params,
            class_weight='balanced',
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
    elif model_type == 'xgboost':
        scale_pos_weight = get_scale_pos_weight(y_train) if y_train is not None else 1.0
        return XGBClassifier(
            **best_params,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_SEED,
            eval_metric='logloss',
            use_label_encoder=False,
            n_jobs=-1
        )


def tune_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    metric: str = 'precision'
) -> Dict[str, Tuple[Dict, float]]:
    """
    Tune all baseline models.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Trials per model
        metric: Metric to optimize

    Returns:
        Dictionary of model_type -> (best_params, best_score)
    """
    results = {}

    model_types = ['logistic_regression', 'random_forest']
    if HAS_XGBOOST:
        model_types.append('xgboost')

    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Tuning {model_type}")
        print(f"{'='*50}")

        best_params, best_score = run_bayesian_optimization(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            n_trials=n_trials,
            metric=metric
        )

        results[model_type] = (best_params, best_score)

    return results


if __name__ == "__main__":
    # Test tuning
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2,
        weights=[0.7, 0.3], random_state=RANDOM_SEED
    )

    # Split
    X_train, X_val = X[:700], X[700:]
    y_train, y_val = y[:700], y[700:]

    # Test XGBoost tuning
    if HAS_XGBOOST and HAS_OPTUNA:
        best_params, best_score = run_bayesian_optimization(
            'xgboost', X_train, y_train, X_val, y_val,
            n_trials=20, metric='precision'
        )
        print(f"\nBest precision: {best_score:.4f}")
