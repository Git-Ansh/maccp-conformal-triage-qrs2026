"""
Shared model utilities for saving, loading, and training models.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


def save_model(
    model: BaseEstimator,
    path: Path,
    model_name: str,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save a trained model with metadata.

    Args:
        model: Trained model to save
        path: Directory to save to
        model_name: Name for the model file
        metadata: Optional metadata to save alongside
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = path / f"{model_name}.joblib"
    joblib.dump(model, model_path)

    # Save metadata
    if metadata is None:
        metadata = {}

    metadata['saved_at'] = datetime.now().isoformat()
    metadata['model_type'] = type(model).__name__

    metadata_path = path / f"{model_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def load_model(path: Path, model_name: str) -> Tuple[BaseEstimator, Dict]:
    """
    Load a saved model with its metadata.

    Args:
        path: Directory containing the model
        model_name: Name of the model file

    Returns:
        Tuple of (model, metadata)
    """
    path = Path(path)

    model_path = path / f"{model_name}.joblib"
    model = joblib.load(model_path)

    metadata_path = path / f"{model_name}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return model, metadata


def save_encoder(
    encoder: Union[LabelEncoder, StandardScaler],
    path: Path,
    encoder_name: str
) -> None:
    """
    Save a fitted encoder/scaler.

    Args:
        encoder: Fitted encoder or scaler
        path: Directory to save to
        encoder_name: Name for the encoder file
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    encoder_path = path / f"{encoder_name}.joblib"
    joblib.dump(encoder, encoder_path)


def load_encoder(path: Path, encoder_name: str) -> Any:
    """
    Load a saved encoder/scaler.

    Args:
        path: Directory containing the encoder
        encoder_name: Name of the encoder file

    Returns:
        Loaded encoder/scaler
    """
    path = Path(path)
    encoder_path = path / f"{encoder_name}.joblib"
    return joblib.load(encoder_path)


def save_feature_names(
    feature_names: List[str],
    path: Path,
    name: str = "feature_names"
) -> None:
    """
    Save feature names for reproducibility.

    Args:
        feature_names: List of feature names
        path: Directory to save to
        name: Name for the file
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    filepath = path / f"{name}.json"
    with open(filepath, 'w') as f:
        json.dump(feature_names, f, indent=2)


def load_feature_names(path: Path, name: str = "feature_names") -> List[str]:
    """
    Load saved feature names.

    Args:
        path: Directory containing the file
        name: Name of the file

    Returns:
        List of feature names
    """
    path = Path(path)
    filepath = path / f"{name}.json"
    with open(filepath, 'r') as f:
        return json.load(f)


def save_results(
    results: Dict,
    path: Path,
    name: str
) -> None:
    """
    Save experiment results as JSON.

    Args:
        results: Dictionary of results
        path: Directory to save to
        name: Name for the file
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    filepath = path / f"{name}.json"
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def save_results_csv(
    results_df: pd.DataFrame,
    path: Path,
    name: str
) -> None:
    """
    Save results DataFrame as CSV.

    Args:
        results_df: DataFrame of results
        path: Directory to save to
        name: Name for the file
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    filepath = path / f"{name}.csv"
    results_df.to_csv(filepath, index=False)


def cross_validate_model(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = 'f1',
    return_estimator: bool = False
) -> Dict[str, Any]:
    """
    Perform stratified k-fold cross-validation.

    Args:
        model: Model to evaluate
        X: Features
        y: Labels
        cv: Number of folds
        scoring: Scoring metric
        return_estimator: Whether to return fitted estimators

    Returns:
        Dictionary with CV results
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)

    results = {
        'scores': scores.tolist(),
        'mean': float(scores.mean()),
        'std': float(scores.std()),
        'cv_folds': cv,
        'scoring': scoring
    }

    return results


def train_with_early_stopping(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    eval_metric: str = 'logloss',
    early_stopping_rounds: int = 50
) -> BaseEstimator:
    """
    Train XGBoost/LightGBM with early stopping.

    Args:
        model: XGBoost or LightGBM model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        eval_metric: Evaluation metric
        early_stopping_rounds: Rounds for early stopping

    Returns:
        Fitted model
    """
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=early_stopping_rounds,
        verbose=False
    )
    return model


def get_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Compute balanced class weights.

    Args:
        y: Target labels

    Returns:
        Dictionary of class -> weight
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)

    return dict(zip(classes, weights))


def get_scale_pos_weight(y: np.ndarray) -> float:
    """
    Compute scale_pos_weight for XGBoost binary classification.

    Args:
        y: Target labels (0 or 1)

    Returns:
        Scale factor for positive class
    """
    n_negative = (y == 0).sum()
    n_positive = (y == 1).sum()

    if n_positive == 0:
        return 1.0

    return n_negative / n_positive


def create_experiment_log(
    experiment_name: str,
    config: Dict,
    results: Dict,
    path: Path
) -> None:
    """
    Create a comprehensive experiment log.

    Args:
        experiment_name: Name of the experiment
        config: Experiment configuration
        results: Experiment results
        path: Directory to save log
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    log = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'results': results
    }

    filepath = path / f"{experiment_name}_log.json"
    with open(filepath, 'w') as f:
        json.dump(log, f, indent=2, default=str)


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
