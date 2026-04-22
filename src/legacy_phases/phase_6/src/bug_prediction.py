"""
Phase 6: Bug Prediction Module

Predict which alerts will result in bug reports.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.data_paths import RANDOM_SEED


class BugPredictionModel:
    """Model to predict if an alert will lead to a bug report."""

    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize model.

        Args:
            model_type: 'xgboost', 'rf', 'gbm', or 'lr'
        """
        self.model_type = model_type
        self.model_ = None
        self.scaler_ = StandardScaler()
        self.imputer_ = SimpleImputer(strategy='median')
        self.feature_names_ = None

    @property
    def name(self) -> str:
        return f"BugPredictor_{self.model_type}"

    def _create_model(self):
        """Create the underlying model."""
        if self.model_type == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=RANDOM_SEED,
                tree_method='hist',
                device='cuda',
                eval_metric='logloss'
            )
        elif self.model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
        elif self.model_type == 'gbm':
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=RANDOM_SEED
            )
        elif self.model_type == 'lr':
            return LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_SEED
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> 'BugPredictionModel':
        """
        Fit the model.

        Args:
            X: Feature matrix
            y: Target labels (1=has bug, 0=no bug)
            feature_names: Names of features

        Returns:
            self
        """
        self.feature_names_ = feature_names

        # Impute and scale
        X = self.imputer_.fit_transform(X)
        X = self.scaler_.fit_transform(X)

        # Create and fit model
        self.model_ = self._create_model()
        self.model_.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict bug labels."""
        X = self.imputer_.transform(X)
        X = self.scaler_.transform(X)
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict bug probabilities."""
        X = self.imputer_.transform(X)
        X = self.scaler_.transform(X)
        return self.model_.predict_proba(X)[:, 1]

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        metrics = {
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0)
        }

        if len(np.unique(y)) > 1:
            try:
                metrics['roc_auc'] = roc_auc_score(y, y_proba)
            except:
                metrics['roc_auc'] = 0.5

        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if self.model_ is None:
            return pd.DataFrame()

        if hasattr(self.model_, 'feature_importances_'):
            importances = self.model_.feature_importances_
        elif hasattr(self.model_, 'coef_'):
            importances = np.abs(self.model_.coef_[0])
        else:
            return pd.DataFrame()

        if self.feature_names_ is None:
            self.feature_names_ = [f'feature_{i}' for i in range(len(importances))]

        return pd.DataFrame({
            'feature': self.feature_names_,
            'importance': importances
        }).sort_values('importance', ascending=False)

    def save(self, path: Path):
        """Save model to file."""
        joblib.dump({
            'model': self.model_,
            'scaler': self.scaler_,
            'imputer': self.imputer_,
            'feature_names': self.feature_names_,
            'model_type': self.model_type
        }, path)

    @classmethod
    def load(cls, path: Path) -> 'BugPredictionModel':
        """Load model from file."""
        data = joblib.load(path)
        model = cls(model_type=data['model_type'])
        model.model_ = data['model']
        model.scaler_ = data['scaler']
        model.imputer_ = data['imputer']
        model.feature_names_ = data['feature_names']
        return model


def prepare_bug_prediction_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare data for bug prediction.

    Args:
        df: DataFrame with features and has_bug column
        feature_cols: Feature columns to use
        test_size: Test split ratio

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    # Filter to valid feature columns
    valid_cols = [c for c in feature_cols if c in df.columns]

    # Extract features and target
    X = df[valid_cols].values
    y = df['has_bug'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y
    )

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Bug ratio - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test, valid_cols


def cross_validate_bug_prediction(
    X: np.ndarray,
    y: np.ndarray,
    model_types: List[str] = ['xgboost', 'rf', 'gbm'],
    cv: int = 5
) -> pd.DataFrame:
    """
    Cross-validate multiple models for bug prediction.

    Args:
        X: Feature matrix
        y: Target labels
        model_types: List of model types to evaluate
        cv: Number of CV folds

    Returns:
        DataFrame with cross-validation results
    """
    results = []

    # Impute and scale once
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_clean = scaler.fit_transform(imputer.fit_transform(X))

    for model_type in model_types:
        print(f"  Cross-validating {model_type}...")

        if model_type == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=100, max_depth=6,
                random_state=RANDOM_SEED, eval_metric='logloss'
            )
        elif model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=100, max_depth=10,
                random_state=RANDOM_SEED, n_jobs=-1
            )
        elif model_type == 'gbm':
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=5,
                random_state=RANDOM_SEED
            )
        elif model_type == 'lr':
            model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        else:
            continue

        try:
            scores = cross_val_score(model, X_clean, y, cv=cv, scoring='f1')
            results.append({
                'model': model_type,
                'f1_mean': np.mean(scores),
                'f1_std': np.std(scores),
                'cv_folds': cv
            })
            print(f"    F1: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
        except Exception as e:
            print(f"    Error: {e}")

    return pd.DataFrame(results)
