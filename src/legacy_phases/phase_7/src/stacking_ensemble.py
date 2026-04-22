"""
Phase 7: Stacking Ensemble

Stacking ensemble combining multiple base models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    classification_report
)
import xgboost as xgb
import joblib
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.data_paths import RANDOM_SEED


class BaseModel:
    """Wrapper for base models in the ensemble."""

    def __init__(self, name: str, model: Any):
        self.name = name
        self.model = model
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        return self.predict(X).astype(float)


class StackingEnsemble:
    """
    Two-level stacking ensemble for regression detection.

    Level 1: Multiple base classifiers
    Level 2: Meta-classifier combining base predictions
    """

    def __init__(
        self,
        base_models: Optional[List[BaseModel]] = None,
        meta_model: Optional[Any] = None,
        cv_folds: int = 5,
        use_probas: bool = True
    ):
        self.cv_folds = cv_folds
        self.use_probas = use_probas

        # Default base models
        if base_models is None:
            self.base_models = self._create_default_base_models()
        else:
            self.base_models = base_models

        # Default meta model
        if meta_model is None:
            self.meta_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=RANDOM_SEED,
                eval_metric='logloss'
            )
        else:
            self.meta_model = meta_model

        self.meta_features_ = None

    def _create_default_base_models(self) -> List[BaseModel]:
        """Create default set of base models."""
        return [
            BaseModel('LogisticRegression', LogisticRegression(
                max_iter=1000, random_state=RANDOM_SEED
            )),
            BaseModel('RandomForest', RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1
            )),
            BaseModel('GradientBoosting', GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=RANDOM_SEED
            )),
            BaseModel('XGBoost', xgb.XGBClassifier(
                n_estimators=100, max_depth=6, random_state=RANDOM_SEED,
                eval_metric='logloss'
            ))
        ]

    def _generate_meta_features(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Generate meta-features using cross-validation.

        Each base model produces out-of-fold predictions that become
        input features for the meta-model.
        """
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        meta_features = np.zeros((n_samples, n_models))

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=RANDOM_SEED)

        for i, base_model in enumerate(self.base_models):
            print(f"  Generating meta-features from {base_model.name}...")

            if self.use_probas and hasattr(base_model.model, 'predict_proba'):
                # Use probability predictions
                try:
                    oof_preds = cross_val_predict(
                        base_model.model, X, y, cv=cv, method='predict_proba'
                    )[:, 1]
                except:
                    oof_preds = cross_val_predict(
                        base_model.model, X, y, cv=cv, method='predict'
                    )
            else:
                # Use class predictions
                oof_preds = cross_val_predict(
                    base_model.model, X, y, cv=cv, method='predict'
                )

            meta_features[:, i] = oof_preds

        return meta_features

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingEnsemble':
        """
        Fit the stacking ensemble.

        1. Generate out-of-fold predictions from base models
        2. Train meta-model on these predictions
        3. Refit base models on full data
        """
        print("Fitting stacking ensemble...")
        print(f"  {len(self.base_models)} base models, {self.cv_folds}-fold CV")

        # Generate meta-features
        self.meta_features_ = self._generate_meta_features(X, y)

        # Train meta-model
        print("  Training meta-classifier...")
        self.meta_model.fit(self.meta_features_, y)

        # NOTE: Refitting base models on full data removed to prevent contamination
        # The base models are already fitted during CV-based meta-feature generation
        # Refitting would create optimistic bias as models see data they'll predict on
        # Original (problematic) code:
        # for base_model in self.base_models:
        #     base_model.fit(X, y)

        print("  Base models remain fitted from CV (prevents contamination)")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the ensemble."""
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict(meta_features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using the ensemble."""
        meta_features = self._get_meta_features(X)
        if hasattr(self.meta_model, 'predict_proba'):
            return self.meta_model.predict_proba(meta_features)[:, 1]
        return self.predict(X).astype(float)

    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Get meta-features from base model predictions."""
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        meta_features = np.zeros((n_samples, n_models))

        for i, base_model in enumerate(self.base_models):
            if self.use_probas:
                meta_features[:, i] = base_model.predict_proba(X)
            else:
                meta_features[:, i] = base_model.predict(X)

        return meta_features

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate ensemble performance."""
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

    def get_base_model_performance(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> pd.DataFrame:
        """Get performance of individual base models."""
        results = []

        for base_model in self.base_models:
            y_pred = base_model.predict(X)
            y_proba = base_model.predict_proba(X)

            results.append({
                'model': base_model.name,
                'precision': precision_score(y, y_pred, zero_division=0),
                'recall': recall_score(y, y_pred, zero_division=0),
                'f1_score': f1_score(y, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.5
            })

        return pd.DataFrame(results)

    def save(self, path: Path):
        """Save ensemble to file."""
        joblib.dump({
            'base_models': self.base_models,
            'meta_model': self.meta_model,
            'cv_folds': self.cv_folds,
            'use_probas': self.use_probas
        }, path)

    @classmethod
    def load(cls, path: Path) -> 'StackingEnsemble':
        """Load ensemble from file."""
        data = joblib.load(path)
        ensemble = cls(
            base_models=data['base_models'],
            meta_model=data['meta_model'],
            cv_folds=data['cv_folds'],
            use_probas=data['use_probas']
        )
        return ensemble


def train_base_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    Train and evaluate individual base models.

    Returns DataFrame with model comparison.
    """
    models = [
        ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ('RandomForest', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_SEED)),
        ('GradientBoosting', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_SEED)),
        ('XGBoost', xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_SEED, eval_metric='logloss'))
    ]

    results = []

    for name, model in models:
        print(f"  Training {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

        results.append({
            'model': name,
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5
        })

    return pd.DataFrame(results)


def train_meta_classifier(
    meta_features: np.ndarray,
    y: np.ndarray,
    model_type: str = 'xgboost'
) -> Any:
    """
    Train the meta-classifier.

    Args:
        meta_features: Stacked predictions from base models
        y: True labels
        model_type: Type of meta-classifier

    Returns:
        Trained meta-classifier
    """
    if model_type == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4,
            random_state=RANDOM_SEED, eval_metric='logloss'
        )
    elif model_type == 'lr':
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)

    model.fit(meta_features, y)
    return model
