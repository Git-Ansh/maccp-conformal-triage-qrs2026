"""
Phase 7: Feature Normalizer

Normalize and prepare features for ensemble modeling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


class FeatureNormalizer:
    """Normalize features for ensemble modeling."""

    def __init__(
        self,
        scaling_method: str = 'standard',
        impute_strategy: str = 'median'
    ):
        self.scaling_method = scaling_method
        self.impute_strategy = impute_strategy
        self.imputer_ = None
        self.scaler_ = None
        self.feature_names_ = None

    def fit_transform(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Fit and transform features.

        Args:
            X: Feature matrix
            feature_names: Names of features

        Returns:
            Normalized feature matrix
        """
        self.feature_names_ = feature_names

        # Impute missing values
        self.imputer_ = SimpleImputer(strategy=self.impute_strategy)
        X = self.imputer_.fit_transform(X)

        # Scale features
        if self.scaling_method == 'standard':
            self.scaler_ = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler_ = MinMaxScaler()
        else:
            self.scaler_ = None

        if self.scaler_ is not None:
            X = self.scaler_.fit_transform(X)

        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using fitted normalizer."""
        X = self.imputer_.transform(X)
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        return X


def normalize_features(
    X: np.ndarray,
    method: str = 'standard'
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalize features.

    Args:
        X: Feature matrix
        method: 'standard' or 'minmax'

    Returns:
        Tuple of (normalized_X, scaler)
    """
    # Impute first
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    # Scale
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    n_features: int = 50,
    method: str = 'f_classif',
    feature_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[int], List[str]]:
    """
    Select top features.

    Args:
        X: Feature matrix
        y: Target labels
        n_features: Number of features to select
        method: 'f_classif' or 'mutual_info'
        feature_names: Names of features

    Returns:
        Tuple of (selected_X, selected_indices, selected_names)
    """
    # Handle n_features > actual features
    n_features = min(n_features, X.shape[1])

    if method == 'f_classif':
        selector = SelectKBest(f_classif, k=n_features)
    else:
        selector = SelectKBest(mutual_info_classif, k=n_features)

    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True).tolist()

    if feature_names is not None:
        selected_names = [feature_names[i] for i in selected_indices]
    else:
        selected_names = [f'feature_{i}' for i in selected_indices]

    return X_selected, selected_indices, selected_names


def reduce_dimensions(
    X: np.ndarray,
    n_components: int = 20,
    method: str = 'pca'
) -> Tuple[np.ndarray, float]:
    """
    Reduce dimensionality.

    Args:
        X: Feature matrix
        n_components: Number of components
        method: 'pca'

    Returns:
        Tuple of (reduced_X, explained_variance_ratio)
    """
    n_components = min(n_components, X.shape[1], X.shape[0])

    if method == 'pca':
        reducer = PCA(n_components=n_components)
        X_reduced = reducer.fit_transform(X)
        explained_var = sum(reducer.explained_variance_ratio_)
        return X_reduced, explained_var

    return X[:, :n_components], 0.0


def prepare_ensemble_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    normalize: bool = True,
    select_k: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare features for ensemble training.

    Args:
        df: DataFrame with features
        feature_cols: Feature column names
        target_col: Target column name
        normalize: Whether to normalize
        select_k: Number of features to select (None = all)

    Returns:
        Tuple of (X, y, feature_names)
    """
    # Filter to valid columns
    valid_cols = [c for c in feature_cols if c in df.columns]

    if not valid_cols:
        raise ValueError("No valid feature columns found")

    X = df[valid_cols].values
    y = df[target_col].fillna(0).astype(int).values

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    # Normalize
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Feature selection
    if select_k is not None and select_k < len(valid_cols):
        X, selected_idx, valid_cols = select_features(
            X, y, n_features=select_k, feature_names=valid_cols
        )

    print(f"Prepared {X.shape[0]} samples x {X.shape[1]} features")

    return X, y, valid_cols
