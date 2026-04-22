"""
Phase 6: Clustering Module

K-Means, DBSCAN, and HDBSCAN clustering for alert grouping.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    homogeneity_score, completeness_score, v_measure_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


class BaseClusterer(ABC):
    """Abstract base class for clusterers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Clusterer name."""
        pass

    @abstractmethod
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict cluster labels."""
        pass

    def get_params(self) -> Dict:
        """Get clusterer parameters."""
        return {}


class KMeansClusterer(BaseClusterer):
    """K-Means clustering."""

    def __init__(self, n_clusters: int = 10, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model_ = None

    @property
    def name(self) -> str:
        return f"KMeans_{self.n_clusters}"

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.model_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        return self.model_.fit_predict(X)

    def get_params(self) -> Dict:
        return {'n_clusters': self.n_clusters}


class DBSCANClusterer(BaseClusterer):
    """DBSCAN clustering."""

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.model_ = None

    @property
    def name(self) -> str:
        return f"DBSCAN_eps{self.eps}_min{self.min_samples}"

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.model_ = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples
        )
        return self.model_.fit_predict(X)

    def get_params(self) -> Dict:
        return {'eps': self.eps, 'min_samples': self.min_samples}


class HDBSCANClusterer(BaseClusterer):
    """HDBSCAN clustering."""

    def __init__(self, min_cluster_size: int = 10, min_samples: int = 5):
        if not HAS_HDBSCAN:
            raise ImportError("hdbscan package not installed")
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.model_ = None

    @property
    def name(self) -> str:
        return f"HDBSCAN_cs{self.min_cluster_size}_ms{self.min_samples}"

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.model_ = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples
        )
        return self.model_.fit_predict(X)

    def get_params(self) -> Dict:
        return {
            'min_cluster_size': self.min_cluster_size,
            'min_samples': self.min_samples
        }


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    true_labels: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluate clustering quality.

    Args:
        X: Feature matrix
        labels: Predicted cluster labels
        true_labels: Ground truth labels (optional)

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Filter out noise points for internal metrics
    mask = labels >= 0
    n_clusters = len(set(labels[mask]))

    metrics['n_clusters'] = n_clusters
    metrics['n_noise'] = np.sum(labels < 0)
    metrics['noise_ratio'] = np.mean(labels < 0)

    # Internal metrics (only if we have at least 2 clusters and non-noise points)
    if n_clusters >= 2 and np.sum(mask) > n_clusters:
        try:
            metrics['silhouette'] = silhouette_score(X[mask], labels[mask])
        except:
            metrics['silhouette'] = -1

        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(X[mask], labels[mask])
        except:
            metrics['calinski_harabasz'] = 0

        try:
            metrics['davies_bouldin'] = davies_bouldin_score(X[mask], labels[mask])
        except:
            metrics['davies_bouldin'] = np.inf

    # External metrics (if ground truth provided)
    if true_labels is not None:
        try:
            metrics['homogeneity'] = homogeneity_score(true_labels, labels)
            metrics['completeness'] = completeness_score(true_labels, labels)
            metrics['v_measure'] = v_measure_score(true_labels, labels)
            metrics['ari'] = adjusted_rand_score(true_labels, labels)
            metrics['nmi'] = normalized_mutual_info_score(true_labels, labels)
        except:
            pass

    return metrics


def get_cluster_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
    profile_cols: List[str]
) -> pd.DataFrame:
    """
    Get summary profiles for each cluster.

    Args:
        df: Original DataFrame
        labels: Cluster labels
        profile_cols: Columns to include in profile

    Returns:
        DataFrame with cluster profiles
    """
    df = df.copy()
    df['cluster'] = labels

    profiles = []
    for cluster_id in sorted(set(labels)):
        if cluster_id < 0:
            continue  # Skip noise

        cluster_data = df[df['cluster'] == cluster_id]
        profile = {'cluster': cluster_id, 'size': len(cluster_data)}

        for col in profile_cols:
            if col not in cluster_data.columns:
                continue

            if cluster_data[col].dtype in ['float64', 'int64']:
                profile[f'{col}_mean'] = cluster_data[col].mean()
                profile[f'{col}_std'] = cluster_data[col].std()
            else:
                # Most common value
                profile[f'{col}_mode'] = cluster_data[col].mode().iloc[0] if len(cluster_data[col].mode()) > 0 else None
                profile[f'{col}_unique'] = cluster_data[col].nunique()

        profiles.append(profile)

    return pd.DataFrame(profiles)


def reduce_dimensions(
    X: np.ndarray,
    n_components: int = 2,
    method: str = 'pca'
) -> np.ndarray:
    """
    Reduce dimensionality for visualization.

    Args:
        X: Feature matrix
        n_components: Number of components
        method: 'pca' or 'umap'

    Returns:
        Reduced feature matrix
    """
    if method == 'pca':
        reducer = PCA(n_components=n_components)
        return reducer.fit_transform(X)
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            return reducer.fit_transform(X)
        except ImportError:
            print("UMAP not installed, falling back to PCA")
            return reduce_dimensions(X, n_components, 'pca')
    else:
        return X[:, :n_components]


def find_optimal_k(
    X: np.ndarray,
    k_range: range = range(2, 21),
    method: str = 'silhouette'
) -> Tuple[int, List[float]]:
    """
    Find optimal number of clusters using elbow method or silhouette.

    Args:
        X: Feature matrix
        k_range: Range of k values to try
        method: 'silhouette' or 'inertia'

    Returns:
        Tuple of (optimal_k, scores)
    """
    scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        if method == 'silhouette':
            score = silhouette_score(X, labels)
        else:
            score = -kmeans.inertia_  # Negative so higher is better

        scores.append(score)

    optimal_idx = np.argmax(scores)
    optimal_k = list(k_range)[optimal_idx]

    return optimal_k, scores
