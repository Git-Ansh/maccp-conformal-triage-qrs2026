"""
Phase 2: Class Balancing Module
Handle class imbalance using SMOTE, class weights, and other techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN, SMOTETomek
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("Warning: imbalanced-learn not installed. Run: pip install imbalanced-learn")


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Compute balanced class weights.

    Args:
        y: Target labels (integer encoded)

    Returns:
        Dictionary of class -> weight
    """
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    weight_dict = dict(zip(classes, weights))

    print("Class weights:")
    for cls, weight in sorted(weight_dict.items()):
        count = (y == cls).sum()
        print(f"  Class {cls}: weight={weight:.4f} (n={count})")

    return weight_dict


def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: str = 'auto',
    k_neighbors: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE oversampling to handle class imbalance.

    Args:
        X: Feature matrix
        y: Target labels
        sampling_strategy: SMOTE sampling strategy
        k_neighbors: Number of neighbors for SMOTE
        random_state: Random state

    Returns:
        Tuple of (resampled X, resampled y)
    """
    if not HAS_IMBLEARN:
        raise ImportError("imbalanced-learn not installed")

    print(f"\nApplying SMOTE (k_neighbors={k_neighbors})...")
    print(f"Before SMOTE: {Counter(y)}")

    # Adjust k_neighbors based on minority class size
    min_samples = min(Counter(y).values())
    k_neighbors = min(k_neighbors, min_samples - 1)
    k_neighbors = max(1, k_neighbors)

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state
    )

    X_resampled, y_resampled = smote.fit_resample(X, y)

    print(f"After SMOTE: {Counter(y_resampled)}")
    print(f"Samples: {len(y)} -> {len(y_resampled)}")

    return X_resampled, y_resampled


def apply_adasyn(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: str = 'auto',
    n_neighbors: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply ADASYN oversampling.

    Args:
        X: Feature matrix
        y: Target labels
        sampling_strategy: Sampling strategy
        n_neighbors: Number of neighbors
        random_state: Random state

    Returns:
        Tuple of (resampled X, resampled y)
    """
    if not HAS_IMBLEARN:
        raise ImportError("imbalanced-learn not installed")

    print(f"\nApplying ADASYN...")
    print(f"Before ADASYN: {Counter(y)}")

    # Adjust n_neighbors
    min_samples = min(Counter(y).values())
    n_neighbors = min(n_neighbors, min_samples - 1)
    n_neighbors = max(1, n_neighbors)

    adasyn = ADASYN(
        sampling_strategy=sampling_strategy,
        n_neighbors=n_neighbors,
        random_state=random_state,
        n_jobs=-1
    )

    try:
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
        print(f"After ADASYN: {Counter(y_resampled)}")
        return X_resampled, y_resampled
    except ValueError as e:
        print(f"ADASYN failed: {e}. Falling back to SMOTE.")
        return apply_smote(X, y, sampling_strategy, n_neighbors, random_state)


def apply_smote_tomek(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE + Tomek links for cleaning.

    Args:
        X: Feature matrix
        y: Target labels
        random_state: Random state

    Returns:
        Tuple of (resampled X, resampled y)
    """
    if not HAS_IMBLEARN:
        raise ImportError("imbalanced-learn not installed")

    print("\nApplying SMOTETomek...")
    print(f"Before: {Counter(y)}")

    # Adjust k_neighbors
    min_samples = min(Counter(y).values())
    k_neighbors = min(5, min_samples - 1)
    k_neighbors = max(1, k_neighbors)

    smt = SMOTETomek(
        smote=SMOTE(k_neighbors=k_neighbors, random_state=random_state),
        random_state=random_state,
        n_jobs=-1
    )

    X_resampled, y_resampled = smt.fit_resample(X, y)

    print(f"After: {Counter(y_resampled)}")

    return X_resampled, y_resampled


def apply_undersampling(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: str = 'auto',
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random undersampling.

    Args:
        X: Feature matrix
        y: Target labels
        sampling_strategy: Sampling strategy
        random_state: Random state

    Returns:
        Tuple of (resampled X, resampled y)
    """
    if not HAS_IMBLEARN:
        raise ImportError("imbalanced-learn not installed")

    print("\nApplying undersampling...")
    print(f"Before: {Counter(y)}")

    rus = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )

    X_resampled, y_resampled = rus.fit_resample(X, y)

    print(f"After: {Counter(y_resampled)}")

    return X_resampled, y_resampled


def get_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Compute per-sample weights for training.

    Args:
        y: Target labels

    Returns:
        Array of sample weights
    """
    class_weights = compute_class_weights(y)
    sample_weights = np.array([class_weights[label] for label in y])
    return sample_weights


def summarize_class_distribution(y: np.ndarray, label_encoder=None) -> pd.DataFrame:
    """
    Create a summary DataFrame of class distribution.

    Args:
        y: Target labels
        label_encoder: Optional LabelEncoder for class names

    Returns:
        DataFrame with class statistics
    """
    counts = Counter(y)
    total = len(y)

    data = []
    for cls in sorted(counts.keys()):
        count = counts[cls]
        name = label_encoder.classes_[cls] if label_encoder else f"Class {cls}"
        data.append({
            'class_id': cls,
            'class_name': name,
            'count': count,
            'percentage': count / total * 100
        })

    return pd.DataFrame(data)


if __name__ == "__main__":
    # Test class balancing
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_classes=5, weights=[0.6, 0.2, 0.1, 0.05, 0.05],
        random_state=42
    )

    print("Testing class balancing utilities...")

    # Test class weights
    weights = compute_class_weights(y)

    # Test SMOTE
    if HAS_IMBLEARN:
        X_smote, y_smote = apply_smote(X, y)
        print(f"\nSMOTE increased samples: {len(y)} -> {len(y_smote)}")

    # Test summary
    summary = summarize_class_distribution(y)
    print("\nClass distribution summary:")
    print(summary)
