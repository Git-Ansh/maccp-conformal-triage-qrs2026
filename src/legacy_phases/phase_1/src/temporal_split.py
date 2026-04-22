"""
Phase 1: Temporal Split Module
Time-based train/validation/test splits to prevent data leakage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import sys

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.data_paths import TIMESTAMP_COL


def temporal_train_val_test_split(
    df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally to avoid data leakage.

    Training data is always BEFORE validation data,
    which is always BEFORE test data.

    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # Sort by timestamp
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    n = len(df)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    # Verify temporal ordering
    train_max = train_df[timestamp_col].max()
    val_min = val_df[timestamp_col].min()
    val_max = val_df[timestamp_col].max()
    test_min = test_df[timestamp_col].min()

    print("\nTemporal Split Summary:")
    print(f"  Train: {len(train_df)} samples ({train_ratio*100:.0f}%)")
    print(f"    Date range: {train_df[timestamp_col].min()} to {train_max}")
    print(f"  Validation: {len(val_df)} samples ({val_ratio*100:.0f}%)")
    print(f"    Date range: {val_min} to {val_max}")
    print(f"  Test: {len(test_df)} samples ({test_ratio*100:.0f}%)")
    print(f"    Date range: {test_min} to {test_df[timestamp_col].max()}")

    # Verify no leakage
    assert train_max <= val_min, "LEAKAGE: Train data overlaps with validation!"
    assert val_max <= test_min, "LEAKAGE: Validation data overlaps with test!"
    print("\n[OK] Temporal ordering verified - no data leakage")

    return train_df, val_df, test_df


def temporal_train_test_split(
    df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL,
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple temporal train/test split.

    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
        train_ratio: Proportion for training

    Returns:
        Tuple of (train_df, test_df)
    """
    # Sort by timestamp
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    n = len(df)
    train_end = int(train_ratio * n)

    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[train_end:].copy()

    # Verify temporal ordering
    train_max = train_df[timestamp_col].max()
    test_min = test_df[timestamp_col].min()

    print("\nTemporal Split Summary:")
    print(f"  Train: {len(train_df)} samples ({train_ratio*100:.0f}%)")
    print(f"    Date range: {train_df[timestamp_col].min()} to {train_max}")
    print(f"  Test: {len(test_df)} samples ({(1-train_ratio)*100:.0f}%)")
    print(f"    Date range: {test_min} to {test_df[timestamp_col].max()}")

    # Verify no leakage
    assert train_max <= test_min, "LEAKAGE: Train data overlaps with test!"
    print("\n[OK] Temporal ordering verified - no data leakage")

    return train_df, test_df


def get_split_indices(
    df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get indices for temporal split (useful for preprocessed data).

    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
        train_ratio: Proportion for training
        val_ratio: Proportion for validation

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    # Sort and get indices
    sorted_indices = df.sort_values(timestamp_col).index.values

    n = len(sorted_indices)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    train_idx = sorted_indices[:train_end]
    val_idx = sorted_indices[train_end:val_end]
    test_idx = sorted_indices[val_end:]

    return train_idx, val_idx, test_idx


def split_by_repository(
    df: pd.DataFrame,
    train_repos: list,
    test_repos: list,
    repo_col: str = 'alert_summary_repository'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by repository (for cross-repository evaluation).

    Args:
        df: DataFrame with repository column
        train_repos: List of repository names for training
        test_repos: List of repository names for testing
        repo_col: Name of repository column

    Returns:
        Tuple of (train_df, test_df)
    """
    train_df = df[df[repo_col].isin(train_repos)].copy()
    test_df = df[df[repo_col].isin(test_repos)].copy()

    print(f"\nRepository Split:")
    print(f"  Train repos: {train_repos}")
    print(f"    Samples: {len(train_df)}")
    print(f"  Test repos: {test_repos}")
    print(f"    Samples: {len(test_df)}")

    return train_df, test_df


def get_cross_repo_splits(
    df: pd.DataFrame,
    repo_col: str = 'alert_summary_repository'
) -> list:
    """
    Generate leave-one-repository-out splits.

    Args:
        df: DataFrame with repository column
        repo_col: Name of repository column

    Returns:
        List of (train_repos, test_repo) tuples
    """
    repos = df[repo_col].unique().tolist()
    splits = []

    for test_repo in repos:
        train_repos = [r for r in repos if r != test_repo]
        splits.append((train_repos, test_repo))

    print(f"\nGenerated {len(splits)} cross-repository splits")

    return splits


def verify_no_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL
) -> bool:
    """
    Verify there is no temporal leakage between train and test.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        timestamp_col: Name of timestamp column

    Returns:
        True if no leakage, raises AssertionError otherwise
    """
    train_max = train_df[timestamp_col].max()
    test_min = test_df[timestamp_col].min()

    if train_max > test_min:
        overlap = train_df[train_df[timestamp_col] > test_min]
        raise AssertionError(
            f"TEMPORAL LEAKAGE DETECTED!\n"
            f"  Train max timestamp: {train_max}\n"
            f"  Test min timestamp: {test_min}\n"
            f"  Overlapping samples: {len(overlap)}"
        )

    print("[OK] No temporal leakage detected")
    return True


if __name__ == "__main__":
    # Test temporal split
    from data_loader import load_and_prepare_data

    df, _ = load_and_prepare_data()

    # Test train/val/test split
    train_df, val_df, test_df = temporal_train_val_test_split(df)

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val: {len(val_df)}")
    print(f"  Test: {len(test_df)}")

    # Verify
    verify_no_leakage(train_df, test_df)
