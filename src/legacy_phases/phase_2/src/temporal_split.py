"""
Phase 2: Temporal Split Module
Reuses Phase 1 temporal splitting logic for multi-class classification.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
import sys

# Add Phase 1 to path for reuse
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'phase_1' / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.data_paths import TIMESTAMP_COL, REPOSITORY_COL, RANDOM_SEED


def temporal_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    timestamp_col: str = TIMESTAMP_COL
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally into train/val/test sets.

    Args:
        df: Input DataFrame
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        timestamp_col: Column with timestamps

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Sort by timestamp
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"\nTemporal split:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/n*100:.1f}%)")
    print(f"  Val:   {len(val_df)} samples ({len(val_df)/n*100:.1f}%)")
    print(f"  Test:  {len(test_df)} samples ({len(test_df)/n*100:.1f}%)")

    # Show date ranges
    if timestamp_col in df.columns:
        print(f"\nDate ranges:")
        print(f"  Train: {train_df[timestamp_col].min()} to {train_df[timestamp_col].max()}")
        print(f"  Val:   {val_df[timestamp_col].min()} to {val_df[timestamp_col].max()}")
        print(f"  Test:  {test_df[timestamp_col].min()} to {test_df[timestamp_col].max()}")

    return train_df, val_df, test_df


def get_split_indices(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    timestamp_col: str = TIMESTAMP_COL
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get indices for temporal split.

    Args:
        df: Input DataFrame
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        timestamp_col: Column with timestamps

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    # Sort by timestamp and get original indices
    sorted_idx = df.sort_values(timestamp_col).index.values

    n = len(sorted_idx)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = sorted_idx[:train_end]
    val_idx = sorted_idx[train_end:val_end]
    test_idx = sorted_idx[val_end:]

    return train_idx, val_idx, test_idx


def split_by_repository(
    df: pd.DataFrame,
    test_repos: List[str],
    repository_col: str = REPOSITORY_COL
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by repository for cross-repository evaluation.

    Args:
        df: Input DataFrame
        test_repos: List of repositories for testing
        repository_col: Column with repository names

    Returns:
        Tuple of (train_df, test_df)
    """
    test_mask = df[repository_col].isin(test_repos)
    train_df = df[~test_mask]
    test_df = df[test_mask]

    print(f"\nRepository split:")
    print(f"  Train repos: {df[~test_mask][repository_col].unique().tolist()}")
    print(f"  Test repos: {test_repos}")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test:  {len(test_df)} samples")

    return train_df, test_df


if __name__ == "__main__":
    # Test splitting
    from data_loader import load_and_prepare_data

    df, summary = load_and_prepare_data()

    train_df, val_df, test_df = temporal_train_val_test_split(df)

    print(f"\nSplit complete:")
    print(f"  Total: {len(df)}")
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")
