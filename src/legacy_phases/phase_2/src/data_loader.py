"""
Phase 2: Data Loader Module
Load and prepare alerts data for multi-class status prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import sys

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.data_paths import ALERTS_DATA_PATH, STATUS_TARGET_COL


def load_and_prepare_data(
    data_path: Path = None,
    min_class_samples: int = 10
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load alerts data and prepare for multi-class classification.

    Args:
        data_path: Path to alerts CSV (uses default if None)
        min_class_samples: Minimum samples required per class

    Returns:
        Tuple of (DataFrame, summary dict)
    """
    if data_path is None:
        data_path = ALERTS_DATA_PATH

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    initial_rows = len(df)
    print(f"Initial rows: {initial_rows}")

    # Filter out rows with missing target
    df = df.dropna(subset=[STATUS_TARGET_COL])
    after_target_filter = len(df)
    print(f"After filtering missing {STATUS_TARGET_COL}: {after_target_filter}")

    # Get class distribution
    class_counts = df[STATUS_TARGET_COL].value_counts()
    print(f"\nClass distribution:")
    for cls, count in class_counts.items():
        pct = count / len(df) * 100
        print(f"  {cls}: {count} ({pct:.2f}%)")

    # Identify rare classes
    rare_classes = class_counts[class_counts < min_class_samples].index.tolist()
    if rare_classes:
        print(f"\nRare classes (< {min_class_samples} samples): {rare_classes}")

    summary = {
        'initial_rows': initial_rows,
        'final_rows': len(df),
        'class_counts': class_counts.to_dict(),
        'n_classes': len(class_counts),
        'rare_classes': rare_classes
    }

    return df, summary


def get_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get detailed class distribution statistics.

    Args:
        df: DataFrame with target column

    Returns:
        DataFrame with class statistics
    """
    class_counts = df[STATUS_TARGET_COL].value_counts()
    class_pcts = df[STATUS_TARGET_COL].value_counts(normalize=True) * 100

    dist_df = pd.DataFrame({
        'count': class_counts,
        'percentage': class_pcts.round(2)
    })
    dist_df = dist_df.sort_values('count', ascending=False)

    return dist_df


def group_rare_classes(
    df: pd.DataFrame,
    min_samples: int = 50,
    other_label: str = 'Other'
) -> pd.DataFrame:
    """
    Group rare classes into 'Other' category.

    Args:
        df: DataFrame with target column
        min_samples: Minimum samples to keep class separate
        other_label: Label for grouped classes

    Returns:
        DataFrame with grouped classes
    """
    df = df.copy()
    class_counts = df[STATUS_TARGET_COL].value_counts()

    rare_classes = class_counts[class_counts < min_samples].index.tolist()

    if rare_classes:
        print(f"Grouping rare classes into '{other_label}': {rare_classes}")
        df.loc[df[STATUS_TARGET_COL].isin(rare_classes), STATUS_TARGET_COL] = other_label

    return df


if __name__ == "__main__":
    df, summary = load_and_prepare_data()
    print(f"\nLoaded {len(df)} alerts with {summary['n_classes']} classes")

    dist = get_class_distribution(df)
    print("\nClass Distribution:")
    print(dist)
