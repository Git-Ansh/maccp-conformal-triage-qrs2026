"""
Phase 1: Data Loading Module
Loads and validates alerts_data.csv for binary classification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.data_paths import (
    ALERTS_DATA_PATH, REGRESSION_TARGET_COL, TIMESTAMP_COL, ALERT_ID_COL
)


def load_alerts_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load alerts_data.csv with proper data types.

    Args:
        filepath: Path to alerts data CSV. Uses default if not specified.

    Returns:
        DataFrame with alerts data
    """
    if filepath is None:
        filepath = ALERTS_DATA_PATH

    print(f"Loading alerts data from: {filepath}")

    # Load with optimized dtypes
    df = pd.read_csv(
        filepath,
        low_memory=False,
        parse_dates=[TIMESTAMP_COL, 'alert_summary_creation_timestamp']
    )

    print(f"Loaded {len(df)} alerts with {len(df.columns)} columns")

    return df


def validate_data_schema(df: pd.DataFrame) -> bool:
    """
    Validate that required columns are present.

    Args:
        df: Alerts DataFrame

    Returns:
        True if valid, raises ValueError otherwise
    """
    required_columns = [
        ALERT_ID_COL,
        TIMESTAMP_COL,
        REGRESSION_TARGET_COL,
        'single_alert_amount_abs',
        'single_alert_amount_pct',
        'single_alert_t_value',
        'alert_summary_repository'
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("Data schema validation passed")
    return True


def get_class_distribution(df: pd.DataFrame, target_col: str = REGRESSION_TARGET_COL) -> Dict:
    """
    Get class distribution statistics.

    Args:
        df: Alerts DataFrame
        target_col: Target column name

    Returns:
        Dictionary with class counts and percentages
    """
    counts = df[target_col].value_counts()
    percentages = df[target_col].value_counts(normalize=True) * 100

    distribution = {
        'counts': counts.to_dict(),
        'percentages': percentages.to_dict(),
        'total': len(df),
        'n_classes': len(counts)
    }

    print(f"\nClass Distribution for '{target_col}':")
    for class_val, count in counts.items():
        pct = percentages[class_val]
        print(f"  {class_val}: {count} ({pct:.2f}%)")

    return distribution


def filter_valid_labels(
    df: pd.DataFrame,
    target_col: str = REGRESSION_TARGET_COL
) -> pd.DataFrame:
    """
    Filter out rows with missing or invalid target labels.

    Args:
        df: Alerts DataFrame
        target_col: Target column name

    Returns:
        Filtered DataFrame
    """
    initial_count = len(df)

    # Remove rows with missing target
    df = df[df[target_col].notna()].copy()

    removed = initial_count - len(df)
    if removed > 0:
        print(f"Removed {removed} rows with missing '{target_col}'")

    return df


def get_repository_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get distribution of alerts across repositories.

    Args:
        df: Alerts DataFrame

    Returns:
        DataFrame with repository counts
    """
    repo_counts = df['alert_summary_repository'].value_counts().reset_index()
    repo_counts.columns = ['repository', 'count']
    repo_counts['percentage'] = repo_counts['count'] / len(df) * 100

    print("\nRepository Distribution:")
    for _, row in repo_counts.iterrows():
        print(f"  {row['repository']}: {row['count']} ({row['percentage']:.2f}%)")

    return repo_counts


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive data summary.

    Args:
        df: Alerts DataFrame

    Returns:
        Dictionary with data summary statistics
    """
    summary = {
        'n_samples': len(df),
        'n_features': len(df.columns),
        'date_range': {
            'start': str(df[TIMESTAMP_COL].min()),
            'end': str(df[TIMESTAMP_COL].max())
        },
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict()
    }

    # Target distribution
    if REGRESSION_TARGET_COL in df.columns:
        summary['target_distribution'] = df[REGRESSION_TARGET_COL].value_counts().to_dict()

    return summary


def load_and_prepare_data(
    filepath: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load, validate, and prepare alerts data for modeling.

    Args:
        filepath: Path to alerts data CSV

    Returns:
        Tuple of (prepared DataFrame, data summary)
    """
    # Load data
    df = load_alerts_data(filepath)

    # Validate schema
    validate_data_schema(df)

    # Filter valid labels
    df = filter_valid_labels(df)

    # Get summary
    summary = get_data_summary(df)

    # Print summary
    print(f"\n{'='*50}")
    print("Data Preparation Summary")
    print(f"{'='*50}")
    print(f"Total samples: {summary['n_samples']}")
    print(f"Features: {summary['n_features']}")
    print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")

    return df, summary


if __name__ == "__main__":
    # Test loading
    df, summary = load_and_prepare_data()
    print("\nData loaded successfully!")
    print(f"Shape: {df.shape}")
    get_class_distribution(df)
    get_repository_distribution(df)
