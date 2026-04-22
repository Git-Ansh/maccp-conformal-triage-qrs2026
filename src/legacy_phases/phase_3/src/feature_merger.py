"""
Phase 3: Feature Merger
Merge time-series features with metadata features from Phase 1.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'phase_1' / 'src'))

from common.data_paths import PHASE_1_DIR, PHASE_3_DIR


def load_phase1_features() -> Tuple[pd.DataFrame, List[str]]:
    """
    Load preprocessed features from Phase 1.

    Returns:
        Tuple of (features DataFrame, feature names)
    """
    feature_table_path = PHASE_1_DIR / 'outputs' / 'feature_tables' / 'processed_features.parquet'

    if not feature_table_path.exists():
        raise FileNotFoundError(f"Phase 1 features not found at {feature_table_path}")

    df = pd.read_parquet(feature_table_path)
    feature_names = df.columns.tolist()

    print(f"Loaded Phase 1 features: {df.shape}")

    return df, feature_names


def create_ts_feature_dataframe(
    matched_alerts: List[Dict],
    ts_features: List[Dict]
) -> pd.DataFrame:
    """
    Create a DataFrame from extracted time-series features.

    Args:
        matched_alerts: List of matched alert dictionaries
        ts_features: List of time-series feature dictionaries

    Returns:
        DataFrame with time-series features indexed by alert_id
    """
    records = []

    for matched, features in zip(matched_alerts, ts_features):
        alert_row = matched['alert_row']
        alert_id = alert_row.get('single_alert_id')

        if alert_id is None:
            continue

        record = {'single_alert_id': int(alert_id)}
        record.update(features)
        records.append(record)

    df = pd.DataFrame(records)
    df = df.set_index('single_alert_id')

    print(f"Created TS feature DataFrame: {df.shape}")

    return df


def merge_features(
    metadata_df: pd.DataFrame,
    ts_features_df: pd.DataFrame,
    alerts_df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Merge metadata features with time-series features.

    Args:
        metadata_df: Phase 1 metadata features
        ts_features_df: Time-series features
        alerts_df: Original alerts DataFrame with alert IDs

    Returns:
        Tuple of (merged DataFrame, metadata feature names, ts feature names)
    """
    # Get alert IDs from original alerts
    if 'single_alert_id' in alerts_df.columns:
        alert_ids = alerts_df['single_alert_id'].values
    else:
        alert_ids = alerts_df.index.values

    # Add alert_id to metadata
    metadata_df = metadata_df.copy()
    metadata_df['single_alert_id'] = alert_ids

    # Get feature names
    metadata_cols = [c for c in metadata_df.columns if c != 'single_alert_id']
    ts_cols = ts_features_df.columns.tolist()

    # Merge on alert_id
    merged = metadata_df.merge(
        ts_features_df.reset_index(),
        on='single_alert_id',
        how='inner'
    )

    print(f"\nMerge results:")
    print(f"  Metadata features: {len(metadata_cols)}")
    print(f"  TS features: {len(ts_cols)}")
    print(f"  Total features: {len(metadata_cols) + len(ts_cols)}")
    print(f"  Matched samples: {len(merged)}")

    return merged, metadata_cols, ts_cols


def handle_missing_ts_features(
    df: pd.DataFrame,
    ts_cols: List[str]
) -> pd.DataFrame:
    """
    Handle missing values in time-series features.

    Args:
        df: DataFrame with merged features
        ts_cols: Time-series column names

    Returns:
        DataFrame with handled missing values
    """
    df = df.copy()

    # Fill NaN with median for each TS column
    for col in ts_cols:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Replace inf values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df


def save_merged_features(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = 'merged_features'
) -> Path:
    """
    Save merged features to disk.

    Args:
        df: Merged features DataFrame
        output_dir: Output directory
        filename: Output filename (without extension)

    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f'{filename}.parquet'
    df.to_parquet(output_path, index=False)

    print(f"Saved merged features to {output_path}")

    return output_path


if __name__ == "__main__":
    # Test feature merger
    print("Testing feature merger...")

    try:
        metadata_df, metadata_cols = load_phase1_features()
        print(f"\nMetadata features: {len(metadata_cols)}")
        print(f"Sample columns: {metadata_cols[:5]}")
    except FileNotFoundError as e:
        print(f"Could not load Phase 1 features: {e}")
