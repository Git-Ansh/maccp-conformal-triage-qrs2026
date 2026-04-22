"""
Phase 6: Feature Aggregation

Combines metadata features with time-series features for RCA.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.data_paths import (
    ALERTS_DATA_PATH, BUGS_DATA_PATH, ALERT_ID_COL,
    MAGNITUDE_FEATURES, CONTEXT_FEATURES, REGRESSION_TARGET_COL,
    SIGNATURE_ID_COL, REPOSITORY_COL
)

from phase_3.src.timeseries_loader import build_signature_index, load_timeseries_from_zip
from phase_3.src.ts_feature_engineering import extract_all_features
from phase_3.src.alert_timeseries_matcher import find_alert_index_in_timeseries


def load_alerts_with_features(
    max_samples: Optional[int] = None,
    include_ts_features: bool = True,
    window_size: int = 20
) -> pd.DataFrame:
    """
    Load alerts with metadata and optional time-series features.

    Args:
        max_samples: Maximum number of samples to load
        include_ts_features: Whether to include time-series features
        window_size: Window size for time-series feature extraction

    Returns:
        DataFrame with combined features
    """
    # Load alerts
    alerts_df = pd.read_csv(ALERTS_DATA_PATH)

    if max_samples:
        alerts_df = alerts_df.head(max_samples)

    print(f"Loaded {len(alerts_df)} alerts")

    # Extract metadata features
    feature_cols = []

    # Magnitude features (numeric)
    for col in MAGNITUDE_FEATURES:
        if col in alerts_df.columns:
            feature_cols.append(col)

    # Context features (categorical - need encoding)
    cat_cols = []
    for col in CONTEXT_FEATURES:
        if col in alerts_df.columns:
            cat_cols.append(col)

    # Encode categorical features
    for col in cat_cols:
        le = LabelEncoder()
        alerts_df[f'{col}_encoded'] = le.fit_transform(
            alerts_df[col].fillna('unknown').astype(str)
        )
        feature_cols.append(f'{col}_encoded')

    # Add workflow features
    if 'single_alert_manually_created' in alerts_df.columns:
        alerts_df['manually_created_encoded'] = alerts_df['single_alert_manually_created'].astype(int)
        feature_cols.append('manually_created_encoded')

    # Add time-series features if requested
    if include_ts_features:
        alerts_df = _add_timeseries_features(alerts_df, window_size)
        # Get TS feature columns
        ts_cols = [c for c in alerts_df.columns if c.startswith('ts_')]
        feature_cols.extend(ts_cols)

    alerts_df['feature_columns'] = [feature_cols] * len(alerts_df)

    return alerts_df


def _add_timeseries_features(
    alerts_df: pd.DataFrame,
    window_size: int = 20
) -> pd.DataFrame:
    """Add time-series features to alerts DataFrame."""
    print("Building signature index...")
    signature_index = build_signature_index()

    print(f"Extracting time-series features (window={window_size})...")

    ts_features_list = []

    for idx, row in alerts_df.iterrows():
        sig_id = row.get(SIGNATURE_ID_COL)

        if pd.isna(sig_id) or int(sig_id) not in signature_index:
            ts_features_list.append({})
            continue

        try:
            sig_id = int(sig_id)
            ts_df = load_timeseries_from_zip(signature_index[sig_id], sig_id)

            if ts_df is None or len(ts_df) < window_size:
                ts_features_list.append({})
                continue

            ts_df = ts_df.sort_values('push_timestamp').reset_index(drop=True)
            alert_idx = find_alert_index_in_timeseries(ts_df, row)

            if alert_idx is None or alert_idx < window_size:
                ts_features_list.append({})
                continue

            # Extract window before alert
            window = ts_df.iloc[alert_idx - window_size:alert_idx]
            values = window['value'].values
            alert_value = row.get('single_alert_new_value', values[-1])

            # Extract features
            features = extract_all_features(values, alert_value, window_size)
            ts_features_list.append(features)

        except Exception:
            ts_features_list.append({})

    # Convert to DataFrame
    ts_features_df = pd.DataFrame(ts_features_list)
    ts_features_df.columns = [f'ts_{c}' for c in ts_features_df.columns]

    # Merge with alerts
    alerts_df = pd.concat([alerts_df.reset_index(drop=True), ts_features_df], axis=1)

    print(f"  Added {len(ts_features_df.columns)} time-series features")

    return alerts_df


def load_bug_data() -> pd.DataFrame:
    """
    Load bug reports data.

    Returns:
        DataFrame with bug data
    """
    bugs_df = pd.read_csv(BUGS_DATA_PATH)
    print(f"Loaded {len(bugs_df)} bug reports")

    # Clean text fields
    bugs_df['summary'] = bugs_df['summary'].fillna('')
    bugs_df['component'] = bugs_df['component'].fillna('Unknown')
    bugs_df['product'] = bugs_df['product'].fillna('Unknown')

    return bugs_df


def merge_alert_bug_data(
    alerts_df: pd.DataFrame,
    bugs_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge alerts with their associated bug reports.

    Args:
        alerts_df: Alerts DataFrame
        bugs_df: Bugs DataFrame

    Returns:
        Merged DataFrame
    """
    # alerts_df has alert_summary_bug_number, bugs_df has id
    if 'alert_summary_bug_number' not in alerts_df.columns:
        print("Warning: No bug_number column in alerts")
        alerts_df['has_bug'] = 0
        return alerts_df

    # Mark alerts that have bugs using proper null check
    alerts_df['has_bug'] = alerts_df['alert_summary_bug_number'].notna().astype(int)

    # Prepare for merge
    bugs_df = bugs_df.copy()
    bugs_df['id'] = bugs_df['id'].astype(str)

    # Convert bug number to string for merge, handling NaN properly
    alerts_df = alerts_df.copy()
    alerts_df['_bug_num_str'] = alerts_df['alert_summary_bug_number'].fillna(-1).astype(int).astype(str)
    alerts_df.loc[alerts_df['alert_summary_bug_number'].isna(), '_bug_num_str'] = ''

    # Merge bug info
    merged = alerts_df.merge(
        bugs_df[['id', 'summary', 'component', 'product', 'keywords', 'resolution']],
        left_on='_bug_num_str',
        right_on='id',
        how='left',
        suffixes=('', '_bug')
    )

    # Clean up temp column
    merged = merged.drop(columns=['_bug_num_str'], errors='ignore')

    print(f"  Alerts with bugs: {merged['has_bug'].sum()}")
    print(f"  Alerts without bugs: {(merged['has_bug'] == 0).sum()}")

    return merged


def prepare_clustering_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare features for clustering.

    Args:
        df: DataFrame with features
        feature_cols: List of columns to use (auto-detect if None)

    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    if feature_cols is None:
        # Auto-detect numeric features
        feature_cols = []
        for col in df.columns:
            if col.endswith('_encoded') or col.startswith('ts_') or col in MAGNITUDE_FEATURES:
                if col in df.columns:
                    feature_cols.append(col)

    # Extract feature matrix
    X = df[feature_cols].values

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Prepared {X_scaled.shape[0]} samples x {X_scaled.shape[1]} features")

    return X_scaled, feature_cols
