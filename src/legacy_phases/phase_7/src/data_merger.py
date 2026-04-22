"""
Phase 7: Data Merger

Merge outputs from all previous phases into a unified dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.data_paths import (
    ALERTS_DATA_PATH, PHASE_1_DIR, PHASE_2_DIR, PHASE_3_DIR,
    PHASE_4_DIR, PHASE_5_DIR, PHASE_6_DIR,
    ALERT_ID_COL, REGRESSION_TARGET_COL, MAGNITUDE_FEATURES, CONTEXT_FEATURES
)


def load_phase_outputs() -> Dict[str, pd.DataFrame]:
    """
    Load outputs from all phases.

    Returns:
        Dictionary mapping phase name to DataFrame
    """
    outputs = {}

    # Load base alerts
    print("Loading base alerts...")
    alerts_df = pd.read_csv(ALERTS_DATA_PATH)
    outputs['alerts'] = alerts_df
    print(f"  Loaded {len(alerts_df)} alerts")

    # Phase 1 outputs (if available)
    phase1_report = PHASE_1_DIR / 'outputs' / 'reports'
    if phase1_report.exists():
        print("Loading Phase 1 outputs...")
        # Load feature importance or predictions if saved
        for f in phase1_report.glob('*.csv'):
            outputs[f'phase1_{f.stem}'] = pd.read_csv(f)

    # Phase 3 outputs (time-series features)
    phase3_report = PHASE_3_DIR / 'outputs' / 'reports'
    if phase3_report.exists():
        print("Loading Phase 3 outputs...")
        for f in phase3_report.glob('*.csv'):
            outputs[f'phase3_{f.stem}'] = pd.read_csv(f)

    # Phase 4 outputs (change-point detection)
    phase4_report = PHASE_4_DIR / 'outputs' / 'reports'
    if phase4_report.exists():
        print("Loading Phase 4 outputs...")
        for f in phase4_report.glob('*.csv'):
            outputs[f'phase4_{f.stem}'] = pd.read_csv(f)

    # Phase 5 outputs (forecasting)
    phase5_report = PHASE_5_DIR / 'outputs' / 'reports'
    if phase5_report.exists():
        print("Loading Phase 5 outputs...")
        for f in phase5_report.glob('*.csv'):
            outputs[f'phase5_{f.stem}'] = pd.read_csv(f)

    # Phase 6 outputs (RCA)
    phase6_report = PHASE_6_DIR / 'outputs' / 'reports'
    if phase6_report.exists():
        print("Loading Phase 6 outputs...")
        for f in phase6_report.glob('*.csv'):
            outputs[f'phase6_{f.stem}'] = pd.read_csv(f)

    return outputs


def merge_all_features(
    alerts_df: pd.DataFrame,
    include_ts_features: bool = True,
    include_rca_features: bool = True,
    max_samples: Optional[int] = None
) -> pd.DataFrame:
    """
    Merge all feature types into a unified dataset.

    Args:
        alerts_df: Base alerts DataFrame
        include_ts_features: Whether to include time-series features
        include_rca_features: Whether to include RCA features
        max_samples: Maximum samples (for testing)

    Returns:
        Unified DataFrame with all features
    """
    if max_samples:
        alerts_df = alerts_df.head(max_samples)

    # Start with metadata features
    feature_cols = []

    # Magnitude features
    for col in MAGNITUDE_FEATURES:
        if col in alerts_df.columns:
            feature_cols.append(col)

    # Encode categorical features
    from sklearn.preprocessing import LabelEncoder
    for col in CONTEXT_FEATURES:
        if col in alerts_df.columns:
            le = LabelEncoder()
            alerts_df[f'{col}_encoded'] = le.fit_transform(
                alerts_df[col].fillna('unknown').astype(str)
            )
            feature_cols.append(f'{col}_encoded')

    # Workflow features
    if 'single_alert_manually_created' in alerts_df.columns:
        alerts_df['manually_created'] = alerts_df['single_alert_manually_created'].astype(int)
        feature_cols.append('manually_created')

    # Add time-series features if requested
    if include_ts_features:
        alerts_df = _add_ts_features(alerts_df)
        ts_cols = [c for c in alerts_df.columns if c.startswith('ts_')]
        feature_cols.extend(ts_cols)

    # Add RCA features if requested
    if include_rca_features:
        alerts_df = _add_rca_features(alerts_df)
        rca_cols = [c for c in alerts_df.columns if c.startswith('rca_')]
        feature_cols.extend(rca_cols)

    alerts_df['_feature_cols'] = [feature_cols] * len(alerts_df)

    return alerts_df


def _add_ts_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-series features from Phase 3."""
    from phase_3.src.timeseries_loader import build_signature_index, load_timeseries_from_zip
    from phase_3.src.ts_feature_engineering import extract_all_features
    from phase_3.src.alert_timeseries_matcher import find_alert_index_in_timeseries
    from common.data_paths import SIGNATURE_ID_COL

    print("Extracting time-series features...")
    signature_index = build_signature_index()

    window_size = 20
    ts_features_list = []

    for idx, row in df.iterrows():
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

            window = ts_df.iloc[alert_idx - window_size:alert_idx]
            values = window['value'].values
            alert_value = row.get('single_alert_new_value', values[-1])

            features = extract_all_features(values, alert_value, window_size)
            ts_features_list.append(features)

        except Exception:
            ts_features_list.append({})

    ts_df = pd.DataFrame(ts_features_list)
    ts_df.columns = [f'ts_{c}' for c in ts_df.columns]

    df = pd.concat([df.reset_index(drop=True), ts_df], axis=1)
    print(f"  Added {len(ts_df.columns)} time-series features")

    return df


def _add_rca_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add RCA features from Phase 6."""
    # Check if Phase 6 cluster assignments exist
    cluster_file = PHASE_6_DIR / 'outputs' / 'reports' / 'E1_cluster_profiles.csv'

    if not cluster_file.exists():
        print("  No RCA cluster data found, skipping")
        return df

    # For now, add placeholder RCA features based on metadata
    # In a full implementation, we would load actual cluster assignments

    # Add has_bug feature
    df['rca_has_bug'] = df['alert_summary_bug_number'].notna().astype(int)

    # Add downstream feature
    df['rca_is_downstream'] = df['single_alert_related_summary_id'].notna().astype(int)

    print("  Added RCA features")

    return df


def create_unified_dataset(
    max_samples: Optional[int] = None,
    include_ts: bool = True,
    include_rca: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create a unified dataset for ensemble training.

    Args:
        max_samples: Maximum samples
        include_ts: Include time-series features
        include_rca: Include RCA features

    Returns:
        Tuple of (DataFrame, feature_columns)
    """
    alerts_df = pd.read_csv(ALERTS_DATA_PATH)

    if max_samples:
        alerts_df = alerts_df.head(max_samples)

    df = merge_all_features(
        alerts_df,
        include_ts_features=include_ts,
        include_rca_features=include_rca
    )

    # Get feature columns
    feature_cols = df['_feature_cols'].iloc[0] if '_feature_cols' in df.columns else []

    return df, feature_cols
