"""
Phase 3: Alert-Timeseries Matcher
Match alerts to their corresponding time series and extract windows.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.data_paths import ALERTS_DATA_PATH, SIGNATURE_ID_COL


def load_alerts_with_signatures() -> pd.DataFrame:
    """
    Load alerts data with signature information.

    Returns:
        DataFrame with alerts and signature IDs
    """
    df = pd.read_csv(ALERTS_DATA_PATH)

    # Ensure we have signature_id
    if SIGNATURE_ID_COL not in df.columns:
        # Try to find alternative column
        if 'single_alert_series_signature_id' in df.columns:
            df[SIGNATURE_ID_COL] = df['single_alert_series_signature_id']
        else:
            raise ValueError(f"Cannot find signature ID column in alerts data")

    # Filter to alerts with valid signature IDs
    df = df.dropna(subset=[SIGNATURE_ID_COL])
    df[SIGNATURE_ID_COL] = df[SIGNATURE_ID_COL].astype(int)

    print(f"Loaded {len(df)} alerts with {df[SIGNATURE_ID_COL].nunique()} unique signatures")

    return df


def find_alert_index_in_timeseries(
    ts_df: pd.DataFrame,
    alert_row: pd.Series
) -> Optional[int]:
    """
    Find the index of an alert in its time series.

    Args:
        ts_df: Time-series DataFrame for the signature
        alert_row: Alert data row

    Returns:
        Index in the time series or None if not found
    """
    # Sort by timestamp
    ts_df = ts_df.sort_values('push_timestamp').reset_index(drop=True)

    # Try to match by push_id first
    if 'alert_summary_push_id' in alert_row and 'push_id' in ts_df.columns:
        push_id = alert_row.get('alert_summary_push_id')
        if pd.notna(push_id):
            matches = ts_df[ts_df['push_id'] == push_id].index
            if len(matches) > 0:
                return matches[0]

    # Try to match by revision
    if 'alert_summary_revision' in alert_row and 'revision' in ts_df.columns:
        revision = alert_row.get('alert_summary_revision')
        if pd.notna(revision):
            matches = ts_df[ts_df['revision'] == revision].index
            if len(matches) > 0:
                return matches[0]

    # Try to match by alert_id in the timeseries
    if 'single_alert_id' in ts_df.columns:
        alert_id = alert_row.get('single_alert_id')
        if pd.notna(alert_id):
            matches = ts_df[ts_df['single_alert_id'] == alert_id].index
            if len(matches) > 0:
                return matches[0]

    # Try to match by value and approximate timestamp
    if 'single_alert_new_value' in alert_row and 'value' in ts_df.columns:
        new_value = alert_row.get('single_alert_new_value')
        if pd.notna(new_value):
            # Find closest value match
            value_diffs = np.abs(ts_df['value'] - new_value)
            min_idx = value_diffs.idxmin()
            if value_diffs[min_idx] < 0.01 * np.abs(new_value):  # Within 1%
                return min_idx

    return None


def extract_pre_alert_window(
    ts_df: pd.DataFrame,
    alert_index: int,
    window_size: int = 20
) -> Tuple[Optional[np.ndarray], float, Dict]:
    """
    Extract the window of values before an alert.

    Args:
        ts_df: Time-series DataFrame
        alert_index: Index of the alert in the time series
        window_size: Number of points before alert to include

    Returns:
        Tuple of (pre_alert_values, alert_value, metadata)
    """
    ts_df = ts_df.sort_values('push_timestamp').reset_index(drop=True)
    values = ts_df['value'].values

    if alert_index < 1:
        # Not enough data before alert
        return None, values[alert_index] if len(values) > 0 else np.nan, {'valid': False}

    # Get alert value
    alert_value = values[alert_index]

    # Extract pre-alert window
    start_idx = max(0, alert_index - window_size)
    pre_values = values[start_idx:alert_index]

    metadata = {
        'valid': True,
        'actual_window_size': len(pre_values),
        'requested_window_size': window_size,
        'alert_index': alert_index,
        'total_series_length': len(values)
    }

    return pre_values, alert_value, metadata


def match_alerts_to_timeseries(
    alerts_df: pd.DataFrame,
    signature_index: Dict[int, Path],
    window_size: int = 20,
    min_window_size: int = 5
) -> Tuple[pd.DataFrame, Dict]:
    """
    Match all alerts to their time series and extract windows.

    Args:
        alerts_df: DataFrame with alerts
        signature_index: Index mapping signature_id -> zip_path
        window_size: Window size for feature extraction
        min_window_size: Minimum required window size

    Returns:
        Tuple of (matched alerts DataFrame, statistics)
    """
    from .timeseries_loader import load_timeseries_from_zip

    matched_alerts = []
    stats = {
        'total_alerts': len(alerts_df),
        'signature_found': 0,
        'index_found': 0,
        'valid_window': 0,
        'skipped_no_signature': 0,
        'skipped_no_index': 0,
        'skipped_small_window': 0
    }

    # Group alerts by signature for efficiency
    grouped = alerts_df.groupby(SIGNATURE_ID_COL)

    for sig_id, group in grouped:
        sig_id = int(sig_id)

        if sig_id not in signature_index:
            stats['skipped_no_signature'] += len(group)
            continue

        # Load time series for this signature
        ts_df = load_timeseries_from_zip(signature_index[sig_id], sig_id)
        if ts_df is None:
            stats['skipped_no_signature'] += len(group)
            continue

        stats['signature_found'] += len(group)

        # Sort timeseries by timestamp
        ts_df = ts_df.sort_values('push_timestamp').reset_index(drop=True)

        for _, alert_row in group.iterrows():
            # Find alert index
            alert_idx = find_alert_index_in_timeseries(ts_df, alert_row)

            if alert_idx is None:
                stats['skipped_no_index'] += 1
                continue

            stats['index_found'] += 1

            # Extract window
            pre_values, alert_value, meta = extract_pre_alert_window(
                ts_df, alert_idx, window_size
            )

            if pre_values is None or len(pre_values) < min_window_size:
                stats['skipped_small_window'] += 1
                continue

            stats['valid_window'] += 1

            # Store matched alert info
            matched_alerts.append({
                'alert_row': alert_row.to_dict(),
                'pre_values': pre_values,
                'alert_value': alert_value,
                'metadata': meta,
                'signature_id': sig_id
            })

    print(f"\nMatching Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return matched_alerts, stats


if __name__ == "__main__":
    # Test matching
    from timeseries_loader import build_signature_index

    alerts_df = load_alerts_with_signatures()
    print(f"\nAlerts loaded: {len(alerts_df)}")

    # Build signature index
    sig_index = build_signature_index()

    # Test on small sample
    sample_alerts = alerts_df.head(100)
    matched, stats = match_alerts_to_timeseries(sample_alerts, sig_index)

    print(f"\nMatched {len(matched)} alerts")
