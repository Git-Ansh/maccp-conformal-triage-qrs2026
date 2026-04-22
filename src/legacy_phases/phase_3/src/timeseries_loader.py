"""
Phase 3: Time-Series Loader Module
Load and manage time-series data from ZIP files.
"""

import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator
from io import StringIO
import sys

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.data_paths import TIMESERIES_DIR, TIMESERIES_ZIPS


def list_signatures_in_zip(zip_path: Path) -> List[int]:
    """
    List all signature IDs in a ZIP file.

    Args:
        zip_path: Path to ZIP file

    Returns:
        List of signature IDs
    """
    signatures = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            # Skip MACOSX metadata and non-CSV files
            if '__MACOSX' in name or not name.endswith('_timeseries_data.csv'):
                continue
            try:
                # Extract signature ID from filename
                filename = name.split('/')[-1]
                sig_id = int(filename.split('_')[0])
                signatures.append(sig_id)
            except (ValueError, IndexError):
                continue
    return signatures


def load_timeseries_from_zip(
    zip_path: Path,
    signature_id: int
) -> Optional[pd.DataFrame]:
    """
    Load time-series data for a specific signature from a ZIP file.

    Args:
        zip_path: Path to ZIP file
        signature_id: Signature ID to load

    Returns:
        DataFrame with time-series data or None if not found
    """
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            # Skip MACOSX metadata
            if '__MACOSX' in name:
                continue
            if name.endswith(f'{signature_id}_timeseries_data.csv'):
                with zf.open(name) as f:
                    content = f.read().decode('utf-8')
                    df = pd.read_csv(StringIO(content))
                    return df
    return None


def find_signature_zip(signature_id: int) -> Optional[Path]:
    """
    Find which ZIP file contains a given signature.

    Args:
        signature_id: Signature ID to find

    Returns:
        Path to ZIP file or None if not found
    """
    for zip_name, zip_path in TIMESERIES_ZIPS.items():
        if zip_path.exists():
            signatures = list_signatures_in_zip(zip_path)
            if signature_id in signatures:
                return zip_path
    return None


def build_signature_index() -> Dict[int, Path]:
    """
    Build an index mapping signature IDs to their ZIP files.

    Returns:
        Dictionary of signature_id -> zip_path
    """
    print("Building signature index...")
    index = {}

    for zip_name, zip_path in TIMESERIES_ZIPS.items():
        if zip_path.exists():
            print(f"  Indexing {zip_name}...")
            signatures = list_signatures_in_zip(zip_path)
            for sig_id in signatures:
                index[sig_id] = zip_path

    print(f"Indexed {len(index)} signatures")
    return index


def load_timeseries_batch(
    signature_ids: List[int],
    signature_index: Dict[int, Path],
    max_signatures: Optional[int] = None
) -> Generator[Tuple[int, pd.DataFrame], None, None]:
    """
    Load time-series data for multiple signatures.

    Args:
        signature_ids: List of signature IDs to load
        signature_index: Index mapping signatures to ZIP files
        max_signatures: Optional limit on number of signatures

    Yields:
        Tuples of (signature_id, DataFrame)
    """
    loaded = 0

    for sig_id in signature_ids:
        if max_signatures and loaded >= max_signatures:
            break

        if sig_id not in signature_index:
            continue

        zip_path = signature_index[sig_id]
        df = load_timeseries_from_zip(zip_path, sig_id)

        if df is not None:
            loaded += 1
            yield sig_id, df


def extract_timeseries_values(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Extract core time-series values from DataFrame.

    Args:
        df: Time-series DataFrame

    Returns:
        Tuple of (values array, timestamps array, metadata DataFrame)
    """
    # Sort by timestamp
    df = df.sort_values('push_timestamp').reset_index(drop=True)

    values = df['value'].values
    timestamps = pd.to_datetime(df['push_timestamp']).values

    # Keep metadata columns
    metadata_cols = ['signature_id', 'push_id', 'revision', 'push_timestamp']
    metadata = df[[c for c in metadata_cols if c in df.columns]].copy()

    return values, timestamps, metadata


def get_alert_indices(df: pd.DataFrame) -> Dict[int, int]:
    """
    Find indices where alerts occurred in the time series.

    Args:
        df: Time-series DataFrame with alert info

    Returns:
        Dictionary of alert_id -> index
    """
    alert_indices = {}

    # Find rows with non-null single_alert_id
    if 'single_alert_id' in df.columns:
        alert_rows = df[df['single_alert_id'].notna()]
        for idx, row in alert_rows.iterrows():
            alert_id = int(row['single_alert_id'])
            alert_indices[alert_id] = idx

    return alert_indices


if __name__ == "__main__":
    # Test loading
    print("Testing time-series loader...")

    # Build index
    index = build_signature_index()
    print(f"Total signatures: {len(index)}")

    # Load sample
    if index:
        sample_sig = list(index.keys())[0]
        df = load_timeseries_from_zip(index[sample_sig], sample_sig)
        if df is not None:
            print(f"\nSample signature {sample_sig}:")
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {list(df.columns)}")

            values, timestamps, metadata = extract_timeseries_values(df)
            print(f"  Values range: {values.min():.2f} - {values.max():.2f}")
