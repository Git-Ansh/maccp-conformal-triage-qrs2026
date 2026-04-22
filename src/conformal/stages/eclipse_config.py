"""
Eclipse Bug Cascade Configuration (Zenodo 2024 dataset).

3-stage sequential cascade for Eclipse bug triage:
  Stage 0 (Noise Gate): Noise (INVALID/DUPLICATE/WONTFIX/WORKSFORME/NOT_ECLIPSE) vs Valid
  Stage 1 (Severity):   Predict severity level (7 classes) -- on non-noise items
  Stage 2 (Component):  Predict component assignment (top-30 + Other) -- on S1-forwarded items

Sequential routing:
  S0: report P(Noise) flag metrics, filter noise manually for S1/S2
  S1: confident severity -> forward to S2 (with severity as feature)
      uncertain          -> defer to human
  S2: confident component -> terminal (auto-assign)
      uncertain           -> defer to human
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

from cascade.framework.cascade_pipeline import StageConfig

# Severity code mapping (for Stage 1)
SEVERITY_CLASSES = {
    0: 'blocker',
    1: 'critical',
    2: 'major',
    3: 'normal',
    4: 'minor',
    5: 'trivial',
    6: 'enhancement',
}
SEVERITY_NAME_TO_CODE = {v: k for k, v in SEVERITY_CLASSES.items()}


def encode_categoricals(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical columns using LabelEncoder fitted on training data.
    Unseen test categories mapped to 'unknown'.
    """
    encoders = {}
    train_df = train_df.copy()
    test_df = test_df.copy()

    for col in cat_columns:
        if col not in train_df.columns:
            continue

        le = LabelEncoder()
        train_vals = train_df[col].astype(str).fillna('unknown')
        le.fit(train_vals)
        encoders[col] = le

        train_df[col + '_enc'] = le.transform(train_vals)

        # Handle unseen categories in test
        test_vals = test_df[col].astype(str).fillna('unknown')
        known = set(le.classes_)
        fallback = 'unknown' if 'unknown' in known else le.classes_[0]
        test_vals = test_vals.apply(lambda x: x if x in known else fallback)
        test_df[col + '_enc'] = le.transform(test_vals)

    return train_df, test_df, encoders


def prepare_stage_0_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: List[str],
    cat_features: List[str],
) -> Dict:
    """
    Prepare Stage 0 (Noise Gate) data.

    Target: is_noise (0=Valid, 1=Noise)
    Text: Summary + first 500 chars of Description
    """
    # Encode categoricals
    train_df, test_df, encoders = encode_categoricals(train_df, test_df, cat_features)

    # Build feature column list
    feature_cols = numeric_features + [c + '_enc' for c in cat_features if c in train_df.columns]
    feature_cols = [c for c in feature_cols if c in train_df.columns]

    # Combined text: Summary + truncated Description
    def _combine_text(df):
        summary = df['Summary'].fillna('')
        desc = df['Description'].fillna('').str[:500]
        return (summary + ' ' + desc).str.strip()

    return {
        'train_X': train_df[feature_cols].fillna(0).values,
        'train_y': train_df['is_noise'].values,
        'test_X': test_df[feature_cols].fillna(0).values,
        'test_y': test_df['is_noise'].values,
        'feature_cols': feature_cols,
        'encoders': encoders,
        'train_text': _combine_text(train_df).values,
        'test_text': _combine_text(test_df).values,
        'train_idx': train_df.index.values,
        'test_idx': test_df.index.values,
    }


def prepare_stage_1_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: List[str],
    cat_features: List[str],
    train_mask: Optional[np.ndarray] = None,
    test_mask: Optional[np.ndarray] = None,
) -> Dict:
    """
    Prepare Stage 1 (Severity Prediction) data.

    Target: Severity encoded to 0-6
    Features: EXCLUDE severity-related features (severity_numeric, is_enhancement,
              is_high_severity, Severity categorical) since Severity IS the target.

    Args:
        train_mask: Boolean mask selecting which training rows to use.
                    If None, uses all non-noise rows.
        test_mask:  Boolean mask selecting which test rows to use.
                    If None, uses all non-noise rows.
    """
    # Apply masks or default to non-noise
    if train_mask is not None:
        train_real = train_df[np.asarray(train_mask, dtype=bool)].copy()
    else:
        train_real = train_df[train_df['is_noise'] == 0].copy()

    if test_mask is not None:
        test_real = test_df[np.asarray(test_mask, dtype=bool)].copy()
    else:
        test_real = test_df[test_df['is_noise'] == 0].copy()

    # Encode severity target
    train_real['severity_code'] = train_real['Severity'].str.lower().map(
        SEVERITY_NAME_TO_CODE).fillna(3).astype(int)
    test_real['severity_code'] = test_real['Severity'].str.lower().map(
        SEVERITY_NAME_TO_CODE).fillna(3).astype(int)

    # Exclude severity-related features (Severity IS the target)
    severity_leak = {'severity_numeric', 'is_enhancement', 'is_high_severity'}
    s1_numeric = [f for f in numeric_features if f not in severity_leak]
    s1_cat = [c for c in cat_features if c != 'Severity']

    # Encode categoricals
    train_real, test_real, encoders = encode_categoricals(train_real, test_real, s1_cat)

    feature_cols = s1_numeric + [c + '_enc' for c in s1_cat if c in train_real.columns]
    feature_cols = [c for c in feature_cols if c in train_real.columns]

    # Add upstream features if available
    for col in ['s0_confidence', 's0_pred']:
        if col in test_real.columns:
            feature_cols.append(col)
            if col not in train_real.columns:
                train_real[col] = 0.0

    # Combined text
    def _combine_text(df):
        summary = df['Summary'].fillna('')
        desc = df['Description'].fillna('').str[:500]
        return (summary + ' ' + desc).str.strip()

    return {
        'train_X': train_real[feature_cols].fillna(0).values,
        'train_y': train_real['severity_code'].values,
        'test_X': test_real[feature_cols].fillna(0).values,
        'test_y': test_real['severity_code'].values,
        'feature_cols': feature_cols,
        'encoders': encoders,
        'train_text': _combine_text(train_real).values,
        'test_text': _combine_text(test_real).values,
        'train_df': train_real,
        'test_df': test_real,
    }


def prepare_stage_2_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: List[str],
    cat_features: List[str],
    top_components: List[str],
    train_mask: Optional[np.ndarray] = None,
    test_mask: Optional[np.ndarray] = None,
) -> Dict:
    """
    Prepare Stage 2 (Component Assignment) data.

    Target: component_target (top-30 + Other)
    Features: EXCLUDE component_size (derived from Component target).
              CAN include severity since it's not the S2 target.

    Args:
        train_mask: Boolean mask selecting which training rows to use.
                    If None, uses all non-noise rows.
        test_mask:  Boolean mask selecting which test rows to use.
                    If None, uses all non-noise rows.
    """
    # Apply masks or default to non-noise
    if train_mask is not None:
        train_real = train_df[np.asarray(train_mask, dtype=bool)].copy()
    else:
        train_real = train_df[train_df['is_noise'] == 0].copy()

    if test_mask is not None:
        test_real = test_df[np.asarray(test_mask, dtype=bool)].copy()
    else:
        test_real = test_df[test_df['is_noise'] == 0].copy()

    # Encode component target
    comp_le = LabelEncoder()
    comp_le.fit(top_components + ['Other'])

    train_real['component_code'] = comp_le.transform(
        train_real['component_target'].fillna('Other')
    )
    test_comp = test_real['component_target'].fillna('Other')
    known_comps = set(comp_le.classes_)
    test_comp = test_comp.apply(lambda x: x if x in known_comps else 'Other')
    test_real['component_code'] = comp_le.transform(test_comp)

    # Build component class mapping
    component_classes = {i: name for i, name in enumerate(comp_le.classes_)}

    # Exclude component_size from numeric features (derived from target)
    s2_numeric = [f for f in numeric_features if f != 'component_size']

    # All categoricals are fine for S2 (including Severity as feature)
    s2_cat = list(cat_features)

    train_real, test_real, encoders = encode_categoricals(train_real, test_real, s2_cat)

    feature_cols = s2_numeric + [c + '_enc' for c in s2_cat if c in train_real.columns]
    feature_cols = [c for c in feature_cols if c in train_real.columns]

    # Add upstream stage features if available
    for extra in ['s0_confidence', 's1_pred', 's1_confidence']:
        if extra in train_real.columns:
            feature_cols.append(extra)

    # Combined text
    def _combine_text(df):
        summary = df['Summary'].fillna('')
        desc = df['Description'].fillna('').str[:500]
        return (summary + ' ' + desc).str.strip()

    return {
        'train_X': train_real[feature_cols].fillna(0).values,
        'train_y': train_real['component_code'].values,
        'test_X': test_real[feature_cols].fillna(0).values,
        'test_y': test_real['component_code'].values,
        'feature_cols': feature_cols,
        'encoders': encoders,
        'component_le': comp_le,
        'component_classes': component_classes,
        'train_text': _combine_text(train_real).values,
        'test_text': _combine_text(test_real).values,
    }
