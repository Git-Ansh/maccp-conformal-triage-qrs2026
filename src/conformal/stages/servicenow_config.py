"""
ServiceNow ITSM Cascade Configuration.

3-stage cascade for IT incident triage:
  Stage 0 (Priority Gate): Predict priority (1-Critical to 4-Low)
  Stage 1 (Category):      Predict incident category
  Stage 2 (Team Routing):  Predict assignment group

Stage routing:
  S0: confident priority -> forward to S1
      uncertain          -> defer to human
  S1: confident category -> forward to S2
      uncertain          -> defer to human
  S2: confident group    -> terminal (auto-assign)
      uncertain          -> defer to human
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

from cascade.framework.cascade_pipeline import StageConfig

# Priority classes
PRIORITY_CLASSES = {
    0: 'Critical',
    1: 'High',
    2: 'Moderate',
    3: 'Low',
}

PRIORITY_MAP = {
    '1 - Critical': 0,
    '2 - High': 1,
    '3 - Moderate': 2,
    '4 - Low': 3,
}


def encode_categoricals(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    """Encode categorical columns using LabelEncoder fitted on training data."""
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
    Prepare Stage 0 (Priority Prediction) data.

    Target: priority encoded to 0-3
    Features: impact, urgency, contact_type, category, temporal
    """
    train_df, test_df, encoders = encode_categoricals(train_df, test_df, cat_features)

    # Encode priority target
    train_df['priority_target'] = train_df['final_priority'].map(PRIORITY_MAP).fillna(2).astype(int)
    test_df['priority_target'] = test_df['final_priority'].map(PRIORITY_MAP).fillna(2).astype(int)

    feature_cols = numeric_features + [c + '_enc' for c in cat_features if c in train_df.columns]
    feature_cols = [c for c in feature_cols if c in train_df.columns]

    return {
        'train_X': train_df[feature_cols].fillna(0).values,
        'train_y': train_df['priority_target'].values,
        'test_X': test_df[feature_cols].fillna(0).values,
        'test_y': test_df['priority_target'].values,
        'feature_cols': feature_cols,
        'encoders': encoders,
    }


def prepare_stage_1_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: List[str],
    cat_features: List[str],
    top_n_categories: int = 20,
    train_mask: Optional[np.ndarray] = None,
    test_mask: Optional[np.ndarray] = None,
) -> Dict:
    """
    Prepare Stage 1 (Category Prediction) data.

    Target: incident category (top-N + Other)

    Args:
        train_mask: Optional boolean mask to filter training rows
        test_mask: Optional boolean mask to filter test rows
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Apply masks if provided (boolean Series or array)
    if train_mask is not None:
        train_df = train_df[np.asarray(train_mask, dtype=bool)]
    if test_mask is not None:
        test_df = test_df[np.asarray(test_mask, dtype=bool)]

    # Top categories from training data
    top_cats = train_df['category'].value_counts().head(top_n_categories).index.tolist()

    train_df['cat_target'] = train_df['category'].apply(
        lambda x: x if x in top_cats else 'Other')
    test_df['cat_target'] = test_df['category'].apply(
        lambda x: x if x in top_cats else 'Other')

    # Encode target
    cat_le = LabelEncoder()
    cat_le.fit(top_cats + ['Other'])
    train_df['cat_code'] = cat_le.transform(train_df['cat_target'])

    test_cat = test_df['cat_target'].apply(
        lambda x: x if x in set(cat_le.classes_) else 'Other')
    test_df['cat_code'] = cat_le.transform(test_cat)

    # Don't use category or subcategory as feature (subcategory is a child of
    # category, so it deterministically predicts the target)
    # But DO use u_symptom and cmdb_ci which are independent signals
    cat_features_s1 = [c for c in cat_features if c not in ('category', 'subcategory')]

    # Also remove category_size from numeric features (derived from category)
    numeric_features = [f for f in numeric_features if f != 'category_size']

    # Add upstream features if available
    upstream_feats = []
    for col in ['s0_confidence', 's0_pred']:
        if col in test_df.columns:
            upstream_feats.append(col)
            if col not in train_df.columns:
                train_df[col] = 0.0  # placeholder for training

    print(f"  S1 features: {numeric_features + cat_features_s1 + upstream_feats}")

    train_df, test_df, encoders = encode_categoricals(train_df, test_df, cat_features_s1)

    feature_cols = numeric_features + [c + '_enc' for c in cat_features_s1 if c in train_df.columns] + upstream_feats
    feature_cols = [c for c in feature_cols if c in train_df.columns]

    cat_classes = {i: name for i, name in enumerate(cat_le.classes_)}

    return {
        'train_X': train_df[feature_cols].fillna(0).values,
        'train_y': train_df['cat_code'].values,
        'test_X': test_df[feature_cols].fillna(0).values,
        'test_y': test_df['cat_code'].values,
        'feature_cols': feature_cols,
        'encoders': encoders,
        'cat_le': cat_le,
        'cat_classes': cat_classes,
        'top_categories': top_cats,
        'test_df': test_df,
    }


def prepare_stage_2_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: List[str],
    cat_features: List[str],
    top_groups: List[str],
    train_mask: Optional[np.ndarray] = None,
    test_mask: Optional[np.ndarray] = None,
) -> Dict:
    """
    Prepare Stage 2 (Team Routing) data.

    Target: assignment_group (top-N + Other)

    Args:
        train_mask: Optional boolean mask to filter training rows
        test_mask: Optional boolean mask to filter test rows
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Apply masks if provided (boolean Series or array)
    if train_mask is not None:
        train_df = train_df[np.asarray(train_mask, dtype=bool)]
    if test_mask is not None:
        test_df = test_df[np.asarray(test_mask, dtype=bool)]

    # Encode group target
    group_le = LabelEncoder()
    group_le.fit(top_groups + ['Other'])

    train_df['group_code'] = group_le.transform(train_df['group_target'])
    test_group = test_df['group_target'].apply(
        lambda x: x if x in set(group_le.classes_) else 'Other')
    test_df['group_code'] = group_le.transform(test_group)

    group_classes = {i: name for i, name in enumerate(group_le.classes_)}

    # Add upstream features if available
    upstream_feats = []
    for col in ['s0_confidence', 's0_pred', 's1_pred', 's1_confidence']:
        if col in test_df.columns:
            upstream_feats.append(col)
            if col not in train_df.columns:
                train_df[col] = 0.0  # placeholder for training

    train_df, test_df, encoders = encode_categoricals(train_df, test_df, cat_features)

    feature_cols = numeric_features + [c + '_enc' for c in cat_features if c in train_df.columns] + upstream_feats
    feature_cols = [c for c in feature_cols if c in train_df.columns]

    return {
        'train_X': train_df[feature_cols].fillna(0).values,
        'train_y': train_df['group_code'].values,
        'test_X': test_df[feature_cols].fillna(0).values,
        'test_y': test_df['group_code'].values,
        'feature_cols': feature_cols,
        'encoders': encoders,
        'group_le': group_le,
        'group_classes': group_classes,
        'test_df': test_df,
    }
