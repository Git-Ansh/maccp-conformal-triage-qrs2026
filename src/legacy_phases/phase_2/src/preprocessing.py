"""
Phase 2: Preprocessing Module
Feature engineering for multi-class status prediction.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.preprocessing import LabelEncoder
import sys

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.data_paths import (
    MAGNITUDE_FEATURES, CONTEXT_FEATURES, WORKFLOW_FEATURES,
    LEAKAGE_COLUMNS, ID_COLUMNS, STATUS_TARGET_COL
)


def select_phase2_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select features for Phase 2 multi-class classification.
    Excludes leakage columns and IDs.

    Args:
        df: Input DataFrame

    Returns:
        Tuple of (feature DataFrame, list of feature column names)
    """
    # Columns to exclude
    exclude_cols = set(LEAKAGE_COLUMNS + ID_COLUMNS)

    # Also exclude target
    exclude_cols.add(STATUS_TARGET_COL)
    exclude_cols.add('single_alert_is_regression')  # Phase 1 target

    # Get all columns except excluded
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Filter to keep only sensible feature types
    valid_features = []
    for col in feature_cols:
        # Skip timestamp columns (potential leakage)
        if 'timestamp' in col.lower() or 'date' in col.lower():
            continue
        # Skip URL columns
        if 'url' in col.lower():
            continue
        # Skip text/notes columns
        if 'notes' in col.lower() or 'title' in col.lower():
            continue
        valid_features.append(col)

    print(f"Selected {len(valid_features)} features for Phase 2")

    return df[valid_features].copy(), valid_features


def encode_target(df: pd.DataFrame) -> Tuple[np.ndarray, LabelEncoder, Dict]:
    """
    Encode multi-class target variable.

    Args:
        df: DataFrame with target column

    Returns:
        Tuple of (encoded target array, label encoder, class mapping)
    """
    le = LabelEncoder()
    y = le.fit_transform(df[STATUS_TARGET_COL])

    class_mapping = dict(zip(le.classes_, range(len(le.classes_))))
    print(f"\nClass mapping:")
    for cls, idx in class_mapping.items():
        count = (y == idx).sum()
        print(f"  {idx}: {cls} ({count} samples)")

    return y, le, class_mapping


def preprocess_features(
    df: pd.DataFrame,
    high_cardinality_threshold: int = 50
) -> Tuple[pd.DataFrame, Dict]:
    """
    Preprocess features for multi-class classification.

    Args:
        df: Feature DataFrame
        high_cardinality_threshold: Threshold for frequency encoding

    Returns:
        Tuple of (processed DataFrame, preprocessing info)
    """
    info = {
        'original_columns': list(df.columns),
        'dropped_columns': [],
        'encoded_columns': [],
        'numeric_columns': [],
        'final_columns': []
    }

    # 0. Drop columns with >90% missing values
    print("\n0. Dropping columns with >90% missing values...")
    cols_to_drop = []
    for col in df.columns:
        missing_pct = df[col].isnull().sum() / len(df)
        if missing_pct > 0.9:
            cols_to_drop.append(col)
            print(f"  Dropping {col}: {missing_pct*100:.1f}% missing")

    df = df.drop(columns=cols_to_drop)
    info['dropped_columns'].extend(cols_to_drop)

    # 1. Identify column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()

    print(f"\n1. Column types identified:")
    print(f"   Numeric: {len(numeric_cols)}")
    print(f"   Categorical: {len(categorical_cols)}")

    info['numeric_columns'] = numeric_cols

    # 2. Handle missing values in numeric columns
    print("\n2. Handling missing values in numeric columns...")
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"   Filled {col} with median: {median_val:.4f}")

    # 3. Encode categorical columns
    print("\n3. Encoding categorical columns...")
    encoded_dfs = []

    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
        n_unique = df[col].nunique()

        if n_unique <= 2:
            # Binary encoding
            df[col] = (df[col] == df[col].mode().iloc[0]).astype(int)
            print(f"   Binary encoded: {col}")
            info['encoded_columns'].append((col, 'binary'))
        elif n_unique <= high_cardinality_threshold:
            # One-hot encoding
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            encoded_dfs.append(dummies)
            df = df.drop(columns=[col])
            print(f"   One-hot encoded: {col} ({n_unique} categories)")
            info['encoded_columns'].append((col, 'onehot', n_unique))
        else:
            # Frequency encoding
            freq_map = df[col].value_counts(normalize=True).to_dict()
            df[col] = df[col].map(freq_map).astype(float)
            print(f"   Frequency encoded: {col} ({n_unique} categories)")
            info['encoded_columns'].append((col, 'frequency', n_unique))

    # Concatenate encoded columns
    if encoded_dfs:
        df = pd.concat([df] + encoded_dfs, axis=1)

    # 4. Handle any remaining inf values
    print("\n4. Handling infinite values...")
    df = df.replace([np.inf, -np.inf], np.nan)

    # 5. Final NaN cleanup
    print("\n5. Final NaN cleanup...")
    if df.isnull().any().any():
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(0)

    info['final_columns'] = list(df.columns)
    print(f"\nFinal feature count: {len(df.columns)}")

    return df, info


def get_feature_groups(feature_names: List[str]) -> Dict[str, List[int]]:
    """
    Get feature group indices for ablation studies.

    Args:
        feature_names: List of feature column names

    Returns:
        Dictionary mapping group names to feature indices
    """
    magnitude_keywords = ['amount_abs', 'amount_pct', 't_value', 'prev_value', 'new_value']
    context_keywords = ['repository', 'framework', 'platform', 'suite', 'lower_is_better']
    workflow_keywords = ['manually_created', 'assignee']

    groups = {
        'magnitude': [],
        'context': [],
        'workflow': []
    }

    for i, name in enumerate(feature_names):
        name_lower = name.lower()
        if any(kw in name_lower for kw in magnitude_keywords):
            groups['magnitude'].append(i)
        elif any(kw in name_lower for kw in context_keywords):
            groups['context'].append(i)
        elif any(kw in name_lower for kw in workflow_keywords):
            groups['workflow'].append(i)

    # Remove empty groups
    groups = {k: v for k, v in groups.items() if v}

    print("\nFeature groups for ablation:")
    for name, indices in groups.items():
        print(f"  {name}: {len(indices)} features")

    return groups


if __name__ == "__main__":
    from data_loader import load_and_prepare_data

    df, summary = load_and_prepare_data()

    X_raw, raw_cols = select_phase2_features(df)
    print(f"\nRaw features: {len(raw_cols)}")

    X_processed, info = preprocess_features(X_raw)
    print(f"\nProcessed features: {len(X_processed.columns)}")

    y, le, class_map = encode_target(df)
    print(f"\nEncoded {len(class_map)} classes")
