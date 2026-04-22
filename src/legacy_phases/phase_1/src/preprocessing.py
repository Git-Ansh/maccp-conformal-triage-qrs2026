"""
Phase 1: Preprocessing Module
Feature engineering, encoding, and imputation for binary classification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sys

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.data_paths import (
    MAGNITUDE_FEATURES, CONTEXT_FEATURES, WORKFLOW_FEATURES,
    LEAKAGE_COLUMNS, ID_COLUMNS, REGRESSION_TARGET_COL, ALERT_ID_COL
)


def identify_leakage_columns() -> List[str]:
    """
    Return columns that could cause data leakage.

    These are columns that contain information that would only be
    available after the alert has been triaged.

    Returns:
        List of column names to exclude
    """
    # Columns that reflect post-triage decisions
    post_triage = [
        'single_alert_classifier',
        'single_alert_classifier_email',
        'alert_summary_first_triaged',
        'alert_summary_bug_number',
        'alert_summary_bug_updated',
        'alert_summary_bug_due_date',
        'alert_summary_notes',
        'alert_summary_status',  # This is what we want to predict in Phase 2
        'single_alert_status',
        'alert_summary_assignee_email',
        'alert_summary_assignee_username',
        'single_alert_starred',
        'single_alert_related_summary_id',
        'alert_summary_performance_tags'
    ]

    # Backfill columns (post-event actions)
    backfill_cols = [col for col in LEAKAGE_COLUMNS if 'backfill' in col.lower()]

    return list(set(post_triage + backfill_cols + LEAKAGE_COLUMNS))


def select_phase1_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select features for Phase 1 binary classification.

    Excludes:
    - ID columns
    - Leakage columns
    - Target column

    Args:
        df: Full alerts DataFrame

    Returns:
        Tuple of (features DataFrame, list of feature names)
    """
    # Columns to exclude
    exclude_cols = set(
        ID_COLUMNS +
        identify_leakage_columns() +
        [REGRESSION_TARGET_COL, 'push_timestamp', 'alert_summary_creation_timestamp']
    )

    # Add any columns ending with '_id' or '_timestamp'
    for col in df.columns:
        if col.endswith('_id') or col.endswith('_timestamp') or col.endswith('_hash'):
            exclude_cols.add(col)

    # Select features
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"Selected {len(feature_cols)} features (excluded {len(exclude_cols)} columns)")

    return df[feature_cols].copy(), feature_cols


def get_feature_groups() -> Dict[str, List[str]]:
    """
    Return predefined feature groups for ablation studies.

    Returns:
        Dictionary mapping group name to feature list
    """
    return {
        'magnitude': MAGNITUDE_FEATURES,
        'context': CONTEXT_FEATURES,
        'workflow': WORKFLOW_FEATURES,
        'magnitude_context': MAGNITUDE_FEATURES + CONTEXT_FEATURES,
        'all': MAGNITUDE_FEATURES + CONTEXT_FEATURES + WORKFLOW_FEATURES
    }


def impute_numeric_features(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    strategy: str = 'median'
) -> pd.DataFrame:
    """
    Impute missing values in numeric columns.

    Args:
        df: DataFrame with features
        cols: Specific columns to impute. If None, impute all numeric.
        strategy: Imputation strategy ('median', 'mean', 'zero')

    Returns:
        DataFrame with imputed values
    """
    df = df.copy()

    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in cols:
        if col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                if strategy == 'median':
                    fill_value = df[col].median()
                elif strategy == 'mean':
                    fill_value = df[col].mean()
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

                df[col] = df[col].fillna(fill_value)
                print(f"  Imputed {missing} missing values in '{col}' with {strategy}={fill_value:.4f}")

    return df


def impute_categorical_features(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    fill_value: str = 'Unknown'
) -> pd.DataFrame:
    """
    Impute missing values in categorical columns.

    Args:
        df: DataFrame with features
        cols: Specific columns to impute. If None, impute all object/category.
        fill_value: Value to fill missing entries

    Returns:
        DataFrame with imputed values
    """
    df = df.copy()

    if cols is None:
        cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in cols:
        if col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                df[col] = df[col].fillna(fill_value)
                print(f"  Imputed {missing} missing values in '{col}' with '{fill_value}'")

    return df


def encode_low_cardinality(
    df: pd.DataFrame,
    cols: List[str],
    max_categories: int = 20
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    One-hot encode low cardinality categorical columns.

    Args:
        df: DataFrame with features
        cols: Columns to encode
        max_categories: Maximum unique values for one-hot encoding

    Returns:
        Tuple of (encoded DataFrame, dictionary of encoders)
    """
    df = df.copy()
    encoders = {}

    for col in cols:
        if col not in df.columns:
            continue

        n_unique = df[col].nunique()

        if n_unique <= max_categories:
            # One-hot encoding
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
            print(f"  One-hot encoded '{col}' ({n_unique} categories -> {len(dummies.columns)} columns)")
        else:
            # Label encoding for high cardinality
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"  Label encoded '{col}' ({n_unique} categories)")

    return df, encoders


def encode_high_cardinality(
    df: pd.DataFrame,
    cols: List[str],
    method: str = 'frequency'
) -> pd.DataFrame:
    """
    Encode high cardinality categorical columns.

    Args:
        df: DataFrame with features
        cols: Columns to encode
        method: Encoding method ('frequency', 'label')

    Returns:
        Encoded DataFrame
    """
    df = df.copy()

    for col in cols:
        if col not in df.columns:
            continue

        if method == 'frequency':
            # Frequency encoding
            freq = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq)
            print(f"  Frequency encoded '{col}'")
        elif method == 'label':
            # Label encoding
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"  Label encoded '{col}'")

    return df


def group_rare_categories(
    df: pd.DataFrame,
    col: str,
    min_frequency: float = 0.01,
    other_label: str = 'Other'
) -> pd.DataFrame:
    """
    Group rare categories into 'Other'.

    Args:
        df: DataFrame with features
        col: Column to process
        min_frequency: Minimum frequency threshold
        other_label: Label for rare categories

    Returns:
        DataFrame with grouped categories
    """
    df = df.copy()

    if col not in df.columns:
        return df

    freq = df[col].value_counts(normalize=True)
    rare_categories = freq[freq < min_frequency].index.tolist()

    if len(rare_categories) > 0:
        df[col] = df[col].apply(lambda x: other_label if x in rare_categories else x)
        print(f"  Grouped {len(rare_categories)} rare categories in '{col}' as '{other_label}'")

    return df


def create_target_variable(
    df: pd.DataFrame,
    target_col: str = REGRESSION_TARGET_COL
) -> pd.Series:
    """
    Create binary target variable.

    Args:
        df: Alerts DataFrame
        target_col: Name of target column

    Returns:
        Binary target Series (0 or 1)
    """
    y = df[target_col].copy()

    # Convert to numeric if needed
    if y.dtype == 'bool':
        y = y.astype(int)
    elif y.dtype == 'object':
        y = y.map({'True': 1, 'False': 0, True: 1, False: 0})

    y = y.astype(int)

    print(f"\nTarget variable '{target_col}':")
    print(f"  Class 0 (Not Regression): {(y == 0).sum()}")
    print(f"  Class 1 (Regression): {(y == 1).sum()}")

    return y


def preprocess_features(
    df: pd.DataFrame,
    numeric_strategy: str = 'median',
    categorical_fill: str = 'Unknown',
    encode_method: str = 'frequency'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Full preprocessing pipeline for Phase 1 features.

    Args:
        df: Raw features DataFrame
        numeric_strategy: Imputation strategy for numeric columns
        categorical_fill: Fill value for categorical columns
        encode_method: Encoding method for high cardinality

    Returns:
        Tuple of (preprocessed DataFrame, preprocessing info dict)
    """
    print("\n" + "="*50)
    print("Preprocessing Features")
    print("="*50)

    # Store original columns
    original_cols = df.columns.tolist()

    # 0. Drop columns that are entirely NaN or have >90% missing
    print("\n0. Dropping columns with >90% missing values...")
    cols_to_drop = []
    for col in df.columns:
        missing_pct = df[col].isnull().sum() / len(df)
        if missing_pct > 0.9:
            cols_to_drop.append(col)
            print(f"  Dropping '{col}' ({missing_pct*100:.1f}% missing)")

    df = df.drop(columns=cols_to_drop)

    # Identify column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    boolean_cols = df.select_dtypes(include=['bool']).columns.tolist()

    print(f"\nColumn types:")
    print(f"  Numeric: {len(numeric_cols)}")
    print(f"  Categorical: {len(categorical_cols)}")
    print(f"  Boolean: {len(boolean_cols)}")

    # 1. Convert booleans to int first
    print("\n1. Converting boolean features...")
    for col in boolean_cols:
        df[col] = df[col].astype(int)
        print(f"  Converted '{col}' to int")

    # Update numeric cols after boolean conversion
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 2. Impute numeric features
    print("\n2. Imputing numeric features...")
    for col in numeric_cols:
        if df[col].isnull().any():
            if numeric_strategy == 'median':
                fill_val = df[col].median()
            elif numeric_strategy == 'mean':
                fill_val = df[col].mean()
            else:
                fill_val = 0

            # Handle case where median/mean is NaN (all values missing)
            if pd.isna(fill_val):
                fill_val = 0

            missing = df[col].isnull().sum()
            df[col] = df[col].fillna(fill_val)
            print(f"  Imputed {missing} missing in '{col}' with {fill_val:.4f}")

    # 3. Impute categorical features
    print("\n3. Imputing categorical features...")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        if df[col].isnull().any():
            missing = df[col].isnull().sum()
            df[col] = df[col].fillna(categorical_fill)
            print(f"  Imputed {missing} missing in '{col}' with '{categorical_fill}'")

    # 4. Group rare categories
    print("\n4. Grouping rare categories...")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        df = group_rare_categories(df, col, min_frequency=0.01)

    # 5. Encode categorical features
    print("\n5. Encoding categorical features...")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Separate low and high cardinality
    low_card = [col for col in categorical_cols if df[col].nunique() <= 10]
    high_card = [col for col in categorical_cols if df[col].nunique() > 10]

    df, encoders = encode_low_cardinality(df, low_card)
    df = encode_high_cardinality(df, high_card, method=encode_method)

    # 6. Final check for remaining non-numeric
    remaining_object = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if remaining_object:
        print(f"\n6. Force encoding remaining columns: {remaining_object}")
        for col in remaining_object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # 7. Final NaN check - fill any remaining with 0
    if df.isnull().any().any():
        print("\n7. Final NaN cleanup...")
        for col in df.columns:
            if df[col].isnull().any():
                missing = df[col].isnull().sum()
                df[col] = df[col].fillna(0)
                print(f"  Filled {missing} remaining NaN in '{col}' with 0")

    # Preprocessing info
    info = {
        'original_columns': original_cols,
        'final_columns': df.columns.tolist(),
        'n_original': len(original_cols),
        'n_final': len(df.columns),
        'dropped_columns': cols_to_drop,
        'numeric_strategy': numeric_strategy,
        'categorical_fill': categorical_fill,
        'encode_method': encode_method
    }

    print(f"\nPreprocessing complete:")
    print(f"  Original features: {info['n_original']}")
    print(f"  Dropped (>90% missing): {len(cols_to_drop)}")
    print(f"  Final features: {info['n_final']}")

    return df, info


def scale_features(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numeric features using StandardScaler.

    Args:
        df: Preprocessed features DataFrame
        scaler: Existing scaler (for transform only)

    Returns:
        Tuple of (scaled DataFrame, scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df)
    else:
        scaled_values = scaler.transform(df)

    scaled_df = pd.DataFrame(scaled_values, columns=df.columns, index=df.index)

    return scaled_df, scaler


if __name__ == "__main__":
    # Test preprocessing
    from data_loader import load_and_prepare_data

    df, _ = load_and_prepare_data()

    # Select features
    X, feature_cols = select_phase1_features(df)
    print(f"\nSelected features shape: {X.shape}")

    # Preprocess
    X_processed, info = preprocess_features(X)
    print(f"\nProcessed features shape: {X_processed.shape}")

    # Create target
    y = create_target_variable(df)
