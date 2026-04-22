"""
Stage 2a: Individual Alert Roles (for confident groups from Stage 1)
4-class classification: Acknowledged(4), Downstream(1), Reassigned(2), Invalid(3).
Confidence-gated: confident → auto-label, uncertain → Investigating.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
from common.data_paths import RANDOM_SEED

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from cascade.evaluation.calibration import calibrate_model, apply_confidence_gate

ALERT_ROLE_CLASSES = {
    1: 'Downstream',
    2: 'Reassigned',
    3: 'Invalid',
    4: 'Acknowledged',
}

# Alert-level features
ALERT_FEATURES = [
    'single_alert_amount_abs',
    'single_alert_amount_pct',
    'single_alert_t_value',
    'single_alert_prev_value',
    'single_alert_new_value',
    'single_alert_is_regression',
    'single_alert_manually_created',
]

ALERT_CAT_FEATURES = [
    'single_alert_series_signature_suite',
    'single_alert_series_signature_machine_platform',
    'single_alert_noise_profile',
    'alert_summary_repository',
]

# Group context features to merge in
GROUP_CONTEXT = [
    'group_size', 'n_unique_suites', 'n_unique_platforms',
    'magnitude_mean', 'magnitude_max',
]


def prepare_stage_2a_data(
    alerts_df: pd.DataFrame,
    summary_df: pd.DataFrame
) -> Tuple[pd.DataFrame, np.ndarray, Dict, StandardScaler, LabelEncoder]:
    """
    Prepare alert-level features for Stage 2a.
    Only includes alerts from resolved, non-Invalid summaries.
    """
    # Filter to non-Invalid resolved summaries
    valid_summaries = summary_df[
        (summary_df['alert_summary_status'] != 3) &
        (~summary_df['alert_summary_status'].isin([0, 5]))  # exclude untriaged/investigating
    ]['alert_summary_id'].values

    df = alerts_df[alerts_df['alert_summary_id'].isin(valid_summaries)].copy()

    # Target: single_alert_status (0-4), filter to status 1-4
    df = df[df['single_alert_status'].isin([1, 2, 3, 4])].copy()

    class_encoder = LabelEncoder()
    y = class_encoder.fit_transform(df['single_alert_status'].values)

    # Merge group context features
    group_cols = ['alert_summary_id'] + GROUP_CONTEXT
    available_group_cols = [c for c in group_cols if c in summary_df.columns]
    if len(available_group_cols) > 1:
        df = df.merge(
            summary_df[available_group_cols],
            on='alert_summary_id', how='left'
        )

    # NOTE: has_related_summary was REMOVED — single_alert_related_summary_id
    # is a LEAKAGE column (it encodes post-triage downstream linkage).

    # Encode categoricals
    cat_encoders = {}
    for col in ALERT_CAT_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_enc'] = le.fit_transform(df[col].astype(str).fillna('unknown'))
            cat_encoders[col] = le

    all_numeric = ALERT_FEATURES + \
                  [c for c in GROUP_CONTEXT if c in df.columns]
    all_encoded = [c + '_enc' for c in ALERT_CAT_FEATURES if c in df.columns]
    feature_cols = all_numeric + all_encoded
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy().fillna(0)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    return X_scaled, y, cat_encoders, scaler, class_encoder


def train_stage_2a(
    alerts_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    calibration_method: str = 'isotonic'
) -> Dict:
    """
    Train Stage 2a alert role classifier.
    """
    X, y, cat_encoders, scaler, class_encoder = prepare_stage_2a_data(alerts_df, summary_df)

    print(f"Stage 2a training: {len(y)} alerts, {len(class_encoder.classes_)} classes")
    for i, cls in enumerate(class_encoder.classes_):
        count = (y == i).sum()
        label = ALERT_ROLE_CLASSES.get(int(cls), str(cls))
        print(f"  {label}: {count}")

    if HAS_XGBOOST:
        base_model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_SEED, eval_metric='mlogloss',
            use_label_encoder=False, n_jobs=-1,
            objective='multi:softprob',
        )
    else:
        base_model = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1
        )

    calibrated_model = calibrate_model(
        base_model, X.values, y,
        method=calibration_method, cv=5
    )

    # Threshold: target ~75% accuracy on confident predictions
    proba = calibrated_model.predict_proba(X.values)
    confidence = np.max(proba, axis=1)
    predicted = np.argmax(proba, axis=1)

    best_threshold = 0.50
    best_score = 0
    for t in np.arange(0.40, 0.90, 0.01):
        mask = confidence >= t
        if mask.sum() < 20:
            continue
        acc = (y[mask] == predicted[mask]).mean()
        cov = mask.mean()
        score = acc * cov
        if acc >= 0.70 and score > best_score:
            best_score = score
            best_threshold = t

    print(f"Stage 2a threshold: {best_threshold:.2f}")

    return {
        'model': calibrated_model,
        'scaler': scaler,
        'cat_encoders': cat_encoders,
        'class_encoder': class_encoder,
        'feature_cols': list(X.columns),
        'threshold': best_threshold,
    }


def predict_stage_2a(
    stage_2a_artifacts: Dict,
    alerts_df: pd.DataFrame,
    summary_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Run Stage 2a predictions on alerts.

    Adds columns:
        - s2a_pred: predicted alert status (or -1 for uncertain)
        - s2a_confidence: calibrated confidence
        - s2a_is_confident: boolean mask
    """
    df = alerts_df.copy()
    model = stage_2a_artifacts['model']
    scaler = stage_2a_artifacts['scaler']
    cat_encoders = stage_2a_artifacts['cat_encoders']
    class_encoder = stage_2a_artifacts['class_encoder']
    threshold = stage_2a_artifacts['threshold']

    # Merge group context if summary_df provided
    if summary_df is not None:
        group_cols = ['alert_summary_id'] + [c for c in GROUP_CONTEXT if c in summary_df.columns]
        available = [c for c in group_cols if c in summary_df.columns]
        if len(available) > 1:
            # Drop existing group columns to avoid conflicts
            existing = [c for c in GROUP_CONTEXT if c in df.columns]
            if existing:
                df = df.drop(columns=existing)
            df = df.merge(summary_df[available], on='alert_summary_id', how='left')

    for col, le in cat_encoders.items():
        if col in df.columns:
            vals = df[col].astype(str).fillna('unknown')
            known = set(le.classes_)
            vals = vals.apply(lambda x: x if x in known else le.classes_[0])
            df[col + '_enc'] = le.transform(vals)

    feature_cols = stage_2a_artifacts['feature_cols']
    missing_cols = [c for c in feature_cols if c not in df.columns]
    for c in missing_cols:
        df[c] = 0
    X = df[feature_cols].copy().fillna(0)
    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled)
    confidence = np.max(proba, axis=1)
    predicted_idx = np.argmax(proba, axis=1)
    predicted_class = class_encoder.inverse_transform(predicted_idx)

    is_confident = confidence >= threshold

    df['s2a_pred'] = np.where(is_confident, predicted_class, -1).astype(int)
    df['s2a_confidence'] = confidence
    df['s2a_is_confident'] = is_confident

    return df
