"""
Stage 2b: Simplified Noise Filter (for Investigating/uncertain groups)
Binary classification at alert level: Invalid (confident noise) vs Investigating (everything else).
Only flags obvious noise; all other alerts inherit "Investigating" status.
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

NOISE_FEATURES = [
    'single_alert_amount_abs',
    'single_alert_amount_pct',
    'single_alert_t_value',
    'single_alert_prev_value',
    'single_alert_new_value',
    'single_alert_is_regression',
    'single_alert_manually_created',
]

NOISE_CAT_FEATURES = [
    'single_alert_series_signature_suite',
    'single_alert_noise_profile',
]


def prepare_stage_2b_data(
    alerts_df: pd.DataFrame
) -> Tuple[pd.DataFrame, np.ndarray, Dict, StandardScaler]:
    """
    Prepare features for binary noise filter.
    Target: 1 = Invalid (noise), 0 = Not Invalid.
    Uses all resolved alerts (status 1-4).
    """
    df = alerts_df.copy()
    df = df[df['single_alert_status'].isin([1, 2, 3, 4])].copy()

    y = (df['single_alert_status'] == 3).astype(int).values

    cat_encoders = {}
    for col in NOISE_CAT_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_enc'] = le.fit_transform(df[col].astype(str).fillna('unknown'))
            cat_encoders[col] = le

    feature_cols = NOISE_FEATURES + [c + '_enc' for c in NOISE_CAT_FEATURES if c in df.columns]
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].copy().fillna(0)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    return X_scaled, y, cat_encoders, scaler


def train_stage_2b(
    alerts_df: pd.DataFrame,
    calibration_method: str = 'isotonic'
) -> Dict:
    """
    Train Stage 2b noise filter.
    Conservative: only flag noise when very confident (high precision).
    """
    X, y, cat_encoders, scaler = prepare_stage_2b_data(alerts_df)

    n_invalid = y.sum()
    n_valid = len(y) - n_invalid
    print(f"Stage 2b training: {n_invalid} Invalid vs {n_valid} non-Invalid alerts")

    if HAS_XGBOOST:
        base_model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=n_valid / n_invalid if n_invalid > 0 else 1,
            random_state=RANDOM_SEED, eval_metric='logloss',
            use_label_encoder=False, n_jobs=-1
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

    # High threshold for noise flagging (conservative — only flag when sure)
    proba = calibrated_model.predict_proba(X.values)
    confidence = np.max(proba, axis=1)
    predicted = np.argmax(proba, axis=1)

    best_threshold = 0.80
    for t in np.arange(0.70, 0.96, 0.01):
        mask = (confidence >= t) & (predicted == 1)
        if mask.sum() == 0:
            continue
        prec = y[mask].mean()
        if prec >= 0.85:
            best_threshold = t
            break

    print(f"Stage 2b threshold: {best_threshold:.2f}")

    return {
        'model': calibrated_model,
        'scaler': scaler,
        'cat_encoders': cat_encoders,
        'feature_cols': list(X.columns),
        'threshold': best_threshold,
    }


def predict_stage_2b(
    stage_2b_artifacts: Dict,
    alerts_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Run Stage 2b: flag confident noise as Invalid, else mark as Investigating.

    Adds columns:
        - s2b_pred: 3 (Invalid/noise) or -1 (Investigating/uncertain)
        - s2b_confidence: calibrated confidence
        - s2b_is_noise: True if flagged as confident noise
    """
    df = alerts_df.copy()
    model = stage_2b_artifacts['model']
    scaler = stage_2b_artifacts['scaler']
    cat_encoders = stage_2b_artifacts['cat_encoders']
    threshold = stage_2b_artifacts['threshold']

    for col, le in cat_encoders.items():
        if col in df.columns:
            vals = df[col].astype(str).fillna('unknown')
            known = set(le.classes_)
            vals = vals.apply(lambda x: x if x in known else le.classes_[0])
            df[col + '_enc'] = le.transform(vals)

    feature_cols = stage_2b_artifacts['feature_cols']
    missing = [c for c in feature_cols if c not in df.columns]
    for c in missing:
        df[c] = 0
    X = df[feature_cols].copy().fillna(0)
    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled)
    confidence = np.max(proba, axis=1)
    predicted = np.argmax(proba, axis=1)

    # Only flag as noise if predicted Invalid AND confident
    is_noise = (predicted == 1) & (confidence >= threshold)

    df['s2b_pred'] = np.where(is_noise, 3, -1)  # 3=Invalid, -1=Investigating
    df['s2b_confidence'] = confidence
    df['s2b_is_noise'] = is_noise

    return df
