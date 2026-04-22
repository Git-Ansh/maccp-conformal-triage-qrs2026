"""
Stage 1: Group Disposition
4-class classification at summary level for non-Invalid groups.
Classes: Actionable (Ack+Reassigned+Backedout), Wontfix, Fixed, Downstream.
Confidence-gated: confident -> auto-label, uncertain -> Investigating -> sheriff.

Improvements applied:
  - Change 1: Class hierarchy simplification (Ack+Reas+Backedout -> Actionable)
  - Change 2: OOF threshold tuning (not in-sample)
  - Change 3: Per-class confidence thresholds
  - Change 4: Soft-voting ensemble (XGB + RF)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
from common.data_paths import RANDOM_SEED

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from cascade.evaluation.calibration import (
    calibrate_model, apply_confidence_gate,
    find_per_class_thresholds, get_oof_predictions
)

# Merge map: Backedout(8) and Reassigned(2) -> Actionable(4)
STATUS_MERGE = {8: 4, 2: 4}

# Class labels after merge (4 classes)
DISPOSITION_CLASSES = {
    1: 'Downstream',
    4: 'Actionable',
    6: 'Wontfix',
    7: 'Fixed',
}

# Features for Stage 1
# NOTE: n_invalid_alerts, n_ack_alerts, n_reassigned_alerts, n_downstream_alerts,
# and invalid_alert_ratio were REMOVED -- all derived from single_alert_status
# (leakage: those ARE the sheriff's triage labels).
STAGE_1_FEATURES = [
    'group_size', 'is_single_alert',
    'magnitude_mean', 'magnitude_max', 'magnitude_min', 'magnitude_std',
    'pct_change_mean', 'pct_change_max',
    't_value_mean', 't_value_max', 't_value_min',
    'n_regressions', 'regression_ratio',
    'n_unique_suites', 'n_unique_platforms',
    'n_manually_created', 'manually_created_ratio',
    'noise_ratio',
    'prev_value_mean', 'new_value_mean', 'value_change_ratio',
    'has_subtests_ratio', 'lower_is_better_ratio',
]

STAGE_1_CAT_FEATURES = ['dominant_suite', 'dominant_platform', 'repository', 'dominant_noise']


def prepare_stage_1_data(
    summary_df: pd.DataFrame
) -> Tuple[pd.DataFrame, np.ndarray, Dict, StandardScaler, LabelEncoder]:
    """
    Prepare features and target for Stage 1.
    Filters to non-Invalid resolved summaries, merges Reas+Backedout into Actionable.

    Returns:
        (X, y_encoded, cat_encoders, scaler, class_encoder)
    """
    df = summary_df.copy()

    # Filter out Invalid (status 3) -- those were handled by Stage 0
    df = df[df['alert_summary_status'] != 3].copy()

    # Merge Backedout(8) and Reassigned(2) -> Actionable(4)
    df['disposition'] = df['alert_summary_status'].replace(STATUS_MERGE)

    # Ensure only valid disposition classes remain
    valid_classes = set(DISPOSITION_CLASSES.keys())
    df = df[df['disposition'].isin(valid_classes)].copy()

    # Encode target to 0..n_classes-1
    class_encoder = LabelEncoder()
    y = class_encoder.fit_transform(df['disposition'].values)

    # Encode categoricals
    cat_encoders = {}
    for col in STAGE_1_CAT_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_enc'] = le.fit_transform(df[col].astype(str).fillna('unknown'))
            cat_encoders[col] = le

    # Build feature matrix
    feature_cols = STAGE_1_FEATURES + [c + '_enc' for c in STAGE_1_CAT_FEATURES if c in df.columns]
    X = df[feature_cols].copy().fillna(0)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    return X_scaled, y, cat_encoders, scaler, class_encoder


def _detect_xgb_gpu():
    """Detect XGBoost GPU support."""
    if not HAS_XGBOOST:
        return {}
    try:
        _t = XGBClassifier(tree_method='hist', device='cuda', n_estimators=1)
        _t.fit(np.random.rand(10, 2), np.random.randint(0, 2, 10))
        del _t
        return {'tree_method': 'hist', 'device': 'cuda'}
    except Exception:
        return {}

_XGB_GPU_PARAMS = _detect_xgb_gpu()


def _build_base_model():
    """Build the base model (XGBoost with GPU preferred, RF fallback)."""
    if HAS_XGBOOST:
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_SEED,
            eval_metric='mlogloss',
            n_jobs=-1,
            objective='multi:softprob',
            **_XGB_GPU_PARAMS,
        )
    else:
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=RANDOM_SEED,
            n_jobs=-1
        )


def _build_ensemble():
    """Build soft-voting ensemble of XGB (GPU) + RF using sklearn VotingClassifier."""
    estimators = []

    if HAS_XGBOOST:
        estimators.append(('xgb', XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_SEED,
            eval_metric='mlogloss',
            n_jobs=-1,
            objective='multi:softprob',
            **_XGB_GPU_PARAMS,
        )))

    estimators.append(('rf', RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )))

    if len(estimators) == 1:
        return estimators[0][1]

    return VotingClassifier(estimators=estimators, voting='soft')


def train_stage_1(
    train_summaries: pd.DataFrame,
    calibration_method: str = 'isotonic'
) -> Dict:
    """
    Train Stage 1 disposition classifier with calibrated confidence.
    Uses soft-voting ensemble, OOF threshold tuning, and per-class thresholds.
    """
    X, y, cat_encoders, scaler, class_encoder = prepare_stage_1_data(train_summaries)

    class_counts = pd.Series(y).value_counts()
    class_names = class_encoder.classes_
    print(f"Stage 1 training: {len(y)} summaries, {len(class_names)} classes")
    for i, name in enumerate(class_names):
        count = class_counts.get(i, 0)
        label = DISPOSITION_CLASSES.get(int(name), str(name))
        print(f"  {label}: {count}")

    # Build ensemble model
    base_model = _build_ensemble()

    # Generate OOF predictions for threshold tuning
    oof_base = _build_base_model()
    oof_proba = get_oof_predictions(oof_base, X.values, y,
                                     n_folds=5, random_state=RANDOM_SEED)

    # Find per-class thresholds on OOF predictions
    per_class_thresholds = find_per_class_thresholds(
        y, oof_proba, target_accuracy=0.80, min_samples=5
    )
    print(f"Stage 1 per-class thresholds:")
    for i, name in enumerate(class_names):
        label = DISPOSITION_CLASSES.get(int(name), str(name))
        print(f"  {label}: {per_class_thresholds[i]:.2f}")

    # Also find a global fallback threshold on OOF
    oof_confidence = np.max(oof_proba, axis=1)
    oof_predicted = np.argmax(oof_proba, axis=1)

    best_global = 0.50
    best_score = 0
    for t in np.arange(0.40, 0.90, 0.01):
        mask = oof_confidence >= t
        if mask.sum() < 20:
            continue
        acc = (y[mask] == oof_predicted[mask]).mean()
        cov = mask.mean()
        score = acc * cov
        if acc >= 0.70 and score > best_score:
            best_score = score
            best_global = t

    print(f"Stage 1 global threshold (OOF): {best_global:.2f}")

    # Train final calibrated model on all data
    calibrated_model = calibrate_model(
        base_model, X.values, y,
        method=calibration_method, cv=5
    )

    return {
        'model': calibrated_model,
        'scaler': scaler,
        'cat_encoders': cat_encoders,
        'class_encoder': class_encoder,
        'feature_cols': list(X.columns),
        'threshold': per_class_thresholds,  # per-class thresholds
        'global_threshold': best_global,
    }


def get_cross_validated_predictions(
    train_summaries: pd.DataFrame,
    n_folds: int = 5
) -> pd.DataFrame:
    """
    Get cross-validated Stage 1 predictions for Stage 3 training.
    Train on k-1 folds, predict fold k. This avoids train-inference mismatch.

    Returns:
        DataFrame with alert_summary_id and predicted disposition probabilities
    """
    X, y, cat_encoders, scaler, class_encoder = prepare_stage_1_data(train_summaries)

    # We need the alert_summary_ids that correspond to these rows
    df = train_summaries.copy()
    df = df[df['alert_summary_status'] != 3].copy()
    df['disposition'] = df['alert_summary_status'].replace(STATUS_MERGE)
    valid_classes = set(DISPOSITION_CLASSES.keys())
    df = df[df['disposition'].isin(valid_classes)].copy()
    summary_ids = df['alert_summary_id'].values

    cv_predictions = np.zeros((len(y), len(class_encoder.classes_)))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.values[train_idx], X.values[val_idx]
        y_tr = y[train_idx]

        model = _build_base_model()
        model.fit(X_tr, y_tr)
        cv_predictions[val_idx] = model.predict_proba(X_val)

    result = pd.DataFrame({
        'alert_summary_id': summary_ids,
        'cv_pred_disposition': np.argmax(cv_predictions, axis=1),
        'cv_pred_confidence': np.max(cv_predictions, axis=1),
    })

    # Add per-class probabilities
    for i, cls in enumerate(class_encoder.classes_):
        result[f'cv_proba_class_{int(cls)}'] = cv_predictions[:, i]

    return result


def predict_stage_1(
    stage_1_artifacts: Dict,
    summary_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Run Stage 1 predictions on summary DataFrame.

    Adds columns:
        - s1_pred: predicted disposition code (or -1 for uncertain/Investigating)
        - s1_confidence: calibrated confidence
        - s1_is_confident: whether prediction is confident
    """
    df = summary_df.copy()
    model = stage_1_artifacts['model']
    scaler = stage_1_artifacts['scaler']
    cat_encoders = stage_1_artifacts['cat_encoders']
    class_encoder = stage_1_artifacts['class_encoder']
    threshold = stage_1_artifacts['threshold']

    # Encode categoricals
    # N1: Map unseen categories to "unknown" (not first training class)
    for col, le in cat_encoders.items():
        if col in df.columns:
            vals = df[col].astype(str).fillna('unknown')
            known = set(le.classes_)
            fallback = 'unknown' if 'unknown' in known else le.classes_[0]
            vals = vals.apply(lambda x: x if x in known else fallback)
            df[col + '_enc'] = le.transform(vals)

    feature_cols = stage_1_artifacts['feature_cols']
    X = df[feature_cols].copy().fillna(0)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)

    proba = model.predict_proba(X_scaled.values)
    confidence = np.max(proba, axis=1)
    predicted_idx = np.argmax(proba, axis=1)
    predicted_class = class_encoder.inverse_transform(predicted_idx)

    # Apply per-class or global threshold
    if isinstance(threshold, np.ndarray):
        per_sample_threshold = threshold[predicted_idx]
        is_confident = confidence >= per_sample_threshold
    else:
        is_confident = confidence >= float(threshold)

    df['s1_pred'] = np.where(is_confident, predicted_class, -1).astype(int)
    df['s1_confidence'] = confidence
    df['s1_is_confident'] = is_confident

    # Store probabilities for downstream use
    for i, cls in enumerate(class_encoder.classes_):
        df[f's1_proba_{int(cls)}'] = proba[:, i]

    return df


if __name__ == '__main__':
    from cascade.data.loader import prepare_cascade_data

    data = prepare_cascade_data()

    print("\n--- Training Stage 1 ---")
    artifacts = train_stage_1(data['train_summaries'])

    print("\n--- Evaluating Stage 1 ---")
    # Filter test to non-Invalid
    test_non_invalid = data['test_summaries'][
        data['test_summaries']['alert_summary_status'] != 3
    ].copy()

    test_with_preds = predict_stage_1(artifacts, test_non_invalid)

    # Map true status with merge
    true_status = test_with_preds['alert_summary_status'].replace(STATUS_MERGE).values
    pred_status = test_with_preds['s1_pred'].values
    confident = np.asarray(test_with_preds['s1_is_confident'].values, dtype=bool)

    n_conf = confident.sum()
    n_total = len(test_with_preds)
    print(f"Coverage: {n_conf}/{n_total} ({n_conf/n_total:.1%})")

    if n_conf > 0:
        acc = (true_status[confident] == pred_status[confident]).mean()
        print(f"Accuracy on confident: {acc:.4f}")

    print(f"\nPrediction breakdown:")
    print(pd.Series(pred_status).value_counts().sort_index())
