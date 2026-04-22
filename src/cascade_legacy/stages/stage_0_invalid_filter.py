"""
Stage 0: Group Invalid Filter
Binary classification at summary level -- Invalid (3) vs Valid (all other resolved).
Confidence-gated: confident Invalid -> auto-label, confident Valid -> Stage 1, uncertain -> Investigating.

Improvements applied:
  - Change 2: OOF threshold tuning (not in-sample)
  - Change 3: Per-class confidence thresholds
  - Change 6: Cost-sensitive reweighting for higher Invalid recall
  - TS features: Phase 3 time-series features (ts_cv, ts_variance_ratio, etc.)
  - Suite heuristic: per-suite Invalid rate learned from training data
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold

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


# Features used for Stage 0
# NOTE: invalid_alert_ratio was REMOVED -- it was derived from single_alert_status
# (leakage: that IS the sheriff's label).
STAGE_0_FEATURES = [
    'group_size', 'is_single_alert',
    'magnitude_mean', 'magnitude_max', 'magnitude_min', 'magnitude_std',
    'pct_change_mean', 'pct_change_max',
    't_value_mean', 't_value_max', 't_value_min',
    'n_regressions', 'regression_ratio',
    'n_unique_suites', 'n_unique_platforms',
    'n_manually_created', 'manually_created_ratio',
    'noise_ratio',
    'prev_value_mean', 'new_value_mean', 'value_change_ratio',
]

# Phase 3 TS features: full set was disabled after ablation (curse of
# dimensionality with 30+ features for 387 Invalid samples).
# Now using a CURATED subset of 3 features that capture noise/stability
# patterns most predictive for Invalid detection.
STAGE_0_TS_FEATURES = [
    'ts_cv_mean',                    # Coefficient of variation - high = noisy = Invalid-like
    'ts_variance_ratio_mean',        # Before/after variance ratio - no real change = Invalid
    'ts_direction_change_rate_mean', # Direction instability - erratic = noise
]

# Suite-level heuristic: per-suite Invalid rate learned from training data.
# This is NOT leakage -- it's computed from training labels only and
# applied as a prior at prediction time.
# Ablation: adds +4 Invalid caught (42 vs 38), precision 97.6% vs 100%.
SUITE_INVALID_RATE_COL = 'suite_invalid_rate'

# Categorical features to encode
STAGE_0_CAT_FEATURES = ['dominant_suite', 'dominant_platform', 'repository', 'dominant_noise']


def compute_suite_invalid_rates(
    train_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Compute per-suite Invalid rate from training data.
    Returns dict: suite_name -> fraction of summaries with that suite that are Invalid.
    NOT leakage: uses only training labels, applied as a prior at prediction time.
    """
    if 'dominant_suite' not in train_df.columns:
        return {}
    is_invalid = (train_df['alert_summary_status'] == 3).astype(int)
    suite_rates = train_df.groupby('dominant_suite').apply(
        lambda g: (g['alert_summary_status'] == 3).mean()
    ).to_dict()
    return suite_rates


def prepare_stage_0_data(
    summary_df: pd.DataFrame,
    suite_invalid_rates: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, Dict, StandardScaler]:
    """
    Prepare features and target for Stage 0.

    Target: 1 = Invalid (status 3), 0 = Valid (all other resolved)

    Args:
        summary_df: Summary-level DataFrame (resolved only)
        suite_invalid_rates: Per-suite Invalid rates from training data (or None)

    Returns:
        (X, y, label_encoders_dict, scaler)
    """
    df = summary_df.copy()

    # Binary target: Invalid vs Valid
    df['is_invalid'] = (df['alert_summary_status'] == 3).astype(int)

    # Encode categoricals
    label_encoders = {}
    for col in STAGE_0_CAT_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_enc'] = le.fit_transform(df[col].astype(str).fillna('unknown'))
            label_encoders[col] = le

    # Suite Invalid rate feature (learned from training data)
    if suite_invalid_rates and 'dominant_suite' in df.columns:
        global_rate = df['is_invalid'].mean()  # fallback for unseen suites
        df[SUITE_INVALID_RATE_COL] = df['dominant_suite'].map(suite_invalid_rates).fillna(global_rate)

    # Build feature matrix
    feature_cols = STAGE_0_FEATURES + [c + '_enc' for c in STAGE_0_CAT_FEATURES if c in df.columns]

    # Add TS features if available
    available_ts = [c for c in STAGE_0_TS_FEATURES if c in df.columns]
    feature_cols += available_ts

    # Add suite Invalid rate if computed
    if SUITE_INVALID_RATE_COL in df.columns:
        feature_cols.append(SUITE_INVALID_RATE_COL)

    X = df[feature_cols].copy()
    X = X.fillna(0)

    y = df['is_invalid'].values

    # Scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    return X_scaled, y, label_encoders, scaler


def train_stage_0(
    train_summaries: pd.DataFrame,
    calibration_method: str = 'isotonic'
) -> Dict:
    """
    Train Stage 0 invalid filter with calibrated confidence.
    Uses cost-sensitive reweighting for higher Invalid recall and OOF threshold tuning.
    Includes TS features and suite-level Invalid rate heuristic.

    Args:
        train_summaries: Training summary DataFrame
        calibration_method: 'isotonic' or 'sigmoid' (Platt scaling)

    Returns:
        Dict with model, scaler, encoders, feature_cols, threshold, suite_invalid_rates
    """
    # Compute suite Invalid rates from training data (NOT leakage)
    suite_invalid_rates = compute_suite_invalid_rates(train_summaries)
    if suite_invalid_rates:
        top_invalid_suites = sorted(suite_invalid_rates.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"Suite Invalid rates (top 5): {[(s, f'{r:.2f}') for s, r in top_invalid_suites]}")

    X, y, encoders, scaler = prepare_stage_0_data(train_summaries, suite_invalid_rates)

    n_invalid = y.sum()
    n_valid = len(y) - n_invalid
    print(f"Stage 0 training: {n_invalid} Invalid vs {n_valid} Valid")

    # Cost-sensitive reweighting: increase Invalid class weight
    # Raised from 1.5x to 2.5x to boost Invalid recall (target ~55%)
    invalid_weight = (n_valid / n_invalid) * 2.5 if n_invalid > 0 else 1

    # Detect GPU for XGBoost
    _xgb_gpu = {}
    if HAS_XGBOOST:
        try:
            _t = XGBClassifier(tree_method='hist', device='cuda', n_estimators=1)
            _t.fit(__import__('numpy').random.rand(10,2), __import__('numpy').random.randint(0,2,10))
            _xgb_gpu = {'tree_method': 'hist', 'device': 'cuda'}
            del _t
        except Exception:
            pass

    if HAS_XGBOOST:
        base_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=invalid_weight,
            random_state=RANDOM_SEED,
            eval_metric='logloss',
            n_jobs=-1,
            **_xgb_gpu,
        )
    else:
        base_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight={0: 1, 1: invalid_weight},
            random_state=RANDOM_SEED,
            n_jobs=-1
        )

    # Generate OOF predictions for threshold tuning
    oof_model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=invalid_weight,
        random_state=RANDOM_SEED, eval_metric='logloss',
        n_jobs=-1, **_xgb_gpu,
    ) if HAS_XGBOOST else RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        class_weight={0: 1, 1: invalid_weight},
        random_state=RANDOM_SEED, n_jobs=-1
    )
    oof_proba = get_oof_predictions(oof_model, X.values, y,
                                     n_folds=5, random_state=RANDOM_SEED)

    # Find per-class thresholds on OOF predictions
    # Lowered from 0.85 to 0.80 to allow more Invalid predictions through
    per_class_thresholds = find_per_class_thresholds(
        y, oof_proba, target_accuracy=0.80, min_samples=5
    )
    print(f"Stage 0 per-class thresholds: Valid={per_class_thresholds[0]:.2f}, Invalid={per_class_thresholds[1]:.2f}")

    # Also find optimal global threshold on OOF for Invalid precision/recall
    oof_confidence = np.max(oof_proba, axis=1)
    oof_predicted = np.argmax(oof_proba, axis=1)

    best_threshold = 0.70
    best_f1 = 0
    for t in np.arange(0.50, 0.96, 0.01):
        mask = oof_confidence >= t
        if mask.sum() < 10:
            continue
        pred_invalid = mask & (oof_predicted == 1)
        if pred_invalid.sum() == 0:
            continue
        prec = y[pred_invalid].mean()
        rec = pred_invalid.sum() / y.sum() if y.sum() > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if prec >= 0.80 and f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print(f"Stage 0 global threshold (OOF): {best_threshold:.2f}")

    # Show OOF performance summary
    oof_mask = oof_confidence >= best_threshold
    oof_invalid_pred = oof_mask & (oof_predicted == 1)
    if oof_invalid_pred.sum() > 0:
        oof_prec = y[oof_invalid_pred].mean()
        oof_rec = oof_invalid_pred.sum() / y.sum() if y.sum() > 0 else 0
        print(f"Stage 0 OOF Invalid: precision={oof_prec:.3f}, recall={oof_rec:.3f}")

    # Train final calibrated model on all data
    calibrated_model = calibrate_model(
        base_model, X.values, y,
        method=calibration_method, cv=5
    )

    return {
        'model': calibrated_model,
        'scaler': scaler,
        'encoders': encoders,
        'feature_cols': list(X.columns),
        'threshold': per_class_thresholds,
        'global_threshold': best_threshold,
        'suite_invalid_rates': suite_invalid_rates,
    }


def predict_stage_0(
    stage_0_artifacts: Dict,
    summary_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Run Stage 0 predictions on summary DataFrame.

    Adds columns:
        - s0_pred: 0=valid, 3=invalid, -1=uncertain (investigating)
        - s0_confidence: calibrated confidence
        - s0_is_confident: whether prediction is confident

    Args:
        stage_0_artifacts: Output of train_stage_0
        summary_df: Summary DataFrame to predict on

    Returns:
        summary_df with Stage 0 predictions added
    """
    df = summary_df.copy()
    model = stage_0_artifacts['model']
    scaler = stage_0_artifacts['scaler']
    encoders = stage_0_artifacts['encoders']
    threshold = stage_0_artifacts['threshold']
    suite_invalid_rates = stage_0_artifacts.get('suite_invalid_rates', {})

    # Encode categoricals using fitted encoders
    # N1: Map unseen categories to "unknown" (not first training class)
    for col, le in encoders.items():
        if col in df.columns:
            vals = df[col].astype(str).fillna('unknown')
            known = set(le.classes_)
            # Map unseen to 'unknown' if it was in training, else first class
            fallback = 'unknown' if 'unknown' in known else le.classes_[0]
            vals = vals.apply(lambda x: x if x in known else fallback)
            df[col + '_enc'] = le.transform(vals)

    # Apply suite Invalid rate from training data
    if suite_invalid_rates and 'dominant_suite' in df.columns:
        # Use global average from training as fallback for unseen suites
        global_rate = np.mean(list(suite_invalid_rates.values()))
        df[SUITE_INVALID_RATE_COL] = df['dominant_suite'].map(suite_invalid_rates).fillna(global_rate)

    # Build feature matrix
    feature_cols = stage_0_artifacts['feature_cols']
    missing_cols = [c for c in feature_cols if c not in df.columns]
    for c in missing_cols:
        df[c] = 0
    X = df[feature_cols].copy().fillna(0)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)

    # Predict
    proba = model.predict_proba(X_scaled.values)
    predicted_idx = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)

    # Apply per-class or global threshold
    if isinstance(threshold, np.ndarray):
        per_sample_threshold = threshold[predicted_idx]
        is_confident = confidence >= per_sample_threshold
    else:
        is_confident = confidence >= float(threshold)

    df['s0_proba_invalid'] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    df['s0_confidence'] = confidence
    df['s0_is_confident'] = is_confident
    df['s0_pred'] = np.where(
        is_confident,
        np.where(predicted_idx == 1, 3, 0),  # 3=Invalid, 0=pass to Stage 1
        -1  # Uncertain -> Investigating
    )

    return df


if __name__ == '__main__':
    from cascade.data.loader import prepare_cascade_data

    data = prepare_cascade_data()

    print("\n--- Training Stage 0 ---")
    artifacts = train_stage_0(data['train_summaries'])

    print("\n--- Evaluating Stage 0 ---")
    test_with_preds = predict_stage_0(artifacts, data['test_summaries'])

    # Evaluate
    true_invalid = (test_with_preds['alert_summary_status'] == 3).astype(int)
    pred_invalid = (test_with_preds['s0_pred'] == 3).astype(int)
    confident = np.asarray(test_with_preds['s0_is_confident'].values, dtype=bool)

    n_conf = confident.sum()
    n_total = len(test_with_preds)
    print(f"Coverage: {n_conf}/{n_total} ({n_conf/n_total:.1%})")

    if n_conf > 0:
        from sklearn.metrics import classification_report
        conf_mask = confident
        print(f"Accuracy on confident: {(true_invalid[conf_mask] == pred_invalid[conf_mask]).mean():.4f}")

    # Invalid-specific metrics
    n_true_invalid = true_invalid.sum()
    n_pred_invalid = pred_invalid.sum()
    if n_pred_invalid > 0:
        prec = true_invalid[pred_invalid == 1].mean()
        rec = pred_invalid[true_invalid == 1].mean() if n_true_invalid > 0 else 0
        print(f"Invalid: precision={prec:.3f}, recall={rec:.3f} ({n_pred_invalid} predicted, {n_true_invalid} true)")

    print(f"\nPrediction breakdown:")
    print(test_with_preds['s0_pred'].value_counts())
