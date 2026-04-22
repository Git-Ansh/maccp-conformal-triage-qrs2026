"""
Stage 3: Bug Linkage Prediction (has_bug)
Summary-level prediction: 0=uncertain, 1=no bug, 2=has bug.
Shortcut rules: Invalid->1 (Reassigned shortcut removed after class merge).
Mode A: confident status available -> use predicted disposition as feature.
Mode B: Investigating groups -> predict without status context (hint for sheriff).

Improvements applied:
  - Change 1: Reassigned shortcut removed (merged into Actionable in Stage 1)
  - Change 2: OOF threshold tuning (not in-sample)
  - Change 3: Per-class confidence thresholds
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

# Shortcut rules: only Invalid has near-zero bug rate
# Reassigned shortcut removed -- merged into Actionable in Stage 1
NO_BUG_STATUSES = [3]  # Invalid only (0.4% bug rate)

# Features for Stage 3
# NOTE: invalid_alert_ratio, n_ack_alerts, n_reassigned_alerts, n_downstream_alerts
# were REMOVED -- all derived from single_alert_status (leakage).
STAGE_3_FEATURES = [
    'group_size', 'is_single_alert',
    'magnitude_mean', 'magnitude_max', 'magnitude_std',
    'pct_change_mean', 'pct_change_max',
    't_value_mean', 't_value_max',
    'n_regressions', 'regression_ratio',
    'n_unique_suites', 'n_unique_platforms',
    'n_manually_created', 'manually_created_ratio',
    'noise_ratio',
    'prev_value_mean', 'new_value_mean', 'value_change_ratio',
]

# Curated TS features for bug prediction (from Phase 3 time-series extraction).
# These capture signal quality and change-point characteristics that correlate
# with real bugs vs noise. Added from cascade_outputs/ts_features_per_summary.csv.
STAGE_3_TS_FEATURES = [
    'ts_cv_mean',                    # Coefficient of variation (noise indicator)
    'ts_direction_change_rate_mean', # How often direction flips (noisy = high)
    'ts_slope_diff_mean',            # Slope change at alert point (sharp = bug)
    'ts_variance_ratio_mean',        # Before/after variance ratio
    'ts_trend_strength_mean',        # Trend strength (strong trend = real change)
    'ts_normalized_change_mean',     # Normalized magnitude of change
    'ts_autocorr_lag1_mean',         # Autocorrelation (persistent change = bug)
    'ts_cusum_max_mean',             # CUSUM statistic (cumulative deviation)
]

STAGE_3_CAT_FEATURES = ['dominant_suite', 'dominant_platform', 'repository']


def prepare_stage_3_data(
    summary_df: pd.DataFrame,
    cv_predictions: Optional[pd.DataFrame] = None,
    mode: str = 'A'
) -> Tuple[pd.DataFrame, np.ndarray, Dict, StandardScaler]:
    """
    Prepare features for Stage 3 bug linkage prediction.

    Mode A: Include predicted disposition as feature (for confident groups).
    Mode B: No disposition feature (for Investigating groups).
    """
    df = summary_df.copy()
    y = df['has_bug'].values

    cat_encoders = {}
    for col in STAGE_3_CAT_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_enc'] = le.fit_transform(df[col].astype(str).fillna('unknown'))
            cat_encoders[col] = le

    feature_cols = STAGE_3_FEATURES.copy()

    # Add curated TS features if available in the DataFrame
    available_ts = [c for c in STAGE_3_TS_FEATURES if c in df.columns]
    feature_cols += available_ts

    feature_cols += [c + '_enc' for c in STAGE_3_CAT_FEATURES if c in df.columns]

    # Mode A: add cross-validated disposition predictions as features
    if mode == 'A' and cv_predictions is not None:
        df = df.merge(cv_predictions, on='alert_summary_id', how='left')
        # Add predicted disposition and confidence as features
        proba_cols = [c for c in cv_predictions.columns if c.startswith('cv_proba_class_')]
        for col in proba_cols:
            if col in df.columns:
                feature_cols.append(col)
        if 'cv_pred_confidence' in df.columns:
            feature_cols.append('cv_pred_confidence')
        if 'cv_pred_disposition' in df.columns:
            feature_cols.append('cv_pred_disposition')

    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].copy().fillna(0)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    return X_scaled, y, cat_encoders, scaler


def train_stage_3(
    train_summaries: pd.DataFrame,
    cv_predictions: Optional[pd.DataFrame] = None,
    mode: str = 'A',
    calibration_method: str = 'isotonic'
) -> Dict:
    """
    Train Stage 3 bug linkage classifier.
    Uses OOF threshold tuning and per-class thresholds.

    Args:
        train_summaries: Training summaries (should exclude shortcuts)
        cv_predictions: Cross-validated Stage 1 predictions (Mode A only)
        mode: 'A' (with status features) or 'B' (without)
        calibration_method: calibration method
    """
    # For training, exclude shortcut statuses (Invalid only after class merge)
    df = train_summaries[
        ~train_summaries['alert_summary_status'].isin(NO_BUG_STATUSES)
    ].copy()

    X, y, cat_encoders, scaler = prepare_stage_3_data(df, cv_predictions, mode)

    n_bug = y.sum()
    n_no_bug = len(y) - n_bug
    raw_ratio = n_no_bug / n_bug if n_bug > 0 else 1
    print(f"Stage 3 (Mode {mode}) training: {n_bug} has_bug vs {n_no_bug} no_bug (ratio={raw_ratio:.1f})")
    n_ts = sum(1 for c in X.columns if c.startswith('ts_'))
    print(f"Stage 3 (Mode {mode}) features: {len(X.columns)} total ({n_ts} TS features)")

    # Detect GPU for XGBoost
    _xgb_gpu = {}
    if HAS_XGBOOST:
        try:
            _t = XGBClassifier(tree_method='hist', device='cuda', n_estimators=1)
            _t.fit(np.random.rand(10,2), np.random.randint(0,2,10))
            _xgb_gpu = {'tree_method': 'hist', 'device': 'cuda'}
            del _t
        except Exception:
            pass

    # Grid search scale_pos_weight for best bug recall
    # Default raw ratio is ~6.7, try multipliers to boost recall
    best_spw = raw_ratio
    best_f1 = 0
    spw_candidates = [raw_ratio * m for m in [0.75, 1.0, 1.25, 1.5, 2.0]]
    print(f"Stage 3 (Mode {mode}) grid searching scale_pos_weight: {[f'{s:.1f}' for s in spw_candidates]}")

    for spw in spw_candidates:
        if HAS_XGBOOST:
            _model = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                scale_pos_weight=spw,
                random_state=RANDOM_SEED, eval_metric='logloss',
                n_jobs=-1, **_xgb_gpu,
            )
        else:
            _model = RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_leaf=5,
                class_weight={0: 1, 1: spw},
                random_state=RANDOM_SEED, n_jobs=-1
            )
        _oof = get_oof_predictions(_model, X.values, y, n_folds=5, random_state=RANDOM_SEED)
        _pred = np.argmax(_oof, axis=1)
        # F1 for has_bug class (class 1)
        tp = ((y == 1) & (_pred == 1)).sum()
        fp = ((y == 0) & (_pred == 1)).sum()
        fn = ((y == 1) & (_pred == 0)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"  spw={spw:.1f}: has_bug precision={prec:.3f}, recall={rec:.3f}, F1={f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_spw = spw

    print(f"Stage 3 (Mode {mode}) best scale_pos_weight: {best_spw:.1f} (F1={best_f1:.3f})")

    if HAS_XGBOOST:
        base_model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=best_spw,
            random_state=RANDOM_SEED, eval_metric='logloss',
            n_jobs=-1, **_xgb_gpu,
        )
    else:
        base_model = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1
        )

    # Generate OOF predictions for threshold tuning with best spw
    oof_model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=best_spw,
        random_state=RANDOM_SEED, eval_metric='logloss',
        n_jobs=-1, **_xgb_gpu,
    ) if HAS_XGBOOST else RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1
    )
    oof_proba = get_oof_predictions(oof_model, X.values, y,
                                     n_folds=5, random_state=RANDOM_SEED)

    # Find per-class thresholds on OOF predictions
    # Lower target_accuracy to allow higher recall on bugs
    per_class_thresholds = find_per_class_thresholds(
        y, oof_proba, target_accuracy=0.70, min_samples=5
    )
    print(f"Stage 3 (Mode {mode}) per-class thresholds: no_bug={per_class_thresholds[0]:.2f}, has_bug={per_class_thresholds[1]:.2f}")

    # Find global threshold tuned for F1 of the has_bug class
    oof_confidence = np.max(oof_proba, axis=1)
    oof_predicted = np.argmax(oof_proba, axis=1)

    best_threshold = 0.50
    best_f1_global = 0
    for t in np.arange(0.40, 0.90, 0.01):
        mask = oof_confidence >= t
        if mask.sum() < 10:
            continue
        # F1 for the has_bug class among confident predictions
        pred_bug = (oof_predicted[mask] == 1)
        true_bug = (y[mask] == 1)
        tp = (pred_bug & true_bug).sum()
        fp = (pred_bug & ~true_bug).sum()
        fn = (~pred_bug & true_bug).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        cov = mask.mean()
        # Weighted: F1 + small coverage bonus
        score = f1 + 0.1 * cov
        if score > best_f1_global:
            best_f1_global = score
            best_threshold = t

    print(f"Stage 3 (Mode {mode}) global threshold (OOF, F1-tuned): {best_threshold:.2f}")

    # Train final calibrated model on all data
    calibrated_model = calibrate_model(
        base_model, X.values, y,
        method=calibration_method, cv=5
    )

    return {
        'model': calibrated_model,
        'scaler': scaler,
        'cat_encoders': cat_encoders,
        'feature_cols': list(X.columns),
        'threshold': per_class_thresholds,
        'global_threshold': best_threshold,
        'mode': mode,
    }


def predict_stage_3(
    stage_3_artifacts: Dict,
    summary_df: pd.DataFrame,
    stage_1_proba: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Run Stage 3 predictions.

    Output encoding: 0=uncertain, 1=no bug, 2=has bug.
    Applies shortcut rules before model prediction.

    Adds columns:
        - s3_pred: 0 (uncertain), 1 (no bug), 2 (has bug)
        - s3_confidence: calibrated confidence
        - s3_is_confident: boolean mask
        - s3_source: 'shortcut_invalid', 'model_A', 'model_B', or 'uncertain'
    """
    df = summary_df.copy()
    model = stage_3_artifacts['model']
    scaler = stage_3_artifacts['scaler']
    cat_encoders = stage_3_artifacts['cat_encoders']
    threshold = stage_3_artifacts['threshold']
    mode = stage_3_artifacts['mode']

    # Initialize outputs
    df['s3_pred'] = 0  # default: uncertain
    df['s3_confidence'] = 0.0
    df['s3_is_confident'] = False
    df['s3_source'] = 'uncertain'

    # Shortcut: Invalid groups -> no bug (0.4% bug rate)
    invalid_mask = df.get('s0_pred', pd.Series(dtype=int)).eq(3) if 's0_pred' in df.columns else pd.Series(False, index=df.index)
    if invalid_mask.any():
        df.loc[invalid_mask, 's3_pred'] = 1
        df.loc[invalid_mask, 's3_confidence'] = 1.0
        df.loc[invalid_mask, 's3_is_confident'] = True
        df.loc[invalid_mask, 's3_source'] = 'shortcut_invalid'

    # Model prediction for remaining (no Reassigned shortcut after class merge)
    remaining_mask = ~invalid_mask
    if remaining_mask.sum() > 0:
        remaining_df = df.loc[remaining_mask].copy()

        # Encode categoricals
        for col, le in cat_encoders.items():
            if col in remaining_df.columns:
                vals = remaining_df[col].astype(str).fillna('unknown')
                known = set(le.classes_)
                fallback = 'unknown' if 'unknown' in known else le.classes_[0]
                vals = vals.apply(lambda x: x if x in known else fallback)
                remaining_df[col + '_enc'] = le.transform(vals)

        # Add Stage 1 probabilities if in Mode A
        if mode == 'A' and stage_1_proba is not None:
            remaining_df = remaining_df.merge(stage_1_proba, on='alert_summary_id', how='left')

        feature_cols = stage_3_artifacts['feature_cols']
        missing = [c for c in feature_cols if c not in remaining_df.columns]
        for c in missing:
            remaining_df[c] = 0

        X = remaining_df[feature_cols].copy().fillna(0)
        X_scaled = scaler.transform(X)

        proba = model.predict_proba(X_scaled)
        confidence = np.max(proba, axis=1)
        predicted = np.argmax(proba, axis=1)

        # Apply per-class or global threshold
        if isinstance(threshold, np.ndarray):
            per_sample_threshold = threshold[predicted]
            is_confident = confidence >= per_sample_threshold
        else:
            is_confident = confidence >= float(threshold)

        # Map: model predicts 0=no_bug, 1=has_bug -> output 1=no_bug, 2=has_bug
        pred_mapped = np.where(predicted == 1, 2, 1)
        pred_with_uncertain = np.where(is_confident, pred_mapped, 0)

        df.loc[remaining_mask, 's3_pred'] = pred_with_uncertain
        df.loc[remaining_mask, 's3_confidence'] = confidence
        df.loc[remaining_mask, 's3_is_confident'] = is_confident
        df.loc[remaining_mask, 's3_source'] = np.where(
            is_confident, f'model_{mode}', 'uncertain'
        )

    return df


if __name__ == '__main__':
    from cascade.data.loader import prepare_cascade_data
    from cascade.stages.stage_1_disposition import get_cross_validated_predictions

    data = prepare_cascade_data()

    # Get CV predictions for Mode A
    print("\n--- Getting cross-validated Stage 1 predictions ---")
    cv_preds = get_cross_validated_predictions(data['train_summaries'])

    print("\n--- Training Stage 3 (Mode A) ---")
    artifacts_a = train_stage_3(data['train_summaries'], cv_preds, mode='A')

    print("\n--- Training Stage 3 (Mode B) ---")
    artifacts_b = train_stage_3(data['train_summaries'], mode='B')

    # Evaluate Mode B on test
    test = data['test_summaries'].copy()
    test_with_preds = predict_stage_3(artifacts_b, test)

    true_bug = test_with_preds['has_bug'].values
    pred_bug = test_with_preds['s3_pred'].values
    confident = np.asarray(test_with_preds['s3_is_confident'].values, dtype=bool)

    n_conf = confident.sum()
    print(f"\nMode B Coverage: {n_conf}/{len(test)} ({n_conf/len(test):.1%})")
    if n_conf > 0:
        pred_mapped = np.where(pred_bug[confident] == 2, 1, 0)
        acc = (true_bug[confident] == pred_mapped).mean()
        print(f"Mode B Accuracy on confident: {acc:.4f}")
