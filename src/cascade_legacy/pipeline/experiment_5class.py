"""
Experiment: 5-class Stage 1 vs 4-class at high confidence thresholds.

Tests whether we can provide granular labels (Ack, Reassigned, Wontfix, Fixed, Downstream)
at t=0.90 while maintaining ~90% accuracy and ~60% coverage.

Key insight: at high thresholds, the model defers ambiguous Ack/Reassigned cases,
so the ones it keeps should be accurate.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
from common.data_paths import RANDOM_SEED

from cascade.data.loader import prepare_cascade_data
from cascade.stages.stage_0_invalid_filter import train_stage_0, predict_stage_0
from cascade.stages.stage_1_disposition import (
    prepare_stage_1_data, _build_ensemble, _build_base_model,
    STAGE_1_FEATURES, STAGE_1_CAT_FEATURES,
    train_stage_1, predict_stage_1,
    STATUS_MERGE, DISPOSITION_CLASSES
)
from cascade.evaluation.calibration import (
    calibrate_model, find_per_class_thresholds, get_oof_predictions
)

# 5-class config: only merge Backedout(8) into Ack(4), keep Reassigned(2) separate
STATUS_MERGE_5CLASS = {8: 4}  # Backedout -> Ack (only 15 samples)
DISPOSITION_5CLASS = {
    1: 'Downstream',
    2: 'Reassigned',
    4: 'Acknowledged',
    6: 'Wontfix',
    7: 'Fixed',
}


def run_experiment():
    data = prepare_cascade_data()
    train = data['train_summaries']
    test = data['test_summaries']

    # Train Stage 0 (shared)
    print("=" * 70)
    print("Training Stage 0 (shared)")
    print("=" * 70)
    s0 = train_stage_0(train)
    test_s0 = predict_stage_0(s0, test)
    valid_mask = test_s0['s0_pred'] == 0
    test_valid = test_s0[valid_mask].copy()

    print(f"\nStage 0: {valid_mask.sum()} valid, {(test_s0['s0_pred'] == 3).sum()} invalid, "
          f"{(test_s0['s0_pred'] == -1).sum()} uncertain")

    # ========================================================
    # Experiment A: Current 4-class
    # ========================================================
    print("\n" + "=" * 70)
    print("A: CURRENT 4-CLASS (Actionable, Wontfix, Fixed, Downstream)")
    print("=" * 70)
    s1_4class = train_stage_1(train)
    test_4class = predict_stage_1(s1_4class, test_valid.copy())

    true_4class = test_4class['alert_summary_status'].replace(STATUS_MERGE).values
    pred_4class = test_4class['s1_pred'].values
    conf_4class = test_4class['s1_confidence'].values

    print_threshold_curve("4-class", true_4class, pred_4class, conf_4class, DISPOSITION_CLASSES)

    # ========================================================
    # Experiment B: 5-class (Ack, Reassigned separate)
    # ========================================================
    print("\n" + "=" * 70)
    print("B: 5-CLASS (Ack, Reassigned, Wontfix, Fixed, Downstream)")
    print("=" * 70)
    s1_5class = train_stage_1_5class(train)
    test_5class = predict_stage_1_custom(s1_5class, test_valid.copy(), STATUS_MERGE_5CLASS, DISPOSITION_5CLASS)

    true_5class = test_5class['alert_summary_status'].replace(STATUS_MERGE_5CLASS).values
    pred_5class = test_5class['s1_pred'].values
    conf_5class = test_5class['s1_confidence'].values

    print_threshold_curve("5-class", true_5class, pred_5class, conf_5class, DISPOSITION_5CLASS)

    # ========================================================
    # End-to-end comparison at t=0.90
    # ========================================================
    print("\n" + "=" * 70)
    print("END-TO-END COMPARISON AT t=0.90")
    print("=" * 70)

    for name, true, pred, conf, classes in [
        ("4-class", true_4class, pred_4class, conf_4class, DISPOSITION_CLASSES),
        ("5-class", true_5class, pred_5class, conf_5class, DISPOSITION_5CLASS),
    ]:
        mask = conf >= 0.90
        n = mask.sum()
        total = len(true)
        if n > 0:
            acc = (true[mask] == pred[mask]).mean()
            print(f"\n  {name}: {acc:.1%} accuracy, {n}/{total} = {n/total:.1%} coverage")
            # Per-class breakdown
            for code, label in sorted(classes.items()):
                cls_mask = mask & (pred == code)
                if cls_mask.sum() > 0:
                    cls_acc = (true[cls_mask] == code).mean()
                    print(f"    {label}: {cls_mask.sum()} predicted, {cls_acc:.1%} correct")

    # ========================================================
    # Including Stage 0 Invalid in end-to-end
    # ========================================================
    print("\n" + "=" * 70)
    print("FULL END-TO-END (Stage 0 + Stage 1) AT VARIOUS THRESHOLDS")
    print("=" * 70)

    # Stage 0 predictions
    s0_invalid = test_s0['s0_pred'] == 3
    s0_uncertain = test_s0['s0_pred'] == -1
    true_status_all = test_s0['alert_summary_status'].values
    n_total = len(test_s0)

    for name, true_s1, pred_s1, conf_s1, merge_map in [
        ("4-class", true_4class, pred_4class, conf_4class, STATUS_MERGE),
        ("5-class", true_5class, pred_5class, conf_5class, STATUS_MERGE_5CLASS),
    ]:
        print(f"\n  {name} end-to-end:")
        true_all_merged = pd.Series(true_status_all).replace(merge_map).values

        for t in [0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
            # Stage 0 confident invalids (always correct at ~100%)
            correct = 0
            automated = 0

            # S0 invalid predictions
            s0_inv_ids = test_s0[s0_invalid].index
            for idx in s0_inv_ids:
                if true_status_all[test_s0.index.get_loc(idx)] == 3:
                    correct += 1
                automated += 1

            # S1 confident predictions at threshold t
            s1_conf_mask = conf_s1 >= t
            correct += (true_s1[s1_conf_mask] == pred_s1[s1_conf_mask]).sum()
            automated += s1_conf_mask.sum()

            # Uncertain from S0 are deferred
            coverage = automated / n_total
            acc = correct / automated if automated > 0 else 0
            print(f"    t={t:.2f}: {acc:.1%} accuracy, {automated}/{n_total} = {coverage:.1%} coverage, "
                  f"({n_total - automated} deferred)")


def train_stage_1_5class(train_summaries, calibration_method='isotonic'):
    """Train Stage 1 with 5 classes (Ack and Reassigned kept separate)."""
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    df = train_summaries.copy()
    df = df[df['alert_summary_status'] != 3].copy()

    # 5-class merge: only Backedout(8) -> Ack(4)
    df['disposition'] = df['alert_summary_status'].replace(STATUS_MERGE_5CLASS)
    valid_classes = set(DISPOSITION_5CLASS.keys())
    df = df[df['disposition'].isin(valid_classes)].copy()

    class_encoder = LabelEncoder()
    y = class_encoder.fit_transform(df['disposition'].values)

    cat_encoders = {}
    for col in STAGE_1_CAT_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_enc'] = le.fit_transform(df[col].astype(str).fillna('unknown'))
            cat_encoders[col] = le

    feature_cols = STAGE_1_FEATURES + [c + '_enc' for c in STAGE_1_CAT_FEATURES if c in df.columns]
    X = df[feature_cols].copy().fillna(0)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    class_counts = pd.Series(y).value_counts()
    class_names = class_encoder.classes_
    print(f"Stage 1 (5-class) training: {len(y)} summaries, {len(class_names)} classes")
    for i, name in enumerate(class_names):
        count = class_counts.get(i, 0)
        label = DISPOSITION_5CLASS.get(int(name), str(name))
        print(f"  {label}: {count}")

    base_model = _build_ensemble()

    # OOF predictions for threshold tuning
    oof_base = _build_base_model()
    oof_proba = get_oof_predictions(oof_base, X_scaled.values, y,
                                     n_folds=5, random_state=RANDOM_SEED)

    per_class_thresholds = find_per_class_thresholds(
        y, oof_proba, target_accuracy=0.80, min_samples=5
    )
    print(f"Stage 1 (5-class) per-class thresholds:")
    for i, name in enumerate(class_names):
        label = DISPOSITION_5CLASS.get(int(name), str(name))
        print(f"  {label}: {per_class_thresholds[i]:.2f}")

    calibrated_model = calibrate_model(
        base_model, X_scaled.values, y,
        method=calibration_method, cv=5
    )

    return {
        'model': calibrated_model,
        'scaler': scaler,
        'cat_encoders': cat_encoders,
        'class_encoder': class_encoder,
        'feature_cols': list(X_scaled.columns),
        'threshold': per_class_thresholds,
        'status_merge': STATUS_MERGE_5CLASS,
        'disposition_classes': DISPOSITION_5CLASS,
    }


def predict_stage_1_custom(artifacts, summary_df, status_merge, disposition_classes):
    """Predict using custom Stage 1 artifacts."""
    df = summary_df.copy()
    model = artifacts['model']
    scaler = artifacts['scaler']
    cat_encoders = artifacts['cat_encoders']
    class_encoder = artifacts['class_encoder']
    threshold = artifacts['threshold']

    for col, le in cat_encoders.items():
        if col in df.columns:
            vals = df[col].astype(str).fillna('unknown')
            known = set(le.classes_)
            vals = vals.apply(lambda x: x if x in known else le.classes_[0])
            df[col + '_enc'] = le.transform(vals)

    feature_cols = artifacts['feature_cols']
    X = df[feature_cols].copy().fillna(0)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)

    proba = model.predict_proba(X_scaled.values)
    confidence = np.max(proba, axis=1)
    predicted_idx = np.argmax(proba, axis=1)
    predicted_class = class_encoder.inverse_transform(predicted_idx)

    if isinstance(threshold, np.ndarray):
        per_sample_threshold = threshold[predicted_idx]
        is_confident = confidence >= per_sample_threshold
    else:
        is_confident = confidence >= float(threshold)

    df['s1_pred'] = np.where(is_confident, predicted_class, -1).astype(int)
    df['s1_confidence'] = confidence
    df['s1_is_confident'] = is_confident

    return df


def print_threshold_curve(name, true, pred, conf, classes):
    """Print accuracy/coverage at various thresholds."""
    print(f"\n  {name} coverage-accuracy curve:")
    print(f"  {'Threshold':>10} {'Coverage':>10} {'Accuracy':>10} {'N':>6}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*6}")

    for t in [0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        mask = conf >= t
        n = mask.sum()
        total = len(true)
        if n > 0:
            acc = (true[mask] == pred[mask]).mean()
            cov = n / total
            print(f"  {t:>10.2f} {cov:>10.1%} {acc:>10.1%} {n:>6}")

    # Per-class breakdown at t=0.90
    t = 0.90
    mask = conf >= t
    if mask.sum() > 0:
        print(f"\n  Per-class breakdown at t={t}:")
        for code, label in sorted(classes.items()):
            cls_pred_mask = mask & (pred == code)
            cls_true_count = (true == code).sum()
            if cls_pred_mask.sum() > 0:
                cls_acc = (true[cls_pred_mask] == code).mean()
                print(f"    {label}: {cls_pred_mask.sum()} predicted ({cls_acc:.1%} precision), "
                      f"{cls_true_count} true total")
            else:
                print(f"    {label}: 0 predicted, {cls_true_count} true total")


if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    run_experiment()
