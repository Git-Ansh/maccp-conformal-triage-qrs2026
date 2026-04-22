"""
JM1 Defect Prediction Cascade Pipeline.

Single-stage cascade demonstrating confidence gating for defect prediction:
  Stage 0: Defective vs clean (binary classification)

The cascade value: reduce false positives via confidence gating.
At 90% confidence, predictions are more precise than flat classifier,
at the cost of deferring uncertain modules to manual review.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from cascade.framework.confidence_stage import ConfidenceStage
from conformal.data.jm1_loader import load_jm1_data
from conformal.stages.jm1_config import JM1_CLASSES

OUTPUT_DIR = PROJECT_ROOT / 'conformal_outputs' / 'jm1'


def run_jm1_cascade(
    target_accuracy: float = 0.85,
    save_results: bool = True,
):
    """Run JM1 defect prediction with confidence gating."""
    print("=" * 70)
    print("JM1 DEFECT PREDICTION CASCADE")
    print("=" * 70)

    # Load data
    data = load_jm1_data()
    train_df = data['train_df']
    test_df = data['test_df']
    feature_cols = data['feature_cols']

    train_X = train_df[feature_cols].fillna(0).values
    train_y = train_df['defective'].values
    test_X = test_df[feature_cols].fillna(0).values
    test_y = test_df['defective'].values

    # Majority baseline
    majority_class = int(np.argmax(np.bincount(train_y)))
    majority_acc = (test_y == majority_class).mean()
    print(f"\nMajority baseline: {majority_acc:.1%} (class {JM1_CLASSES.get(majority_class, majority_class)})")

    # Stage 0: Defect prediction with confidence gating
    print("\n" + "-" * 50)
    print("STAGE 0: DEFECT PREDICTION")
    print("-" * 50)

    stage = ConfidenceStage(
        name='S0_defect',
        classes=JM1_CLASSES,
        target_accuracy=target_accuracy,
    )
    stage.fit(train_X, train_y, feature_names=feature_cols)

    # Predictions
    preds = stage.predict(test_X)
    curve = stage.coverage_accuracy_curve(test_X, test_y)

    # Evaluate
    is_conf = preds['is_confident']
    if is_conf.any():
        cascade_acc = (test_y[is_conf] == preds['class'][is_conf]).mean()
        cascade_cov = is_conf.mean()
        lift = cascade_acc - majority_acc
        print(f"\nCascade: {cascade_acc:.1%} accuracy, {cascade_cov:.1%} coverage "
              f"(lift: {lift:+.1%})")

        # Per-class results
        for cls_code, cls_name in JM1_CLASSES.items():
            cls_mask = is_conf & (preds['class'] == cls_code)
            if cls_mask.any():
                cls_prec = (test_y[cls_mask] == cls_code).mean()
                print(f"  {cls_name}: precision={cls_prec:.1%}, n={cls_mask.sum()}")
    else:
        cascade_acc = 0
        cascade_cov = 0
        lift = 0

    # Flat baseline
    print("\n" + "-" * 50)
    print("FLAT BASELINE")
    print("-" * 50)

    try:
        from xgboost import XGBClassifier
        gpu_params = {}
        try:
            _t = XGBClassifier(tree_method='hist', device='cuda', n_estimators=1)
            _t.fit(np.random.rand(10, 2), np.random.randint(0, 2, 10))
            gpu_params = {'tree_method': 'hist', 'device': 'cuda'}
            del _t
        except Exception:
            pass
        flat_model = XGBClassifier(n_estimators=200, max_depth=6, random_state=42,
                                    eval_metric='logloss', n_jobs=-1, **gpu_params)
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        flat_model = RandomForestClassifier(n_estimators=200, max_depth=10,
                                             class_weight='balanced', random_state=42)

    flat_model.fit(train_X, train_y)
    flat_pred = flat_model.predict(test_X)
    flat_acc = accuracy_score(test_y, flat_pred)
    print(f"Flat XGBoost: {flat_acc:.1%} accuracy (100% coverage)")
    print(f"Cascade advantage: {cascade_acc - flat_acc:+.1%} accuracy at "
          f"{cascade_cov:.1%} coverage")

    # Coverage-accuracy curve
    print("\nCoverage-accuracy curve:")
    print(f"  {'Threshold':>10} {'Coverage':>10} {'Accuracy':>10} {'Lift':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for _, row in curve.iterrows():
        lift_val = row['accuracy'] - majority_acc
        print(f"  {row['threshold']:>10.2f} {row['coverage']:>10.1%} "
              f"{row['accuracy']:>10.1%} {lift_val:>+8.1%}")

    # Deferral analysis
    if is_conf.any():
        n_deferred = (~is_conf).sum()
        deferred_defect_rate = test_y[~is_conf].mean() if n_deferred > 0 else 0
        overall_defect_rate = test_y.mean()
        print(f"\nDeferral analysis:")
        print(f"  Deferred: {n_deferred} ({n_deferred/len(test_y):.1%})")
        print(f"  Deferred defect rate: {deferred_defect_rate:.1%} "
              f"(overall: {overall_defect_rate:.1%})")

    results = {
        'dataset': 'JM1 (PROMISE)',
        'n_train': len(train_df),
        'n_test': len(test_df),
        'n_features': len(feature_cols),
        'cascade_accuracy': float(cascade_acc),
        'cascade_coverage': float(cascade_cov),
        'flat_accuracy': float(flat_acc),
        'majority_baseline': float(majority_acc),
        'accuracy_lift': float(lift),
    }

    if save_results:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        results_path = OUTPUT_DIR / f'jm1_results_{datetime.now():%Y%m%d_%H%M%S}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")

        curve.to_csv(OUTPUT_DIR / 'jm1_curve.csv', index=False)

    return results


if __name__ == '__main__':
    run_jm1_cascade()
