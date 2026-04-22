"""
Baseline Gauntlet: flat model comparisons for all datasets.

For each dataset, runs:
  1. Majority class baseline
  2. Flat Logistic Regression
  3. Flat Random Forest
  4. Flat XGBoost (no confidence gating)

All baselines use the same features and train/test split as the cascade.
Results saved to conformal_outputs/{dataset}/baselines.json.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

OUTPUT_DIR = PROJECT_ROOT / 'conformal_outputs'


def _detect_gpu():
    """Detect XGBoost GPU availability."""
    if not HAS_XGBOOST:
        return {}
    try:
        _t = XGBClassifier(tree_method='hist', device='cuda', n_estimators=1)
        _t.fit(np.random.rand(10, 2), np.random.randint(0, 2, 10))
        del _t
        return {'tree_method': 'hist', 'device': 'cuda'}
    except Exception:
        return {}


def run_baselines(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    dataset_name: str,
    feature_names: Optional[List[str]] = None,
    save_results: bool = True,
) -> Dict:
    """
    Run all flat baselines on a dataset.

    Args:
        train_X: Training features
        train_y: Training labels
        test_X: Test features
        test_y: Test labels
        dataset_name: Name for output directory and reporting
        feature_names: Optional feature names for reporting
        save_results: Whether to save results to disk

    Returns:
        Dict of baseline_name -> {accuracy, precision, recall, f1}
    """
    print("=" * 70)
    print(f"BASELINE GAUNTLET: {dataset_name.upper()}")
    print("=" * 70)

    n_classes = len(np.unique(np.concatenate([train_y, test_y])))
    avg = 'weighted' if n_classes > 2 else 'binary'
    gpu_params = _detect_gpu()

    results = {}

    # 1. Majority class baseline
    majority_class = int(np.argmax(np.bincount(train_y)))
    majority_pred = np.full_like(test_y, majority_class)
    majority_acc = accuracy_score(test_y, majority_pred)
    p, r, f, _ = precision_recall_fscore_support(
        test_y, majority_pred, average=avg, zero_division=0)
    results['majority'] = {
        'accuracy': float(majority_acc),
        'precision': float(p), 'recall': float(r), 'f1': float(f),
        'majority_class': int(majority_class),
    }
    print(f"\n  Majority class: {majority_acc:.1%} (class {majority_class})")

    # 2. Logistic Regression
    print("  Training Logistic Regression...")
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    test_X_scaled = scaler.transform(test_X)

    lr = LogisticRegression(
        max_iter=1000, random_state=42, n_jobs=-1,
        class_weight='balanced',
    )
    lr.fit(train_X_scaled, train_y)
    lr_pred = lr.predict(test_X_scaled)
    lr_acc = accuracy_score(test_y, lr_pred)
    p, r, f, _ = precision_recall_fscore_support(
        test_y, lr_pred, average=avg, zero_division=0)
    results['logistic_regression'] = {
        'accuracy': float(lr_acc),
        'precision': float(p), 'recall': float(r), 'f1': float(f),
    }
    print(f"  Logistic Regression: {lr_acc:.1%}")

    # 3. Random Forest
    print("  Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1,
    )
    rf.fit(train_X, train_y)
    rf_pred = rf.predict(test_X)
    rf_acc = accuracy_score(test_y, rf_pred)
    p, r, f, _ = precision_recall_fscore_support(
        test_y, rf_pred, average=avg, zero_division=0)
    results['random_forest'] = {
        'accuracy': float(rf_acc),
        'precision': float(p), 'recall': float(r), 'f1': float(f),
    }
    print(f"  Random Forest: {rf_acc:.1%}")

    # 4. XGBoost (no gating)
    if HAS_XGBOOST:
        print("  Training XGBoost...")
        eval_metric = 'mlogloss' if n_classes > 2 else 'logloss'
        xgb = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric=eval_metric, n_jobs=-1,
            **gpu_params,
        )
        xgb.fit(train_X, train_y)
        xgb_pred = xgb.predict(test_X)
        xgb_acc = accuracy_score(test_y, xgb_pred)
        p, r, f, _ = precision_recall_fscore_support(
            test_y, xgb_pred, average=avg, zero_division=0)
        results['xgboost'] = {
            'accuracy': float(xgb_acc),
            'precision': float(p), 'recall': float(r), 'f1': float(f),
        }
        print(f"  XGBoost: {xgb_acc:.1%}")

    # Summary table
    print(f"\n  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for name, res in results.items():
        print(f"  {name:<25} {res['accuracy']:>10.1%} {res['precision']:>10.3f} "
              f"{res['recall']:>10.3f} {res['f1']:>10.3f}")

    # Save
    if save_results:
        out_dir = OUTPUT_DIR / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / 'baselines.json'
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {path}")

    return results


def run_all_baselines():
    """Run baselines for all available datasets."""
    # JM1
    try:
        from conformal.data.jm1_loader import load_jm1_data
        data = load_jm1_data()
        train_df = data['train_df']
        test_df = data['test_df']
        feature_cols = data['feature_cols']
        run_baselines(
            train_df[feature_cols].fillna(0).values,
            train_df['defective'].values,
            test_df[feature_cols].fillna(0).values,
            test_df['defective'].values,
            'jm1', feature_cols,
        )
    except Exception as e:
        print(f"JM1 baselines failed: {e}")

    # Eclipse
    try:
        from conformal.data.eclipse_loader import prepare_eclipse_data
        from conformal.stages.eclipse_config import prepare_stage_0_data
        data = prepare_eclipse_data()
        s0_data = prepare_stage_0_data(
            data['train_df'], data['test_df'],
            data['numeric_features'], data['categorical_features'],
        )
        run_baselines(
            s0_data['train_X'], s0_data['train_y'],
            s0_data['test_X'], s0_data['test_y'],
            'eclipse', s0_data['feature_cols'],
        )
    except Exception as e:
        print(f"Eclipse baselines failed: {e}")


if __name__ == '__main__':
    run_all_baselines()
