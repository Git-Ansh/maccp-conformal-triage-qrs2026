"""
Cross-Repository Transfer Experiment

Train the cascade on autoland data, test on mozilla-beta + firefox-android.
Tests whether triage patterns transfer across Mozilla sub-projects.

Ground truth is collected separately and stored out of the model's reach.
The model never sees mozilla-beta/firefox-android labels during training.

Leakage prevention:
  - Training data: only autoland summaries
  - Test data: only mozilla-beta + firefox-android summaries
  - No label information from test repos leaks into training
  - Suite encoders, scalers, thresholds all fitted on autoland only
  - Suite Invalid rates computed from autoland only
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
from common.data_paths import RANDOM_SEED, PROJECT_ROOT

from cascade.data.loader import (
    load_alerts, build_has_bug, build_summary_df,
    EXCLUDED_SUMMARY_STATUSES
)
from cascade.stages.stage_0_invalid_filter import train_stage_0, predict_stage_0
from cascade.stages.stage_1_disposition import (
    train_stage_1, predict_stage_1, get_cross_validated_predictions,
    STATUS_MERGE, DISPOSITION_CLASSES
)
from cascade.stages.stage_3_bug_linkage import train_stage_3, predict_stage_3


GROUND_TRUTH_PATH = PROJECT_ROOT / 'cascade_outputs' / 'cross_repo_ground_truth.csv'


def prepare_cross_repo_split() -> Dict:
    """
    Split data by repository instead of time.
    Train: autoland only. Test: mozilla-beta + firefox-android.
    Ground truth stored separately.
    """
    print("Loading and preparing cross-repo data...")
    alerts = load_alerts()
    alerts = build_has_bug(alerts)

    print(f"  Total alerts: {len(alerts)}")

    summary_df = build_summary_df(alerts)

    # Merge TS features if available
    ts_cache = PROJECT_ROOT / 'cascade_outputs' / 'ts_features_per_summary.csv'
    if ts_cache.exists():
        ts_df = pd.read_csv(ts_cache)
        summary_df = summary_df.merge(ts_df, on='alert_summary_id', how='left')
        print(f"  Merged TS features")

    # Filter to resolved only
    summary_resolved = summary_df[
        ~summary_df['alert_summary_status'].isin(EXCLUDED_SUMMARY_STATUSES)
    ].copy()

    # Split by repository
    train_repos = ['autoland']
    test_repos = ['mozilla-beta', 'firefox-android']

    train_summaries = summary_resolved[
        summary_resolved['repository'].isin(train_repos)
    ].copy()
    test_summaries = summary_resolved[
        summary_resolved['repository'].isin(test_repos)
    ].copy()

    print(f"\n  TRAIN (autoland): {len(train_summaries)} summaries")
    print(f"    Status distribution:")
    for status, count in train_summaries['alert_summary_status'].value_counts().sort_index().items():
        print(f"      {status}: {count}")

    print(f"\n  TEST (mozilla-beta + firefox-android): {len(test_summaries)} summaries")
    print(f"    By repo:")
    for repo, count in test_summaries['repository'].value_counts().items():
        print(f"      {repo}: {count}")
    print(f"    Status distribution:")
    for status, count in test_summaries['alert_summary_status'].value_counts().sort_index().items():
        print(f"      {status}: {count}")

    # Store ground truth SEPARATELY before any model sees it
    ground_truth = test_summaries[['alert_summary_id', 'alert_summary_status',
                                    'has_bug', 'repository']].copy()
    ground_truth['true_disposition'] = ground_truth['alert_summary_status'].replace(STATUS_MERGE)
    ground_truth['true_is_invalid'] = (ground_truth['alert_summary_status'] == 3).astype(int)

    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    ground_truth.to_csv(GROUND_TRUTH_PATH, index=False)
    print(f"\n  Ground truth saved to {GROUND_TRUTH_PATH}")
    print(f"  (Contains {len(ground_truth)} test summaries -- model never sees these labels)")

    # Get corresponding alerts
    train_alert_ids = set(train_summaries['alert_summary_id'])
    test_alert_ids = set(test_summaries['alert_summary_id'])
    train_alerts = alerts[alerts['alert_summary_id'].isin(train_alert_ids)].copy()
    test_alerts = alerts[alerts['alert_summary_id'].isin(test_alert_ids)].copy()

    return {
        'train_summaries': train_summaries,
        'test_summaries': test_summaries,
        'train_alerts': train_alerts,
        'test_alerts': test_alerts,
        'ground_truth': ground_truth,
    }


def run_cross_repo_cascade(data: Dict) -> Dict:
    """
    Train cascade on autoland, run inference on mozilla-beta + firefox-android.
    Evaluate against separately stored ground truth.
    """
    train = data['train_summaries']
    test = data['test_summaries']
    ground_truth = data['ground_truth']

    print("\n" + "=" * 70)
    print("TRAINING CASCADE ON AUTOLAND ONLY")
    print("=" * 70)

    # Stage 0
    print("\n[Stage 0] Training Invalid Filter...")
    s0_artifacts = train_stage_0(train)

    # Stage 1
    print("\n[Stage 1] Training Disposition Classifier...")
    s1_artifacts = train_stage_1(train)

    # CV predictions for Stage 3
    print("\n[Stage 1-CV] Getting cross-validated predictions...")
    cv_preds = get_cross_validated_predictions(train)

    # Stage 3 Mode B (no status context -- more fair for cross-repo)
    print("\n[Stage 3] Training Bug Linkage (Mode B)...")
    s3_artifacts_b = train_stage_3(train, mode='B')

    # Also train Mode A for comparison
    print("\n[Stage 3] Training Bug Linkage (Mode A)...")
    s3_artifacts_a = train_stage_3(train, cv_preds, mode='A')

    print("\n" + "=" * 70)
    print("RUNNING INFERENCE ON MOZILLA-BETA + FIREFOX-ANDROID")
    print("=" * 70)

    # Stage 0
    test_s0 = predict_stage_0(s0_artifacts, test)

    # Stage 1 (on non-Invalid from Stage 0)
    valid_mask = test_s0['s0_pred'] != 3
    uncertain_mask = test_s0['s0_pred'] == -1
    pass_to_s1 = test_s0[valid_mask & ~uncertain_mask].copy()

    if len(pass_to_s1) > 0:
        test_s1 = predict_stage_1(s1_artifacts, pass_to_s1)
    else:
        test_s1 = pass_to_s1.copy()

    # Stage 3 Mode B on all test
    test_s3 = predict_stage_3(s3_artifacts_b, test)

    print("\n" + "=" * 70)
    print("EVALUATION AGAINST GROUND TRUTH")
    print("=" * 70)

    results = {}

    # --- Stage 0 evaluation ---
    print("\n--- Stage 0: Invalid Filter ---")
    gt = ground_truth.set_index('alert_summary_id')
    s0_results = test_s0[['alert_summary_id', 's0_pred', 's0_confidence', 's0_is_confident']].copy()
    s0_results = s0_results.set_index('alert_summary_id')
    merged_s0 = s0_results.join(gt[['true_is_invalid']])

    confident_s0 = np.asarray(test_s0['s0_is_confident'].values, dtype=bool)
    n_conf_s0 = confident_s0.sum()
    n_total = len(test_s0)

    pred_invalid = (test_s0['s0_pred'] == 3).values
    true_invalid = gt.loc[test_s0['alert_summary_id'].values, 'true_is_invalid'].values

    print(f"  Total test: {n_total}")
    print(f"  Confident: {n_conf_s0} ({n_conf_s0/n_total:.1%})")

    if pred_invalid.sum() > 0:
        prec = true_invalid[pred_invalid].mean()
        rec = pred_invalid[true_invalid == 1].mean() if true_invalid.sum() > 0 else 0
        print(f"  Invalid: precision={prec:.3f}, recall={rec:.3f}")
        print(f"    Predicted Invalid: {pred_invalid.sum()}, True Invalid: {true_invalid.sum()}")
        results['s0_precision'] = prec
        results['s0_recall'] = rec
    else:
        print(f"  No Invalid predictions (True Invalid: {true_invalid.sum()})")

    if n_conf_s0 > 0:
        pred_binary = pred_invalid[confident_s0].astype(int)
        true_binary = true_invalid[confident_s0]
        acc = (pred_binary == true_binary).mean()
        print(f"  Accuracy on confident: {acc:.4f}")
        results['s0_accuracy'] = acc
        results['s0_coverage'] = n_conf_s0 / n_total

    # --- Stage 1 evaluation ---
    print("\n--- Stage 1: Disposition ---")
    if len(test_s1) > 0:
        s1_gt_status = gt.loc[test_s1['alert_summary_id'].values, 'true_disposition'].values
        s1_pred = test_s1['s1_pred'].values
        s1_confident = np.asarray(test_s1['s1_is_confident'].values, dtype=bool)

        n_conf_s1 = s1_confident.sum()
        print(f"  Passed to Stage 1: {len(test_s1)}")
        print(f"  Confident: {n_conf_s1} ({n_conf_s1/len(test_s1):.1%})")

        if n_conf_s1 > 0:
            acc_s1 = (s1_gt_status[s1_confident] == s1_pred[s1_confident]).mean()
            print(f"  Accuracy on confident: {acc_s1:.4f}")
            results['s1_accuracy'] = acc_s1
            results['s1_coverage'] = n_conf_s1 / len(test_s1)

            # Breakdown by predicted class
            print(f"  Prediction breakdown:")
            for cls_code, cls_name in DISPOSITION_CLASSES.items():
                mask = s1_pred[s1_confident] == cls_code
                if mask.sum() > 0:
                    cls_acc = (s1_gt_status[s1_confident][mask] == cls_code).mean()
                    print(f"    {cls_name}: {mask.sum()} predictions, {cls_acc:.1%} correct")

        # Majority baseline
        majority_class = pd.Series(s1_gt_status).value_counts().index[0]
        majority_acc = (s1_gt_status == majority_class).mean()
        print(f"  Majority baseline (always {DISPOSITION_CLASSES.get(majority_class, majority_class)}): {majority_acc:.4f}")
        results['s1_majority_baseline'] = majority_acc
    else:
        print("  No summaries passed to Stage 1")

    # --- Stage 3 evaluation (Mode B) ---
    print("\n--- Stage 3: Bug Linkage (Mode B) ---")
    s3_pred = test_s3['s3_pred'].values
    s3_confident = np.asarray(test_s3['s3_is_confident'].values, dtype=bool)
    true_bug = gt.loc[test_s3['alert_summary_id'].values, 'has_bug'].values if 'has_bug' in gt.columns else test_s3['has_bug'].values

    n_conf_s3 = s3_confident.sum()
    print(f"  Total: {len(test_s3)}")
    print(f"  Confident: {n_conf_s3} ({n_conf_s3/len(test_s3):.1%})")

    if n_conf_s3 > 0:
        pred_mapped = np.where(s3_pred[s3_confident] == 2, 1, 0)
        true_mapped = true_bug[s3_confident]
        acc_s3 = (pred_mapped == true_mapped).mean()
        print(f"  Accuracy on confident: {acc_s3:.4f}")
        results['s3_accuracy'] = acc_s3
        results['s3_coverage'] = n_conf_s3 / len(test_s3)

    # Majority baseline for has_bug
    majority_bug = int(pd.Series(true_bug).value_counts().index[0])
    majority_bug_acc = (true_bug == majority_bug).mean()
    print(f"  Majority baseline (always {majority_bug}): {majority_bug_acc:.4f}")
    results['s3_majority_baseline'] = majority_bug_acc

    # --- End-to-end ---
    print("\n--- End-to-End Summary ---")
    # How many groups are fully automated?
    automated_mask = np.asarray(test_s0['s0_is_confident'].values, dtype=bool)
    n_automated = automated_mask.sum()

    # For automated groups, check accuracy
    auto_pred = test_s0.loc[automated_mask, 's0_pred'].values
    auto_true_status = gt.loc[test_s0.loc[automated_mask, 'alert_summary_id'].values, 'alert_summary_status'].values
    auto_true_status_merged = pd.Series(auto_true_status).replace(STATUS_MERGE).values

    # Stage 0 says 3 (Invalid) or 0 (Valid->Stage 1 handles)
    # For valid ones that went to Stage 1, use Stage 1's prediction
    auto_correct = 0
    auto_total = 0
    for i, (s0_pred_val, summ_id) in enumerate(zip(auto_pred, test_s0.loc[automated_mask, 'alert_summary_id'].values)):
        true_status = gt.loc[summ_id, 'alert_summary_status']
        true_merged = gt.loc[summ_id, 'true_disposition']

        if s0_pred_val == 3:
            # Stage 0 predicted Invalid
            auto_correct += int(true_status == 3)
            auto_total += 1
        elif s0_pred_val == 0:
            # Stage 0 said Valid -> check if Stage 1 got it right
            if summ_id in test_s1['alert_summary_id'].values:
                s1_row = test_s1[test_s1['alert_summary_id'] == summ_id]
                if len(s1_row) > 0:
                    s1_pred_val = s1_row['s1_pred'].values[0]
                    if s1_pred_val != -1:
                        auto_correct += int(s1_pred_val == true_merged)
                        auto_total += 1

    if auto_total > 0:
        e2e_acc = auto_correct / auto_total
        print(f"  Automated groups: {auto_total}/{n_total} ({auto_total/n_total:.1%})")
        print(f"  End-to-end accuracy: {e2e_acc:.4f}")
        results['e2e_accuracy'] = e2e_acc
        results['e2e_coverage'] = auto_total / n_total

    # Per-repo breakdown
    print("\n--- Per-Repository Breakdown ---")
    for repo in test['repository'].unique():
        repo_mask = test['repository'] == repo
        repo_ids = test.loc[repo_mask, 'alert_summary_id'].values
        n_repo = len(repo_ids)
        repo_gt = gt.loc[gt.index.isin(repo_ids)]
        n_invalid = repo_gt['true_is_invalid'].sum()
        n_bug = repo_gt['has_bug'].sum() if 'has_bug' in repo_gt.columns else 0
        print(f"  {repo}: {n_repo} summaries ({n_invalid} Invalid, {n_bug} has_bug)")

    results['n_train'] = len(train)
    results['n_test'] = len(test)

    return results


if __name__ == '__main__':
    data = prepare_cross_repo_split()
    results = run_cross_repo_cascade(data)

    print("\n" + "=" * 70)
    print("CROSS-REPO TRANSFER RESULTS SUMMARY")
    print("=" * 70)
    for k, v in sorted(results.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
