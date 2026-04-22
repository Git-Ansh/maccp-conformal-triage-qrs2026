"""
ServiceNow ITSM Sequential Cascade Pipeline.

Runs a 3-stage SEQUENTIAL cascade on ServiceNow incident data:
  S0: Priority prediction (4-class)
      -> confident: priority assigned, forward to S1
      -> uncertain: defer to human

  S1: Category prediction (top-10 + Other) on S0-forwarded items
      -> confident: category assigned, forward to S2
      -> uncertain: defer to human

  S2: Team routing (top-10 + Other) on S1-forwarded items
      -> confident: team assigned (fully automated!)
      -> uncertain: defer to human

End-to-end: an item is "fully automated" only if ALL 3 stages were confident.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from cascade.framework.confidence_stage import ConfidenceStage
from conformal.data.servicenow_loader import prepare_servicenow_data
from conformal.stages.servicenow_config import (
    prepare_stage_0_data, prepare_stage_1_data, prepare_stage_2_data,
    PRIORITY_CLASSES,
)

OUTPUT_DIR = PROJECT_ROOT / 'conformal_outputs' / 'servicenow'


def run_servicenow_cascade(
    target_accuracy: float = 0.85,
    save_results: bool = True,
):
    """Run full ServiceNow sequential cascade pipeline."""
    print("=" * 70)
    print("SERVICENOW ITSM INCIDENT TRIAGE CASCADE (SEQUENTIAL)")
    print("=" * 70)
    print(f"  Target accuracy: {target_accuracy}")
    print()

    # Load data (top-10 groups for better accuracy)
    data = prepare_servicenow_data(top_n_groups=10)
    train_df = data['train_df']
    test_df = data['test_df']
    numeric_features = data['numeric_features']
    cat_features = data['categorical_features']
    n_test = len(test_df)

    results = {
        'dataset': 'ServiceNow ITSM (UCI 498)',
        'n_train': len(train_df),
        'n_test': n_test,
        'target_accuracy': target_accuracy,
        'stages': {},
    }

    # =========================================================
    # STAGE 0: Priority Prediction
    # =========================================================
    print("\n" + "=" * 70)
    print("STAGE 0: PRIORITY PREDICTION")
    print("=" * 70)

    s0_data = prepare_stage_0_data(train_df, test_df, numeric_features, cat_features)

    majority_class_s0 = int(np.argmax(np.bincount(s0_data['train_y'])))
    majority_acc_s0 = (s0_data['test_y'] == majority_class_s0).mean()
    print(f"Majority baseline (S0): {majority_acc_s0:.1%} "
          f"(class {PRIORITY_CLASSES.get(majority_class_s0, majority_class_s0)})")

    s0_stage = ConfidenceStage(
        name='S0_priority',
        classes=PRIORITY_CLASSES,
        target_accuracy=target_accuracy,
    )
    s0_stage.fit(s0_data['train_X'], s0_data['train_y'],
                 feature_names=s0_data['feature_cols'])
    s0_preds = s0_stage.predict(s0_data['test_X'])
    s0_curve = s0_stage.coverage_accuracy_curve(
        s0_data['test_X'], s0_data['test_y'])

    s0_conf = np.asarray(s0_preds['is_confident'], dtype=bool)
    s0_class = s0_preds['class']
    s0_confidence = s0_preds['confidence']

    if s0_conf.any():
        s0_acc = (s0_data['test_y'][s0_conf] == s0_class[s0_conf]).mean()
        s0_cov = s0_conf.mean()
        s0_lift = s0_acc - majority_acc_s0
        print(f"\nS0 Results: {s0_acc:.1%} accuracy, {s0_cov:.1%} coverage "
              f"(lift: {s0_lift:+.1%})")
    else:
        s0_acc = s0_cov = s0_lift = 0.0

    # Per-class breakdown
    for cls_code, cls_name in PRIORITY_CLASSES.items():
        cls_mask = s0_conf & (s0_class == cls_code)
        if cls_mask.any():
            cls_correct = (s0_data['test_y'][cls_mask] == cls_code).mean()
            print(f"  {cls_name}: {cls_correct:.1%} precision, {cls_mask.sum()} predictions")

    # S0 confident -> forward to S1, uncertain -> defer
    s0_forward = s0_conf  # only confident items proceed
    s0_deferred = ~s0_conf
    print(f"\nS0 confident (forwarded to S1): {s0_forward.sum()} items")
    print(f"S0 deferred: {s0_deferred.sum()} items")

    results['stages']['S0_priority'] = {
        'accuracy': float(s0_acc),
        'coverage': float(s0_cov),
        'majority_baseline': float(majority_acc_s0),
        'accuracy_lift': float(s0_lift),
        'n_forwarded': int(s0_forward.sum()),
        'n_deferred': int(s0_deferred.sum()),
    }

    # --- Attach S0 results to test_df for downstream use ---
    test_df = test_df.copy()
    test_df['s0_confidence'] = s0_confidence
    test_df['s0_pred'] = s0_class

    # S1 test mask: only S0-confident items
    test_s1_mask = pd.Series(s0_forward, index=test_df.index)

    # =========================================================
    # STAGE 1: Category Prediction (on S0-forwarded items)
    # =========================================================
    print("\n" + "=" * 70)
    print("STAGE 1: CATEGORY PREDICTION (on S0-confident items)")
    print("=" * 70)
    print(f"  Training on: {len(train_df)} items (all)")
    print(f"  Testing on:  {test_s1_mask.sum()} S0-confident items")

    if test_s1_mask.sum() == 0:
        print("  No items forwarded from S0, skipping S1.")
        return results

    s1_data = prepare_stage_1_data(
        train_df, test_df, numeric_features, cat_features,
        top_n_categories=10,
        test_mask=test_s1_mask,
    )

    majority_class_s1 = int(np.argmax(np.bincount(s1_data['train_y'])))
    majority_acc_s1 = (s1_data['test_y'] == majority_class_s1).mean()
    majority_name_s1 = s1_data['cat_classes'].get(majority_class_s1, str(majority_class_s1))
    print(f"Majority baseline (S1): {majority_acc_s1:.1%} (class {majority_name_s1})")
    print(f"Number of categories: {len(s1_data['cat_classes'])}")

    s1_stage = ConfidenceStage(
        name='S1_category',
        classes=s1_data['cat_classes'],
        target_accuracy=min(target_accuracy, 0.70),
    )
    s1_stage.fit(s1_data['train_X'], s1_data['train_y'],
                 feature_names=s1_data['feature_cols'])
    s1_preds = s1_stage.predict(s1_data['test_X'])
    s1_curve = s1_stage.coverage_accuracy_curve(
        s1_data['test_X'], s1_data['test_y'])

    s1_conf = np.asarray(s1_preds['is_confident'], dtype=bool)
    s1_class = s1_preds['class']
    s1_confidence = s1_preds['confidence']

    if s1_conf.any():
        s1_acc = (s1_data['test_y'][s1_conf] == s1_class[s1_conf]).mean()
        s1_cov = s1_conf.mean()
        s1_lift = s1_acc - majority_acc_s1
        print(f"\nS1 Results: {s1_acc:.1%} accuracy, {s1_cov:.1%} coverage "
              f"(lift: {s1_lift:+.1%})")
    else:
        s1_acc = s1_cov = s1_lift = 0.0

    s1_deferred_count = (~s1_conf).sum()
    print(f"\nS1 deferred: {s1_deferred_count} items")
    print(f"S1 confident (forwarded to S2): {s1_conf.sum()} items")

    results['stages']['S1_category'] = {
        'accuracy': float(s1_acc),
        'coverage': float(s1_cov),
        'majority_baseline': float(majority_acc_s1),
        'accuracy_lift': float(s1_lift),
        'n_classes': len(s1_data['cat_classes']),
        'n_forwarded': int(s1_conf.sum()),
        'n_deferred': int(s1_deferred_count),
    }

    # --- Attach S1 results for S2 ---
    s1_test_df = s1_data['test_df']
    test_df_s2 = test_df.copy()
    test_df_s2['s1_pred'] = np.nan
    test_df_s2['s1_confidence'] = np.nan
    test_df_s2.loc[s1_test_df.index, 's1_pred'] = s1_class
    test_df_s2.loc[s1_test_df.index, 's1_confidence'] = s1_confidence

    # S2 mask: only S1-confident items
    test_s2_indices = s1_test_df.index[s1_conf]
    test_s2_mask = pd.Series(test_df_s2.index.isin(test_s2_indices), index=test_df_s2.index)

    # =========================================================
    # STAGE 2: Team Routing (on S1-forwarded items)
    # =========================================================
    print("\n" + "=" * 70)
    print("STAGE 2: TEAM ROUTING (on S1-confident items)")
    print("=" * 70)
    print(f"  Training on: {len(train_df)} items (all)")
    print(f"  Testing on:  {test_s2_mask.sum()} S1-confident items")

    if test_s2_mask.sum() == 0:
        print("  No items forwarded from S1, skipping S2.")
    else:
        s2_data = prepare_stage_2_data(
            train_df, test_df_s2, numeric_features, cat_features,
            data['top_groups'],
            test_mask=test_s2_mask,
        )

        majority_class_s2 = int(np.argmax(np.bincount(s2_data['train_y'])))
        majority_acc_s2 = (s2_data['test_y'] == majority_class_s2).mean()
        majority_name_s2 = s2_data['group_classes'].get(majority_class_s2, str(majority_class_s2))
        print(f"Majority baseline (S2): {majority_acc_s2:.1%} (class {majority_name_s2})")
        print(f"Number of groups: {len(s2_data['group_classes'])}")

        s2_stage = ConfidenceStage(
            name='S2_team_routing',
            classes=s2_data['group_classes'],
            target_accuracy=min(target_accuracy, 0.70),
        )
        s2_stage.fit(s2_data['train_X'], s2_data['train_y'],
                     feature_names=s2_data['feature_cols'])
        s2_preds = s2_stage.predict(s2_data['test_X'])
        s2_curve = s2_stage.coverage_accuracy_curve(
            s2_data['test_X'], s2_data['test_y'])

        s2_conf = np.asarray(s2_preds['is_confident'], dtype=bool)
        s2_class = s2_preds['class']

        if s2_conf.any():
            s2_acc = (s2_data['test_y'][s2_conf] == s2_class[s2_conf]).mean()
            s2_cov = s2_conf.mean()
            s2_lift = s2_acc - majority_acc_s2
            print(f"\nS2 Results: {s2_acc:.1%} accuracy, {s2_cov:.1%} coverage "
                  f"(lift: {s2_lift:+.1%})")
        else:
            s2_acc = s2_cov = s2_lift = 0.0

        results['stages']['S2_team_routing'] = {
            'accuracy': float(s2_acc),
            'coverage': float(s2_cov),
            'majority_baseline': float(majority_acc_s2),
            'accuracy_lift': float(s2_lift),
            'n_classes': len(s2_data['group_classes']),
            'n_forwarded': int(s2_conf.sum()),
            'n_deferred': int((~s2_conf).sum()),
        }

    # =========================================================
    # END-TO-END SEQUENTIAL CASCADE EVALUATION
    # =========================================================
    print("\n" + "=" * 70)
    print("END-TO-END SEQUENTIAL CASCADE EVALUATION")
    print("=" * 70)

    # Build full-pipeline decision array for all test items
    # Fully automated = S0 confident + S1 confident + S2 confident
    decision = np.full(n_test, 'deferred', dtype=object)
    correct = np.zeros(n_test, dtype=bool)

    # S0 confident items get priority assigned
    decision[s0_conf] = 's0_only'

    # S1 confident items (subset of S0 confident) get category assigned
    s1_test_global_idx = np.where(test_s1_mask.values)[0]
    s1_confident_global = s1_test_global_idx[s1_conf]
    decision[s1_confident_global] = 's1_only'

    # S2 confident items (subset of S1 confident) are fully automated
    if test_s2_mask.sum() > 0 and 'S2_team_routing' in results['stages']:
        s2_test_global_idx = np.where(test_s2_mask.values)[0]
        s2_confident_global = s2_test_global_idx[s2_conf]
        decision[s2_confident_global] = 'fully_automated'

        # Correctness: all 3 stages must be correct
        # S0 correctness for fully automated items
        s0_correct_full = (s0_data['test_y'][s2_confident_global] ==
                           s0_class[s2_confident_global])
        # S2 correctness (S1 correctness implied by reaching S2)
        s2_correct_local = (s2_data['test_y'][s2_conf] == s2_class[s2_conf])
        # S1 correctness for the S2-confident items
        # s2_conf indexes into S1-confident items, need S1 correctness there
        s1_correct_for_s2 = (s1_data['test_y'][s1_conf] == s1_class[s1_conf])
        # s2_conf is within the S1-confident set
        s1_correct_at_s2 = s1_correct_for_s2[s2_conf]

        correct[s2_confident_global] = s0_correct_full & s1_correct_at_s2 & s2_correct_local

    n_deferred = (decision == 'deferred').sum()
    n_s0_only = (decision == 's0_only').sum()
    n_s1_only = (decision == 's1_only').sum()
    n_fully_automated = (decision == 'fully_automated').sum()
    n_any_automated = n_s0_only + n_s1_only + n_fully_automated

    # Partial automation: items that got at least priority (S0)
    partial_auto = (decision != 'deferred')
    partial_acc = (s0_data['test_y'][partial_auto] ==
                   s0_class[partial_auto]).mean() if partial_auto.any() else 0.0

    # Full automation: all 3 stages confident
    full_auto_rate = n_fully_automated / n_test
    full_auto_acc = correct[decision == 'fully_automated'].mean() if n_fully_automated > 0 else 0.0

    print(f"\nTotal test items: {n_test}")
    print(f"  S0 deferred (no priority):    {n_deferred} ({n_deferred/n_test:.1%})")
    print(f"  S0 only (priority, no cat):   {n_s0_only} ({n_s0_only/n_test:.1%})")
    print(f"  S0+S1 (priority+cat, no team):{n_s1_only} ({n_s1_only/n_test:.1%})")
    print(f"  Fully automated (all 3):      {n_fully_automated} ({n_fully_automated/n_test:.1%})")
    print(f"\n  Partial automation rate (S0+): {partial_auto.mean():.1%}")
    print(f"  Partial accuracy (S0 priority): {partial_acc:.1%}")
    print(f"  Full automation rate (all 3):  {full_auto_rate:.1%}")
    print(f"  Full automation accuracy:      {full_auto_acc:.1%}")

    results['end_to_end'] = {
        'n_deferred': int(n_deferred),
        'n_s0_only': int(n_s0_only),
        'n_s1_only': int(n_s1_only),
        'n_fully_automated': int(n_fully_automated),
        'partial_automation_rate': float(partial_auto.mean()),
        'partial_accuracy': float(partial_acc),
        'full_automation_rate': float(full_auto_rate),
        'full_automation_accuracy': float(full_auto_acc),
    }

    # =========================================================
    # Coverage-accuracy curves
    # =========================================================
    print("\n" + "=" * 70)
    print("COVERAGE-ACCURACY CURVES")
    print("=" * 70)

    print("\nS0 Priority curve:")
    for _, row in s0_curve.iterrows():
        print(f"  t={row['threshold']:.2f}: {row['accuracy']:.1%} acc, "
              f"{row['coverage']:.1%} cov")

    if s1_conf.any():
        print("\nS1 Category curve (on S0-forwarded items):")
        for _, row in s1_curve.iterrows():
            print(f"  t={row['threshold']:.2f}: {row['accuracy']:.1%} acc, "
                  f"{row['coverage']:.1%} cov")

    if test_s2_mask.sum() > 0 and 'S2_team_routing' in results['stages']:
        print("\nS2 Team Routing curve (on S1-forwarded items):")
        for _, row in s2_curve.iterrows():
            print(f"  t={row['threshold']:.2f}: {row['accuracy']:.1%} acc, "
                  f"{row['coverage']:.1%} cov")

    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 70)
    print("SERVICENOW CASCADE SUMMARY")
    print("=" * 70)
    for stage_name, stage_res in results['stages'].items():
        print(f"  {stage_name}: {stage_res['accuracy']:.1%} acc, "
              f"{stage_res['coverage']:.1%} cov, "
              f"lift={stage_res['accuracy_lift']:+.1%}")
    if 'end_to_end' in results:
        e2e = results['end_to_end']
        print(f"\n  End-to-end: {e2e['full_automation_accuracy']:.1%} accuracy, "
              f"{e2e['full_automation_rate']:.1%} full automation rate")
        print(f"  Partial:    {e2e['partial_accuracy']:.1%} accuracy, "
              f"{e2e['partial_automation_rate']:.1%} partial automation rate")

    # Save results
    if save_results:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        results_path = OUTPUT_DIR / f'servicenow_results_{datetime.now():%Y%m%d_%H%M%S}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")

        s0_curve.to_csv(OUTPUT_DIR / 'S0_priority_curve.csv', index=False)
        if s1_conf.any():
            s1_curve.to_csv(OUTPUT_DIR / 'S1_category_curve.csv', index=False)
        if test_s2_mask.sum() > 0 and 'S2_team_routing' in results['stages']:
            s2_curve.to_csv(OUTPUT_DIR / 'S2_team_routing_curve.csv', index=False)

    return results


if __name__ == '__main__':
    run_servicenow_cascade()
