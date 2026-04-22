"""
Eclipse Bug Sequential Cascade Pipeline (Zenodo 2024).

Runs a 3-stage SEQUENTIAL cascade on Eclipse Zenodo 2024 bug data (304K bugs, 9 projects):
  S0: Noise gate -- report P(Noise) flag metrics for future LLM rescue
      Noise is filtered manually (ground truth) for S1/S2 evaluation.

  S1: Severity prediction (7 classes) on non-noise items
      -> confident: forward to S2 (with severity as feature)
      -> uncertain: defer to human

  S2: Component assignment (top-30 + Other) on S1-forwarded items
      -> confident: terminal (auto-assign)
      -> uncertain: defer to human

End-to-end metrics: combined S1+S2 performance on non-noise items.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from cascade.framework.confidence_stage import ConfidenceStage
from conformal.data.eclipse_zenodo_loader import prepare_eclipse_zenodo_data
from conformal.stages.eclipse_config import (
    prepare_stage_0_data, prepare_stage_1_data, prepare_stage_2_data,
    SEVERITY_CLASSES,
)

OUTPUT_DIR = PROJECT_ROOT / 'conformal_outputs' / 'eclipse'


def _fit_predict(stage, train_X, train_y, test_X, feature_cols,
                 train_text=None, test_text=None, use_text=True):
    """Fit a stage and return predictions."""
    text_kw_fit = {'text_data': train_text} if use_text and train_text is not None else {}
    text_kw_pred = {'text_data': test_text} if use_text and test_text is not None else {}

    stage.fit(train_X, train_y, feature_names=feature_cols, **text_kw_fit)
    preds = stage.predict(test_X, **text_kw_pred)
    return preds


def run_eclipse_cascade(
    use_text: bool = True,
    target_accuracy: float = 0.85,
    save_results: bool = True,
    max_per_project: int = 20000,
):
    """Run full Eclipse sequential cascade pipeline on Zenodo 2024 data."""
    print("=" * 70)
    print("ECLIPSE BUG TRIAGE CASCADE (Zenodo 2024, SEQUENTIAL)")
    print("=" * 70)
    print(f"  Text features: {use_text}")
    print(f"  Target accuracy: {target_accuracy}")
    print(f"  Max per project: {max_per_project}")
    print()

    # Load data (limit per project for memory; 20K x 9 = 180K bugs)
    data = prepare_eclipse_zenodo_data(max_per_project=max_per_project)
    train_df = data['train_df']
    test_df = data['test_df']
    numeric_features = data['numeric_features']
    cat_features = data['categorical_features']
    n_test = len(test_df)

    results = {
        'dataset': 'Eclipse Zenodo 2024',
        'n_train': len(train_df),
        'n_test': n_test,
        'n_projects': data['stats']['n_projects'],
        'use_text': use_text,
        'target_accuracy': target_accuracy,
        'stages': {},
    }

    # =========================================================
    # STAGE 0: Noise Gate (flag metrics only, no auto-close)
    # =========================================================
    print("\n" + "=" * 70)
    print("STAGE 0: NOISE GATE (flag metrics for LLM rescue)")
    print("=" * 70)

    s0_data = prepare_stage_0_data(train_df, test_df, numeric_features, cat_features)

    noise_train = s0_data['train_y'].mean()
    noise_test = s0_data['test_y'].mean()
    total_noise = (s0_data['test_y'] == 1).sum()
    total_valid = (s0_data['test_y'] == 0).sum()
    print(f"Noise ratio: train={noise_train:.1%}, test={noise_test:.1%}")
    print(f"Test: {total_noise} noise, {total_valid} valid")

    # Train S0 model to get P(Noise) for flag metrics
    s0_stage = ConfidenceStage(
        name='S0_noise_gate',
        classes={0: 'Valid', 1: 'Noise'},
        target_accuracy=target_accuracy,
    )

    s0_preds = _fit_predict(
        s0_stage, s0_data['train_X'], s0_data['train_y'],
        s0_data['test_X'], s0_data['feature_cols'],
        s0_data['train_text'], s0_data['test_text'], use_text,
    )

    s0_conf = np.asarray(s0_preds['is_confident'], dtype=bool)
    s0_class = s0_preds['class']
    s0_confidence = s0_preds['confidence']

    # Report S0 accuracy for reference
    if s0_conf.any():
        s0_acc = (s0_data['test_y'][s0_conf] == s0_class[s0_conf]).mean()
        s0_cov = s0_conf.mean()
    else:
        s0_acc = s0_cov = 0.0
    majority_acc_s0 = (s0_data['test_y'] == 0).mean()  # majority = Valid
    print(f"\nS0 Reference: {s0_acc:.1%} acc, {s0_cov:.1%} cov "
          f"(majority baseline: {majority_acc_s0:.1%})")

    # P(Noise) flag metrics for LLM rescue
    print("\n--- P(Noise) Flag Metrics (for future LLM rescue) ---")
    # Get full probability matrix from predict
    text_kw_s0 = {'text_data': s0_data['test_text']} if use_text else {}
    s0_preds_full = s0_stage.predict(s0_data['test_X'], return_proba=True, **text_kw_s0)
    proba = s0_preds_full['proba']
    noise_idx = list(s0_stage._label_encoder.classes_).index(1)
    p_noise = proba[:, noise_idx]

    print(f"P(Noise) stats: mean={p_noise.mean():.3f}, "
          f"median={np.median(p_noise):.3f}, max={p_noise.max():.3f}")
    print(f"P(Noise) for actual noise: mean={p_noise[s0_data['test_y']==1].mean():.3f}, "
          f"median={np.median(p_noise[s0_data['test_y']==1]):.3f}")

    flag_results = {}
    for threshold in [0.08, 0.10, 0.15, 0.20, 0.30]:
        flagged = p_noise >= threshold
        flag_count = flagged.sum()
        if total_noise > 0:
            flag_recall = (s0_data['test_y'][flagged] == 1).sum() / total_noise
        else:
            flag_recall = 0.0
        flag_pct = flag_count / len(p_noise)
        print(f"  P(Noise)>={threshold:.2f}: {flag_count} flagged ({flag_pct:.1%} of items), "
              f"noise recall={flag_recall:.1%}")
        flag_results[f't_{threshold}'] = {
            'flagged': int(flag_count), 'pct': float(flag_pct),
            'recall': float(flag_recall),
        }

    print("\n  NOTE: S0 noise is semantically complex. Noise items flagged above "
          "will be sent to LLM for classification in a future step.")

    results['stages']['S0_noise_gate'] = {
        'accuracy': float(s0_acc),
        'coverage': float(s0_cov),
        'majority_baseline': float(majority_acc_s0),
        'noise_ratio': float(noise_test),
        'flag_metrics': flag_results,
        'note': 'Noise filtered manually for S1/S2. LLM rescue TBD.',
    }

    # =========================================================
    # STAGE 1: Severity Prediction (on non-noise items)
    # =========================================================
    print("\n" + "=" * 70)
    print("STAGE 1: SEVERITY PREDICTION (on non-noise items)")
    print("=" * 70)

    # Filter noise manually using ground truth
    train_s1_mask = (train_df['is_noise'] == 0).values
    test_s1_mask = (test_df['is_noise'] == 0).values
    print(f"  Training on: {train_s1_mask.sum()} non-noise items")
    print(f"  Testing on:  {test_s1_mask.sum()} non-noise items (manual noise filter)")

    # Add S0 info to test_df for downstream features
    test_df = test_df.copy()
    test_df['s0_confidence'] = s0_confidence
    test_df['s0_pred'] = s0_class

    s1_data = prepare_stage_1_data(
        train_df, test_df, numeric_features, cat_features,
        train_mask=train_s1_mask, test_mask=test_s1_mask,
    )

    if len(s1_data['train_y']) == 0 or len(s1_data['test_y']) == 0:
        print("  No data for S1, skipping.")
        return results

    majority_class_s1 = int(np.argmax(np.bincount(s1_data['train_y'])))
    majority_acc_s1 = (s1_data['test_y'] == majority_class_s1).mean()
    print(f"Majority baseline (S1): {majority_acc_s1:.1%} "
          f"(class {SEVERITY_CLASSES.get(majority_class_s1, majority_class_s1)})")

    # Severity distribution
    print(f"Severity distribution (S1 test):")
    for code, name in SEVERITY_CLASSES.items():
        n = (s1_data['test_y'] == code).sum()
        if n > 0:
            print(f"  {name}: {n} ({n/len(s1_data['test_y']):.1%})")

    s1_stage = ConfidenceStage(
        name='S1_severity',
        classes=SEVERITY_CLASSES,
        target_accuracy=min(target_accuracy, 0.70),  # Lower for 7-class
    )

    s1_preds = _fit_predict(
        s1_stage, s1_data['train_X'], s1_data['train_y'],
        s1_data['test_X'], s1_data['feature_cols'],
        s1_data['train_text'], s1_data['test_text'], use_text,
    )

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

    # Per-class recall
    print("S1 per-class recall (confident predictions):")
    for code, name in SEVERITY_CLASSES.items():
        actual = (s1_data['test_y'] == code).sum()
        if actual > 0:
            pred_correct = (s1_conf & (s1_class == code) & (s1_data['test_y'] == code)).sum()
            recall = pred_correct / actual
            print(f"  {name}: {recall:.1%} ({pred_correct}/{actual})")

    # Prediction diversity
    unique_preds = np.unique(s1_class[s1_conf]) if s1_conf.any() else []
    print(f"  Unique predictions: {len(unique_preds)} / {len(SEVERITY_CLASSES)} classes")

    s1_deferred_count = (~s1_conf).sum()
    print(f"\nS1 deferred: {s1_deferred_count} items")
    print(f"S1 confident (forwarded to S2): {s1_conf.sum()} items")

    results['stages']['S1_severity'] = {
        'accuracy': float(s1_acc),
        'coverage': float(s1_cov),
        'majority_baseline': float(majority_acc_s1),
        'accuracy_lift': float(s1_lift),
        'n_forwarded': int(s1_conf.sum()),
        'n_deferred': int(s1_deferred_count),
        'unique_classes_predicted': int(len(unique_preds)),
    }

    # --- Attach S1 results for S2 ---
    s1_test_df = s1_data['test_df']
    test_df_s2 = test_df.copy()
    test_df_s2['s1_pred'] = np.nan
    test_df_s2['s1_confidence'] = np.nan
    test_df_s2.loc[s1_test_df.index, 's1_pred'] = s1_class
    test_df_s2.loc[s1_test_df.index, 's1_confidence'] = s1_confidence

    # S2 mask: only S1-confident non-noise items
    test_s2_indices = s1_test_df.index[s1_conf]
    test_s2_mask = test_df_s2.index.isin(test_s2_indices)

    # =========================================================
    # STAGE 2: Component Assignment (on S1-forwarded items)
    # =========================================================
    print("\n" + "=" * 70)
    print("STAGE 2: COMPONENT ASSIGNMENT (on S1-confident items)")
    print("=" * 70)
    print(f"  Training on: {train_s1_mask.sum()} non-noise items")
    print(f"  Testing on:  {test_s2_mask.sum()} S1-confident items")

    if test_s2_mask.sum() == 0:
        print("  No items forwarded from S1, skipping S2.")
    else:
        s2_data = prepare_stage_2_data(
            train_df, test_df_s2, numeric_features, cat_features,
            data['top_components'],
            train_mask=train_s1_mask, test_mask=test_s2_mask,
        )

        majority_class_s2 = int(np.argmax(np.bincount(s2_data['train_y'])))
        majority_acc_s2 = (s2_data['test_y'] == majority_class_s2).mean()
        majority_name_s2 = s2_data['component_classes'].get(majority_class_s2, str(majority_class_s2))
        print(f"Majority baseline (S2): {majority_acc_s2:.1%} (class {majority_name_s2})")
        print(f"Number of classes: {len(s2_data['component_classes'])}")

        s2_stage = ConfidenceStage(
            name='S2_component',
            classes=s2_data['component_classes'],
            target_accuracy=min(target_accuracy, 0.70),  # Lower for many classes
        )

        s2_preds = _fit_predict(
            s2_stage, s2_data['train_X'], s2_data['train_y'],
            s2_data['test_X'], s2_data['feature_cols'],
            s2_data['train_text'], s2_data['test_text'], use_text,
        )

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

        # Per-class recall for top components
        print("S2 per-class recall (top-5 components):")
        top5_comps = sorted(
            [(code, name, (s2_data['test_y'] == code).sum())
             for code, name in s2_data['component_classes'].items()],
            key=lambda x: -x[2]
        )[:5]
        for code, name, actual in top5_comps:
            if actual > 0:
                pred_correct = (s2_conf & (s2_class == code) & (s2_data['test_y'] == code)).sum()
                recall = pred_correct / actual
                print(f"  {name}: {recall:.1%} ({pred_correct}/{actual})")

        results['stages']['S2_component'] = {
            'accuracy': float(s2_acc),
            'coverage': float(s2_cov),
            'majority_baseline': float(majority_acc_s2),
            'accuracy_lift': float(s2_lift),
            'n_classes': len(s2_data['component_classes']),
        }

    # =========================================================
    # END-TO-END EVALUATION (S1 + S2 on non-noise items)
    # =========================================================
    print("\n" + "=" * 70)
    print("END-TO-END EVALUATION (non-noise items, S1 -> S2)")
    print("=" * 70)

    n_non_noise_test = test_s1_mask.sum()
    print(f"\nNon-noise test items: {n_non_noise_test}")

    # S1 confident items
    n_s1_confident = s1_conf.sum()
    n_s1_deferred = (~s1_conf).sum()

    # S2 results (subset of S1-confident)
    if test_s2_mask.sum() > 0 and 'S2_component' in results['stages']:
        n_s2_confident = s2_conf.sum()
        n_s2_deferred = (~s2_conf).sum()
        s2_correct = (s2_data['test_y'][s2_conf] == s2_class[s2_conf]).sum() if s2_conf.any() else 0

        # Fully automated = S1 confident AND S2 confident (both severity + component assigned)
        n_fully_automated = n_s2_confident
        n_partial = n_s1_confident - n_s2_confident  # severity only
        n_deferred = n_s1_deferred  # no severity prediction

        e2e_full_rate = n_fully_automated / n_non_noise_test
        e2e_partial_rate = n_s1_confident / n_non_noise_test

        # Accuracy of fully automated items
        # S1 accuracy on S2-confident items
        s1_on_s2_conf = (s1_data['test_y'][s1_conf][s2_conf] == s1_class[s1_conf][s2_conf])
        s1_s2_combined_acc = (s1_on_s2_conf & (s2_data['test_y'][s2_conf] == s2_class[s2_conf])).mean()

        # S1-only accuracy (partial automation)
        s1_only_acc = (s1_data['test_y'][s1_conf] == s1_class[s1_conf]).mean()

        print(f"  Fully automated (S1+S2 confident): {n_fully_automated} ({e2e_full_rate:.1%})")
        print(f"  Partial (S1 only):                 {n_partial}")
        print(f"  Deferred (S1 uncertain):           {n_deferred} ({n_deferred/n_non_noise_test:.1%})")
        print(f"\n  S1 accuracy (partial): {s1_only_acc:.1%}")
        print(f"  S1+S2 combined accuracy (full): {s1_s2_combined_acc:.1%}")
        print(f"  S1 coverage: {e2e_partial_rate:.1%}")
        print(f"  S1+S2 coverage (full): {e2e_full_rate:.1%}")

        results['end_to_end'] = {
            'n_non_noise_test': int(n_non_noise_test),
            'n_fully_automated': int(n_fully_automated),
            'full_automation_rate': float(e2e_full_rate),
            'partial_automation_rate': float(e2e_partial_rate),
            's1_accuracy': float(s1_only_acc),
            's1_s2_combined_accuracy': float(s1_s2_combined_acc),
        }
    else:
        print(f"  S1 confident: {n_s1_confident} ({n_s1_confident/n_non_noise_test:.1%})")
        print(f"  S1 deferred: {n_s1_deferred}")
        results['end_to_end'] = {
            'n_non_noise_test': int(n_non_noise_test),
            's1_coverage': float(n_s1_confident / n_non_noise_test),
        }

    # =========================================================
    # Flat baseline comparison
    # =========================================================
    print("\n" + "=" * 70)
    print("FLAT BASELINE COMPARISON")
    print("=" * 70)

    _print_flat_comparison(s0_data, s1_data,
                           s2_data if test_s2_mask.sum() > 0 and 'S2_component' in results['stages'] else None,
                           use_text)

    # =========================================================
    # Coverage-accuracy curves
    # =========================================================
    print("\n" + "=" * 70)
    print("COVERAGE-ACCURACY CURVES")
    print("=" * 70)

    text_kw_s1 = {}
    if use_text:
        text_kw_s1['text_data'] = s1_data['test_text']
    s1_curve = s1_stage.coverage_accuracy_curve(
        s1_data['test_X'], s1_data['test_y'], **text_kw_s1)
    if len(s1_curve) > 0:
        print("S1 curve (severity):")
        for _, row in s1_curve.iterrows():
            print(f"  t={row['threshold']:.2f}: {row['accuracy']:.1%} acc, "
                  f"{row['coverage']:.1%} cov")

    if test_s2_mask.sum() > 0 and 'S2_component' in results['stages']:
        text_kw_s2 = {}
        if use_text:
            text_kw_s2['text_data'] = s2_data['test_text']
        s2_curve = s2_stage.coverage_accuracy_curve(
            s2_data['test_X'], s2_data['test_y'], **text_kw_s2)
        if len(s2_curve) > 0:
            print("\nS2 curve (component):")
            for _, row in s2_curve.iterrows():
                print(f"  t={row['threshold']:.2f}: {row['accuracy']:.1%} acc, "
                      f"{row['coverage']:.1%} cov")

    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 70)
    print("ECLIPSE CASCADE SUMMARY")
    print("=" * 70)
    for stage_name, stage_res in results['stages'].items():
        if 'accuracy' in stage_res:
            cov = stage_res.get('coverage', 0)
            lift = stage_res.get('accuracy_lift', 0)
            print(f"  {stage_name}: {stage_res['accuracy']:.1%} acc, "
                  f"{cov:.1%} cov, lift={lift:+.1%}")
    if 'end_to_end' in results:
        e2e = results['end_to_end']
        if 'full_automation_rate' in e2e:
            print(f"\n  End-to-end (non-noise): "
                  f"{e2e['s1_s2_combined_accuracy']:.1%} acc, "
                  f"{e2e['full_automation_rate']:.1%} full automation")
            print(f"  S1 partial: {e2e['s1_accuracy']:.1%} acc, "
                  f"{e2e['partial_automation_rate']:.1%} coverage")

    # Save results
    if save_results:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        results_path = OUTPUT_DIR / f'eclipse_results_{datetime.now():%Y%m%d_%H%M%S}.json'

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")

        if len(s1_curve) > 0:
            s1_curve.to_csv(OUTPUT_DIR / 'S1_severity_curve.csv', index=False)

    return results


def _build_flat_model(eval_metric='logloss'):
    """Build a flat XGBoost model with GPU if available."""
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
        return XGBClassifier(n_estimators=200, max_depth=6, random_state=42,
                             eval_metric=eval_metric, n_jobs=-1, **gpu_params)
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, max_depth=10,
                                      class_weight='balanced', random_state=42)


def _print_flat_comparison(s0_data, s1_data, s2_data, use_text):
    """Compare cascade vs flat classifier at matched coverage."""
    from sklearn.metrics import accuracy_score

    flat_s1 = _build_flat_model('mlogloss')
    flat_s1.fit(s1_data['train_X'], s1_data['train_y'])
    flat_s1_pred = flat_s1.predict(s1_data['test_X'])
    flat_s1_acc = accuracy_score(s1_data['test_y'], flat_s1_pred)
    print(f"  Flat S1: {flat_s1_acc:.1%} accuracy (100% coverage, no gating)")

    if s2_data is not None:
        flat_s2 = _build_flat_model('mlogloss')
        flat_s2.fit(s2_data['train_X'], s2_data['train_y'])
        flat_s2_pred = flat_s2.predict(s2_data['test_X'])
        flat_s2_acc = accuracy_score(s2_data['test_y'], flat_s2_pred)
        print(f"  Flat S2: {flat_s2_acc:.1%} accuracy (100% coverage, no gating)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Eclipse Bug Cascade Pipeline')
    parser.add_argument('--no-text', action='store_true', help='Disable text features')
    parser.add_argument('--target-accuracy', type=float, default=0.85)
    parser.add_argument('--max-per-project', type=int, default=20000,
                        help='Max bugs per project (default 20000, use 0 for no limit)')
    args = parser.parse_args()

    run_eclipse_cascade(
        use_text=not args.no_text,
        target_accuracy=args.target_accuracy,
        max_per_project=args.max_per_project if args.max_per_project > 0 else None,
    )
