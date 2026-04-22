"""
Demo: Applying the General Cascade Framework to Mozilla Perfherder.

Shows that the entire cascade can be configured declaratively --
no dataset-specific code needed beyond defining:
1. Stage configurations (classes, features, thresholds)
2. Label merge rules
3. Routing between stages

This same framework can be applied to ANY CI triage system by changing
only the configuration. The calibration, threshold tuning, and routing
logic are handled automatically.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
from common.data_paths import RANDOM_SEED

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from cascade.data.loader import prepare_cascade_data
from cascade.framework.cascade_pipeline import GeneralCascade, StageConfig


# ============================================================
# CONFIGURATION ONLY -- no dataset-specific code below
# ============================================================

# Features shared across stages (summary-level)
SUMMARY_FEATURES = [
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


def _build_binary_model():
    """Build ensemble model for binary classification."""
    estimators = []
    if HAS_XGBOOST:
        estimators.append(('xgb', XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_SEED, eval_metric='logloss',
            use_label_encoder=False, n_jobs=-1,
        )))
    estimators.append(('rf', RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=3,
        class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1,
    )))
    if len(estimators) == 1:
        return estimators[0][1]
    return VotingClassifier(estimators=estimators, voting='soft')


def _build_multiclass_model():
    """Build ensemble model for multiclass classification."""
    estimators = []
    if HAS_XGBOOST:
        estimators.append(('xgb', XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_SEED, eval_metric='mlogloss',
            use_label_encoder=False, n_jobs=-1, objective='multi:softprob',
        )))
    estimators.append(('rf', RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=3,
        class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1,
    )))
    if len(estimators) == 1:
        return estimators[0][1]
    return VotingClassifier(estimators=estimators, voting='soft')


# Stage 0: Invalid vs Valid (binary at summary level)
STAGE_0 = StageConfig(
    name='invalid_filter',
    classes={0: 'Valid', 1: 'Invalid'},
    target_accuracy=0.85,
    feature_columns=SUMMARY_FEATURES,
    target_column='is_invalid',
    routing={
        1: 'terminal',   # Confident Invalid -> done
        0: 'next',       # Confident Valid -> Stage 1
        -1: 'defer',     # Uncertain -> human review
    },
    model=_build_binary_model(),
    output_prefix='s0',
)

# Stage 1: Disposition (4-class at summary level, non-Invalid only)
STAGE_1 = StageConfig(
    name='disposition',
    classes={4: 'Actionable', 6: 'Wontfix', 7: 'Fixed', 1: 'Downstream'},
    target_accuracy=0.80,
    feature_columns=SUMMARY_FEATURES + ['has_subtests_ratio', 'lower_is_better_ratio'],
    target_column='disposition',
    label_merge={8: 4, 2: 4},  # Backedout, Reassigned -> Actionable
    input_filter=lambda df: df[df.get('alert_summary_status', df.get('status', pd.Series())) != 3],
    routing={
        4: 'terminal',   # Confident Actionable -> done
        6: 'terminal',   # Confident Wontfix -> done
        7: 'terminal',   # Confident Fixed -> done
        1: 'terminal',   # Confident Downstream -> done
        -1: 'defer',     # Uncertain -> human review
    },
    model=_build_multiclass_model(),
    output_prefix='s1',
)

# Stage 3: Bug linkage (binary at summary level)
# has_bug at summary level: 0 = no bug, 1 = has bug
STAGE_3 = StageConfig(
    name='bug_linkage',
    classes={0: 'No Bug', 1: 'Has Bug'},
    target_accuracy=0.85,
    feature_columns=SUMMARY_FEATURES,
    target_column='has_bug',
    routing={
        0: 'terminal',
        1: 'terminal',
        -1: 'defer',
    },
    model=_build_binary_model(),
    output_prefix='s3',
)


def run_demo():
    """Run the general cascade on Mozilla Perfherder data."""
    # Load data
    data = prepare_cascade_data()
    train = data['train_summaries'].copy()
    test = data['test_summaries'].copy()

    # Prepare derived columns needed by the config
    for df in [train, test]:
        df['is_invalid'] = (df['alert_summary_status'] == 3).astype(int)
        df['disposition'] = df['alert_summary_status']

    # Create and train cascade
    cascade = GeneralCascade(
        stages=[STAGE_0, STAGE_1, STAGE_3],
        random_state=RANDOM_SEED,
    )
    cascade.fit(train, calibration_method='isotonic', n_cv_folds=5)

    # Predict on test set
    predictions = cascade.predict(test)

    # Evaluate
    eval_results = cascade.evaluate(
        test, predictions,
        true_label_column='alert_summary_status',
        label_merge={8: 4, 2: 4},
    )
    cascade.print_evaluation(eval_results)

    # Per-stage coverage-accuracy curves
    print("\n" + "=" * 70)
    print("PER-STAGE COVERAGE-ACCURACY CURVES")
    print("=" * 70)

    # Stage 0 curve
    stage0 = cascade.get_stage('invalid_filter')
    if stage0 is not None:
        test_s0 = test.copy()
        X_s0 = test_s0[SUMMARY_FEATURES].fillna(0).values
        y_s0 = test_s0['is_invalid'].values
        curve_s0 = stage0.coverage_accuracy_curve(X_s0, y_s0)
        print(f"\nStage 0 (Invalid Filter):")
        for _, row in curve_s0.iterrows():
            print(f"  t={row['threshold']:.2f}: {row['accuracy']:.1%} acc, "
                  f"{row['coverage']:.1%} cov ({row['n_predicted']} predicted)")

    # Stage 1 curve (on non-Invalid)
    stage1 = cascade.get_stage('disposition')
    if stage1 is not None:
        test_s1 = test[test['alert_summary_status'] != 3].copy()
        test_s1['disposition'] = test_s1['alert_summary_status'].replace({8: 4, 2: 4})
        feature_cols_s1 = SUMMARY_FEATURES + ['has_subtests_ratio', 'lower_is_better_ratio']
        available = [c for c in feature_cols_s1 if c in test_s1.columns]
        X_s1 = test_s1[available].fillna(0).values
        y_s1 = test_s1['disposition'].values
        curve_s1 = stage1.coverage_accuracy_curve(X_s1, y_s1)
        print(f"\nStage 1 (Disposition):")
        for _, row in curve_s1.iterrows():
            print(f"  t={row['threshold']:.2f}: {row['accuracy']:.1%} acc, "
                  f"{row['coverage']:.1%} cov ({row['n_predicted']} predicted)")

    # Stage 3 curve
    stage3 = cascade.get_stage('bug_linkage')
    if stage3 is not None:
        X_s3 = test[SUMMARY_FEATURES].fillna(0).values
        y_s3 = test['has_bug'].values
        curve_s3 = stage3.coverage_accuracy_curve(X_s3, y_s3)
        print(f"\nStage 3 (Bug Linkage):")
        for _, row in curve_s3.iterrows():
            print(f"  t={row['threshold']:.2f}: {row['accuracy']:.1%} acc, "
                  f"{row['coverage']:.1%} cov ({row['n_predicted']} predicted)")

    # Summary
    print("\n" + "=" * 70)
    print("FRAMEWORK GENERALIZABILITY")
    print("=" * 70)
    print("""
    This cascade was configured entirely through StageConfig objects.
    To apply to a different CI system:

    1. Define your stages:
       - What classes does each stage predict?
       - What accuracy threshold do you target?
       - What features are available?

    2. Define routing rules:
       - Which predictions are terminal (final decisions)?
       - Which pass to the next stage?
       - Which get deferred to humans?

    3. Provide labeled data and call cascade.fit()

    The framework handles:
       - Probability calibration (isotonic regression)
       - OOF threshold tuning (no in-sample leakage)
       - Per-class confidence gates
       - Coverage-accuracy tradeoff curves
       - Automatic routing between stages
    """)

    return cascade, predictions, eval_results


if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    cascade, predictions, results = run_demo()
