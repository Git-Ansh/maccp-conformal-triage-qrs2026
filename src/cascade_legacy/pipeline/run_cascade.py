"""
Main entry point for the Cascading Confidence-Gated Triage System.
Loads data, trains all stages, runs the cascade, evaluates, and saves results.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
from common.data_paths import RANDOM_SEED, PROJECT_ROOT

from cascade.data.loader import prepare_cascade_data
from cascade.pipeline.cascade import CascadePipeline
from cascade.evaluation.selective import coverage_accuracy_curve

# Output directory
OUTPUT_DIR = PROJECT_ROOT / 'cascade_outputs'


def run_full_cascade():
    """Run the complete cascade pipeline: train, predict, evaluate."""

    # Prepare data
    data = prepare_cascade_data()

    # Initialize and train pipeline
    pipeline = CascadePipeline()
    pipeline.train(
        data['train_summaries'],
        data['train_alerts'],
        calibration_method='isotonic'
    )

    # Run inference on test set
    test_summaries, test_alerts = pipeline.predict(
        data['test_summaries'],
        data['test_alerts']
    )

    # Evaluate
    results = pipeline.evaluate(test_summaries, test_alerts)

    # Coverage-accuracy analysis
    print("\n" + "=" * 70)
    print("COVERAGE-ACCURACY ANALYSIS")
    print("=" * 70)

    if 's0_confidence' in test_summaries.columns:
        # Overall system coverage-accuracy at different thresholds
        # Use group_auto_status predictions vs true status
        auto_mask = test_summaries['group_is_automated']
        if auto_mask.sum() > 0:
            from cascade.stages.stage_1_disposition import STATUS_MERGE
            true = test_summaries['alert_summary_status'].replace(STATUS_MERGE).values
            pred = test_summaries['group_auto_status'].values
            conf = np.where(
                test_summaries['s1_confidence'].notna(),
                test_summaries['s1_confidence'].fillna(0),
                test_summaries['s0_confidence'].fillna(0)
            )

            # Manual curve computation (since we have heterogeneous confidence sources)
            thresholds = np.arange(0.40, 0.96, 0.05)
            print(f"\n  {'Threshold':>10} {'Coverage':>10} {'Accuracy':>10} {'Automated':>10}")
            print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

            for t in thresholds:
                mask = conf >= t
                n_auto = mask.sum()
                coverage = n_auto / len(true)
                if n_auto > 0:
                    acc = (true[mask] == pred[mask]).mean()
                    print(f"  {t:>10.2f} {coverage:>10.1%} {acc:>10.4f} {n_auto:>10}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save summary predictions
    summary_output = test_summaries[[
        'alert_summary_id', 'alert_summary_status', 'has_bug',
        's0_pred', 's0_confidence', 's0_is_confident',
    ]].copy()
    for col in ['s1_pred', 's1_confidence', 's1_is_confident',
                'group_auto_status', 'group_is_automated',
                's3_pred', 's3_confidence', 's3_is_confident', 's3_source']:
        if col in test_summaries.columns:
            summary_output[col] = test_summaries[col]

    summary_output.to_csv(OUTPUT_DIR / 'cascade_summary_predictions.csv', index=False)

    # Save alert predictions
    alert_cols = ['single_alert_id', 'alert_summary_id', 'single_alert_status']
    for col in ['alert_pred_status', 'alert_is_automated']:
        if col in test_alerts.columns:
            alert_cols.append(col)
    test_alerts[alert_cols].to_csv(OUTPUT_DIR / 'cascade_alert_predictions.csv', index=False)

    # Save metrics
    serializable_results = {}
    for key, val in results.items():
        if isinstance(val, dict):
            serializable_results[key] = {
                k: float(v) if isinstance(v, (np.floating, float)) else
                int(v) if isinstance(v, (np.integer, int)) else
                bool(v) if isinstance(v, (np.bool_, bool)) else str(v)
                for k, v in val.items()
            }

    serializable_results['timestamp'] = datetime.now().isoformat()
    serializable_results['train_size'] = len(data['train_summaries'])
    serializable_results['test_size'] = len(data['test_summaries'])

    with open(OUTPUT_DIR / 'cascade_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    return results, pipeline, test_summaries, test_alerts


if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    results, pipeline, summaries, alerts = run_full_cascade()
