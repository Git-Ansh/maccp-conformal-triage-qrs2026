"""
Run all external dataset cascade evaluations.

Runs Eclipse (Zenodo 2024), JM1, and ServiceNow, then generates
cross-dataset comparison figures and summary tables.
"""

import sys
import json
import traceback
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

OUTPUT_DIR = PROJECT_ROOT / 'conformal_outputs'


def run_all(skip_errors: bool = True):
    """Run all external dataset evaluations."""
    print("=" * 70)
    print("EXTERNAL DATASET CASCADE EVALUATION")
    print(f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    all_results = {}

    # Eclipse (Zenodo 2024)
    print("\n\n" + "#" * 70)
    print("# ECLIPSE BUG DATASET (Zenodo 2024)")
    print("#" * 70)
    try:
        from conformal.pipeline.run_eclipse import run_eclipse_cascade
        all_results['Eclipse'] = run_eclipse_cascade(save_results=True)
    except Exception as e:
        print(f"Eclipse FAILED: {e}")
        if not skip_errors:
            raise
        traceback.print_exc()

    # JM1
    print("\n\n" + "#" * 70)
    print("# JM1 DEFECT PREDICTION")
    print("#" * 70)
    try:
        from conformal.pipeline.run_jm1 import run_jm1_cascade
        jm1_results = run_jm1_cascade(save_results=True)
        # Reshape JM1 results to match Eclipse format
        all_results['JM1'] = {
            'dataset': 'JM1',
            'stages': {
                'S0_defect': {
                    'accuracy': jm1_results.get('cascade_accuracy', 0),
                    'coverage': jm1_results.get('cascade_coverage', 0),
                    'majority_baseline': jm1_results.get('majority_baseline', 0),
                    'accuracy_lift': jm1_results.get('accuracy_lift', 0),
                }
            }
        }
    except Exception as e:
        print(f"JM1 FAILED: {e}")
        if not skip_errors:
            raise
        traceback.print_exc()

    # ServiceNow
    print("\n\n" + "#" * 70)
    print("# SERVICENOW ITSM")
    print("#" * 70)
    try:
        from conformal.pipeline.run_servicenow import run_servicenow_cascade
        all_results['ServiceNow'] = run_servicenow_cascade(save_results=True)
    except Exception as e:
        print(f"ServiceNow FAILED: {e}")
        if not skip_errors:
            raise
        traceback.print_exc()

    # Cross-dataset comparison
    if len(all_results) > 0:
        print("\n\n" + "#" * 70)
        print("# CROSS-DATASET COMPARISON")
        print("#" * 70)

        try:
            from conformal.evaluation.comparison import (
                print_cross_dataset_summary,
                plot_accuracy_lift_comparison,
            )

            print_cross_dataset_summary(all_results)

            # Generate comparison figures
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            # Accuracy lift chart
            lift_data = {}
            for ds_name, ds_res in all_results.items():
                lift_data[ds_name] = ds_res.get('stages', {})

            plot_accuracy_lift_comparison(
                lift_data,
                output_path=OUTPUT_DIR / 'accuracy_lift_comparison.png',
            )
        except Exception as e:
            print(f"Comparison generation failed: {e}")
            traceback.print_exc()

        # Save combined results
        summary_path = OUTPUT_DIR / f'all_results_{datetime.now():%Y%m%d_%H%M%S}.json'

        # Make JSON-serializable
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()
                        if k != 'curve'}  # Skip curve data
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        import numpy as np
        with open(summary_path, 'w') as f:
            json.dump(make_serializable(all_results), f, indent=2)
        print(f"\nCombined results saved to {summary_path}")

    return all_results


if __name__ == '__main__':
    run_all()
