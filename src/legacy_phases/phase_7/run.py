#!/usr/bin/env python3
"""
Phase 7: Integration and End-to-End System - Run Script

Experiments:
E1: Integrated vs individual model comparison
E2: Cross-suite integrated evaluation
E3: Ablation of integrated components
E4: RCA usefulness analysis
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime
from typing import List
import gc

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_merger import create_unified_dataset, load_phase_outputs
from src.feature_normalizer import (
    normalize_features, select_features, prepare_ensemble_features
)
from src.stacking_ensemble import (
    StackingEnsemble, train_base_models
)
from src.evaluation import (
    evaluate_integrated_model, cross_phase_comparison,
    ablation_study, repository_evaluation, generate_comparison_report
)

from common.data_paths import (
    PHASE_7_DIR, REGRESSION_TARGET_COL, RANDOM_SEED,
    MAGNITUDE_FEATURES, CONTEXT_FEATURES
)
from common.model_utils import save_results, set_random_seeds

warnings.filterwarnings('ignore')


def run_experiment_E1(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path
) -> pd.DataFrame:
    """
    E1: Integrated vs Individual Model Comparison

    Compare stacking ensemble with individual base models.
    """
    print("\n" + "="*60)
    print("EXPERIMENT E1: Integrated vs Individual Model Comparison")
    print("="*60)

    # Train and evaluate individual base models
    print("\nTraining individual base models...")
    base_results = train_base_models(X_train, y_train, X_test, y_test)
    print("\nBase model results:")
    print(base_results.to_string(index=False))

    # Train stacking ensemble
    print("\nTraining stacking ensemble...")
    ensemble = StackingEnsemble(cv_folds=5)
    ensemble.fit(X_train, y_train)

    # Evaluate ensemble
    ensemble_metrics = ensemble.evaluate(X_test, y_test)
    print(f"\nStacking Ensemble - Test:")
    print(f"  Precision: {ensemble_metrics['precision']:.4f}")
    print(f"  Recall: {ensemble_metrics['recall']:.4f}")
    print(f"  F1: {ensemble_metrics['f1_score']:.4f}")
    print(f"  AUC: {ensemble_metrics.get('roc_auc', 0):.4f}")

    # Add ensemble to results
    ensemble_row = pd.DataFrame([{
        'model': 'StackingEnsemble',
        'precision': ensemble_metrics['precision'],
        'recall': ensemble_metrics['recall'],
        'f1_score': ensemble_metrics['f1_score'],
        'roc_auc': ensemble_metrics.get('roc_auc', 0.5)
    }])

    all_results = pd.concat([base_results, ensemble_row], ignore_index=True)
    all_results = all_results.sort_values('f1_score', ascending=False)
    all_results.to_csv(output_dir / 'reports' / 'E1_model_comparison.csv', index=False)

    # Save ensemble
    ensemble.save(output_dir / 'models' / 'stacking_ensemble.joblib')

    print("\nFinal ranking:")
    print(all_results[['model', 'f1_score', 'roc_auc']].to_string(index=False))

    return all_results, ensemble


def run_experiment_E2(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path
) -> pd.DataFrame:
    """
    E2: Cross-Suite Integrated Evaluation

    Evaluate model performance across different repositories and platforms.
    """
    print("\n" + "="*60)
    print("EXPERIMENT E2: Cross-Suite Integrated Evaluation")
    print("="*60)

    # Repository-level evaluation
    print("\nEvaluating per repository...")
    repo_results = repository_evaluation(df, X, y, 'alert_summary_repository')

    if len(repo_results) > 0:
        print("\nPer-repository results:")
        print(repo_results.to_string(index=False))
        repo_results.to_csv(output_dir / 'reports' / 'E2_cross_repo.csv', index=False)

    # Platform-level evaluation
    print("\nEvaluating per platform...")
    if 'single_alert_series_signature_machine_platform' in df.columns:
        platform_results = repository_evaluation(
            df, X, y, 'single_alert_series_signature_machine_platform'
        )
        if len(platform_results) > 0:
            platform_results.columns = ['platform', 'n_samples', 'precision', 'recall', 'f1_score']
            print("\nPer-platform results:")
            print(platform_results.to_string(index=False))
            platform_results.to_csv(output_dir / 'reports' / 'E2_cross_platform.csv', index=False)

    return repo_results


def run_experiment_E3(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    output_dir: Path
) -> pd.DataFrame:
    """
    E3: Ablation of Integrated Components

    Test contribution of different feature groups.
    """
    print("\n" + "="*60)
    print("EXPERIMENT E3: Feature Group Ablation")
    print("="*60)

    # Define feature groups to ablate
    ablation_groups = {
        'Magnitude': ['amount_abs', 'amount_pct', 't_value', 'prev_value', 'new_value'],
        'Context': ['repository', 'framework', 'suite', 'platform'],
        'TimeSeries': ['ts_'],
        'RCA': ['rca_']
    }

    print("\nRunning ablation study...")
    ablation_results = ablation_study(X, y, feature_names, ablation_groups, cv=5)

    print("\nAblation results:")
    print(ablation_results.to_string(index=False))

    ablation_results.to_csv(output_dir / 'reports' / 'E3_ablation.csv', index=False)

    # Identify most important feature groups
    print("\nFeature group importance (by F1 delta when removed):")
    for _, row in ablation_results.iterrows():
        if row['configuration'] != 'Full Model':
            print(f"  {row['configuration']}: {row['f1_delta']:+.4f}")

    return ablation_results


def run_experiment_E4(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path
) -> pd.DataFrame:
    """
    E4: RCA Usefulness Analysis

    Analyze whether RCA features improve triage.
    """
    print("\n" + "="*60)
    print("EXPERIMENT E4: RCA Usefulness Analysis")
    print("="*60)

    results = []

    # Check if alerts with bugs are predicted correctly
    if 'rca_has_bug' in df.columns:
        print("\nAnalyzing bug prediction correlation...")

        # Train model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score
        model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
        model.fit(X, y)

        y_pred = model.predict(X)

        # Compare predictions for alerts with/without bugs
        has_bug_mask = df['rca_has_bug'] == 1
        no_bug_mask = df['rca_has_bug'] == 0

        has_bug_f1 = f1_score(y[has_bug_mask], y_pred[has_bug_mask], zero_division=0) if has_bug_mask.sum() > 0 else 0
        no_bug_f1 = f1_score(y[no_bug_mask], y_pred[no_bug_mask], zero_division=0) if no_bug_mask.sum() > 0 else 0

        results.append({
            'segment': 'Alerts with bugs',
            'n_samples': has_bug_mask.sum(),
            'f1_score': has_bug_f1
        })
        results.append({
            'segment': 'Alerts without bugs',
            'n_samples': no_bug_mask.sum(),
            'f1_score': no_bug_f1
        })

        print(f"  Alerts with bugs (n={has_bug_mask.sum()}): F1={has_bug_f1:.4f}")
        print(f"  Alerts without bugs (n={no_bug_mask.sum()}): F1={no_bug_f1:.4f}")

    # Check downstream alerts
    if 'rca_is_downstream' in df.columns:
        print("\nAnalyzing downstream alert predictions...")

        downstream_mask = df['rca_is_downstream'] == 1
        primary_mask = df['rca_is_downstream'] == 0

        downstream_f1 = f1_score(y[downstream_mask], y_pred[downstream_mask], zero_division=0) if downstream_mask.sum() > 0 else 0
        primary_f1 = f1_score(y[primary_mask], y_pred[primary_mask], zero_division=0) if primary_mask.sum() > 0 else 0

        results.append({
            'segment': 'Downstream alerts',
            'n_samples': downstream_mask.sum(),
            'f1_score': downstream_f1
        })
        results.append({
            'segment': 'Primary alerts',
            'n_samples': primary_mask.sum(),
            'f1_score': primary_f1
        })

        print(f"  Downstream alerts (n={downstream_mask.sum()}): F1={downstream_f1:.4f}")
        print(f"  Primary alerts (n={primary_mask.sum()}): F1={primary_f1:.4f}")

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df.to_csv(output_dir / 'reports' / 'E4_rca_analysis.csv', index=False)

    return results_df


def main():
    """Main execution function."""
    print("\n" + "#"*60)
    print("PHASE 7: Integration and End-to-End System")
    print("#"*60)
    print(f"Started at: {datetime.now().isoformat()}")

    set_random_seeds(RANDOM_SEED)

    # Output directory
    output_dir = PHASE_7_DIR / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)
    (output_dir / 'reports').mkdir(exist_ok=True)
    (output_dir / 'models').mkdir(exist_ok=True)

    # ========================================
    # Load and Prepare Data
    # ========================================
    print("\n" + "="*60)
    print("LOADING AND PREPARING DATA")
    print("="*60)

    # Create unified dataset (skip TS features for speed)
    print("\nCreating unified dataset...")
    df, feature_cols = create_unified_dataset(
        max_samples=None,
        include_ts=False,  # Skip TS extraction for faster demo
        include_rca=True
    )

    # Filter valid samples
    df = df.dropna(subset=[REGRESSION_TARGET_COL])
    print(f"Total samples after filtering: {len(df)}")

    # Prepare features
    X, y, feature_names = prepare_ensemble_features(
        df, feature_cols, REGRESSION_TARGET_COL,
        normalize=True, select_k=None
    )

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Also split the DataFrame for repository evaluation
    train_idx, test_idx = train_test_split(
        range(len(df)), test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    df_test = df.iloc[test_idx].reset_index(drop=True)

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': len(df),
        'n_features': X.shape[1],
        'train_size': len(X_train),
        'test_size': len(X_test)
    }

    # ========================================
    # Run Experiments
    # ========================================

    # E1: Model comparison
    e1_results, ensemble = run_experiment_E1(
        X_train, y_train, X_test, y_test, output_dir
    )
    all_results['E1'] = e1_results.to_dict(orient='records')
    gc.collect()

    # E2: Cross-suite evaluation
    e2_results = run_experiment_E2(df_test, X_test, y_test, output_dir)
    all_results['E2'] = e2_results.to_dict(orient='records') if len(e2_results) > 0 else []
    gc.collect()

    # E3: Ablation study
    e3_results = run_experiment_E3(X_train, y_train, feature_names, output_dir)
    all_results['E3'] = e3_results.to_dict(orient='records')
    gc.collect()

    # E4: RCA analysis
    e4_results = run_experiment_E4(df_test, X_test, y_test, output_dir)
    all_results['E4'] = e4_results.to_dict(orient='records') if len(e4_results) > 0 else []
    gc.collect()

    # ========================================
    # Generate Summary Report
    # ========================================
    print("\n" + "="*60)
    print("GENERATING SUMMARY")
    print("="*60)

    # Best model summary
    best_model = e1_results.iloc[0]
    print(f"\nBest model: {best_model['model']}")
    print(f"  F1 Score: {best_model['f1_score']:.4f}")
    print(f"  Precision: {best_model['precision']:.4f}")
    print(f"  Recall: {best_model['recall']:.4f}")
    print(f"  ROC AUC: {best_model['roc_auc']:.4f}")

    # Ensemble improvement
    ensemble_row = e1_results[e1_results['model'] == 'StackingEnsemble']
    if len(ensemble_row) > 0:
        ensemble_f1 = ensemble_row.iloc[0]['f1_score']
        base_avg_f1 = e1_results[e1_results['model'] != 'StackingEnsemble']['f1_score'].mean()
        improvement = ensemble_f1 - base_avg_f1
        print(f"\nEnsemble improvement over base average: {improvement:+.4f}")

    # Save all results
    save_results(all_results, output_dir / 'reports', 'experiment_summary')

    print(f"\nPhase 7 complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Finished at: {datetime.now().isoformat()}")

    return all_results


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Phase 7 failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
