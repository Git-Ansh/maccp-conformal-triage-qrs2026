"""
Phase 7: Evaluation

Cross-phase comparison and ablation studies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import xgboost as xgb
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.data_paths import RANDOM_SEED


def evaluate_integrated_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate integrated model on train and test sets.

    Returns:
        Dictionary with train and test metrics
    """
    results = {}

    for name, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
        y_pred = model.predict(X)

        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)
            if len(y_proba.shape) > 1:
                y_proba = y_proba[:, 1]
        else:
            y_proba = y_pred.astype(float)

        metrics = {
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0)
        }

        if len(np.unique(y)) > 1:
            try:
                metrics['roc_auc'] = roc_auc_score(y, y_proba)
            except:
                metrics['roc_auc'] = 0.5

        results[name] = metrics

    return results


def cross_phase_comparison(
    X: np.ndarray,
    y: np.ndarray,
    feature_groups: Dict[str, List[int]],
    cv: int = 5
) -> pd.DataFrame:
    """
    Compare performance using different feature groups.

    Args:
        X: Full feature matrix
        y: Target labels
        feature_groups: Dict mapping group name to feature indices
        cv: Number of CV folds

    Returns:
        DataFrame with comparison results
    """
    results = []

    for group_name, indices in feature_groups.items():
        print(f"  Evaluating {group_name} ({len(indices)} features)...")

        X_group = X[:, indices]

        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6,
            random_state=RANDOM_SEED, eval_metric='logloss'
        )

        try:
            scores = cross_val_score(model, X_group, y, cv=cv, scoring='f1')
            results.append({
                'feature_group': group_name,
                'n_features': len(indices),
                'f1_mean': np.mean(scores),
                'f1_std': np.std(scores)
            })
            print(f"    F1: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
        except Exception as e:
            print(f"    Error: {e}")

    return pd.DataFrame(results)


def ablation_study(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    ablation_groups: Dict[str, List[str]],
    cv: int = 5
) -> pd.DataFrame:
    """
    Perform ablation study by removing feature groups.

    Args:
        X: Feature matrix
        y: Target labels
        feature_names: Names of all features
        ablation_groups: Dict mapping group name to feature names to remove
        cv: Number of CV folds

    Returns:
        DataFrame with ablation results
    """
    results = []

    # Full model baseline
    print("  Training full model...")
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6,
        random_state=RANDOM_SEED, eval_metric='logloss'
    )
    full_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    results.append({
        'configuration': 'Full Model',
        'removed_features': 0,
        'remaining_features': X.shape[1],
        'f1_mean': np.mean(full_scores),
        'f1_std': np.std(full_scores),
        'f1_delta': 0.0
    })
    print(f"    Full Model F1: {np.mean(full_scores):.4f}")

    # Ablate each group
    for group_name, features_to_remove in ablation_groups.items():
        print(f"  Ablating {group_name}...")

        # Find indices to keep
        keep_indices = []
        for i, name in enumerate(feature_names):
            if not any(rem in name for rem in features_to_remove):
                keep_indices.append(i)

        if len(keep_indices) == 0:
            print(f"    Skipping (would remove all features)")
            continue

        X_ablated = X[:, keep_indices]

        try:
            scores = cross_val_score(model, X_ablated, y, cv=cv, scoring='f1')
            delta = np.mean(scores) - np.mean(full_scores)
            results.append({
                'configuration': f'Without {group_name}',
                'removed_features': X.shape[1] - len(keep_indices),
                'remaining_features': len(keep_indices),
                'f1_mean': np.mean(scores),
                'f1_std': np.std(scores),
                'f1_delta': delta
            })
            print(f"    F1: {np.mean(scores):.4f} (delta: {delta:+.4f})")
        except Exception as e:
            print(f"    Error: {e}")

    return pd.DataFrame(results)


def repository_evaluation(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    repo_col: str = 'alert_summary_repository'
) -> pd.DataFrame:
    """
    Evaluate model performance per repository.

    LEGACY VERSION with DATA LEAKAGE:
    Trains on all data, tests on subsets. Kept for backward compatibility
    but marked as deprecated. Use cross_repository_evaluation() instead.

    Args:
        df: DataFrame with repository column
        X: Feature matrix
        y: Target labels
        repo_col: Repository column name

    Returns:
        DataFrame with per-repository results
    """
    print("WARNING: repository_evaluation() has data leakage.")
    print("         Use cross_repository_evaluation() for valid results.")

    results = []

    # Train global model (LEAKAGE: trains on all data including test repos)
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6,
        random_state=RANDOM_SEED, eval_metric='logloss'
    )
    model.fit(X, y)

    # Evaluate per repository
    for repo in df[repo_col].unique():
        mask = df[repo_col] == repo
        if mask.sum() < 50:
            continue

        X_repo = X[mask]
        y_repo = y[mask]

        y_pred = model.predict(X_repo)

        results.append({
            'repository': repo,
            'n_samples': mask.sum(),
            'precision': precision_score(y_repo, y_pred, zero_division=0),
            'recall': recall_score(y_repo, y_pred, zero_division=0),
            'f1_score': f1_score(y_repo, y_pred, zero_division=0),
            'note': 'LEAKAGE: trained on all repos'
        })

    return pd.DataFrame(results)


def cross_repository_evaluation(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    repo_col: str = 'alert_summary_repository'
) -> pd.DataFrame:
    """
    TRUE cross-repository evaluation (NO DATA LEAKAGE).

    For each repository:
    1. Train on all OTHER repositories
    2. Test on the held-out repository
    3. Report metrics

    This evaluates true generalization across projects.

    Args:
        df: DataFrame with repository column
        X: Feature matrix
        y: Target labels
        repo_col: Repository column name

    Returns:
        DataFrame with leave-one-repo-out results
    """
    results = []

    unique_repos = df[repo_col].unique()

    for test_repo in unique_repos:
        # Split: train on all OTHER repos, test on this repo
        train_mask = df[repo_col] != test_repo
        test_mask = df[repo_col] == test_repo

        if train_mask.sum() < 100 or test_mask.sum() < 50:
            print(f"  Skipping {test_repo}: insufficient data")
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        print(f"  Training on {train_mask.sum()} samples (excluding {test_repo})")
        print(f"  Testing on {test_mask.sum()} samples ({test_repo})")

        # Train model on OTHER repos
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6,
            random_state=RANDOM_SEED, eval_metric='logloss'
        )
        model.fit(X_train, y_train)

        # Test on held-out repo
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.5

        results.append({
            'test_repository': test_repo,
            'train_n': train_mask.sum(),
            'test_n': test_mask.sum(),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': auc,
            'mcc': matthews_corrcoef(y_test, y_pred),
            'note': 'No leakage: trained on other repos only'
        })

    return pd.DataFrame(results)


def generate_comparison_report(
    individual_results: Dict[str, Dict[str, float]],
    ensemble_results: Dict[str, float],
    output_path: Path
):
    """
    Generate comparison report between individual and ensemble models.

    Args:
        individual_results: Results from individual phase models
        ensemble_results: Results from stacking ensemble
        output_path: Path to save report
    """
    report_lines = [
        "# Cross-Phase Model Comparison Report",
        "",
        "## Individual Phase Models",
        ""
    ]

    for phase, metrics in individual_results.items():
        report_lines.append(f"### {phase}")
        for metric, value in metrics.items():
            report_lines.append(f"- {metric}: {value:.4f}")
        report_lines.append("")

    report_lines.extend([
        "## Stacking Ensemble",
        ""
    ])
    for metric, value in ensemble_results.items():
        report_lines.append(f"- {metric}: {value:.4f}")

    report_lines.extend([
        "",
        "## Key Findings",
        "",
        "1. The stacking ensemble combines predictions from multiple base models.",
        "2. Feature groups from different phases contribute complementary signals.",
        "3. Cross-repository evaluation shows model generalization capability."
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Report saved to: {output_path}")
