"""
Phase 1: Evaluation Module
Comprehensive evaluation for binary classification.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve
)

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.evaluation_utils import (
    compute_binary_metrics, find_optimal_threshold, compare_models
)
from common.visualization_utils import (
    plot_confusion_matrix, plot_roc_curves, plot_precision_recall_curves,
    plot_model_comparison
)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate a single model's predictions.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        model_name: Name of the model

    Returns:
        Dictionary of metrics
    """
    metrics = compute_binary_metrics(y_true, y_pred, y_prob)
    metrics['model'] = model_name

    print(f"\n{model_name} Results:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")

    if y_prob is not None and not np.isnan(metrics.get('roc_auc', np.nan)):
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")

    return metrics


def evaluate_with_threshold_tuning(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_precision: float = 0.9,
    model_name: str = "Model"
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate with threshold tuned for target precision.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        target_precision: Target precision to achieve
        model_name: Name of the model

    Returns:
        Tuple of (optimal_threshold, metrics)
    """
    optimal_threshold, metrics = find_optimal_threshold(
        y_true, y_prob, target_precision
    )
    metrics['model'] = model_name

    print(f"\n{model_name} (Threshold={optimal_threshold:.3f}):")
    print(f"  Target Precision: {target_precision}")
    print(f"  Achieved Precision: {metrics['precision']:.4f}")
    print(f"  Recall at threshold: {metrics['recall']:.4f}")
    print(f"  F1 at threshold: {metrics['f1_score']:.4f}")

    return optimal_threshold, metrics


def run_full_evaluation(
    models: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Run comprehensive evaluation of all models.

    Args:
        models: Dictionary of model_name -> fitted_model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        output_dir: Directory to save outputs

    Returns:
        DataFrame with all results
    """
    results = []
    roc_data = {}
    pr_data = {}

    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Standard evaluation
        metrics = evaluate_model(y_test, y_pred, y_prob, name)
        results.append(metrics)

        # Store data for curves
        roc_data[name] = (y_test, y_prob, metrics['roc_auc'])
        pr_data[name] = (y_test, y_prob, metrics['pr_auc'])

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('precision', ascending=False)

    # Save outputs if directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save results CSV
        results_df.to_csv(output_dir / 'evaluation_results.csv', index=False)

        # Save figures
        figures_dir = output_dir.parent / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)

        # ROC curves
        plot_roc_curves(
            roc_data,
            title="ROC Curves - Model Comparison",
            save_path=figures_dir / 'roc_curves_comparison.png'
        )

        # PR curves
        plot_precision_recall_curves(
            pr_data,
            title="Precision-Recall Curves - Model Comparison",
            save_path=figures_dir / 'pr_curves_comparison.png'
        )

        # Model comparison bar chart
        plot_model_comparison(
            results_df,
            metrics=['precision', 'recall', 'f1_score'],
            title="Model Comparison",
            save_path=figures_dir / 'model_comparison.png'
        )

        # Confusion matrices for each model
        for name, model in models.items():
            y_pred = model.predict(X_test)
            plot_confusion_matrix(
                y_test, y_pred,
                labels=['Not Regression', 'Regression'],
                title=f"Confusion Matrix - {name}",
                save_path=figures_dir / f'confusion_matrix_{name.lower().replace(" ", "_")}.png'
            )

        print(f"\nResults saved to {output_dir}")

    return results_df


def cross_repository_evaluation(
    model,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    train_repos: List[str],
    test_repo: str
) -> Dict[str, float]:
    """
    Evaluate model generalization across repositories.

    Args:
        model: Model to evaluate
        train_data: Training DataFrame
        test_data: Test DataFrame
        feature_cols: Feature column names
        target_col: Target column name
        train_repos: Training repositories
        test_repo: Test repository

    Returns:
        Dictionary of metrics
    """
    X_train = train_data[feature_cols].values
    y_train = train_data[target_col].values
    X_test = test_data[feature_cols].values
    y_test = test_data[target_col].values

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate
    metrics = compute_binary_metrics(y_test, y_pred, y_prob)
    metrics['train_repos'] = str(train_repos)
    metrics['test_repo'] = test_repo
    metrics['n_train'] = len(y_train)
    metrics['n_test'] = len(y_test)

    return metrics


def ablation_study(
    model_class,
    model_params: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_groups: Dict[str, List[int]],
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Run feature ablation study.

    Args:
        model_class: Model class to use
        model_params: Model parameters
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        feature_groups: Dict of group_name -> list of feature indices
        feature_names: List of feature names

    Returns:
        DataFrame with ablation results
    """
    results = []

    for group_name, feature_indices in feature_groups.items():
        print(f"\nEvaluating feature group: {group_name}")

        # Select features
        X_train_subset = X_train[:, feature_indices]
        X_test_subset = X_test[:, feature_indices]

        # Train model
        model = model_class(**model_params)
        model.fit(X_train_subset, y_train)

        # Predict
        y_pred = model.predict(X_test_subset)
        y_prob = model.predict_proba(X_test_subset)[:, 1]

        # Evaluate
        metrics = compute_binary_metrics(y_test, y_pred, y_prob)
        metrics['feature_group'] = group_name
        metrics['n_features'] = len(feature_indices)
        metrics['features'] = [feature_names[i] for i in feature_indices]

        results.append(metrics)

        print(f"  Features: {len(feature_indices)}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  F1: {metrics['f1_score']:.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1_score', ascending=False)

    return results_df


def generate_evaluation_report(
    results_df: pd.DataFrame,
    output_path: Path,
    experiment_name: str = "Phase 1 Evaluation"
) -> str:
    """
    Generate markdown evaluation report.

    Args:
        results_df: DataFrame with results
        output_path: Path to save report
        experiment_name: Name of the experiment

    Returns:
        Report content as string
    """
    report = f"""# {experiment_name} Report

## Summary

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
"""

    for _, row in results_df.iterrows():
        report += f"| {row['model']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1_score']:.4f} | {row.get('roc_auc', 'N/A'):.4f if isinstance(row.get('roc_auc'), float) else 'N/A'} |\n"

    best_model = results_df.iloc[0]
    report += f"""

## Best Model

The best performing model based on **precision** is **{best_model['model']}**:
- Precision: {best_model['precision']:.4f}
- Recall: {best_model['recall']:.4f}
- F1-Score: {best_model['f1_score']:.4f}

## Observations

1. Focus on precision (minimize false positives) as per user requirements
2. Trade-off between precision and recall is managed through threshold tuning
3. See figures/ directory for visualization plots

"""

    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Report saved to {output_path}")

    return report


if __name__ == "__main__":
    # Test evaluation
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2,
        weights=[0.7, 0.3], random_state=42
    )

    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    # Train models
    models = {
        'LogisticRegression': LogisticRegression(class_weight='balanced').fit(X_train, y_train),
        'RandomForest': RandomForestClassifier(class_weight='balanced', n_estimators=100).fit(X_train, y_train)
    }

    # Evaluate
    results = run_full_evaluation(models, X_train, y_train, X_test, y_test)
    print("\nResults:")
    print(results[['model', 'precision', 'recall', 'f1_score']])
