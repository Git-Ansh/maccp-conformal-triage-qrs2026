"""
Phase 2: Evaluation Module
Multi-class classification evaluation metrics and reporting.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, balanced_accuracy_score
)

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.visualization_utils import plot_confusion_matrix


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute multi-class classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }

    return metrics


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute per-class precision, recall, F1.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names

    Returns:
        DataFrame with per-class metrics
    """
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))

    if class_names is None:
        class_names = [f"Class_{i}" for i in unique_classes]

    per_class = []
    for i, cls in enumerate(unique_classes):
        cls_mask_true = y_true == cls
        cls_mask_pred = y_pred == cls

        tp = np.sum(cls_mask_true & cls_mask_pred)
        fp = np.sum(~cls_mask_true & cls_mask_pred)
        fn = np.sum(cls_mask_true & ~cls_mask_pred)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        name = class_names[cls] if cls < len(class_names) else f"Class_{cls}"

        per_class.append({
            'class_id': cls,
            'class_name': name,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': int(cls_mask_true.sum())
        })

    return pd.DataFrame(per_class)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    model_name: str = "Model"
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Full evaluation of a multi-class model.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names
        model_name: Name of the model

    Returns:
        Tuple of (metrics dict, per-class DataFrame)
    """
    metrics = compute_multiclass_metrics(y_true, y_pred, class_names)
    metrics['model'] = model_name

    per_class = compute_per_class_metrics(y_true, y_pred, class_names)

    print(f"\n{model_name} Results:")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1:          {metrics['f1_macro']:.4f}")
    print(f"  Weighted F1:       {metrics['f1_weighted']:.4f}")

    return metrics, per_class


def run_full_evaluation(
    models: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Run comprehensive evaluation of all models.

    Args:
        models: Dictionary of model_name -> fitted_model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        class_names: Optional list of class names
        output_dir: Directory to save outputs

    Returns:
        Tuple of (results DataFrame, dict of per-class DataFrames)
    """
    results = []
    per_class_results = {}

    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        y_pred = model.predict(X_test)

        metrics, per_class = evaluate_model(y_test, y_pred, class_names, name)
        results.append(metrics)
        per_class_results[name] = per_class

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1_macro', ascending=False)

    # Save outputs
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        figures_dir = output_dir.parent / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Save results CSV
        results_df.to_csv(output_dir / 'multiclass_results.csv', index=False)

        # Confusion matrices
        for name, model in models.items():
            y_pred = model.predict(X_test)
            labels = class_names if class_names else [f"Class_{i}" for i in np.unique(y_test)]

            plot_confusion_matrix(
                y_test, y_pred,
                labels=labels,
                title=f"Confusion Matrix - {name}",
                save_path=figures_dir / f'confusion_matrix_{name.lower().replace(" ", "_")}.png'
            )

            # Save per-class results
            per_class_results[name].to_csv(
                output_dir / f'per_class_{name.lower().replace(" ", "_")}.csv',
                index=False
            )

        print(f"\nResults saved to {output_dir}")

    return results_df, per_class_results


def cross_repository_evaluation(
    model,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    train_repos: List[str],
    test_repo: str,
    label_encoder=None
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
        label_encoder: Label encoder for target

    Returns:
        Dictionary of metrics
    """
    X_train = train_data[feature_cols].values
    y_train = train_data[target_col].values
    X_test = test_data[feature_cols].values
    y_test = test_data[target_col].values

    # Encode if needed
    if label_encoder:
        y_train = label_encoder.transform(y_train)
        y_test = label_encoder.transform(y_test)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    metrics = compute_multiclass_metrics(y_test, y_pred)
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
    Run feature ablation study for multi-class.

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

        X_train_subset = X_train[:, feature_indices]
        X_test_subset = X_test[:, feature_indices]

        model = model_class(**model_params)
        model.fit(X_train_subset, y_train)

        y_pred = model.predict(X_test_subset)

        metrics = compute_multiclass_metrics(y_test, y_pred)
        metrics['feature_group'] = group_name
        metrics['n_features'] = len(feature_indices)
        metrics['features'] = [feature_names[i] for i in feature_indices]

        results.append(metrics)

        print(f"  Features: {len(feature_indices)}")
        print(f"  Macro F1: {metrics['f1_macro']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1_macro', ascending=False)

    return results_df


def generate_multiclass_report(
    results_df: pd.DataFrame,
    per_class_results: Dict[str, pd.DataFrame],
    output_path: Path,
    experiment_name: str = "Phase 2 Multi-Class Evaluation"
) -> str:
    """
    Generate markdown evaluation report.

    Args:
        results_df: DataFrame with overall results
        per_class_results: Dict of model -> per-class DataFrame
        output_path: Path to save report
        experiment_name: Name of the experiment

    Returns:
        Report content as string
    """
    report = f"""# {experiment_name} Report

## Overall Results

| Model | Accuracy | Balanced Acc | Macro F1 | Weighted F1 |
|-------|----------|--------------|----------|-------------|
"""

    for _, row in results_df.iterrows():
        report += f"| {row['model']} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | {row['f1_macro']:.4f} | {row['f1_weighted']:.4f} |\n"

    # Best model details
    best_model = results_df.iloc[0]['model']
    report += f"""

## Best Model: {best_model}

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
"""

    best_per_class = per_class_results[best_model]
    for _, row in best_per_class.iterrows():
        report += f"| {row['class_name']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1_score']:.4f} | {row['support']} |\n"

    report += """

## Observations

1. Multi-class classification is more challenging than binary due to class imbalance
2. Classes with few samples may have low recall
3. See confusion matrices in figures/ for detailed error analysis

"""

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
        n_samples=1000, n_features=20, n_classes=5,
        n_clusters_per_class=1, random_state=42
    )

    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000).fit(X_train, y_train),
        'RandomForest': RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
    }

    results_df, per_class = run_full_evaluation(
        models, X_train, y_train, X_test, y_test
    )

    print("\nOverall Results:")
    print(results_df[['model', 'accuracy', 'f1_macro', 'f1_weighted']])
