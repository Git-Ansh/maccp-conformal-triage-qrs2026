"""
Shared evaluation utilities for all phases.
Focus on high precision as per user requirements.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute comprehensive binary classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }

    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        except ValueError:
            # Handle edge cases where only one class is present
            metrics['roc_auc'] = np.nan
            metrics['pr_auc'] = np.nan

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0

    return metrics


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    labels: Optional[List] = None
) -> Dict[str, float]:
    """
    Compute comprehensive multi-class classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        labels: List of class labels

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

    if y_prob is not None:
        try:
            metrics['roc_auc_ovr'] = roc_auc_score(
                y_true, y_prob, multi_class='ovr', average='macro'
            )
        except ValueError:
            metrics['roc_auc_ovr'] = np.nan

    return metrics


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List
) -> pd.DataFrame:
    """
    Compute per-class precision, recall, and F1.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of class labels

    Returns:
        DataFrame with per-class metrics
    """
    precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    # Count support
    support = [(y_true == label).sum() for label in labels]

    df = pd.DataFrame({
        'class': labels,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support
    })

    return df


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_precision: float = 0.9
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal decision threshold to achieve target precision.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        target_precision: Target precision to achieve

    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # Find threshold where precision >= target_precision
    valid_indices = np.where(precisions[:-1] >= target_precision)[0]

    if len(valid_indices) == 0:
        # If target precision not achievable, use highest precision point
        optimal_idx = np.argmax(precisions[:-1])
    else:
        # Among valid thresholds, pick one with highest recall
        recall_at_valid = recalls[:-1][valid_indices]
        optimal_idx = valid_indices[np.argmax(recall_at_valid)]

    optimal_threshold = thresholds[optimal_idx]

    # Compute metrics at optimal threshold
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    metrics = compute_binary_metrics(y_true, y_pred_optimal, y_prob)
    metrics['threshold'] = optimal_threshold

    return optimal_threshold, metrics


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create comparison DataFrame from multiple model results.

    Args:
        results: Dictionary of model_name -> metrics

    Returns:
        DataFrame comparing all models
    """
    df = pd.DataFrame(results).T
    df.index.name = 'model'
    df = df.reset_index()

    # Sort by precision (user priority) then F1
    if 'precision' in df.columns:
        df = df.sort_values(['precision', 'f1_score'], ascending=[False, False])

    return df


def generate_classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    model_name: str = "Model"
) -> str:
    """
    Generate a text summary of classification results.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        model_name: Name of the model

    Returns:
        Formatted text summary
    """
    metrics = compute_binary_metrics(y_true, y_pred, y_prob)

    summary = f"""
{'='*60}
Classification Report: {model_name}
{'='*60}

Performance Metrics:
  - Accuracy:  {metrics['accuracy']:.4f}
  - Precision: {metrics['precision']:.4f}  (priority metric)
  - Recall:    {metrics['recall']:.4f}
  - F1-Score:  {metrics['f1_score']:.4f}
"""

    if y_prob is not None and not np.isnan(metrics.get('roc_auc', np.nan)):
        summary += f"""  - ROC-AUC:   {metrics['roc_auc']:.4f}
  - PR-AUC:    {metrics['pr_auc']:.4f}
"""

    summary += f"""
Confusion Matrix:
  - True Positives:  {metrics['true_positives']}
  - True Negatives:  {metrics['true_negatives']}
  - False Positives: {metrics['false_positives']}  (minimize this)
  - False Negatives: {metrics['false_negatives']}

False Positive Rate: {metrics['false_positive_rate']:.4f}
{'='*60}
"""
    return summary
