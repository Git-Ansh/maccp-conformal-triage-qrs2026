"""
Shared visualization utilities for all phases.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve


def set_style():
    """Set consistent plotting style across all phases."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def plot_class_distribution(
    y: pd.Series,
    title: str = "Class Distribution",
    labels: Optional[List[str]] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot class distribution bar chart.

    Args:
        y: Target variable series
        title: Plot title
        labels: Optional custom labels
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    value_counts = y.value_counts().sort_index()

    if labels:
        value_counts.index = labels[:len(value_counts)]

    bars = ax.bar(value_counts.index.astype(str), value_counts.values, color=sns.color_palette("husl", len(value_counts)))

    # Add value labels on bars
    for bar, val in zip(bars, value_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val}\n({val/len(y)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot confusion matrix heatmap.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Class labels
        title: Plot title
        normalize: Whether to normalize
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_roc_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    title: str = "ROC Curves Comparison",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot multiple ROC curves for model comparison.

    Args:
        results: Dict of model_name -> (y_true, y_prob, auc_score)
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = sns.color_palette("husl", len(results))

    for (model_name, (y_true, y_prob, auc)), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{model_name} (AUC = {auc:.3f})')

    # Random classifier baseline
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_precision_recall_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    title: str = "Precision-Recall Curves Comparison",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot multiple PR curves for model comparison.

    Args:
        results: Dict of model_name -> (y_true, y_prob, pr_auc)
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = sns.color_palette("husl", len(results))

    for (model_name, (y_true, y_prob, pr_auc)), color in zip(results.items(), colors):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ax.plot(recall, precision, color=color, lw=2, label=f'{model_name} (AP = {pr_auc:.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot feature importance bar chart.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    set_style()
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get top N features
    top_features = importance_df.nlargest(top_n, 'importance')

    # Horizontal bar plot
    bars = ax.barh(top_features['feature'], top_features['importance'], color=sns.color_palette("viridis", top_n))

    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(title)
    ax.invert_yaxis()  # Highest importance at top

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_model_comparison(
    results_df: pd.DataFrame,
    metrics: List[str] = ['precision', 'recall', 'f1_score'],
    title: str = "Model Comparison",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot grouped bar chart comparing models across metrics.

    Args:
        results_df: DataFrame with model results
        metrics: List of metrics to compare
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    set_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(results_df))
    width = 0.8 / len(metrics)

    colors = sns.color_palette("husl", len(metrics))

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (i - len(metrics)/2 + 0.5) * width
        bars = ax.bar(x + offset, results_df[metric], width, label=metric.replace('_', ' ').title(), color=color)

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['model'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_ablation_results(
    results: Dict[str, Dict[str, float]],
    metric: str = 'f1_score',
    title: str = "Feature Ablation Study",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot ablation study results.

    Args:
        results: Dict of feature_group -> metrics
        metric: Metric to plot
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    set_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    groups = list(results.keys())
    values = [results[g][metric] for g in groups]

    colors = sns.color_palette("viridis", len(groups))
    bars = ax.bar(groups, values, color=colors)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Feature Group')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.15)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_timeseries_with_alert(
    series: np.ndarray,
    alert_idx: int,
    window_before: int = 20,
    window_after: int = 10,
    title: str = "Time Series with Alert",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot time series with alert point highlighted.

    Args:
        series: Time series values
        alert_idx: Index of alert point
        window_before: Points before alert to show
        window_after: Points after alert to show
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    set_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    start_idx = max(0, alert_idx - window_before)
    end_idx = min(len(series), alert_idx + window_after + 1)

    x = np.arange(start_idx, end_idx)
    y = series[start_idx:end_idx]

    ax.plot(x, y, 'b-', linewidth=1.5, label='Performance metric')
    ax.axvline(x=alert_idx, color='red', linestyle='--', linewidth=2, label='Alert point')
    ax.scatter([alert_idx], [series[alert_idx]], color='red', s=100, zorder=5)

    ax.set_xlabel('Revision Index')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Feature Correlation Matrix",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot correlation heatmap for numeric features.

    Args:
        df: DataFrame with features
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    set_style()

    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(14, 12))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8})

    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
