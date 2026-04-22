"""
Stage-specific and system-wide evaluation metrics for the cascade.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def stage_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_confident: np.ndarray,
    stage_name: str = "Stage"
) -> Dict:
    """
    Compute metrics for a single cascade stage.

    Reports:
        - Overall metrics (on all predictions including uncertain=0)
        - Selective metrics (only on confident predictions)
        - Coverage and deferral stats

    Args:
        y_true: True labels
        y_pred: Predicted labels (0 = deferred/uncertain)
        is_confident: Boolean mask of confident predictions
        stage_name: Name for reporting

    Returns:
        Dict of metrics
    """
    n_total = len(y_true)
    n_confident = is_confident.sum()
    coverage = n_confident / n_total if n_total > 0 else 0

    result = {
        'stage': stage_name,
        'n_total': n_total,
        'n_confident': int(n_confident),
        'n_deferred': int(n_total - n_confident),
        'coverage': coverage,
    }

    if n_confident > 0:
        y_true_conf = y_true[is_confident]
        y_pred_conf = y_pred[is_confident]

        result['accuracy_confident'] = accuracy_score(y_true_conf, y_pred_conf)
        result['precision_macro'] = precision_score(y_true_conf, y_pred_conf, average='macro', zero_division=0)
        result['recall_macro'] = recall_score(y_true_conf, y_pred_conf, average='macro', zero_division=0)
        result['f1_macro'] = f1_score(y_true_conf, y_pred_conf, average='macro', zero_division=0)
        result['f1_weighted'] = f1_score(y_true_conf, y_pred_conf, average='weighted', zero_division=0)
    else:
        result['accuracy_confident'] = np.nan
        result['precision_macro'] = np.nan
        result['recall_macro'] = np.nan
        result['f1_macro'] = np.nan
        result['f1_weighted'] = np.nan

    return result


def cascade_summary(stage_results: List[Dict]) -> pd.DataFrame:
    """
    Aggregate per-stage metrics into a summary table.

    Args:
        stage_results: List of dicts from stage_metrics()

    Returns:
        DataFrame with one row per stage
    """
    return pd.DataFrame(stage_results)


def end_to_end_metrics(
    summary_true_status: np.ndarray,
    summary_pred_status: np.ndarray,
    summary_is_automated: np.ndarray,
    alert_true_status: Optional[np.ndarray] = None,
    alert_pred_status: Optional[np.ndarray] = None,
    alert_is_automated: Optional[np.ndarray] = None,
    has_bug_true: Optional[np.ndarray] = None,
    has_bug_pred: Optional[np.ndarray] = None,
    has_bug_is_confident: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compute end-to-end cascade metrics.

    Args:
        summary_true_status: True summary status labels
        summary_pred_status: Predicted summary status (0=investigating)
        summary_is_automated: Boolean mask for auto-labeled summaries
        alert_true_status: True alert status (optional)
        alert_pred_status: Predicted alert status (optional)
        alert_is_automated: Boolean mask for auto-labeled alerts (optional)
        has_bug_true: True has_bug labels (optional)
        has_bug_pred: Predicted has_bug (0=uncertain, 1=no, 2=yes) (optional)
        has_bug_is_confident: Boolean mask for confident has_bug predictions (optional)

    Returns:
        Dict of end-to-end metrics
    """
    n_summaries = len(summary_true_status)
    n_automated_summaries = summary_is_automated.sum()

    result = {
        'n_summaries_total': n_summaries,
        'n_summaries_automated': int(n_automated_summaries),
        'n_summaries_deferred': int(n_summaries - n_automated_summaries),
        'summary_automation_rate': n_automated_summaries / n_summaries if n_summaries > 0 else 0,
    }

    if n_automated_summaries > 0:
        result['summary_accuracy_auto'] = accuracy_score(
            summary_true_status[summary_is_automated],
            summary_pred_status[summary_is_automated]
        )

    # Alert-level metrics
    if alert_true_status is not None and alert_pred_status is not None:
        n_alerts = len(alert_true_status)
        n_automated_alerts = alert_is_automated.sum() if alert_is_automated is not None else 0
        result['n_alerts_total'] = n_alerts
        result['n_alerts_automated'] = int(n_automated_alerts)
        result['alert_automation_rate'] = n_automated_alerts / n_alerts if n_alerts > 0 else 0

        if n_automated_alerts > 0:
            result['alert_accuracy_auto'] = accuracy_score(
                alert_true_status[alert_is_automated],
                alert_pred_status[alert_is_automated]
            )

    # has_bug metrics
    if has_bug_true is not None and has_bug_pred is not None and has_bug_is_confident is not None:
        n_bug_confident = has_bug_is_confident.sum()
        result['has_bug_n_confident'] = int(n_bug_confident)
        result['has_bug_coverage'] = n_bug_confident / len(has_bug_true) if len(has_bug_true) > 0 else 0

        if n_bug_confident > 0:
            result['has_bug_accuracy'] = accuracy_score(
                has_bug_true[has_bug_is_confident],
                has_bug_pred[has_bug_is_confident]
            )

    return result


def print_stage_report(metrics: Dict) -> str:
    """Format a stage metrics dict as a readable report."""
    lines = [
        f"{'='*60}",
        f"  {metrics['stage']}",
        f"{'='*60}",
        f"  Total samples:     {metrics['n_total']}",
        f"  Confident:         {metrics['n_confident']} ({metrics['coverage']:.1%})",
        f"  Deferred:          {metrics['n_deferred']}",
    ]
    if not np.isnan(metrics.get('accuracy_confident', np.nan)):
        lines.extend([
            f"  Accuracy (conf):   {metrics['accuracy_confident']:.4f}",
            f"  F1 macro (conf):   {metrics['f1_macro']:.4f}",
            f"  F1 weighted (conf):{metrics['f1_weighted']:.4f}",
        ])
    return '\n'.join(lines)
