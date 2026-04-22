"""
Bootstrap confidence intervals and statistical tests for cascade evaluation.

Provides:
- bootstrap_metric(): 95% CIs via 1000 resamples for any metric
- mcnemar_test(): McNemar's test for paired classifier comparison
- bootstrap_coverage_accuracy(): CIs for coverage-accuracy curves
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple
from scipy import stats


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        metric_fn: Function (y_true, y_pred) -> float
        n_bootstrap: Number of bootstrap resamples
        confidence_level: Confidence level (default 0.95)
        random_state: Random seed

    Returns:
        Dict with 'mean', 'ci_lower', 'ci_upper', 'std'
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    scores = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        try:
            scores[i] = metric_fn(y_true[idx], y_pred[idx])
        except Exception:
            scores[i] = np.nan

    scores = scores[~np.isnan(scores)]
    if len(scores) == 0:
        return {'mean': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'std': np.nan}

    alpha = (1 - confidence_level) / 2
    ci_lower = np.percentile(scores, 100 * alpha)
    ci_upper = np.percentile(scores, 100 * (1 - alpha))

    return {
        'mean': float(np.mean(scores)),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'std': float(np.std(scores)),
    }


def bootstrap_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, float]:
    """Shorthand for bootstrap CI on accuracy."""
    def accuracy(y_t, y_p):
        return (y_t == y_p).mean()
    return bootstrap_metric(y_true, y_pred, accuracy, n_bootstrap,
                            confidence_level, random_state)


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Dict[str, float]:
    """
    McNemar's test for comparing two classifiers on the same data.

    Tests whether the two classifiers have the same error rate.
    Significant p-value means classifiers differ significantly.

    Args:
        y_true: True labels
        y_pred_a: Predictions from classifier A
        y_pred_b: Predictions from classifier B

    Returns:
        Dict with 'statistic', 'p_value', 'n_a_correct_b_wrong',
        'n_a_wrong_b_correct', 'significant_0.05'
    """
    correct_a = (y_true == y_pred_a)
    correct_b = (y_true == y_pred_b)

    # Contingency: A right & B wrong vs A wrong & B right
    b = (correct_a & ~correct_b).sum()  # A correct, B wrong
    c = (~correct_a & correct_b).sum()  # A wrong, B correct

    # McNemar's test with continuity correction
    if b + c == 0:
        return {
            'statistic': 0.0,
            'p_value': 1.0,
            'n_a_correct_b_wrong': int(b),
            'n_a_wrong_b_correct': int(c),
            'significant_0.05': False,
        }

    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'n_a_correct_b_wrong': int(b),
        'n_a_wrong_b_correct': int(c),
        'significant_0.05': p_value < 0.05,
    }


def bootstrap_coverage_accuracy(
    y_true: np.ndarray,
    confidence: np.ndarray,
    predicted: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Bootstrap CIs for coverage-accuracy curve.

    Args:
        y_true: True labels
        confidence: Calibrated confidence scores
        predicted: Predicted labels (before gating)
        thresholds: Confidence thresholds to evaluate
        n_bootstrap: Number of bootstrap resamples
        confidence_level: CI level
        random_state: Random seed

    Returns:
        DataFrame with threshold, coverage_mean, coverage_ci_lower/upper,
        accuracy_mean, accuracy_ci_lower/upper
    """
    if thresholds is None:
        thresholds = np.arange(0.50, 0.96, 0.05)

    rng = np.random.RandomState(random_state)
    n = len(y_true)
    alpha = (1 - confidence_level) / 2

    results = []
    for t in thresholds:
        cov_scores = []
        acc_scores = []

        for _ in range(n_bootstrap):
            idx = rng.randint(0, n, size=n)
            mask = confidence[idx] >= t
            cov = mask.mean()
            cov_scores.append(cov)
            if mask.sum() > 0:
                acc = (y_true[idx][mask] == predicted[idx][mask]).mean()
                acc_scores.append(acc)

        cov_arr = np.array(cov_scores)
        acc_arr = np.array(acc_scores) if acc_scores else np.array([np.nan])

        results.append({
            'threshold': t,
            'coverage_mean': float(np.mean(cov_arr)),
            'coverage_ci_lower': float(np.percentile(cov_arr, 100 * alpha)),
            'coverage_ci_upper': float(np.percentile(cov_arr, 100 * (1 - alpha))),
            'accuracy_mean': float(np.nanmean(acc_arr)),
            'accuracy_ci_lower': float(np.nanpercentile(acc_arr, 100 * alpha)),
            'accuracy_ci_upper': float(np.nanpercentile(acc_arr, 100 * (1 - alpha))),
        })

    return pd.DataFrame(results)


def bootstrap_cascade_results(
    y_true: np.ndarray,
    cascade_pred: np.ndarray,
    cascade_conf: np.ndarray,
    cascade_automated: np.ndarray,
    majority_class: int,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Bootstrap CIs for end-to-end cascade metrics.

    Computes CIs for:
    - cascade accuracy (on automated items)
    - cascade coverage
    - majority baseline accuracy
    - accuracy lift (cascade - majority)

    Args:
        y_true: True labels
        cascade_pred: Final cascade predictions
        cascade_conf: Final cascade confidences
        cascade_automated: Boolean mask of automated items
        majority_class: Majority class label
        n_bootstrap: Number of resamples
        confidence_level: CI level
        random_state: Random seed

    Returns:
        Dict of metric_name -> {mean, ci_lower, ci_upper, std}
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    alpha = (1 - confidence_level) / 2
    automated = np.asarray(cascade_automated, dtype=bool)

    acc_scores = []
    cov_scores = []
    maj_scores = []
    lift_scores = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        auto_mask = automated[idx]

        cov = auto_mask.mean()
        cov_scores.append(cov)

        maj_acc = (y_true[idx] == majority_class).mean()
        maj_scores.append(maj_acc)

        if auto_mask.sum() > 0:
            acc = (y_true[idx][auto_mask] == cascade_pred[idx][auto_mask]).mean()
            acc_scores.append(acc)
            lift_scores.append(acc - maj_acc)

    def _ci(arr):
        arr = np.array(arr)
        return {
            'mean': float(np.mean(arr)),
            'ci_lower': float(np.percentile(arr, 100 * alpha)),
            'ci_upper': float(np.percentile(arr, 100 * (1 - alpha))),
            'std': float(np.std(arr)),
        }

    return {
        'accuracy': _ci(acc_scores),
        'coverage': _ci(cov_scores),
        'majority_baseline': _ci(maj_scores),
        'accuracy_lift': _ci(lift_scores),
    }


if __name__ == '__main__':
    # Quick demo
    rng = np.random.RandomState(42)
    y = rng.randint(0, 3, 500)
    pred = y.copy()
    pred[rng.rand(500) < 0.15] = rng.randint(0, 3, 500)[rng.rand(500) < 0.15]

    ci = bootstrap_accuracy(y, pred)
    print(f"Accuracy: {ci['mean']:.3f} [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")

    # McNemar test
    pred2 = y.copy()
    pred2[rng.rand(500) < 0.25] = rng.randint(0, 3, 500)[rng.rand(500) < 0.25]
    mc = mcnemar_test(y, pred, pred2)
    print(f"McNemar: stat={mc['statistic']:.2f}, p={mc['p_value']:.4f}, "
          f"sig={mc['significant_0.05']}")
