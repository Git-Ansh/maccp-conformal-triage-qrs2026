"""
Phase 4: Change-Point Detection Evaluation
Evaluate detected change points against ground truth.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


def match_detected_to_true(
    detected: List[int],
    true_points: List[int],
    tolerance: int = 5
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match detected change points to true change points.

    Args:
        detected: List of detected change point indices
        true_points: List of true change point indices
        tolerance: Maximum distance for a match

    Returns:
        Tuple of (matches, unmatched_detected, unmatched_true)
    """
    matches = []
    unmatched_detected = list(detected)
    unmatched_true = list(true_points)

    for true_cp in true_points:
        best_match = None
        best_dist = float('inf')

        for det_cp in unmatched_detected:
            dist = abs(det_cp - true_cp)
            if dist <= tolerance and dist < best_dist:
                best_match = det_cp
                best_dist = dist

        if best_match is not None:
            matches.append((true_cp, best_match))
            unmatched_detected.remove(best_match)
            unmatched_true.remove(true_cp)

    return matches, unmatched_detected, unmatched_true


def compute_detection_metrics(
    detected: List[int],
    true_points: List[int],
    tolerance: int = 5
) -> Dict[str, float]:
    """
    Compute precision, recall, F1 for change point detection.

    Args:
        detected: Detected change points
        true_points: True change points
        tolerance: Matching tolerance

    Returns:
        Dictionary of metrics
    """
    if not true_points:
        return {
            'precision': 1.0 if not detected else 0.0,
            'recall': 1.0,
            'f1_score': 1.0 if not detected else 0.0,
            'n_detected': len(detected),
            'n_true': 0,
            'n_matches': 0,
            'false_positives': len(detected),
            'false_negatives': 0
        }

    matches, unmatched_det, unmatched_true = match_detected_to_true(
        detected, true_points, tolerance
    )

    tp = len(matches)
    fp = len(unmatched_det)
    fn = len(unmatched_true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Compute detection delay for matches
    delays = [det - true for true, det in matches]
    avg_delay = np.mean(delays) if delays else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'n_detected': len(detected),
        'n_true': len(true_points),
        'n_matches': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'avg_delay': avg_delay
    }


def evaluate_detector_on_series(
    detector,
    signals: List[np.ndarray],
    true_points_list: List[List[int]],
    tolerance: int = 5
) -> Dict[str, float]:
    """
    Evaluate a detector across multiple time series.

    Args:
        detector: Change point detector
        signals: List of time series
        true_points_list: List of true change points for each series
        tolerance: Matching tolerance

    Returns:
        Aggregated metrics
    """
    all_metrics = []

    for signal, true_points in zip(signals, true_points_list):
        if len(signal) < 10:
            continue

        detected = detector.fit_predict(signal)
        metrics = compute_detection_metrics(detected, true_points, tolerance)
        all_metrics.append(metrics)

    if not all_metrics:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'n_series': 0
        }

    # Aggregate metrics
    df = pd.DataFrame(all_metrics)

    return {
        'precision': df['precision'].mean(),
        'recall': df['recall'].mean(),
        'f1_score': df['f1_score'].mean(),
        'precision_std': df['precision'].std(),
        'recall_std': df['recall'].std(),
        'f1_std': df['f1_score'].std(),
        'avg_delay': df['avg_delay'].mean(),
        'total_detected': df['n_detected'].sum(),
        'total_true': df['n_true'].sum(),
        'total_matches': df['n_matches'].sum(),
        'total_fp': df['false_positives'].sum(),
        'total_fn': df['false_negatives'].sum(),
        'n_series': len(all_metrics)
    }


def benchmark_detectors(
    detectors: Dict,
    signals: List[np.ndarray],
    true_points_list: List[List[int]],
    tolerance: int = 5
) -> pd.DataFrame:
    """
    Benchmark multiple detectors.

    Args:
        detectors: Dictionary of detector_name -> detector
        signals: List of time series
        true_points_list: True change points for each series
        tolerance: Matching tolerance

    Returns:
        DataFrame with benchmark results
    """
    results = []

    for name, detector in detectors.items():
        print(f"  Evaluating {name}...")
        metrics = evaluate_detector_on_series(
            detector, signals, true_points_list, tolerance
        )
        metrics['algorithm'] = name
        results.append(metrics)

    return pd.DataFrame(results)


def evaluate_by_group(
    detectors: Dict,
    signals: List[np.ndarray],
    true_points_list: List[List[int]],
    group_labels: List[str],
    tolerance: int = 5
) -> pd.DataFrame:
    """
    Evaluate detectors grouped by some category (suite, platform, etc).

    Args:
        detectors: Dictionary of detectors
        signals: List of time series
        true_points_list: True change points
        group_labels: Group label for each series
        tolerance: Matching tolerance

    Returns:
        DataFrame with results by group
    """
    results = []

    # Group data
    groups = {}
    for signal, true_pts, label in zip(signals, true_points_list, group_labels):
        if label not in groups:
            groups[label] = {'signals': [], 'true_points': []}
        groups[label]['signals'].append(signal)
        groups[label]['true_points'].append(true_pts)

    for group_name, data in groups.items():
        for det_name, detector in detectors.items():
            metrics = evaluate_detector_on_series(
                detector, data['signals'], data['true_points'], tolerance
            )
            metrics['algorithm'] = det_name
            metrics['group'] = group_name
            results.append(metrics)

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Test evaluation
    np.random.seed(42)

    # Create test signals
    signals = []
    true_points = []

    for _ in range(10):
        cp = np.random.randint(30, 70)
        signal = np.concatenate([
            np.random.normal(100, 5, cp),
            np.random.normal(120, 5, 100 - cp)
        ])
        signals.append(signal)
        true_points.append([cp])

    from algorithms import get_all_detectors
    detectors = get_all_detectors()

    results = benchmark_detectors(detectors, signals, true_points)
    print("\nBenchmark Results:")
    print(results[['algorithm', 'precision', 'recall', 'f1_score']].to_string())
