#!/usr/bin/env python3
"""
Phase 4: Change-Point Detection Benchmarking - Run Script
Benchmark classical and modern change-point detection algorithms.
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime
import gc

import numpy as np
import pandas as pd

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms import get_all_detectors, HAS_RUPTURES
from src.evaluation import (
    benchmark_detectors, compute_detection_metrics, evaluate_by_group
)

from phase_3.src.timeseries_loader import (
    build_signature_index, load_timeseries_from_zip
)
from phase_3.src.alert_timeseries_matcher import (
    load_alerts_with_signatures, find_alert_index_in_timeseries
)

from common.data_paths import (
    PHASE_4_DIR, REGRESSION_TARGET_COL, SIGNATURE_ID_COL, RANDOM_SEED
)
from common.model_utils import save_results, set_random_seeds

warnings.filterwarnings('ignore')


def preprocess_signal(
    signal: np.ndarray,
    smooth: bool = False,
    smooth_window: int = 3,
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess time series for change-point detection.

    Args:
        signal: Raw signal
        smooth: Whether to apply smoothing
        smooth_window: Window for rolling mean
        normalize: Whether to z-score normalize

    Returns:
        Preprocessed signal
    """
    signal = signal.copy().astype(float)

    # Remove NaN
    signal = signal[~np.isnan(signal)]

    if len(signal) < 5:
        return signal

    # Smooth with rolling mean
    if smooth and smooth_window > 1:
        signal = pd.Series(signal).rolling(smooth_window, center=True, min_periods=1).mean().values

    # Z-score normalize
    if normalize:
        mean = np.mean(signal)
        std = np.std(signal)
        if std > 0:
            signal = (signal - mean) / std

    return signal


def extract_regression_series(
    alerts_df: pd.DataFrame,
    signature_index: dict,
    window_before: int = 30,
    window_after: int = 20,
    max_series: int = None
) -> tuple:
    """
    Extract time series windows around regression alerts.

    Returns:
        Tuple of (signals, true_change_points, metadata)
    """
    # Filter to regressions only
    regressions = alerts_df[alerts_df[REGRESSION_TARGET_COL] == True].copy()
    print(f"Regression alerts: {len(regressions)}")

    signals = []
    true_points = []
    metadata_list = []

    count = 0
    for sig_id, group in regressions.groupby(SIGNATURE_ID_COL):
        if max_series and count >= max_series:
            break

        sig_id = int(sig_id)
        if sig_id not in signature_index:
            continue

        ts_df = load_timeseries_from_zip(signature_index[sig_id], sig_id)
        if ts_df is None:
            continue

        ts_df = ts_df.sort_values('push_timestamp').reset_index(drop=True)
        values = ts_df['value'].values

        for _, alert_row in group.iterrows():
            if max_series and count >= max_series:
                break

            alert_idx = find_alert_index_in_timeseries(ts_df, alert_row)
            if alert_idx is None:
                continue

            # Extract window
            start = max(0, alert_idx - window_before)
            end = min(len(values), alert_idx + window_after)

            if end - start < 20:
                continue

            signal = values[start:end]
            # True change point relative to window
            true_cp = alert_idx - start

            signals.append(signal)
            true_points.append([true_cp])
            metadata_list.append({
                'signature_id': sig_id,
                'alert_id': alert_row.get('single_alert_id'),
                'repository': alert_row.get('alert_summary_repository', 'unknown'),
                'suite': alert_row.get('single_alert_series_signature_suite', 'unknown'),
                'platform': alert_row.get('single_alert_series_signature_machine_platform', 'unknown')
            })

            count += 1

        del ts_df
        gc.collect()

    print(f"Extracted {len(signals)} time series windows")
    return signals, true_points, metadata_list


def run_experiment_E1(signals, true_points, output_dir, tolerance=5):
    """
    E1: Single-change detection comparison
    """
    print("\n" + "="*60)
    print("EXPERIMENT E1: Single-Change Detection Comparison")
    print("="*60)

    detectors = get_all_detectors(
        pen_values=[1.0, 3.0, 5.0],
        window_sizes=[5, 10]
    )

    results = benchmark_detectors(detectors, signals, true_points, tolerance)
    results = results.sort_values('f1_score', ascending=False)

    results.to_csv(output_dir / 'reports' / 'E1_algorithm_comparison.csv', index=False)

    print("\nTop algorithms by F1 score:")
    print(results[['algorithm', 'precision', 'recall', 'f1_score', 'n_series']].head(10).to_string())

    return results


def run_experiment_E2(signals, true_points, output_dir, tolerance=5):
    """
    E2: Sensitivity to tolerance window
    """
    print("\n" + "="*60)
    print("EXPERIMENT E2: Tolerance Sensitivity")
    print("="*60)

    # Use best detectors
    detectors = {
        'PELT_l2_pen3': get_all_detectors()['PELT_l2_pen3.0'],
        'MeanShift_2.5': get_all_detectors()['MeanShift_2.5'],
        'BOCD': get_all_detectors()['BOCD']
    }

    results = []
    for tol in [3, 5, 7, 10]:
        tol_results = benchmark_detectors(detectors, signals, true_points, tol)
        tol_results['tolerance'] = tol
        results.append(tol_results)

    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv(output_dir / 'reports' / 'E2_tolerance_sensitivity.csv', index=False)

    print("\nF1 by tolerance:")
    pivot = results_df.pivot_table(values='f1_score', index='algorithm', columns='tolerance')
    print(pivot.to_string())

    return results_df


def run_experiment_E3(signals_raw, true_points, output_dir, tolerance=5):
    """
    E3: Sensitivity to smoothing
    """
    print("\n" + "="*60)
    print("EXPERIMENT E3: Smoothing Sensitivity")
    print("="*60)

    detectors = {
        'PELT_l2_pen3': get_all_detectors()['PELT_l2_pen3.0'],
        'MeanShift_2.5': get_all_detectors()['MeanShift_2.5']
    }

    results = []

    # Raw (normalized only)
    signals_norm = [preprocess_signal(s, smooth=False, normalize=True) for s in signals_raw]
    raw_results = benchmark_detectors(detectors, signals_norm, true_points, tolerance)
    raw_results['preprocessing'] = 'normalized'
    results.append(raw_results)

    # Smoothed
    for window in [3, 5]:
        signals_smooth = [preprocess_signal(s, smooth=True, smooth_window=window, normalize=True)
                         for s in signals_raw]
        smooth_results = benchmark_detectors(detectors, signals_smooth, true_points, tolerance)
        smooth_results['preprocessing'] = f'smooth_{window}'
        results.append(smooth_results)

    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv(output_dir / 'reports' / 'E3_smoothing_sensitivity.csv', index=False)

    print("\nF1 by preprocessing:")
    pivot = results_df.pivot_table(values='f1_score', index='algorithm', columns='preprocessing')
    print(pivot.to_string())

    return results_df


def run_experiment_E4(signals, true_points, metadata, output_dir, tolerance=5):
    """
    E4: Cross-suite performance
    """
    print("\n" + "="*60)
    print("EXPERIMENT E4: Cross-Suite Performance")
    print("="*60)

    detectors = {
        'PELT_l2_pen3': get_all_detectors()['PELT_l2_pen3.0'],
        'MeanShift_2.5': get_all_detectors()['MeanShift_2.5']
    }

    # Group by repository
    repo_labels = [m.get('repository', 'unknown') for m in metadata]

    results = evaluate_by_group(detectors, signals, true_points, repo_labels, tolerance)
    results.to_csv(output_dir / 'reports' / 'E4_cross_repo_performance.csv', index=False)

    print("\nF1 by repository:")
    pivot = results.pivot_table(values='f1_score', index='algorithm', columns='group')
    print(pivot.to_string())

    return results


def main():
    """Main execution function."""
    print("\n" + "#"*60)
    print("PHASE 4: Change-Point Detection Benchmarking")
    print("#"*60)
    print(f"Started at: {datetime.now().isoformat()}")

    if not HAS_RUPTURES:
        print("WARNING: ruptures library not installed. Installing...")
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'ruptures'], check=True)
        import ruptures as rpt

    set_random_seeds(RANDOM_SEED)

    # Output directory
    output_dir = PHASE_4_DIR / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)
    (output_dir / 'reports').mkdir(exist_ok=True)

    # ========================================
    # Build Signature Index
    # ========================================
    print("\n" + "="*60)
    print("BUILDING SIGNATURE INDEX")
    print("="*60)

    signature_index = build_signature_index()

    # ========================================
    # Load and Filter Alerts
    # ========================================
    print("\n" + "="*60)
    print("LOADING REGRESSION ALERTS")
    print("="*60)

    alerts_df = load_alerts_with_signatures()
    alerts_df = alerts_df.dropna(subset=[REGRESSION_TARGET_COL])

    # ========================================
    # Extract Time Series Windows
    # ========================================
    print("\n" + "="*60)
    print("EXTRACTING TIME SERIES WINDOWS")
    print("="*60)

    WINDOW_BEFORE = 30
    WINDOW_AFTER = 20
    MAX_SERIES = 2000  # Limit for efficiency

    signals_raw, true_points, metadata = extract_regression_series(
        alerts_df, signature_index,
        window_before=WINDOW_BEFORE,
        window_after=WINDOW_AFTER,
        max_series=MAX_SERIES
    )

    # Preprocess signals
    signals = [preprocess_signal(s, smooth=False, normalize=True) for s in signals_raw]

    print(f"Processed {len(signals)} time series")

    # ========================================
    # Run Experiments
    # ========================================
    TOLERANCE = 5
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'n_series': len(signals),
        'window_before': WINDOW_BEFORE,
        'window_after': WINDOW_AFTER,
        'tolerance': TOLERANCE
    }

    # E1: Algorithm comparison
    e1_results = run_experiment_E1(signals, true_points, output_dir, TOLERANCE)
    all_results['E1'] = e1_results.to_dict(orient='records')

    # E2: Tolerance sensitivity
    e2_results = run_experiment_E2(signals, true_points, output_dir, TOLERANCE)
    all_results['E2'] = e2_results.to_dict(orient='records')

    # E3: Smoothing sensitivity
    e3_results = run_experiment_E3(signals_raw, true_points, output_dir, TOLERANCE)
    all_results['E3'] = e3_results.to_dict(orient='records')

    # E4: Cross-suite performance
    e4_results = run_experiment_E4(signals, true_points, metadata, output_dir, TOLERANCE)
    all_results['E4'] = e4_results.to_dict(orient='records')

    # ========================================
    # Save Results
    # ========================================
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    save_results(all_results, output_dir / 'reports', 'experiment_summary')

    print(f"\nPhase 4 complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Finished at: {datetime.now().isoformat()}")

    return all_results


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Phase 4 failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
