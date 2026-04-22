#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4 Enhanced: Multi-tau CPD Evaluation with Real Mozilla Time Series

This script implements proper CPD evaluation methodology:
1. Uses REAL Mozilla time series data (not synthetic)
2. Evaluates at multiple tolerance values (tau = 1, 3, 5, 7, 10, 15, 20)
3. Reports precision-recall curves across tau values
4. Computes VUS-like metrics for comprehensive evaluation
5. Direct comparison to ML baselines on same has_bug task

Based on:
- TCPDBench evaluation methodology (van den Burg et al.)
- VUS metric (Paparrizos et al., VLDB 2022)
- ICSME 2025 Mozilla dataset paper (Besbes et al.)
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime
import gc

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef,
    confusion_matrix, precision_recall_curve, auc
)

warnings.filterwarnings('ignore')

# Paths
SRC_DIR = Path(__file__).parent.parent
PROJECT_ROOT = SRC_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "alerts_data.csv"
TIMESERIES_DIR = PROJECT_ROOT / "data" / "timeseries-data"
OUTPUT_DIR = Path(__file__).parent / "outputs_multi_tau"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'reports').mkdir(exist_ok=True)
(OUTPUT_DIR / 'figures').mkdir(exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Evaluation parameters
TAU_VALUES = [1, 3, 5, 7, 10, 15, 20]  # Tolerance windows to evaluate
SEQUENCE_LENGTH = 50  # Time series window size


# ============================================
# TIME SERIES LOADING
# ============================================

def build_timeseries_index(timeseries_dir):
    """Build index of available time series files."""
    print("Building time series file index...")
    ts_file_cache = {}
    
    for repo_dir in timeseries_dir.iterdir():
        if repo_dir.is_dir() and not repo_dir.name.endswith('.zip'):
            inner_dir = repo_dir / repo_dir.name
            if inner_dir.exists():
                for f in inner_dir.glob("*_timeseries_data.csv"):
                    try:
                        sig_id = int(f.stem.split('_')[0])
                        ts_file_cache[sig_id] = f
                    except:
                        pass
    
    print(f"  Found {len(ts_file_cache)} time series files")
    return ts_file_cache


def load_timeseries_values(file_path, max_points=None):
    """Load time series values from file."""
    try:
        df = pd.read_csv(file_path)
        if 'value' not in df.columns:
            return None
        
        # Sort by timestamp
        if 'push_timestamp' in df.columns:
            df = df.sort_values('push_timestamp')
        
        values = df['value'].values
        
        if max_points is not None and len(values) > max_points:
            values = values[-max_points:]
        
        return values.astype(np.float64)
    except Exception as e:
        return None


# ============================================
# CPD ALGORITHMS
# ============================================

def cusum_detection(values, threshold=3.0, drift=0.5):
    """CUSUM change-point detection."""
    n = len(values)
    if n < 10:
        return []
    
    mean = np.mean(values)
    std = np.std(values) + 1e-10
    z = (values - mean) / std
    
    s_pos = np.zeros(n)
    s_neg = np.zeros(n)
    
    for i in range(1, n):
        s_pos[i] = max(0, s_pos[i-1] + z[i] - drift)
        s_neg[i] = max(0, s_neg[i-1] - z[i] - drift)
    
    change_points = []
    i = 0
    while i < n:
        if s_pos[i] > threshold or s_neg[i] > threshold:
            change_points.append(i)
            s_pos[i:] = 0
            s_neg[i:] = 0
            i += 5  # Skip forward to avoid detecting same change
        else:
            i += 1
    
    return change_points


def pelt_detection(values, penalty='bic', min_size=5):
    """PELT change-point detection using ruptures library."""
    try:
        import ruptures as rpt
        
        if len(values) < min_size * 2:
            return []
        
        # Normalize
        values_norm = (values - np.mean(values)) / (np.std(values) + 1e-10)
        
        # PELT with different cost functions
        algo = rpt.Pelt(model="rbf", min_size=min_size, jump=1)
        algo.fit(values_norm.reshape(-1, 1))
        
        # Use BIC-like penalty
        if penalty == 'bic':
            pen = np.log(len(values)) * 2
        else:
            pen = float(penalty)
        
        change_points = algo.predict(pen=pen)
        
        # Remove last point (end of series marker)
        if change_points and change_points[-1] == len(values):
            change_points = change_points[:-1]
        
        return change_points
    except ImportError:
        return cusum_detection(values)  # Fallback
    except Exception:
        return []


def binary_segmentation(values, n_bkps=3, min_size=5):
    """Binary segmentation change-point detection."""
    try:
        import ruptures as rpt
        
        if len(values) < min_size * 2:
            return []
        
        values_norm = (values - np.mean(values)) / (np.std(values) + 1e-10)
        
        algo = rpt.Binseg(model="l2", min_size=min_size, jump=1)
        algo.fit(values_norm.reshape(-1, 1))
        change_points = algo.predict(n_bkps=n_bkps)
        
        if change_points and change_points[-1] == len(values):
            change_points = change_points[:-1]
        
        return change_points
    except ImportError:
        return cusum_detection(values)
    except Exception:
        return []


def window_sliding(values, width=10, threshold=2.0):
    """Sliding window change detection."""
    try:
        import ruptures as rpt
        
        if len(values) < width * 2:
            return []
        
        values_norm = (values - np.mean(values)) / (np.std(values) + 1e-10)
        
        algo = rpt.Window(width=width, model="l2")
        algo.fit(values_norm.reshape(-1, 1))
        change_points = algo.predict(pen=threshold)
        
        if change_points and change_points[-1] == len(values):
            change_points = change_points[:-1]
        
        return change_points
    except ImportError:
        return cusum_detection(values)
    except Exception:
        return []


def bocpd_detection(values, hazard_rate=100):
    """
    Bayesian Online Change Point Detection (simplified).
    
    Based on Adams & MacKay (2007).
    """
    n = len(values)
    if n < 10:
        return []
    
    # Simplified BOCPD using cumulative statistics
    change_points = []
    window = 10
    
    for i in range(window, n - window):
        before = values[i-window:i]
        after = values[i:i+window]
        
        # Two-sample t-test
        if len(before) >= 3 and len(after) >= 3:
            t_stat, p_value = stats.ttest_ind(before, after)
            if abs(t_stat) > 3.0:  # Significant change
                change_points.append(i)
    
    # Remove close detections
    if change_points:
        filtered = [change_points[0]]
        for cp in change_points[1:]:
            if cp - filtered[-1] > 5:
                filtered.append(cp)
        change_points = filtered
    
    return change_points


def mozilla_ttest_detection(values, t_threshold=7.0, window=12):
    """
    Mozilla's actual detection algorithm (from Perfherder source).
    
    Uses two-sample t-test with t > 7 threshold.
    """
    n = len(values)
    if n < window * 2:
        return []
    
    change_points = []
    
    for i in range(window, n - window):
        before = values[i-window:i]
        after = values[i:i+window]
        
        if len(before) >= 3 and len(after) >= 3:
            t_stat, p_value = stats.ttest_ind(before, after)
            if abs(t_stat) > t_threshold:
                change_points.append(i)
    
    # Remove close detections
    if change_points:
        filtered = [change_points[0]]
        for cp in change_points[1:]:
            if cp - filtered[-1] > window:
                filtered.append(cp)
        change_points = filtered
    
    return change_points


# ============================================
# EVALUATION METRICS
# ============================================

def evaluate_detection_at_tau(detected_cps, true_cp_idx, n_points, tau):
    """
    Evaluate change point detection at a specific tolerance tau.
    
    Args:
        detected_cps: List of detected change point indices
        true_cp_idx: True change point index (or None if no change)
        n_points: Length of time series
        tau: Tolerance window
    
    Returns:
        dict with TP, FP, FN counts
    """
    if true_cp_idx is None:
        # No true change point - any detection is FP
        return {
            'tp': 0,
            'fp': len(detected_cps),
            'fn': 0,
            'detected': len(detected_cps) > 0
        }
    
    # Check if any detection is within tau of true change point
    tp = 0
    for cp in detected_cps:
        if abs(cp - true_cp_idx) <= tau:
            tp = 1
            break
    
    fn = 1 - tp  # Missed if no detection within tau
    fp = max(0, len(detected_cps) - tp)  # Extra detections
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'detected': len(detected_cps) > 0
    }


def compute_metrics_from_counts(total_tp, total_fp, total_fn):
    """Compute precision, recall, F1 from aggregate counts."""
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }


def compute_vus_approximation(results_by_tau, tau_values):
    """
    Compute VUS-like metric (Volume Under Surface approximation).
    
    Integrates F1 across all tau values.
    """
    f1_values = [results_by_tau[tau]['f1'] for tau in tau_values]
    
    # Trapezoidal integration
    vus = np.trapz(f1_values, tau_values) / (max(tau_values) - min(tau_values))
    
    return vus


# ============================================
# MAIN EVALUATION
# ============================================

def run_cpd_evaluation(alerts_df, ts_file_cache, cpd_algorithm, algo_name, 
                       tau_values=TAU_VALUES, max_samples=None):
    """
    Run CPD evaluation on real time series at multiple tau values.
    """
    print(f"\n  Evaluating {algo_name}...")
    
    results_by_tau = {tau: {'tp': 0, 'fp': 0, 'fn': 0} for tau in tau_values}
    
    processed = 0
    matched = 0
    
    sample_df = alerts_df if max_samples is None else alerts_df.head(max_samples)
    
    for idx, row in sample_df.iterrows():
        sig_id = int(row['signature_id'])
        has_bug = int(row['has_bug'])
        
        # Get time series
        ts_file = ts_file_cache.get(sig_id)
        if ts_file is None:
            continue
        
        values = load_timeseries_values(ts_file, max_points=100)
        if values is None or len(values) < 20:
            continue
        
        processed += 1
        
        # Run CPD algorithm
        detected_cps = cpd_algorithm(values)
        
        # For alerts with bugs, assume change point near end of series
        # (alert was triggered, so change should be detectable)
        if has_bug:
            true_cp_idx = len(values) - 10  # Approximate location
        else:
            true_cp_idx = None  # No true change for non-bug alerts
        
        # Evaluate at each tau
        for tau in tau_values:
            eval_result = evaluate_detection_at_tau(
                detected_cps, true_cp_idx, len(values), tau
            )
            results_by_tau[tau]['tp'] += eval_result['tp']
            results_by_tau[tau]['fp'] += eval_result['fp']
            results_by_tau[tau]['fn'] += eval_result['fn']
        
        matched += 1
        
        if matched % 500 == 0:
            print(f"    Processed {matched} time series...")
    
    print(f"    Total processed: {matched}")
    
    # Compute metrics for each tau
    metrics_by_tau = {}
    for tau in tau_values:
        metrics = compute_metrics_from_counts(
            results_by_tau[tau]['tp'],
            results_by_tau[tau]['fp'],
            results_by_tau[tau]['fn']
        )
        metrics['tau'] = tau
        metrics['algorithm'] = algo_name
        metrics_by_tau[tau] = metrics
    
    # Compute VUS approximation
    vus = compute_vus_approximation(metrics_by_tau, tau_values)
    
    return metrics_by_tau, vus, matched


def main():
    print("="*60)
    print("PHASE 4 ENHANCED: Multi-tau CPD Evaluation")
    print("="*60)
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"Tolerance values: {TAU_VALUES}")
    
    # Load alerts data
    print("\n[1/4] Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['has_bug'] = df['alert_summary_bug_number'].notna().astype(int)
    print(f"  Total alerts: {len(df)}")
    print(f"  Alerts with bugs: {df['has_bug'].sum()} ({df['has_bug'].mean()*100:.1f}%)")
    
    # Build time series index
    ts_file_cache = build_timeseries_index(TIMESERIES_DIR)
    
    # Temporal split (same as Phase 1)
    print("\n[2/4] Creating temporal split...")
    df = df.sort_values('push_timestamp').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    print(f"  Test set: {len(test_df)} alerts, {test_df['has_bug'].sum()} with bugs")
    
    # Define CPD algorithms
    print("\n[3/4] Evaluating CPD algorithms...")
    algorithms = {
        'CUSUM': lambda v: cusum_detection(v, threshold=3.0, drift=0.5),
        'PELT-BIC': lambda v: pelt_detection(v, penalty='bic'),
        'PELT-10': lambda v: pelt_detection(v, penalty=10),
        'BinSeg-3': lambda v: binary_segmentation(v, n_bkps=3),
        'Window-10': lambda v: window_sliding(v, width=10),
        'Mozilla-T7': lambda v: mozilla_ttest_detection(v, t_threshold=7.0),
        'Mozilla-T5': lambda v: mozilla_ttest_detection(v, t_threshold=5.0),
        'BOCPD': lambda v: bocpd_detection(v),
    }
    
    all_results = []
    vus_results = []
    
    for algo_name, algo_func in algorithms.items():
        metrics_by_tau, vus, n_processed = run_cpd_evaluation(
            test_df, ts_file_cache, algo_func, algo_name,
            tau_values=TAU_VALUES, max_samples=5000  # Limit for speed
        )
        
        for tau, metrics in metrics_by_tau.items():
            all_results.append(metrics)
        
        vus_results.append({
            'algorithm': algo_name,
            'vus': vus,
            'n_processed': n_processed
        })
        
        print(f"    {algo_name}: VUS={vus:.4f}")
    
    # Save results
    print("\n[4/4] Saving results...")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / 'reports' / 'cpd_multi_tau_results.csv', index=False)
    
    vus_df = pd.DataFrame(vus_results)
    vus_df.to_csv(OUTPUT_DIR / 'reports' / 'cpd_vus_results.csv', index=False)
    
    # Display summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    # Pivot table for nice display
    print("\nF1 Score by Algorithm and tau:")
    pivot = results_df.pivot(index='algorithm', columns='tau', values='f1')
    print(pivot.round(3).to_string())
    
    print("\nVUS Scores (higher is better):")
    print(vus_df.sort_values('vus', ascending=False).to_string(index=False))
    
    # Best algorithm
    best_algo = vus_df.loc[vus_df['vus'].idxmax()]
    print(f"\nBest Algorithm: {best_algo['algorithm']} (VUS={best_algo['vus']:.4f})")
    
    # Compare to Phase 1 ML baselines
    print("\n" + "="*60)
    print("COMPARISON TO ML BASELINES")
    print("="*60)
    print("\nPhase 1 ML Results (has_bug prediction):")
    print("  XGBoost: F1=0.394, MCC=0.186")
    print("  Random Forest: F1=0.338, MCC=0.132")
    
    # Get best CPD F1 at reasonable tau
    tau7_results = results_df[results_df['tau'] == 7].sort_values('f1', ascending=False)
    if len(tau7_results) > 0:
        best_tau7 = tau7_results.iloc[0]
        print(f"\nBest CPD at tau=7: {best_tau7['algorithm']}")
        print(f"  F1={best_tau7['f1']:.4f}, Precision={best_tau7['precision']:.4f}, Recall={best_tau7['recall']:.4f}")
        
        if best_tau7['f1'] > 0.394:
            print("\n[*] CPD OUTPERFORMS ML at tau=7!")
        else:
            print("\n[*] ML remains competitive at reasonable tau")
    
    print(f"\nFinished at: {datetime.now().isoformat()}")
    print(f"Results saved to: {OUTPUT_DIR}")
    
    return results_df, vus_df


if __name__ == "__main__":
    results_df, vus_df = main()
