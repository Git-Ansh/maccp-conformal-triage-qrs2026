#!/usr/bin/env python3
"""
Phase 5: Forecasting-Based Regression Detection - Run Script
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

from src.models import get_all_forecasters, HAS_TF
from src.anomaly_detector import (
    evaluate_forecast_detection, find_optimal_threshold,
    compute_detection_metrics
)

from phase_3.src.timeseries_loader import (
    build_signature_index, load_timeseries_from_zip
)
from phase_3.src.alert_timeseries_matcher import (
    load_alerts_with_signatures, find_alert_index_in_timeseries
)

from common.data_paths import (
    PHASE_5_DIR, REGRESSION_TARGET_COL, SIGNATURE_ID_COL, RANDOM_SEED
)
from common.model_utils import save_results, set_random_seeds

warnings.filterwarnings('ignore')


def extract_forecast_data(
    alerts_df: pd.DataFrame,
    signature_index: dict,
    window_size: int = 30,
    horizon: int = 5,
    max_samples: int = 1000
) -> tuple:
    """
    Extract time series windows for forecasting evaluation.

    Returns:
        Tuple of (series_list, labels, metadata)
    """
    series_list = []
    labels = []
    metadata_list = []

    count = 0
    for sig_id, group in alerts_df.groupby(SIGNATURE_ID_COL):
        if count >= max_samples:
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
            if count >= max_samples:
                break

            alert_idx = find_alert_index_in_timeseries(ts_df, alert_row)
            if alert_idx is None:
                continue

            # Need enough data before and after
            if alert_idx < window_size or alert_idx + horizon > len(values):
                continue

            # Extract window including some post-alert data for evaluation
            series = values[alert_idx - window_size:alert_idx + horizon]

            if len(series) < window_size + horizon:
                continue

            series_list.append(series)
            labels.append(int(alert_row.get(REGRESSION_TARGET_COL, 0)))
            metadata_list.append({
                'signature_id': sig_id,
                'alert_id': alert_row.get('single_alert_id'),
                'repository': alert_row.get('alert_summary_repository', 'unknown')
            })

            count += 1

        del ts_df
        gc.collect()

    print(f"Extracted {len(series_list)} series for forecasting")
    print(f"  Regressions: {sum(labels)}")
    print(f"  Non-regressions: {len(labels) - sum(labels)}")

    return series_list, labels, metadata_list


def run_experiment_E1(series_list, labels, window_size, horizon, output_dir):
    """
    E1: Forecast-based detection vs ground truth
    """
    print("\n" + "="*60)
    print("EXPERIMENT E1: Forecast Detection vs Ground Truth")
    print("="*60)

    forecasters = get_all_forecasters(include_neural=False)  # Start without neural
    results = []

    for name, forecaster in forecasters.items():
        print(f"  Evaluating {name}...")

        try:
            # Compute scores for all series
            scores = []
            for series in series_list:
                train = series[:window_size]
                actual = series[window_size:window_size + horizon]

                try:
                    forecaster.fit(train)
                    forecast = forecaster.predict(horizon)
                    residual = np.mean(np.abs(actual - forecast))
                    train_std = np.std(train)
                    score = residual / train_std if train_std > 0 else residual
                    scores.append(score)
                except:
                    scores.append(0)

            scores = np.array(scores)
            y_true = np.array(labels)

            # Find optimal threshold
            opt_thresh, metrics = find_optimal_threshold(scores, y_true)
            metrics['model'] = name
            metrics['optimal_threshold'] = opt_thresh
            results.append(metrics)

            print(f"    F1={metrics['f1_score']:.4f}, AUC={metrics.get('roc_auc', 0):.4f}")

        except Exception as e:
            print(f"    Error: {e}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1_score', ascending=False)
    results_df.to_csv(output_dir / 'reports' / 'E1_forecast_detection.csv', index=False)

    print("\nTop models:")
    print(results_df[['model', 'precision', 'recall', 'f1_score']].head().to_string())

    return results_df


def run_experiment_E2(series_list, labels, window_size, horizon, output_dir):
    """
    E2: Comparison of forecasting models
    """
    print("\n" + "="*60)
    print("EXPERIMENT E2: Model Comparison")
    print("="*60)

    # Include neural if available
    forecasters = get_all_forecasters(include_neural=HAS_TF)
    results = []

    for name, forecaster in forecasters.items():
        print(f"  Training {name}...")

        try:
            # Compute forecast errors
            mae_list = []
            for series in series_list[:200]:  # Limit for speed
                train = series[:window_size]
                actual = series[window_size:window_size + horizon]

                try:
                    forecaster.fit(train)
                    forecast = forecaster.predict(horizon)
                    mae = np.mean(np.abs(actual - forecast))
                    mae_list.append(mae)
                except:
                    pass

            if mae_list:
                results.append({
                    'model': name,
                    'mae_mean': np.mean(mae_list),
                    'mae_std': np.std(mae_list),
                    'mae_median': np.median(mae_list),
                    'n_evaluated': len(mae_list)
                })
                print(f"    MAE={np.mean(mae_list):.4f}")

        except Exception as e:
            print(f"    Error: {e}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mae_mean')
    results_df.to_csv(output_dir / 'reports' / 'E2_model_comparison.csv', index=False)

    return results_df


def run_experiment_E3(series_list, labels, output_dir):
    """
    E3: Effect of window size
    """
    print("\n" + "="*60)
    print("EXPERIMENT E3: Window Size Effect")
    print("="*60)

    from src.models import RandomForestForecaster

    results = []
    horizon = 3

    for window_size in [15, 20, 30, 40]:
        print(f"  Window size = {window_size}...")

        forecaster = RandomForestForecaster(n_lags=min(10, window_size - 5))

        scores = []
        valid_labels = []

        for series, label in zip(series_list, labels):
            if len(series) < window_size + horizon:
                continue

            train = series[:window_size]
            actual = series[window_size:window_size + horizon]

            try:
                forecaster.fit(train)
                forecast = forecaster.predict(horizon)
                residual = np.mean(np.abs(actual - forecast))
                train_std = np.std(train)
                score = residual / train_std if train_std > 0 else residual
                scores.append(score)
                valid_labels.append(label)
            except:
                pass

        if scores:
            scores = np.array(scores)
            y_true = np.array(valid_labels)
            opt_thresh, metrics = find_optimal_threshold(scores, y_true)
            metrics['window_size'] = window_size
            results.append(metrics)
            print(f"    F1={metrics['f1_score']:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'reports' / 'E3_window_size.csv', index=False)

    return results_df


def run_experiment_E4(series_list, labels, metadata, window_size, horizon, output_dir):
    """
    E4: Repository-level generalization
    """
    print("\n" + "="*60)
    print("EXPERIMENT E4: Cross-Repository Performance")
    print("="*60)

    from src.models import RandomForestForecaster

    # Group by repository
    repo_data = {}
    for series, label, meta in zip(series_list, labels, metadata):
        repo = meta.get('repository', 'unknown')
        if repo not in repo_data:
            repo_data[repo] = {'series': [], 'labels': []}
        repo_data[repo]['series'].append(series)
        repo_data[repo]['labels'].append(label)

    results = []
    forecaster = RandomForestForecaster(n_lags=10)

    for repo, data in repo_data.items():
        if len(data['series']) < 20:
            continue

        print(f"  Repository: {repo} ({len(data['series'])} samples)")

        scores = []
        for series in data['series']:
            if len(series) < window_size + horizon:
                scores.append(0)
                continue

            train = series[:window_size]
            actual = series[window_size:window_size + horizon]

            try:
                forecaster.fit(train)
                forecast = forecaster.predict(horizon)
                residual = np.mean(np.abs(actual - forecast))
                train_std = np.std(train)
                scores.append(residual / train_std if train_std > 0 else residual)
            except:
                scores.append(0)

        scores = np.array(scores)
        y_true = np.array(data['labels'])

        opt_thresh, metrics = find_optimal_threshold(scores, y_true)
        metrics['repository'] = repo
        metrics['n_samples'] = len(data['series'])
        results.append(metrics)

        print(f"    F1={metrics['f1_score']:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'reports' / 'E4_cross_repo.csv', index=False)

    return results_df


def main():
    """Main execution function."""
    print("\n" + "#"*60)
    print("PHASE 5: Forecasting-Based Regression Detection")
    print("#"*60)
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"TensorFlow available: {HAS_TF}")

    set_random_seeds(RANDOM_SEED)

    # Output directory
    output_dir = PHASE_5_DIR / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)
    (output_dir / 'reports').mkdir(exist_ok=True)
    (output_dir / 'models').mkdir(exist_ok=True)

    # ========================================
    # Build Signature Index
    # ========================================
    print("\n" + "="*60)
    print("BUILDING SIGNATURE INDEX")
    print("="*60)

    signature_index = build_signature_index()

    # ========================================
    # Load Alerts
    # ========================================
    print("\n" + "="*60)
    print("LOADING ALERTS")
    print("="*60)

    alerts_df = load_alerts_with_signatures()
    alerts_df = alerts_df.dropna(subset=[REGRESSION_TARGET_COL])

    # ========================================
    # Extract Forecast Data
    # ========================================
    print("\n" + "="*60)
    print("EXTRACTING FORECAST DATA")
    print("="*60)

    WINDOW_SIZE = 30
    HORIZON = 5
    MAX_SAMPLES = 1500

    series_list, labels, metadata = extract_forecast_data(
        alerts_df, signature_index,
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
        max_samples=MAX_SAMPLES
    )

    # ========================================
    # Run Experiments
    # ========================================
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': len(series_list),
        'window_size': WINDOW_SIZE,
        'horizon': HORIZON,
        'has_tensorflow': HAS_TF
    }

    # E1: Forecast detection
    e1_results = run_experiment_E1(
        series_list, labels, WINDOW_SIZE, HORIZON, output_dir
    )
    all_results['E1'] = e1_results.to_dict(orient='records')

    # E2: Model comparison
    e2_results = run_experiment_E2(
        series_list, labels, WINDOW_SIZE, HORIZON, output_dir
    )
    all_results['E2'] = e2_results.to_dict(orient='records')

    # E3: Window size effect
    e3_results = run_experiment_E3(series_list, labels, output_dir)
    all_results['E3'] = e3_results.to_dict(orient='records')

    # E4: Cross-repository
    e4_results = run_experiment_E4(
        series_list, labels, metadata, WINDOW_SIZE, HORIZON, output_dir
    )
    all_results['E4'] = e4_results.to_dict(orient='records')

    # ========================================
    # Save Results
    # ========================================
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    save_results(all_results, output_dir / 'reports', 'experiment_summary')

    print(f"\nPhase 5 complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Finished at: {datetime.now().isoformat()}")

    return all_results


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Phase 5 failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
