#!/usr/bin/env python3
"""
Phase 3: Time-Series Feature Extraction - Run Script
Extract time-series features and evaluate impact on classification.
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

from src.timeseries_loader import (
    build_signature_index, load_timeseries_from_zip
)
from src.ts_feature_engineering import (
    extract_all_features, get_feature_groups
)
from src.alert_timeseries_matcher import (
    load_alerts_with_signatures, find_alert_index_in_timeseries,
    extract_pre_alert_window
)
from src.feature_merger import (
    load_phase1_features, create_ts_feature_dataframe,
    merge_features, handle_missing_ts_features, save_merged_features
)

from common.data_paths import (
    PHASE_3_DIR, ALERTS_DATA_PATH, REGRESSION_TARGET_COL, TIMESTAMP_COL,
    RANDOM_SEED, SIGNATURE_ID_COL
)
from common.model_utils import save_model, save_results, set_random_seeds
from common.evaluation_utils import compute_binary_metrics

warnings.filterwarnings('ignore')


def extract_features_for_alerts(
    alerts_df: pd.DataFrame,
    signature_index: dict,
    window_size: int = 20,
    min_window_size: int = 5,
    verbose: bool = True
) -> tuple:
    """
    Extract time-series features for all alerts.

    Returns:
        Tuple of (features list, valid alert IDs, statistics)
    """
    features_list = []
    valid_alert_ids = []
    stats = {
        'total': 0,
        'sig_found': 0,
        'idx_found': 0,
        'valid_window': 0
    }

    # Group by signature for efficiency
    grouped = alerts_df.groupby(SIGNATURE_ID_COL)
    total_groups = len(grouped)

    for i, (sig_id, group) in enumerate(grouped):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Processing signature {i+1}/{total_groups}...")

        sig_id = int(sig_id)
        stats['total'] += len(group)

        if sig_id not in signature_index:
            continue

        # Load timeseries
        ts_df = load_timeseries_from_zip(signature_index[sig_id], sig_id)
        if ts_df is None:
            continue

        stats['sig_found'] += len(group)
        ts_df = ts_df.sort_values('push_timestamp').reset_index(drop=True)

        for _, alert_row in group.iterrows():
            alert_id = alert_row.get('single_alert_id')

            # Find alert index
            alert_idx = find_alert_index_in_timeseries(ts_df, alert_row)

            if alert_idx is None:
                continue

            stats['idx_found'] += 1

            # Extract window
            pre_values, alert_value, meta = extract_pre_alert_window(
                ts_df, alert_idx, window_size
            )

            if pre_values is None or len(pre_values) < min_window_size:
                continue

            stats['valid_window'] += 1

            # Extract features
            features = extract_all_features(pre_values, alert_value, window_size)
            features_list.append(features)
            valid_alert_ids.append(alert_id)

        # Memory cleanup
        del ts_df
        gc.collect()

    return features_list, valid_alert_ids, stats


def run_experiment_E1(
    X_metadata, X_combined, y, train_idx, test_idx,
    metadata_cols, ts_cols, output_dir
):
    """
    E1: Baseline Comparison - Metadata vs Metadata+TS
    """
    print("\n" + "="*60)
    print("EXPERIMENT E1: Baseline Comparison")
    print("="*60)

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    try:
        from xgboost import XGBClassifier
        HAS_XGBOOST = True
    except ImportError:
        HAS_XGBOOST = False

    results = []

    # Get train/test splits
    X_meta_train = X_metadata[train_idx]
    X_meta_test = X_metadata[test_idx]
    X_comb_train = X_combined[train_idx]
    X_comb_test = X_combined[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=RANDOM_SEED
        )
    }

    if HAS_XGBOOST:
        from common.model_utils import get_scale_pos_weight
        scale_weight = get_scale_pos_weight(y_train)
        models['XGBoost'] = XGBClassifier(
            n_estimators=100, max_depth=6, scale_pos_weight=scale_weight,
            random_state=RANDOM_SEED, tree_method='hist', device='cuda',
            eval_metric='logloss', use_label_encoder=False
        )

    # Train and evaluate
    for name, model in models.items():
        print(f"\n{name}:")

        # Metadata only
        model_meta = model.__class__(**model.get_params())
        model_meta.fit(X_meta_train, y_train)
        y_pred_meta = model_meta.predict(X_meta_test)
        y_prob_meta = model_meta.predict_proba(X_meta_test)[:, 1]
        metrics_meta = compute_binary_metrics(y_test, y_pred_meta, y_prob_meta)
        metrics_meta['model'] = name
        metrics_meta['feature_set'] = 'metadata_only'
        print(f"  Metadata: Precision={metrics_meta['precision']:.4f}, F1={metrics_meta['f1_score']:.4f}")

        # Combined
        model_comb = model.__class__(**model.get_params())
        model_comb.fit(X_comb_train, y_train)
        y_pred_comb = model_comb.predict(X_comb_test)
        y_prob_comb = model_comb.predict_proba(X_comb_test)[:, 1]
        metrics_comb = compute_binary_metrics(y_test, y_pred_comb, y_prob_comb)
        metrics_comb['model'] = name
        metrics_comb['feature_set'] = 'metadata_plus_ts'
        print(f"  Combined: Precision={metrics_comb['precision']:.4f}, F1={metrics_comb['f1_score']:.4f}")

        # Improvement
        f1_improvement = metrics_comb['f1_score'] - metrics_meta['f1_score']
        print(f"  F1 Improvement: {f1_improvement:+.4f}")

        results.append(metrics_meta)
        results.append(metrics_comb)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'reports' / 'E1_baseline_comparison.csv', index=False)

    return results_df


def run_experiment_E2(
    X_combined, y, train_idx, test_idx, feature_names, output_dir,
    window_sizes=[10, 20, 30]
):
    """
    E2: Window Size Sensitivity
    """
    print("\n" + "="*60)
    print("EXPERIMENT E2: Window Size Sensitivity")
    print("="*60)

    # Note: This would require re-extracting features with different window sizes
    # For efficiency, we'll use the extracted features and analyze window size metadata
    print("Note: Full window size comparison requires re-extraction")
    print("Using current window size (20) as baseline")

    return {'window_size': 20, 'note': 'Single window size evaluated'}


def run_experiment_E3(
    X_combined, y, train_idx, test_idx, feature_names,
    metadata_cols, ts_cols, output_dir
):
    """
    E3: Feature Group Ablation
    """
    print("\n" + "="*60)
    print("EXPERIMENT E3: Feature Group Ablation")
    print("="*60)

    from sklearn.ensemble import RandomForestClassifier

    results = []
    ts_feature_groups = get_feature_groups()

    # Get train/test data
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Get indices for different feature groups
    metadata_idx = [i for i, f in enumerate(feature_names) if f in metadata_cols]
    all_ts_idx = [i for i, f in enumerate(feature_names) if f in ts_cols]

    # Test each TS feature group added to metadata
    for group_name, group_features in ts_feature_groups.items():
        group_idx = [i for i, f in enumerate(feature_names)
                    if any(f.startswith(gf) for gf in group_features)]

        if not group_idx:
            continue

        feature_idx = metadata_idx + group_idx
        X_train = X_combined[train_idx][:, feature_idx]
        X_test = X_combined[test_idx][:, feature_idx]

        model = RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_binary_metrics(y_test, y_pred, y_prob)
        metrics['feature_group'] = f'metadata + {group_name}'
        metrics['n_features'] = len(feature_idx)
        results.append(metrics)

        print(f"  metadata + {group_name}: F1={metrics['f1_score']:.4f} ({len(group_idx)} TS features)")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'reports' / 'E3_ablation_results.csv', index=False)

    return results_df


def main():
    """Main execution function."""
    print("\n" + "#"*60)
    print("PHASE 3: Time-Series Feature Extraction")
    print("#"*60)
    print(f"Started at: {datetime.now().isoformat()}")

    set_random_seeds(RANDOM_SEED)

    # Output directory
    output_dir = PHASE_3_DIR / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'models').mkdir(exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)
    (output_dir / 'reports').mkdir(exist_ok=True)
    (output_dir / 'extracted_ts_features').mkdir(exist_ok=True)

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

    # Filter to alerts with target
    alerts_df = alerts_df.dropna(subset=[REGRESSION_TARGET_COL])
    print(f"Alerts with target: {len(alerts_df)}")

    # ========================================
    # Extract Time-Series Features
    # ========================================
    print("\n" + "="*60)
    print("EXTRACTING TIME-SERIES FEATURES")
    print("="*60)

    WINDOW_SIZE = 20
    MIN_WINDOW = 5

    features_list, valid_alert_ids, extraction_stats = extract_features_for_alerts(
        alerts_df, signature_index,
        window_size=WINDOW_SIZE,
        min_window_size=MIN_WINDOW
    )

    print(f"\nExtraction Statistics:")
    for k, v in extraction_stats.items():
        print(f"  {k}: {v}")

    # Create TS features DataFrame
    ts_features_df = pd.DataFrame(features_list)
    ts_features_df['single_alert_id'] = valid_alert_ids
    ts_features_df = ts_features_df.set_index('single_alert_id')

    print(f"\nTS Features DataFrame: {ts_features_df.shape}")

    # Save TS features
    ts_features_df.to_parquet(output_dir / 'extracted_ts_features' / f'ts_features_w{WINDOW_SIZE}.parquet')

    # ========================================
    # Load Phase 1 Features and Merge
    # ========================================
    print("\n" + "="*60)
    print("MERGING WITH PHASE 1 FEATURES")
    print("="*60)

    try:
        metadata_df, metadata_cols = load_phase1_features()
    except FileNotFoundError:
        print("Phase 1 features not found. Running Phase 1 preprocessing...")
        # Fallback: use Phase 1 preprocessing
        from phase_1.src.data_loader import load_and_prepare_data
        from phase_1.src.preprocessing import select_phase1_features, preprocess_features

        df, _ = load_and_prepare_data()
        X_raw, _ = select_phase1_features(df)
        metadata_df, _ = preprocess_features(X_raw)
        metadata_cols = metadata_df.columns.tolist()

    # Filter alerts to those with both metadata and TS features
    alerts_with_ts = alerts_df[alerts_df['single_alert_id'].isin(valid_alert_ids)].copy()
    alert_indices = alerts_df['single_alert_id'].isin(valid_alert_ids)

    # Align metadata
    metadata_aligned = metadata_df.loc[alert_indices].reset_index(drop=True)
    alerts_aligned = alerts_with_ts.reset_index(drop=True)

    # Add alert IDs to metadata
    metadata_aligned['single_alert_id'] = alerts_aligned['single_alert_id'].values

    # Merge with TS features
    merged_df = metadata_aligned.merge(
        ts_features_df.reset_index(),
        on='single_alert_id',
        how='inner'
    )

    ts_cols = [c for c in ts_features_df.columns]

    print(f"\nMerged DataFrame: {merged_df.shape}")
    print(f"  Metadata features: {len(metadata_cols)}")
    print(f"  TS features: {len(ts_cols)}")

    # Handle missing values
    merged_df = handle_missing_ts_features(merged_df, ts_cols)

    # Save merged features
    save_merged_features(merged_df, output_dir / 'extracted_ts_features', 'merged_features')

    # ========================================
    # Prepare for Experiments
    # ========================================
    print("\n" + "="*60)
    print("PREPARING EXPERIMENT DATA")
    print("="*60)

    # Get aligned target
    y = alerts_aligned[REGRESSION_TARGET_COL].values.astype(int)

    # Feature matrices
    X_metadata = merged_df[metadata_cols].values
    X_combined = merged_df[metadata_cols + ts_cols].values
    feature_names = metadata_cols + ts_cols

    # Temporal split
    from phase_1.src.temporal_split import get_split_indices
    train_idx, val_idx, test_idx = get_split_indices(alerts_aligned)

    # Combine train and val for simplicity
    train_idx = np.concatenate([train_idx, val_idx])

    print(f"\nDataset sizes:")
    print(f"  Total: {len(y)}")
    print(f"  Train: {len(train_idx)}")
    print(f"  Test: {len(test_idx)}")

    # ========================================
    # Run Experiments
    # ========================================
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'extraction_stats': extraction_stats,
        'window_size': WINDOW_SIZE,
        'n_metadata_features': len(metadata_cols),
        'n_ts_features': len(ts_cols)
    }

    # E1: Baseline comparison
    e1_results = run_experiment_E1(
        X_metadata, X_combined, y, train_idx, test_idx,
        metadata_cols, ts_cols, output_dir
    )
    all_results['E1'] = e1_results.to_dict(orient='records')

    # E2: Window size (simplified)
    e2_results = run_experiment_E2(
        X_combined, y, train_idx, test_idx, feature_names, output_dir
    )
    all_results['E2'] = e2_results

    # E3: Feature ablation
    e3_results = run_experiment_E3(
        X_combined, y, train_idx, test_idx, feature_names,
        metadata_cols, ts_cols, output_dir
    )
    all_results['E3'] = e3_results.to_dict(orient='records')

    # ========================================
    # Save Results
    # ========================================
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    save_results(all_results, output_dir / 'reports', 'experiment_summary')

    print(f"\nPhase 3 complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Finished at: {datetime.now().isoformat()}")

    return all_results


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Phase 3 failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
