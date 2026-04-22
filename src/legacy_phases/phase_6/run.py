#!/usr/bin/env python3
"""
Phase 6: Automated Root Cause Analysis - Run Script

Experiments:
E1: Metadata + time-series clustering
E2: Predictive models for bug filing
E3: Text-derived root-cause categories
E4: Downstream vs primary alert analysis
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

from src.feature_aggregator import (
    load_alerts_with_features, load_bug_data,
    merge_alert_bug_data, prepare_clustering_features
)
from src.clustering import (
    KMeansClusterer, DBSCANClusterer, HDBSCANClusterer,
    evaluate_clustering, get_cluster_profiles, reduce_dimensions,
    find_optimal_k
)
from src.bug_prediction import (
    BugPredictionModel, prepare_bug_prediction_data,
    cross_validate_bug_prediction
)
from src.text_analysis import (
    TextAnalyzer, extract_keywords, extract_topics_lda,
    extract_topics_nmf, categorize_by_keywords, MOZILLA_CATEGORIES
)
from src.downstream_analysis import (
    build_alert_graph, identify_downstream_alerts,
    analyze_downstream_clusters, get_summary_statistics
)

from common.data_paths import (
    PHASE_6_DIR, ALERTS_DATA_PATH, BUGS_DATA_PATH,
    REGRESSION_TARGET_COL, RANDOM_SEED, MAGNITUDE_FEATURES, CONTEXT_FEATURES
)
from common.model_utils import save_results, set_random_seeds

warnings.filterwarnings('ignore')


def run_experiment_E1(output_dir: Path) -> pd.DataFrame:
    """
    E1: Alert Clustering

    Cluster alerts based on metadata + time-series features.
    Evaluate cluster quality against bug IDs and components.
    """
    print("\n" + "="*60)
    print("EXPERIMENT E1: Alert Clustering")
    print("="*60)

    # Load data (without TS features for speed, can enable if needed)
    print("\nLoading alerts with metadata features...")
    alerts_df = pd.read_csv(ALERTS_DATA_PATH)

    # Prepare features
    feature_cols = []

    # Magnitude features
    for col in MAGNITUDE_FEATURES:
        if col in alerts_df.columns:
            feature_cols.append(col)

    # Encoded categorical features
    from sklearn.preprocessing import LabelEncoder
    for col in CONTEXT_FEATURES:
        if col in alerts_df.columns:
            le = LabelEncoder()
            alerts_df[f'{col}_encoded'] = le.fit_transform(
                alerts_df[col].fillna('unknown').astype(str)
            )
            feature_cols.append(f'{col}_encoded')

    print(f"Using {len(feature_cols)} features for clustering")

    # Prepare feature matrix
    X, _ = prepare_clustering_features(alerts_df, feature_cols)

    # Find optimal k
    print("\nFinding optimal number of clusters...")
    optimal_k, silhouette_scores = find_optimal_k(X, k_range=range(5, 31, 5))
    print(f"Optimal k (silhouette): {optimal_k}")

    # Run multiple clusterers
    results = []

    clusterers = [
        KMeansClusterer(n_clusters=10),
        KMeansClusterer(n_clusters=optimal_k),
        KMeansClusterer(n_clusters=20),
        DBSCANClusterer(eps=1.0, min_samples=5),
        DBSCANClusterer(eps=1.5, min_samples=5),
    ]

    # Try HDBSCAN if available
    try:
        clusterers.append(HDBSCANClusterer(min_cluster_size=50, min_samples=10))
    except ImportError:
        print("HDBSCAN not available, skipping")

    for clusterer in clusterers:
        print(f"\n  Running {clusterer.name}...")
        try:
            labels = clusterer.fit_predict(X)

            # Evaluate
            metrics = evaluate_clustering(X, labels)
            metrics['clusterer'] = clusterer.name
            metrics.update(clusterer.get_params())
            results.append(metrics)

            print(f"    Clusters: {metrics['n_clusters']}, Noise: {metrics.get('n_noise', 0)}")
            if 'silhouette' in metrics:
                print(f"    Silhouette: {metrics['silhouette']:.4f}")

        except Exception as e:
            print(f"    Error: {e}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('silhouette', ascending=False)
    results_df.to_csv(output_dir / 'reports' / 'E1_clustering_results.csv', index=False)

    print("\nTop clustering results:")
    print(results_df[['clusterer', 'n_clusters', 'silhouette', 'noise_ratio']].head())

    # Get best clusterer and save cluster assignments
    best_clusterer = KMeansClusterer(n_clusters=optimal_k)
    best_labels = best_clusterer.fit_predict(X)
    alerts_df['cluster'] = best_labels

    # Cluster profiles
    profile_cols = ['alert_summary_repository', 'single_alert_series_signature_suite']
    profiles = get_cluster_profiles(alerts_df, best_labels, profile_cols)
    profiles.to_csv(output_dir / 'reports' / 'E1_cluster_profiles.csv', index=False)

    return results_df


def run_experiment_E2(output_dir: Path) -> pd.DataFrame:
    """
    E2: Bug Prediction

    Predict which alerts will lead to bug reports.
    """
    print("\n" + "="*60)
    print("EXPERIMENT E2: Bug Prediction")
    print("="*60)

    # Load alerts and bugs
    print("\nLoading alerts...")
    alerts_df = pd.read_csv(ALERTS_DATA_PATH)
    print(f"  Loaded {len(alerts_df)} alerts")

    print("\nLoading bugs...")
    bugs_df = load_bug_data()

    # Merge to get has_bug label
    alerts_df = merge_alert_bug_data(alerts_df, bugs_df)

    # Prepare features
    feature_cols = []

    # Magnitude features
    for col in MAGNITUDE_FEATURES:
        if col in alerts_df.columns:
            feature_cols.append(col)

    # Encode categorical features
    from sklearn.preprocessing import LabelEncoder
    for col in CONTEXT_FEATURES:
        if col in alerts_df.columns:
            le = LabelEncoder()
            alerts_df[f'{col}_encoded'] = le.fit_transform(
                alerts_df[col].fillna('unknown').astype(str)
            )
            feature_cols.append(f'{col}_encoded')

    # Add regression label as feature
    if REGRESSION_TARGET_COL in alerts_df.columns:
        alerts_df['is_regression_encoded'] = alerts_df[REGRESSION_TARGET_COL].fillna(0).astype(int)
        feature_cols.append('is_regression_encoded')

    print(f"Using {len(feature_cols)} features")
    print(f"Bug ratio: {alerts_df['has_bug'].mean():.3f}")

    # Prepare data
    X_train, X_test, y_train, y_test, valid_cols = prepare_bug_prediction_data(
        alerts_df, feature_cols, test_size=0.2
    )

    # Cross-validate models
    print("\nCross-validating models...")
    cv_results = cross_validate_bug_prediction(
        X_train, y_train,
        model_types=['xgboost', 'rf', 'gbm', 'lr'],
        cv=5
    )
    cv_results.to_csv(output_dir / 'reports' / 'E2_cv_results.csv', index=False)

    # Train best model (XGBoost) and evaluate on test
    print("\nTraining final XGBoost model...")
    model = BugPredictionModel(model_type='xgboost')
    model.fit(X_train, y_train, feature_names=valid_cols)

    # Evaluate
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)

    print(f"\nTrain - Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}, F1: {train_metrics['f1_score']:.4f}")
    print(f"Test  - Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1_score']:.4f}")

    # Feature importance
    importance_df = model.get_feature_importance()
    importance_df.to_csv(output_dir / 'reports' / 'E2_feature_importance.csv', index=False)
    print("\nTop 10 features:")
    print(importance_df.head(10).to_string(index=False))

    # Save model
    model.save(output_dir / 'models' / 'bug_predictor.joblib')

    # Compile results
    results = [{
        'model': 'XGBoost',
        'train_precision': train_metrics['precision'],
        'train_recall': train_metrics['recall'],
        'train_f1': train_metrics['f1_score'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1_score'],
        'test_auc': test_metrics.get('roc_auc', 0)
    }]

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'reports' / 'E2_final_results.csv', index=False)

    return results_df


def run_experiment_E3(output_dir: Path) -> pd.DataFrame:
    """
    E3: Text Analysis

    Extract root-cause categories from bug summaries using NLP.
    """
    print("\n" + "="*60)
    print("EXPERIMENT E3: Text-Based Root Cause Analysis")
    print("="*60)

    # Load bugs
    bugs_df = load_bug_data()

    # Get texts
    summaries = bugs_df['summary'].fillna('').tolist()
    print(f"\nAnalyzing {len(summaries)} bug summaries")

    # 1. TF-IDF Analysis
    print("\n--- TF-IDF Analysis ---")
    analyzer = TextAnalyzer(max_features=500)
    tfidf_matrix = analyzer.fit_tfidf(summaries)

    top_terms = analyzer.get_top_terms(n_terms=20)
    print("Top 20 TF-IDF terms:")
    for term, score in top_terms[:10]:
        print(f"  {term}: {score:.4f}")

    # 2. Keyword Extraction
    print("\n--- Keyword Extraction ---")
    keywords = extract_keywords(summaries, n_keywords=30, method='tfidf')
    print("Top keywords:", keywords[:15])

    # 3. Topic Modeling (LDA)
    print("\n--- LDA Topic Modeling ---")
    doc_topics_lda, topic_words_lda = extract_topics_lda(summaries, n_topics=8, n_top_words=8)

    print("LDA Topics:")
    for i, words in enumerate(topic_words_lda):
        print(f"  Topic {i}: {', '.join(words[:5])}")

    # 4. NMF Topics
    print("\n--- NMF Topic Modeling ---")
    doc_topics_nmf, topic_words_nmf = extract_topics_nmf(summaries, n_topics=8, n_top_words=8)

    print("NMF Topics:")
    for i, words in enumerate(topic_words_nmf):
        print(f"  Topic {i}: {', '.join(words[:5])}")

    # 5. Keyword-Based Categorization
    print("\n--- Keyword-Based Categorization ---")
    categories = categorize_by_keywords(summaries, MOZILLA_CATEGORIES)
    bugs_df['rca_category'] = categories

    category_counts = pd.Series(categories).value_counts()
    print("Category distribution:")
    print(category_counts.to_string())

    # 6. Component Analysis
    print("\n--- Component Analysis ---")
    component_counts = bugs_df['component'].value_counts().head(15)
    print("Top 15 components:")
    print(component_counts.to_string())

    # 7. Text Clustering
    print("\n--- Text Clustering ---")
    text_labels = analyzer.cluster_texts(n_clusters=10)
    bugs_df['text_cluster'] = text_labels

    # Compile results
    results = {
        'top_tfidf_terms': [t[0] for t in top_terms[:20]],
        'top_keywords': keywords[:20],
        'lda_topics': topic_words_lda,
        'nmf_topics': topic_words_nmf,
        'category_distribution': category_counts.to_dict(),
        'component_distribution': component_counts.to_dict()
    }

    # Save detailed results
    topic_df = pd.DataFrame({
        'topic_id': range(len(topic_words_lda)),
        'lda_words': [', '.join(w) for w in topic_words_lda],
        'nmf_words': [', '.join(w) for w in topic_words_nmf]
    })
    topic_df.to_csv(output_dir / 'reports' / 'E3_topics.csv', index=False)

    category_df = pd.DataFrame({
        'category': category_counts.index,
        'count': category_counts.values
    })
    category_df.to_csv(output_dir / 'reports' / 'E3_categories.csv', index=False)

    # Save bugs with categories
    bugs_df.to_csv(output_dir / 'reports' / 'E3_bugs_categorized.csv', index=False)

    print(f"\nText analysis complete!")
    print(f"  {len(set(categories))} root-cause categories identified")
    print(f"  {len(topic_words_lda)} LDA topics extracted")

    return category_df


def run_experiment_E4(output_dir: Path) -> pd.DataFrame:
    """
    E4: Downstream Alert Analysis

    Analyze relationships between primary and downstream alerts.
    """
    print("\n" + "="*60)
    print("EXPERIMENT E4: Downstream Alert Analysis")
    print("="*60)

    # Load alerts
    alerts_df = pd.read_csv(ALERTS_DATA_PATH)

    # Get summary statistics
    print("\n--- Summary Statistics ---")
    stats = get_summary_statistics(alerts_df)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Identify primary vs downstream
    print("\n--- Identifying Primary/Downstream Alerts ---")
    alerts_df = identify_downstream_alerts(alerts_df)

    # Build alert graph
    print("\n--- Building Alert Graph ---")
    graph = build_alert_graph(alerts_df)

    # Prepare features for clustering
    feature_cols = []
    for col in MAGNITUDE_FEATURES:
        if col in alerts_df.columns:
            feature_cols.append(col)

    from sklearn.preprocessing import LabelEncoder
    for col in CONTEXT_FEATURES:
        if col in alerts_df.columns:
            le = LabelEncoder()
            alerts_df[f'{col}_encoded'] = le.fit_transform(
                alerts_df[col].fillna('unknown').astype(str)
            )
            feature_cols.append(f'{col}_encoded')

    X, _ = prepare_clustering_features(alerts_df, feature_cols)

    # Cluster alerts
    print("\n--- Clustering Alerts ---")
    clusterer = KMeansClusterer(n_clusters=15)
    labels = clusterer.fit_predict(X)

    # Analyze downstream clustering
    print("\n--- Analyzing Downstream Clusters ---")
    cluster_analysis = analyze_downstream_clusters(alerts_df, labels)

    if len(cluster_analysis) > 0:
        print(f"\nCluster coherence statistics:")
        print(f"  Mean coherence: {cluster_analysis['cluster_coherence'].mean():.3f}")
        print(f"  Median coherence: {cluster_analysis['cluster_coherence'].median():.3f}")
        print(f"  Summaries with perfect coherence: {cluster_analysis['all_same_cluster'].sum()}")

        cluster_analysis.to_csv(output_dir / 'reports' / 'E4_downstream_clustering.csv', index=False)

    # Compare primary vs downstream statistics
    print("\n--- Primary vs Downstream Comparison ---")
    primary_alerts = alerts_df[alerts_df['is_primary']]
    downstream_alerts = alerts_df[alerts_df['is_downstream']]

    comparison = {
        'metric': ['count', 'is_regression_mean', 'amount_abs_mean'],
        'primary': [
            len(primary_alerts),
            primary_alerts[REGRESSION_TARGET_COL].mean() if REGRESSION_TARGET_COL in primary_alerts.columns else np.nan,
            primary_alerts['single_alert_amount_abs'].mean() if 'single_alert_amount_abs' in primary_alerts.columns else np.nan
        ],
        'downstream': [
            len(downstream_alerts),
            downstream_alerts[REGRESSION_TARGET_COL].mean() if REGRESSION_TARGET_COL in downstream_alerts.columns else np.nan,
            downstream_alerts['single_alert_amount_abs'].mean() if 'single_alert_amount_abs' in downstream_alerts.columns else np.nan
        ]
    }
    comparison_df = pd.DataFrame(comparison)
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv(output_dir / 'reports' / 'E4_primary_vs_downstream.csv', index=False)

    # Save summary statistics
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(output_dir / 'reports' / 'E4_summary_stats.csv', index=False)

    return cluster_analysis


def main():
    """Main execution function."""
    print("\n" + "#"*60)
    print("PHASE 6: Automated Root Cause Analysis")
    print("#"*60)
    print(f"Started at: {datetime.now().isoformat()}")

    set_random_seeds(RANDOM_SEED)

    # Output directory
    output_dir = PHASE_6_DIR / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)
    (output_dir / 'reports').mkdir(exist_ok=True)
    (output_dir / 'models').mkdir(exist_ok=True)

    all_results = {
        'timestamp': datetime.now().isoformat()
    }

    # E1: Clustering
    e1_results = run_experiment_E1(output_dir)
    all_results['E1'] = e1_results.to_dict(orient='records')
    gc.collect()

    # E2: Bug Prediction
    e2_results = run_experiment_E2(output_dir)
    all_results['E2'] = e2_results.to_dict(orient='records')
    gc.collect()

    # E3: Text Analysis
    e3_results = run_experiment_E3(output_dir)
    all_results['E3'] = e3_results.to_dict(orient='records')
    gc.collect()

    # E4: Downstream Analysis
    e4_results = run_experiment_E4(output_dir)
    all_results['E4'] = e4_results.to_dict(orient='records') if len(e4_results) > 0 else []
    gc.collect()

    # Save all results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    save_results(all_results, output_dir / 'reports', 'experiment_summary')

    print(f"\nPhase 6 complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Finished at: {datetime.now().isoformat()}")

    return all_results


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Phase 6 failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
