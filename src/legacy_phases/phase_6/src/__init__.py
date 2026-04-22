"""
Phase 6: Automated Root Cause Analysis

Modules for clustering, bug prediction, text analysis, and downstream analysis.
"""

from .feature_aggregator import (
    load_alerts_with_features,
    load_bug_data,
    merge_alert_bug_data,
    prepare_clustering_features
)

from .clustering import (
    KMeansClusterer,
    HDBSCANClusterer,
    DBSCANClusterer,
    evaluate_clustering
)

from .bug_prediction import (
    BugPredictionModel,
    prepare_bug_prediction_data
)

from .text_analysis import (
    TextAnalyzer,
    extract_keywords,
    extract_topics_lda
)

from .downstream_analysis import (
    build_alert_graph,
    analyze_downstream_clusters
)
