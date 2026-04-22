"""
Phase 7: Integration and End-to-End System

Combines all phases into a unified regression detection pipeline.
"""

from .data_merger import (
    load_phase_outputs,
    merge_all_features,
    create_unified_dataset
)

from .feature_normalizer import (
    normalize_features,
    select_features,
    prepare_ensemble_features
)

from .stacking_ensemble import (
    StackingEnsemble,
    train_base_models,
    train_meta_classifier
)

from .evaluation import (
    evaluate_integrated_model,
    cross_phase_comparison,
    ablation_study
)
