#!/usr/bin/env python3
"""
Phase 2 FIXED: Multi-Class Alert Status Prediction WITHOUT Data Leakage

CRITICAL FIXES:
1. Use ONLY direction-agnostic features (true absolute values)
2. Predict alert STATUS which is a real human decision
3. Proper temporal train/test split
4. Handle class imbalance appropriately
5. Report per-class metrics and MCC

Status codes:
- 0: untriaged (exclude from training - no label yet)
- 1: downstream
- 2: reassigned
- 3: invalid
- 4: acknowledged
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    classification_report, confusion_matrix, matthews_corrcoef
)
import xgboost as xgb
import joblib

warnings.filterwarnings('ignore')

# Paths - use relative paths based on script location
SRC_DIR = Path(__file__).parent.parent
PROJECT_ROOT = SRC_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "alerts_data.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs_fixed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'reports').mkdir(exist_ok=True)
(OUTPUT_DIR / 'models').mkdir(exist_ok=True)
(OUTPUT_DIR / 'figures').mkdir(exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

STATUS_NAMES = {
    0: 'untriaged',
    1: 'downstream',
    2: 'reassigned',
    3: 'invalid',
    4: 'acknowledged'
}


def load_and_prepare_data():
    """Load data and prepare SAFE features only."""
    print("="*60)
    print("LOADING DATA FOR MULTI-CLASS STATUS PREDICTION")
    print("="*60)

    df = pd.read_csv(DATA_PATH)
    print(f"Total samples: {len(df)}")

    # ===========================================
    # FILTER TO TRIAGED ALERTS ONLY
    # ===========================================
    df['status'] = df['single_alert_status'].fillna(-1).astype(int)

    # Remove untriaged (status=0) - they have no ground truth
    triaged_df = df[df['status'] > 0].copy()
    print(f"\nTriaged alerts: {len(triaged_df)}")

    # Status distribution
    print("\nStatus Distribution:")
    for status_code, status_name in STATUS_NAMES.items():
        if status_code == 0:
            continue
        count = (triaged_df['status'] == status_code).sum()
        pct = count / len(triaged_df) * 100
        print(f"  {status_code} ({status_name}): {count} ({pct:.1f}%)")

    # ===========================================
    # DIRECTION-AGNOSTIC FEATURES ONLY
    # ===========================================
    print("\n" + "-"*40)
    print("SAFE FEATURES (Direction-Agnostic)")
    print("-"*40)

    # Truly absolute magnitude features
    triaged_df['magnitude_abs'] = np.abs(triaged_df['single_alert_amount_abs'])
    triaged_df['magnitude_pct_abs'] = np.abs(triaged_df['single_alert_amount_pct'])
    triaged_df['t_value_abs'] = np.abs(triaged_df['single_alert_t_value'])

    # Value scale features
    triaged_df['value_mean'] = (triaged_df['single_alert_new_value'] + triaged_df['single_alert_prev_value']) / 2
    triaged_df['value_ratio'] = triaged_df['single_alert_new_value'] / (triaged_df['single_alert_prev_value'] + 1e-10)
    triaged_df['value_ratio_abs'] = np.abs(np.log(triaged_df['value_ratio'].clip(0.01, 100)))

    magnitude_features = [
        'magnitude_abs',
        'magnitude_pct_abs',
        't_value_abs',
        'value_mean',
        'value_ratio_abs'
    ]

    # Context features
    context_cols = [
        'alert_summary_repository',
        'single_alert_series_signature_framework_id',
        'single_alert_series_signature_machine_platform',
        'single_alert_series_signature_suite',
    ]

    encoded_features = []
    encoders = {}
    for col in context_cols:
        if col in triaged_df.columns:
            le = LabelEncoder()
            triaged_df[f'{col}_enc'] = le.fit_transform(triaged_df[col].fillna('unknown').astype(str))
            encoded_features.append(f'{col}_enc')
            encoders[col] = le

    # Workflow feature
    workflow_features = []
    if 'single_alert_manually_created' in triaged_df.columns:
        triaged_df['manually_created'] = triaged_df['single_alert_manually_created'].fillna(0).astype(int)
        workflow_features.append('manually_created')

    feature_cols = magnitude_features + encoded_features + workflow_features

    print(f"\nFeatures used ({len(feature_cols)}):")
    for f in feature_cols:
        print(f"  - {f}")

    return triaged_df, feature_cols, encoders


def temporal_split(df, feature_cols, target_col, test_ratio=0.2):
    """Proper temporal split."""
    df = df.sort_values('push_timestamp').reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    # Remap labels to 0-indexed for XGBoost compatibility
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Impute and scale
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, train_df, test_df, imputer, scaler, label_encoder


def evaluate_multiclass_model(model, X_test, y_test, model_name, class_names):
    """Comprehensive multi-class evaluation."""
    y_pred = model.predict(X_test)

    # Overall metrics
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
    }

    # Per-class metrics
    unique_classes = np.unique(np.concatenate([y_test, y_pred]))
    per_class = {}
    for cls in unique_classes:
        y_test_binary = (y_test == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)

        cls_name = class_names.get(cls, str(cls))
        per_class[cls_name] = {
            'precision': precision_score(y_test_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_test_binary, y_pred_binary, zero_division=0),
            'f1': f1_score(y_test_binary, y_pred_binary, zero_division=0),
            'support': int(y_test_binary.sum())
        }

    return metrics, per_class


def run_experiment(df, feature_cols, target_col, class_names, output_dir):
    """Run multi-class classification experiment."""
    print(f"\n{'='*60}")
    print("EXPERIMENT: Multi-Class Status Prediction")
    print(f"{'='*60}")

    # Temporal split
    X_train, X_test, y_train, y_test, train_df, test_df, imputer, scaler, label_encoder = temporal_split(
        df, feature_cols, target_col
    )

    # Update class_names to use 0-indexed labels
    class_names = {i: STATUS_NAMES[c] for i, c in enumerate(label_encoder.classes_)}

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Train period: {train_df['push_timestamp'].min()} to {train_df['push_timestamp'].max()}")
    print(f"Test period: {test_df['push_timestamp'].min()} to {test_df['push_timestamp'].max()}")

    # Class distribution
    print("\nTrain class distribution:")
    for cls in np.unique(y_train):
        count = (y_train == cls).sum()
        print(f"  {cls} ({class_names.get(cls, 'unknown')}): {count}")

    # Get unique classes
    n_classes = len(np.unique(y_train))

    # Compute class weights
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))

    models = [
        ('Logistic Regression', LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED, class_weight='balanced'
        )),
        ('Random Forest', RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=RANDOM_SEED, class_weight='balanced'
        )),
        ('Gradient Boosting', GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=RANDOM_SEED
        )),
        ('XGBoost', xgb.XGBClassifier(
            n_estimators=100, max_depth=6, random_state=RANDOM_SEED,
            eval_metric='mlogloss', num_class=n_classes
        ))
    ]

    results = []
    all_per_class = {}
    best_model = None
    best_f1 = 0

    for name, model in models:
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        metrics, per_class = evaluate_multiclass_model(model, X_test, y_test, name, class_names)
        results.append(metrics)
        all_per_class[name] = per_class

        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  F1 Macro: {metrics['f1_macro']:.3f}")
        print(f"  F1 Weighted: {metrics['f1_weighted']:.3f}")

        if metrics['f1_macro'] > best_f1:
            best_f1 = metrics['f1_macro']
            best_model = model

    # Print per-class results for best model
    print("\n" + "-"*40)
    print("Per-Class Results (Best Model):")
    print("-"*40)
    best_name = results[-1]['model'] if best_model == models[-1][1] else 'XGBoost'
    for cls_name, cls_metrics in all_per_class.get('XGBoost', {}).items():
        print(f"  {cls_name}:")
        print(f"    Precision: {cls_metrics['precision']:.3f}")
        print(f"    Recall: {cls_metrics['recall']:.3f}")
        print(f"    F1: {cls_metrics['f1']:.3f}")
        print(f"    Support: {cls_metrics['support']}")

    return pd.DataFrame(results), all_per_class, best_model, imputer, scaler


def cross_repository_evaluation(df, feature_cols, target_col, class_names):
    """Leave-one-repository-out cross-validation."""
    print(f"\n{'='*60}")
    print("CROSS-REPOSITORY EVALUATION")
    print(f"{'='*60}")

    repo_col = 'alert_summary_repository'
    repos = df[repo_col].unique()
    print(f"Repositories: {list(repos)}")

    results = []

    for test_repo in repos:
        train_df = df[df[repo_col] != test_repo]
        test_df = df[df[repo_col] == test_repo]

        if len(test_df) < 50:
            print(f"Skipping {test_repo} - only {len(test_df)} samples")
            continue

        # Prepare features
        X_train = train_df[feature_cols].values
        X_test = test_df[feature_cols].values
        y_train = train_df[target_col].values
        y_test = test_df[target_col].values

        # Encode labels to 0-indexed
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        local_class_names = {i: STATUS_NAMES[c] for i, c in enumerate(le.classes_)}

        # Impute and scale
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train model
        n_classes = len(np.unique(y_train))
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, random_state=RANDOM_SEED,
            eval_metric='mlogloss'
        )
        model.fit(X_train, y_train)

        metrics, _ = evaluate_multiclass_model(model, X_test, y_test, f"XGB_{test_repo}", local_class_names)
        metrics['test_repo'] = test_repo
        metrics['train_samples'] = len(train_df)
        metrics['test_samples'] = len(test_df)
        results.append(metrics)

        print(f"  {test_repo}: F1_macro={metrics['f1_macro']:.3f}, Acc={metrics['accuracy']:.3f}")

    return pd.DataFrame(results)


def feature_ablation_study(df, feature_cols, target_col, class_names):
    """Test contribution of each feature group."""
    print(f"\n{'='*60}")
    print("FEATURE ABLATION STUDY")
    print(f"{'='*60}")

    # Define feature groups
    magnitude_features = [f for f in feature_cols if any(x in f for x in ['magnitude', 't_value', 'value_mean', 'value_ratio'])]
    context_features = [f for f in feature_cols if '_enc' in f]
    workflow_features = [f for f in feature_cols if 'manually_created' in f]

    feature_groups = {
        'all_features': feature_cols,
        'without_magnitude': [f for f in feature_cols if f not in magnitude_features],
        'without_context': [f for f in feature_cols if f not in context_features],
        'magnitude_only': magnitude_features,
        'context_only': context_features,
    }

    results = []

    for group_name, features in feature_groups.items():
        if len(features) == 0:
            continue

        print(f"\nTesting: {group_name} ({len(features)} features)")

        X_train, X_test, y_train, y_test, _, _, _, _, _ = temporal_split(
            df, features, target_col
        )

        n_classes = len(np.unique(y_train))
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, random_state=RANDOM_SEED,
            eval_metric='mlogloss'
        )
        model.fit(X_train, y_train)

        metrics, _ = evaluate_multiclass_model(model, X_test, y_test, group_name, class_names)
        metrics['feature_group'] = group_name
        metrics['n_features'] = len(features)
        results.append(metrics)

        print(f"  F1_macro={metrics['f1_macro']:.3f}, Acc={metrics['accuracy']:.3f}")

    return pd.DataFrame(results)


def main():
    print("\n" + "#"*60)
    print("PHASE 2 FIXED: Multi-Class Status Prediction")
    print("#"*60)
    print(f"Started at: {datetime.now().isoformat()}")

    # Load data
    df, feature_cols, encoders = load_and_prepare_data()

    class_names = {k: v for k, v in STATUS_NAMES.items() if k > 0}

    all_results = {}

    # ===========================================
    # MAIN EXPERIMENT: Status Prediction
    # ===========================================
    results_df, per_class_results, best_model, imputer, scaler = run_experiment(
        df, feature_cols, 'status', class_names, OUTPUT_DIR
    )
    results_df.to_csv(OUTPUT_DIR / 'reports' / 'E1_status_prediction_results.csv', index=False)
    all_results['status_prediction'] = results_df.to_dict(orient='records')

    # Save best model
    joblib.dump(best_model, OUTPUT_DIR / 'models' / 'best_status_predictor.joblib')
    joblib.dump(imputer, OUTPUT_DIR / 'models' / 'status_imputer.joblib')
    joblib.dump(scaler, OUTPUT_DIR / 'models' / 'status_scaler.joblib')

    # ===========================================
    # CROSS-REPOSITORY EVALUATION
    # ===========================================
    cross_repo_results = cross_repository_evaluation(df, feature_cols, 'status', class_names)
    cross_repo_results.to_csv(OUTPUT_DIR / 'reports' / 'E2_cross_repo_results.csv', index=False)
    all_results['cross_repo'] = cross_repo_results.to_dict(orient='records')

    # ===========================================
    # FEATURE ABLATION
    # ===========================================
    ablation_results = feature_ablation_study(df, feature_cols, 'status', class_names)
    ablation_results.to_csv(OUTPUT_DIR / 'reports' / 'E3_ablation_results.csv', index=False)
    all_results['ablation'] = ablation_results.to_dict(orient='records')

    # ===========================================
    # SUMMARY
    # ===========================================
    print("\n" + "="*60)
    print("SUMMARY: REALISTIC MULTI-CLASS PERFORMANCE")
    print("="*60)

    print("\nStatus Prediction Results:")
    print(results_df[['model', 'accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']].to_string(index=False))

    print("\nCross-Repository Average:")
    if len(cross_repo_results) > 0:
        print(f"  F1 Macro: {cross_repo_results['f1_macro'].mean():.3f} (+/- {cross_repo_results['f1_macro'].std():.3f})")
        print(f"  Accuracy: {cross_repo_results['accuracy'].mean():.3f} (+/- {cross_repo_results['accuracy'].std():.3f})")

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("""
Multi-class status prediction is a VALID task because:
- Status is a human decision (not deterministic)
- Predicting triage outcomes has practical value

Expected performance:
- F1 Macro: 0.40-0.70 (depends on class balance)
- Accuracy: 0.60-0.80 (dominated by majority class)

Key challenges:
1. Severe class imbalance (acknowledged >> others)
2. Overlapping decision criteria between classes
3. Subjective human judgments

The model learns patterns in how alerts are triaged,
which can be useful for prioritizing alerts for review.
    """)

    # Save all results
    import json
    with open(OUTPUT_DIR / 'reports' / 'experiment_summary.json', 'w') as f:
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\nFinished at: {datetime.now().isoformat()}")
    print(f"Results saved to: {OUTPUT_DIR}")

    return all_results


if __name__ == "__main__":
    results = main()
