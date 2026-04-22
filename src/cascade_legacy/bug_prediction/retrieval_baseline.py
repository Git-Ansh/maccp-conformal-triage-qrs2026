"""
Bug Description Prediction: Retrieval + Template Baseline.

Given an alert summary (group) predicted to have a bug (has_bug=1),
predict the bug's component, product, and type using k-nearest-neighbor
retrieval from historical bugs.

Approach:
1. Build feature vectors from alert-level features aggregated per summary.
2. Build TF-IDF vectors from bug summary text for the 482 known bugs.
3. For a new summary, find k most similar past summaries (by alert features).
4. Majority vote for component, product, type from the k neighbors.
5. Use the nearest bug's summary as a template for the draft description.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
from common.data_paths import RANDOM_SEED, PROJECT_ROOT

BUGS_DATA_PATH = PROJECT_ROOT / 'data' / 'bugs_data.csv'
ALERTS_DATA_PATH = PROJECT_ROOT / 'data' / 'alerts_data.csv'

# Alert features used for matching summaries to bugs
MATCHING_FEATURES = [
    'group_size', 'n_regressions', 'regression_ratio',
    'magnitude_mean', 'magnitude_max', 'magnitude_std',
    'pct_change_mean', 'pct_change_max',
    't_value_mean', 't_value_max',
    'n_unique_suites', 'n_unique_platforms',
    'n_manually_created',
]


class RetrievalBugPredictor:
    """
    Retrieval-based bug triage predictor.
    Uses alert features to find similar historical bugs via kNN.
    """

    def __init__(self, k: int = 5, use_text: bool = True):
        self.k = k
        self.use_text = use_text
        self.scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(max_features=500, stop_words='english',
                                      ngram_range=(1, 2), min_df=1)
        self.nn_model = None
        self.train_bugs = None
        self.train_features = None
        self.train_text_features = None

    def _build_summary_features(self, alerts_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate alert-level features into summary-level features."""
        grouped = alerts_df.groupby('alert_summary_id')

        summary = pd.DataFrame({
            'alert_summary_id': grouped.ngroup().index if hasattr(grouped, 'ngroup') else grouped.groups.keys(),
        })
        summary = pd.DataFrame({'alert_summary_id': list(grouped.groups.keys())})

        agg_dict = {
            'single_alert_amount_abs': ['mean', 'max', 'std'],
            'single_alert_amount_pct': ['mean', 'max'],
            'single_alert_t_value': ['mean', 'max'],
            'single_alert_is_regression': ['sum', 'mean'],
            'single_alert_manually_created': 'sum',
        }
        available_agg = {k: v for k, v in agg_dict.items() if k in alerts_df.columns}
        agg = grouped.agg(available_agg)
        agg.columns = ['_'.join(col).strip('_') for col in agg.columns]
        agg = agg.reset_index()

        # Group size
        agg['group_size'] = grouped.size().values

        # Unique suites/platforms
        if 'single_alert_series_signature_suite' in alerts_df.columns:
            agg['n_unique_suites'] = grouped['single_alert_series_signature_suite'].nunique().values
            # Dominant suite for text matching
            agg['dominant_suite'] = grouped['single_alert_series_signature_suite'].agg(
                lambda x: x.value_counts().index[0] if len(x) > 0 else 'unknown'
            ).values

        if 'single_alert_series_signature_machine_platform' in alerts_df.columns:
            agg['n_unique_platforms'] = grouped['single_alert_series_signature_machine_platform'].nunique().values
            agg['dominant_platform'] = grouped['single_alert_series_signature_machine_platform'].agg(
                lambda x: x.value_counts().index[0] if len(x) > 0 else 'unknown'
            ).values

        # Repository
        if 'alert_summary_repository' in alerts_df.columns:
            agg['repository'] = grouped['alert_summary_repository'].first().values

        return agg

    def _get_feature_matrix(self, summary_df: pd.DataFrame) -> np.ndarray:
        """Extract numeric feature matrix from summary DataFrame."""
        # Rename aggregated columns to match MATCHING_FEATURES
        rename_map = {
            'single_alert_amount_abs_mean': 'magnitude_mean',
            'single_alert_amount_abs_max': 'magnitude_max',
            'single_alert_amount_abs_std': 'magnitude_std',
            'single_alert_amount_pct_mean': 'pct_change_mean',
            'single_alert_amount_pct_max': 'pct_change_max',
            'single_alert_t_value_mean': 't_value_mean',
            'single_alert_t_value_max': 't_value_max',
            'single_alert_is_regression_sum': 'n_regressions',
            'single_alert_is_regression_mean': 'regression_ratio',
            'single_alert_manually_created_sum': 'n_manually_created',
        }
        df = summary_df.rename(columns=rename_map)

        available = [c for c in MATCHING_FEATURES if c in df.columns]
        X = df[available].fillna(0).values
        return X

    def _build_text_descriptors(self, summary_df: pd.DataFrame) -> List[str]:
        """Build text descriptors from alert features for text-based matching."""
        texts = []
        for _, row in summary_df.iterrows():
            parts = []
            if 'dominant_suite' in row.index and pd.notna(row.get('dominant_suite')):
                parts.append(str(row['dominant_suite']))
            if 'dominant_platform' in row.index and pd.notna(row.get('dominant_platform')):
                parts.append(str(row['dominant_platform']))
            if 'repository' in row.index and pd.notna(row.get('repository')):
                parts.append(str(row['repository']))
            texts.append(' '.join(parts) if parts else 'unknown')
        return texts

    def fit(
        self,
        alerts_df: pd.DataFrame,
        bugs_df: pd.DataFrame,
        summary_bug_map: pd.DataFrame
    ):
        """
        Fit the retrieval model.

        Args:
            alerts_df: Alert-level features
            bugs_df: Bug metadata (id, type, summary, component, product)
            summary_bug_map: DataFrame with alert_summary_id -> bug_number mapping
        """
        # Build summary features from alerts
        summary_features = self._build_summary_features(alerts_df)

        # Link summaries to bugs
        linked = summary_features.merge(
            summary_bug_map, on='alert_summary_id', how='inner'
        )
        linked = linked.merge(
            bugs_df[['id', 'type', 'summary', 'component', 'product']],
            left_on='alert_summary_bug_number', right_on='id', how='inner'
        )

        # Store training data
        self.train_bugs = linked[['alert_summary_id', 'alert_summary_bug_number',
                                   'type', 'summary', 'component', 'product']].copy()
        self.train_summary_ids = linked['alert_summary_id'].values

        # Build feature matrix
        X = self._get_feature_matrix(linked)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.train_features = X_scaled

        # Text features (TF-IDF on bug summaries + alert descriptors)
        if self.use_text:
            # Combine bug summary text with alert descriptors
            alert_texts = self._build_text_descriptors(linked)
            combined_texts = [
                f"{row['summary']} {alert_text}"
                for (_, row), alert_text in zip(linked.iterrows(), alert_texts)
            ]
            self.train_text_features = self.tfidf.fit_transform(combined_texts)

        # Fit kNN on numeric features
        self.nn_model = NearestNeighbors(n_neighbors=min(self.k, len(X_scaled)),
                                          metric='euclidean', n_jobs=-1)
        self.nn_model.fit(X_scaled)

        print(f"RetrievalBugPredictor fitted on {len(linked)} summary-bug pairs")
        print(f"  Components: {linked['component'].nunique()}, Products: {linked['product'].nunique()}")

        return self

    def predict(
        self,
        summary_df: pd.DataFrame,
        alerts_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Predict bug component, product, type for new summaries.

        Args:
            summary_df: Summary-level features (from cascade or freshly built)
            alerts_df: If provided, build summary features from alerts

        Returns:
            DataFrame with predictions: pred_component, pred_product, pred_type,
            pred_summary_template, neighbor_bug_ids, confidence
        """
        if alerts_df is not None:
            features_df = self._build_summary_features(alerts_df)
            # Match by alert_summary_id
            features_df = features_df[
                features_df['alert_summary_id'].isin(summary_df['alert_summary_id'])
            ]
        else:
            features_df = summary_df.copy()

        X = self._get_feature_matrix(features_df)
        X_scaled = self.scaler.transform(X)

        distances, indices = self.nn_model.kneighbors(X_scaled)

        results = []
        for i in range(len(X_scaled)):
            neighbor_idx = indices[i]
            neighbor_dists = distances[i]
            neighbor_bugs = self.train_bugs.iloc[neighbor_idx]

            # Majority vote (weighted by inverse distance)
            weights = 1.0 / (neighbor_dists + 1e-6)

            # Component prediction
            comp_votes = Counter()
            for bug, w in zip(neighbor_bugs['component'], weights):
                comp_votes[bug] += w
            pred_component = comp_votes.most_common(1)[0][0]

            # Product prediction
            prod_votes = Counter()
            for bug, w in zip(neighbor_bugs['product'], weights):
                prod_votes[bug] += w
            pred_product = prod_votes.most_common(1)[0][0]

            # Type prediction
            type_votes = Counter()
            for bug, w in zip(neighbor_bugs['type'], weights):
                type_votes[bug] += w
            pred_type = type_votes.most_common(1)[0][0]

            # Template: nearest bug's summary
            pred_summary = neighbor_bugs.iloc[0]['summary']

            # Confidence: agreement ratio
            total_weight = weights.sum()
            comp_conf = comp_votes[pred_component] / total_weight
            prod_conf = prod_votes[pred_product] / total_weight

            results.append({
                'alert_summary_id': features_df.iloc[i]['alert_summary_id'],
                'pred_component': pred_component,
                'pred_product': pred_product,
                'pred_type': pred_type,
                'pred_summary_template': pred_summary,
                'neighbor_bug_ids': list(neighbor_bugs['alert_summary_bug_number'].values),
                'component_confidence': comp_conf,
                'product_confidence': prod_conf,
            })

        return pd.DataFrame(results)

    def evaluate_leave_one_out(self) -> Dict:
        """
        Leave-one-out evaluation on training data.
        For each summary, predict using k nearest neighbors (excluding itself).
        """
        n = len(self.train_features)
        true_components = self.train_bugs['component'].values
        true_products = self.train_bugs['product'].values
        true_types = self.train_bugs['type'].values

        pred_components = []
        pred_products = []
        pred_types = []

        # Use k+1 neighbors and exclude self
        nn = NearestNeighbors(n_neighbors=min(self.k + 1, n),
                               metric='euclidean', n_jobs=-1)
        nn.fit(self.train_features)
        distances, indices = nn.kneighbors(self.train_features)

        for i in range(n):
            # Exclude self from neighbors
            neighbor_idx = [j for j in indices[i] if j != i][:self.k]
            neighbor_dists = [distances[i][list(indices[i]).index(j)] for j in neighbor_idx]

            if len(neighbor_idx) == 0:
                pred_components.append('unknown')
                pred_products.append('unknown')
                pred_types.append('unknown')
                continue

            weights = 1.0 / (np.array(neighbor_dists) + 1e-6)

            # Component
            comp_votes = Counter()
            for j, w in zip(neighbor_idx, weights):
                comp_votes[true_components[j]] += w
            pred_components.append(comp_votes.most_common(1)[0][0])

            # Product
            prod_votes = Counter()
            for j, w in zip(neighbor_idx, weights):
                prod_votes[true_products[j]] += w
            pred_products.append(prod_votes.most_common(1)[0][0])

            # Type
            type_votes = Counter()
            for j, w in zip(neighbor_idx, weights):
                type_votes[true_types[j]] += w
            pred_types.append(type_votes.most_common(1)[0][0])

        # Compute metrics
        comp_acc = accuracy_score(true_components, pred_components)
        prod_acc = accuracy_score(true_products, pred_products)
        type_acc = accuracy_score(true_types, pred_types)

        comp_f1 = f1_score(true_components, pred_components, average='weighted', zero_division=0)
        prod_f1 = f1_score(true_products, pred_products, average='weighted', zero_division=0)
        type_f1 = f1_score(true_types, pred_types, average='weighted', zero_division=0)

        results = {
            'component_accuracy': comp_acc,
            'component_f1_weighted': comp_f1,
            'product_accuracy': prod_acc,
            'product_f1_weighted': prod_f1,
            'type_accuracy': type_acc,
            'type_f1_weighted': type_f1,
            'n_samples': n,
            'k': self.k,
        }

        return results, {
            'component': classification_report(true_components, pred_components, zero_division=0),
            'product': classification_report(true_products, pred_products, zero_division=0),
            'type': classification_report(true_types, pred_types, zero_division=0),
        }

    def evaluate_text_retrieval(self) -> Dict:
        """
        Evaluate text-based retrieval using TF-IDF cosine similarity.
        Leave-one-out: for each bug, find most similar bugs by text.
        """
        if self.train_text_features is None:
            return {}

        n = self.train_text_features.shape[0]
        true_components = self.train_bugs['component'].values
        true_products = self.train_bugs['product'].values

        # Compute pairwise similarity
        sim_matrix = cosine_similarity(self.train_text_features)
        np.fill_diagonal(sim_matrix, -1)  # exclude self

        pred_components = []
        pred_products = []

        for i in range(n):
            top_k_idx = np.argsort(sim_matrix[i])[-self.k:][::-1]
            weights = sim_matrix[i][top_k_idx]
            weights = np.maximum(weights, 0)  # clip negatives

            comp_votes = Counter()
            prod_votes = Counter()
            for j, w in zip(top_k_idx, weights):
                comp_votes[true_components[j]] += max(w, 1e-6)
                prod_votes[true_products[j]] += max(w, 1e-6)

            pred_components.append(comp_votes.most_common(1)[0][0])
            pred_products.append(prod_votes.most_common(1)[0][0])

        return {
            'text_component_accuracy': accuracy_score(true_components, pred_components),
            'text_product_accuracy': accuracy_score(true_products, pred_products),
            'text_component_f1': f1_score(true_components, pred_components, average='weighted', zero_division=0),
            'text_product_f1': f1_score(true_products, pred_products, average='weighted', zero_division=0),
        }


def load_bug_prediction_data() -> Dict:
    """Load and prepare data for bug prediction evaluation."""
    alerts = pd.read_csv(ALERTS_DATA_PATH)
    bugs = pd.read_csv(BUGS_DATA_PATH)

    # Build summary -> bug mapping
    summary_bug_map = alerts.dropna(subset=['alert_summary_bug_number'])[
        ['alert_summary_id', 'alert_summary_bug_number']
    ].drop_duplicates()
    summary_bug_map['alert_summary_bug_number'] = summary_bug_map['alert_summary_bug_number'].astype(int)

    return {
        'alerts': alerts,
        'bugs': bugs,
        'summary_bug_map': summary_bug_map,
    }


class MLBugClassifier:
    """
    ML-based bug attribute classifier.
    Trains separate classifiers for product, component, and type prediction
    using alert summary features + TF-IDF text features.
    """

    def __init__(self, min_class_count: int = 5):
        self.min_class_count = min_class_count
        self.scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(max_features=300, stop_words='english',
                                      ngram_range=(1, 2), min_df=2)
        self.product_model = None
        self.component_model = None
        self.type_model = None
        self.product_encoder = LabelEncoder()
        self.component_encoder = LabelEncoder()
        self.type_encoder = LabelEncoder()

    def _prepare_features(
        self, alerts_df: pd.DataFrame, bugs_df: pd.DataFrame,
        summary_bug_map: pd.DataFrame
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Build feature matrix and labels."""
        # Build summary features
        grouped = alerts_df.groupby('alert_summary_id')
        agg_dict = {
            'single_alert_amount_abs': ['mean', 'max', 'std'],
            'single_alert_amount_pct': ['mean', 'max'],
            'single_alert_t_value': ['mean', 'max'],
            'single_alert_is_regression': ['sum', 'mean'],
            'single_alert_manually_created': 'sum',
        }
        available_agg = {k: v for k, v in agg_dict.items() if k in alerts_df.columns}
        agg = grouped.agg(available_agg)
        agg.columns = ['_'.join(col).strip('_') for col in agg.columns]
        agg = agg.reset_index()
        agg['group_size'] = grouped.size().values

        if 'single_alert_series_signature_suite' in alerts_df.columns:
            agg['n_unique_suites'] = grouped['single_alert_series_signature_suite'].nunique().values
            agg['dominant_suite'] = grouped['single_alert_series_signature_suite'].agg(
                lambda x: x.value_counts().index[0] if len(x) > 0 else 'unknown'
            ).values

        if 'single_alert_series_signature_machine_platform' in alerts_df.columns:
            agg['n_unique_platforms'] = grouped['single_alert_series_signature_machine_platform'].nunique().values

        if 'alert_summary_repository' in alerts_df.columns:
            agg['repository'] = grouped['alert_summary_repository'].first().values

        # Link to bugs
        linked = agg.merge(summary_bug_map, on='alert_summary_id', how='inner')
        linked = linked.merge(
            bugs_df[['id', 'type', 'summary', 'component', 'product']],
            left_on='alert_summary_bug_number', right_on='id', how='inner'
        )

        # Numeric features
        numeric_cols = [c for c in linked.columns if linked[c].dtype in ['float64', 'int64', 'int32']
                       and c not in ['alert_summary_id', 'alert_summary_bug_number', 'id']]
        X_numeric = linked[numeric_cols].fillna(0).values

        # Text features from bug summaries + suite info
        texts = []
        for _, row in linked.iterrows():
            parts = [str(row.get('summary', ''))]
            if 'dominant_suite' in row.index and pd.notna(row.get('dominant_suite')):
                parts.append(str(row['dominant_suite']))
            if 'repository' in row.index and pd.notna(row.get('repository')):
                parts.append(str(row['repository']))
            texts.append(' '.join(parts))

        X_text = self.tfidf.fit_transform(texts).toarray()

        # Combine
        X_numeric_scaled = self.scaler.fit_transform(X_numeric)
        X = np.hstack([X_numeric_scaled, X_text])

        return X, linked

    def evaluate_cv(
        self,
        alerts_df: pd.DataFrame,
        bugs_df: pd.DataFrame,
        summary_bug_map: pd.DataFrame,
        cv: int = 5
    ) -> Dict:
        """
        Cross-validated evaluation of ML classifiers.

        Returns accuracy and F1 for product, component (top classes), and type.
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.ensemble import RandomForestClassifier

        try:
            from xgboost import XGBClassifier
            has_xgb = True
        except ImportError:
            has_xgb = False

        X, linked = self._prepare_features(alerts_df, bugs_df, summary_bug_map)

        results = {}

        for target_col, label in [('product', 'Product'), ('component', 'Component'), ('type', 'Type')]:
            y_raw = linked[target_col].values

            # For component: only keep classes with >= min_class_count samples
            if target_col == 'component':
                counts = Counter(y_raw)
                valid_classes = {c for c, n in counts.items() if n >= self.min_class_count}
                mask = np.array([c in valid_classes for c in y_raw])
                X_target = X[mask]
                y_raw_target = y_raw[mask]
                n_filtered = len(y_raw) - mask.sum()
                print(f"  {label}: {len(valid_classes)} classes with >= {self.min_class_count} samples "
                      f"({n_filtered} samples filtered)")
            else:
                X_target = X
                y_raw_target = y_raw

            le = LabelEncoder()
            y = le.fit_transform(y_raw_target)
            n_classes = len(le.classes_)

            # Use RF for multi-class to avoid non-contiguous label issues in CV
            model = RandomForestClassifier(
                n_estimators=200, max_depth=12, min_samples_leaf=3,
                class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1
            )

            # Stratified K-fold CV
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
            all_true = []
            all_pred = []

            for train_idx, test_idx in skf.split(X_target, y):
                model.fit(X_target[train_idx], y[train_idx])
                preds = model.predict(X_target[test_idx])
                all_true.extend(y[test_idx])
                all_pred.extend(preds)

            all_true = np.array(all_true)
            all_pred = np.array(all_pred)

            acc = accuracy_score(all_true, all_pred)
            f1_w = f1_score(all_true, all_pred, average='weighted', zero_division=0)
            f1_m = f1_score(all_true, all_pred, average='macro', zero_division=0)

            results[f'{target_col}_accuracy'] = acc
            results[f'{target_col}_f1_weighted'] = f1_w
            results[f'{target_col}_f1_macro'] = f1_m
            results[f'{target_col}_n_classes'] = n_classes
            results[f'{target_col}_n_samples'] = len(y)

            print(f"  {label} ({n_classes} classes): Acc={acc:.3f}, F1w={f1_w:.3f}, F1m={f1_m:.3f}")

        return results


def run_retrieval_baseline(k_values: List[int] = None) -> Dict:
    """
    Run the full retrieval baseline evaluation.

    Args:
        k_values: List of k values to evaluate (default: [1, 3, 5, 7, 10])

    Returns:
        Dict with evaluation results for each k
    """
    if k_values is None:
        k_values = [1, 3, 5, 7, 10]

    data = load_bug_prediction_data()
    print(f"Loaded {len(data['bugs'])} bugs, {len(data['summary_bug_map'])} linked summaries")

    all_results = {}
    best_k = None
    best_score = 0

    for k in k_values:
        print(f"\n--- k={k} ---")
        predictor = RetrievalBugPredictor(k=k, use_text=True)
        predictor.fit(data['alerts'], data['bugs'], data['summary_bug_map'])

        # Feature-based LOO
        metrics, reports = predictor.evaluate_leave_one_out()
        print(f"  Feature-based: Component={metrics['component_accuracy']:.3f}, "
              f"Product={metrics['product_accuracy']:.3f}, "
              f"Type={metrics['type_accuracy']:.3f}")

        # Text-based LOO
        text_metrics = predictor.evaluate_text_retrieval()
        if text_metrics:
            print(f"  Text-based:    Component={text_metrics['text_component_accuracy']:.3f}, "
                  f"Product={text_metrics['text_product_accuracy']:.3f}")

        combined = {**metrics, **text_metrics}
        all_results[f'k={k}'] = combined

        # Track best
        score = metrics['component_accuracy'] + metrics['product_accuracy']
        if score > best_score:
            best_score = score
            best_k = k

    print(f"\nBest k={best_k}")
    all_results['best_k'] = best_k

    return all_results


def run_ml_classifier(cv: int = 5) -> Dict:
    """Run the ML classifier evaluation."""
    data = load_bug_prediction_data()
    print(f"Loaded {len(data['bugs'])} bugs, {len(data['summary_bug_map'])} linked summaries")

    clf = MLBugClassifier(min_class_count=5)
    results = clf.evaluate_cv(
        data['alerts'], data['bugs'], data['summary_bug_map'], cv=cv
    )
    return results


if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)

    # Retrieval baseline
    print("=" * 70)
    print("RETRIEVAL BASELINE (kNN)")
    print("=" * 70)
    retrieval_results = run_retrieval_baseline()

    # ML classifier
    print("\n" + "=" * 70)
    print("ML CLASSIFIER (XGBoost/RF, 5-fold CV)")
    print("=" * 70)
    ml_results = run_ml_classifier(cv=5)

    # Summary
    print("\n" + "=" * 70)
    print("COMBINED RESULTS SUMMARY")
    print("=" * 70)

    print("\nRetrieval (best k):")
    best_k_key = f"k={retrieval_results.get('best_k', 5)}"
    if best_k_key in retrieval_results:
        r = retrieval_results[best_k_key]
        print(f"  Feature-based: Product={r['product_accuracy']:.3f}, Component={r['component_accuracy']:.3f}")
        if 'text_product_accuracy' in r:
            print(f"  Text-based:    Product={r['text_product_accuracy']:.3f}, Component={r['text_component_accuracy']:.3f}")

    print("\nML Classifier:")
    print(f"  Product:   Acc={ml_results['product_accuracy']:.3f}, F1w={ml_results['product_f1_weighted']:.3f}")
    print(f"  Component: Acc={ml_results['component_accuracy']:.3f}, F1w={ml_results['component_f1_weighted']:.3f}")
    print(f"  Type:      Acc={ml_results['type_accuracy']:.3f}, F1w={ml_results['type_f1_weighted']:.3f}")
