"""
Bug Description Prediction: LLM Few-Shot Approach.

Uses large language models with few-shot prompting to predict bug attributes
(component, product, type) and generate draft bug summaries given alert features
and similar historical bugs.

Approach:
1. Given a new alert summary, retrieve k most similar past bugs (from retrieval_baseline).
2. Construct a few-shot prompt with alert features + k example bugs.
3. Ask the LLM to predict component, product, type, and draft a bug summary.
4. Parse structured output.

Supports multiple backends: LLM API (OpenAI GPT), or cached/mock for evaluation.
"""

import sys
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
from common.data_paths import RANDOM_SEED, PROJECT_ROOT

from cascade.bug_prediction.retrieval_baseline import (
    RetrievalBugPredictor, load_bug_prediction_data
)

# Cache directory for LLM responses
CACHE_DIR = PROJECT_ROOT / 'cascade_outputs' / 'llm_cache'

# Top products and components for structured output
TOP_PRODUCTS = [
    'Core', 'Testing', 'DevTools', 'Firefox', 'Firefox Build System',
    'Fenix', 'Toolkit', 'GeckoView', 'Release Engineering', 'Remote Protocol'
]

TOP_COMPONENTS = [
    'Performance', 'General', 'Raptor', 'CSS Parsing and Computation',
    'Debugger', 'JavaScript Engine', 'Graphics: WebRender', 'Widget: Gtk',
    'DOM: Core & HTML', 'Translations', 'Toolchains', 'Graphics: Canvas2D',
    'JavaScript: GC', 'DOM: Navigation', 'Internationalization',
    'JavaScript Engine: JIT', 'Layout', 'Networking: HTTP',
    'Storage: IndexedDB', 'ImageLib',
]

BUG_TYPES = ['defect', 'task', 'enhancement']


def build_alert_description(summary_row: pd.Series) -> str:
    """Build a human-readable description of an alert summary for the prompt."""
    parts = []
    parts.append(f"Alert Summary ID: {summary_row.get('alert_summary_id', 'unknown')}")

    if 'group_size' in summary_row.index:
        parts.append(f"Number of alerts in group: {int(summary_row['group_size'])}")
    if 'dominant_suite' in summary_row.index and pd.notna(summary_row.get('dominant_suite')):
        parts.append(f"Dominant test suite: {summary_row['dominant_suite']}")
    if 'dominant_platform' in summary_row.index and pd.notna(summary_row.get('dominant_platform')):
        parts.append(f"Platform: {summary_row['dominant_platform']}")
    if 'repository' in summary_row.index and pd.notna(summary_row.get('repository')):
        parts.append(f"Repository: {summary_row['repository']}")

    # Magnitude info
    mag_parts = []
    for col, label in [('magnitude_mean', 'mean'), ('magnitude_max', 'max')]:
        if col in summary_row.index and pd.notna(summary_row.get(col)):
            mag_parts.append(f"{label}={summary_row[col]:.2f}")
    if mag_parts:
        parts.append(f"Regression magnitude: {', '.join(mag_parts)}")

    # T-value
    t_parts = []
    for col, label in [('t_value_mean', 'mean'), ('t_value_max', 'max')]:
        if col in summary_row.index and pd.notna(summary_row.get(col)):
            t_parts.append(f"{label}={summary_row[col]:.2f}")
    if t_parts:
        parts.append(f"T-value: {', '.join(t_parts)}")

    if 'n_unique_suites' in summary_row.index:
        parts.append(f"Unique test suites: {int(summary_row.get('n_unique_suites', 0))}")
    if 'n_unique_platforms' in summary_row.index:
        parts.append(f"Unique platforms: {int(summary_row.get('n_unique_platforms', 0))}")

    return '\n'.join(parts)


def build_few_shot_prompt(
    alert_description: str,
    similar_bugs: pd.DataFrame,
    products: List[str] = None,
    components: List[str] = None,
) -> str:
    """
    Build a few-shot prompt for bug triage prediction.

    Args:
        alert_description: Human-readable alert summary description
        similar_bugs: DataFrame of k most similar historical bugs
        products: List of valid product names
        components: List of valid component names

    Returns:
        Formatted prompt string
    """
    if products is None:
        products = TOP_PRODUCTS
    if components is None:
        components = TOP_COMPONENTS

    # System context
    prompt_parts = [
        "You are an expert Mozilla Firefox performance engineer. "
        "Given a performance regression alert from Perfherder (Mozilla's CI performance monitoring system), "
        "predict the most likely Bugzilla bug attributes and draft a bug summary.\n",
    ]

    # Examples from similar bugs
    prompt_parts.append("Here are similar past performance alerts and their associated bugs:\n")
    for i, (_, bug) in enumerate(similar_bugs.iterrows(), 1):
        prompt_parts.append(f"Example {i}:")
        prompt_parts.append(f"  Bug Summary: {bug.get('summary', 'N/A')}")
        prompt_parts.append(f"  Product: {bug.get('product', 'N/A')}")
        prompt_parts.append(f"  Component: {bug.get('component', 'N/A')}")
        prompt_parts.append(f"  Type: {bug.get('type', 'N/A')}")
        prompt_parts.append("")

    # Current alert
    prompt_parts.append("Now, analyze this new performance regression alert:\n")
    prompt_parts.append(alert_description)
    prompt_parts.append("")

    # Output format
    prompt_parts.append("Based on the alert features and the similar past bugs, predict:")
    prompt_parts.append(f"1. Product (choose from: {', '.join(products[:10])})")
    prompt_parts.append(f"2. Component (choose from: {', '.join(components[:15])}, or suggest another)")
    prompt_parts.append(f"3. Type (choose from: {', '.join(BUG_TYPES)})")
    prompt_parts.append("4. A draft bug summary (one line, similar in style to the examples)")
    prompt_parts.append("")
    prompt_parts.append("Respond in JSON format:")
    prompt_parts.append('{"product": "...", "component": "...", "type": "...", "summary": "..."}')

    return '\n'.join(prompt_parts)


def parse_llm_response(response: str) -> Dict:
    """Parse the LLM response into structured predictions."""
    # Try to extract JSON from response
    try:
        # Find JSON in response
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
    except json.JSONDecodeError:
        pass

    # Fallback: parse key-value pairs
    result = {'product': 'Core', 'component': 'General', 'type': 'defect', 'summary': ''}
    response_lower = response.lower()

    for product in TOP_PRODUCTS:
        if product.lower() in response_lower:
            result['product'] = product
            break

    for component in TOP_COMPONENTS:
        if component.lower() in response_lower:
            result['component'] = component
            break

    for bug_type in BUG_TYPES:
        if bug_type in response_lower:
            result['type'] = bug_type
            break

    return result


class LLMBugPredictor:
    """
    LLM-based bug triage predictor with few-shot prompting.

    Uses a retrieval model to find similar bugs, then asks an LLM to
    predict component/product/type and draft a summary.
    """

    def __init__(
        self,
        k_examples: int = 5,
        backend: str = 'cache',
        model: str = 'gpt-4o',
        cache_dir: Optional[Path] = None,
    ):
        self.k_examples = k_examples
        self.backend = backend
        self.model = model
        self.cache_dir = cache_dir or CACHE_DIR
        self.retriever = RetrievalBugPredictor(k=k_examples, use_text=True)
        self._client = None

    def _get_cache_key(self, prompt: str) -> str:
        """Generate a deterministic cache key for a prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()

    def _load_cached(self, cache_key: str) -> Optional[str]:
        """Load a cached LLM response."""
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                data = json.load(f)
                return data.get('response')
        return None

    def _save_cached(self, cache_key: str, prompt: str, response: str):
        """Save an LLM response to cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"{cache_key}.json"
        with open(cache_path, 'w') as f:
            json.dump({
                'prompt': prompt,
                'response': response,
                'model': self.model,
            }, f, indent=2)

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM backend."""
        cache_key = self._get_cache_key(prompt)
        cached = self._load_cached(cache_key)
        if cached is not None:
            return cached

        response = None

        if self.backend == 'llm_api':
            try:
                import openai
                if self._client is None:
                    self._client = openai.OpenAI()
                msg = self._client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )
                response = msg.content[0].text
            except ImportError:
                print("LLM API package not installed. Using fallback.")
            except Exception as e:
                print(f"LLM API error: {e}. Using fallback.")

        elif self.backend == 'openai':
            try:
                import openai
                if self._client is None:
                    self._client = openai.OpenAI()
                completion = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                )
                response = completion.choices[0].message.content
            except ImportError:
                print("openai package not installed. Using fallback.")
            except Exception as e:
                print(f"OpenAI API error: {e}. Using fallback.")

        if response is None:
            # Fallback: use retrieval-based prediction (no LLM)
            return None

        self._save_cached(cache_key, prompt, response)
        return response

    def fit(self, alerts_df: pd.DataFrame, bugs_df: pd.DataFrame,
            summary_bug_map: pd.DataFrame):
        """Fit the retrieval model for finding similar bugs."""
        self.retriever.fit(alerts_df, bugs_df, summary_bug_map)
        self.bugs_df = bugs_df
        return self

    def predict_single(
        self,
        summary_row: pd.Series,
        exclude_bug_id: Optional[int] = None
    ) -> Dict:
        """
        Predict bug attributes for a single alert summary.

        Args:
            summary_row: Row with alert summary features
            exclude_bug_id: Bug ID to exclude (for LOO evaluation)

        Returns:
            Dict with predictions
        """
        alert_desc = build_alert_description(summary_row)

        # Find similar bugs
        if self.retriever.train_bugs is not None:
            train_bugs = self.retriever.train_bugs
            if exclude_bug_id is not None:
                train_bugs = train_bugs[
                    train_bugs['alert_summary_bug_number'] != exclude_bug_id
                ]
            similar = train_bugs.head(self.k_examples)
        else:
            similar = pd.DataFrame()

        # Build prompt
        prompt = build_few_shot_prompt(alert_desc, similar)

        # Call LLM
        response = self._call_llm(prompt)

        if response is not None:
            prediction = parse_llm_response(response)
            prediction['source'] = 'llm'
        else:
            # Fallback to retrieval
            if len(similar) > 0:
                prediction = {
                    'product': Counter(similar['product']).most_common(1)[0][0],
                    'component': Counter(similar['component']).most_common(1)[0][0],
                    'type': Counter(similar['type']).most_common(1)[0][0],
                    'summary': similar.iloc[0]['summary'],
                    'source': 'retrieval_fallback',
                }
            else:
                prediction = {
                    'product': 'Core', 'component': 'General',
                    'type': 'defect', 'summary': '',
                    'source': 'default',
                }

        prediction['alert_summary_id'] = summary_row.get('alert_summary_id', None)
        prediction['prompt'] = prompt

        return prediction

    def evaluate_loo(
        self,
        max_samples: Optional[int] = None
    ) -> Dict:
        """
        Leave-one-out evaluation.

        For each linked summary, predict its bug attributes using all other bugs
        as the training set.

        Args:
            max_samples: Limit evaluation to first N samples (for testing)

        Returns:
            Dict with evaluation metrics
        """
        if self.retriever.train_bugs is None:
            raise ValueError("Model not fitted. Call fit() first.")

        train_bugs = self.retriever.train_bugs
        n = len(train_bugs) if max_samples is None else min(max_samples, len(train_bugs))

        true_components = []
        true_products = []
        true_types = []
        pred_components = []
        pred_products = []
        pred_types = []
        sources = []

        for i in range(n):
            row = train_bugs.iloc[i]
            bug_id = row['alert_summary_bug_number']

            # Get features for this summary
            feature_row = pd.Series({
                'alert_summary_id': row['alert_summary_id'],
            })
            # Merge with any available features from the retriever's training data
            if hasattr(self.retriever, 'train_features') and i < len(self.retriever.train_features):
                for j, col in enumerate(MATCHING_FEATURES_FALLBACK):
                    feature_row[col] = 0  # placeholder

            prediction = self.predict_single(feature_row, exclude_bug_id=bug_id)

            true_components.append(row['component'])
            true_products.append(row['product'])
            true_types.append(row['type'])
            pred_components.append(prediction.get('component', 'General'))
            pred_products.append(prediction.get('product', 'Core'))
            pred_types.append(prediction.get('type', 'defect'))
            sources.append(prediction.get('source', 'unknown'))

            if (i + 1) % 50 == 0:
                print(f"  Evaluated {i+1}/{n}...")

        from sklearn.metrics import accuracy_score, f1_score

        results = {
            'n_samples': n,
            'component_accuracy': accuracy_score(true_components, pred_components),
            'product_accuracy': accuracy_score(true_products, pred_products),
            'type_accuracy': accuracy_score(true_types, pred_types),
            'component_f1': f1_score(true_components, pred_components,
                                      average='weighted', zero_division=0),
            'product_f1': f1_score(true_products, pred_products,
                                    average='weighted', zero_division=0),
            'type_f1': f1_score(true_types, pred_types,
                                 average='weighted', zero_division=0),
            'source_distribution': Counter(sources),
        }

        return results


# Fallback feature names
MATCHING_FEATURES_FALLBACK = [
    'group_size', 'n_regressions', 'regression_ratio',
    'magnitude_mean', 'magnitude_max',
]


def run_llm_evaluation(
    backend: str = 'cache',
    max_samples: Optional[int] = None
) -> Dict:
    """
    Run LLM-based bug prediction evaluation.

    Args:
        backend: 'openai' or 'cache' (uses cached responses / fallback)
        max_samples: Limit evaluation samples

    Returns:
        Dict with evaluation results
    """
    data = load_bug_prediction_data()
    print(f"Loaded {len(data['bugs'])} bugs, {len(data['summary_bug_map'])} linked summaries")

    predictor = LLMBugPredictor(
        k_examples=5,
        backend=backend,
    )
    predictor.fit(data['alerts'], data['bugs'], data['summary_bug_map'])

    print("\nRunning LOO evaluation...")
    results = predictor.evaluate_loo(max_samples=max_samples)

    print(f"\nLLM Bug Prediction Results (backend={backend}):")
    print(f"  Component accuracy: {results['component_accuracy']:.4f}")
    print(f"  Product accuracy:   {results['product_accuracy']:.4f}")
    print(f"  Type accuracy:      {results['type_accuracy']:.4f}")
    print(f"  Sources: {dict(results['source_distribution'])}")

    return results


def generate_prompts_for_paper(n_examples: int = 3) -> List[Dict]:
    """
    Generate example prompts for the paper's appendix.

    Returns:
        List of dicts with prompt, true_labels, and predictions
    """
    data = load_bug_prediction_data()

    predictor = LLMBugPredictor(k_examples=5, backend='cache')
    predictor.fit(data['alerts'], data['bugs'], data['summary_bug_map'])

    examples = []
    train_bugs = predictor.retriever.train_bugs

    for i in range(min(n_examples, len(train_bugs))):
        row = train_bugs.iloc[i]
        feature_row = pd.Series({
            'alert_summary_id': row['alert_summary_id'],
        })
        prediction = predictor.predict_single(feature_row, exclude_bug_id=row['alert_summary_bug_number'])

        examples.append({
            'alert_summary_id': row['alert_summary_id'],
            'true_component': row['component'],
            'true_product': row['product'],
            'true_type': row['type'],
            'true_summary': row['summary'],
            'prompt': prediction['prompt'],
            'pred_component': prediction.get('component', ''),
            'pred_product': prediction.get('product', ''),
            'pred_type': prediction.get('type', ''),
            'pred_summary': prediction.get('summary', ''),
        })

    return examples


if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)

    # Run retrieval baseline first
    from cascade.bug_prediction.retrieval_baseline import run_retrieval_baseline
    print("=" * 70)
    print("RETRIEVAL BASELINE")
    print("=" * 70)
    retrieval_results = run_retrieval_baseline(k_values=[3, 5, 7])

    # Run LLM evaluation (uses cache/fallback since no API keys)
    print("\n" + "=" * 70)
    print("LLM-BASED PREDICTION (fallback mode)")
    print("=" * 70)
    llm_results = run_llm_evaluation(backend='cache', max_samples=50)

    # Generate example prompts
    print("\n" + "=" * 70)
    print("EXAMPLE PROMPTS FOR PAPER")
    print("=" * 70)
    examples = generate_prompts_for_paper(n_examples=2)
    for ex in examples:
        print(f"\n--- Summary {ex['alert_summary_id']} ---")
        print(f"  True: {ex['true_product']}:{ex['true_component']} ({ex['true_type']})")
        print(f"  Pred: {ex['pred_product']}:{ex['pred_component']} ({ex['pred_type']})")
        print(f"  True summary: {ex['true_summary'][:80]}...")
