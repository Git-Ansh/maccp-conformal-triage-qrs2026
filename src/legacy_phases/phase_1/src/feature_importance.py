"""
Phase 1: Feature Importance Module
Permutation importance and SHAP analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: SHAP not installed. Run: pip install shap")

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.visualization_utils import plot_feature_importance


def compute_permutation_importance(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
    scoring: str = 'f1',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compute permutation importance for features.

    Args:
        model: Fitted model
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        n_repeats: Number of permutation repeats
        scoring: Scoring metric
        random_state: Random state

    Returns:
        DataFrame with feature importances
    """
    print(f"Computing permutation importance (n_repeats={n_repeats})...")

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1
    )

    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': result.importances_mean,
        'std': result.importances_std
    })

    importance_df = importance_df.sort_values('importance', ascending=False)
    importance_df = importance_df.reset_index(drop=True)

    print(f"\nTop 10 features by permutation importance:")
    print(importance_df.head(10).to_string(index=False))

    return importance_df


def compute_tree_feature_importance(
    model: BaseEstimator,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Get feature importance from tree-based models.

    Args:
        model: Fitted tree-based model
        feature_names: List of feature names

    Returns:
        DataFrame with feature importances
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })

    importance_df = importance_df.sort_values('importance', ascending=False)
    importance_df = importance_df.reset_index(drop=True)

    print(f"\nTop 10 features by tree importance:")
    print(importance_df.head(10).to_string(index=False))

    return importance_df


def compute_shap_values(
    model: BaseEstimator,
    X: np.ndarray,
    feature_names: List[str],
    max_samples: int = 1000
) -> Tuple['shap.Explanation', pd.DataFrame]:
    """
    Compute SHAP values for model interpretation.

    Args:
        model: Fitted model
        X: Features to explain
        feature_names: List of feature names
        max_samples: Maximum samples to use

    Returns:
        Tuple of (SHAP explanation, importance DataFrame)
    """
    if not HAS_SHAP:
        raise ImportError("SHAP not installed. Run: pip install shap")

    print(f"Computing SHAP values...")

    # Subsample if needed
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X = X[idx]

    # Create explainer based on model type
    model_type = type(model).__name__

    if model_type in ['XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier']:
        explainer = shap.TreeExplainer(model)
    elif model_type in ['RandomForestClassifier', 'GradientBoostingClassifier']:
        explainer = shap.TreeExplainer(model)
    else:
        # Use KernelExplainer for other models
        explainer = shap.KernelExplainer(
            model.predict_proba,
            shap.sample(X, min(100, len(X)))
        )

    # Compute SHAP values
    shap_values = explainer.shap_values(X)

    # Handle binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class

    # Create explanation object
    explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value if not isinstance(explainer.expected_value, list)
                    else explainer.expected_value[1],
        data=X,
        feature_names=feature_names
    )

    # Compute mean absolute SHAP values for importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    })

    importance_df = importance_df.sort_values('importance', ascending=False)
    importance_df = importance_df.reset_index(drop=True)

    print(f"\nTop 10 features by SHAP importance:")
    print(importance_df.head(10).to_string(index=False))

    return explanation, importance_df


def plot_shap_summary(
    shap_explanation: 'shap.Explanation',
    save_path: Optional[Path] = None,
    max_display: int = 20
) -> None:
    """
    Plot SHAP summary plot.

    Args:
        shap_explanation: SHAP explanation object
        save_path: Path to save figure
        max_display: Maximum features to display
    """
    if not HAS_SHAP:
        raise ImportError("SHAP not installed")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_explanation,
        max_display=max_display,
        show=False
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"SHAP summary plot saved to {save_path}")

    plt.close()


def plot_shap_bar(
    shap_explanation: 'shap.Explanation',
    save_path: Optional[Path] = None,
    max_display: int = 20
) -> None:
    """
    Plot SHAP bar plot (feature importance).

    Args:
        shap_explanation: SHAP explanation object
        save_path: Path to save figure
        max_display: Maximum features to display
    """
    if not HAS_SHAP:
        raise ImportError("SHAP not installed")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    shap.plots.bar(
        shap_explanation,
        max_display=max_display,
        show=False
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"SHAP bar plot saved to {save_path}")

    plt.close()


def plot_shap_dependence(
    shap_explanation: 'shap.Explanation',
    feature: str,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot SHAP dependence plot for a specific feature.

    Args:
        shap_explanation: SHAP explanation object
        feature: Feature name
        save_path: Path to save figure
    """
    if not HAS_SHAP:
        raise ImportError("SHAP not installed")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature,
        shap_explanation.values,
        shap_explanation.data,
        feature_names=shap_explanation.feature_names,
        show=False
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"SHAP dependence plot saved to {save_path}")

    plt.close()


def run_full_importance_analysis(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    output_dir: Path
) -> Dict[str, pd.DataFrame]:
    """
    Run full feature importance analysis.

    Args:
        model: Fitted model
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        output_dir: Directory for outputs

    Returns:
        Dictionary with importance DataFrames
    """
    output_dir = Path(output_dir)
    reports_dir = output_dir / 'reports'
    figures_dir = output_dir / 'figures'
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 1. Permutation importance
    print("\n" + "="*50)
    print("Permutation Importance Analysis")
    print("="*50)

    perm_importance = compute_permutation_importance(
        model, X_test, y_test, feature_names
    )
    results['permutation'] = perm_importance
    perm_importance.to_csv(reports_dir / 'permutation_importance.csv', index=False)

    plot_feature_importance(
        perm_importance, top_n=20,
        title="Permutation Feature Importance",
        save_path=figures_dir / 'permutation_importance.png'
    )

    # 2. Tree importance (if available)
    if hasattr(model, 'feature_importances_'):
        print("\n" + "="*50)
        print("Tree-based Feature Importance")
        print("="*50)

        tree_importance = compute_tree_feature_importance(model, feature_names)
        results['tree'] = tree_importance
        tree_importance.to_csv(reports_dir / 'tree_importance.csv', index=False)

        plot_feature_importance(
            tree_importance, top_n=20,
            title="Tree-based Feature Importance",
            save_path=figures_dir / 'tree_importance.png'
        )

    # 3. SHAP analysis (if available)
    if HAS_SHAP:
        print("\n" + "="*50)
        print("SHAP Feature Importance")
        print("="*50)

        try:
            shap_explanation, shap_importance = compute_shap_values(
                model, X_test, feature_names
            )
            results['shap'] = shap_importance
            shap_importance.to_csv(reports_dir / 'shap_importance.csv', index=False)

            # SHAP plots
            plot_shap_summary(
                shap_explanation,
                save_path=figures_dir / 'shap_summary.png'
            )

            plot_shap_bar(
                shap_explanation,
                save_path=figures_dir / 'shap_bar.png'
            )

            # Top 5 dependence plots
            for i, feat in enumerate(shap_importance['feature'].head(5)):
                plot_shap_dependence(
                    shap_explanation, feat,
                    save_path=figures_dir / f'shap_dependence_{i+1}_{feat[:20]}.png'
                )

        except Exception as e:
            print(f"SHAP analysis failed: {e}")

    print(f"\nImportance analysis complete. Results saved to {output_dir}")

    return results


if __name__ == "__main__":
    # Test importance analysis
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    # Create test data
    X, y = make_classification(
        n_samples=500, n_features=10, n_classes=2,
        n_informative=5, random_state=42
    )
    feature_names = [f'feature_{i}' for i in range(10)]

    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Test permutation importance
    perm_df = compute_permutation_importance(model, X_test, y_test, feature_names)

    # Test tree importance
    tree_df = compute_tree_feature_importance(model, feature_names)
