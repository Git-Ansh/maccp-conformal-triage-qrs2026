"""
Explore Dirichlet uncertainty structure on Mozilla Perfherder has_bug data.

Uses k-means clustering on (quantile-transformed) features, then overlays
clusters onto the aleatoric-vs-epistemic scatter to identify what kinds of
alerts the model struggles with.

Outputs to cascade_outputs/dirichlet_experiment/:
  - cluster_uncertainty.png   : aleatoric vs epistemic colored by cluster
  - cluster_profiles.png      : per-cluster feature profiles (radar/bar)
  - cluster_table.txt         : per-cluster stats (size, bug rate, accuracy, uncertainty)
  - hardest_clusters.png      : zoom into worst-performing clusters
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

SRC_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(SRC_ROOT))

from cascade.data.loader import prepare_cascade_data
from cascade.stages.stage_3_bug_linkage import (
    STAGE_3_FEATURES, STAGE_3_TS_FEATURES, STAGE_3_CAT_FEATURES,
    NO_BUG_STATUSES,
)
from cascade.framework.dirichlet_classifier import DirichletClassifier
from common.data_paths import RANDOM_SEED

OUTPUT_DIR = SRC_ROOT.parent / 'cascade_outputs' / 'dirichlet_experiment'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = 8


# ---------------------------------------------------------------------------
# Data prep (same as dirichlet_experiment.py)
# ---------------------------------------------------------------------------

def prepare_features(summary_df):
    df = summary_df.copy()
    y = df['has_bug'].values
    cat_encoders = {}
    for col in STAGE_3_CAT_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_enc'] = le.fit_transform(df[col].astype(str).fillna('unknown'))
            cat_encoders[col] = le
    feature_cols = STAGE_3_FEATURES.copy()
    feature_cols += [c for c in STAGE_3_TS_FEATURES if c in df.columns]
    feature_cols += [c + '_enc' for c in STAGE_3_CAT_FEATURES if c in df.columns]
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].copy().fillna(0).values.astype(np.float32)
    return X, y, feature_cols, cat_encoders


def prepare_test_features(summary_df, feature_cols, cat_encoders):
    df = summary_df.copy()
    y = df['has_bug'].values
    for col, le in cat_encoders.items():
        if col in df.columns:
            vals = df[col].astype(str).fillna('unknown')
            known = set(le.classes_)
            fallback = 'unknown' if 'unknown' in known else le.classes_[0]
            vals = vals.apply(lambda x: x if x in known else fallback)
            df[col + '_enc'] = le.transform(vals)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    X = df[feature_cols].copy().fillna(0).values.astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_cluster_uncertainty(ale, epi, clusters, y_true, y_pred, n_clusters,
                            save_path):
    """Aleatoric vs epistemic scatter, colored by k-means cluster.
    Shape encodes correctness: circle=correct, X=incorrect."""
    correct = y_pred == y_true
    colors = cm.tab10(np.linspace(0, 1, n_clusters))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: colored by cluster
    ax = axes[0]
    for k in range(n_clusters):
        mask = clusters == k
        ax.scatter(ale[mask & correct], epi[mask & correct],
                   c=[colors[k]], marker='o', s=20, alpha=0.5,
                   label=f'C{k}', edgecolors='none')
        ax.scatter(ale[mask & ~correct], epi[mask & ~correct],
                   c=[colors[k]], marker='X', s=40, alpha=0.8,
                   edgecolors='black', linewidths=0.5)
    ax.set_xlabel('Aleatoric uncertainty')
    ax.set_ylabel('Epistemic uncertainty')
    ax.set_title('Clusters in Uncertainty Space\n(X = incorrect prediction)')
    ax.legend(fontsize=8, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Right: colored by true label
    ax = axes[1]
    bug_mask = y_true == 1
    ax.scatter(ale[~bug_mask & correct], epi[~bug_mask & correct],
               c='tab:blue', marker='o', s=15, alpha=0.4,
               label='no_bug (correct)', edgecolors='none')
    ax.scatter(ale[~bug_mask & ~correct], epi[~bug_mask & ~correct],
               c='tab:blue', marker='X', s=35, alpha=0.7,
               label='no_bug (wrong)', edgecolors='black', linewidths=0.5)
    ax.scatter(ale[bug_mask & correct], epi[bug_mask & correct],
               c='tab:red', marker='o', s=25, alpha=0.5,
               label='has_bug (correct)', edgecolors='none')
    ax.scatter(ale[bug_mask & ~correct], epi[bug_mask & ~correct],
               c='tab:red', marker='X', s=45, alpha=0.8,
               label='has_bug (wrong)', edgecolors='black', linewidths=0.5)
    ax.set_xlabel('Aleatoric uncertainty')
    ax.set_ylabel('Epistemic uncertainty')
    ax.set_title('True Labels in Uncertainty Space')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_cluster_profiles(X_test_scaled, feature_cols, clusters, y_true,
                          y_pred, result, n_clusters, save_path):
    """Per-cluster bar chart of mean feature values (z-scored), plus
    accuracy, bug rate, and mean uncertainty."""
    # Pick a readable subset of features (drop encoded categoricals)
    display_feats = [f for f in feature_cols if not f.endswith('_enc')]
    display_idx = [feature_cols.index(f) for f in display_feats]
    # Shorten names for display
    short = [f.replace('_mean', '').replace('_max', '_mx')
              .replace('ts_', '').replace('_rate', '_r') for f in display_feats]

    n_cols = 4
    n_rows = (n_clusters + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    correct = y_pred == y_true

    for k in range(n_clusters):
        ax = axes[k]
        mask = clusters == k
        n = mask.sum()
        acc = accuracy_score(y_true[mask], y_pred[mask]) if n > 0 else 0
        bug_rate = y_true[mask].mean() if n > 0 else 0
        ale_m = result['uncertainty']['aleatoric'][mask].mean()
        epi_m = result['uncertainty']['epistemic'][mask].mean()

        # Mean z-scored feature values for this cluster
        means = X_test_scaled[mask][:, display_idx].mean(axis=0)

        bars = ax.barh(range(len(short)), means, color='steelblue', alpha=0.7)
        # Color bars by sign
        for bar, v in zip(bars, means):
            if v < 0:
                bar.set_color('coral')
        ax.set_yticks(range(len(short)))
        ax.set_yticklabels(short, fontsize=6)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_title(
            f'C{k}: n={n}, acc={acc:.0%}, bug={bug_rate:.0%}\n'
            f'ale={ale_m:.3f}, epi={epi_m:.3f}',
            fontsize=9
        )
        ax.set_xlim(-2.5, 2.5)
        ax.grid(True, alpha=0.2)

    for i in range(n_clusters, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Cluster Feature Profiles (z-scored means)', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_hardest_clusters(X_test_scaled, feature_cols, clusters, y_true,
                          y_pred, result, n_clusters, save_path):
    """Zoom into the 2-3 worst-performing clusters: show their feature
    distributions compared to the dataset average, split by correct/wrong."""
    correct = y_pred == y_true

    # Find worst clusters by error rate (min 10 samples)
    cluster_err = []
    for k in range(n_clusters):
        mask = clusters == k
        n = mask.sum()
        if n >= 10:
            err = 1 - accuracy_score(y_true[mask], y_pred[mask])
            cluster_err.append((k, err, n))
    cluster_err.sort(key=lambda x: -x[1])
    worst = cluster_err[:3]

    if not worst:
        print("  No clusters with 10+ samples, skipping hardest_clusters plot")
        return

    # Pick top differentiating features per cluster
    display_feats = [f for f in feature_cols if not f.endswith('_enc')]
    display_idx = [feature_cols.index(f) for f in display_feats]
    global_means = X_test_scaled[:, display_idx].mean(axis=0)

    fig, axes = plt.subplots(1, len(worst), figsize=(7 * len(worst), 6))
    if len(worst) == 1:
        axes = [axes]

    for ax, (k, err, n) in zip(axes, worst):
        mask = clusters == k
        bug_rate = y_true[mask].mean()
        ale_m = result['uncertainty']['aleatoric'][mask].mean()
        epi_m = result['uncertainty']['epistemic'][mask].mean()

        cluster_means = X_test_scaled[mask][:, display_idx].mean(axis=0)
        diff = cluster_means - global_means

        # Sort by absolute difference, show top 12
        top_idx = np.argsort(np.abs(diff))[-12:][::-1]
        names = [display_feats[i] for i in top_idx]
        vals = diff[top_idx]
        short = [n.replace('_mean', '').replace('ts_', '')
                  .replace('_rate', '_r') for n in names]

        colors = ['coral' if v < 0 else 'steelblue' for v in vals]
        ax.barh(range(len(short)), vals, color=colors, alpha=0.7)
        ax.set_yticks(range(len(short)))
        ax.set_yticklabels(short, fontsize=8)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Deviation from dataset mean (z-scored)')
        ax.set_title(
            f'Cluster {k}: {n} samples, {err:.0%} error rate\n'
            f'bug_rate={bug_rate:.0%}, ale={ale_m:.3f}, epi={epi_m:.3f}',
            fontsize=10
        )
        ax.grid(True, alpha=0.2)

    fig.suptitle('Hardest Clusters: Feature Deviations from Average',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    plt.style.use('seaborn-v0_8-whitegrid')
    np.random.seed(RANDOM_SEED)

    # --- Load & prepare ---
    print("Loading data...")
    data = prepare_cascade_data()
    train_sum = data['train_summaries']
    test_sum = data['test_summaries']
    train_sum = train_sum[
        ~train_sum['alert_summary_status'].isin(NO_BUG_STATUSES)
    ].copy()

    X_train, y_train, feature_cols, cat_encoders = prepare_features(train_sum)
    X_test, y_test = prepare_test_features(test_sum, feature_cols, cat_encoders)
    print(f"Train: {len(y_train)}, Test: {len(y_test)}, Features: {len(feature_cols)}")

    # --- Train Dirichlet ---
    print("Training Dirichlet classifier...")
    dir_clf = DirichletClassifier(
        n_classes=2, hidden_dims=(128, 64), lr=1e-3,
        epochs=200, batch_size=256, kl_weight=0.1, annealing_steps=50,
    )
    dir_clf.fit(X_train, y_train)
    result = dir_clf.predict(X_test)
    y_pred = result['class']
    ale = result['uncertainty']['aleatoric']
    epi = result['uncertainty']['epistemic']

    correct = y_pred == y_test
    print(f"Test accuracy: {correct.mean():.1%}")

    # --- Quantile-transform for clustering (same reason as Dirichlet: extreme skew) ---
    qt = QuantileTransformer(
        output_distribution='normal', random_state=RANDOM_SEED,
        n_quantiles=min(len(X_train), 1000),
    )
    X_train_q = qt.fit_transform(X_train)
    X_test_q = qt.transform(X_test)
    # Additional z-score on top of quantile for cluster profile display
    scaler = StandardScaler()
    scaler.fit(X_train_q)
    X_test_s = scaler.transform(X_test_q)

    # --- K-Means on test set (quantile-transformed features) ---
    print(f"\nClustering test set into {N_CLUSTERS} clusters...")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
    clusters = km.fit_predict(X_test_q)

    # --- Cluster summary table ---
    print(f"\n{'Clust':>5} {'N':>5} {'Bug%':>6} {'Acc':>6} "
          f"{'Ale':>7} {'Epi':>7} {'Strength':>8}  Description")
    print("-" * 75)

    # Get dominant suite/platform per cluster for description
    test_df = test_sum.copy()
    test_df = test_df.reset_index(drop=True)

    for k in range(N_CLUSTERS):
        mask = clusters == k
        n = mask.sum()
        bug_rate = y_test[mask].mean()
        acc = accuracy_score(y_test[mask], y_pred[mask]) if n > 0 else 0
        ale_m = ale[mask].mean()
        epi_m = epi[mask].mean()
        str_m = result['strength'][mask].mean()

        # Find dominant suite for this cluster
        cluster_df = test_df.loc[mask]
        if 'dominant_suite' in cluster_df.columns:
            top_suite = cluster_df['dominant_suite'].mode().iloc[0] if n > 0 else '?'
        else:
            top_suite = '?'

        # Find most distinctive feature
        cluster_means = X_test_s[mask].mean(axis=0)
        global_means = X_test_s.mean(axis=0)
        diff = cluster_means - global_means
        non_enc = [i for i, f in enumerate(feature_cols) if not f.endswith('_enc')]
        best_feat_idx = non_enc[np.argmax(np.abs(diff[non_enc]))]
        best_feat = feature_cols[best_feat_idx]
        feat_dir = '+' if diff[best_feat_idx] > 0 else '-'

        print(f"  C{k:>2}  {n:>4}  {bug_rate:>5.0%}  {acc:>5.0%}  "
              f"{ale_m:>6.3f}  {epi_m:>6.3f}  {str_m:>7.1f}  "
              f"{top_suite[:20]:20s} ({feat_dir}{best_feat})")

    # Save table to file
    with open(OUTPUT_DIR / 'cluster_table.txt', 'w') as f:
        f.write(f"K-Means Clustering (K={N_CLUSTERS}) on test set (N={len(y_test)})\n")
        f.write(f"{'Clust':>5} {'N':>5} {'Bug%':>6} {'Acc':>6} "
                f"{'Ale':>7} {'Epi':>7} {'Strength':>8}\n")
        f.write("-" * 55 + "\n")
        for k in range(N_CLUSTERS):
            mask = clusters == k
            n = mask.sum()
            bug_rate = y_test[mask].mean()
            acc = accuracy_score(y_test[mask], y_pred[mask]) if n > 0 else 0
            ale_m = ale[mask].mean()
            epi_m = epi[mask].mean()
            str_m = result['strength'][mask].mean()
            f.write(f"  C{k:>2}  {n:>4}  {bug_rate:>5.0%}  {acc:>5.0%}  "
                    f"{ale_m:>6.3f}  {epi_m:>6.3f}  {str_m:>7.1f}\n")

    # --- Plots ---
    print("\nGenerating plots...")

    plot_cluster_uncertainty(
        ale, epi, clusters, y_test, y_pred, N_CLUSTERS,
        OUTPUT_DIR / 'cluster_uncertainty.png'
    )

    plot_cluster_profiles(
        X_test_s, feature_cols, clusters, y_test, y_pred, result,
        N_CLUSTERS, OUTPUT_DIR / 'cluster_profiles.png'
    )

    plot_hardest_clusters(
        X_test_s, feature_cols, clusters, y_test, y_pred, result,
        N_CLUSTERS, OUTPUT_DIR / 'cluster_table_hardest.png'
    )

    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
