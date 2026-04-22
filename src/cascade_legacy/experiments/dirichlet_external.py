"""
Dirichlet Evidential Classifier on external datasets.

For each dataset, trains a DirichletClassifier, prints metrics vs majority
baseline, and plots the predicted Dirichlet/Beta distribution for one
representative sample per class.

Datasets:
  - JM1 (binary, 2 classes) -> Beta plot
  - ServiceNow S0 Priority (4 classes) -> Dirichlet bar plot
  - Eclipse S1 Severity (7 classes) -> Dirichlet bar plot

Outputs to cascade_outputs/dirichlet_experiment/external/
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist, dirichlet as dirichlet_dist
from scipy.special import digamma
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, log_loss

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from cascade.framework.dirichlet_classifier import DirichletClassifier

OUTPUT_DIR = PROJECT_ROOT / 'cascade_outputs' / 'dirichlet_experiment' / 'external'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


def compute_uncertainty(alpha):
    """Compute uncertainty decomposition from alpha parameters."""
    S = alpha.sum(axis=1)
    proba = alpha / S[:, None]
    total = -np.sum(proba * np.log(proba + 1e-10), axis=1)
    aleatoric = -np.sum(
        (alpha / S[:, None]) * (digamma(alpha + 1) - digamma(S[:, None] + 1)),
        axis=1
    )
    epistemic = total - aleatoric
    return {'total': total, 'aleatoric': aleatoric, 'epistemic': epistemic}


def pick_representative(indices, strength):
    """Pick the sample closest to median strength among given indices."""
    if len(indices) == 0:
        return None
    s = strength[indices]
    return indices[np.argsort(np.abs(s - np.median(s)))[0]]


def plot_beta_distributions(result, y_test, class_names, dataset_name, save_path):
    """For K=2 (binary), plot Beta PDF for one representative per class + one wrong."""
    alpha = result['alpha']
    y_pred = result['class']
    strength = result['strength']
    correct = y_pred == y_test
    unc = compute_uncertainty(alpha)

    examples = []
    for cls in range(2):
        idx_correct = np.where((y_test == cls) & correct)[0]
        rep = pick_representative(idx_correct, strength)
        if rep is not None:
            examples.append((f'True: {class_names[cls]} (correct)', rep))

    # Add a wrong prediction
    wrong_idx = np.where(~correct)[0]
    rep = pick_representative(wrong_idx, strength)
    if rep is not None:
        true_cls = class_names[y_test[rep]]
        pred_cls = class_names[y_pred[rep]]
        examples.append((f'True: {true_cls}, Pred: {pred_cls}', rep))

    # Highest epistemic
    high_epi = np.argmax(unc['epistemic'])
    examples.append(('Highest epistemic', high_epi))

    x = np.linspace(0.001, 0.999, 500)
    n = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (title, idx) in zip(axes, examples):
        a0, a1 = alpha[idx, 0], alpha[idx, 1]
        pdf = beta_dist.pdf(x, a1, a0)  # P(class 1) ~ Beta(a1, a0)
        ax.fill_between(x, pdf, alpha=0.3, color='steelblue')
        ax.plot(x, pdf, color='steelblue', linewidth=2)

        p1 = a1 / (a0 + a1)
        ax.axvline(p1, color='black', linestyle='--', linewidth=1, alpha=0.6)
        ax.axvline(y_test[idx], color='red', linestyle='-', linewidth=2, alpha=0.4,
                   label=f'True: {class_names[y_test[idx]]}')

        ax.set_xlim(0, 1)
        ax.set_xlabel(f'P({class_names[1]})')
        ax.set_ylabel('Density')
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.legend(fontsize=7)

        txt = (f'a=[{a0:.1f}, {a1:.1f}]\n'
               f'S={a0+a1:.1f}\n'
               f'epi={unc["epistemic"][idx]:.3f}')
        ax.text(0.97 if p1 < 0.5 else 0.03, 0.95, txt,
                transform=ax.transAxes, fontsize=7, va='top',
                ha='right' if p1 < 0.5 else 'left',
                bbox=dict(boxstyle='round', fc='wheat', alpha=0.8))

    fig.suptitle(f'{dataset_name}: Beta Distribution over P({class_names[1]})',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_dirichlet_bars(result, y_test, class_names, dataset_name, save_path):
    """For K>2, show alpha bar charts for one representative per class + wrong + high epi."""
    alpha = result['alpha']
    y_pred = result['class']
    strength = result['strength']
    correct = y_pred == y_test
    K = len(class_names)
    unc = compute_uncertainty(alpha)

    examples = []
    # One correct representative per class (skip classes with <5 correct)
    for cls in range(K):
        idx_correct = np.where((y_test == cls) & correct)[0]
        if len(idx_correct) >= 5:
            rep = pick_representative(idx_correct, strength)
            examples.append((f'True: {class_names[cls]}', rep))

    # One misclassification
    wrong_idx = np.where(~correct)[0]
    if len(wrong_idx) > 0:
        rep = pick_representative(wrong_idx, strength)
        examples.append((f'WRONG (true={class_names[y_test[rep]]})', rep))

    # Highest epistemic
    high_epi = np.argmax(unc['epistemic'])
    examples.append(('Highest epistemic', high_epi))

    n = len(examples)
    n_cols = min(n, 4)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Short class names for x-axis
    short_names = [n[:8] for n in class_names.values()]

    for ax_i, (title, idx) in enumerate(zip(range(len(examples)), examples)):
        ax = axes[ax_i]
        title, idx = examples[ax_i]
        a = alpha[idx]
        p = result['proba'][idx]
        pred = y_pred[idx]
        true = y_test[idx]

        colors = ['#dd8452' if i == true else '#4c72b0' for i in range(K)]
        bars = ax.bar(range(K), a - 1, bottom=1, color=colors, alpha=0.7,
                      edgecolor='black', linewidth=0.4)
        ax.set_xticks(range(K))
        ax.set_xticklabels(short_names, fontsize=6, rotation=45, ha='right')
        ax.set_ylabel('alpha', fontsize=8)
        ax.axhline(1, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

        check = 'Y' if pred == true else 'X'
        ax.set_title(
            f'{title}\npred={class_names[pred]} [{check}]\n'
            f'S={strength[idx]:.1f}  epi={unc["epistemic"][idx]:.3f}',
            fontsize=7,
        )

    for i in range(len(examples), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'{dataset_name}: Dirichlet Alpha Parameters\n'
                 f'(orange = true class, blue = other classes)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def run_dataset(name, X_train, y_train, X_test, y_test, class_names, epochs=200):
    """Train Dirichlet classifier and report results."""
    n_classes = len(class_names)
    print(f"\n{'='*60}")
    print(f"{name}: {n_classes} classes, {len(X_train)} train, {len(X_test)} test")
    print(f"{'='*60}")

    # Majority baseline
    majority = int(np.argmax(np.bincount(y_train)))
    maj_acc = (y_test == majority).mean()
    print(f"  Majority baseline: {maj_acc:.1%} (class={class_names[majority]})")

    clf = DirichletClassifier(
        n_classes=n_classes, hidden_dims=(128, 64), lr=1e-3,
        epochs=epochs, batch_size=256, kl_weight=0.1, annealing_steps=50,
    )
    clf.fit(X_train.astype(np.float32), y_train)
    result = clf.predict(X_test.astype(np.float32))

    y_pred = result['class']
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, result['proba'], labels=list(range(n_classes)))
    unc = compute_uncertainty(result['alpha'])

    print(f"  Dirichlet accuracy: {acc:.1%} (lift: {acc - maj_acc:+.1%})")
    print(f"  Log-loss: {ll:.4f}")
    print(f"  Strength: mean={result['strength'].mean():.1f}, median={np.median(result['strength']):.1f}")
    print(f"  Epistemic: mean={unc['epistemic'].mean():.4f}")
    print(f"  Aleatoric: mean={unc['aleatoric'].mean():.4f}")

    correct = y_pred == y_test
    if (~correct).any():
        print(f"  Epistemic (correct):   {unc['epistemic'][correct].mean():.4f}")
        print(f"  Epistemic (incorrect): {unc['epistemic'][~correct].mean():.4f}")

    return result, class_names


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_jm1():
    from cascade_external.data.jm1_loader import load_jm1_data
    data = load_jm1_data()
    fc = data['feature_cols']
    X_train = data['train_df'][fc].fillna(0).values
    y_train = data['train_df']['defective'].values
    X_test = data['test_df'][fc].fillna(0).values
    y_test = data['test_df']['defective'].values
    classes = {0: 'clean', 1: 'defective'}
    return X_train, y_train, X_test, y_test, classes


def load_servicenow_s0():
    from cascade_external.data.servicenow_loader import prepare_servicenow_data
    from cascade_external.stages.servicenow_config import (
        prepare_stage_0_data, PRIORITY_CLASSES,
    )
    data = prepare_servicenow_data(top_n_groups=10)
    s0 = prepare_stage_0_data(
        data['train_df'], data['test_df'],
        data['numeric_features'], data['categorical_features'],
    )
    return s0['train_X'], s0['train_y'], s0['test_X'], s0['test_y'], PRIORITY_CLASSES


def load_eclipse_s1():
    from cascade_external.data.eclipse_zenodo_loader import prepare_eclipse_zenodo_data
    from cascade_external.stages.eclipse_config import (
        prepare_stage_1_data, SEVERITY_CLASSES,
    )
    data = prepare_eclipse_zenodo_data(max_per_project=5000)
    # Filter to non-noise for S1
    train_df = data['train_df'][data['train_df']['is_noise'] == 0].copy()
    test_df = data['test_df'][data['test_df']['is_noise'] == 0].copy()
    s1 = prepare_stage_1_data(
        train_df, test_df,
        data['numeric_features'], data['categorical_features'],
    )
    return s1['train_X'], s1['train_y'], s1['test_X'], s1['test_y'], SEVERITY_CLASSES


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    plt.style.use('seaborn-v0_8-whitegrid')
    np.random.seed(RANDOM_SEED)

    # --- JM1 (binary) ---
    try:
        X_tr, y_tr, X_te, y_te, classes = load_jm1()
        result, cls = run_dataset('JM1 Defect', X_tr, y_tr, X_te, y_te, classes)
        plot_beta_distributions(
            result, y_te, classes, 'JM1 Defect Prediction',
            OUTPUT_DIR / 'jm1_dirichlet.png',
        )
    except Exception as e:
        print(f"JM1 failed: {e}")

    # --- ServiceNow S0 (4-class) ---
    try:
        X_tr, y_tr, X_te, y_te, classes = load_servicenow_s0()
        result, cls = run_dataset('ServiceNow Priority', X_tr, y_tr, X_te, y_te, classes)
        plot_dirichlet_bars(
            result, y_te, classes, 'ServiceNow Priority',
            OUTPUT_DIR / 'servicenow_dirichlet.png',
        )
    except Exception as e:
        print(f"ServiceNow failed: {e}")

    # --- Eclipse S1 (7-class severity) ---
    try:
        X_tr, y_tr, X_te, y_te, classes = load_eclipse_s1()
        result, cls = run_dataset('Eclipse Severity', X_tr, y_tr, X_te, y_te, classes,
                                   epochs=100)  # large dataset, fewer epochs
        plot_dirichlet_bars(
            result, y_te, classes, 'Eclipse Severity',
            OUTPUT_DIR / 'eclipse_dirichlet.png',
        )
    except Exception as e:
        print(f"Eclipse failed: {e}")

    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
