"""
Dirichlet Evidential Classifier experiment on Mozilla Perfherder has_bug data.

Compares DirichletClassifier vs XGBoost baseline on Stage 3 (bug linkage)
using Mode B (no upstream disposition features) for independence.

Outputs:
  - Coverage-accuracy curves (both models)
  - Calibration reliability diagram
  - Uncertainty decomposition (aleatoric vs epistemic)
  - Example prediction distributions (Dirichlet simplex views)
  - All figures saved to cascade_outputs/dirichlet_experiment/
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, f1_score, brier_score_loss, log_loss,
    precision_score, recall_score
)

SRC_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(SRC_ROOT))

from cascade.data.loader import prepare_cascade_data
from cascade.stages.stage_3_bug_linkage import (
    STAGE_3_FEATURES, STAGE_3_TS_FEATURES, STAGE_3_CAT_FEATURES,
    NO_BUG_STATUSES,
)
from cascade.framework.dirichlet_classifier import DirichletClassifier
from common.data_paths import RANDOM_SEED

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

OUTPUT_DIR = SRC_ROOT.parent / 'cascade_outputs' / 'dirichlet_experiment'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data preparation (mirrors stage_3_bug_linkage.prepare_stage_3_data, Mode B)
# ---------------------------------------------------------------------------

def prepare_features(summary_df):
    """Extract Stage 3 features from summary DataFrame (Mode B)."""
    df = summary_df.copy()
    y = df['has_bug'].values

    cat_encoders = {}
    for col in STAGE_3_CAT_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_enc'] = le.fit_transform(df[col].astype(str).fillna('unknown'))
            cat_encoders[col] = le

    feature_cols = STAGE_3_FEATURES.copy()
    available_ts = [c for c in STAGE_3_TS_FEATURES if c in df.columns]
    feature_cols += available_ts
    feature_cols += [c + '_enc' for c in STAGE_3_CAT_FEATURES if c in df.columns]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy().fillna(0).values.astype(np.float32)
    return X, y, feature_cols, cat_encoders


def prepare_test_features(summary_df, feature_cols, cat_encoders):
    """Encode test set using train-fitted encoders."""
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
# Training
# ---------------------------------------------------------------------------

def train_xgboost_baseline(X_train, y_train):
    """Train XGBoost with scale_pos_weight grid search (matches Stage 3)."""
    n_bug = y_train.sum()
    n_no = len(y_train) - n_bug
    raw_ratio = n_no / n_bug if n_bug > 0 else 1

    # Detect GPU
    xgb_gpu = {}
    if HAS_XGBOOST:
        try:
            _t = XGBClassifier(tree_method='hist', device='cuda', n_estimators=1)
            _t.fit(np.random.rand(10, 2), np.random.randint(0, 2, 10))
            xgb_gpu = {'tree_method': 'hist', 'device': 'cuda'}
            del _t
        except Exception:
            pass

    best_spw, best_f1 = raw_ratio, 0
    for mult in [0.75, 1.0, 1.25, 1.5, 2.0]:
        spw = raw_ratio * mult
        model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=spw, random_state=RANDOM_SEED,
            eval_metric='logloss', n_jobs=-1, **xgb_gpu,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_train)
        f1 = f1_score(y_train, pred)
        if f1 > best_f1:
            best_f1, best_spw = f1, spw

    print(f"  XGBoost best scale_pos_weight: {best_spw:.1f}")
    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=best_spw, random_state=RANDOM_SEED,
        eval_metric='logloss', n_jobs=-1, **xgb_gpu,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def coverage_accuracy_curve(y_true, confidence, y_pred,
                            thresholds=np.arange(0.50, 1.00, 0.01)):
    """Coverage-accuracy at various confidence thresholds."""
    rows = []
    for t in thresholds:
        mask = confidence >= t
        n = mask.sum()
        if n == 0:
            continue
        acc = accuracy_score(y_true[mask], y_pred[mask])
        rows.append({'threshold': t, 'coverage': mask.mean(),
                     'accuracy': acc, 'n': n})
    return pd.DataFrame(rows)


def reliability_diagram_data(y_true, proba_positive, n_bins=10):
    """Compute calibration curve data."""
    fraction_pos, mean_pred = calibration_curve(
        y_true, proba_positive, n_bins=n_bins, strategy='uniform'
    )
    return mean_pred, fraction_pos


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_coverage_accuracy(curves, title, save_path):
    """Coverage-accuracy curves for multiple models."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, df in curves.items():
        ax.plot(df['coverage'], df['accuracy'], 'o-', label=name, markersize=3)
    ax.set_xlabel('Coverage (fraction predicted)')
    ax.set_ylabel('Accuracy (on predicted)')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.5, 1.01)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_reliability(models_data, save_path):
    """Reliability diagram (calibration) for multiple models."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.5)
    for name, (mean_pred, frac_pos) in models_data.items():
        ax.plot(mean_pred, frac_pos, 's-', label=name, markersize=5)
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Calibration: Reliability Diagram (has_bug)')
    ax.legend(loc='lower right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_uncertainty_decomposition(result, y_true, save_path):
    """Scatter: aleatoric vs epistemic, colored by correctness."""
    correct = result['class'] == y_true

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: aleatoric vs epistemic
    ax = axes[0]
    sc = ax.scatter(
        result['uncertainty']['aleatoric'][correct],
        result['uncertainty']['epistemic'][correct],
        c='tab:blue', alpha=0.4, s=15, label='Correct', edgecolors='none'
    )
    ax.scatter(
        result['uncertainty']['aleatoric'][~correct],
        result['uncertainty']['epistemic'][~correct],
        c='tab:red', alpha=0.6, s=25, label='Incorrect', edgecolors='none'
    )
    ax.set_xlabel('Aleatoric uncertainty')
    ax.set_ylabel('Epistemic uncertainty')
    ax.set_title('Uncertainty Decomposition')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: strength histogram by correctness (log scale to handle outliers)
    ax = axes[1]
    log_s_correct = np.log10(result['strength'][correct] + 1)
    log_s_wrong = np.log10(result['strength'][~correct] + 1)
    bins = np.linspace(0, max(log_s_correct.max(), log_s_wrong.max()) + 0.2, 31)
    ax.hist(log_s_correct, bins=bins, alpha=0.6,
            label='Correct', color='tab:blue', density=True)
    ax.hist(log_s_wrong, bins=bins, alpha=0.6,
            label='Incorrect', color='tab:red', density=True)
    ax.set_xlabel('log10(Dirichlet strength S)')
    ax.set_ylabel('Density')
    ax.set_title('Evidence Strength Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Dirichlet Classifier: Uncertainty Analysis', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_example_predictions(result, y_true, feature_names, X_test, save_path,
                             n_examples=8):
    """Show Dirichlet prediction distributions for selected examples.

    Picks examples across the uncertainty spectrum: 2 high-confidence correct,
    2 high-confidence incorrect, 2 high-epistemic, 2 high-aleatoric.
    For each, shows the Dirichlet alpha bar + predicted vs true class.
    """
    epi = result['uncertainty']['epistemic']
    ale = result['uncertainty']['aleatoric']
    correct = result['class'] == y_true
    conf = result['proba'].max(axis=1)
    strength = result['strength']

    # Select examples
    indices = {}

    # High confidence correct: pick from p75-p95 of strength (avoid extreme outliers)
    correct_idx = np.where(correct)[0]
    if len(correct_idx) >= 2:
        s_corr = strength[correct_idx]
        p75, p95 = np.percentile(s_corr, [75, 95])
        band = correct_idx[(s_corr >= p75) & (s_corr <= p95)]
        if len(band) >= 2:
            top = band[np.argsort(-conf[band])[:2]]
        else:
            top = correct_idx[np.argsort(-conf[correct_idx])[:2]]
        indices['Confident correct'] = top

    # High confidence incorrect (same approach)
    wrong_idx = np.where(~correct)[0]
    if len(wrong_idx) >= 2:
        s_wrong = strength[wrong_idx]
        p95 = np.percentile(s_wrong, 95)
        band = wrong_idx[s_wrong <= p95]
        if len(band) >= 2:
            top = band[np.argsort(-conf[band])[:2]]
        else:
            top = wrong_idx[np.argsort(-conf[wrong_idx])[:2]]
        indices['Confident incorrect'] = top

    # Highest epistemic
    top_epi = np.argsort(-epi)[:2]
    indices['High epistemic'] = top_epi

    # Highest aleatoric
    top_ale = np.argsort(-ale)[:2]
    indices['High aleatoric'] = top_ale

    all_idx = []
    labels = []
    for label, idx_arr in indices.items():
        for i in idx_arr:
            all_idx.append(i)
            labels.append(label)

    n = len(all_idx)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.5 * ((n + 1) // 2), 7))
    axes = axes.flatten()
    class_names = ['no_bug', 'has_bug']

    for ax_i, (idx, label) in enumerate(zip(all_idx, labels)):
        ax = axes[ax_i]
        alpha = result['alpha'][idx]
        proba = result['proba'][idx]
        pred = result['class'][idx]
        true = y_true[idx]

        colors = ['#4c72b0', '#dd8452']
        bars = ax.bar(class_names, alpha - 1, bottom=1, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=0.5)

        ax.set_ylabel('alpha')
        true_name = class_names[true]
        pred_name = class_names[pred]
        check = 'Y' if pred == true else 'X'
        ax.set_title(
            f'{label}\n'
            f'pred={pred_name} [{check}]  true={true_name}\n'
            f'S={strength[idx]:.1f}  epi={epi[idx]:.3f}  ale={ale[idx]:.3f}',
            fontsize=8,
        )
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        # Annotate alpha values
        for bar, a, p in zip(bars, alpha, proba):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'a={a:.1f}\np={p:.2f}', ha='center', va='bottom', fontsize=7)

    # Hide unused axes
    for ax_i in range(n, len(axes)):
        axes[ax_i].set_visible(False)

    fig.suptitle('Example Dirichlet Predictions (alpha bars above baseline=1)',
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    plt.style.use('seaborn-v0_8-whitegrid')
    np.random.seed(RANDOM_SEED)

    # --- Load data ---
    print("=" * 60)
    print("Loading data...")
    data = prepare_cascade_data()
    train_sum = data['train_summaries']
    test_sum = data['test_summaries']

    # Exclude Invalid from training (same as Stage 3)
    train_sum = train_sum[
        ~train_sum['alert_summary_status'].isin(NO_BUG_STATUSES)
    ].copy()

    # Prepare features
    X_train, y_train, feature_cols, cat_encoders = prepare_features(train_sum)
    X_test, y_test = prepare_test_features(test_sum, feature_cols, cat_encoders)

    n_bug_train = y_train.sum()
    n_bug_test = y_test.sum()
    print(f"Train: {len(y_train)} ({n_bug_train} has_bug, {len(y_train)-n_bug_train} no_bug)")
    print(f"Test:  {len(y_test)} ({n_bug_test} has_bug, {len(y_test)-n_bug_test} no_bug)")
    print(f"Features: {len(feature_cols)} ({sum(1 for c in feature_cols if c.startswith('ts_'))} TS)")

    # --- Scale features (fit on train only) ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ==================================================================
    # 1. XGBoost baseline
    # ==================================================================
    print("\n" + "=" * 60)
    print("Training XGBoost baseline...")
    xgb_model = train_xgboost_baseline(X_train_s, y_train)

    xgb_proba_test = xgb_model.predict_proba(X_test_s)
    xgb_pred = np.argmax(xgb_proba_test, axis=1)
    xgb_conf = np.max(xgb_proba_test, axis=1)

    # ==================================================================
    # 2. Dirichlet classifier
    # ==================================================================
    print("\n" + "=" * 60)
    print("Training Dirichlet classifier...")
    # DirichletClassifier does its own internal normalization, so pass raw features
    dir_clf = DirichletClassifier(
        n_classes=2, hidden_dims=(128, 64), lr=1e-3,
        epochs=200, batch_size=256, kl_weight=0.1,
        annealing_steps=50,
    )
    dir_clf.fit(X_train, y_train)

    dir_result = dir_clf.predict(X_test)
    dir_pred = dir_result['class']
    dir_proba = dir_result['proba']
    dir_conf = dir_proba.max(axis=1)

    # ==================================================================
    # 3. Metrics comparison
    # ==================================================================
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)

    for name, pred, proba in [('XGBoost', xgb_pred, xgb_proba_test),
                               ('Dirichlet', dir_pred, dir_proba)]:
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred)
        brier = brier_score_loss(y_test, proba[:, 1])
        ll = log_loss(y_test, proba)
        print(f"\n{name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  F1 (bug):  {f1:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  Brier:     {brier:.4f}")
        print(f"  Log-loss:  {ll:.4f}")

    # Dirichlet-specific stats
    print(f"\nDirichlet uncertainty stats (test):")
    print(f"  Strength:  mean={dir_result['strength'].mean():.1f}, "
          f"median={np.median(dir_result['strength']):.1f}")
    print(f"  Epistemic: mean={dir_result['uncertainty']['epistemic'].mean():.4f}")
    print(f"  Aleatoric: mean={dir_result['uncertainty']['aleatoric'].mean():.4f}")

    correct = dir_pred == y_test
    print(f"  Correct   epistemic: {dir_result['uncertainty']['epistemic'][correct].mean():.4f}")
    print(f"  Incorrect epistemic: {dir_result['uncertainty']['epistemic'][~correct].mean():.4f}")

    # ==================================================================
    # 4. Coverage-accuracy curves
    # ==================================================================
    print("\n" + "=" * 60)
    print("Generating plots...")

    thresholds = np.arange(0.50, 1.00, 0.01)

    xgb_curve = coverage_accuracy_curve(y_test, xgb_conf, xgb_pred, thresholds)
    dir_conf_curve = coverage_accuracy_curve(y_test, dir_conf, dir_pred, thresholds)

    # Epistemic curve: use percentile-based thresholds (epistemic is on a different scale)
    epi = dir_result['uncertainty']['epistemic']
    epi_rows = []
    for pct in np.arange(5, 100, 5):
        thresh = np.percentile(epi, pct)
        mask = epi <= thresh  # keep low-epistemic samples
        if mask.sum() == 0:
            continue
        acc = accuracy_score(y_test[mask], dir_pred[mask])
        epi_rows.append({'threshold': pct / 100, 'coverage': mask.mean(),
                         'accuracy': acc, 'n': mask.sum()})
    dir_epi_curve = pd.DataFrame(epi_rows)

    # Total uncertainty curve (same percentile approach)
    total = dir_result['uncertainty']['total']
    total_rows = []
    for pct in np.arange(5, 100, 5):
        thresh = np.percentile(total, pct)
        mask = total <= thresh
        if mask.sum() == 0:
            continue
        acc = accuracy_score(y_test[mask], dir_pred[mask])
        total_rows.append({'threshold': pct / 100, 'coverage': mask.mean(),
                           'accuracy': acc, 'n': mask.sum()})
    dir_total_curve = pd.DataFrame(total_rows)

    curves = {
        'XGBoost (max prob)': xgb_curve,
        'Dirichlet (max prob)': dir_conf_curve,
        'Dirichlet (epistemic)': dir_epi_curve,
        'Dirichlet (total unc.)': dir_total_curve,
    }
    plot_coverage_accuracy(
        curves,
        'Coverage-Accuracy: has_bug Prediction',
        OUTPUT_DIR / 'coverage_accuracy.png'
    )

    # Print key operating points
    print("\n  Coverage-accuracy summary:")
    for name, df in curves.items():
        for cov_target in [0.90, 0.80, 0.70]:
            row = df.iloc[(df['coverage'] - cov_target).abs().argsort()[:1]]
            if len(row):
                r = row.iloc[0]
                print(f"    {name}: cov={r['coverage']:.1%} -> acc={r['accuracy']:.1%}")

    # ==================================================================
    # 5. Calibration reliability diagram
    # ==================================================================
    xgb_cal = reliability_diagram_data(y_test, xgb_proba_test[:, 1])
    dir_cal = reliability_diagram_data(y_test, dir_proba[:, 1])

    plot_reliability(
        {'XGBoost': xgb_cal, 'Dirichlet': dir_cal},
        OUTPUT_DIR / 'calibration_reliability.png'
    )

    # Compute ECE (Expected Calibration Error)
    for name, (mean_pred, frac_pos) in [('XGBoost', xgb_cal), ('Dirichlet', dir_cal)]:
        ece = np.mean(np.abs(mean_pred - frac_pos))
        print(f"  {name} ECE: {ece:.4f}")

    # ==================================================================
    # 6. Uncertainty decomposition plot
    # ==================================================================
    plot_uncertainty_decomposition(
        dir_result, y_test,
        OUTPUT_DIR / 'uncertainty_decomposition.png'
    )

    # ==================================================================
    # 7. Example predictions
    # ==================================================================
    plot_example_predictions(
        dir_result, y_test, feature_cols, X_test,
        OUTPUT_DIR / 'example_predictions.png'
    )

    print("\n" + "=" * 60)
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
