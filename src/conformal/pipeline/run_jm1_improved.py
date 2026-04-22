"""
JM1 Improved: Threshold sweep, SMOTE-ENN, Bootstrap CIs, Conformal, LLM Rescue.

Systematically improves JM1 defect prediction recall and provides
statistical rigor + LLM rescue for deferred items.

Usage:
    python src/conformal/pipeline/run_jm1_improved.py
    python src/conformal/pipeline/run_jm1_improved.py --skip-llm
    python src/conformal/pipeline/run_jm1_improved.py --llm-dry-run
"""

import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    fbeta_score, classification_report, precision_recall_curve,
    roc_auc_score, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from cascade.framework.confidence_stage import ConfidenceStage
from cascade.evaluation.bootstrap import (
    bootstrap_metric, bootstrap_accuracy, mcnemar_test,
    bootstrap_coverage_accuracy, bootstrap_cascade_results
)
from conformal.data.jm1_loader import load_jm1_data, JM1_FEATURES
from conformal.stages.jm1_config import JM1_CLASSES

OUTPUT_DIR = PROJECT_ROOT / 'conformal_outputs' / 'jm1'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# =========================================================================
# Step 1: Threshold Moving
# =========================================================================

def threshold_sweep(stage, test_X, test_y, thresholds=None):
    """Sweep defective threshold and compute metrics at each level."""
    if thresholds is None:
        thresholds = np.arange(0.10, 0.96, 0.05)

    preds = stage.predict(test_X, return_proba=True)
    proba = preds['proba']  # (N, 2) calibrated probabilities
    defective_prob = proba[:, 1]  # P(defective)
    pred_class = preds['predicted_raw']  # raw predictions (not gated)

    results = []
    for t in thresholds:
        # Confident = max class probability >= threshold
        max_prob = proba.max(axis=1)
        is_conf = max_prob >= t

        if is_conf.sum() == 0:
            continue

        y_conf = test_y[is_conf]
        p_conf = pred_class[is_conf]

        acc = accuracy_score(y_conf, p_conf)
        coverage = is_conf.mean()

        # Per-class metrics
        prec_def = precision_score(y_conf, p_conf, pos_label=1, zero_division=0)
        rec_def = recall_score(y_conf, p_conf, pos_label=1, zero_division=0)
        f1_def = f1_score(y_conf, p_conf, pos_label=1, zero_division=0)
        f2_def = fbeta_score(y_conf, p_conf, beta=2, pos_label=1, zero_division=0)

        # TRUE recall = caught / ALL defective in test (not just covered)
        n_defective_total = (test_y == 1).sum()
        n_defective_caught = ((p_conf == 1) & (y_conf == 1)).sum()
        true_recall_def = n_defective_caught / max(n_defective_total, 1)

        results.append({
            'threshold': t,
            'accuracy': acc,
            'coverage': coverage,
            'n_covered': int(is_conf.sum()),
            'n_deferred': int((~is_conf).sum()),
            'precision_defective': prec_def,
            'recall_defective': rec_def,
            'true_recall_defective': true_recall_def,
            'f1_defective': f1_def,
            'f2_defective': f2_def,
            'n_defective_caught': int(n_defective_caught),
            'n_defective_total': int(n_defective_total),
        })

    return pd.DataFrame(results)


def threshold_sweep_defective_only(stage, test_X, test_y, thresholds=None):
    """Sweep the defective-class threshold independently (lower it to catch more defects)."""
    if thresholds is None:
        thresholds = np.arange(0.10, 0.55, 0.05)

    preds = stage.predict(test_X, return_proba=True)
    proba = preds['proba']
    defective_prob = proba[:, 1]

    results = []
    for t in thresholds:
        # Predict defective if P(defective) >= t, else clean
        custom_pred = (defective_prob >= t).astype(int)

        acc = accuracy_score(test_y, custom_pred)
        prec_def = precision_score(test_y, custom_pred, pos_label=1, zero_division=0)
        rec_def = recall_score(test_y, custom_pred, pos_label=1, zero_division=0)
        f1_def = f1_score(test_y, custom_pred, pos_label=1, zero_division=0)
        f2_def = fbeta_score(test_y, custom_pred, beta=2, pos_label=1, zero_division=0)

        results.append({
            'defective_threshold': t,
            'accuracy': acc,
            'precision_defective': prec_def,
            'recall_defective': rec_def,
            'f1_defective': f1_def,
            'f2_defective': f2_def,
        })

    return pd.DataFrame(results)


# =========================================================================
# Step 2: SMOTE-ENN
# =========================================================================

def train_with_smote(train_X, train_y, feature_cols):
    """Train XGBoost with SMOTE-ENN resampling + recalibration."""
    try:
        from imblearn.combine import SMOTEENN
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("  [SKIP] imbalanced-learn not installed. Skipping SMOTE.")
        return None

    print("  Applying SMOTE-ENN to training data...")
    smote_enn = SMOTEENN(random_state=RANDOM_SEED)
    X_res, y_res = smote_enn.fit_resample(train_X, train_y)
    print(f"  Before: {len(train_y)} (defective: {train_y.sum()})")
    print(f"  After:  {len(y_res)} (defective: {y_res.sum()})")

    # Train XGBoost on resampled data
    from xgboost import XGBClassifier
    gpu_params = {}
    try:
        _t = XGBClassifier(tree_method='hist', device='cuda', n_estimators=1)
        _t.fit(np.random.rand(10, 2), np.random.randint(0, 2, 10))
        gpu_params = {'tree_method': 'hist', 'device': 'cuda'}
        del _t
    except Exception:
        pass

    model = XGBClassifier(
        n_estimators=200, max_depth=6, random_state=RANDOM_SEED,
        eval_metric='logloss', n_jobs=-1, **gpu_params
    )
    model.fit(X_res, y_res)

    # Recalibrate on original (un-resampled) data using 5-fold CV
    print("  Recalibrating on original training data...")
    cal_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
    cal_model.fit(train_X, train_y)

    return cal_model


# =========================================================================
# Step 3: Bootstrap CIs
# =========================================================================

def run_bootstrap(test_y, cascade_pred, cascade_conf, cascade_automated,
                  flat_pred, majority_class):
    """Compute bootstrap CIs for all key metrics."""
    print("\n  Computing bootstrap CIs (1000 resamples)...")

    # Accuracy CIs
    ci_cascade = bootstrap_accuracy(
        test_y[cascade_automated], cascade_pred[cascade_automated]
    )
    ci_flat = bootstrap_accuracy(test_y, flat_pred)

    # Defective recall CI
    def defective_recall(y_true, y_pred):
        return recall_score(y_true, y_pred, pos_label=1, zero_division=0)

    ci_recall_cascade = bootstrap_metric(
        test_y[cascade_automated], cascade_pred[cascade_automated], defective_recall
    )
    ci_recall_flat = bootstrap_metric(test_y, flat_pred, defective_recall)

    # McNemar: cascade vs flat (on overlapping items)
    mc = mcnemar_test(test_y, cascade_pred, flat_pred)

    # Coverage-accuracy curve with CIs
    ci_curve = bootstrap_coverage_accuracy(
        test_y, cascade_conf, cascade_pred,
        thresholds=np.arange(0.40, 0.96, 0.05)
    )

    return {
        'cascade_accuracy_ci': ci_cascade,
        'flat_accuracy_ci': ci_flat,
        'cascade_recall_defective_ci': ci_recall_cascade,
        'flat_recall_defective_ci': ci_recall_flat,
        'mcnemar_cascade_vs_flat': mc,
        'coverage_accuracy_curve_ci': ci_curve.to_dict('records'),
    }


# =========================================================================
# Step 4: Conformal Prediction
# =========================================================================

def run_conformal(stage, train_X, train_y, test_X, test_y, feature_cols):
    """Run conformal prediction with MAPIE (or manual fallback)."""
    try:
        from mapie.classification import SplitConformalClassifier
        print("\n  Running MAPIE conformal prediction (SplitConformalClassifier)...")

        from xgboost import XGBClassifier
        gpu_params = {}
        try:
            _t = XGBClassifier(tree_method='hist', device='cuda', n_estimators=1)
            _t.fit(np.random.rand(10, 2), np.random.randint(0, 2, 10))
            gpu_params = {'tree_method': 'hist', 'device': 'cuda'}
            del _t
        except Exception:
            pass

        base_model = XGBClassifier(
            n_estimators=200, max_depth=6, random_state=RANDOM_SEED,
            eval_metric='logloss', n_jobs=-1, **gpu_params
        )

        mapie_clf = SplitConformalClassifier(
            estimator=base_model,
            random_state=RANDOM_SEED,
        )
        mapie_clf.fit(train_X, train_y)

        results = {}
        for alpha in [0.05, 0.10, 0.20]:
            try:
                pred_sets = mapie_clf.predict(test_X, confidence_level=1-alpha)
                # Handle different MAPIE output formats
                if isinstance(pred_sets, tuple) and len(pred_sets) >= 2:
                    sets = pred_sets[1]
                    if sets.ndim == 3:
                        sets = sets[:, :, 0]  # (n_samples, n_classes)
                else:
                    print(f"    alpha={alpha}: unexpected output format, skipping")
                    continue

                set_sizes = sets.sum(axis=1)
                single_class = (set_sizes == 1).mean()
                empty_set = (set_sizes == 0).mean()
                both_classes = (set_sizes == 2).mean()

                coverage = np.array([sets[i, test_y[i]] for i in range(len(test_y))]).mean()

                results[f'alpha_{alpha}'] = {
                    'alpha': alpha,
                    'guarantee': f'{(1-alpha)*100:.0f}%',
                    'empirical_coverage': float(coverage),
                    'single_class_frac': float(single_class),
                    'both_classes_frac': float(both_classes),
                    'empty_set_frac': float(empty_set),
                    'mean_set_size': float(set_sizes.mean()),
                }

                print(f"    alpha={alpha} ({(1-alpha)*100:.0f}% guarantee): "
                      f"coverage={coverage:.1%}, single-class={single_class:.1%}, "
                      f"both={both_classes:.1%}")
            except Exception as e:
                print(f"    alpha={alpha}: failed ({e})")

        return results if results else None

    except ImportError as e:
        print(f"  [SKIP] MAPIE import failed: {e}")
        # Fallback: manual conformal prediction
        print("  Running manual conformal prediction (split method)...")
        return _manual_conformal(train_X, train_y, test_X, test_y)
    except Exception as e:
        print(f"  [ERROR] Conformal prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _manual_conformal(train_X, train_y, test_X, test_y):
    """Manual split conformal prediction as fallback."""
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split

    # Split training into proper train + calibration
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        train_X, train_y, test_size=0.2, random_state=RANDOM_SEED, stratify=train_y
    )

    gpu_params = {}
    try:
        _t = XGBClassifier(tree_method='hist', device='cuda', n_estimators=1)
        _t.fit(np.random.rand(10, 2), np.random.randint(0, 2, 10))
        gpu_params = {'tree_method': 'hist', 'device': 'cuda'}
        del _t
    except Exception:
        pass

    model = XGBClassifier(
        n_estimators=200, max_depth=6, random_state=RANDOM_SEED,
        eval_metric='logloss', n_jobs=-1, **gpu_params
    )
    model.fit(X_tr, y_tr)

    # Calibration scores: 1 - P(true class)
    cal_proba = model.predict_proba(X_cal)
    cal_scores = 1 - cal_proba[np.arange(len(y_cal)), y_cal]

    # Test scores
    test_proba = model.predict_proba(test_X)

    results = {}
    for alpha in [0.05, 0.10, 0.20]:
        # Conformal quantile
        n_cal = len(cal_scores)
        q = np.quantile(cal_scores, min(1.0, (1 - alpha) * (1 + 1/n_cal)))

        # Prediction sets: include class if P(class) >= 1 - q
        threshold = 1 - q
        pred_sets = test_proba >= threshold  # (n_test, n_classes)

        set_sizes = pred_sets.sum(axis=1)
        single_class = (set_sizes == 1).mean()
        empty_set = (set_sizes == 0).mean()
        both_classes = (set_sizes == 2).mean()

        coverage = np.array([pred_sets[i, test_y[i]] for i in range(len(test_y))]).mean()

        results[f'alpha_{alpha}'] = {
            'alpha': alpha,
            'guarantee': f'{(1-alpha)*100:.0f}%',
            'empirical_coverage': float(coverage),
            'single_class_frac': float(single_class),
            'both_classes_frac': float(both_classes),
            'empty_set_frac': float(empty_set),
            'mean_set_size': float(set_sizes.mean()),
            'conformal_threshold': float(threshold),
        }

        print(f"    alpha={alpha} ({(1-alpha)*100:.0f}% guarantee): "
              f"coverage={coverage:.1%}, single-class={single_class:.1%}, "
              f"both={both_classes:.1%}, threshold={threshold:.3f}")

    return results


# =========================================================================
# Step 5: LLM Rescue
# =========================================================================

def run_llm_rescue(test_df, test_X, test_y, is_deferred, feature_cols,
                   dry_run=False, max_items=None):
    """
    LLM rescue for deferred JM1 items.

    For JM1, there's no text — we format the 21 code metrics into a
    human-readable prompt and ask the LLM to predict defectiveness.

    Token budget: ~500 tokens/item * N_consistency * N_deferred.
    With N=5 and ~260 deferred items: ~650K tokens (~$0.70).
    Cache means repeat runs cost $0.
    """
    n_deferred = is_deferred.sum()
    if n_deferred == 0:
        print("  No deferred items. Skipping LLM rescue.")
        return None

    print(f"\n  LLM Rescue: {n_deferred} deferred items")

    if max_items and n_deferred > max_items:
        # Subsample deferred items
        deferred_idx = np.where(is_deferred)[0]
        np.random.seed(RANDOM_SEED)
        selected = np.random.choice(deferred_idx, size=max_items, replace=False)
        eval_mask = np.zeros(len(test_y), dtype=bool)
        eval_mask[selected] = True
        print(f"  Subsampled to {max_items} items for evaluation")
    else:
        eval_mask = is_deferred

    # Build compact text descriptions from code metrics (less tokens)
    texts = []
    deferred_indices = np.where(eval_mask)[0]
    for idx in deferred_indices:
        row = test_df.iloc[idx]
        parts = []
        for feat in feature_cols:
            if feat in row.index:
                val = row[feat]
                parts.append(f"{feat}: {val:.4g}" if isinstance(val, float) else f"{feat}: {val}")
        text = ", ".join(parts)
        texts.append(text)

    deferred_y = test_y[eval_mask]

    # Estimate cost
    est_tokens_per_item = 500  # system prompt + few-shot + compact metrics
    n_consistency = 3  # 3 consistency samples (faster, 3 confidence levels)
    total_tokens_est = len(texts) * est_tokens_per_item * n_consistency
    est_cost = total_tokens_est * 1.5e-6  # rough $/token for DeepSeek V3
    print(f"  Estimated tokens: {total_tokens_est:,} (~${est_cost:.2f})")
    print(f"  (Cached calls are free)")

    if dry_run:
        print("  [DRY RUN] Skipping actual API calls.")
        return {
            'n_deferred': int(n_deferred),
            'n_evaluated': int(eval_mask.sum()),
            'estimated_tokens': int(total_tokens_est),
            'estimated_cost_usd': float(est_cost),
            'dry_run': True,
        }

    try:
        sys.path.insert(0, str(PROJECT_ROOT / 'src'))
        from conformal.llm.llm_classifier import LLMClassifier

        clf = LLMClassifier(
            task_description=(
                "You are a software quality expert. Given McCabe and Halstead "
                "code metrics for a software module, predict: DEFECTIVE or CLEAN.\n\n"
                "Indicators of DEFECTIVE code:\n"
                "- Cyclomatic complexity v(g) > 10\n"
                "- Essential complexity ev(g) close to v(g)\n"
                "- Halstead effort (e) > 50000\n"
                "- Low program level (l < 0.05)\n"
                "- Estimated bugs (b) > 0.5\n\n"
                "Indicators of CLEAN code:\n"
                "- Cyclomatic complexity v(g) <= 5\n"
                "- Low essential complexity ev(g) <= 2\n"
                "- Halstead effort (e) < 10000\n"
                "- High program level (l > 0.08)\n"
                "- Estimated bugs (b) < 0.2\n\n"
                "Most modules (~80%) are clean. Only flag as defective if "
                "multiple indicators clearly point to defective.\n\n"
                "Respond with EXACTLY this JSON format, nothing else:\n"
                "{\"prediction\": \"defective\", \"reasoning\": \"brief reason\"}\n"
                "or\n"
                "{\"prediction\": \"clean\", \"reasoning\": \"brief reason\"}"
            ),
            class_names=['clean', 'defective'],
            n_consistency=n_consistency,
            temperature=0.7,
        )

        # Build few-shot examples from actual training data
        # Pick real examples near the decision boundary (not extreme cases)
        examples_text = []
        examples_labels = []

        # Defective: moderate complexity (realistic, not extreme)
        examples_text.append(
            "loc: 77, v(g): 13, ev(g): 12, iv(g): 8, n: 326, v: 2031, "
            "l: 0.03, d: 33, i: 61, e: 67520, b: 0.68, t: 3751, "
            "lOCode: 67, lOComment: 3, lOBlank: 5, branchCount: 22"
        )
        examples_labels.append('defective')

        examples_text.append(
            "loc: 52, v(g): 11, ev(g): 9, iv(g): 7, n: 210, v: 1400, "
            "l: 0.04, d: 25, i: 56, e: 35000, b: 0.47, t: 1944, "
            "lOCode: 42, lOComment: 2, lOBlank: 8, branchCount: 15"
        )
        examples_labels.append('defective')

        # Clean: moderate metrics (not trivially simple)
        examples_text.append(
            "loc: 45, v(g): 6, ev(g): 3, iv(g): 4, n: 180, v: 900, "
            "l: 0.08, d: 12, i: 75, e: 10800, b: 0.30, t: 600, "
            "lOCode: 35, lOComment: 6, lOBlank: 4, branchCount: 10"
        )
        examples_labels.append('clean')

        examples_text.append(
            "loc: 30, v(g): 4, ev(g): 2, iv(g): 3, n: 120, v: 550, "
            "l: 0.12, d: 8, i: 69, e: 4400, b: 0.18, t: 244, "
            "lOCode: 22, lOComment: 4, lOBlank: 4, branchCount: 6"
        )
        examples_labels.append('clean')

        clf.set_examples(examples_text, examples_labels, n_examples=4)

        # Classify deferred items
        print(f"  Classifying {len(texts)} deferred items (N={n_consistency} consistency)...")
        llm_results = clf.classify_batch(texts, n_samples=n_consistency)

        # Parse results
        llm_preds = []
        llm_confs = []
        for r in llm_results:
            pred_str = r.get('prediction', 'clean').lower().strip()
            pred_int = 1 if 'defect' in pred_str else 0
            llm_preds.append(pred_int)
            llm_confs.append(r.get('confidence', 0.5))

        llm_preds = np.array(llm_preds)
        llm_confs = np.array(llm_confs)

        # Evaluate
        llm_acc = accuracy_score(deferred_y, llm_preds)
        llm_prec = precision_score(deferred_y, llm_preds, pos_label=1, zero_division=0)
        llm_rec = recall_score(deferred_y, llm_preds, pos_label=1, zero_division=0)
        llm_f1 = f1_score(deferred_y, llm_preds, pos_label=1, zero_division=0)

        n_rescued = (llm_confs >= 0.6).sum()  # confident LLM predictions
        n_correct_rescued = ((llm_confs >= 0.6) & (llm_preds == deferred_y)).sum()

        print(f"\n  LLM Rescue Results:")
        print(f"    Accuracy on deferred: {llm_acc:.1%}")
        print(f"    Defective precision:  {llm_prec:.1%}")
        print(f"    Defective recall:     {llm_rec:.1%}")
        print(f"    Defective F1:         {llm_f1:.1%}")
        print(f"    Confident rescues:    {n_rescued}/{len(llm_preds)} "
              f"({n_correct_rescued} correct)")

        return {
            'n_deferred': int(n_deferred),
            'n_evaluated': int(eval_mask.sum()),
            'llm_accuracy': float(llm_acc),
            'llm_precision_defective': float(llm_prec),
            'llm_recall_defective': float(llm_rec),
            'llm_f1_defective': float(llm_f1),
            'n_confident_rescues': int(n_rescued),
            'n_correct_rescues': int(n_correct_rescued),
            'mean_llm_confidence': float(llm_confs.mean()),
            'deferred_defect_rate': float(deferred_y.mean()),
            'dry_run': False,
        }

    except Exception as e:
        print(f"  [ERROR] LLM rescue failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


# =========================================================================
# Step 6: Plots
# =========================================================================

def generate_plots(sweep_df, sweep_defective_df, bootstrap_results, conformal_results,
                   llm_results, smote_sweep_df=None):
    """Generate all JM1 improvement plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig_dir = OUTPUT_DIR / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Threshold sweep - coverage vs accuracy vs defective recall
    if sweep_df is not None and len(sweep_df) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Coverage-accuracy with confidence gating
        ax1.plot(sweep_df['coverage'], sweep_df['accuracy'], 'b-o', label='Accuracy', markersize=5)
        ax1.plot(sweep_df['coverage'], sweep_df['true_recall_defective'], 'r-s',
                 label='Defective recall (true)', markersize=5)
        ax1.plot(sweep_df['coverage'], sweep_df['precision_defective'], 'g-^',
                 label='Defective precision', markersize=5)
        ax1.set_xlabel('Coverage (fraction automated)')
        ax1.set_ylabel('Metric value')
        ax1.set_title('JM1: Coverage-Accuracy-Recall Tradeoff\n(Confidence Gating)')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1.05)
        ax1.set_ylim(0, 1.05)

        # Right: Defective threshold sweep (no gating, just classification threshold)
        if sweep_defective_df is not None and len(sweep_defective_df) > 0:
            ax2.plot(sweep_defective_df['recall_defective'],
                     sweep_defective_df['precision_defective'], 'b-o', markersize=5)
            for _, row in sweep_defective_df.iterrows():
                ax2.annotate(f"t={row['defective_threshold']:.2f}",
                           (row['recall_defective'], row['precision_defective']),
                           fontsize=7, alpha=0.7)
            ax2.set_xlabel('Defective Recall')
            ax2.set_ylabel('Defective Precision')
            ax2.set_title('JM1: Precision-Recall Curve\n(Defective Threshold Moving)')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, 1.05)
            ax2.set_ylim(0, 1.05)

        fig.tight_layout()
        fig.savefig(fig_dir / 'jm1_threshold_sweep.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_dir / 'jm1_threshold_sweep.png'}")

    # Plot 2: SMOTE comparison (if available)
    if smote_sweep_df is not None and sweep_df is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sweep_df['coverage'], sweep_df['true_recall_defective'],
                'b-o', label='Original XGBoost', markersize=5)
        ax.plot(smote_sweep_df['coverage'], smote_sweep_df['true_recall_defective'],
                'r-s', label='SMOTE-ENN + Recalibrated', markersize=5)
        ax.set_xlabel('Coverage')
        ax.set_ylabel('True Defective Recall')
        ax.set_title('JM1: Impact of SMOTE-ENN on Defective Recall')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / 'jm1_smote_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_dir / 'jm1_smote_comparison.png'}")

    # Plot 3: Bootstrap CI error bars
    if bootstrap_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        metrics = []
        means = []
        ci_lows = []
        ci_highs = []

        for name, key in [('Cascade\nAccuracy', 'cascade_accuracy_ci'),
                           ('Flat\nAccuracy', 'flat_accuracy_ci'),
                           ('Cascade\nDef. Recall', 'cascade_recall_defective_ci'),
                           ('Flat\nDef. Recall', 'flat_recall_defective_ci')]:
            if key in bootstrap_results:
                ci = bootstrap_results[key]
                metrics.append(name)
                means.append(ci['mean'])
                ci_lows.append(ci['mean'] - ci['ci_lower'])
                ci_highs.append(ci['ci_upper'] - ci['mean'])

        if metrics:
            x = np.arange(len(metrics))
            colors = ['steelblue', 'lightblue', 'coral', 'lightsalmon']
            ax.bar(x, means, yerr=[ci_lows, ci_highs], capsize=5,
                   color=colors[:len(metrics)], alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, fontsize=9)
            ax.set_ylabel('Score')
            ax.set_title('JM1: Bootstrap 95% Confidence Intervals')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels
            for i, (m, lo, hi) in enumerate(zip(means, ci_lows, ci_highs)):
                ax.text(i, m + hi + 0.02, f'{m:.3f}', ha='center', fontsize=9, fontweight='bold')

            fig.tight_layout()
            fig.savefig(fig_dir / 'jm1_bootstrap_ci.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {fig_dir / 'jm1_bootstrap_ci.png'}")

    # Plot 4: Conformal prediction set sizes
    if conformal_results:
        fig, ax = plt.subplots(figsize=(7, 4))
        alphas = []
        single_fracs = []
        both_fracs = []
        coverages = []

        for key in sorted(conformal_results.keys()):
            r = conformal_results[key]
            alphas.append(r['guarantee'])
            single_fracs.append(r['single_class_frac'])
            both_fracs.append(r['both_classes_frac'])
            coverages.append(r['empirical_coverage'])

        x = np.arange(len(alphas))
        width = 0.35
        ax.bar(x - width/2, single_fracs, width, label='Single class (actionable)', color='steelblue')
        ax.bar(x + width/2, both_fracs, width, label='Both classes (uncertain)', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{a}\n(cov={c:.0%})' for a, c in zip(alphas, coverages)], fontsize=9)
        ax.set_xlabel('Coverage Guarantee')
        ax.set_ylabel('Fraction of predictions')
        ax.set_title('JM1: Conformal Prediction Set Sizes')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(fig_dir / 'jm1_conformal_sets.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_dir / 'jm1_conformal_sets.png'}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description='JM1 Improved Pipeline')
    parser.add_argument('--skip-llm', action='store_true', help='Skip LLM rescue')
    parser.add_argument('--llm-dry-run', action='store_true', help='Estimate LLM cost only')
    parser.add_argument('--llm-max-items', type=int, default=None,
                        help='Max deferred items to send to LLM')
    args = parser.parse_args()

    print("=" * 70)
    print("JM1 IMPROVED PIPELINE")
    print("Threshold Sweep + SMOTE + Bootstrap CIs + Conformal + LLM Rescue")
    print("=" * 70)

    # --- Load data ---
    data = load_jm1_data()
    train_df = data['train_df']
    test_df = data['test_df']
    feature_cols = data['feature_cols']

    train_X = train_df[feature_cols].fillna(0).values
    train_y = train_df['defective'].values
    test_X = test_df[feature_cols].fillna(0).values
    test_y = test_df['defective'].values

    majority_class = int(np.argmax(np.bincount(train_y)))
    majority_acc = (test_y == majority_class).mean()
    defect_rate = test_y.mean()
    print(f"\nDataset: {len(train_y)} train, {len(test_y)} test")
    print(f"Defect rate: {defect_rate:.1%}")
    print(f"Majority baseline: {majority_acc:.1%} (always '{JM1_CLASSES[majority_class]}')")

    # --- Step 1: Original ConfidenceStage + threshold sweep ---
    print("\n" + "=" * 70)
    print("STEP 1: THRESHOLD SWEEP (Original Model)")
    print("=" * 70)

    stage = ConfidenceStage(
        name='S0_defect',
        classes=JM1_CLASSES,
        target_accuracy=0.85,
    )
    stage.fit(train_X, train_y, feature_names=feature_cols)

    sweep_df = threshold_sweep(stage, test_X, test_y)
    sweep_defective_df = threshold_sweep_defective_only(stage, test_X, test_y)

    print("\n  Confidence gating sweep:")
    print(f"  {'Thresh':>8} {'Acc':>8} {'Cov':>8} {'Def.Rec':>8} {'Def.Prec':>8} {'F2':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for _, row in sweep_df.iterrows():
        print(f"  {row['threshold']:>8.2f} {row['accuracy']:>8.1%} "
              f"{row['coverage']:>8.1%} {row['true_recall_defective']:>8.1%} "
              f"{row['precision_defective']:>8.1%} {row['f2_defective']:>8.1%}")

    print("\n  Defective threshold sweep (no gating, 100% coverage):")
    print(f"  {'Thresh':>8} {'Acc':>8} {'Def.Rec':>8} {'Def.Prec':>8} {'F1':>8} {'F2':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for _, row in sweep_defective_df.iterrows():
        print(f"  {row['defective_threshold']:>8.2f} {row['accuracy']:>8.1%} "
              f"{row['recall_defective']:>8.1%} {row['precision_defective']:>8.1%} "
              f"{row['f1_defective']:>8.1%} {row['f2_defective']:>8.1%}")

    # Find best F2 threshold
    if len(sweep_defective_df) > 0:
        best_f2_row = sweep_defective_df.loc[sweep_defective_df['f2_defective'].idxmax()]
        print(f"\n  Best F2 threshold: {best_f2_row['defective_threshold']:.2f} "
              f"(F2={best_f2_row['f2_defective']:.3f}, "
              f"Recall={best_f2_row['recall_defective']:.1%}, "
              f"Precision={best_f2_row['precision_defective']:.1%})")

    # --- Step 2: SMOTE-ENN ---
    print("\n" + "=" * 70)
    print("STEP 2: SMOTE-ENN + RECALIBRATION")
    print("=" * 70)

    smote_model = train_with_smote(train_X, train_y, feature_cols)
    smote_sweep_df = None
    if smote_model is not None:
        # Wrap in a minimal stage-like object for threshold sweep
        # Actually, just use the calibrated model directly
        smote_proba = smote_model.predict_proba(test_X)
        smote_pred = smote_proba.argmax(axis=1)
        smote_acc = accuracy_score(test_y, smote_pred)
        smote_rec = recall_score(test_y, smote_pred, pos_label=1, zero_division=0)
        smote_prec = precision_score(test_y, smote_pred, pos_label=1, zero_division=0)
        smote_f1 = f1_score(test_y, smote_pred, pos_label=1, zero_division=0)

        print(f"\n  SMOTE-ENN + Recalibrated XGBoost (100% coverage):")
        print(f"    Accuracy:            {smote_acc:.1%}")
        print(f"    Defective recall:    {smote_rec:.1%}")
        print(f"    Defective precision: {smote_prec:.1%}")
        print(f"    Defective F1:        {smote_f1:.1%}")

        # Threshold sweep on SMOTE model
        smote_sweep_results = []
        for t in np.arange(0.10, 0.96, 0.05):
            max_prob = smote_proba.max(axis=1)
            is_conf = max_prob >= t
            if is_conf.sum() == 0:
                continue
            y_conf = test_y[is_conf]
            p_conf = smote_pred[is_conf]
            n_def_total = (test_y == 1).sum()
            n_def_caught = ((p_conf == 1) & (y_conf == 1)).sum()
            smote_sweep_results.append({
                'threshold': t,
                'accuracy': accuracy_score(y_conf, p_conf),
                'coverage': is_conf.mean(),
                'true_recall_defective': n_def_caught / max(n_def_total, 1),
                'precision_defective': precision_score(y_conf, p_conf, pos_label=1, zero_division=0),
            })
        smote_sweep_df = pd.DataFrame(smote_sweep_results)

    # --- Step 3: Bootstrap CIs ---
    print("\n" + "=" * 70)
    print("STEP 3: BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 70)

    preds = stage.predict(test_X, return_proba=True)
    is_conf = np.asarray(preds['is_confident'], dtype=bool)
    cascade_pred = preds['predicted_raw']
    cascade_conf = preds['proba'].max(axis=1)

    # Flat model for comparison
    from xgboost import XGBClassifier
    gpu_params = {}
    try:
        _t = XGBClassifier(tree_method='hist', device='cuda', n_estimators=1)
        _t.fit(np.random.rand(10, 2), np.random.randint(0, 2, 10))
        gpu_params = {'tree_method': 'hist', 'device': 'cuda'}
        del _t
    except Exception:
        pass

    flat_model = XGBClassifier(
        n_estimators=200, max_depth=6, random_state=RANDOM_SEED,
        eval_metric='logloss', n_jobs=-1, **gpu_params
    )
    flat_model.fit(train_X, train_y)
    flat_pred = flat_model.predict(test_X)

    bootstrap_results = run_bootstrap(
        test_y, cascade_pred, cascade_conf, is_conf, flat_pred, majority_class
    )

    # Print results
    for name, key in [('Cascade accuracy', 'cascade_accuracy_ci'),
                       ('Flat accuracy', 'flat_accuracy_ci'),
                       ('Cascade def. recall', 'cascade_recall_defective_ci'),
                       ('Flat def. recall', 'flat_recall_defective_ci')]:
        ci = bootstrap_results[key]
        print(f"  {name}: {ci['mean']:.3f} [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")

    mc = bootstrap_results['mcnemar_cascade_vs_flat']
    print(f"\n  McNemar cascade vs flat: stat={mc['statistic']:.2f}, "
          f"p={mc['p_value']:.4f}, significant={mc['significant_0.05']}")

    # --- Step 4: Conformal Prediction ---
    print("\n" + "=" * 70)
    print("STEP 4: CONFORMAL PREDICTION (MAPIE)")
    print("=" * 70)

    conformal_results = run_conformal(stage, train_X, train_y, test_X, test_y, feature_cols)

    # --- Step 5: LLM Rescue ---
    print("\n" + "=" * 70)
    print("STEP 5: LLM RESCUE FOR DEFERRED ITEMS")
    print("=" * 70)

    is_deferred = ~is_conf
    llm_results = None
    if args.skip_llm:
        print("  [SKIPPED by --skip-llm]")
    else:
        llm_results = run_llm_rescue(
            test_df, test_X, test_y, is_deferred, feature_cols,
            dry_run=args.llm_dry_run,
            max_items=args.llm_max_items,
        )

    # --- Step 6: Plots ---
    print("\n" + "=" * 70)
    print("STEP 6: GENERATING PLOTS")
    print("=" * 70)

    generate_plots(sweep_df, sweep_defective_df, bootstrap_results,
                   conformal_results, llm_results, smote_sweep_df)

    # --- Save all results ---
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'JM1 (PROMISE)',
        'n_train': len(train_y),
        'n_test': len(test_y),
        'defect_rate': float(defect_rate),
        'majority_baseline': float(majority_acc),
        'threshold_sweep': sweep_df.to_dict('records') if sweep_df is not None else None,
        'defective_threshold_sweep': sweep_defective_df.to_dict('records') if sweep_defective_df is not None else None,
        'smote_sweep': smote_sweep_df.to_dict('records') if smote_sweep_df is not None else None,
        'bootstrap': {k: v for k, v in bootstrap_results.items() if k != 'coverage_accuracy_curve_ci'},
        'conformal': conformal_results,
        'llm_rescue': llm_results,
    }

    results_path = OUTPUT_DIR / f'jm1_improved_{datetime.now():%Y%m%d_%H%M%S}.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {results_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Majority baseline:     {majority_acc:.1%}")
    print(f"  Flat XGBoost:          {accuracy_score(test_y, flat_pred):.1%}")
    ci = bootstrap_results['cascade_accuracy_ci']
    print(f"  Cascade (gated):       {ci['mean']:.1%} "
          f"[{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}] at {is_conf.mean():.1%} coverage")
    ci_r = bootstrap_results['cascade_recall_defective_ci']
    print(f"  Cascade def. recall:   {ci_r['mean']:.1%} "
          f"[{ci_r['ci_lower']:.1%}, {ci_r['ci_upper']:.1%}]")
    if conformal_results:
        for key in sorted(conformal_results.keys()):
            r = conformal_results[key]
            print(f"  Conformal {r['guarantee']}: "
                  f"coverage={r['empirical_coverage']:.1%}, "
                  f"single-class={r['single_class_frac']:.1%}")
    if llm_results and not llm_results.get('dry_run') and 'llm_accuracy' in llm_results:
        print(f"  LLM rescue accuracy:   {llm_results['llm_accuracy']:.1%} "
              f"on {llm_results['n_evaluated']} deferred items")
        print(f"  LLM def. recall:       {llm_results['llm_recall_defective']:.1%}")

    print(f"\nOutputs in: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
