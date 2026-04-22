"""
run_mozilla_core_maccp.py -- Full MACCP pipeline for Mozilla Core bug triage.

Steps:
1. Load DeBERTa and XGBoost predictions
2. Agreement analysis (key test: within-confidence-bin gaps)
3. RAPS conformal on DeBERTa alone
4. RAPS conformal on XGBoost alone
5. MACCP: agreement-conditioned conformal prediction
6. Save all results

Usage:
    python run_mozilla_core_maccp.py

Requires:
    - conformal_outputs/mozilla_core/deberta/{cal,test}_{probs,labels,preds}.npy
    - conformal_outputs/mozilla_core/xgb_{cal,test}_{probs,labels,preds}.npy
    - data/mozilla_core/label_mapping.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ---- Paths ----
DATA_DIR = Path("data/mozilla_core")
DEB_DIR = Path("conformal_outputs/mozilla_core/deberta")
XGB_DIR = Path("conformal_outputs/mozilla_core")
OUTPUT_DIR = Path("conformal_outputs/mozilla_core/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load data ----
print("=" * 70)
print("MOZILLA CORE MACCP ANALYSIS")
print("=" * 70)

label_map = json.load(open(DATA_DIR / "label_mapping.json"))
inv_map = {v: k for k, v in label_map.items()}
num_classes = len(label_map)
print(f"Classes: {num_classes}")

# DeBERTa predictions
deb_cal_probs = np.load(DEB_DIR / "cal_probs.npy")
deb_cal_labels = np.load(DEB_DIR / "cal_labels.npy")
deb_cal_preds = np.load(DEB_DIR / "cal_preds.npy")
deb_test_probs = np.load(DEB_DIR / "test_probs.npy")
deb_test_labels = np.load(DEB_DIR / "test_labels.npy")
deb_test_preds = np.load(DEB_DIR / "test_preds.npy")

# XGBoost predictions
xgb_cal_probs = np.load(XGB_DIR / "xgb_cal_probs.npy")
xgb_cal_labels = np.load(XGB_DIR / "xgb_cal_labels.npy")
xgb_cal_preds = np.load(XGB_DIR / "xgb_cal_preds.npy")
xgb_test_probs = np.load(XGB_DIR / "xgb_test_probs.npy")
xgb_test_labels = np.load(XGB_DIR / "xgb_test_labels.npy")
xgb_test_preds = np.load(XGB_DIR / "xgb_test_preds.npy")

# Verify labels match
assert np.array_equal(deb_cal_labels, xgb_cal_labels), "Cal labels mismatch!"
assert np.array_equal(deb_test_labels, xgb_test_labels), "Test labels mismatch!"

y_cal = deb_cal_labels
y_test = deb_test_labels

print(f"Cal: {len(y_cal):,} | Test: {len(y_test):,}")
print(f"DeBERTa probs shape: {deb_test_probs.shape}")
print(f"XGBoost probs shape: {xgb_test_probs.shape}")

# ================================================================
# SECTION 1: BASE MODEL PERFORMANCE
# ================================================================
print("\n" + "=" * 70)
print("SECTION 1: BASE MODEL PERFORMANCE")
print("=" * 70)

deb_test_acc = accuracy_score(y_test, deb_test_preds)
xgb_test_acc = accuracy_score(y_test, xgb_test_preds)
deb_cal_acc = accuracy_score(y_cal, deb_cal_preds)
xgb_cal_acc = accuracy_score(y_cal, xgb_cal_preds)

majority_class = np.bincount(y_cal).argmax()
majority_acc = (y_test == majority_class).mean()

print(f"DeBERTa test accuracy:  {deb_test_acc:.4f}")
print(f"XGBoost test accuracy:  {xgb_test_acc:.4f}")
print(f"DeBERTa cal accuracy:   {deb_cal_acc:.4f}")
print(f"XGBoost cal accuracy:   {xgb_cal_acc:.4f}")
print(f"Majority baseline:      {majority_acc:.4f} (class {majority_class}={inv_map[majority_class]})")

# ================================================================
# SECTION 2: AGREEMENT ANALYSIS
# ================================================================
print("\n" + "=" * 70)
print("SECTION 2: CROSS-MODEL AGREEMENT ANALYSIS (DeBERTa vs XGBoost)")
print("=" * 70)

agree_mask_test = deb_test_preds == xgb_test_preds
agree_mask_cal = deb_cal_preds == xgb_cal_preds
n_total = len(y_test)
n_agree = agree_mask_test.sum()
n_disagree = (~agree_mask_test).sum()

print(f"\n1. OVERALL AGREEMENT")
print(f"   Test agreement rate:  {n_agree}/{n_total} ({n_agree/n_total:.1%})")
print(f"   Cal agreement rate:   {agree_mask_cal.sum()}/{len(y_cal)} ({agree_mask_cal.mean():.1%})")

# Accuracy when agree vs disagree
deb_correct = deb_test_preds == y_test
xgb_correct = xgb_test_preds == y_test

acc_agree = deb_correct[agree_mask_test].mean() if agree_mask_test.sum() > 0 else 0
acc_disagree = deb_correct[~agree_mask_test].mean() if (~agree_mask_test).sum() > 0 else 0
xgb_acc_agree = xgb_correct[agree_mask_test].mean() if agree_mask_test.sum() > 0 else 0
xgb_acc_disagree = xgb_correct[~agree_mask_test].mean() if (~agree_mask_test).sum() > 0 else 0

print(f"\n2. ACCURACY BY AGREEMENT STATUS")
print(f"   DeBERTa overall:     {deb_correct.mean():.4f}")
print(f"   XGBoost overall:     {xgb_correct.mean():.4f}")
print(f"   When AGREE:          DeBERTa={acc_agree:.4f}, XGBoost={xgb_acc_agree:.4f} (n={n_agree:,})")
print(f"   When DISAGREE:       DeBERTa={acc_disagree:.4f}, XGBoost={xgb_acc_disagree:.4f} (n={n_disagree:,})")
print(f"   Agreement acc gap:   {acc_agree - acc_disagree:+.4f}")

# ---- THE KEY TEST: Confidence-binned agreement analysis ----
print(f"\n3. THE KEY TEST: Agreement value WITHIN DeBERTa confidence bins")
print(f"   (Does XGBoost agreement add info beyond DeBERTa's max softmax prob?)")
print()

deb_confidence = deb_test_probs.max(axis=1)
bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]

print(f"   {'Conf Bin':>12} | {'N':>6} | {'DeBERTa':>8} | {'Agree':>6} | {'Acc(agree)':>10} | {'Acc(disagree)':>13} | {'GAP':>7} | {'Signal?':>8}")
print(f"   {'-'*12}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*10}-+-{'-'*13}-+-{'-'*7}-+-{'-'*8}")

results_by_bin = []
for lo, hi in bins:
    bin_mask = (deb_confidence >= lo) & (deb_confidence < hi)
    n_bin = bin_mask.sum()
    if n_bin < 10:
        continue

    bin_agree = agree_mask_test[bin_mask]
    bin_deb_correct = deb_correct[bin_mask]
    bin_deb_acc = bin_deb_correct.mean()
    n_agree_bin = bin_agree.sum()
    n_disagree_bin = (~bin_agree).sum()

    if n_agree_bin > 5:
        acc_a = bin_deb_correct[bin_agree].mean()
    else:
        acc_a = float('nan')

    if n_disagree_bin > 5:
        acc_d = bin_deb_correct[~bin_agree].mean()
    else:
        acc_d = float('nan')

    gap = acc_a - acc_d if not (np.isnan(acc_a) or np.isnan(acc_d)) else float('nan')
    signal = ""
    if not np.isnan(gap):
        if abs(gap) >= 0.15:
            signal = "STRONG"
        elif abs(gap) >= 0.05:
            signal = "moderate"
        else:
            signal = "weak"

    print(f"   [{lo:.2f},{hi:.2f}) | {n_bin:6,} | {bin_deb_acc:8.1%} | {n_agree_bin/n_bin:5.1%} | {acc_a:10.1%} | {acc_d:13.1%} | {gap:+7.1%} | {signal:>8}")

    results_by_bin.append({
        "bin": f"[{lo:.2f},{hi:.2f})",
        "n": int(n_bin),
        "deberta_acc": float(bin_deb_acc),
        "agree_rate": float(n_agree_bin / n_bin),
        "acc_when_agree": float(acc_a) if not np.isnan(acc_a) else None,
        "acc_when_disagree": float(acc_d) if not np.isnan(acc_d) else None,
        "gap": float(gap) if not np.isnan(gap) else None,
    })

# ---- Agreement pattern analysis ----
print(f"\n4. AGREEMENT PATTERN ANALYSIS")
both_correct = (deb_correct & xgb_correct).sum()
only_deb = (deb_correct & ~xgb_correct).sum()
only_xgb = (~deb_correct & xgb_correct).sum()
both_wrong = (~deb_correct & ~xgb_correct).sum()

print(f"   Both correct:      {both_correct:6,} ({both_correct/n_total:5.1%})")
print(f"   Only DeBERTa:      {only_deb:6,} ({only_deb/n_total:5.1%})")
print(f"   Only XGBoost:      {only_xgb:6,} ({only_xgb/n_total:5.1%})")
print(f"   Both wrong:        {both_wrong:6,} ({both_wrong/n_total:5.1%})")
print(f"   Complementarity:   {only_xgb/(only_xgb+both_wrong):.1%} of DeBERTa errors caught by XGBoost")

# Oracle
oracle_preds = np.where(deb_correct, deb_test_preds, xgb_test_preds)
oracle_acc = accuracy_score(y_test, oracle_preds)
print(f"\n5. ORACLE ANALYSIS")
print(f"   DeBERTa alone:     {deb_correct.mean():.4f}")
print(f"   XGBoost alone:     {xgb_correct.mean():.4f}")
print(f"   Oracle (best of):  {oracle_acc:.4f}")
print(f"   Oracle headroom:   +{oracle_acc - deb_correct.mean():.4f} over DeBERTa")

# ================================================================
# SECTION 3: RAPS CONFORMAL PREDICTION
# ================================================================
print("\n" + "=" * 70)
print("SECTION 3: RAPS CONFORMAL PREDICTION")
print("=" * 70)

RAPS_LAM = 0.01
RAPS_K_REG = 5
ALPHA_LEVELS = [0.01, 0.05, 0.10, 0.20]


def raps_nonconformity_scores(probs, labels, lam=0.01, k_reg=5):
    """RAPS nonconformity scores (Romano et al. 2020 + Angelopoulos et al. 2021)."""
    n, k = probs.shape
    scores = np.zeros(n)
    for i in range(n):
        sorted_idx = np.argsort(-probs[i])
        cumsum = 0.0
        for rank, class_idx in enumerate(sorted_idx):
            cumsum += probs[i, class_idx]
            penalty = lam * max(0, rank + 1 - k_reg)
            if class_idx == labels[i]:
                rand_u = np.random.uniform(0, probs[i, class_idx] + penalty)
                scores[i] = cumsum + penalty - rand_u
                break
    return scores


def compute_prediction_sets(test_probs, quantile, lam=0.01, k_reg=5):
    """Construct RAPS prediction sets."""
    n, k = test_probs.shape
    sets_bool = np.zeros((n, k), dtype=bool)
    for i in range(n):
        sorted_idx = np.argsort(-test_probs[i])
        cumsum = 0.0
        for rank, class_idx in enumerate(sorted_idx):
            cumsum += test_probs[i, class_idx]
            penalty = lam * max(0, rank + 1 - k_reg)
            cumsum_reg = cumsum + penalty
            sets_bool[i, class_idx] = True
            if cumsum_reg >= quantile:
                break
    return sets_bool


def evaluate_conformal_sets(pred_sets, labels, alpha):
    """Evaluate conformal prediction sets."""
    n = len(labels)
    set_sizes = pred_sets.sum(axis=1)
    coverage = np.array([pred_sets[i, labels[i]] for i in range(n)]).mean()
    singleton_mask = set_sizes == 1
    singleton_rate = singleton_mask.mean()
    if singleton_mask.sum() > 0:
        singleton_preds = pred_sets[singleton_mask].argmax(axis=1)
        singleton_acc = accuracy_score(labels[singleton_mask], singleton_preds)
    else:
        singleton_acc = 0.0
    return {
        "alpha": alpha,
        "nominal_coverage": 1 - alpha,
        "empirical_coverage": float(coverage),
        "mean_set_size": float(set_sizes.mean()),
        "median_set_size": float(np.median(set_sizes)),
        "singleton_rate": float(singleton_rate),
        "singleton_accuracy": float(singleton_acc),
        "n_singletons": int(singleton_mask.sum()),
    }


def run_raps(cal_probs, cal_labels, test_probs, test_labels, model_name, alpha_levels):
    """Run full RAPS conformal prediction for a model."""
    print(f"\n--- {model_name} RAPS ---")
    cal_scores = raps_nonconformity_scores(cal_probs, cal_labels, lam=RAPS_LAM, k_reg=RAPS_K_REG)
    results = {}
    for alpha in alpha_levels:
        n_cal = len(cal_scores)
        q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
        q_level = min(q_level, 1.0)
        quantile = np.quantile(cal_scores, q_level, method="higher")
        pred_sets = compute_prediction_sets(test_probs, quantile, lam=RAPS_LAM, k_reg=RAPS_K_REG)
        r = evaluate_conformal_sets(pred_sets, test_labels, alpha)
        print(f"  alpha={alpha:.2f}: coverage={r['empirical_coverage']:.4f} (target {1-alpha:.2f}), "
              f"mean_size={r['mean_set_size']:.2f}, singletons={r['singleton_rate']:.1%} "
              f"(acc={r['singleton_accuracy']:.1%}, n={r['n_singletons']})")
        results[str(alpha)] = r
    return results, cal_scores


# Run RAPS for both models
deb_raps_results, deb_cal_scores = run_raps(
    deb_cal_probs, y_cal, deb_test_probs, y_test, "DeBERTa", ALPHA_LEVELS
)
xgb_raps_results, xgb_cal_scores = run_raps(
    xgb_cal_probs, y_cal, xgb_test_probs, y_test, "XGBoost", ALPHA_LEVELS
)

# ================================================================
# SECTION 4: MACCP (Model-Agreement-Conditioned Conformal Prediction)
# ================================================================
print("\n" + "=" * 70)
print("SECTION 4: MACCP (Model-Agreement-Conditioned Conformal Prediction)")
print("=" * 70)

# Check if MACCP is viable
key_bins = [r for r in results_by_bin if r["gap"] is not None and r["bin"] in ["[0.30,0.50)", "[0.50,0.70)"]]
avg_gap = np.mean([abs(r["gap"]) for r in key_bins]) if key_bins else 0
agreement_rate = agree_mask_test.mean()

print(f"\nViability check:")
print(f"  Average gap in medium-confidence bins: {avg_gap:.1%}")
print(f"  Agreement rate: {agreement_rate:.1%}")

maccp_viable = avg_gap >= 0.05 and agreement_rate > 0.15
print(f"  MACCP viable: {maccp_viable} (need gap>=5% and agreement>15%)")

maccp_results = {}
if maccp_viable:
    print("\n--- Running MACCP ---")

    # Split calibration data by agreement
    deb_cal_scores_agree = deb_cal_scores[agree_mask_cal]
    deb_cal_scores_disagree = deb_cal_scores[~agree_mask_cal]
    print(f"  Cal agree: {len(deb_cal_scores_agree):,} | Cal disagree: {len(deb_cal_scores_disagree):,}")

    for alpha in ALPHA_LEVELS:
        # Separate quantiles for agree/disagree groups
        n_agree_cal = len(deb_cal_scores_agree)
        n_disagree_cal = len(deb_cal_scores_disagree)

        if n_agree_cal < 20 or n_disagree_cal < 20:
            print(f"  alpha={alpha:.2f}: SKIP (insufficient calibration data)")
            continue

        q_agree = np.ceil((n_agree_cal + 1) * (1 - alpha)) / n_agree_cal
        q_agree = min(q_agree, 1.0)
        quantile_agree = np.quantile(deb_cal_scores_agree, q_agree, method="higher")

        q_disagree = np.ceil((n_disagree_cal + 1) * (1 - alpha)) / n_disagree_cal
        q_disagree = min(q_disagree, 1.0)
        quantile_disagree = np.quantile(deb_cal_scores_disagree, q_disagree, method="higher")

        print(f"  alpha={alpha:.2f}: q_agree={quantile_agree:.4f}, q_disagree={quantile_disagree:.4f} "
              f"(gap={quantile_disagree - quantile_agree:+.4f})")

        # Build prediction sets with group-specific thresholds
        pred_sets_maccp = np.zeros((len(y_test), num_classes), dtype=bool)

        # Agree group: tighter threshold (more confident -> smaller sets)
        agree_test_idx = np.where(agree_mask_test)[0]
        disagree_test_idx = np.where(~agree_mask_test)[0]

        sets_agree = compute_prediction_sets(
            deb_test_probs[agree_test_idx], quantile_agree, lam=RAPS_LAM, k_reg=RAPS_K_REG
        )
        sets_disagree = compute_prediction_sets(
            deb_test_probs[disagree_test_idx], quantile_disagree, lam=RAPS_LAM, k_reg=RAPS_K_REG
        )

        pred_sets_maccp[agree_test_idx] = sets_agree
        pred_sets_maccp[disagree_test_idx] = sets_disagree

        # Evaluate overall
        r_overall = evaluate_conformal_sets(pred_sets_maccp, y_test, alpha)

        # Evaluate per group
        r_agree = evaluate_conformal_sets(sets_agree, y_test[agree_test_idx], alpha)
        r_disagree = evaluate_conformal_sets(sets_disagree, y_test[disagree_test_idx], alpha)

        # Compare with standard RAPS
        std_r = deb_raps_results[str(alpha)]

        print(f"    MACCP overall: cov={r_overall['empirical_coverage']:.4f}, "
              f"size={r_overall['mean_set_size']:.2f}, "
              f"singletons={r_overall['singleton_rate']:.1%} (acc={r_overall['singleton_accuracy']:.1%})")
        print(f"    MACCP agree:   cov={r_agree['empirical_coverage']:.4f}, "
              f"size={r_agree['mean_set_size']:.2f}, "
              f"singletons={r_agree['singleton_rate']:.1%} (acc={r_agree['singleton_accuracy']:.1%})")
        print(f"    MACCP disagree: cov={r_disagree['empirical_coverage']:.4f}, "
              f"size={r_disagree['mean_set_size']:.2f}, "
              f"singletons={r_disagree['singleton_rate']:.1%} (acc={r_disagree['singleton_accuracy']:.1%})")
        print(f"    Standard RAPS: cov={std_r['empirical_coverage']:.4f}, "
              f"size={std_r['mean_set_size']:.2f}, "
              f"singletons={std_r['singleton_rate']:.1%}")

        # Key metric: singleton improvement
        singleton_diff = r_overall['singleton_rate'] - std_r['singleton_rate']
        singleton_acc_agree_vs_disagree = r_agree['singleton_accuracy'] - r_disagree['singleton_accuracy']

        print(f"    Singleton rate change: {singleton_diff:+.1%} (MACCP vs standard)")
        print(f"    Agree singleton acc vs disagree: {singleton_acc_agree_vs_disagree:+.1%}")

        maccp_results[str(alpha)] = {
            "overall": r_overall,
            "agree": r_agree,
            "disagree": r_disagree,
            "standard_raps": std_r,
            "quantile_agree": float(quantile_agree),
            "quantile_disagree": float(quantile_disagree),
            "singleton_rate_change": float(singleton_diff),
        }
else:
    print("\n  MACCP not viable -- running anyway for completeness.")
    # Run MACCP even if gaps are small, for reporting
    deb_cal_scores_agree = deb_cal_scores[agree_mask_cal]
    deb_cal_scores_disagree = deb_cal_scores[~agree_mask_cal]

    if len(deb_cal_scores_agree) >= 20 and len(deb_cal_scores_disagree) >= 20:
        for alpha in ALPHA_LEVELS:
            n_agree_cal = len(deb_cal_scores_agree)
            n_disagree_cal = len(deb_cal_scores_disagree)

            q_agree = np.ceil((n_agree_cal + 1) * (1 - alpha)) / n_agree_cal
            q_agree = min(q_agree, 1.0)
            quantile_agree = np.quantile(deb_cal_scores_agree, q_agree, method="higher")

            q_disagree = np.ceil((n_disagree_cal + 1) * (1 - alpha)) / n_disagree_cal
            q_disagree = min(q_disagree, 1.0)
            quantile_disagree = np.quantile(deb_cal_scores_disagree, q_disagree, method="higher")

            agree_test_idx = np.where(agree_mask_test)[0]
            disagree_test_idx = np.where(~agree_mask_test)[0]

            pred_sets_maccp = np.zeros((len(y_test), num_classes), dtype=bool)
            sets_agree = compute_prediction_sets(
                deb_test_probs[agree_test_idx], quantile_agree, lam=RAPS_LAM, k_reg=RAPS_K_REG
            )
            sets_disagree = compute_prediction_sets(
                deb_test_probs[disagree_test_idx], quantile_disagree, lam=RAPS_LAM, k_reg=RAPS_K_REG
            )
            pred_sets_maccp[agree_test_idx] = sets_agree
            pred_sets_maccp[disagree_test_idx] = sets_disagree

            r_overall = evaluate_conformal_sets(pred_sets_maccp, y_test, alpha)
            r_agree = evaluate_conformal_sets(sets_agree, y_test[agree_test_idx], alpha)
            r_disagree = evaluate_conformal_sets(sets_disagree, y_test[disagree_test_idx], alpha)
            std_r = deb_raps_results[str(alpha)]

            singleton_diff = r_overall['singleton_rate'] - std_r['singleton_rate']
            print(f"  alpha={alpha:.2f}: MACCP singletons={r_overall['singleton_rate']:.1%}, "
                  f"std={std_r['singleton_rate']:.1%}, change={singleton_diff:+.1%}")

            maccp_results[str(alpha)] = {
                "overall": r_overall,
                "agree": r_agree,
                "disagree": r_disagree,
                "standard_raps": std_r,
                "quantile_agree": float(quantile_agree),
                "quantile_disagree": float(quantile_disagree),
                "singleton_rate_change": float(singleton_diff),
            }

# ================================================================
# SECTION 5: AUGRC (Selective Classification Quality)
# ================================================================
print("\n" + "=" * 70)
print("SECTION 5: AUGRC (Area Under Generalized Risk-Coverage Curve)")
print("=" * 70)


def compute_augrc(probs, labels, preds):
    """Compute AUGRC (Traub et al. NeurIPS 2024)."""
    confidences = probs.max(axis=1)
    correct = (preds == labels).astype(float)
    sorted_idx = np.argsort(-confidences)
    sorted_correct = correct[sorted_idx]
    n = len(sorted_correct)
    coverages = np.arange(1, n + 1) / n
    cumulative_errors = np.cumsum(1 - sorted_correct)
    generalized_risks = cumulative_errors / n
    augrc = np.trapz(generalized_risks, coverages)
    return augrc


def bootstrap_augrc(probs, labels, preds, n_bootstrap=10000):
    """Bootstrap CI for AUGRC."""
    n = len(labels)
    samples = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        samples.append(compute_augrc(probs[idx], labels[idx], preds[idx]))
    samples = np.array(samples)
    return float(np.mean(samples)), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


deb_augrc = compute_augrc(deb_test_probs, y_test, deb_test_preds)
xgb_augrc = compute_augrc(xgb_test_probs, y_test, xgb_test_preds)
print(f"DeBERTa AUGRC: {deb_augrc:.4f}")
print(f"XGBoost AUGRC: {xgb_augrc:.4f}")

# Bootstrap CIs
deb_augrc_mean, deb_augrc_lo, deb_augrc_hi = bootstrap_augrc(deb_test_probs, y_test, deb_test_preds)
xgb_augrc_mean, xgb_augrc_lo, xgb_augrc_hi = bootstrap_augrc(xgb_test_probs, y_test, xgb_test_preds)
print(f"DeBERTa AUGRC 95% CI: [{deb_augrc_lo:.4f}, {deb_augrc_hi:.4f}]")
print(f"XGBoost AUGRC 95% CI: [{xgb_augrc_lo:.4f}, {xgb_augrc_hi:.4f}]")

# ================================================================
# SECTION 6: PER-CLASS AGREEMENT RATES
# ================================================================
print("\n" + "=" * 70)
print("SECTION 6: PER-CLASS AGREEMENT RATES (sorted by support)")
print("=" * 70)

print(f"   {'Class':>30} | {'N':>5} | {'Agree':>6} | {'DEB':>5} | {'XGB':>5}")
print(f"   {'-'*30}-+-{'-'*5}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}")

per_class_stats = []
for c in range(num_classes):
    mask = y_test == c
    n_c = mask.sum()
    if n_c < 5:
        continue
    agree_c = agree_mask_test[mask].mean()
    deb_acc_c = deb_correct[mask].mean()
    xgb_acc_c = xgb_correct[mask].mean()
    per_class_stats.append({
        "class": inv_map[c], "n": int(n_c),
        "agree_rate": float(agree_c),
        "deb_acc": float(deb_acc_c), "xgb_acc": float(xgb_acc_c),
    })

per_class_stats.sort(key=lambda x: x["n"], reverse=True)
for s in per_class_stats:
    print(f"   {s['class']:>30} | {s['n']:5,} | {s['agree_rate']:5.1%} | {s['deb_acc']:5.1%} | {s['xgb_acc']:5.1%}")

# ================================================================
# SECTION 7: SAVE ALL RESULTS
# ================================================================
print("\n" + "=" * 70)
print("SECTION 7: SAVING RESULTS")
print("=" * 70)

all_results = {
    "dataset": "mozilla_core",
    "num_classes": num_classes,
    "test_size": len(y_test),
    "cal_size": len(y_cal),
    "base_models": {
        "deberta_test_acc": float(deb_test_acc),
        "xgboost_test_acc": float(xgb_test_acc),
        "majority_baseline": float(majority_acc),
    },
    "agreement": {
        "agreement_rate": float(agreement_rate),
        "acc_when_agree": float(acc_agree),
        "acc_when_disagree": float(acc_disagree),
        "gap": float(acc_agree - acc_disagree),
        "oracle_accuracy": float(oracle_acc),
        "both_correct_rate": float(both_correct / n_total),
        "complementarity": float(only_xgb / (only_xgb + both_wrong)) if (only_xgb + both_wrong) > 0 else 0,
        "bins": results_by_bin,
    },
    "conformal": {
        "deberta_raps": deb_raps_results,
        "xgboost_raps": xgb_raps_results,
    },
    "maccp": maccp_results,
    "maccp_viable": maccp_viable,
    "augrc": {
        "deberta": {"augrc": deb_augrc, "ci_lo": deb_augrc_lo, "ci_hi": deb_augrc_hi},
        "xgboost": {"augrc": xgb_augrc, "ci_lo": xgb_augrc_lo, "ci_hi": xgb_augrc_hi},
    },
    "per_class": per_class_stats,
}

results_path = OUTPUT_DIR / "maccp_results.json"
json.dump(all_results, open(results_path, "w"), indent=2)
print(f"[OK] Saved all results to {results_path}")

# Print summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"DeBERTa test acc:      {deb_test_acc:.4f}")
print(f"XGBoost test acc:      {xgb_test_acc:.4f}")
print(f"Agreement rate:        {agreement_rate:.1%}")
print(f"Acc gap (agree-disagree): {acc_agree - acc_disagree:+.4f}")
if key_bins:
    print(f"Avg gap in medium bins: {avg_gap:.1%}")
if "0.1" in maccp_results:
    m = maccp_results["0.1"]
    print(f"MACCP alpha=0.10:      singletons={m['overall']['singleton_rate']:.1%} "
          f"(std={m['standard_raps']['singleton_rate']:.1%}, "
          f"change={m['singleton_rate_change']:+.1%})")
print(f"DeBERTa AUGRC:         {deb_augrc:.4f} [{deb_augrc_lo:.4f}, {deb_augrc_hi:.4f}]")
print(f"XGBoost AUGRC:         {xgb_augrc:.4f} [{xgb_augrc_lo:.4f}, {xgb_augrc_hi:.4f}]")
