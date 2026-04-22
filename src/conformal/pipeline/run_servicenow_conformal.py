"""
run_servicenow_conformal.py -- RAPS conformal prediction on ServiceNow ITSM data.

Evaluates conformal prediction (RAPS) on the ServiceNow UCI 498 ITSM dataset
using XGBoost as the base classifier. ServiceNow has NO text fields, making
this a structured-data-only conformal evaluation.

Classification target: incident category (Stage 1, top-10 + Other = 11 classes)

Split: 60% train / 20% calibration / 20% test (temporal)

Usage:
    python src/conformal/pipeline/run_servicenow_conformal.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from conformal.data.servicenow_loader import (
    load_servicenow_events,
    aggregate_to_incidents,
    engineer_features,
)

OUTPUT_DIR = PROJECT_ROOT / 'conformal_outputs' / 'servicenow_conformal'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────
# RAPS nonconformity scores (from run_conformal.py)
# ────────────────────────────────────────────────────────

def raps_nonconformity_scores(probs, labels, lam=0.01, k_reg=5):
    """
    RAPS (Regularized APS) nonconformity scores.
    Romano et al. 2020 + Angelopoulos et al. 2021.
    """
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
    """
    Construct RAPS prediction sets for test data.
    Returns boolean array (n_test, n_classes).
    """
    n, k = test_probs.shape
    sets = np.zeros((n, k), dtype=bool)
    for i in range(n):
        sorted_idx = np.argsort(-test_probs[i])
        cumsum = 0.0
        for rank, class_idx in enumerate(sorted_idx):
            cumsum += test_probs[i, class_idx]
            penalty = lam * max(0, rank + 1 - k_reg)
            cumsum_reg = cumsum + penalty
            sets[i, class_idx] = True
            if cumsum_reg >= quantile:
                break
    return sets


def evaluate_conformal(prediction_sets, test_labels, test_preds, alpha, label_map):
    """Compute all conformal prediction metrics."""
    n = len(test_labels)
    num_classes = prediction_sets.shape[1]
    inv_map = {v: k for k, v in label_map.items()}

    set_sizes = prediction_sets.sum(axis=1)

    # Coverage
    coverage = np.array([
        prediction_sets[i, test_labels[i]] for i in range(n)
    ]).mean()

    # Singleton metrics
    singleton_mask = set_sizes == 1
    singleton_rate = singleton_mask.mean()

    if singleton_mask.sum() > 0:
        singleton_preds = prediction_sets[singleton_mask].argmax(axis=1)
        singleton_acc = accuracy_score(test_labels[singleton_mask], singleton_preds)
        singleton_correct = np.array([
            prediction_sets[i, test_labels[i]]
            for i in range(n) if singleton_mask[i]
        ])
        wrong_singleton_rate = 1.0 - singleton_correct.mean()
    else:
        singleton_acc = 0.0
        wrong_singleton_rate = 0.0

    # Class-conditional coverage
    class_coverage = {}
    for c in range(num_classes):
        mask = test_labels == c
        if mask.sum() > 0:
            cov_c = np.array([
                prediction_sets[i, c] for i in range(n) if test_labels[i] == c
            ]).mean()
            class_name = inv_map.get(c, f"class_{c}")
            class_coverage[class_name] = {
                "coverage": float(cov_c),
                "count": int(mask.sum()),
                "mean_set_size": float(set_sizes[mask].mean()),
            }

    # Set size distribution
    size_dist = {}
    for s in range(1, min(num_classes + 2, 16)):
        size_dist[str(s)] = float((set_sizes == s).mean())
    size_dist["16+"] = float((set_sizes >= 16).mean())

    return {
        "alpha": alpha,
        "nominal_coverage": 1 - alpha,
        "empirical_coverage": float(coverage),
        "coverage_gap": float((1 - alpha) - coverage),
        "mean_set_size": float(set_sizes.mean()),
        "median_set_size": float(np.median(set_sizes)),
        "singleton_rate": float(singleton_rate),
        "singleton_accuracy": float(singleton_acc),
        "wrong_singleton_rate": float(wrong_singleton_rate),
        "set_size_std": float(set_sizes.std()),
        "size_distribution": size_dist,
        "class_conditional_coverage": class_coverage,
    }


def compute_augrc(test_probs, test_labels, test_preds):
    """
    AUGRC (Area Under Generalized Risk-Coverage Curve).
    Traub et al. NeurIPS 2024. Lower is better.
    """
    confidences = test_probs.max(axis=1)
    correct = (test_preds == test_labels).astype(float)
    sorted_idx = np.argsort(-confidences)
    sorted_correct = correct[sorted_idx]
    n = len(sorted_correct)
    coverages = np.arange(1, n + 1) / n
    cumulative_errors = np.cumsum(1 - sorted_correct)
    generalized_risks = cumulative_errors / n
    augrc = np.trapezoid(generalized_risks, coverages)
    return augrc, coverages, generalized_risks


def bootstrap_augrc(test_probs, test_labels, test_preds, n_bootstrap=10000, ci=0.95):
    """Bootstrap confidence interval for AUGRC."""
    n = len(test_labels)
    samples = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        aug, _, _ = compute_augrc(test_probs[idx], test_labels[idx], test_preds[idx])
        samples.append(aug)
    samples = np.array(samples)
    lo = np.percentile(samples, (1 - ci) / 2 * 100)
    hi = np.percentile(samples, (1 + ci) / 2 * 100)
    return float(np.mean(samples)), float(lo), float(hi)


# ────────────────────────────────────────────────────────
# Feature encoding helpers (mirror servicenow_config.py)
# ────────────────────────────────────────────────────────

def encode_categoricals(train_df, cal_df, test_df, cat_columns):
    """Fit LabelEncoder on train, apply to cal and test."""
    encoders = {}
    train_df = train_df.copy()
    cal_df = cal_df.copy()
    test_df = test_df.copy()
    for col in cat_columns:
        if col not in train_df.columns:
            continue
        le = LabelEncoder()
        train_vals = train_df[col].astype(str).fillna('unknown')
        le.fit(train_vals)
        encoders[col] = le
        train_df[col + '_enc'] = le.transform(train_vals)
        known = set(le.classes_)
        fallback = 'unknown' if 'unknown' in known else le.classes_[0]
        for df in [cal_df, test_df]:
            vals = df[col].astype(str).fillna('unknown')
            vals = vals.apply(lambda x: x if x in known else fallback)
            df[col + '_enc'] = le.transform(vals)
    return train_df, cal_df, test_df, encoders


# ────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SERVICENOW ITSM -- RAPS CONFORMAL PREDICTION")
    print("Target: S1 Category (11 classes, structured data only)")
    print("Split: 60% train / 20% calibration / 20% test (temporal)")
    print("=" * 70)

    # ─── Step 1: Load and prepare data ───────────────────────
    print("\n[Step 1] Loading ServiceNow data ...")
    events = load_servicenow_events()
    incidents = aggregate_to_incidents(events)
    incidents = engineer_features(incidents)

    # Filter to resolved incidents with valid category/assignment
    resolved = incidents[
        incidents['final_state'].isin(['Closed', 'Resolved'])
    ].copy()
    resolved = resolved[resolved['category'] != '?'].copy()
    resolved = resolved[resolved['assignment_group'] != '?'].copy()
    resolved = resolved.sort_values('opened_at').reset_index(drop=True)
    print(f"  Resolved incidents: {len(resolved)}")

    # 60/20/20 temporal split
    n = len(resolved)
    train_end = int(n * 0.60)
    cal_end = int(n * 0.80)
    train_df = resolved.iloc[:train_end].copy()
    cal_df = resolved.iloc[train_end:cal_end].copy()
    test_df = resolved.iloc[cal_end:].copy()

    print(f"  Train: {len(train_df)} ({len(train_df)/n:.0%})")
    print(f"  Cal:   {len(cal_df)} ({len(cal_df)/n:.0%})")
    print(f"  Test:  {len(test_df)} ({len(test_df)/n:.0%})")
    print(f"  Date range train: {train_df['opened_at'].min().date()} to "
          f"{train_df['opened_at'].max().date()}")
    print(f"  Date range cal:   {cal_df['opened_at'].min().date()} to "
          f"{cal_df['opened_at'].max().date()}")
    print(f"  Date range test:  {test_df['opened_at'].min().date()} to "
          f"{test_df['opened_at'].max().date()}")

    # Recompute train-derived statistics on training split only
    train_caller_counts = train_df['caller_id'].value_counts()
    train_df['caller_incident_count'] = train_df['caller_id'].map(train_caller_counts).fillna(0).astype(int)
    cal_df['caller_incident_count'] = cal_df['caller_id'].map(train_caller_counts).fillna(0).astype(int)
    test_df['caller_incident_count'] = test_df['caller_id'].map(train_caller_counts).fillna(0).astype(int)

    train_cat_size = train_df['category'].value_counts()
    train_df['category_size'] = train_df['category'].map(train_cat_size).fillna(0).astype(int)
    cal_df['category_size'] = cal_df['category'].map(train_cat_size).fillna(0).astype(int)
    test_df['category_size'] = test_df['category'].map(train_cat_size).fillna(0).astype(int)

    # ─── Step 2: Build S1 Category features ─────────────────
    print("\n[Step 2] Building Stage 1 (Category) features ...")

    # S1 target: top-10 categories + Other (mirrors servicenow_config.py)
    TOP_N_CATEGORIES = 10
    top_cats = train_df['category'].value_counts().head(TOP_N_CATEGORIES).index.tolist()
    all_cat_classes = sorted(top_cats + ['Other'])

    for df in [train_df, cal_df, test_df]:
        df['cat_target'] = df['category'].apply(
            lambda x: x if x in top_cats else 'Other'
        )

    cat_le = LabelEncoder()
    cat_le.fit(all_cat_classes)
    for df in [train_df, cal_df, test_df]:
        df['cat_code'] = cat_le.transform(df['cat_target'])

    n_classes = len(cat_le.classes_)
    label_map = {name: int(code) for code, name in enumerate(cat_le.classes_)}
    inv_label_map = {v: k for k, v in label_map.items()}
    print(f"  Classes ({n_classes}): {list(cat_le.classes_)}")

    # Distribution
    train_dist = train_df['cat_target'].value_counts()
    print("\n  Train category distribution:")
    for cat, cnt in train_dist.head(12).items():
        print(f"    {cat}: {cnt} ({cnt/len(train_df):.1%})")

    # Numeric features (exclude category_size: derived from category target)
    # Exclude subcategory (child of category -> leakage)
    NUMERIC_FEATURES = [
        'priority_num', 'impact_num', 'urgency_num',
        'caller_incident_count',
        'made_sla_num', 'is_phone', 'is_email',
        'open_hour', 'open_dayofweek', 'open_month',
        'is_weekend', 'is_business_hours',
    ]
    numeric_features = [f for f in NUMERIC_FEATURES if f in train_df.columns]

    # Categorical features for S1: exclude 'category' (is the target),
    # 'subcategory' (leakage: child of category)
    CAT_FEATURES_S1 = ['contact_type', 'location', 'u_symptom', 'cmdb_ci']
    cat_features_s1 = [c for c in CAT_FEATURES_S1 if c in train_df.columns]

    train_df, cal_df, test_df, encoders = encode_categoricals(
        train_df, cal_df, test_df, cat_features_s1
    )

    feature_cols = (
        numeric_features +
        [c + '_enc' for c in cat_features_s1 if c + '_enc' in train_df.columns]
    )
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    print(f"\n  Feature columns ({len(feature_cols)}): {feature_cols}")

    train_X = train_df[feature_cols].fillna(0).values
    train_y = train_df['cat_code'].values
    cal_X = cal_df[feature_cols].fillna(0).values
    cal_y = cal_df['cat_code'].values
    test_X = test_df[feature_cols].fillna(0).values
    test_y = test_df['cat_code'].values

    print(f"\n  Train X: {train_X.shape}, Cal X: {cal_X.shape}, Test X: {test_X.shape}")

    # Majority baseline on test
    majority_class = int(np.argmax(np.bincount(train_y)))
    majority_name = inv_label_map[majority_class]
    majority_acc = (test_y == majority_class).mean()
    print(f"\n  Majority baseline (test): {majority_acc:.4f} (class: {majority_name})")

    # ─── Step 3: Train XGBoost ───────────────────────────────
    print("\n[Step 3] Training XGBoost ...")
    try:
        import xgboost as xgb
    except ImportError:
        print("ERROR: xgboost not installed.")
        sys.exit(1)

    model = xgb.XGBClassifier(
        n_estimators=1000,
        early_stopping_rounds=30,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        train_X, train_y,
        eval_set=[(cal_X, cal_y)],
        verbose=100,
    )

    best_round = model.best_iteration
    print(f"  Best iteration: {best_round}")

    # Generate probabilities
    cal_probs = model.predict_proba(cal_X)
    cal_preds = model.predict(cal_X)
    test_probs = model.predict_proba(test_X)
    test_preds = model.predict(test_X)

    # Flatten to shape (n, n_classes) with full probability columns
    # XGBoost may skip missing classes; ensure shape matches n_classes
    if cal_probs.shape[1] != n_classes:
        print(f"  WARNING: XGBoost output {cal_probs.shape[1]} classes, "
              f"expected {n_classes}. Padding with zeros.")
        def pad_probs(probs, n_cls):
            padded = np.zeros((probs.shape[0], n_cls))
            padded[:, :probs.shape[1]] = probs
            return padded
        cal_probs = pad_probs(cal_probs, n_classes)
        test_probs = pad_probs(test_probs, n_classes)

    cal_acc = accuracy_score(cal_y, cal_preds)
    test_acc = accuracy_score(test_y, test_preds)
    print(f"\n  XGBoost calibration accuracy: {cal_acc:.4f}")
    print(f"  XGBoost test accuracy:        {test_acc:.4f}")
    print(f"  Majority baseline (test):     {majority_acc:.4f}")
    print(f"  Lift over majority:           {test_acc - majority_acc:+.4f}")

    # Save arrays
    np.save(OUTPUT_DIR / 'cal_probs.npy', cal_probs)
    np.save(OUTPUT_DIR / 'cal_labels.npy', cal_y)
    np.save(OUTPUT_DIR / 'cal_preds.npy', cal_preds)
    np.save(OUTPUT_DIR / 'test_probs.npy', test_probs)
    np.save(OUTPUT_DIR / 'test_labels.npy', test_y)
    np.save(OUTPUT_DIR / 'test_preds.npy', test_preds)
    print(f"\n  Arrays saved to {OUTPUT_DIR}/")

    # ─── Step 4: RAPS conformal prediction ──────────────────
    print("\n[Step 4] RAPS conformal prediction ...")
    ALPHA_LEVELS = [0.01, 0.05, 0.10, 0.20]
    LAM = 0.01
    K_REG = 5

    print(f"  Computing RAPS nonconformity scores on {len(cal_y)} calibration samples ...")
    cal_scores = raps_nonconformity_scores(cal_probs, cal_y, lam=LAM, k_reg=K_REG)
    print(f"  Score range: [{cal_scores.min():.4f}, {cal_scores.max():.4f}]")
    print(f"  Score mean:  {cal_scores.mean():.4f}")
    print(f"  Score median:{np.median(cal_scores):.4f}")

    all_results = {
        "dataset": "ServiceNow ITSM (UCI 498)",
        "model": "XGBoost",
        "conformal_method": "RAPS",
        "split": "60/20/20 temporal",
        "n_train": int(len(train_df)),
        "n_cal": int(len(cal_df)),
        "n_test": int(len(test_df)),
        "n_classes": int(n_classes),
        "classes": list(cat_le.classes_),
        "feature_cols": feature_cols,
        "xgb_test_accuracy": float(test_acc),
        "majority_baseline": float(majority_acc),
        "raps_lam": LAM,
        "raps_k_reg": K_REG,
        "conformal_results": {},
    }

    print(f"\n  {'alpha':<8} {'target':<10} {'actual':<10} {'gap':<10} "
          f"{'mean set':<12} {'singleton':<12} {'wrong_sing':<12}")
    print("  " + "-" * 78)

    for alpha in ALPHA_LEVELS:
        # Conformal quantile: (ceil((n_cal+1)*(1-alpha)) / n_cal)-th quantile
        n_cal = len(cal_scores)
        q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
        q_level = min(q_level, 1.0)
        quantile = np.quantile(cal_scores, q_level, method='higher')

        pred_sets = compute_prediction_sets(test_probs, quantile, lam=LAM, k_reg=K_REG)
        res = evaluate_conformal(pred_sets, test_y, test_preds, alpha, label_map)
        res['quantile'] = float(quantile)

        print(f"  {alpha:<8.2f} {res['nominal_coverage']:<10.3f} "
              f"{res['empirical_coverage']:<10.4f} {res['coverage_gap']:+.4f}    "
              f"{res['mean_set_size']:<12.2f} {res['singleton_rate']:<12.4f} "
              f"{res['wrong_singleton_rate']:.4f}")

        all_results["conformal_results"][str(alpha)] = res

        # Detailed output for each alpha
        print(f"\n  --- alpha={alpha} (target {1-alpha:.0%}) ---")
        print(f"    Quantile q_hat:     {quantile:.4f}")
        print(f"    Empirical coverage: {res['empirical_coverage']:.4f} "
              f"(target: {res['nominal_coverage']:.4f}, gap: {res['coverage_gap']:+.4f})")
        print(f"    Mean set size:      {res['mean_set_size']:.2f}")
        print(f"    Median set size:    {res['median_set_size']:.1f}")
        print(f"    Singleton rate:     {res['singleton_rate']:.4f} "
              f"({res['singleton_rate']*100:.1f}%)")
        print(f"    Singleton accuracy: {res['singleton_accuracy']:.4f}")
        print(f"    Wrong singleton:    {res['wrong_singleton_rate']:.4f} "
              f"({res['wrong_singleton_rate']*100:.1f}%)")

        # Set size distribution
        print(f"    Set size distribution:")
        for sz, frac in res['size_distribution'].items():
            if frac > 0.001:
                bar = '#' * int(frac * 40)
                print(f"      {sz:>3}: {frac:6.1%}  {bar}")

        # Worst 3 class-conditional coverage
        cc = res['class_conditional_coverage']
        worst_3 = sorted(cc.items(), key=lambda x: x[1]['coverage'])[:3]
        print(f"    Worst class-conditional coverage (3 worst):")
        for cls_name, info in worst_3:
            print(f"      {cls_name}: coverage={info['coverage']:.3f}, "
                  f"mean_set={info['mean_set_size']:.1f}, n={info['count']}")
        print()

    # ─── Step 5: AUGRC with bootstrap CIs ───────────────────
    print("\n[Step 5] AUGRC with bootstrap CIs (10,000 resamples) ...")
    augrc, coverages, risks = compute_augrc(test_probs, test_y, test_preds)
    augrc_mean, augrc_lo, augrc_hi = bootstrap_augrc(
        test_probs, test_y, test_preds, n_bootstrap=10000
    )

    print(f"  AUGRC:              {augrc:.6f}")
    print(f"  Bootstrap mean:     {augrc_mean:.6f}")
    print(f"  Bootstrap 95% CI:   [{augrc_lo:.6f}, {augrc_hi:.6f}]")

    # Random classifier AUGRC (for reference): integral of p/n from 0 to 1 ~= (1-acc)/2
    random_augrc_approx = (1 - majority_acc) / 2
    print(f"  Majority AUGRC approx: {random_augrc_approx:.6f}")

    np.savez(
        OUTPUT_DIR / 'augrc_curve.npz',
        coverages=coverages,
        risks=risks,
        augrc=augrc,
        augrc_ci_lo=augrc_lo,
        augrc_ci_hi=augrc_hi,
    )

    all_results['augrc'] = {
        "augrc": float(augrc),
        "augrc_bootstrap_mean": float(augrc_mean),
        "augrc_ci_lo": float(augrc_lo),
        "augrc_ci_hi": float(augrc_hi),
        "majority_augrc_approx": float(random_augrc_approx),
    }

    # ─── Step 6: Selective accuracy ─────────────────────────
    print("\n[Step 6] Selective accuracy at fixed coverage levels ...")
    confidences = test_probs.max(axis=1)
    correct = (test_preds == test_y)
    sorted_idx = np.argsort(-confidences)
    n_test_size = len(test_y)

    selective_results = {}
    print(f"  {'Coverage':<12} {'Accuracy':<12} {'n':<10} {'Lift over full':<16}")
    print("  " + "-" * 52)
    full_acc = correct.mean()

    for target_cov in [0.70, 0.80, 0.90, 0.95]:
        n_keep = int(n_test_size * target_cov)
        selected = sorted_idx[:n_keep]
        sel_acc = correct[selected].mean()
        lift = sel_acc - full_acc
        print(f"  {target_cov:<12.0%} {sel_acc:<12.4f} {n_keep:<10} {lift:+.4f}")
        selective_results[str(target_cov)] = {
            "coverage": target_cov,
            "accuracy": float(sel_acc),
            "n_samples": int(n_keep),
            "lift_over_full": float(lift),
        }

    all_results['selective_accuracy'] = selective_results
    all_results['full_test_accuracy'] = float(full_acc)

    # ─── Step 7: Save results ────────────────────────────────
    results_path = OUTPUT_DIR / 'conformal_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # ─── Final summary ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: SERVICENOW S1 CATEGORY -- RAPS CONFORMAL PREDICTION")
    print("=" * 70)
    print(f"  XGBoost base accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"  Majority baseline:     {majority_acc:.4f} ({majority_acc*100:.1f}%)")
    print(f"  Lift over majority:    {test_acc - majority_acc:+.4f}")
    print(f"  AUGRC: {augrc:.6f} (95% CI: [{augrc_lo:.6f}, {augrc_hi:.6f}])")
    print()
    print(f"  {'alpha':<8} {'coverage (target->actual)':<30} {'mean_set':<12} {'singleton%'}")
    print("  " + "-" * 70)
    for alpha in ALPHA_LEVELS:
        r = all_results["conformal_results"][str(alpha)]
        print(f"  {alpha:<8.2f} {r['nominal_coverage']:.2f} -> {r['empirical_coverage']:.4f}         "
              f"{r['mean_set_size']:<12.2f} {r['singleton_rate']*100:.1f}%")

    print()
    print(f"  Selective accuracy (confidence-sorted, top-N%):")
    for target_cov in [0.70, 0.80, 0.90, 0.95]:
        sr = selective_results[str(target_cov)]
        print(f"    {target_cov:.0%} coverage: {sr['accuracy']:.4f} accuracy "
              f"(lift {sr['lift_over_full']:+.4f})")

    print(f"\n  [OK] All artifacts saved to {OUTPUT_DIR}/")
    print(f"  [OK] conformal_results.json")
    print(f"  [OK] cal_probs.npy, cal_labels.npy, cal_preds.npy")
    print(f"  [OK] test_probs.npy, test_labels.npy, test_preds.npy")
    print(f"  [OK] augrc_curve.npz")


if __name__ == '__main__':
    main()
