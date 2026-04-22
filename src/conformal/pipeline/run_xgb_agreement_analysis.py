"""
Cross-model agreement analysis: DeBERTa vs XGBoost on Eclipse 30-class (no Other).

Key question: Does XGBoost agreement add information beyond DeBERTa's confidence?
If yes -> model-agreement-conditioned conformal prediction is a novel contribution.
If no -> agreement is redundant with confidence.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    print("XGBoost available")
except ImportError:
    print("XGBoost not available")
    exit(1)

# ─── Paths (use env vars on TamIA, fallback to local) ───
import os
DATA_DIR = Path(os.environ.get("ECLIPSE_DATA_DIR", "conformal_outputs/eclipse_no_other"))
DEBERTA_DIR = Path(os.environ.get("DEBERTA_DIR", "conformal_outputs/deberta_no_other"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "conformal_outputs/agreement_analysis"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Load data ───
print("=== Loading Data ===")
label_map = json.load(open(DATA_DIR / "label_mapping.json"))
inv_map = {v: k for k, v in label_map.items()}
num_classes = len(label_map)
print(f"Classes: {num_classes}")

train_df = pd.read_parquet(DATA_DIR / "train.parquet")
cal_df = pd.read_parquet(DATA_DIR / "cal.parquet")
test_df = pd.read_parquet(DATA_DIR / "test.parquet")
print(f"Train: {len(train_df):,} | Cal: {len(cal_df):,} | Test: {len(test_df):,}")

# ─── Load DeBERTa predictions ───
print("\n=== Loading DeBERTa Predictions ===")
deb_test_probs = np.load(DEBERTA_DIR / "test_probs.npy")
deb_test_labels = np.load(DEBERTA_DIR / "test_labels.npy")
deb_test_preds = np.load(DEBERTA_DIR / "test_preds.npy")
print(f"DeBERTa test shape: {deb_test_probs.shape}")
print(f"DeBERTa test accuracy: {accuracy_score(deb_test_labels, deb_test_preds):.4f}")

# Verify labels match
test_labels = test_df["label"].values
assert len(test_labels) == len(deb_test_labels), f"Label count mismatch: {len(test_labels)} vs {len(deb_test_labels)}"
assert np.array_equal(test_labels, deb_test_labels), "Label values don't match!"
print("Labels verified: match between parquet and DeBERTa npy")

# ─── Feature Engineering (matching existing Eclipse S2 pipeline) ───
print("\n=== Feature Engineering (full S2 pipeline) ===")

SEVERITY_MAP = {
    'blocker': 5, 'critical': 4, 'major': 3, 'normal': 2,
    'minor': 1, 'trivial': 0, 'enhancement': -1
}
PRIORITY_MAP = {'P1': 4, 'P2': 3, 'P3': 2, 'P4': 1, 'P5': 0}


def engineer_features(df):
    """Replicate eclipse_zenodo_loader.engineer_features() on our parquet data."""
    df = df.copy()

    # Text length features
    df['summary_length'] = df['text'].str.split(' \\[SEP\\] ').str[0].str.len()
    df['summary_word_count'] = df['text'].str.split(' \\[SEP\\] ').str[0].str.split().str.len()
    desc = df['text'].str.split(' \\[SEP\\] ').str[1].fillna('')
    df['desc_length'] = desc.str.len()
    df['desc_word_count'] = desc.str.split().str.len()
    df['has_description'] = (df['desc_length'] > 10).astype(int)

    # Severity features
    if 'severity' in df.columns:
        sev_lower = df['severity'].fillna('normal').str.lower()
        df['severity_numeric'] = sev_lower.map(SEVERITY_MAP).fillna(2).astype(int)
        df['is_enhancement'] = (sev_lower == 'enhancement').astype(int)
        df['is_high_severity'] = sev_lower.isin(['blocker', 'critical']).astype(int)
    else:
        df['severity_numeric'] = 2
        df['is_enhancement'] = 0
        df['is_high_severity'] = 0

    # Priority features
    if 'priority' in df.columns:
        df['priority_numeric'] = df['priority'].fillna('P3').map(PRIORITY_MAP).fillna(2).astype(int)
    else:
        df['priority_numeric'] = 2

    # Temporal features
    if 'creation_time' in df.columns:
        dt = pd.to_datetime(df['creation_time'], errors='coerce')
        df['open_hour'] = dt.dt.hour.fillna(12).astype(int)
        df['open_dayofweek'] = dt.dt.dayofweek.fillna(0).astype(int)
        df['open_month'] = dt.dt.month.fillna(1).astype(int)
    else:
        df['open_hour'] = 12
        df['open_dayofweek'] = 0
        df['open_month'] = 1

    return df


def encode_categoricals(train_df, cal_df, test_df, cat_cols):
    """Label-encode categoricals, fitting on train only."""
    from sklearn.preprocessing import LabelEncoder
    encoders = {}
    for col in cat_cols:
        if col not in train_df.columns:
            continue
        le = LabelEncoder()
        # Fit on train values + 'unknown'
        all_vals = list(train_df[col].fillna('unknown').unique()) + ['unknown']
        le.fit(all_vals)
        enc_col = f'{col}_enc'
        for df in [train_df, cal_df, test_df]:
            vals = df[col].fillna('unknown').copy()
            # Map unseen to 'unknown'
            vals = vals.where(vals.isin(le.classes_), other='unknown')
            df[enc_col] = le.transform(vals)
        encoders[col] = le
    return encoders


# Engineer features for all splits
for name, df in [("train", train_df), ("cal", cal_df), ("test", test_df)]:
    engineered = engineer_features(df)
    for col in engineered.columns:
        if col not in df.columns:
            df[col] = engineered[col]

# Creator bug count (from training data only)
if 'creator' in train_df.columns:
    creator_counts = train_df['creator'].value_counts().to_dict()
    for df in [train_df, cal_df, test_df]:
        df['creator_bug_count'] = df['creator'].map(creator_counts).fillna(1).astype(int)
    print(f"  creator_bug_count: mean={train_df['creator_bug_count'].mean():.0f}")

# Encode categoricals
cat_cols = ['severity', 'priority', 'platform', 'op_sys', 'product']
cat_cols = [c for c in cat_cols if c in train_df.columns]
encode_categoricals(train_df, cal_df, test_df, cat_cols)
print(f"  Encoded categoricals: {cat_cols}")

# Numeric feature list (matching S2 pipeline, excluding component_size)
numeric_features = [
    'summary_length', 'summary_word_count', 'desc_length', 'desc_word_count',
    'has_description', 'severity_numeric', 'priority_numeric',
    'is_enhancement', 'is_high_severity', 'open_hour', 'open_dayofweek', 'open_month',
]
if 'creator_bug_count' in train_df.columns:
    numeric_features.append('creator_bug_count')

cat_enc_features = [f'{c}_enc' for c in cat_cols if f'{c}_enc' in train_df.columns]
structured_features = numeric_features + cat_enc_features
print(f"  Structured features ({len(structured_features)}): {structured_features}")

# TF-IDF (matching S2: 500 features, bigrams, stop_words, min_df=2)
print("Building TF-IDF features (500, matching S2 config)...")
tfidf = TfidfVectorizer(
    max_features=500, ngram_range=(1, 2),
    stop_words='english', min_df=2, sublinear_tf=True
)
X_train_tfidf = tfidf.fit_transform(train_df["text"].values)
X_cal_tfidf = tfidf.transform(cal_df["text"].values)
X_test_tfidf = tfidf.transform(test_df["text"].values)
print(f"  TF-IDF shape: {X_train_tfidf.shape}")

# Scale structured features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_struct = scaler.fit_transform(train_df[structured_features].values.astype(float))
X_cal_struct = scaler.transform(cal_df[structured_features].values.astype(float))
X_test_struct = scaler.transform(test_df[structured_features].values.astype(float))

# Combine: structured + TF-IDF
X_train = hstack([csr_matrix(X_train_struct), X_train_tfidf])
X_cal = hstack([csr_matrix(X_cal_struct), X_cal_tfidf])
X_test = hstack([csr_matrix(X_test_struct), X_test_tfidf])

y_train = train_df["label"].values
y_cal = cal_df["label"].values
y_test = test_df["label"].values

print(f"\n=== Training XGBoost ===")
print(f"Features: {X_train.shape[1]} ({len(structured_features)} structured + {X_train_tfidf.shape[1]} TF-IDF)")
print(f"Samples: {X_train.shape[0]:,} train, {X_cal.shape[0]:,} cal, {X_test.shape[0]:,} test")

xgb = XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    early_stopping_rounds=30,
)
xgb.fit(X_train, y_train, eval_set=[(X_cal, y_cal)], verbose=50)

print(f"\nBest iteration: {xgb.best_iteration}")

xgb_test_preds = xgb.predict(X_test)
xgb_test_probs = xgb.predict_proba(X_test)
xgb_cal_preds = xgb.predict(X_cal)
xgb_cal_probs = xgb.predict_proba(X_cal)
xgb_test_acc = accuracy_score(y_test, xgb_test_preds)
xgb_cal_acc = accuracy_score(y_cal, xgb_cal_preds)
print(f"XGBoost cal accuracy: {xgb_cal_acc:.4f}")
print(f"XGBoost test accuracy: {xgb_test_acc:.4f}")
print(f"XGBoost test F1 (macro): {f1_score(y_test, xgb_test_preds, average='macro', zero_division=0):.4f}")
print(f"XGBoost test F1 (weighted): {f1_score(y_test, xgb_test_preds, average='weighted', zero_division=0):.4f}")

# Save XGBoost predictions
np.save(OUTPUT_DIR / "xgb_test_preds.npy", xgb_test_preds)
np.save(OUTPUT_DIR / "xgb_test_probs.npy", xgb_test_probs)

# ─── Agreement Analysis ───
print("\n" + "=" * 70)
print("CROSS-MODEL AGREEMENT ANALYSIS: DeBERTa vs XGBoost")
print("=" * 70)

agree_mask = deb_test_preds == xgb_test_preds
n_total = len(y_test)
n_agree = agree_mask.sum()
n_disagree = (~agree_mask).sum()

print(f"\n1. OVERALL AGREEMENT")
print(f"   Agreement rate: {n_agree}/{n_total} ({n_agree/n_total:.1%})")
print(f"   Disagreement rate: {n_disagree}/{n_total} ({n_disagree/n_total:.1%})")

# Accuracy when agree vs disagree
deb_correct = deb_test_preds == y_test
xgb_correct = xgb_test_preds == y_test

acc_agree = deb_correct[agree_mask].mean() if agree_mask.sum() > 0 else 0
acc_disagree = deb_correct[~agree_mask].mean() if (~agree_mask).sum() > 0 else 0
xgb_acc_agree = xgb_correct[agree_mask].mean() if agree_mask.sum() > 0 else 0
xgb_acc_disagree = xgb_correct[~agree_mask].mean() if (~agree_mask).sum() > 0 else 0

print(f"\n2. ACCURACY BY AGREEMENT STATUS")
print(f"   DeBERTa overall:     {deb_correct.mean():.4f}")
print(f"   XGBoost overall:     {xgb_correct.mean():.4f}")
print(f"   When AGREE:          DeBERTa={acc_agree:.4f}, XGBoost={xgb_acc_agree:.4f} (n={n_agree:,})")
print(f"   When DISAGREE:       DeBERTa={acc_disagree:.4f}, XGBoost={xgb_acc_disagree:.4f} (n={n_disagree:,})")
print(f"   Agreement accuracy gap: {acc_agree - acc_disagree:+.4f}")

# Either model correct when they agree vs disagree
either_correct_agree = (deb_correct | xgb_correct)[agree_mask].mean()
either_correct_disagree = (deb_correct | xgb_correct)[~agree_mask].mean()
print(f"   Either correct (agree):    {either_correct_agree:.4f}")
print(f"   Either correct (disagree): {either_correct_disagree:.4f}")

# ─── THE KEY TEST: Confidence-binned agreement analysis ───
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

    bin_agree = agree_mask[bin_mask]
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

# ─── Additional: XGBoost confidence bins ───
print(f"\n4. REVERSE TEST: Agreement value WITHIN XGBoost confidence bins")
print(f"   (Does XGBoost's own confidence correlate with agreement?)")
print()

xgb_confidence = xgb_test_probs.max(axis=1)
print(f"   {'Conf Bin':>12} | {'N':>6} | {'XGB Acc':>8} | {'Agree':>6} | {'DEB Acc(agree)':>14} | {'DEB Acc(disagree)':>17} | {'GAP':>7}")
print(f"   {'-'*12}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*14}-+-{'-'*17}-+-{'-'*7}")

for lo, hi in bins:
    bin_mask = (xgb_confidence >= lo) & (xgb_confidence < hi)
    n_bin = bin_mask.sum()
    if n_bin < 10:
        continue

    bin_agree = agree_mask[bin_mask]
    bin_xgb_correct = xgb_correct[bin_mask]
    bin_deb_correct_here = deb_correct[bin_mask]
    n_agree_bin = bin_agree.sum()
    n_disagree_bin = (~bin_agree).sum()

    xgb_acc = bin_xgb_correct.mean()

    acc_a = bin_deb_correct_here[bin_agree].mean() if n_agree_bin > 5 else float('nan')
    acc_d = bin_deb_correct_here[~bin_agree].mean() if n_disagree_bin > 5 else float('nan')
    gap = acc_a - acc_d if not (np.isnan(acc_a) or np.isnan(acc_d)) else float('nan')

    print(f"   [{lo:.2f},{hi:.2f}) | {n_bin:6,} | {xgb_acc:8.1%} | {n_agree_bin/n_bin:5.1%} | {acc_a:14.1%} | {acc_d:17.1%} | {gap:+7.1%}")

# ─── Confusion analysis: agreement patterns ───
print(f"\n5. AGREEMENT PATTERN ANALYSIS")
both_correct = (deb_correct & xgb_correct).sum()
only_deb = (deb_correct & ~xgb_correct).sum()
only_xgb = (~deb_correct & xgb_correct).sum()
both_wrong = (~deb_correct & ~xgb_correct).sum()

print(f"   Both correct:      {both_correct:6,} ({both_correct/n_total:5.1%})")
print(f"   Only DeBERTa:      {only_deb:6,} ({only_deb/n_total:5.1%})")
print(f"   Only XGBoost:      {only_xgb:6,} ({only_xgb/n_total:5.1%})")
print(f"   Both wrong:        {both_wrong:6,} ({both_wrong/n_total:5.1%})")
print(f"   Complementarity:   {only_xgb/(only_xgb+both_wrong):.1%} of DeBERTa errors are caught by XGBoost")

# ─── Oracle analysis ───
# If we could perfectly know when to trust each model
oracle_preds = np.where(deb_correct, deb_test_preds, xgb_test_preds)
oracle_acc = accuracy_score(y_test, oracle_preds)
print(f"\n6. ORACLE ANALYSIS")
print(f"   DeBERTa alone:     {deb_correct.mean():.4f}")
print(f"   XGBoost alone:     {xgb_correct.mean():.4f}")
print(f"   Oracle (best of):  {oracle_acc:.4f}")
print(f"   Oracle headroom:   +{oracle_acc - deb_correct.mean():.4f} over DeBERTa")

# ─── Per-class agreement ───
print(f"\n7. PER-CLASS AGREEMENT RATES (top 10 by support)")
class_stats = []
for c in range(num_classes):
    mask = y_test == c
    n_c = mask.sum()
    if n_c < 10:
        continue
    agree_c = agree_mask[mask].mean()
    deb_acc_c = deb_correct[mask].mean()
    xgb_acc_c = xgb_correct[mask].mean()
    class_stats.append({
        "class": inv_map[c],
        "n": n_c,
        "agree_rate": agree_c,
        "deb_acc": deb_acc_c,
        "xgb_acc": xgb_acc_c,
    })

class_stats.sort(key=lambda x: x["n"], reverse=True)
print(f"   {'Class':>25} | {'N':>5} | {'Agree':>6} | {'DEB':>5} | {'XGB':>5} | {'Gap':>6}")
print(f"   {'-'*25}-+-{'-'*5}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}")
for s in class_stats[:15]:
    gap = s["deb_acc"] - s["xgb_acc"]
    print(f"   {s['class']:>25} | {s['n']:5,} | {s['agree_rate']:5.1%} | {s['deb_acc']:5.1%} | {s['xgb_acc']:5.1%} | {gap:+5.1%}")

# ─── Verdict ───
print(f"\n{'='*70}")
print("VERDICT")
print(f"{'='*70}")

# Check the key bins
key_bins = [r for r in results_by_bin if r["gap"] is not None and r["bin"] in ["[0.30,0.50)", "[0.50,0.70)"]]
if key_bins:
    avg_gap = np.mean([abs(r["gap"]) for r in key_bins])
    if avg_gap >= 0.15:
        print(f"STRONG SIGNAL: Average gap in medium-confidence bins = {avg_gap:.1%}")
        print("XGBoost agreement adds substantial information beyond DeBERTa confidence.")
        print("-> Model-agreement-conditioned conformal prediction IS a novel contribution.")
    elif avg_gap >= 0.05:
        print(f"MODERATE SIGNAL: Average gap in medium-confidence bins = {avg_gap:.1%}")
        print("XGBoost agreement adds some information beyond DeBERTa confidence.")
        print("-> Worth reporting but not the main contribution.")
    else:
        print(f"WEAK SIGNAL: Average gap in medium-confidence bins = {avg_gap:.1%}")
        print("XGBoost agreement is mostly redundant with DeBERTa confidence.")
        print("-> Agreement adds little beyond confidence. Focus on conformal prediction alone.")

# Save results
results = {
    "deberta_accuracy": float(deb_correct.mean()),
    "xgboost_accuracy": float(xgb_test_acc),
    "agreement_rate": float(n_agree / n_total),
    "acc_when_agree": float(acc_agree),
    "acc_when_disagree": float(acc_disagree),
    "oracle_accuracy": float(oracle_acc),
    "bins": results_by_bin,
}
json.dump(results, open(OUTPUT_DIR / "agreement_analysis.json", "w"), indent=2)
print(f"\nResults saved to {OUTPUT_DIR}/agreement_analysis.json")
