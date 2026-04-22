"""
run_maccp.py -- Model-Agreement-Conditioned Conformal Prediction (MACCP)

Novel contribution for QRS 2026: condition conformal prediction thresholds
on whether DeBERTa and XGBoost agree, producing tighter prediction sets
when models agree and wider sets (better coverage) when they disagree.

Steps:
  A) Retrain XGBoost (same feature engineering) to get cal predictions
  B) MACCP: split cal into agree/disagree, compute separate RAPS thresholds
  C) Baselines: standard RAPS on DeBERTa alone, XGBoost alone
  D) Report: coverage, set size, singleton rate/accuracy, AUGRC
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except ImportError:
    print("ERROR: XGBoost not available")
    sys.exit(1)

np.random.seed(42)

# ============================================================
# Paths
# ============================================================
BASE = Path(os.environ.get("PROJECT_ROOT", "."))
DATA_DIR = BASE / "conformal_outputs" / "eclipse_no_other"
DEBERTA_DIR = BASE / "conformal_outputs" / "deberta_no_other"
AGREEMENT_DIR = BASE / "conformal_outputs" / "agreement_analysis"
OUTPUT_DIR = BASE / "conformal_outputs" / "maccp_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# RAPS hyperparameters
LAM = 0.01
K_REG = 5
ALPHA_LEVELS = [0.01, 0.05, 0.10, 0.20]
N_BOOTSTRAP = 1000


# ============================================================
# RAPS functions (from run_conformal.py)
# ============================================================
def raps_nonconformity_scores(probs, labels, lam=LAM, k_reg=K_REG):
    """Compute RAPS nonconformity scores."""
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


def compute_prediction_sets(test_probs, quantile, lam=LAM, k_reg=K_REG):
    """Construct RAPS prediction sets given a threshold quantile."""
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


def compute_prediction_sets_maccp(test_probs, agree_mask, q_agree, q_disagree,
                                   lam=LAM, k_reg=K_REG):
    """Construct MACCP prediction sets: use q_agree when models agree, q_disagree otherwise."""
    n, k = test_probs.shape
    sets = np.zeros((n, k), dtype=bool)
    for i in range(n):
        q = q_agree if agree_mask[i] else q_disagree
        sorted_idx = np.argsort(-test_probs[i])
        cumsum = 0.0
        for rank, class_idx in enumerate(sorted_idx):
            cumsum += test_probs[i, class_idx]
            penalty = lam * max(0, rank + 1 - k_reg)
            cumsum_reg = cumsum + penalty
            sets[i, class_idx] = True
            if cumsum_reg >= q:
                break
    return sets


def conformal_quantile(scores, alpha):
    """Compute conformal quantile with finite-sample correction."""
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return np.quantile(scores, q_level, method="higher")


# ============================================================
# Evaluation helpers
# ============================================================
def evaluate_sets(pred_sets, labels, description=""):
    """Compute coverage, set size, singleton rate/accuracy for prediction sets."""
    n = len(labels)
    set_sizes = pred_sets.sum(axis=1)

    # Coverage
    coverage = np.mean([pred_sets[i, labels[i]] for i in range(n)])

    # Singleton analysis
    singleton_mask = set_sizes == 1
    singleton_rate = singleton_mask.mean()
    if singleton_mask.sum() > 0:
        singleton_preds = pred_sets[singleton_mask].argmax(axis=1)
        singleton_acc = accuracy_score(labels[singleton_mask], singleton_preds)
    else:
        singleton_acc = 0.0

    return {
        "description": description,
        "coverage": float(coverage),
        "mean_set_size": float(set_sizes.mean()),
        "median_set_size": float(np.median(set_sizes)),
        "singleton_rate": float(singleton_rate),
        "singleton_accuracy": float(singleton_acc),
        "n": int(n),
    }


def _trapz(y, x):
    """Trapezoidal integration (compatible with numpy 1.x and 2.x)."""
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def compute_augrc(probs, labels, preds):
    """Compute AUGRC (Area Under Generalized Risk-Coverage Curve)."""
    confidences = probs.max(axis=1)
    correct = (preds == labels).astype(float)
    sorted_idx = np.argsort(-confidences)
    sorted_correct = correct[sorted_idx]
    n = len(sorted_correct)
    coverages = np.arange(1, n + 1) / n
    cumulative_errors = np.cumsum(1 - sorted_correct)
    generalized_risks = cumulative_errors / n
    augrc = _trapz(generalized_risks, coverages)
    return augrc


def bootstrap_augrc_ci(probs, labels, preds, n_bootstrap=N_BOOTSTRAP, ci=0.95):
    """Bootstrap 95% CI for AUGRC."""
    n = len(labels)
    samples = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        samples.append(compute_augrc(probs[idx], labels[idx], preds[idx]))
    samples = np.array(samples)
    lo = np.percentile(samples, (1 - ci) / 2 * 100)
    hi = np.percentile(samples, (1 + ci) / 2 * 100)
    return float(np.mean(samples)), float(lo), float(hi)


# ============================================================
# Feature engineering (copied from run_xgb_agreement_analysis.py)
# ============================================================
SEVERITY_MAP = {
    'blocker': 5, 'critical': 4, 'major': 3, 'normal': 2,
    'minor': 1, 'trivial': 0, 'enhancement': -1
}
PRIORITY_MAP = {'P1': 4, 'P2': 3, 'P3': 2, 'P4': 1, 'P5': 0}


def engineer_features(df):
    """Replicate eclipse_zenodo_loader.engineer_features() on our parquet data."""
    df = df.copy()
    df['summary_length'] = df['text'].str.split(' \\[SEP\\] ').str[0].str.len()
    df['summary_word_count'] = df['text'].str.split(' \\[SEP\\] ').str[0].str.split().str.len()
    desc = df['text'].str.split(' \\[SEP\\] ').str[1].fillna('')
    df['desc_length'] = desc.str.len()
    df['desc_word_count'] = desc.str.split().str.len()
    df['has_description'] = (df['desc_length'] > 10).astype(int)

    if 'severity' in df.columns:
        sev_lower = df['severity'].fillna('normal').str.lower()
        df['severity_numeric'] = sev_lower.map(SEVERITY_MAP).fillna(2).astype(int)
        df['is_enhancement'] = (sev_lower == 'enhancement').astype(int)
        df['is_high_severity'] = sev_lower.isin(['blocker', 'critical']).astype(int)
    else:
        df['severity_numeric'] = 2
        df['is_enhancement'] = 0
        df['is_high_severity'] = 0

    if 'priority' in df.columns:
        df['priority_numeric'] = df['priority'].fillna('P3').map(PRIORITY_MAP).fillna(2).astype(int)
    else:
        df['priority_numeric'] = 2

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
    encoders = {}
    for col in cat_cols:
        if col not in train_df.columns:
            continue
        le = LabelEncoder()
        all_vals = list(train_df[col].fillna('unknown').unique()) + ['unknown']
        le.fit(all_vals)
        enc_col = f'{col}_enc'
        for df in [train_df, cal_df, test_df]:
            vals = df[col].fillna('unknown').copy()
            vals = vals.where(vals.isin(le.classes_), other='unknown')
            df[enc_col] = le.transform(vals)
        encoders[col] = le
    return encoders


# ============================================================
# PART A: Retrain XGBoost to get calibration predictions
# ============================================================
print("=" * 70)
print("PART A: Retraining XGBoost (need calibration predictions)")
print("=" * 70)

label_map = json.load(open(DATA_DIR / "label_mapping.json"))
inv_map = {v: k for k, v in label_map.items()}
num_classes = len(label_map)
print(f"Classes: {num_classes}")

train_df = pd.read_parquet(DATA_DIR / "train.parquet")
cal_df = pd.read_parquet(DATA_DIR / "cal.parquet")
test_df = pd.read_parquet(DATA_DIR / "test.parquet")
print(f"Train: {len(train_df):,} | Cal: {len(cal_df):,} | Test: {len(test_df):,}")

# Engineer features
for name, df in [("train", train_df), ("cal", cal_df), ("test", test_df)]:
    engineered = engineer_features(df)
    for col in engineered.columns:
        if col not in df.columns:
            df[col] = engineered[col]

# Creator bug count from training data only
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

# Numeric features
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

# TF-IDF
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
scaler = StandardScaler()
X_train_struct = scaler.fit_transform(train_df[structured_features].values.astype(float))
X_cal_struct = scaler.transform(cal_df[structured_features].values.astype(float))
X_test_struct = scaler.transform(test_df[structured_features].values.astype(float))

# Combine
X_train = hstack([csr_matrix(X_train_struct), X_train_tfidf])
X_cal = hstack([csr_matrix(X_cal_struct), X_cal_tfidf])
X_test = hstack([csr_matrix(X_test_struct), X_test_tfidf])

y_train = train_df["label"].values
y_cal = cal_df["label"].values
y_test = test_df["label"].values

print(f"\nTraining XGBoost...")
print(f"  Features: {X_train.shape[1]} ({len(structured_features)} structured + {X_train_tfidf.shape[1]} TF-IDF)")
print(f"  Samples: {X_train.shape[0]:,} train, {X_cal.shape[0]:,} cal, {X_test.shape[0]:,} test")

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
print(f"Best iteration: {xgb.best_iteration}")

# Get predictions for BOTH cal and test
xgb_cal_preds = xgb.predict(X_cal)
xgb_cal_probs = xgb.predict_proba(X_cal)
xgb_test_preds = xgb.predict(X_test)
xgb_test_probs = xgb.predict_proba(X_test)

xgb_cal_acc = accuracy_score(y_cal, xgb_cal_preds)
xgb_test_acc = accuracy_score(y_test, xgb_test_preds)
print(f"XGBoost cal accuracy: {xgb_cal_acc:.4f}")
print(f"XGBoost test accuracy: {xgb_test_acc:.4f}")

# Save XGBoost cal and test predictions
np.save(OUTPUT_DIR / "xgb_cal_preds.npy", xgb_cal_preds)
np.save(OUTPUT_DIR / "xgb_cal_probs.npy", xgb_cal_probs)
np.save(OUTPUT_DIR / "xgb_test_preds.npy", xgb_test_preds)
np.save(OUTPUT_DIR / "xgb_test_probs.npy", xgb_test_probs)

# Load DeBERTa predictions
print("\nLoading DeBERTa predictions...")
deb_cal_probs = np.load(DEBERTA_DIR / "cal_probs.npy")
deb_cal_labels = np.load(DEBERTA_DIR / "cal_labels.npy")
deb_cal_preds = np.load(DEBERTA_DIR / "cal_preds.npy")
deb_test_probs = np.load(DEBERTA_DIR / "test_probs.npy")
deb_test_labels = np.load(DEBERTA_DIR / "test_labels.npy")
deb_test_preds = np.load(DEBERTA_DIR / "test_preds.npy")

# Verify labels match (allow tiny discrepancies from parquet row ordering)
cal_match = np.array_equal(y_cal, deb_cal_labels)
test_match = np.array_equal(y_test, deb_test_labels)
if not cal_match:
    n_diff = (y_cal != deb_cal_labels).sum()
    print(f"  WARNING: {n_diff}/{len(y_cal)} cal labels differ (parquet vs DeBERTa npy)")
    print(f"  Using DeBERTa labels as ground truth for calibration")
    y_cal = deb_cal_labels
if not test_match:
    n_diff = (y_test != deb_test_labels).sum()
    print(f"  WARNING: {n_diff}/{len(y_test)} test labels differ (parquet vs DeBERTa npy)")
    print(f"  Using DeBERTa labels as ground truth for test")
    y_test = deb_test_labels
deb_test_acc = accuracy_score(y_test, deb_test_preds)
deb_cal_acc = accuracy_score(y_cal, deb_cal_preds)
print(f"DeBERTa cal accuracy: {deb_cal_acc:.4f}")
print(f"DeBERTa test accuracy: {deb_test_acc:.4f}")

# ============================================================
# Agreement masks
# ============================================================
cal_agree_mask = (xgb_cal_preds == deb_cal_preds)
test_agree_mask = (xgb_test_preds == deb_test_preds)

n_cal_agree = cal_agree_mask.sum()
n_cal_disagree = (~cal_agree_mask).sum()
n_test_agree = test_agree_mask.sum()
n_test_disagree = (~test_agree_mask).sum()

print(f"\nAgreement statistics:")
print(f"  Cal:  {n_cal_agree:,} agree ({n_cal_agree/len(y_cal):.1%}), "
      f"{n_cal_disagree:,} disagree ({n_cal_disagree/len(y_cal):.1%})")
print(f"  Test: {n_test_agree:,} agree ({n_test_agree/len(y_test):.1%}), "
      f"{n_test_disagree:,} disagree ({n_test_disagree/len(y_test):.1%})")

# DeBERTa accuracy by agreement status
deb_cal_correct = deb_cal_preds == y_cal
deb_test_correct = deb_test_preds == y_test
print(f"  DeBERTa cal acc when agree:    {deb_cal_correct[cal_agree_mask].mean():.4f}")
print(f"  DeBERTa cal acc when disagree: {deb_cal_correct[~cal_agree_mask].mean():.4f}")
print(f"  DeBERTa test acc when agree:    {deb_test_correct[test_agree_mask].mean():.4f}")
print(f"  DeBERTa test acc when disagree: {deb_test_correct[~test_agree_mask].mean():.4f}")


# ============================================================
# PART B: MACCP Implementation
# ============================================================
print("\n" + "=" * 70)
print("PART B: Model-Agreement-Conditioned Conformal Prediction (MACCP)")
print("=" * 70)

# Compute RAPS scores on DeBERTa calibration probabilities
print("\nComputing RAPS nonconformity scores on DeBERTa calibration set...")
deb_cal_scores = raps_nonconformity_scores(deb_cal_probs, deb_cal_labels, lam=LAM, k_reg=K_REG)
print(f"  Score range: [{deb_cal_scores.min():.4f}, {deb_cal_scores.max():.4f}]")
print(f"  Score mean: {deb_cal_scores.mean():.4f}")

# Split scores into agree/disagree groups
scores_agree = deb_cal_scores[cal_agree_mask]
scores_disagree = deb_cal_scores[~cal_agree_mask]
print(f"  Agree scores: n={len(scores_agree):,}, mean={scores_agree.mean():.4f}")
print(f"  Disagree scores: n={len(scores_disagree):,}, mean={scores_disagree.mean():.4f}")

# Compute RAPS scores on XGBoost calibration probabilities (for baseline)
print("\nComputing RAPS nonconformity scores on XGBoost calibration set...")
xgb_cal_scores = raps_nonconformity_scores(xgb_cal_probs, y_cal, lam=LAM, k_reg=K_REG)
print(f"  Score range: [{xgb_cal_scores.min():.4f}, {xgb_cal_scores.max():.4f}]")
print(f"  Score mean: {xgb_cal_scores.mean():.4f}")


# ============================================================
# PART C & D: Evaluate all methods at all alpha levels
# ============================================================
all_results = {}

for alpha in ALPHA_LEVELS:
    print(f"\n{'=' * 70}")
    print(f"ALPHA = {alpha:.2f}  (target coverage = {1-alpha:.0%})")
    print(f"{'=' * 70}")

    results_this_alpha = {}

    # --- 1. DeBERTa RAPS (baseline) ---
    q_deb = conformal_quantile(deb_cal_scores, alpha)
    sets_deb = compute_prediction_sets(deb_test_probs, q_deb, lam=LAM, k_reg=K_REG)
    r_deb = evaluate_sets(sets_deb, y_test, "DeBERTa RAPS")
    r_deb["quantile"] = float(q_deb)
    results_this_alpha["deberta_raps"] = r_deb

    # --- 2. XGBoost RAPS (baseline) ---
    q_xgb = conformal_quantile(xgb_cal_scores, alpha)
    sets_xgb = compute_prediction_sets(xgb_test_probs, q_xgb, lam=LAM, k_reg=K_REG)
    r_xgb = evaluate_sets(sets_xgb, y_test, "XGBoost RAPS")
    r_xgb["quantile"] = float(q_xgb)
    results_this_alpha["xgboost_raps"] = r_xgb

    # --- 3. MACCP ---
    q_agree = conformal_quantile(scores_agree, alpha)
    q_disagree = conformal_quantile(scores_disagree, alpha)
    print(f"\n  MACCP thresholds: q_agree={q_agree:.4f}, q_disagree={q_disagree:.4f}")
    print(f"  Threshold ratio (disagree/agree): {q_disagree/q_agree:.2f}x")

    sets_maccp = compute_prediction_sets_maccp(
        deb_test_probs, test_agree_mask, q_agree, q_disagree, lam=LAM, k_reg=K_REG
    )

    # MACCP overall
    r_maccp = evaluate_sets(sets_maccp, y_test, "MACCP overall")
    r_maccp["q_agree"] = float(q_agree)
    r_maccp["q_disagree"] = float(q_disagree)
    results_this_alpha["maccp_overall"] = r_maccp

    # MACCP agree subset
    r_maccp_agree = evaluate_sets(
        sets_maccp[test_agree_mask], y_test[test_agree_mask], "MACCP agree"
    )
    results_this_alpha["maccp_agree"] = r_maccp_agree

    # MACCP disagree subset
    r_maccp_disagree = evaluate_sets(
        sets_maccp[~test_agree_mask], y_test[~test_agree_mask], "MACCP disagree"
    )
    results_this_alpha["maccp_disagree"] = r_maccp_disagree

    # DeBERTa RAPS on agree/disagree subsets (for comparison)
    r_deb_agree = evaluate_sets(
        sets_deb[test_agree_mask], y_test[test_agree_mask], "DeBERTa RAPS (agree subset)"
    )
    results_this_alpha["deberta_raps_agree"] = r_deb_agree

    r_deb_disagree = evaluate_sets(
        sets_deb[~test_agree_mask], y_test[~test_agree_mask], "DeBERTa RAPS (disagree subset)"
    )
    results_this_alpha["deberta_raps_disagree"] = r_deb_disagree

    # --- Print comparison table ---
    print(f"\n  {'Method':<30} | {'Coverage':>8} | {'Mean SS':>7} | {'Med SS':>6} | {'Sing%':>6} | {'SingAcc':>7} | {'N':>7}")
    print(f"  {'-'*30}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}")
    for key, r in results_this_alpha.items():
        print(f"  {r['description']:<30} | {r['coverage']:8.4f} | {r['mean_set_size']:7.2f} | "
              f"{r['median_set_size']:6.1f} | {r['singleton_rate']:5.1%} | {r['singleton_accuracy']:7.4f} | {r['n']:7,}")

    all_results[f"alpha_{alpha}"] = results_this_alpha


# ============================================================
# AUGRC with bootstrap CIs
# ============================================================
print(f"\n{'=' * 70}")
print("AUGRC ANALYSIS (selective classification quality)")
print(f"{'=' * 70}")

print(f"\nComputing AUGRC with {N_BOOTSTRAP} bootstrap resamples...")

# DeBERTa AUGRC
augrc_deb = compute_augrc(deb_test_probs, y_test, deb_test_preds)
augrc_deb_mean, augrc_deb_lo, augrc_deb_hi = bootstrap_augrc_ci(
    deb_test_probs, y_test, deb_test_preds
)
print(f"  DeBERTa AUGRC: {augrc_deb:.4f} [{augrc_deb_lo:.4f}, {augrc_deb_hi:.4f}]")

# XGBoost AUGRC
augrc_xgb = compute_augrc(xgb_test_probs, y_test, xgb_test_preds)
augrc_xgb_mean, augrc_xgb_lo, augrc_xgb_hi = bootstrap_augrc_ci(
    xgb_test_probs, y_test, xgb_test_preds
)
print(f"  XGBoost AUGRC: {augrc_xgb:.4f} [{augrc_xgb_lo:.4f}, {augrc_xgb_hi:.4f}]")

# MACCP AUGRC -- use DeBERTa probs but with agreement-aware confidence:
# For AUGRC, we need a scalar confidence per sample. We use DeBERTa's max prob
# but boost it when models agree (equivalent to reranking).
# MACCP effective confidence = DeBERTa max_prob + agreement_bonus
# This is principled: agreement indicates we can use tighter threshold.
deb_max_prob = deb_test_probs.max(axis=1)
# For AUGRC, we use DeBERTa's prediction as the point prediction,
# but order by an agreement-boosted confidence:
# agree -> higher effective confidence (tighter sets -> more singletons)
maccp_confidence = deb_max_prob.copy()
maccp_confidence[test_agree_mask] += 1.0  # ensure agree always ranked above disagree at same prob

augrc_maccp = compute_augrc(
    # Need to pass probs with max = our maccp_confidence for AUGRC computation
    # Create synthetic probs array where max(axis=1) = maccp_confidence
    np.column_stack([maccp_confidence, np.zeros((len(y_test), num_classes - 1))]),
    y_test, deb_test_preds
)
augrc_maccp_mean, augrc_maccp_lo, augrc_maccp_hi = bootstrap_augrc_ci(
    np.column_stack([maccp_confidence, np.zeros((len(y_test), num_classes - 1))]),
    y_test, deb_test_preds
)
print(f"  MACCP AUGRC:   {augrc_maccp:.4f} [{augrc_maccp_lo:.4f}, {augrc_maccp_hi:.4f}]")

all_results["augrc"] = {
    "deberta": {
        "augrc": float(augrc_deb),
        "ci_lo": float(augrc_deb_lo),
        "ci_hi": float(augrc_deb_hi),
    },
    "xgboost": {
        "augrc": float(augrc_xgb),
        "ci_lo": float(augrc_xgb_lo),
        "ci_hi": float(augrc_xgb_hi),
    },
    "maccp": {
        "augrc": float(augrc_maccp),
        "ci_lo": float(augrc_maccp_lo),
        "ci_hi": float(augrc_maccp_hi),
    },
}


# ============================================================
# Summary comparison table
# ============================================================
print(f"\n{'=' * 70}")
print("SUMMARY: MACCP vs Baselines (set size reduction at guaranteed coverage)")
print(f"{'=' * 70}")

print(f"\n  {'Alpha':<6} | {'Method':<15} | {'Coverage':>8} | {'MeanSS':>7} | {'MedSS':>6} | {'Sing%':>6} | {'SingAcc':>7}")
print(f"  {'-'*6}-+-{'-'*15}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}")

for alpha in ALPHA_LEVELS:
    key = f"alpha_{alpha}"
    for method in ["deberta_raps", "xgboost_raps", "maccp_overall"]:
        r = all_results[key][method]
        label = {"deberta_raps": "DeBERTa", "xgboost_raps": "XGBoost", "maccp_overall": "MACCP"}[method]
        print(f"  {alpha:<6.2f} | {label:<15} | {r['coverage']:8.4f} | {r['mean_set_size']:7.2f} | "
              f"{r['median_set_size']:6.1f} | {r['singleton_rate']:5.1%} | {r['singleton_accuracy']:7.4f}")
    print(f"  {'-'*6}-+-{'-'*15}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}")

# Set size reduction %
print(f"\n  SET SIZE REDUCTION: MACCP vs DeBERTa RAPS")
print(f"  {'Alpha':<6} | {'DeBERTa SS':>10} | {'MACCP SS':>9} | {'Reduction':>9} | {'Reduction%':>10}")
print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*10}")
for alpha in ALPHA_LEVELS:
    key = f"alpha_{alpha}"
    deb_ss = all_results[key]["deberta_raps"]["mean_set_size"]
    mac_ss = all_results[key]["maccp_overall"]["mean_set_size"]
    reduction = deb_ss - mac_ss
    reduction_pct = (reduction / deb_ss) * 100 if deb_ss > 0 else 0
    print(f"  {alpha:<6.2f} | {deb_ss:10.2f} | {mac_ss:9.2f} | {reduction:9.2f} | {reduction_pct:9.1f}%")

# Agree vs Disagree breakdown
print(f"\n  MACCP AGREE vs DISAGREE BREAKDOWN")
print(f"  {'Alpha':<6} | {'Group':<10} | {'N':>7} | {'Coverage':>8} | {'MeanSS':>7} | {'Sing%':>6} | {'SingAcc':>7}")
print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}")
for alpha in ALPHA_LEVELS:
    key = f"alpha_{alpha}"
    for group in ["maccp_agree", "maccp_disagree"]:
        r = all_results[key][group]
        label = "Agree" if group == "maccp_agree" else "Disagree"
        print(f"  {alpha:<6.2f} | {label:<10} | {r['n']:7,} | {r['coverage']:8.4f} | "
              f"{r['mean_set_size']:7.2f} | {r['singleton_rate']:5.1%} | {r['singleton_accuracy']:7.4f}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}")


# ============================================================
# KEY FINDINGS for paper
# ============================================================
print(f"\n{'=' * 70}")
print("KEY FINDINGS FOR PAPER")
print(f"{'=' * 70}")

# At alpha=0.10 (most common in literature)
alpha_key = "alpha_0.1"
r_deb = all_results[alpha_key]["deberta_raps"]
r_xgb = all_results[alpha_key]["xgboost_raps"]
r_mac = all_results[alpha_key]["maccp_overall"]
r_mac_a = all_results[alpha_key]["maccp_agree"]
r_mac_d = all_results[alpha_key]["maccp_disagree"]

print(f"\n  At alpha=0.10 (90% target coverage):")
print(f"  1. DeBERTa RAPS: coverage={r_deb['coverage']:.1%}, mean SS={r_deb['mean_set_size']:.2f}, "
      f"singleton rate={r_deb['singleton_rate']:.1%}")
print(f"  2. XGBoost RAPS: coverage={r_xgb['coverage']:.1%}, mean SS={r_xgb['mean_set_size']:.2f}, "
      f"singleton rate={r_xgb['singleton_rate']:.1%}")
print(f"  3. MACCP overall: coverage={r_mac['coverage']:.1%}, mean SS={r_mac['mean_set_size']:.2f}, "
      f"singleton rate={r_mac['singleton_rate']:.1%}")
print(f"  4. MACCP agree:   coverage={r_mac_a['coverage']:.1%}, mean SS={r_mac_a['mean_set_size']:.2f}, "
      f"singleton rate={r_mac_a['singleton_rate']:.1%}, accuracy={r_mac_a['singleton_accuracy']:.1%}")
print(f"  5. MACCP disagree: coverage={r_mac_d['coverage']:.1%}, mean SS={r_mac_d['mean_set_size']:.2f}, "
      f"singleton rate={r_mac_d['singleton_rate']:.1%}")

deb_ss = r_deb['mean_set_size']
mac_ss = r_mac['mean_set_size']
reduction_pct = (deb_ss - mac_ss) / deb_ss * 100
print(f"\n  Set size reduction: {deb_ss:.2f} -> {mac_ss:.2f} ({reduction_pct:+.1f}%)")
print(f"  Agreement rate (test): {test_agree_mask.mean():.1%}")
print(f"  AUGRC: DeBERTa={augrc_deb:.4f}, XGBoost={augrc_xgb:.4f}, MACCP={augrc_maccp:.4f}")

# ============================================================
# Save results
# ============================================================
# Add metadata
all_results["metadata"] = {
    "deberta_test_accuracy": float(deb_test_acc),
    "xgboost_test_accuracy": float(xgb_test_acc),
    "deberta_cal_accuracy": float(deb_cal_acc),
    "xgboost_cal_accuracy": float(xgb_cal_acc),
    "n_cal": int(len(y_cal)),
    "n_test": int(len(y_test)),
    "n_classes": int(num_classes),
    "cal_agreement_rate": float(cal_agree_mask.mean()),
    "test_agreement_rate": float(test_agree_mask.mean()),
    "raps_lambda": LAM,
    "raps_k_reg": K_REG,
    "random_seed": 42,
    "n_bootstrap": N_BOOTSTRAP,
}

json.dump(all_results, open(OUTPUT_DIR / "maccp_results.json", "w"), indent=2)
print(f"\nResults saved to {OUTPUT_DIR}/maccp_results.json")
print("Done.")
