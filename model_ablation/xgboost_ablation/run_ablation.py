"""
XGBoost feature-channel ablation for MACCP independence claim.

Trains three XGBoost variants on Eclipse 30-class:
  1. Metadata only (no TF-IDF)
  2. TF-IDF only (no metadata)
  3. Full = Metadata + TF-IDF (existing)

For each variant: top-1 accuracy, agreement rate with DeBERTa, agree-acc,
disagree-acc, gap. Saves results.json + prints a markdown table.

Isolated -- does not modify existing files.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# ─── Paths ───
PROJ_ROOT = Path("/home/zephyrus/perf-regression-ci")
DATA_DIR = PROJ_ROOT / "outputs" / "eclipse_no_other"
DEBERTA_DIR = PROJ_ROOT / "outputs" / "deberta_no_other"
OUT_DIR = PROJ_ROOT / "model_ablation" / "xgboost_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Load data ───
print("=== Loading data ===")
label_map = json.load(open(DATA_DIR / "label_mapping.json"))
num_classes = len(label_map)
print(f"Classes: {num_classes}")

train_df = pd.read_parquet(DATA_DIR / "train.parquet")
cal_df = pd.read_parquet(DATA_DIR / "cal.parquet")
test_df = pd.read_parquet(DATA_DIR / "test.parquet")
print(f"Train: {len(train_df):,} | Cal: {len(cal_df):,} | Test: {len(test_df):,}")

deb_test_preds = np.load(DEBERTA_DIR / "test_preds.npy")
deb_test_labels = np.load(DEBERTA_DIR / "test_labels.npy")
y_test = test_df["label"].values
assert np.array_equal(y_test, deb_test_labels), "Label alignment failed"
deb_correct = deb_test_preds == y_test
print(f"DeBERTa test accuracy: {deb_correct.mean():.4f}")

# ─── Feature engineering (mirror run_xgb_agreement_analysis.py) ───
SEVERITY_MAP = {
    "blocker": 5,
    "critical": 4,
    "major": 3,
    "normal": 2,
    "minor": 1,
    "trivial": 0,
    "enhancement": -1,
}
PRIORITY_MAP = {"P1": 4, "P2": 3, "P3": 2, "P4": 1, "P5": 0}


def engineer_features(df):
    df = df.copy()
    df["summary_length"] = df["text"].str.split(" \\[SEP\\] ").str[0].str.len()
    df["summary_word_count"] = df["text"].str.split(" \\[SEP\\] ").str[0].str.split().str.len()
    desc = df["text"].str.split(" \\[SEP\\] ").str[1].fillna("")
    df["desc_length"] = desc.str.len()
    df["desc_word_count"] = desc.str.split().str.len()
    df["has_description"] = (df["desc_length"] > 10).astype(int)

    if "severity" in df.columns:
        sev = df["severity"].fillna("normal").str.lower()
        df["severity_numeric"] = sev.map(SEVERITY_MAP).fillna(2).astype(int)
        df["is_enhancement"] = (sev == "enhancement").astype(int)
        df["is_high_severity"] = sev.isin(["blocker", "critical"]).astype(int)
    else:
        df["severity_numeric"] = 2
        df["is_enhancement"] = 0
        df["is_high_severity"] = 0

    if "priority" in df.columns:
        df["priority_numeric"] = df["priority"].fillna("P3").map(PRIORITY_MAP).fillna(2).astype(int)
    else:
        df["priority_numeric"] = 2

    if "creation_time" in df.columns:
        dt = pd.to_datetime(df["creation_time"], errors="coerce")
        df["open_hour"] = dt.dt.hour.fillna(12).astype(int)
        df["open_dayofweek"] = dt.dt.dayofweek.fillna(0).astype(int)
        df["open_month"] = dt.dt.month.fillna(1).astype(int)
    else:
        df["open_hour"] = 12
        df["open_dayofweek"] = 0
        df["open_month"] = 1
    return df


def encode_categoricals(train_df, cal_df, test_df, cat_cols):
    encoders = {}
    for col in cat_cols:
        if col not in train_df.columns:
            continue
        le = LabelEncoder()
        all_vals = list(train_df[col].fillna("unknown").unique()) + ["unknown"]
        le.fit(all_vals)
        enc_col = f"{col}_enc"
        for df in [train_df, cal_df, test_df]:
            vals = df[col].fillna("unknown").copy()
            vals = vals.where(vals.isin(le.classes_), other="unknown")
            df[enc_col] = le.transform(vals)
        encoders[col] = le
    return encoders


for df in [train_df, cal_df, test_df]:
    eng = engineer_features(df)
    for col in eng.columns:
        if col not in df.columns:
            df[col] = eng[col]

if "creator" in train_df.columns:
    creator_counts = train_df["creator"].value_counts().to_dict()
    for df in [train_df, cal_df, test_df]:
        df["creator_bug_count"] = df["creator"].map(creator_counts).fillna(1).astype(int)

cat_cols = [c for c in ["severity", "priority", "platform", "op_sys", "product"] if c in train_df.columns]
encode_categoricals(train_df, cal_df, test_df, cat_cols)

numeric_features = [
    "summary_length", "summary_word_count", "desc_length", "desc_word_count",
    "has_description", "severity_numeric", "priority_numeric",
    "is_enhancement", "is_high_severity", "open_hour", "open_dayofweek", "open_month",
]
if "creator_bug_count" in train_df.columns:
    numeric_features.append("creator_bug_count")
cat_enc_features = [f"{c}_enc" for c in cat_cols if f"{c}_enc" in train_df.columns]

# ── Metadata-only feature set ──
# Per task spec: product encoding, creator bug count, severity, priority, OS, platform,
# component size. Drop pure text length features (summary/desc length, word counts) which
# are also "text-derived." Keep numeric encodings of severity/priority/temporal.
METADATA_FEATURES = [
    "severity_numeric", "priority_numeric", "is_enhancement", "is_high_severity",
    "open_hour", "open_dayofweek", "open_month",
] + ([f for f in ["creator_bug_count"] if f in train_df.columns]) + cat_enc_features
print(f"\nMetadata features ({len(METADATA_FEATURES)}): {METADATA_FEATURES}")

# Full feature set (matches existing pipeline) -- includes the text-length features
FULL_STRUCTURED = numeric_features + cat_enc_features
print(f"Full structured features ({len(FULL_STRUCTURED)})")

# ─── TF-IDF (matching existing config) ───
print("\n=== Building TF-IDF (500-dim, ngram (1,2), min_df=2) ===")
tfidf = TfidfVectorizer(
    max_features=500, ngram_range=(1, 2),
    stop_words="english", min_df=2, sublinear_tf=True,
)
X_train_tfidf = tfidf.fit_transform(train_df["text"].values)
X_cal_tfidf = tfidf.transform(cal_df["text"].values)
X_test_tfidf = tfidf.transform(test_df["text"].values)
print(f"TF-IDF shape: {X_train_tfidf.shape}")

# ─── Build feature matrices ───
def build_metadata_only():
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train_df[METADATA_FEATURES].values.astype(float))
    Xc = scaler.transform(cal_df[METADATA_FEATURES].values.astype(float))
    Xt = scaler.transform(test_df[METADATA_FEATURES].values.astype(float))
    return csr_matrix(Xtr), csr_matrix(Xc), csr_matrix(Xt)


def build_tfidf_only():
    return X_train_tfidf, X_cal_tfidf, X_test_tfidf


def build_full():
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(train_df[FULL_STRUCTURED].values.astype(float))
    Xc_s = scaler.transform(cal_df[FULL_STRUCTURED].values.astype(float))
    Xt_s = scaler.transform(test_df[FULL_STRUCTURED].values.astype(float))
    return (
        hstack([csr_matrix(Xtr_s), X_train_tfidf]).tocsr(),
        hstack([csr_matrix(Xc_s), X_cal_tfidf]).tocsr(),
        hstack([csr_matrix(Xt_s), X_test_tfidf]).tocsr(),
    )


VARIANTS = {
    "metadata_only": build_metadata_only,
    "tfidf_only": build_tfidf_only,
    "full": build_full,
}

y_train = train_df["label"].values
y_cal = cal_df["label"].values

results = {}

for name, builder in VARIANTS.items():
    print(f"\n{'=' * 70}")
    print(f"Variant: {name}")
    print(f"{'=' * 70}")
    Xtr, Xc, Xt = builder()
    print(f"  shape: train={Xtr.shape}, cal={Xc.shape}, test={Xt.shape}")

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
    xgb.fit(Xtr, y_train, eval_set=[(Xc, y_cal)], verbose=False)
    print(f"  best_iter: {xgb.best_iteration}")

    test_preds = xgb.predict(Xt)
    test_acc = accuracy_score(y_test, test_preds)
    xgb_correct = test_preds == y_test

    agree_mask = test_preds == deb_test_preds
    n_agree = int(agree_mask.sum())
    n_total = len(y_test)
    agree_rate = n_agree / n_total

    if n_agree > 0:
        agree_acc_deb = float(deb_correct[agree_mask].mean())
    else:
        agree_acc_deb = float("nan")
    if (n_total - n_agree) > 0:
        disagree_acc_deb = float(deb_correct[~agree_mask].mean())
    else:
        disagree_acc_deb = float("nan")
    gap = agree_acc_deb - disagree_acc_deb

    print(f"  XGBoost test acc:        {test_acc:.4f}")
    print(f"  Agreement rate:          {agree_rate:.4f} ({n_agree}/{n_total})")
    print(f"  DeBERTa acc | agree:     {agree_acc_deb:.4f}")
    print(f"  DeBERTa acc | disagree:  {disagree_acc_deb:.4f}")
    print(f"  Gap (agree - disagree):  {gap:+.4f}")

    results[name] = {
        "n_features": int(Xtr.shape[1]),
        "best_iteration": int(xgb.best_iteration),
        "xgb_test_accuracy": float(test_acc),
        "agreement_rate": float(agree_rate),
        "n_agree": n_agree,
        "n_total": n_total,
        "deberta_acc_when_agree": agree_acc_deb,
        "deberta_acc_when_disagree": disagree_acc_deb,
        "gap": float(gap),
    }

# ─── Save and print summary table ───
out_path = OUT_DIR / "results.json"
json.dump(results, open(out_path, "w"), indent=2)
print(f"\nSaved: {out_path}")

print("\n\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"{'Variant':<22}{'Acc':>8}{'AgreeRate':>12}{'AgreeAcc':>12}{'DisagAcc':>12}{'Gap':>10}")
print("-" * 80)
for name in ["metadata_only", "tfidf_only", "full"]:
    r = results[name]
    print(
        f"{name:<22}"
        f"{r['xgb_test_accuracy']*100:>7.1f}%"
        f"{r['agreement_rate']*100:>11.1f}%"
        f"{r['deberta_acc_when_agree']*100:>11.1f}%"
        f"{r['deberta_acc_when_disagree']*100:>11.1f}%"
        f"{r['gap']*100:>+9.1f}pp"
    )

# ─── Verdict for paper inclusion ───
print("\n" + "=" * 80)
metadata_gap = results["metadata_only"]["gap"]
if metadata_gap >= 0.30:
    print(f"VERDICT: metadata-only gap = {metadata_gap*100:.1f}pp >= 30pp")
    print("-> Independence claim SUPPORTED. Add ablation table to paper.")
elif metadata_gap >= 0.15:
    print(f"VERDICT: metadata-only gap = {metadata_gap*100:.1f}pp (15-30pp range)")
    print("-> Mixed; report cautiously.")
else:
    print(f"VERDICT: metadata-only gap = {metadata_gap*100:.1f}pp < 15pp")
    print("-> Independence claim WEAKENED. Do NOT add to paper without revision.")
print("=" * 80)
