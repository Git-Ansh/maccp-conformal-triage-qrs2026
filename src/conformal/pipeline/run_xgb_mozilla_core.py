"""
run_xgb_mozilla_core.py -- Train XGBoost on Mozilla Core bug component assignment.

Matches the feature engineering from run_xgb_agreement_analysis.py (Eclipse).
Mozilla Core data only has: Bug ID, Summary, Component, Number of Comments.
No severity/priority/description/reporter fields.

Usage:
    python run_xgb_mozilla_core.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
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

# ---- Paths ----
DATA_DIR = Path("data/mozilla_core")
OUTPUT_DIR = Path("conformal_outputs/mozilla_core")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load data ----
print("=== Loading Data ===")
label_map = json.load(open(DATA_DIR / "label_mapping.json"))
inv_map = {v: k for k, v in label_map.items()}
num_classes = len(label_map)
print(f"Classes: {num_classes}")

train_df = pd.read_parquet(DATA_DIR / "train.parquet")
cal_df = pd.read_parquet(DATA_DIR / "cal.parquet")
test_df = pd.read_parquet(DATA_DIR / "test.parquet")
print(f"Train: {len(train_df):,} | Cal: {len(cal_df):,} | Test: {len(test_df):,}")

# ---- Feature Engineering ----
print("\n=== Feature Engineering ===")


def engineer_features(df):
    """Engineer structured features from available Mozilla Core columns."""
    df = df.copy()

    # Text length features (only have summary, no description)
    summary_text = df['text'].fillna('')
    df['summary_length'] = summary_text.str.len()
    df['summary_word_count'] = summary_text.str.split().str.len().fillna(0).astype(int)

    # Number of comments (if available -- comes from raw data via 'component' column join)
    # Not available in parquet, so skip

    # No severity/priority/description/reporter available for this dataset
    # Set defaults for consistency with the XGBoost pipeline
    df['severity_numeric'] = 2  # default normal
    df['priority_numeric'] = 2  # default P3
    df['is_enhancement'] = 0
    df['is_high_severity'] = 0

    # No temporal features (creation_ts is synthetic from Bug ID)
    df['open_hour'] = 12
    df['open_dayofweek'] = 0
    df['open_month'] = 1

    return df


for name, df in [("train", train_df), ("cal", cal_df), ("test", test_df)]:
    engineered = engineer_features(df)
    for col in engineered.columns:
        if col not in df.columns:
            df[col] = engineered[col]

# Numeric feature list
numeric_features = [
    'summary_length', 'summary_word_count',
    'severity_numeric', 'priority_numeric',
    'is_enhancement', 'is_high_severity',
    'open_hour', 'open_dayofweek', 'open_month',
]

structured_features = numeric_features
print(f"  Structured features ({len(structured_features)}): {structured_features}")

# TF-IDF (500 features, bigrams, stop_words, min_df=2 -- matching S2 config)
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

# Combine: structured + TF-IDF
X_train = hstack([csr_matrix(X_train_struct), X_train_tfidf])
X_cal = hstack([csr_matrix(X_cal_struct), X_cal_tfidf])
X_test = hstack([csr_matrix(X_test_struct), X_test_tfidf])

y_train = train_df["label"].values
y_cal = cal_df["label"].values
y_test = test_df["label"].values

# ---- Train XGBoost ----
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

# ---- Predictions ----
xgb_cal_preds = xgb.predict(X_cal)
xgb_cal_probs = xgb.predict_proba(X_cal)
xgb_test_preds = xgb.predict(X_test)
xgb_test_probs = xgb.predict_proba(X_test)

xgb_cal_acc = accuracy_score(y_cal, xgb_cal_preds)
xgb_test_acc = accuracy_score(y_test, xgb_test_preds)
xgb_test_f1_macro = f1_score(y_test, xgb_test_preds, average='macro', zero_division=0)
xgb_test_f1_weighted = f1_score(y_test, xgb_test_preds, average='weighted', zero_division=0)

print(f"\n=== XGBoost Results ===")
print(f"Cal accuracy:     {xgb_cal_acc:.4f}")
print(f"Test accuracy:    {xgb_test_acc:.4f}")
print(f"Test F1 (macro):  {xgb_test_f1_macro:.4f}")
print(f"Test F1 (weighted): {xgb_test_f1_weighted:.4f}")

# Majority baseline
majority_class = np.bincount(y_train).argmax()
majority_acc = (y_test == majority_class).mean()
print(f"Majority baseline (class {majority_class}={inv_map[majority_class]}): {majority_acc:.4f}")
print(f"Lift over majority: +{xgb_test_acc - majority_acc:.4f}")

# Per-class report
target_names = [inv_map[i] for i in range(num_classes)]
report = classification_report(y_test, xgb_test_preds, target_names=target_names, digits=3, zero_division=0)
print(f"\nPer-class report:\n{report}")

# ---- Save predictions ----
np.save(OUTPUT_DIR / "xgb_cal_preds.npy", xgb_cal_preds)
np.save(OUTPUT_DIR / "xgb_cal_probs.npy", xgb_cal_probs)
np.save(OUTPUT_DIR / "xgb_cal_labels.npy", y_cal)
np.save(OUTPUT_DIR / "xgb_test_preds.npy", xgb_test_preds)
np.save(OUTPUT_DIR / "xgb_test_probs.npy", xgb_test_probs)
np.save(OUTPUT_DIR / "xgb_test_labels.npy", y_test)

print(f"\n[OK] Saved XGBoost predictions to {OUTPUT_DIR}/")
print(f"  xgb_cal_probs.npy shape: {xgb_cal_probs.shape}")
print(f"  xgb_test_probs.npy shape: {xgb_test_probs.shape}")

# ---- Save results summary ----
results = {
    "model": "XGBoost",
    "num_classes": num_classes,
    "train_size": len(train_df),
    "cal_size": len(cal_df),
    "test_size": len(test_df),
    "n_features": X_train.shape[1],
    "best_iteration": int(xgb.best_iteration),
    "cal_accuracy": float(xgb_cal_acc),
    "test_accuracy": float(xgb_test_acc),
    "test_f1_macro": float(xgb_test_f1_macro),
    "test_f1_weighted": float(xgb_test_f1_weighted),
    "majority_baseline": float(majority_acc),
    "lift_over_majority": float(xgb_test_acc - majority_acc),
}
json.dump(results, open(OUTPUT_DIR / "xgb_results.json", "w"), indent=2)
print(f"[OK] Saved xgb_results.json")
