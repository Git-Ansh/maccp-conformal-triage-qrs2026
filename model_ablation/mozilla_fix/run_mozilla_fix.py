"""
Mozilla Core XGBoost Feature Engineering + Hybrid MACCP Re-evaluation.

Problem: Mozilla XGBoost only 36.3% because product encoding is constant ("Core").
Solution: Engineer reporter/assignee profiles, keyword features, text structure features.
Target: 45-50% accuracy so hybrid MACCP Config C works on Mozilla.
"""

import sys, os, json, re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

np.random.seed(42)

SCRIPT_DIR = Path(__file__).parent.resolve()
ABLATION_DIR = SCRIPT_DIR.parent
DATA_DIR = ABLATION_DIR / "data" / "mozilla"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Add parent scripts for utils
sys.path.insert(0, str(ABLATION_DIR / "scripts"))

# ============================================================
# STEP 1: Load data
# ============================================================
print("=" * 70)
print("MOZILLA CORE XGBOOST FEATURE ENGINEERING")
print("=" * 70)

label_map = json.load(open(DATA_DIR / "label_mapping.json"))
n_classes = len(label_map)
inv_map = {v: k for k, v in label_map.items()}

train_df = pd.read_parquet(DATA_DIR / "train.parquet")
cal_df = pd.read_parquet(DATA_DIR / "cal.parquet")
test_df = pd.read_parquet(DATA_DIR / "test.parquet")

print(f"Train: {len(train_df)}, Cal: {len(cal_df)}, Test: {len(test_df)}, Classes: {n_classes}")

# Load raw data with Assignee and Number of Comments
import zipfile
PROJECT_ROOT = ABLATION_DIR.parent
dfs_raw = []
with zipfile.ZipFile(PROJECT_ROOT / "data" / "external" / "mozilla_firefox" / "buglist.zip") as z:
    for name in sorted(z.namelist()):
        if name.endswith('.csv'):
            with z.open(name) as f:
                dfs_raw.append(pd.read_csv(f, low_memory=False))
raw_all = pd.concat(dfs_raw, ignore_index=True)
raw_core = raw_all[raw_all['Product'] == 'Core'].copy()
raw_core.columns = raw_core.columns.str.lower().str.strip().str.replace(' ', '_')
print(f"Raw Core bugs: {len(raw_core)}")

# Join raw metadata to train/cal/test via bug_id matching
# The parquet 'text' column is the Summary. Match by text content.
# Actually, the parquet has creation_ts which we can use as bug_id proxy
# Let's match by Summary text (unique enough)
raw_core['summary_clean'] = raw_core['summary'].str.strip()

for split_name, split_df in [('train', train_df), ('cal', cal_df), ('test', test_df)]:
    # Extract summary from text (before [SEP])
    split_df['summary_clean'] = split_df['text'].str.split(' \\[SEP\\] ').str[0].str.strip()

# Create a lookup from summary to raw metadata
raw_lookup = raw_core.drop_duplicates(subset='summary_clean', keep='first').set_index('summary_clean')

for split_name, split_df in [('train', train_df), ('cal', cal_df), ('test', test_df)]:
    matched = split_df['summary_clean'].map(raw_lookup['assignee']).fillna('unknown')
    split_df['assignee'] = matched
    matched_comments = split_df['summary_clean'].map(raw_lookup['number_of_comments']).fillna(0)
    split_df['n_comments'] = matched_comments.astype(int)
    match_rate = (split_df['assignee'] != 'unknown').mean()
    print(f"  {split_name}: assignee match rate = {match_rate:.1%}")

# ============================================================
# STEP 2: Feature Engineering (from training data ONLY)
# ============================================================
print("\n=== Feature Engineering ===")

# --- Feature 1: Assignee component history ---
print("  Building assignee component profiles...")
assignee_profiles = {}
for assignee, group in train_df.groupby('assignee'):
    if assignee == 'unknown':
        continue
    profile = np.zeros(n_classes)
    for lbl in group['label'].values:
        profile[lbl] += 1
    total = profile.sum()
    if total > 0:
        profile = profile / total
    assignee_profiles[assignee] = profile
print(f"    {len(assignee_profiles)} unique assignees with profiles")

def get_assignee_features(assignee):
    if assignee in assignee_profiles:
        prof = assignee_profiles[assignee]
        entropy = -np.sum(prof[prof > 0] * np.log2(prof[prof > 0]))
        return np.concatenate([prof, [entropy, prof.max(), prof.sum()]])
    # Unknown: uniform + high entropy
    uniform = np.ones(n_classes) / n_classes
    return np.concatenate([uniform, [np.log2(n_classes), 1.0/n_classes, 0.0]])

# --- Feature 2: Keyword-component co-occurrence ---
print("  Building keyword-component matrix...")
from sklearn.feature_extraction.text import CountVectorizer

kw_vectorizer = CountVectorizer(max_features=200, stop_words='english', binary=True, min_df=5)
kw_matrix = kw_vectorizer.fit_transform(train_df['summary_clean'].values)
kw_profiles = np.zeros((200, n_classes))
for word_idx in range(200):
    mask = kw_matrix[:, word_idx].toarray().flatten() > 0
    if mask.sum() < 5:
        continue
    for comp_idx in range(n_classes):
        kw_profiles[word_idx, comp_idx] = (train_df.loc[mask, 'label'].values == comp_idx).mean()
print(f"    {kw_vectorizer.get_feature_names_out()[:5]}... ({len(kw_vectorizer.get_feature_names_out())} keywords)")

def get_keyword_features(summary):
    word_vec = kw_vectorizer.transform([summary]).toarray().flatten()
    present = np.where(word_vec > 0)[0]
    if len(present) == 0:
        return np.ones(n_classes) / n_classes
    agg = kw_profiles[present].mean(axis=0)
    total = agg.sum()
    if total > 0:
        agg = agg / total
    return agg

# --- Feature 3: Text structure features ---
print("  Building text structure features...")
crash_words = {'crash', 'assertion', 'abort', 'segfault', 'sigsegv'}
error_words = {'error', 'exception', 'fail', 'broken', 'failure'}
regression_words = {'regression', 'regress', 'regressed'}

def get_text_features(summary):
    words = summary.lower().split()
    word_set = set(words)
    return np.array([
        len(words),                                  # word count
        len(summary),                                # char count
        int(bool(word_set & crash_words)),           # has crash keyword
        int(bool(word_set & error_words)),           # has error keyword
        int(bool(word_set & regression_words)),      # has regression keyword
        int('::' in summary or bool(re.search(r'\w+\.\w+\(', summary))),  # has API reference
        int('/' in summary or '.cpp' in summary or '.js' in summary or '.h' in summary),  # has file path
        sum(1 for w in words if w.isupper() and len(w) > 1),  # uppercase words
        int(summary.startswith('[') or summary.startswith('(')),  # starts with bracket
    ], dtype=float)

TEXT_FEAT_NAMES = ['word_count', 'char_count', 'has_crash', 'has_error', 'has_regression',
                   'has_api_ref', 'has_file_path', 'n_uppercase', 'starts_bracket']

# --- Feature 4: Assignee severity interaction (no severity available, use n_comments instead) ---
print("  Building comment-based features...")
assignee_avg_comments = train_df.groupby('assignee')['n_comments'].mean().to_dict()

def get_comment_features(assignee, n_comments):
    avg = assignee_avg_comments.get(assignee, train_df['n_comments'].mean())
    return np.array([n_comments, avg, n_comments - avg], dtype=float)

# ============================================================
# STEP 3: Build feature matrices
# ============================================================
print("\n=== Building Feature Matrices ===")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

# TF-IDF (same as original: 500 features)
tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words='english', min_df=2, sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(train_df['text'].values)
X_cal_tfidf = tfidf.transform(cal_df['text'].values)
X_test_tfidf = tfidf.transform(test_df['text'].values)

def build_engineered_features(df):
    feats = []
    for _, row in df.iterrows():
        assignee_f = get_assignee_features(row['assignee'])         # n_classes + 3
        keyword_f = get_keyword_features(row['summary_clean'])      # n_classes
        text_f = get_text_features(row['summary_clean'])            # 9
        comment_f = get_comment_features(row['assignee'], row['n_comments'])  # 3
        feats.append(np.concatenate([assignee_f, keyword_f, text_f, comment_f]))
    return np.array(feats)

print("  Engineering train features...")
X_train_eng = build_engineered_features(train_df)
print("  Engineering cal features...")
X_cal_eng = build_engineered_features(cal_df)
print("  Engineering test features...")
X_test_eng = build_engineered_features(test_df)

n_eng = X_train_eng.shape[1]
print(f"  Engineered features: {n_eng} ({n_classes}+3 assignee + {n_classes} keyword + 9 text + 3 comment)")

# Scale engineered features
scaler = StandardScaler()
X_train_eng_s = scaler.fit_transform(X_train_eng)
X_cal_eng_s = scaler.transform(X_cal_eng)
X_test_eng_s = scaler.transform(X_test_eng)

# Combine
X_train = hstack([csr_matrix(X_train_eng_s), X_train_tfidf])
X_cal = hstack([csr_matrix(X_cal_eng_s), X_cal_tfidf])
X_test = hstack([csr_matrix(X_test_eng_s), X_test_tfidf])

y_train = train_df['label'].values
y_cal = cal_df['label'].values
y_test = test_df['label'].values

print(f"  Total features: {X_train.shape[1]} ({n_eng} engineered + {X_train_tfidf.shape[1]} TF-IDF)")

# ============================================================
# STEP 4: Train XGBoost
# ============================================================
print("\n=== Training XGBoost ===")
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

# Version A: same hyperparameters as paper
xgb_a = XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    tree_method='hist', early_stopping_rounds=30)
xgb_a.fit(X_train, y_train, eval_set=[(X_cal, y_cal)], verbose=50)
cal_acc_a = accuracy_score(y_cal, xgb_a.predict(X_cal))
test_acc_a = accuracy_score(y_test, xgb_a.predict(X_test))
print(f"Version A: cal={cal_acc_a:.4f}, test={test_acc_a:.4f}, best_iter={xgb_a.best_iteration}")

# Version B: more capacity
xgb_b = XGBClassifier(n_estimators=1000, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=3, random_state=42,
    n_jobs=-1, tree_method='hist', early_stopping_rounds=30)
xgb_b.fit(X_train, y_train, eval_set=[(X_cal, y_cal)], verbose=50)
cal_acc_b = accuracy_score(y_cal, xgb_b.predict(X_cal))
test_acc_b = accuracy_score(y_test, xgb_b.predict(X_test))
print(f"Version B: cal={cal_acc_b:.4f}, test={test_acc_b:.4f}, best_iter={xgb_b.best_iteration}")

# Pick best on cal
if cal_acc_b > cal_acc_a:
    best_xgb = xgb_b
    best_version = 'B'
    best_cal = cal_acc_b
    best_test = test_acc_b
else:
    best_xgb = xgb_a
    best_version = 'A'
    best_cal = cal_acc_a
    best_test = test_acc_a

print(f"\nBest: Version {best_version} (cal={best_cal:.4f}, test={best_test:.4f})")

# Save predictions
new_cal_preds = best_xgb.predict(X_cal)
new_cal_probs = best_xgb.predict_proba(X_cal)
new_test_preds = best_xgb.predict(X_test)
new_test_probs = best_xgb.predict_proba(X_test)

np.save(RESULTS_DIR / "new_xgb_cal_preds.npy", new_cal_preds)
np.save(RESULTS_DIR / "new_xgb_cal_probs.npy", new_cal_probs)
np.save(RESULTS_DIR / "new_xgb_test_preds.npy", new_test_preds)
np.save(RESULTS_DIR / "new_xgb_test_probs.npy", new_test_probs)

# ============================================================
# STEP 5: Feature importance
# ============================================================
print("\n=== Feature Importance (top 20 by gain) ===")
importance = best_xgb.get_booster().get_score(importance_type='gain')
sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
for feat, gain in sorted_imp[:20]:
    print(f"  {feat}: {gain:.1f}")

json.dump(dict(sorted_imp[:50]), open(RESULTS_DIR / "feature_importance.json", "w"), indent=2)

# ============================================================
# STEP 6: Evaluation + Hybrid MACCP
# ============================================================
print("\n=== Evaluation ===")

# Load old DeBERTa predictions (unchanged)
deb_cal_preds = np.load(DATA_DIR / "deberta_cal_preds.npy")
deb_cal_probs = np.load(DATA_DIR / "deberta_cal_probs.npy")
deb_test_preds = np.load(DATA_DIR / "deberta_test_preds.npy")
deb_test_probs = np.load(DATA_DIR / "deberta_test_probs.npy")

# Load old XGBoost predictions for comparison
old_test_preds = np.load(DATA_DIR / "xgb_test_preds.npy")
old_test_acc = accuracy_score(y_test, old_test_preds)

# New agreement
new_agree_cal = deb_cal_preds == new_cal_preds
new_agree_test = deb_test_preds == new_test_preds
old_agree_test = deb_test_preds == old_test_preds

deb_correct = deb_test_preds == y_test
new_xgb_correct = new_test_preds == y_test
old_xgb_correct = old_test_preds == y_test

print(f"Old XGBoost test acc: {old_test_acc:.4f}")
print(f"New XGBoost test acc: {best_test:.4f} (delta: {best_test - old_test_acc:+.4f})")
print(f"DeBERTa test acc: {accuracy_score(y_test, deb_test_preds):.4f}")
print(f"Old agreement rate: {old_agree_test.mean():.1%}")
print(f"New agreement rate: {new_agree_test.mean():.1%}")
print(f"Old disagree XGB acc: {old_xgb_correct[~old_agree_test].mean():.4f}")
print(f"New disagree XGB acc: {new_xgb_correct[~new_agree_test].mean():.4f}")
print(f"DeBERTa disagree acc (old): {deb_correct[~old_agree_test].mean():.4f}")
print(f"DeBERTa disagree acc (new): {deb_correct[~new_agree_test].mean():.4f}")

# Within-confidence-bin gaps
print("\n=== Within-Confidence-Bin Gaps (New XGBoost) ===")
deb_conf = deb_test_probs.max(axis=1)
bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]
for lo, hi in bins:
    mask = (deb_conf >= lo) & (deb_conf < hi)
    if mask.sum() < 10: continue
    a = new_agree_test[mask]
    c = deb_correct[mask]
    na, nd = a.sum(), (~a).sum()
    acc_a = c[a].mean() if na > 5 else float('nan')
    acc_d = c[~a].mean() if nd > 5 else float('nan')
    gap = acc_a - acc_d if not (np.isnan(acc_a) or np.isnan(acc_d)) else float('nan')
    print(f"  [{lo:.2f},{hi:.2f}): n={mask.sum():4d}, agree={na/mask.sum():.1%}, acc_agree={acc_a:.1%}, acc_disagree={acc_d:.1%}, gap={gap:+.1%}")

# ============================================================
# STEP 7: Hybrid MACCP
# ============================================================
print("\n=== Hybrid MACCP (New XGBoost) ===")

def raps_scores(probs, labels, lam=0.01, kreg=5):
    n = len(labels)
    scores = np.zeros(n)
    for i in range(n):
        si = np.argsort(-probs[i])
        cs = 0.0
        for rank, ci in enumerate(si):
            cs += probs[i, ci]
            pen = lam * max(0, rank + 1 - kreg)
            if ci == labels[i]:
                u = np.random.uniform(0, probs[i, ci] + pen)
                scores[i] = cs + pen - u
                break
    return scores

def conformal_q(scores, alpha):
    n = len(scores)
    return np.quantile(scores, min(np.ceil((n+1)*(1-alpha))/n, 1.0), method='higher')

def make_sets(probs, q, lam=0.01, kreg=5):
    n, k = probs.shape
    sets_list = []
    for i in range(n):
        si = np.argsort(-probs[i])
        cs = 0.0
        pset = []
        for rank, ci in enumerate(si):
            cs += probs[i, ci]
            pen = lam * max(0, rank + 1 - kreg)
            pset.append(int(ci))
            if cs + pen >= q: break
        sets_list.append(pset)
    return sets_list

configs = [
    ('A: DEB/DEB', deb_cal_probs, deb_cal_probs, deb_test_probs, deb_test_probs),
    ('B: newXGB/newXGB', new_cal_probs, new_cal_probs, new_test_probs, new_test_probs),
    ('C: DEB/newXGB', deb_cal_probs, new_cal_probs, deb_test_probs, new_test_probs),
    ('D: newXGB/DEB', new_cal_probs, deb_cal_probs, new_test_probs, deb_test_probs),
]

hybrid_results = {}
for alpha in [0.05, 0.10, 0.20]:
    print(f"\nalpha = {alpha}")
    print(f"{'Config':<22} | {'Agr.Cov':>7} | {'Agr.Mean':>8} | {'Agr.Sing%':>9} | {'Agr.S.Acc':>9} | {'Dis.Mean':>8} | {'Dis.Cov':>7}")
    print('-' * 90)

    for config_name, cal_agree_p, cal_disagree_p, test_agree_p, test_disagree_p in configs:
        np.random.seed(42)
        # Cal scores
        agree_cal_scores = raps_scores(cal_agree_p[new_agree_cal], y_cal[new_agree_cal])
        disagree_cal_scores = raps_scores(cal_disagree_p[~new_agree_cal], y_cal[~new_agree_cal])

        q_a = conformal_q(agree_cal_scores, alpha) if len(agree_cal_scores) > 5 else 1.0
        q_d = conformal_q(disagree_cal_scores, alpha) if len(disagree_cal_scores) > 5 else 1.0

        # Test sets
        agree_idx = np.where(new_agree_test)[0]
        disagree_idx = np.where(~new_agree_test)[0]

        a_sets = [make_sets(test_agree_p[i:i+1], q_a)[0] for i in agree_idx]
        d_sets = [make_sets(test_disagree_p[i:i+1], q_d)[0] for i in disagree_idx]

        a_labels = y_test[agree_idx]
        d_labels = y_test[disagree_idx]

        a_cov = np.mean([l in s for s, l in zip(a_sets, a_labels)]) if a_sets else 0
        a_sizes = [len(s) for s in a_sets]
        a_mean = np.mean(a_sizes) if a_sizes else 0
        a_singletons = [(s, l) for s, l in zip(a_sets, a_labels) if len(s) == 1]
        a_sing_rate = len(a_singletons) / len(a_sets) if a_sets else 0
        a_sing_acc = np.mean([l in s for s, l in a_singletons]) if a_singletons else 0

        d_cov = np.mean([l in s for s, l in zip(d_sets, d_labels)]) if d_sets else 0
        d_mean = np.mean([len(s) for s in d_sets]) if d_sets else 0

        print(f"{config_name:<22} | {a_cov*100:6.1f}% | {a_mean:8.2f} | {a_sing_rate*100:8.1f}% | {a_sing_acc*100:8.1f}% | {d_mean:8.2f} | {d_cov*100:6.1f}%")

        hybrid_results[f'{config_name}_alpha{alpha}'] = {
            'agree': {'n': len(agree_idx), 'coverage': float(a_cov), 'mean_set': float(a_mean), 'singleton_rate': float(a_sing_rate), 'singleton_acc': float(a_sing_acc)},
            'disagree': {'n': len(disagree_idx), 'coverage': float(d_cov), 'mean_set': float(d_mean)},
        }

# ============================================================
# STEP 8: Summary comparison
# ============================================================
print("\n" + "=" * 70)
print("MOZILLA CORE: OLD vs NEW XGBOOST")
print("=" * 70)

# Get old Config A and C at alpha=0.10 from saved results
old_hybrid = json.load(open(ABLATION_DIR / "results" / "hybrid" / "mozilla_hybrid.json"))

print(f"{'Metric':<30} | {'Old (36.3%)':<12} | {'New ({best_test:.1%})':<12} | {'Delta':<10}")
print("-" * 70)
print(f"{'Test accuracy':<30} | {old_test_acc:.1%}{'':>7} | {best_test:.1%}{'':>7} | {best_test - old_test_acc:+.1%}")
print(f"{'Agreement rate':<30} | {old_agree_test.mean():.1%}{'':>7} | {new_agree_test.mean():.1%}{'':>7} | {new_agree_test.mean() - old_agree_test.mean():+.1%}")
print(f"{'Agree accuracy':<30} | {deb_correct[old_agree_test].mean():.1%}{'':>7} | {deb_correct[new_agree_test].mean():.1%}{'':>7} |")
print(f"{'Disagree acc (XGB)':<30} | {old_xgb_correct[~old_agree_test].mean():.1%}{'':>7} | {new_xgb_correct[~new_agree_test].mean():.1%}{'':>7} |")
print(f"{'Disagree acc (DeBERTa)':<30} | {deb_correct[~old_agree_test].mean():.1%}{'':>7} | {deb_correct[~new_agree_test].mean():.1%}{'':>7} | (same model)")

# Config C comparisons
old_c = old_hybrid.get("C: DEB/XGB_alpha0.1", {})
new_c = hybrid_results.get("C: DEB/newXGB_alpha0.1", {})
if old_c and new_c:
    print(f"{'Hybrid C disagree mean':<30} | {old_c['disagree']['mean_set']:.2f}{'':>8} | {new_c['disagree']['mean_set']:.2f}{'':>8} |")
    print(f"{'Hybrid C disagree cov':<30} | {old_c['disagree']['coverage']*100:.1f}%{'':>7} | {new_c['disagree']['coverage']*100:.1f}%{'':>7} |")
    print(f"{'Hybrid C agree singletons':<30} | {old_c['agree']['singleton_rate']*100:.1f}%{'':>7} | {new_c['agree']['singleton_rate']*100:.1f}%{'':>7} |")

print(f"\n=== HYBRID WORKING? ===")
new_dis_xgb = new_xgb_correct[~new_agree_test].mean()
new_dis_deb = deb_correct[~new_agree_test].mean()
print(f"XGB disagree acc ({new_dis_xgb:.1%}) > DeBERTa disagree acc ({new_dis_deb:.1%})? {'YES' if new_dis_xgb > new_dis_deb else 'NO'}")
if new_c:
    old_a_dis = old_hybrid.get("A: DEB/DEB_alpha0.1", {}).get('disagree', {}).get('mean_set', 0)
    new_c_dis = new_c['disagree']['mean_set']
    print(f"Hybrid C disagree mean ({new_c_dis:.2f}) < Config A disagree mean ({old_a_dis:.2f})? {'YES' if new_c_dis < old_a_dis else 'NO'}")

# Save all results
all_results = {
    'old_test_acc': float(old_test_acc),
    'new_test_acc': float(best_test),
    'new_version': best_version,
    'new_best_iteration': int(best_xgb.best_iteration),
    'old_agreement_rate': float(old_agree_test.mean()),
    'new_agreement_rate': float(new_agree_test.mean()),
    'n_engineered_features': n_eng,
    'hybrid_results': hybrid_results,
}
json.dump(all_results, open(RESULTS_DIR / "mozilla_fix_results.json", "w"), indent=2, default=str)
print(f"\nResults saved to {RESULTS_DIR}/")
