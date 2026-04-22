"""
Eclipse S0 Decomposed: Three sub-stages with all ML + DL models.

S0a: NOT_ECLIPSE Detection (binary classification)
  - Is this bug about Eclipse or something else?
  - Confident NOT_ECLIPSE -> auto-reject
  - Confident ECLIPSE -> forward to S0b
  - Uncertain -> defer to human

S0b: Duplicate Detection (similarity retrieval)
  - Has this bug been reported before?
  - Confident DUPLICATE -> auto-link
  - Confident UNIQUE -> forward to S0c
  - Uncertain -> defer to human with candidate match

S0c: Bug Report Quality Scoring
  - Is this a well-formed, actionable bug report?
  - High quality -> forward to S1
  - Low quality -> defer to human (likely INVALID)
  - Medium quality -> forward but flagged

Models tested per sub-stage:
  ML: XGBoost+TF-IDF, XGBoost+SBERT, RandomForest+TF-IDF, LR+TF-IDF
  ML Ensemble: VotingClassifier(XGB+RF+LR) with TF-IDF and SBERT
  DL: TextCNN, SBERT-Classifier head

Also runs monolithic S0 (current approach) for comparison.

Usage:
    python src/conformal/pipeline/run_eclipse_s0_experiment.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from conformal.data.eclipse_zenodo_loader import prepare_eclipse_zenodo_data

OUTPUT_DIR = PROJECT_ROOT / 'conformal_outputs' / 'eclipse' / 's0_experiment'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUTPUT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# =========================================================================
# Data Preparation
# =========================================================================

def prepare_data(data):
    """Prepare all S0 sub-task targets and features."""
    train_df = data['train_df'].copy()
    test_df = data['test_df'].copy()

    # --- Targets ---
    # S0a: NOT_ECLIPSE detection
    train_df['is_not_eclipse'] = (train_df['Resolution'] == 'NOT_ECLIPSE').astype(int)
    test_df['is_not_eclipse'] = (test_df['Resolution'] == 'NOT_ECLIPSE').astype(int)

    # S0c: INVALID detection (narrower than full noise)
    train_df['is_invalid'] = (train_df['Resolution'] == 'INVALID').astype(int)
    test_df['is_invalid'] = (test_df['Resolution'] == 'INVALID').astype(int)

    # Broad noise (all 5 types — current monolithic S0)
    noise_resolutions = {'INVALID', 'DUPLICATE', 'WONTFIX', 'WORKSFORME', 'NOT_ECLIPSE'}
    train_df['is_noise'] = train_df['Resolution'].isin(noise_resolutions).astype(int)
    test_df['is_noise'] = test_df['Resolution'].isin(noise_resolutions).astype(int)

    # S0b: Duplicate detection
    train_df['is_duplicate'] = (train_df['Resolution'] == 'DUPLICATE').astype(int)
    test_df['is_duplicate'] = (test_df['Resolution'] == 'DUPLICATE').astype(int)

    # S0c: Quality features (rule-based)
    for df in [train_df, test_df]:
        text = df['Summary'].fillna('') + ' ' + df['Description'].fillna('').str[:500]
        df['text'] = text.str.strip()
        desc = df['Description'].fillna('')
        df['has_steps'] = desc.str.contains(r'(?:steps?\s*to\s*reproduce|1\.\s|step\s*1)', case=False, regex=True).astype(int)
        df['has_stacktrace'] = desc.str.contains(r'(?:Exception|at\s+org\.|Caused\s+by|stacktrace|traceback)', case=False, regex=True).astype(int)
        df['has_expected_actual'] = desc.str.contains(r'(?:expected|actual|should\s+be|instead\s+of)', case=False, regex=True).astype(int)
        df['has_version'] = desc.str.contains(r'(?:\d+\.\d+\.\d+|version\s+\d|eclipse\s+\d|build\s+\d)', case=False, regex=True).astype(int)
        df['desc_length'] = desc.str.len()
        df['desc_word_count'] = desc.str.split().str.len().fillna(0).astype(int)
        df['has_code_ref'] = desc.str.contains(r'(?:org\.eclipse\.|java\.|javax\.)', case=False, regex=True).astype(int)
        df['summary_length'] = df['Summary'].fillna('').str.len()

    # Quality score (simple additive)
    for df in [train_df, test_df]:
        df['quality_score'] = (
            df['has_steps'] +
            df['has_stacktrace'] +
            df['has_expected_actual'] +
            df['has_version'] +
            (df['desc_length'] > 200).astype(int) +
            df['has_code_ref']
        )

    # Numeric features (from loader)
    numeric_feats = [c for c in data['numeric_features'] if c in train_df.columns]
    cat_feats = [c for c in data['categorical_features'] if c in train_df.columns]

    # Encode categoricals
    for col in cat_feats:
        le = LabelEncoder()
        train_df[col + '_enc'] = le.fit_transform(train_df[col].astype(str).fillna('unknown'))
        vals = test_df[col].astype(str).fillna('unknown')
        known = set(le.classes_)
        vals = vals.apply(lambda x: x if x in known else le.classes_[0])
        test_df[col + '_enc'] = le.transform(vals)

    base_feat_cols = numeric_feats + [c + '_enc' for c in cat_feats]
    quality_feat_cols = ['has_steps', 'has_stacktrace', 'has_expected_actual',
                         'has_version', 'desc_length', 'desc_word_count',
                         'has_code_ref', 'summary_length', 'quality_score']

    stats = {
        'n_train': len(train_df),
        'n_test': len(test_df),
        'resolutions_train': dict(Counter(train_df['Resolution'])),
        'resolutions_test': dict(Counter(test_df['Resolution'])),
        'not_eclipse_rate_test': float(test_df['is_not_eclipse'].mean()),
        'duplicate_rate_test': float(test_df['is_duplicate'].mean()),
        'invalid_rate_test': float(test_df['is_invalid'].mean()),
        'noise_rate_test': float(test_df['is_noise'].mean()),
    }

    return train_df, test_df, base_feat_cols, quality_feat_cols, stats


# =========================================================================
# Feature Builders
# =========================================================================

def build_tfidf_features(train_text, test_text, max_features=500):
    """Build TF-IDF feature matrices."""
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english',
                             ngram_range=(1, 2), min_df=2)
    train_tfidf = tfidf.fit_transform(train_text).toarray()
    test_tfidf = tfidf.transform(test_text).toarray()
    return train_tfidf, test_tfidf, tfidf


def build_sbert_features(train_text, test_text):
    """Build sentence-transformer embedding features."""
    from sentence_transformers import SentenceTransformer
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"    SBERT device: {device}")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    print(f"    Encoding {len(train_text)} train texts...")
    train_emb = model.encode(train_text.tolist(), batch_size=512, show_progress_bar=True)
    print(f"    Encoding {len(test_text)} test texts...")
    test_emb = model.encode(test_text.tolist(), batch_size=512, show_progress_bar=True)
    return train_emb, test_emb


def build_combined_features(train_df, test_df, feat_cols, text_features='tfidf', sbert_cache=None):
    """Build combined tabular + text features."""
    train_tab = train_df[feat_cols].fillna(0).values.astype(np.float32)
    test_tab = test_df[feat_cols].fillna(0).values.astype(np.float32)

    if text_features == 'tfidf':
        train_text, test_text, _ = build_tfidf_features(train_df['text'], test_df['text'])
        train_X = np.hstack([train_tab, train_text])
        test_X = np.hstack([test_tab, test_text])
    elif text_features == 'sbert':
        if sbert_cache is not None:
            train_text, test_text = sbert_cache
        else:
            train_text, test_text = build_sbert_features(train_df['text'], test_df['text'])
        train_X = np.hstack([train_tab, train_text])
        test_X = np.hstack([test_tab, test_text])
    elif text_features == 'none':
        train_X = train_tab
        test_X = test_tab
    else:
        raise ValueError(f"Unknown text_features: {text_features}")

    return train_X, test_X


# =========================================================================
# ML Models
# =========================================================================

def get_xgboost():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_SEED,
                          eval_metric='logloss', n_jobs=-1, tree_method='hist',
                          device='cuda')


def run_ml_model(model, train_X, train_y, test_X, test_y, name, calibrate=True):
    """Train, calibrate, predict, evaluate a single ML model."""
    print(f"    [{name}] Training...")
    if calibrate:
        cal = CalibratedClassifierCV(model, method='isotonic', cv=3)
        cal.fit(train_X, train_y)
        proba = cal.predict_proba(test_X)[:, 1]
    else:
        model.fit(train_X, train_y)
        proba = model.predict_proba(test_X)[:, 1]

    pred = (proba >= 0.5).astype(int)
    return evaluate(test_y, pred, proba, name)


def run_ensemble(train_X, train_y, test_X, test_y, name='Ensemble'):
    """Train ensemble of XGB+RF+LR with calibration."""
    from xgboost import XGBClassifier
    print(f"    [{name}] Training...")
    ens = VotingClassifier(estimators=[
        ('xgb', XGBClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_SEED,
                               eval_metric='logloss', n_jobs=-1, tree_method='hist', device='cuda')),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_SEED,
                                       n_jobs=-1, class_weight='balanced')),
        ('lr', make_pipeline(StandardScaler(), LogisticRegression(random_state=RANDOM_SEED,
                                                                    max_iter=1000, class_weight='balanced'))),
    ], voting='soft')
    cal = CalibratedClassifierCV(ens, method='isotonic', cv=3)
    cal.fit(train_X, train_y)
    proba = cal.predict_proba(test_X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return evaluate(test_y, pred, proba, name)


# =========================================================================
# DL Models
# =========================================================================

def run_textcnn(train_df, test_df, target_col, name='TextCNN'):
    """1D CNN on tokenized text."""
    print(f"    [{name}] Training...")
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from collections import Counter as Ctr

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    word_counts = Ctr()
    for text in train_df['text']:
        word_counts.update(str(text).lower().split())

    vocab = {w: i + 2 for i, (w, c) in enumerate(word_counts.most_common(15000))}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    max_len = 256

    def text_to_ids(text):
        words = str(text).lower().split()[:max_len]
        ids = [vocab.get(w, 1) for w in words]
        return ids + [0] * (max_len - len(ids))

    train_ids = np.array([text_to_ids(t) for t in train_df['text']])
    test_ids = np.array([text_to_ids(t) for t in test_df['text']])
    train_y = train_df[target_col].values
    test_y = test_df[target_col].values

    class TextCNN(nn.Module):
        def __init__(self, vocab_size, embed_dim=128, num_filters=128, filter_sizes=(2, 3, 4, 5)):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes])
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(num_filters * len(filter_sizes), 1)

        def forward(self, x):
            x = self.embedding(x).permute(0, 2, 1)
            conv_outs = [torch.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
            x = torch.cat(conv_outs, dim=1)
            x = self.dropout(x)
            return self.fc(x).squeeze(1)

    model = TextCNN(len(vocab) + 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_weight = torch.tensor([(train_y == 0).sum() / max((train_y == 1).sum(), 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(
        TensorDataset(torch.LongTensor(train_ids), torch.FloatTensor(train_y)),
        batch_size=512, shuffle=True
    )

    model.train()
    for epoch in range(15):
        total_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"      Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        probas = []
        for i in range(0, len(test_ids), 1024):
            batch = torch.LongTensor(test_ids[i:i+1024]).to(device)
            probas.append(torch.sigmoid(model(batch)).cpu().numpy())
        proba = np.concatenate(probas)

    pred = (proba >= 0.5).astype(int)
    return evaluate(test_y, pred, proba, name)


def run_sbert_classifier(train_emb, test_emb, train_y, test_y, name='SBERT-Clf'):
    """Classification head on frozen SBERT embeddings."""
    print(f"    [{name}] Training...")
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class Head(nn.Module):
        def __init__(self, dim=384):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 1))

        def forward(self, x):
            return self.net(x).squeeze(1)

    model = Head().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pos_weight = torch.tensor([(train_y == 0).sum() / max((train_y == 1).sum(), 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(train_emb), torch.FloatTensor(train_y)),
        batch_size=512, shuffle=True
    )

    model.train()
    for epoch in range(30):
        total_loss = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(torch.FloatTensor(test_emb).to(device))).cpu().numpy()

    pred = (proba >= 0.5).astype(int)
    return evaluate(test_y, pred, proba, name)


# =========================================================================
# S0b: Duplicate Detection (Retrieval)
# =========================================================================

def run_duplicate_detection(train_df, test_df, train_emb, test_emb):
    """Duplicate detection via nearest-neighbor retrieval."""
    print("\n  [S0b: Duplicate Detection]")

    test_y = test_df['is_duplicate'].values
    dup_rate = test_y.mean()
    print(f"    Duplicate rate in test: {dup_rate:.1%}")

    # For each test bug, find most similar training bug
    # If similarity > threshold, flag as potential duplicate
    nn = NearestNeighbors(n_neighbors=5, metric='cosine')
    nn.fit(train_emb)
    distances, indices = nn.kneighbors(test_emb)

    # distance=0 means identical, distance=2 means opposite
    # similarity = 1 - distance
    top1_sim = 1 - distances[:, 0]

    results = []
    for threshold in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        pred = (top1_sim >= threshold).astype(int)
        prec = precision_score(test_y, pred, zero_division=0)
        rec = recall_score(test_y, pred, zero_division=0)
        f1 = f1_score(test_y, pred, zero_division=0)
        n_flagged = pred.sum()

        results.append({
            'threshold': threshold,
            'n_flagged': int(n_flagged),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
        })
        print(f"    sim>={threshold:.2f}: flagged={n_flagged}, prec={prec:.1%}, rec={rec:.1%}, F1={f1:.3f}")

    # Also compute AUC using similarity as score
    try:
        auc = roc_auc_score(test_y, top1_sim)
    except:
        auc = 0.5
    print(f"    AUC (similarity as score): {auc:.4f}")

    return {'retrieval_results': results, 'auc': float(auc), 'duplicate_rate': float(dup_rate)}


# =========================================================================
# Evaluation
# =========================================================================

def evaluate(y_true, y_pred, y_proba, model_name):
    """Compute binary classification metrics with confidence gating."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except:
        auc = 0.5

    # Coverage-accuracy at different confidence thresholds
    coverage_curve = []
    for t in [0.50, 0.60, 0.70, 0.80, 0.90]:
        confident = np.maximum(y_proba, 1 - y_proba) >= t
        if confident.sum() > 0:
            cov = confident.mean()
            conf_pred = (y_proba[confident] >= 0.5).astype(int)
            conf_acc = accuracy_score(y_true[confident], conf_pred)
            coverage_curve.append({'threshold': t, 'coverage': float(cov), 'accuracy': float(conf_acc)})

    print(f"    {model_name}: Acc={acc:.1%}, Prec={prec:.1%}, Rec={rec:.1%}, F1={f1:.3f}, AUC={auc:.4f}")

    return {
        'model': model_name,
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'auc': float(auc),
        'coverage_curve': coverage_curve,
    }


# =========================================================================
# Run All Models for One Task
# =========================================================================

def run_all_models(train_df, test_df, feat_cols, target_col, task_name, sbert_cache=None):
    """Run all ML + DL models on a single binary task."""
    print(f"\n{'='*60}")
    print(f"  {task_name}")
    print(f"{'='*60}")

    train_y = train_df[target_col].values
    test_y = test_df[target_col].values
    pos_rate = test_y.mean()
    majority_acc = max(pos_rate, 1 - pos_rate)
    print(f"  Positive rate: {pos_rate:.1%}, Majority baseline: {majority_acc:.1%}")

    results = []

    # --- ML Models with TF-IDF ---
    print("\n  ML Models (TF-IDF features):")
    train_X_tfidf, test_X_tfidf = build_combined_features(train_df, test_df, feat_cols, 'tfidf')

    r = run_ml_model(get_xgboost(), train_X_tfidf, train_y, test_X_tfidf, test_y, 'XGB+TF-IDF')
    results.append(r)

    r = run_ml_model(
        RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_SEED,
                                n_jobs=-1, class_weight='balanced'),
        train_X_tfidf, train_y, test_X_tfidf, test_y, 'RF+TF-IDF')
    results.append(r)

    r = run_ml_model(
        make_pipeline(StandardScaler(), LogisticRegression(random_state=RANDOM_SEED,
                                                            max_iter=1000, class_weight='balanced')),
        train_X_tfidf, train_y, test_X_tfidf, test_y, 'LR+TF-IDF', calibrate=False)
    results.append(r)

    # TF-IDF Ensemble
    r = run_ensemble(train_X_tfidf, train_y, test_X_tfidf, test_y, 'Ensemble+TF-IDF')
    results.append(r)

    # --- ML Models with SBERT ---
    print("\n  ML Models (SBERT features):")
    if sbert_cache is None:
        sbert_cache = build_sbert_features(train_df['text'], test_df['text'])
    train_X_sbert, test_X_sbert = build_combined_features(train_df, test_df, feat_cols, 'sbert',
                                                            sbert_cache=sbert_cache)

    r = run_ml_model(get_xgboost(), train_X_sbert, train_y, test_X_sbert, test_y, 'XGB+SBERT')
    results.append(r)

    r = run_ml_model(
        RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_SEED,
                                n_jobs=-1, class_weight='balanced'),
        train_X_sbert, train_y, test_X_sbert, test_y, 'RF+SBERT')
    results.append(r)

    r = run_ensemble(train_X_sbert, train_y, test_X_sbert, test_y, 'Ensemble+SBERT')
    results.append(r)

    # --- DL Models ---
    print("\n  DL Models:")
    r = run_textcnn(train_df, test_df, target_col, 'TextCNN')
    results.append(r)

    train_emb, test_emb = sbert_cache
    r = run_sbert_classifier(train_emb, test_emb, train_y, test_y, 'SBERT-Clf')
    results.append(r)

    return results, sbert_cache


# =========================================================================
# Plots
# =========================================================================

def plot_comparison(all_results, stats):
    """Generate comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    task_order = ['S0a: NOT_ECLIPSE', 'S0b: DUPLICATE (retrieval)', 'S0c: INVALID', 'Monolithic: ALL NOISE']
    colors = {'XGB+TF-IDF': '#1976d2', 'RF+TF-IDF': '#388e3c', 'LR+TF-IDF': '#f57c00',
              'Ensemble+TF-IDF': '#d32f2f', 'XGB+SBERT': '#1565c0', 'RF+SBERT': '#2e7d32',
              'Ensemble+SBERT': '#b71c1c', 'TextCNN': '#7b1fa2', 'SBERT-Clf': '#c2185b'}

    for idx, (task_name, ax) in enumerate(zip(task_order, axes.flatten())):
        if task_name not in all_results:
            ax.set_visible(False)
            continue

        results = all_results[task_name]
        if isinstance(results, dict) and 'retrieval_results' in results:
            # S0b retrieval plot
            rr = results['retrieval_results']
            thresholds = [r['threshold'] for r in rr]
            recalls = [r['recall'] for r in rr]
            precs = [r['precision'] for r in rr]
            ax.plot(thresholds, recalls, 'b-o', label='Recall')
            ax.plot(thresholds, precs, 'r-s', label='Precision')
            ax.set_xlabel('Similarity Threshold')
            ax.set_ylabel('Score')
            ax.set_title(f'{task_name}\nAUC={results["auc"]:.4f}')
            ax.legend()
        else:
            # Classification comparison
            models = [r['model'] for r in results]
            aucs = [r['auc'] for r in results]
            f1s = [r['f1'] for r in results]

            x = np.arange(len(models))
            width = 0.35
            ax.bar(x - width/2, aucs, width, label='AUC', alpha=0.8)
            ax.bar(x + width/2, f1s, width, label='F1', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=7)
            ax.set_title(task_name)
            ax.legend()
            ax.set_ylim(0, 1)

        ax.grid(True, alpha=0.3)

    fig.suptitle('Eclipse S0 Decomposed: Model Comparison', fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 's0_decomposed_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 's0_decomposed_comparison.png'}")


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 70)
    print("ECLIPSE S0 DECOMPOSED: FULL MODEL COMPARISON")
    print("=" * 70)

    # Load data
    print("\nLoading Eclipse Zenodo data...")
    data = prepare_eclipse_zenodo_data(max_per_project=20000)
    train_df, test_df, base_feat_cols, quality_feat_cols, stats = prepare_data(data)

    print(f"\n  Train: {stats['n_train']}, Test: {stats['n_test']}")
    print(f"  NOT_ECLIPSE rate: {stats['not_eclipse_rate_test']:.1%}")
    print(f"  DUPLICATE rate:   {stats['duplicate_rate_test']:.1%}")
    print(f"  INVALID rate:     {stats['invalid_rate_test']:.1%}")
    print(f"  ALL NOISE rate:   {stats['noise_rate_test']:.1%}")

    all_results = {}

    # Build SBERT embeddings once (reuse across all tasks)
    print("\n  Building SBERT embeddings (shared across all tasks)...")
    sbert_cache = build_sbert_features(train_df['text'], test_df['text'])

    # ================================================================
    # S0a: NOT_ECLIPSE Detection
    # ================================================================
    results_s0a, _ = run_all_models(
        train_df, test_df, base_feat_cols, 'is_not_eclipse',
        'S0a: NOT_ECLIPSE', sbert_cache=sbert_cache)
    all_results['S0a: NOT_ECLIPSE'] = results_s0a

    # ================================================================
    # S0b: Duplicate Detection (Retrieval)
    # ================================================================
    train_emb, test_emb = sbert_cache
    results_s0b = run_duplicate_detection(train_df, test_df, train_emb, test_emb)
    all_results['S0b: DUPLICATE (retrieval)'] = results_s0b

    # ================================================================
    # S0c: INVALID Detection (with quality features)
    # ================================================================
    all_feat_cols = base_feat_cols + quality_feat_cols
    results_s0c, _ = run_all_models(
        train_df, test_df, all_feat_cols, 'is_invalid',
        'S0c: INVALID', sbert_cache=sbert_cache)
    all_results['S0c: INVALID'] = results_s0c

    # ================================================================
    # Monolithic S0 (current approach, for comparison)
    # ================================================================
    results_mono, _ = run_all_models(
        train_df, test_df, base_feat_cols, 'is_noise',
        'Monolithic: ALL NOISE', sbert_cache=sbert_cache)
    all_results['Monolithic: ALL NOISE'] = results_mono

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for task_name, results in all_results.items():
        print(f"\n  {task_name}:")
        if isinstance(results, dict) and 'retrieval_results' in results:
            print(f"    Retrieval AUC: {results['auc']:.4f}")
            best = max(results['retrieval_results'], key=lambda r: r['f1'])
            print(f"    Best F1: {best['f1']:.3f} at threshold={best['threshold']}")
        else:
            print(f"    {'Model':<20} {'AUC':>7} {'Recall':>8} {'Prec':>8} {'F1':>7}")
            print(f"    {'-'*50}")
            for r in sorted(results, key=lambda x: -x['auc']):
                print(f"    {r['model']:<20} {r['auc']:>7.4f} {r['recall']:>8.1%} {r['precision']:>8.1%} {r['f1']:>7.3f}")

    # ================================================================
    # Save & Plot
    # ================================================================
    print("\nGenerating plots...")
    plot_comparison(all_results, stats)

    output = {
        'timestamp': datetime.now().isoformat(),
        'stats': stats,
        'results': {},
    }
    for k, v in all_results.items():
        if isinstance(v, list):
            output['results'][k] = v
        else:
            output['results'][k] = v

    path = OUTPUT_DIR / f's0_decomposed_{datetime.now():%Y%m%d_%H%M%S}.json'
    with open(path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {path}")
    print("Done.")


if __name__ == '__main__':
    main()
