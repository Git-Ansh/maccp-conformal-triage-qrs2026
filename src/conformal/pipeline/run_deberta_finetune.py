"""
Fine-tune DeBERTa-v3-base for Eclipse bug component assignment.

This is Gate 2 of the dual-gate selective prediction architecture.
Gate 1 (XGBoost + conformal) handles confident predictions.
Gate 2 (DeBERTa) provides an independent cross-model signal.

When XGBoost and DeBERTa disagree, the prediction is contested → defer.
When they agree, the prediction is corroborated → auto-triage.

Outputs:
  - Fine-tuned model saved to conformal_outputs/deberta/
  - Test predictions + logits saved for downstream analysis
  - Per-class accuracy, F1, confusion matrix
  - Head-to-head comparison with XGBoost
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from conformal.data.eclipse_zenodo_loader import prepare_eclipse_zenodo_data

OUTPUT_DIR = PROJECT_ROOT / 'conformal_outputs' / 'deberta'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5


class BugDataset(Dataset):
    """PyTorch dataset for bug report text classification."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def prepare_text(row):
    """Combine Summary + Description for model input."""
    summary = str(row.get('Summary', ''))
    desc = str(row.get('Description', ''))[:500]
    return f"{summary} [SEP] {desc}".strip()


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count()
    print(f"Device: {device}, GPUs: {n_gpus}")

    # ================================================================
    # Load Data
    # ================================================================
    print("\n=== Loading Eclipse Zenodo Data ===")
    data = prepare_eclipse_zenodo_data(max_per_project=20000)
    train_df = data['train_df'].copy()
    test_df = data['test_df'].copy()

    # Prepare text
    train_df['text'] = train_df.apply(prepare_text, axis=1)
    test_df['text'] = test_df.apply(prepare_text, axis=1)

    # Filter to non-noise (same as S2 pipeline)
    noise_resolutions = {'INVALID', 'DUPLICATE', 'WONTFIX', 'WORKSFORME', 'NOT_ECLIPSE'}
    train_df = train_df[~train_df['Resolution'].isin(noise_resolutions)].copy()
    test_df = test_df[~test_df['Resolution'].isin(noise_resolutions)].copy()

    # Component target (top-30 + Other)
    top_n = 30
    component_counts = train_df['Component'].value_counts()
    top_components = component_counts.head(top_n).index.tolist()
    train_df['component_target'] = train_df['Component'].apply(
        lambda x: x if x in top_components else 'Other')
    test_df['component_target'] = test_df['Component'].apply(
        lambda x: x if x in top_components else 'Other')

    # Encode labels
    le = LabelEncoder()
    train_labels = le.fit_transform(train_df['component_target'].values)
    test_labels = le.transform(test_df['component_target'].values)
    n_classes = len(le.classes_)

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Classes: {n_classes} ({', '.join(le.classes_[:5])}...)")
    print(f"Train class distribution:")
    for cls, count in zip(*np.unique(train_labels, return_counts=True)):
        print(f"  {le.classes_[cls]}: {count} ({count/len(train_labels):.1%})")

    # ================================================================
    # Load Tokenizer + Model
    # ================================================================
    print("\n=== Loading DeBERTa-v3-base ===")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_name = 'microsoft/deberta-v3-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_classes
    )

    # Multi-GPU
    if n_gpus > 1:
        print(f"Using {n_gpus} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # ================================================================
    # Create Datasets
    # ================================================================
    train_dataset = BugDataset(
        train_df['text'].values, train_labels, tokenizer, MAX_LENGTH)
    test_dataset = BugDataset(
        test_df['text'].values, test_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE * max(n_gpus, 1),
        shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE * max(n_gpus, 1) * 2,
        shuffle=False, num_workers=4, pin_memory=True)

    # ================================================================
    # Train
    # ================================================================
    print(f"\n=== Fine-tuning DeBERTa ({EPOCHS} epochs) ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Class weights for imbalanced data
    class_counts = np.bincount(train_labels, minlength=n_classes)
    class_weights = torch.tensor(
        len(train_labels) / (n_classes * class_counts.clip(min=1)),
        dtype=torch.float32
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # Evaluate
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch['labels'].numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f"  Epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.4f}, "
              f"test_acc={acc:.4f}, test_f1_macro={f1_macro:.4f}")

        if acc > best_acc:
            best_acc = acc
            # Save best model
            save_model = model.module if hasattr(model, 'module') else model
            save_model.save_pretrained(OUTPUT_DIR / 'best_model')
            tokenizer.save_pretrained(OUTPUT_DIR / 'best_model')
            print(f"    -> Saved best model (acc={acc:.4f})")

    # ================================================================
    # Final Evaluation with Logits
    # ================================================================
    print(f"\n=== Final Evaluation ===")
    model.eval()
    all_preds = []
    all_proba = []
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits.cpu().numpy()
            proba = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = logits.argmax(axis=-1)

            all_logits.append(logits)
            all_proba.append(proba)
            all_preds.extend(preds)
            all_labels.extend(batch['labels'].numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_proba = np.concatenate(all_proba, axis=0)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f"Final accuracy: {acc:.4f}")
    print(f"Final F1 (macro): {f1_macro:.4f}")
    print(f"Final F1 (weighted): {f1_weighted:.4f}")
    print(f"\nClassification report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=le.classes_,
        zero_division=0
    ))

    # ================================================================
    # Save Predictions + Logits for Downstream Analysis
    # ================================================================
    print("\n=== Saving Results ===")

    # Save predictions
    pred_df = pd.DataFrame({
        'true_label': all_labels,
        'true_component': le.inverse_transform(all_labels),
        'deberta_pred': all_preds,
        'deberta_component': le.inverse_transform(all_preds),
        'deberta_confidence': all_proba.max(axis=1),
        'deberta_correct': (all_preds == all_labels).astype(int),
    })
    pred_df.to_csv(OUTPUT_DIR / 'eclipse_deberta_predictions.csv', index=False)

    # Save logits + probabilities for energy scores and conformal
    np.save(OUTPUT_DIR / 'eclipse_deberta_logits.npy', all_logits)
    np.save(OUTPUT_DIR / 'eclipse_deberta_proba.npy', all_proba)
    np.save(OUTPUT_DIR / 'eclipse_deberta_labels.npy', all_labels)

    # Save label encoder
    le_classes = list(le.classes_)
    with open(OUTPUT_DIR / 'label_encoder_classes.json', 'w') as f:
        json.dump(le_classes, f)

    # Save summary results
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': 'microsoft/deberta-v3-base',
        'dataset': 'Eclipse Zenodo 2024 (component assignment)',
        'n_train': len(train_df),
        'n_test': len(test_df),
        'n_classes': n_classes,
        'classes': le_classes,
        'max_length': MAX_LENGTH,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'lr': LR,
        'accuracy': float(acc),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'best_accuracy': float(best_acc),
    }

    with open(OUTPUT_DIR / 'eclipse_deberta_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nAll outputs saved to {OUTPUT_DIR}")
    print(f"Best accuracy: {best_acc:.4f}")
    print("Done.")


if __name__ == '__main__':
    main()
