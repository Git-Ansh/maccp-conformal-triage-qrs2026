"""
experiment3_finetune_deberta_large.py -- Fine-tune DeBERTa-v3-large for component prediction.

Uses HuggingFace Trainer (handles DDP properly) instead of custom training loop.

Model: microsoft/deberta-v3-large (304M params)
Training: lr=1e-5, batch=16, grad_accum=2, epochs=3, warmup=0.1, bf16=True
Eval: on cal set each epoch, save best by accuracy.

After training:
  - Run inference on cal+test sets, save probabilities as .npy
  - Compute accuracy, agreement with XGBoost, within-bin gaps, full MACCP (4 configs)

IMPORTANT: torch_dtype=torch.float32 for model loading (same fix as DeBERTa-base),
but bf16=True in TrainingArguments (Trainer handles the casting properly).

Usage:
    python experiment3_finetune_deberta_large.py --dataset eclipse
    python experiment3_finetune_deberta_large.py --dataset mozilla
    torchrun --nproc_per_node=4 experiment3_finetune_deberta_large.py --dataset eclipse
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from utils import (
    load_data,
    compute_agreement,
    run_maccp_pipeline,
    evaluate_prediction_sets,
    compute_raps_scores_batch,
    compute_conformal_quantile,
    compute_prediction_set,
    save_results,
    print_comparison_row,
    print_table_header,
)

# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 512
LEARNING_RATE = 1e-5
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1
SEED = 42

BASE_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(BASE_DIR, "data")


def get_output_dirs(dataset_name):
    model_dir = os.path.join(BASE_DIR, "models", "deberta_large", dataset_name)
    results_dir = os.path.join(BASE_DIR, "results", "deberta_large")
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints", "deberta_large", dataset_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    return model_dir, results_dir, ckpt_dir


def train_and_evaluate(dataset_name):
    """Fine-tune DeBERTa-v3-large and evaluate on cal+test sets."""
    import torch
    import pandas as pd
    from torch.utils.data import Dataset as TorchDataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    from sklearn.metrics import accuracy_score

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model_dir, results_dir, ckpt_dir = get_output_dirs(dataset_name)
    data = load_data(dataset_name, DATA_DIR)
    num_classes = data["num_classes"]

    # Load parquet data
    train_df = pd.read_parquet(os.path.join(DATA_DIR, dataset_name, "train.parquet"))
    cal_df = pd.read_parquet(os.path.join(DATA_DIR, dataset_name, "cal.parquet"))
    test_df = pd.read_parquet(os.path.join(DATA_DIR, dataset_name, "test.parquet"))

    print(f"Dataset: {dataset_name}")
    print(f"  Train: {len(train_df):,} | Cal: {len(cal_df):,} | Test: {len(test_df):,}")
    print(f"  Classes: {num_classes}")
    print(f"  Model: {MODEL_NAME}")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    class BugReportDataset(TorchDataset):
        def __init__(self, texts, labels):
            self.texts = list(texts)
            self.labels = list(labels)

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = tokenizer(
                self.texts[idx],
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    train_dataset = BugReportDataset(train_df["text"].tolist(), train_df["label"].tolist())
    cal_dataset = BugReportDataset(cal_df["text"].tolist(), cal_df["label"].tolist())
    test_dataset = BugReportDataset(test_df["text"].tolist(), test_df["label"].tolist())

    # ---- Model ----
    # IMPORTANT: load in float32 (same fix as DeBERTa-base)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        torch_dtype=torch.float32,
    )

    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ---- Training ----
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}

    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        bf16=True,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        dataloader_num_workers=4,
        seed=SEED,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=cal_dataset,
        compute_metrics=compute_metrics,
    )

    print("\n  Starting training...")
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.0f}s ({train_time/60:.1f}min)")

    # Save best model
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"  Model saved to: {model_dir}")

    # ---- Inference ----
    print("\n  Running inference on cal and test sets...")

    def run_inference(dataset):
        predictions = trainer.predict(dataset)
        logits = predictions.predictions
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
        preds = np.argmax(probs, axis=-1)
        return probs.astype(np.float32), preds

    cal_probs, cal_preds = run_inference(cal_dataset)
    test_probs, test_preds = run_inference(test_dataset)

    # Save probabilities
    for name, arr in [
        ("deberta_large_cal_probs", cal_probs),
        ("deberta_large_cal_preds", cal_preds),
        ("deberta_large_test_probs", test_probs),
        ("deberta_large_test_preds", test_preds),
    ]:
        fpath = os.path.join(DATA_DIR, dataset_name, f"{name}.npy")
        np.save(fpath, arr)
        print(f"    Saved: {fpath} (shape={arr.shape})")

    # ---- Evaluation ----
    cal_labels = data["cal_labels"]
    test_labels = data["test_labels"]

    cal_acc = (cal_preds == cal_labels).mean()
    test_acc = (test_preds == test_labels).mean()

    print(f"\n  DeBERTa-large cal accuracy:  {cal_acc:.4f}")
    print(f"  DeBERTa-large test accuracy: {test_acc:.4f}")

    # Compare with base model
    deb_base_acc = (data["deberta_test_preds"] == test_labels).mean()
    xgb_acc = (data["xgb_test_preds"] == test_labels).mean()
    print(f"  DeBERTa-base test accuracy:  {deb_base_acc:.4f}")
    print(f"  XGBoost test accuracy:       {xgb_acc:.4f}")
    print(f"  Improvement over base:       {test_acc - deb_base_acc:+.4f}")

    # Agreement with XGBoost
    cal_agree = compute_agreement(cal_preds, data["xgb_cal_preds"])
    test_agree = compute_agreement(test_preds, data["xgb_test_preds"])
    print(f"  Cal agreement (large-XGB):   {cal_agree.mean():.3f}")
    print(f"  Test agreement (large-XGB):  {test_agree.mean():.3f}")

    # Agreement with base DeBERTa
    cal_agree_base = compute_agreement(cal_preds, data["deberta_cal_preds"])
    test_agree_base = compute_agreement(test_preds, data["deberta_test_preds"])
    print(f"  Cal agreement (large-base):  {cal_agree_base.mean():.3f}")
    print(f"  Test agreement (large-base): {test_agree_base.mean():.3f}")

    # ---- Within-bin gap analysis ----
    print(f"\n  Within-bin gap analysis (DeBERTa-large confidence bins):")
    deb_confidence = test_probs.max(axis=1)
    deb_correct = test_preds == test_labels
    bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]

    print(f"    {'Bin':>12} | {'N':>6} | {'Acc':>6} | {'AgreeRate':>9} | {'AccAgree':>8} | {'AccDisag':>8} | {'Gap':>7}")
    print(f"    {'-'*12}-+-{'-'*6}-+-{'-'*6}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}")

    for lo, hi in bins:
        mask = (deb_confidence >= lo) & (deb_confidence < hi)
        n_bin = mask.sum()
        if n_bin < 10:
            continue
        bin_agree = test_agree[mask]
        bin_correct = deb_correct[mask]
        acc_bin = bin_correct.mean()
        agree_rate = bin_agree.mean()

        if bin_agree.sum() > 5:
            acc_a = bin_correct[bin_agree].mean()
        else:
            acc_a = float("nan")
        if (~bin_agree).sum() > 5:
            acc_d = bin_correct[~bin_agree].mean()
        else:
            acc_d = float("nan")
        gap = acc_a - acc_d if not (np.isnan(acc_a) or np.isnan(acc_d)) else float("nan")

        gap_str = f"{gap:+.3f}" if not np.isnan(gap) else "   N/A"
        acc_a_str = f"{acc_a:.3f}" if not np.isnan(acc_a) else "    N/A"
        acc_d_str = f"{acc_d:.3f}" if not np.isnan(acc_d) else "    N/A"
        print(f"    [{lo:.2f},{hi:.2f}) | {n_bin:6,} | {acc_bin:.3f} | {agree_rate:9.3f} | {acc_a_str:>8} | {acc_d_str:>8} | {gap_str:>7}")

    # ---- MACCP with 4 configs ----
    print(f"\n  MACCP results (DeBERTa-large):")

    # Agreement for MACCP conditioning: large vs XGB
    maccp_results = {}
    configs = {
        "A: Large/Large": ("large", "large"),
        "B: XGBoost/XGBoost": ("xgb", "xgb"),
        "C: Large/XGBoost": ("large", "xgb"),
        "D: XGBoost/Large": ("xgb", "large"),
    }

    def get_model_probs(model_key, split):
        if model_key == "large":
            return cal_probs if split == "cal" else test_probs
        elif model_key == "xgb":
            return data[f"xgb_{split}_probs"]
        else:
            return data[f"deberta_{split}_probs"]

    for alpha in [0.05, 0.10, 0.20]:
        print(f"\n    alpha = {alpha}")
        print_table_header()
        for config_name, (agree_model, disagree_model) in configs.items():
            result = run_maccp_pipeline(
                cal_probs_agree_model=get_model_probs(agree_model, "cal"),
                cal_probs_disagree_model=get_model_probs(disagree_model, "cal"),
                test_probs_agree_model=get_model_probs(agree_model, "test"),
                test_probs_disagree_model=get_model_probs(disagree_model, "test"),
                cal_labels=cal_labels,
                test_labels=test_labels,
                cal_agreement=cal_agree,
                test_agreement=test_agree,
                alpha=alpha,
            )
            key = f"{config_name}_alpha{alpha}"
            maccp_results[key] = result
            print_comparison_row(config_name, dataset_name, result["overall"])

    # ---- Save all results ----
    results = {
        "dataset": dataset_name,
        "model": MODEL_NAME,
        "num_classes": num_classes,
        "train_time_seconds": train_time,
        "cal_accuracy": float(cal_acc),
        "test_accuracy": float(test_acc),
        "deberta_base_test_accuracy": float(deb_base_acc),
        "xgb_test_accuracy": float(xgb_acc),
        "improvement_over_base": float(test_acc - deb_base_acc),
        "agreement_large_xgb_cal": float(cal_agree.mean()),
        "agreement_large_xgb_test": float(test_agree.mean()),
        "agreement_large_base_cal": float(cal_agree_base.mean()),
        "agreement_large_base_test": float(test_agree_base.mean()),
        "maccp": maccp_results,
    }

    save_results(results, os.path.join(results_dir, f"{dataset_name}_deberta_large.json"))
    print(f"\n  All results saved to {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa-v3-large")
    parser.add_argument(
        "--dataset", required=True, choices=["eclipse", "mozilla"],
        help="Dataset to fine-tune on",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("EXPERIMENT 3: FINE-TUNE DeBERTa-v3-LARGE")
    print("=" * 70)

    train_and_evaluate(args.dataset)
    print("\nDone.")


if __name__ == "__main__":
    main()
