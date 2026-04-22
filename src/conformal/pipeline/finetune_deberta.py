"""
finetune_deberta.py — Fine-tune DeBERTa-v3-base for multi-class bug triage

This is the go/no-go experiment. If accuracy < 70%, the paper is in trouble.

Usage (single GPU):
    python finetune_deberta.py \
        --data_dir /path/to/processed/ \
        --output_dir /path/to/models/deberta_eclipse/ \
        --max_length 512 \
        --batch_size 16 \
        --epochs 5 \
        --lr 2e-5

Usage (4x H100 via torchrun):
    torchrun --nproc_per_node=4 finetune_deberta.py \
        --data_dir /path/to/processed/ \
        --output_dir /path/to/models/deberta_eclipse/ \
        --max_length 512 \
        --batch_size 16 \
        --epochs 5 \
        --lr 2e-5 \
        --distributed

SLURM (TamIA HPC):
    sbatch --gres=gpu:4 --mem=128G --time=02:00:00 \
        --wrap="torchrun --nproc_per_node=4 finetune_deberta.py ..."

Expected runtime on 4x H100 with 60K samples: ~25-40 minutes.
Expected runtime on 1x H100: ~60-90 minutes.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, 
    confusion_matrix, top_k_accuracy_score
)
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)


# ─── Dataset ───

class BugReportDataset(Dataset):
    """Tokenized bug report dataset for DeBERTa."""
    
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─── Class weights ───

def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """
    Inverse frequency weights, capped at 10x to prevent rare-class explosion.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)  # avoid division by zero
    weights = len(labels) / (num_classes * counts)
    weights = np.minimum(weights, 10.0)  # cap at 10x
    return torch.tensor(weights, dtype=torch.float32)


# ─── Training ───

def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits.float(), labels)
        loss.backward()
        
        # Gradient clipping — important for DeBERTa stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item() * labels.size(0)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {total_loss/total:.4f} | Acc: {correct/total:.4f}")
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, num_classes):
    """
    Full evaluation returning predictions, probabilities, and metrics.
    Probabilities are needed for conformal prediction downstream.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits_f32 = outputs.logits.float()
        loss = criterion(logits_f32, labels)

        total_loss += loss.item() * labels.size(0)
        probs = torch.softmax(logits_f32, dim=-1)
        preds = probs.argmax(dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    
    # Top-3 accuracy — relevant for conformal set evaluation
    if num_classes > 3:
        top3_acc = top_k_accuracy_score(all_labels, all_probs, k=3, labels=range(num_classes))
    else:
        top3_acc = acc
    
    return {
        "loss": total_loss / len(all_labels),
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "top3_accuracy": top3_acc,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory with train/cal/test.parquet + label_mapping.json")
    parser.add_argument("--output_dir", required=True, help="Where to save model + predictions")
    parser.add_argument("--model_name", default="microsoft/deberta-v3-base")
    parser.add_argument("--max_length", type=int, default=512, help="Max token length. 256=fast, 512=better accuracy")
    parser.add_argument("--batch_size", type=int, default=16, help="Per-GPU batch size")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--distributed", action="store_true", help="Use DistributedDataParallel")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # ─── Setup ───
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device setup
    if args.distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        is_main = local_rank == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True
    
    if is_main:
        print(f"Device: {device}")
        print(f"Model: {args.model_name}")
        print(f"Max length: {args.max_length}")
        print(f"Batch size: {args.batch_size} per GPU")
    
    # ─── Load data ───
    label_map = json.load(open(data_dir / "label_mapping.json"))
    num_classes = len(label_map)
    if is_main:
        print(f"Classes: {num_classes}")
    
    train_df = pd.read_parquet(data_dir / "train.parquet")
    cal_df = pd.read_parquet(data_dir / "cal.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    
    if is_main:
        print(f"Train: {len(train_df):,} | Cal: {len(cal_df):,} | Test: {len(test_df):,}")
    
    # ─── Tokenizer ───
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # ─── Datasets ───
    train_dataset = BugReportDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    cal_dataset = BugReportDataset(
        texts=cal_df["text"].tolist(),
        labels=cal_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    test_dataset = BugReportDataset(
        texts=test_df["text"].tolist(),
        labels=test_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    
    # ─── DataLoaders ───
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
    cal_loader = DataLoader(
        cal_dataset, batch_size=args.batch_size * 2,  # eval can use bigger batches
        shuffle=False, num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=4, pin_memory=True,
    )
    
    # ─── Model ───
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_classes,
        problem_type="single_label_classification",
        torch_dtype=torch.float32,  # Force fp32 — H100 may default to fp16/bf16
    )
    model.to(device)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], find_unused_parameters=False
        )
    
    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {total_params/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")
    
    # ─── Loss with class weights ───
    class_weights = compute_class_weights(
        train_df["label"].tolist(), num_classes
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    if is_main:
        print(f"Class weight range: [{class_weights.min():.2f}, {class_weights.max():.2f}]")
    
    # ─── Optimizer + Scheduler ───
    # Separate weight decay for bias and LayerNorm (standard practice)
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    if is_main:
        print(f"Total steps: {total_steps}, Warmup: {warmup_steps}")
    
    # ─── Training loop with early stopping ───
    best_cal_acc = 0.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, epoch
        )
        
        # Evaluate on calibration set (NOT test — avoid data leakage)
        raw_model = model.module if args.distributed else model
        cal_results = evaluate(raw_model, cal_loader, criterion, device, num_classes)
        
        epoch_time = time.time() - start_time
        
        if is_main:
            print(f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.0f}s)")
            print(f"  Train — Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            print(f"  Cal   — Loss: {cal_results['loss']:.4f} | Acc: {cal_results['accuracy']:.4f} | "
                  f"F1(macro): {cal_results['f1_macro']:.4f} | Top-3: {cal_results['top3_accuracy']:.4f}")
        
        # Early stopping on calibration accuracy
        if cal_results["accuracy"] > best_cal_acc:
            best_cal_acc = cal_results["accuracy"]
            patience_counter = 0
            if is_main:
                # Save best model
                save_model = model.module if args.distributed else model
                save_model.save_pretrained(output_dir / "best_model")
                tokenizer.save_pretrained(output_dir / "best_model")
                print(f"  [OK] Saved best model (cal acc: {best_cal_acc:.4f})")
        else:
            patience_counter += 1
            if is_main:
                print(f"  No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                if is_main:
                    print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # ─── Final evaluation on test set using best model ───
    if is_main:
        print("\n" + "="*60)
        print("FINAL EVALUATION ON TEST SET (using best model)")
        print("="*60)
        
        # Reload best model
        best_model = AutoModelForSequenceClassification.from_pretrained(
            output_dir / "best_model",
            torch_dtype=torch.float32,
        ).to(device)
        
        # No-weight criterion for fair test eval
        criterion_unweighted = nn.CrossEntropyLoss()
        
        # Evaluate on calibration (needed for conformal prediction)
        cal_results = evaluate(best_model, cal_loader, criterion_unweighted, device, num_classes)
        print(f"\nCalibration set:")
        print(f"  Accuracy: {cal_results['accuracy']:.4f}")
        print(f"  F1 (macro): {cal_results['f1_macro']:.4f}")
        print(f"  F1 (weighted): {cal_results['f1_weighted']:.4f}")
        
        # Evaluate on test
        test_results = evaluate(best_model, test_loader, criterion_unweighted, device, num_classes)
        print(f"\nTest set:")
        print(f"  Accuracy: {test_results['accuracy']:.4f}")
        print(f"  F1 (macro): {test_results['f1_macro']:.4f}")
        print(f"  F1 (weighted): {test_results['f1_weighted']:.4f}")
        print(f"  Top-3 accuracy: {test_results['top3_accuracy']:.4f}")
        
        # ─── GO/NO-GO CHECK ───
        acc = test_results["accuracy"]
        print(f"\n{'='*60}")
        if acc >= 0.75:
            print(f"[GO] Test accuracy {acc:.1%}. Conformal sets will be practical.")
        elif acc >= 0.70:
            print(f"[VIABLE] Test accuracy {acc:.1%}. Sets will be moderate (4-6).")
        elif acc >= 0.65:
            print(f"[BORDERLINE] Test accuracy {acc:.1%}. Consider fewer components.")
        else:
            print(f"[PROBLEM] Test accuracy {acc:.1%}. Need to regroup.")
        print(f"{'='*60}")
        
        # ─── Per-class classification report ───
        inv_map = {v: k for k, v in label_map.items()}
        target_names = [inv_map[i] for i in range(num_classes)]
        
        report = classification_report(
            test_results["labels"],
            test_results["predictions"],
            target_names=target_names,
            digits=3,
            zero_division=0,
        )
        print(f"\nPer-class report:\n{report}")
        
        # ─── Save predictions + probabilities (needed for conformal prediction) ───
        np.save(output_dir / "cal_probs.npy", cal_results["probabilities"])
        np.save(output_dir / "cal_labels.npy", cal_results["labels"])
        np.save(output_dir / "cal_preds.npy", cal_results["predictions"])
        
        np.save(output_dir / "test_probs.npy", test_results["probabilities"])
        np.save(output_dir / "test_labels.npy", test_results["labels"])
        np.save(output_dir / "test_preds.npy", test_results["predictions"])
        
        print(f"\n[OK] Saved calibration probabilities -> {output_dir}/cal_probs.npy")
        print(f"[OK] Saved test probabilities -> {output_dir}/test_probs.npy")
        print(f"  Shape: {test_results['probabilities'].shape}")
        print(f"  (These feed directly into MAPIE for conformal prediction)")
        
        # ─── Save results summary ───
        summary = {
            "model": args.model_name,
            "max_length": args.max_length,
            "num_classes": num_classes,
            "train_size": len(train_df),
            "epochs_trained": epoch + 1,
            "best_cal_accuracy": float(best_cal_acc),
            "test_accuracy": float(test_results["accuracy"]),
            "test_f1_macro": float(test_results["f1_macro"]),
            "test_f1_weighted": float(test_results["f1_weighted"]),
            "test_top3_accuracy": float(test_results["top3_accuracy"]),
        }
        json.dump(summary, open(output_dir / "results_summary.json", "w"), indent=2)
        print(f"[OK] Saved results_summary.json")
    
    if args.distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
