"""
experiment5_inference_finetuned.py -- Standalone inference for fine-tuned models.

Load a fine-tuned model from ../models/{model}/{dataset}/ and run inference
on cal+test sets.  Saves probabilities as .npy files and computes accuracy,
agreement, within-bin gaps, and MACCP metrics.

Usage:
    python experiment5_inference_finetuned.py --model deberta_large --dataset eclipse
    python experiment5_inference_finetuned.py --model qwen32b --dataset mozilla
"""

import argparse
import json
import os
import sys
import time

import numpy as np

np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from utils import (
    load_data,
    compute_agreement,
    run_maccp_pipeline,
    save_results,
    print_comparison_row,
    print_table_header,
)

BASE_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
MAX_LENGTH = 512
SEED = 42


def run_deberta_large_inference(dataset_name, model_path):
    """Run inference with fine-tuned DeBERTa-v3-large."""
    import torch
    import pandas as pd
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.utils.data import Dataset as TorchDataset, DataLoader

    torch.manual_seed(SEED)

    data = load_data(dataset_name, DATA_DIR)
    num_classes = data["num_classes"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_classes,
        torch_dtype=torch.float32,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"  Model loaded from: {model_path}")
    print(f"  Device: {device}")
    print(f"  Classes: {num_classes}")

    class TextDataset(TorchDataset):
        def __init__(self, texts):
            self.texts = list(texts)

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
            }

    results = {}
    for split_name in ["cal", "test"]:
        df = pd.read_parquet(os.path.join(DATA_DIR, dataset_name, f"{split_name}.parquet"))
        labels = data[f"{split_name}_labels"]

        dataset = TextDataset(df["text"].tolist())
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

        all_probs = []
        t0 = time.time()
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

        probs = np.concatenate(all_probs, axis=0).astype(np.float32)
        preds = np.argmax(probs, axis=-1)
        elapsed = time.time() - t0
        acc = (preds == labels).mean()

        print(f"  {split_name}: {len(df):,} examples, {elapsed:.0f}s, accuracy={acc:.4f}")

        results[f"{split_name}_probs"] = probs
        results[f"{split_name}_preds"] = preds

    return results


def run_qwen_inference(dataset_name, model_path):
    """Run inference with fine-tuned Qwen2.5-32B-Instruct (LoRA adapter)."""
    import torch
    import pandas as pd
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    # Import the probability extraction functions
    from experiment4_finetune_qwen import (
        build_chat_messages,
        check_first_token_uniqueness,
        extract_probs_first_token,
        extract_probs_multi_token,
    )

    torch.manual_seed(SEED)

    data = load_data(dataset_name, DATA_DIR)
    num_classes = data["num_classes"]
    label_mapping = data["label_mapping"]

    # Load base model + LoRA adapter
    print(f"  Loading base model: Qwen/Qwen2.5-32B-Instruct")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-32B-Instruct",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"  Loading LoRA adapter from: {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"  Device: {device}")
    print(f"  Classes: {num_classes}")

    # Check first-token uniqueness
    first_tok_unique, _, conflicts = check_first_token_uniqueness(label_mapping, tokenizer)
    extract_fn = extract_probs_first_token if first_tok_unique else extract_probs_multi_token
    method_name = "first-token" if first_tok_unique else "multi-token"
    print(f"  Extraction method: {method_name}")

    results = {}
    for split_name in ["cal", "test"]:
        df = pd.read_parquet(os.path.join(DATA_DIR, dataset_name, f"{split_name}.parquet"))
        labels = data[f"{split_name}_labels"]
        n = len(df)

        probs = np.zeros((n, num_classes), dtype=np.float32)
        preds = np.zeros(n, dtype=np.int64)

        t0 = time.time()
        for i in range(n):
            text = str(df.iloc[i]["text"])
            messages = build_chat_messages(text, dataset_name, label_mapping, label=None)
            probs[i] = extract_fn(model, tokenizer, messages, label_mapping, device)
            preds[i] = np.argmax(probs[i])

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                acc_so_far = (preds[:i+1] == labels[:i+1]).mean()
                print(
                    f"    {split_name}: {i+1}/{n} "
                    f"({elapsed:.0f}s, acc={acc_so_far:.4f})"
                )

        elapsed = time.time() - t0
        acc = (preds == labels).mean()
        print(f"  {split_name}: {n:,} examples, {elapsed:.0f}s, accuracy={acc:.4f}")

        results[f"{split_name}_probs"] = probs
        results[f"{split_name}_preds"] = preds

    return results


def run_evaluation(dataset_name, model_name, inference_results):
    """Compute agreement, within-bin gaps, and MACCP for inference results."""
    data = load_data(dataset_name, DATA_DIR)
    cal_labels = data["cal_labels"]
    test_labels = data["test_labels"]

    cal_probs = inference_results["cal_probs"]
    cal_preds = inference_results["cal_preds"]
    test_probs = inference_results["test_probs"]
    test_preds = inference_results["test_preds"]

    cal_acc = (cal_preds == cal_labels).mean()
    test_acc = (test_preds == test_labels).mean()
    deb_base_acc = (data["deberta_test_preds"] == test_labels).mean()
    xgb_acc = (data["xgb_test_preds"] == test_labels).mean()

    print(f"\n  Accuracy summary:")
    print(f"    {model_name} cal:       {cal_acc:.4f}")
    print(f"    {model_name} test:      {test_acc:.4f}")
    print(f"    DeBERTa-base test: {deb_base_acc:.4f}")
    print(f"    XGBoost test:      {xgb_acc:.4f}")

    # Agreement
    cal_agree_xgb = compute_agreement(cal_preds, data["xgb_cal_preds"])
    test_agree_xgb = compute_agreement(test_preds, data["xgb_test_preds"])
    cal_agree_deb = compute_agreement(cal_preds, data["deberta_cal_preds"])
    test_agree_deb = compute_agreement(test_preds, data["deberta_test_preds"])

    print(f"\n  Agreement:")
    print(f"    {model_name}-XGB:    cal={cal_agree_xgb.mean():.3f}, test={test_agree_xgb.mean():.3f}")
    print(f"    {model_name}-DeBERTa: cal={cal_agree_deb.mean():.3f}, test={test_agree_deb.mean():.3f}")

    # Within-bin gap analysis
    print(f"\n  Within-bin gap analysis:")
    confidence = test_probs.max(axis=1)
    correct = test_preds == test_labels
    bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]

    print(f"    {'Bin':>12} | {'N':>6} | {'Acc':>6} | {'AgreeRate':>9} | {'AccAgree':>8} | {'AccDisag':>8} | {'Gap':>7}")
    print(f"    {'-'*12}-+-{'-'*6}-+-{'-'*6}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}")

    for lo, hi in bins:
        mask = (confidence >= lo) & (confidence < hi)
        n_bin = mask.sum()
        if n_bin < 10:
            continue
        bin_agree = test_agree_xgb[mask]
        bin_correct = correct[mask]
        acc_bin = bin_correct.mean()
        agree_rate = bin_agree.mean()

        acc_a = bin_correct[bin_agree].mean() if bin_agree.sum() > 5 else float("nan")
        acc_d = bin_correct[~bin_agree].mean() if (~bin_agree).sum() > 5 else float("nan")
        gap = acc_a - acc_d if not (np.isnan(acc_a) or np.isnan(acc_d)) else float("nan")

        gap_str = f"{gap:+.3f}" if not np.isnan(gap) else "   N/A"
        acc_a_str = f"{acc_a:.3f}" if not np.isnan(acc_a) else "    N/A"
        acc_d_str = f"{acc_d:.3f}" if not np.isnan(acc_d) else "    N/A"
        print(f"    [{lo:.2f},{hi:.2f}) | {n_bin:6,} | {acc_bin:.3f} | {agree_rate:9.3f} | {acc_a_str:>8} | {acc_d_str:>8} | {gap_str:>7}")

    # MACCP
    print(f"\n  MACCP results:")
    maccp_results = {}

    for alpha in [0.05, 0.10, 0.20]:
        print(f"\n    alpha = {alpha}")
        print_table_header()

        def get_probs(key, split):
            if key == "finetuned":
                return cal_probs if split == "cal" else test_probs
            return data[f"xgb_{split}_probs"]

        configs = {
            f"A: {model_name}/{model_name}": ("finetuned", "finetuned"),
            "B: XGBoost/XGBoost": ("xgb", "xgb"),
            f"C: {model_name}/XGBoost": ("finetuned", "xgb"),
            f"D: XGBoost/{model_name}": ("xgb", "finetuned"),
        }

        for config_name, (agree_model, disagree_model) in configs.items():
            result = run_maccp_pipeline(
                cal_probs_agree_model=get_probs(agree_model, "cal"),
                cal_probs_disagree_model=get_probs(disagree_model, "cal"),
                test_probs_agree_model=get_probs(agree_model, "test"),
                test_probs_disagree_model=get_probs(disagree_model, "test"),
                cal_labels=cal_labels,
                test_labels=test_labels,
                cal_agreement=cal_agree_xgb,
                test_agreement=test_agree_xgb,
                alpha=alpha,
            )
            key = f"{config_name}_alpha{alpha}"
            maccp_results[key] = result
            print_comparison_row(config_name, dataset_name, result["overall"])

    return {
        "model": model_name,
        "dataset": dataset_name,
        "cal_accuracy": float(cal_acc),
        "test_accuracy": float(test_acc),
        "deberta_base_test_accuracy": float(deb_base_acc),
        "xgb_test_accuracy": float(xgb_acc),
        "agreement_xgb_cal": float(cal_agree_xgb.mean()),
        "agreement_xgb_test": float(test_agree_xgb.mean()),
        "agreement_deberta_cal": float(cal_agree_deb.mean()),
        "agreement_deberta_test": float(test_agree_deb.mean()),
        "maccp": maccp_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Standalone inference for fine-tuned models")
    parser.add_argument(
        "--model", required=True, choices=["deberta_large", "qwen32b"],
        help="Model to run inference with",
    )
    parser.add_argument(
        "--dataset", required=True, choices=["eclipse", "mozilla"],
        help="Dataset to evaluate on",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"EXPERIMENT 5: INFERENCE ({args.model} on {args.dataset})")
    print("=" * 70)

    model_path = os.path.join(BASE_DIR, "models", args.model, args.dataset)
    if not os.path.exists(model_path):
        print(f"  ERROR: Model not found at {model_path}")
        print(f"  Run experiment3 or experiment4 first to fine-tune the model.")
        sys.exit(1)

    # Run inference
    if args.model == "deberta_large":
        inference_results = run_deberta_large_inference(args.dataset, model_path)
    else:
        inference_results = run_qwen_inference(args.dataset, model_path)

    # Save probabilities
    prefix = "deberta_large" if args.model == "deberta_large" else "qwen32b"
    for split in ["cal", "test"]:
        for kind in ["probs", "preds"]:
            arr = inference_results[f"{split}_{kind}"]
            fpath = os.path.join(DATA_DIR, args.dataset, f"{prefix}_{split}_{kind}.npy")
            np.save(fpath, arr)
            print(f"  Saved: {fpath} (shape={arr.shape})")

    # Run evaluation
    eval_results = run_evaluation(args.dataset, args.model, inference_results)

    # Save results
    results_dir = os.path.join(BASE_DIR, "results", args.model)
    os.makedirs(results_dir, exist_ok=True)
    save_results(eval_results, os.path.join(results_dir, f"{args.dataset}_inference.json"))

    print("\nDone.")


if __name__ == "__main__":
    main()
