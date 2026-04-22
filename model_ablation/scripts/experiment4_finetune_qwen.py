"""
experiment4_finetune_qwen.py -- Fine-tune Qwen2.5-32B-Instruct with LoRA.

Uses peft LoRA for parameter-efficient fine-tuning of a 32B parameter model.
Training uses chat-format messages with system/user/assistant roles.

LoRA: r=16, alpha=32, targets=[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
Training: bf16, DeepSpeed ZeRO Stage 2 with CPU offloading, gradient checkpointing
          2 epochs, lr=2e-4, batch=2, grad_accum=8

After training: extract probabilities via first-token logits method, with
fallback to multi-token scoring if component names share first tokens.

Usage:
    python experiment4_finetune_qwen.py --dataset eclipse
    python experiment4_finetune_qwen.py --dataset mozilla
    deepspeed --num_gpus=4 experiment4_finetune_qwen.py --dataset eclipse
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
    save_results,
    print_comparison_row,
    print_table_header,
)

# ============================================================
# Configuration
# ============================================================
MODEL_NAME = os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen2.5-32B-Instruct")
MAX_LENGTH = 512
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
NUM_EPOCHS = 2
WARMUP_RATIO = 0.1
SEED = 42

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

BASE_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
DS_CONFIG = os.path.join(SCRIPT_DIR, "ds_config_zero2.json")


def get_output_dirs(dataset_name):
    model_dir = os.path.join(BASE_DIR, "models", "qwen32b", dataset_name)
    results_dir = os.path.join(BASE_DIR, "results", "qwen32b")
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints", "qwen32b", dataset_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    return model_dir, results_dir, ckpt_dir


def build_chat_messages(text, dataset_name, label_mapping, label=None):
    """Build chat messages for Qwen fine-tuning.

    Parameters
    ----------
    text : str
        Bug report text.
    dataset_name : str
        'eclipse' or 'mozilla'.
    label_mapping : dict
        Component name -> index mapping.
    label : int or None
        Ground-truth label (for training). None for inference.

    Returns
    -------
    list of dict
        Chat messages with system/user/assistant roles.
    """
    inv_mapping = {v: k for k, v in label_mapping.items()}
    component_names = sorted(label_mapping.keys())
    component_list = ", ".join(component_names)

    if dataset_name == "eclipse":
        system_msg = (
            f"You are a bug triage expert for Eclipse IDE. "
            f"Classify bug reports into one of these {len(component_names)} components: "
            f"{component_list}. "
            f"Respond with ONLY the component name."
        )
        # Parse title and description from [SEP] format
        parts = text.split(" [SEP] ", 1)
        title = parts[0] if len(parts) > 0 else text
        description = parts[1][:500] if len(parts) > 1 else ""
        user_msg = f"Title: {title}"
        if description:
            user_msg += f"\nDescription: {description}"
    else:
        system_msg = (
            f"You are a bug triage expert for Mozilla Firefox. "
            f"Classify bug reports into one of these {len(component_names)} components: "
            f"{component_list}. "
            f"Respond with ONLY the component name."
        )
        user_msg = f"Title: {text}"

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    if label is not None:
        assistant_msg = inv_mapping.get(label, "unknown")
        messages.append({"role": "assistant", "content": assistant_msg})

    return messages


def check_first_token_uniqueness(label_mapping, tokenizer):
    """Check if all component names have unique first tokens.

    Returns
    -------
    bool
        True if all first tokens are unique (can use first-token logits method).
    dict
        Mapping from first token ID to component index.
    """
    inv_mapping = {v: k for k, v in label_mapping.items()}
    first_tokens = {}
    conflicts = []

    for idx, name in inv_mapping.items():
        tokens = tokenizer.encode(name, add_special_tokens=False)
        first_tok = tokens[0] if tokens else None
        if first_tok in first_tokens:
            conflicts.append((name, inv_mapping[first_tokens[first_tok]]))
        else:
            first_tokens[first_tok] = idx

    token_to_class = {tok: idx for tok, idx in first_tokens.items() if tok is not None}
    is_unique = len(conflicts) == 0

    return is_unique, token_to_class, conflicts


def extract_probs_first_token(model, tokenizer, messages, label_mapping, device):
    """Extract probabilities via first-token logits method.

    For each class, look at the logit of the first token of that class name
    in the model's output distribution.

    Parameters
    ----------
    model : PreTrainedModel
        The fine-tuned model.
    tokenizer : PreTrainedTokenizer
        The tokenizer.
    messages : list of dict
        Chat messages (system + user only, no assistant).
    label_mapping : dict
        Component name -> index.
    device : torch.device
        Device for inference.

    Returns
    -------
    ndarray, shape (num_classes,)
        Probability distribution over classes.
    """
    import torch

    # Tokenize the prompt (without assistant response)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # Last token logits (next token prediction)
        logits = outputs.logits[:, -1, :]  # shape: (1, vocab_size)

    # Map each class to its first token
    inv_mapping = {v: k for k, v in label_mapping.items()}
    num_classes = len(label_mapping)
    class_logits = np.zeros(num_classes)

    for idx in range(num_classes):
        name = inv_mapping[idx]
        tokens = tokenizer.encode(name, add_special_tokens=False)
        if tokens:
            class_logits[idx] = logits[0, tokens[0]].float().cpu().item()
        else:
            class_logits[idx] = -1e9

    # Softmax over class logits
    class_logits = class_logits - class_logits.max()  # numerical stability
    probs = np.exp(class_logits) / np.exp(class_logits).sum()
    return probs.astype(np.float32)


def extract_probs_multi_token(model, tokenizer, messages, label_mapping, device):
    """Extract probabilities via multi-token scoring (fallback method).

    Scores each component name by computing the log-probability of its
    full token sequence under the model.

    Parameters
    ----------
    model, tokenizer, messages, label_mapping, device : same as above.

    Returns
    -------
    ndarray, shape (num_classes,)
    """
    import torch

    inv_mapping = {v: k for k, v in label_mapping.items()}
    num_classes = len(label_mapping)
    class_scores = np.zeros(num_classes)

    # Tokenize the base prompt
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    base_ids = tokenizer.encode(prompt, add_special_tokens=False)

    for idx in range(num_classes):
        name = inv_mapping[idx]
        name_ids = tokenizer.encode(name, add_special_tokens=False)

        # Full sequence: prompt + class name tokens
        full_ids = base_ids + name_ids
        input_ids = torch.tensor([full_ids[-MAX_LENGTH:]], device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits  # (1, seq_len, vocab_size)

        # Sum log-probs of the class name tokens
        log_prob = 0.0
        start = len(full_ids) - len(name_ids) - 1
        if start < 0:
            start = 0
        for t, token_id in enumerate(name_ids):
            pos = start + t
            if pos >= logits.shape[1]:
                break
            token_logits = logits[0, pos, :].float()
            token_logprobs = torch.log_softmax(token_logits, dim=-1)
            log_prob += token_logprobs[token_id].item()

        class_scores[idx] = log_prob

    # Convert log-probs to probabilities
    class_scores = class_scores - class_scores.max()
    probs = np.exp(class_scores) / np.exp(class_scores).sum()
    return probs.astype(np.float32)


def train_and_evaluate(dataset_name):
    """Fine-tune Qwen2.5-32B-Instruct with LoRA and evaluate."""
    import torch
    import pandas as pd
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model_dir, results_dir, ckpt_dir = get_output_dirs(dataset_name)
    data = load_data(dataset_name, DATA_DIR)
    num_classes = data["num_classes"]
    label_mapping = data["label_mapping"]
    inv_mapping = data["inv_mapping"]

    # Load parquet data
    train_df = pd.read_parquet(os.path.join(DATA_DIR, dataset_name, "train.parquet"))
    cal_df = pd.read_parquet(os.path.join(DATA_DIR, dataset_name, "cal.parquet"))
    test_df = pd.read_parquet(os.path.join(DATA_DIR, dataset_name, "test.parquet"))

    print(f"Dataset: {dataset_name}")
    print(f"  Train: {len(train_df):,} | Cal: {len(cal_df):,} | Test: {len(test_df):,}")
    print(f"  Classes: {num_classes}")
    print(f"  Model: {MODEL_NAME}")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check first-token uniqueness
    first_tok_unique, token_to_class, conflicts = check_first_token_uniqueness(
        label_mapping, tokenizer
    )
    if first_tok_unique:
        print("  First-token method: AVAILABLE (all component names have unique first tokens)")
    else:
        print(f"  First-token method: UNAVAILABLE ({len(conflicts)} conflicts)")
        for c1, c2 in conflicts[:5]:
            print(f"    Conflict: '{c1}' vs '{c2}'")
        print("  Will use multi-token scoring fallback")

    # ---- Prepare training data ----
    print("\n  Preparing chat-format training data...")

    def format_example(row, include_label=True):
        text = str(row["text"])
        label = int(row["label"]) if include_label else None
        messages = build_chat_messages(text, dataset_name, label_mapping, label)
        formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        return formatted

    train_texts = [format_example(train_df.iloc[i], include_label=True) for i in range(len(train_df))]
    print(f"  Formatted {len(train_texts):,} training examples")

    # Tokenize
    from torch.utils.data import Dataset as TorchDataset

    class ChatDataset(TorchDataset):
        def __init__(self, formatted_texts, max_length):
            self.formatted_texts = formatted_texts
            self.max_length = max_length

        def __len__(self):
            return len(self.formatted_texts)

        def __getitem__(self, idx):
            encoding = tokenizer(
                self.formatted_texts[idx],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            # For causal LM, labels = input_ids (shifted internally by the model)
            labels = input_ids.clone()
            # Mask padding tokens in labels
            labels[attention_mask == 0] = -100
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

    train_dataset = ChatDataset(train_texts, MAX_LENGTH)

    # ---- Model + LoRA ----
    print("\n  Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Base model params: {total_params:,}")

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    # ---- Training ----
    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        bf16=True,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        dataloader_num_workers=2,
        seed=SEED,
        report_to="none",
        remove_unused_columns=False,
        deepspeed=DS_CONFIG if os.path.exists(DS_CONFIG) else None,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print("\n  Starting LoRA fine-tuning...")
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.0f}s ({train_time/60:.1f}min)")

    # Save LoRA adapter
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"  LoRA adapter saved to: {model_dir}")

    # ---- Inference ----
    print("\n  Running inference on cal and test sets...")
    device = next(model.parameters()).device
    model.eval()

    extract_fn = extract_probs_first_token if first_tok_unique else extract_probs_multi_token

    def run_inference_split(df, labels, split_name):
        n = len(df)
        probs_all = np.zeros((n, num_classes), dtype=np.float32)
        preds_all = np.zeros(n, dtype=np.int64)

        t0_inf = time.time()
        for i in range(n):
            text = str(df.iloc[i]["text"])
            messages = build_chat_messages(text, dataset_name, label_mapping, label=None)
            probs_all[i] = extract_fn(model, tokenizer, messages, label_mapping, device)
            preds_all[i] = np.argmax(probs_all[i])

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0_inf
                acc_so_far = (preds_all[:i+1] == labels[:i+1]).mean()
                print(
                    f"    {split_name}: {i+1}/{n} "
                    f"({elapsed:.0f}s, {(i+1)/elapsed:.1f}/s, "
                    f"acc={acc_so_far:.4f})"
                )

        elapsed = time.time() - t0_inf
        print(f"    {split_name} inference: {n} in {elapsed:.0f}s ({n/max(elapsed,1):.1f}/s)")
        return probs_all, preds_all

    cal_probs, cal_preds = run_inference_split(cal_df, data["cal_labels"], "cal")
    test_probs, test_preds = run_inference_split(test_df, data["test_labels"], "test")

    # Save probabilities
    for name, arr in [
        ("qwen32b_cal_probs", cal_probs),
        ("qwen32b_cal_preds", cal_preds),
        ("qwen32b_test_probs", test_probs),
        ("qwen32b_test_preds", test_preds),
    ]:
        fpath = os.path.join(DATA_DIR, dataset_name, f"{name}.npy")
        np.save(fpath, arr)
        print(f"    Saved: {fpath} (shape={arr.shape})")

    # ---- Evaluation ----
    cal_labels = data["cal_labels"]
    test_labels = data["test_labels"]

    cal_acc = (cal_preds == cal_labels).mean()
    test_acc = (test_preds == test_labels).mean()

    print(f"\n  Qwen32B cal accuracy:  {cal_acc:.4f}")
    print(f"  Qwen32B test accuracy: {test_acc:.4f}")

    deb_base_acc = (data["deberta_test_preds"] == test_labels).mean()
    xgb_acc = (data["xgb_test_preds"] == test_labels).mean()
    print(f"  DeBERTa-base test acc: {deb_base_acc:.4f}")
    print(f"  XGBoost test acc:      {xgb_acc:.4f}")

    # Agreement
    cal_agree = compute_agreement(cal_preds, data["xgb_cal_preds"])
    test_agree = compute_agreement(test_preds, data["xgb_test_preds"])
    print(f"  Agreement (Qwen-XGB):  cal={cal_agree.mean():.3f}, test={test_agree.mean():.3f}")

    cal_agree_deb = compute_agreement(cal_preds, data["deberta_cal_preds"])
    test_agree_deb = compute_agreement(test_preds, data["deberta_test_preds"])
    print(f"  Agreement (Qwen-DeBERTa): cal={cal_agree_deb.mean():.3f}, test={test_agree_deb.mean():.3f}")

    # ---- MACCP ----
    print(f"\n  MACCP results (Qwen32B):")
    maccp_results = {}

    for alpha in [0.05, 0.10, 0.20]:
        print(f"\n    alpha = {alpha}")
        print_table_header()

        configs = {
            "A: Qwen/Qwen": ("qwen", "qwen"),
            "B: XGBoost/XGBoost": ("xgb", "xgb"),
            "C: Qwen/XGBoost": ("qwen", "xgb"),
            "D: XGBoost/Qwen": ("xgb", "qwen"),
        }

        def get_probs(model_key, split):
            if model_key == "qwen":
                return cal_probs if split == "cal" else test_probs
            return data[f"xgb_{split}_probs"]

        for config_name, (agree_model, disagree_model) in configs.items():
            result = run_maccp_pipeline(
                cal_probs_agree_model=get_probs(agree_model, "cal"),
                cal_probs_disagree_model=get_probs(disagree_model, "cal"),
                test_probs_agree_model=get_probs(agree_model, "test"),
                test_probs_disagree_model=get_probs(disagree_model, "test"),
                cal_labels=cal_labels,
                test_labels=test_labels,
                cal_agreement=cal_agree,
                test_agreement=test_agree,
                alpha=alpha,
            )
            key = f"{config_name}_alpha{alpha}"
            maccp_results[key] = result
            print_comparison_row(config_name, dataset_name, result["overall"])

    # ---- Save results ----
    results = {
        "dataset": dataset_name,
        "model": MODEL_NAME,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "num_classes": num_classes,
        "train_time_seconds": train_time,
        "cal_accuracy": float(cal_acc),
        "test_accuracy": float(test_acc),
        "deberta_base_test_accuracy": float(deb_base_acc),
        "xgb_test_accuracy": float(xgb_acc),
        "agreement_qwen_xgb_cal": float(cal_agree.mean()),
        "agreement_qwen_xgb_test": float(test_agree.mean()),
        "agreement_qwen_deberta_cal": float(cal_agree_deb.mean()),
        "agreement_qwen_deberta_test": float(test_agree_deb.mean()),
        "first_token_unique": first_tok_unique,
        "maccp": maccp_results,
    }

    save_results(results, os.path.join(results_dir, f"{dataset_name}_qwen32b.json"))
    print(f"\n  All results saved to {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-32B-Instruct with LoRA")
    parser.add_argument(
        "--dataset", required=True, choices=["eclipse", "mozilla"],
        help="Dataset to fine-tune on",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="DeepSpeed local rank")
    args = parser.parse_args()

    print("=" * 70)
    print("EXPERIMENT 4: FINE-TUNE Qwen2.5-32B-Instruct (LoRA)")
    print("=" * 70)

    train_and_evaluate(args.dataset)
    print("\nDone.")


if __name__ == "__main__":
    main()
