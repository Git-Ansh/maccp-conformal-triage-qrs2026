"""
experiment2_llm_zeroshot.py -- LLM zero-shot classification via Fireworks API.

Runs DeepSeek-v3 and GLM-5 on Eclipse and Mozilla datasets for component
prediction.  Uses agreement with XGBoost for MACCP conditioning.

Models:
  - accounts/fireworks/models/deepseek-v3 (cheaper, run first)
  - accounts/fireworks/models/glm-5 (run second)

Runs in order: DeepSeek-Eclipse, DeepSeek-Mozilla, GLM-5-Eclipse, GLM-5-Mozilla.

API key: from env var FIREWORKS_API_KEY or file ../../.fireworks_key
If no API key found, prints cost estimate and exits cleanly.

Usage:
    python experiment2_llm_zeroshot.py
    python experiment2_llm_zeroshot.py --models deepseek-v3
    python experiment2_llm_zeroshot.py --datasets eclipse
    python experiment2_llm_zeroshot.py --max-examples 100  # for testing
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

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
from cost_tracker import CostTracker

# ============================================================
# Configuration
# ============================================================
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1/chat/completions"

MODEL_IDS = {
    "deepseek-v3": "accounts/fireworks/models/deepseek-v3",
    "glm-5": "accounts/fireworks/models/glm-5",
}

MAX_TOKENS = 30
TEMPERATURE = 0.0
NUM_WORKERS = 16
CHECKPOINT_EVERY = 200

BASE_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "llm_zeroshot")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Approximate probability for predicted class
PRED_PROB = 0.80


def get_api_key():
    """Try to get API key from env var or file."""
    key = os.environ.get("FIREWORKS_API_KEY", "").strip()
    if key:
        return key

    key_file = os.path.join(SCRIPT_DIR, "..", "..", ".fireworks_key")
    if os.path.exists(key_file):
        with open(key_file) as f:
            key = f.read().strip()
        if key:
            return key

    return None


def build_system_prompt(dataset_name, label_mapping):
    """Build the system prompt listing all component names."""
    component_names = sorted(label_mapping.keys())
    component_list = "\n".join(f"- {name}" for name in component_names)
    n_classes = len(component_names)

    if dataset_name == "eclipse":
        return (
            f"You are a bug triage expert for the Eclipse IDE project. "
            f"Given a bug report, predict which component it belongs to. "
            f"Respond with ONLY the component name, exactly as listed below. "
            f"Do not add any explanation or extra text.\n\n"
            f"There are {n_classes} possible components:\n{component_list}"
        )
    else:
        return (
            f"You are a bug triage expert for Mozilla Firefox. "
            f"Given a bug report, predict which component it belongs to. "
            f"Respond with ONLY the component name, exactly as listed below. "
            f"Do not add any explanation or extra text.\n\n"
            f"There are {n_classes} possible components:\n{component_list}"
        )


def build_user_prompt(row, dataset_name):
    """Build the user prompt from a parquet row.

    Eclipse: Title + Description + Product + Severity
    Mozilla: Title + Severity (default 'normal' if missing)
    """
    text = str(row.get("text", ""))

    if dataset_name == "eclipse":
        # text format: "summary [SEP] description"
        parts = text.split(" [SEP] ", 1)
        title = parts[0] if len(parts) > 0 else text
        description = parts[1] if len(parts) > 1 else ""
        product = str(row.get("product", "unknown"))
        severity = str(row.get("severity", "normal"))

        prompt = f"Title: {title}\n"
        if description:
            # Truncate long descriptions
            if len(description) > 1000:
                description = description[:1000] + "..."
            prompt += f"Description: {description}\n"
        prompt += f"Product: {product}\n"
        prompt += f"Severity: {severity}"
        return prompt
    else:
        # Mozilla: text is just "summary"
        title = text
        severity = str(row.get("severity", "normal")) if "severity" in row.index else "normal"

        prompt = f"Title: {title}\n"
        prompt += f"Severity: {severity}"
        return prompt


def fuzzy_match(response_text, label_mapping):
    """Match LLM response to a component name.

    Tries exact match first, then case-insensitive, then stripped/cleaned.

    Returns
    -------
    int or None
        Matched label index, or None if no match.
    str
        The matched component name, or empty string.
    """
    cleaned = response_text.strip().strip('"').strip("'").strip()

    # Exact match
    if cleaned in label_mapping:
        return label_mapping[cleaned], cleaned

    # Case-insensitive match
    lower_map = {k.lower(): (v, k) for k, v in label_mapping.items()}
    if cleaned.lower() in lower_map:
        idx, name = lower_map[cleaned.lower()]
        return idx, name

    # Partial match (response contains or is contained by a component name)
    for name, idx in label_mapping.items():
        if cleaned.lower() in name.lower() or name.lower() in cleaned.lower():
            return idx, name

    return None, ""


def call_fireworks_api(api_key, model_id, system_prompt, user_prompt):
    """Make a single Fireworks API call.

    Returns
    -------
    tuple
        (response_text, prompt_tokens, completion_tokens, error_msg)
    """
    import urllib.request
    import urllib.error

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(FIREWORKS_BASE_URL, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        text = result["choices"][0]["message"]["content"].strip()
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        return text, prompt_tokens, completion_tokens, None

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        return "", 0, 0, f"HTTP {e.code}: {body[:200]}"
    except Exception as e:
        return "", 0, 0, str(e)[:200]


def build_approximate_probs(pred_label, num_classes):
    """Build approximate probability vector: PRED_PROB for predicted class, rest uniform.

    Parameters
    ----------
    pred_label : int
        Predicted class index.
    num_classes : int
        Total number of classes.

    Returns
    -------
    ndarray, shape (num_classes,)
    """
    probs = np.full(num_classes, (1.0 - PRED_PROB) / (num_classes - 1))
    probs[pred_label] = PRED_PROB
    return probs


def run_llm_on_dataset(
    api_key,
    model_short_name,
    dataset_name,
    cost_tracker,
    max_examples=None,
):
    """Run LLM zero-shot classification on cal+test sets.

    Parameters
    ----------
    api_key : str
        Fireworks API key.
    model_short_name : str
        Short model name: 'deepseek-v3' or 'glm-5'.
    dataset_name : str
        'eclipse' or 'mozilla'.
    cost_tracker : CostTracker
        Cost tracker instance.
    max_examples : int or None
        Maximum examples per split (for testing). None = all.

    Returns
    -------
    dict
        Results including accuracy, agreement, MACCP metrics.
    """
    model_id = MODEL_IDS[model_short_name]
    data = load_data(dataset_name, DATA_DIR)
    label_mapping = data["label_mapping"]
    num_classes = data["num_classes"]

    # Load parquet files for prompts
    cal_df = pd.read_parquet(os.path.join(DATA_DIR, dataset_name, "cal.parquet"))
    test_df = pd.read_parquet(os.path.join(DATA_DIR, dataset_name, "test.parquet"))

    system_prompt = build_system_prompt(dataset_name, label_mapping)

    # Checkpoint directory
    ckpt_dir = os.path.join(RESULTS_DIR, "checkpoints", f"{model_short_name}_{dataset_name}")
    os.makedirs(ckpt_dir, exist_ok=True)

    all_split_results = {}

    for split_name, df, labels in [("cal", cal_df, data["cal_labels"]),
                                    ("test", test_df, data["test_labels"])]:
        n = len(df)
        if max_examples is not None:
            n = min(n, max_examples)

        print(f"\n  Running {model_short_name} on {dataset_name}/{split_name} ({n:,} examples)")

        # Check for existing checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"{split_name}_preds.json")
        if os.path.exists(ckpt_path):
            with open(ckpt_path) as f:
                cached = json.load(f)
            if len(cached.get("preds", [])) >= n:
                print(f"    Loaded checkpoint with {len(cached['preds'])} predictions")
                all_split_results[split_name] = cached
                continue
            else:
                start_idx = len(cached.get("preds", []))
                preds = cached["preds"]
                raw_responses = cached.get("raw_responses", [])
                errors = cached.get("errors", 0)
                unmatched = cached.get("unmatched", 0)
                print(f"    Resuming from checkpoint at index {start_idx}")
        else:
            start_idx = 0
            preds = []
            raw_responses = []
            errors = 0
            unmatched = 0

        # Build prompts for remaining examples
        indices = list(range(start_idx, n))

        def process_single(idx):
            row = df.iloc[idx]
            user_prompt = build_user_prompt(row, dataset_name)
            text, pt, ct, err = call_fireworks_api(api_key, model_id, system_prompt, user_prompt)
            return idx, text, pt, ct, err

        # Process with ThreadPoolExecutor
        batch_preds = [None] * len(indices)
        batch_responses = [None] * len(indices)
        batch_errors = 0
        batch_unmatched = 0

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {
                executor.submit(process_single, idx): i
                for i, idx in enumerate(indices)
            }

            completed = 0
            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    orig_idx, text, pt, ct, err = future.result()
                except Exception as e:
                    batch_preds[batch_idx] = -1
                    batch_responses[batch_idx] = ""
                    batch_errors += 1
                    continue

                cost_tracker.add_usage(model_short_name, pt, ct)

                if err:
                    batch_preds[batch_idx] = -1
                    batch_responses[batch_idx] = f"ERROR: {err}"
                    batch_errors += 1
                else:
                    matched_idx, matched_name = fuzzy_match(text, label_mapping)
                    if matched_idx is not None:
                        batch_preds[batch_idx] = matched_idx
                    else:
                        batch_preds[batch_idx] = -1
                        batch_unmatched += 1
                    batch_responses[batch_idx] = text

                completed += 1
                if completed % CHECKPOINT_EVERY == 0:
                    # Save checkpoint
                    checkpoint_preds = preds + [p for p in batch_preds[:completed] if p is not None]
                    checkpoint_data = {
                        "preds": checkpoint_preds,
                        "raw_responses": raw_responses + [r for r in batch_responses[:completed] if r is not None],
                        "errors": errors + batch_errors,
                        "unmatched": unmatched + batch_unmatched,
                    }
                    with open(ckpt_path, "w") as f:
                        json.dump(checkpoint_data, f)

                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(
                        f"    Progress: {completed}/{len(indices)} "
                        f"({rate:.1f}/s, {elapsed:.0f}s elapsed)"
                    )

        # Merge batch results
        preds.extend(batch_preds)
        raw_responses.extend(batch_responses)
        errors += batch_errors
        unmatched += batch_unmatched

        elapsed = time.time() - t0
        print(f"    Completed {len(indices)} in {elapsed:.0f}s ({len(indices)/max(elapsed,1):.1f}/s)")
        print(f"    Errors: {errors}, Unmatched: {unmatched}")

        # Save final checkpoint
        split_result = {
            "preds": preds[:n],
            "raw_responses": raw_responses[:n],
            "errors": errors,
            "unmatched": unmatched,
        }
        with open(ckpt_path, "w") as f:
            json.dump(split_result, f)

        all_split_results[split_name] = split_result

    # ---- Compute metrics ----
    print(f"\n  Computing metrics for {model_short_name}/{dataset_name}...")

    cal_preds_llm = np.array(all_split_results["cal"]["preds"][:len(data["cal_labels"])])
    test_preds_llm = np.array(all_split_results["test"]["preds"][:len(data["test_labels"])])

    cal_labels = data["cal_labels"]
    test_labels = data["test_labels"]

    # Mask out errors/unmatched for accuracy
    cal_valid = cal_preds_llm >= 0
    test_valid = test_preds_llm >= 0

    cal_acc = (cal_preds_llm[cal_valid] == cal_labels[cal_valid]).mean() if cal_valid.sum() > 0 else 0.0
    test_acc = (test_preds_llm[test_valid] == test_labels[test_valid]).mean() if test_valid.sum() > 0 else 0.0

    print(f"    Cal accuracy:  {cal_acc:.4f} ({cal_valid.sum():,}/{len(cal_labels):,} valid)")
    print(f"    Test accuracy: {test_acc:.4f} ({test_valid.sum():,}/{len(test_labels):,} valid)")

    # Agreement with XGBoost
    xgb_cal_preds = data["xgb_cal_preds"]
    xgb_test_preds = data["xgb_test_preds"]

    cal_agree_xgb = compute_agreement(cal_preds_llm, xgb_cal_preds)
    test_agree_xgb = compute_agreement(test_preds_llm, xgb_test_preds)

    print(f"    Cal LLM-XGB agreement:  {cal_agree_xgb.mean():.3f}")
    print(f"    Test LLM-XGB agreement: {test_agree_xgb.mean():.3f}")

    # Build approximate probability matrices for MACCP
    cal_probs_llm = np.zeros((len(cal_labels), num_classes), dtype=np.float32)
    test_probs_llm = np.zeros((len(test_labels), num_classes), dtype=np.float32)

    for i in range(len(cal_labels)):
        if cal_preds_llm[i] >= 0:
            cal_probs_llm[i] = build_approximate_probs(cal_preds_llm[i], num_classes)
        else:
            # Uniform for errors
            cal_probs_llm[i] = 1.0 / num_classes

    for i in range(len(test_labels)):
        if test_preds_llm[i] >= 0:
            test_probs_llm[i] = build_approximate_probs(test_preds_llm[i], num_classes)
        else:
            test_probs_llm[i] = 1.0 / num_classes

    # MACCP: LLM agree / XGB disagree (and vice versa)
    maccp_results = {}

    # Use DeBERTa-XGB agreement for conditioning
    deb_cal_preds = data["deberta_cal_preds"]
    deb_test_preds = data["deberta_test_preds"]
    cal_agree_deb_xgb = compute_agreement(deb_cal_preds, xgb_cal_preds)
    test_agree_deb_xgb = compute_agreement(deb_test_preds, xgb_test_preds)

    for alpha in [0.05, 0.10, 0.20]:
        # Config: LLM agree / XGB disagree (conditioned on DeBERTa-XGB agreement)
        result = run_maccp_pipeline(
            cal_probs_agree_model=cal_probs_llm,
            cal_probs_disagree_model=data["xgb_cal_probs"],
            test_probs_agree_model=test_probs_llm,
            test_probs_disagree_model=data["xgb_test_probs"],
            cal_labels=cal_labels,
            test_labels=test_labels,
            cal_agreement=cal_agree_deb_xgb,
            test_agreement=test_agree_deb_xgb,
            alpha=alpha,
        )
        maccp_results[f"llm_agree_xgb_disagree_alpha{alpha}"] = result

    results = {
        "model": model_short_name,
        "dataset": dataset_name,
        "cal_accuracy": float(cal_acc),
        "test_accuracy": float(test_acc),
        "cal_valid_rate": float(cal_valid.mean()),
        "test_valid_rate": float(test_valid.mean()),
        "cal_agreement_with_xgb": float(cal_agree_xgb.mean()),
        "test_agreement_with_xgb": float(test_agree_xgb.mean()),
        "n_cal": int(len(cal_labels)),
        "n_test": int(len(test_labels)),
        "errors_cal": int(all_split_results["cal"]["errors"]),
        "errors_test": int(all_split_results["test"]["errors"]),
        "unmatched_cal": int(all_split_results["cal"]["unmatched"]),
        "unmatched_test": int(all_split_results["test"]["unmatched"]),
        "maccp": maccp_results,
    }

    # Save approximate probs for downstream use
    np.save(
        os.path.join(RESULTS_DIR, f"{model_short_name}_{dataset_name}_cal_probs.npy"),
        cal_probs_llm,
    )
    np.save(
        os.path.join(RESULTS_DIR, f"{model_short_name}_{dataset_name}_test_probs.npy"),
        test_probs_llm,
    )
    np.save(
        os.path.join(RESULTS_DIR, f"{model_short_name}_{dataset_name}_cal_preds.npy"),
        cal_preds_llm,
    )
    np.save(
        os.path.join(RESULTS_DIR, f"{model_short_name}_{dataset_name}_test_preds.npy"),
        test_preds_llm,
    )

    return results


def print_cost_estimates(datasets, models):
    """Print estimated costs for all planned runs."""
    print("\n" + "=" * 70)
    print("COST ESTIMATES (no API key found)")
    print("=" * 70)

    total_est = 0.0
    for model in models:
        for dataset in datasets:
            # Approximate sizes
            if dataset == "eclipse":
                n_cal, n_test = 30017, 25499
                avg_prompt = 500
            else:
                n_cal, n_test = 8289, 8480
                avg_prompt = 300

            n_total = n_cal + n_test
            est = CostTracker.estimate_cost(model, n_total, avg_prompt, 15)
            total_est += est["cost_total"]
            print(
                f"  {model:<15s} x {dataset:<8s}: "
                f"{n_total:>7,} calls, "
                f"~{est['total_input_tokens']:>10,} in + {est['total_output_tokens']:>8,} out = "
                f"${est['cost_total']:.4f}"
            )

    print(f"\n  TOTAL ESTIMATED: ${total_est:.4f}")
    print(f"  Budget: ${CostTracker().budget:.2f}")
    print(f"  Headroom: ${CostTracker().budget - total_est:.4f}")


def main():
    parser = argparse.ArgumentParser(description="LLM zero-shot classification via Fireworks API")
    parser.add_argument(
        "--models", nargs="+", default=["deepseek-v3", "glm-5"],
        choices=["deepseek-v3", "glm-5"],
        help="Models to run (default: both)",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["eclipse", "mozilla"],
        choices=["eclipse", "mozilla"],
        help="Datasets to run (default: both)",
    )
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Max examples per split (for testing)",
    )
    args = parser.parse_args()

    api_key = get_api_key()
    if api_key is None:
        print("No Fireworks API key found.")
        print("Set FIREWORKS_API_KEY env var or create ../../.fireworks_key file.")
        print_cost_estimates(args.datasets, args.models)
        print("\nExiting cleanly (no API key).")
        sys.exit(0)

    print("=" * 70)
    print("LLM ZERO-SHOT CLASSIFICATION EXPERIMENT")
    print("=" * 70)
    print(f"  Models: {args.models}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Max examples: {args.max_examples or 'all'}")
    print(f"  Workers: {NUM_WORKERS}")
    print(f"  Checkpoint every: {CHECKPOINT_EVERY}")

    # Print cost estimates
    print_cost_estimates(args.datasets, args.models)

    cost_tracker = CostTracker()
    all_results = {}

    # Run order: DeepSeek first (cheaper), then GLM-5
    run_order = []
    for model in ["deepseek-v3", "glm-5"]:
        if model in args.models:
            for dataset in args.datasets:
                run_order.append((model, dataset))

    for model_name, dataset_name in run_order:
        print(f"\n{'='*70}")
        print(f"Running: {model_name} on {dataset_name}")
        print(f"{'='*70}")

        # Check budget before starting
        if cost_tracker.is_over_budget():
            print("  BUDGET EXCEEDED -- skipping remaining runs")
            break

        results = run_llm_on_dataset(
            api_key=api_key,
            model_short_name=model_name,
            dataset_name=dataset_name,
            cost_tracker=cost_tracker,
            max_examples=args.max_examples,
        )

        key = f"{model_name}_{dataset_name}"
        all_results[key] = results

        save_results(results, os.path.join(RESULTS_DIR, f"{key}_results.json"))

        cost_tracker.print_summary()

    # Save combined results
    save_results(all_results, os.path.join(RESULTS_DIR, "all_llm_results.json"))
    cost_tracker.save(os.path.join(RESULTS_DIR, "cost_tracker.json"))

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for key, res in all_results.items():
        print(f"\n  {key}:")
        print(f"    Test accuracy: {res['test_accuracy']:.4f}")
        print(f"    Valid rate: {res['test_valid_rate']:.3f}")
        print(f"    XGB agreement: {res['test_agreement_with_xgb']:.3f}")

    cost_tracker.print_summary()
    print("Done.")


if __name__ == "__main__":
    main()
