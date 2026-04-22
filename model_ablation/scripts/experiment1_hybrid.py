"""
experiment1_hybrid.py -- Hybrid MACCP experiment.

Tests 4 model configurations on both Eclipse (30 classes) and Mozilla (20 classes)
at multiple alpha levels to evaluate hybrid agree/disagree model routing.

Configurations:
  A: DeBERTa/DeBERTa (paper baseline -- same model for both partitions)
  B: XGBoost/XGBoost (structured-only baseline)
  C: DeBERTa agree / XGBoost disagree (hybrid)
  D: XGBoost agree / DeBERTa disagree (control)

Agreement is always binary argmax: deberta_top1 == xgb_top1.

Usage:
    python experiment1_hybrid.py
"""

import json
import os
import sys

import numpy as np

np.random.seed(42)

# Add scripts directory to path
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
DATASETS = ["eclipse", "mozilla"]
ALPHA_LEVELS = [0.05, 0.10, 0.20]
LAM = 0.01
KREG = 5

BASE_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "hybrid")
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIGS = {
    "A: DeBERTa/DeBERTa": {
        "agree_model": "deberta",
        "disagree_model": "deberta",
        "description": "Paper baseline (DeBERTa for both partitions)",
    },
    "B: XGBoost/XGBoost": {
        "agree_model": "xgb",
        "disagree_model": "xgb",
        "description": "Structured-only baseline (XGBoost for both partitions)",
    },
    "C: DeBERTa/XGBoost": {
        "agree_model": "deberta",
        "disagree_model": "xgb",
        "description": "Hybrid: DeBERTa when agree, XGBoost when disagree",
    },
    "D: XGBoost/DeBERTa": {
        "agree_model": "xgb",
        "disagree_model": "deberta",
        "description": "Control: XGBoost when agree, DeBERTa when disagree",
    },
}


def get_probs(data, model_name, split):
    """Get probability matrix for a given model and split."""
    return data[f"{model_name}_{split}_probs"]


def run_single_model_baseline(data, model_name, alpha):
    """Run standard RAPS (non-MACCP) on a single model for comparison."""
    cal_probs = get_probs(data, model_name, "cal")
    test_probs = get_probs(data, model_name, "test")
    cal_labels = data["cal_labels"]
    test_labels = data["test_labels"]

    scores = compute_raps_scores_batch(cal_probs, cal_labels, LAM, KREG)
    q = compute_conformal_quantile(scores, alpha)
    pred_sets = compute_prediction_set(test_probs, q, LAM, KREG)
    metrics = evaluate_prediction_sets(
        pred_sets, test_labels, f"{model_name} RAPS (alpha={alpha})"
    )
    metrics["quantile"] = float(q)
    metrics["alpha"] = float(alpha)
    return metrics


def run_experiment():
    """Run the full hybrid MACCP experiment across datasets, configs, and alphas."""
    all_results = {}

    for dataset in DATASETS:
        print("=" * 80)
        print(f"DATASET: {dataset.upper()}")
        print("=" * 80)

        data = load_data(dataset, DATA_DIR)
        cal_labels = data["cal_labels"]
        test_labels = data["test_labels"]
        num_classes = data["num_classes"]

        # Compute agreement
        cal_agree = compute_agreement(data["deberta_cal_preds"], data["xgb_cal_preds"])
        test_agree = compute_agreement(data["deberta_test_preds"], data["xgb_test_preds"])

        print(f"  Classes: {num_classes}")
        print(f"  Cal: {len(cal_labels):,} | Test: {len(test_labels):,}")
        print(f"  Cal agreement: {cal_agree.sum():,}/{len(cal_agree):,} ({cal_agree.mean():.1%})")
        print(f"  Test agreement: {test_agree.sum():,}/{len(test_agree):,} ({test_agree.mean():.1%})")

        # Base model accuracies
        deb_acc = (data["deberta_test_preds"] == test_labels).mean()
        xgb_acc = (data["xgb_test_preds"] == test_labels).mean()
        majority_class = np.bincount(cal_labels).argmax()
        majority_acc = (test_labels == majority_class).mean()
        print(f"  DeBERTa accuracy: {deb_acc:.4f}")
        print(f"  XGBoost accuracy: {xgb_acc:.4f}")
        print(f"  Majority baseline: {majority_acc:.4f}")

        dataset_results = {
            "meta": {
                "num_classes": num_classes,
                "n_cal": len(cal_labels),
                "n_test": len(test_labels),
                "cal_agreement_rate": float(cal_agree.mean()),
                "test_agreement_rate": float(test_agree.mean()),
                "deberta_accuracy": float(deb_acc),
                "xgb_accuracy": float(xgb_acc),
                "majority_accuracy": float(majority_acc),
            },
            "baselines": {},
            "configs": {},
        }

        for alpha in ALPHA_LEVELS:
            print(f"\n  --- alpha = {alpha} ---")

            # Single-model baselines
            for model_name, label in [("deberta", "DeBERTa"), ("xgb", "XGBoost")]:
                baseline = run_single_model_baseline(data, model_name, alpha)
                key = f"{model_name}_alpha{alpha}"
                dataset_results["baselines"][key] = baseline
                print(
                    f"    {label} RAPS: "
                    f"Cov={baseline['coverage']:.3f} "
                    f"Size={baseline['mean_set_size']:.2f} "
                    f"Sing={baseline['singleton_rate']:.3f} "
                    f"SingAcc={baseline['singleton_accuracy']:.3f}"
                )

            # MACCP configs
            print()
            for config_name, config in CONFIGS.items():
                agree_model = config["agree_model"]
                disagree_model = config["disagree_model"]

                result = run_maccp_pipeline(
                    cal_probs_agree_model=get_probs(data, agree_model, "cal"),
                    cal_probs_disagree_model=get_probs(data, disagree_model, "cal"),
                    test_probs_agree_model=get_probs(data, agree_model, "test"),
                    test_probs_disagree_model=get_probs(data, disagree_model, "test"),
                    cal_labels=cal_labels,
                    test_labels=test_labels,
                    cal_agreement=cal_agree,
                    test_agreement=test_agree,
                    alpha=alpha,
                    lam=LAM,
                    kreg=KREG,
                )

                key = f"{config_name}_alpha{alpha}"
                dataset_results["configs"][key] = result

                overall = result["overall"]
                agree_info = result.get("agree", {})
                disagree_info = result.get("disagree", {})

                print(
                    f"    {config_name}: "
                    f"Cov={overall['coverage']:.3f} "
                    f"Size={overall['mean_set_size']:.2f} "
                    f"Sing={overall['singleton_rate']:.3f} "
                    f"SingAcc={overall['singleton_accuracy']:.3f} "
                    f"| q_a={overall['q_agree']:.3f} q_d={overall['q_disagree']:.3f}"
                )
                if agree_info:
                    print(
                        f"      Agree  (n={agree_info['n']:,}): "
                        f"Cov={agree_info['coverage']:.3f} "
                        f"Size={agree_info['mean_set_size']:.2f} "
                        f"Sing={agree_info['singleton_rate']:.3f}"
                    )
                if disagree_info:
                    print(
                        f"      Disagree (n={disagree_info['n']:,}): "
                        f"Cov={disagree_info['coverage']:.3f} "
                        f"Size={disagree_info['mean_set_size']:.2f} "
                        f"Sing={disagree_info['singleton_rate']:.3f}"
                    )

        all_results[dataset] = dataset_results

    # ============================================================
    # Print summary comparison table
    # ============================================================
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 80)

    for alpha in ALPHA_LEVELS:
        print(f"\n  alpha = {alpha}")
        print_table_header()

        for dataset in DATASETS:
            ds_res = all_results[dataset]

            # Baselines
            for model_name, label in [("deberta", "DeBERTa RAPS"), ("xgb", "XGBoost RAPS")]:
                key = f"{model_name}_alpha{alpha}"
                print_comparison_row(label, dataset, ds_res["baselines"][key])

            # MACCP configs
            for config_name in CONFIGS:
                key = f"{config_name}_alpha{alpha}"
                overall = ds_res["configs"][key]["overall"]
                print_comparison_row(config_name, dataset, overall)

            print()

    # Save results
    save_results(all_results, os.path.join(RESULTS_DIR, "hybrid_results.json"))

    # Per-dataset files
    for dataset in DATASETS:
        save_results(
            all_results[dataset],
            os.path.join(RESULTS_DIR, f"{dataset}_hybrid.json"),
        )

    print("\nDone. Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    run_experiment()
