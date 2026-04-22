"""
generate_explanations.py -- Generate example MACCP explanations and summary statistics.

Produces:
  conformal_outputs/maccp_results/example_explanations.json   (10 examples)
  conformal_outputs/maccp_results/explanation_summary.json    (full test-set stats)

Run with:
  PYTHONUNBUFFERED=1 venv/Scripts/python.exe -u src/conformal/pipeline/generate_explanations.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------
# Path setup
# ------------------------------------------------------------------
BASE = Path(os.environ.get("PROJECT_ROOT", "."))
sys.path.insert(0, str(BASE))

from src.conformal.explanation_generator import TriageExplanationGenerator

DATA_DIR = BASE / "conformal_outputs" / "eclipse_no_other"
DEBERTA_DIR = BASE / "conformal_outputs" / "deberta_no_other"
MACCP_DIR = BASE / "conformal_outputs" / "maccp_results"
OUTPUT_DIR = MACCP_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# RAPS hyperparameters -- must match run_maccp.py
LAM = 0.01
K_REG = 5
ALPHA = 0.10


# ------------------------------------------------------------------
# Reconstruct MACCP prediction sets
# ------------------------------------------------------------------
def compute_maccp_sets(test_probs, agree_mask, q_agree, q_disagree,
                       lam=LAM, k_reg=K_REG):
    """Re-derive MACCP prediction sets from stored quantile thresholds."""
    sets = []
    for i in range(len(test_probs)):
        q = q_agree if agree_mask[i] else q_disagree
        sorted_idx = np.argsort(-test_probs[i])
        cumsum = 0.0
        s = []
        for rank, class_idx in enumerate(sorted_idx):
            cumsum += test_probs[i, class_idx]
            penalty = lam * max(0, rank + 1 - k_reg)
            cumsum_reg = cumsum + penalty
            s.append(int(class_idx))
            if cumsum_reg >= q:
                break
        sets.append(s)
    return sets


def main():
    print("Loading data...")

    # Labels and predictions
    deb_probs = np.load(str(DEBERTA_DIR / "test_probs.npy"))
    deb_preds = np.load(str(DEBERTA_DIR / "test_preds.npy"))
    deb_labels = np.load(str(DEBERTA_DIR / "test_labels.npy"))
    xgb_preds = np.load(str(MACCP_DIR / "xgb_test_preds.npy"))
    xgb_probs = np.load(str(MACCP_DIR / "xgb_test_probs.npy"))

    # Label map
    with open(str(DATA_DIR / "label_mapping.json")) as f:
        label_map = json.load(f)
    inv_map = {v: k for k, v in label_map.items()}

    # MACCP quantile thresholds
    with open(str(MACCP_DIR / "maccp_results.json")) as f:
        maccp_results = json.load(f)
    alpha_key = f"alpha_{str(ALPHA).replace('.', '_').rstrip('0')}"
    # Handle 0.1 -> alpha_0.1 vs 0.10
    if alpha_key not in maccp_results:
        alpha_key = f"alpha_{ALPHA}"
    alpha_data = maccp_results[alpha_key]
    q_agree = alpha_data["maccp_overall"]["q_agree"]
    q_disagree = alpha_data["maccp_overall"]["q_disagree"]

    print(f"q_agree={q_agree:.6f}  q_disagree={q_disagree:.6f}")

    agree_mask = deb_preds == xgb_preds
    print(f"Test set: {len(deb_labels)} samples, agreement rate: {agree_mask.mean():.3f}")

    # Reconstruct prediction sets
    print("Reconstructing MACCP prediction sets...")
    sets = compute_maccp_sets(deb_probs, agree_mask, q_agree, q_disagree)

    # ------------------------------------------------------------------
    # Instantiate generator (agree/disagree accuracy from agreement_analysis.json)
    # ------------------------------------------------------------------
    gen = TriageExplanationGenerator(
        label_map=label_map,
        agree_accuracy=0.857,
        disagree_accuracy=0.271,
        enabled=True,
    )

    # ------------------------------------------------------------------
    # Select 10 representative examples
    # ------------------------------------------------------------------
    # 3 agree singletons (correct)
    agree_singletons_correct = [
        i for i in range(len(sets))
        if len(sets[i]) == 1 and agree_mask[i] and deb_labels[i] in sets[i]
    ]
    # 3 agree small sets (size 2-3)
    agree_small = [
        i for i in range(len(sets))
        if 2 <= len(sets[i]) <= 3 and agree_mask[i]
    ]
    # 2 disagree correct
    disagree_correct = [
        i for i in range(len(sets))
        if not agree_mask[i] and deb_labels[i] in sets[i]
    ]
    # 2 disagree incorrect
    disagree_incorrect = [
        i for i in range(len(sets))
        if not agree_mask[i] and deb_labels[i] not in sets[i]
    ]

    selected_indices = (
        agree_singletons_correct[:3]
        + agree_small[:3]
        + disagree_correct[:2]
        + disagree_incorrect[:2]
    )

    category_labels = (
        ["agree_singleton_correct"] * 3
        + ["agree_small_set"] * 3
        + ["disagree_correct"] * 2
        + ["disagree_incorrect"] * 2
    )

    print(f"\nSelected {len(selected_indices)} examples:")
    print("  agree_singleton_correct:", agree_singletons_correct[:3])
    print("  agree_small_set:", agree_small[:3])
    print("  disagree_correct:", disagree_correct[:2])
    print("  disagree_incorrect:", disagree_incorrect[:2])

    # ------------------------------------------------------------------
    # Generate 10 example explanations
    # ------------------------------------------------------------------
    examples = []
    for idx, (test_idx, category) in enumerate(zip(selected_indices, category_labels)):
        ex = gen.explain(
            conformal_set=sets[test_idx],
            deberta_probs=deb_probs[test_idx],
            xgboost_probs=xgb_probs[test_idx],
            deberta_pred=int(deb_preds[test_idx]),
            xgboost_pred=int(xgb_preds[test_idx]),
            agreement=bool(agree_mask[test_idx]),
            alpha=ALPHA,
        )
        true_label = inv_map[int(deb_labels[test_idx])]
        correct = true_label in ex["prediction_set"]
        record = {
            "example_id": idx + 1,
            "test_index": int(test_idx),
            "category": category,
            "true_label": true_label,
            "true_label_in_set": correct,
            "explanation": ex,
        }
        examples.append(record)
        print(f"\n  Example {idx + 1} [{category}]  idx={test_idx}")
        print(f"    True: {true_label}  |  In set: {correct}  |  "
              f"Set: {ex['prediction_set']}")
        print(f"    Action: {ex['action_recommendation']}")
        print(f"    Reliability: {ex['reliability_estimate']}")

    # Save examples
    out_path = OUTPUT_DIR / "example_explanations.json"
    with open(str(out_path), "w") as f:
        json.dump(examples, f, indent=2)
    print(f"\nSaved {len(examples)} examples -> {out_path}")

    # ------------------------------------------------------------------
    # Generate summary stats over the FULL test set
    # ------------------------------------------------------------------
    print("\nGenerating all explanations for summary statistics...")
    n = len(deb_labels)
    all_explanations = []
    for i in range(n):
        ex = gen.explain(
            conformal_set=sets[i],
            deberta_probs=deb_probs[i],
            xgboost_probs=xgb_probs[i],
            deberta_pred=int(deb_preds[i]),
            xgboost_pred=int(xgb_preds[i]),
            agreement=bool(agree_mask[i]),
            alpha=ALPHA,
        )
        all_explanations.append(ex)
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{n}...")

    print(f"  Done -- {n} explanations generated.")

    summary = gen.generate_summary_stats(all_explanations)

    # Save summary
    summary_path = OUTPUT_DIR / "explanation_summary.json"
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary -> {summary_path}")

    # Print summary
    print("\n=== EXPLANATION SUMMARY STATISTICS ===")
    print(f"Total:          {summary['total']}")
    print(f"Avg set size:   {summary['avg_set_size']:.3f}")
    print(f"Diff top cand:  {summary['different_top_candidates']:.3f}")
    print("\nAction distribution:")
    for k, v in summary["action_distribution"].items():
        pct = v / summary["total"] * 100
        print(f"  {k:<35} {v:>6} ({pct:.1f}%)")
    print("\nAgreement distribution:")
    for k, v in summary["agreement_distribution"].items():
        pct = v / summary["total"] * 100
        print(f"  {k:<10} {v:>6} ({pct:.1f}%)")
    print("\nConfidence distribution:")
    for k, v in summary["confidence_distribution"].items():
        pct = v / summary["total"] * 100
        print(f"  {k:<10} {v:>6} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
