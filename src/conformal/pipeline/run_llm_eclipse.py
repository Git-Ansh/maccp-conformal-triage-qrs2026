"""
LLM Evaluation on Eclipse Bug Triage.

Runs DeepSeek V3 via Fireworks AI on two tasks:

1. FULL COMPARISON: LLM vs XGBoost on Eclipse S2 Component Assignment
   - Same test set, same classes (29 components + Other)
   - LLM uses bug title (short_desc) only
   - XGBoost uses tabular + TF-IDF features
   - Reports accuracy, coverage, and confidence curves for both

2. DEFERRED ITEM RESCUE: Run LLM on items XGBoost deferred
   - XGBoost cascade defers uncertain items
   - LLM classifies those deferred items
   - If LLM is confident, it "rescues" them from human review
   - Reports rescue rate and accuracy on rescued items

Confidence estimation uses consistency sampling (5 runs with temperature=0.7).
Agreement rate = confidence.
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cascade.framework.confidence_stage import ConfidenceStage
from conformal.data.eclipse_loader import prepare_eclipse_data
from conformal.stages.eclipse_config import (
    prepare_stage_2_data,
    SEVERITY_CLASSES,
)
from conformal.llm.fireworks_client import FireworksClient
from conformal.llm.llm_classifier import LLMClassifier

OUTPUT_DIR = PROJECT_ROOT / "conformal_outputs" / "eclipse"


def run_llm_component_classification(
    max_test_samples: int = 500,
    n_consistency: int = 5,
    n_few_shot: int = 5,
    confidence_threshold: float = 0.6,
    run_xgboost_comparison: bool = True,
    save_results: bool = True,
):
    """
    Run LLM on Eclipse S2 Component Assignment and compare with XGBoost.

    Args:
        max_test_samples: Limit test samples (for cost control, full=~28K)
        n_consistency: Number of consistency samples per item
        n_few_shot: Number of few-shot examples in prompt
        confidence_threshold: Min confidence for "confident" prediction
        run_xgboost_comparison: Also run XGBoost for comparison
        save_results: Save results to disk
    """
    print("=" * 70)
    print("LLM vs XGBOOST: ECLIPSE COMPONENT ASSIGNMENT")
    print("=" * 70)
    print(f"  Model: DeepSeek V3 via Fireworks AI")
    print(f"  Consistency samples: {n_consistency}")
    print(f"  Few-shot examples: {n_few_shot}")
    print(f"  Max test samples: {max_test_samples}")
    print()

    # Load Eclipse data
    data = prepare_eclipse_data()
    train_df = data["train_df"]
    test_df = data["test_df"]

    # Filter to real bugs only (same as S2)
    train_real = train_df[train_df["is_noise"] == 0].copy()
    test_real = test_df[test_df["is_noise"] == 0].copy()

    # Get component target and classes
    top_components = data["top_components"]
    all_classes = top_components + ["Other"]

    # Subsample test set for cost control
    if max_test_samples and len(test_real) > max_test_samples:
        test_real = test_real.sample(n=max_test_samples, random_state=42)
        test_real = test_real.reset_index(drop=True)
        print(f"  Subsampled test set: {len(test_real)} items")
    else:
        print(f"  Full test set: {len(test_real)} items")

    test_texts = test_real["short_desc"].fillna("").tolist()
    test_labels = test_real["component_target"].fillna("Other").tolist()

    # Majority baseline
    label_counts = Counter(test_labels)
    majority_class = label_counts.most_common(1)[0][0]
    majority_acc = label_counts[majority_class] / len(test_labels)
    print(f"  Majority baseline: {majority_acc:.1%} (class: {majority_class})")
    print(f"  Number of classes: {len(set(test_labels))}")
    print()

    # =========================================================
    # LLM Classification
    # =========================================================
    print("-" * 50)
    print("LLM CLASSIFICATION (DeepSeek V3)")
    print("-" * 50)

    client = FireworksClient()
    clf = LLMClassifier(
        task_description=(
            "You are classifying Eclipse IDE bug reports into their correct component. "
            "Given a bug title, predict which Eclipse component this bug belongs to. "
            "These components are parts of the Eclipse IDE platform."
        ),
        class_names=all_classes,
        n_consistency=n_consistency,
        temperature=0.7,
        client=client,
    )

    # Set few-shot examples from training data (diverse: one per class)
    train_texts = train_real["short_desc"].fillna("").tolist()
    train_labels_list = train_real["component_target"].fillna("Other").tolist()
    clf.set_examples(train_texts, train_labels_list, n_examples=n_few_shot)

    print(f"  Few-shot examples set: {len(clf._examples)}")
    for ex in clf._examples[:3]:
        print(f"    {ex['label']}: {ex['text'][:60]}...")
    print()

    start_time = time.time()
    llm_results = clf.evaluate(
        texts=test_texts,
        true_labels=test_labels,
        n_samples=n_consistency,
        confidence_threshold=confidence_threshold,
    )
    llm_time = time.time() - start_time

    print(f"\n  LLM Results:")
    print(f"    Overall accuracy: {llm_results['overall_accuracy']:.1%}")
    print(f"    Confident accuracy: {llm_results['confident_accuracy']:.1%}")
    print(f"    Coverage: {llm_results['coverage']:.1%}")
    print(f"    Deferred accuracy: {llm_results['deferred_accuracy']:.1%}")
    print(f"    Time: {llm_time:.1f}s")
    print(f"    API calls: {llm_results['client_stats']['api_calls']}")
    print(f"    Cache hits: {llm_results['client_stats']['cache_hits']}")
    print(f"    Total tokens: {llm_results['client_stats']['total_tokens']:,}")

    print(f"\n  Coverage-accuracy curve (LLM):")
    print(f"  {'Threshold':>10} {'Coverage':>10} {'Accuracy':>10} {'N':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for row in llm_results["curve"]:
        print(
            f"  {row['threshold']:10.2f} {row['coverage']:10.1%} "
            f"{row['accuracy']:10.1%} {row['n']:8d}"
        )

    # =========================================================
    # XGBoost Comparison (optional)
    # =========================================================
    xgb_results = None
    if run_xgboost_comparison:
        print(f"\n{'-' * 50}")
        print("XGBOOST + TF-IDF COMPARISON")
        print("-" * 50)

        s2_data = prepare_stage_2_data(
            train_df, test_df, data["numeric_features"], data["categorical_features"],
            top_components,
        )

        s2_stage = ConfidenceStage(
            name="S2_component_comparison",
            classes=s2_data["component_classes"],
            target_accuracy=0.70,
        )
        s2_stage.fit(
            s2_data["train_X"],
            s2_data["train_y"],
            feature_names=s2_data["feature_cols"],
            text_data=s2_data["train_text"],
        )

        # If we subsampled, match the test indices
        if max_test_samples and len(test_real) < len(test_df[test_df["is_noise"] == 0]):
            # Use the subsampled indices to filter XGBoost test data
            test_real_full = test_df[test_df["is_noise"] == 0]
            sub_idx = test_real.index
            # Need to map back to position in the full real test set
            full_idx = test_real_full.index
            pos_mask = full_idx.isin(sub_idx)

            xgb_test_X = s2_data["test_X"][pos_mask]
            xgb_test_y = s2_data["test_y"][pos_mask]
            xgb_test_text = s2_data["test_text"][pos_mask]
        else:
            xgb_test_X = s2_data["test_X"]
            xgb_test_y = s2_data["test_y"]
            xgb_test_text = s2_data["test_text"]

        xgb_preds = s2_stage.predict(xgb_test_X, text_data=xgb_test_text)

        # XGBoost accuracy at same coverage as LLM
        xgb_conf = xgb_preds["is_confident"]
        if xgb_conf.any():
            xgb_acc = (xgb_test_y[xgb_conf] == xgb_preds["class"][xgb_conf]).mean()
            xgb_cov = xgb_conf.mean()
        else:
            xgb_acc = 0.0
            xgb_cov = 0.0

        # XGBoost overall (forced prediction, no gating)
        xgb_overall_acc = (xgb_test_y == xgb_preds["predicted_raw"]).mean()

        xgb_results = {
            "overall_accuracy": float(xgb_overall_acc),
            "confident_accuracy": float(xgb_acc),
            "coverage": float(xgb_cov),
        }

        print(f"  XGBoost Results:")
        print(f"    Overall accuracy (no gating): {xgb_overall_acc:.1%}")
        print(f"    Confident accuracy: {xgb_acc:.1%}")
        print(f"    Coverage: {xgb_cov:.1%}")

        # XGBoost coverage-accuracy curve
        xgb_curve = s2_stage.coverage_accuracy_curve(
            xgb_test_X, xgb_test_y, text_data=xgb_test_text
        )
        print(f"\n  Coverage-accuracy curve (XGBoost):")
        print(f"  {'Threshold':>10} {'Coverage':>10} {'Accuracy':>10}")
        print(f"  {'-'*10} {'-'*10} {'-'*10}")
        for _, row in xgb_curve.iterrows():
            print(
                f"  {row['threshold']:10.2f} {row['coverage']:10.1%} "
                f"{row['accuracy']:10.1%}"
            )

        # Store deferred items for rescue analysis
        xgb_deferred_mask = ~xgb_conf
        xgb_results["n_deferred"] = int(xgb_deferred_mask.sum())

    # =========================================================
    # DEFERRED ITEM RESCUE
    # =========================================================
    if run_xgboost_comparison and xgb_results and xgb_results["n_deferred"] > 0:
        print(f"\n{'-' * 50}")
        print("DEFERRED ITEM RESCUE (LLM on XGBoost-deferred items)")
        print("-" * 50)

        deferred_mask = ~xgb_conf
        n_deferred = deferred_mask.sum()
        print(f"  XGBoost deferred {n_deferred} items")

        # Get deferred texts and labels
        deferred_texts = [
            t for t, m in zip(test_texts[:len(deferred_mask)], deferred_mask) if m
        ]
        deferred_labels = [
            l for l, m in zip(test_labels[:len(deferred_mask)], deferred_mask) if m
        ]

        # Limit for cost
        max_deferred = min(200, len(deferred_texts))
        if len(deferred_texts) > max_deferred:
            # Random subsample of deferred items
            rng = np.random.RandomState(42)
            idx = rng.choice(len(deferred_texts), size=max_deferred, replace=False)
            deferred_texts = [deferred_texts[i] for i in idx]
            deferred_labels = [deferred_labels[i] for i in idx]
            print(f"  Subsampled to {max_deferred} deferred items")

        rescue_results = clf.evaluate(
            texts=deferred_texts,
            true_labels=deferred_labels,
            n_samples=n_consistency,
            confidence_threshold=confidence_threshold,
        )

        rescued_n = rescue_results["n_confident"]
        rescued_acc = rescue_results["confident_accuracy"]
        rescue_rate = rescued_n / len(deferred_texts) if deferred_texts else 0

        print(f"\n  Rescue Results:")
        print(f"    Deferred items tested: {len(deferred_texts)}")
        print(f"    LLM rescued: {rescued_n} ({rescue_rate:.1%})")
        print(f"    Rescue accuracy: {rescued_acc:.1%}")
        print(f"    Still deferred: {rescue_results['n_deferred']}")
        print(f"    Deferred accuracy: {rescue_results['deferred_accuracy']:.1%}")

        xgb_results["rescue"] = {
            "n_tested": len(deferred_texts),
            "n_rescued": rescued_n,
            "rescue_rate": rescue_rate,
            "rescue_accuracy": float(rescued_acc),
            "n_still_deferred": rescue_results["n_deferred"],
        }

    # =========================================================
    # SUMMARY
    # =========================================================
    print(f"\n{'=' * 70}")
    print("SUMMARY: LLM vs XGBoost on Eclipse Component Assignment")
    print("=" * 70)
    print(f"  {'Method':<25} {'Overall Acc':>12} {'Confident Acc':>14} {'Coverage':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*14} {'-'*10}")
    print(f"  {'Majority baseline':<25} {majority_acc:>12.1%} {'-':>14} {'100.0%':>10}")
    print(
        f"  {'LLM (DeepSeek V3)':<25} {llm_results['overall_accuracy']:>12.1%} "
        f"{llm_results['confident_accuracy']:>14.1%} {llm_results['coverage']:>10.1%}"
    )
    if xgb_results:
        print(
            f"  {'XGBoost + TF-IDF':<25} {xgb_results['overall_accuracy']:>12.1%} "
            f"{xgb_results['confident_accuracy']:>14.1%} {xgb_results['coverage']:>10.1%}"
        )

    # Save results
    if save_results:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        results = {
            "task": "Eclipse S2 Component Assignment",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "max_test_samples": max_test_samples,
                "n_consistency": n_consistency,
                "n_few_shot": n_few_shot,
                "confidence_threshold": confidence_threshold,
                "model": "accounts/fireworks/models/deepseek-v3p2",
            },
            "majority_baseline": majority_acc,
            "n_classes": len(set(test_labels)),
            "n_test": len(test_labels),
            "llm": {
                "overall_accuracy": llm_results["overall_accuracy"],
                "confident_accuracy": llm_results["confident_accuracy"],
                "coverage": llm_results["coverage"],
                "deferred_accuracy": llm_results["deferred_accuracy"],
                "n_confident": llm_results["n_confident"],
                "n_deferred": llm_results["n_deferred"],
                "curve": llm_results["curve"],
                "client_stats": llm_results["client_stats"],
                "time_seconds": llm_time,
            },
        }
        if xgb_results:
            results["xgboost"] = xgb_results

        results_path = OUTPUT_DIR / f"llm_comparison_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=lambda x: int(x) if hasattr(x, 'item') else str(x))
        print(f"\nResults saved to {results_path}")

    return {
        "llm": llm_results,
        "xgboost": xgb_results,
        "majority_baseline": majority_acc,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Eclipse Component Classification")
    parser.add_argument(
        "--max-samples", type=int, default=500,
        help="Max test samples (default 500, use -1 for all)",
    )
    parser.add_argument(
        "--n-consistency", type=int, default=5,
        help="Number of consistency samples (default 5)",
    )
    parser.add_argument(
        "--n-few-shot", type=int, default=30,
        help="Number of few-shot examples (default 30, one per class)",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.6,
        help="Confidence threshold for gating (default 0.6)",
    )
    parser.add_argument(
        "--no-xgboost", action="store_true",
        help="Skip XGBoost comparison",
    )
    args = parser.parse_args()

    max_samples = None if args.max_samples == -1 else args.max_samples

    run_llm_component_classification(
        max_test_samples=max_samples,
        n_consistency=args.n_consistency,
        n_few_shot=args.n_few_shot,
        confidence_threshold=args.confidence_threshold,
        run_xgboost_comparison=not args.no_xgboost,
    )
