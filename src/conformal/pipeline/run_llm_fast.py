"""
Fast LLM evaluation on Eclipse Component Assignment.

Uses pre-extracted CSV (no JSON parsing needed).
Reads from conformal_outputs/eclipse/llm_test_data.csv
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conformal.llm.llm_classifier import LLMClassifier

OUTPUT_DIR = PROJECT_ROOT / "conformal_outputs" / "eclipse"
TEST_CSV = OUTPUT_DIR / "llm_test_data.csv"
EXAMPLES_JSON = OUTPUT_DIR / "llm_few_shot_examples.json"


def run(max_samples=200, n_consistency=5, confidence_threshold=0.6):
    print("=" * 70)
    print("LLM ECLIPSE COMPONENT ASSIGNMENT (Fast)")
    print("=" * 70)

    # Load pre-extracted data
    test_df = pd.read_csv(TEST_CSV)
    with open(EXAMPLES_JSON, "r") as f:
        examples = json.load(f)

    print(f"  Test bugs: {len(test_df)}")
    print(f"  Few-shot examples: {len(examples)}")

    # Subsample
    if max_samples and len(test_df) > max_samples:
        test_df = test_df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"  Subsampled to: {len(test_df)}")

    test_texts = test_df["short_desc"].fillna("").tolist()
    test_labels = test_df["component_target"].fillna("Other").tolist()

    # Classes
    all_classes = sorted(set([e["label"] for e in examples] + test_labels))
    print(f"  Classes: {len(all_classes)}")

    # Majority baseline
    counts = Counter(test_labels)
    majority_class, majority_n = counts.most_common(1)[0]
    majority_acc = majority_n / len(test_labels)
    print(f"  Majority baseline: {majority_acc:.1%} ({majority_class})")

    # Create LLM classifier
    clf = LLMClassifier(
        task_description=(
            "Classify Eclipse IDE bug reports into their correct component. "
            "Each component is a part of the Eclipse IDE platform."
        ),
        class_names=all_classes,
        n_consistency=n_consistency,
        temperature=0.7,
    )

    # Set examples
    clf.set_examples(
        texts=[e["text"] for e in examples],
        labels=[e["label"] for e in examples],
        n_examples=len(examples),
        strategy="diverse",
    )

    # Run evaluation
    print(f"\n  Running LLM evaluation ({len(test_df)} samples x {n_consistency} consistency)...")
    start = time.time()
    results = clf.evaluate(
        texts=test_texts,
        true_labels=test_labels,
        n_samples=n_consistency,
        confidence_threshold=confidence_threshold,
    )
    elapsed = time.time() - start

    # Print results
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print("=" * 70)
    print(f"  Overall accuracy:   {results['overall_accuracy']:.1%}")
    print(f"  Confident accuracy: {results['confident_accuracy']:.1%}")
    print(f"  Coverage:           {results['coverage']:.1%}")
    print(f"  Deferred accuracy:  {results['deferred_accuracy']:.1%}")
    print(f"  Majority baseline:  {majority_acc:.1%}")
    print(f"  Lift over majority: {results['overall_accuracy'] - majority_acc:+.1%}")
    print(f"  N confident:        {results['n_confident']}")
    print(f"  N deferred:         {results['n_deferred']}")
    print(f"  Time:               {elapsed:.0f}s ({elapsed/len(test_df):.1f}s/sample)")
    print(f"  API calls:          {results['client_stats']['api_calls']}")
    print(f"  Cache hits:         {results['client_stats']['cache_hits']}")
    print(f"  Total tokens:       {results['client_stats']['total_tokens']:,}")

    # Coverage-accuracy curve
    print(f"\n  Coverage-accuracy curve:")
    print(f"  {'Threshold':>10} {'Coverage':>10} {'Accuracy':>10} {'N':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for row in results["curve"]:
        print(
            f"  {row['threshold']:10.2f} {row['coverage']:10.1%} "
            f"{row['accuracy']:10.1%} {row['n']:8d}"
        )

    # Top-5 classes by accuracy
    print(f"\n  Per-class accuracy (top 10):")
    sorted_classes = sorted(
        results["per_class"].items(),
        key=lambda x: x[1]["n"], reverse=True,
    )
    for cls, info in sorted_classes[:10]:
        print(f"    {cls:<35} {info['accuracy']:.1%} (n={info['n']}, conf={info['avg_confidence']:.0%})")

    # Save
    save_data = {
        "task": "Eclipse S2 Component (LLM)",
        "model": "accounts/fireworks/models/deepseek-v3p2",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "max_samples": max_samples,
            "n_consistency": n_consistency,
            "confidence_threshold": confidence_threshold,
            "n_few_shot": len(examples),
        },
        "majority_baseline": majority_acc,
        "overall_accuracy": results["overall_accuracy"],
        "confident_accuracy": results["confident_accuracy"],
        "coverage": results["coverage"],
        "n_confident": results["n_confident"],
        "n_deferred": results["n_deferred"],
        "time_seconds": elapsed,
        "curve": results["curve"],
        "client_stats": results["client_stats"],
    }
    path = OUTPUT_DIR / f"llm_results_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {path}")

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--n-consistency", type=int, default=5)
    p.add_argument("--confidence-threshold", type=float, default=0.6)
    args = p.parse_args()
    run(args.max_samples, args.n_consistency, args.confidence_threshold)
