"""
Quick LLM evaluation on pre-saved Eclipse test data.

Reads from conformal_outputs/eclipse/llm_test_subset.csv
No Eclipse JSON loading needed -- runs in seconds.
"""

import sys
import json
import time
import pandas as pd
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conformal.llm.fireworks_client import FireworksClient
from conformal.llm.llm_classifier import LLMClassifier

# Pre-saved test data
TEST_CSV = PROJECT_ROOT / "conformal_outputs" / "eclipse" / "llm_test_subset.csv"
OUTPUT_DIR = PROJECT_ROOT / "conformal_outputs" / "eclipse"

# Eclipse top-30 components + Other
ALL_CLASSES = [
    "UI", "Core", "Debug", "SWT", "Team", "Text", "Resources", "Ant",
    "cdt-core", "User Assistance", "Update (deprecated - use RT>Equinox>p2)",
    "CVS", "Runtime", "Build", "deprecated7", "cdt-parser", "APT",
    "Compare", "cdt-debug", "Releng", "cdt-build", "Search", "VCM",
    "deprecated8", "deprecated6", "cdt-debug-gdb", "WebDAV", "cdt-launch",
    "Scripting", "cdt-managedbuild", "Other",
]

# Hardcoded few-shot examples (one per major class, from training data)
FEW_SHOT = [
    {"text": "Usability issue with external editors (1GE6IRL)", "label": "Team"},
    {"text": "Need to see team stream label for a project set (1GL3SSQ)", "label": "VCM"},
    {"text": "VCM UI: warn about overwriting changes (1G8PD27)", "label": "Compare"},
    {"text": "Preferences > Fonts -- Windows system font changes are lost (1GF3KNH)", "label": "SWT"},
    {"text": "DCR: Console output of a launched program should be viewable", "label": "Debug"},
    {"text": "Source > Organize Imports doesn't add needed imports", "label": "Core"},
    {"text": "Project meta information should be consolidated (1GIRW80)", "label": "Resources"},
    {"text": "Plug-in Registry view should have option to sort alphabetically", "label": "UI"},
    {"text": "IDocument.replace(0, 0, text) does not trigger textChanged", "label": "Text"},
    {"text": "Ant buildfile editor should provide content assist", "label": "Ant"},
    {"text": "CDT parser fails to parse template specialization", "label": "cdt-parser"},
    {"text": "CDT core indexer performance issue with large projects", "label": "cdt-core"},
    {"text": "CDT debugger does not show local variables in some cases", "label": "cdt-debug"},
    {"text": "Search dialog should support regular expressions", "label": "Search"},
    {"text": "Build path entry is incorrect after rename refactoring", "label": "Build"},
    {"text": "Help system search returns no results for some queries", "label": "User Assistance"},
    {"text": "CDT managed build does not detect toolchain properly", "label": "cdt-build"},
    {"text": "Update manager fails to install features from remote site", "label": "Update (deprecated - use RT>Equinox>p2)"},
    {"text": "CVS commit fails silently on binary files", "label": "CVS"},
    {"text": "Runtime classloader issue with fragment bundles", "label": "Runtime"},
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50, help="Test samples")
    parser.add_argument("--n-consistency", type=int, default=3, help="Consistency runs")
    args = parser.parse_args()

    # Load test data
    df = pd.read_csv(TEST_CSV)
    df = df.head(args.n_samples)
    print(f"Test items: {len(df)}")

    texts = df["short_desc"].fillna("").tolist()
    labels = df["component_target"].fillna("Other").tolist()

    # Majority baseline
    counts = Counter(labels)
    majority = counts.most_common(1)[0]
    majority_acc = majority[1] / len(labels)
    print(f"Majority baseline: {majority_acc:.1%} ({majority[0]})")
    print(f"Classes in sample: {len(counts)}")

    # Create classifier
    clf = LLMClassifier(
        task_description=(
            "Classify Eclipse IDE bug reports into their correct component. "
            "Given a bug title, predict which Eclipse component this bug belongs to."
        ),
        class_names=ALL_CLASSES,
        n_consistency=args.n_consistency,
        temperature=0.7,
    )
    clf._examples = FEW_SHOT

    # Run evaluation
    print(f"\nClassifying {len(texts)} items ({args.n_consistency} consistency each)...")
    print(f"Expected API calls: {len(texts) * args.n_consistency}")
    t0 = time.time()

    results = clf.evaluate(
        texts=texts,
        true_labels=labels,
        n_samples=args.n_consistency,
        confidence_threshold=0.6,
    )
    elapsed = time.time() - t0

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS (n={len(texts)}, consistency={args.n_consistency})")
    print(f"{'='*60}")
    print(f"Overall accuracy:   {results['overall_accuracy']:.1%}")
    print(f"Confident accuracy: {results['confident_accuracy']:.1%}")
    print(f"Coverage:           {results['coverage']:.1%}")
    print(f"Deferred accuracy:  {results['deferred_accuracy']:.1%}")
    print(f"Majority baseline:  {majority_acc:.1%}")
    print(f"Lift over majority: {results['overall_accuracy'] - majority_acc:+.1%}")
    print(f"Time: {elapsed:.0f}s")
    print(f"API stats: {results['client_stats']}")

    print(f"\nCoverage-accuracy curve:")
    print(f"  {'Threshold':>10} {'Coverage':>10} {'Accuracy':>10} {'N':>6}")
    for row in results["curve"]:
        print(f"  {row['threshold']:10.1f} {row['coverage']:10.1%} {row['accuracy']:10.1%} {row['n']:6d}")

    # Save
    out = {
        "n_test": len(texts),
        "n_consistency": args.n_consistency,
        "majority_baseline": majority_acc,
        "results": {
            k: v for k, v in results.items()
            if k != "per_class"
        },
    }
    out_path = OUTPUT_DIR / "llm_quick_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
