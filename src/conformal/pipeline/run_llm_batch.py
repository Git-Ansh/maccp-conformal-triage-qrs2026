"""
Batch LLM evaluation on Eclipse component assignment.

Sends ALL test items in a single prompt (or small number of batches)
instead of one API call per item. Much faster and cheaper.

For confidence: uses 3 batch calls with temperature=0.7,
measures per-item agreement across the 3 runs.
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conformal.llm.fireworks_client import FireworksClient

TEST_CSV = PROJECT_ROOT / "conformal_outputs" / "eclipse" / "llm_test_subset.csv"
OUTPUT_DIR = PROJECT_ROOT / "conformal_outputs" / "eclipse"

ALL_CLASSES = [
    "UI", "Core", "Debug", "SWT", "Team", "Text", "Resources", "Ant",
    "cdt-core", "User Assistance", "CVS", "Runtime", "Build", "Releng",
    "cdt-parser", "APT", "Compare", "cdt-debug", "cdt-build", "Search",
    "VCM", "cdt-debug-gdb", "cdt-launch", "cdt-managedbuild",
    "User Assistance", "Other",
]

FEW_SHOT_TEXT = """Examples of bug titles and their components:
- "Preferences > Fonts -- system font changes lost" -> SWT
- "DCR: Console output of launched program should be viewable" -> Debug
- "Source > Organize Imports doesn't add needed imports" -> Core
- "IDocument.replace(0,0,text) does not trigger textChanged" -> Text
- "Ant buildfile editor should provide content assist" -> Ant
- "CDT parser fails to parse template specialization" -> cdt-parser
- "Search dialog should support regular expressions" -> Search
- "Help system search returns no results" -> User Assistance
- "CVS commit fails silently on binary files" -> CVS
- "Runtime classloader issue with fragment bundles" -> Runtime
- "Plug-in Registry view should sort alphabetically" -> UI
- "Project meta information should be consolidated" -> Resources
- "VCM UI: warn about overwriting changes" -> Compare
- "Need to see team stream label for a project set" -> Team
- "Build path entry incorrect after rename refactoring" -> Build"""


def build_batch_prompt(bug_titles: list, batch_id: int) -> str:
    """Build a single prompt that classifies ALL bugs at once."""
    class_list = ", ".join(sorted(set(ALL_CLASSES)))

    bugs_text = "\n".join(
        f'{i+1}. "{title}"' for i, title in enumerate(bug_titles)
    )

    return f"""{FEW_SHOT_TEXT}

Valid components: [{class_list}]

Now classify each of these {len(bug_titles)} bugs. For EACH bug, predict which Eclipse component it belongs to.

{bugs_text}

Respond with a JSON array of objects, one per bug, in order:
[{{"id": 1, "class": "..."}}, {{"id": 2, "class": "..."}}, ...]

IMPORTANT: Return ONLY the JSON array. Include ALL {len(bug_titles)} bugs."""


def parse_batch_response(response: str, n_expected: int) -> list:
    """Parse the batch JSON response into a list of predictions."""
    # Find JSON array
    start = response.find("[")
    end = response.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            parsed = json.loads(response[start:end])
            predictions = []
            for item in parsed:
                cls = item.get("class", "Other")
                # Validate
                if cls not in ALL_CLASSES:
                    # Fuzzy match
                    cls_lower = {c.lower(): c for c in ALL_CLASSES}
                    cls = cls_lower.get(cls.lower(), "Other")
                predictions.append(cls)
            return predictions
        except json.JSONDecodeError:
            pass

    # Fallback: try to extract line by line
    predictions = []
    lines = response.split("\n")
    for line in lines:
        for cls in ALL_CLASSES:
            if cls.lower() in line.lower():
                predictions.append(cls)
                break

    # Pad or trim to expected length
    while len(predictions) < n_expected:
        predictions.append("Other")
    return predictions[:n_expected]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--n-consistency", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=25,
                        help="Items per API call (25 fits in 4096 output tokens)")
    args = parser.parse_args()

    df = pd.read_csv(TEST_CSV).head(args.n_samples)
    texts = df["short_desc"].fillna("").tolist()
    labels = df["component_target"].fillna("Other").tolist()
    n = len(texts)

    counts = Counter(labels)
    majority = counts.most_common(1)[0]
    majority_acc = majority[1] / n
    print(f"Test items: {n}")
    print(f"Majority baseline: {majority_acc:.1%} ({majority[0]})")
    print(f"Batches: {(n + args.batch_size - 1) // args.batch_size} x {args.n_consistency} consistency = "
          f"{((n + args.batch_size - 1) // args.batch_size) * args.n_consistency} API calls total")

    client = FireworksClient()

    # Split into batches
    batches = []
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        batches.append(texts[start:end])

    # Run n_consistency times for confidence estimation
    all_runs = []  # list of lists of predictions
    t0 = time.time()

    for run_idx in range(args.n_consistency):
        run_preds = []
        for batch_idx, batch in enumerate(batches):
            prompt = build_batch_prompt(batch, batch_idx)

            # Use different seed per run for diversity
            if run_idx == 0:
                temp = 0.0
                seed = 42
            else:
                temp = 0.7
                seed = run_idx

            print(f"  Run {run_idx+1}/{args.n_consistency}, "
                  f"batch {batch_idx+1}/{len(batches)} "
                  f"({len(batch)} items)...", end=" ", flush=True)

            resp = client.chat(
                prompt=prompt,
                system="You classify Eclipse IDE bug reports into components. Output ONLY JSON.",
                temperature=temp,
                max_tokens=4000,  # Fireworks limit is 4096 for non-streaming
                seed=seed,
                use_cache=True,
            )
            preds = parse_batch_response(resp, len(batch))
            run_preds.extend(preds)
            print(f"got {len(preds)} predictions")

        all_runs.append(run_preds)

    elapsed = time.time() - t0

    # Compute per-item majority vote and confidence
    predictions = []
    confidences = []
    for i in range(n):
        item_preds = [run[i] for run in all_runs]
        counter = Counter(item_preds)
        best, best_count = counter.most_common(1)[0]
        predictions.append(best)
        confidences.append(best_count / len(item_preds))

    # Evaluate
    correct = sum(p == t for p, t in zip(predictions, labels))
    overall_acc = correct / n

    # At different confidence thresholds
    print(f"\n{'='*60}")
    print(f"RESULTS: LLM (DeepSeek V3) on Eclipse Component Assignment")
    print(f"{'='*60}")
    print(f"Overall accuracy: {overall_acc:.1%} ({correct}/{n})")
    print(f"Majority baseline: {majority_acc:.1%}")
    print(f"Lift: {overall_acc - majority_acc:+.1%}")
    print(f"Time: {elapsed:.0f}s")
    print(f"API calls: {client.stats()}")

    # Coverage-accuracy at different thresholds
    print(f"\nCoverage-accuracy curve:")
    print(f"  {'Threshold':>10} {'Coverage':>10} {'Accuracy':>10} {'N':>6}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
    curve = []
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        mask = [c >= t for c in confidences]
        n_conf = sum(mask)
        if n_conf > 0:
            conf_correct = sum(
                p == l for p, l, m in zip(predictions, labels, mask) if m
            )
            conf_acc = conf_correct / n_conf
            cov = n_conf / n
            print(f"  {t:10.1f} {cov:10.1%} {conf_acc:10.1%} {n_conf:6d}")
            curve.append({"threshold": t, "coverage": cov, "accuracy": conf_acc, "n": n_conf})

    # Per-class breakdown (top 5)
    print(f"\nPer-class accuracy (top classes):")
    for cls, cnt in counts.most_common(8):
        cls_mask = [l == cls for l in labels]
        cls_correct = sum(p == l for p, l, m in zip(predictions, labels, cls_mask) if m)
        cls_n = sum(cls_mask)
        print(f"  {cls:<25} {cls_correct}/{cls_n} ({cls_correct/cls_n:.0%})")

    # Sample wrong predictions
    print(f"\nSample wrong predictions:")
    wrong_count = 0
    for i in range(n):
        if predictions[i] != labels[i] and wrong_count < 5:
            print(f"  True={labels[i]}, Pred={predictions[i]} ({confidences[i]:.0%}): {texts[i][:60]}...")
            wrong_count += 1

    # Save results
    results = {
        "n_test": n,
        "n_consistency": args.n_consistency,
        "majority_baseline": float(majority_acc),
        "overall_accuracy": float(overall_acc),
        "lift": float(overall_acc - majority_acc),
        "curve": curve,
        "api_stats": client.stats(),
        "time_seconds": elapsed,
    }
    out_path = OUTPUT_DIR / "llm_batch_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
