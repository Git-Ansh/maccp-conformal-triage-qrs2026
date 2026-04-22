"""
experiment6_comparison.py -- Final comparison across all model ablation experiments.

Loads results from ../results/*/ and builds a master comparison table.
Prints formatted tables and generates a one-paragraph summary.

Usage:
    python experiment6_comparison.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

BASE_DIR = os.path.join(SCRIPT_DIR, "..")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

DATASETS = ["eclipse", "mozilla"]
ALPHA_LEVELS = [0.05, 0.10, 0.20]


def load_json_safe(filepath):
    """Load JSON file, return None if not found."""
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        return json.load(f)


def collect_all_results():
    """Collect results from all experiment directories.

    Returns
    -------
    dict
        Nested dict with structure:
        {experiment_name: {dataset: results_dict}}
    """
    all_results = {}

    # Experiment 1: Hybrid MACCP
    hybrid_results = load_json_safe(os.path.join(RESULTS_DIR, "hybrid", "hybrid_results.json"))
    if hybrid_results:
        all_results["hybrid"] = hybrid_results

    # Experiment 2: LLM zero-shot
    llm_results = load_json_safe(os.path.join(RESULTS_DIR, "llm_zeroshot", "all_llm_results.json"))
    if llm_results:
        all_results["llm_zeroshot"] = llm_results

    # Experiment 3: DeBERTa-large
    for dataset in DATASETS:
        path = os.path.join(RESULTS_DIR, "deberta_large", f"{dataset}_deberta_large.json")
        res = load_json_safe(path)
        if res:
            all_results.setdefault("deberta_large", {})[dataset] = res

    # Experiment 4: Qwen32B
    for dataset in DATASETS:
        path = os.path.join(RESULTS_DIR, "qwen32b", f"{dataset}_qwen32b.json")
        res = load_json_safe(path)
        if res:
            all_results.setdefault("qwen32b", {})[dataset] = res

    # Experiment 5: Standalone inference (may overlap with 3/4)
    for model in ["deberta_large", "qwen32b"]:
        for dataset in DATASETS:
            path = os.path.join(RESULTS_DIR, model, f"{dataset}_inference.json")
            res = load_json_safe(path)
            if res:
                key = f"{model}_inference"
                all_results.setdefault(key, {})[dataset] = res

    return all_results


def print_accuracy_table(all_results):
    """Print the master accuracy comparison table."""
    print("\n" + "=" * 90)
    print("TABLE 1: TEST ACCURACY COMPARISON")
    print("=" * 90)

    header = f"  {'Model':<25s} | {'Eclipse':>10s} | {'Mozilla':>10s} | {'Source':>15s}"
    print(header)
    print("  " + "-" * 70)

    # Collect accuracy rows
    rows = []

    # From hybrid results
    if "hybrid" in all_results:
        for dataset in DATASETS:
            if dataset in all_results["hybrid"]:
                meta = all_results["hybrid"][dataset].get("meta", {})
                if dataset == "eclipse":
                    rows.append(("DeBERTa-base", meta.get("deberta_accuracy"), dataset, "hybrid"))
                    rows.append(("XGBoost", meta.get("xgb_accuracy"), dataset, "hybrid"))
                    rows.append(("Majority baseline", meta.get("majority_accuracy"), dataset, "hybrid"))

    # From DeBERTa-large
    if "deberta_large" in all_results:
        for dataset in DATASETS:
            if dataset in all_results["deberta_large"]:
                res = all_results["deberta_large"][dataset]
                rows.append(("DeBERTa-large", res.get("test_accuracy"), dataset, "deberta_large"))

    # From Qwen32B
    if "qwen32b" in all_results:
        for dataset in DATASETS:
            if dataset in all_results["qwen32b"]:
                res = all_results["qwen32b"][dataset]
                rows.append(("Qwen2.5-32B (LoRA)", res.get("test_accuracy"), dataset, "qwen32b"))

    # From LLM zero-shot
    if "llm_zeroshot" in all_results:
        llm = all_results["llm_zeroshot"]
        for model_name, display_name in [("deepseek-v3", "DeepSeek-v3 (0-shot)"), ("glm-5", "GLM-5 (0-shot)")]:
            for dataset in DATASETS:
                key = f"{model_name}_{dataset}"
                if key in llm:
                    rows.append((display_name, llm[key].get("test_accuracy"), dataset, "llm_zeroshot"))

    # Pivot into a table
    model_accs = {}  # model_name -> {eclipse: acc, mozilla: acc}
    model_sources = {}
    for name, acc, dataset, source in rows:
        if name not in model_accs:
            model_accs[name] = {}
            model_sources[name] = source
        if acc is not None:
            model_accs[name][dataset] = acc

    # Print in a consistent order
    model_order = [
        "Majority baseline", "XGBoost", "DeBERTa-base",
        "DeBERTa-large", "Qwen2.5-32B (LoRA)",
        "DeepSeek-v3 (0-shot)", "GLM-5 (0-shot)",
    ]

    for model_name in model_order:
        if model_name in model_accs:
            accs = model_accs[model_name]
            ecl = f"{accs['eclipse']:.4f}" if "eclipse" in accs else "   --"
            moz = f"{accs['mozilla']:.4f}" if "mozilla" in accs else "   --"
            source = model_sources.get(model_name, "")
            print(f"  {model_name:<25s} | {ecl:>10s} | {moz:>10s} | {source:>15s}")

    # Print any models not in the predefined order
    for model_name in model_accs:
        if model_name not in model_order:
            accs = model_accs[model_name]
            ecl = f"{accs['eclipse']:.4f}" if "eclipse" in accs else "   --"
            moz = f"{accs['mozilla']:.4f}" if "mozilla" in accs else "   --"
            source = model_sources.get(model_name, "")
            print(f"  {model_name:<25s} | {ecl:>10s} | {moz:>10s} | {source:>15s}")

    return model_accs


def print_agreement_table(all_results):
    """Print agreement rates between model pairs."""
    print("\n" + "=" * 90)
    print("TABLE 2: CROSS-MODEL AGREEMENT RATES (TEST SET)")
    print("=" * 90)

    header = f"  {'Model Pair':<35s} | {'Eclipse':>10s} | {'Mozilla':>10s}"
    print(header)
    print("  " + "-" * 60)

    # From hybrid results: DeBERTa-base vs XGBoost
    if "hybrid" in all_results:
        for dataset in DATASETS:
            if dataset in all_results["hybrid"]:
                meta = all_results["hybrid"][dataset].get("meta", {})
                rate = meta.get("test_agreement_rate")
                if rate is not None:
                    print(f"  {'DeBERTa-base vs XGBoost':<35s} | {rate:10.3f} |")

    # From DeBERTa-large
    if "deberta_large" in all_results:
        pairs = [
            ("agreement_large_xgb_test", "DeBERTa-large vs XGBoost"),
            ("agreement_large_base_test", "DeBERTa-large vs DeBERTa-base"),
        ]
        for key, label in pairs:
            vals = {}
            for dataset in DATASETS:
                if dataset in all_results["deberta_large"]:
                    v = all_results["deberta_large"][dataset].get(key)
                    if v is not None:
                        vals[dataset] = v
            if vals:
                ecl = f"{vals['eclipse']:.3f}" if "eclipse" in vals else "   --"
                moz = f"{vals['mozilla']:.3f}" if "mozilla" in vals else "   --"
                print(f"  {label:<35s} | {ecl:>10s} | {moz:>10s}")

    # From Qwen32B
    if "qwen32b" in all_results:
        pairs = [
            ("agreement_qwen_xgb_test", "Qwen32B vs XGBoost"),
            ("agreement_qwen_deberta_test", "Qwen32B vs DeBERTa-base"),
        ]
        for key, label in pairs:
            vals = {}
            for dataset in DATASETS:
                if dataset in all_results["qwen32b"]:
                    v = all_results["qwen32b"][dataset].get(key)
                    if v is not None:
                        vals[dataset] = v
            if vals:
                ecl = f"{vals['eclipse']:.3f}" if "eclipse" in vals else "   --"
                moz = f"{vals['mozilla']:.3f}" if "mozilla" in vals else "   --"
                print(f"  {label:<35s} | {ecl:>10s} | {moz:>10s}")


def print_maccp_table(all_results):
    """Print MACCP comparison at alpha=0.10."""
    print("\n" + "=" * 90)
    print("TABLE 3: MACCP COMPARISON (alpha=0.10)")
    print("=" * 90)

    alpha = 0.10

    for dataset in DATASETS:
        print(f"\n  --- {dataset.upper()} ---")
        header = (
            f"  {'Configuration':<35s} | "
            f"{'Cov':>6s} | {'Size':>6s} | {'Sing%':>6s} | {'SingAcc':>7s} | "
            f"{'q_agree':>7s} | {'q_disag':>7s}"
        )
        print(header)
        print("  " + "-" * 90)

        # Hybrid results
        if "hybrid" in all_results and dataset in all_results["hybrid"]:
            ds_res = all_results["hybrid"][dataset]

            # Baselines
            for model_key, label in [("deberta", "DeBERTa-base RAPS"), ("xgb", "XGBoost RAPS")]:
                bkey = f"{model_key}_alpha{alpha}"
                if bkey in ds_res.get("baselines", {}):
                    b = ds_res["baselines"][bkey]
                    print(
                        f"  {label:<35s} | "
                        f"{b['coverage']:6.3f} | {b['mean_set_size']:6.2f} | "
                        f"{b['singleton_rate']:6.3f} | {b['singleton_accuracy']:7.3f} | "
                        f"{'':>7s} | {'':>7s}"
                    )

            # MACCP configs
            for config_name in ["A: DeBERTa/DeBERTa", "B: XGBoost/XGBoost",
                                 "C: DeBERTa/XGBoost", "D: XGBoost/DeBERTa"]:
                ckey = f"{config_name}_alpha{alpha}"
                if ckey in ds_res.get("configs", {}):
                    r = ds_res["configs"][ckey]["overall"]
                    print(
                        f"  {config_name:<35s} | "
                        f"{r['coverage']:6.3f} | {r['mean_set_size']:6.2f} | "
                        f"{r['singleton_rate']:6.3f} | {r['singleton_accuracy']:7.3f} | "
                        f"{r.get('q_agree', 0):7.3f} | {r.get('q_disagree', 0):7.3f}"
                    )

        # DeBERTa-large MACCP
        if "deberta_large" in all_results and dataset in all_results["deberta_large"]:
            maccp = all_results["deberta_large"][dataset].get("maccp", {})
            for config_name in ["A: Large/Large", "C: Large/XGBoost"]:
                ckey = f"{config_name}_alpha{alpha}"
                if ckey in maccp:
                    r = maccp[ckey]["overall"]
                    label = config_name.replace("Large", "DeBERTa-L")
                    print(
                        f"  {label:<35s} | "
                        f"{r['coverage']:6.3f} | {r['mean_set_size']:6.2f} | "
                        f"{r['singleton_rate']:6.3f} | {r['singleton_accuracy']:7.3f} | "
                        f"{r.get('q_agree', 0):7.3f} | {r.get('q_disagree', 0):7.3f}"
                    )

        # Qwen32B MACCP
        if "qwen32b" in all_results and dataset in all_results["qwen32b"]:
            maccp = all_results["qwen32b"][dataset].get("maccp", {})
            for config_name in ["A: Qwen/Qwen", "C: Qwen/XGBoost"]:
                ckey = f"{config_name}_alpha{alpha}"
                if ckey in maccp:
                    r = maccp[ckey]["overall"]
                    print(
                        f"  {config_name:<35s} | "
                        f"{r['coverage']:6.3f} | {r['mean_set_size']:6.2f} | "
                        f"{r['singleton_rate']:6.3f} | {r['singleton_accuracy']:7.3f} | "
                        f"{r.get('q_agree', 0):7.3f} | {r.get('q_disagree', 0):7.3f}"
                    )


def generate_summary(all_results, model_accs):
    """Generate a one-paragraph text summary of key findings."""
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    findings = []

    # Best model per dataset
    for dataset in DATASETS:
        if dataset in model_accs.get("DeBERTa-base", {}):
            base_acc = model_accs["DeBERTa-base"][dataset]
            best_name = "DeBERTa-base"
            best_acc = base_acc

            for model_name, accs in model_accs.items():
                if dataset in accs and accs[dataset] > best_acc:
                    best_acc = accs[dataset]
                    best_name = model_name

            if best_name != "DeBERTa-base":
                improvement = best_acc - base_acc
                findings.append(
                    f"On {dataset}, {best_name} ({best_acc:.1%}) improves over "
                    f"DeBERTa-base ({base_acc:.1%}) by {improvement:+.1%}."
                )
            else:
                findings.append(
                    f"On {dataset}, DeBERTa-base ({best_acc:.1%}) remains the strongest model."
                )

    # Agreement insight
    if "hybrid" in all_results:
        for dataset in DATASETS:
            if dataset in all_results["hybrid"]:
                meta = all_results["hybrid"][dataset].get("meta", {})
                agree_rate = meta.get("test_agreement_rate")
                if agree_rate is not None:
                    findings.append(
                        f"{dataset} DeBERTa-XGBoost agreement: {agree_rate:.1%}."
                    )

    # MACCP insight at alpha=0.10
    alpha = 0.10
    if "hybrid" in all_results:
        for dataset in DATASETS:
            if dataset in all_results["hybrid"]:
                configs = all_results["hybrid"][dataset].get("configs", {})
                # Compare config A vs C
                a_key = f"A: DeBERTa/DeBERTa_alpha{alpha}"
                c_key = f"C: DeBERTa/XGBoost_alpha{alpha}"
                if a_key in configs and c_key in configs:
                    a_size = configs[a_key]["overall"]["mean_set_size"]
                    c_size = configs[c_key]["overall"]["mean_set_size"]
                    a_sing = configs[a_key]["overall"]["singleton_rate"]
                    c_sing = configs[c_key]["overall"]["singleton_rate"]
                    findings.append(
                        f"{dataset} MACCP hybrid (alpha={alpha}): "
                        f"Config C set size {c_size:.1f} vs A {a_size:.1f}, "
                        f"singleton rate {c_sing:.1%} vs {a_sing:.1%}."
                    )

    if findings:
        print("\n  " + " ".join(findings))
    else:
        print("\n  No results available to summarise. Run experiments 1-5 first.")

    print()


def main():
    print("=" * 90)
    print("EXPERIMENT 6: FINAL COMPARISON ACROSS ALL MODEL ABLATION EXPERIMENTS")
    print("=" * 90)

    all_results = collect_all_results()

    # Report what was found
    found = list(all_results.keys())
    print(f"\n  Found results from: {found}")

    if not found:
        print("  No results found in", RESULTS_DIR)
        print("  Run experiments 1-5 first.")
        sys.exit(0)

    model_accs = print_accuracy_table(all_results)
    print_agreement_table(all_results)
    print_maccp_table(all_results)
    generate_summary(all_results, model_accs)

    # Save combined summary
    summary_path = os.path.join(RESULTS_DIR, "comparison_summary.json")
    summary = {
        "experiments_found": found,
        "model_accuracies": {
            name: {ds: float(v) for ds, v in accs.items()}
            for name, accs in model_accs.items()
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to: {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
