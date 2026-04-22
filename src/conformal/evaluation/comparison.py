"""
Cross-dataset comparison for cascade evaluation.

Generates:
1. Side-by-side coverage-accuracy curves
2. Accuracy lift over majority baseline (normalized comparison)
3. Per-dataset cascade vs flat model comparison
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional


def plot_coverage_accuracy_comparison(
    datasets: Dict[str, pd.DataFrame],
    output_path: Optional[Path] = None,
    title: str = 'Coverage-Accuracy Curves: Cascade vs Flat Baseline',
):
    """
    Plot coverage-accuracy curves for multiple datasets side by side.

    Args:
        datasets: Dict of dataset_name -> DataFrame with threshold, coverage, accuracy
        output_path: Where to save the figure
        title: Plot title
    """
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4),
                             sharey=True, squeeze=False)

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    for idx, (name, curve_df) in enumerate(datasets.items()):
        ax = axes[0][idx]
        color = colors[idx % len(colors)]

        ax.plot(curve_df['coverage'], curve_df['accuracy'],
                'o-', color=color, markersize=3, label='Cascade')

        # Mark operating point if available
        if 'is_operating_point' in curve_df.columns:
            op = curve_df[curve_df['is_operating_point']]
            if len(op) > 0:
                ax.scatter(op['coverage'].values, op['accuracy'].values,
                          s=100, color='red', zorder=5, marker='*',
                          label='Operating point')

        ax.set_xlabel('Coverage')
        if idx == 0:
            ax.set_ylabel('Accuracy')
        ax.set_title(name)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0.4, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def plot_accuracy_lift_comparison(
    results: Dict[str, Dict],
    output_path: Optional[Path] = None,
):
    """
    Bar chart comparing accuracy lift over majority baseline across datasets.

    Args:
        results: Dict of dataset_name -> {stage_name -> {accuracy, majority_baseline, ...}}
        output_path: Where to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    datasets = list(results.keys())
    x = np.arange(len(datasets))
    width = 0.25

    # Collect stage names across all datasets
    all_stages = set()
    for r in results.values():
        all_stages.update(r.keys())
    stages = sorted(all_stages)

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    for i, stage in enumerate(stages):
        lifts = []
        for ds in datasets:
            stage_data = results[ds].get(stage, {})
            lift = stage_data.get('accuracy_lift', 0)
            lifts.append(lift * 100)  # Convert to percentage points

        ax.bar(x + i * width, lifts, width, label=stage,
               color=colors[i % len(colors)], alpha=0.8)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Accuracy Lift over Majority Baseline (pp)')
    ax.set_title('Cascade Accuracy Lift Across Datasets')
    ax.set_xticks(x + width * (len(stages) - 1) / 2)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def print_cross_dataset_summary(all_results: Dict[str, Dict]):
    """Print a cross-dataset summary table."""
    print("\n" + "=" * 90)
    print("CROSS-DATASET CASCADE COMPARISON")
    print("=" * 90)

    print(f"\n{'Dataset':<20} {'Stage':<20} {'Accuracy':>10} {'Coverage':>10} "
          f"{'Majority':>10} {'Lift':>8}")
    print("-" * 90)

    for ds_name, ds_results in all_results.items():
        stages = ds_results.get('stages', {})
        for stage_name, stage_data in stages.items():
            acc = stage_data.get('accuracy', 0)
            cov = stage_data.get('coverage', 0)
            maj = stage_data.get('majority_baseline', 0)
            lift = stage_data.get('accuracy_lift', 0)
            print(f"{ds_name:<20} {stage_name:<20} {acc:>10.1%} {cov:>10.1%} "
                  f"{maj:>10.1%} {lift:>+8.1%}")

    print("=" * 90)
