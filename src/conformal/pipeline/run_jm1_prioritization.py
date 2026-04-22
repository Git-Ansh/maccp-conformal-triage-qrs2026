"""
JM1 Prioritization Cascade: 2-Stage XGBoost + RAG-LLM rescue.

Stage 0: XGBoost screens extremes (CRITICAL / MINIMAL)
Stage 1: XGBoost assigns tiers (HIGH / ELEVATED / LOW) with tree-variance confidence
LLM:     Only called on high-variance modules where trees disagree (~10%)

Usage:
    python src/conformal/pipeline/run_jm1_prioritization.py
    python src/conformal/pipeline/run_jm1_prioritization.py --llm-items 15
    python src/conformal/pipeline/run_jm1_prioritization.py --skip-llm
"""

import sys
import argparse
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from scipy.special import expit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from cascade.framework.confidence_stage import ConfidenceStage
from conformal.data.jm1_loader import load_jm1_data
from conformal.stages.jm1_config import JM1_CLASSES
from xgboost import XGBClassifier

OUTPUT_DIR = PROJECT_ROOT / 'conformal_outputs' / 'jm1'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUTPUT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TIER_BOUNDARIES = {'CRITICAL': 0.50, 'HIGH': 0.35, 'ELEVATED': 0.20, 'LOW': 0.10}


def compute_tree_variance(xgb_model, X, n_trees=200, window=50):
    """Measure how much the last `window` trees shift the prediction."""
    margins = np.zeros((len(X), n_trees))
    for i in range(n_trees):
        margins[:, i] = xgb_model.predict(X, iteration_range=(0, i + 1), output_margin=True)
    probs = expit(margins)
    return probs[:, -window:].std(axis=1)


def assign_tiers(p_def, tree_var, var_percentile=85):
    """Assign priority tiers with confidence check via tree variance."""
    n = len(p_def)
    tier = np.full(n, '', dtype='U10')
    confident = np.ones(n, dtype=bool)

    # Stage 0: extremes
    tier[p_def >= TIER_BOUNDARIES['CRITICAL']] = 'CRITICAL'
    tier[p_def < TIER_BOUNDARIES['LOW']] = 'MINIMAL'

    # Stage 1: uncertain band
    mid = (p_def >= TIER_BOUNDARIES['LOW']) & (p_def < TIER_BOUNDARIES['CRITICAL'])
    mid_p = p_def[mid]
    mid_var = tree_var[mid]

    sub_tier = np.where(mid_p >= TIER_BOUNDARIES['HIGH'], 'HIGH',
               np.where(mid_p >= TIER_BOUNDARIES['ELEVATED'], 'ELEVATED', 'LOW'))

    # Confidence: tree variance threshold on the middle band only
    if mid.sum() > 0:
        var_thresh = np.percentile(mid_var, var_percentile)
        sub_confident = mid_var <= var_thresh
    else:
        sub_confident = np.ones(mid.sum(), dtype=bool)

    tier[mid] = sub_tier
    confident[mid] = sub_confident

    # Extremes are always confident
    confident[p_def >= TIER_BOUNDARIES['CRITICAL']] = True
    confident[p_def < TIER_BOUNDARIES['LOW']] = True

    return tier, confident


def run_llm_on_subset(test_df, test_X, test_y, p_def, llm_indices,
                      train_X, train_y, feature_cols, n_items=15):
    """RAG-augmented LLM tier assignment on a subset."""
    from conformal.llm.fireworks_client import FireworksClient

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_X)
    test_scaled = scaler.transform(test_X)

    nn = NearestNeighbors(n_neighbors=7, metric='euclidean')
    nn.fit(train_scaled)

    np.random.seed(RANDOM_SEED)
    if len(llm_indices) > n_items:
        sample = np.random.choice(llm_indices, size=n_items, replace=False)
    else:
        sample = llm_indices

    client = FireworksClient()
    results = []

    for i, idx in enumerate(sample):
        distances, nbr_idx = nn.kneighbors(test_scaled[idx:idx + 1])
        nbrs = nbr_idx[0]
        n_def_nbrs = train_y[nbrs].sum()

        nbr_lines = []
        for j, ni in enumerate(nbrs):
            label = 'DEFECTIVE' if train_y[ni] == 1 else 'CLEAN'
            key = (f"loc={train_X[ni][feature_cols.index('loc')]:.0f}, "
                   f"v(g)={train_X[ni][feature_cols.index('v(g)')]:.0f}, "
                   f"ev(g)={train_X[ni][feature_cols.index('ev(g)')]:.0f}, "
                   f"e={train_X[ni][feature_cols.index('e')]:.0f}, "
                   f"b={train_X[ni][feature_cols.index('b')]:.2f}")
            nbr_lines.append(f"  #{j+1}: {key} -> {label}")

        module_metrics = ', '.join(
            f"{f}={test_X[idx][k]:.4g}" for k, f in enumerate(feature_cols))

        prompt = f"""You are prioritizing NASA software modules for code review based on defect likelihood.

Assign one of these priority tiers:
- HIGH: Most similar historical modules were defective. Metrics are in the high-risk range. Review soon.
- ELEVATED: Mixed evidence from similar modules. Some risk indicators present. Review if resources allow.
- LOW: Most similar historical modules were clean. Metrics are moderate/low. Deprioritize.

Context: ~20% of modules in this codebase have defects (base rate ~1.4/7 neighbors).
The ML model gave P(defective) = {p_def[idx]:.2f} but its internal trees disagree, so it is uncertain.

TARGET MODULE: {module_metrics}

7 most similar historical modules:
{chr(10).join(nbr_lines)}

{n_def_nbrs}/7 similar modules had defects.

Respond ONLY with: {{"priority": "HIGH"}}, {{"priority": "ELEVATED"}}, or {{"priority": "LOW"}}"""

        response = client.chat(prompt, temperature=0.0, seed=42)
        resp_lower = response.lower()

        if 'high' in resp_lower and 'elevated' not in resp_lower:
            llm_tier = 'HIGH'
        elif 'low' in resp_lower and 'elevated' not in resp_lower:
            llm_tier = 'LOW'
        else:
            llm_tier = 'ELEVATED'

        # What XGBoost would have assigned
        xgb_tier = ('HIGH' if p_def[idx] >= 0.35
                     else 'ELEVATED' if p_def[idx] >= 0.20
                     else 'LOW')

        true_label = test_y[idx]
        results.append({
            'idx': int(idx),
            'p_def': float(p_def[idx]),
            'n_def_nbrs': int(n_def_nbrs),
            'xgb_tier': xgb_tier,
            'llm_tier': llm_tier,
            'changed': xgb_tier != llm_tier,
            'true_label': int(true_label),
        })

        changed_str = f" CHANGED {xgb_tier}->{llm_tier}" if xgb_tier != llm_tier else ""
        print(f"  [{i+1}/{len(sample)}] P={p_def[idx]:.2f}, nbrs={n_def_nbrs}/7, "
              f"XGB={xgb_tier:>8}, LLM={llm_tier:>8}, "
              f"truth={'DEF' if true_label == 1 else 'CLN'}{changed_str}")

    return results


def plot_pipeline(p_def, test_y, tier, confident, llm_results=None):
    """Generate prioritization cascade plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Probability distribution with tier boundaries
    ax = axes[0]
    ax.hist(p_def[test_y == 0], bins=50, alpha=0.6, color='steelblue',
            label='Clean', density=True)
    ax.hist(p_def[test_y == 1], bins=50, alpha=0.6, color='coral',
            label='Defective', density=True)
    for name, thresh in TIER_BOUNDARIES.items():
        ax.axvline(thresh, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(thresh + 0.01, ax.get_ylim()[1] * 0.9, name, fontsize=7, rotation=90)
    ax.set_xlabel('P(defective)')
    ax.set_ylabel('Density')
    ax.set_title('Probability Distribution by True Label')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Cumulative gain chart
    ax = axes[1]
    tier_order = ['CRITICAL', 'HIGH', 'ELEVATED', 'LOW', 'MINIMAL']
    cum_modules = 0
    cum_defects = 0
    total_def = (test_y == 1).sum()
    total_mod = len(test_y)
    xs, ys = [0], [0]

    for t in tier_order:
        mask = tier == t
        n = mask.sum()
        n_def = test_y[mask].sum()
        cum_modules += n
        cum_defects += n_def
        xs.append(cum_modules / total_mod)
        ys.append(cum_defects / total_def)

    ax.plot(xs, ys, 'b-o', markersize=5, label='Cascade priority')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
    for t, x, y in zip(tier_order, xs[1:], ys[1:]):
        ax.annotate(t, (x, y), textcoords="offset points",
                    xytext=(5, -10), fontsize=7)
    ax.set_xlabel('Fraction of modules reviewed')
    ax.set_ylabel('Fraction of defects found')
    ax.set_title('Cumulative Gain Chart')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    # Plot 3: Tier summary bar chart
    ax = axes[2]
    tier_names = []
    tier_sizes = []
    tier_def_rates = []
    for t in tier_order:
        mask = tier == t
        if mask.sum() > 0:
            tier_names.append(t)
            tier_sizes.append(mask.sum())
            tier_def_rates.append(test_y[mask].mean())

    colors = ['#d32f2f', '#ff9800', '#ffc107', '#8bc34a', '#4caf50']
    x = np.arange(len(tier_names))
    bars = ax.bar(x, tier_def_rates, color=colors[:len(tier_names)], alpha=0.8,
                  edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(n={s})" for n, s in zip(tier_names, tier_sizes)],
                       fontsize=8)
    ax.set_ylabel('Defect Rate')
    ax.set_title('Defect Rate by Priority Tier')
    ax.axhline(test_y.mean(), color='black', linestyle='--', alpha=0.4,
               label=f'Base rate ({test_y.mean():.0%})')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bar, rate in zip(bars, tier_def_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{rate:.0%}', ha='center', fontsize=9, fontweight='bold')

    fig.suptitle('JM1 Prioritization Cascade', fontsize=13, y=1.02)
    fig.tight_layout()
    path = FIG_DIR / 'jm1_prioritization_cascade.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-llm', action='store_true')
    parser.add_argument('--llm-items', type=int, default=15)
    parser.add_argument('--var-percentile', type=int, default=85,
                        help='Percentile of tree variance above which -> LLM (default 85 = top 15%%)')
    args = parser.parse_args()

    print("=" * 60)
    print("JM1 PRIORITIZATION CASCADE")
    print("=" * 60)

    # Load data
    data = load_jm1_data()
    train_X = data['train_df'][data['feature_cols']].fillna(0).values
    train_y = data['train_df']['defective'].values
    test_X = data['test_df'][data['feature_cols']].fillna(0).values
    test_y = data['test_df']['defective'].values
    feature_cols = data['feature_cols']

    print(f"\nDataset: {len(train_y)} train, {len(test_y)} test")
    print(f"Defect rate: {test_y.mean():.1%}")

    # Train XGBoost
    print("\nTraining XGBoost...")
    xgb = XGBClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_SEED,
                         eval_metric='logloss', n_jobs=-1)
    xgb.fit(train_X, train_y)

    # Get calibrated probabilities via ConfidenceStage
    print("Calibrating probabilities...")
    stage = ConfidenceStage(name='S0', classes=JM1_CLASSES, target_accuracy=0.85)
    stage.fit(train_X, train_y, feature_names=feature_cols)
    preds = stage.predict(test_X, return_proba=True)
    p_def = preds['proba'][:, 1]

    # Compute tree variance
    print("Computing tree variance...")
    tree_var = compute_tree_variance(xgb, test_X)

    # Assign tiers
    tier, confident = assign_tiers(p_def, tree_var, var_percentile=args.var_percentile)
    llm_mask = ~confident

    # ================================================================
    # RESULTS
    # ================================================================
    print("\n" + "=" * 60)
    print("STAGE 0 + STAGE 1: TIER ASSIGNMENT")
    print("=" * 60)

    tier_order = ['CRITICAL', 'HIGH', 'ELEVATED', 'LOW', 'MINIMAL']
    cum_modules = 0
    cum_defects = 0
    total_def = (test_y == 1).sum()

    print(f"\n{'Tier':>10} {'N':>6} {'%':>6} {'Defects':>8} {'Def Rate':>9} {'Lift':>6} {'Cum Recall':>11} {'Source'}")
    print("-" * 80)

    for t in tier_order:
        mask = (tier == t) & confident
        n = mask.sum()
        n_def = test_y[mask].sum()
        def_rate = test_y[mask].mean() if n > 0 else 0
        lift = def_rate / test_y.mean() if test_y.mean() > 0 else 0
        cum_modules += n
        cum_defects += n_def
        cum_recall = cum_defects / total_def if total_def > 0 else 0
        print(f"{t:>10} {n:>6} {n/len(test_y):>6.1%} {n_def:>8} {def_rate:>9.0%} {lift:>6.1f}x {cum_recall:>11.0%}  XGBoost")

    # LLM bucket
    n_llm = llm_mask.sum()
    n_llm_def = test_y[llm_mask].sum()
    cum_modules += n_llm
    cum_defects += n_llm_def
    print(f"{'-> LLM':>10} {n_llm:>6} {n_llm/len(test_y):>6.1%} {n_llm_def:>8} "
          f"{test_y[llm_mask].mean():>9.0%} {'':>6} {cum_defects/total_def:>11.0%}  Trees disagree")
    print("-" * 80)
    print(f"{'TOTAL':>10} {len(test_y):>6} {'100%':>6} {total_def:>8}")

    auto_n = confident.sum()
    print(f"\nAuto-tiered: {auto_n}/{len(test_y)} ({auto_n/len(test_y):.1%})")
    print(f"LLM needed:  {n_llm}/{len(test_y)} ({n_llm/len(test_y):.1%})")

    # Gain chart numbers
    print("\nCUMULATIVE GAIN (review top tiers first):")
    cum_m = 0
    cum_d = 0
    for t in tier_order:
        mask = tier == t  # all modules in tier (confident + LLM)
        n = mask.sum()
        n_def = test_y[mask].sum()
        cum_m += n
        cum_d += n_def
        print(f"  Review through {t:>8}: {cum_m:>5}/{len(test_y)} modules "
              f"({cum_m/len(test_y):>5.1%}), "
              f"catch {cum_d}/{total_def} defects ({cum_d/total_def:.0%})")

    # ================================================================
    # LLM RESCUE
    # ================================================================
    llm_results = None
    if not args.skip_llm and n_llm > 0:
        print("\n" + "=" * 60)
        print(f"LLM RESCUE: {args.llm_items} items from {n_llm} uncertain modules")
        print("=" * 60)

        llm_indices = np.where(llm_mask)[0]
        llm_results = run_llm_on_subset(
            data['test_df'], test_X, test_y, p_def, llm_indices,
            train_X, train_y, feature_cols, n_items=args.llm_items)

        # Analyze LLM tier changes
        n_changed = sum(1 for r in llm_results if r['changed'])
        print(f"\nLLM changed tier for {n_changed}/{len(llm_results)} modules")

        # Did the changes improve ranking?
        for r in llm_results:
            if r['changed']:
                # Was the change correct?
                # A "correct" change: defective module moved UP, or clean module moved DOWN
                tier_rank = {'HIGH': 0, 'ELEVATED': 1, 'LOW': 2}
                xgb_rank = tier_rank[r['xgb_tier']]
                llm_rank = tier_rank[r['llm_tier']]
                if r['true_label'] == 1 and llm_rank < xgb_rank:
                    quality = 'GOOD (defective moved up)'
                elif r['true_label'] == 0 and llm_rank > xgb_rank:
                    quality = 'GOOD (clean moved down)'
                elif r['true_label'] == 1 and llm_rank > xgb_rank:
                    quality = 'BAD (defective moved down)'
                elif r['true_label'] == 0 and llm_rank < xgb_rank:
                    quality = 'BAD (clean moved up)'
                else:
                    quality = 'NEUTRAL'
                print(f"  {r['xgb_tier']:>8} -> {r['llm_tier']:>8}: "
                      f"truth={'DEF' if r['true_label'] == 1 else 'CLN'} | {quality}")
    elif args.skip_llm:
        print("\n[LLM skipped by --skip-llm]")

    # ================================================================
    # PLOTS
    # ================================================================
    print("\nGenerating plots...")
    plot_pipeline(p_def, test_y, tier, confident, llm_results)

    # ================================================================
    # SAVE RESULTS
    # ================================================================
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'JM1 (PROMISE)',
        'n_train': len(train_y),
        'n_test': len(test_y),
        'defect_rate': float(test_y.mean()),
        'var_percentile': args.var_percentile,
        'tiers': {},
        'auto_tiered_pct': float(auto_n / len(test_y)),
        'llm_needed_pct': float(n_llm / len(test_y)),
    }
    for t in tier_order:
        mask = tier == t
        results['tiers'][t] = {
            'n': int(mask.sum()),
            'defect_rate': float(test_y[mask].mean()) if mask.sum() > 0 else 0,
        }
    if llm_results:
        results['llm_rescue'] = llm_results

    path = OUTPUT_DIR / f'jm1_prioritization_{datetime.now():%Y%m%d_%H%M%S}.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {path}")


if __name__ == '__main__':
    main()
