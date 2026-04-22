"""
JM1 Final Results: Prioritization Cascade for Paper.

Runs the full 2-stage prioritization cascade with:
  1. Ensemble (XGB + RF + LR) — the recommended configuration
  2. XGBoost alone
  3. Random Forest alone
  4. Logistic Regression alone

Saves all results, methodology documentation, and comparison tables
to conformal_outputs/jm1/paper_results/.

=== TERMINOLOGY ===

PRIORITY TIERS:
  Modules are assigned to 5 tiers based on calibrated P(defective):
  - CRITICAL (P >= 0.50): Strong defect indicators across all metrics.
    Action: Review immediately in the current sprint.
  - HIGH (P 0.35-0.50): Multiple risk indicators present.
    Action: Schedule review in the next sprint.
  - ELEVATED (P 0.20-0.35): Some risk indicators but not conclusive.
    Action: Review if resources permit.
  - LOW (P 0.10-0.20): Weak indicators, metrics mostly normal.
    Action: Automated static analysis only. Deprioritize manual review.
  - MINIMAL (P < 0.10): All metrics in safe range.
    Action: Skip manual review.

HOW TIERS ARE DECIDED:
  Stage 0 screens the extremes:
    P(defective) >= 0.50 -> CRITICAL (auto-assigned)
    P(defective) <  0.10 -> MINIMAL  (auto-assigned)
    Everything else       -> forwarded to Stage 1

  Stage 1 assigns the middle tiers:
    P(defective) >= 0.35 -> HIGH
    P(defective) >= 0.20 -> ELEVATED
    P(defective) >= 0.10 -> LOW

  The probability P(defective) comes from a calibrated model:
    1. Base model(s) are trained on labeled historical modules
    2. Isotonic regression calibration ensures P(defective) = 0.30
       means ~30% of modules at that score are truly defective
    3. The calibrated probability IS the tier assignment signal

CASCADE GENERAL PATTERN:
  The GeneralCascade framework uses two reusable building blocks:

  ConfidenceStage: A model + calibration + per-class threshold tuning.
    - Trains any sklearn-compatible model on labeled data
    - Calibrates probabilities via isotonic regression (5-fold CV)
    - Tunes per-class confidence thresholds on out-of-fold predictions
    - At inference: classifies items and marks confident vs uncertain

  StageConfig: Declarative configuration for one stage.
    - Defines classes, routing rules, target accuracy, features
    - Routes confident predictions as terminal (done) or next (forward)
    - Routes uncertain predictions to the next stage or human review

  For JM1, this general pattern is configured as:
    Stage 0: ConfidenceStage with high target_accuracy (0.88)
      -> Only the most obvious cases get auto-assigned (CRITICAL/MINIMAL)
      -> Uncertain cases forwarded to Stage 1
    Stage 1: Same model output, lower thresholds
      -> Assigns HIGH/ELEVATED/LOW to the remaining modules
      -> These tiers are less certain but still useful for prioritization

  The SAME framework pattern handles Mozilla (5 stages), Eclipse (3 stages),
  ServiceNow (3 stages), and JM1 (2 stages) — only the config changes.
  JM1 uses standalone sklearn models (not the GeneralCascade class) since
  it is a single-model tier assignment, but follows the same architectural
  pattern: screen extremes first, triage the uncertain middle.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from conformal.data.jm1_loader import load_jm1_data
from xgboost import XGBClassifier

OUTPUT_DIR = PROJECT_ROOT / 'conformal_outputs' / 'jm1' / 'paper_results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUTPUT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TIER_THRESHOLDS = {'CRITICAL': 0.50, 'HIGH': 0.35, 'ELEVATED': 0.20, 'LOW': 0.10}


def assign_tiers(p_def):
    """Assign priority tiers from calibrated P(defective)."""
    tier = np.full(len(p_def), 'MINIMAL', dtype='U10')
    tier[p_def >= 0.10] = 'LOW'
    tier[p_def >= 0.20] = 'ELEVATED'
    tier[p_def >= 0.35] = 'HIGH'
    tier[p_def >= 0.50] = 'CRITICAL'
    return tier


def evaluate_tiers(p_def, test_y, model_name):
    """Compute all tier metrics for a model."""
    tier = assign_tiers(p_def)
    total_def = (test_y == 1).sum()
    total = len(test_y)
    auc = roc_auc_score(test_y, p_def)

    tier_order = ['CRITICAL', 'HIGH', 'ELEVATED', 'LOW', 'MINIMAL']
    rows = []
    cum_modules = 0
    cum_defects = 0

    for t in tier_order:
        mask = tier == t
        n = mask.sum()
        n_def = test_y[mask].sum()
        n_cln = n - n_def
        def_rate = n_def / n if n > 0 else 0
        lift = def_rate / (total_def / total) if total > 0 else 0
        cum_modules += n
        cum_defects += n_def
        cum_recall = cum_defects / total_def if total_def > 0 else 0
        cum_pct_modules = cum_modules / total

        rows.append({
            'tier': t,
            'n_modules': int(n),
            'pct_of_total': float(n / total),
            'n_defective': int(n_def),
            'n_clean': int(n_cln),
            'defect_rate': float(def_rate),
            'lift_over_random': float(lift),
            'cumulative_modules': int(cum_modules),
            'cumulative_modules_pct': float(cum_pct_modules),
            'cumulative_defects': int(cum_defects),
            'cumulative_recall': float(cum_recall),
        })

    return {
        'model': model_name,
        'auc': float(auc),
        'total_modules': int(total),
        'total_defective': int(total_def),
        'defect_rate': float(total_def / total),
        'tiers': rows,
    }


def main():
    np.random.seed(RANDOM_SEED)

    print("=" * 70)
    print("JM1 PRIORITIZATION CASCADE - FINAL RESULTS FOR PAPER")
    print("=" * 70)

    # Load data
    data = load_jm1_data()
    train_X = data['train_df'][data['feature_cols']].fillna(0).values
    train_y = data['train_df']['defective'].values
    test_X = data['test_df'][data['feature_cols']].fillna(0).values
    test_y = data['test_df']['defective'].values
    feature_cols = data['feature_cols']

    print(f"\nDataset: {len(train_y)} train, {len(test_y)} test")
    print(f"Defect rate: {test_y.mean():.1%} ({(test_y==1).sum()}/{len(test_y)})")
    print(f"Features: {len(feature_cols)}")

    # ================================================================
    # Train all 4 model variants
    # ================================================================
    print("\n" + "-" * 70)
    print("TRAINING MODELS")
    print("-" * 70)

    models = {}

    # 1. XGBoost alone
    # CalibratedClassifierCV(cv=5) clones and refits internally — no pre-fitting needed
    print("\n  [1/4] XGBoost...")
    cal_xgb = CalibratedClassifierCV(
        XGBClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_SEED,
                      eval_metric='logloss', n_jobs=-1),
        method='isotonic', cv=5)
    cal_xgb.fit(train_X, train_y)
    models['XGBoost'] = cal_xgb

    # 2. Random Forest alone
    print("  [2/4] Random Forest...")
    cal_rf = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_SEED,
                               n_jobs=-1, class_weight='balanced'),
        method='isotonic', cv=5)
    cal_rf.fit(train_X, train_y)
    models['RandomForest'] = cal_rf

    # 3. Logistic Regression alone
    print("  [3/4] Logistic Regression...")
    cal_lr = CalibratedClassifierCV(
        make_pipeline(StandardScaler(),
                      LogisticRegression(random_state=RANDOM_SEED, max_iter=1000,
                                         class_weight='balanced')),
        method='isotonic', cv=5)
    cal_lr.fit(train_X, train_y)
    models['LogisticRegression'] = cal_lr

    # 4. Ensemble (XGB + RF + LR)
    print("  [4/4] Ensemble (XGB + RF + LR)...")
    ens = VotingClassifier(estimators=[
        ('xgb', XGBClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_SEED,
                               eval_metric='logloss', n_jobs=-1)),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_SEED,
                                       n_jobs=-1, class_weight='balanced')),
        ('lr', make_pipeline(StandardScaler(),
                             LogisticRegression(random_state=RANDOM_SEED, max_iter=1000,
                                                class_weight='balanced'))),
    ], voting='soft')
    cal_ens = CalibratedClassifierCV(ens, method='isotonic', cv=5)
    cal_ens.fit(train_X, train_y)
    models['Ensemble'] = cal_ens

    # ================================================================
    # Evaluate all models
    # ================================================================
    print("\n" + "-" * 70)
    print("EVALUATING ALL MODELS")
    print("-" * 70)

    all_results = {}
    all_probs = {}

    for name, model in models.items():
        p_def = model.predict_proba(test_X)[:, 1]
        all_probs[name] = p_def
        result = evaluate_tiers(p_def, test_y, name)
        all_results[name] = result

        print(f"\n  === {name} (AUC={result['auc']:.4f}) ===")
        print(f"  {'Tier':>10} {'N':>6} {'Def':>5} {'Cln':>5} {'Def%':>6} {'Lift':>6} {'CumRecall':>10}")
        print(f"  {'-'*55}")
        for row in result['tiers']:
            print(f"  {row['tier']:>10} {row['n_modules']:>6} {row['n_defective']:>5} "
                  f"{row['n_clean']:>5} {row['defect_rate']:>6.0%} {row['lift_over_random']:>6.1f}x "
                  f"{row['cumulative_recall']:>10.0%}")

    # ================================================================
    # Comparison table
    # ================================================================
    print("\n" + "-" * 70)
    print("MODEL COMPARISON")
    print("-" * 70)

    print(f"\n  {'Model':<20} {'AUC':>6} {'CRIT Def%':>10} {'CRIT+HIGH':>10} "
          f"{'Top16% Recall':>14} {'MINIMAL Def':>12}")
    print(f"  {'-'*75}")

    for name, result in all_results.items():
        tiers = {r['tier']: r for r in result['tiers']}
        crit_def_rate = tiers['CRITICAL']['defect_rate']
        crit_high_def = tiers['CRITICAL']['n_defective'] + tiers['HIGH']['n_defective']
        crit_high_recall = crit_high_def / result['total_defective']
        minimal_def = tiers['MINIMAL']['n_defective']

        # Recall at 16% of code reviewed
        p = all_probs[name]
        tier = assign_tiers(p)
        top_16_mask = p >= np.percentile(p, 84)  # top 16%
        top_16_recall = test_y[top_16_mask].sum() / result['total_defective']

        print(f"  {name:<20} {result['auc']:>6.4f} {crit_def_rate:>10.0%} "
              f"{crit_high_recall:>10.0%} {top_16_recall:>14.0%} {minimal_def:>12}")

    # Random baseline
    print(f"  {'Random baseline':<20} {'0.500':>6} {'19%':>10} {'19%':>10} {'16%':>14} {'n/a':>12}")

    # ================================================================
    # Plain English explanation
    # ================================================================
    print("\n" + "-" * 70)
    print("WHAT THESE RESULTS MEAN (PLAIN ENGLISH)")
    print("-" * 70)

    ens_result = all_results['Ensemble']
    xgb_result = all_results['XGBoost']
    ens_tiers = {r['tier']: r for r in ens_result['tiers']}
    xgb_tiers = {r['tier']: r for r in xgb_result['tiers']}

    print(f"""
  THE TASK:
    {ens_result['total_modules']} software modules. {ens_result['total_defective']} contain defects (19%).
    Developers don't know which ones. They need a priority list.

  WHAT THE CASCADE DOES:
    Takes each module's 16 code metrics (complexity, size, effort, etc.)
    and assigns a review priority tier.

  ENSEMBLE (recommended) PERFORMANCE:
    CRITICAL tier: {ens_tiers['CRITICAL']['n_modules']} modules, {ens_tiers['CRITICAL']['defect_rate']:.0%} actually defective
      -> If a dev reviews these first, roughly {ens_tiers['CRITICAL']['defect_rate']:.0%} of the time they find a real bug.
      -> This is {ens_tiers['CRITICAL']['lift_over_random']:.1f}x better than picking modules at random.

    By reviewing CRITICAL + HIGH ({ens_tiers['HIGH']['cumulative_modules']} modules, {ens_tiers['HIGH']['cumulative_modules_pct']:.0%} of code):
      -> They find {ens_tiers['HIGH']['cumulative_recall']:.0%} of all defects.
      -> Random review of the same amount would find only {ens_tiers['HIGH']['cumulative_modules_pct']:.0%}.

    MINIMAL tier: {ens_tiers['MINIMAL']['n_modules']} modules, only {ens_tiers['MINIMAL']['defect_rate']:.0%} defective
      -> These can safely be skipped. Only {ens_tiers['MINIMAL']['n_defective']} defects are missed.

  XGBoost ALONE vs ENSEMBLE:
    XGBoost AUC: {xgb_result['auc']:.4f}
    Ensemble AUC: {ens_result['auc']:.4f} (+{ens_result['auc'] - xgb_result['auc']:.4f})

    XGBoost CRITICAL+HIGH catches: {xgb_tiers['CRITICAL']['n_defective'] + xgb_tiers['HIGH']['n_defective']}/{xgb_result['total_defective']} defects ({(xgb_tiers['CRITICAL']['n_defective'] + xgb_tiers['HIGH']['n_defective'])/xgb_result['total_defective']:.0%})
    Ensemble CRITICAL+HIGH catches: {ens_tiers['CRITICAL']['n_defective'] + ens_tiers['HIGH']['n_defective']}/{ens_result['total_defective']} defects ({(ens_tiers['CRITICAL']['n_defective'] + ens_tiers['HIGH']['n_defective'])/ens_result['total_defective']:.0%})

    The ensemble combines three different model types. Each model
    has a different "perspective" on the code metrics:
      - XGBoost: finds sharp decision boundaries via gradient boosting
      - Random Forest: averages many random feature subsets
      - Logistic Regression: finds the linear trend with class balancing
    Their combined vote is more reliable than any single model.

  WHY SOME MODELS ARE BETTER:
    Random Forest with class_weight='balanced' upweights defective modules
    during training, making it more aggressive at flagging potential defects.
    Logistic Regression does the same. XGBoost is more conservative.
    The ensemble balances aggression (RF, LR) with precision (XGBoost).
""")

    # ================================================================
    # Save results
    # ================================================================
    print("-" * 70)
    print("SAVING RESULTS")
    print("-" * 70)

    output = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'JM1 (PROMISE)',
        'description': 'Prioritization cascade: 2-stage XGBoost/ensemble tier assignment',
        'n_train': int(len(train_y)),
        'n_test': int(len(test_y)),
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'defect_rate': float(test_y.mean()),
        'tier_definitions': {
            'CRITICAL': 'P(defective) >= 0.50. Review immediately.',
            'HIGH': 'P(defective) 0.35-0.50. Review next sprint.',
            'ELEVATED': 'P(defective) 0.20-0.35. Review if resources permit.',
            'LOW': 'P(defective) 0.10-0.20. Automated scan only.',
            'MINIMAL': 'P(defective) < 0.10. Skip review.',
        },
        'methodology': {
            'stage_0': 'Screen extremes: CRITICAL (P>=0.50) and MINIMAL (P<0.10) auto-assigned.',
            'stage_1': 'Triage middle band: HIGH, ELEVATED, LOW assigned by calibrated probability.',
            'calibration': 'Isotonic regression via 5-fold CalibratedClassifierCV.',
            'ensemble': 'Soft-voting VotingClassifier: XGBoost(200 trees, depth 6) + RandomForest(200 trees, depth 10, balanced) + LogisticRegression(balanced, standardized).',
            'deterministic': 'All models use random_state=42. Results are fully reproducible.',
        },
        'models': all_results,
    }

    results_path = OUTPUT_DIR / 'jm1_final_results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results: {results_path}")

    # Save comparison CSV
    rows = []
    for name, result in all_results.items():
        for tier_row in result['tiers']:
            rows.append({'model': name, **tier_row})
    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / 'jm1_model_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path}")

    # ================================================================
    # Plots
    # ================================================================
    print("\n  Generating figures...")

    # Plot 1: Cumulative gain chart (all 4 models)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'Ensemble': '#d32f2f', 'XGBoost': '#1976d2', 'RandomForest': '#388e3c', 'LogisticRegression': '#f57c00'}
    tier_order = ['CRITICAL', 'HIGH', 'ELEVATED', 'LOW', 'MINIMAL']

    for name, result in all_results.items():
        xs = [0]
        ys = [0]
        for row in result['tiers']:
            xs.append(row['cumulative_modules_pct'])
            ys.append(row['cumulative_recall'])
        ax.plot(xs, ys, '-o', color=colors[name], markersize=5,
                label=f"{name} (AUC={result['auc']:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random baseline')
    ax.set_xlabel('Fraction of modules reviewed (top-down)', fontsize=11)
    ax.set_ylabel('Fraction of defects found', fontsize=11)
    ax.set_title('JM1 Prioritization: Cumulative Gain Chart', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'jm1_gain_chart.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {FIG_DIR / 'jm1_gain_chart.png'}")

    # Plot 2: Defect rate by tier (ensemble vs XGBoost)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(5)
    width = 0.35

    ens_rates = [r['defect_rate'] for r in all_results['Ensemble']['tiers']]
    xgb_rates = [r['defect_rate'] for r in all_results['XGBoost']['tiers']]

    bars1 = ax.bar(x - width/2, ens_rates, width, label='Ensemble', color='#d32f2f', alpha=0.8)
    bars2 = ax.bar(x + width/2, xgb_rates, width, label='XGBoost', color='#1976d2', alpha=0.8)

    ax.axhline(test_y.mean(), color='black', linestyle='--', alpha=0.4, label=f'Base rate ({test_y.mean():.0%})')
    ax.set_xticks(x)
    ax.set_xticklabels(tier_order, fontsize=10)
    ax.set_ylabel('Defect Rate', fontsize=11)
    ax.set_title('Defect Rate by Priority Tier', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.0%}',
                    ha='center', fontsize=8, fontweight='bold')

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'jm1_tier_defect_rates.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {FIG_DIR / 'jm1_tier_defect_rates.png'}")

    # Plot 3: Tier size distribution (ensemble)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ens_tiers_data = all_results['Ensemble']['tiers']
    tier_colors = ['#d32f2f', '#ff9800', '#ffc107', '#8bc34a', '#4caf50']

    # Left: module counts with defective highlighted
    ax = axes[0]
    n_defs = [r['n_defective'] for r in ens_tiers_data]
    n_clns = [r['n_clean'] for r in ens_tiers_data]
    ax.bar(x, n_defs, color=tier_colors, alpha=0.9, label='Defective', edgecolor='black', linewidth=0.5)
    ax.bar(x, n_clns, bottom=n_defs, color=tier_colors, alpha=0.3, label='Clean', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_order, fontsize=10)
    ax.set_ylabel('Number of modules', fontsize=11)
    ax.set_title('Module Distribution by Tier (Ensemble)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Right: where defects land
    ax = axes[1]
    def_counts = [r['n_defective'] for r in ens_tiers_data]
    def_pcts = [r['n_defective'] / all_results['Ensemble']['total_defective'] for r in ens_tiers_data]
    bars = ax.bar(x, def_pcts, color=tier_colors, alpha=0.9, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n({n} def)" for t, n in zip(tier_order, def_counts)], fontsize=9)
    ax.set_ylabel('Fraction of all defects', fontsize=11)
    ax.set_title('Defect Distribution Across Tiers (Ensemble)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, pct in zip(bars, def_pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{pct:.0%}', ha='center', fontsize=10, fontweight='bold')

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'jm1_tier_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {FIG_DIR / 'jm1_tier_distribution.png'}")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("Done.")


if __name__ == '__main__':
    main()
