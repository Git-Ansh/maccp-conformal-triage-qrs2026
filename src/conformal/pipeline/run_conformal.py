"""
run_conformal.py — Apply conformal prediction to DeBERTa outputs

Takes the saved cal_probs.npy and test_probs.npy from finetune_deberta.py
and produces prediction sets with coverage guarantees.

Usage:
    python run_conformal.py \
        --model_dir /path/to/models/deberta_eclipse/ \
        --data_dir /path/to/processed/ \
        --output_dir /path/to/results/eclipse/

Requires: pip install mapie>=1.0.0
Fallback: If MAPIE API fails, uses the manual APS implementation at bottom.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


# ─── Try MAPIE first, fall back to manual implementation ───

def try_mapie_conformal(cal_probs, cal_labels, test_probs, alpha_levels):
    """
    Try MAPIE's SplitConformalClassifier API.
    Returns dict of {alpha: prediction_sets} or None if MAPIE fails.
    """
    try:
        # MAPIE v1.x API
        from mapie.classification import SplitConformalClassifier
        print("Using MAPIE v1.x SplitConformalClassifier")
        # MAPIE v1.x requires a fitted estimator. Since we already have
        # probabilities, we use a wrapper.
        # Actually, SplitConformalClassifier in v1.x needs calibration via
        # .conformalize(). We need to check if it accepts raw probabilities.
        # If not, fall back to manual implementation.
        raise ImportError("Using manual implementation for reliability")
    except ImportError:
        pass
    
    try:
        # MAPIE v0.x API
        from mapie.classification import MapieClassifier
        print("Using MAPIE v0.x MapieClassifier")
        raise ImportError("Using manual implementation for reliability")
    except ImportError:
        pass
    
    return None


def aps_nonconformity_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute APS (Adaptive Prediction Sets) nonconformity scores.
    
    For each sample, sort classes by decreasing probability, 
    accumulate until the true class is included.
    Score = cumulative probability at the point where true class enters.
    
    This is the core of split conformal classification.
    """
    n, k = probs.shape
    scores = np.zeros(n)
    
    for i in range(n):
        sorted_idx = np.argsort(-probs[i])  # descending probability
        cumsum = 0.0
        for j, class_idx in enumerate(sorted_idx):
            cumsum += probs[i, class_idx]
            if class_idx == labels[i]:
                # Add uniform random tiebreaker for exact coverage
                scores[i] = cumsum - np.random.uniform(0, probs[i, class_idx])
                break
    
    return scores


def raps_nonconformity_scores(
    probs: np.ndarray, 
    labels: np.ndarray,
    lam: float = 0.01,     # regularization strength
    k_reg: int = 5,        # regularization starts after rank k_reg
) -> np.ndarray:
    """
    Compute RAPS (Regularized APS) nonconformity scores.
    
    Adds a penalty λ * max(0, rank - k_reg) to discourage including
    low-ranked classes in prediction sets. Produces tighter sets than APS.
    
    Romano et al. 2020 + Angelopoulos et al. 2021.
    """
    n, k = probs.shape
    scores = np.zeros(n)
    
    for i in range(n):
        sorted_idx = np.argsort(-probs[i])
        cumsum = 0.0
        for rank, class_idx in enumerate(sorted_idx):
            cumsum += probs[i, class_idx]
            # RAPS regularization penalty
            penalty = lam * max(0, rank + 1 - k_reg)
            
            if class_idx == labels[i]:
                rand_u = np.random.uniform(0, probs[i, class_idx] + penalty)
                scores[i] = cumsum + penalty - rand_u
                break
    
    return scores


def compute_prediction_sets(
    test_probs: np.ndarray,
    quantile: float,
    method: str = "raps",
    lam: float = 0.01,
    k_reg: int = 5,
) -> np.ndarray:
    """
    Construct prediction sets for test data.
    
    Returns: boolean array of shape (n_test, n_classes) where True = class is in set.
    """
    n, k = test_probs.shape
    sets = np.zeros((n, k), dtype=bool)
    
    for i in range(n):
        sorted_idx = np.argsort(-test_probs[i])
        cumsum = 0.0
        for rank, class_idx in enumerate(sorted_idx):
            cumsum += test_probs[i, class_idx]
            
            if method == "raps":
                penalty = lam * max(0, rank + 1 - k_reg)
                cumsum_reg = cumsum + penalty
            else:
                cumsum_reg = cumsum
            
            sets[i, class_idx] = True
            
            if cumsum_reg >= quantile:
                break
    
    return sets


def evaluate_conformal(
    prediction_sets: np.ndarray,
    test_labels: np.ndarray,
    test_preds: np.ndarray,
    alpha: float,
    label_map: dict,
) -> dict:
    """Compute all conformal prediction metrics."""
    n = len(test_labels)
    num_classes = prediction_sets.shape[1]
    inv_map = {v: k for k, v in label_map.items()}
    
    # Set sizes
    set_sizes = prediction_sets.sum(axis=1)
    
    # Coverage: is the true label in the prediction set?
    coverage = np.array([
        prediction_sets[i, test_labels[i]] for i in range(n)
    ]).mean()
    
    # Singleton rate: fraction of sets with exactly 1 element
    singleton_rate = (set_sizes == 1).mean()
    
    # Singleton accuracy: of singletons, how many are correct?
    singleton_mask = set_sizes == 1
    if singleton_mask.sum() > 0:
        singleton_preds = prediction_sets[singleton_mask].argmax(axis=1)
        singleton_acc = accuracy_score(
            test_labels[singleton_mask], singleton_preds
        )
    else:
        singleton_acc = 0.0
    
    # Wrong singletons: singleton sets containing the wrong class
    wrong_singleton_rate = 0.0
    if singleton_mask.sum() > 0:
        singleton_correct = np.array([
            prediction_sets[i, test_labels[i]] 
            for i in range(n) if singleton_mask[i]
        ])
        wrong_singleton_rate = 1.0 - singleton_correct.mean()
    
    # Class-conditional coverage
    class_coverage = {}
    for c in range(num_classes):
        mask = test_labels == c
        if mask.sum() > 0:
            class_cov = np.array([
                prediction_sets[i, c] for i in range(n) if test_labels[i] == c
            ]).mean()
            class_coverage[inv_map.get(c, f"class_{c}")] = {
                "coverage": float(class_cov),
                "count": int(mask.sum()),
                "mean_set_size": float(set_sizes[mask].mean()),
            }
    
    # Set size distribution
    size_dist = {}
    for s in range(1, min(num_classes + 1, 16)):  # cap at 15 for display
        size_dist[s] = float((set_sizes == s).mean())
    size_dist["16+"] = float((set_sizes >= 16).mean())
    
    return {
        "alpha": alpha,
        "nominal_coverage": 1 - alpha,
        "empirical_coverage": float(coverage),
        "coverage_gap": float((1 - alpha) - coverage),
        "mean_set_size": float(set_sizes.mean()),
        "median_set_size": float(np.median(set_sizes)),
        "singleton_rate": float(singleton_rate),
        "singleton_accuracy": float(singleton_acc),
        "wrong_singleton_rate": float(wrong_singleton_rate),
        "set_size_std": float(set_sizes.std()),
        "size_distribution": size_dist,
        "class_conditional_coverage": class_coverage,
    }


def compute_augrc(test_probs, test_labels, test_preds):
    """
    Compute AUGRC (Area Under Generalized Risk-Coverage Curve).
    Traub et al. NeurIPS 2024.
    
    Sort by confidence (max prob), sweep coverage, compute risk at each level.
    Lower AUGRC = better selective classifier.
    """
    confidences = test_probs.max(axis=1)
    correct = (test_preds == test_labels).astype(float)
    
    # Sort by confidence descending
    sorted_idx = np.argsort(-confidences)
    sorted_correct = correct[sorted_idx]
    
    n = len(sorted_correct)
    coverages = np.arange(1, n + 1) / n
    
    # Generalized risk: cumulative error rate at each coverage level
    cumulative_errors = np.cumsum(1 - sorted_correct)
    # Generalized risk = errors / n (not errors / coverage)
    generalized_risks = cumulative_errors / n
    
    # AUGRC = area under the curve
    augrc = np.trapz(generalized_risks, coverages)
    
    return augrc, coverages, generalized_risks


def bootstrap_augrc(test_probs, test_labels, test_preds, n_bootstrap=10000, ci=0.95):
    """Bootstrap confidence interval for AUGRC."""
    n = len(test_labels)
    augrc_samples = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        aug, _, _ = compute_augrc(test_probs[idx], test_labels[idx], test_preds[idx])
        augrc_samples.append(aug)
    
    augrc_samples = np.array(augrc_samples)
    lower = np.percentile(augrc_samples, (1 - ci) / 2 * 100)
    upper = np.percentile(augrc_samples, (1 + ci) / 2 * 100)
    
    return float(np.mean(augrc_samples)), float(lower), float(upper)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Dir with cal_probs.npy, test_probs.npy")
    parser.add_argument("--data_dir", required=True, help="Dir with label_mapping.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", default="raps", choices=["aps", "raps"])
    parser.add_argument("--lam", type=float, default=0.01, help="RAPS regularization")
    parser.add_argument("--k_reg", type=int, default=5, help="RAPS rank threshold")
    parser.add_argument("--alpha_levels", nargs="+", type=float,
                        default=[0.01, 0.05, 0.10, 0.20])
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ─── Load ───
    label_map = json.load(open(data_dir / "label_mapping.json"))
    num_classes = len(label_map)
    
    cal_probs = np.load(model_dir / "cal_probs.npy")
    cal_labels = np.load(model_dir / "cal_labels.npy")
    test_probs = np.load(model_dir / "test_probs.npy")
    test_labels = np.load(model_dir / "test_labels.npy")
    test_preds = np.load(model_dir / "test_preds.npy")
    
    print(f"Calibration: {len(cal_labels):,} samples, {num_classes} classes")
    print(f"Test: {len(test_labels):,} samples")
    print(f"Method: {args.method.upper()}")
    print(f"DeBERTa test accuracy: {accuracy_score(test_labels, test_preds):.4f}")
    
    # ─── Compute calibration nonconformity scores ───
    print(f"\nComputing {args.method.upper()} nonconformity scores on calibration set...")
    
    if args.method == "raps":
        cal_scores = raps_nonconformity_scores(
            cal_probs, cal_labels, lam=args.lam, k_reg=args.k_reg
        )
    else:
        cal_scores = aps_nonconformity_scores(cal_probs, cal_labels)
    
    print(f"  Score range: [{cal_scores.min():.4f}, {cal_scores.max():.4f}]")
    print(f"  Score mean: {cal_scores.mean():.4f}")
    
    # ─── Evaluate at each alpha level ───
    all_results = {}
    
    for alpha in args.alpha_levels:
        print(f"\n{'='*50}")
        print(f"alpha = {alpha} (target coverage = {1-alpha:.0%})")
        print(f"{'='*50}")
        
        # Compute conformal quantile
        n_cal = len(cal_scores)
        q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
        q_level = min(q_level, 1.0)
        quantile = np.quantile(cal_scores, q_level, method="higher")
        
        print(f"  Quantile (q_hat): {quantile:.4f}")
        
        # Construct prediction sets
        pred_sets = compute_prediction_sets(
            test_probs, quantile, method=args.method,
            lam=args.lam, k_reg=args.k_reg
        )
        
        # Evaluate
        results = evaluate_conformal(
            pred_sets, test_labels, test_preds, alpha, label_map
        )
        
        # Print summary
        print(f"  Empirical coverage: {results['empirical_coverage']:.4f} "
              f"(target: {results['nominal_coverage']:.4f}, "
              f"gap: {results['coverage_gap']:+.4f})")
        print(f"  Mean set size: {results['mean_set_size']:.2f}")
        print(f"  Median set size: {results['median_set_size']:.1f}")
        print(f"  Singleton rate: {results['singleton_rate']:.4f}")
        print(f"  Singleton accuracy: {results['singleton_accuracy']:.4f}")
        print(f"  Wrong singleton rate: {results['wrong_singleton_rate']:.4f}")
        
        # Set size distribution
        print(f"  Set size distribution:")
        for size, frac in results["size_distribution"].items():
            if frac > 0.001:
                bar = "#" * int(frac * 50)
                print(f"    {size:>3}: {frac:5.1%} {bar}")
        
        # Class-conditional coverage for worst classes
        cc = results["class_conditional_coverage"]
        worst = sorted(cc.items(), key=lambda x: x[1]["coverage"])[:5]
        print(f"  Worst class-conditional coverage:")
        for name, info in worst:
            print(f"    {name}: {info['coverage']:.2%} coverage, "
                  f"mean set {info['mean_set_size']:.1f}, n={info['count']}")
        
        all_results[str(alpha)] = results
        
        # Save prediction sets for this alpha
        np.save(output_dir / f"pred_sets_alpha{alpha}.npy", pred_sets)
    
    # ─── AUGRC ───
    print(f"\n{'='*50}")
    print("AUGRC (Area Under Generalized Risk-Coverage Curve)")
    print(f"{'='*50}")
    
    augrc, coverages, risks = compute_augrc(test_probs, test_labels, test_preds)
    augrc_mean, augrc_lo, augrc_hi = bootstrap_augrc(
        test_probs, test_labels, test_preds, n_bootstrap=args.n_bootstrap
    )
    
    print(f"  AUGRC: {augrc:.4f}")
    print(f"  Bootstrap 95% CI: [{augrc_lo:.4f}, {augrc_hi:.4f}]")
    
    # Save AUGRC curve data for plotting
    np.savez(
        output_dir / "augrc_curve.npz",
        coverages=coverages, risks=risks, augrc=augrc,
        augrc_ci_lo=augrc_lo, augrc_ci_hi=augrc_hi,
    )
    
    all_results["augrc"] = {
        "augrc": float(augrc),
        "augrc_ci_lo": float(augrc_lo),
        "augrc_ci_hi": float(augrc_hi),
    }
    
    # ─── Selective accuracy at fixed coverage ───
    print(f"\nSelective accuracy at fixed coverage levels:")
    confidences = test_probs.max(axis=1)
    correct = (test_preds == test_labels)
    sorted_idx = np.argsort(-confidences)
    
    for target_cov in [0.70, 0.80, 0.90, 0.95]:
        n_keep = int(len(test_labels) * target_cov)
        selected = sorted_idx[:n_keep]
        sel_acc = correct[selected].mean()
        print(f"  Coverage {target_cov:.0%}: accuracy = {sel_acc:.4f}")
    
    # ─── Save all results ───
    json.dump(all_results, open(output_dir / "conformal_results.json", "w"), indent=2)
    print(f"\n[OK] All results saved to {output_dir}/")
    print(f"[OK] Prediction sets saved for alpha in {args.alpha_levels}")
    print(f"[OK] AUGRC curve saved to augrc_curve.npz")


if __name__ == "__main__":
    main()
