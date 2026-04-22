"""
utils.py -- Shared RAPS / MACCP functions for model ablation experiments.

Provides:
  - compute_raps_score: RAPS nonconformity score for a single example
  - compute_raps_scores_batch: Vectorized batch version
  - compute_conformal_quantile: Finite-sample-corrected quantile
  - compute_prediction_set: Build prediction sets given a threshold
  - run_maccp_pipeline: Full MACCP with agreement-conditioned thresholds
  - save_results / print_comparison_row: I/O helpers
"""

import json
import os
import numpy as np

np.random.seed(42)


# ============================================================
# RAPS nonconformity scores
# ============================================================

def compute_raps_score(probs, true_label, lam=0.01, kreg=5):
    """Compute the RAPS nonconformity score for a single example.

    Parameters
    ----------
    probs : ndarray, shape (K,)
        Softmax probabilities over K classes.
    true_label : int
        Ground-truth class index.
    lam : float
        Regularisation penalty weight (controls set size).
    kreg : int
        Number of top classes exempt from the penalty.

    Returns
    -------
    float
        The RAPS score (lower = more conformal).
    """
    sorted_idx = np.argsort(-probs)
    cumsum = 0.0
    for rank, class_idx in enumerate(sorted_idx):
        cumsum += probs[class_idx]
        penalty = lam * max(0, rank + 1 - kreg)
        if class_idx == true_label:
            rand_u = np.random.uniform(0, probs[class_idx] + penalty)
            return cumsum + penalty - rand_u
    # Should never reach here if true_label is valid
    return cumsum


def compute_raps_scores_batch(probs_matrix, labels, lam=0.01, kreg=5):
    """Compute RAPS nonconformity scores for a batch of examples.

    Parameters
    ----------
    probs_matrix : ndarray, shape (N, K)
        Softmax probabilities for N examples over K classes.
    labels : ndarray, shape (N,)
        Ground-truth class indices.
    lam : float
        Regularisation penalty weight.
    kreg : int
        Number of top classes exempt from the penalty.

    Returns
    -------
    ndarray, shape (N,)
        RAPS scores for each example.
    """
    n = len(labels)
    scores = np.zeros(n)
    for i in range(n):
        scores[i] = compute_raps_score(probs_matrix[i], labels[i], lam, kreg)
    return scores


# ============================================================
# Conformal quantile
# ============================================================

def compute_conformal_quantile(scores, alpha):
    """Compute the conformal quantile with finite-sample correction.

    Uses ceil((n+1)*(1-alpha)) / n as the quantile level, capped at 1.0.

    Parameters
    ----------
    scores : ndarray, shape (N,)
        Calibration nonconformity scores.
    alpha : float
        Target miscoverage rate (e.g. 0.10 for 90% coverage).

    Returns
    -------
    float
        The quantile threshold for prediction set construction.
    """
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


# ============================================================
# Prediction set construction
# ============================================================

def compute_prediction_set(probs, threshold, lam=0.01, kreg=5):
    """Construct RAPS prediction sets for a batch of examples.

    Parameters
    ----------
    probs : ndarray, shape (N, K)
        Softmax probabilities.
    threshold : float
        Conformal quantile threshold.
    lam : float
        Regularisation penalty weight.
    kreg : int
        Number of top classes exempt from the penalty.

    Returns
    -------
    ndarray, shape (N, K), dtype=bool
        Boolean mask where True indicates class is in the prediction set.
    """
    n, k = probs.shape
    sets = np.zeros((n, k), dtype=bool)
    for i in range(n):
        sorted_idx = np.argsort(-probs[i])
        cumsum = 0.0
        for rank, class_idx in enumerate(sorted_idx):
            cumsum += probs[i, class_idx]
            penalty = lam * max(0, rank + 1 - kreg)
            cumsum_reg = cumsum + penalty
            sets[i, class_idx] = True
            if cumsum_reg >= threshold:
                break
    return sets


# ============================================================
# Evaluation helpers
# ============================================================

def evaluate_prediction_sets(pred_sets, labels, description=""):
    """Compute coverage, set size, singleton rate/accuracy for prediction sets.

    Parameters
    ----------
    pred_sets : ndarray, shape (N, K), dtype=bool
        Boolean prediction sets.
    labels : ndarray, shape (N,)
        Ground-truth class indices.
    description : str
        Label for the results dict.

    Returns
    -------
    dict
        Metrics including coverage, mean/median set size, singleton rate/accuracy.
    """
    n = len(labels)
    set_sizes = pred_sets.sum(axis=1)

    # Coverage: fraction of examples where true label is in the set
    coverage = np.mean([pred_sets[i, labels[i]] for i in range(n)])

    # Singleton analysis
    singleton_mask = set_sizes == 1
    singleton_rate = float(singleton_mask.mean())
    if singleton_mask.sum() > 0:
        singleton_preds = pred_sets[singleton_mask].argmax(axis=1)
        singleton_acc = float((singleton_preds == labels[singleton_mask]).mean())
    else:
        singleton_acc = 0.0

    return {
        "description": description,
        "coverage": float(coverage),
        "mean_set_size": float(set_sizes.mean()),
        "median_set_size": float(np.median(set_sizes)),
        "singleton_rate": singleton_rate,
        "singleton_accuracy": singleton_acc,
        "n": int(n),
    }


# ============================================================
# MACCP pipeline
# ============================================================

def run_maccp_pipeline(
    cal_probs_agree_model,
    cal_probs_disagree_model,
    test_probs_agree_model,
    test_probs_disagree_model,
    cal_labels,
    test_labels,
    cal_agreement,
    test_agreement,
    alpha=0.10,
    lam=0.01,
    kreg=5,
):
    """Run the full MACCP pipeline with agreement-conditioned thresholds.

    Uses one model for the "agree" partition and (possibly different) model
    for the "disagree" partition.  Agreement is a boolean mask.

    Parameters
    ----------
    cal_probs_agree_model : ndarray (N_cal, K)
        Calibration probabilities from the model used when models agree.
    cal_probs_disagree_model : ndarray (N_cal, K)
        Calibration probabilities from the model used when models disagree.
    test_probs_agree_model : ndarray (N_test, K)
        Test probabilities from the agree-model.
    test_probs_disagree_model : ndarray (N_test, K)
        Test probabilities from the disagree-model.
    cal_labels : ndarray (N_cal,)
        Calibration ground-truth labels.
    test_labels : ndarray (N_test,)
        Test ground-truth labels.
    cal_agreement : ndarray (N_cal,), dtype=bool
        Whether models agree on calibration examples.
    test_agreement : ndarray (N_test,), dtype=bool
        Whether models agree on test examples.
    alpha : float
        Target miscoverage rate.
    lam : float
        RAPS lambda.
    kreg : int
        RAPS kreg.

    Returns
    -------
    dict
        Full results including per-partition and overall metrics.
    """
    cal_agreement = np.asarray(cal_agreement, dtype=bool)
    test_agreement = np.asarray(test_agreement, dtype=bool)

    # Split calibration into agree/disagree
    cal_agree_idx = np.where(cal_agreement)[0]
    cal_disagree_idx = np.where(~cal_agreement)[0]

    n_cal_agree = len(cal_agree_idx)
    n_cal_disagree = len(cal_disagree_idx)

    # Compute RAPS scores on each partition using the respective model
    if n_cal_agree > 0:
        scores_agree = compute_raps_scores_batch(
            cal_probs_agree_model[cal_agree_idx],
            cal_labels[cal_agree_idx],
            lam, kreg,
        )
        q_agree = compute_conformal_quantile(scores_agree, alpha)
    else:
        q_agree = 1.0

    if n_cal_disagree > 0:
        scores_disagree = compute_raps_scores_batch(
            cal_probs_disagree_model[cal_disagree_idx],
            cal_labels[cal_disagree_idx],
            lam, kreg,
        )
        q_disagree = compute_conformal_quantile(scores_disagree, alpha)
    else:
        q_disagree = 1.0

    # Build test prediction sets
    n_test = len(test_labels)
    k = test_probs_agree_model.shape[1]
    pred_sets = np.zeros((n_test, k), dtype=bool)

    test_agree_idx = np.where(test_agreement)[0]
    test_disagree_idx = np.where(~test_agreement)[0]

    if len(test_agree_idx) > 0:
        agree_sets = compute_prediction_set(
            test_probs_agree_model[test_agree_idx], q_agree, lam, kreg
        )
        pred_sets[test_agree_idx] = agree_sets

    if len(test_disagree_idx) > 0:
        disagree_sets = compute_prediction_set(
            test_probs_disagree_model[test_disagree_idx], q_disagree, lam, kreg
        )
        pred_sets[test_disagree_idx] = disagree_sets

    # Evaluate
    overall = evaluate_prediction_sets(pred_sets, test_labels, "MACCP overall")
    overall["q_agree"] = float(q_agree)
    overall["q_disagree"] = float(q_disagree)
    overall["n_cal_agree"] = int(n_cal_agree)
    overall["n_cal_disagree"] = int(n_cal_disagree)
    overall["n_test_agree"] = int(len(test_agree_idx))
    overall["n_test_disagree"] = int(len(test_disagree_idx))
    overall["alpha"] = float(alpha)

    results = {"overall": overall}

    if len(test_agree_idx) > 0:
        results["agree"] = evaluate_prediction_sets(
            pred_sets[test_agree_idx], test_labels[test_agree_idx], "MACCP agree"
        )

    if len(test_disagree_idx) > 0:
        results["disagree"] = evaluate_prediction_sets(
            pred_sets[test_disagree_idx], test_labels[test_disagree_idx], "MACCP disagree"
        )

    return results


# ============================================================
# I/O helpers
# ============================================================

def save_results(results, filepath):
    """Save results dict to JSON, creating parent directories as needed.

    Parameters
    ----------
    results : dict
        Results to save (must be JSON-serializable).
    filepath : str or Path
        Output file path.
    """
    filepath = str(filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {filepath}")


def print_comparison_row(name, dataset, metrics):
    """Print a single formatted comparison row.

    Parameters
    ----------
    name : str
        Configuration name (e.g. "Config A").
    dataset : str
        Dataset name (e.g. "Eclipse").
    metrics : dict
        Must contain: coverage, mean_set_size, singleton_rate, singleton_accuracy.
    """
    print(
        f"  {name:<25s} | {dataset:<8s} | "
        f"Cov={metrics['coverage']:.3f} | "
        f"Size={metrics['mean_set_size']:.2f} | "
        f"Sing={metrics['singleton_rate']:.3f} | "
        f"SingAcc={metrics['singleton_accuracy']:.3f}"
    )


def print_table_header():
    """Print the header row for a comparison table."""
    print(
        f"  {'Configuration':<25s} | {'Dataset':<8s} | "
        f"{'Coverage':>8s} | {'MeanSize':>8s} | "
        f"{'SingRate':>8s} | {'SingAcc':>8s}"
    )
    print("  " + "-" * 85)


def load_data(dataset_name, base_dir=None):
    """Load all data files for a dataset.

    Parameters
    ----------
    dataset_name : str
        Either 'eclipse' or 'mozilla'.
    base_dir : str or None
        Base directory. Defaults to script's parent/../data.

    Returns
    -------
    dict
        Keys: deberta_cal_probs, deberta_test_probs, xgb_cal_probs, xgb_test_probs,
              deberta_cal_preds, deberta_test_preds, xgb_cal_preds, xgb_test_preds,
              cal_labels, test_labels, label_mapping, inv_mapping, num_classes.
    """
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    data_dir = os.path.join(base_dir, dataset_name)

    data = {}
    for prefix in ["deberta", "xgb"]:
        for split in ["cal", "test"]:
            for kind in ["probs", "preds"]:
                key = f"{prefix}_{split}_{kind}"
                fpath = os.path.join(data_dir, f"{key}.npy")
                data[key] = np.load(fpath)

    data["cal_labels"] = np.load(os.path.join(data_dir, "cal_labels.npy"))
    data["test_labels"] = np.load(os.path.join(data_dir, "test_labels.npy"))

    with open(os.path.join(data_dir, "label_mapping.json")) as f:
        data["label_mapping"] = json.load(f)
    data["inv_mapping"] = {v: k for k, v in data["label_mapping"].items()}
    data["num_classes"] = len(data["label_mapping"])

    return data


def compute_agreement(deberta_preds, xgb_preds):
    """Compute binary agreement mask: argmax(DeBERTa) == argmax(XGBoost).

    Parameters
    ----------
    deberta_preds : ndarray (N,)
        DeBERTa predicted class indices.
    xgb_preds : ndarray (N,)
        XGBoost predicted class indices.

    Returns
    -------
    ndarray (N,), dtype=bool
        True where both models predict the same class.
    """
    return np.asarray(deberta_preds == xgb_preds, dtype=bool)
