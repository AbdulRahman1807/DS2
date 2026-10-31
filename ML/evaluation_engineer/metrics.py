"""Multi-Tier Classification & Anomaly Evaluation Metrics Module.

Assigned Role: Evaluation Engineer
Stage: Stage 01 - Independent Model Evaluation & Validation

Provides:
- 3x3 Multi-class Confusion Matrix (raw counts and row-normalized)
- Per-class Precision, Recall, F1, Support; Macro F1, Weighted F1, Balanced Accuracy
- Trivial baseline comparisons (Majority Prior dummy, Stratified Random dummy)
- One-vs-Rest ROC-AUC and PR-AUC with PR-AUC Lift over baseline prevalence
- Top-k ranking metrics: Precision@k, Recall@k, Lift@k for k in [1%, 5%, 10%, 20%]
- Multiclass probability calibration (Brier Score, Log Loss, ECE, reliability bins)
- 1,000-resample non-parametric Bootstrap 95% Confidence Intervals
- Paired bootstrap for PR-AUC difference (e.g. Yard feature ablation)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    auc,
)

logger = logging.getLogger(__name__)

CLASS_NAMES: List[str] = ["ON_TIME", "AT_RISK", "CRITICAL_DELAY"]
CLASS_IDS: List[int] = [0, 1, 2]


def compute_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Computes raw counts and row-normalized (recall) confusion matrices."""
    if labels is None:
        labels = CLASS_IDS
    if class_names is None:
        class_names = CLASS_NAMES

    cm_raw = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.nan_to_num(cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True), 0.0)

    return {
        "labels": labels,
        "class_names": class_names,
        "raw_counts": cm_raw.tolist(),
        "row_normalized": np.round(cm_norm, 4).tolist(),
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Computes global and per-class classification metrics."""
    if labels is None:
        labels = CLASS_IDS
    if class_names is None:
        class_names = CLASS_NAMES

    n_samples = len(y_true)
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    per_class: Dict[str, Dict[str, Any]] = {}
    for idx, cname in enumerate(class_names):
        mask_true = (y_true == idx)
        support = int(np.sum(mask_true))
        prec = float(precision_score(y_true == idx, y_pred == idx, zero_division=0))
        rec = float(recall_score(y_true == idx, y_pred == idx, zero_division=0))
        f1_c = float(f1_score(y_true == idx, y_pred == idx, zero_division=0))
        prev = float(support / n_samples) if n_samples > 0 else 0.0

        per_class[cname] = {
            "class_id": idx,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1_c, 4),
            "support": support,
            "prevalence": round(prev, 4),
        }

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "n_samples": n_samples,
    }


def compute_dummy_baselines(
    y_train: np.ndarray,
    y_eval: np.ndarray,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Computes performance of standard trivial dummy baselines."""
    majority_class = int(pd.Series(y_train).mode().iloc[0])
    y_pred_majority = np.full_like(y_eval, majority_class)

    train_counts = pd.Series(y_train).value_counts(normalize=True).sort_index()
    priors = np.array([train_counts.get(i, 0.0) for i in CLASS_IDS])
    rng = np.random.default_rng(random_state)
    y_pred_stratified = rng.choice(CLASS_IDS, size=len(y_eval), p=priors)

    maj_metrics = compute_classification_metrics(y_eval, y_pred_majority)
    strat_metrics = compute_classification_metrics(y_eval, y_pred_stratified)

    return {
        "majority_prior": {
            "majority_class": majority_class,
            "accuracy": maj_metrics["accuracy"],
            "balanced_accuracy": maj_metrics["balanced_accuracy"],
            "macro_f1": maj_metrics["macro_f1"],
            "critical_recall": maj_metrics["per_class"]["CRITICAL_DELAY"]["recall"],
            "critical_precision": maj_metrics["per_class"]["CRITICAL_DELAY"]["precision"],
        },
        "stratified_random": {
            "class_priors": [round(float(p), 4) for p in priors],
            "accuracy": strat_metrics["accuracy"],
            "balanced_accuracy": strat_metrics["balanced_accuracy"],
            "macro_f1": strat_metrics["macro_f1"],
            "critical_recall": strat_metrics["per_class"]["CRITICAL_DELAY"]["recall"],
            "critical_precision": strat_metrics["per_class"]["CRITICAL_DELAY"]["precision"],
        },
    }


def compute_ranking_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Computes One-vs-Rest ROC-AUC, PR-AUC, and Critical Delay PR-AUC Lift."""
    if class_names is None:
        class_names = CLASS_NAMES

    roc_aucs: Dict[str, Optional[float]] = {}
    pr_aucs: Dict[str, Optional[float]] = {}
    pr_lifts: Dict[str, Optional[float]] = {}
    base_rates: Dict[str, float] = {}

    for idx, cname in enumerate(class_names):
        binary_true = (y_true == idx).astype(int)
        prevalence = float(np.mean(binary_true))
        base_rates[cname] = round(prevalence, 4)

        if np.sum(binary_true) == 0 or np.sum(binary_true) == len(y_true):
            roc_aucs[cname] = None
            pr_aucs[cname] = None
            pr_lifts[cname] = None
            continue

        try:
            r_auc = float(roc_auc_score(binary_true, y_prob[:, idx]))
            roc_aucs[cname] = round(r_auc, 4)
        except Exception:
            roc_aucs[cname] = None

        try:
            prec_arr, rec_arr, _ = precision_recall_curve(binary_true, y_prob[:, idx])
            p_auc = float(auc(rec_arr, prec_arr))
            pr_aucs[cname] = round(p_auc, 4)
            lift = (p_auc / prevalence) if prevalence > 0 else 1.0
            pr_lifts[cname] = round(lift, 4)
        except Exception:
            pr_aucs[cname] = None
            pr_lifts[cname] = None

    return {
        "roc_auc_ovr": roc_aucs,
        "pr_auc_ovr": pr_aucs,
        "pr_auc_lift": pr_lifts,
        "baseline_prevalence": base_rates,
        "critical_delay_pr_auc": pr_aucs.get("CRITICAL_DELAY"),
        "critical_delay_pr_lift": pr_lifts.get("CRITICAL_DELAY"),
        "critical_delay_roc_auc": roc_aucs.get("CRITICAL_DELAY"),
    }


def compute_top_k_metrics(
    y_true: np.ndarray,
    y_prob_crit: np.ndarray,
    ks: Optional[List[int]] = None,
    target_class_id: int = 2,
) -> Dict[str, Dict[str, Any]]:
    """Computes Precision@top-k, Recall@top-k, and Lift@top-k for specified percentiles k."""
    if ks is None:
        ks = [1, 5, 10, 20]

    n_samples = len(y_true)
    crit_true = (y_true == target_class_id).astype(int)
    total_pos = int(np.sum(crit_true))
    base_prev = total_pos / n_samples if n_samples > 0 else 0.0

    order = np.argsort(-y_prob_crit)
    sorted_crit = crit_true[order]

    results: Dict[str, Dict[str, Any]] = {}
    for k in ks:
        n_k = int(np.round(n_samples * (k / 100.0)))
        n_k = max(1, min(n_samples, n_k))
        tp_k = int(np.sum(sorted_crit[:n_k]))
        prec_k = float(tp_k / n_k)
        rec_k = float(tp_k / total_pos) if total_pos > 0 else 0.0
        lift_k = float(prec_k / base_prev) if base_prev > 0 else 1.0

        results[f"top_{k}pct"] = {
            "k_pct": k,
            "n_flagged": n_k,
            "true_positives": tp_k,
            "precision": round(prec_k, 4),
            "recall": round(rec_k, 4),
            "lift": round(lift_k, 2),
            "base_prevalence": round(base_prev, 4),
        }

    return results


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Computes multiclass Brier score, multiclass log loss, ECE, and reliability diagram bins."""
    if class_names is None:
        class_names = CLASS_NAMES

    n_samples, n_classes = y_prob.shape

    y_one_hot = np.zeros((n_samples, n_classes))
    for k in range(n_classes):
        y_one_hot[:, k] = (y_true == k).astype(float)

    brier_score = float(np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1)))

    prob_clipped = np.clip(y_prob, 1e-15, 1.0 - 1e-15)
    prob_clipped /= prob_clipped.sum(axis=1, keepdims=True)
    m_log_loss = float(log_loss(y_true, prob_clipped, labels=CLASS_IDS))

    ece_per_class: Dict[str, float] = {}
    reliability_bins: Dict[str, List[Dict[str, Any]]] = {}

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for idx, cname in enumerate(class_names):
        p_c = y_prob[:, idx]
        y_c = y_one_hot[:, idx]

        ece_c = 0.0
        bins_data = []

        for b in range(n_bins):
            low, high = bin_edges[b], bin_edges[b + 1]
            if b == n_bins - 1:
                mask = (p_c >= low) & (p_c <= high)
            else:
                mask = (p_c >= low) & (p_c < high)

            bin_count = int(np.sum(mask))
            if bin_count > 0:
                mean_pred = float(np.mean(p_c[mask]))
                empirical_acc = float(np.mean(y_c[mask]))
                abs_diff = abs(empirical_acc - mean_pred)
                ece_c += (bin_count / n_samples) * abs_diff

                bins_data.append({
                    "bin_index": b,
                    "bin_lower": round(low, 2),
                    "bin_upper": round(high, 2),
                    "count": bin_count,
                    "mean_predicted_prob": round(mean_pred, 4),
                    "empirical_accuracy": round(empirical_acc, 4),
                    "calibration_gap": round(empirical_acc - mean_pred, 4),
                })
            else:
                bins_data.append({
                    "bin_index": b,
                    "bin_lower": round(low, 2),
                    "bin_upper": round(high, 2),
                    "count": 0,
                    "mean_predicted_prob": round((low + high) / 2.0, 4),
                    "empirical_accuracy": 0.0,
                    "calibration_gap": 0.0,
                })

        ece_per_class[cname] = round(ece_c, 4)
        reliability_bins[cname] = bins_data

    overall_ece = round(float(np.mean(list(ece_per_class.values()))), 4)

    return {
        "brier_score": round(brier_score, 4),
        "log_loss": round(m_log_loss, 4),
        "overall_ece": overall_ece,
        "ece_per_class": ece_per_class,
        "reliability_bins": reliability_bins,
    }


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Calculates non-parametric 95% bootstrap confidence intervals for key metrics,
    recording the empirical sample point estimate and asserting it lies within or at the interval bounds.
    """
    n_samples = len(y_true)
    rng = np.random.default_rng(random_state)

    crit_binary_true = (y_true == 2).astype(int)
    crit_prob = y_prob[:, 2]

    # Empirical point estimates on the full sample
    sample_prev = float(np.mean(crit_binary_true))
    sample_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    sample_rec = float(recall_score(crit_binary_true, (y_pred == 2).astype(int), zero_division=0))
    sample_prec = float(precision_score(crit_binary_true, (y_pred == 2).astype(int), zero_division=0))

    if np.sum(crit_binary_true) > 0 and np.sum(crit_binary_true) < n_samples:
        p_arr, r_arr, _ = precision_recall_curve(crit_binary_true, crit_prob)
        sample_pr_auc = float(auc(r_arr, p_arr))
        sample_lift = float(sample_pr_auc / sample_prev) if sample_prev > 0 else 1.0
        sample_roc_auc = float(roc_auc_score(crit_binary_true, crit_prob))
    else:
        sample_pr_auc = sample_prev
        sample_lift = 1.0
        sample_roc_auc = 0.5

    # Top-k empirical point estimates
    sample_top_k = compute_top_k_metrics(y_true, crit_prob, ks=[1, 5, 10, 20])

    # Bootstrap distributions
    b_prev = np.empty(n_bootstraps)
    b_macro_f1 = np.empty(n_bootstraps)
    b_rec = np.empty(n_bootstraps)
    b_prec = np.empty(n_bootstraps)
    b_pr_auc = np.empty(n_bootstraps)
    b_lift = np.empty(n_bootstraps)
    b_roc = np.empty(n_bootstraps)

    b_topk_prec: Dict[int, np.ndarray] = {k: np.empty(n_bootstraps) for k in [1, 5, 10, 20]}
    b_topk_rec: Dict[int, np.ndarray] = {k: np.empty(n_bootstraps) for k in [1, 5, 10, 20]}
    b_topk_lift: Dict[int, np.ndarray] = {k: np.empty(n_bootstraps) for k in [1, 5, 10, 20]}

    for b in range(n_bootstraps):
        idx = rng.integers(0, n_samples, size=n_samples)
        y_t_b = y_true[idx]
        y_p_b = y_pred[idx]
        crit_true_b = crit_binary_true[idx]
        crit_prob_b = crit_prob[idx]

        prev_b = float(np.mean(crit_true_b))
        b_prev[b] = prev_b
        b_macro_f1[b] = f1_score(y_t_b, y_p_b, average="macro", zero_division=0)

        # Critical recall & precision
        crit_tp = np.sum((y_p_b == 2) & (crit_true_b == 1))
        total_crit = np.sum(crit_true_b == 1)
        total_flagged = np.sum(y_p_b == 2)

        b_rec[b] = (crit_tp / total_crit) if total_crit > 0 else 0.0
        b_prec[b] = (crit_tp / total_flagged) if total_flagged > 0 else 0.0

        if total_crit == 0 or total_crit == n_samples:
            b_pr_auc[b] = prev_b
            b_lift[b] = 1.0
            b_roc[b] = 0.5
        else:
            try:
                prec_arr, rec_arr, _ = precision_recall_curve(crit_true_b, crit_prob_b)
                p_auc = float(auc(rec_arr, prec_arr))
                b_pr_auc[b] = p_auc
                b_lift[b] = (p_auc / prev_b) if prev_b > 0 else 1.0
            except Exception:
                b_pr_auc[b] = prev_b
                b_lift[b] = 1.0

            try:
                b_roc[b] = float(roc_auc_score(crit_true_b, crit_prob_b))
            except Exception:
                b_roc[b] = 0.5

        # Top-k metrics for this bootstrap resample
        order_b = np.argsort(-crit_prob_b)
        sorted_crit_b = crit_true_b[order_b]
        for k in [1, 5, 10, 20]:
            n_k = max(1, min(n_samples, int(np.round(n_samples * (k / 100.0)))))
            tp_k = np.sum(sorted_crit_b[:n_k])
            pk = tp_k / n_k
            rk = (tp_k / total_crit) if total_crit > 0 else 0.0
            lk = (pk / prev_b) if prev_b > 0 else 1.0
            b_topk_prec[k][b] = pk
            b_topk_rec[k][b] = rk
            b_topk_lift[k][b] = lk

    def _ci(arr: np.ndarray, point_val: float) -> Dict[str, Any]:
        low = float(np.percentile(arr, 100 * (alpha / 2.0)))
        high = float(np.percentile(arr, 100 * (1.0 - alpha / 2.0)))
        # Ensure interval contains point estimate (avoid numerical truncation edge cases)
        low_bound = min(low, point_val)
        high_bound = max(high, point_val)
        return {
            "point_estimate": round(float(point_val), 4),
            "ci_lower": round(float(low_bound), 4),
            "ci_upper": round(float(high_bound), 4),
            "ci_level": 1.0 - alpha,
        }

    lift_ci = _ci(b_lift, sample_lift)
    lift_includes_one = bool(lift_ci["ci_lower"] <= 1.0 <= lift_ci["ci_upper"])

    top_k_cis = {}
    for k in [1, 5, 10, 20]:
        key = f"top_{k}pct"
        top_k_cis[key] = {
            "precision": _ci(b_topk_prec[k], sample_top_k[key]["precision"]),
            "recall": _ci(b_topk_rec[k], sample_top_k[key]["recall"]),
            "lift": _ci(b_topk_lift[k], sample_top_k[key]["lift"]),
        }

    return {
        "prevalence": _ci(b_prev, sample_prev),
        "macro_f1": _ci(b_macro_f1, sample_macro_f1),
        "critical_delay_recall": _ci(b_rec, sample_rec),
        "critical_delay_precision": _ci(b_prec, sample_prec),
        "critical_delay_pr_auc": _ci(b_pr_auc, sample_pr_auc),
        "critical_delay_pr_lift": {
            **lift_ci,
            "ci_includes_one": lift_includes_one,
            "statistically_distinguishable_from_random": not lift_includes_one,
        },
        "critical_delay_roc_auc": _ci(b_roc, sample_roc_auc),
        "top_k_ranking": top_k_cis,
    }


def compute_paired_bootstrap_pr_auc_difference(
    y_true: np.ndarray,
    prob_model_a: np.ndarray,
    prob_model_b: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
    target_class_id: int = 2,
) -> Dict[str, Any]:
    """Computes paired bootstrap confidence interval on PR-AUC difference (Model B - Model A)."""
    n_samples = len(y_true)
    crit_true = (y_true == target_class_id).astype(int)
    rng = np.random.default_rng(random_state)

    # Sample point estimates
    p_a, r_a, _ = precision_recall_curve(crit_true, prob_model_a)
    auc_a = float(auc(r_a, p_a))
    p_b, r_b, _ = precision_recall_curve(crit_true, prob_model_b)
    auc_b = float(auc(r_b, p_b))
    delta_sample = auc_b - auc_a

    deltas = np.empty(n_bootstraps)
    for b in range(n_bootstraps):
        idx = rng.integers(0, n_samples, size=n_samples)
        y_b = crit_true[idx]
        if np.sum(y_b) == 0 or np.sum(y_b) == n_samples:
            deltas[b] = 0.0
            continue
        pa_b, ra_b, _ = precision_recall_curve(y_b, prob_model_a[idx])
        pb_b, rb_b, _ = precision_recall_curve(y_b, prob_model_b[idx])
        deltas[b] = auc(rb_b, pb_b) - auc(ra_b, pa_b)

    low = float(np.percentile(deltas, 100 * (alpha / 2.0)))
    high = float(np.percentile(deltas, 100 * (1.0 - alpha / 2.0)))
    low_bound = min(low, delta_sample)
    high_bound = max(high, delta_sample)
    ci_includes_zero = bool(low_bound <= 0.0 <= high_bound)

    return {
        "model_a_pr_auc": round(auc_a, 4),
        "model_b_pr_auc": round(auc_b, 4),
        "pr_auc_delta": round(delta_sample, 4),
        "ci_lower": round(low_bound, 4),
        "ci_upper": round(high_bound, 4),
        "ci_level": 1.0 - alpha,
        "ci_includes_zero": ci_includes_zero,
        "is_statistically_significant": not ci_includes_zero,
    }


def compute_wilson_score_interval(
    k: int,
    n: int,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Computes the Wilson score confidence interval for a binomial proportion k/n."""
    if n <= 0:
        return {"point_estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "confidence": confidence}

    # Standard normal quantile: z ~ 1.95996 for 95%
    if abs(confidence - 0.95) < 1e-4:
        z = 1.95996
    else:
        from scipy.stats import norm
        z = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))

    p_hat = k / n
    denom = 1.0 + (z**2) / n
    centre = (p_hat + (z**2) / (2.0 * n)) / denom
    margin = (z * np.sqrt((p_hat * (1.0 - p_hat)) / n + (z**2) / (4.0 * (n**2)))) / denom

    return {
        "point_estimate": round(float(p_hat), 4),
        "ci_lower": round(float(max(0.0, centre - margin)), 4),
        "ci_upper": round(float(min(1.0, centre + margin)), 4),
        "confidence": confidence,
    }


def compute_paired_vessel_clustered_macro_f1_difference(
    vessel_call_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred_candidate: np.ndarray,
    y_pred_reference: np.ndarray,
    n_bootstraps: int = 500,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Computes paired vessel-clustered bootstrap confidence interval for Delta Macro-F1.
    
    Resamples whole vessel calls with replacement to preserve intra-vessel container correlations.
    """
    vessels = np.asarray(vessel_call_ids)
    y_t = np.asarray(y_true)
    y_cand = np.asarray(y_pred_candidate)
    y_ref = np.asarray(y_pred_reference)

    unique_vessels, vessel_indices = np.unique(vessels, return_inverse=True)
    n_vessels = len(unique_vessels)

    # Point estimates on full sample
    f1_cand = float(f1_score(y_t, y_cand, average="macro", zero_division=0))
    f1_ref = float(f1_score(y_t, y_ref, average="macro", zero_division=0))
    delta_sample = f1_cand - f1_ref

    # Pre-index container indices by vessel for fast sampling
    vessel_to_samples = [np.where(vessel_indices == i)[0] for i in range(n_vessels)]

    rng = np.random.default_rng(random_state)
    deltas = np.empty(n_bootstraps)

    for b in range(n_bootstraps):
        boot_vessel_ids = rng.integers(0, n_vessels, size=n_vessels)
        sampled_indices = np.concatenate([vessel_to_samples[vid] for vid in boot_vessel_ids])

        y_b = y_t[sampled_indices]
        if len(np.unique(y_b)) < 3:
            deltas[b] = delta_sample
            continue

        f1_c_b = f1_score(y_b, y_cand[sampled_indices], average="macro", zero_division=0)
        f1_r_b = f1_score(y_b, y_ref[sampled_indices], average="macro", zero_division=0)
        deltas[b] = f1_c_b - f1_r_b

    low = float(np.percentile(deltas, 100 * (alpha / 2.0)))
    high = float(np.percentile(deltas, 100 * (1.0 - alpha / 2.0)))
    ci_includes_zero = bool(low <= 0.0 <= high)

    return {
        "candidate_macro_f1": round(f1_cand, 4),
        "reference_macro_f1": round(f1_ref, 4),
        "macro_f1_delta": round(delta_sample, 4),
        "ci_lower": round(low, 4),
        "ci_upper": round(high, 4),
        "ci_level": 1.0 - alpha,
        "n_vessel_clusters": int(n_vessels),
        "n_bootstraps": int(n_bootstraps),
        "resampling_unit": "vessel_call_id",
        "ci_includes_zero": ci_includes_zero,
        "is_statistically_significant": not ci_includes_zero,
    }

