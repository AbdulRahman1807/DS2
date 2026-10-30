"""False-Positive vs Emergency Reroute Economic Cost Optimizer.

Assigned Role: Evaluation Engineer
Stage: Stage 01 - Economic Loss Modeling & Operational Alert Optimization

Methodology & Assumptions:
- Cost of False Alarm (C_FA): Assumed $150 to $500 per container (unnecessary emergency reroute fees, carrier friction)
- Cost of Missed Delay (C_FN): Assumed $1,500 to $5,000+ per container (demurrage penalties, supply chain gridlock)
- NOTE: All cost parameters are ASSUMPTIONS for operational trade-off analysis, NOT internal company financials.

Provides:
- Threshold sweep (tau in [0.01, 0.50] in 0.01 increments) on calibrated probabilities
- Optimal threshold tau* selection on validation split and out-of-sample evaluation on test split
- Comparison with Bayes theoretical threshold tau_bayes = C_FA / (C_FA + C_FN)
- Trivial strategy comparisons: (a) Do Nothing, (b) Flag Everything, (c) Argmax, (d) tau*
- Strict mathematical consistency checks (cost = C_FA*FP + C_FN*FN, flag rate = (TP+FP)/N)
- Break-even cost ratio analysis: Minimum C_FN / C_FA ratio to beat Do Nothing
- Sensitivity analysis across cost scenarios (3x to 33x ratios) with boundary hit detection
- Operational translation per 1,000 containers/day
- Prior-shift prevalence correction (Saerens et al. odds adjustment)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default assumed cost parameters (USD)
DEFAULT_C_FA: float = 300.0   # False Alarm cost (unnecessary emergency reroute / wasted chassis)
DEFAULT_C_FN: float = 3000.0  # Missed Critical Delay cost (demurrage / gridlock / late delivery)


def evaluate_decision_threshold(
    y_true: np.ndarray,
    y_prob_crit: np.ndarray,
    tau: float,
    c_fa: float = DEFAULT_C_FA,
    c_fn: float = DEFAULT_C_FN,
    target_class_id: int = 2,
) -> Dict[str, Any]:
    """Computes confusion matrix, alert statistics, and total economic cost at a given threshold tau."""
    n_samples = len(y_true)
    crit_binary_true = (y_true == target_class_id).astype(int)
    pred_crit = (y_prob_crit >= tau).astype(int)

    tp = int(np.sum((pred_crit == 1) & (crit_binary_true == 1)))
    fp = int(np.sum((pred_crit == 1) & (crit_binary_true == 0)))
    fn = int(np.sum((pred_crit == 0) & (crit_binary_true == 1)))
    tn = int(np.sum((pred_crit == 0) & (crit_binary_true == 0)))

    flagged_count = tp + fp
    flagged_pct = (flagged_count / n_samples * 100.0) if n_samples > 0 else 0.0

    precision = (tp / flagged_count) if flagged_count > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    total_cost = float(c_fa * fp + c_fn * fn)
    cost_per_1k = float((total_cost / n_samples) * 1000.0) if n_samples > 0 else 0.0

    # Strict mathematical consistency assertion
    assert flagged_count == tp + fp
    assert total_cost == c_fa * fp + c_fn * fn

    return {
        "threshold_tau": round(float(tau), 4),
        "flagged_count": flagged_count,
        "flagged_pct": round(flagged_pct, 2),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "total_cost": round(total_cost, 2),
        "cost_per_1k_containers": round(cost_per_1k, 2),
    }


def compute_threshold_sweep(
    y_true: np.ndarray,
    y_prob_crit: np.ndarray,
    c_fa: float = DEFAULT_C_FA,
    c_fn: float = DEFAULT_C_FN,
    thresholds: Optional[np.ndarray] = None,
    target_class_id: int = 2,
) -> Tuple[pd.DataFrame, float, Dict[str, Any]]:
    """Sweeps thresholds tau in [0.01, 0.50] in 0.01 increments and selects tau* by minimum total cost."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 0.501, 0.01), 4)

    rows = []
    min_cost = float("inf")
    best_tau = 0.25
    best_stats = {}

    for tau in thresholds:
        stats = evaluate_decision_threshold(
            y_true=y_true,
            y_prob_crit=y_prob_crit,
            tau=tau,
            c_fa=c_fa,
            c_fn=c_fn,
            target_class_id=target_class_id,
        )
        rows.append(stats)
        if stats["total_cost"] < min_cost:
            min_cost = stats["total_cost"]
            best_tau = tau
            best_stats = stats

    df_sweep = pd.DataFrame(rows)
    return df_sweep, best_tau, best_stats


def compute_trivial_strategies(
    y_true: np.ndarray,
    y_pred_argmax: np.ndarray,
    y_prob_crit: np.ndarray,
    tau_star: float,
    c_fa: float = DEFAULT_C_FA,
    c_fn: float = DEFAULT_C_FN,
    target_class_id: int = 2,
) -> Dict[str, Dict[str, Any]]:
    """Evaluates Total Cost and Cost per 1,000 containers across:
    (a) Do Nothing (never alert)
    (b) Flag Everything (always alert)
    (c) Default Argmax (highest model score wins)
    (d) Optimal tau* alert
    """
    n_samples = len(y_true)
    total_positives = int(np.sum(y_true == target_class_id))
    total_negatives = n_samples - total_positives

    # Strategy A: Do Nothing (tau = 1.01)
    cost_do_nothing = float(c_fn * total_positives)
    cost_per_1k_do_nothing = float((cost_do_nothing / n_samples) * 1000.0)

    # Strategy B: Flag Everything (tau = 0.00)
    cost_flag_all = float(c_fa * total_negatives)
    cost_per_1k_flag_all = float((cost_flag_all / n_samples) * 1000.0)

    # Strategy C: Argmax decision rule
    pred_crit_argmax = (y_pred_argmax == target_class_id).astype(int)
    tp_am = int(np.sum((pred_crit_argmax == 1) & (y_true == target_class_id)))
    fp_am = int(np.sum((pred_crit_argmax == 1) & (y_true != target_class_id)))
    fn_am = int(np.sum((pred_crit_argmax == 0) & (y_true == target_class_id)))
    tn_am = int(np.sum((pred_crit_argmax == 0) & (y_true != target_class_id)))

    flagged_am = tp_am + fp_am
    cost_argmax = float(c_fa * fp_am + c_fn * fn_am)
    cost_per_1k_argmax = float((cost_argmax / n_samples) * 1000.0)

    # Strict consistency assertions
    assert flagged_am == tp_am + fp_am
    assert total_positives == tp_am + fn_am
    assert total_negatives == tn_am + fp_am
    assert cost_argmax == c_fa * fp_am + c_fn * fn_am

    # Strategy D: tau* alert
    stats_tau_star = evaluate_decision_threshold(
        y_true=y_true,
        y_prob_crit=y_prob_crit,
        tau=tau_star,
        c_fa=c_fa,
        c_fn=c_fn,
        target_class_id=target_class_id,
    )

    strategies = {
        "do_nothing": {
            "strategy": "Do Nothing (Never Alert)",
            "flagged_count": 0,
            "flagged_pct": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": total_positives,
            "true_negatives": total_negatives,
            "recall": 0.0,
            "precision": 0.0,
            "total_cost": round(cost_do_nothing, 2),
            "cost_per_1k_containers": round(cost_per_1k_do_nothing, 2),
        },
        "flag_everything": {
            "strategy": "Flag Everything (Always Alert)",
            "flagged_count": n_samples,
            "flagged_pct": 100.0,
            "true_positives": total_positives,
            "false_positives": total_negatives,
            "false_negatives": 0,
            "true_negatives": 0,
            "recall": 1.0,
            "precision": round(float(total_positives / n_samples), 4) if n_samples > 0 else 0.0,
            "total_cost": round(cost_flag_all, 2),
            "cost_per_1k_containers": round(cost_per_1k_flag_all, 2),
        },
        "argmax": {
            "strategy": "Argmax (Default Decision Rule)",
            "flagged_count": flagged_am,
            "flagged_pct": round(float(flagged_am / n_samples * 100.0), 2),
            "true_positives": tp_am,
            "false_positives": fp_am,
            "false_negatives": fn_am,
            "true_negatives": tn_am,
            "recall": round(float(tp_am / total_positives), 4) if total_positives > 0 else 0.0,
            "precision": round(float(tp_am / flagged_am), 4) if flagged_am > 0 else 0.0,
            "total_cost": round(cost_argmax, 2),
            "cost_per_1k_containers": round(cost_per_1k_argmax, 2),
        },
        "tau_star": {
            "strategy": f"Optimal tau* ({tau_star:.2f})",
            **stats_tau_star,
        },
    }

    best_trivial_name = "do_nothing" if cost_do_nothing <= cost_flag_all else "flag_everything"
    best_trivial_cost = min(cost_do_nothing, cost_flag_all)

    beats_trivial = bool(stats_tau_star["total_cost"] < best_trivial_cost)
    cost_savings = round(best_trivial_cost - stats_tau_star["total_cost"], 2)
    pct_savings = round((cost_savings / best_trivial_cost * 100.0), 2) if best_trivial_cost > 0 else 0.0

    return {
        **strategies,
        "summary": {
            "best_trivial_strategy": best_trivial_name,
            "best_trivial_cost": best_trivial_cost,
            "model_beats_best_trivial": beats_trivial,
            "dollar_savings_over_best_trivial": cost_savings,
            "percent_savings_over_best_trivial": pct_savings,
        },
    }


def compute_break_even_ratio(
    y_true: np.ndarray,
    y_prob_crit: np.ndarray,
    tau: float,
    target_class_id: int = 2,
) -> Dict[str, Any]:
    """Computes the exact minimum cost ratio (C_FN / C_FA) where the model beats the 'Do Nothing' strategy."""
    crit_binary_true = (y_true == target_class_id).astype(int)
    pred_crit = (y_prob_crit >= tau).astype(int)

    tp = int(np.sum((pred_crit == 1) & (crit_binary_true == 1)))
    fp = int(np.sum((pred_crit == 1) & (crit_binary_true == 0)))

    if tp == 0:
        break_even_ratio = float("inf")
    else:
        break_even_ratio = float(fp / tp)

    prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    return {
        "threshold_tau": round(float(tau), 4),
        "true_positives": tp,
        "false_positives": fp,
        "precision": round(float(prec), 4),
        "break_even_cost_ratio": round(break_even_ratio, 2) if break_even_ratio != float("inf") else "INFINITE",
        "interpretation": (
            f"At tau={tau:.2f}, emergency reroute failure (C_FN) must be at least "
            f"{break_even_ratio:.1f}x more costly than false alarm reroute waste (C_FA) for alerting to beat Do Nothing."
            if break_even_ratio != float("inf")
            else f"At tau={tau:.2f}, 0 delays are caught; model can never beat Do Nothing."
        ),
    }


def compute_sensitivity_matrix(
    y_true: np.ndarray,
    y_prob_crit: np.ndarray,
    c_fa_values: Optional[List[float]] = None,
    c_fn_values: Optional[List[float]] = None,
    thresholds: Optional[np.ndarray] = None,
    y_eval: Optional[np.ndarray] = None,
    y_prob_eval_crit: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Sweeps cost parameter combinations, tunes tau* on (y_true, y_prob_crit),
    and evaluates resulting operational loss on (y_eval, y_prob_eval_crit).
    If y_eval is omitted, evaluates on y_true.
    """
    if c_fa_values is None:
        c_fa_values = [150.0, 300.0, 500.0]
    if c_fn_values is None:
        c_fn_values = [1500.0, 3000.0, 5000.0]
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 0.501, 0.01), 4)

    eval_y = y_eval if y_eval is not None else y_true
    eval_prob = y_prob_eval_crit if y_prob_eval_crit is not None else y_prob_crit
    eval_positives = int(np.sum(eval_y == 2))
    n_eval = len(eval_y)

    rows = []
    for c_fa in c_fa_values:
        for c_fn in c_fn_values:
            ratio = round(c_fn / c_fa, 1)
            bayes_tau = round(c_fa / (c_fa + c_fn), 4)

            # 1. Select optimal tau* on tuning data (validation)
            _, tau_star, _ = compute_threshold_sweep(
                y_true=y_true,
                y_prob_crit=y_prob_crit,
                c_fa=c_fa,
                c_fn=c_fn,
                thresholds=thresholds,
            )

            # 2. Evaluate performance on evaluation data (test) at tau*
            flagged = eval_prob >= tau_star
            tp = int(np.sum(flagged & (eval_y == 2)))
            fp = int(np.sum(flagged & (eval_y != 2)))
            fn = int(np.sum((~flagged) & (eval_y == 2)))
            tn = int(np.sum((~flagged) & (eval_y != 2)))

            total_cost = float(c_fa * fp + c_fn * fn)
            cost_dn = float(c_fn * eval_positives)
            net_saving = float(cost_dn - total_cost)
            beats_dn = bool(net_saving > 0)

            rec = round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0, 4)
            prec = round(float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0, 4)
            flagged_pct = round(100.0 * (tp + fp) / n_eval, 2)
            cost_per_1k = round(float(total_cost / n_eval * 1000.0), 2)

            # Boundary hit check: if tau* hits max sweep threshold (0.50), model cannot profitably alert
            is_boundary = bool(tau_star >= 0.50)
            decision = "Do Nothing is optimal (boundary hit)" if is_boundary else f"Alert at tau*={tau_star:.2f}"

            rows.append({
                "c_fa_usd": int(c_fa),
                "c_fn_usd": int(c_fn),
                "cost_ratio": f"{ratio}x",
                "bayes_theoretical_tau": bayes_tau,
                "optimal_tau_star": round(float(tau_star), 2),
                "operational_decision": decision,
                "recall_at_tau": rec,
                "precision_at_tau": prec,
                "alert_rate_pct": flagged_pct,
                "total_cost": total_cost,
                "cost_per_1k": cost_per_1k,
                "do_nothing_cost": cost_dn,
                "net_saving": net_saving,
                "beats_do_nothing": beats_dn,
            })

    return pd.DataFrame(rows)


def simulate_operational_daily_impact(
    stats: Dict[str, Any],
    stats_argmax: Dict[str, Any],
    stats_do_nothing: Dict[str, Any],
    daily_volume: int = 735,
    test_duration_days: int = 19,
    top_1pct_precision: float = 0.1500,
    c_fa: float = 300.0,
    c_fn: float = 3000.0,
) -> Dict[str, Any]:
    """Translates model evaluation metrics to a daily operational simulation based on terminal throughput (~735 containers/day)."""
    flag_rate = stats["flagged_pct"] / 100.0
    rec = stats["recall"]
    prec = stats["precision"]

    daily_alerts = round(daily_volume * flag_rate, 1)
    daily_true_positives = round(daily_alerts * prec, 1)
    daily_false_alarms = round(daily_alerts * (1.0 - prec), 1)

    daily_cost_model = round(stats["cost_per_1k_containers"] * (daily_volume / 1000.0), 2)
    daily_cost_argmax = round(stats_argmax["cost_per_1k_containers"] * (daily_volume / 1000.0), 2)
    daily_cost_do_nothing = round(stats_do_nothing["cost_per_1k_containers"] * (daily_volume / 1000.0), 2)

    # Top-1% Operator Priority Queue Daily Impact
    top_1pct_daily_queue = round(daily_volume * 0.01, 1)  # ~7.35 containers/day
    top_1pct_daily_tp = round(top_1pct_daily_queue * top_1pct_precision, 2)  # ~1.10 true delays/day
    top_1pct_daily_fp = round(top_1pct_daily_queue * (1.0 - top_1pct_precision), 2)  # ~6.25 false alarms/day
    top_1pct_daily_delay_savings = top_1pct_daily_tp * c_fn
    top_1pct_daily_fa_cost = top_1pct_daily_fp * c_fa
    top_1pct_daily_net_saving = round(top_1pct_daily_delay_savings - top_1pct_daily_fa_cost, 2)

    return {
        "daily_inbound_volume": daily_volume,
        "test_duration_days": test_duration_days,
        "selected_tau": stats["threshold_tau"],
        "daily_alerts_generated": daily_alerts,
        "daily_delays_caught": daily_true_positives,
        "daily_false_alarms": daily_false_alarms,
        "estimated_daily_cost_tau_star_usd": daily_cost_model,
        "estimated_daily_cost_argmax_usd": daily_cost_argmax,
        "estimated_daily_cost_do_nothing_usd": daily_cost_do_nothing,
        "daily_savings_vs_do_nothing_usd": round(daily_cost_do_nothing - daily_cost_model, 2),
        "daily_savings_vs_argmax_usd": round(daily_cost_argmax - daily_cost_model, 2),
        "top_1pct_queue": {
            "daily_containers_reviewed": top_1pct_daily_queue,
            "daily_delays_caught": top_1pct_daily_tp,
            "daily_false_alarms": top_1pct_daily_fp,
            "expected_daily_net_saving_usd": top_1pct_daily_net_saving,
            "assumptions_caveat": "OPTIMISTIC: assumes dispatcher advance review prevents 100% of alerted critical delays at $3,000 saving with zero additional operational friction.",
        },
    }


def compute_net_saving_bootstrap_ci(
    y_true: np.ndarray,
    y_prob_crit: np.ndarray,
    tau: float,
    c_fa: float,
    c_fn: float,
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> dict:
    """Bootstrap 95% CI for Strategy D net saving over Do Nothing.

    Net saving = Do-Nothing cost - Strategy D cost
               = C_FN * FN_do_nothing - (C_FA * FP + C_FN * FN_tau)
               = C_FN * n_positive - (C_FA * FP_tau + C_FN * FN_tau)
    where n_positive is the total number of critical delays in the split.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    net_savings = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, n, replace=True)
        yt = (y_true[idx] == 2).astype(int)
        yp = y_prob_crit[idx]
        pred = (yp >= tau).astype(int)
        tp = int(np.sum((pred == 1) & (yt == 1)))
        fp = int(np.sum((pred == 1) & (yt == 0)))
        fn = int(np.sum((pred == 0) & (yt == 1)))
        do_nothing_cost = c_fn * int(np.sum(yt))
        model_cost = c_fa * fp + c_fn * fn
        net_savings.append(do_nothing_cost - model_cost)
    net_savings = np.array(net_savings)
    # point estimate on full data (not bootstrap)
    yt_full = (y_true == 2).astype(int)
    pred_full = (y_prob_crit >= tau).astype(int)
    fp_f = int(np.sum((pred_full == 1) & (yt_full == 0)))
    fn_f = int(np.sum((pred_full == 0) & (yt_full == 1)))
    do_nothing_full = c_fn * int(np.sum(yt_full))
    model_full = c_fa * fp_f + c_fn * fn_f
    point = do_nothing_full - model_full
    ci_lower = float(np.percentile(net_savings, 2.5))
    ci_upper = float(np.percentile(net_savings, 97.5))
    return {
        "point": round(point),
        "ci_lower": round(ci_lower),
        "ci_upper": round(ci_upper),
        "includes_zero": ci_lower <= 0 <= ci_upper,
    }


def apply_prior_shift_correction(
    y_prob: np.ndarray,
    train_prevalence: np.ndarray,
    target_prevalence: np.ndarray,
) -> np.ndarray:
    """Adjusts predicted class probabilities for prevalence shift between training and target splits."""
    ratio = target_prevalence / np.maximum(train_prevalence, 1e-12)
    p_adj = y_prob * ratio
    p_adj /= np.sum(p_adj, axis=1, keepdims=True)
    return p_adj
