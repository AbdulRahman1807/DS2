"""Evaluation Engineer Master Metrics Calculation Engine.

Provides standardized, mathematically rigorous regression and supply chain evaluation metrics:
1. Huber Loss (configurable delta: delta=1.0 for vessel calls, delta=50000.0 for cargo tonnage)
2. Mean Absolute Error (MAE)
3. Median Absolute Error (Median AE)
4. Root Mean Squared Error (RMSE)
5. Coefficient of Determination (R^2)
6. Mean Prediction Bias (pred - actual)
7. Weighted Absolute Percentage Error (WAPE)
8. Pinball / Quantile Loss (P10, P50, P90)
9. Stratified Activity-Bucket Error Profiling
10. Temporal Monthly Stability Profiling
11. Spatial Port Win / Tie / Regression Analysis
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def compute_metrics(
    actual: Union[np.ndarray, pd.Series],
    pred: Union[np.ndarray, pd.Series],
    delta: float = 1.0,
) -> Dict[str, float]:
    """Compute comprehensive regression metrics against ground truth.
    
    Args:
        actual: 1D array of ground truth target values.
        pred: 1D array of predicted values.
        delta: Threshold parameter for Huber loss calculation.
        
    Returns:
        Dictionary containing huber, mae, median_ae, rmse, r2, bias, and wape.
    """
    y = np.asarray(actual, dtype=np.float64).ravel()
    y_hat = np.asarray(pred, dtype=np.float64).ravel()

    if len(y) != len(y_hat):
        raise ValueError(f"Shape mismatch: actual ({len(y)}) vs pred ({len(y_hat)})")

    if len(y) == 0:
        return {
            "huber": 0.0,
            "mae": 0.0,
            "median_ae": 0.0,
            "rmse": 0.0,
            "r2": 0.0,
            "bias": 0.0,
            "wape": 0.0,
        }

    err = y_hat - y
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    med_ae = float(np.median(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))

    # Huber Loss
    huber_mask = abs_err <= delta
    huber = np.where(huber_mask, 0.5 * (err ** 2), delta * (abs_err - 0.5 * delta))
    huber_loss = float(np.mean(huber))

    # Coefficient of Determination (R^2)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum(err ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-12 else 0.0

    # Weighted Absolute Percentage Error (WAPE)
    sum_y = float(np.sum(np.abs(y)))
    wape = float(np.sum(abs_err) / sum_y) if sum_y > 1e-12 else 0.0

    return {
        "huber": round(huber_loss, 4),
        "mae": round(mae, 4),
        "median_ae": round(med_ae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "bias": round(bias, 4),
        "wape": round(wape, 4),
    }


def pinball_loss(
    actual: Union[np.ndarray, pd.Series],
    pred: Union[np.ndarray, pd.Series],
    quantile: float = 0.5,
) -> float:
    """Compute pinball / quantile loss for a specified quantile tau in (0, 1)."""
    y = np.asarray(actual, dtype=np.float64).ravel()
    y_hat = np.asarray(pred, dtype=np.float64).ravel()
    err = y - y_hat
    loss = np.maximum(quantile * err, (quantile - 1.0) * err)
    return float(np.mean(loss))


def distribution_statistics(arr: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """Compute summary distribution statistics for targets or predictions."""
    a = np.asarray(arr, dtype=np.float64).ravel()
    if len(a) == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "q25": 0.0, "median": 0.0, "q75": 0.0, "max": 0.0}
    return {
        "count": int(len(a)),
        "mean": round(float(np.mean(a)), 4),
        "std": round(float(np.std(a)), 4),
        "min": round(float(np.min(a)), 4),
        "q25": round(float(np.percentile(a, 25)), 4),
        "median": round(float(np.median(a)), 4),
        "q75": round(float(np.percentile(a, 75)), 4),
        "max": round(float(np.max(a)), 4),
    }


def compute_activity_buckets(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    naive_col: str,
    bins: List[float],
    labels: List[str],
    delta: float = 1.0,
) -> Dict[str, Dict[str, Any]]:
    """Compute diagnostic error breakdown across stratified activity buckets."""
    total_n = len(df)
    bucketed = pd.cut(df[actual_col], bins=bins, labels=labels, include_lowest=True)
    results = {}
    for label, group in df.groupby(bucketed, observed=True):
        m_model = compute_metrics(group[actual_col].values, group[pred_col].values, delta=delta)
        m_naive = compute_metrics(group[actual_col].values, group[naive_col].values, delta=delta)
        mae_delta = float(m_naive["mae"] - m_model["mae"])
        pct_improvement = round(mae_delta / m_naive["mae"] * 100, 2) if m_naive["mae"] > 0 else 0.0
        results[str(label)] = {
            "count": int(len(group)),
            "pct_of_total": round(float(len(group) / total_n * 100), 2),
            "model_mae": m_model["mae"],
            "naive_mae": m_naive["mae"],
            "mae_delta": round(mae_delta, 4),
            "mae_improvement_pct": pct_improvement,
            "model_rmse": m_model["rmse"],
            "naive_rmse": m_naive["rmse"],
            "model_huber": m_model["huber"],
            "naive_huber": m_naive["huber"],
            "model_median_ae": m_model["median_ae"],
            "naive_median_ae": m_naive["median_ae"],
        }
    return results


def compute_monthly_stability(
    df: pd.DataFrame,
    date_col: str,
    actual_col: str,
    pred_col: str,
    naive_col: str,
    delta: float = 1.0,
) -> Dict[str, Dict[str, Any]]:
    """Assess temporal stability and concept drift across calendar months."""
    months = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m")
    results = {}
    for month_str, group in df.groupby(months):
        m_model = compute_metrics(group[actual_col].values, group[pred_col].values, delta=delta)
        m_naive = compute_metrics(group[actual_col].values, group[naive_col].values, delta=delta)
        mae_delta = float(m_naive["mae"] - m_model["mae"])
        pct_improvement = round(mae_delta / m_naive["mae"] * 100, 2) if m_naive["mae"] > 0 else 0.0
        results[str(month_str)] = {
            "count": int(len(group)),
            "model_mae": m_model["mae"],
            "naive_mae": m_naive["mae"],
            "mae_delta": round(mae_delta, 4),
            "mae_improvement_pct": pct_improvement,
            "model_rmse": m_model["rmse"],
            "naive_rmse": m_naive["rmse"],
            "model_r2": m_model["r2"],
            "naive_r2": m_naive["r2"],
        }
    return results


def compute_port_spatial_summary(
    df: pd.DataFrame,
    port_id_col: str,
    port_name_col: str,
    actual_col: str,
    pred_col: str,
    naive_col: str,
    delta_threshold: float = 1e-4,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute per-port error aggregations and win/tie/loss classifications."""
    records = []
    n_improved, n_tied, n_worsened = 0, 0, 0

    for pid, group in df.groupby(port_id_col):
        pname = group[port_name_col].iloc[0] if port_name_col in group.columns else str(pid)
        act = group[actual_col].values
        prd = group[pred_col].values
        nve = group[naive_col].values

        mae_m = float(np.mean(np.abs(prd - act)))
        mae_n = float(np.mean(np.abs(nve - act)))
        diff = mae_n - mae_m  # Positive means model improved over naive

        if diff > delta_threshold:
            n_improved += 1
            verdict = "Improved"
        elif diff < -delta_threshold:
            n_worsened += 1
            verdict = "Worsened"
        else:
            n_tied += 1
            verdict = "Tied"

        pct_imp = round(diff / mae_n * 100, 2) if mae_n > 0 else 0.0
        records.append({
            "port_id": pid,
            "port_name": pname,
            "sequence_count": int(len(group)),
            "target_mean": round(float(np.mean(act)), 2),
            "model_mae": round(mae_m, 4),
            "naive_mae": round(mae_n, 4),
            "mae_delta": round(diff, 4),
            "pct_improvement": pct_imp,
            "verdict": verdict,
        })

    df_summary = pd.DataFrame(records)
    total_ports = len(df_summary)
    summary_meta = {
        "total_ports": total_ports,
        "ports_improved": n_improved,
        "ports_improved_pct": round(n_improved / total_ports * 100, 2) if total_ports > 0 else 0.0,
        "ports_tied": n_tied,
        "ports_tied_pct": round(n_tied / total_ports * 100, 2) if total_ports > 0 else 0.0,
        "ports_worsened": n_worsened,
        "ports_worsened_pct": round(n_worsened / total_ports * 100, 2) if total_ports > 0 else 0.0,
        "ports_improved_or_tied_pct": round((n_improved + n_tied) / total_ports * 100, 2) if total_ports > 0 else 0.0,
        "top_10_improved": df_summary.sort_values("mae_delta", ascending=False).head(10).to_dict(orient="records"),
        "top_10_worsened": df_summary.sort_values("mae_delta", ascending=True).head(10).to_dict(orient="records"),
    }
    return df_summary, summary_meta

