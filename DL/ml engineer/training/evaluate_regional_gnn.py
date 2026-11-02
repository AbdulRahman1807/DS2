"""Comprehensive Diagnostic Evaluation Suite for Milestone 3.3 Regional GNNs.

Computes rigorous statistical comparisons between:
1. Naive Persistence Baseline
2. Milestone 3.2 Exp 03 Reference Temporal Baseline (Residual LSTM)
3. Caribbean ST-GNN (Actual GATv2 Topology)
4. Caribbean ST-GNN (Shuffled Edge Ablation)
5. Caribbean ST-GNN (No-Edge / Identity Ablation)
6. Caribbean ST-GNN (GCN Topology)

Evaluates:
- Global metrics: Huber loss (delta=1.0), MAE, RMSE, R-squared
- Paired sequence-level differences: mean delta AE, win rate (%), clustered t-stat, p-value
- Port-level breakdown: win rate (% ports improved), best/worst ports
- Volume quartile breakdown: Q1 (low) to Q4 (mega-hubs)
- Graph centrality correlation: Spearman/Pearson r between degree/voyages and predictive delta
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from data.caribbean_dataset import load_caribbean_topology
from data.storage import MemmapStorageReader


def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
    err = np.abs(y_true - y_pred)
    linear = err - 0.5 * delta
    quadratic = 0.5 * (err ** 2) / delta
    return float(np.mean(np.where(err <= delta, quadratic, linear)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_true - y_pred
    abs_err = np.abs(err)
    sq_err = err ** 2
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(sq_err)))
    huber = huber_loss(y_true, y_pred, delta=1.0)
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    ss_res = float(np.sum(sq_err))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {
        "huber": huber,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mean_error": float(np.mean(err)),
        "std_error": float(np.std(err)),
    }


def compare_paired(
    y_true: np.ndarray,
    y_ref: np.ndarray,
    y_cand: np.ndarray,
    cluster_ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    ae_ref = np.abs(y_true - y_ref)
    ae_cand = np.abs(y_true - y_cand)
    diff = ae_ref - ae_cand  # positive means candidate improved over reference

    mean_diff = float(np.mean(diff))
    median_diff = float(np.median(diff))
    pct_improved = float(np.mean(diff > 0.0) * 100.0)
    pct_tied = float(np.mean(diff == 0.0) * 100.0)
    pct_worsened = float(np.mean(diff < 0.0) * 100.0)

    # Clustered standard error if cluster_ids provided
    if cluster_ids is not None:
        unique_clusters = np.unique(cluster_ids)
        G = len(unique_clusters)
        cluster_means = np.array([np.mean(diff[cluster_ids == c]) for c in unique_clusters])
        se_clustered = np.std(cluster_means, ddof=1) / np.sqrt(G)
        t_stat = mean_diff / se_clustered if se_clustered > 0 else 0.0
        p_val = float(2 * (1 - stats.t.cdf(np.abs(t_stat), df=G - 1)))
    else:
        se = np.std(diff, ddof=1) / np.sqrt(len(diff))
        t_stat = mean_diff / se if se > 0 else 0.0
        p_val = float(2 * (1 - stats.norm.cdf(np.abs(t_stat))))

    # Wilcoxon signed-rank test
    non_zero_diff = diff[diff != 0]
    if len(non_zero_diff) > 0:
        w_res = stats.wilcoxon(non_zero_diff, alternative="two-sided")
        wilcoxon_stat = float(w_res.statistic)
        wilcoxon_p = float(w_res.pvalue)
    else:
        wilcoxon_stat = 0.0
        wilcoxon_p = 1.0

    return {
        "mean_ae_diff": mean_diff,
        "median_ae_diff": median_diff,
        "pct_improved": pct_improved,
        "pct_tied": pct_tied,
        "pct_worsened": pct_worsened,
        "t_stat": t_stat,
        "p_val": p_val,
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_p": wilcoxon_p,
    }


def analyze_regional_models(
    exp03_val_parquet: str,
    gnn_val_parquets: Dict[str, str],
    edges_parquet: str,
    meta_parquet: str,
    storage_dir: str,
) -> Dict[str, Any]:
    """Execute end-to-end multi-model diagnostic audit across Caribbean ports."""
    topo = load_caribbean_topology(edges_parquet)
    carib_ports = set(topo["ports"])

    # Load Exp 03 validation predictions and filter to Caribbean
    df_exp03 = pd.read_parquet(exp03_val_parquet)
    df_carib_exp03 = df_exp03[df_exp03["portwatch_id"].isin(carib_ports)].copy()
    df_carib_exp03 = df_carib_exp03.sort_values(["reference_date", "portwatch_id"]).reset_index(drop=True)

    # Compute unscaled naive persistence directly from storage memmap
    sr = MemmapStorageReader(storage_dir)
    roll7_idx = 12
    roll7_mean = 2.4068180554238894
    roll7_std = 6.3483944613159915

    y_naive_list = []
    for _, row in df_carib_exp03.iterrows():
        p_idx = sr.port2idx[row["portwatch_id"]]
        d_idx = sr.date2idx[row["reference_date"]]
        obs_end_idx = d_idx - 2
        val_scaled = sr.array[p_idx, obs_end_idx, roll7_idx]
        val_unscaled = val_scaled * roll7_std + roll7_mean
        y_naive_list.append(float(7.0 * val_unscaled))

    y_true = df_carib_exp03["y_true"].to_numpy()
    y_naive = np.array(y_naive_list, dtype=np.float32)
    y_exp03 = df_carib_exp03["y_pred"].to_numpy()
    port_ids = df_carib_exp03["portwatch_id"].to_numpy()
    dates = df_carib_exp03["reference_date"].to_numpy()

    # Compute Naive and Exp 03 benchmark metrics
    benchmarks = {
        "naive_persistence": compute_metrics(y_true, y_naive),
        "exp03_temporal_residual": compute_metrics(y_true, y_exp03),
    }

    # Paired: Exp 03 vs Naive
    benchmarks["exp03_vs_naive"] = compare_paired(y_true, y_naive, y_exp03, cluster_ids=port_ids)

    results = {
        "num_sequences": len(y_true),
        "num_ports": len(carib_ports),
        "benchmarks": benchmarks,
        "models": {},
        "comparisons": {},
        "volume_breakdowns": {},
        "per_port_breakdowns": {},
        "graph_centrality_correlation": {},
    }

    # Evaluate each GNN model
    for model_name, gnn_path in gnn_val_parquets.items():
        if not Path(gnn_path).exists():
            print(f"Warning: File {gnn_path} not found. Skipping {model_name}.")
            continue

        df_gnn = pd.read_parquet(gnn_path)
        df_gnn = df_gnn.sort_values(["reference_date", "portwatch_id"]).reset_index(drop=True)

        if len(df_gnn) != len(df_carib_exp03):
            raise ValueError(f"Mismatch in sequence count: GNN={len(df_gnn)}, Exp03={len(df_carib_exp03)}")

        # Check alignment
        if not (df_gnn["sequence_id"].to_numpy() == df_carib_exp03["sequence_id"].to_numpy()).all():
            raise ValueError("Sequence ID alignment failed between GNN and Exp03!")

        y_gnn = df_gnn["y_pred"].to_numpy()
        results["models"][model_name] = compute_metrics(y_true, y_gnn)

        # Compare vs Exp 03
        results["comparisons"][f"{model_name}_vs_exp03"] = compare_paired(
            y_true, y_exp03, y_gnn, cluster_ids=port_ids
        )
        # Compare vs Naive
        results["comparisons"][f"{model_name}_vs_naive"] = compare_paired(
            y_true, y_naive, y_gnn, cluster_ids=port_ids
        )

        # Volume Quartile Breakdown
        # Determine port-level volume quartile based on median validation true port calls
        port_medians = df_carib_exp03.groupby("portwatch_id")["y_true"].mean().to_dict()
        q_labels = ["Q1 (<25%)", "Q2 (25-50%)", "Q3 (50-75%)", "Q4 (>75%)"]
        quartiles = pd.qcut(list(port_medians.values()), q=4, labels=q_labels)
        port_to_quartile = dict(zip(port_medians.keys(), quartiles))

        df_eval = df_carib_exp03[["sequence_id", "portwatch_id", "reference_date", "y_true"]].copy()
        df_eval["y_naive"] = y_naive
        df_eval["y_exp03"] = y_exp03
        df_eval["y_gnn"] = y_gnn
        df_eval["quartile"] = df_eval["portwatch_id"].map(port_to_quartile)

        vol_res = {}
        for q in q_labels:
            df_q = df_eval[df_eval["quartile"] == q]
            y_t_q = df_q["y_true"].to_numpy()
            y_n_q = df_q["y_naive"].to_numpy()
            y_e_q = df_q["y_exp03"].to_numpy()
            y_g_q = df_q["y_gnn"].to_numpy()
            vol_res[q] = {
                "num_ports": int(df_q["portwatch_id"].nunique()),
                "num_seqs": len(df_q),
                "naive": compute_metrics(y_t_q, y_n_q),
                "exp03": compute_metrics(y_t_q, y_e_q),
                "gnn": compute_metrics(y_t_q, y_g_q),
                "gnn_vs_exp03_pct_improved": float(np.mean(np.abs(y_t_q - y_e_q) > np.abs(y_t_q - y_g_q)) * 100.0),
                "gnn_mae_delta_vs_exp03": float(np.mean(np.abs(y_t_q - y_e_q)) - np.mean(np.abs(y_t_q - y_g_q))),
            }
        results["volume_breakdowns"][model_name] = vol_res

        # Per-Port Breakdown
        port_summary = []
        for p in sorted(list(carib_ports)):
            df_p = df_eval[df_eval["portwatch_id"] == p]
            y_t_p = df_p["y_true"].to_numpy()
            y_n_p = df_p["y_naive"].to_numpy()
            y_e_p = df_p["y_exp03"].to_numpy()
            y_g_p = df_p["y_gnn"].to_numpy()

            m_naive = compute_metrics(y_t_p, y_n_p)
            m_exp03 = compute_metrics(y_t_p, y_e_p)
            m_gnn = compute_metrics(y_t_p, y_g_p)

            port_summary.append({
                "portwatch_id": p,
                "mean_y_true": float(np.mean(y_t_p)),
                "naive_mae": m_naive["mae"],
                "exp03_mae": m_exp03["mae"],
                "gnn_mae": m_gnn["mae"],
                "delta_mae_gnn_vs_exp03": m_exp03["mae"] - m_gnn["mae"],  # positive means GNN is better
                "delta_mae_gnn_vs_naive": m_naive["mae"] - m_gnn["mae"],
                "pct_seqs_improved_vs_exp03": float(np.mean(np.abs(y_t_p - y_e_p) > np.abs(y_t_p - y_g_p)) * 100.0),
            })
        df_port_sum = pd.DataFrame(port_summary)
        results["per_port_breakdowns"][model_name] = df_port_sum.to_dict(orient="records")

        # Topology Correlation Analysis
        # Compute degrees from topology
        A_raw = topo["adj_weights"].numpy()
        in_degrees = (A_raw > 0).sum(axis=0)
        out_degrees = (A_raw > 0).sum(axis=1)
        tot_degrees = in_degrees + out_degrees
        in_voyages = A_raw.sum(axis=0)
        out_voyages = A_raw.sum(axis=1)
        tot_voyages = in_voyages + out_voyages

        df_topo = pd.DataFrame({
            "portwatch_id": topo["ports"],
            "in_degree": in_degrees,
            "out_degree": out_degrees,
            "tot_degree": tot_degrees,
            "in_voyages": in_voyages,
            "out_voyages": out_voyages,
            "tot_voyages": tot_voyages,
        })
        df_merged = df_port_sum.merge(df_topo, on="portwatch_id")

        corr_res = {}
        for topo_col in ["tot_degree", "in_degree", "out_degree", "tot_voyages"]:
            r_val, p_val = stats.pearsonr(df_merged[topo_col], df_merged["delta_mae_gnn_vs_exp03"])
            rho_val, rho_p = stats.spearmanr(df_merged[topo_col], df_merged["delta_mae_gnn_vs_exp03"])
            corr_res[topo_col] = {
                "pearson_r": float(r_val),
                "pearson_p": float(p_val),
                "spearman_rho": float(rho_val),
                "spearman_p": float(rho_p),
            }
        results["graph_centrality_correlation"][model_name] = corr_res

    return results


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp03-parquet", default="DL/ml engineer/artifacts/lstm/experiments/exp03_residual_persistence/validation_predictions.parquet")
    parser.add_argument("--gnn-dir", default="DL/ml engineer/artifacts/gnn/caribbean")
    parser.add_argument("--edges-parquet", default="DL/data/staging/supplementary_acquisition/processed/graph_ready/caribbean_graph_edges.parquet")
    parser.add_argument("--meta-parquet", default="DL/data/staging/supplementary_acquisition/processed/temporal_sequences/sequence_metadata.parquet")
    parser.add_argument("--storage-dir", default="DL/ml engineer/artifacts/preprocessing/storage")
    parser.add_argument("--output-json", default="DL/ml engineer/artifacts/gnn/caribbean/diagnostic_results.json")
    args = parser.parse_args()

    gnn_models = {}
    base_gnn_dir = Path(args.gnn_dir)
    for model_dir in base_gnn_dir.iterdir():
        if model_dir.is_dir():
            val_p = model_dir / "validation_predictions.parquet"
            if val_p.exists():
                gnn_models[model_dir.name] = str(val_p)

    print(f"Found {len(gnn_models)} GNN model prediction sets: {list(gnn_models.keys())}")
    res = analyze_regional_models(
        args.exp03_parquet,
        gnn_models,
        args.edges_parquet,
        args.meta_parquet,
        args.storage_dir,
    )

    with open(args.output_json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Diagnostic results successfully saved to {args.output_json}")

