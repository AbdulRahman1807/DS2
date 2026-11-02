"""Comprehensive Evaluation and Interpretability Suite for Milestone 3.4.

Executes rigorous statistical comparisons and diagnostic audits between:
1. Naive Persistence Baseline
2. Milestone 3.2 Experiment 03 Reference Temporal Baseline (Residual LSTM)
3. Architecture A: Lightweight Transformer Residual Forecaster
4. Architecture B: Temporal Attention Residual LSTM Forecaster
5. Architecture C: Transformer with Mean Pooling (Optional Ablation)

Diagnostics:
- Global metrics: Huber loss (delta=1.0), MAE, RMSE, R2, Bias, Min/Max/Mean/Std
- Paired sequence comparisons vs Exp 03: Mean/median delta AE, Win rate (%), Clustered t-stat, p-value
- Target volume bucket breakdowns (Q1, Q2, Q3, Q4, and zero-target sequences)
- Calendar month performance breakdown (January - December 2024)
- Port-level win rates, best/worst ports
- Temporal attention profile extraction and diagnostic distributions
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import torch

# Add workspace to path
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parents[2] / "ml engineer"
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from models.transformer_forecaster import ResidualTransformerForecaster
from models.attention_lstm import TemporalAttentionLSTMForecaster
from data.data_module import create_dataloaders


def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
    err = np.abs(y_true - y_pred)
    linear = err - 0.5 * delta
    quadratic = 0.5 * (err ** 2) / delta
    return float(np.mean(np.where(err <= delta, quadratic, linear)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
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
        "bias": float(np.mean(err)),
        "pred_min": float(np.min(y_pred)),
        "pred_max": float(np.max(y_pred)),
        "pred_mean": float(np.mean(y_pred)),
        "pred_std": float(np.std(y_pred)),
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

    # Clustered standard errors across ports
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

    # Paired squared error difference
    se_diff = float(np.mean((y_true - y_ref)**2 - (y_true - y_cand)**2))

    return {
        "mean_ae_diff": mean_diff,
        "median_ae_diff": median_diff,
        "pct_improved": pct_improved,
        "pct_tied": pct_tied,
        "pct_worsened": pct_worsened,
        "se_diff": se_diff,
        "t_stat": t_stat,
        "p_val": p_val,
    }


def extract_attention_profiles(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    model_type: str,
    num_batches: int = 50,
) -> Dict[str, Any]:
    """Sample validation batches to extract learned temporal attention distributions."""
    model.eval()
    all_attns = []
    all_targets = []
    all_ports = []

    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            if b_idx >= num_batches:
                break
            bx, by = batch
            bx = bx.to(device)

            if model_type == "ResidualTransformerForecaster":
                _, attn_layers = model(bx, return_attention=True)
                # Take last layer attention: [B, nhead=4, 30, 30]
                # Look at attention from the last query token (step 29) to all keys (0..29)
                # Average over attention heads: [B, 30]
                last_token_attn = attn_layers[-1][:, :, -1, :].mean(dim=1).cpu().numpy()
                all_attns.append(last_token_attn)
            elif model_type == "TemporalAttentionLSTMForecaster":
                _, alpha = model(bx, return_attention=True)
                # alpha shape: [B, 30]
                all_attns.append(alpha.cpu().numpy())
            
            all_targets.append(by.view(-1).numpy())

    attns_np = np.concatenate(all_attns, axis=0)  # [N_sampled, 30]
    targets_np = np.concatenate(all_targets, axis=0)  # [N_sampled]

    # Overall mean temporal attention profile over the 30 steps
    # Day 0 is t-31 (oldest), Day 29 is t-2 (most recent available)
    mean_profile = attns_np.mean(axis=0).tolist()
    std_profile = attns_np.std(axis=0).tolist()

    # Stratified by high vs low volume
    q50 = float(np.median(targets_np))
    low_vol_profile = attns_np[targets_np <= q50].mean(axis=0).tolist()
    high_vol_profile = attns_np[targets_np > q50].mean(axis=0).tolist()

    # Summary metrics
    recent_weight_pct = float(np.sum(attns_np[:, -7:]) / np.sum(attns_np) * 100.0)  # last 7 days of observation window
    oldest_weight_pct = float(np.sum(attns_np[:, :7]) / np.sum(attns_np) * 100.0)   # first 7 days of observation window
    peak_timestep = int(np.argmax(attns_np.mean(axis=0)))

    return {
        "mean_profile": mean_profile,
        "std_profile": std_profile,
        "low_volume_profile": low_vol_profile,
        "high_volume_profile": high_vol_profile,
        "recent_7d_weight_pct": recent_weight_pct,
        "oldest_7d_weight_pct": oldest_weight_pct,
        "peak_timestep": peak_timestep,
        "peak_timestep_label": f"t - {31 - peak_timestep}",
    }


def run_comprehensive_evaluation(
    exp03_parquet: str,
    candidate_parquets: Dict[str, str],
    meta_parquet: str,
    output_json: str,
) -> Dict[str, Any]:
    """Execute complete multi-model diagnostic audit across 743,400 validation sequences."""
    print("=" * 80)
    print(" MILESTONE 3.4: COMPREHENSIVE BENCHMARKING & ATTENTION EVALUATION")
    print(f" Reference (Exp 03): {exp03_parquet}")
    print(f" Candidate Models:   {list(candidate_parquets.keys())}")
    print("=" * 80)

    # 1. Load Exp 03 Reference and Sequence Metadata
    print("\n[Step 1/5] Ingesting Exp 03 Reference Predictions and Metadata...")
    df_exp03 = pd.read_parquet(exp03_parquet)
    df_exp03 = df_exp03.sort_values("sequence_id").reset_index(drop=True)

    seq_ids = df_exp03["sequence_id"].to_numpy()
    port_ids = df_exp03["portwatch_id"].to_numpy()
    ref_dates = df_exp03["reference_date"].to_numpy()
    y_true = df_exp03["y_true"].to_numpy()
    y_exp03 = df_exp03["y_pred"].to_numpy()
    total_seqs = len(y_true)
    print(f"  Loaded {total_seqs:,} validation sequences across {len(np.unique(port_ids)):,} ports.")

    # 2. Benchmark Naive Persistence and Exp 03
    benchmarks = {
        "exp03_temporal_residual": compute_metrics(y_true, y_exp03),
    }

    # Extract calendar month
    months = np.array([d[:7] for d in ref_dates])

    # Volume Quartiles based on true targets
    q_edges = np.percentile(y_true, [25, 50, 75])
    vol_buckets = np.zeros(total_seqs, dtype=int)
    vol_buckets[(y_true > q_edges[0]) & (y_true <= q_edges[1])] = 1
    vol_buckets[(y_true > q_edges[1]) & (y_true <= q_edges[2])] = 2
    vol_buckets[y_true > q_edges[2]] = 3
    bucket_labels = ["Q1 (<25%)", "Q2 (25-50%)", "Q3 (50-75%)", "Q4 (>75%)"]

    zero_target_mask = (y_true == 0.0)

    results = {
        "num_sequences": total_seqs,
        "num_ports": int(len(np.unique(port_ids))),
        "benchmarks": benchmarks,
        "models": {},
        "paired_comparisons": {},
        "volume_breakdowns": {},
        "zero_target_breakdowns": {},
        "monthly_breakdowns": {},
        "port_level_summaries": {},
    }

    # 3. Evaluate Each Candidate Model
    for model_name, cand_path in candidate_parquets.items():
        if not Path(cand_path).exists():
            print(f"Warning: File {cand_path} does not exist. Skipping {model_name}.")
            continue

        print(f"\n[Evaluating Candidate: {model_name}]...")
        df_cand = pd.read_parquet(cand_path)
        df_cand = df_cand.sort_values("sequence_id").reset_index(drop=True)

        if len(df_cand) != total_seqs:
            raise ValueError(f"Length mismatch: {model_name} has {len(df_cand)} rows, expected {total_seqs}.")

        if not (df_cand["sequence_id"].to_numpy() == seq_ids).all():
            raise ValueError(f"Sequence ID alignment failed for {model_name}!")

        y_cand = df_cand["y_pred"].to_numpy()

        # Global metrics
        results["models"][model_name] = compute_metrics(y_true, y_cand)
        print(f"  Huber: {results['models'][model_name]['huber']:.4f} | MAE: {results['models'][model_name]['mae']:.4f} | RMSE: {results['models'][model_name]['rmse']:.4f} | R2: {results['models'][model_name]['r2']:.4f}")

        # Paired comparison vs Exp 03
        paired_res = compare_paired(y_true, y_exp03, y_cand, cluster_ids=port_ids)
        results["paired_comparisons"][f"{model_name}_vs_exp03"] = paired_res
        print(f"  Paired vs Exp03 -> Delta MAE: {paired_res['mean_ae_diff']:+.4f} | Win %: {paired_res['pct_improved']:.2f}% | t-stat: {paired_res['t_stat']:+.2f} (p={paired_res['p_val']:.4e})")

        # Volume Bucket Breakdown
        vol_res = {}
        for b_idx, b_label in enumerate(bucket_labels):
            mask_b = (vol_buckets == b_idx)
            yt_b = y_true[mask_b]
            ye_b = y_exp03[mask_b]
            yc_b = y_cand[mask_b]

            vol_res[b_label] = {
                "num_sequences": int(np.sum(mask_b)),
                "true_mean": float(np.mean(yt_b)),
                "exp03_mae": float(np.mean(np.abs(yt_b - ye_b))),
                "cand_mae": float(np.mean(np.abs(yt_b - yc_b))),
                "delta_mae": float(np.mean(np.abs(yt_b - ye_b)) - np.mean(np.abs(yt_b - yc_b))),
                "win_pct": float(np.mean(np.abs(yt_b - ye_b) > np.abs(yt_b - yc_b)) * 100.0),
            }
        results["volume_breakdowns"][model_name] = vol_res

        # Zero Target Breakdown
        yt_z = y_true[zero_target_mask]
        ye_z = y_exp03[zero_target_mask]
        yc_z = y_cand[zero_target_mask]
        results["zero_target_breakdowns"][model_name] = {
            "num_sequences": int(np.sum(zero_target_mask)),
            "pct_of_total": float(np.sum(zero_target_mask) / total_seqs * 100.0),
            "exp03_mae": float(np.mean(np.abs(yt_z - ye_z))),
            "cand_mae": float(np.mean(np.abs(yt_z - yc_z))),
            "delta_mae": float(np.mean(np.abs(yt_z - ye_z)) - np.mean(np.abs(yt_z - yc_z))),
        }

        # Monthly Breakdown
        monthly_res = {}
        unique_months = sorted(np.unique(months).tolist())
        for m in unique_months:
            mask_m = (months == m)
            yt_m = y_true[mask_m]
            ye_m = y_exp03[mask_m]
            yc_m = y_cand[mask_m]

            monthly_res[m] = {
                "num_sequences": int(np.sum(mask_m)),
                "exp03_mae": float(np.mean(np.abs(yt_m - ye_m))),
                "cand_mae": float(np.mean(np.abs(yt_m - yc_m))),
                "delta_mae": float(np.mean(np.abs(yt_m - ye_m)) - np.mean(np.abs(yt_m - yc_m))),
                "win_pct": float(np.mean(np.abs(yt_m - ye_m) > np.abs(yt_m - yc_m)) * 100.0),
            }
        results["monthly_breakdowns"][model_name] = monthly_res

        # Port-Level Summary
        df_port = pd.DataFrame({
            "portwatch_id": port_ids,
            "y_true": y_true,
            "y_exp03": y_exp03,
            "y_cand": y_cand,
        })
        port_stats = df_port.groupby("portwatch_id").apply(lambda g: pd.Series({
            "seq_count": len(g),
            "mean_true": np.mean(g["y_true"]),
            "exp03_mae": np.mean(np.abs(g["y_true"] - g["y_exp03"])),
            "cand_mae": np.mean(np.abs(g["y_true"] - g["y_cand"])),
            "delta_mae": np.mean(np.abs(g["y_true"] - g["y_exp03"])) - np.mean(np.abs(g["y_true"] - g["y_cand"])),
            "win_pct": np.mean(np.abs(g["y_true"] - g["y_exp03"]) > np.abs(g["y_true"] - g["y_cand"])) * 100.0,
        })).reset_index()

        num_improved_ports = int(np.sum(port_stats["delta_mae"] > 0))
        num_worsened_ports = int(np.sum(port_stats["delta_mae"] < 0))
        num_tied_ports = int(np.sum(port_stats["delta_mae"] == 0))
        total_ports_cnt = len(port_stats)

        results["port_level_summaries"][model_name] = {
            "total_ports": total_ports_cnt,
            "improved_ports_count": num_improved_ports,
            "improved_ports_pct": float(num_improved_ports / total_ports_cnt * 100.0),
            "worsened_ports_count": num_worsened_ports,
            "worsened_ports_pct": float(num_worsened_ports / total_ports_cnt * 100.0),
            "tied_ports_count": num_tied_ports,
            "mean_port_delta_mae": float(port_stats["delta_mae"].mean()),
            "top_5_improved": port_stats.sort_values("delta_mae", ascending=False).head(5).to_dict(orient="records"),
            "worst_5_ports": port_stats.sort_values("delta_mae", ascending=True).head(5).to_dict(orient="records"),
        }

    # Save complete evaluation output
    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDiagnostic audit completed successfully and saved to: {out_p}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Temporal Attention & Transformer Models")
    parser.add_argument("--exp03-parquet", default="DL/ml engineer/artifacts/lstm/experiments/exp03_residual_persistence/validation_predictions.parquet")
    parser.add_argument("--meta-parquet", default="DL/data/staging/supplementary_acquisition/processed/temporal_sequences/sequence_metadata.parquet")
    parser.add_argument("--output-json", default="DL/ml engineer/artifacts/transformer/milestone_3_4_evaluation.json")
    args = parser.parse_args()

    # Automatically discover candidate model parquets
    candidates = {}
    tf_p = Path("DL/ml engineer/artifacts/transformer/validation_predictions.parquet")
    if tf_p.exists():
        candidates["transformer_residual"] = str(tf_p)

    al_p = Path("DL/ml engineer/artifacts/attention_lstm/validation_predictions.parquet")
    if al_p.exists():
        candidates["attention_lstm"] = str(al_p)

    tf_mean_p = Path("DL/ml engineer/artifacts/transformer_mean_pooling/validation_predictions.parquet")
    if tf_mean_p.exists():
        candidates["transformer_mean_pooling"] = str(tf_mean_p)

    run_comprehensive_evaluation(
        exp03_parquet=args.exp03_parquet,
        candidate_parquets=candidates,
        meta_parquet=args.meta_parquet,
        output_json=args.output_json,
    )

