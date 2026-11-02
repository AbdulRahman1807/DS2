"""Phase B: Controlled Transformer-GNN Feasibility Benchmark Runner.

Executes the 3 controlled models on identical Caribbean validation splits:
1. Model B: Transformer + Actual 2019 AIS Graph
2. Model C: Transformer + Shuffled Graph (random rewiring negative control)
3. Model D: Transformer + No Edges (isolated adapter negative control)
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from training.train_transformer_gnn import run_transformer_gnn_training

def main():
    cfg_p = str(WORKSPACE_DIR / "configs/transformer_gnn.yaml")
    results = {}

    experiments = [
        ("actual", "none", WORKSPACE_DIR / "artifacts/gnn/caribbean/trans_gatv2_actual"),
        ("shuffled", "shuffled", WORKSPACE_DIR / "artifacts/gnn/caribbean/trans_gatv2_shuffled"),
        ("no_edges", "no_edges", WORKSPACE_DIR / "artifacts/gnn/caribbean/trans_gatv2_no_edges"),
    ]

    for label, abl, out_dir in experiments:
        print(f"\n{'='*40} RUNNING ABLATION: {label.upper()} {'='*40}")
        t0 = time.time()
        res = run_transformer_gnn_training(
            config_path=cfg_p,
            ablation=abl,
            gnn_type="gatv2",
            artifacts_dir=str(out_dir),
        )
        res["duration_sec"] = time.time() - t0
        results[label] = res

    # Also load Model A (Regional Frozen Transformer Reference)
    val_trans = pd.read_parquet(str(WORKSPACE_DIR / "artifacts/transformer_mean_pooling/validation_predictions.parquet"))
    # Caribbean ports
    from data.caribbean_dataset import load_caribbean_topology
    topo = load_caribbean_topology(str(WORKSPACE_DIR / "../data/staging/supplementary_acquisition/processed/graph_ready/caribbean_graph_edges.parquet"))
    carib_ports = topo["ports"]
    df_a = val_trans[val_trans["portwatch_id"].isin(carib_ports)].copy()
    
    y_t = df_a["y_true"].values
    y_p = df_a["y_pred"].values
    d = y_p - y_t
    ae = np.abs(d)
    se = d ** 2
    mae_a = float(np.mean(ae))
    rmse_a = float(np.sqrt(np.mean(se)))
    huber_a = float(np.mean(np.where(ae <= 1.0, 0.5 * se, ae - 0.5)))
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    r2_a = float(1.0 - (np.sum(se) / ss_tot)) if ss_tot > 0 else 0.0
    bias_a = float(np.mean(d))

    summary_rows = [
        {
            "Model": "Model A — Regional Transformer (Frozen M3.4)",
            "Ablation": "No Graph (Reference)",
            "Huber": round(huber_a, 4),
            "MAE": round(mae_a, 4),
            "RMSE": round(rmse_a, 4),
            "R2": round(r2_a, 4),
            "Bias": round(bias_a, 4),
        },
        {
            "Model": "Model B — Transformer + Actual Graph",
            "Ablation": "Actual 2019 AIS",
            "Huber": round(results["actual"]["best_val_huber"], 4),
            "MAE": round(results["actual"]["val_mae"], 4),
            "RMSE": round(results["actual"]["val_rmse"], 4),
            "R2": round(results["actual"]["val_r2"], 4),
            "Bias": round(results["actual"]["val_bias"], 4),
        },
        {
            "Model": "Model C — Transformer + Shuffled Graph",
            "Ablation": "Random Rewiring",
            "Huber": round(results["shuffled"]["best_val_huber"], 4),
            "MAE": round(results["shuffled"]["val_mae"], 4),
            "RMSE": round(results["shuffled"]["val_rmse"], 4),
            "R2": round(results["shuffled"]["val_r2"], 4),
            "Bias": round(results["shuffled"]["val_bias"], 4),
        },
        {
            "Model": "Model D — Transformer + No Edges",
            "Ablation": "Zero Edges (Adapter)",
            "Huber": round(results["no_edges"]["best_val_huber"], 4),
            "MAE": round(results["no_edges"]["val_mae"], 4),
            "RMSE": round(results["no_edges"]["val_rmse"], 4),
            "R2": round(results["no_edges"]["val_r2"], 4),
            "Bias": round(results["no_edges"]["val_bias"], 4),
        },
    ]

    sum_df = pd.DataFrame(summary_rows)
    print("\n" + "="*80)
    print("PHASE B: CARIBBEAN BENCHMARK CONTROLLED RESULTS (N = 18,360)")
    print("="*80)
    print(sum_df.to_string(index=False))

    sum_df.to_csv(str(WORKSPACE_DIR / "artifacts/gnn/caribbean/phase_b_results.csv"), index=False)

if __name__ == "__main__":
    main()
