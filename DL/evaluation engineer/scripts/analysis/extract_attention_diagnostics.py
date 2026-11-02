"""Temporal Attention Extraction and Interpretability Diagnostics (Milestone 3.4).

Quantifies and inspects the learned temporal attention distributions for:
1. ResidualTransformerForecaster (Multi-Head Self-Attention)
2. TemporalAttentionLSTMForecaster (Additive Temporal Attention)

Answers the 5 core diagnostic questions from Section 11:
1. Recency concentration: Is attention concentrated on recent observations (t-2 to t-8)?
2. Long-range retention: Does the model assign meaningful weight to older observations (t-25 to t-31)?
3. Cross-port heterogeneity: Do attention patterns vary across ports?
4. Volume-stratified dynamics: Do high-volume hubs and low-volume berths exhibit different temporal focus?
5. Temporal stability & entropy: How dispersed vs focused is the learned attention distribution?
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch

# Add workspace to path
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parents[2] / "ml engineer"
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from models.transformer_forecaster import ResidualTransformerForecaster
from models.attention_lstm import TemporalAttentionLSTMForecaster
from data.data_module import create_dataloaders


def compute_entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Shannon entropy of probability distribution."""
    p_safe = np.clip(probs, eps, 1.0)
    return float(-np.sum(p_safe * np.log(p_safe)))


def extract_model_attention(
    checkpoint_path: str,
    config_path: str,
    output_json: str,
    num_eval_batches: int = 100,
    device_name: str = "mps",
) -> Dict[str, Any]:
    device = torch.device(device_name if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")
    print(f"Loading checkpoint {checkpoint_path} onto {device}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    m_cfg = ckpt.get("model_config", {})
    model_type = m_cfg.get("model_type", "ResidualTransformerForecaster")

    if model_type == "ResidualTransformerForecaster":
        model = ResidualTransformerForecaster(
            input_size=m_cfg.get("input_size", 20),
            seq_len=m_cfg.get("seq_len", 30),
            d_model=m_cfg.get("d_model", 64),
            nhead=m_cfg.get("nhead", 4),
            num_layers=m_cfg.get("num_layers", 2),
            dim_feedforward=m_cfg.get("dim_feedforward", 128),
            dropout=m_cfg.get("dropout", 0.1),
            pooling=m_cfg.get("pooling", "last"),
            pe_type=m_cfg.get("pe_type", "sinusoidal"),
            roll7_idx=m_cfg.get("roll7_idx", 12),
            roll7_mean=m_cfg.get("roll7_mean", 2.4068180554238894),
            roll7_std=m_cfg.get("roll7_std", 6.3483944613159915),
            constraint=m_cfg.get("constraint", "relu"),
        ).to(device)
    elif model_type == "TemporalAttentionLSTMForecaster":
        model = TemporalAttentionLSTMForecaster(
            input_size=m_cfg.get("input_size", 20),
            hidden_size=m_cfg.get("hidden_size", 64),
            attention_dim=m_cfg.get("attention_dim", 32),
            num_layers=m_cfg.get("num_layers", 2),
            dropout=m_cfg.get("dropout", 0.2),
            roll7_idx=m_cfg.get("roll7_idx", 12),
            roll7_mean=m_cfg.get("roll7_mean", 2.4068180554238894),
            roll7_std=m_cfg.get("roll7_std", 6.3483944613159915),
            constraint=m_cfg.get("constraint", "relu"),
        ).to(device)
    else:
        raise ValueError(f"Unsupported model type '{model_type}'.")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loaders = create_dataloaders(
        config_path=config_path,
        batch_size=512,
        val_batch_size=512,
        num_workers=0,
        return_port_idx=False,
    )
    val_loader = loaders["val"]
    dataset = val_loader.dataset
    df_meta = dataset.df_meta

    all_attns = []
    all_targets = []
    all_ports = []
    all_head_attns = []  # for transformer: [N, nheads=4, 30]

    print(f"Sampling {num_eval_batches} validation batches ({num_eval_batches * 512:,} sequences)...")
    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            if b_idx >= num_eval_batches:
                break
            bx, by = batch
            bx = bx.to(device)
            b_len = bx.size(0)

            # Metadata slice for these sequences
            sub_meta = df_meta.iloc[b_idx * 512 : b_idx * 512 + b_len]
            all_ports.extend(sub_meta["portwatch_id"].tolist())
            all_targets.extend(by.view(-1).numpy().tolist())

            if model_type == "ResidualTransformerForecaster":
                _, attn_layers = model(bx, return_attention=True)
                # Last layer attention: [B, 4, 30, 30]
                # Look at query 29 (the latest available day t-2) attending to keys 0..29
                last_layer_attn = attn_layers[-1][:, :, -1, :]  # [B, 4, 30]
                all_head_attns.append(last_layer_attn.cpu().numpy())
                # Mean over heads for overall temporal profile
                mean_over_heads = last_layer_attn.mean(dim=1).cpu().numpy()  # [B, 30]
                all_attns.append(mean_over_heads)
            elif model_type == "TemporalAttentionLSTMForecaster":
                _, alpha = model(bx, return_attention=True)  # [B, 30]
                all_attns.append(alpha.cpu().numpy())

    attns_np = np.concatenate(all_attns, axis=0)  # [N_sampled, 30]
    targets_np = np.array(all_targets)  # [N_sampled]
    ports_np = np.array(all_ports)      # [N_sampled]

    # Overall 30-step profile
    # Timestep 0 = t-31 (oldest), Timestep 29 = t-2 (most recent)
    timesteps_labels = [f"t-{31 - i}" for i in range(30)]
    mean_profile = attns_np.mean(axis=0)
    std_profile = attns_np.std(axis=0)

    # 1. Recency vs Long-Range Weighting
    recent_7d_weight = float(np.sum(mean_profile[-7:]) * 100.0)    # t-2 through t-8
    mid_16d_weight = float(np.sum(mean_profile[7:-7]) * 100.0)      # t-9 through t-24
    oldest_7d_weight = float(np.sum(mean_profile[:7]) * 100.0)     # t-25 through t-31
    peak_idx = int(np.argmax(mean_profile))

    # 2. Entropy
    sample_entropies = [compute_entropy(attns_np[i]) for i in range(len(attns_np))]
    mean_entropy = float(np.mean(sample_entropies))
    max_possible_entropy = float(np.log(30.0))  # ~3.4012 (uniform distribution)
    dispersion_ratio = mean_entropy / max_possible_entropy

    # 3. Volume Stratification
    q25, q50, q75 = np.percentile(targets_np, [25, 50, 75])
    q1_mask = targets_np <= q25
    q4_mask = targets_np >= q75

    q1_profile = attns_np[q1_mask].mean(axis=0) if np.any(q1_mask) else mean_profile
    q4_profile = attns_np[q4_mask].mean(axis=0) if np.any(q4_mask) else mean_profile

    # 4. Cross-Port Heterogeneity
    df_sample = pd.DataFrame({"portwatch_id": ports_np})
    for t_i in range(30):
        df_sample[f"attn_{t_i}"] = attns_np[:, t_i]
    port_agg = df_sample.groupby("portwatch_id").mean()
    port_recent_weights = port_agg.iloc[:, -7:].sum(axis=1)

    port_heterogeneity = {
        "num_ports_sampled": len(port_agg),
        "mean_port_recent_weight_pct": float(port_recent_weights.mean() * 100.0),
        "std_port_recent_weight_pct": float(port_recent_weights.std() * 100.0),
        "min_port_recent_weight_pct": float(port_recent_weights.min() * 100.0),
        "max_port_recent_weight_pct": float(port_recent_weights.max() * 100.0),
    }

    # 5. Multi-head breakdown if Transformer
    head_profiles = {}
    if len(all_head_attns) > 0:
        head_attns_np = np.concatenate(all_head_attns, axis=0)  # [N, 4, 30]
        for h_i in range(4):
            head_m = head_attns_np[:, h_i, :].mean(axis=0)
            head_profiles[f"head_{h_i+1}"] = {
                "profile": head_m.tolist(),
                "recent_7d_pct": float(np.sum(head_m[-7:]) * 100.0),
                "oldest_7d_pct": float(np.sum(head_m[:7]) * 100.0),
                "peak_timestep": timesteps_labels[int(np.argmax(head_m))],
            }

    summary = {
        "model_type": model_type,
        "num_sequences_evaluated": len(attns_np),
        "timesteps_labels": timesteps_labels,
        "mean_attention_profile": mean_profile.tolist(),
        "std_attention_profile": std_profile.tolist(),
        "recency_concentration": {
            "recent_7d_weight_pct": recent_7d_weight,
            "mid_16d_weight_pct": mid_16d_weight,
            "oldest_7d_weight_pct": oldest_7d_weight,
            "peak_timestep": timesteps_labels[peak_idx],
            "peak_weight": float(mean_profile[peak_idx]),
        },
        "dispersion_and_entropy": {
            "mean_entropy": mean_entropy,
            "max_possible_entropy": max_possible_entropy,
            "dispersion_ratio": dispersion_ratio,
            "interpretation": "Focused/Spiked" if dispersion_ratio < 0.85 else "Broad/Distributed",
        },
        "volume_stratification": {
            "q1_low_volume_profile": q1_profile.tolist(),
            "q1_recent_7d_pct": float(np.sum(q1_profile[-7:]) * 100.0),
            "q4_high_volume_profile": q4_profile.tolist(),
            "q4_recent_7d_pct": float(np.sum(q4_profile[-7:]) * 100.0),
        },
        "cross_port_heterogeneity": port_heterogeneity,
        "head_profiles": head_profiles,
    }

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAttention diagnostics successfully extracted and saved to: {out_p}")
    print(f"  Recent 7d Weight:  {recent_7d_weight:.2f}% | Oldest 7d Weight: {oldest_7d_weight:.2f}%")
    print(f"  Peak Timestep:     {timesteps_labels[peak_idx]} ({mean_profile[peak_idx]:.4f})")
    print(f"  Mean Entropy:      {mean_entropy:.3f} / {max_possible_entropy:.3f} ({dispersion_ratio*100:.1f}% uniform)")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Attention Diagnostics")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--config", default="DL/ml engineer/configs/sequence_pipeline.yaml")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--batches", type=int, default=100)
    args = parser.parse_args()

    extract_model_attention(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_json=args.output,
        num_eval_batches=args.batches,
    )

