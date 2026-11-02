"""Lightweight Transformer Temporal Forecaster for Milestone 3.4.

Implements a compact, availability-aware Transformer encoder with a residual prediction head
for 7-day cumulative port call forecasting from 30-day 20-feature input sequences.

Architecture:
1. Input Projection: Linear(20, 64)
2. Positional Encoding: Sinusoidal (default) or Learned Positional Embedding
3. Transformer Encoder: 2 layers of Pre-LN Multi-Head Self-Attention (4 heads) + Feed-Forward (128)
4. Temporal Pooling: Last-token (default) or Mean-pooling
5. Residual Head: Linear(64, 1) mapping pooled representation to residual delta
6. Persistence Sum: y_hat = ReLU( y_naive + delta )
"""

import math
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 60):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T, d_model]
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderLayerWithAttn(nn.Module):
    """Transformer Encoder Layer with Pre-LayerNorm and explicit attention weight extraction."""

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-LN Self-Attention
        src2 = self.norm1(src)
        attn_out, attn_weights = self.self_attn(
            src2, src2, src2,
            need_weights=need_weights,
            average_attn_weights=False if need_weights else True,
        )
        src = src + self.dropout1(attn_out)

        # Pre-LN Feed-Forward
        src2 = self.norm2(src)
        ff_out = self.linear2(self.dropout(F.gelu(self.linear1(src2))))
        src = src + self.dropout2(ff_out)

        return src, attn_weights


class ResidualTransformerForecaster(nn.Module):
    """Transformer Forecaster formulating prediction as an inductive residual over naive persistence.

    Mathematical formulation:
        y_hat = ReLU( y_naive + delta )
    where:
        y_naive = 7.0 * (x[:, 29, roll7_idx] * sigma_roll7 + mu_roll7)
        delta = head(Transformer(x))
    """

    def __init__(
        self,
        input_size: int = 20,
        seq_len: int = 30,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        pooling: str = "last",
        pe_type: str = "sinusoidal",
        roll7_idx: int = 12,
        roll7_mean: float = 2.4068180554238894,
        roll7_std: float = 6.3483944613159915,
        constraint: str = "relu",
    ):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.pooling = pooling.lower()
        self.pe_type = pe_type.lower()
        self.roll7_idx = roll7_idx
        self.constraint = constraint.lower()

        # Register unscaling constants
        self.register_buffer("roll7_mean", torch.tensor(roll7_mean, dtype=torch.float32))
        self.register_buffer("roll7_std", torch.tensor(roll7_std, dtype=torch.float32))

        # 1. Input Linear Projection
        self.input_proj = nn.Linear(input_size, d_model)

        # 2. Positional Encoding
        if self.pe_type == "learned":
            self.pos_emb = nn.Embedding(seq_len, d_model)
            nn.init.normal_(self.pos_emb.weight, std=0.02)
        else:
            self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=seq_len + 10)

        self.input_dropout = nn.Dropout(dropout)

        # 3. Stacked Transformer Encoder Layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithAttn(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # 4. Residual Regression Head
        self.head = nn.Linear(d_model, 1)

        # Conservative initialization: delta starts near zero
        nn.init.xavier_uniform_(self.head.weight, gain=0.01)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Forward pass computing persistence forecast plus Transformer residual.

        Args:
            x: Input tensor [batch_size, seq_len=30, input_size=20].
            return_attention: If True, also returns attention weight tensors from each layer.

        Returns:
            If return_attention=False: predictions tensor [batch_size, 1].
            If return_attention=True: tuple of (predictions, list of attention weights).
        """
        B, T, D = x.shape

        # Step 1: Analytical Persistence Forecast from latest available observation (timestep 29, t-2)
        unscaled_roll7 = x[:, -1, self.roll7_idx:self.roll7_idx + 1] * self.roll7_std + self.roll7_mean
        y_naive = 7.0 * unscaled_roll7  # [B, 1]

        # Step 2: Input Projection & Positional Encoding
        h = self.input_proj(x)  # [B, T, d_model]
        if self.pe_type == "learned":
            pos_indices = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
            h = h + self.pos_emb(pos_indices)
        else:
            h = self.pos_encoder(h)
        h = self.input_dropout(h)

        # Step 3: Transformer Encoder Layers
        attn_weights_list = []
        for layer in self.layers:
            h, attn_w = layer(h, need_weights=return_attention)
            if return_attention and attn_w is not None:
                attn_weights_list.append(attn_w)

        h = self.final_norm(h)  # [B, T, d_model]

        # Step 4: Temporal Pooling
        if self.pooling == "mean":
            pooled = h.mean(dim=1)  # [B, d_model]
        elif self.pooling == "last":
            pooled = h[:, -1, :]    # [B, d_model]
        else:
            raise ValueError(f"Unsupported pooling mode '{self.pooling}'. Choose 'last' or 'mean'.")

        # Step 5: Learned Residual Correction
        delta = self.head(pooled)  # [B, 1]

        # Step 6: Additive Residual with physical non-negativity
        if self.constraint == "relu":
            y_pred = F.relu(y_naive + delta)
        else:
            y_pred = y_naive + delta

        if return_attention:
            return y_pred, attn_weights_list
        return y_pred

    def get_config(self) -> Dict[str, Any]:
        """Return model architecture configuration dictionary."""
        return {
            "model_type": "ResidualTransformerForecaster",
            "input_size": self.input_size,
            "seq_len": self.seq_len,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout_rate,
            "pooling": self.pooling,
            "pe_type": self.pe_type,
            "roll7_idx": self.roll7_idx,
            "roll7_mean": float(self.roll7_mean.item()),
            "roll7_std": float(self.roll7_std.item()),
            "constraint": self.constraint,
        }

