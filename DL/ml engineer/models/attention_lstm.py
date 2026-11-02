"""Temporal Attention Residual LSTM Forecaster for Milestone 3.4.

Implements an attention-enhanced recurrent forecasting architecture:
1. 2-layer LSTM temporal encoder generating hidden states for all 30 historical time steps [B, 30, 64].
2. Additive temporal attention mechanism computing dynamic importance weights alpha_t over the 30 days.
3. Weighted temporal context vector c = sum(alpha_t * h_t).
4. Residual regression head mapping context vector to residual delta.
5. Inductive residual formulation: y_hat = ReLU( y_naive + delta ).
"""

from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAdditiveAttention(nn.Module):
    """Additive (Bahdanau-style) temporal attention mechanism over sequence hidden states."""

    def __init__(self, hidden_size: int = 64, attention_dim: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim
        self.W_a = nn.Linear(hidden_size, attention_dim, bias=True)
        self.v_a = nn.Linear(attention_dim, 1, bias=False)

        nn.init.xavier_uniform_(self.W_a.weight)
        nn.init.xavier_uniform_(self.v_a.weight)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention weights and context vector over temporal hidden states.

        Args:
            H: Tensor of shape [batch_size, seq_len=30, hidden_size=64].

        Returns:
            context: Weighted representation tensor [batch_size, hidden_size=64].
            alpha: Attention probability distribution [batch_size, seq_len=30].
        """
        # score = v_a^T tanh(W_a h_t)
        u = torch.tanh(self.W_a(H))  # [B, T, attention_dim]
        scores = self.v_a(u).squeeze(-1)  # [B, T]

        # Normalized attention coefficients across the 30 temporal positions
        alpha = F.softmax(scores, dim=-1)  # [B, T]

        # Context vector: c = sum_t (alpha_t * h_t)
        # [B, 1, T] @ [B, T, hidden_size] -> [B, 1, hidden_size] -> [B, hidden_size]
        context = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)

        return context, alpha


class TemporalAttentionLSTMForecaster(nn.Module):
    """Residual LSTM augmented with explicit temporal self-attention over historical timesteps."""

    def __init__(
        self,
        input_size: int = 20,
        hidden_size: int = 64,
        attention_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        roll7_idx: int = 12,
        roll7_mean: float = 2.4068180554238894,
        roll7_std: float = 6.3483944613159915,
        constraint: str = "relu",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim
        self.num_layers = num_layers
        self.dropout = dropout if num_layers > 1 else 0.0
        self.roll7_idx = roll7_idx
        self.constraint = constraint.lower()

        # Register unscaling constants
        self.register_buffer("roll7_mean", torch.tensor(roll7_mean, dtype=torch.float32))
        self.register_buffer("roll7_std", torch.tensor(roll7_std, dtype=torch.float32))

        # 1. Temporal Recurrent Encoder
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=False,
        )

        # 2. Temporal Attention Layer over all 30 time steps
        self.attention = TemporalAdditiveAttention(
            hidden_size=self.hidden_size,
            attention_dim=self.attention_dim,
        )

        # 3. Residual Regression Head
        self.head = nn.Linear(self.hidden_size, 1)

        # Conservative initialization: delta starts near zero
        nn.init.xavier_uniform_(self.head.weight, gain=0.01)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass computing persistence forecast plus temporal attention residual.

        Args:
            x: Input tensor [batch_size, seq_len=30, input_size=20].
            return_attention: If True, returns (y_pred, alpha).

        Returns:
            If return_attention=False: predictions tensor [batch_size, 1].
            If return_attention=True: tuple of (predictions, attention_weights).
        """
        # Step 1: Analytical Persistence Forecast from latest available observation (timestep 29, t-2)
        unscaled_roll7 = x[:, -1, self.roll7_idx:self.roll7_idx + 1] * self.roll7_std + self.roll7_mean
        y_naive = 7.0 * unscaled_roll7  # [B, 1]

        # Step 2: Temporal Recurrent Encoding (all 30 timesteps)
        lstm_out, _ = self.lstm(x)  # [B, 30, hidden_size]

        # Step 3: Temporal Attention Weighting
        context, alpha = self.attention(lstm_out)  # context: [B, hidden_size], alpha: [B, 30]

        # Step 4: Learned Residual Correction
        delta = self.head(context)  # [B, 1]

        # Step 5: Additive Residual with physical non-negativity
        if self.constraint == "relu":
            y_pred = F.relu(y_naive + delta)
        else:
            y_pred = y_naive + delta

        if return_attention:
            return y_pred, alpha
        return y_pred

    def get_config(self) -> Dict[str, Any]:
        """Return model architecture configuration dictionary."""
        return {
            "model_type": "TemporalAttentionLSTMForecaster",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "attention_dim": self.attention_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "roll7_idx": self.roll7_idx,
            "roll7_mean": float(self.roll7_mean.item()),
            "roll7_std": float(self.roll7_std.item()),
            "constraint": self.constraint,
        }

