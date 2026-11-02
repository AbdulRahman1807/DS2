"""LSTM Forecasting Baseline Model for Milestone 3.2.

Implements a clean, configurable recurrent neural network encoder with a regression head
predicting cumulative 7-day future port calls from 30 days of 20-feature input sequences.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMForecaster(nn.Module):
    """LSTM encoder with linear regression head for temporal bottleneck forecasting.
    
    Attributes:
        input_size: Number of input features per timestep (default 20).
        hidden_size: Hidden dimension of LSTM recurrent units (default 64).
        num_layers: Number of stacked LSTM recurrent layers (default 2).
        dropout: Inter-layer dropout probability (default 0.2).
        output_size: Dimension of output prediction (default 1).
        output_activation: Output activation ('softplus', 'none', 'relu').
    """

    def __init__(
        self,
        input_size: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
        output_activation: str = "softplus",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout if num_layers > 1 else 0.0
        self.output_size = output_size
        self.output_activation = output_activation.lower()

        # Recurrent Encoder
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=False,
        )

        # Regression Head mapping final hidden representation to scalar target
        self.head = nn.Linear(self.hidden_size, self.output_size)

        # Initialize head weights conservatively
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            # Initialize bias near empirical target mean log (~2.8) if softplus, or 0.0
            if self.output_activation == "softplus":
                nn.init.constant_(self.head.bias, 2.8)
            else:
                nn.init.constant_(self.head.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM encoder and regression head.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len=30, input_size=20].
            
        Returns:
            Predictions tensor of shape [batch_size, output_size=1].
        """
        # x shape: [B, T, D]
        lstm_out, _ = self.lstm(x)  # lstm_out: [B, T, hidden_size]
        
        # Take final timestep's hidden state as sequence summary representation
        last_hidden = lstm_out[:, -1, :]  # [B, hidden_size]
        
        # Linear projection
        raw_pred = self.head(last_hidden)  # [B, output_size]

        # Output activation
        if self.output_activation == "softplus":
            # Smooth non-negative mapping: softplus(z) = ln(1 + e^z)
            return F.softplus(raw_pred)
        elif self.output_activation == "relu":
            return F.relu(raw_pred)
        elif self.output_activation == "none":
            return raw_pred
        else:
            raise ValueError(f"Unsupported output_activation '{self.output_activation}'.")

    def get_config(self) -> Dict[str, Any]:
        """Return model architecture configuration dictionary."""
        return {
            "model_type": "LSTMForecaster",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "output_size": self.output_size,
            "output_activation": self.output_activation,
        }

