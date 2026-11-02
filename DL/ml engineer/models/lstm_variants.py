"""Experimental LSTM Architecture Variants for Milestone 3.2 Diagnostics.

Implements two specialized experimental architectures to test core diagnostic hypotheses:
1. PortEmbeddedLSTMForecaster: Evaluates whether adding explicit static port entity
   embeddings resolves cross-sectional dynamic range compression.
2. ResidualLSTMForecaster: Evaluates whether predicting a residual correction delta over
   the strong naive persistence baseline (y_hat = ReLU(y_naive + delta)) enables effective learning.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PortEmbeddedLSTMForecaster(nn.Module):
    """LSTM forecaster augmented with learned static port entity embeddings.
    
    Attributes:
        num_ports: Total number of unique ports (default 2065).
        embedding_dim: Dimension of port identity embedding vector (default 16).
        input_size: Number of input features per timestep (default 20).
        hidden_size: Hidden dimension of LSTM recurrent units (default 64).
        num_layers: Number of stacked LSTM recurrent layers (default 2).
        dropout: Inter-layer dropout probability (default 0.2).
        output_size: Dimension of output prediction (default 1).
        output_activation: Output activation ('softplus', 'none', 'relu').
    """

    def __init__(
        self,
        num_ports: int = 2065,
        embedding_dim: int = 16,
        input_size: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
        output_activation: str = "softplus",
    ):
        super().__init__()
        self.num_ports = num_ports
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout if num_layers > 1 else 0.0
        self.output_size = output_size
        self.output_activation = output_activation.lower()

        # Port entity embedding table: 0..num_ports-1 for valid ports, index num_ports for unknown/unseen
        self.port_embedding = nn.Embedding(
            num_embeddings=self.num_ports + 1,
            embedding_dim=self.embedding_dim,
            padding_idx=self.num_ports,
        )
        nn.init.normal_(self.port_embedding.weight, mean=0.0, std=0.05)

        # Recurrent temporal encoder
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=False,
        )

        # Regression head combining temporal hidden state and static port embedding
        combined_dim = self.hidden_size + self.embedding_dim
        self.head = nn.Linear(combined_dim, self.output_size)

        nn.init.xavier_uniform_(self.head.weight)
        if self.output_activation == "softplus":
            nn.init.constant_(self.head.bias, 2.8)
        else:
            nn.init.constant_(self.head.bias, 0.0)

    def forward(self, x: torch.Tensor, port_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through LSTM temporal encoder and port-embedded head.
        
        Args:
            x: Input tensor [batch_size, seq_len=30, input_size=20].
            port_idx: Optional port index tensor [batch_size]. If None, uses unknown embedding.
            
        Returns:
            Predictions tensor [batch_size, 1].
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # [B, hidden_size]

        if port_idx is not None:
            safe_port_idx = torch.clamp(port_idx.view(-1), 0, self.num_ports)
            port_emb = self.port_embedding(safe_port_idx)  # [B, embedding_dim]
        else:
            unknown_idx = torch.full((x.size(0),), self.num_ports, dtype=torch.long, device=x.device)
            port_emb = self.port_embedding(unknown_idx)

        combined = torch.cat([last_hidden, port_emb], dim=-1)  # [B, hidden_size + embedding_dim]
        raw_pred = self.head(combined)

        if self.output_activation == "softplus":
            return F.softplus(raw_pred)
        elif self.output_activation == "relu":
            return F.relu(raw_pred)
        elif self.output_activation == "none":
            return raw_pred
        else:
            raise ValueError(f"Unsupported output_activation '{self.output_activation}'.")

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_type": "PortEmbeddedLSTMForecaster",
            "num_ports": self.num_ports,
            "embedding_dim": self.embedding_dim,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "output_size": self.output_size,
            "output_activation": self.output_activation,
        }


class ResidualLSTMForecaster(nn.Module):
    """LSTM model formulating prediction as a residual correction over naive persistence.
    
    Formula:
        y_hat = ReLU( y_naive + delta )
    where:
        y_naive = 7.0 * (x[:, 29, roll7_idx] * sigma_roll7 + mu_roll7)
        delta = head(LSTM(x))
        
    Attributes:
        input_size: Number of input features per timestep (default 20).
        hidden_size: Hidden dimension of LSTM recurrent units (default 64).
        num_layers: Number of stacked LSTM recurrent layers (default 2).
        dropout: Inter-layer dropout probability (default 0.2).
        roll7_idx: Index of pw_calls_rolling_7d_mean feature (default 12).
        roll7_mean: Training set mean for standard unscaling (default 2.406818).
        roll7_std: Training set std for standard unscaling (default 6.348394).
        constraint: Physical non-negativity constraint ('relu' or 'none').
    """

    def __init__(
        self,
        input_size: int = 20,
        hidden_size: int = 64,
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
        self.num_layers = num_layers
        self.dropout = dropout if num_layers > 1 else 0.0
        self.roll7_idx = roll7_idx
        self.constraint = constraint.lower()

        # Register unscaling constants as non-trainable buffers
        self.register_buffer("roll7_mean", torch.tensor(roll7_mean, dtype=torch.float32))
        self.register_buffer("roll7_std", torch.tensor(roll7_std, dtype=torch.float32))

        # Temporal recurrent encoder
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=False,
        )

        # Residual projection head
        self.head = nn.Linear(self.hidden_size, 1)

        # Initialize residual head with small weights and zero bias so delta starts near 0.0
        nn.init.xavier_uniform_(self.head.weight, gain=0.01)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, x: torch.Tensor, port_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass computing persistence forecast plus learned residual.
        
        Args:
            x: Input tensor [batch_size, seq_len=30, input_size=20].
            port_idx: Unused, kept for unified trainer interface.
            
        Returns:
            Non-negative predictions tensor [batch_size, 1].
        """
        # Step 1: Extract latest available 7-day rolling mean at timestep 29 (t-2)
        # Unscale back to raw calls/day using training set scaler parameters
        unscaled_roll7 = x[:, -1, self.roll7_idx:self.roll7_idx + 1] * self.roll7_std + self.roll7_mean
        
        # Naive persistence 7-day cumulative prediction
        y_naive = 7.0 * unscaled_roll7

        # Step 2: Temporal recurrent representation
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # [B, hidden_size]

        # Step 3: Residual correction delta
        delta = self.head(last_hidden)  # [B, 1]

        # Step 4: Combined forecast
        pred = y_naive + delta

        # Step 5: Enforce physical non-negativity
        if self.constraint == "relu":
            return F.relu(pred)
        elif self.constraint == "none":
            return pred
        else:
            raise ValueError(f"Unsupported constraint '{self.constraint}'.")

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_type": "ResidualLSTMForecaster",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "roll7_idx": self.roll7_idx,
            "roll7_mean": float(self.roll7_mean.item()),
            "roll7_std": float(self.roll7_std.item()),
            "constraint": self.constraint,
        }

