"""
Residual Dwell and Waiting-Time Forecaster
==========================================
Specialized neural sequence model predicting Weekly Vessel Anchorage Waiting Time
(and berth service duration) from historical port-call sequences.

Formulation:
    y_hat_{t+H} = ReLU( y_t + Delta_theta( X_{t-L+1:t}, port_id ) )

This residual formulation guarantees non-negativity and grounds predictions in
the local persistence benchmark while learning non-linear multi-port dynamics.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any


class ResidualDwellGRUForecaster(nn.Module):
    """
    Lightweight GRU Forecaster with static port entity embeddings and
    residual persistence connection for weekly vessel waiting time forecasting.
    """

    def __init__(
        self,
        num_ports: int = 19,
        port_embedding_dim: int = 8,
        num_temporal_features: int = 7,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        predict_service_time: bool = False,
    ):
        super().__init__()
        self.num_ports = num_ports
        self.port_embedding_dim = port_embedding_dim
        self.num_temporal_features = num_temporal_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.predict_service_time = predict_service_time

        # Port entity embedding
        self.port_embedding = nn.Embedding(num_ports, port_embedding_dim)

        # Temporal sequence encoder (GRU)
        self.gru = nn.GRU(
            input_size=num_temporal_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Combined representation dimension: GRU hidden state + Port embedding
        fusion_dim = hidden_size + port_embedding_dim

        # Primary Residual Prediction Head: 1-week ahead waiting time residual Delta
        self.waiting_head_h1 = nn.Sequential(
            nn.Linear(fusion_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        # Multi-horizon Head: 4-week mean waiting time residual Delta
        self.waiting_head_h4 = nn.Sequential(
            nn.Linear(fusion_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        # Optional Service Time Prediction Head
        if predict_service_time:
            self.service_head = nn.Sequential(
                nn.Linear(fusion_dim, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )
        else:
            self.service_head = None

    def forward(
        self,
        x_seq: torch.Tensor,
        port_ids: torch.Tensor,
        naive_waiting: torch.Tensor,
        naive_service: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x_seq: (batch_size, seq_len, num_temporal_features)
            port_ids: (batch_size,) long tensor of port indices (0..num_ports-1)
            naive_waiting: (batch_size,) float tensor of unscaled waiting time at time t
            naive_service: (batch_size,) float tensor of unscaled service time at time t

        Returns:
            Dict containing:
                - 'waiting_pred_h1': 1-week ahead waiting time prediction (hours)
                - 'waiting_pred_h4': 4-week mean waiting time prediction (hours)
                - 'service_pred_h1': 1-week ahead service time prediction (hours) if enabled
        """
        batch_size = x_seq.size(0)

        # GRU encoding
        gru_out, h_n = self.gru(x_seq)
        # h_n shape: (num_layers, batch_size, hidden_size) -> take last layer
        h_last = h_n[-1]  # (batch_size, hidden_size)

        # Port entity embedding
        port_emb = self.port_embedding(port_ids)  # (batch_size, port_embedding_dim)

        # Fuse temporal state with static port identity
        fused = torch.cat([h_last, port_emb], dim=-1)  # (batch_size, fusion_dim)

        # Compute residuals
        delta_waiting_h1 = self.waiting_head_h1(fused).squeeze(-1)  # (batch_size,)
        delta_waiting_h4 = self.waiting_head_h4(fused).squeeze(-1)  # (batch_size,)

        # Non-negative residual prediction
        waiting_pred_h1 = torch.relu(naive_waiting + delta_waiting_h1)
        waiting_pred_h4 = torch.relu(naive_waiting + delta_waiting_h4)

        results = {
            "waiting_pred_h1": waiting_pred_h1,
            "waiting_pred_h4": waiting_pred_h4,
            "delta_waiting_h1": delta_waiting_h1,
            "delta_waiting_h4": delta_waiting_h4,
        }

        if self.service_head is not None and naive_service is not None:
            delta_service = self.service_head(fused).squeeze(-1)
            service_pred_h1 = torch.relu(naive_service + delta_service)
            results["service_pred_h1"] = service_pred_h1

        return results

