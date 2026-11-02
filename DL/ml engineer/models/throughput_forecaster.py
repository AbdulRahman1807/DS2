"""
Direct Throughput Residual Forecaster
=====================================
Milestone 3.6 Phase C: Direct Throughput Forecasting

Predicts direct port logistics throughput measures:
1. Cumulative 7-day container-carrying vessel calls (target_container_calls_7d_future_sum)
2. Cumulative 7-day metric cargo tonnage (target_cargo_tonnage_7d_future_sum)

Architecture:
- Frozen Transformer Encoder Backbone (from Milestone 3.4/3.5 reference checkpoint)
- Non-negative residual prediction heads grounded in local persistence benchmarks:
    y_pred_container = ReLU( y_naive_container + Head_container(h) )
    y_pred_tonnage   = ReLU( y_naive_tonnage + Head_tonnage(h) * sigma_tonnage )
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .transformer_forecaster import ResidualTransformerForecaster


class DirectThroughputResidualForecaster(nn.Module):
    """
    Direct Throughput Forecaster coupling a frozen Transformer representation
    with dedicated residual throughput projection heads.
    """

    def __init__(
        self,
        backbone: ResidualTransformerForecaster,
        head_hidden_dim: int = 32,
        dropout: float = 0.1,
        mu_container_roll7: float = 0.6108,
        std_container_roll7: float = 2.3581,
        mu_tonnage: float = 21934.4852,
        std_tonnage: float = 96941.1915,
    ):
        super().__init__()
        self.backbone = backbone

        # Freeze entire backbone to guarantee zero weight modification
        for p in self.backbone.parameters():
            p.requires_grad = False

        d_model = backbone.d_model

        # Register unscaling constants as buffers
        self.register_buffer("mu_c", torch.tensor(mu_container_roll7, dtype=torch.float32))
        self.register_buffer("std_c", torch.tensor(std_container_roll7, dtype=torch.float32))
        self.register_buffer("mu_t", torch.tensor(mu_tonnage, dtype=torch.float32))
        self.register_buffer("std_t", torch.tensor(std_tonnage, dtype=torch.float32))

        # Container Calls Residual Head (calls scale ~ 0 to 300)
        self.container_head = nn.Sequential(
            nn.Linear(d_model, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )

        # Cargo Tonnage Residual Head (normalized scale)
        self.tonnage_head = nn.Sequential(
            nn.Linear(d_model, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )

        # Small initialization so predictions start exactly at persistence
        nn.init.xavier_uniform_(self.container_head[-1].weight, gain=0.01)
        nn.init.constant_(self.container_head[-1].bias, 0.0)
        nn.init.xavier_uniform_(self.tonnage_head[-1].weight, gain=0.01)
        nn.init.constant_(self.tonnage_head[-1].bias, 0.0)

    def extract_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled d_model representation from frozen Transformer backbone."""
        with torch.no_grad():
            h = self.backbone.input_proj(x)
            if self.backbone.pe_type == "learned":
                pos_indices = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
                h = h + self.backbone.pos_emb(pos_indices)
            else:
                h = self.backbone.pos_encoder(h)

            for layer in self.backbone.layers:
                h, _ = layer(h, need_weights=False)

            h = self.backbone.final_norm(h)
            if self.backbone.pooling == "mean":
                pooled = h.mean(dim=1)
            else:
                pooled = h[:, -1, :]
        return pooled

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Input tensor [B, 30, 20] adhering strictly to M3.1 20-feature core.
        Returns:
            Dict containing predictions and analytical persistence values for container calls and tonnage.
        """
        # 1. Analytical persistence for container vessel calls (Index 15 = pw_container_rolling_7d_mean at t-2)
        unscaled_roll7_container = x[:, -1, 15:16] * self.std_c + self.mu_c
        y_naive_container = torch.clamp(7.0 * unscaled_roll7_container, min=0.0)

        # 2. Analytical persistence for cargo tonnage (Index 11 = total_cargo_tonnage trailing 7 days)
        unscaled_daily_tonnage = x[:, 23:, 11:12] * self.std_t + self.mu_t
        y_naive_tonnage = torch.clamp(unscaled_daily_tonnage.sum(dim=1), min=0.0)

        # 3. Extract latent representation from frozen backbone
        h_pooled = self.extract_representation(x)

        # 4. Residual projection
        delta_container = self.container_head(h_pooled)
        delta_tonnage_norm = self.tonnage_head(h_pooled)

        # 5. Non-negative residual prediction
        y_pred_container = torch.relu(y_naive_container + delta_container)
        y_pred_tonnage = torch.relu(y_naive_tonnage + delta_tonnage_norm * self.std_t)

        return {
            "pred_container": y_pred_container,
            "naive_container": y_naive_container,
            "pred_tonnage": y_pred_tonnage,
            "naive_tonnage": y_naive_tonnage,
            "delta_container": delta_container,
            "delta_tonnage": delta_tonnage_norm * self.std_t,
        }

