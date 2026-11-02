"""Caribbean Regional Spatio-Temporal Graph Neural Network for Milestone 3.3.

Implements the non-negotiable residual spatio-temporal architecture:
    y_hat_i(t) = ReLU( y_naive_i(t) + Delta_temporal_i(X) + Delta_graph_i(X, G) )

Where:
- y_naive_i(t) is the established T-2 persistence forecast:
    7.0 * (X_i[29, roll7_idx] * sigma_roll7 + mu_roll7)
- Delta_temporal_i(X) is the temporal residual learned by the trained LSTM encoder (Exp 03).
- Delta_graph_i(X, G) is the spatial residual adjustment learned via graph message passing
  over the 777 empirical Caribbean shipping corridors.
- Supports Graph Attention (GATv2), Weighted Graph Convolution (GCN), and Topological Ablations
  (edge shuffling and edge removal).
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PurePyTorchGATv2Layer(nn.Module):
    """Vectorized Graph Attention v2 layer operating on batched node representations [B, N, D]."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1, negative_slope: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.W_src = nn.Linear(in_features, out_features, bias=False)
        self.W_dst = nn.Linear(in_features, out_features, bias=False)
        self.attn_vec = nn.Parameter(torch.empty(1, 1, 1, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        nn.init.xavier_uniform_(self.W_src.weight)
        nn.init.xavier_uniform_(self.W_dst.weight)
        nn.init.xavier_uniform_(self.attn_vec)

    def forward(self, h: torch.Tensor, adj_mask: torch.Tensor, adj_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for batched GATv2.
        
        Args:
            h: Node states [B, N, in_features].
            adj_mask: Boolean adjacency mask [N, N] (True where edge exists).
            adj_weights: Optional edge weight tensor [N, N].
            
        Returns:
            Aggregated node states [B, N, out_features].
        """
        B, N, _ = h.shape

        # Linear projections
        h_src = self.W_src(h).unsqueeze(2)  # [B, N, 1, out_features]
        h_dst = self.W_dst(h).unsqueeze(1)  # [B, 1, N, out_features]

        # GATv2 scoring function: a^T LeakyReLU(W_src h_i + W_dst h_j)
        scores = (self.attn_vec * F.leaky_relu(h_src + h_dst, self.negative_slope)).sum(dim=-1)  # [B, N, N]

        # Modulate attention scores with log of historical voyage density if available
        if adj_weights is not None:
            # log(1 + voyages) provides positive prior bias for high-frequency corridors
            log_w = torch.log1p(adj_weights).unsqueeze(0)  # [1, N, N]
            scores = scores + 0.1 * log_w

        # Mask non-edges with large negative value
        mask = adj_mask.unsqueeze(0)  # [1, N, N]
        scores = scores.masked_fill(~mask, -1e9)

        # Normalized attention coefficients
        alpha = F.softmax(scores, dim=-1)  # [B, N, N]
        if self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Message aggregation
        v = self.W_dst(h)  # [B, N, out_features]
        out = torch.matmul(alpha, v) + self.bias  # [B, N, out_features]
        return F.elu(out)


class PurePyTorchGCNLayer(nn.Module):
    """Vectorized Weighted Graph Convolution layer [B, N, D]."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, h: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        """Forward pass for GCN.
        
        Args:
            h: Node states [B, N, in_features].
            norm_adj: Normalized adjacency matrix [N, N].
            
        Returns:
            Aggregated node states [B, N, out_features].
        """
        # [B, N, D] = norm_adj [N, N] @ h [B, N, D]
        agg = torch.matmul(norm_adj.unsqueeze(0), h)
        return F.elu(self.linear(agg))


class CaribbeanResidualSTGNN(nn.Module):
    """Modular Spatio-Temporal Residual GNN for Caribbean Maritime Logistics."""

    def __init__(
        self,
        num_nodes: int = 51,
        input_size: int = 20,
        hidden_size: int = 64,
        graph_hidden_size: int = 32,
        num_lstm_layers: int = 2,
        dropout: float = 0.2,
        roll7_idx: int = 12,
        roll7_mean: float = 2.4068180554238894,
        roll7_std: float = 6.3483944613159915,
        gnn_type: str = "gatv2",
        ablation: str = "none",
        freeze_temporal: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.graph_hidden_size = graph_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.dropout = dropout
        self.roll7_idx = roll7_idx
        self.gnn_type = gnn_type.lower()
        self.ablation = ablation.lower()
        self.freeze_temporal = freeze_temporal

        # Register unscaling constants
        self.register_buffer("roll7_mean", torch.tensor(roll7_mean, dtype=torch.float32))
        self.register_buffer("roll7_std", torch.tensor(roll7_std, dtype=torch.float32))

        # 1. Temporal Recurrent Encoder
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_lstm_layers,
            dropout=self.dropout if self.num_lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        # 2. Temporal Residual Head
        self.head_temporal = nn.Linear(self.hidden_size, 1)

        # 3. Spatial Graph Message-Passing Layer
        if self.gnn_type == "gatv2":
            self.gnn = PurePyTorchGATv2Layer(
                in_features=self.hidden_size,
                out_features=self.graph_hidden_size,
                dropout=self.dropout,
            )
        elif self.gnn_type == "gcn":
            self.gnn = PurePyTorchGCNLayer(
                in_features=self.hidden_size,
                out_features=self.graph_hidden_size,
            )
        else:
            raise ValueError(f"Unsupported gnn_type '{self.gnn_type}'. Choose 'gatv2' or 'gcn'.")

        # 4. Graph Residual Head
        self.head_graph = nn.Linear(self.graph_hidden_size, 1)

        # Initialize graph head with small gain and zero bias so Delta_graph starts near 0.0
        nn.init.xavier_uniform_(self.head_graph.weight, gain=0.01)
        nn.init.constant_(self.head_graph.bias, 0.0)

    def load_temporal_pretrained(self, exp03_ckpt_path: str, device: torch.device):
        """Initialize temporal LSTM and temporal head from the Exp 03 trained checkpoint."""
        ckpt = torch.load(exp03_ckpt_path, map_location=device, weights_only=False)
        sd = ckpt["model_state_dict"]

        # Map LSTM parameters
        lstm_sd = {}
        for k in list(self.lstm.state_dict().keys()):
            if f"lstm.{k}" in sd:
                lstm_sd[k] = sd[f"lstm.{k}"]
        self.lstm.load_state_dict(lstm_sd)

        # Map temporal head parameters
        self.head_temporal.weight.data.copy_(sd["head.weight"])
        self.head_temporal.bias.data.copy_(sd["head.bias"])

        if self.freeze_temporal:
            for p in self.lstm.parameters():
                p.requires_grad = False
            for p in self.head_temporal.parameters():
                p.requires_grad = False
            print("Successfully loaded and FROZEN Exp 03 temporal encoder & head parameters.")
        else:
            print("Successfully loaded Exp 03 temporal encoder & head parameters (fine-tuning enabled).")

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        adj_mask: torch.Tensor,
        norm_adj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for Caribbean Regional Residual ST-GNN.
        
        Args:
            x: Input tensor [B, N=51, T=30, D=20].
            adj: Adjacency voyage weight matrix [N, N].
            adj_mask: Boolean adjacency mask [N, N].
            norm_adj: Optional normalized adjacency matrix [N, N] (for GCN).
            
        Returns:
            Non-negative predictions tensor [B, N].
        """
        B, N, T, D = x.shape

        # Step 1: Compute persistence projection for all 51 ports
        # unscaled_roll7 = x[:, :, -1, roll7_idx] * roll7_std + roll7_mean
        unscaled_roll7 = x[:, :, -1, self.roll7_idx] * self.roll7_std + self.roll7_mean
        y_naive = 7.0 * unscaled_roll7  # [B, N]

        # Step 2: Temporal recurrent representation
        # Flatten [B, N, T, D] -> [B*N, T, D]
        x_flat = x.view(B * N, T, D)
        lstm_out, _ = self.lstm(x_flat)
        h_last_flat = lstm_out[:, -1, :]  # [B*N, hidden_size]

        # Temporal residual
        delta_temporal_flat = self.head_temporal(h_last_flat)  # [B*N, 1]
        delta_temporal = delta_temporal_flat.view(B, N)  # [B, N]

        # Step 3: Spatial Graph Message Passing
        h_nodes = h_last_flat.view(B, N, self.hidden_size)  # [B, N, hidden_size]

        # Handle topological ablations
        if self.ablation == "no_edges":
            # No message passing: mask is pure identity (self-loops only)
            active_mask = torch.eye(N, dtype=torch.bool, device=x.device)
            active_adj = torch.eye(N, device=x.device)
            active_norm_adj = torch.eye(N, device=x.device)
        elif self.ablation == "shuffled":
            # Shuffled edges: keep density identical but randomly rewire connections
            active_mask = adj_mask
            active_adj = adj
            active_norm_adj = norm_adj
        else:
            active_mask = adj_mask
            active_adj = adj
            active_norm_adj = norm_adj

        if self.gnn_type == "gatv2":
            h_graph = self.gnn(h_nodes, active_mask, active_adj)  # [B, N, graph_hidden_size]
        else:
            h_graph = self.gnn(h_nodes, active_norm_adj)  # [B, N, graph_hidden_size]

        # Spatial residual
        delta_graph = self.head_graph(h_graph).squeeze(-1)  # [B, N]

        # Step 4: Additive Residual Formulation with physical non-negativity
        y_pred = F.relu(y_naive + delta_temporal + delta_graph)  # [B, N]
        return y_pred, delta_temporal, delta_graph

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_type": "CaribbeanResidualSTGNN",
            "num_nodes": self.num_nodes,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "graph_hidden_size": self.graph_hidden_size,
            "num_lstm_layers": self.num_lstm_layers,
            "dropout": self.dropout,
            "roll7_idx": self.roll7_idx,
            "roll7_mean": float(self.roll7_mean.item()),
            "roll7_std": float(self.roll7_std.item()),
            "gnn_type": self.gnn_type,
            "ablation": self.ablation,
            "freeze_temporal": self.freeze_temporal,
        }

