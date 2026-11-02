"""Caribbean Regional Spatio-Temporal Transformer-GNN for Milestone 3.5.

Combines the leading pre-trained and frozen Residual Transformer encoder (Milestone 3.4)
with spatial graph message passing across the 51 canonical Caribbean ports (777 AIS corridors):
    y_hat_i(t) = ReLU( y_naive_i(t) + Delta_transformer_i(X) + Delta_graph_i(X, G) )

Where:
- y_naive_i(t) is the established T-2 persistence forecast:
    7.0 * (X_i[29, roll7_idx] * sigma_roll7 + mu_roll7)
- Delta_transformer_i(X) is the temporal residual predicted by the frozen M3.4 Transformer.
- Delta_graph_i(X, G) is the spatial residual adjustment learned via graph message passing.
- Supports GATv2, GCN, and negative control topological ablations (shuffled edges, zero edges).
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_forecaster import ResidualTransformerForecaster


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

        h_src = self.W_src(h).unsqueeze(2)  # [B, N, 1, out_features]
        h_dst = self.W_dst(h).unsqueeze(1)  # [B, 1, N, out_features]

        scores = (self.attn_vec * F.leaky_relu(h_src + h_dst, self.negative_slope)).sum(dim=-1)  # [B, N, N]

        if adj_weights is not None:
            log_w = torch.log1p(adj_weights).unsqueeze(0)  # [1, N, N]
            scores = scores + 0.1 * log_w

        mask = adj_mask.unsqueeze(0)  # [1, N, N]
        scores = scores.masked_fill(~mask, -1e9)

        alpha = F.softmax(scores, dim=-1)  # [B, N, N]
        if self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

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
        agg = torch.matmul(norm_adj.unsqueeze(0), h)
        return F.elu(self.linear(agg))


class CaribbeanResidualTransformerGNN(nn.Module):
    """Modular Spatio-Temporal Residual Transformer-GNN for Caribbean Maritime Logistics."""

    def __init__(
        self,
        num_nodes: int = 51,
        input_size: int = 20,
        seq_len: int = 30,
        d_model: int = 64,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        dim_feedforward: int = 128,
        graph_hidden_size: int = 32,
        dropout: float = 0.1,
        pooling: str = "mean",
        pe_type: str = "sinusoidal",
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
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_transformer_layers = num_transformer_layers
        self.dim_feedforward = dim_feedforward
        self.graph_hidden_size = graph_hidden_size
        self.dropout = dropout
        self.pooling = pooling
        self.pe_type = pe_type
        self.gnn_type = gnn_type.lower()
        self.ablation = ablation.lower()
        self.freeze_temporal = freeze_temporal

        self.register_buffer("roll7_mean", torch.tensor(roll7_mean, dtype=torch.float32))
        self.register_buffer("roll7_std", torch.tensor(roll7_std, dtype=torch.float32))
        self.roll7_idx = roll7_idx

        # 1. Temporal Transformer Backbone
        self.transformer = ResidualTransformerForecaster(
            input_size=input_size,
            seq_len=seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_transformer_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pooling=pooling,
            pe_type=pe_type,
            roll7_idx=roll7_idx,
            roll7_mean=roll7_mean,
            roll7_std=roll7_std,
            constraint="relu",
        )

        # 2. Spatial GNN Layer
        if self.gnn_type == "gatv2":
            self.gnn = PurePyTorchGATv2Layer(
                in_features=d_model,
                out_features=self.graph_hidden_size,
                dropout=dropout,
            )
        elif self.gnn_type == "gcn":
            self.gnn = PurePyTorchGCNLayer(
                in_features=d_model,
                out_features=self.graph_hidden_size,
            )
        else:
            raise ValueError(f"Unsupported gnn_type: '{gnn_type}'. Choose 'gatv2' or 'gcn'.")

        # 3. Spatial Residual Head
        self.head_graph = nn.Linear(self.graph_hidden_size, 1)

        # Initialize graph head with small gain and zero bias so Delta_graph starts near 0.0
        nn.init.xavier_uniform_(self.head_graph.weight, gain=0.01)
        nn.init.constant_(self.head_graph.bias, 0.0)

    def load_temporal_pretrained(self, transformer_ckpt_path: str, device: torch.device):
        """Initialize Transformer backbone from the M3.4 trained checkpoint."""
        ckpt = torch.load(transformer_ckpt_path, map_location=device, weights_only=False)
        sd = ckpt["model_state_dict"]

        self.transformer.load_state_dict(sd)

        if self.freeze_temporal:
            for p in self.transformer.parameters():
                p.requires_grad = False
            print("Successfully loaded and FROZEN M3.4 Transformer backbone parameters.")
        else:
            print("Successfully loaded M3.4 Transformer backbone (fine-tuning enabled).")

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        adj_mask: torch.Tensor,
        norm_adj: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for Caribbean Regional Residual Transformer-GNN.
        
        Args:
            x: Input tensor [B, N=51, T=30, D=20].
            adj: Adjacency voyage weight matrix [N, N].
            adj_mask: Boolean adjacency mask [N, N].
            norm_adj: Optional normalized adjacency matrix [N, N] (for GCN).
            
        Returns:
            Tuple of (y_pred, delta_temporal, delta_graph) each [B, N].
        """
        B, N, T, D = x.shape

        # Step 1: Persistence projection for all 51 ports
        unscaled_roll7 = x[:, :, -1, self.roll7_idx] * self.roll7_std + self.roll7_mean
        y_naive = 7.0 * unscaled_roll7  # [B, N]

        # Step 2: Temporal Transformer Representation
        x_flat = x.view(B * N, T, D)
        
        # Ingest into Transformer
        proj = self.transformer.input_proj(x_flat)  # [B*N, T, d_model]
        pos = self.transformer.pos_encoder(proj)    # [B*N, T, d_model]

        h = pos
        for layer in self.transformer.layers:
            h, _ = layer(h)
        h = self.transformer.final_norm(h)  # [B*N, T, d_model]

        if self.pooling == "mean":
            h_pooled = h.mean(dim=1)  # [B*N, d_model]
        else:
            h_pooled = h[:, -1, :]    # [B*N, d_model]

        # Temporal residual
        delta_temporal_flat = self.transformer.head(h_pooled)  # [B*N, 1]
        delta_temporal = delta_temporal_flat.view(B, N)        # [B, N]

        # Step 3: Spatial Graph Message Passing
        h_nodes = h_pooled.view(B, N, self.d_model)  # [B, N, d_model]

        # Handle topological ablations
        if self.ablation == "no_edges":
            # No message passing: mask is pure identity (self-loops only)
            active_mask = torch.eye(N, dtype=torch.bool, device=x.device)
            active_adj = torch.eye(N, device=x.device)
            active_norm_adj = torch.eye(N, device=x.device)
        elif self.ablation == "shuffled":
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
            h_graph = self.gnn(h_nodes, active_norm_adj)          # [B, N, graph_hidden_size]

        # Spatial residual
        delta_graph = self.head_graph(h_graph).squeeze(-1)  # [B, N]

        # Step 4: Additive Residual Formulation with physical non-negativity
        y_pred = F.relu(y_naive + delta_temporal + delta_graph)  # [B, N]
        return y_pred, delta_temporal, delta_graph

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_type": "CaribbeanResidualTransformerGNN",
            "num_nodes": self.num_nodes,
            "input_size": self.input_size,
            "seq_len": self.seq_len,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_transformer_layers": self.num_transformer_layers,
            "dim_feedforward": self.dim_feedforward,
            "graph_hidden_size": self.graph_hidden_size,
            "dropout": self.dropout,
            "pooling": self.pooling,
            "pe_type": self.pe_type,
            "roll7_idx": self.roll7_idx,
            "roll7_mean": float(self.roll7_mean.item()),
            "roll7_std": float(self.roll7_std.item()),
            "gnn_type": self.gnn_type,
            "ablation": self.ablation,
            "freeze_temporal": self.freeze_temporal,
        }
