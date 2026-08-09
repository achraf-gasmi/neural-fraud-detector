"""
FraudShield — Temporal Graph Neural Network
============================================
Heterogeneous GNN that aggregates information from transaction neighborhoods:
users, merchants, devices, IPs, and temporally adjacent transactions.
Uses Graph Attention Networks (GAT) for adaptive neighbor weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv, Linear, to_hetero
from torch_geometric.data import HeteroData


class TemporalEncoding(nn.Module):
    """
    Sinusoidal temporal position encoding for transaction timestamps.
    Injects time-of-day and recency information into node features.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch,) float tensor of normalized timestamps [0, 1]
        Returns:
            encoding: (batch, d_model)
        """
        half = self.d_model // 2
        freqs = torch.arange(half, device=t.device).float()
        freqs = 1.0 / (10000 ** (freqs / half))
        t_enc = t.unsqueeze(1) * freqs.unsqueeze(0)      # (B, half)
        encoding = torch.cat([torch.sin(t_enc), torch.cos(t_enc)], dim=-1)
        return self.linear(encoding)


class GNNLayer(nn.Module):
    """Single heterogeneous GAT layer across all edge types."""

    def __init__(self, hidden_channels: int, heads: int, dropout: float):
        super().__init__()
        self.conv = HeteroConv({
            ("transaction", "sent_by", "user"):    GATConv((-1, -1), hidden_channels, heads=heads, dropout=dropout, add_self_loops=False),
            ("user", "has_txn", "transaction"):    GATConv((-1, -1), hidden_channels, heads=heads, dropout=dropout, add_self_loops=False),
            ("transaction", "at", "merchant"):     GATConv((-1, -1), hidden_channels, heads=heads, dropout=dropout, add_self_loops=False),
            ("merchant", "received", "transaction"): GATConv((-1, -1), hidden_channels, heads=heads, dropout=dropout, add_self_loops=False),
            ("transaction", "used_device", "device"): GATConv((-1, -1), hidden_channels, heads=heads, dropout=dropout, add_self_loops=False),
            ("device", "used_in", "transaction"): GATConv((-1, -1), hidden_channels, heads=heads, dropout=dropout, add_self_loops=False),
            ("transaction", "from_ip", "ip"):      GATConv((-1, -1), hidden_channels, heads=heads, dropout=dropout, add_self_loops=False),
            ("ip", "origin_of", "transaction"):    GATConv((-1, -1), hidden_channels, heads=heads, dropout=dropout, add_self_loops=False),
            ("transaction", "precedes", "transaction"): GATConv((-1, -1), hidden_channels, heads=heads, dropout=dropout, add_self_loops=False),
        }, aggr="sum")

        self.norms = nn.ModuleDict({
            "transaction": nn.LayerNorm(hidden_channels * heads),
            "user":        nn.LayerNorm(hidden_channels * heads),
            "merchant":    nn.LayerNorm(hidden_channels * heads),
            "device":      nn.LayerNorm(hidden_channels * heads),
            "ip":          nn.LayerNorm(hidden_channels * heads),
        })

    def forward(self, x_dict, edge_index_dict):
        out = self.conv(x_dict, edge_index_dict)
        # Apply norm + residual where shapes match
        for node_type, h in out.items():
            h = self.norms[node_type](h)
            if node_type in x_dict and x_dict[node_type].shape == h.shape:
                h = h + x_dict[node_type]
            out[node_type] = F.gelu(h)
        return out


class TemporalGNN(nn.Module):
    """
    Multi-layer heterogeneous GAT encoder.
    Only returns transaction node embeddings (the target nodes for classification).
    """

    def __init__(
        self,
        txn_in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        time_encoding_dim: int = 32,
        entity_embedding_dim: int = 64,
        n_users: int = 5000,
        n_merchants: int = 500,
        n_devices: int = 15000,
        n_ips: int = 20000,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = hidden_channels * heads

        # Temporal encoding for transaction nodes
        self.time_encoding = TemporalEncoding(time_encoding_dim)

        # Project transaction features + time encoding → hidden_channels
        self.txn_proj = nn.Linear(txn_in_channels + time_encoding_dim, hidden_channels * heads)

        # Learnable entity embeddings (entities have no fixed features)
        self.user_emb = nn.Embedding(n_users + 1, hidden_channels * heads)
        self.merchant_emb = nn.Embedding(n_merchants + 1, hidden_channels * heads)
        self.device_emb = nn.Embedding(n_devices + 1, hidden_channels * heads)
        self.ip_emb = nn.Embedding(n_ips + 1, hidden_channels * heads)

        # GNN layers
        self.layers = nn.ModuleList([
            GNNLayer(hidden_channels, heads, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _entity_ids(store, device: torch.device) -> torch.Tensor:
        """
        Global entity IDs for embedding lookup.

        A NeighborLoader-sampled subgraph re-indexes nodes locally (0..k-1),
        so local position no longer matches the entity's row in the global
        embedding table. PyG preserves the original global index in `n_id`
        for sampled subgraphs; the full (unsampled) graph built by
        graph_builder.py has no `n_id`, so fall back to arange there.
        """
        if hasattr(store, "n_id"):
            return store.n_id.to(device)
        return torch.arange(store.num_nodes, device=device)

    def forward(self, data: HeteroData, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data: HeteroData graph batch (full graph or NeighborLoader sample)
            timestamps: (n_txns,) normalized timestamps [0, 1], aligned to
                every "transaction" node present in `data`
        Returns:
            txn_embeddings: (n_txns, out_channels)
        """
        # ── Initial node features ──
        t_enc = self.time_encoding(timestamps)
        txn_feats = torch.cat([data["transaction"].x, t_enc], dim=-1)
        txn_h = F.gelu(self.txn_proj(txn_feats))

        x_dict = {
            "transaction": txn_h,
            "user":        self.user_emb(self._entity_ids(data["user"], txn_h.device)),
            "merchant":    self.merchant_emb(self._entity_ids(data["merchant"], txn_h.device)),
            "device":      self.device_emb(self._entity_ids(data["device"], txn_h.device)),
            "ip":          self.ip_emb(self._entity_ids(data["ip"], txn_h.device)),
        }

        # ── Message passing ──
        edge_index_dict = data.edge_index_dict
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
            x_dict = {k: self.dropout(v) for k, v in x_dict.items()}

        return x_dict["transaction"]  # (n_txns, out_channels)
