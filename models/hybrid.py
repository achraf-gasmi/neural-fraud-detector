"""
FraudShield — Hybrid Fraud Detection Model
==========================================
Fuses:
  1. FT-Transformer  → tabular feature interactions
  2. Temporal GNN    → graph-based neighborhood signals
  3. Anomaly Head    → unsupervised reconstruction loss (learns "normal")

Classification head outputs P(fraud) per transaction.
Anomaly head outputs reconstruction for auxiliary loss during training.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData

from models.gnn import TemporalGNN
from models.transformer import FTTransformer


class AnomalyHead(nn.Module):
    """
    Autoencoder branch: reconstructs transaction features from fused embedding.
    High reconstruction error at inference → likely anomaly / fraud.
    """

    def __init__(self, input_dim: int, n_features: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_features),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """Returns reconstructed features (batch, n_features)."""
        z = self.encoder(fused)
        return self.decoder(z)

    def reconstruction_loss(self, recon: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(recon, original)


class FusionLayer(nn.Module):
    """
    Fuses Transformer and GNN embeddings.

    Each modality is gated (scaled by a learned, per-sample weight in [0, 1])
    before concatenation, so the model can learn to lean on the tabular
    signal or the graph-neighborhood signal depending on the transaction —
    e.g. a brand-new user has little graph history, so the gate should learn
    to downweight the GNN branch for them.
    """

    def __init__(self, transformer_dim: int, gnn_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        combined_dim = transformer_dim + gnn_dim

        self.gate = nn.Sequential(
            nn.Linear(combined_dim, 2),
            nn.Softmax(dim=-1),
        )

        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, transformer_emb: torch.Tensor, gnn_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            transformer_emb: (B, transformer_dim)
            gnn_emb: (B, gnn_dim)
        Returns:
            fused: (B, hidden_dim)
        """
        combined = torch.cat([transformer_emb, gnn_emb], dim=-1)  # (B, combined)
        gates = self.gate(combined)  # (B, 2)

        gated = torch.cat([
            gates[:, 0:1] * transformer_emb,
            gates[:, 1:2] * gnn_emb,
        ], dim=-1)  # (B, combined)

        return self.fusion(gated)


class FraudDetector(nn.Module):
    """
    Full FraudShield model.

    Forward pass:
      1. FT-Transformer encodes tabular features → (B, d_token)
      2. Temporal GNN encodes graph context → (B, gnn_out)
      3. Fusion layer combines both → (B, hidden_dim)
      4. Classification head → P(fraud)
      5. Anomaly head → reconstruction (for auxiliary loss)
    """

    def __init__(
        self,
        n_features: int,
        # Transformer config
        transformer_d_token: int = 192,
        transformer_n_blocks: int = 3,
        transformer_n_heads: int = 8,
        transformer_attn_dropout: float = 0.2,
        transformer_ffn_d_hidden_mult: float = 1.333,
        transformer_ffn_dropout: float = 0.1,
        # GNN config
        gnn_hidden_channels: int = 128,
        gnn_num_layers: int = 3,
        gnn_heads: int = 4,
        gnn_dropout: float = 0.2,
        gnn_time_encoding_dim: int = 32,
        # Entity counts for GNN embeddings
        n_users: int = 5000,
        n_merchants: int = 500,
        n_devices: int = 15000,
        n_ips: int = 20000,
        # Fusion + heads
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.3,
        anomaly_hidden_dim: int = 128,
        use_anomaly_head: bool = True,
    ):
        super().__init__()
        self.use_anomaly_head = use_anomaly_head
        self.n_features = n_features

        # ── Encoders ──
        self.transformer = FTTransformer(
            n_features=n_features,
            d_token=transformer_d_token,
            n_blocks=transformer_n_blocks,
            attention_n_heads=transformer_n_heads,
            attention_dropout=transformer_attn_dropout,
            ffn_d_hidden_multiplier=transformer_ffn_d_hidden_mult,
            ffn_dropout=transformer_ffn_dropout,
        )

        gnn_out_dim = gnn_hidden_channels * gnn_heads
        self.gnn = TemporalGNN(
            txn_in_channels=n_features,
            hidden_channels=gnn_hidden_channels,
            num_layers=gnn_num_layers,
            heads=gnn_heads,
            dropout=gnn_dropout,
            time_encoding_dim=gnn_time_encoding_dim,
            n_users=n_users,
            n_merchants=n_merchants,
            n_devices=n_devices,
            n_ips=n_ips,
        )

        # ── Fusion ──
        self.fusion = FusionLayer(
            transformer_dim=transformer_d_token,
            gnn_dim=gnn_out_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=fusion_dropout,
        )

        # ── Classification Head ──
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim // 2, 1),
        )

        # ── Anomaly Head ──
        if use_anomaly_head:
            self.anomaly_head = AnomalyHead(
                input_dim=fusion_hidden_dim,
                n_features=n_features,
                hidden_dim=anomaly_hidden_dim,
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, graph: HeteroData) -> dict:
        """
        Args:
            graph: HeteroData batch of transaction nodes plus their sampled
                neighborhood (as produced by graph_builder.py for the full
                graph, or by a PyG NeighborLoader for mini-batches). The
                "transaction" node store must carry `.x` (features) and
                `.time_norm` (normalized timestamp) for every node in the
                subgraph, not just the seed transactions being scored.

        Seed transactions — the ones this batch is actually scoring — are
        the first `batch_size` rows of graph["transaction"].x. NeighborLoader
        guarantees this ordering for its `input_nodes`; graph_builder.py's
        full, unsampled graph has no separate neighborhood so every node is
        its own seed.

        Returns:
            dict with keys:
              - logits:       (B, 1) raw logits
              - probs:        (B,) fraud probability
              - reconstruction: (B, n_features) if anomaly head enabled
              - fused:        (B, hidden_dim) fused representation
        """
        txn_store = graph["transaction"]
        batch_size = getattr(txn_store, "batch_size", txn_store.x.size(0))
        x_seed = txn_store.x[:batch_size]                       # (B, n_features)

        # Tabular path — only the seed transactions being scored
        transformer_emb = self.transformer(x_seed)              # (B, d_token)

        # Graph path — runs over the full sampled subgraph, then we keep
        # only the embeddings for the seed transactions
        gnn_emb_all = self.gnn(graph, txn_store.time_norm)       # (n_txns_in_subgraph, gnn_out)
        gnn_emb = gnn_emb_all[:batch_size]                       # (B, gnn_out)

        # Fusion
        fused = self.fusion(transformer_emb, gnn_emb)           # (B, hidden_dim)

        # Classification
        logits = self.classifier(fused)                         # (B, 1)
        probs = torch.sigmoid(logits.squeeze(-1))               # (B,)

        out = {
            "logits": logits,
            "probs": probs,
            "fused": fused,
            "transformer_emb": transformer_emb,
            "gnn_emb": gnn_emb,
        }

        # Anomaly reconstruction — target is the seed transactions' own features
        if self.use_anomaly_head:
            reconstruction = self.anomaly_head(fused)           # (B, n_features)
            out["reconstruction"] = reconstruction

        return out

    @torch.no_grad()
    def predict(self, graph: HeteroData) -> torch.Tensor:
        """Inference-only: returns fraud probabilities for the seed transactions."""
        self.eval()
        return self.forward(graph)["probs"]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
