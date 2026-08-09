"""
FraudShield — Real-Time Tabular Model
========================================
FT-Transformer + classification head + anomaly head, with no graph
dependency. This is what api/main.py serves — it needs to import cleanly
without pulling in training-only packages (hydra, mlflow, tqdm), so it lives
here rather than in training/train.py.

For the full FT-Transformer + Temporal GNN hybrid (fraud-ring detection,
run offline/in batch — see scripts/score_with_graph.py), see models/hybrid.py.
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from models.transformer import FTTransformer


class TabularFraudDetector(nn.Module):
    """
    Tabular-only version for fast training and low-latency serving.
    Uses only FT-Transformer + classification head + anomaly head.
    Swap for models.hybrid.FraudDetector when graph context is available.
    """

    def __init__(self, n_features: int, cfg: DictConfig):
        super().__init__()
        tc = cfg.model.transformer

        self.transformer = FTTransformer(
            n_features=n_features,
            d_token=tc.d_token,
            n_blocks=tc.n_blocks,
            attention_n_heads=tc.attention_n_heads,
            attention_dropout=tc.attention_dropout,
            ffn_d_hidden_multiplier=tc.ffn_d_hidden_multiplier,
            ffn_dropout=tc.ffn_dropout,
        )

        fc = cfg.model.fusion
        self.classifier = nn.Sequential(
            nn.Linear(tc.d_token, fc.hidden_dim),
            nn.GELU(),
            nn.Dropout(fc.dropout),
            nn.Linear(fc.hidden_dim, 1),
        )

        # Anomaly head
        self.anomaly_head = nn.Sequential(
            nn.Linear(tc.d_token, 128),
            nn.GELU(),
            nn.Linear(128, n_features),
        )

    def forward(self, x):
        emb = self.transformer(x)
        logits = self.classifier(emb)
        recon = self.anomaly_head(emb)
        return {
            "logits": logits,
            "probs": torch.sigmoid(logits.squeeze(-1)),
            "reconstruction": recon,
            "fused": emb,
        }
