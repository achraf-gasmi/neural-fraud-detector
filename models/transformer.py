"""
FraudShield — FT-Transformer (Feature Tokenizer + Transformer)
===============================================================
Based on "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., 2021).
Tokenizes each feature into a learned embedding, then applies Transformer blocks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FeatureTokenizer(nn.Module):
    """
    Converts flat tabular features into per-feature token embeddings.
    Each feature gets its own linear projection + bias (learnable).
    """

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        # One weight per feature (value → token)
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        # CLS token (aggregation token prepended to sequence)
        self.cls_token = nn.Parameter(torch.empty(1, d_token))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_features)
        Returns:
            tokens: (batch, n_features + 1, d_token)  — +1 for CLS
        """
        # x[:, :, None]: (B, F, 1) * weight: (F, D) → (B, F, D)
        tokens = x[:, :, None] * self.weight[None] + self.bias[None]  # (B, F, D)
        cls = self.cls_token.expand(x.size(0), -1, -1)                # (B, 1, D)
        return torch.cat([cls, tokens], dim=1)                         # (B, F+1, D)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_token: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_token % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_token // n_heads
        self.scale = self.d_head ** -0.5

        self.W_q = nn.Linear(d_token, d_token)
        self.W_k = nn.Linear(d_token, d_token)
        self.W_v = nn.Linear(d_token, d_token)
        self.W_o = nn.Linear(d_token, d_token)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H, Dh = self.n_heads, self.d_head

        def split_heads(t):
            return t.view(B, S, H, Dh).transpose(1, 2)  # (B, H, S, Dh)

        Q = split_heads(self.W_q(x))
        K = split_heads(self.W_k(x))
        V = split_heads(self.W_v(x))

        scores = (Q @ K.transpose(-2, -1)) * self.scale
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = (attn @ V).transpose(1, 2).contiguous().view(B, S, D)
        return self.W_o(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_token: int,
        n_heads: int,
        ffn_d_hidden: int,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.norm2 = nn.LayerNorm(d_token)
        self.attn = MultiHeadAttention(d_token, n_heads, attention_dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, ffn_d_hidden),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_d_hidden, d_token),
        )
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture (more stable than post-norm)
        x = x + self.residual_dropout(self.attn(self.norm1(x)))
        x = x + self.residual_dropout(self.ffn(self.norm2(x)))
        return x


class FTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer for tabular fraud features.
    Output: CLS token embedding of shape (batch, d_token)
    """

    def __init__(
        self,
        n_features: int,
        d_token: int = 192,
        n_blocks: int = 3,
        attention_n_heads: int = 8,
        attention_dropout: float = 0.2,
        ffn_d_hidden_multiplier: float = 1.333,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_token = d_token
        ffn_d_hidden = int(d_token * ffn_d_hidden_multiplier)

        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_token=d_token,
                n_heads=attention_n_heads,
                ffn_d_hidden=ffn_d_hidden,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
                residual_dropout=residual_dropout,
            )
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_features) tabular features
        Returns:
            embedding: (batch, d_token) — CLS token representation
        """
        tokens = self.tokenizer(x)           # (B, F+1, D)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens[:, 0]                  # CLS token → (B, D)
