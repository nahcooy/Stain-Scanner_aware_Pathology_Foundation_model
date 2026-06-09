#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 Step 2: Bag-level Stain Vector Learning — Model

StainVecStage2: aggregator + projector on top of frozen patch features.
Input:  [B, N, feat_dim] pre-computed patch features
Output: [B, embed_dim] L2-normalized stain vectors

Aggregators:
  ABMILAggregator       — gated attention MIL (Ilse et al. 2018)
  CrossAttentionAggregator — learnable query token + nn.MultiheadAttention
  MeanPoolAggregator    — simple mean pool baseline

Config keys:
  FEAT_DIM               int   (default 1024)
  MODEL.AGG_TYPE         str   "abmil" | "cross_attention" | "mean_pool"
  MODEL.EMBED_DIM        int   (default 256)
  MODEL.PROJ_HIDDEN_DIM  int   (default 1024)
  MODEL.PROJ_DROPOUT     float (default 0.1)
  MODEL.AGG_HIDDEN_DIM   int   (default 256)  — ABMIL hidden dim
  MODEL.AGG_HEADS        int   (default 8)    — CrossAttention heads
  MODEL.AGG_DROPOUT      float (default 0.1)
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Aggregators
# ─────────────────────────────────────────────────────────────────────────────

class ABMILAggregator(nn.Module):
    """
    Gated Attention-based Multiple Instance Learning pooling (Ilse et al. 2018).

    Attention weights:
        a_k = softmax( W_v * tanh(V * h_k) ⊙ sigmoid(U * h_k) )
    Pooled representation:
        z = Σ_k a_k * h_k

    Args:
        dim:        input feature dimension
        hidden_dim: internal attention hidden size (default 256)
        dropout:    dropout applied to both V and U projections (default 0.0)
    """

    def __init__(self, dim: int, hidden_dim: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        self.V = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Dropout(dropout))
        self.U = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Dropout(dropout))
        self.w = nn.Linear(hidden_dim, 1, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim]

        Returns:
            pooled: [B, dim]
        """
        v = torch.tanh(self.V(x))          # [B, N, H]
        u = torch.sigmoid(self.U(x))       # [B, N, H]
        attn_logits = self.w(v * u)        # [B, N, 1]
        attn = torch.softmax(attn_logits, dim=1)   # [B, N, 1]
        pooled = (attn * x).sum(dim=1)     # [B, dim]
        return self.norm(pooled)


class CrossAttentionAggregator(nn.Module):
    """
    Single learnable query token attends to all patch features via MHA.

    Args:
        dim:       input feature dimension
        num_heads: number of attention heads (default 8)
        dropout:   attention dropout (default 0.0)
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim]

        Returns:
            pooled: [B, dim]
        """
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)          # [B, 1, dim]
        out, _ = self.attn(q, x, x, need_weights=False)  # [B, 1, dim]
        pooled = out.squeeze(1)                   # [B, dim]
        return self.norm(pooled)


class MeanPoolAggregator(nn.Module):
    """
    Simple mean pooling over the patch dimension.

    Args:
        dim: input feature dimension
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim]

        Returns:
            pooled: [B, dim]
        """
        return self.norm(x.mean(dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class StainVecStage2(nn.Module):
    """
    Bag-level stain vector encoder.

    Aggregates N patch features into a single L2-normalized stain vector.

    Args:
        cfg: experiment config dict (see module docstring for keys)
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        feat_dim: int = int(cfg.get("FEAT_DIM", 1024))
        model_cfg: Dict[str, Any] = cfg["MODEL"]
        agg_type: str = str(model_cfg.get("AGG_TYPE", "abmil")).lower()
        embed_dim: int = int(model_cfg.get("EMBED_DIM", 256))
        proj_hidden_dim: int = int(model_cfg.get("PROJ_HIDDEN_DIM", 1024))
        proj_dropout: float = float(model_cfg.get("PROJ_DROPOUT", 0.1))
        agg_dropout: float = float(model_cfg.get("AGG_DROPOUT", 0.1))

        if agg_type == "abmil":
            agg_hidden_dim: int = int(model_cfg.get("AGG_HIDDEN_DIM", 256))
            self.aggregator: nn.Module = ABMILAggregator(
                dim=feat_dim,
                hidden_dim=agg_hidden_dim,
                dropout=agg_dropout,
            )
        elif agg_type == "cross_attention":
            agg_heads: int = int(model_cfg.get("AGG_HEADS", 8))
            self.aggregator = CrossAttentionAggregator(
                dim=feat_dim,
                num_heads=agg_heads,
                dropout=agg_dropout,
            )
        elif agg_type == "mean_pool":
            self.aggregator = MeanPoolAggregator(dim=feat_dim)
        else:
            raise ValueError(
                f"Unknown AGG_TYPE: {agg_type!r}. "
                "Expected 'abmil', 'cross_attention', or 'mean_pool'."
            )

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, proj_hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(proj_hidden_dim, embed_dim),
        )

        self.agg_type = agg_type
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, feat_dim] pre-computed patch features (float32)

        Returns:
            z: [B, embed_dim] L2-normalized stain vectors
        """
        pooled = self.aggregator(x)    # [B, feat_dim]
        z = self.projector(pooled)     # [B, embed_dim]
        return F.normalize(z, dim=-1)
