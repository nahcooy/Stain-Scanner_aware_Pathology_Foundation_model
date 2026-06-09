#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 Step 1: Patch-level Supervised Contrastive Learning — Model

PatchEncoder: ViT-L backbone (timm) + MLP projection head.
Forward returns L2-normalized embeddings. backbone is exposed for feature extraction.
"""

from __future__ import annotations

from typing import Any, Dict

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

TIMM_NAME = "vit_large_patch16_224.augreg_in21k"


class PatchEncoder(nn.Module):
    """
    ViT-L backbone with a two-layer MLP projection head.

    Config keys used:
        IMAGE_SIZE        : int  (default 256)
        MODEL.PRETRAINED  : bool (default False)
        MODEL.EMBED_DIM   : int  (default 1024)
        MODEL.PROJ_HIDDEN_DIM : int (default 2048)
        MODEL.PROJ_DROPOUT    : float (default 0.0)
        MODEL.CHANNELS_LAST   : bool (default True)
        MODEL.TORCH_COMPILE   : bool (default False) — applied externally by train.py
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        image_size: int = int(cfg.get("IMAGE_SIZE", 256))
        pretrained: bool = bool(cfg["MODEL"].get("PRETRAINED", False))
        embed_dim: int = int(cfg["MODEL"]["EMBED_DIM"])
        proj_hidden_dim: int = int(cfg["MODEL"]["PROJ_HIDDEN_DIM"])
        proj_dropout: float = float(cfg["MODEL"].get("PROJ_DROPOUT", 0.0))

        try:
            self.backbone = timm.create_model(
                TIMM_NAME,
                pretrained=pretrained,
                num_classes=0,
                img_size=image_size,
            )
        except TypeError:
            # fallback for timm versions that don't accept img_size
            self.backbone = timm.create_model(
                TIMM_NAME,
                pretrained=pretrained,
                num_classes=0,
            )

        feat_dim: int = self.backbone.num_features
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, proj_hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(proj_hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] float32 in [0, 1]

        Returns:
            z: [B, embed_dim] L2-normalized float32
        """
        feat = self.backbone(x)
        z = self.projector(feat)
        return F.normalize(z, dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw backbone features without projection head. [B, feat_dim]"""
        return self.backbone(x)
