#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stain-aware ViT + DINO head."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def _clean_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        nk = k
        for prefix in ("module.", "model.", "backbone.", "encoder.backbone."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        out[nk] = v
    return out


def _load_local_backbone_weights(backbone: nn.Module, ckpt_path: str | Path) -> Tuple[List[str], List[str]]:
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"UNI/backbone checkpoint not found: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        for key in ("state_dict", "model", "teacher", "student"):
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported checkpoint format for {path}")
    sd = _clean_state_dict(raw)
    return backbone.load_state_dict(sd, strict=False)


def _create_timm_backbone(
    timm_name: str,
    *,
    image_size: int,
    pretrained: bool,
    dynamic_img_size: bool,
) -> nn.Module:
    kwargs = {"pretrained": pretrained, "num_classes": 0}
    for extra in (
        {"img_size": image_size, "dynamic_img_size": dynamic_img_size},
        {"img_size": image_size},
        {"dynamic_img_size": dynamic_img_size},
        {},
    ):
        try:
            return timm.create_model(timm_name, **kwargs, **extra)
        except TypeError:
            continue
    raise RuntimeError(f"Failed to build timm model: {timm_name}")


class AdaLNConditioner(nn.Module):
    def __init__(self, dim: int, stain_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.modulation = nn.Sequential(
            nn.Linear(stain_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * dim),
        )
        self.gate = nn.Sequential(nn.Linear(stain_dim, dim), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, stain_vec: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.modulation(stain_vec)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        h = self.norm(x)
        h = h * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        g = self.gate(stain_vec).unsqueeze(1)
        return x + self.dropout(h * g)


class PromptGenerator(nn.Module):
    def __init__(self, stain_dim: int, embed_dim: int, num_tokens: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.embed_dim = int(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(stain_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_tokens * self.embed_dim),
        )

    def forward(self, stain_vec: torch.Tensor) -> torch.Tensor:
        x = self.mlp(stain_vec)
        return x.view(stain_vec.shape[0], self.num_tokens, self.embed_dim)


class CrossAttentionConditioner(nn.Module):
    def __init__(self, dim: int, stain_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.key_value_norm = nn.LayerNorm(dim)
        self.stain_proj = nn.Sequential(
            nn.Linear(stain_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, patch_tokens: torch.Tensor, stain_vec: torch.Tensor) -> torch.Tensor:
        q = self.query_norm(patch_tokens)
        kv = self.key_value_norm(self.stain_proj(stain_vec).unsqueeze(1))
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return patch_tokens + self.dropout(out)


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        nlayers: int = 3,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(nlayers):
            d_in = in_dim if i == 0 else hidden_dim
            d_out = bottleneck_dim if i == nlayers - 1 else hidden_dim
            is_last = i == nlayers - 1
            layers.append(nn.Linear(d_in, d_out, bias=not is_last))
            if not is_last:
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)
        self.last_weight_v = nn.Parameter(torch.empty(out_dim, bottleneck_dim))
        self.last_weight_g = nn.Parameter(torch.ones(out_dim, 1))
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.last_weight_v, std=0.02)
        nn.init.ones_(self.last_weight_g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        v = F.normalize(self.last_weight_v, dim=1, p=2)
        w = self.last_weight_g * v
        return F.linear(x, w)


class StainAwareVisionTransformer(nn.Module):
    """
    ViT with one of:
      - adaln
      - prompt
      - cross_attention
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["MODEL"]
        backbone_cfg = model_cfg["BACKBONE"]
        cond_cfg = model_cfg["CONDITIONING"]

        self.train_mode = str(model_cfg.get("TRAIN_MODE", "scratch")).lower()
        self.stain_dim = int(cfg["DATA"]["STAIN_DIM"])
        self.method = str(cond_cfg["METHOD"]).lower()
        self.cond_hidden = int(cond_cfg.get("HIDDEN_DIM", 512))
        self.dropout = float(cond_cfg.get("DROPOUT", 0.1))
        self.prompt_tokens = int(cond_cfg.get("PROMPT_TOKENS", 4))
        self.prompt_depth = int(cond_cfg.get("PROMPT_DEPTH", 0))
        self.cross_heads = int(cond_cfg.get("CROSS_ATTN_HEADS", 8))

        timm_name = str(backbone_cfg["TIMM_NAME"])
        image_size = int(backbone_cfg.get("IMAGE_SIZE", 224))
        dynamic_img_size = bool(backbone_cfg.get("DYNAMIC_IMG_SIZE", True))
        pretrained = bool(backbone_cfg.get("PRETRAINED", False))

        # UNI shortcut: pass hf-hub name directly as timm_name and enable pretrained.
        uni_source = backbone_cfg.get("UNI_SOURCE", None)
        if self.train_mode == "uni_frozen" and uni_source and str(uni_source).startswith(("hf-hub:", "timm/")):
            timm_name = str(uni_source)
            pretrained = True

        self.backbone = _create_timm_backbone(
            timm_name=timm_name,
            image_size=image_size,
            pretrained=pretrained,
            dynamic_img_size=dynamic_img_size,
        )
        self.embed_dim = int(self.backbone.embed_dim)
        self.depth = len(self.backbone.blocks)
        self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))

        if self.train_mode == "uni_frozen" and uni_source and not str(uni_source).startswith(("hf-hub:", "timm/")):
            missing, unexpected = _load_local_backbone_weights(self.backbone, str(uni_source))
            print(
                f"[UNI load] source={uni_source} missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )

        if self.method == "adaln":
            self.adaln_layers = nn.ModuleList(
                [
                    AdaLNConditioner(
                        dim=self.embed_dim,
                        stain_dim=self.stain_dim,
                        hidden_dim=self.cond_hidden,
                        dropout=self.dropout,
                    )
                    for _ in range(self.depth)
                ]
            )
            self.prompt_pos_embed = None
            self.prompt_generator = None
            self.deep_prompt_generators = None
            self.cross_layers = None
        elif self.method == "prompt":
            self.prompt_pos_embed = nn.Parameter(torch.zeros(1, self.prompt_tokens, self.embed_dim))
            nn.init.trunc_normal_(self.prompt_pos_embed, std=0.02)
            self.prompt_generator = PromptGenerator(
                stain_dim=self.stain_dim,
                embed_dim=self.embed_dim,
                num_tokens=self.prompt_tokens,
                hidden_dim=self.cond_hidden,
                dropout=self.dropout,
            )
            if self.prompt_depth > 0:
                self.deep_prompt_generators = nn.ModuleList(
                    [
                        PromptGenerator(
                            stain_dim=self.stain_dim,
                            embed_dim=self.embed_dim,
                            num_tokens=self.prompt_tokens,
                            hidden_dim=self.cond_hidden,
                            dropout=self.dropout,
                        )
                        for _ in range(self.prompt_depth)
                    ]
                )
            else:
                self.deep_prompt_generators = None
            self.adaln_layers = None
            self.cross_layers = None
        elif self.method == "cross_attention":
            self.cross_layers = nn.ModuleList(
                [
                    CrossAttentionConditioner(
                        dim=self.embed_dim,
                        stain_dim=self.stain_dim,
                        num_heads=self.cross_heads,
                        dropout=self.dropout,
                    )
                    for _ in range(self.depth)
                ]
            )
            self.adaln_layers = None
            self.prompt_pos_embed = None
            self.prompt_generator = None
            self.deep_prompt_generators = None
        else:
            raise ValueError(
                f"Unknown CONDITIONING.METHOD={self.method!r}. "
                "Expected one of: adaln | prompt | cross_attention."
            )

        if self.train_mode == "uni_frozen":
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def _embed_with_pos(self, x: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        patches = b.patch_embed(x)
        if hasattr(b, "_pos_embed"):
            tokens = b._pos_embed(patches)
            if hasattr(b, "pos_drop") and b.pos_drop is not None:
                tokens = b.pos_drop(tokens)
            return tokens

        if patches.dim() == 4:
            patches = patches.flatten(1, 2)
        cls = b.cls_token.expand(patches.shape[0], -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        if getattr(b, "pos_embed", None) is not None:
            pe = b.pos_embed[:, : tokens.shape[1], :]
            if pe.shape[1] != tokens.shape[1]:
                raise ValueError(
                    f"Fallback pos_embed path does not support dynamic token count: "
                    f"tokens={tokens.shape[1]} pos_embed={pe.shape[1]}"
                )
            tokens = tokens + pe
        if hasattr(b, "pos_drop") and b.pos_drop is not None:
            tokens = b.pos_drop(tokens)
        return tokens

    def _insert_prompt_tokens(self, tokens: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        prefix = tokens[:, : self.num_prefix_tokens, :]
        patches = tokens[:, self.num_prefix_tokens :, :]
        prompt = prompt + self.prompt_pos_embed
        return torch.cat([prefix, prompt, patches], dim=1)

    def _replace_prompt_tokens(self, tokens: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        prefix = tokens[:, : self.num_prefix_tokens, :]
        patch_start = self.num_prefix_tokens + self.prompt_tokens
        patches = tokens[:, patch_start:, :]
        prompt = prompt + self.prompt_pos_embed
        return torch.cat([prefix, prompt, patches], dim=1)

    def forward(self, x: torch.Tensor, stain_vec: torch.Tensor) -> torch.Tensor:
        tokens = self._embed_with_pos(x)

        if self.method == "prompt":
            prompt = self.prompt_generator(stain_vec)
            tokens = self._insert_prompt_tokens(tokens, prompt)

        if getattr(self.backbone, "norm_pre", None) is not None:
            tokens = self.backbone.norm_pre(tokens)

        for i, block in enumerate(self.backbone.blocks):
            if (
                self.method == "prompt"
                and self.deep_prompt_generators is not None
                and i < len(self.deep_prompt_generators)
            ):
                deep_prompt = self.deep_prompt_generators[i](stain_vec)
                tokens = self._replace_prompt_tokens(tokens, deep_prompt)

            tokens = block(tokens)

            if self.method == "adaln":
                tokens = self.adaln_layers[i](tokens, stain_vec)
            elif self.method == "cross_attention":
                patch_start = self.num_prefix_tokens
                patch_tokens = tokens[:, patch_start:, :]
                patch_tokens = self.cross_layers[i](patch_tokens, stain_vec)
                tokens = torch.cat([tokens[:, :patch_start, :], patch_tokens], dim=1)

        tokens = self.backbone.norm(tokens)
        cls = tokens[:, 0]
        return cls


class StainAwareDINO(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["MODEL"]
        dino_cfg = model_cfg["DINO"]

        self.encoder = StainAwareVisionTransformer(cfg)
        self.head = DINOHead(
            in_dim=self.encoder.embed_dim,
            out_dim=int(dino_cfg["OUT_DIM"]),
            hidden_dim=int(dino_cfg.get("HEAD_HIDDEN_DIM", 2048)),
            bottleneck_dim=int(dino_cfg.get("HEAD_BOTTLENECK_DIM", 256)),
            nlayers=int(dino_cfg.get("HEAD_NLAYERS", 3)),
        )

    def forward_views(self, views: List[torch.Tensor], stain_vec: torch.Tensor) -> torch.Tensor:
        logits = []
        for view in views:
            feat = self.encoder(view, stain_vec)
            logits.append(self.head(feat))
        return torch.cat(logits, dim=0)

    def encode(self, images: torch.Tensor, stain_vec: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.encoder(images, stain_vec)
            return F.normalize(feat, dim=-1)


def build_student_teacher(cfg: Dict[str, Any], device: torch.device) -> Tuple[StainAwareDINO, StainAwareDINO]:
    student = StainAwareDINO(cfg).to(device)
    teacher = copy.deepcopy(student).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return student, teacher

