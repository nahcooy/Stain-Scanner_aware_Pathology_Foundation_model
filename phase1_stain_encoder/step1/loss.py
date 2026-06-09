#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 Step 1: Patch-level Supervised Contrastive Learning — Loss

DeviceStainSupConLoss:
  Given a batch of 2N embeddings (N patches × 2 augmented views, interleaved
  as [v1_0, v1_1, ..., v1_{N-1}, v2_0, ..., v2_{N-1}]) and their N pos_keys,
  positive pairs are any two embeddings that share the same pos_key (excluding self).
  The loss is SupCon-style with an optional cosine-attraction auxiliary term.

  Training convention: caller concatenates view1 [N] and view2 [N] along dim=0
  to form z [2N]. pos_keys is the N-length list repeated once (all_keys = keys + keys).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


class DeviceStainSupConLoss(nn.Module):
    """
    Supervised contrastive loss keyed on (device, stain) identity.

    Args:
        temperature:           Logit scaling factor (default 0.07).
        positive_loss_weight:  Weight for auxiliary cosine-attraction term (default 0.1).
                               Set to 0 to use pure SupCon.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        positive_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature
        self.positive_loss_weight = positive_loss_weight

    def forward(
        self,
        z: torch.Tensor,
        pos_keys: List[str],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            z:        [2N, embed_dim] — view1 [N] followed by view2 [N], L2-normalized.
            pos_keys: list[str] of length N, one key per patch (repeated for both views).

        Returns:
            loss:  scalar tensor
            stats: dict with float metrics for logging
        """
        two_n = z.shape[0]
        n = two_n // 2
        device = z.device

        all_keys = pos_keys + pos_keys  # length 2N

        # Cosine similarity matrix [2N, 2N] (z is already L2-normalized)
        z32 = z.float()
        cosine_sim = torch.matmul(z32, z32.T)          # [2N, 2N]

        # Numerically stable shift
        sim = cosine_sim / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        # Build positive / self masks
        key_arr = np.asarray(all_keys, dtype=object)
        pos_mask_np: np.ndarray = key_arr[:, None] == key_arr[None, :]  # [2N, 2N]
        pos_mask = torch.from_numpy(pos_mask_np).to(device=device, dtype=torch.bool)
        self_mask = torch.eye(two_n, device=device, dtype=torch.bool)
        pos_mask = pos_mask & ~self_mask
        neg_mask = ~pos_mask & ~self_mask

        # SupCon log-softmax
        exp_sim = torch.exp(sim) * (~self_mask).to(z32.dtype)
        denom = exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12)
        log_prob = sim - torch.log(denom)                # [2N, 2N]

        pos_count = pos_mask.sum(dim=1).clamp_min(1)
        mean_log_prob_pos = (log_prob * pos_mask.to(z32.dtype)).sum(dim=1) / pos_count
        supcon_loss = -mean_log_prob_pos.mean()

        # Auxiliary cosine-attraction term: push positive pairs toward cosine sim = 1
        if pos_mask.any():
            positive_attract_loss = (1.0 - cosine_sim[pos_mask]).mean()
        else:
            positive_attract_loss = cosine_sim.new_zeros(())

        loss = supcon_loss + self.positive_loss_weight * positive_attract_loss

        # Diagnostics (no grad)
        with torch.no_grad():
            sim_det = cosine_sim.detach()
            pos_sim = sim_det[pos_mask].mean().item() if pos_mask.any() else 0.0
            neg_sim = sim_det[neg_mask].mean().item() if neg_mask.any() else 0.0
            # pair accuracy: is the nearest neighbour (excluding self) a positive?
            sim_for_top1 = sim.masked_fill(self_mask, -1e9)
            top1_idx = sim_for_top1.argmax(dim=1)
            pair_acc = pos_mask[torch.arange(two_n, device=device), top1_idx].float().mean().item()

        stats: Dict[str, float] = {
            "loss": loss.item(),
            "supcon_loss": supcon_loss.item(),
            "positive_attract_loss": positive_attract_loss.item(),
            "pos_sim": pos_sim,
            "neg_sim": neg_sim,
            "pair_acc": pair_acc,
        }
        return loss, stats
