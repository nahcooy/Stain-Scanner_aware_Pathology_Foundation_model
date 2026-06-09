#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DINO loss (CLS only) with EMA center."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t / dist.get_world_size()
    return t


class DINOLoss(nn.Module):
    """
    DINO self-distillation loss for CLS logits.

    Teacher consumes only global views.
    Student consumes global + local views.
    """

    def __init__(
        self,
        out_dim: int,
        n_global_crops: int,
        n_local_crops: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.n_global = int(n_global_crops)
        self.n_local = int(n_local_crops)
        self.student_temp = float(student_temp)
        self.center_momentum = float(center_momentum)
        self.register_buffer("center", torch.zeros(1, out_dim, dtype=torch.float32))

    @torch.no_grad()
    def _update_center(self, teacher_logits: torch.Tensor) -> None:
        batch_center = teacher_logits.float().mean(dim=0, keepdim=True)
        batch_center = _all_reduce_mean(batch_center)
        self.center.mul_(self.center_momentum).add_(batch_center * (1.0 - self.center_momentum))

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_temp: float,
        *,
        update_center: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if teacher_logits.ndim != 2 or student_logits.ndim != 2:
            raise ValueError("student_logits and teacher_logits must be 2D tensors")
        if teacher_logits.shape[1] != self.out_dim or student_logits.shape[1] != self.out_dim:
            raise ValueError(
                f"logit dim mismatch: expected out_dim={self.out_dim}, "
                f"got student={student_logits.shape}, teacher={teacher_logits.shape}"
            )
        if teacher_logits.shape[0] % self.n_global != 0:
            raise ValueError("teacher batch size must be divisible by n_global_crops")
        batch_size = teacher_logits.shape[0] // self.n_global

        n_student_crops = self.n_global + self.n_local
        if student_logits.shape[0] != batch_size * n_student_crops:
            raise ValueError(
                "student logits shape mismatch: expected "
                f"{batch_size * n_student_crops}, got {student_logits.shape[0]}"
            )

        student_chunks = student_logits.float().chunk(n_student_crops, dim=0)
        teacher_probs = F.softmax((teacher_logits.float() - self.center) / teacher_temp, dim=-1).detach()
        teacher_chunks = teacher_probs.chunk(self.n_global, dim=0)

        total_loss = torch.tensor(0.0, device=student_logits.device, dtype=torch.float32)
        n_terms = 0
        for t_idx, t_prob in enumerate(teacher_chunks):
            for s_idx, s_logit in enumerate(student_chunks):
                if s_idx == t_idx:
                    # Skip same-index global pair.
                    continue
                s_logp = F.log_softmax(s_logit / self.student_temp, dim=-1)
                total_loss = total_loss + (-(t_prob * s_logp).sum(dim=-1).mean())
                n_terms += 1
        total_loss = total_loss / max(1, n_terms)

        if update_center:
            self._update_center(teacher_logits)

        with torch.no_grad():
            teacher_entropy = (-(teacher_probs.clamp_min(1.0e-9).log() * teacher_probs).sum(dim=-1).mean()).item()
            student_entropy = (
                -(F.softmax(student_logits.float() / self.student_temp, dim=-1).clamp_min(1.0e-9).log()
                  * F.softmax(student_logits.float() / self.student_temp, dim=-1)).sum(dim=-1).mean()
            ).item()

        stats = {
            "teacher_entropy": float(teacher_entropy),
            "student_entropy": float(student_entropy),
            "center_norm": float(self.center.norm().item()),
        }
        return total_loss, stats

