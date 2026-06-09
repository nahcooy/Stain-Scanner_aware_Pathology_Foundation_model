#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility helpers for Phase 2 training/evaluation."""

from __future__ import annotations

import gc
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from torch.optim import AdamW


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def amp_dtype_from_str(name: str) -> torch.dtype:
    key = str(name).lower()
    if key == "bf16":
        return torch.bfloat16
    if key == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported AMP dtype: {name!r}")


def maybe_scaler(use_amp: bool, amp_dtype: torch.dtype) -> Optional[torch.amp.GradScaler]:
    if use_amp and torch.cuda.is_available() and amp_dtype == torch.float16:
        return torch.amp.GradScaler("cuda")
    return None


def cleanup_memory(device: Optional[torch.device] = None) -> None:
    gc.collect()
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


class FileLogger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.log_path = run_dir / "train.log"
        self.metrics_path = run_dir / "metrics.jsonl"

    def log(self, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_json(self, payload: Dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def dump_config(self, cfg: Dict[str, Any]) -> None:
        with (self.run_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False, default=str)


class CosineSchedule:
    """Step-level warmup + cosine decay schedule."""

    def __init__(
        self,
        base_value: float,
        final_value: float,
        total_steps: int,
        warmup_steps: int = 0,
        start_warmup_value: float = 0.0,
    ) -> None:
        self.base_value = float(base_value)
        self.final_value = float(final_value)
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.start_warmup_value = float(start_warmup_value)

    def value_at(self, step: int) -> float:
        s = int(step)
        if self.warmup_steps > 0 and s < self.warmup_steps:
            alpha = s / max(1, self.warmup_steps)
            return self.start_warmup_value + alpha * (self.base_value - self.start_warmup_value)

        if self.total_steps <= self.warmup_steps:
            return self.final_value

        progress = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final_value + (self.base_value - self.final_value) * cosine


def build_optimizer(model: torch.nn.Module, train_cfg: Dict[str, Any]) -> AdamW:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    groups = [
        {"params": decay, "weight_decay": float(train_cfg["WEIGHT_DECAY"])},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(
        groups,
        lr=float(train_cfg["BASE_LR"]),
        betas=tuple(train_cfg.get("BETAS", (0.9, 0.999))),
        eps=float(train_cfg.get("EPS", 1.0e-8)),
    )


def set_optimizer_lr_wd(optimizer: AdamW, lr: float, wd: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
        if group.get("weight_decay", 0.0) > 0:
            group["weight_decay"] = wd


def count_parameters(module: torch.nn.Module) -> Dict[str, float]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {
        "total_m": total / 1.0e6,
        "trainable_m": trainable / 1.0e6,
        "trainable_ratio": (trainable / total) if total > 0 else 0.0,
    }


def copy_ema_weights(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    momentum: float,
) -> None:
    with torch.no_grad():
        for ps, pt in zip(student.parameters(), teacher.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)
        for bs, bt in zip(student.buffers(), teacher.buffers()):
            bt.copy_(bs)


def infer_steps_per_epoch(
    dataloader_len: int,
    requested: Optional[int],
) -> int:
    if requested is None:
        return int(dataloader_len)
    return max(1, min(int(requested), int(dataloader_len)))


def resolve_optional_path(path_like: Optional[str]) -> Optional[Path]:
    if path_like is None:
        return None
    text = str(path_like).strip()
    if not text or text.lower() == "null":
        return None
    return Path(text)


def flatten_dict(prefix: str, dct: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in dct.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(key, v))
        else:
            out[key] = v
    return out


def pretty_trainable_modules(module: torch.nn.Module) -> Iterable[str]:
    for name, p in module.named_parameters():
        if p.requires_grad:
            yield name

