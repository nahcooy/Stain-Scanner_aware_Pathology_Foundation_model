#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 Step 1: Patch-level Supervised Contrastive Learning — Training Script

Usage:
    python train.py --config configs/sm_stage4.yaml --domain sm
    python train.py --config configs/wsi_stage4.yaml --domain wsi

Example YAML (sm):
    SEED: 27
    PROCESSED_DIR: /path/to/PLISM-sm-processed_np_float32_optimized_4way_ver3
    RUN_ROOT: /path/to/saver
    DEVICE: cuda:3
    IMAGE_SIZE: 256
    TRAIN_BATCH_SIZE: 256
    VAL_BATCH_SIZE: 256
    NUM_WORKERS: 32
    PIN_MEMORY: true
    PERSISTENT_WORKERS: true
    PREFETCH_FACTOR: 4
    AUG:
      STAGE: 4
    MODEL:
      PRETRAINED: false
      EMBED_DIM: 1024
      PROJ_HIDDEN_DIM: 2048
      PROJ_DROPOUT: 0.0
      CHANNELS_LAST: true
      TORCH_COMPILE: true
    TRAIN:
      EPOCHS: 100
      BASE_LR: 1.0e-4
      WEIGHT_DECAY: 1.0e-4
      BETAS: [0.9, 0.999]
      WARMUP_EPOCHS: 5
      MIN_LR_RATIO: 0.01
      GRAD_CLIP_NORM: 1.0
    AMP:
      USE_AMP: true
      DTYPE: bf16
    LOSS:
      TEMPERATURE: 0.07
      POSITIVE_LOSS_WEIGHT: 0.1
    LOG_EVERY_N_STEPS: 10
    SAVE_EVERY_N_EPOCHS: 10
    RESUME_PATH: null

Additional WSI-specific keys:
    STORAGE_MODE: npy_individual   # or npy_shard
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    PLISMSMDataset,
    PLISMWSIDataset,
    RepeatKeySampler,
    TwoViewTransform,
    make_loader,
)
from loss import DeviceStainSupConLoss
from model import PatchEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def amp_dtype_from_str(name: str) -> torch.dtype:
    name = name.lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported AMP dtype: {name!r}")


def maybe_scaler(use_amp: bool, amp_dtype: torch.dtype) -> Optional[torch.amp.GradScaler]:
    if use_amp and torch.cuda.is_available() and amp_dtype == torch.float16:
        return torch.amp.GradScaler("cuda")
    return None


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()

    def isatty(self) -> bool:
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)


class FileLogger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.log_path = run_dir / "train.log"
        self.jsonl_path = run_dir / "metrics.jsonl"

    def log(self, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_json(self, payload: Dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def dump_config(self, cfg: Dict[str, Any]) -> None:
        with (self.run_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False, default=str)


def cleanup_memory(device: Optional[torch.device] = None,
                   logger: Optional[FileLogger] = None,
                   tag: str = "") -> None:
    gc.collect()
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        if logger is not None:
            alloc = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            logger.log(f"[memory] {tag} alloc={alloc:.2f}GB reserved={reserved:.2f}GB")


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer / Scheduler
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> AdamW:
    """AdamW with weight decay applied only to non-bias, non-norm parameters."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    param_groups = [
        {"params": decay, "weight_decay": float(cfg["TRAIN"]["WEIGHT_DECAY"])},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(
        param_groups,
        lr=float(cfg["TRAIN"]["BASE_LR"]),
        betas=tuple(cfg["TRAIN"]["BETAS"]),
    )


class WarmupCosineLR:
    """Step-level warmup + cosine decay scheduler (not a torch.optim.lr_scheduler)."""

    def __init__(
        self,
        optimizer: AdamW,
        total_steps: int,
        warmup_steps: int,
        min_lr_ratio: float,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.warmup_steps = max(0, warmup_steps)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_num = 0

    def _compute_lr(self, base_lr: float) -> float:
        if self.warmup_steps > 0 and self.step_num <= self.warmup_steps:
            return base_lr * self.step_num / self.warmup_steps
        progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return base_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine)

    def step(self) -> None:
        self.step_num += 1
        for base_lr, pg in zip(self.base_lrs, self.optimizer.param_groups):
            pg["lr"] = self._compute_lr(base_lr)

    def state_dict(self) -> Dict[str, Any]:
        return {"step_num": self.step_num}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.step_num = int(state.get("step_num", 0))
        for base_lr, pg in zip(self.base_lrs, self.optimizer.param_groups):
            pg["lr"] = self._compute_lr(base_lr)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: WarmupCosineLR,
    epoch: int,
    best_score: float,
    cfg: Dict[str, Any],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "cfg": cfg,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: WarmupCosineLR,
    device: torch.device,
) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt["epoch"]) + 1, float(ckpt["best_score"])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builders
# ─────────────────────────────────────────────────────────────────────────────

def build_sm_datasets(cfg: Dict[str, Any], logger: FileLogger):
    processed_dir = Path(cfg["PROCESSED_DIR"])
    train_aug = TwoViewTransform(stage=cfg["AUG"]["STAGE"], image_size=cfg["IMAGE_SIZE"])
    eval_aug = TwoViewTransform(stage=1, image_size=cfg["IMAGE_SIZE"])

    logger.log("[stage] loading SM datasets")
    train_ds = PLISMSMDataset(processed_dir, "train", train_aug)
    val_ds = PLISMSMDataset(processed_dir, "val", eval_aug)
    logger.log(f"[startup] SM train={len(train_ds)} val={len(val_ds)}")
    return train_ds, val_ds


def build_wsi_datasets(cfg: Dict[str, Any], logger: FileLogger):
    processed_dir = Path(cfg["PROCESSED_DIR"])
    storage_mode: str = cfg.get("STORAGE_MODE", "npy_individual")
    train_aug = TwoViewTransform(stage=cfg["AUG"]["STAGE"], image_size=cfg["IMAGE_SIZE"])
    eval_aug = TwoViewTransform(stage=1, image_size=cfg["IMAGE_SIZE"])

    logger.log("[stage] loading WSI datasets")
    train_ds = PLISMWSIDataset(processed_dir, "train", train_aug, storage_mode)
    val_ds = PLISMWSIDataset(processed_dir, "val", eval_aug, storage_mode)
    logger.log(f"[startup] WSI train={len(train_ds)} val={len(val_ds)}")
    return train_ds, val_ds


# ─────────────────────────────────────────────────────────────────────────────
# Train / Eval loops
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler: WarmupCosineLR,
    criterion: DeviceStainSupConLoss,
    device: torch.device,
    epoch: int,
    cfg: Dict[str, Any],
    logger: FileLogger,
    scaler: Optional[torch.amp.GradScaler],
) -> Dict[str, float]:
    model.train()
    use_amp: bool = bool(cfg["AMP"]["USE_AMP"])
    amp_dtype: torch.dtype = amp_dtype_from_str(cfg["AMP"]["DTYPE"])
    channels_last: bool = bool(cfg["MODEL"].get("CHANNELS_LAST", True))
    grad_clip: float = float(cfg["TRAIN"]["GRAD_CLIP_NORM"])
    log_every: int = int(cfg.get("LOG_EVERY_N_STEPS", 10))

    meters: Dict[str, float] = defaultdict(float)
    steps = 0
    pbar = tqdm(loader, desc=f"train {epoch:03d}", dynamic_ncols=True)

    for step, batch in enumerate(pbar, start=1):
        view1 = batch["view1"].to(device, non_blocking=True)
        view2 = batch["view2"].to(device, non_blocking=True)
        pos_keys: list[str] = batch["pos_key"]

        if channels_last:
            view1 = view1.contiguous(memory_format=torch.channels_last)
            view2 = view2.contiguous(memory_format=torch.channels_last)

        x = torch.cat([view1, view2], dim=0)

        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            z = model(x)
            loss, stats = criterion(z, pos_keys)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()

        for k, v in stats.items():
            meters[k] += v
        steps += 1

        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{meters['loss'] / steps:.4f}", lr=f"{lr:.2e}")

        if step % log_every == 0:
            logger.log(
                f"epoch={epoch:03d} step={step:05d}/{len(loader):05d} "
                f"loss={meters['loss'] / steps:.4f} "
                f"pair_acc={meters['pair_acc'] / steps:.4f} "
                f"lr={lr:.6e}"
            )

        del batch, view1, view2, x, z, loss

    cleanup_memory(device, tag=f"after_train_epoch_{epoch:03d}")
    return {k: v / steps for k, v in meters.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: DeviceStainSupConLoss,
    device: torch.device,
    cfg: Dict[str, Any],
    desc: str,
) -> Dict[str, float]:
    model.eval()
    use_amp: bool = bool(cfg["AMP"]["USE_AMP"])
    amp_dtype: torch.dtype = amp_dtype_from_str(cfg["AMP"]["DTYPE"])
    channels_last: bool = bool(cfg["MODEL"].get("CHANNELS_LAST", True))

    meters: Dict[str, float] = defaultdict(float)
    steps = 0

    for batch in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        view1 = batch["view1"].to(device, non_blocking=True)
        view2 = batch["view2"].to(device, non_blocking=True)
        pos_keys: list[str] = batch["pos_key"]

        if channels_last:
            view1 = view1.contiguous(memory_format=torch.channels_last)
            view2 = view2.contiguous(memory_format=torch.channels_last)

        x = torch.cat([view1, view2], dim=0)
        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            z = model(x)
            _, stats = criterion(z, pos_keys)

        for k, v in stats.items():
            meters[k] += v
        steps += 1
        del batch, view1, view2, x, z

    cleanup_memory(device, tag=desc.replace(" ", "_"))
    return {k: v / steps for k, v in meters.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 Step 1 — Patch-level SupCon training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--domain", required=True, choices=["sm", "wsi"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    domain: str = args.domain
    seed: int = int(cfg.get("SEED", 27))
    seed_everything(seed)

    device_str: str = str(cfg.get("DEVICE", "cuda"))
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {device_str} but CUDA is not available.")
    device = torch.device(device_str)

    # ── Run directory ─────────────────────────────────────────────────────────
    run_root = Path(cfg.get("RUN_ROOT", "./runs"))
    run_dir = ensure_dir(run_root / f"{datetime.now():%Y%m%d_%H%M%S}_vit_l_{domain}_step1")
    ckpt_dir = ensure_dir(run_dir / "checkpoints")

    stdout_log = (run_dir / "stdout.log").open("a", encoding="utf-8", buffering=1)
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = TeeStream(orig_stdout, stdout_log)
    sys.stderr = TeeStream(orig_stderr, stdout_log)

    try:
        logger = FileLogger(run_dir)
        logger.dump_config(cfg)
        logger.log(f"[startup] domain={domain} device={device} run_dir={run_dir}")

        # ── Datasets ──────────────────────────────────────────────────────────
        if domain == "sm":
            train_ds, val_ds = build_sm_datasets(cfg, logger)
        else:
            train_ds, val_ds = build_wsi_datasets(cfg, logger)

        # ── DataLoaders ───────────────────────────────────────────────────────
        loader_kwargs = dict(
            num_workers=int(cfg.get("NUM_WORKERS", 8)),
            pin_memory=bool(cfg.get("PIN_MEMORY", True)),
            persistent_workers=bool(cfg.get("PERSISTENT_WORKERS", True)),
            prefetch_factor=cfg.get("PREFETCH_FACTOR", 4),
            seed=seed,
        )
        train_bs: int = int(cfg.get("TRAIN_BATCH_SIZE", 256))
        val_bs: int = int(cfg.get("VAL_BATCH_SIZE", 256))

        train_loader = make_loader(train_ds, train_bs, shuffle=True, drop_last=True, **loader_kwargs)
        val_loader = make_loader(val_ds, val_bs, shuffle=False, drop_last=False, **loader_kwargs)

        logger.log(
            f"[startup] train_steps={len(train_loader)} val_steps={len(val_loader)} "
            f"train_bs={train_bs} val_bs={val_bs}"
        )

        # ── Model ─────────────────────────────────────────────────────────────
        model = PatchEncoder(cfg).to(device)
        if cfg["MODEL"].get("CHANNELS_LAST", True):
            model = model.to(memory_format=torch.channels_last)
            logger.log("[startup] channels_last enabled")
        if cfg["MODEL"].get("TORCH_COMPILE", False):
            logger.log("[startup] torch.compile enabled — compiling model")
            model = torch.compile(model)

        # ── Criterion / Optimizer / Scheduler ─────────────────────────────────
        criterion = DeviceStainSupConLoss(
            temperature=float(cfg["LOSS"]["TEMPERATURE"]),
            positive_loss_weight=float(cfg["LOSS"]["POSITIVE_LOSS_WEIGHT"]),
        )
        optimizer = build_optimizer(model, cfg)
        epochs: int = int(cfg["TRAIN"]["EPOCHS"])
        total_steps = len(train_loader) * epochs
        warmup_steps = len(train_loader) * int(cfg["TRAIN"]["WARMUP_EPOCHS"])
        scheduler = WarmupCosineLR(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=float(cfg["TRAIN"]["MIN_LR_RATIO"]),
        )
        amp_dtype = amp_dtype_from_str(cfg["AMP"]["DTYPE"])
        scaler = maybe_scaler(bool(cfg["AMP"]["USE_AMP"]), amp_dtype)
        logger.log(
            f"[startup] epochs={epochs} total_steps={total_steps} warmup_steps={warmup_steps} "
            f"base_lr={cfg['TRAIN']['BASE_LR']} temperature={cfg['LOSS']['TEMPERATURE']} "
            f"positive_loss_weight={cfg['LOSS']['POSITIVE_LOSS_WEIGHT']}"
        )

        # ── Resume ────────────────────────────────────────────────────────────
        start_epoch = 1
        best_val_loss = float("inf")
        best_pair_acc = float("-inf")
        best_margin = float("-inf")

        resume_path: Optional[str] = cfg.get("RESUME_PATH")
        if resume_path and Path(resume_path).is_file():
            start_epoch, best_pair_acc = load_checkpoint(
                Path(resume_path), model, optimizer, scheduler, device
            )
            logger.log(f"[startup] resumed from {resume_path} epoch={start_epoch}")

        save_every_n: int = int(cfg.get("SAVE_EVERY_N_EPOCHS", 0))

        # ── Training loop ─────────────────────────────────────────────────────
        logger.log("[stage] training loop start")
        for epoch in range(start_epoch, epochs + 1):
            logger.log(f"[epoch-start] {epoch:03d}/{epochs:03d}")

            train_metrics = train_one_epoch(
                model, train_loader, optimizer, scheduler, criterion,
                device, epoch, cfg, logger, scaler,
            )
            val_metrics = evaluate(
                model, val_loader, criterion, device, cfg,
                desc=f"val {epoch:03d}",
            )
            cleanup_memory(device, logger, tag=f"after_epoch_{epoch:03d}")

            val_margin = val_metrics["pos_sim"] - val_metrics["neg_sim"]
            logger.log(
                f"epoch={epoch:03d} | "
                f"train_loss={train_metrics['loss']:.4f} train_pair_acc={train_metrics['pair_acc']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_pair_acc={val_metrics['pair_acc']:.4f} "
                f"val_margin={val_margin:.4f}"
            )
            logger.log_json({
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "val_margin": val_margin,
            })

            # ── Checkpointing ─────────────────────────────────────────────────
            save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, epoch, best_pair_acc, cfg)

            if save_every_n > 0 and epoch % save_every_n == 0:
                save_checkpoint(
                    ckpt_dir / f"epoch_{epoch:03d}.pt",
                    model, optimizer, scheduler, epoch, best_pair_acc, cfg,
                )
                logger.log(f"[checkpoint] saved epoch_{epoch:03d}.pt")

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(ckpt_dir / "best_loss.pt", model, optimizer, scheduler, epoch, best_val_loss, cfg)
                logger.log(f"[checkpoint] updated best_loss.pt val_loss={best_val_loss:.6f}")

            if val_metrics["pair_acc"] > best_pair_acc:
                best_pair_acc = val_metrics["pair_acc"]
                save_checkpoint(ckpt_dir / "best_pair_acc.pt", model, optimizer, scheduler, epoch, best_pair_acc, cfg)
                logger.log(f"[checkpoint] updated best_pair_acc.pt val_pair_acc={best_pair_acc:.6f}")

            if val_margin > best_margin:
                best_margin = val_margin
                save_checkpoint(ckpt_dir / "best_margin.pt", model, optimizer, scheduler, epoch, best_margin, cfg)
                logger.log(f"[checkpoint] updated best_margin.pt val_margin={best_margin:.6f}")

            del train_metrics, val_metrics
            cleanup_memory(device, logger, tag=f"after_epoch_{epoch:03d}_ckpt")

        logger.log("[stage] training complete")

    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        stdout_log.close()


if __name__ == "__main__":
    main()
