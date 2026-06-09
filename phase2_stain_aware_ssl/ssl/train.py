#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2: Stain-Aware ViT DINO SSL Training (PNG patches, train/val directory).

Supports:
  - conditioning method: adaln | prompt | cross_attention
  - train mode: scratch | uni_frozen
  - distributed training via torchrun (default 4 GPUs in scripts)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from augmentations import build_train_transform, build_val_transform
from dataset import PatchMultiCropDataset, collate_multicrop, make_loader
from loss import DINOLoss
from model import build_student_teacher
from utils import (
    CosineSchedule,
    FileLogger,
    amp_dtype_from_str,
    build_optimizer,
    cleanup_memory,
    copy_ema_weights,
    count_parameters,
    ensure_dir,
    infer_steps_per_epoch,
    maybe_scaler,
    pretty_trainable_modules,
    resolve_optional_path,
    seed_everything,
    set_optimizer_lr_wd,
)


class NoOpLogger:
    def log(self, _msg: str) -> None:
        return

    def log_json(self, _payload: Dict[str, Any]) -> None:
        return

    def dump_config(self, _cfg: Dict[str, Any]) -> None:
        return


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def dist_reduce_mean_scalar(value: float, device: torch.device) -> float:
    t = torch.tensor([float(value)], dtype=torch.float32, device=device)
    if is_dist_avail_and_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t / dist.get_world_size()
    return float(t.item())


def dist_reduce_sum_count(value_sum: float, value_count: int, device: torch.device) -> Tuple[float, int]:
    t = torch.tensor([float(value_sum), float(value_count)], dtype=torch.float64, device=device)
    if is_dist_avail_and_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t[0].item()), int(t[1].item())


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _forward_views(model: torch.nn.Module, views: list[torch.Tensor], stain_vec: torch.Tensor) -> torch.Tensor:
    return unwrap_model(model).forward_views(views, stain_vec)


def _to_device(batch: Dict[str, Any], device: torch.device) -> Tuple[list[torch.Tensor], torch.Tensor]:
    views = [v.to(device, non_blocking=True) for v in batch["views"]]
    stain_vec = batch["stain_vec"].to(device, non_blocking=True)
    return views, stain_vec


def save_checkpoint(
    path: Path,
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    criterion: DINOLoss,
    cfg: Dict[str, Any],
    epoch: int,
    global_step: int,
    best_val_loss: float,
) -> None:
    ckpt = {
        "student": unwrap_model(student).state_dict(),
        "teacher": unwrap_model(teacher).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "criterion_center": criterion.center.detach().cpu(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_loss": float(best_val_loss),
        "cfg": cfg,
    }
    torch.save(ckpt, path)


def load_checkpoint(
    path: Path,
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    criterion: DINOLoss,
    device: torch.device,
) -> Tuple[int, int, float]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    unwrap_model(student).load_state_dict(ckpt["student"], strict=True)
    unwrap_model(teacher).load_state_dict(ckpt["teacher"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if "criterion_center" in ckpt:
        criterion.center.copy_(ckpt["criterion_center"].to(criterion.center.device))
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    global_step = int(ckpt.get("global_step", 0))
    best_val_loss = float(ckpt.get("best_val_loss", math.inf))
    return start_epoch, global_step, best_val_loss


@torch.no_grad()
def run_validation(
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    criterion: DINOLoss,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    n_global_crops: int,
    teacher_temp: float,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    unwrap_model(student).eval()
    unwrap_model(teacher).eval()
    losses = []
    t_entropy = []
    s_entropy = []

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        views, stain_vec = _to_device(batch, device)
        global_views = views[:n_global_crops]

        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            student_logits = _forward_views(student, views, stain_vec)
            teacher_logits = _forward_views(teacher, global_views, stain_vec)
            loss, stats = criterion(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                teacher_temp=teacher_temp,
                update_center=False,
            )

        losses.append(float(loss.detach().item()))
        t_entropy.append(float(stats["teacher_entropy"]))
        s_entropy.append(float(stats["student_entropy"]))

    unwrap_model(student).train()
    unwrap_model(teacher).eval()

    local = {
        "loss_sum": float(sum(losses)),
        "loss_count": int(len(losses)),
        "t_sum": float(sum(t_entropy)),
        "t_count": int(len(t_entropy)),
        "s_sum": float(sum(s_entropy)),
        "s_count": int(len(s_entropy)),
    }
    loss_sum, loss_count = dist_reduce_sum_count(local["loss_sum"], local["loss_count"], device)
    t_sum, t_count = dist_reduce_sum_count(local["t_sum"], local["t_count"], device)
    s_sum, s_count = dist_reduce_sum_count(local["s_sum"], local["s_count"], device)

    if loss_count == 0:
        return {"val_loss": math.nan, "val_teacher_entropy": math.nan, "val_student_entropy": math.nan}
    return {
        "val_loss": loss_sum / max(1, loss_count),
        "val_teacher_entropy": t_sum / max(1, t_count),
        "val_student_entropy": s_sum / max(1, s_count),
    }


def build_run_dir(cfg: Dict[str, Any]) -> Path:
    run_root = ensure_dir(cfg["RUN_ROOT"])
    run_name = str(cfg["RUN_NAME"])
    return ensure_dir(run_root / run_name)


def init_distributed(cfg: Dict[str, Any]) -> Tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
        requested = str(cfg.get("DEVICE", "cuda:0"))
        device = torch.device(requested if torch.cuda.is_available() else "cpu")
    return distributed, rank, local_rank, world_size, device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg: Dict[str, Any] = yaml.safe_load(args.config.read_text())
    distributed, rank, local_rank, world_size, device = init_distributed(cfg)
    seed_everything(int(cfg.get("SEED", 42)))

    amp_cfg = cfg.get("AMP", {})
    amp_enabled = bool(amp_cfg.get("USE_AMP", True)) and device.type == "cuda"
    amp_dtype = amp_dtype_from_str(str(amp_cfg.get("DTYPE", "bf16")))
    scaler = maybe_scaler(amp_enabled, amp_dtype)

    run_dir = build_run_dir(cfg)
    if distributed:
        dist.barrier()
    logger = FileLogger(run_dir) if rank == 0 else NoOpLogger()
    logger.dump_config(cfg)
    logger.log(f"config={args.config}")
    logger.log(
        f"distributed={distributed} world_size={world_size} rank={rank} local_rank={local_rank} "
        f"device={device} amp={amp_enabled} amp_dtype={amp_dtype}"
    )

    train_tf = build_train_transform(cfg)
    val_tf = build_val_transform(cfg)
    data_cfg = cfg["DATA"]
    dl_cfg = cfg["DATALOADER"]

    train_set = PatchMultiCropDataset(
        root_dir=data_cfg["TRAIN_DIR"],
        transform=train_tf,
        stain_dim=int(data_cfg["STAIN_DIM"]),
        stain_vector_dir=data_cfg.get("STAIN_VECTOR_DIR"),
        stain_index_csv=data_cfg.get("STAIN_INDEX_CSV"),
        require_stain_vector=bool(data_cfg.get("REQUIRE_STAIN_VECTOR", True)),
        recursive=bool(data_cfg.get("RECURSIVE", True)),
        extensions=data_cfg.get("EXTENSIONS"),
    )
    val_set = PatchMultiCropDataset(
        root_dir=data_cfg["VAL_DIR"],
        transform=val_tf,
        stain_dim=int(data_cfg["STAIN_DIM"]),
        stain_vector_dir=data_cfg.get("STAIN_VECTOR_DIR"),
        stain_index_csv=data_cfg.get("STAIN_INDEX_CSV"),
        require_stain_vector=bool(data_cfg.get("REQUIRE_STAIN_VECTOR", True)),
        recursive=bool(data_cfg.get("RECURSIVE", True)),
        extensions=data_cfg.get("EXTENSIONS"),
    )

    per_gpu_train_bs = int(dl_cfg["TRAIN_BATCH_SIZE"])
    per_gpu_val_bs = int(dl_cfg.get("VAL_BATCH_SIZE", dl_cfg["TRAIN_BATCH_SIZE"]))
    train_sampler = (
        DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        if distributed
        else None
    )
    val_sampler = (
        DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        if distributed
        else None
    )

    train_loader = make_loader(
        train_set,
        batch_size=per_gpu_train_bs,
        num_workers=int(dl_cfg.get("NUM_WORKERS", 8)),
        pin_memory=bool(dl_cfg.get("PIN_MEMORY", True)),
        drop_last=True,
        shuffle=True,
        persistent_workers=bool(dl_cfg.get("PERSISTENT_WORKERS", False)),
        prefetch_factor=int(dl_cfg.get("PREFETCH_FACTOR", 2)),
        collate_fn=collate_multicrop,
        sampler=train_sampler,
    )
    val_loader = make_loader(
        val_set,
        batch_size=per_gpu_val_bs,
        num_workers=int(dl_cfg.get("NUM_WORKERS", 8)),
        pin_memory=bool(dl_cfg.get("PIN_MEMORY", True)),
        drop_last=False,
        shuffle=False,
        persistent_workers=bool(dl_cfg.get("PERSISTENT_WORKERS", False)),
        prefetch_factor=int(dl_cfg.get("PREFETCH_FACTOR", 2)),
        collate_fn=collate_multicrop,
        sampler=val_sampler,
    )
    logger.log(
        f"train_samples={len(train_set)} val_samples={len(val_set)} "
        f"per_gpu_batch={per_gpu_train_bs} effective_batch={per_gpu_train_bs * world_size}"
    )

    student, teacher = build_student_teacher(cfg, device=device)

    if bool(cfg["MODEL"].get("TORCH_COMPILE", False)) and hasattr(torch, "compile") and not distributed:
        student = torch.compile(student)  # type: ignore[assignment]
        logger.log("torch.compile enabled for student (single GPU mode)")
    elif bool(cfg["MODEL"].get("TORCH_COMPILE", False)) and distributed:
        logger.log("torch.compile is skipped in distributed mode for stability")

    if distributed:
        student = DDP(student, device_ids=[local_rank] if device.type == "cuda" else None, broadcast_buffers=False)

    counts = count_parameters(unwrap_model(student))
    logger.log(
        "params total={:.2f}M trainable={:.2f}M ({:.2%})".format(
            counts["total_m"], counts["trainable_m"], counts["trainable_ratio"]
        )
    )
    trainable_preview = list(pretty_trainable_modules(unwrap_model(student)))[:25]
    logger.log("trainable_param_preview=" + json.dumps(trainable_preview, ensure_ascii=False))

    aug_cfg = cfg["AUG"]
    dino_cfg = cfg["MODEL"]["DINO"]
    train_cfg = cfg["TRAIN"]
    n_global = int(aug_cfg.get("N_GLOBAL_CROPS", 2))
    n_local = int(aug_cfg.get("N_LOCAL_CROPS", 6))

    criterion = DINOLoss(
        out_dim=int(dino_cfg["OUT_DIM"]),
        n_global_crops=n_global,
        n_local_crops=n_local,
        student_temp=float(train_cfg.get("STUDENT_TEMP", 0.1)),
        center_momentum=float(train_cfg.get("CENTER_MOMENTUM", 0.9)),
    ).to(device)

    optimizer = build_optimizer(unwrap_model(student), train_cfg)
    epochs = int(train_cfg["EPOCHS"])
    steps_per_epoch = infer_steps_per_epoch(len(train_loader), train_cfg.get("STEPS_PER_EPOCH"))
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * int(train_cfg.get("WARMUP_EPOCHS", 10))
    teacher_temp_warmup_steps = steps_per_epoch * int(train_cfg.get("TEACHER_TEMP_WARMUP_EPOCHS", 20))

    lr_sched = CosineSchedule(
        base_value=float(train_cfg["BASE_LR"]),
        final_value=float(train_cfg.get("MIN_LR", 1.0e-6)),
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        start_warmup_value=float(train_cfg.get("WARMUP_START_LR", 1.0e-6)),
    )
    wd_sched = CosineSchedule(
        base_value=float(train_cfg["WEIGHT_DECAY"]),
        final_value=float(train_cfg.get("WEIGHT_DECAY_END", train_cfg["WEIGHT_DECAY"])),
        total_steps=total_steps,
    )
    momentum_sched = CosineSchedule(
        base_value=float(train_cfg.get("EMA_MOMENTUM_START", 0.996)),
        final_value=float(train_cfg.get("EMA_MOMENTUM_END", 1.0)),
        total_steps=total_steps,
    )
    teacher_temp_sched = CosineSchedule(
        base_value=float(train_cfg.get("TEACHER_TEMP_END", 0.07)),
        final_value=float(train_cfg.get("TEACHER_TEMP_END", 0.07)),
        total_steps=max(1, teacher_temp_warmup_steps),
        warmup_steps=max(1, teacher_temp_warmup_steps),
        start_warmup_value=float(train_cfg.get("TEACHER_TEMP_START", 0.04)),
    )

    start_epoch = 0
    global_step = 0
    best_val_loss = math.inf

    resume_path = resolve_optional_path(cfg.get("RESUME_PATH"))
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"RESUME_PATH not found: {resume_path}")
        start_epoch, global_step, best_val_loss = load_checkpoint(
            resume_path,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
        )
        logger.log(
            f"resumed from {resume_path} start_epoch={start_epoch} "
            f"global_step={global_step} best_val_loss={best_val_loss:.6f}"
        )
    if distributed:
        dist.barrier()

    save_every = int(cfg.get("SAVE_EVERY_N_EPOCHS", 5))
    log_every = int(cfg.get("LOG_EVERY_N_STEPS", 20))
    val_max_steps = cfg.get("VAL_MAX_STEPS", None)
    if val_max_steps is not None:
        val_max_steps = int(val_max_steps)

    logger.log(
        "training start "
        f"epochs={epochs} steps_per_epoch={steps_per_epoch} total_steps={total_steps} "
        f"n_global={n_global} n_local={n_local}"
    )

    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        unwrap_model(student).train()
        unwrap_model(teacher).eval()
        running_loss = 0.0
        running_t_ent = 0.0
        running_s_ent = 0.0
        epoch_start = time.time()

        show_pbar = rank == 0
        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch:03d}", dynamic_ncols=True, disable=not show_pbar)
        for step, batch in enumerate(train_loader):
            if step >= steps_per_epoch:
                break

            lr = lr_sched.value_at(global_step)
            wd = wd_sched.value_at(global_step)
            set_optimizer_lr_wd(optimizer, lr=lr, wd=wd)
            teacher_momentum = momentum_sched.value_at(global_step)
            teacher_temp = teacher_temp_sched.value_at(global_step)

            views, stain_vec = _to_device(batch, device)
            global_views = views[:n_global]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                student_logits = _forward_views(student, views, stain_vec)
                with torch.no_grad():
                    teacher_logits = _forward_views(teacher, global_views, stain_vec)
                loss, loss_stats = criterion(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    teacher_temp=teacher_temp,
                    update_center=True,
                )

            if scaler is not None:
                scaler.scale(loss).backward()
                if float(train_cfg.get("CLIP_GRAD_NORM", 0.0)) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        unwrap_model(student).parameters(),
                        max_norm=float(train_cfg["CLIP_GRAD_NORM"]),
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if float(train_cfg.get("CLIP_GRAD_NORM", 0.0)) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        unwrap_model(student).parameters(),
                        max_norm=float(train_cfg["CLIP_GRAD_NORM"]),
                    )
                optimizer.step()

            copy_ema_weights(student=unwrap_model(student), teacher=unwrap_model(teacher), momentum=teacher_momentum)

            loss_value = float(loss.detach().item())
            running_loss += loss_value
            running_t_ent += float(loss_stats["teacher_entropy"])
            running_s_ent += float(loss_stats["student_entropy"])
            global_step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss_value:.4f}", lr=f"{lr:.2e}", ttemp=f"{teacher_temp:.3f}")

            if rank == 0 and ((step + 1) % log_every == 0 or step == 0):
                logger.log(
                    f"ep={epoch:03d} step={step+1:04d}/{steps_per_epoch:04d} "
                    f"loss={loss_value:.4f} lr={lr:.2e} wd={wd:.3f} "
                    f"m={teacher_momentum:.5f} ttemp={teacher_temp:.4f} "
                    f"t_ent={loss_stats['teacher_entropy']:.3f} s_ent={loss_stats['student_entropy']:.3f}"
                )
                logger.log_json(
                    {
                        "epoch": epoch,
                        "step": step + 1,
                        "global_step": global_step,
                        "loss": loss_value,
                        "lr": lr,
                        "wd": wd,
                        "teacher_momentum": teacher_momentum,
                        "teacher_temp": teacher_temp,
                        "teacher_entropy": float(loss_stats["teacher_entropy"]),
                        "student_entropy": float(loss_stats["student_entropy"]),
                        "center_norm": float(loss_stats["center_norm"]),
                    }
                )
        pbar.close()

        local_epoch_loss = running_loss / max(1, steps_per_epoch)
        local_epoch_t_ent = running_t_ent / max(1, steps_per_epoch)
        local_epoch_s_ent = running_s_ent / max(1, steps_per_epoch)
        epoch_loss = dist_reduce_mean_scalar(local_epoch_loss, device)
        epoch_t_ent = dist_reduce_mean_scalar(local_epoch_t_ent, device)
        epoch_s_ent = dist_reduce_mean_scalar(local_epoch_s_ent, device)

        last_teacher_temp = teacher_temp_sched.value_at(max(0, global_step - 1))
        val_metrics = run_validation(
            student=student,
            teacher=teacher,
            criterion=criterion,
            loader=val_loader,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            n_global_crops=n_global,
            teacher_temp=last_teacher_temp,
            max_steps=val_max_steps,
        )

        elapsed = time.time() - epoch_start
        if rank == 0:
            logger.log(
                f"[epoch {epoch:03d}] train_loss={epoch_loss:.4f} "
                f"val_loss={val_metrics['val_loss']:.4f} "
                f"train_t_ent={epoch_t_ent:.3f} train_s_ent={epoch_s_ent:.3f} "
                f"time={elapsed/60.0:.1f}m"
            )
            logger.log_json(
                {
                    "epoch": epoch,
                    "train_loss": epoch_loss,
                    "val_loss": val_metrics["val_loss"],
                    "train_teacher_entropy": epoch_t_ent,
                    "train_student_entropy": epoch_s_ent,
                    "val_teacher_entropy": val_metrics["val_teacher_entropy"],
                    "val_student_entropy": val_metrics["val_student_entropy"],
                    "elapsed_sec": elapsed,
                }
            )

            save_checkpoint(
                run_dir / "checkpoint_latest.pth",
                student=student,
                teacher=teacher,
                optimizer=optimizer,
                scaler=scaler,
                criterion=criterion,
                cfg=cfg,
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
            )

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                save_checkpoint(
                    run_dir / "best.pth",
                    student=student,
                    teacher=teacher,
                    optimizer=optimizer,
                    scaler=scaler,
                    criterion=criterion,
                    cfg=cfg,
                    epoch=epoch,
                    global_step=global_step,
                    best_val_loss=best_val_loss,
                )
                logger.log(f"new best checkpoint saved: val_loss={best_val_loss:.6f}")

            if (epoch + 1) % save_every == 0:
                save_checkpoint(
                    run_dir / f"checkpoint_ep{epoch+1:03d}.pth",
                    student=student,
                    teacher=teacher,
                    optimizer=optimizer,
                    scaler=scaler,
                    criterion=criterion,
                    cfg=cfg,
                    epoch=epoch,
                    global_step=global_step,
                    best_val_loss=best_val_loss,
                )

        if distributed:
            dist.barrier()
        cleanup_memory(device=device)

    if rank == 0:
        logger.log("training completed")
        logger.log(f"best_val_loss={best_val_loss:.6f}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

