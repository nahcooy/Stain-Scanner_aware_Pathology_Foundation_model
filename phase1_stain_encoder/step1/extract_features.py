#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 Step 1: Patch-level Supervised Contrastive Learning — Feature Extraction

Loads a trained PatchEncoder checkpoint, strips the projection head, and runs
the backbone over all splits to produce per-pos_key feature files.

Output layout:
    {output_dir}/{domain}/{split}/{pos_key}.npy  — float32 [N_patches, 1024]

Usage:
    python extract_features.py \\
        --config configs/sm_stage4.yaml \\
        --domain sm \\
        --checkpoint /path/to/best_loss.pt \\
        --output_dir /path/to/features \\
        --device cuda:3 \\
        --batch_size 512
"""

from __future__ import annotations

import argparse
import gc
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

SPLITS = ("train", "val", "internal_test", "unseen_test")
FEAT_DIM = 1024  # ViT-L num_features
TIMM_NAME = "vit_large_patch16_224.augreg_in21k"


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def load_backbone(ckpt_path: str | Path, device: torch.device, image_size: int = 256) -> nn.Module:
    """
    Load backbone weights from a PatchEncoder checkpoint.
    Handles torch.compile prefix (_orig_mod.) and projector key stripping.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    raw_state: Dict[str, Any] = ckpt["model"]

    # Strip torch.compile wrapper prefix
    state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}

    # Extract backbone-only keys
    bb_state = {
        k[len("backbone."):]: v
        for k, v in state.items()
        if k.startswith("backbone.")
    }

    backbone = timm.create_model(
        TIMM_NAME,
        pretrained=False,
        num_classes=0,
        img_size=image_size,
    )
    missing, unexpected = backbone.load_state_dict(bb_state, strict=True)
    if missing:
        raise RuntimeError(f"Missing keys in backbone state_dict: {missing}")
    backbone.eval().to(device)
    return backbone


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_key_to_filename(key: str) -> str:
    """Convert pos_key (e.g. 'GT450||GIV') to a filesystem-safe filename."""
    return re.sub(r"[^\w\-]", "_", key)


@torch.no_grad()
def extract_batch(
    backbone: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    channels_last: bool = True,
) -> np.ndarray:
    """images: [B, C, H, W] float32 → features: [B, FEAT_DIM] float32"""
    x = images.to(device, non_blocking=True)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        feats = backbone(x)
    return feats.float().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# SM extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_sm_split(
    backbone: nn.Module,
    processed_dir: Path,
    split: str,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    channels_last: bool,
) -> None:
    npy_path = processed_dir / f"plism_sm_256_{split}_float32.npy"
    meta_path = processed_dir / f"plism_sm_{split}_meta.pkl"

    if not npy_path.is_file() or not meta_path.is_file():
        print(f"[skip] SM split={split}: missing files ({npy_path.name} or {meta_path.name})")
        return

    npy: np.ndarray = np.load(str(npy_path), mmap_mode="r")
    meta: pd.DataFrame = pd.read_pickle(meta_path).reset_index(drop=True)

    if len(meta) != len(npy):
        raise ValueError(f"SM length mismatch split={split}: meta={len(meta)} npy={len(npy)}")

    device_col = meta["device"].astype(str).str.strip()
    stain_col = meta["stain"].astype(str).str.strip()
    meta["pos_key"] = device_col + "||" + stain_col

    # Group row indices by pos_key
    key_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, key in enumerate(meta["pos_key"].to_numpy()):
        key_to_indices[key].append(i)

    split_out = output_dir / split
    split_out.mkdir(parents=True, exist_ok=True)

    for key, indices in tqdm(key_to_indices.items(), desc=f"SM {split}", unit="key"):
        fname = split_out / f"{_safe_key_to_filename(key)}.npy"
        if fname.is_file():
            continue

        # Batch inference
        all_feats: List[np.ndarray] = []
        for start in range(0, len(indices), batch_size):
            chunk_idx = indices[start : start + batch_size]
            # SM images are float32 [0,1]; shape [B, H, W, C] → [B, C, H, W]
            imgs_np = np.array([npy[i] for i in chunk_idx], dtype=np.float32)
            imgs = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).clamp_(0.0, 1.0)
            all_feats.append(extract_batch(backbone, imgs, device, channels_last))

        feats = np.concatenate(all_feats, axis=0).astype(np.float32)  # [N, 1024]
        np.save(str(fname), feats)

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"[done] SM split={split} keys={len(key_to_indices)} -> {split_out}")


# ─────────────────────────────────────────────────────────────────────────────
# WSI extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_wsi_split(
    backbone: nn.Module,
    processed_dir: Path,
    split: str,
    storage_mode: str,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    channels_last: bool,
) -> None:
    meta_path = processed_dir / "shared" / f"{split}_meta.csv"
    manifest_path = processed_dir / storage_mode / split / f"{split}_storage_manifest.pkl"

    if not meta_path.is_file() or not manifest_path.is_file():
        print(f"[skip] WSI split={split}: missing files")
        return

    meta_df = pd.read_csv(meta_path)
    manifest_df = pd.read_pickle(manifest_path).reset_index(drop=True)

    df = manifest_df.merge(
        meta_df[["sample_id", "pos_key", "stain", "device"]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )

    # Detect storage mode from manifest columns
    if "individual_npy_path" in manifest_df.columns:
        mode = "individual"
    elif "shard_path" in manifest_df.columns:
        mode = "shard"
    else:
        raise ValueError(f"Cannot detect WSI storage mode from columns: {manifest_df.columns.tolist()}")

    key_to_rows: Dict[str, List[int]] = defaultdict(list)
    for i, key in enumerate(df["pos_key"].to_numpy()):
        key_to_rows[key].append(i)

    split_out = output_dir / split
    split_out.mkdir(parents=True, exist_ok=True)

    shard_cache: Dict[str, np.ndarray] = {}

    def load_patch(row_idx: int) -> np.ndarray:
        row = df.iloc[row_idx]
        if mode == "individual":
            return np.load(str(row["individual_npy_path"])).astype(np.uint8)
        else:
            sp = str(row["shard_path"])
            if sp not in shard_cache:
                shard_cache[sp] = np.load(sp, mmap_mode="r")
            return np.array(shard_cache[sp][int(row["local_idx"])], dtype=np.uint8)

    for key, row_indices in tqdm(key_to_rows.items(), desc=f"WSI {split}", unit="key"):
        fname = split_out / f"{_safe_key_to_filename(key)}.npy"
        if fname.is_file():
            continue

        all_feats: List[np.ndarray] = []
        for start in range(0, len(row_indices), batch_size):
            chunk = row_indices[start : start + batch_size]
            imgs_np = np.stack([load_patch(i) for i in chunk])  # [B, H, W, C] uint8
            imgs = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).float().div_(255.0)
            all_feats.append(extract_batch(backbone, imgs, device, channels_last))

        feats = np.concatenate(all_feats, axis=0).astype(np.float32)
        np.save(str(fname), feats)

    shard_cache.clear()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"[done] WSI split={split} keys={len(key_to_rows)} -> {split_out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 Step 1 — Backbone feature extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config used during training")
    parser.add_argument("--domain", required=True, choices=["sm", "wsi"])
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint (e.g. best_loss.pt)")
    parser.add_argument("--output_dir", required=True, help="Root directory for output features")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--batch_size", default=512, type=int, help="Inference batch size")
    parser.add_argument(
        "--splits",
        default=",".join(SPLITS),
        help="Comma-separated list of splits to process",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {args.device} but CUDA is unavailable.")
    device = torch.device(args.device)

    image_size: int = int(cfg.get("IMAGE_SIZE", 256))
    channels_last: bool = bool(cfg.get("MODEL", {}).get("CHANNELS_LAST", True))
    processed_dir = Path(cfg["PROCESSED_DIR"])
    storage_mode: str = cfg.get("STORAGE_MODE", "npy_individual")

    domain_out = Path(args.output_dir) / args.domain
    domain_out.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    print(f"[extract] domain={args.domain}  device={device}  batch_size={args.batch_size}")
    print(f"[extract] checkpoint={args.checkpoint}")
    print(f"[extract] processed_dir={processed_dir}")
    print(f"[extract] output_dir={domain_out}")
    print(f"[extract] splits={splits}")

    backbone = load_backbone(args.checkpoint, device, image_size)
    if channels_last:
        backbone = backbone.to(memory_format=torch.channels_last)
    backbone.eval()

    for split in splits:
        print(f"\n[extract] processing split={split}")
        if args.domain == "sm":
            extract_sm_split(
                backbone=backbone,
                processed_dir=processed_dir,
                split=split,
                output_dir=domain_out,
                device=device,
                batch_size=args.batch_size,
                channels_last=channels_last,
            )
        else:
            extract_wsi_split(
                backbone=backbone,
                processed_dir=processed_dir,
                split=split,
                storage_mode=storage_mode,
                output_dir=domain_out,
                device=device,
                batch_size=args.batch_size,
                channels_last=channels_last,
            )

    print(f"\n[extract] all splits complete. Features saved to: {domain_out}")


if __name__ == "__main__":
    main()
