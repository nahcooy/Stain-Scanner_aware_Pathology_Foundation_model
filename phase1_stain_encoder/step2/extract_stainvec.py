#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 Step 2: Bag-level Stain Vector Learning — Stain Vector Extraction

Loads a trained StainVecStage2 checkpoint, then for each pos_key in each split
samples N_BAGS bags and computes stain vectors, saving a per-split .npy array
and a .pkl metadata file.

Output layout:
    {output_dir}/{domain}/{split}/stainvecs.npy   — float32 [N_keys * N_bags, embed_dim]
    {output_dir}/{domain}/{split}/metadata.pkl    — list[dict] with keys:
        pos_key, bag_idx, key_idx, device, stain

Usage:
    python extract_stainvec.py \\
        --config  configs/sm_step2.yaml \\
        --domain  sm \\
        --checkpoint /path/to/checkpoints/best_loss.pt \\
        --output_dir /path/to/stainvecs \\
        --device cuda:0 \\
        --n_bags 16 \\
        --bag_size 32 \\
        --batch_size 256 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import gc
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

from dataset import BagFeatureDataset
from model import StainVecStage2

SPLITS = ("train", "val", "internal_test", "unseen_test")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    ckpt_path: str | Path,
    cfg: Dict[str, Any],
    device: torch.device,
) -> StainVecStage2:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    # Merge checkpoint config with CLI config (CLI takes precedence for FEAT_DIM / MODEL)
    ckpt_cfg: Dict[str, Any] = ckpt.get("cfg", {})
    merged = {**ckpt_cfg, **cfg}
    # Ensure MODEL sub-dict is fully merged
    if "MODEL" in ckpt_cfg and "MODEL" in cfg:
        merged["MODEL"] = {**ckpt_cfg["MODEL"], **cfg["MODEL"]}
    elif "MODEL" in ckpt_cfg:
        merged["MODEL"] = ckpt_cfg["MODEL"]

    model = StainVecStage2(merged)
    raw_state: Dict[str, Any] = ckpt["model"]
    state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# pos_key → (device, stain) parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_pos_key(stem: str) -> Dict[str, str]:
    """
    Split a pos_key filename stem (device_stain) into device/stain strings.
    The original pos_key used || as separator; extract_features.py replaced
    non-word chars with _. We split on the first _ that divides two non-empty
    tokens.  For keys with no _ the whole stem is treated as the device.
    """
    parts = stem.split("_", 1)
    if len(parts) == 2:
        return {"device": parts[0], "stain": parts[1]}
    return {"device": stem, "stain": "unknown"}


# ─────────────────────────────────────────────────────────────────────────────
# Per-split extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_split(
    model: StainVecStage2,
    feat_dir: Path,
    split: str,
    output_dir: Path,
    device: torch.device,
    n_bags: int,
    bag_size: int,
    batch_size: int,
    seed: int,
    use_amp: bool,
) -> None:
    split_feat_dir = feat_dir / split
    if not split_feat_dir.is_dir():
        print(f"[skip] split={split}: feature dir not found ({split_feat_dir})")
        return

    npy_files = sorted(split_feat_dir.glob("*.npy"))
    if not npy_files:
        print(f"[skip] split={split}: no .npy files in {split_feat_dir}")
        return

    split_out = output_dir / split
    split_out.mkdir(parents=True, exist_ok=True)

    # Build a lightweight dataset for bag sampling — reuse BagFeatureDataset
    # but we need all n_bags per key in one pass.
    ds = BagFeatureDataset(
        feat_dir=feat_dir,
        split=split,
        bag_size=bag_size,
        bags_per_key=n_bags,
        seed=seed,
    )

    total = ds.n_keys * n_bags
    all_vecs: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []

    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    with tqdm(total=total, desc=f"extract {split}", unit="bag") as pbar:
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            bags_list: List[torch.Tensor] = []

            for idx in range(start, end):
                item = ds[idx]
                bags_list.append(item["bag"])
                key_idx = item["key_idx"]
                bag_idx = idx % n_bags
                parsed = _parse_pos_key(item["pos_key"])
                metadata.append({
                    "pos_key": item["pos_key"],
                    "key_idx": int(key_idx),
                    "bag_idx": int(bag_idx),
                    "device": parsed["device"],
                    "stain": parsed["stain"],
                })

            bags = torch.stack(bags_list).to(device, non_blocking=True)  # [B, N, D]

            with torch.autocast(
                device_type=device.type,
                enabled=use_amp,
                dtype=amp_dtype,
            ):
                z = model(bags)   # [B, embed_dim]

            all_vecs.append(z.float().cpu().numpy())
            pbar.update(end - start)
            del bags, z

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    stainvecs = np.concatenate(all_vecs, axis=0).astype(np.float32)
    np.save(str(split_out / "stainvecs.npy"), stainvecs)

    with (split_out / "metadata.pkl").open("wb") as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"[done] split={split} keys={ds.n_keys} bags_per_key={n_bags} "
        f"total={len(stainvecs)} -> {split_out}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 Step 2 — Stain vector extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config used during training")
    parser.add_argument("--domain", required=True, choices=["sm", "wsi"])
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint (e.g. best_loss.pt)")
    parser.add_argument("--output_dir", required=True, help="Root directory for output stain vectors")
    parser.add_argument("--device", default="cuda:0", help="Torch device string")
    parser.add_argument("--n_bags", default=16, type=int, help="Number of bags to sample per pos_key")
    parser.add_argument("--bag_size", default=32, type=int, help="Patches per bag")
    parser.add_argument("--batch_size", default=256, type=int, help="Bags per inference batch")
    parser.add_argument(
        "--splits",
        default=",".join(SPLITS),
        help="Comma-separated list of splits to process",
    )
    parser.add_argument("--seed", default=42, type=int, help="RNG seed for bag sampling")
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable bfloat16 autocast (use float32 throughout)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {args.device} but CUDA is unavailable.")
    device = torch.device(args.device)

    feat_dir = Path(cfg["FEAT_DIR"]) / args.domain
    output_dir = Path(args.output_dir) / args.domain
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    use_amp = not args.no_amp

    print(f"[extract] domain={args.domain}  device={device}")
    print(f"[extract] checkpoint={args.checkpoint}")
    print(f"[extract] feat_dir={feat_dir}")
    print(f"[extract] output_dir={output_dir}")
    print(f"[extract] splits={splits}")
    print(f"[extract] n_bags={args.n_bags}  bag_size={args.bag_size}  batch_size={args.batch_size}")
    print(f"[extract] seed={args.seed}  amp={use_amp}")

    model = load_model(Path(args.checkpoint), cfg, device)
    print(f"[extract] model loaded: agg_type={model.agg_type}")

    for split in splits:
        print(f"\n[extract] processing split={split}")
        extract_split(
            model=model,
            feat_dir=feat_dir,
            split=split,
            output_dir=output_dir,
            device=device,
            n_bags=args.n_bags,
            bag_size=args.bag_size,
            batch_size=args.batch_size,
            seed=args.seed,
            use_amp=use_amp,
        )

    print(f"\n[extract] all splits complete. Stain vectors saved to: {output_dir}")


if __name__ == "__main__":
    main()
