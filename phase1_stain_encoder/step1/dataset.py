#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 Step 1: Patch-level Supervised Contrastive Learning — Datasets

Two dataset classes:
  PLISMSMDataset   — SM domain: large npy array + pandas metadata
  PLISMWSIDataset  — WSI domain: individual .npy patch files + storage manifest

Both return two independently augmented views of a patch from a given pos_key.

Sampler:
  RepeatKeySampler — for each epoch, samples `total_samples` indices, each drawn
  uniformly from the full dataset but ensuring diversity across pos_keys.

Transform:
  TwoViewTransform — staged albumentations pipeline (stage 1-4).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


# ─────────────────────────────────────────────────────────────────────────────
# Transform
# ─────────────────────────────────────────────────────────────────────────────

class TwoViewTransform:
    """
    Staged albumentations augmentation pipeline.

    Stage 1: geometric only (RandomResizedCrop, Flip, Rotate90)
    Stage 2: + elastic / grid / optical distortion
    Stage 3: + blur / noise (Gaussian, Motion, ISONoise)
    Stage 4: + color jitter (ColorJitter)

    Input:  np.ndarray [H, W, C] uint8 or float32 [0, 1]
    Output: torch.Tensor [C, H, W] float32 [0, 1]
    """

    def __init__(self, stage: int, image_size: int = 256) -> None:
        if stage not in range(5):
            raise ValueError(f"stage must be in 0..4, got {stage}")
        self.stage = stage
        self.image_size = image_size
        self._pipeline = self._build(stage, image_size)

    @staticmethod
    def _geometric_ops(h: int, w: int) -> List[Any]:
        return [
            A.OneOf(
                [
                    A.RandomResizedCrop(size=(h, w), scale=(0.7, 1.0), ratio=(0.85, 1.15), p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ],
                p=1.0,
            )
        ]

    @staticmethod
    def _distortion_ops() -> List[Any]:
        return [
            A.OneOf(
                [
                    A.ElasticTransform(
                        alpha=300.0, sigma=10.0,
                        interpolation=cv2.INTER_LINEAR,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=0.4,
                    ),
                    A.GridDistortion(
                        num_steps=5, distort_limit=0.3,
                        interpolation=cv2.INTER_LINEAR,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=0.4,
                    ),
                    A.OpticalDistortion(
                        distort_limit=0.3,
                        interpolation=cv2.INTER_LINEAR,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=0.4,
                    ),
                ],
                p=1.0,
            )
        ]

    @staticmethod
    def _blur_noise_ops() -> List[Any]:
        return [
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 9), p=0.4),
                    A.MotionBlur(blur_limit=(3, 11), p=0.2),
                    A.GaussNoise(std_range=(0.01, 0.1), mean_range=(0.0, 0.0), p=0.4),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.4), p=0.2),
                ],
                p=1.0,
            )
        ]

    @staticmethod
    def _color_ops() -> List[Any]:
        return [
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03, p=0.4)
        ]

    @classmethod
    def _build(cls, stage: int, image_size: int) -> A.Compose:
        ops: List[Any] = []
        if stage >= 1:
            ops.extend(cls._geometric_ops(image_size, image_size))
        if stage >= 2:
            ops.extend(cls._distortion_ops())
        if stage >= 3:
            ops.extend(cls._blur_noise_ops())
        if stage >= 4:
            ops.extend(cls._color_ops())
        return A.Compose(ops)

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        aug = self._pipeline(image=img)["image"]
        arr = np.array(aug, copy=True)
        t = torch.from_numpy(arr).permute(2, 0, 1).float()
        # support both uint8 and pre-normalised float32 inputs
        if t.max().item() > 1.5:
            t = t.div_(255.0)
        else:
            t = t.clamp_(0.0, 1.0)
        return t


# ─────────────────────────────────────────────────────────────────────────────
# SM Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PLISMSMDataset(Dataset):
    """
    SM domain patch dataset.

    Loads:
      - plism_sm_256_{split}_float32.npy  : [N, H, W, C] float32
      - plism_sm_{split}_meta.pkl         : DataFrame with columns [stain, device, ...]

    Indexed by individual patch (row index). __getitem__ returns two independently
    augmented views of the same patch.

    pos_key = device + "||" + stain

    Args:
        processed_dir: directory containing the .npy and .pkl files
        split:         "train" | "val" | "internal_test" | "unseen_test"
        transform:     TwoViewTransform (or compatible callable)
        mmap:          whether to use mmap_mode="r" (default True, saves RAM)
    """

    def __init__(
        self,
        processed_dir: str | Path,
        split: str,
        transform: TwoViewTransform,
        mmap: bool = True,
    ) -> None:
        processed_dir = Path(processed_dir)
        npy_path = processed_dir / f"plism_sm_256_{split}_float32.npy"
        meta_path = processed_dir / f"plism_sm_{split}_meta.pkl"

        for p in (npy_path, meta_path):
            if not p.is_file():
                raise FileNotFoundError(f"Required SM artifact not found: {p}")

        self.npy: np.ndarray = np.load(str(npy_path), mmap_mode="r" if mmap else None)
        self.meta: pd.DataFrame = pd.read_pickle(meta_path).reset_index(drop=True)

        if len(self.meta) != len(self.npy):
            raise ValueError(
                f"SM length mismatch split={split}: meta={len(self.meta)} npy={len(self.npy)}"
            )

        device_col = self.meta["device"].astype(str).str.strip()
        stain_col = self.meta["stain"].astype(str).str.strip()
        self.meta["pos_key"] = device_col + "||" + stain_col

        self.pos_key: np.ndarray = self.meta["pos_key"].to_numpy()
        self.transform = transform
        self.split = split

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img = np.array(self.npy[idx], copy=True)
        return {
            "view1": self.transform(img),
            "view2": self.transform(img),
            "pos_key": self.pos_key[idx],
            "idx": idx,
        }

    def key_index_map(self) -> Dict[str, List[int]]:
        """Returns {pos_key: [row_indices]} for all patches."""
        mapping: Dict[str, List[int]] = {}
        for i, key in enumerate(self.pos_key):
            mapping.setdefault(key, []).append(i)
        return mapping


# ─────────────────────────────────────────────────────────────────────────────
# WSI Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PLISMWSIDataset(Dataset):
    """
    WSI domain patch dataset.

    Loads:
      - {split}_meta.csv (or .pkl)            : shared metadata with pos_key / sample_id
      - {storage_mode}/{split}/{split}_storage_manifest.pkl : per-sample npy paths

    For npy_individual mode the manifest has columns [sample_id, individual_npy_path].
    For npy_shard mode the manifest has columns [sample_id, shard_path, local_idx].

    Auto-detects the storage format from the manifest columns.

    pos_key column is taken from the metadata CSV directly (already computed as
    device + "||" + stain during preprocessing).

    Args:
        processed_dir:  root of PLISM-wsi-processed/
        split:          "train" | "val" | "internal_test" | "unseen_test"
        transform:      TwoViewTransform (or compatible callable)
        storage_mode:   "npy_individual" | "npy_shard" (default "npy_individual")
    """

    def __init__(
        self,
        processed_dir: str | Path,
        split: str,
        transform: TwoViewTransform,
        storage_mode: str = "npy_individual",
    ) -> None:
        processed_dir = Path(processed_dir)
        meta_path = processed_dir / "shared" / f"{split}_meta.csv"
        manifest_path = processed_dir / storage_mode / split / f"{split}_storage_manifest.pkl"

        for p in (meta_path, manifest_path):
            if not p.is_file():
                raise FileNotFoundError(f"Required WSI artifact not found: {p}")

        meta_df = pd.read_csv(meta_path)
        manifest_df = pd.read_pickle(manifest_path).reset_index(drop=True)

        self.df = manifest_df.merge(
            meta_df[["sample_id", "pos_key", "stain", "device"]],
            on="sample_id",
            how="inner",
            validate="one_to_one",
        )
        if len(self.df) != len(manifest_df):
            raise ValueError(
                f"WSI manifest/meta merge mismatch split={split}: "
                f"manifest={len(manifest_df)} merged={len(self.df)}"
            )

        # Detect storage format from manifest columns
        if "individual_npy_path" in manifest_df.columns:
            self._mode = "individual"
        elif "shard_path" in manifest_df.columns and "local_idx" in manifest_df.columns:
            self._mode = "shard"
        else:
            raise ValueError(
                f"Cannot determine WSI storage mode from manifest columns: "
                f"{manifest_df.columns.tolist()}"
            )

        self.pos_key: np.ndarray = self.df["pos_key"].to_numpy()
        self.transform = transform
        self.split = split

        if self._mode == "shard":
            self._shard_cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _load_patch(self, idx: int) -> np.ndarray:
        if self._mode == "individual":
            path: str = self.df.iloc[idx]["individual_npy_path"]
            return np.load(path).astype(np.uint8)
        else:
            shard_path: str = str(self.df.iloc[idx]["shard_path"])
            local_idx: int = int(self.df.iloc[idx]["local_idx"])
            if shard_path not in self._shard_cache:
                self._shard_cache[shard_path] = np.load(shard_path, mmap_mode="r")
            return np.array(self._shard_cache[shard_path][local_idx], dtype=np.uint8)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img = self._load_patch(idx)
        return {
            "view1": self.transform(img),
            "view2": self.transform(img),
            "pos_key": self.pos_key[idx],
            "idx": idx,
        }

    def key_index_map(self) -> Dict[str, List[int]]:
        mapping: Dict[str, List[int]] = {}
        for i, key in enumerate(self.pos_key):
            mapping.setdefault(key, []).append(i)
        return mapping


# ─────────────────────────────────────────────────────────────────────────────
# Sampler
# ─────────────────────────────────────────────────────────────────────────────

class RepeatKeySampler(Sampler[int]):
    """
    Yields `total_samples` indices per epoch, sampling keys uniformly.

    For each draw: pick a random pos_key, then pick a random patch index from
    that key. This ensures each (device, stain) combination gets proportional
    representation regardless of class imbalance.

    Args:
        key_to_indices: mapping from pos_key → list of dataset row indices
        total_samples:  number of indices to yield per epoch
        seed:           base random seed (incremented per epoch via set_epoch)
    """

    def __init__(
        self,
        key_to_indices: Dict[str, List[int]],
        total_samples: int,
        seed: int = 0,
    ) -> None:
        self._keys: List[str] = list(key_to_indices.keys())
        self._indices: List[List[int]] = [key_to_indices[k] for k in self._keys]
        self.total_samples = total_samples
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self._epoch)
        n_keys = len(self._keys)
        for _ in range(self.total_samples):
            ki = rng.randrange(n_keys)
            yield rng.choice(self._indices[ki])

    def __len__(self) -> int:
        return self.total_samples


# ─────────────────────────────────────────────────────────────────────────────
# Collate & DataLoader helpers
# ─────────────────────────────────────────────────────────────────────────────

def _collate_two_view(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "view1": torch.stack([b["view1"] for b in batch]),
        "view2": torch.stack([b["view2"] for b in batch]),
        "pos_key": [b["pos_key"] for b in batch],
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
    }


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: Optional[int] = 4,
    sampler: Optional[Sampler] = None,
    seed: Optional[int] = None,
) -> DataLoader:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=_collate_two_view,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
