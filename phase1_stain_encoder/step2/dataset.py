#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 Step 2: Bag-level Stain Vector Learning — Dataset

BagFeatureDataset:
  Loads all pre-extracted per-pos_key .npy feature files from
  features/{domain}/{split}/ into memory at construction, then samples
  fixed-size bags on-the-fly.

  Feature files were written by Step 1 extract_features.py with the
  convention that || in pos_key is replaced by _ in the filename, e.g.:
      GT450||GIV  →  GT450_GIV.npy

  Dataset length = n_keys * bags_per_key.
  __getitem__ for logical index i returns the (i // bags_per_key)-th key's
  (i % bags_per_key)-th bag, sampled with replacement if N < bag_size.

RepeatStepSampler:
  Yields BATCH_KEYS * STEPS_PER_EPOCH key indices per epoch.
  Each draw picks one key uniformly at random; the DataLoader collects
  BATCH_KEYS such draws to form a step batch (via batch_size=BATCH_KEYS in
  the DataLoader).  bag_idx within each key is determined by __getitem__,
  so multiple bags per key are assembled at collation by the caller
  sampling bags_per_key times the same key.

collate_fn:
  Stacks bags into [B, bag_size, feat_dim] and collects pos_keys.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def _filename_to_pos_key(stem: str) -> str:
    r"""
    Reverse the _safe_key_to_filename transform from extract_features.py.
    The only replacement is || -> _  (re.sub(r"[^\w\-]", "_", key)).
    We restore the first _ that follows a device token as ||.
    Since the pattern is always DEVICE_STAIN (one separator), we find the
    first _ and treat everything before/after as device/stain.
    """
    return stem


def _safe_key_to_filename(key: str) -> str:
    return re.sub(r"[^\w\-]", "_", key)


class BagFeatureDataset(Dataset):
    """
    In-memory bag sampling dataset over pre-extracted patch features.

    Args:
        feat_dir:     root features directory (e.g. features/sm/)
        split:        "train" | "val" | "internal_test" | "unseen_test"
        bag_size:     number of patches per bag
        bags_per_key: virtual number of bags per key (controls dataset length)
        seed:         base RNG seed; per-item seed derived as seed + key_idx * bags_per_key + bag_idx
    """

    def __init__(
        self,
        feat_dir: str | Path,
        split: str,
        bag_size: int,
        bags_per_key: int,
        seed: int = 0,
    ) -> None:
        feat_dir = Path(feat_dir)
        split_dir = feat_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Feature split directory not found: {split_dir}")

        npy_files = sorted(split_dir.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy feature files found in: {split_dir}")

        self.pos_keys: List[str] = []
        self.features: List[np.ndarray] = []

        for p in npy_files:
            arr = np.load(str(p))
            if arr.ndim != 2:
                raise ValueError(
                    f"Expected 2-D feature array [N, D], got shape {arr.shape} in {p}"
                )
            self.features.append(arr.astype(np.float32))
            # Reconstruct pos_key: stem uses _ as separator (written by extract_features)
            self.pos_keys.append(p.stem)

        self.bag_size = bag_size
        self.bags_per_key = bags_per_key
        self.seed = seed
        self.split = split
        self.n_keys: int = len(self.pos_keys)

    def __len__(self) -> int:
        return self.n_keys * self.bags_per_key

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        key_idx = idx // self.bags_per_key
        bag_idx = idx % self.bags_per_key
        feats = self.features[key_idx]   # [N, D]
        n_patches = len(feats)

        rng = np.random.default_rng(self.seed + key_idx * self.bags_per_key + bag_idx)

        if n_patches >= self.bag_size:
            chosen = rng.choice(n_patches, size=self.bag_size, replace=False)
        else:
            chosen = rng.choice(n_patches, size=self.bag_size, replace=True)

        bag = torch.from_numpy(feats[chosen])   # [bag_size, D]
        return {
            "bag": bag,
            "pos_key": self.pos_keys[key_idx],
            "key_idx": key_idx,
        }

    def key_index_map(self) -> Dict[str, int]:
        """Returns {pos_key: key_idx} for all keys."""
        return {k: i for i, k in enumerate(self.pos_keys)}


# ─────────────────────────────────────────────────────────────────────────────
# Sampler
# ─────────────────────────────────────────────────────────────────────────────

class RepeatStepSampler(Sampler[int]):
    """
    Yields BATCH_KEYS * STEPS_PER_EPOCH dataset indices per epoch.

    Each draw selects a random key, then a random bag_idx for that key,
    producing a dataset index = key_idx * bags_per_key + bag_idx.

    The DataLoader is configured with batch_size=BATCH_KEYS; each step
    therefore processes BATCH_KEYS distinct key draws.  Since bags_per_key
    virtual slots exist per key, repeated draws of the same key naturally
    produce different bags (different bag_idx offsets).

    Args:
        n_keys:           number of distinct pos_keys in the dataset
        bags_per_key:     virtual bags per key (must match BagFeatureDataset)
        batch_keys:       number of keys per training step
        steps_per_epoch:  training steps per epoch
        seed:             base RNG seed (incremented per epoch via set_epoch)
    """

    def __init__(
        self,
        n_keys: int,
        bags_per_key: int,
        batch_keys: int,
        steps_per_epoch: int,
        seed: int = 0,
    ) -> None:
        self.n_keys = n_keys
        self.bags_per_key = bags_per_key
        self.batch_keys = batch_keys
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self._epoch = 0

    @property
    def total_samples(self) -> int:
        return self.batch_keys * self.steps_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self._epoch)
        for _ in range(self.total_samples):
            key_idx = rng.randrange(self.n_keys)
            bag_idx = rng.randrange(self.bags_per_key)
            yield key_idx * self.bags_per_key + bag_idx

    def __len__(self) -> int:
        return self.total_samples


# ─────────────────────────────────────────────────────────────────────────────
# Collate & DataLoader helper
# ─────────────────────────────────────────────────────────────────────────────

def collate_bags(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate a list of bag samples into a batch.

    Returns:
        bags:     [B, bag_size, feat_dim] float32 tensor
        pos_keys: list[str] of length B
        key_idxs: [B] long tensor
    """
    return {
        "bags": torch.stack([b["bag"] for b in batch]),
        "pos_keys": [b["pos_key"] for b in batch],
        "key_idxs": torch.tensor([b["key_idx"] for b in batch], dtype=torch.long),
    }


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_bag_loader(
    dataset: BagFeatureDataset,
    batch_size: int,
    sampler: Optional[Sampler] = None,
    shuffle: bool = False,
    drop_last: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: Optional[int] = 4,
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
        collate_fn=collate_bags,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
