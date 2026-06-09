#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PNG patch datasets for Phase 2 Stain-Aware SSL."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def discover_images(
    root_dir: str | Path,
    recursive: bool = True,
    extensions: Optional[Iterable[str]] = None,
) -> List[Path]:
    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Patch directory not found: {root}")

    exts = tuple(str(x).lower() for x in (extensions or DEFAULT_EXTENSIONS))
    iterator = root.rglob("*") if recursive else root.glob("*")
    paths = [p for p in iterator if p.is_file() and p.suffix.lower() in exts]
    paths.sort()
    if not paths:
        raise RuntimeError(f"No image patches found in {root} (extensions={exts})")
    return paths


def _resolve_csv_path(raw_path: str, base_dir: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _build_stain_index_from_csv(
    csv_path: str | Path,
    image_root: str | Path,
) -> Dict[str, Path]:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"stain index csv not found: {csv_path}")
    image_root = Path(image_root).resolve()

    img_cols = ("image_path", "patch_path", "path", "relative_path")
    vec_cols = ("stain_vector_path", "vector_path", "stain_path")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        fields = {k.strip() for k in reader.fieldnames}

        image_col = next((c for c in img_cols if c in fields), None)
        vec_col = next((c for c in vec_cols if c in fields), None)
        if image_col is None or vec_col is None:
            raise ValueError(
                f"CSV must contain one image column {img_cols} and one vector column {vec_cols}. "
                f"Got fields={sorted(fields)}"
            )

        mapping: Dict[str, Path] = {}
        for row in reader:
            img_raw = str(row[image_col]).strip()
            vec_raw = str(row[vec_col]).strip()
            img_path = _resolve_csv_path(img_raw, csv_path.parent)
            vec_path = _resolve_csv_path(vec_raw, csv_path.parent)

            if not img_path.exists():
                maybe = (image_root / img_raw).resolve()
                if maybe.exists():
                    img_path = maybe
            mapping[str(img_path.resolve())] = vec_path.resolve()
        return mapping


def _vector_path_from_mirror_root(image_path: Path, image_root: Path, vec_root: Path) -> Path:
    rel = image_path.resolve().relative_to(image_root.resolve())
    return (vec_root / rel).with_suffix(".npy")


class PatchMultiCropDataset(Dataset):
    """Dataset returning multi-crop views + stain vectors for SSL training."""

    def __init__(
        self,
        root_dir: str | Path,
        transform: Any,
        *,
        stain_dim: int,
        stain_vector_dir: Optional[str | Path] = None,
        stain_index_csv: Optional[str | Path] = None,
        require_stain_vector: bool = True,
        recursive: bool = True,
        extensions: Optional[Iterable[str]] = None,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.paths = discover_images(self.root_dir, recursive=recursive, extensions=extensions)
        self.transform = transform
        self.stain_dim = int(stain_dim)
        self.require_stain_vector = bool(require_stain_vector)
        self._zero_stain = np.zeros((self.stain_dim,), dtype=np.float32)

        self.stain_vector_dir = Path(stain_vector_dir).resolve() if stain_vector_dir else None
        self.stain_index = (
            _build_stain_index_from_csv(stain_index_csv, self.root_dir) if stain_index_csv else {}
        )

    def __len__(self) -> int:
        return len(self.paths)

    def _resolve_stain_vec_path(self, img_path: Path) -> Optional[Path]:
        key = str(img_path.resolve())
        if key in self.stain_index:
            return self.stain_index[key]
        if self.stain_vector_dir is not None:
            return _vector_path_from_mirror_root(img_path, self.root_dir, self.stain_vector_dir)
        return None

    def _load_stain_vec(self, img_path: Path) -> torch.Tensor:
        vec_path = self._resolve_stain_vec_path(img_path)
        if vec_path is None:
            if self.require_stain_vector:
                raise FileNotFoundError(f"Stain vector mapping not configured for image: {img_path}")
            return torch.from_numpy(self._zero_stain.copy())

        if not vec_path.is_file():
            if self.require_stain_vector:
                raise FileNotFoundError(f"Stain vector not found: {vec_path}")
            return torch.from_numpy(self._zero_stain.copy())

        vec = np.load(vec_path)
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.stain_dim:
            raise ValueError(
                f"Stain vector dim mismatch for {vec_path}: got {vec.shape[0]}, expected {self.stain_dim}"
            )
        return torch.from_numpy(vec)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        views = self.transform(img)
        if not isinstance(views, list) or not views:
            raise TypeError("transform must return a non-empty list[Tensor] for SSL training")
        return {"views": views, "stain_vec": self._load_stain_vec(img_path), "path": str(img_path)}


class PatchEvalDataset(Dataset):
    """Single-view deterministic dataset for feature extraction/evaluation."""

    def __init__(
        self,
        root_dir: str | Path,
        transform: Any,
        *,
        stain_dim: int,
        stain_vector_dir: Optional[str | Path] = None,
        stain_index_csv: Optional[str | Path] = None,
        require_stain_vector: bool = True,
        recursive: bool = True,
        extensions: Optional[Iterable[str]] = None,
        label_mode: str = "parent_dir",
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.paths = discover_images(self.root_dir, recursive=recursive, extensions=extensions)
        self.transform = transform
        self.label_mode = str(label_mode).lower()

        self._ssl_dataset = PatchMultiCropDataset(
            root_dir=self.root_dir,
            transform=lambda x: [x],
            stain_dim=stain_dim,
            stain_vector_dir=stain_vector_dir,
            stain_index_csv=stain_index_csv,
            require_stain_vector=require_stain_vector,
            recursive=recursive,
            extensions=extensions,
        )
        self._ssl_dataset.paths = self.paths

    def __len__(self) -> int:
        return len(self.paths)

    def _infer_label(self, path: Path) -> str:
        if self.label_mode == "parent_dir":
            return path.parent.name
        if self.label_mode == "none":
            return ""
        return path.parent.name

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(img)
        if not torch.is_tensor(image_tensor):
            raise TypeError("eval transform must return torch.Tensor")
        return {
            "image": image_tensor,
            "stain_vec": self._ssl_dataset._load_stain_vec(img_path),
            "path": str(img_path),
            "label": self._infer_label(img_path),
        }


def collate_multicrop(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_views = len(batch[0]["views"])
    views = [torch.stack([sample["views"][i] for sample in batch], dim=0) for i in range(n_views)]
    stain_vec = torch.stack([sample["stain_vec"] for sample in batch], dim=0).float()
    paths = [sample["path"] for sample in batch]
    return {"views": views, "stain_vec": stain_vec, "paths": paths}


def collate_eval(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([sample["image"] for sample in batch], dim=0)
    stain_vec = torch.stack([sample["stain_vec"] for sample in batch], dim=0).float()
    paths = [sample["path"] for sample in batch]
    labels = [sample["label"] for sample in batch]
    return {"images": images, "stain_vec": stain_vec, "paths": paths, "labels": labels}


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    shuffle: bool,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    collate_fn: Any = collate_multicrop,
    sampler: Optional[DistributedSampler] = None,
) -> DataLoader:
    kwargs: Dict[str, Any] = {}
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)
        kwargs["persistent_workers"] = bool(persistent_workers)

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=bool(drop_last),
        collate_fn=collate_fn,
        **kwargs,
    )

