#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2 evaluation: embedding extraction + kNN / linear probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

from ssl.augmentations import build_eval_transform
from ssl.dataset import PatchEvalDataset, collate_eval, make_loader
from ssl.model import StainAwareDINO


def _extract_embeddings(
    model: StainAwareDINO,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> Tuple[np.ndarray, list[str], list[str]]:
    model.eval()
    embs: list[np.ndarray] = []
    labels: list[str] = []
    paths: list[str] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extract", dynamic_ncols=True):
            images = batch["images"].to(device, non_blocking=True)
            stain_vec = batch["stain_vec"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                feat = model.encode(images, stain_vec)
            embs.append(feat.detach().cpu().numpy())
            labels.extend(batch["labels"])
            paths.extend(batch["paths"])

    return np.concatenate(embs, axis=0), labels, paths


def _compute_metrics(
    train_emb: np.ndarray,
    train_labels: list[str],
    val_emb: np.ndarray,
    val_labels: list[str],
    knn_k: int,
    linear_probe_c: float,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if len(set(train_labels)) <= 1 or len(set(val_labels)) <= 1:
        metrics["warning"] = "Not enough class diversity for classification metrics."
        return metrics

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover
        metrics["warning"] = f"scikit-learn not available: {exc}"
        return metrics

    knn = KNeighborsClassifier(n_neighbors=int(knn_k), metric="cosine", weights="distance")
    knn.fit(train_emb, train_labels)
    pred_knn = knn.predict(val_emb)
    metrics["knn_top1"] = float(accuracy_score(val_labels, pred_knn))
    metrics["knn_balanced_acc"] = float(balanced_accuracy_score(val_labels, pred_knn))

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(linear_probe_c),
            max_iter=3000,
            n_jobs=-1,
            multi_class="auto",
        ),
    )
    clf.fit(train_emb, train_labels)
    pred_lp = clf.predict(val_emb)
    metrics["linear_probe_acc"] = float(accuracy_score(val_labels, pred_lp))
    metrics["linear_probe_balanced_acc"] = float(balanced_accuracy_score(val_labels, pred_lp))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_dir", type=Path, default=None)
    parser.add_argument("--val_dir", type=Path, default=None)
    args = parser.parse_args()

    cfg: Dict[str, Any] = yaml.safe_load(args.config.read_text())
    device = torch.device(str(cfg.get("DEVICE", "cuda:0")) if torch.cuda.is_available() else "cpu")
    amp_cfg = cfg.get("AMP", {})
    amp_enabled = bool(amp_cfg.get("USE_AMP", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(amp_cfg.get("DTYPE", "bf16")).lower() == "bf16" else torch.float16

    model = StainAwareDINO(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["student"], strict=True)

    data_cfg = cfg["DATA"]
    eval_cfg = cfg.get("EVAL", {})
    train_dir = args.train_dir if args.train_dir is not None else Path(data_cfg["TRAIN_DIR"])
    val_dir = args.val_dir if args.val_dir is not None else Path(data_cfg["VAL_DIR"])

    transform = build_eval_transform(cfg)
    train_set = PatchEvalDataset(
        root_dir=train_dir,
        transform=transform,
        stain_dim=int(data_cfg["STAIN_DIM"]),
        stain_vector_dir=data_cfg.get("STAIN_VECTOR_DIR"),
        stain_index_csv=data_cfg.get("STAIN_INDEX_CSV"),
        require_stain_vector=bool(data_cfg.get("REQUIRE_STAIN_VECTOR", True)),
        recursive=bool(data_cfg.get("RECURSIVE", True)),
        extensions=data_cfg.get("EXTENSIONS"),
        label_mode=str(eval_cfg.get("LABEL_MODE", "parent_dir")),
    )
    val_set = PatchEvalDataset(
        root_dir=val_dir,
        transform=transform,
        stain_dim=int(data_cfg["STAIN_DIM"]),
        stain_vector_dir=data_cfg.get("STAIN_VECTOR_DIR"),
        stain_index_csv=data_cfg.get("STAIN_INDEX_CSV"),
        require_stain_vector=bool(data_cfg.get("REQUIRE_STAIN_VECTOR", True)),
        recursive=bool(data_cfg.get("RECURSIVE", True)),
        extensions=data_cfg.get("EXTENSIONS"),
        label_mode=str(eval_cfg.get("LABEL_MODE", "parent_dir")),
    )

    batch_size = int(eval_cfg.get("BATCH_SIZE", 256))
    num_workers = int(eval_cfg.get("NUM_WORKERS", 8))
    pin_memory = bool(eval_cfg.get("PIN_MEMORY", True))
    persistent = bool(eval_cfg.get("PERSISTENT_WORKERS", False))
    prefetch = int(eval_cfg.get("PREFETCH_FACTOR", 2))

    train_loader = make_loader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        shuffle=False,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        collate_fn=collate_eval,
    )
    val_loader = make_loader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        shuffle=False,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        collate_fn=collate_eval,
    )

    train_emb, train_labels, train_paths = _extract_embeddings(
        model=model,
        loader=train_loader,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    val_emb, val_labels, val_paths = _extract_embeddings(
        model=model,
        loader=val_loader,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )

    metrics = _compute_metrics(
        train_emb=train_emb,
        train_labels=train_labels,
        val_emb=val_emb,
        val_labels=val_labels,
        knn_k=int(eval_cfg.get("KNN_K", 20)),
        linear_probe_c=float(eval_cfg.get("LINEAR_PROBE_C", 1.0)),
    )
    metrics["n_train"] = int(train_emb.shape[0])
    metrics["n_val"] = int(val_emb.shape[0])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "embeddings_train.npz",
        emb=train_emb,
        labels=np.array(train_labels, dtype=object),
        paths=np.array(train_paths, dtype=object),
    )
    np.savez_compressed(
        args.output_dir / "embeddings_val.npz",
        emb=val_emb,
        labels=np.array(val_labels, dtype=object),
        paths=np.array(val_paths, dtype=object),
    )
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

