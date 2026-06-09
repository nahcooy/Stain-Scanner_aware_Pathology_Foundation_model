"""
run_eval.py — unified evaluation entry point for Phase 1 stain vectors.

Usage
-----
python eval/run_eval.py \\
    --stainvec_dir  /path/to/stainvec_output \\
    --output_dir    /path/to/eval_results \\
    --splits        val internal_test unseen_test \\
    --eval_pairs    val:internal_test val:unseen_test internal_test:unseen_test \\
    --skip_viz

Expected stainvec_dir layout
-----------------------------
<stainvec_dir>/
  <split>/
    stainvecs.npy    # np.ndarray [N, 256]  float32, L2-normalised
    metadata.pkl     # dict with keys:
                     #   "pos_key"  : list[str]  e.g. "AT2||HE"
                     #   "device"   : list[str]  e.g. "AT2"
                     #   "stain"    : list[str]  e.g. "HE"
                     #   (optional) "tissue": list[str]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

# local imports — adjust sys.path so this script is runnable from anywhere
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from metrics import clustering_metrics, knn_purity, linear_probe, retrieval_metrics
from visualize import compute_tsne, compute_umap, plot_embedding, plot_embedding_grid


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_split(stainvec_dir: Path, split: str) -> tuple[np.ndarray, dict[str, list]]:
    """Return (vecs, metadata) for a given split."""
    split_dir = stainvec_dir / split
    vecs_path = split_dir / "stainvecs.npy"
    meta_path = split_dir / "metadata.pkl"

    if not vecs_path.exists():
        raise FileNotFoundError(f"stainvecs.npy not found: {vecs_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.pkl not found: {meta_path}")

    vecs = np.load(vecs_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    print(f"  Loaded '{split}': {vecs.shape[0]} vectors, dim={vecs.shape[1]}")
    return vecs, meta


def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Per-split evaluation
# ---------------------------------------------------------------------------

def eval_single_split(
    split: str,
    vecs: np.ndarray,
    meta: dict[str, list],
) -> dict[str, Any]:
    """
    Run clustering, kNN purity, and (train == test) linear probe for one split.
    """
    results: dict[str, Any] = {}

    pos_keys = meta["pos_key"]
    devices  = meta["device"]
    stains   = meta["stain"]

    # --- clustering (pos_key) ------------------------------------------------
    print(f"  [{split}] clustering metrics (pos_key) …")
    results["clustering_pos_key"] = clustering_metrics(vecs, pos_keys)

    # --- kNN purity ----------------------------------------------------------
    print(f"  [{split}] kNN purity (pos_key) …")
    results["knn_pos_key"] = knn_purity(
        vecs, pos_keys, vecs, pos_keys, k_list=[1, 5, 10]
    )

    print(f"  [{split}] kNN purity (device) …")
    results["knn_device"] = knn_purity(
        vecs, devices, vecs, devices, k_list=[1, 5, 10]
    )

    print(f"  [{split}] kNN purity (stain) …")
    results["knn_stain"] = knn_purity(
        vecs, stains, vecs, stains, k_list=[1, 5, 10]
    )

    # --- linear probe (within-split, for completeness) ----------------------
    # Note: for proper generalisation, use cross-split probes below.
    print(f"  [{split}] linear probe within split (device) …")
    results["linear_probe_device_within"] = linear_probe(
        vecs, devices, vecs, devices
    )
    print(f"  [{split}] linear probe within split (stain) …")
    results["linear_probe_stain_within"] = linear_probe(
        vecs, stains, vecs, stains
    )

    return results


# ---------------------------------------------------------------------------
# Cross-split evaluation
# ---------------------------------------------------------------------------

def eval_cross_split(
    split_a: str,
    vecs_a: np.ndarray,
    meta_a: dict[str, list],
    split_b: str,
    vecs_b: np.ndarray,
    meta_b: dict[str, list],
) -> dict[str, Any]:
    """
    Run cross-split retrieval and cross-split linear probes.
    """
    results: dict[str, Any] = {}

    # --- cross-split retrieval (pos_key) ------------------------------------
    print(f"  [{split_a} → {split_b}] retrieval (pos_key) …")
    results[f"retrieval_{split_a}_to_{split_b}"] = retrieval_metrics(
        vecs_a, meta_a["pos_key"],
        vecs_b, meta_b["pos_key"],
        k_list=[1, 5, 10],
    )
    print(f"  [{split_b} → {split_a}] retrieval (pos_key) …")
    results[f"retrieval_{split_b}_to_{split_a}"] = retrieval_metrics(
        vecs_b, meta_b["pos_key"],
        vecs_a, meta_a["pos_key"],
        k_list=[1, 5, 10],
    )

    # --- cross-split linear probes ------------------------------------------
    for label_key in ("device", "stain"):
        print(f"  [{split_a}→{split_b}] linear probe ({label_key}) …")
        results[f"linear_probe_{label_key}_{split_a}_to_{split_b}"] = linear_probe(
            vecs_a, meta_a[label_key],
            vecs_b, meta_b[label_key],
        )
        print(f"  [{split_b}→{split_a}] linear probe ({label_key}) …")
        results[f"linear_probe_{label_key}_{split_b}_to_{split_a}"] = linear_probe(
            vecs_b, meta_b[label_key],
            vecs_a, meta_a[label_key],
        )

    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def run_visualisations(
    split: str,
    vecs: np.ndarray,
    meta: dict[str, list],
    output_dir: Path,
) -> None:
    """Generate UMAP and t-SNE plots coloured by pos_key, stain, and device."""
    plot_dir = output_dir / "plots" / split
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [{split}] computing UMAP …")
    umap_coords = compute_umap(vecs, seed=42)
    print(f"  [{split}] computing t-SNE …")
    tsne_coords = compute_tsne(vecs, seed=42)

    for label_key in ("pos_key", "stain", "device"):
        if label_key not in meta:
            continue
        labels = meta[label_key]

        # individual plots
        for method_name, coords in [("UMAP", umap_coords), ("tSNE", tsne_coords)]:
            plot_embedding(
                coords,
                labels,
                title=f"{method_name} — {split} ({label_key})",
                save_path=plot_dir / f"{method_name.lower()}_{label_key}.png",
                label_col=label_key,
            )

        # grid (UMAP + t-SNE side by side)
        plot_embedding_grid(
            {"UMAP": umap_coords, "t-SNE": tsne_coords},
            labels,
            title=f"Stain Vectors — {split} ({label_key})",
            save_path=plot_dir / f"grid_{label_key}.png",
            label_col=label_key,
        )

    print(f"  [{split}] plots saved to {plot_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1 stain-vector evaluation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--stainvec_dir",
        required=True,
        type=Path,
        help="Root directory containing per-split stainvec outputs.",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory where eval_results.json and plots will be saved.",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["val", "internal_test", "unseen_test"],
        help="Splits to evaluate.",
    )
    p.add_argument(
        "--eval_pairs",
        nargs="*",
        default=None,
        help=(
            "Cross-split pairs as 'A:B' strings.  "
            "Defaults to all combinations of --splits."
        ),
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Unused (kept for interface parity with training scripts).",
    )
    p.add_argument(
        "--skip_viz",
        action="store_true",
        help="Skip UMAP/t-SNE visualisation (faster for quick metric runs).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    stainvec_dir: Path = args.stainvec_dir
    output_dir: Path   = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ load
    print("\n=== Loading stain vectors ===")
    data: dict[str, tuple[np.ndarray, dict]] = {}
    for split in args.splits:
        try:
            data[split] = _load_split(stainvec_dir, split)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}  — skipping split '{split}'")

    if not data:
        print("ERROR: no splits could be loaded.  Exiting.")
        sys.exit(1)

    # ------------------------------------------------ cross-split pair list
    if args.eval_pairs is None:
        cross_pairs = list(combinations(list(data.keys()), 2))
    else:
        cross_pairs = []
        for pair_str in args.eval_pairs:
            a, b = pair_str.split(":")
            if a in data and b in data:
                cross_pairs.append((a, b))
            else:
                print(f"  WARNING: skipping pair '{pair_str}' — split(s) not loaded")

    # -------------------------------------------------------- per-split eval
    print("\n=== Per-split evaluation ===")
    all_results: dict[str, Any] = {}
    for split, (vecs, meta) in data.items():
        print(f"\n--- {split} ---")
        all_results[split] = eval_single_split(split, vecs, meta)

    # ------------------------------------------------------ cross-split eval
    print("\n=== Cross-split evaluation ===")
    cross_results: dict[str, Any] = {}
    for split_a, split_b in cross_pairs:
        print(f"\n--- {split_a} ↔ {split_b} ---")
        vecs_a, meta_a = data[split_a]
        vecs_b, meta_b = data[split_b]
        pair_key = f"{split_a}_x_{split_b}"
        cross_results[pair_key] = eval_cross_split(
            split_a, vecs_a, meta_a,
            split_b, vecs_b, meta_b,
        )

    # ------------------------------------------------------------ save JSON
    final = {"per_split": all_results, "cross_split": cross_results}
    _save_json(final, output_dir / "eval_results.json")

    # --------------------------------------------------------- visualisation
    if not args.skip_viz:
        print("\n=== Generating visualisations ===")
        for split, (vecs, meta) in data.items():
            print(f"\n--- {split} ---")
            run_visualisations(split, vecs, meta, output_dir)

    print(f"\n=== Evaluation complete — results in {output_dir} ===\n")


if __name__ == "__main__":
    main()
