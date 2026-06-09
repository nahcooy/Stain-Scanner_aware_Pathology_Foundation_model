"""
Dimensionality-reduction visualisation utilities for stain vectors.

Functions
---------
compute_umap      – UMAP 2-D projection
compute_tsne      – t-SNE 2-D projection
plot_embedding    – single scatter plot coloured by a label column
plot_embedding_grid – UMAP + t-SNE side-by-side multi-panel figure
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless backend — must be set before pyplot import

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.preprocessing import normalize


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def compute_umap(
    vecs: np.ndarray,
    n_components: int = 2,
    seed: int = 42,
    **umap_kwargs: Any,
) -> np.ndarray:
    """
    Compute UMAP embedding.

    Parameters
    ----------
    vecs         : np.ndarray [N, D]
    n_components : output dimensionality (default 2)
    seed         : random seed
    **umap_kwargs: forwarded to ``umap.UMAP``

    Returns
    -------
    np.ndarray [N, n_components]
    """
    try:
        import umap  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required for UMAP visualisation.  "
            "Install with:  pip install umap-learn"
        ) from exc

    vecs_n = normalize(vecs.astype(np.float32), norm="l2")
    reducer = umap.UMAP(
        n_components=n_components,
        random_state=seed,
        **umap_kwargs,
    )
    return reducer.fit_transform(vecs_n)


def compute_tsne(
    vecs: np.ndarray,
    n_components: int = 2,
    seed: int = 42,
    **tsne_kwargs: Any,
) -> np.ndarray:
    """
    Compute t-SNE embedding.

    Parameters
    ----------
    vecs         : np.ndarray [N, D]
    n_components : output dimensionality (default 2)
    seed         : random seed
    **tsne_kwargs: forwarded to ``sklearn.manifold.TSNE``

    Returns
    -------
    np.ndarray [N, n_components]
    """
    from sklearn.manifold import TSNE

    vecs_n = normalize(vecs.astype(np.float32), norm="l2")
    tsne = TSNE(
        n_components=n_components,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        **tsne_kwargs,
    )
    return tsne.fit_transform(vecs_n)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Colour palette — up to 91 distinct colours (7 scanners × 13 stains)
_TAB20   = plt.cm.get_cmap("tab20")
_TAB20B  = plt.cm.get_cmap("tab20b")
_TAB20C  = plt.cm.get_cmap("tab20c")

def _build_palette(n: int) -> list:
    """Return *n* visually distinct RGBA colours."""
    colours = (
        [_TAB20(i)  for i in np.linspace(0, 1, 20, endpoint=False)]
        + [_TAB20B(i) for i in np.linspace(0, 1, 20, endpoint=False)]
        + [_TAB20C(i) for i in np.linspace(0, 1, 20, endpoint=False)]
    )
    # repeat if more labels than colours
    while len(colours) < n:
        colours += colours
    return colours[:n]


def plot_embedding(
    coords: np.ndarray,
    labels,
    title: str,
    save_path: str | Path,
    *,
    figsize: tuple[float, float] = (12, 10),
    dpi: int = 150,
    label_col: str = "pos_key",
    alpha: float = 0.5,
    point_size: float = 4.0,
) -> None:
    """
    Scatter plot of 2-D *coords* coloured by *labels*.

    Parameters
    ----------
    coords    : np.ndarray [N, 2]
    labels    : array-like [N] — string or integer labels
    title     : figure title string
    save_path : where to save the PNG
    figsize   : matplotlib figure size
    dpi       : output resolution
    label_col : legend header text
    alpha     : point transparency
    point_size: marker size
    """
    labels = np.asarray(labels)
    unique_labels = sorted(set(labels.tolist()))
    palette = _build_palette(len(unique_labels))
    colour_map = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}

    colours = [colour_map[lbl] for lbl in labels]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=colours,
        s=point_size,
        alpha=alpha,
        linewidths=0,
        rasterized=True,
    )

    # legend — show at most 40 entries to keep the figure readable
    n_legend = min(len(unique_labels), 40)
    handles = [
        mpatches.Patch(color=colour_map[lbl], label=str(lbl))
        for lbl in unique_labels[:n_legend]
    ]
    if len(unique_labels) > 40:
        handles.append(mpatches.Patch(color="none", label=f"… (+{len(unique_labels)-40} more)"))

    ax.legend(
        handles=handles,
        title=label_col,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=6,
        title_fontsize=7,
        markerscale=2,
        frameon=False,
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Component 1", fontsize=9)
    ax.set_ylabel("Component 2", fontsize=9)
    ax.tick_params(labelsize=7)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_grid(
    coords_dict: dict[str, np.ndarray],
    labels,
    title: str,
    save_path: str | Path,
    *,
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
    label_col: str = "pos_key",
    alpha: float = 0.5,
    point_size: float = 4.0,
) -> None:
    """
    Multi-panel scatter plot — one panel per entry in *coords_dict*.

    Typical usage::

        plot_embedding_grid(
            {"UMAP": umap_coords, "t-SNE": tsne_coords},
            labels=metadata["pos_key"],
            title="Stain-Scanner Vectors — val split",
            save_path="plots/val_grid.png",
        )

    Parameters
    ----------
    coords_dict : mapping from panel name → np.ndarray [N, 2]
    labels      : array-like [N]
    title       : suptitle
    save_path   : output PNG path
    figsize     : auto-derived if None
    dpi         : output resolution
    label_col   : legend header text
    alpha, point_size : aesthetics
    """
    panels = list(coords_dict.items())
    n_panels = len(panels)

    if figsize is None:
        figsize = (10 * n_panels, 9)

    labels_arr = np.asarray(labels)
    unique_labels = sorted(set(labels_arr.tolist()))
    palette = _build_palette(len(unique_labels))
    colour_map = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}
    colours = [colour_map[lbl] for lbl in labels_arr]

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, dpi=dpi)
    if n_panels == 1:
        axes = [axes]

    for ax, (panel_name, coords) in zip(axes, panels):
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=colours,
            s=point_size,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(panel_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Component 1", fontsize=9)
        ax.set_ylabel("Component 2", fontsize=9)
        ax.tick_params(labelsize=7)

    # shared legend on the right-most axis
    n_legend = min(len(unique_labels), 40)
    handles = [
        mpatches.Patch(color=colour_map[lbl], label=str(lbl))
        for lbl in unique_labels[:n_legend]
    ]
    if len(unique_labels) > 40:
        handles.append(mpatches.Patch(color="none", label=f"… (+{len(unique_labels)-40} more)"))

    axes[-1].legend(
        handles=handles,
        title=label_col,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=6,
        title_fontsize=7,
        markerscale=2,
        frameon=False,
    )

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
