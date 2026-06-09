"""
Evaluation metrics for stain vectors.

Covers:
  - Clustering quality  (silhouette, Davies-Bouldin, Calinski-Harabasz, KMeans ARI/NMI)
  - kNN purity
  - Linear probe accuracy / F1
  - Cross-split retrieval Recall@k
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder, normalize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2(vecs: np.ndarray) -> np.ndarray:
    """Return L2-normalised copy of *vecs* (float64 for sklearn stability)."""
    return normalize(vecs.astype(np.float64), norm="l2")


def _encode_labels(labels) -> np.ndarray:
    """Convert arbitrary label sequence to contiguous integers."""
    le = LabelEncoder()
    return le.fit_transform(labels)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clustering_metrics(
    vecs: np.ndarray,
    labels,
    *,
    n_clusters: int | None = None,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Compute unsupervised clustering quality metrics.

    Parameters
    ----------
    vecs : np.ndarray, shape [N, D]
        Feature vectors.  Will be L2-normalised internally.
    labels : array-like, length N
        Ground-truth cluster labels (e.g. pos_key strings).
    n_clusters : int | None
        Number of KMeans clusters.  Defaults to the number of unique labels.
    random_state : int
        RNG seed for KMeans.

    Returns
    -------
    dict with keys:
        silhouette, davies_bouldin, calinski_harabasz, kmeans_ari, kmeans_nmi
    """
    vecs_n = _l2(vecs)
    y = _encode_labels(labels)
    n_cls = n_clusters if n_clusters is not None else int(y.max()) + 1

    km = KMeans(n_clusters=n_cls, random_state=random_state, n_init=10)
    km_labels = km.fit_predict(vecs_n)

    return {
        "silhouette": float(silhouette_score(vecs_n, y, sample_size=min(len(y), 10_000), random_state=random_state)),
        "davies_bouldin": float(davies_bouldin_score(vecs_n, y)),
        "calinski_harabasz": float(calinski_harabasz_score(vecs_n, y)),
        "kmeans_ari": float(adjusted_rand_score(y, km_labels)),
        "kmeans_nmi": float(normalized_mutual_info_score(y, km_labels, average_method="arithmetic")),
    }


def knn_purity(
    vecs_query: np.ndarray,
    labels_query,
    vecs_ref: np.ndarray,
    labels_ref,
    k_list: list[int] | None = None,
) -> dict[str, float]:
    """
    kNN purity: for each query vector, find the k nearest neighbours in the
    reference set and report the fraction whose label matches the query label.

    When query == ref (same split), the query itself is excluded from the
    neighbour search automatically (self-retrieval guard via distance > 0).

    Parameters
    ----------
    vecs_query : np.ndarray [Nq, D]
    labels_query : array-like [Nq]
    vecs_ref    : np.ndarray [Nr, D]
    labels_ref  : array-like [Nr]
    k_list      : list of k values to evaluate (default [1, 5, 10])

    Returns
    -------
    dict  e.g. {"knn_purity@1": 0.95, "knn_purity@5": 0.92, ...}
    """
    if k_list is None:
        k_list = [1, 5, 10]

    q = _l2(vecs_query)
    r = _l2(vecs_ref)
    lq = np.asarray(labels_query)
    lr = np.asarray(labels_ref)

    # cosine similarity matrix  [Nq, Nr]
    sim = q @ r.T

    # self-retrieval guard: mask identical vectors (query == ref)
    same_split = (q.shape == r.shape) and np.allclose(q, r, atol=1e-6)

    results: dict[str, float] = {}
    max_k = max(k_list)
    # retrieve max_k+1 to allow self-exclusion
    k_retrieve = max_k + 1 if same_split else max_k
    top_idx = np.argpartition(-sim, kth=min(k_retrieve, sim.shape[1] - 1), axis=1)[
        :, :k_retrieve
    ]

    for k in k_list:
        hits = []
        for i in range(len(q)):
            idxs = top_idx[i]
            # sort by descending similarity
            idxs_sorted = idxs[np.argsort(-sim[i, idxs])]
            if same_split:
                idxs_sorted = idxs_sorted[idxs_sorted != i]
            nn = idxs_sorted[:k]
            hits.append(np.mean(lr[nn] == lq[i]))
        results[f"knn_purity@{k}"] = float(np.mean(hits))

    return results


def linear_probe(
    vecs_train: np.ndarray,
    labels_train,
    vecs_test: np.ndarray,
    labels_test,
    *,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Train a logistic regression linear probe on *vecs_train* and evaluate on
    *vecs_test*.

    Parameters
    ----------
    vecs_train : np.ndarray [N_train, D]
    labels_train : array-like [N_train]
    vecs_test    : np.ndarray [N_test, D]
    labels_test  : array-like [N_test]
    C            : regularisation inverse strength (default 1.0)
    max_iter     : solver iterations (default 1000)

    Returns
    -------
    dict with keys: accuracy, f1_macro, f1_weighted
    """
    X_tr = _l2(vecs_train)
    X_te = _l2(vecs_test)

    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
        multi_class="auto",
    )
    clf.fit(X_tr, labels_train)
    y_pred = clf.predict(X_te)

    return {
        "accuracy": float(np.mean(y_pred == np.asarray(labels_test))),
        "f1_macro": float(f1_score(labels_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels_test, y_pred, average="weighted", zero_division=0)),
    }


def retrieval_metrics(
    vecs_a: np.ndarray,
    keys_a,
    vecs_b: np.ndarray,
    keys_b,
    k_list: list[int] | None = None,
) -> dict[str, float]:
    """
    Cross-split retrieval Recall@k.

    For each query vector in split A, retrieve the top-k most similar vectors
    from split B and check whether *any* retrieved vector shares the same key
    (pos_key = device||stain).

    Parameters
    ----------
    vecs_a : np.ndarray [Na, D]  — query set
    keys_a : array-like [Na]     — pos_key labels for queries
    vecs_b : np.ndarray [Nb, D]  — gallery set
    keys_b : array-like [Nb]     — pos_key labels for gallery
    k_list : list of k values (default [1, 5, 10])

    Returns
    -------
    dict  e.g. {"recall@1": 0.98, "recall@5": 0.99, ...}
    """
    if k_list is None:
        k_list = [1, 5, 10]

    qa = _l2(vecs_a)
    qb = _l2(vecs_b)
    ka = np.asarray(keys_a)
    kb = np.asarray(keys_b)

    # cosine similarity  [Na, Nb]
    sim = qa @ qb.T

    results: dict[str, float] = {}
    max_k = max(k_list)
    top_idx = np.argpartition(-sim, kth=min(max_k, sim.shape[1] - 1), axis=1)[
        :, :max_k
    ]

    for k in k_list:
        recalls = []
        for i in range(len(qa)):
            idxs = top_idx[i]
            idxs_sorted = idxs[np.argsort(-sim[i, idxs])][:k]
            recalled = ka[i] in kb[idxs_sorted]
            recalls.append(float(recalled))
        results[f"recall@{k}"] = float(np.mean(recalls))

    return results
