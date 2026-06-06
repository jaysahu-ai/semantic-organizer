"""
clusterer.py
------------
Clusters document embeddings using KMeans.

Two modes (per design doc):
  1. Auto   — tries k=2..max_k, picks the k with the best Silhouette score
  2. Manual — user supplies k directly (override mode)

Silhouette score ranges from -1 to 1:
  ~1.0  → documents are well inside their own cluster, far from others (great)
  ~0.0  → documents are on the boundary between clusters (ambiguous)
  ~-1.0 → documents are probably in the wrong cluster (bad)
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def _run_kmeans(embeddings: np.ndarray, k: int, random_state: int = 42) -> tuple[np.ndarray, float]:
    """
    Fit KMeans with k clusters and return (labels, silhouette_score).

    Args:
        embeddings:   Normalized embedding matrix of shape (N, 384).
        k:            Number of clusters.
        random_state: Seed for reproducibility.

    Returns:
        labels:           np.ndarray of shape (N,) — cluster index per document.
        silhouette_score: Float in [-1, 1]. Higher is better.
                          Returns -1.0 if k=1 (silhouette undefined for single cluster).
    """
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(embeddings)

    # Silhouette score requires at least 2 clusters and 2 samples per cluster
    if k < 2:
        return labels, -1.0

    score = silhouette_score(embeddings, labels, metric="cosine")
    return labels, score


def auto_cluster(
    embeddings: np.ndarray,
    min_k: int = 2,
    max_k: int = 10,
) -> tuple[np.ndarray, int, float, list[dict]]:
    """
    Automatically find the optimal number of clusters using Silhouette Analysis.

    Tries every k from min_k to max_k (inclusive), picks the one
    with the highest silhouette score.

    Args:
        embeddings: Normalized embedding matrix of shape (N, 384).
        min_k:      Minimum number of clusters to try.
        max_k:      Maximum number of clusters to try.
                    Auto-capped at N-1 (can't have more clusters than documents).

    Returns:
        labels:        np.ndarray of shape (N,) — cluster index per document.
        best_k:        The optimal number of clusters found.
        best_score:    The silhouette score for best_k.
        score_history: List of {"k": int, "score": float} for all tried values.
                       Useful for visualizing the silhouette curve in the GUI later.
    """
    n_docs = len(embeddings)

    # Guard: can't cluster fewer than 2 documents
    if n_docs < 2:
        raise ValueError(f"[clusterer] Need at least 2 documents to cluster, got {n_docs}.")

    # Cap max_k — can't have more clusters than documents
    max_k = min(max_k, n_docs - 1)

    if min_k > max_k:
        # Edge case: very small folder — just put everything in one cluster
        print(f"[clusterer] Only {n_docs} document(s) — using k=1.")
        labels = np.zeros(n_docs, dtype=int)
        return labels, 1, -1.0, [{"k": 1, "score": -1.0}]

    print(f"[clusterer] Running Silhouette Analysis for k={min_k}..{max_k}...")

    best_k = min_k
    best_score = -1.0
    best_labels = None
    score_history = []

    for k in range(min_k, max_k + 1):
        labels, score = _run_kmeans(embeddings, k)
        score_history.append({"k": k, "score": round(float(score), 4)})
        print(f"[clusterer]   k={k:2d}  silhouette={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    print(f"[clusterer] Best k={best_k} (silhouette={best_score:.4f})")
    return best_labels, best_k, best_score, score_history



def build_clusters(
    metadata: list[dict],
    labels: np.ndarray,
    k: int,
) -> list[dict]:
    """
    Group metadata by cluster label into a structured result.

    Args:
        metadata: List of dicts from store.load() — index-aligned with labels.
        labels:   np.ndarray of shape (N,) — cluster index per document.
        k:        Number of clusters.

    Returns:
        List of cluster dicts:
        [
            {
                "cluster_id": 0,
                "label": None,          ← filled in by labeler.py
                "documents": [
                    {"filename": "...", "path": "..."},
                    ...
                ]
            },
            ...
        ]
    """
    clusters = [
        {"cluster_id": i, "label": None, "documents": []}
        for i in range(k)
    ]

    for doc, cluster_id in zip(metadata, labels):
        clusters[cluster_id]["documents"].append({
            "filename": doc["filename"],
            "path": doc["path"],
        })

    # Log cluster sizes
    for cluster in clusters:
        print(f"[clusterer] Cluster {cluster['cluster_id']}: {len(cluster['documents'])} document(s)")

    return clusters
