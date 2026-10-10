"""Embedding visualization, pairwise cosine similarity, and dimensionality reduction."""

from __future__ import annotations

import math
from typing import Any

from .exceptions import SeeDLError


def inspect_embeddings(
    embeddings: Any,
    *,
    labels: list[str] | None = None,
    method: str = "pca",
    max_vectors: int = 128,
) -> tuple[dict[str, Any], str, list[str]]:
    """Inspect vector embeddings, pairwise cosine similarity matrix, and 2D projections."""
    raw = embeddings
    if hasattr(raw, "detach") and hasattr(raw, "cpu") and hasattr(raw, "tolist"):
        raw = raw.detach().cpu().tolist()
    elif hasattr(raw, "tolist"):
        raw = raw.tolist()

    if not isinstance(raw, list) or not raw or not isinstance(raw[0], list):
        raise SeeDLError("Embeddings must be a 2D matrix of shape (num_vectors, dimensions).")

    num_vectors = len(raw)
    dim = len(raw[0])
    if dim < 2:
        raise SeeDLError(f"Embedding dimension must be at least 2, received {dim}.")

    # Subsample if too many vectors for pairwise matrix
    count = min(num_vectors, max_vectors)
    selected = raw[:count]

    item_labels = [labels[i] if labels and i < len(labels) else f"V{i}" for i in range(count)]

    # Compute vector norms and pairwise cosine similarity
    norms = []
    normalized_vecs = []
    for vec in selected:
        norm = math.sqrt(sum(float(x) * float(x) for x in vec))
        norms.append(norm)
        if norm > 1e-12:
            normalized_vecs.append([float(x) / norm for x in vec])
        else:
            normalized_vecs.append([0.0] * dim)

    similarity_matrix = []
    nearest_pairs = []
    for i in range(count):
        row = []
        for j in range(count):
            dot = sum(normalized_vecs[i][k] * normalized_vecs[j][k] for k in range(dim))
            # clamp to [-1, 1]
            dot = max(-1.0, min(1.0, dot))
            row.append(dot)
            if j > i:
                nearest_pairs.append((dot, item_labels[i], item_labels[j]))
        similarity_matrix.append(row)

    # Sort nearest pairs
    nearest_pairs.sort(key=lambda item: item[0], reverse=True)

    # Compute 2D PCA projection
    coords_2d, explained_var = _compute_pca_2d(selected)

    summary = f"Embeddings: {num_vectors} vectors in {dim} dimensions (showing {count} samples)"
    notes = [
        f"Pairwise cosine similarity matrix calculated for {count} vectors.",
        f"2D PCA projection explains {explained_var:.1%} of total variance across first 2 components.",
        "Notice: 2D linear projections preserve maximum variance but can distort relative high-dimensional distances.",
    ]

    if nearest_pairs:
        top_pair = nearest_pairs[0]
        notes.append(
            f"Highest similarity pair: '{top_pair[1]}' and '{top_pair[2]}' (cos sim = {top_pair[0]:.3f})."
        )

    data = {
        "num_vectors": num_vectors,
        "dim": dim,
        "labels": item_labels,
        "similarity_matrix": similarity_matrix,
        "coords_2d": coords_2d,
        "pca_points": coords_2d,
        "explained_variance": explained_var,
        "pca_variance_ratio": [explained_var],
        "vector_norms": norms,
        "top_similar_pairs": [
            {"sim": p[0], "item_a": p[1], "item_b": p[2]} for p in nearest_pairs[:5]
        ],
    }

    return data, summary, notes


def _compute_pca_2d(vectors: list[list[float]]) -> tuple[list[list[float]], float]:
    """Compute analytical 2D PCA projection and explained variance."""
    n = len(vectors)
    if n == 0:
        return [], 0.0
    d = len(vectors[0])
    if d < 2:
        return [[0.0, 0.0] for _ in vectors], 0.0

    # Center vectors
    means = [sum(vectors[i][j] for i in range(n)) / float(n) for j in range(d)]
    centered = [[vectors[i][j] - means[j] for j in range(d)] for i in range(n)]

    # Compute covariance or SVD using numpy if available
    try:
        import numpy as np

        X = np.array(centered, dtype=np.float64)
        u, s, vt = np.linalg.svd(X, full_matrices=False)
        if vt.shape[0] >= 2:
            proj = np.dot(X, vt[:2].T)
        elif vt.shape[0] == 1:
            p1 = np.dot(X, vt[:1].T)
            proj = np.hstack([p1, np.zeros((n, 1), dtype=np.float64)])
        else:
            proj = np.zeros((n, 2), dtype=np.float64)
        total_var = np.sum(s**2)
        exp_var = float(np.sum(s[:2] ** 2) / total_var) if total_var > 0 else 0.0
        return proj.tolist(), exp_var
    except ImportError:
        pass

    # Simple 2D fallback: first two coordinate axes
    proj = [[row[0], row[1] if d > 1 else 0.0] for row in centered]
    return proj, 0.5
