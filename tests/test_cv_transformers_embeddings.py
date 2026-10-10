"""Tests for computer vision convolution simulations, transformer attention heatmaps, and vector embeddings."""

from __future__ import annotations

import pytest

import seedl as sd
from seedl._embeddings import inspect_embeddings
from seedl._transformers import inspect_attention
from seedl._vision import inspect_convolution
from seedl.exceptions import SeeDLError


def test_convolution_exact_arithmetic_and_output():
    # 3x3 input image
    image = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]
    # 2x2 kernel (ones) -> sum of each 2x2 patch
    kernel = [
        [1.0, 1.0],
        [1.0, 1.0],
    ]

    data, summary, notes = inspect_convolution(image, kernel, stride=1, padding=0)

    # Output dimensions: (3 - 2)/1 + 1 = 2x2
    assert data["h_out"] == 2
    assert data["w_out"] == 2
    # Top-left patch: 1+2+4+5 = 12
    assert abs(data["output"][0][0] - 12.0) < 1e-4
    # Bottom-right patch: 5+6+8+9 = 28
    assert abs(data["output"][1][1] - 28.0) < 1e-4

    # Test invalid stride
    with pytest.raises(SeeDLError, match="stride"):
        inspect_convolution(image, kernel, stride=0)

    # Test sd.convolution wrapper
    view = sd.convolution(image, kernel, backend="svg")
    assert view.kind == "convolution_simulation"
    assert "<svg" in view.to_figure()


def test_transformer_attention_heads_and_causal_mask():
    # 4-token sequence with causal mask (upper triangular is 0)
    causal_weights = [
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.3, 0.3, 0.4, 0.0],
        [0.2, 0.2, 0.3, 0.3],
    ]
    tokens = ["The", "quick", "brown", "fox"]

    data, summary, notes = inspect_attention(causal_weights, tokens=tokens)
    assert data["seq_len"] == 4
    assert any("causal" in note.lower() for note in notes)

    # 3D multi-head attention: (2 heads, 3, 3)
    head_weights = [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]],
    ]
    data_head0, _, _ = inspect_attention(head_weights, head=0)
    assert data_head0["active_head"] == "Head 0"

    # Averaged heads
    data_avg, _, _ = inspect_attention(head_weights, head=None)
    assert "Average" in data_avg["active_head"]

    # Public API wrapper
    view = sd.attention(causal_weights, tokens=tokens, backend="svg")
    assert view.kind == "attention"
    assert "<svg" in view.to_figure()


def test_embeddings_cosine_similarity_and_pca():
    # 3 vectors in 3D space:
    # v0 along X, v1 along X (identical direction), v2 along Y (orthogonal)
    vectors = [
        [2.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
    ]
    labels = ["X1", "X2", "Y"]

    data, summary, notes = inspect_embeddings(vectors, labels=labels)

    # Cosine similarity between X1 and X2 is 1.0
    assert abs(data["similarity_matrix"][0][1] - 1.0) < 1e-4
    # Cosine similarity between X1 and Y is 0.0 (orthogonal)
    assert abs(data["similarity_matrix"][0][2] - 0.0) < 1e-4

    # PCA 2D coordinates exists
    assert len(data["pca_points"]) == 3
    assert 0.0 <= data["explained_variance"] <= 1.0001

    # Public API wrapper
    view = sd.embeddings(vectors, labels=labels, backend="svg")
    assert view.kind == "embeddings"
    assert "<svg" in view.to_figure()


def test_cross_attention_rectangular_matrix():
    # 2 queries, 4 keys (cross-attention)
    weights = [
        [
            [0.1, 0.4, 0.3, 0.2],
            [0.2, 0.3, 0.1, 0.4],
        ]
    ]  # (1 head, 2 queries, 4 keys)
    data, summary, notes = inspect_attention(weights)
    assert len(data["matrix"]) == 2  # 2 rows (queries)
    assert len(data["matrix"][0]) == 4  # 4 columns (keys)
    assert data["is_causal"] is False  # Non-square cannot be causal


def test_embeddings_degenerate_and_rank1_vectors():
    # Single vector (rank 1)
    data_1, _, _ = inspect_embeddings([[1.0, 2.0, 3.0]])
    assert len(data_1["pca_points"]) == 1
    assert len(data_1["pca_points"][0]) == 2  # Guaranteed 2D coordinates

    # Collinear vectors (rank 1)
    collinear = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]
    data_col, _, _ = inspect_embeddings(collinear)
    assert len(data_col["pca_points"]) == 3
    assert all(len(p) == 2 for p in data_col["pca_points"])


def test_convolution_empty_and_malformed_inputs():
    with pytest.raises(SeeDLError, match="image must be a non-empty 2D"):
        inspect_convolution([])

    with pytest.raises(SeeDLError, match="image must be a non-empty 2D"):
        inspect_convolution([[]])

    with pytest.raises(SeeDLError, match="kernel must be a non-empty 2D"):
        inspect_convolution([[1.0, 2.0], [3.0, 4.0]], kernel=[])
