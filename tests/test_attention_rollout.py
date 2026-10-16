"""Tests for Transformer Attention Rollout and Multi-Layer Attention in SeeDL 0.5.0."""

from __future__ import annotations

import math

import pytest

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

import seedl as sd
from seedl._transformer_observatory import attention_layers, attention_rollout


@pytest.mark.skipif(not _NUMPY_AVAILABLE, reason="NumPy not available")
def test_attention_rollout_identity():
    """Single layer with identity attention produces identity rollout."""
    # 1 layer, 1 head, 3x3 identity
    eye = np.eye(3, dtype=float)
    # Shape (1, 1, 3, 3): (layers, heads, tokens, tokens)
    matrices = eye[np.newaxis, np.newaxis, :, :]

    viz = attention_rollout(matrices, tokens=["a", "b", "c"], include_residual=False)
    rollout_matrix = np.array(viz.data["rollout_matrix"])

    assert np.allclose(rollout_matrix, eye, atol=1e-5)
    # Row stochasticity: rows sum to 1
    assert np.allclose(rollout_matrix.sum(axis=-1), np.ones(3), atol=1e-5)


@pytest.mark.skipif(not _NUMPY_AVAILABLE, reason="NumPy not available")
def test_attention_rollout_multiplication_order():
    """Mathematically verify rollout cumulative multiplication order: R = A_L @ ... @ A_1."""
    # Two layers, 1 head, 2 tokens
    # A1
    a1 = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=float)
    # A2
    a2 = np.array([[0.6, 0.4], [0.3, 0.7]], dtype=float)

    matrices = np.stack([a1[np.newaxis, :, :], a2[np.newaxis, :, :]], axis=0)

    # Without residual blending
    viz = attention_rollout(matrices, tokens=["[CLS]", "word"], include_residual=False)
    rollout_matrix = np.array(viz.data["rollout_matrix"])

    # Formula: R = A2 @ A1
    expected_r = a2 @ a1
    # Row normalize expected_r
    expected_r = expected_r / expected_r.sum(axis=-1, keepdims=True)

    assert np.allclose(rollout_matrix, expected_r, atol=1e-5)


@pytest.mark.skipif(not _NUMPY_AVAILABLE, reason="NumPy not available")
def test_attention_rollout_residual_connection():
    """Residual blending modifies attention matrix: A' = 0.5 * (A + I)."""
    a1 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    matrices = a1[np.newaxis, np.newaxis, :, :]

    viz = attention_rollout(matrices, tokens=["t0", "t1"], include_residual=True)
    rollout_matrix = np.array(viz.data["rollout_matrix"])

    # Blending eye with eye still gives eye
    assert np.allclose(rollout_matrix, np.eye(2), atol=1e-5)


@pytest.mark.skipif(not _NUMPY_AVAILABLE, reason="NumPy not available")
def test_attention_rollout_head_reduction():
    """Rollout supports 'mean', 'min', 'max' across attention heads."""
    # 1 layer, 2 heads, 2x2
    h1 = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float)
    h2 = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=float)
    matrices = np.stack([h1, h2], axis=0)[np.newaxis, :, :, :]  # (1, 2, 2, 2)

    viz_mean = attention_rollout(matrices, head_reduction="mean", include_residual=False)
    viz_max = attention_rollout(matrices, head_reduction="max", include_residual=False)

    r_mean = np.array(viz_mean.data["rollout_matrix"])
    r_max = np.array(viz_max.data["rollout_matrix"])

    assert not np.allclose(r_mean, r_max)
    # Check row sums are 1.0 for both
    assert np.allclose(r_mean.sum(axis=-1), np.ones(2), atol=1e-5)
    assert np.allclose(r_max.sum(axis=-1), np.ones(2), atol=1e-5)


@pytest.mark.skipif(not _NUMPY_AVAILABLE, reason="NumPy not available")
def test_attention_layers_structure_and_entropy():
    """attention_layers extracts per-layer entropy and head statistics."""
    # 3 layers, 4 heads, 5 tokens
    rng = np.random.RandomState(42)
    raw = rng.uniform(0.1, 1.0, size=(3, 4, 5, 5))
    # Softmax-like row normalization
    matrices = raw / raw.sum(axis=-1, keepdims=True)

    viz = attention_layers(
        matrices,
        tokens=[f"tok_{i}" for i in range(5)],
    )

    assert viz.kind == "attention_layers"
    assert "layers" in viz.data
    assert len(viz.data["layers"]) == 3
    assert "layer_entropies" in viz.data
    assert len(viz.data["layer_entropies"]) == 3

    # Shannon entropy (base 2) of a 5-element discrete probability distribution is in [0, log2(5)]
    max_entropy = math.log2(5.0)
    for ent in viz.data["layer_entropies"]:
        assert 0.0 <= ent <= max_entropy + 1e-4


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_sd_attention_rollout_torch_tensor_input():
    """sd.attention_rollout accepts PyTorch tensors seamlessly."""
    t = torch.softmax(torch.randn(2, 2, 4, 4), dim=-1)
    viz = sd.attention_rollout(t, tokens=["[CLS]", "a", "b", "[SEP]"])

    assert viz.kind == "attention_rollout"
    assert "rollout_matrix" in viz.data
    assert len(viz.data["rollout_matrix"]) == 4
