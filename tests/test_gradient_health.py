"""Tests for Gradient Health Intelligence in SeeDL 0.5.0."""

from __future__ import annotations

import math

import pytest

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

import seedl as sd
from seedl._gradient_health import inspect_gradient_health


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 2, bias=True)
        self.fc2 = nn.Linear(2, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


class TiedWeightModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 4)
        self.lm_head = nn.Linear(4, 10, bias=False)
        # Tie weights: lm_head.weight shares storage with embed.weight
        self.lm_head.weight = self.embed.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.embed(idx))


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_gradient_health_exact_norms():
    """Verify that computed L1 and L2 norms exactly match manual PyTorch norms."""
    model = SimpleModel()
    x = torch.tensor([[1.0, 2.0, 3.0]])
    y = model(x).sum()
    y.backward()

    viz = sd.gradient_health(model)
    assert viz.kind == "gradient_health"
    data = viz.data

    # Check each parameter
    for rec in data["parameters"]:
        p_name = rec["name"]
        param = dict(model.named_parameters())[p_name]
        assert param.grad is not None
        expected_l1 = float(param.grad.norm(1).item())
        expected_l2 = float(param.grad.norm(2).item())

        assert math.isclose(rec["l1_norm"], expected_l1, rel_tol=1e-5)
        assert math.isclose(rec["l2_norm"], expected_l2, rel_tol=1e-5)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_gradient_health_tied_weights_deduplication():
    """Tied parameters are detected and deduplicated from global gradient norms."""
    model = TiedWeightModel()
    idx = torch.tensor([[1, 2]])
    loss = model(idx).sum()
    loss.backward()

    data = inspect_gradient_health(model)
    records = data["parameters"]

    assert len(records) == 2
    # One of them should be marked as tied
    tied_count = sum(1 for r in records if r["is_tied"])
    assert tied_count == 1

    # Check that global L2 equals the unique parameter's L2 norm (since there's only 1 weight)
    unique_rec = next(r for r in records if not r["is_tied"])
    assert math.isclose(data["global_l2"], unique_rec["l2_norm"], rel_tol=1e-5)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_gradient_health_missing_gradients():
    """Models with missing or uncomputed gradients report missing params gracefully without error."""
    model = SimpleModel()
    # No backward pass performed: param.grad is None
    viz = sd.gradient_health(model)

    assert viz is not None
    data = viz.data
    assert len(data["missing_grad_parameters"]) == len(list(model.parameters()))
    for rec in data["parameters"]:
        assert rec["has_grad"] is False
        assert rec["l1_norm"] is None


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_gradient_health_vanishing_and_nonfinite_observations():
    """Descriptive diagnostics identify vanishing and nonfinite gradients accurately."""
    model = SimpleModel()
    with torch.no_grad():
        for p in model.parameters():
            p.grad = torch.zeros_like(p.data)

    data_zero = inspect_gradient_health(model)
    assert any(
        "vanishing" in obs.lower() or "near-zero" in obs.lower()
        for obs in data_zero["observations"]
    )

    # Now simulate nonfinite gradients
    with torch.no_grad():
        model.fc1.weight.grad[0, 0] = float("nan")

    data_nan = inspect_gradient_health(model)
    assert data_nan["total_nonfinite"] > 0
    assert any("nonfinite" in obs.lower() for obs in data_nan["observations"])


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_gradient_health_frozen_parameters_and_zero_weight_norm():
    """Verify frozen parameters and zero parameter norms handle division by zero safely."""
    model = SimpleModel()
    # Freeze fc2
    model.fc2.weight.requires_grad = False
    # Set fc1 weight data to zero
    with torch.no_grad():
        model.fc1.weight.zero_()
        model.fc1.weight.grad = torch.randn_like(model.fc1.weight)

    data = inspect_gradient_health(model)
    rec_fc1 = next(r for r in data["parameters"] if r["name"] == "fc1.weight")
    rec_fc2 = next(r for r in data["parameters"] if r["name"] == "fc2.weight")

    # fc1 weight norm is 0.0, so grad_to_weight_ratio must safely be None without dividing by zero
    assert rec_fc1["weight_norm"] == 0.0
    assert rec_fc1["grad_to_weight_ratio"] is None

    # fc2 is frozen: requires_grad=False, grad is None, should not be in missing_grad_parameters
    assert rec_fc2["requires_grad"] is False
    assert rec_fc2["has_grad"] is False
    assert "fc2.weight" not in data["missing_grad_parameters"]


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_gradient_health_sparse_and_bfloat16_gradients():
    """Verify sparse COO gradients and bfloat16 mixed-precision gradients are inspected accurately."""
    # Test bfloat16
    model_bf16 = SimpleModel().to(torch.bfloat16)
    x = torch.randn(2, 3, dtype=torch.bfloat16)
    loss = model_bf16(x).sum()
    loss.backward()

    data_bf16 = inspect_gradient_health(model_bf16)
    assert data_bf16["global_l2"] > 0.0
    for r in data_bf16["parameters"]:
        assert r["has_grad"] is True
        assert r["l1_norm"] is not None

    # Test sparse gradient on Embedding
    emb = nn.Embedding(5, 4, sparse=True)
    idx = torch.tensor([0, 2])
    loss_sparse = emb(idx).sum()
    loss_sparse.backward()

    data_sparse = inspect_gradient_health(emb)
    rec_emb = data_sparse["parameters"][0]
    assert rec_emb["has_grad"] is True
    assert rec_emb["l1_norm"] is not None
    assert rec_emb["l2_norm"] is not None
