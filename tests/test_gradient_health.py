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
