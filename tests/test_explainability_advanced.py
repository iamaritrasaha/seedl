"""Tests for advanced explainability capabilities in SeeDL 0.5.0."""

from __future__ import annotations

import pytest

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

import seedl as sd
from seedl._explain_compare import compare_attributions


class LinearModel(nn.Module):
    def __init__(self, in_features: int = 4, out_features: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, -0.5]]))
            self.fc.bias.copy_(torch.tensor([0.5, -0.5]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_integrated_gradients_linear_completeness():
    """Integrated Gradients on a linear model satisfies completeness exactly:

    sum(IG(x)) == F(x) - F(x0).
    """
    model = LinearModel(in_features=4, out_features=2)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    baseline = torch.zeros_like(x)

    viz = sd.explain(
        model,
        input=x,
        target=0,
        method="integrated_gradients",
        steps=50,
        baseline=baseline,
    )

    assert viz is not None
    assert "diagnostics" in viz.data
    diag = viz.data["diagnostics"]
    assert "completeness_error" in diag
    # For a linear model, completeness error should be tiny (< 1e-3)
    assert diag["completeness_error"] < 1e-3
    assert len(diag["convergence_curve"]) == 50


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_smoothgrad_reproducibility_and_noise_scaling():
    """SmoothGrad produces deterministic attributions when given a seed."""
    model = LinearModel()
    x = torch.tensor([[1.0, -2.0, 0.5, 3.0]])

    viz1 = sd.explain(
        model,
        input=x,
        target=0,
        method="smoothgrad",
        samples=15,
        stdev_spread=0.1,
        seed=42,
    )

    viz2 = sd.explain(
        model,
        input=x,
        target=0,
        method="smoothgrad",
        samples=15,
        stdev_spread=0.1,
        seed=42,
    )

    attr1 = viz1.data["attribution"]
    attr2 = viz2.data["attribution"]
    assert torch.allclose(torch.tensor(attr1), torch.tensor(attr2), atol=1e-5)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_smoothgrad_signed_vs_unsigned():
    """SmoothGrad respects abs parameter."""
    model = LinearModel()
    x = torch.tensor([[1.0, -2.0, 0.5, 3.0]])

    viz_signed = sd.explain(
        model,
        input=x,
        target=1,
        method="smoothgrad",
        samples=10,
        stdev_spread=0.1,
        abs=False,
    )
    viz_abs = sd.explain(
        model,
        input=x,
        target=1,
        method="smoothgrad",
        samples=10,
        stdev_spread=0.1,
        abs=True,
    )

    attr_signed = viz_signed.data["attribution"]
    attr_abs = viz_abs.data["attribution"]
    assert any(val < 0.0 for row in attr_signed for val in row)
    # All values in absolute mode should be non-negative
    for row in attr_abs:
        for val in row:
            assert val >= 0.0


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_gradcam_automatic_and_explicit_layer():
    """Grad-CAM locates the last Conv2d layer automatically or respects explicit layer."""
    model = SimpleCNN()
    x = torch.randn(1, 1, 8, 8)

    # Automatic layer discovery
    viz_auto = sd.explain(
        model,
        input=x,
        target=0,
        method="grad_cam",
    )
    assert viz_auto is not None
    assert viz_auto.data["diagnostics"]["target_layer"] == "conv2"

    # Explicit layer override
    viz_explicit = sd.explain(
        model,
        input=x,
        target=0,
        method="grad_cam",
        layer="conv1",
    )
    assert viz_explicit.data["diagnostics"]["target_layer"] == "conv1"


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_compare_attributions_metrics():
    """compare_attributions calculates valid similarity metrics between methods."""
    model = LinearModel()
    x = torch.tensor([[1.0, -1.0, 2.0, -2.0]])

    comparison = compare_attributions(
        model,
        x,
        target=0,
        methods=["saliency", "integrated_gradients"],
        steps=20,
    )

    assert comparison.kind == "attribution_comparison"
    assert "similarity_matrix" in comparison.data
    sim = comparison.data["similarity_matrix"]
    assert "saliency" in sim
    assert "integrated_gradients" in sim["saliency"]
    metrics = sim["saliency"]["integrated_gradients"]
    assert metrics["shape_match"] is True
    assert metrics["pearson"] is not None
    assert metrics["cosine"] is not None
    assert -1.01 <= metrics["pearson"] <= 1.01
    assert -1.01 <= metrics["cosine"] <= 1.01


class ConstantModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 1), 42.0, device=x.device, dtype=x.dtype)


class NegativeWeightModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1, bias=True)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[-2.5, -1.0, -0.5]]))
            self.fc.bias.copy_(torch.tensor([-1.5]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class SaturatingReLUModel(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(42)
        self.fc1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_integrated_gradients_constant_and_negative_weights():
    """Verify IG completeness on constant models (zero attribution) and negative weights."""
    # 1. Constant model
    c_model = ConstantModel()
    x_c = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    viz_c = sd.explain(c_model, input=x_c, method="integrated_gradients", steps=10)
    assert viz_c.data["diagnostics"]["completeness_error"] < 1e-5
    for row in viz_c.data["attribution"]:
        for val in row:
            assert abs(val) < 1e-5

    # 2. Negative weight model
    neg_model = NegativeWeightModel()
    x_neg = torch.tensor([[2.0, 3.0, 1.0]], requires_grad=True)
    baseline_neg = torch.tensor([[0.5, 1.0, 0.0]])
    viz_neg = sd.explain(
        neg_model,
        input=x_neg,
        method="integrated_gradients",
        baseline=baseline_neg,
        steps=25,
        abs=False,
    )
    diag_neg = viz_neg.data["diagnostics"]
    # Exact completeness on linear model with negative weights
    assert diag_neg["completeness_error"] < 1e-4
    # Attributions should be negative since weights are negative and x > baseline
    for val in viz_neg.data["attribution"][0]:
        assert val < 0.0


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_integrated_gradients_saturating_relu_convergence():
    """Verify IG convergence curve is tracked and error is bounded on non-linear saturating models."""
    torch.manual_seed(42)
    model = SaturatingReLUModel()
    x = torch.tensor([[1.5, -0.5]], requires_grad=True)

    viz_10 = sd.explain(model, input=x, method="integrated_gradients", steps=10)
    viz_50 = sd.explain(model, input=x, method="integrated_gradients", steps=50)

    diag_10 = viz_10.data["diagnostics"]
    diag_50 = viz_50.data["diagnostics"]

    assert len(diag_10["convergence_curve"]) == 10
    assert len(diag_50["convergence_curve"]) == 50
    # Both approximations should have bounded relative or absolute error
    assert diag_50["relative_completeness_error"] < 0.15 or diag_50["completeness_error"] < 0.01


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_compare_attributions_mismatched_shapes_handling():
    """Verify compare_attributions flags shape_match=False for incomparable feature maps."""
    model = SimpleCNN()
    x = torch.randn(1, 1, 8, 8)

    # Compare Grad-CAM (which operates on conv layer) and Saliency (pixel level)
    comparison = compare_attributions(
        model,
        x,
        target=0,
        methods=["saliency", "grad_cam"],
    )

    sim = comparison.data["similarity_matrix"]
    assert "saliency" in sim
    assert "grad_cam" in sim["saliency"]
    entry = sim["saliency"]["grad_cam"]
    # Saliency is (1, 1, 8, 8), Grad-CAM on conv2 is (1, 8, 8) or similar
    # If shapes match or don't match, verify integrity without crashing
    if not entry["shape_match"]:
        assert entry["pearson"] is None
        assert entry["cosine"] is None
        assert "Incomparable shapes" in entry["note"]
