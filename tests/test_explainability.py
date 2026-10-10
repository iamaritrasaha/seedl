"""Tests for model explainability, feature attribution, and state preservation."""

from __future__ import annotations

import pytest

import seedl as sd
from seedl._explain import explain_prediction
from seedl.exceptions import SeeDLError


def test_saliency_and_state_preservation():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc1(x)

    model = SimpleNet()
    # Put in train mode to verify restoration
    model.train()

    # Pre-populate some gradients to verify they are restored
    dummy_x = torch.randn(1, 4)
    model(dummy_x).sum().backward()
    orig_grad = model.fc1.weight.grad.clone()

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    orig_x = x.clone()

    data, summary, notes = explain_prediction(model, x, method="saliency")

    # State preservation checks
    assert model.training is True  # Train mode was restored
    assert torch.equal(model.fc1.weight.grad, orig_grad)  # Original grad restored
    assert torch.equal(x, orig_x)  # User input untouched

    # Attribution shape matches input
    assert len(data["attribution"]) == 1
    assert len(data["attribution"][0]) == 4


def test_integrated_gradients():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    model = nn.Sequential(
        nn.Linear(3, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
    )
    x = torch.tensor([[0.5, -0.5, 1.0]])

    data, summary, notes = explain_prediction(
        model, x, method="integrated_gradients", steps=10, target=1
    )
    assert data["target_class"] == 1
    assert "integrated gradients" in summary.lower()
    assert len(data["attribution"][0]) == 3


def test_grad_cam_on_convnet():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class MiniConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 4, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            f = torch.relu(self.conv(x))
            p = self.pool(f).flatten(1)
            return self.fc(p)

    model = MiniConv()
    x = torch.randn(1, 1, 8, 8)

    view = sd.explain(model, input=x, method="grad_cam", backend="svg")
    assert view.kind == "attribution"
    assert "<svg" in view.to_figure()


def test_grad_cam_error_without_conv():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    linear_model = nn.Linear(4, 2)
    x = torch.randn(1, 4)

    with pytest.raises(SeeDLError, match="convolutional"):
        explain_prediction(linear_model, x, method="grad_cam")


def test_unsupported_method_error():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2)
    x = torch.randn(1, 2)

    with pytest.raises(SeeDLError, match="Unsupported explain method"):
        explain_prediction(model, x, method="magical_explain")
