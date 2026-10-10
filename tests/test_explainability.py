from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import seedl as sd
from seedl._explain import explain_prediction
from seedl.exceptions import SeeDLError

if TYPE_CHECKING:
    pass


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


def test_integrated_gradients_completeness_linear_model():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    # Linear model with positive and negative weights
    model = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[-2.5, 4.0, -1.0]]))

    x = torch.tensor([[1.5, 2.0, 3.0]])
    data, _, _ = explain_prediction(model, x, method="integrated_gradients", steps=20)
    attr = data["attribution"][0]

    # Analytical attributions: x_i * w_i
    expected = [1.5 * -2.5, 2.0 * 4.0, 3.0 * -1.0]  # [-3.75, 8.0, -3.0]
    for a, exp in zip(attr, expected, strict=True):
        assert abs(a - exp) < 1e-4

    # Completeness: sum(attr) == f(x) - f(0)
    fx = model(x).item()
    f0 = model(torch.zeros_like(x)).item()
    assert abs(sum(attr) - (fx - f0)) < 1e-4


def test_integrated_gradients_custom_baseline_and_convergence():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    # Cubic model f(x) = x_0^3 + 2*x_1
    class PolyNet(nn.Module):
        def forward(self, inp):
            return (inp[:, 0:1] ** 3 + 2.0 * inp[:, 1:2]).squeeze(-1)

    model = PolyNet()
    x = torch.tensor([[3.0, 2.0]])
    baseline = torch.tensor([[1.0, 0.5]])

    # Baseline comparison: f(x) - f(baseline) = (27 + 4) - (1 + 1) = 31 - 2 = 29
    f_x = model(x).item()
    f_b = model(baseline).item()
    expected_diff = f_x - f_b

    # Test convergence as steps increase: error should decrease
    data_10, _, _ = explain_prediction(
        model, x, baseline=baseline, method="integrated_gradients", steps=10
    )
    data_50, _, _ = explain_prediction(
        model, x, baseline=baseline, method="integrated_gradients", steps=50
    )

    err_10 = abs(sum(data_10["attribution"][0]) - expected_diff)
    err_50 = abs(sum(data_50["attribution"][0]) - expected_diff)

    assert err_50 < err_10
    assert err_50 < 0.05


def test_integrated_gradients_invalid_parameters():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1)
    x = torch.randn(1, 2)

    # Invalid steps
    with pytest.raises(SeeDLError, match="steps must be a positive integer"):
        explain_prediction(model, x, method="integrated_gradients", steps=0)

    # Mismatched baseline shape
    with pytest.raises(SeeDLError, match="Baseline shape"):
        explain_prediction(model, x, method="integrated_gradients", baseline=torch.randn(1, 4))

    # Target out of bounds
    with pytest.raises(SeeDLError, match="out of bounds"):
        explain_prediction(model, x, method="integrated_gradients", target=99)


def test_saliency_signed_and_target_bounds():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[-3.0, 2.0]]))
    x = torch.tensor([[1.0, 1.0]])

    # Signed saliency (abs=False)
    data_signed, _, _ = explain_prediction(model, x, method="saliency", abs=False)
    assert abs(data_signed["attribution"][0][0] - (-3.0)) < 1e-4
    assert abs(data_signed["attribution"][0][1] - 2.0) < 1e-4

    # Absolute saliency (abs=True or default)
    data_abs, _, _ = explain_prediction(model, x, method="saliency")
    assert abs(data_abs["attribution"][0][0] - 3.0) < 1e-4

    # Out of bounds target
    with pytest.raises(SeeDLError, match="out of bounds"):
        explain_prediction(model, x, method="saliency", target=-1)


def test_grad_cam_custom_layer_and_disconnected():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class MultiConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
            self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(8, 2)

        def forward(self, x):
            f1 = torch.relu(self.conv1(x))
            f2 = torch.relu(self.conv2(f1))
            return self.fc(self.pool(f2).flatten(1))

    model = MultiConv()
    x = torch.randn(1, 1, 8, 8)

    # Target specific layer by name
    view = sd.explain(model, input=x, method="grad_cam", layer="conv1", backend="svg")
    assert view.kind == "attribution"

    # Non-existent layer
    with pytest.raises(SeeDLError, match="Target layer 'non_existent' not found"):
        sd.explain(model, input=x, method="grad_cam", layer="non_existent")
