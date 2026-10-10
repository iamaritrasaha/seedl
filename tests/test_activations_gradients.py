from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import seedl as sd
from seedl._activations import capture_activations
from seedl._gradients import inspect_gradients

if TYPE_CHECKING:
    pass


def test_activations_dead_units_and_saturation():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class DiagnosticNet(nn.Module):
        def __init__(self):
            super().__init__()
            # Force negative inputs to ReLU to trigger dead unit diagnostic
            self.fc1 = nn.Linear(5, 10)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

            # Initialize weights to negative values
            with torch.no_grad():
                self.fc1.weight.fill_(-10.0)
                self.fc1.bias.fill_(-10.0)

        def forward(self, x):
            z = self.fc1(x)
            a = self.relu(z)
            s = self.sigmoid(z)
            return a + s

    net = DiagnosticNet()
    x = torch.ones(2, 5)  # All positive input, but weights are -10 -> z will be deeply negative
    data, summary, notes = capture_activations(net, x)

    assert len(data["layers"]) >= 1
    # Check that dead units were detected
    assert any(
        "dead unit" in note.lower() or "dead" in str(layer.get("saturation_note", "")).lower()
        for note in notes
        for layer in data["layers"]
    )


def test_activations_feature_maps_for_conv2d():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)

        def forward(self, x):
            return self.conv(x)

    model = ConvNet()
    x = torch.randn(1, 3, 16, 16)
    view = sd.activations(model, input=x, backend="svg")

    assert view.kind == "activations"
    conv_layer = next(
        layer
        for layer in view.data["layers"]
        if "conv" in layer["name"].lower() or "conv" in layer["type"].lower()
    )
    # Should contain feature_maps grid
    assert conv_layer.get("feature_maps") is not None


def test_gradients_diagnostics_and_state_preservation():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class LinearNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 4)
            self.fc2 = nn.Linear(4, 1)

        def forward(self, x):
            return self.fc2(self.fc1(x))

    model = LinearNet()
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()

    # Artificially set fc1 grad to tiny value to test vanishing alert
    with torch.no_grad():
        model.fc1.weight.grad.fill_(1e-9)

    orig_fc2_grad = model.fc2.weight.grad.clone()

    data, summary, notes = inspect_gradients(model)

    assert data["present_count"] > 0
    # Original grad must be preserved
    assert torch.equal(model.fc2.weight.grad, orig_fc2_grad)
    # Vanishing gradient note should appear
    assert any("vanishing" in note.lower() for note in notes)


def test_gradients_zero_and_missing():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    model = nn.Linear(5, 2)
    # No backward called yet
    data, summary, notes = inspect_gradients(model)
    assert data["present_count"] == 0
    assert any(
        "no parameter gradients" in note.lower() or "no gradients" in note.lower() for note in notes
    )

    # Set grad to zeros
    model.weight.grad = torch.zeros_like(model.weight)
    data2, summary2, notes2 = inspect_gradients(model)
    assert data2["zero_count"] >= 1
    assert any("zero" in note.lower() for note in notes2)


def test_gradients_l2_norm_and_nonfinite_alert():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    model = nn.Linear(3, 1, bias=False)
    # Norm of [3.0, 4.0, 0.0] is 5.0
    model.weight.grad = torch.tensor([[3.0, 4.0, 0.0]])

    data, summary, notes = inspect_gradients(model)
    row = data["layers"][0]
    assert row["norm"] is not None
    assert abs(row["norm"] - 5.0) < 1e-4
    assert row["trainable"] is True

    # Test non-finite gradient note
    model.weight.grad = torch.tensor([[float("nan"), 1.0, 0.0]])
    data_nan, _, notes_nan = inspect_gradients(model)
    assert any("non-finite" in note.lower() for note in notes_nan)


def test_dead_unit_heuristics_do_not_flag_non_activation_layers():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    # A Linear layer whose weights happen to be mostly zero (e.g. pruned/sparse)
    class PrunedNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.fc(x))

    model = PrunedNet()
    # Positive inputs, normal ReLU activation
    x = torch.ones(2, 10) * 5.0
    data, _, notes = capture_activations(model, x)

    # Should not flag fc as dead units if fc is not an activation layer
    assert not any("fc" in note for note in notes if "dead unit" in note.lower())
