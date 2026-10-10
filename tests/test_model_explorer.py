"""Tests for model architecture inspection, parameter counting, and shape tracing."""

from __future__ import annotations

import pytest

import seedl as sd
from seedl._models import inspect_model_architecture


def test_model_architecture_parameter_counting():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)  # 10*20 + 20 = 220
            self.fc2 = nn.Linear(20, 5)  # 20*5 + 5 = 105
            # Freeze fc2
            for p in self.fc2.parameters():
                p.requires_grad = False

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    net = SimpleNet()
    data, summary, notes = inspect_model_architecture(net)

    assert data["parameters"] == 325
    assert data["trainable"] == 220
    assert data["frozen"] == 105
    assert any("frozen" in note for note in notes)


def test_model_shared_parameters_detection():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class SharedNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 8)
            self.fc2 = nn.Linear(8, 8)
            # Share weight matrix
            self.fc2.weight = self.fc1.weight

        def forward(self, x):
            return self.fc2(self.fc1(x))

    net = SharedNet()
    data, summary, notes = inspect_model_architecture(net)

    # 8*8 weight (shared) + 8 bias + 8 bias = 64 + 8 + 8 = 80 total deduplicated params
    assert data["parameters"] == 80
    assert len(data["shared_parameters"]) >= 1
    assert any("shared" in note.lower() for note in notes)


def test_model_shape_tracing_and_state_preservation():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class ResNetBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.shortcut = nn.Conv2d(3, 16, 1)

        def forward(self, x):
            return self.bn1(self.conv1(x)) + self.shortcut(x)

    model = ResNetBlock()
    model.train()  # Start in training mode
    orig_mean = model.bn1.running_mean.clone()

    x = torch.randn(2, 3, 16, 16)
    view = sd.see(model, input=x, backend="svg")

    assert view.kind == "model"
    # Mode must be preserved back to train
    assert model.training is True
    # Running mean must be preserved
    assert torch.equal(model.bn1.running_mean, orig_mean)
    # Check that shapes were captured
    assert any(row["input_shape"] != "—" for row in view.data["layers"])
    assert any(row["output_shape"] != "—" for row in view.data["layers"])
