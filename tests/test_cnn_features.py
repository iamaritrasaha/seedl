"""Scientific correctness and integration tests for CNN Feature Map Explorer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import seedl as sd
from seedl._features import capture_feature_maps, compare_feature_maps
from seedl.exceptions import SeeDLError


class CustomConvNet(nn.Module):
    """Custom model with multiple conv layers and reused module."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.reused_conv = nn.Conv2d(4, 4, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.reused_conv(h)  # Call 1
        h = self.reused_conv(h)  # Call 2 (reused)
        h = self.pool(h)
        return h


def test_cnn_known_weights_and_exact_activation_values():
    # Construct Conv2d with deterministic known weights
    conv = nn.Conv2d(1, 1, kernel_size=2, bias=False)
    with torch.no_grad():
        conv.weight.copy_(torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]))

    # Known input: 1x1x3x3
    x = torch.tensor([[[[1.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 1.0]]]])

    # Ground truth manual convolution output: (3 - 2 + 1) = 2x2
    # top-left: 1*1 + 1*2 + 0*3 + 2*4 = 1 + 2 + 0 + 8 = 11
    # top-right: 1*1 + 0*2 + 2*3 + 1*4 = 1 + 0 + 6 + 4 = 11
    # bottom-left: 0*1 + 2*2 + 1*3 + 0*4 = 0 + 4 + 3 + 0 = 7
    # bottom-right: 2*1 + 1*2 + 0*3 + 1*4 = 2 + 2 + 0 + 4 = 8
    expected = [[11.0, 11.0], [7.0, 8.0]]

    data, summary, notes = capture_feature_maps(conv, example_input=x)

    assert len(data["layers"]) == 1
    layer = data["layers"][0]
    grid = layer["channels"][0]["grid"]

    assert len(grid) == 2
    assert len(grid[0]) == 2
    for r in range(2):
        for c in range(2):
            assert abs(grid[r][c] - expected[r][c]) < 1e-4

    # Verify channel statistics
    stats = layer["channels"][0]["stats"]
    assert stats["min"] == 7.0
    assert stats["max"] == 11.0
    assert stats["mean"] == (11.0 + 11.0 + 7.0 + 8.0) / 4.0
    assert stats["has_negatives"] is False


def test_cnn_sequential_model_and_layer_names():
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
    )
    x = torch.randn(2, 3, 16, 16)

    view = sd.features(model, input=x)
    assert view.kind == "feature_maps"
    assert "layers" in view.data

    captured = view.data["layers"]
    # Conv2d, ReLU, MaxPool2d, Conv2d
    assert len(captured) >= 3

    layer_types = [lyr["type"] for lyr in captured]
    assert "Conv2d" in layer_types
    assert "MaxPool2d" in layer_types

    # First conv has 8 channels, second has 16 channels
    conv_layers = [lyr for lyr in captured if lyr["type"] == "Conv2d"]
    assert conv_layers[0]["total_channels"] == 8
    assert conv_layers[1]["total_channels"] == 16


def test_cnn_reused_module_calls_preservation():
    """Verify that multiple invocations of the same module in one forward pass are both captured."""
    model = CustomConvNet()
    x = torch.randn(1, 1, 8, 8)

    data, summary, notes = capture_feature_maps(model, example_input=x)
    names = [lyr["name"] for lyr in data["layers"]]

    # Both call 1 and call 2 of reused_conv must be present
    assert any("reused_conv" in n and "call 2" in n for n in names)
    assert any("reused_conv" in n for n in names)
    assert len(names) >= 4


def test_cnn_invalid_inputs_raise_seedl_error():
    """Verify appropriate SeeDLError is raised when model has no input or invalid dimensions."""
    model = nn.Conv2d(1, 1, 3)
    with pytest.raises(SeeDLError, match="input tensor"):
        sd.features(model)

    with pytest.raises(SeeDLError, match="3D .* or 4D"):
        sd.features(torch.tensor([1.0, 2.0]))


def test_cnn_model_state_preservation_and_hook_cleanup():
    """Verify model training mode is restored, hooks removed, and no autograd graph retained."""
    model = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1))
    model.train(True)
    assert model.training is True

    x = torch.randn(1, 1, 6, 6, requires_grad=True)

    data, summary, notes = capture_feature_maps(model, example_input=x)

    # 1. Training mode restored
    assert model.training is True

    # 2. Hooks completely removed
    assert len(model[0]._forward_hooks) == 0

    # 3. No autograd graph retained on input
    assert x.grad is None


def test_cnn_signed_and_negative_activations():
    """Verify that negative feature map values are properly tracked and marked."""
    conv = nn.Conv2d(1, 1, 1, bias=False)
    with torch.no_grad():
        conv.weight.fill_(-2.0)  # Negates all inputs

    x = torch.ones(1, 1, 4, 4)
    data, _, _ = capture_feature_maps(conv, example_input=x)

    stats = data["layers"][0]["channels"][0]["stats"]
    assert stats["min"] == -2.0
    assert stats["max"] == -2.0
    assert stats["has_negatives"] is True
    assert stats["positive_fraction"] == 0.0
    assert stats["negative_fraction"] == 1.0


def test_cnn_nonfinite_value_safety():
    """Verify non-finite values (NaN/Inf) do not crash feature capture and are reported."""
    x = torch.tensor([[[[1.0, float("nan")], [float("inf"), 2.0]]]])
    data, _, _ = capture_feature_maps(x)

    stats = data["layers"][0]["channels"][0]["stats"]
    assert stats["nonfinite_count"] == 2
    assert stats["min"] == 1.0
    assert stats["max"] == 2.0


def test_cnn_bounded_sampling_and_downsampling():
    """Verify high resolution maps are downsampled and channels capped at max_channels."""
    # Large 256x256 image with 10 channels
    large_tensor = torch.randn(1, 10, 256, 256)

    data, summary, notes = capture_feature_maps(large_tensor, max_channels=4, max_spatial_dim=32)

    layer = data["layers"][0]
    assert layer["total_channels"] == 10
    assert layer["captured_channels"] == 4
    assert layer["downsampled"] is True
    assert layer["channels"][0]["h"] == 32
    assert layer["channels"][0]["w"] == 32
    assert layer["original_spatial"] == (256, 256)
    assert any("downsampled" in n.lower() for n in notes)


def test_cnn_feature_comparison_same_shape():
    """Compare two feature maps of matching dimensions."""
    feat1 = torch.zeros(1, 1, 4, 4)
    feat2 = torch.ones(1, 1, 4, 4) * 2.0

    cmp_data, summary, notes = compare_feature_maps(feat1, feat2, names=("Zero", "Two"))

    assert cmp_data["shape_match"] is True
    assert cmp_data["diff_stats"] is not None
    assert abs(cmp_data["diff_stats"]["mae"] - 2.0) < 1e-4
    assert abs(cmp_data["diff_stats"]["max_abs_diff"] - 2.0) < 1e-4

    # Using sd.compare()
    v1 = sd.features(feat1)
    v2 = sd.features(feat2)
    view_cmp = sd.compare(v1, v2)
    assert view_cmp.kind == "feature_comparison"
    assert "<svg" in view_cmp.to_figure("svg")


def test_cnn_feature_comparison_mismatched_shape():
    """Compare two feature maps of different dimensions; verify no elementwise difference."""
    feat1 = torch.zeros(1, 1, 8, 8)
    feat2 = torch.zeros(1, 1, 4, 4)

    cmp_data, summary, notes = compare_feature_maps(feat1, feat2, names=("Layer1", "Layer2"))

    assert cmp_data["shape_match"] is False
    assert cmp_data["diff_grid"] is None
    assert any("mismatch" in n.lower() for n in notes)


def test_cnn_renderers_and_export():
    """Verify Plotly, SVG, Matplotlib, and HTML export for feature maps."""
    x = torch.randn(1, 2, 8, 8)
    view = sd.features(x, backend="svg")

    # SVG export
    svg_str = view.to_figure("svg")
    assert "<svg" in svg_str
    assert "FEATURE MAP" in svg_str

    # Plotly export
    fig = view.to_figure("plotly")
    assert fig is not None

    # Matplotlib export
    mpl_fig = view.to_figure("matplotlib")
    assert mpl_fig is not None

    # HTML export
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        out_path = view.save(tmp.name)
        content = Path(out_path).read_text(encoding="utf-8")
        assert "SeeDL · Vision Intelligence" in content
        assert "featureCanvas" in content
        assert "channelSelect" in content
