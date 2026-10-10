"""Tests for advanced 3D neural network features, direct neuron selection, themes, and CNN 3D."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

import seedl as sd
from seedl._network_renderers import build_network_interactive_js


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        return x


def test_3d_nodes_have_customdata_ids():
    """Verify that all 3D node markers have customdata with exact neuron IDs for click interaction."""
    pytest.importorskip("plotly")
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
    v = sd.see(model, view="3d")
    fig = v.figure

    node_traces = [t for t in fig.data if t.meta and t.meta.get("role") == "nodes"]
    assert len(node_traces) > 0
    for trace in node_traces:
        assert hasattr(trace, "customdata")
        assert trace.customdata is not None
        assert len(trace.customdata) == len(trace.x)
        for nid in trace.customdata:
            assert isinstance(nid, str)
            assert ":" in nid


def test_build_network_interactive_js():
    """Verify client-side interactive JavaScript generator."""
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
    x = torch.tensor([[1.0, 2.0]])
    v = sd.see(model, input=x, view="3d")
    graph = v.data["neuron_graph"]
    js_code = build_network_interactive_js(graph, spatial=True)

    assert "plotly_click" in js_code
    assert "__seedl_select_neuron" in js_code
    assert "__seedl_clear_selection" in js_code
    assert "nodeMap" in js_code
    assert "0:0" in js_code


def test_cnn_3d_feature_map_visualization():
    """Verify 3D CNN spatial feature map extraction and rendering."""
    pytest.importorskip("plotly")
    model = SmallCNN()
    img = torch.randn(1, 1, 16, 16)

    v = sd.see(model, input=img, view="3d")
    assert v.data.get("presentation") == "cnn_3d"
    assert v.data.get("dimensions") == 3
    assert len(v.data["layers"]) > 0

    # Ensure intermediate feature maps are captured
    captured_layers = [lyr for lyr in v.data["layers"] if lyr.get("captured", {}).get("channels")]
    assert len(captured_layers) >= 2

    # Check Plotly figure
    fig = v.figure
    assert fig is not None
    assert len(fig.data) > 0


@pytest.mark.parametrize("theme_name", ["scientific", "dark", "light", "minimal", "presentation"])
def test_all_themes_supported_on_3d_network(theme_name):
    """Verify all 5 supported visual themes on 3D neural network."""
    pytest.importorskip("plotly")
    model = nn.Sequential(nn.Linear(2, 2))
    v = sd.see(model, view="3d", theme=theme_name)
    assert v.data["neuron_graph"]["theme"] == theme_name
    fig = v.figure
    assert fig is not None


def test_html_export_embeds_interactive_js():
    """Verify that saving a 3D network to HTML embeds the interactive JS post_script."""
    pytest.importorskip("plotly")
    model = nn.Sequential(nn.Linear(2, 2))
    v = sd.see(model, view="3d")

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        v.save(tmp_path)
        content = open(tmp_path, encoding="utf-8").read()
        assert "plotly_click" in content
        assert "__seedl_select_neuron" in content
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_visualization_to_html_method():
    """Verify Visualization.to_html() returns a complete interactive HTML string."""
    pytest.importorskip("plotly")
    model = nn.Sequential(nn.Linear(2, 2))
    v = sd.see(model, view="3d")
    html_str = v.to_html()
    assert isinstance(html_str, str)
    assert "<html" in html_str.lower()
    assert "plotly_click" in html_str
