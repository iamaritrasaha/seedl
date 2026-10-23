"""Independent mathematical and rendering acceptance for model visualizations."""

from __future__ import annotations

import importlib.util

import pytest

import seedl as sd
from seedl.exceptions import MissingDependencyError, SeeDLError

torch = pytest.importorskip("torch")
nn = torch.nn


def tiny():
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, -2.0], [0.0, 3.0], [-1.0, 2.0]]))
        model[0].bias.copy_(torch.tensor([0.5, -1.0, 0.0]))
        model[2].weight.copy_(torch.tensor([[2.0, -1.0, 0.5]]))
        model[2].bias.fill_(0.25)
    return model


def graph(model=None, **kwargs):
    return sd.see(tiny() if model is None else model, backend="svg", **kwargs).data["neuron_graph"]


def test_width_count_weight_indexing_bias_and_default():
    g = graph()
    assert g["widths"] == [2, 3, 1]
    assert len(g["nodes"]) == 6
    assert len(g["edges"]) == 9
    mapping = {(e["source"], e["target"]): e["weight"] for e in g["edges"]}
    assert mapping == {
        ("0:0", "1:0"): 1.0,
        ("0:1", "1:0"): -2.0,
        ("0:0", "1:1"): 0.0,
        ("0:1", "1:1"): 3.0,
        ("0:0", "1:2"): -1.0,
        ("0:1", "1:2"): 2.0,
        ("1:0", "2:0"): 2.0,
        ("1:1", "2:0"): -1.0,
        ("1:2", "2:0"): 0.5,
    }
    assert [n["bias"] for n in g["layers"][1]["nodes"]] == [0.5, -1.0, 0.0]
    assert g["layers"][1]["operation"] == "ReLU()"
    assert all(n["activation"] is None for n in g["nodes"])


@pytest.mark.parametrize("batch,sample", [(1, 0), (3, 2), (7, 4)])
def test_real_post_activation_and_sample(batch, sample):
    # Each row differs; this catches flatten-and-take-first sampling bugs.
    x = torch.arange(batch * 2, dtype=torch.float32).reshape(batch, 2)
    g = graph(input=x, sample=sample)
    a, b = x[sample].tolist()
    z = [a - 2 * b + 0.5, 3 * b - 1, -a + 2 * b]
    relu = [max(0, v) for v in z]
    assert [n["activation"] for n in g["layers"][0]["nodes"]] == [a, b]
    assert [n["preactivation"] for n in g["layers"][1]["nodes"]] == z
    assert [n["activation"] for n in g["layers"][1]["nodes"]] == relu
    assert g["layers"][2]["nodes"][0]["activation"] == pytest.approx(
        2 * relu[0] - relu[1] + 0.5 * relu[2] + 0.25
    )


def test_unbatched_inplace_and_state():
    m = tiny()
    m[1] = nn.ReLU(inplace=True)
    m.train()
    m[1].eval()
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    saved = {k: v.clone() for k, v in m.state_dict().items()}
    modes = [v.training for v in m.modules()]
    for p in m.parameters():
        p.grad = torch.zeros_like(p)
    g = graph(m, input=x)
    assert [n["preactivation"] for n in g["layers"][1]["nodes"]] == [-2.5, 5.0, 3.0]
    assert [n["activation"] for n in g["layers"][1]["nodes"]] == [0.0, 5.0, 3.0]
    assert x.tolist() == [1.0, 2.0] and x.grad is None
    assert [v.training for v in m.modules()] == modes
    assert all(torch.equal(saved[k], v) for k, v in m.state_dict().items())
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in m.parameters())
    assert all(not v._forward_hooks and not v._forward_pre_hooks for v in m.modules())


def test_shared_occurrences_capture_distinct_outputs_and_deduplicate_parameters():
    layer = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(2 * torch.eye(2))
    m = nn.Sequential(layer, nn.ReLU(), layer)
    v = sd.see(m, input=torch.tensor([1.0, 2.0]), backend="svg")
    g = v.data["neuron_graph"]
    assert v.data["parameters"] == 4
    assert g["widths"] == [2, 2, 2]
    assert [n["activation"] for n in g["layers"][1]["nodes"]] == [2.0, 4.0]
    assert [n["activation"] for n in g["layers"][2]["nodes"]] == [4.0, 8.0]
    assert len({e["parameter"] for e in g["edges"]}) == 1


def test_frozen_and_missing_versus_zero_gradients():
    m = tiny()
    m[0].weight.requires_grad_(False)
    m[0].bias.requires_grad_(False)
    m[2].weight.grad = torch.zeros_like(m[2].weight)
    m[2].bias.grad = torch.zeros_like(m[2].bias)
    g = graph(m, color_by="gradient")
    assert all(e["gradient"] is None and not e["trainable"] for e in g["edges"] if e["layer"] == 1)
    assert all(
        e["gradient"] == 0 and e["gradient_status"] == "available"
        for e in g["edges"]
        if e["layer"] == 2
    )
    assert g["layers"][1]["nodes"][0]["gradient_status"] == "missing"
    assert g["layers"][2]["nodes"][0]["gradient"] == 0


def test_limits_filtering_and_deterministic_layout():
    a = graph(max_connections=4)
    b = graph(max_connections=4)
    assert a == b and len(a["edges"]) <= 4 and a["sampled"]
    assert all(abs(e["weight"]) >= 2 for e in graph(min_weight=2)["edges"])
    v = sd.see(nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 10)), backend="svg")
    assert "neuron_graph" not in v.data
    assert v.data["presentation"] == "architecture"
    assert "collapsed" in str(v)
    assert len(v.figure) < 20000
    assert "neuron_graph" not in sd.see(tiny(), max_neurons=3, backend="svg").data


def test_dense_sampling_exact_mapping():
    m = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 16))
    v = sd.see(m, max_connections=50, backend="svg")
    g = v.data["neuron_graph"]
    assert len(g["edges"]) <= 50 and g["total_connections"] == 1536
    for e in g["edges"]:
        layer = m[0] if e["layer"] == 1 else m[2]
        i, j = int(e["source"].split(":")[1]), int(e["target"].split(":")[1])
        assert e["weight"] == layer.weight[j, i].item()
    assert "sampled" in v.figure


def test_scatter3d_real_depth_and_export(tmp_path):
    pytest.importorskip("plotly")
    v = sd.see(tiny(), view="3d", input=torch.ones(2), theme="dark")
    assert v.backend == "plotly"
    assert all(t.type == "scatter3d" for t in v.figure.data)
    nodes = [t for t in v.figure.data if t.meta["role"] == "nodes"]
    assert len({z for t in nodes for z in t.z}) > 1
    assert v.figure.layout.scene.dragmode == "orbit"
    assert v.figure.layout.scene.camera.eye.y < 0
    path = v.save(tmp_path / "3d.html")
    text = path.read_text(encoding="utf-8")
    assert '"type":"scatter3d"' in text and '"scrollZoom": true' in text
    assert '<script src="https://cdn.plot.ly' not in text
    assert "2D projection" in v.to_figure("svg")
    fig = v.explore()
    assert len(fig.layout.sliders[0].steps) == 4
    assert (
        fig.layout.sliders[0].steps[2].args[0]["opacity"]
        != fig.layout.sliders[0].steps[0].args[0]["opacity"]
    )


def test_2d_svg_plotly_matplotlib():
    pytest.importorskip("plotly")
    plt = pytest.importorskip("matplotlib.pyplot")
    v = sd.see(tiny(), backend="svg")
    assert v.figure.count("<circle") == 12
    assert v.figure.count("<line ") == 9
    assert "Weight: -2" in v.figure
    assert all(t.type == "scatter" for t in v.to_figure("plotly").data)
    fig = v.to_figure("matplotlib")
    assert len(fig.axes[0].collections) == 2
    plt.close(fig)
    assert sd.see(tiny(), view="summary", backend="plotly").figure.data[0].type == "table"


def test_plotly_absent_keeps_svg_and_actionable_3d_error(monkeypatch):
    original = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: None if name == "plotly" else original(name)
    )
    assert sd.see(tiny()).backend == "svg"
    with pytest.raises(MissingDependencyError, match="seedl\\[plotly\\]"):
        sd.see(tiny(), view="3d")


def test_unsupported_residual_and_conv_hierarchy():
    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 2)

        def forward(self, x):
            return self.fc(x) + x

    v = sd.see(Residual(), input=torch.ones(2), backend="svg")
    assert "neuron_graph" not in v.data
    assert not v.data["architecture"]["sequential"]
    assert "containment" in v.figure
    cnn = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(), nn.Flatten(), nn.Linear(16, 2))
    v = sd.see(cnn, input=torch.ones(1, 3, 4, 4), backend="svg")
    assert v.data["architecture"]["sequential"]
    assert v.data["architecture"]["blocks"][0]["shape"] == "(1, 4, 2, 2)"
    assert "out_channels=4" in v.figure


@pytest.mark.parametrize(
    "options",
    [
        {"sample": -1},
        {"max_connections": 0},
        {"max_neurons": True},
        {"min_weight": float("nan")},
        {"edge_opacity": 2},
        {"theme": "pink"},
        {"view": "fake"},
        {"color_by": "fake"},
    ],
)
def test_invalid_options(options):
    with pytest.raises(ValueError):
        sd.see(tiny(), backend="svg", **options)


def test_capture_error_cleanup_and_wrong_dimensions():
    m = tiny()
    for x, sample in [(torch.ones(2, 2), 2), (torch.ones(1, 2, 2), 0)]:
        with pytest.raises(SeeDLError):
            sd.see(m, input=x, sample=sample, backend="svg")
        assert m.training
        assert all(not mod._forward_hooks for mod in m.modules())


def test_nonfinite_values_are_not_zero():
    m = tiny()
    with torch.no_grad():
        m[0].weight[0, 0] = float("nan")
    v = sd.see(m, input=torch.ones(2), backend="svg")
    assert v.data["neuron_graph"]["nonfinite_weights"] == 1
    n = v.data["neuron_graph"]["layers"][1]["nodes"][0]
    assert n["activation"] is None and n["activation_status"] == "non-finite"
    assert "Non-finite" in v.figure


def test_nested_sequential_and_overridden_linear_not_invented():
    m = nn.Sequential(nn.Sequential(nn.Linear(2, 3), nn.Tanh()), nn.Linear(3, 1))
    assert graph(m)["widths"] == [2, 3, 1]

    class Other(nn.Linear):
        def forward(self, x):
            return x

    assert "neuron_graph" not in sd.see(Other(2, 2), backend="svg").data


def test_standalone_linear_keyword_example_input_keeps_existing_call_contract():
    m = nn.Linear(2, 1)
    with torch.no_grad():
        m.weight.copy_(torch.tensor([[2.0, -1.0]]))
        m.bias.fill_(0.5)
    g = graph(m, input={"input": torch.tensor([[3.0, 1.0]])})
    assert [n["activation"] for n in g["layers"][0]["nodes"]] == [3.0, 1.0]
    assert g["layers"][1]["nodes"][0]["activation"] == 5.5
    assert not m._forward_hooks


def test_forward_readout_uses_captured_values_with_fixed_scale():
    pytest.importorskip("plotly")
    v = sd.see(tiny(), input=torch.tensor([1.0, 2.0]), backend="plotly")
    figure = v.explore()
    slider = figure.layout.sliders[0]
    assert slider.y > 1  # Controls stay above the graph, not below the readout.
    hidden = slider.steps[2]
    detail = hidden.args[1]["annotations"][-1]["text"]
    assert "Input a = [1, 2]" in detail
    assert "Pre-activation z = [-2.5, 5, 3]" in detail
    assert "Output = [0, 5, 3]" in detail
    assert "ReLU()" in detail
    assert set(hidden.args[0]) == {"opacity"}  # No altered values, scales or legend state.
    for trace, opacity in zip(figure.data, hidden.args[0]["opacity"], strict=True):
        if trace.meta["role"] == "edges":
            assert opacity == (0.8 if trace.meta["layer"] == 1 else 0.04)
    reset = slider.steps[0].args[0]["opacity"]
    assert all(o == 1 for t, o in zip(figure.data, reset, strict=True) if t.meta["role"] == "nodes")


def test_no_input_explore_is_explicit_about_unavailable_values():
    pytest.importorskip("plotly")
    figure = sd.see(tiny(), backend="plotly").explore()
    assert not figure.layout.sliders
    assert any("Values unavailable" in a.text for a in figure.layout.annotations)
    nodes = [t for t in figure.data if t.meta["role"] == "nodes"]
    assert "Input feature: unavailable" in nodes[0].hovertext[0]
    assert "sample 0" not in nodes[0].hovertext[0]


def test_3d_guides_camera_controls_and_legend_filtering_contract():
    pytest.importorskip("plotly")
    v = sd.see(tiny(), view="3d")
    figure = v.figure
    guides = [t for t in figure.data if t.meta["role"] == "guide"]
    assert len(guides) == 3
    assert all(t.hoverinfo == "skip" and not t.showlegend for t in guides)
    assert all(len(set(t.x)) == 1 and t.line.dash == "dot" for t in guides)
    buttons = figure.layout.updatemenus[0].buttons
    assert [b.label for b in buttons] == ["Orbit", "Pan", "Reset view"]
    assert buttons[-1].args[0]["scene.camera"] == figure.layout.scene.camera.to_plotly_json()
    assert figure.layout.legend.groupclick == "togglegroup"
    edges = [t for t in figure.data if t.meta["role"] == "edges"]
    assert {t.legendgroup for t in edges} == {
        "Positive weights",
        "Negative weights",
        "Zero weights",
    }
    assert len([t for t in edges if t.showlegend]) == 3
    assert sum(sum(x is None for x in t.x) for t in edges) == 9


def test_wide_forward_readout_is_bounded_and_honest():
    pytest.importorskip("plotly")
    model = nn.Linear(16, 16)
    fig = sd.see(model, input=torch.arange(16, dtype=torch.float32)).explore()
    text = fig.layout.sliders[0].steps[1].args[1]["annotations"][-1]["text"]
    assert "[0, 1, 2, 3, 4, 5, … (16 total; hover for each value)]" in text


def test_neuron_selection_validation():
    # Valid formats
    v1 = sd.see(tiny(), select_neuron=(1, 0))
    assert v1.data["neuron_graph"]["selected_neuron"] == "1:0"

    v2 = sd.see(tiny(), select_neuron="1:0")
    assert v2.data["neuron_graph"]["selected_neuron"] == "1:0"

    # Out of bounds layer
    with pytest.raises(ValueError, match="layer index 5 is out of bounds"):
        sd.see(tiny(), select_neuron=(5, 0))

    # Out of bounds neuron
    with pytest.raises(ValueError, match="neuron index 9 is out of bounds"):
        sd.see(tiny(), select_neuron=(0, 9))

    # Invalid formats
    with pytest.raises(ValueError, match="select_neuron must be"):
        sd.see(tiny(), select_neuron="invalid")

    with pytest.raises(ValueError, match="must be integers"):
        sd.see(tiny(), select_neuron=(True, 0))


def test_neuron_selection_graph_data_and_connections():
    x = torch.tensor([1.0, 2.0])
    v = sd.see(tiny(), input=x, select_neuron=(1, 0))
    g = v.data["neuron_graph"]
    assert g["selected_neuron"] == "1:0"

    node = next(n for n in g["nodes"] if n["id"] == "1:0")
    assert len(node["incoming_edges"]) == 2
    assert [e["source"] for e in node["incoming_edges"]] == ["0:0", "0:1"]
    assert len(node["outgoing_edges"]) == 1
    assert [e["target"] for e in node["outgoing_edges"]] == ["2:0"]


def test_neuron_selection_plotly_traces_and_dropdown():
    pytest.importorskip("plotly")
    x = torch.tensor([1.0, 2.0])
    v = sd.see(tiny(), input=x, select_neuron=(1, 0))
    fig = v.figure

    roles = [t.meta.get("role") for t in fig.data if t.meta]
    assert "highlight_incoming" in roles
    assert "highlight_outgoing" in roles
    assert "highlight_node" in roles

    # In 2D, updatemenus[0] is the neuron selection dropdown
    neuron_menu = fig.layout.updatemenus[0]
    button_labels = [b.label for b in neuron_menu.buttons]
    assert button_labels[0] == "Overview (All Neurons)"
    assert any("[1:0]" in lbl for lbl in button_labels)

    # Footer inspection card
    footer_text = fig.layout.annotations[-1].text
    assert "NEURON INSPECTION" in footer_text
    assert "Neuron #0" in footer_text
    assert "Incoming" in footer_text
    assert "Outgoing" in footer_text


def test_neuron_selection_3d_contract():
    pytest.importorskip("plotly")
    x = torch.tensor([1.0, 2.0])
    v = sd.see(tiny(), input=x, view="3d", select_neuron=(1, 0))
    fig = v.figure

    # In 3D, updatemenus[0] MUST remain Camera controls for backward compatibility
    assert [b.label for b in fig.layout.updatemenus[0].buttons] == ["Orbit", "Pan", "Reset view"]
    # updatemenus[1] is the neuron selection dropdown
    assert fig.layout.updatemenus[1].buttons[0].label == "Overview (All Neurons)"

    # Highlight traces in 3D must have 3D coordinates
    hl_node = next(t for t in fig.data if t.meta and t.meta.get("role") == "highlight_node")
    assert hasattr(hl_node, "z")
    assert len(hl_node.z) == 1


def test_forward_animation_frames_and_playback_controls():
    pytest.importorskip("plotly")
    x = torch.tensor([1.0, 2.0])
    fig = sd.see(tiny(), input=x).explore()

    # Frames for playback
    assert len(fig.frames) == 4  # reset + 3 layers
    frame_names = [f.name for f in fig.frames]
    assert frame_names == ["reset", "step_0", "step_1", "step_2"]

    # Play/Pause/Reset controls
    menu_button_labels = [[b.label for b in m.buttons] for m in fig.layout.updatemenus]
    flat_labels = [lbl for sublist in menu_button_labels for lbl in sublist]
    assert "▶ Play" in flat_labels
    assert "⏸ Pause" in flat_labels
    assert "⏮ Reset" in flat_labels
    assert "1x" in flat_labels
    assert "2x" in flat_labels
    assert "0.5x" in flat_labels


def test_svg_selected_neuron_rendering():
    x = torch.tensor([1.0, 2.0])
    v = sd.see(tiny(), input=x, select_neuron=(1, 0), backend="svg")
    svg_str = v.to_figure("svg")

    assert "#00e5ff" in svg_str  # Cyan incoming highlight
    assert "#fbbf24" in svg_str  # Amber outgoing highlight
    assert "Selected Neuron [1:0]" in svg_str
    assert "Incoming (cyan): 2" in svg_str
    assert "Outgoing (amber): 1" in svg_str


def test_explore_with_select_neuron():
    pytest.importorskip("plotly")
    x = torch.tensor([1.0, 2.0])
    fig = sd.see(tiny(), input=x).explore(select_neuron=(1, 0))

    # Should have both explore frames and selected neuron highlight
    assert len(fig.frames) == 4
    roles = [t.meta.get("role") for t in fig.data if t.meta]
    assert "highlight_node" in roles
    hl_node = next(t for t in fig.data if t.meta and t.meta.get("role") == "highlight_node")
    assert hl_node.visible is True
