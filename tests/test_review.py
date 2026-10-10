"""Regression coverage for defects reproduced during the release review."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import pytest

import seedl as sd
from seedl.exceptions import SeeDLError


def test_sampled_vectors_keep_original_zero_based_indices() -> None:
    for obj in (
        list(range(100)),
        pytest.importorskip("numpy").arange(100),
        pytest.importorskip("torch").arange(100),
    ):
        view = sd.see(obj, backend="svg", max_points=4)
        assert view.data["x"] == [0, 25, 50, 75]
        figure = view.to_figure("plotly")
        assert list(figure.data[0].x) == [0, 25, 50, 75]
        assert list(view.to_figure("matplotlib").axes[0].lines[0].get_xdata()) == [0, 25, 50, 75]


@pytest.mark.parametrize("framework", ["numpy", "torch", "python"])
def test_matrix_preview_bounds_and_source_coordinates(framework: str) -> None:
    values = [[float(i * 16 + j) for j in range(16)] for i in range(20)]
    obj = values
    if framework == "numpy":
        obj = pytest.importorskip("numpy").array(values)
    elif framework == "torch":
        obj = pytest.importorskip("torch").tensor(values)
    view = sd.see(obj, max_points=7, backend="svg")
    assert sum(map(len, view.data["values"])) <= 7
    for i, row in enumerate(view.data["values"]):
        for j, value in enumerate(row):
            assert value == values[view.data["row_indices"][i]][view.data["column_indices"][j]]


@pytest.mark.parametrize(
    "values,mean,std",
    [
        ([1e300, 2e300], 1.5e300, 5e299),
        ([1e-300, 2e-300], 1.5e-300, 5e-301),
        ([-1e308, 1e308], 0.0, 1e308),
    ],
)
def test_numpy_extreme_finite_statistics(values: list[float], mean: float, std: float) -> None:
    np = pytest.importorskip("numpy")
    with np.errstate(over="raise", invalid="raise"):
        view = sd.see(np.array(values), backend="svg")
    assert float(view.metadata["Mean"]) == pytest.approx(mean, abs=0)
    assert float(view.metadata["Std. deviation"]) == pytest.approx(std, abs=0)
    root = ET.fromstring(view.figure)
    assert "nan" not in str(root.attrib).lower()
    for node in root.iter():
        for attribute in ("points", "x", "y", "cx", "cy"):
            assert "nan" not in node.get(attribute, "").lower()
            assert "inf" not in node.get(attribute, "").lower()


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_image_channels_and_plotly_color_bounds(channels: int) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("plotly")
    view = sd.see(np.ones((16, 16, channels)), backend="plotly", max_points=17)
    assert len(view.data["pixels"]) * len(view.data["pixels"][0]) <= 17
    trace = view.figure.data[0]
    if channels == 1:
        assert trace.type == "heatmap"
        assert trace.zmin == 0 and trace.zmax == 1
    else:
        assert list(trace.zmax) == [1] * 4
    if channels == 4:
        assert trace.colormodel == "rgba"
        assert 'fill-opacity="1.0"' in view.to_figure("svg")


def test_numpy_and_torch_image_scaling_agree() -> None:
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    for values in (
        np.full((16, 16, 3), 1, dtype=np.uint8),
        np.linspace(0, 2, 16 * 16 * 3).reshape(16, 16, 3),
        np.linspace(-1e300, 1e300, 16 * 16 * 3).reshape(16, 16, 3),
    ):
        a = sd.see(values, backend="svg").data["pixels"]
        b = sd.see(torch.from_numpy(values), backend="svg").data["pixels"]
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8)


def test_history_sampling_retains_epochs_and_distinct_legends() -> None:
    view = sd.see(
        {
            "epochs": list(range(10, 110)),
            "loss": list(range(100)),
            "val_loss": list(range(100)),
            "validation_loss": list(range(100)),
        },
        max_points=4,
        backend="svg",
    )
    series = view.data["groups"][0]["series"]
    assert [s["name"] for s in series] == ["loss", "val_loss", "validation_loss"]
    assert all(s["x"] == [10, 35, 60, 85] for s in series)
    assert all(len(s["values"]) <= 4 for s in series)
    with pytest.raises(SeeDLError, match="finite"):
        sd.see({"epochs": [1, math.nan], "loss": [1, 2]}, backend="svg")


def test_overfitting_explanation_requires_matching_epoch_coordinates() -> None:
    view = sd.see(
        {"epochs": [10, 20, 30, 40], "loss": [1.0, 0.6, 0.3, 0.2], "val_loss": [1.1, 0.5, 0.9]},
        backend="svg",
    )
    assert not any("Possible overfitting" in n for n in view.notes)


def test_svg_exposes_metadata_and_accessible_description() -> None:
    np = pytest.importorskip("numpy")
    view = sd.see(np.array([[1.0, math.nan], [-1.0, 3.0]]), backend="svg")
    root = ET.fromstring(view.figure)
    ns = {"s": "http://www.w3.org/2000/svg"}
    assert root.find("s:title", ns) is not None
    assert "Non-finite values: 1" in root.findtext("s:desc", namespaces=ns, default="")
    text = " ".join(n.text or "" for n in root.findall(".//s:text", ns))
    assert "Shape: (2, 2)" in text
    assert "Color scale:" in text
    assert "Non-finite values: 1" in text


@pytest.mark.parametrize("backend", ["svg", "plotly", "matplotlib"])
def test_missing_gradient_status_and_shape_are_visible(backend: str) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("plotly" if backend == "plotly" else "matplotlib")
    model = torch.nn.Linear(3, 2)
    model.weight.grad = torch.ones_like(model.weight)
    view = sd.gradients(model, backend=backend)
    figure = view.figure
    if backend == "svg":
        assert "no gradient" in figure and "(2,)" in figure
    elif backend == "plotly":
        table = next(trace for trace in figure.data if trace.type == "table")
        assert "no gradient" in table.cells.values[2]
        assert "(2,)" in table.cells.values[1]
    else:
        text = " ".join(
            c.get_text().get_text()
            for table in figure.axes[-1].tables
            for c in table.get_celld().values()
        )
        assert "no gradient" in text and "(2,)" in text


def test_empty_activation_outputs_and_resized_buffer_are_restored() -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("state", torch.tensor([1.0]))

        def forward(self, x):
            self.state.resize_(3).fill_(9)
            self.register_buffer("temporary", torch.ones(1), persistent=False)
            return x[:0]

    model = Model()
    original = model.state
    view = sd.activations(model, input=torch.ones(2), backend="svg")
    assert view.data["layers"][0]["sample"] == []
    assert model.state is original
    assert model.state.tolist() == [1.0]
    assert model._non_persistent_buffers_set == set()
    assert not model._forward_hooks


def test_backend_errors_do_not_execute_model() -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        calls = 0

        def forward(self, x):
            self.calls += 1
            return x

    model = Model()
    with pytest.raises(ValueError, match="backend"):
        sd.see(model, input=torch.ones(1), backend="unknown")
    assert model.calls == 0


def test_custom_forward_side_effects_are_not_sandboxed() -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        calls = 0

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):
            self.calls += 1
            self.weight.add_(1)
            return x + self.weight

    model = Model()
    sd.see(model, input=torch.ones(1), backend="svg")
    assert model.calls == 1
    assert model.weight.item() == 2


def test_large_model_all_backends_disclose_row_limit() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(*[torch.nn.ReLU() for _ in range(105)])
    view = sd.see(model, backend="svg")
    assert len(view.data["layers"]) == 106
    assert "Showing first 100 of 106" in view.figure
    assert len(view.to_figure("plotly").data[0].cells.values[0]) == 100
    figure = view.to_figure("matplotlib")
    assert figure.get_figheight() < 40


def test_matplotlib_notebook_output_is_png_without_html_override() -> None:
    pytest.importorskip("matplotlib")
    view = sd.see([1, 2, 3], backend="matplotlib")
    assert view._repr_mimebundle_()["image/png"].startswith(b"\x89PNG")
    assert view._repr_html_() is None


def test_plotly_html_formatter_does_not_replace_plotly_mime_output() -> None:
    pytest.importorskip("plotly")
    view = sd.see([1, 2, 3], backend="plotly")
    assert view._repr_html_() is None
    bundle = view._repr_mimebundle_()
    assert "application/vnd.plotly.v1+json" in bundle or "text/html" in bundle


def test_plain_numeric_calls_do_not_load_frameworks() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import seedl as sd, sys; sd.see(.7, backend='svg'); "
            "sd.see([1,2,3], backend='svg'); sd.see({'loss':[1.,.5]}, backend='svg'); "
            "assert 'torch' not in sys.modules and 'numpy' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_torch_high_rank_statistics_chunks_respect_memory_bound() -> None:
    torch = pytest.importorskip("torch")
    from seedl._inspect import _torch_chunks

    tensor = torch.arange(4096).reshape((4,) * 6).transpose(0, 1)
    chunks = list(_torch_chunks(tensor, chunk_size=13))
    assert all(c.numel() <= 13 for c in chunks)
    assert sum(c.numel() for c in chunks) == tensor.numel()
    assert sum(c.sum().item() for c in chunks) == tensor.sum().item()


def test_numpy_high_rank_statistics_with_small_chunks() -> None:
    np = pytest.importorskip("numpy")
    from seedl._inspect import _numpy_statistics

    tensor = np.arange(4096).reshape((4,) * 6).transpose(1, 0, 2, 3, 4, 5)
    _, _, mean, std, nonfinite = _numpy_statistics(tensor, chunk_size=13)
    assert mean == pytest.approx(tensor.mean())
    assert std == pytest.approx(tensor.std())
    assert nonfinite == 0


def test_numeric_module_names_keep_separate_activation_distributions() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("plotly")
    model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.ReLU())
    view = sd.activations(model, input=torch.tensor([-2.0, -1.0, 1.0, 2.0]), backend="plotly")
    trace = view.figure.data[0]
    relu_values = [y for x, y in zip(trace.x, trace.y, strict=True) if x == "1"]
    assert relu_values and min(relu_values) >= 0
    assert view.figure.layout.xaxis.type == "category"
