from __future__ import annotations

import importlib.util
import math
import subprocess
import sys

import pytest

import seedl as sd
from seedl.exceptions import SeeDLError, UnsupportedObjectError
from seedl.view import Visualization


def test_package_import_does_not_load_optional_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, seedl; names = ('torch', 'numpy', 'plotly', 'matplotlib'); "
            "assert not any(name in sys.modules for name in names)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_scalar_and_sequence_use_builtin_svg(capsys: pytest.CaptureFixture[str]) -> None:
    scalar = sd.see(0.73, backend="svg")
    sequence = sd.see([0.9, 0.62, 0.41], backend="svg")

    assert scalar.kind == "scalar"
    assert scalar.data["value"] == 0.73
    assert "Numeric value" in capsys.readouterr().out
    assert sequence.kind == "sequence"
    assert "3 value(s)" in str(sequence)
    assert sequence.to_figure().startswith("<svg")


def test_matrix_sequence_is_rectangular_and_numeric() -> None:
    matrix = sd.see([[1, 2], [3, 4]], backend="svg")
    assert matrix.kind == "matrix"
    assert matrix.data["shape"] == (2, 2)
    with pytest.raises(SeeDLError, match="rectangular"):
        sd.see([[1, 2], [3]], backend="svg")
    with pytest.raises(SeeDLError, match="real number"):
        sd.see([1, "two"], backend="svg")


def test_training_history_pairs_validation_and_accepts_unequal_lengths() -> None:
    view = sd.see(
        {
            "epochs": [1, 2, 3],
            "loss": [1.0, 0.6, 0.4],
            "val_loss": [1.2, 0.8],
            "accuracy": [0.4, 0.7, 0.9],
        },
        backend="svg",
    )

    assert view.kind == "training"
    assert len(view.data["groups"]) == 2
    loss_group = view.data["groups"][0]
    assert loss_group["name"] == "loss"
    assert [series["name"] for series in loss_group["series"]] == ["loss", "validation"]
    assert len(loss_group["series"][0]["values"]) == 3
    assert len(loss_group["series"][1]["values"]) == 2
    assert "index labels are used" in " ".join(view.notes)


def test_history_overfitting_note_requires_paired_loss_curves() -> None:
    view = sd.see(
        {"loss": [1.0, 0.7, 0.4, 0.3], "val_loss": [1.1, 0.8, 0.7, 0.9]},
        backend="svg",
    )
    assert any("Possible overfitting clue" in note for note in view.notes)
    short = sd.see({"loss": [1.0, 0.3], "val_loss": [1.1, 0.9]}, backend="svg")
    assert not any("Possible overfitting clue" in note for note in short.notes)


def test_history_nonfinite_values_are_reported_and_empty_history_errors() -> None:
    view = sd.see({"loss": [1.0, math.nan]}, backend="svg")
    assert any("Non-finite" in note for note in view.notes)
    with pytest.raises(SeeDLError, match="empty"):
        sd.see({}, backend="svg")
    with pytest.raises(SeeDLError, match="at least one value"):
        sd.see({"loss": []}, backend="svg")


def test_svg_line_chart_leaves_a_gap_for_nonfinite_metrics() -> None:
    view = sd.see({"loss": [1.0, math.nan, 0.2]}, backend="svg")
    svg = view.to_figure()
    assert svg.count("<polyline") == 2


def test_unknown_object_and_invalid_options_are_actionable() -> None:
    with pytest.raises(UnsupportedObjectError, match="Cannot visualize object"):
        sd.see(object(), backend="svg")
    with pytest.raises(ValueError, match="max_points"):
        sd.see([1, 2], max_points=0)
    with pytest.raises(ValueError, match="backend must"):
        sd.see([1, 2], backend="unknown")


def test_optional_backend_error_or_figure_creation() -> None:
    if importlib.util.find_spec("plotly") is None:
        with pytest.raises(ImportError, match=r"pip install 'seedl\[plotly\]'"):
            sd.see([1, 2], backend="plotly")
    else:
        view = sd.see([1, 2], backend="plotly")
        figure = view.to_figure()
        assert figure.data
        assert figure.data[0].type == "scatter"


def test_plotly_builds_semantic_views_when_available() -> None:
    if importlib.util.find_spec("plotly") is None:
        pytest.skip("Plotly is an optional dependency")
    matrix = sd.see([[1, 2], [3, 4]], backend="plotly").to_figure()
    history = sd.see(
        {"epochs": [1, 3, 6], "loss": [1.0, 0.5, 0.2], "val_loss": [1.2, 0.8, 0.7]},
        backend="plotly",
    ).to_figure()
    scalar = sd.see(math.inf, backend="plotly").to_figure()
    gradients = Visualization(
        kind="gradients",
        title="Gradient sample",
        summary="2 parameter tensors",
        data={
            "layers": [
                {"name": "weight", "max_abs": 0.7, "sample": [-0.2, 0.1, 0.7]},
                {"name": "bias", "max_abs": 0.1, "sample": [-0.1, 0.0, 0.1]},
            ]
        },
        backend="plotly",
    ).to_figure()
    assert matrix.data[0].type == "heatmap"
    assert list(history.data[0].x) == [1, 3, 6]
    assert not any(trace.type == "indicator" for trace in scalar.data)
    assert any(trace.type == "box" for trace in gradients.data)


def test_matplotlib_backend_when_available() -> None:
    if importlib.util.find_spec("matplotlib") is None:
        with pytest.raises(ImportError, match=r"seedl\[matplotlib\]"):
            sd.see([1, 2], backend="matplotlib")
    else:
        view = sd.see([1, 2], backend="matplotlib")
        figure = view.to_figure()
        assert len(figure.axes) == 1
        gradient_view = Visualization(
            kind="gradients",
            title="Gradient sample",
            summary="1 parameter tensor",
            data={"layers": [{"name": "weight", "max_abs": 0.7, "sample": [-0.2, 0.1, 0.7]}]},
            backend="matplotlib",
        )
        assert len(gradient_view.to_figure().axes) == 2


def test_numpy_arrays_and_tensor_values_are_not_modified() -> None:
    np = pytest.importorskip("numpy")
    original = np.array([[1.0, np.nan], [3.0, 4.0]])
    saved = original.copy()
    matrix = sd.see(original, backend="svg")
    assert matrix.kind == "matrix"
    assert matrix.metadata["Non-finite values"] == 1
    assert np.array_equal(original, saved, equal_nan=True)

    boolean = sd.see(np.array([True, False]), backend="svg")
    assert boolean.kind == "sequence"
    assert boolean.data["values"] == [True, False]


def test_numpy_statistics_process_large_arrays() -> None:
    np = pytest.importorskip("numpy")
    values = np.arange(1_050_000, dtype=np.float64)
    view = sd.see(values, backend="svg")
    assert view.kind == "sequence"
    assert float(view.metadata["Mean"]) == pytest.approx(525_000)
    assert view.metadata["Non-finite values"] == 0


def test_optional_numpy_image_scaling_is_explicit() -> None:
    np = pytest.importorskip("numpy")
    image = np.arange(3 * 32 * 32, dtype=np.uint8).reshape(3, 32, 32)
    original = image.copy()
    view = sd.see(image, backend="svg")
    assert view.kind == "image"
    assert view.data["channels"] == 3
    assert any("layout was inferred" in note for note in view.notes)
    assert np.array_equal(image, original)

    small_feature_map = sd.see(np.zeros((3, 8, 8)), backend="svg")
    assert small_feature_map.kind == "distribution"

    nonfinite_image = np.zeros((3, 32, 32), dtype=np.float32)
    nonfinite_image[0, 0, 0] = np.nan
    preview = sd.see(nonfinite_image, backend="svg")
    assert preview.kind == "image"
    assert preview.to_figure().startswith("<svg")


def test_torch_tensor_model_activation_and_gradient_paths() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    class CountingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 3)
            self.norm = nn.BatchNorm1d(3)
            self.calls = 0

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return self.norm(self.linear(value))

    model = CountingModel()
    model.train()
    model.norm.eval()
    model.linear.weight.grad = torch.ones_like(model.linear.weight)
    input_tensor = torch.randn(2, 4, requires_grad=True)
    input_copy = input_tensor.detach().clone()
    grad_copy = model.linear.weight.grad.clone()
    buffer_copies = [buffer.clone() for buffer in model.buffers()]
    training_states = [module.training for module in model.modules()]
    hook_counts = [
        (len(module._forward_hooks), len(module._forward_pre_hooks)) for module in model.modules()
    ]

    hierarchy = sd.see(model, backend="svg")
    assert hierarchy.kind == "model"
    assert model.calls == 0
    assert hierarchy.data["parameters"] == sum(p.numel() for p in model.parameters())

    inspected = sd.see(model, input=input_tensor, backend="svg")
    assert model.calls == 1
    assert any(row["output_shape"] != "—" for row in inspected.data["layers"])
    assert [module.training for module in model.modules()] == training_states
    assert [
        (len(module._forward_hooks), len(module._forward_pre_hooks)) for module in model.modules()
    ] == hook_counts
    assert torch.equal(input_tensor.detach(), input_copy)
    assert torch.equal(model.linear.weight.grad, grad_copy)
    assert all(
        torch.equal(buffer, saved)
        for buffer, saved in zip(model.buffers(), buffer_copies, strict=True)
    )

    activation_view = sd.activations(model, input=input_tensor, backend="svg")
    assert activation_view.kind == "activations"
    assert activation_view.data["layers"]
    assert [
        (len(module._forward_hooks), len(module._forward_pre_hooks)) for module in model.modules()
    ] == hook_counts

    gradient_view = sd.gradients(model, backend="svg")
    assert gradient_view.kind == "gradients"
    assert gradient_view.data["layers"][0]["status"] == "available"


def test_torch_forward_error_cleans_hooks_and_restores_mode() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    model = nn.Sequential(nn.Linear(4, 2))
    model.train()
    states = [module.training for module in model.modules()]
    with pytest.raises(SeeDLError, match="Check the input structure"):
        sd.see(model, input=torch.randn(2, 3), backend="svg")
    assert [module.training for module in model.modules()] == states
    assert all(
        not module._forward_hooks and not module._forward_pre_hooks for module in model.modules()
    )
