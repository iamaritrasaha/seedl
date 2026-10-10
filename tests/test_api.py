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
    assert [series["name"] for series in loss_group["series"]] == ["loss", "val_loss"]
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


def test_plotly_notebook_mimebundle_uses_plotly_output() -> None:
    if importlib.util.find_spec("plotly") is None:
        pytest.skip("Plotly is an optional dependency")
    view = sd.see([1, 2, 3], backend="plotly")
    bundle = view._repr_mimebundle_()
    assert "text/html" in bundle or "application/vnd.plotly.v1+json" in bundle
    assert any("plotly" in str(value).lower() for value in bundle.values())


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
        assert len(gradient_view.to_figure().axes) == 3


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


def test_torch_nested_convolutional_model_shapes_and_state_are_preserved() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    class NestedConvModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(4),
                nn.ReLU(),
            )
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.head(self.encoder(value))

    model = NestedConvModel().train()
    model.encoder[1].eval()
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.25)
    example = torch.randn(2, 3, 20, 20, requires_grad=True)
    example_copy = example.detach().clone()
    parameter_copies = [parameter.detach().clone() for parameter in model.parameters()]
    gradient_copies = [parameter.grad.clone() for parameter in model.parameters()]
    buffer_copies = [buffer.detach().clone() for buffer in model.buffers()]
    training_states = [module.training for module in model.modules()]
    hook_counts = [
        (len(module._forward_hooks), len(module._forward_pre_hooks)) for module in model.modules()
    ]

    architecture = sd.see(model, input=example, backend="svg")
    rows = {row["name"]: row for row in architecture.data["layers"]}
    assert rows["encoder.0"]["input_shape"] == "((2, 3, 20, 20),)"
    assert rows["encoder.0"]["output_shape"] == "(2, 4, 20, 20)"
    assert rows["head.2"]["output_shape"] == "(2, 2)"

    activations = sd.activations(model, input=example, backend="svg", max_points=7)
    assert activations.data["layers"]
    assert sum(len(row["sample"]) for row in activations.data["layers"]) <= 7
    assert all(
        not isinstance(value, torch.Tensor)
        for row in activations.data["layers"]
        for value in row["sample"]
    )
    assert [module.training for module in model.modules()] == training_states
    assert [
        (len(module._forward_hooks), len(module._forward_pre_hooks)) for module in model.modules()
    ] == hook_counts
    assert torch.equal(example.detach(), example_copy)
    assert all(
        torch.equal(parameter, saved)
        for parameter, saved in zip(model.parameters(), parameter_copies, strict=True)
    )
    assert all(
        torch.equal(parameter.grad, saved)
        for parameter, saved in zip(model.parameters(), gradient_copies, strict=True)
    )
    assert all(
        torch.equal(buffer, saved)
        for buffer, saved in zip(model.buffers(), buffer_copies, strict=True)
    )


def test_torch_activation_failure_cleans_temporary_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    import seedl._inspect as inspection

    model = nn.Sequential(nn.Linear(3, 2), nn.ReLU()).train()
    states = [module.training for module in model.modules()]
    counts = [
        (len(module._forward_hooks), len(module._forward_pre_hooks)) for module in model.modules()
    ]

    def fail_visualization(_tensor: object, _max_points: int) -> object:
        raise ValueError("forced activation rendering failure")

    monkeypatch.setattr(inspection, "_torch_tensor_data", fail_visualization)
    with pytest.raises(SeeDLError, match="forced activation rendering failure"):
        sd.activations(model, input=torch.randn(2, 3), backend="svg")

    assert [module.training for module in model.modules()] == states
    assert [
        (len(module._forward_hooks), len(module._forward_pre_hooks)) for module in model.modules()
    ] == counts


def test_torch_replaced_or_added_buffers_are_restored() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    class BufferMutatingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("state", torch.tensor([1.0]))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            self.state = torch.tensor([9.0])
            self.register_buffer("temporary", torch.tensor([2.0]))
            return value + self.state

    model = BufferMutatingModel().train()
    original_state = model.state
    original_value = original_state.clone()

    sd.see(model, input=torch.tensor([0.0]), backend="svg")

    assert model.state is original_state
    assert torch.equal(model.state, original_value)
    assert "temporary" not in model._buffers
    assert model.training


def test_torch_dtype_statistics_and_unsupported_quantized_tensors() -> None:
    torch = pytest.importorskip("torch")

    large = sd.see(torch.tensor([1e300, 2e300], dtype=torch.float64), backend="svg")
    assert large.metadata["Non-finite values"] == 0
    assert float(large.metadata["Mean"]) == pytest.approx(1.5e300)
    assert float(large.metadata["Std. deviation"]) == pytest.approx(5e299)

    tiny = sd.see(torch.tensor([1e-300, 2e-300], dtype=torch.float64), backend="svg")
    assert float(tiny.metadata["Mean"]) == pytest.approx(1.5e-300)
    assert float(tiny.metadata["Std. deviation"]) == pytest.approx(5e-301)

    large_integer = sd.see(torch.tensor([2**40, 2**40 + 1]), backend="svg")
    assert float(large_integer.metadata["Std. deviation"]) == pytest.approx(0.5)
    scalar = sd.see(torch.tensor(1e300, dtype=torch.float64), backend="svg")
    assert scalar.data["value"] == 1e300
    assert scalar.metadata["Non-finite values"] == 0

    image = torch.zeros((3, 16, 16), dtype=torch.float64)
    image[..., 8:] = 1e300
    image_view = sd.see(image, backend="svg")
    assert image_view.kind == "image"
    assert image_view.data["pixels"][0][0] == [0.0, 0.0, 0.0]
    assert image_view.data["pixels"][0][8] == [1.0, 1.0, 1.0]

    empty = sd.see(torch.empty((0, 3)), backend="svg")
    assert empty.data["sample"] == []
    nonfinite = sd.see(torch.tensor([float("nan"), float("inf")]), backend="svg")
    assert nonfinite.metadata["Non-finite values"] == 2
    assert nonfinite.metadata["Mean"] == "n/a"

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quantized = torch.quantize_per_tensor(
            torch.tensor([1.0, 2.0]), scale=0.1, zero_point=0, dtype=torch.qint8
        )
    with pytest.raises(SeeDLError, match="dequantize"):
        sd.see(quantized, backend="svg")


def test_torch_tensor_and_gradient_samples_respect_max_points() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    source = torch.tensor([1.0, 2.0], requires_grad=True)
    source.grad = torch.tensor([3.0, 4.0])
    source_value = source.detach().clone()
    source_gradient = source.grad.clone()
    source_view = sd.see(source, backend="svg")
    assert source_view.metadata["Requires gradients"] is True
    assert torch.equal(source.detach(), source_value)
    assert torch.equal(source.grad, source_gradient)
    assert source.requires_grad

    large = torch.arange(1_100_000, dtype=torch.float32)
    original = large.clone()
    view = sd.see(large, backend="svg", max_points=37)
    assert len(view.data["values"]) <= 37
    assert view.metadata["Elements"] == 1_100_000
    assert torch.equal(large, original)

    sliced = torch.arange(20_000, dtype=torch.float32)[::2]
    sampled = sd.see(sliced, backend="svg", max_points=13)
    assert not sliced.is_contiguous()
    assert len(sampled.data["values"]) <= 13
    assert sampled.data["values"][0] == 0

    model = nn.Sequential(nn.Embedding(100, 4, sparse=True), nn.Flatten(), nn.Linear(8, 1))
    model(torch.tensor([[1, 2], [3, 4]])).sum().backward()
    gradient_view = sd.gradients(model, backend="svg", max_points=3)
    assert sum(len(row["sample"]) for row in gradient_view.data["layers"]) <= 3
    assert any("Sparse gradient" in note for note in gradient_view.notes)


def test_torch_missing_and_nonfinite_gradients_are_read_only() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
    parameters = list(model.parameters())
    parameters[0].grad = torch.tensor(
        [[float("nan"), 1.0, 2.0], [float("inf"), 3.0, 4.0]], dtype=parameters[0].dtype
    )
    parameters[2].grad = torch.ones_like(parameters[2])
    parameter_copies = [parameter.detach().clone() for parameter in parameters]
    gradient_copies = [
        None if parameter.grad is None else parameter.grad.clone() for parameter in parameters
    ]

    view = sd.gradients(model, backend="svg", max_points=2)
    rows = {row["name"]: row for row in view.data["layers"]}
    assert rows["0.weight"]["finite"] == "no (2)"
    assert rows["0.bias"]["status"] == "no gradient"
    assert all(
        torch.equal(parameter, saved)
        for parameter, saved in zip(parameters, parameter_copies, strict=True)
    )
    for parameter, saved in zip(parameters, gradient_copies, strict=True):
        if saved is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None
            torch.testing.assert_close(parameter.grad, saved, rtol=0, atol=0, equal_nan=True)


def test_torch_gpu_tensor_model_and_gradient_paths_when_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA or ROCm accelerator is available to this PyTorch build")
    from torch import nn

    device = torch.device("cuda:0")
    tensor = torch.arange(128, dtype=torch.float32, device=device)
    tensor_view = sd.see(tensor, backend="svg", max_points=11)
    assert tensor_view.metadata["Device"] == str(device)
    assert len(tensor_view.data["values"]) <= 11

    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 2),
    ).to(device)
    model.train()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    example = torch.randn(2, 3, 16, 16, device=device, requires_grad=True)
    example_copy = example.detach().clone()
    parameter_copies = [parameter.detach().clone() for parameter in model.parameters()]
    gradient_copies = [parameter.grad.clone() for parameter in model.parameters()]
    states = [module.training for module in model.modules()]

    architecture = sd.see(model, input=example, backend="svg")
    activations = sd.activations(model, input=example, backend="svg", max_points=17)
    gradients = sd.gradients(model, backend="svg", max_points=17)

    assert architecture.kind == "model"
    assert activations.data["layers"]
    assert sum(len(row["sample"]) for row in activations.data["layers"]) <= 17
    assert sum(len(row["sample"]) for row in gradients.data["layers"]) <= 17
    assert [module.training for module in model.modules()] == states
    assert torch.equal(example.detach(), example_copy)
    assert all(
        torch.equal(parameter, saved)
        for parameter, saved in zip(model.parameters(), parameter_copies, strict=True)
    )
    assert all(
        torch.equal(parameter.grad, saved)
        for parameter, saved in zip(model.parameters(), gradient_copies, strict=True)
    )
