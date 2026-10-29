"""Focused regression and feature verification tests for SeeDL model observability.

Validates:
1. Repeated-module invocation tracking and distinct diagnostics.
2. Filtered layer results consistency with reported summary and notes.
3. Layer-level activation diagnostics: mean, std, min, max, exact zero fractions,
   ReLU inactivity, Sigmoid/Tanh saturation, NaN/Inf, and numerical instability.
4. Hook cleanup, training mode preservation, buffer preservation, and exception safety.
5. Sparse-aware tensor statistics and memory safety.
6. Native Windows AMD ROCm / CUDA GPU execution when available.
"""

from __future__ import annotations

import math

import pytest

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

import seedl as sd
from seedl._execution_graph import compute_tensor_stats
from seedl.exceptions import SeeDLError


class RecurrentBlockModel(nn.Module):
    """Model that reuses the same Linear and ReLU modules across multiple loop steps."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(4, 4)
        self.relu = nn.ReLU()
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = self.relu(self.cell(x))
        return self.head(x)


class MultiActivationModel(nn.Module):
    """Model with ReLU, Sigmoid, and Tanh activation stages."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 8)
        self.sigmoid = nn.Sigmoid()
        self.fc3 = nn.Linear(8, 4)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x


class NanLayer(nn.Module):
    """Layer injecting non-finite activations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        out[0, 0] = float("nan")
        out[0, 1] = float("inf")
        out[0, 2] = float("-inf")
        return out


class NanInfModel(nn.Module):
    """Model producing NaN, +Inf, and -Inf intermediate activations."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.nan_layer = NanLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nan_layer(self.fc(x))


class CrashingModel(nn.Module):
    """Model that intentionally raises a RuntimeError during forward pass."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        raise RuntimeError("Simulated internal forward execution failure")


class NonTensorOutputModel(nn.Module):
    """Model with a module returning structured non-tensor container outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, str], int]:
        out = self.fc(x)
        return out, {"metadata": "extra_info"}, 42


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch required")
def test_repeated_module_invocations_tracked_individually():
    """Verify repeated executions of a reused module are not silently overwritten."""
    torch.manual_seed(42)
    model = RecurrentBlockModel()
    x = torch.randn(2, 4)

    view = sd.activations(model, input=x, backend="svg")
    layers = view.data["layers"]

    # cell and relu were called 3 times each
    cell_layers = [layer for layer in layers if layer["module_name"] == "cell"]
    relu_layers = [layer for layer in layers if layer["module_name"] == "relu"]

    assert len(cell_layers) == 3
    assert len(relu_layers) == 3

    # Check invocation indices and call naming
    for i, cell_layer in enumerate(cell_layers):
        assert cell_layer["invocation_index"] == i
        assert cell_layer["total_invocations"] == 3
        assert cell_layer["name"] == f"cell (call {i + 1})"

    for i, relu_layer in enumerate(relu_layers):
        assert relu_layer["invocation_index"] == i
        assert relu_layer["total_invocations"] == 3
        assert relu_layer["name"] == f"relu (call {i + 1})"

    # Head is called once: total_invocations == 1 and name has no call suffix
    head_layer = next(layer for layer in layers if layer["module_name"] == "head")
    assert head_layer["invocation_index"] == 0
    assert head_layer["total_invocations"] == 1
    assert head_layer["name"] == "head"

    # Repeated invocations note present
    assert any("repeated module invocations detected" in n.lower() for n in view.notes)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch required")
def test_filtered_layer_results_match_summary_and_notes():
    """Verify filtering by layer name or type keeps summary and notes perfectly consistent."""
    model = MultiActivationModel()
    x = torch.randn(2, 4)

    # 1. Filter with single string name
    view_single = sd.activations(model, input=x, layers="fc1", backend="svg")
    layers_single = view_single.data["layers"]
    assert len(layers_single) == 1
    assert layers_single[0]["name"] == "fc1"
    assert "Captured 1 module activation(s)" in view_single.summary
    assert any("filtered" in n.lower() for n in view_single.notes)

    # 2. Filter with list of names
    view_multi = sd.activations(model, input=x, layers=["fc1", "fc2"], backend="svg")
    layers_multi = view_multi.data["layers"]
    assert len(layers_multi) == 2
    assert {layer["name"] for layer in layers_multi} == {"fc1", "fc2"}
    assert "Captured 2 module activation(s)" in view_multi.summary

    # 3. Filter with type name
    view_type = sd.activations(model, input=x, layers="ReLU", backend="svg")
    layers_type = view_type.data["layers"]
    assert len(layers_type) == 1
    assert layers_type[0]["type"] == "ReLU"


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch required")
def test_activation_diagnostics_relu_inactivity_and_saturation():
    """Verify ReLU inactivity and Sigmoid/Tanh saturation diagnostics on tailored inputs."""
    model = MultiActivationModel()
    with torch.no_grad():
        # Set fc1 weights and bias to ensure negative input to ReLU
        model.fc1.weight.fill_(1.0)
        model.fc1.bias.fill_(0.0)

    # Feed negative values so ReLU receives all negative inputs and outputs 100% zeros
    x_neg = torch.full((2, 4), -10.0)
    view_inact = sd.activations(model, input=x_neg, zero_threshold=0.8, backend="svg")
    relu_layer = next(layer for layer in view_inact.data["layers"] if layer["name"] == "relu")

    assert relu_layer["relu_inactivity"] == 1.0
    assert relu_layer["zero_fraction"] == 1.0
    assert relu_layer["zero_fraction_sampled"] is False
    assert relu_layer["dead_fraction"] == 1.0
    assert "zero/inactive units on this input" in relu_layer["saturation_note"]
    assert any("dead unit alert" in n.lower() for n in view_inact.notes)
    # Never describe as permanently dead
    assert any("not a permanent diagnosis" in n.lower() for n in view_inact.notes)

    # Test configurable zero_threshold suppresses alert when above threshold
    view_high_thresh = sd.activations(model, input=x_neg, zero_threshold=1.5, backend="svg")
    assert not any("dead unit alert" in n.lower() for n in view_high_thresh.notes)

    # Sigmoid saturation: feed large positive inputs
    with torch.no_grad():
        model.fc2.weight.fill_(1.0)
        model.fc2.bias.fill_(0.0)
    x_pos = torch.full((2, 4), 10.0)
    view_sat = sd.activations(model, input=x_pos, saturation_threshold=0.5, backend="svg")
    sig_layer = next(layer for layer in view_sat.data["layers"] if layer["name"] == "sigmoid")
    assert sig_layer["saturation_fraction"] is not None
    assert sig_layer["saturation_fraction"] > 0.5
    assert "saturated" in sig_layer["saturation_note"]
    assert any("saturation detected" in n.lower() for n in view_sat.notes)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch required")
def test_numerical_instability_and_nonfinite_activations():
    """Verify NaN, Inf, and large magnitude activations trigger warnings and clean reporting."""
    model = NanInfModel()
    x = torch.randn(2, 4)

    view = sd.activations(model, input=x, backend="svg")
    nan_layer = next(layer for layer in view.data["layers"] if layer["name"] == "nan_layer")

    assert nan_layer["nonfinite_count"] == 3
    assert nan_layer["nan_count"] == 1
    assert nan_layer["inf_count"] == 2
    assert "no (3)" in nan_layer["finite"]
    assert nan_layer["instability_warning"] is not None
    assert "Non-finite activations" in nan_layer["instability_warning"]
    assert any("numerical instability detected" in n.lower() for n in view.notes)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch required")
def test_execution_correctness_and_exception_safety():
    """Verify forward-hooks are removed and training/buffers restored even on exception."""
    model = CrashingModel()
    model.train()
    original_training_state = model.training
    x = torch.randn(2, 4)

    with pytest.raises(SeeDLError, match="Could not inspect this model"):
        sd.activations(model, input=x, backend="svg")

    # Verify ALL forward hooks and pre-hooks were safely cleaned up
    for mod in model.modules():
        assert len(mod._forward_hooks) == 0
        assert len(mod._forward_pre_hooks) == 0

    # Verify training state was preserved
    assert model.training == original_training_state


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch required")
def test_non_tensor_and_nested_outputs():
    """Verify models returning non-tensor tuples and dictionaries are handled gracefully."""
    model = NonTensorOutputModel()
    x = torch.randn(2, 4)

    view = sd.activations(model, input=x, backend="svg")
    assert view is not None
    fc_layer = next(layer for layer in view.data["layers"] if layer["name"] == "fc")
    assert fc_layer["status"] == "available"
    assert fc_layer["shape"] == "(2, 4)"


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch required")
def test_compute_tensor_stats_sparse_vs_dense_exactness():
    """Verify compute_tensor_stats on sparse COO matches dense PyTorch statistics exactly."""
    indices = torch.tensor([[0, 1, 1], [0, 1, 2]])
    values = torch.tensor([3.0, 4.0, 5.0])
    sparse = torch.sparse_coo_tensor(indices, values, (2, 3)).coalesce()
    dense = sparse.to_dense().float()

    s_stats, s_samp, s_nonf, s_sparse = compute_tensor_stats(sparse, torch=torch)
    d_stats, d_samp, d_nonf, d_sparse = compute_tensor_stats(dense, torch=torch)

    assert s_nonf == 0
    assert d_nonf == 0
    assert math.isclose(s_stats["min"], d_stats["min"], rel_tol=1e-5)
    assert math.isclose(s_stats["max"], d_stats["max"], rel_tol=1e-5)
    assert math.isclose(s_stats["mean"], d_stats["mean"], rel_tol=1e-5)
    assert math.isclose(s_stats["std"], d_stats["std"], rel_tol=1e-5)
    assert math.isclose(s_stats["sparsity"], d_stats["sparsity"], rel_tol=1e-5)
    assert s_stats["num_elements"] == 6


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch required")
def test_large_tensor_bounded_memory_safety():
    """Verify compute_tensor_stats processes 2M elements with bounded allocations."""
    large_t = torch.randn(2_000_000)
    stats, samples, nonf, sparsity = compute_tensor_stats(large_t, max_samples=32, torch=torch)

    assert stats["num_elements"] == 2_000_000
    assert nonf == 0
    assert len(samples) == 32
    assert all(math.isfinite(s) for s in samples)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch required")
def test_rocm_gpu_smoke_and_diagnostics():
    """Verify execution on native Windows AMD ROCm / CUDA GPU when accelerator is present."""
    if not torch.cuda.is_available():
        pytest.skip("No CUDA / ROCm GPU device available")

    device = torch.device("cuda:0")
    model = RecurrentBlockModel().to(device)
    x = torch.randn(2, 4, device=device)

    view = sd.activations(model, input=x, backend="svg")
    assert view.data["layers"]
    for layer in view.data["layers"]:
        assert layer["status"] == "available"
        # Check samples are serializable python floats, not GPU tensors
        assert all(isinstance(val, float) for val in layer["sample"])
