"""Tests for Neural Execution Tracer and Tensor Flow Explorer."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import seedl as sd
from seedl._execution_graph import ExecutionTrace


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out + res


class MultiInputOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(4, 8)
        self.fc_b = nn.Linear(6, 8)
        self.fc_out = nn.Linear(16, 2)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out_a = torch.relu(self.fc_a(a))
        out_b = torch.relu(self.fc_b(b))
        combined = torch.cat([out_a, out_b], dim=-1)
        out = self.fc_out(combined)
        return out, combined


class ReusedModuleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(x)  # 3rd invocation of self.relu
        return x


class ExceptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        raise RuntimeError("Intentional error during forward pass")


def test_sequential_execution_trace():
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )
    x = torch.randn(2, 10)
    trace = sd.trace(model, x)

    assert isinstance(trace, ExecutionTrace)
    assert len(trace.nodes) >= 3
    leaf_nodes = trace.get_leaf_nodes()
    assert len(leaf_nodes) == 3
    assert leaf_nodes[0].module_type == "Linear"
    assert leaf_nodes[1].module_type == "ReLU"
    assert leaf_nodes[2].module_type == "Linear"

    assert leaf_nodes[0].output_shapes == [[2, 20]]
    assert leaf_nodes[1].output_shapes == [[2, 20]]
    assert leaf_nodes[2].output_shapes == [[2, 5]]


def test_residual_model_trace():
    model = ResidualBlock(8)
    x = torch.randn(3, 8)
    trace = sd.trace(model, x)

    assert isinstance(trace, ExecutionTrace)
    leaf_nodes = trace.get_leaf_nodes()
    types = [n.module_type for n in leaf_nodes]
    assert "Linear" in types
    assert "ReLU" in types


def test_reused_module_trace():
    model = ReusedModuleModel()
    x = torch.randn(1, 4)
    trace = sd.trace(model, x)

    relu_nodes = [n for n in trace.nodes if n.module_type == "ReLU"]
    assert len(relu_nodes) == 3
    # Check distinct invocation indices
    indices = [n.invocation_index for n in relu_nodes]
    assert indices == [0, 1, 2]
    assert all(n.is_reused is True for n in relu_nodes[1:])


def test_multi_input_multi_output_trace():
    model = MultiInputOutputModel()
    a = torch.randn(2, 4)
    b = torch.randn(2, 6)
    trace = sd.trace(model, (a, b))

    assert len(trace.input_metadata) == 2
    assert trace.input_metadata[0]["shape"] == [2, 4]
    assert trace.input_metadata[1]["shape"] == [2, 6]
    assert len(trace.output_metadata) == 2


def test_model_state_preservation():
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))
    model.train()  # Explicitly set to train mode

    x = torch.randn(3, 4)
    # Give parameter gradients
    loss = model(x).sum()
    loss.backward()

    # Save gradient clones
    orig_grads = {
        name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None
    }
    orig_training_state = {name: m.training for name, m in model.named_modules()}

    # Run trace
    _ = sd.trace(model, x)

    # Verify model is back in train mode
    for name, m in model.named_modules():
        assert m.training == orig_training_state[name], f"Training mode not restored for {name}"

    # Verify parameter gradients were restored
    for name, p in model.named_parameters():
        if name in orig_grads:
            assert torch.equal(p.grad, orig_grads[name]), f"Gradient mutated for {name}"


def test_exception_hook_cleanup():
    model = ExceptionModel()
    x = torch.randn(2, 4)

    # Tracing should propagate the exception or handle safely
    with pytest.raises(RuntimeError, match="Intentional error"):
        sd.trace(model, x)

    # Verify all temporary hooks were removed from model submodules
    for m in model.modules():
        assert len(m._forward_hooks) == 0
        assert len(m._forward_pre_hooks) == 0


def test_sd_flow_visualization():
    model = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU())
    x = torch.randn(1, 3, 16, 16)

    # Via sd.trace -> sd.flow
    t = sd.trace(model, x)
    v1 = sd.flow(t)
    assert v1.kind == "execution_trace"
    assert "Conv2d" in str(v1.data)

    # Direct sd.flow(model, input=x)
    v2 = sd.flow(model, input=x)
    assert v2.kind == "execution_trace"

    # Export to HTML
    html_content = v1.to_html()
    assert "Neural Observatory" in html_content
    assert "Flow Graph" in html_content
    assert "Tensor Table" in html_content

    # Export to SVG
    svg_content = v1.to_figure("svg")
    assert "<svg" in svg_content
    assert "SeeDL / NEURAL EXECUTION TRACER" in svg_content


def test_see_with_execution_trace_and_flow_view():
    model = nn.Sequential(nn.Linear(5, 10), nn.ReLU())
    x = torch.randn(2, 5)

    # Via sd.see(model, input=x, view="flow")
    v = sd.see(model, input=x, view="flow")
    assert v.kind == "execution_trace"

    # Via sd.see(trace)
    t = sd.trace(model, x)
    v2 = sd.see(t)
    assert v2.kind == "execution_trace"


class ViewSliceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2)
        self.fc = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        # Flatten via view creates a storage alias (shares data_ptr, distinct object)
        flat = h.view(h.size(0), -1)
        return self.fc(flat)


def test_storage_alias_detection():
    """Verify that tensor views/slices sharing underlying storage are classified as storage_alias."""
    model = ViewSliceModel()
    x = torch.randn(1, 1, 3, 3)
    trace = sd.trace(model, x, use_fx=False)  # Test hooks path directly

    # Find edge from conv to fc
    alias_edges = [e for e in trace.edges if e.edge_type == "storage_alias"]
    assert len(alias_edges) >= 1
    assert alias_edges[0].source.startswith("conv")
    assert alias_edges[0].target.startswith("fc")
    assert alias_edges[0].is_uncertain is False


class SkipConnectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.fc1(x)
        h2 = self.fc2(h1)
        # Residual skip addition
        merged = h2 + h1
        out = self.fc3(merged)
        return out


def test_residual_skip_connection_fx_topology():
    """Verify torch.fx correctly traces the multi-input skip connection into fc3."""
    model = SkipConnectionModel()
    x = torch.randn(2, 4)
    trace = sd.trace(model, x, use_fx=True)

    assert trace.trace_mode == "torch.fx+hooks"
    # fc3 should have incoming dependencies from BOTH fc2 and fc1 (the skip)
    fc3_incoming = [e.source for e in trace.edges if e.target.startswith("fc3")]
    assert any("fc1" in s for s in fc3_incoming)
    assert any("fc2" in s for s in fc3_incoming)


class DynamicControlFlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_pos = nn.Linear(4, 2)
        self.fc_neg = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Data-dependent Python conditional breaks torch.fx symbolic trace
        if bool(x.sum().item() > 0):
            return self.fc_pos(x)
        else:
            return self.fc_neg(x)


def test_dynamic_control_flow_fallback_to_hooks():
    """Verify that models with dynamic control flow fallback cleanly to hooks mode."""
    model = DynamicControlFlowModel()
    x = torch.randn(2, 4)
    trace = sd.trace(model, x, use_fx=True)

    # Should fallback to hooks mode without throwing
    assert trace.trace_mode == "hooks"
    assert len(trace.nodes) >= 1


class DictOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_feat = nn.Linear(4, 8)
        self.fc_cls = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.fc_feat(x)
        logits = self.fc_cls(feat)
        return {"features": feat, "logits": logits}


def test_dict_outputs_and_inputs():
    """Verify recursive tensor extraction handles dictionary outputs and inputs."""
    model = DictOutputModel()
    x = torch.randn(3, 4)
    trace = sd.trace(model, x)

    assert len(trace.output_metadata) == 2
    out_shapes = [m["shape"] for m in trace.output_metadata]
    assert [3, 8] in out_shapes
    assert [3, 2] in out_shapes


class BufferAndInplaceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.register_buffer("running_stat", torch.tensor([1.0, 2.0, 3.0, 4.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        # Inplace operation outside submodules
        x = x.clone()
        x.add_(1.0)
        return x * self.running_stat


def test_buffer_preservation_and_non_mutation():
    """Verify model buffers and original user input are preserved without mutation."""
    model = BufferAndInplaceModel()
    orig_buffer = model.running_stat.clone()
    orig_x = torch.randn(2, 4)
    x_input = orig_x.clone()

    trace = sd.trace(model, x_input)
    assert trace is not None

    # Verify buffer untouched
    assert torch.equal(model.running_stat, orig_buffer)
    # Verify input tensor untouched
    assert torch.equal(x_input, orig_x)


class LargeBufferMutatingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        # Large buffer exceeding 8192 elements
        self.register_buffer("large_buf", torch.zeros(16384))
        # Small buffer
        self.register_buffer("small_buf", torch.ones(4))
        # None-valued buffer
        self.register_buffer("none_buf", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In-place buffer mutations during forward execution
        self.large_buf.add_(10.0)
        self.small_buf.fill_(42.0)
        return self.fc(x) + self.small_buf


class BufferMutatingExceptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("critical_buf", torch.full((10000,), 3.14159))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.critical_buf.zero_()
        raise RuntimeError("Crash inside forward after buffer zeroing")


class InplaceInputMutatingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Malicious / aggressive in-place modification of input argument
        x.add_(999.0)
        x[0, 0] = -12345.0
        return self.fc(x)


class MixedModeContainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_train = nn.Linear(4, 4)
        self.layer_eval = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_eval(self.layer_train(x))


def test_large_buffer_mutation_restoration():
    """Buffers > 8192 elements and None-valued buffers mutated during forward are fully restored."""
    model = LargeBufferMutatingModel()
    orig_large = model.large_buf.clone()
    orig_small = model.small_buf.clone()

    x = torch.randn(2, 4)
    trace = sd.trace(model, x)
    assert trace is not None

    # Verify large buffer restored
    assert torch.equal(model.large_buf, orig_large)
    assert model.large_buf.numel() == 16384
    # Verify small buffer restored
    assert torch.equal(model.small_buf, orig_small)
    # Verify None buffer remains None
    assert model.none_buf is None


def test_buffer_restoration_on_forward_exception():
    """Registered buffers mutated prior to an exception are guaranteed restored in finally block."""
    model = BufferMutatingExceptionModel()
    orig_buf = model.critical_buf.clone()
    x = torch.randn(2, 4)

    with pytest.raises(RuntimeError, match="Crash inside forward"):
        sd.trace(model, x)

    # Verify buffer restored despite forward exception
    assert torch.equal(model.critical_buf, orig_buf)
    assert model.critical_buf[0].item() == pytest.approx(3.14159, abs=1e-4)


def test_input_tensor_non_mutation():
    """Caller input tensors are never mutated even if forward pass executes in-place operations."""
    model = InplaceInputMutatingModel()
    orig_x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    x_input = orig_x.clone()

    trace = sd.trace(model, x_input)
    assert trace is not None

    # Caller's input tensor must remain exactly equal to orig_x
    assert torch.equal(x_input, orig_x)
    assert x_input[0, 0].item() == 1.0


def test_mixed_training_modes_preservation():
    """Mixed submodule training modes (train vs eval) are preserved individually."""
    model = MixedModeContainer()
    model.layer_train.train()
    model.layer_eval.eval()

    assert model.layer_train.training is True
    assert model.layer_eval.training is False

    x = torch.randn(2, 4)
    _ = sd.trace(model, x)

    # Verify both training flags returned to their exact original states
    assert model.layer_train.training is True
    assert model.layer_eval.training is False


def test_repeated_tracing_stability():
    """Tracing a model multiple times in succession preserves state across all runs."""
    model = LargeBufferMutatingModel()
    orig_large = model.large_buf.clone()
    x = torch.randn(2, 4)

    for _ in range(3):
        t = sd.trace(model, x)
        assert len(t.nodes) >= 1
        assert torch.equal(model.large_buf, orig_large)
