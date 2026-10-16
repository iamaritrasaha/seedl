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
