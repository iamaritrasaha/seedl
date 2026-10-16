"""Execution graph data structures and topology analysis for SeeDL."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TraceNode:
    """Detailed observation of a single executed module or operation."""

    id: str
    name: str
    module_type: str
    invocation_index: int
    execution_order: int
    parent_name: str = ""
    is_leaf: bool = True
    input_shapes: list[list[int]] = field(default_factory=list)
    output_shapes: list[list[int]] = field(default_factory=list)
    dtypes: list[str] = field(default_factory=list)
    devices: list[str] = field(default_factory=list)
    num_elements: int = 0
    num_parameters: int = 0
    trainable_parameters: int = 0
    activation_stats: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None
    is_reused: bool = False
    sampled_values: list[float] = field(default_factory=list)
    is_sampled: bool = False
    sample_size: int = 0
    nonfinite_count: int = 0
    sparsity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraceEdge:
    """Explicit or inferred data connection between executed nodes."""

    source: str
    target: str
    edge_type: str  # "explicit_tensor", "storage_alias", "symbolic_fx", "inferred_sequential", "unknown_dependency"
    tensor_shape: list[int] | None = None
    tensor_dtype: str | None = None
    description: str = ""
    is_uncertain: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionTrace:
    """Complete recorded execution trace for a neural network forward pass."""

    nodes: list[TraceNode] = field(default_factory=list)
    edges: list[TraceEdge] = field(default_factory=list)
    input_metadata: list[dict[str, Any]] = field(default_factory=list)
    output_metadata: list[dict[str, Any]] = field(default_factory=list)
    total_parameters: int = 0
    trainable_parameters: int = 0
    duration_ms: float = 0.0
    device: str = "cpu"
    trace_mode: str = "hooks"  # "hooks", "torch.fx+hooks", "hooks_fallback"
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "input_metadata": self.input_metadata,
            "output_metadata": self.output_metadata,
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "duration_ms": self.duration_ms,
            "device": self.device,
            "trace_mode": self.trace_mode,
            "warnings": self.warnings,
            "limitations": self.limitations,
        }

    def get_node(self, node_id: str) -> TraceNode | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def get_leaf_nodes(self) -> list[TraceNode]:
        return [n for n in self.nodes if n.is_leaf]


def compute_tensor_stats(
    t: Any,
    max_samples: int = 32,
    torch: Any = None,
) -> tuple[dict[str, Any], list[float], int, float]:
    """Compute bounded numerical statistics on an observed tensor without autograd graph retention."""
    if torch is not None and isinstance(t, torch.Tensor):
        detached = t.detach()
        num_elements = detached.numel()
        if num_elements == 0:
            return (
                {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
                [],
                0,
                0.0,
            )

        float_tensor = detached.float()
        # Count non-finite
        is_finite = torch.isfinite(float_tensor)
        nonfinite_count = int((~is_finite).sum().item())

        finite_vals = float_tensor[is_finite]
        if finite_vals.numel() > 0:
            t_min = float(finite_vals.min().item())
            t_max = float(finite_vals.max().item())
            t_mean = float(finite_vals.mean().item())
            t_std = float(finite_vals.std().item()) if finite_vals.numel() > 1 else 0.0
            zeros = int((finite_vals == 0.0).sum().item())
            sparsity = zeros / float(finite_vals.numel())
        else:
            t_min, t_max, t_mean, t_std = float("nan"), float("nan"), float("nan"), float("nan")
            sparsity = 0.0

        # Bounded sampling
        flat = float_tensor.reshape(-1)
        if num_elements <= max_samples:
            sampled_raw = flat.cpu().tolist()
        else:
            step = max(1, num_elements // max_samples)
            sampled_raw = flat[::step][:max_samples].cpu().tolist()

        # Sanitize for JSON (handle nan / inf)
        sampled_sanitized = [
            val if math.isfinite(val) else (999999.0 if val > 0 else -999999.0)
            for val in sampled_raw
        ]

        stats = {
            "min": t_min if math.isfinite(t_min) else None,
            "max": t_max if math.isfinite(t_max) else None,
            "mean": t_mean if math.isfinite(t_mean) else None,
            "std": t_std if math.isfinite(t_std) else None,
            "num_elements": num_elements,
            "nonfinite_count": nonfinite_count,
            "sparsity": sparsity,
        }
        return stats, sampled_sanitized, nonfinite_count, sparsity

    return {}, [], 0, 0.0
