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
                {
                    "min": 0.0,
                    "max": 0.0,
                    "mean": 0.0,
                    "std": 0.0,
                    "num_elements": 0,
                    "nonfinite_count": 0,
                    "sparsity": 0.0,
                },
                [],
                0,
                0.0,
            )

        is_sparse = bool(
            detached.is_sparse or (hasattr(detached, "layout") and detached.layout != torch.strided)
        )
        if is_sparse:
            coalesced = (
                detached.coalesce()
                if hasattr(detached, "coalesce")
                else detached.to_sparse_coo().coalesce()
            )
            vals = coalesced.values()
            stored_count = vals.numel()
            implicit_zeros = num_elements - stored_count
            is_fp = vals.dtype.is_floating_point

            nonfinite_count = int((~torch.isfinite(vals)).sum().item()) if is_fp else 0
            if nonfinite_count == 0:
                vals_float = vals.float() if is_fp else vals.to(dtype=torch.float32)
                v_min = float(vals_float.min().item()) if stored_count > 0 else 0.0
                v_max = float(vals_float.max().item()) if stored_count > 0 else 0.0
                t_min = min(0.0, v_min) if implicit_zeros > 0 else v_min
                t_max = max(0.0, v_max) if implicit_zeros > 0 else v_max
                t_mean = float(vals_float.sum().item()) / float(num_elements)

                if num_elements > 1:
                    stored_dev_sq = float(((vals_float - t_mean) ** 2).sum().item())
                    zero_dev_sq = float(implicit_zeros * (t_mean**2))
                    var_val = max(0.0, (stored_dev_sq + zero_dev_sq) / float(num_elements - 1))
                    t_std = math.sqrt(var_val)
                else:
                    t_std = 0.0

                stored_zeros = int((vals == 0).sum().item())
                zeros = implicit_zeros + stored_zeros
                sparsity = zeros / float(num_elements)
            else:
                t_min, t_max, t_mean, t_std = (
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                )
                sparsity = 0.0

            sampled_raw: list[float] = (
                vals[:max_samples].float().cpu().tolist() if stored_count > 0 else []
            )
            if len(sampled_raw) < max_samples and implicit_zeros > 0:
                sampled_raw.extend([0.0] * min(max_samples - len(sampled_raw), implicit_zeros))

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

        # Dense tensor processing with bounded allocations
        is_fp = detached.dtype.is_floating_point
        nonfinite_count = int((~torch.isfinite(detached)).sum().item()) if is_fp else 0

        if nonfinite_count == 0:
            # Memory-safe fast path: no boolean mask allocation needed
            t_min = float(detached.min().item())
            t_max = float(detached.max().item())
            t_mean = (
                float(detached.mean().item())
                if detached.dtype in (torch.float32, torch.float64)
                else float(detached.float().mean().item())
            )
            t_std = (
                (
                    float(detached.std().item())
                    if detached.dtype in (torch.float32, torch.float64)
                    else float(detached.float().std().item())
                )
                if num_elements > 1
                else 0.0
            )
            zeros = int((detached == 0).sum().item())
            sparsity = zeros / float(num_elements)
        elif nonfinite_count == num_elements:
            t_min, t_max, t_mean, t_std = (
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
            )
            sparsity = 0.0
        else:
            # Bounded processing of finite values only when nonfinite elements are present
            mask = torch.isfinite(detached)
            finite_vals = detached[mask]
            finite_numel = finite_vals.numel()
            if finite_numel > 0:
                t_min = float(finite_vals.min().item())
                t_max = float(finite_vals.max().item())
                t_mean = (
                    float(finite_vals.mean().item())
                    if finite_vals.dtype in (torch.float32, torch.float64)
                    else float(finite_vals.float().mean().item())
                )
                t_std = (
                    (
                        float(finite_vals.std().item())
                        if finite_vals.dtype in (torch.float32, torch.float64)
                        else float(finite_vals.float().std().item())
                    )
                    if finite_numel > 1
                    else 0.0
                )
                zeros = int((finite_vals == 0).sum().item())
                sparsity = zeros / float(finite_numel)
            else:
                t_min, t_max, t_mean, t_std = (
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                )
                sparsity = 0.0

        # Memory-safe bounded sampling: only transfer max_samples elements to CPU
        if num_elements <= max_samples:
            sampled_tensor = detached.view(-1) if detached.is_contiguous() else detached.reshape(-1)
        else:
            step = max(1, num_elements // max_samples)
            flat = detached.view(-1) if detached.is_contiguous() else detached.reshape(-1)
            sampled_tensor = flat[::step][:max_samples]

        sampled_raw = (
            sampled_tensor.float().cpu().tolist()
            if is_fp
            else sampled_tensor.to(dtype=torch.float32).cpu().tolist()
        )

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
