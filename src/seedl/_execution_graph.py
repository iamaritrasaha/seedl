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


def _tensor_chunks(tensor: Any, chunk_size: int = 1_048_576):
    """Yield bounded flattened views without allocating full-tensor copies."""
    count = int(tensor.numel())
    if count <= chunk_size or tensor.ndim == 0:
        yield tensor.reshape(-1) if not tensor.is_contiguous() else tensor.view(-1)
        return
    axis = max(range(tensor.ndim), key=lambda idx: int(tensor.shape[idx]))
    other_elements = count // int(tensor.shape[axis])
    width = max(1, chunk_size // max(other_elements, 1))
    for start in range(0, int(tensor.shape[axis]), width):
        slices = [slice(None)] * tensor.ndim
        slices[axis] = slice(start, start + width)
        yield from _tensor_chunks(tensor[tuple(slices)], chunk_size)


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

        # Dense tensor processing with bounded chunked allocations
        is_fp = detached.dtype.is_floating_point
        chunk_size = 1_048_576

        dense_min: float | None = None
        dense_max: float | None = None
        finite_count = 0
        mean_acc = 0.0
        m2_acc = 0.0
        zero_count = 0
        nonfinite_count = 0

        for chunk in _tensor_chunks(detached, chunk_size=chunk_size):
            zero_count += int((chunk == 0).sum().item())
            if is_fp:
                chunk_nan = int(torch.isnan(chunk).sum().item())
                chunk_inf = int(torch.isinf(chunk).sum().item())
                chunk_nonfinite = chunk_nan + chunk_inf
                nonfinite_count += chunk_nonfinite
                if chunk_nonfinite > 0:
                    mask = torch.isfinite(chunk)
                    valid = chunk[mask]
                else:
                    valid = chunk
            else:
                valid = chunk

            valid_size = int(valid.numel())
            if valid_size == 0:
                continue

            valid_float = (
                valid
                if valid.dtype in (torch.float32, torch.float64)
                else valid.to(dtype=torch.float32)
            )
            c_min = float(valid_float.min().item())
            c_max = float(valid_float.max().item())
            dense_min = c_min if dense_min is None else min(dense_min, c_min)
            dense_max = c_max if dense_max is None else max(dense_max, c_max)

            c_mean = float(valid_float.mean().item())
            c_m2 = float(((valid_float - c_mean) ** 2).sum().item())

            if finite_count == 0:
                mean_acc, m2_acc = c_mean, c_m2
                finite_count = valid_size
            else:
                comb = finite_count + valid_size
                delta = c_mean - mean_acc
                m2_acc += c_m2 + delta**2 * finite_count * valid_size / comb
                mean_acc += delta * valid_size / comb
                finite_count = comb

        if finite_count > 0:
            t_min = dense_min if dense_min is not None else float("nan")
            t_max = dense_max if dense_max is not None else float("nan")
            t_mean = mean_acc
            t_std = (
                math.sqrt(max(0.0, m2_acc / float(finite_count - 1))) if finite_count > 1 else 0.0
            )
            sparsity = zero_count / float(num_elements)
        else:
            t_min, t_max, t_mean, t_std = (
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
            )
            sparsity = 0.0

        # Memory-safe bounded sampling: only transfer max_samples elements to CPU
        sample_count = min(max_samples, num_elements)
        dense_samples: list[float] = []
        if sample_count == 0:
            dense_samples = []
        elif detached.is_contiguous():
            step = max(1, num_elements // sample_count)
            indices = torch.arange(0, num_elements, step, device=detached.device)[:sample_count]
            sampled_tensor = detached.view(-1)[indices]
            dense_samples = (
                sampled_tensor.float().cpu().tolist()
                if is_fp
                else sampled_tensor.to(dtype=torch.float32).cpu().tolist()
            )
        else:
            step = max(1, num_elements // sample_count)
            indices = torch.arange(0, num_elements, step, device=detached.device)[:sample_count]
            coords = []
            rem = indices
            for size in reversed(detached.shape):
                coords.append(torch.remainder(rem, int(size)))
                rem = torch.div(rem, int(size), rounding_mode="floor")
            sampled_tensor = detached[tuple(reversed(coords))]
            dense_samples = (
                sampled_tensor.float().cpu().tolist()
                if is_fp
                else sampled_tensor.to(dtype=torch.float32).cpu().tolist()
            )

        sampled_sanitized = [
            val if math.isfinite(val) else (999999.0 if val > 0 else -999999.0)
            for val in dense_samples
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
