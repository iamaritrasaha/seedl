"""Interactive Tensor Flow Explorer and execution flow visualization."""

from __future__ import annotations

from typing import Any

from ._execution_graph import ExecutionTrace
from ._trace import trace as trace_model
from .exceptions import SeeDLError
from .renderers import resolve_backend
from .view import Visualization


def flow(
    trace_or_model: Any,
    input: Any = None,
    *,
    record: tuple[str, ...] | list[str] = ("shapes", "activations"),
    max_nodes: int = 256,
    backend: str = "auto",
    title: str | None = None,
    mode: str = "graph",  # "graph", "timeline", "table", "summary"
) -> Visualization:
    """Create an interactive Tensor Flow and Neural Execution visualization.

    Parameters:
        trace_or_model: An ExecutionTrace produced by ``sd.trace()``, or a PyTorch ``nn.Module``.
        input: Required example input when passing an ``nn.Module``.
        record: Data to capture during trace ("shapes", "activations").
        max_nodes: Bounded maximum operations to record (default 256).
        backend: Rendering backend ('auto', 'svg', 'plotly', 'matplotlib').
        title: Optional custom visualization title.
        mode: Initial view mode ('graph', 'timeline', 'table', 'summary').
    """
    resolved_backend = resolve_backend(backend)

    if isinstance(trace_or_model, ExecutionTrace):
        exec_trace = trace_or_model
    else:
        # Assume model passed directly
        if input is None:
            raise SeeDLError(
                "Calling sd.flow(model) requires an input argument: sd.flow(model, input=example_input)"
            )
        exec_trace = trace_model(trace_or_model, input=input, record=record, max_nodes=max_nodes)

    data = exec_trace.to_dict()
    data["presentation"] = "execution_trace"
    data["initial_mode"] = mode

    total_ops = len(exec_trace.nodes)
    leaf_ops = len(exec_trace.get_leaf_nodes())
    vis_title = title or "Neural Execution Trace & Tensor Flow"
    summary = (
        f"{leaf_ops} executed leaf operation(s) ({total_ops} total) · "
        f"{exec_trace.total_parameters:,} parameters · {exec_trace.duration_ms:.2f} ms"
    )

    notes = [
        f"Execution trace captured via {exec_trace.trace_mode}.",
        f"Primary device: {exec_trace.device}.",
        f"Total model parameters: {exec_trace.total_parameters:,} ({exec_trace.trainable_parameters:,} trainable).",
    ]
    notes.extend(exec_trace.warnings)
    notes.extend(exec_trace.limitations)

    metadata = {
        "Leaf Operations": leaf_ops,
        "Total Nodes": total_ops,
        "Parameters": f"{exec_trace.total_parameters:,}",
        "Device": exec_trace.device,
        "Mode": exec_trace.trace_mode,
        "Duration (ms)": f"{exec_trace.duration_ms:.2f}",
    }

    return Visualization(
        kind="execution_trace",
        title=vis_title,
        summary=summary,
        data=data,
        metadata=metadata,
        notes=notes,
        backend=resolved_backend,
    )
