"""Public SeeDL operations."""

from __future__ import annotations

from typing import Any

from ._inspect import activation_payload, gradient_payload, inspect_object, is_torch_module
from .exceptions import SeeDLError
from .renderers import resolve_backend
from .view import Visualization


def _validate_max_points(max_points: int) -> None:
    if isinstance(max_points, bool) or not isinstance(max_points, int) or max_points < 1:
        raise ValueError("max_points must be a positive integer.")


def _summary(kind: str, data: dict[str, Any], metadata: dict[str, Any]) -> str:
    if kind == "scalar":
        return str(data.get("caption", metadata.get("Type", "scalar")))
    if kind == "sequence":
        values = data.get("values", [])
        return f"{len(values):,} value(s)"
    if kind == "matrix":
        shape = data.get("shape", (len(data.get("values", [])), 0))
        return f"Matrix with shape {shape}"
    if kind == "image":
        return f"Image-like tensor with shape {data.get('shape')}"
    if kind == "distribution":
        return f"Sampled value distribution · shape {data.get('shape', 'unknown')}"
    if kind in {"model", "training"}:
        return str(data.get("summary", metadata.get("Summary", "")))
    if kind == "activations":
        return f"{len(data.get('layers', []))} layer output(s)"
    if kind == "gradients":
        return f"{len(data.get('layers', []))} parameter tensor(s)"
    return ""


def _show_result(view: Visualization) -> None:
    """Display in an IPython frontend or provide a short terminal fallback."""
    try:
        from IPython import get_ipython  # type: ignore[import-not-found]

        if get_ipython() is not None:
            from IPython.display import display  # type: ignore[import-not-found]

            display(view)
            return
    except ImportError:
        pass
    print(view)


def see(
    obj: Any,
    *,
    input: Any | None = None,
    backend: str = "auto",
    interactive: bool | None = None,
    max_points: int = 2_048,
) -> Visualization:
    """Automatically visualize a supported deep-learning or numeric object.

    Supported objects include PyTorch modules and tensors, NumPy arrays, real
    numbers, numeric sequences, and metric-history dictionaries. The function
    returns a :class:`Visualization` and also displays it in notebooks or prints
    a compact summary in a terminal.

    ``input`` is used only for PyTorch model inspection. The model is called once
    with that input, in eval mode and without gradients; temporary hooks and
    registered buffer changes are restored. No forward pass occurs without it.

    ``backend`` accepts ``"auto"`` (Plotly when installed, otherwise SVG),
    ``"plotly"``, ``"matplotlib"``, or ``"svg"``. Plotly and Matplotlib are
    optional extras. Large values are sampled for display; ``max_points`` sets
    the sampling limit.
    """
    _validate_max_points(max_points)
    if interactive is not None and not isinstance(interactive, bool):
        raise TypeError("interactive must be True, False, or None.")
    if interactive is True:
        if backend == "auto":
            backend = "plotly"
        elif backend != "plotly":
            raise ValueError("interactive=True requires backend='plotly' or backend='auto'.")
    elif interactive is False and backend == "auto":
        backend = "svg"
    if input is not None and not is_torch_module(obj):
        raise SeeDLError("The input= argument is only used when visualizing a PyTorch nn.Module.")
    kind, title, data, metadata, notes = inspect_object(
        obj, example_input=input, max_points=max_points
    )
    view = Visualization(
        kind=kind,
        title=title,
        summary=_summary(kind, data, metadata),
        data=data,
        metadata=metadata,
        notes=notes,
        backend=resolve_backend(backend),
    )
    _show_result(view)
    return view


def activations(
    model: Any,
    *,
    input: Any,
    backend: str = "auto",
    max_points: int = 2_048,
) -> Visualization:
    """Capture layer outputs from one explicit input-driven forward pass.

    Temporary forward hooks are removed even when the model raises an error.
    Model training flags and registered buffers are restored afterward. Each
    layer's first tensor output is sampled without retaining its graph.
    """
    _validate_max_points(max_points)
    data, summary, notes = activation_payload(model, input, max_points)
    view = Visualization(
        kind="activations",
        title=f"{type(model).__name__} activations",
        summary=summary,
        data=data,
        notes=notes,
        backend=resolve_backend(backend),
    )
    _show_result(view)
    return view


def gradients(
    model: Any,
    *,
    backend: str = "auto",
    max_points: int = 2_048,
) -> Visualization:
    """Visualize gradients that already exist on a PyTorch model's parameters.

    This function is read-only. It does not call ``backward()`` or modify model
    parameters. If no gradients are present, the view explains how to create them.
    """
    _validate_max_points(max_points)
    data, summary, notes = gradient_payload(model, max_points)
    view = Visualization(
        kind="gradients",
        title=f"{type(model).__name__} gradients",
        summary=summary,
        data=data,
        notes=notes,
        backend=resolve_backend(backend),
    )
    _show_result(view)
    return view
