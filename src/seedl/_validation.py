"""Validation rules and backend resolution for SeeDL operations."""

from __future__ import annotations

import math
from typing import Any

from .exceptions import SeeDLError
from .renderers import resolve_backend

VALID_VIEWS = {
    "auto",
    "neurons",
    "3d",
    "architecture",
    "summary",
    "features",
    "feature_maps",
    "flow",
    "trace",
    "interactive",
    "explorer",
}

VALID_THEMES = {"light", "dark", "scientific", "minimal", "presentation"}

VALID_COLOR_BY = {"auto", "layer", "activation", "gradient"}

VALID_DETAILS = {"summary", "full", "leaves"}


def validate_max_points(max_points: int) -> None:
    """Validate that max_points is a positive integer."""
    if isinstance(max_points, bool) or not isinstance(max_points, int) or max_points < 1:
        raise ValueError("max_points must be a positive integer.")


def validate_see_options(
    *,
    view: str,
    theme: str,
    color_by: str,
    detail: str,
    max_neurons: int,
    max_connections: int,
    sample: int,
    min_weight: float,
    edge_opacity: float,
) -> None:
    """Validate common keyword options for the see() visualization function."""
    if view not in VALID_VIEWS:
        raise ValueError(
            "view must be auto, neurons, 3d, architecture, summary, features, feature_maps, flow, trace, interactive, or explorer."
        )
    if theme not in VALID_THEMES:
        raise ValueError("theme must be light, dark, scientific, minimal, or presentation.")
    if color_by not in VALID_COLOR_BY:
        raise ValueError("color_by must be auto, layer, activation, or gradient.")
    if detail not in VALID_DETAILS:
        raise ValueError("detail must be summary, full, or leaves.")
    for name, value, upper in (
        ("max_neurons", max_neurons, 1000),
        ("max_connections", max_connections, 20000),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
            raise ValueError(f"{name} must be an integer between 1 and {upper}.")
    if isinstance(sample, bool) or not isinstance(sample, int) or sample < 0:
        raise ValueError("sample must be a nonnegative integer.")
    if not math.isfinite(min_weight) or min_weight < 0:
        raise ValueError("min_weight must be finite and nonnegative.")
    if not math.isfinite(edge_opacity) or not 0 <= edge_opacity <= 1:
        raise ValueError("edge_opacity must be between 0 and 1.")


def validate_tensor_view_options(
    obj: Any,
    *,
    view: str,
    dim_names: list[str] | tuple[str, ...] | None,
    max_frames: int,
    compare: Any | None,
    is_tensor: bool,
    is_multidim_tensor: bool,
    is_module: bool,
) -> None:
    """Validate view, dim_names, and frame settings against target data structures."""
    if view in {"interactive", "explorer"}:
        if is_module:
            raise SeeDLError(
                f"view='{view}' is only supported for PyTorch tensors and NumPy arrays, not nn.Module models."
            )
        if not is_tensor:
            raise SeeDLError(
                f"view='{view}' is only supported for PyTorch tensors and NumPy arrays, got {type(obj).__name__}."
            )
        if not is_multidim_tensor:
            raise SeeDLError(
                f"view='{view}' requires at least 2 dimensions, got shape {getattr(obj, 'shape', ())}."
            )

    if dim_names is not None:
        if is_module:
            raise SeeDLError(
                "The dim_names= argument is only supported for PyTorch tensors and NumPy arrays, not nn.Module models."
            )
        if not is_tensor:
            raise SeeDLError(
                f"The dim_names= argument is only supported for PyTorch tensors and NumPy arrays, got {type(obj).__name__}."
            )

    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0:
        raise ValueError("max_frames must be a positive integer.")

    if view not in {"auto", "features", "feature_maps", "interactive", "explorer"} and (
        not is_module or compare is not None
    ):
        raise ValueError("Explicit view options require a model without compare=.")


def resolve_see_backend(
    backend: str,
    *,
    view: str,
    interactive: bool | None,
    is_multidim_tensor: bool,
) -> str:
    """Validate interactive options and determine the appropriate rendering backend."""
    if view == "3d":
        if backend not in {"auto", "plotly"} or interactive is False:
            raise ValueError("view='3d' requires Plotly and interactive rendering.")
        backend = "plotly"
    if interactive is not None and not isinstance(interactive, bool):
        raise TypeError("interactive must be True, False, or None.")

    launch_interactive_tensor = (
        interactive is True or view in {"interactive", "explorer"}
    ) and is_multidim_tensor

    if interactive is True:
        if launch_interactive_tensor:
            if backend == "auto":
                backend = "svg"
        else:
            if backend == "auto":
                backend = "plotly"
            elif backend != "plotly":
                raise ValueError("interactive=True requires backend='plotly' or backend='auto'.")
    elif interactive is False and backend == "auto":
        backend = "svg"
    return resolve_backend(backend)


def validate_object_arguments(
    obj: Any,
    *,
    input: Any | None,
    slice: Any | None,
    layout: str | None,
    is_module: bool,
    is_tensor: bool,
) -> None:
    """Validate that input, slice, and layout arguments are used with compatible targets."""
    if input is not None and not is_module:
        raise SeeDLError("The input= argument is only used when visualizing a PyTorch nn.Module.")

    if is_module:
        if slice is not None:
            raise SeeDLError(
                "The slice= argument is only supported for PyTorch tensors and NumPy arrays, not nn.Module models."
            )
        if layout is not None:
            raise SeeDLError(
                "The layout= argument is only supported for PyTorch tensors and NumPy arrays, not nn.Module models."
            )
    elif layout is not None and not is_tensor:
        raise SeeDLError(
            f"The layout= argument is only supported for PyTorch tensors and NumPy arrays, got {type(obj).__name__}."
        )
