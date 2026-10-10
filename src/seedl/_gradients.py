"""Gradient flow inspection, L2 norm analysis, and vanishing/exploding diagnostics."""

from __future__ import annotations

from typing import Any

from .exceptions import MissingDependencyError


def inspect_gradients(
    model: Any,
    *,
    max_points: int = 2_048,
) -> tuple[dict[str, Any], str, list[str]]:
    """Inspect parameter gradients with diagnostics for zero, vanishing, or exploding values."""
    from ._inspect import _gradient_data, _torch_module

    torch = _torch_module()
    if torch is None or not isinstance(model, torch.nn.Module):
        raise MissingDependencyError(
            "Gradient inspection requires PyTorch. Install it with `pip install 'seedl[torch]'`."
        )

    return _gradient_data(model, max_points)
