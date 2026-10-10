"""Intermediate activation and feature map extraction with diagnostics."""

from __future__ import annotations

from typing import Any

from .exceptions import MissingDependencyError


def capture_activations(
    model: Any,
    example_input: Any,
    *,
    layers: str | list[str] = "all",
    max_points: int = 2_048,
) -> tuple[dict[str, Any], str, list[str]]:
    """Capture intermediate layer activations, feature maps, and saturation diagnostics."""
    from ._inspect import _activation_data, _torch_module

    torch = _torch_module()
    if torch is None or not isinstance(model, torch.nn.Module):
        raise MissingDependencyError(
            "Activation inspection requires PyTorch. Install it with `pip install 'seedl[torch]'`."
        )

    data, summary, notes = _activation_data(model, example_input, max_points)

    if layers != "all" and isinstance(layers, (list, tuple, set)):
        data["layers"] = [
            layer for layer in data["layers"] if layer["name"] in layers or layer["type"] in layers
        ]

    return data, summary, notes
