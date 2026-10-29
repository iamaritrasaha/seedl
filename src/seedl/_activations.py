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
    zero_threshold: float = 0.8,
    saturation_threshold: float = 0.5,
) -> tuple[dict[str, Any], str, list[str]]:
    """Capture intermediate layer activations, feature maps, and saturation diagnostics."""
    from ._inspect import _activation_data, _torch_module

    torch = _torch_module()
    if torch is None or not isinstance(model, torch.nn.Module):
        raise MissingDependencyError(
            "Activation inspection requires PyTorch. Install it with `pip install 'seedl[torch]'`."
        )

    data, summary, notes = _activation_data(
        model,
        example_input,
        max_points,
        zero_threshold=zero_threshold,
        saturation_threshold=saturation_threshold,
    )
    total_captured = len(data["layers"])

    if layers != "all":
        target = [layers] if isinstance(layers, str) else list(layers)
        data["layers"] = [
            layer
            for layer in data["layers"]
            if layer["name"] in target
            or layer.get("base_name") in target
            or layer.get("module_name") in target
            or layer["type"] in target
        ]

        summary = f"Captured {len(data['layers'])} module activation(s) (filtered from {total_captured} total)"
        notes = [
            n
            for n in notes
            if not n.startswith("Dead unit alert")
            and not n.startswith("Saturation detected")
            and not n.startswith("Numerical instability")
        ]
        dead_layers = [
            layer["name"]
            for layer in data["layers"]
            if layer.get("zero_fraction", 0.0) > zero_threshold
            and any(
                act in layer.get("type", "").lower()
                for act in ("relu", "gelu", "silu", "sigmoid", "tanh")
            )
        ]
        if dead_layers:
            notes.append(
                f"Dead unit alert (heuristic): {len(dead_layers)} activation layer(s) have >{int(zero_threshold * 100)}% inactive zeros: {', '.join(dead_layers[:5])}. Note: observed response pattern on this input, not a permanent diagnosis."
            )
        sat_layers = [layer["name"] for layer in data["layers"] if layer.get("saturation_note")]
        if sat_layers:
            notes.append(
                f"Saturation detected in {len(sat_layers)} layer(s): {', '.join(sat_layers[:5])}"
            )
        unstable_layers = [
            layer["name"] for layer in data["layers"] if layer.get("instability_warning")
        ]
        if unstable_layers:
            notes.append(
                f"Numerical instability detected in {len(unstable_layers)} layer(s): {', '.join(unstable_layers[:5])}"
            )
        notes.append(
            f"Filtered to {len(data['layers'])} of {total_captured} captured module activation(s)."
        )

    return data, summary, notes
