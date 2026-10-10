"""PyTorch model architecture and computation inspection."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from ._tensors import format_memory_bytes
from .exceptions import MissingDependencyError, SeeDLError


def _torch():
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError:
        return None


def inspect_model_architecture(
    model: Any,
    example_input: Any | None = None,
    *,
    detail: str = "summary",
    max_points: int = 2_048,
) -> tuple[dict[str, Any], str, list[str]]:
    """Inspect model hierarchy, parameter counts, and shape propagation."""
    torch = _torch()
    if torch is None or not isinstance(model, torch.nn.Module):
        raise MissingDependencyError(
            "Model inspection requires PyTorch. Install it with `pip install 'seedl[torch]'`."
        )

    # Check for uninitialized lazy parameters
    if any(isinstance(p, torch.nn.parameter.UninitializedParameter) for p in model.parameters()):
        raise SeeDLError(
            "This model has uninitialized lazy parameters. Run it once yourself to "
            "initialize them, then call SeeDL again."
        )

    observations: dict[int, dict[str, Any]] = {}
    execution_order: list[str] = []
    residual_clues: list[str] = []

    if example_input is not None:
        observations, execution_order, residual_clues = _trace_model_execution(
            model, example_input, torch
        )

    # Parameter accounting
    modules = list(model.named_modules())
    seen_params: dict[int, str] = {}
    shared_params: list[str] = []
    total_numel = 0
    trainable_numel = 0
    frozen_numel = 0
    total_bytes = 0

    for name, param in model.named_parameters(remove_duplicate=False):
        p_id = id(param)
        if p_id in seen_params:
            shared_params.append(f"{name} (shared with {seen_params[p_id]})")
            continue
        seen_params[p_id] = name
        try:
            num = int(param.numel())
            elem_size = param.element_size() if hasattr(param, "element_size") else 4
        except (RuntimeError, ValueError):
            num = 0
            elem_size = 4
        total_numel += num
        total_bytes += num * elem_size
        if param.requires_grad:
            trainable_numel += num
        else:
            frozen_numel += num

    direct_counts: dict[int, tuple[int, int, int]] = {}
    for _, module in modules:
        direct_params = list(module.parameters(recurse=False))
        d_total = sum(int(p.numel()) for p in direct_params if hasattr(p, "numel"))
        d_train = sum(
            int(p.numel()) for p in direct_params if p.requires_grad and hasattr(p, "numel")
        )
        d_frozen = d_total - d_train
        direct_counts[id(module)] = (d_total, d_train, d_frozen)

    layers: list[dict[str, Any]] = []
    max_depth = 0
    leaf_count = 0

    for name, module in modules:
        d_total, d_train, d_frozen = direct_counts[id(module)]
        depth = name.count(".") + (1 if name else 0)
        max_depth = max(max_depth, depth)
        is_leaf = len(list(module.children())) == 0
        if is_leaf:
            leaf_count += 1

        obs = observations.get(id(module), {})
        in_shape = obs.get("input_shape", "—")
        out_shape = obs.get("output_shape", "—")

        layer_info = {
            "name": name,
            "depth": depth,
            "type": type(module).__name__,
            "parameters": d_total,
            "trainable": d_train,
            "frozen": d_frozen,
            "is_leaf": is_leaf,
            "input_shape": in_shape,
            "output_shape": out_shape,
        }
        layers.append(layer_info)

    # Filter by detail level if requested
    filtered_layers = layers
    if detail == "leaves":
        filtered_layers = [layer for layer in layers if layer["is_leaf"]]

    summary = (
        f"{len(layers)} module(s) ({leaf_count} leaves) · "
        f"{total_numel:,} parameter(s) · {trainable_numel:,} trainable"
    )

    notes = [
        "Per-layer counts include direct parameters only; shared parameters are deduplicated in total counts.",
    ]
    if frozen_numel > 0:
        notes.append(f"Model contains {frozen_numel:,} frozen (non-trainable) parameters.")
    if shared_params:
        notes.append(f"Detected {len(shared_params)} shared parameter instance(s).")
    if total_bytes > 0:
        notes.append(f"Estimated parameter memory footprint: {format_memory_bytes(total_bytes)}.")

    if example_input is None:
        notes.append("Pass input=... to trace runtime input and output shapes through the layers.")
    else:
        notes.append(
            "An eval-mode forward pass under no_grad() was executed once with the supplied input; "
            "temporary hooks were removed and ordinary buffer updates were restored."
        )
        if residual_clues:
            notes.append(
                f"Architecture structure: {len(residual_clues)} branch/residual connection clue(s) observed."
            )

    data = {
        "layers": filtered_layers,
        "all_layers": layers,
        "parameters": total_numel,
        "trainable": trainable_numel,
        "frozen": frozen_numel,
        "memory_bytes": total_bytes,
        "max_depth": max_depth,
        "leaf_count": leaf_count,
        "shared_parameters": shared_params,
        "execution_order": execution_order,
        "residual_clues": residual_clues,
        "summary": summary,
    }

    return data, summary, notes


def _trace_model_execution(
    model: Any, example_input: Any, torch: Any
) -> tuple[dict[int, dict[str, Any]], list[str], list[str]]:
    """Safely execute a forward pass with hooks to capture shapes and execution graph."""
    modules = list(model.named_modules())
    states = [(m, bool(m.training)) for _, m in modules]
    buffers = [
        (
            m,
            dict(m._buffers),
            set(m._non_persistent_buffers_set),
            {k: v.detach().clone() if v is not None else None for k, v in m._buffers.items()},
        )
        for _, m in modules
    ]

    observations: dict[int, dict[str, Any]] = {}
    execution_order: list[str] = []
    residual_clues: list[str] = []
    handles = []

    def make_pre(m_id: int, m_name: str):
        def hook(_m: Any, args: Any) -> None:
            obs = observations.setdefault(m_id, {})
            obs["input_shape"] = _format_shape(args, torch)
            execution_order.append(m_name or "(root)")

        return hook

    def make_post(m_id: int, m_name: str):
        def hook(_m: Any, _args: Any, output: Any) -> None:
            obs = observations.setdefault(m_id, {})
            out_shape = _format_shape(output, torch)
            obs["output_shape"] = out_shape
            # Detect residual clues (e.g. shortcut, downsample, add, or matching shapes)
            if any(term in m_name.lower() for term in ("shortcut", "downsample", "res", "skip")):
                residual_clues.append(f"Skip/residual path: {m_name} -> {out_shape}")

        return hook

    try:
        for name, module in modules:
            handles.append(module.register_forward_pre_hook(make_pre(id(module), name)))
            handles.append(module.register_forward_hook(make_post(id(module), name)))
        model.eval()
        copied_input = _clone_val(example_input, torch)
        with torch.no_grad():
            if isinstance(copied_input, dict):
                model(**copied_input)
            elif isinstance(copied_input, tuple):
                model(*copied_input)
            else:
                model(copied_input)
    except Exception as exc:
        raise SeeDLError(
            "Could not inspect this model with the supplied input. "
            "Check input shape, dtype, and device. "
            f"Original error: {exc}"
        ) from exc
    finally:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass
        with torch.no_grad():
            for m, orig_buf, non_pers, saved_vals in buffers:
                try:
                    m._non_persistent_buffers_set = non_pers
                    m._buffers = orig_buf
                    for k, saved in saved_vals.items():
                        buf = orig_buf.get(k)
                        if buf is not None and saved is not None:
                            if buf.shape != saved.shape:
                                buf.resize_(saved.shape)
                            buf.copy_(saved)
                except Exception:
                    pass
        for m, training in states:
            m.training = training

    return observations, execution_order, residual_clues


def _format_shape(value: Any, torch: Any) -> str:
    if isinstance(value, torch.Tensor):
        return str(tuple(int(s) for s in value.shape))
    if isinstance(value, Mapping):
        parts = [f"{k}: {_format_shape(v, torch)}" for k, v in list(value.items())[:4]]
        return "{" + ", ".join(parts) + (", …" if len(value) > 4 else "") + "}"
    if isinstance(value, (tuple, list)):
        parts = [_format_shape(v, torch) for v in value[:4]]
        return "(" + ", ".join(parts) + (", …" if len(value) > 4 else "") + ")"
    return type(value).__name__


def _clone_val(value: Any, torch: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_val(v, torch) for v in value)
    if isinstance(value, list):
        return [_clone_val(v, torch) for v in value]
    if isinstance(value, dict):
        return {k: _clone_val(v, torch) for k, v in value.items()}
    return value
