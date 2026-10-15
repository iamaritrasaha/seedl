"""Interactive CNN Feature Map Explorer and Spatial Activation Intelligence.

Provides genuine capture and visualization of intermediate convolutional activations,
spatial heatmaps, multi-channel galleries, channel distributions, and spatial comparisons.
"""

from __future__ import annotations

import html
import json
import math
from collections import defaultdict
from typing import Any

from .exceptions import MissingDependencyError, SeeDLError


def _torch():
    try:
        import torch

        return torch
    except ModuleNotFoundError:
        return None


def _is_conv_or_feature_module(module: Any) -> bool:
    torch = _torch()
    if torch is None:
        return False
    # Check for Conv2d, pooling, activations, or modules that process spatial feature maps
    conv_classes = (
        torch.nn.Conv2d,
        torch.nn.modules.conv._ConvNd,
        torch.nn.MaxPool2d,
        torch.nn.AvgPool2d,
        torch.nn.AdaptiveAvgPool2d,
        torch.nn.ReLU,
        torch.nn.LeakyReLU,
        torch.nn.GELU,
        torch.nn.SiLU,
        torch.nn.Sigmoid,
        torch.nn.Tanh,
        torch.nn.BatchNorm2d,
        torch.nn.Dropout2d,
    )
    return isinstance(module, conv_classes)


def _compute_channel_stats(grid: list[list[float]]) -> dict[str, Any]:
    """Compute exact numerical statistics for a 2D channel grid."""
    all_vals: list[float] = []
    nonfinite_count = 0
    zero_count = 0
    positive_count = 0
    negative_count = 0

    for row in grid:
        for val in row:
            if math.isfinite(val):
                all_vals.append(val)
                if val == 0.0:
                    zero_count += 1
                elif val > 0.0:
                    positive_count += 1
                else:
                    negative_count += 1
            else:
                nonfinite_count += 1

    total = len(all_vals)
    if total == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "sparsity": 1.0,
            "positive_fraction": 0.0,
            "negative_fraction": 0.0,
            "nonfinite_count": nonfinite_count,
            "has_negatives": False,
        }

    min_v = min(all_vals)
    max_v = max(all_vals)
    mean_v = sum(all_vals) / total
    variance = sum((v - mean_v) ** 2 for v in all_vals) / total
    std_v = math.sqrt(variance)

    return {
        "min": min_v,
        "max": max_v,
        "mean": mean_v,
        "std": std_v,
        "sparsity": zero_count / total,
        "positive_fraction": positive_count / total,
        "negative_fraction": negative_count / total,
        "nonfinite_count": nonfinite_count,
        "has_negatives": negative_count > 0,
    }


def capture_feature_maps(
    model_or_tensor: Any,
    example_input: Any = None,
    *,
    layers: str | list[str] | None = None,
    channel: int | None = None,
    batch_index: int = 0,
    max_channels: int = 64,
    max_spatial_dim: int = 128,
    theme: str = "scientific",
) -> tuple[dict[str, Any], str, list[str]]:
    """Capture real feature maps from a PyTorch model or directly from a 4D/3D tensor.

    Parameters:
        model_or_tensor: PyTorch nn.Module or 4D/3D Tensor/array.
        example_input: Model input tensor, tuple of args, or dict of kwargs.
        layers: Target layer name(s) or None for all candidate layers.
        channel: Default channel index to inspect (default 0).
        batch_index: Batch index to inspect (default 0).
        max_channels: Maximum number of channels to capture per layer (default 64).
        max_spatial_dim: Maximum spatial dimension before downsampling preview (default 128).
        theme: UI theme ('scientific', 'light', 'dark').
    """
    torch = _torch()

    # Case 1: Direct Tensor / Array input
    is_tensor = (torch is not None and isinstance(model_or_tensor, torch.Tensor)) or hasattr(
        model_or_tensor, "ndim"
    )

    if is_tensor or isinstance(model_or_tensor, (list, tuple)):
        return _capture_from_tensor(
            model_or_tensor,
            channel=channel,
            batch_index=batch_index,
            max_channels=max_channels,
            max_spatial_dim=max_spatial_dim,
            theme=theme,
        )

    # Case 2: PyTorch nn.Module
    if torch is None or not isinstance(model_or_tensor, torch.nn.Module):
        raise SeeDLError(
            "sd.features() expects a PyTorch nn.Module, a 4D/3D activation tensor, or a NumPy array."
        )

    if example_input is None:
        raise SeeDLError(
            "Visualizing feature maps from a PyTorch nn.Module requires an input tensor: "
            "sd.features(model, input=example_input)"
        )

    model = model_or_tensor
    candidate_modules: list[tuple[str, Any]] = []

    for name, module in model.named_modules():
        if name and _is_conv_or_feature_module(module):
            candidate_modules.append((name, module))

    if not candidate_modules:
        # Check if the root module itself can process 4D input
        candidate_modules.append(("(model)", model))

    # Filter layers if requested
    if layers is not None and layers != "all":
        target_names = {layers} if isinstance(layers, str) else set(layers)
        filtered = [
            (name, mod)
            for name, mod in candidate_modules
            if name in target_names or type(mod).__name__ in target_names
        ]
        if filtered:
            candidate_modules = filtered

    # Multi-call tracking
    call_counts: dict[str, int] = defaultdict(int)
    captured_layers: list[dict[str, Any]] = []
    handles = []

    # Save original training mode for every submodule to preserve mixed modes
    orig_training_modes = [(module, module.training) for module in model.modules()]
    saved_buffers = [
        (
            module,
            dict(module._buffers),
            set(module._non_persistent_buffers_set),
            {
                b_name: b_val.detach().clone()
                for b_name, b_val in module._buffers.items()
                if b_val is not None and b_val.numel() <= 8192
            },
        )
        for _, module in model.named_modules()
        if module._buffers
    ]

    def make_hook(layer_name: str, module_obj: Any):
        def hook(_mod: Any, _args: Any, output: Any):
            # Resolve tensor output
            out_tensor = output
            if isinstance(out_tensor, (tuple, list)):
                for item in out_tensor:
                    if isinstance(item, torch.Tensor) and item.ndim in {3, 4}:
                        out_tensor = item
                        break
                else:
                    return

            if not isinstance(out_tensor, torch.Tensor) or out_tensor.ndim not in {3, 4}:
                return

            call_counts[layer_name] += 1
            call_idx = call_counts[layer_name]
            display_name = f"{layer_name} (call {call_idx})" if call_idx > 1 else layer_name

            raw_shape = tuple(int(s) for s in out_tensor.shape)
            # Normalize to 4D [B, C, H, W]
            t_4d = out_tensor if out_tensor.ndim == 4 else out_tensor.unsqueeze(0)
            b_size, num_c, h, w = t_4d.shape

            b_idx = min(max(0, batch_index), b_size - 1)
            batch_slice = t_4d[b_idx].detach()

            num_captured = min(num_c, max_channels)
            downsampled = h > max_spatial_dim or w > max_spatial_dim
            target_h = min(h, max_spatial_dim)
            target_w = min(w, max_spatial_dim)

            channels_data: list[dict[str, Any]] = []
            layer_min: float | None = None
            layer_max: float | None = None
            layer_sum = 0.0
            layer_count = 0

            for c in range(num_captured):
                ch_mat = batch_slice[c].to(dtype=torch.float32)
                if downsampled:
                    ch_mat = torch.nn.functional.adaptive_avg_pool2d(
                        ch_mat.unsqueeze(0).unsqueeze(0),
                        (target_h, target_w),
                    )[0, 0]

                ch_grid = ch_mat.cpu().tolist()
                stats = _compute_channel_stats(ch_grid)

                if layer_min is None or stats["min"] < layer_min:
                    layer_min = stats["min"]
                if layer_max is None or stats["max"] > layer_max:
                    layer_max = stats["max"]
                layer_sum += stats["mean"] * (target_h * target_w)
                layer_count += target_h * target_w

                channels_data.append(
                    {
                        "index": c,
                        "grid": ch_grid,
                        "h": target_h,
                        "w": target_w,
                        "stats": stats,
                    }
                )

            layer_mean = (layer_sum / layer_count) if layer_count > 0 else 0.0

            captured_layers.append(
                {
                    "name": display_name,
                    "module_name": layer_name,
                    "type": type(module_obj).__name__,
                    "original_shape": raw_shape,
                    "captured_shape": (1, num_captured, target_h, target_w),
                    "total_channels": num_c,
                    "captured_channels": num_captured,
                    "batch_size": b_size,
                    "batch_index": b_idx,
                    "downsampled": downsampled,
                    "original_spatial": (h, w),
                    "channels": channels_data,
                    "layer_stats": {
                        "min": layer_min if layer_min is not None else 0.0,
                        "max": layer_max if layer_max is not None else 0.0,
                        "mean": layer_mean,
                    },
                }
            )

        return hook

    try:
        for name, module in candidate_modules:
            handles.append(module.register_forward_hook(make_hook(name, module)))

        model.eval()
        with torch.no_grad():
            if isinstance(example_input, dict):
                model(**example_input)
            elif isinstance(example_input, (tuple, list)):
                model(*example_input)
            else:
                model(example_input)
    except SeeDLError:
        raise
    except Exception as exc:
        raise SeeDLError(f"Failed to inspect CNN feature maps during forward pass: {exc}") from exc
    finally:
        for h in handles:
            h.remove()
        with torch.no_grad():
            for module, orig_buffers, non_persistent, saved_vals in saved_buffers:
                module._non_persistent_buffers_set = non_persistent
                module._buffers = orig_buffers
                for b_name, b_saved in saved_vals.items():
                    if orig_buffers.get(b_name) is not None and b_saved is not None:
                        if orig_buffers[b_name].shape != b_saved.shape:
                            orig_buffers[b_name].resize_(b_saved.shape)
                        orig_buffers[b_name].copy_(b_saved)
        for module, orig_mode in orig_training_modes:
            module.training = orig_mode

    if not captured_layers:
        raise SeeDLError(
            "No 2D convolutional or feature-map activations were observed during model execution. "
            "Ensure the model contains Conv2d or spatial layers, and check the input shape."
        )

    # Assemble data payload
    sel_channel = channel if channel is not None and 0 <= channel < max_channels else 0
    first_layer = captured_layers[0]

    summary = (
        f"CNN Feature Maps: {len(captured_layers)} layer(s) captured · "
        f"{first_layer['name']} shape {first_layer['original_shape']}"
    )

    notes = [
        "Genuine intermediate activation maps captured under eval mode without gradient tracking.",
        "Spatial activation heatmaps show channel responses across the input field.",
    ]
    if any(lyr["downsampled"] for lyr in captured_layers):
        notes.append(
            f"High-resolution activations downsampled to {max_spatial_dim}×{max_spatial_dim} for responsive visualization; original spatial dimensions preserved."
        )
    if any(lyr["total_channels"] > max_channels for lyr in captured_layers):
        notes.append(
            f"Bounded preview: first {max_channels} channels captured per layer to conserve memory."
        )

    data = {
        "presentation": "feature_maps",
        "theme": theme,
        "layers": captured_layers,
        "selected_layer_idx": 0,
        "selected_channel_idx": min(sel_channel, len(first_layer["channels"]) - 1),
        "total_layers": len(captured_layers),
        "batch_index": batch_index,
        "normalization_modes": ["per_channel", "layer_global", "zero_centered"],
    }

    return data, summary, notes


def _capture_from_tensor(
    tensor_data: Any,
    *,
    channel: int | None = None,
    batch_index: int = 0,
    max_channels: int = 64,
    max_spatial_dim: int = 128,
    theme: str = "scientific",
) -> tuple[dict[str, Any], str, list[str]]:
    """Capture feature map payload directly from a 3D or 4D tensor / array."""
    torch = _torch()

    # Convert to torch tensor if available, otherwise NumPy/Python list
    if torch is not None and isinstance(tensor_data, torch.Tensor):
        t = tensor_data.detach()
    elif hasattr(tensor_data, "tolist"):
        if torch is not None:
            t = torch.as_tensor(tensor_data)
        else:
            t = tensor_data
    else:
        if torch is not None:
            t = torch.tensor(tensor_data, dtype=torch.float32)
        else:
            t = tensor_data

    # Extract shape
    if hasattr(t, "shape"):
        shape = tuple(int(s) for s in t.shape)
    else:
        # Measure nested list shape
        dims = []
        curr = t
        while isinstance(curr, list) and curr:
            dims.append(len(curr))
            curr = curr[0]
        shape = tuple(dims)

    if len(shape) < 2 or len(shape) > 4:
        raise SeeDLError(
            f"Feature map tensor must be 3D [C, H, W] or 4D [B, C, H, W]; received shape {shape}."
        )

    t_4d: Any
    if len(shape) == 2:
        # [H, W] -> treat as 1 batch, 1 channel
        h, w = shape
        b_size, num_c = 1, 1
        t_4d = t.unsqueeze(0).unsqueeze(0) if hasattr(t, "unsqueeze") else [[t]]
    elif len(shape) == 3:
        num_c, h, w = shape
        b_size = 1
        t_4d = t.unsqueeze(0) if hasattr(t, "unsqueeze") else [t]
    else:
        b_size, num_c, h, w = shape
        t_4d = t

    b_idx = min(max(0, batch_index), b_size - 1)
    num_captured = min(num_c, max_channels)
    downsampled = h > max_spatial_dim or w > max_spatial_dim
    target_h = min(h, max_spatial_dim)
    target_w = min(w, max_spatial_dim)

    channels_data: list[dict[str, Any]] = []
    layer_min: float | None = None
    layer_max: float | None = None
    layer_sum = 0.0
    layer_count = 0

    for c in range(num_captured):
        if hasattr(t_4d, "shape") and torch is not None:
            ch_mat = t_4d[b_idx, c].to(dtype=torch.float32)
            if downsampled:
                ch_mat = torch.nn.functional.adaptive_avg_pool2d(
                    ch_mat.unsqueeze(0).unsqueeze(0),
                    (target_h, target_w),
                )[0, 0]
            ch_grid = ch_mat.cpu().tolist() if hasattr(ch_mat, "cpu") else ch_mat.tolist()
        else:
            batch_item = t_4d[b_idx]
            raw_grid = batch_item[c]
            if downsampled:
                step_r = max(1, h // target_h)
                step_c = max(1, w // target_w)
                ch_grid = [
                    [raw_grid[r][col] for col in range(0, w, step_c)[:target_w]]
                    for r in range(0, h, step_r)[:target_h]
                ]
            else:
                ch_grid = raw_grid

        stats = _compute_channel_stats(ch_grid)

        if layer_min is None or stats["min"] < layer_min:
            layer_min = stats["min"]
        if layer_max is None or stats["max"] > layer_max:
            layer_max = stats["max"]
        layer_sum += stats["mean"] * (target_h * target_w)
        layer_count += target_h * target_w

        channels_data.append(
            {
                "index": c,
                "grid": ch_grid,
                "h": target_h,
                "w": target_w,
                "stats": stats,
            }
        )

    layer_mean = (layer_sum / layer_count) if layer_count > 0 else 0.0

    layer_payload = {
        "name": "Activation Tensor",
        "module_name": "Tensor",
        "type": "Tensor",
        "original_shape": shape,
        "captured_shape": (1, num_captured, target_h, target_w),
        "total_channels": num_c,
        "captured_channels": num_captured,
        "batch_size": b_size,
        "batch_index": b_idx,
        "downsampled": downsampled,
        "original_spatial": (h, w),
        "channels": channels_data,
        "layer_stats": {
            "min": layer_min if layer_min is not None else 0.0,
            "max": layer_max if layer_max is not None else 0.0,
            "mean": layer_mean,
        },
    }

    sel_channel = channel if channel is not None and 0 <= channel < num_captured else 0

    summary = (
        f"CNN Feature Map Tensor: shape {shape} ({num_c} channels, {h}×{w} spatial) · "
        f"Batch [{b_idx}]"
    )

    notes = [
        "Feature map values provided directly from tensor input.",
        f"Spatial resolution: {h}×{w} with {num_c} channels.",
    ]
    if downsampled:
        notes.append(
            f"Downsampled spatially to {target_h}×{target_w} for preview; original shape {shape}."
        )

    data = {
        "presentation": "feature_maps",
        "theme": theme,
        "layers": [layer_payload],
        "selected_layer_idx": 0,
        "selected_channel_idx": min(sel_channel, len(channels_data) - 1),
        "total_layers": 1,
        "batch_index": b_idx,
        "normalization_modes": ["per_channel", "layer_global", "zero_centered"],
    }

    return data, summary, notes


# ==============================================================================
# CNN FEATURE COMPARISON
# ==============================================================================


def compare_feature_maps(
    feat_a: Any,
    feat_b: Any,
    *,
    names: tuple[str, str] = ("State A", "State B"),
    theme: str = "scientific",
) -> tuple[dict[str, Any], str, list[str]]:
    """Compare two CNN feature map states (same input across layers, same layer across inputs, or training states).

    Validates compatible spatial dimensions and warns when dimensions mismatch.
    """
    # Extract channel data if passed as Visualization or raw payload
    data_a = feat_a.data if hasattr(feat_a, "data") else feat_a
    data_b = feat_b.data if hasattr(feat_b, "data") else feat_b

    # Fallback capture if passed raw tensors or models
    if not isinstance(data_a, dict) or "layers" not in data_a:
        data_a, _, _ = capture_feature_maps(feat_a)
    if not isinstance(data_b, dict) or "layers" not in data_b:
        data_b, _, _ = capture_feature_maps(feat_b)

    layer_a = data_a["layers"][0]
    layer_b = data_b["layers"][0]

    ch_a = layer_a["channels"][0]
    ch_b = layer_b["channels"][0]

    grid_a = ch_a["grid"]
    grid_b = ch_b["grid"]

    h_a, w_a = len(grid_a), len(grid_a[0]) if grid_a else 0
    h_b, w_b = len(grid_b), len(grid_b[0]) if grid_b else 0

    shape_match = (h_a == h_b) and (w_a == w_b)

    notes = [
        f"Comparing feature maps: {names[0]} ({layer_a['name']}) vs {names[1]} ({layer_b['name']}).",
    ]

    diff_grid: list[list[float]] | None = None
    diff_stats: dict[str, Any] | None = None

    if shape_match and h_a > 0 and w_a > 0:
        diff_grid = []
        abs_diffs = []
        sq_diffs = []

        for r in range(h_a):
            row_diff = []
            for c in range(w_a):
                va = grid_a[r][c]
                vb = grid_b[r][c]
                d = vb - va
                row_diff.append(d)
                abs_diffs.append(abs(d))
                sq_diffs.append(d * d)
            diff_grid.append(row_diff)

        mae = sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0.0
        mse = sum(sq_diffs) / len(sq_diffs) if sq_diffs else 0.0
        rmse = math.sqrt(mse)
        max_diff = max(abs_diffs) if abs_diffs else 0.0

        diff_stats = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "max_abs_diff": max_diff,
        }

        summary = (
            f"Feature Map Comparison: {names[0]} vs {names[1]} · "
            f"Shape ({h_a}×{w_a}) · MAE: {mae:.4f} · Max Diff: {max_diff:.4f}"
        )
        notes.append(
            f"Aligned spatial dimensions match ({h_a}×{w_a}). Element-wise difference computed: State B - State A."
        )
    else:
        summary = (
            f"Feature Map Comparison: {names[0]} ({h_a}×{w_a}) vs {names[1]} ({h_b}×{w_b}) · "
            f"Shape Mismatch"
        )
        notes.append(
            f"Spatial dimension mismatch: {names[0]} is {h_a}×{w_a} while {names[1]} is {h_b}×{w_b}. "
            "Element-wise spatial subtraction is withheld to prevent misleading unaligned comparisons."
        )

    payload = {
        "presentation": "feature_comparison",
        "theme": theme,
        "names": names,
        "shape_match": shape_match,
        "layer_a": layer_a,
        "layer_b": layer_b,
        "ch_a": ch_a,
        "ch_b": ch_b,
        "diff_grid": diff_grid,
        "diff_stats": diff_stats,
    }

    return payload, summary, notes


# ==============================================================================
# RENDERERS: PLOTLY, SVG, MATPLOTLIB, HTML
# ==============================================================================


def plotly_features(view: Any) -> Any:
    """Render interactive Plotly CNN Feature Map explorer with layer, channel, and normalization dropdowns."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise MissingDependencyError("Interactive feature map inspection requires Plotly.") from exc

    data = view.data
    layers = data.get("layers", [])
    if not layers:
        raise SeeDLError("No layers present in feature map data.")

    theme = data.get("theme", "scientific")
    bg = "#0f172a" if theme in {"dark", "scientific"} else "#f8fafc"
    fg = "#f8fafc" if theme in {"dark", "scientific"} else "#0f172a"
    grid_col = "#334155" if theme in {"dark", "scientific"} else "#e2e8f0"

    # Default to first layer and first channel
    first_layer = layers[0]
    channels = first_layer["channels"]
    first_ch = channels[0]
    stats = first_ch["stats"]

    # Choose colorscale based on whether activations are signed
    is_signed = stats.get("has_negatives", False)
    default_colorscale = "RdBu_r" if is_signed else "Teal"

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.65, 0.35],
        subplot_titles=[
            f"{first_layer['name']} · Channel {first_ch['index']}",
            "Activation Distribution",
        ],
        horizontal_spacing=0.12,
    )

    # 1. Heatmap trace
    grid = first_ch["grid"]
    h, w = len(grid), len(grid[0]) if grid else 0

    z_min = -max(abs(stats["min"]), abs(stats["max"])) if is_signed else stats["min"]
    z_max = max(abs(stats["min"]), abs(stats["max"])) if is_signed else stats["max"]

    fig.add_trace(
        go.Heatmap(
            z=grid,
            x=list(range(w)),
            y=list(range(h)),
            colorscale=default_colorscale,
            zmin=z_min,
            zmax=z_max,
            colorbar=dict(title="Activation", x=0.62, len=0.85),
            hoverongaps=False,
            hovertemplate=(
                f"<b>{first_layer['name']} (Ch {first_ch['index']})</b><br>"
                "Coord: (y=%{y}, x=%{x})<br>"
                "Activation: %{z:.4f}<extra></extra>"
            ),
            name=f"Ch {first_ch['index']}",
        ),
        row=1,
        col=1,
    )

    # 2. Distribution Histogram trace
    flat_vals = [v for row in grid for v in row if math.isfinite(v)]
    fig.add_trace(
        go.Histogram(
            x=flat_vals,
            nbinsx=24,
            marker=dict(color="#38bdf8", line=dict(color=bg, width=1)),
            name="Values",
            hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Build Layer and Channel Dropdown Menus if multiple layers/channels exist
    updatemenus = []

    # Channel buttons for the first layer
    ch_buttons = []
    for ch in channels[:32]:
        c_stats = ch["stats"]
        c_grid = ch["grid"]
        c_signed = c_stats.get("has_negatives", False)
        c_zmin = -max(abs(c_stats["min"]), abs(c_stats["max"])) if c_signed else c_stats["min"]
        c_zmax = max(abs(c_stats["min"]), abs(c_stats["max"])) if c_signed else c_stats["max"]
        c_cs = "RdBu_r" if c_signed else "Teal"
        c_flat = [v for row in c_grid for v in row if math.isfinite(v)]

        ch_buttons.append(
            dict(
                method="update",
                label=f"Channel {ch['index']}",
                args=[
                    {
                        "z": [c_grid, None],
                        "x": [list(range(len(c_grid[0]) if c_grid else 0)), c_flat],
                        "y": [list(range(len(c_grid))), None],
                        "colorscale": [c_cs, None],
                        "zmin": [c_zmin, None],
                        "zmax": [c_zmax, None],
                    },
                    {
                        "subplot_titles": [
                            f"{first_layer['name']} · Channel {ch['index']}",
                            "Activation Distribution",
                        ]
                    },
                ],
            )
        )

    if len(ch_buttons) > 1:
        updatemenus.append(
            dict(
                buttons=ch_buttons,
                direction="down",
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.18,
                yanchor="top",
                bgcolor=bg,
                font=dict(color=fg, size=11),
                bordercolor=grid_col,
            )
        )

    # Normalization mode toggle
    norm_buttons = [
        dict(
            method="restyle",
            label="Per-Channel Scale",
            args=[
                {
                    "zmin": [z_min, None],
                    "zmax": [z_max, None],
                    "colorscale": [default_colorscale, None],
                }
            ],
        ),
        dict(
            method="restyle",
            label="Layer Global Scale",
            args=[
                {
                    "zmin": [first_layer["layer_stats"]["min"], None],
                    "zmax": [first_layer["layer_stats"]["max"], None],
                }
            ],
        ),
        dict(
            method="restyle",
            label="Zero-Centered (Symmetric)",
            args=[
                {
                    "zmin": [-max(abs(z_min), abs(z_max)), None],
                    "zmax": [max(abs(z_min), abs(z_max)), None],
                    "colorscale": ["RdBu_r", None],
                }
            ],
        ),
    ]

    updatemenus.append(
        dict(
            buttons=norm_buttons,
            direction="down",
            showactive=True,
            x=0.22,
            xanchor="left",
            y=1.18,
            yanchor="top",
            bgcolor=bg,
            font=dict(color=fg, size=11),
            bordercolor=grid_col,
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{view.title}</b><br><sup>Shape {first_layer['original_shape']} · Min: {stats['min']:.3f}, Max: {stats['max']:.3f}, Mean: {stats['mean']:.3f}, Zero%: {stats['sparsity']:.1%}</sup>",
            x=0.04,
            y=0.96,
            font=dict(size=18, color=fg),
        ),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(family="Inter, system-ui, sans-serif", color=fg),
        height=540,
        margin=dict(l=50, r=40, t=110, b=50),
        updatemenus=updatemenus,
    )

    # Invert y-axis on heatmap so row 0 is at top (standard visual matrix convention)
    fig.update_yaxes(autorange="reversed", title_text="Height (row)", row=1, col=1)
    fig.update_xaxes(title_text="Width (column)", row=1, col=1)
    fig.update_xaxes(title_text="Activation Value", row=1, col=2)
    fig.update_yaxes(title_text="Pixel Count", row=1, col=2)

    return fig


def plotly_feature_comparison(view: Any) -> Any:
    """Render Plotly side-by-side comparison of two CNN feature maps."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise MissingDependencyError("Feature map comparison requires Plotly.") from exc

    data = view.data
    theme = data.get("theme", "scientific")
    bg = "#0f172a" if theme in {"dark", "scientific"} else "#f8fafc"
    fg = "#f8fafc" if theme in {"dark", "scientific"} else "#0f172a"

    names = data.get("names", ("State A", "State B"))
    shape_match = data.get("shape_match", False)

    grid_a = data["ch_a"]["grid"]
    grid_b = data["ch_b"]["grid"]
    diff_grid = data.get("diff_grid")

    cols = 3 if shape_match and diff_grid else 2
    titles = [f"{names[0]}", f"{names[1]}"]
    if cols == 3:
        titles.append(f"Difference ({names[1]} - {names[0]})")

    fig = make_subplots(
        rows=1,
        cols=cols,
        subplot_titles=titles,
        horizontal_spacing=0.08,
    )

    # Trace A
    fig.add_trace(
        go.Heatmap(
            z=grid_a,
            colorscale="Teal",
            colorbar=dict(title=names[0], x=0.28, len=0.7),
            name=names[0],
        ),
        row=1,
        col=1,
    )

    # Trace B
    fig.add_trace(
        go.Heatmap(
            z=grid_b,
            colorscale="Teal",
            colorbar=dict(title=names[1], x=0.62 if cols == 3 else 0.98, len=0.7),
            name=names[1],
        ),
        row=1,
        col=2,
    )

    # Difference trace if available
    if cols == 3 and diff_grid:
        max_d = data["diff_stats"]["max_abs_diff"] if data.get("diff_stats") else 1.0
        fig.add_trace(
            go.Heatmap(
                z=diff_grid,
                colorscale="RdBu_r",
                zmin=-max_d,
                zmax=max_d,
                colorbar=dict(title="Diff", x=0.98, len=0.7),
                name="Difference",
            ),
            row=1,
            col=3,
        )

    for c in range(1, cols + 1):
        fig.update_yaxes(autorange="reversed", title_text="Height", row=1, col=c)
        fig.update_xaxes(title_text="Width", row=1, col=c)

    fig.update_layout(
        title=dict(
            text=f"<b>{view.title}</b><br><sup>{view.summary}</sup>",
            x=0.04,
            y=0.96,
            font=dict(size=18, color=fg),
        ),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(family="Inter, system-ui, sans-serif", color=fg),
        height=480,
        margin=dict(l=40, r=40, t=100, b=40),
    )

    return fig


def svg_features(view: Any) -> str:
    """Render clean static SVG of the selected CNN feature map."""
    data = view.data
    theme = data.get("theme", "scientific")
    bg = "#0f172a" if theme in {"dark", "scientific"} else "#f8fafc"
    fg = "#f8fafc" if theme in {"dark", "scientific"} else "#0f172a"
    muted = "#94a3b8" if theme in {"dark", "scientific"} else "#64748b"
    card_bg = "#1e293b" if theme in {"dark", "scientific"} else "#ffffff"
    border = "#334155" if theme in {"dark", "scientific"} else "#e2e8f0"

    layers = data.get("layers", [])
    if not layers:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="600" height="200"><rect width="600" height="200" fill="{bg}"/><text x="20" y="40" fill="{fg}">No feature maps captured</text></svg>'

    layer = layers[data.get("selected_layer_idx", 0)]
    ch_idx = min(data.get("selected_channel_idx", 0), len(layer["channels"]) - 1)
    ch = layer["channels"][ch_idx]
    grid = ch["grid"]
    stats = ch["stats"]

    gh = len(grid)
    gw = len(grid[0]) if gh > 0 else 0

    width = 780
    height = 520

    # Grid cell size with singleton dimension support
    max_grid_px = 320.0
    if gh == 1 and gw > 1:
        cell_x = max_grid_px / max(gw, 1)
        cell_y = min(40.0, max_grid_px)
    elif gw == 1 and gh > 1:
        cell_x = min(40.0, max_grid_px)
        cell_y = max_grid_px / max(gh, 1)
    elif gh == 1 and gw == 1:
        cell_x = 120.0
        cell_y = 120.0
    else:
        cell = min(max_grid_px / max(gw, 1), max_grid_px / max(gh, 1))
        cell_x = cell
        cell_y = cell

    grid_render_w = cell_x * gw
    grid_render_h = cell_y * gh

    gx0 = 40
    gy0 = 130

    min_v = stats["min"]
    max_v = stats["max"]
    v_range = (max_v - min_v) if max_v > min_v else 1.0

    rects = []
    for r in range(gh):
        for c in range(gw):
            val = grid[r][c]
            norm = (val - min_v) / v_range if v_range > 0 else 0.5
            norm = max(0.0, min(1.0, norm))

            # Color calculation (Teal or cool warm)
            if stats.get("has_negatives", False):
                # Diverging: blue for neg, red for pos
                if val < 0:
                    r_col = int(56 + (1 - abs(val / (abs(min_v) or 1))) * 100)
                    g_col = int(189 + (1 - abs(val / (abs(min_v) or 1))) * 40)
                    b_col = 248
                else:
                    r_col = 244
                    g_col = int(63 + (1 - val / (max_v or 1)) * 120)
                    b_col = int(94 + (1 - val / (max_v or 1)) * 120)
            else:
                # Teal colormap: from dark to bright cyan
                r_col = int(15 + norm * 30)
                g_col = int(40 + norm * 180)
                b_col = int(80 + norm * 160)

            fill = f"rgb({r_col},{g_col},{b_col})"
            rx = gx0 + c * cell_x
            ry = gy0 + r * cell_y
            rects.append(
                f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{cell_x:.1f}" height="{cell_y:.1f}" fill="{fill}">'
                f"<title>({r}, {c}): {val:.4f}</title></rect>"
            )

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        f'<rect width="{width}" height="{height}" fill="{bg}"/>',
        f'<text x="40" y="38" font-family="system-ui, sans-serif" font-size="12" fill="{muted}" font-weight="600">SeeDL / CNN FEATURE MAP EXPLORER</text>',
        f'<text x="40" y="70" font-family="system-ui, sans-serif" font-size="20" fill="{fg}" font-weight="700">{html.escape(layer["name"])} · Channel {ch["index"]}</text>',
        f'<text x="40" y="92" font-family="system-ui, sans-serif" font-size="13" fill="{muted}">Shape: {layer["original_shape"]} · Spatial: {gh}×{gw} pixels · Total Channels: {layer["total_channels"]}</text>',
        # Heatmap grid container
        f'<g transform="translate(0, 0)">{"".join(rects)}</g>',
        f'<rect x="{gx0}" y="{gy0}" width="{grid_render_w:.1f}" height="{grid_render_h:.1f}" fill="none" stroke="{border}" stroke-width="1.5"/>',
        # Stats panel on right
        f'<g transform="translate({gx0 + grid_render_w + 40}, {gy0})">',
        f'<rect width="320" height="300" rx="8" fill="{card_bg}" stroke="{border}" stroke-width="1"/>',
        f'<text x="20" y="32" font-family="system-ui, sans-serif" font-size="14" font-weight="700" fill="{fg}">Channel Diagnostics</text>',
        f'<text x="20" y="65" font-family="system-ui, sans-serif" font-size="12" fill="{muted}">Minimum Value</text>',
        f'<text x="200" y="65" font-family="system-ui, sans-serif" font-size="12" font-weight="600" fill="{fg}" text-anchor="end">{stats["min"]:.4f}</text>',
        f'<text x="20" y="95" font-family="system-ui, sans-serif" font-size="12" fill="{muted}">Maximum Value</text>',
        f'<text x="200" y="95" font-family="system-ui, sans-serif" font-size="12" font-weight="600" fill="{fg}" text-anchor="end">{stats["max"]:.4f}</text>',
        f'<text x="20" y="125" font-family="system-ui, sans-serif" font-size="12" fill="{muted}">Mean Activation</text>',
        f'<text x="200" y="125" font-family="system-ui, sans-serif" font-size="12" font-weight="600" fill="{fg}" text-anchor="end">{stats["mean"]:.4f}</text>',
        f'<text x="20" y="155" font-family="system-ui, sans-serif" font-size="12" fill="{muted}">Std. Deviation</text>',
        f'<text x="200" y="155" font-family="system-ui, sans-serif" font-size="12" font-weight="600" fill="{fg}" text-anchor="end">{stats["std"]:.4f}</text>',
        f'<text x="20" y="185" font-family="system-ui, sans-serif" font-size="12" fill="{muted}">Zero/Inactive Units</text>',
        f'<text x="200" y="185" font-family="system-ui, sans-serif" font-size="12" font-weight="600" fill="{fg}" text-anchor="end">{stats["sparsity"]:.1%}</text>',
        f'<text x="20" y="215" font-family="system-ui, sans-serif" font-size="12" fill="{muted}">Positive Fraction</text>',
        f'<text x="200" y="215" font-family="system-ui, sans-serif" font-size="12" font-weight="600" fill="{fg}" text-anchor="end">{stats["positive_fraction"]:.1%}</text>',
        f'<text x="20" y="255" font-family="system-ui, sans-serif" font-size="11" fill="{muted}">Downsampled: {"Yes" if layer["downsampled"] else "No (native resolution)"}</text>',
        "</g>",
        "</svg>",
    ]

    return "\n".join(svg_lines)


def svg_feature_comparison(view: Any) -> str:
    """Render static SVG of CNN feature map comparison."""
    data = view.data
    theme = data.get("theme", "scientific")
    bg = "#0f172a" if theme in {"dark", "scientific"} else "#f8fafc"
    fg = "#f8fafc" if theme in {"dark", "scientific"} else "#0f172a"
    muted = "#94a3b8" if theme in {"dark", "scientific"} else "#64748b"

    names = data.get("names", ("State A", "State B"))
    width = 800
    height = 420

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        f'<rect width="{width}" height="{height}" fill="{bg}"/>',
        f'<text x="40" y="38" font-family="system-ui, sans-serif" font-size="12" fill="{muted}" font-weight="600">SeeDL / FEATURE MAP COMPARISON</text>',
        f'<text x="40" y="70" font-family="system-ui, sans-serif" font-size="20" fill="{fg}" font-weight="700">{html.escape(view.title)}</text>',
        f'<text x="40" y="92" font-family="system-ui, sans-serif" font-size="13" fill="{muted}">{html.escape(view.summary)}</text>',
        f'<text x="140" y="130" font-family="system-ui, sans-serif" font-size="14" font-weight="700" fill="{fg}">{html.escape(str(names[0]))}</text>',
        f'<text x="440" y="130" font-family="system-ui, sans-serif" font-size="14" font-weight="700" fill="{fg}">{html.escape(str(names[1]))}</text>',
    ]

    # Draw previews of grid A and grid B
    for grid, gx0, _label in (
        (data["ch_a"]["grid"], 60, names[0]),
        (data["ch_b"]["grid"], 360, names[1]),
    ):
        gh = len(grid)
        gw = len(grid[0]) if gh > 0 else 0
        cell = min(220 / max(gh, 1), 220 / max(gw, 1))
        rects = []
        vals = [v for row in grid for v in row if math.isfinite(v)]
        min_v = min(vals) if vals else 0.0
        max_v = max(vals) if vals else 1.0
        v_rng = max_v - min_v if max_v > min_v else 1.0

        for r in range(gh):
            for c in range(gw):
                val = grid[r][c]
                norm = max(0.0, min(1.0, (val - min_v) / v_rng))
                c_val = int(40 + norm * 200)
                rects.append(
                    f'<rect x="{gx0 + c * cell:.1f}" y="{150 + r * cell:.1f}" width="{cell:.1f}" height="{cell:.1f}" fill="rgb(20,{c_val},{int(c_val * 0.8)})"/>'
                )
        svg_lines.append("".join(rects))

    svg_lines.append("</svg>")
    return "\n".join(svg_lines)


def matplotlib_features(view: Any) -> Any:
    """Render Matplotlib figure of CNN feature map and value distribution."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingDependencyError("Matplotlib rendering requires matplotlib.") from exc

    data = view.data
    layers = data.get("layers", [])
    if not layers:
        raise SeeDLError("No layers to render.")

    layer = layers[data.get("selected_layer_idx", 0)]
    ch = layer["channels"][min(data.get("selected_channel_idx", 0), len(layer["channels"]) - 1)]
    grid = ch["grid"]
    stats = ch["stats"]

    fig, (ax_map, ax_hist) = plt.subplots(
        1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [1.4, 1]}
    )

    is_signed = stats.get("has_negatives", False)
    cmap = "RdBu_r" if is_signed else "viridis"

    im = ax_map.imshow(grid, cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax_map, label="Activation Value")
    ax_map.set_title(f"{layer['name']} · Ch {ch['index']}", fontsize=11, fontweight="bold")
    ax_map.set_xlabel("Width (col)")
    ax_map.set_ylabel("Height (row)")

    flat_vals = [v for row in grid for v in row if math.isfinite(v)]
    ax_hist.hist(flat_vals, bins=25, color="#0284c7", edgecolor="black", alpha=0.85)
    ax_hist.set_title("Activation Distribution", fontsize=11, fontweight="bold")
    ax_hist.set_xlabel("Value")
    ax_hist.set_ylabel("Count")

    fig.suptitle(f"{view.title}\n{view.summary}", fontsize=12)
    fig.tight_layout()
    return fig


def matplotlib_feature_comparison(view: Any) -> Any:
    """Render Matplotlib comparison of two CNN feature maps."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingDependencyError("Matplotlib rendering requires matplotlib.") from exc

    data = view.data
    names = data.get("names", ("State A", "State B"))
    grid_a = data["ch_a"]["grid"]
    grid_b = data["ch_b"]["grid"]
    diff_grid = data.get("diff_grid")

    n_cols = 3 if diff_grid else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    im_a = axes[0].imshow(grid_a, cmap="viridis", aspect="auto")
    axes[0].set_title(names[0])
    fig.colorbar(im_a, ax=axes[0])

    im_b = axes[1].imshow(grid_b, cmap="viridis", aspect="auto")
    axes[1].set_title(names[1])
    fig.colorbar(im_b, ax=axes[1])

    if diff_grid and n_cols == 3:
        max_d = data.get("diff_stats", {}).get("max_abs_diff", 1.0)
        im_d = axes[2].imshow(diff_grid, cmap="RdBu_r", vmin=-max_d, vmax=max_d, aspect="auto")
        axes[2].set_title(f"Diff ({names[1]} - {names[0]})")
        fig.colorbar(im_d, ax=axes[2])

    fig.suptitle(view.title, fontsize=12)
    fig.tight_layout()
    return fig


def features_html(view: Any) -> str:
    """Generate self-contained, offline interactive HTML for CNN feature map exploration."""
    data = view.data
    title = html.escape(view.title)
    summary = html.escape(view.summary)
    payload_json = (
        json.dumps(data).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{title} — SeeDL CNN Feature Explorer</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-hover: #0284c7;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 24px;
      line-height: 1.5;
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 24px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    }}
    header {{
      margin-bottom: 20px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
    }}
    .badge {{
      display: inline-block;
      font-size: 11px;
      font-weight: 700;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 4px;
    }}
    h1 {{ font-size: 22px; color: var(--text); margin-bottom: 6px; }}
    .summary {{ font-size: 14px; color: var(--muted); }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      align-items: center;
      margin-bottom: 24px;
      background: rgba(15, 23, 42, 0.6);
      padding: 12px 16px;
      border-radius: 8px;
      border: 1px solid var(--border);
    }}
    .control-group {{
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    label {{ font-size: 12px; font-weight: 600; color: var(--muted); }}
    select, button {{
      background: var(--card);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 6px 12px;
      font-size: 13px;
      cursor: pointer;
    }}
    select:focus, button:focus {{ outline: 2px solid var(--accent); }}
    .btn-active {{
      background: var(--accent);
      color: #0f172a;
      font-weight: 600;
      border-color: var(--accent);
    }}
    .view-grid {{
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 24px;
      margin-bottom: 24px;
    }}
    @media (max-width: 900px) {{
      .view-grid {{ grid-template-columns: 1fr; }}
    }}
    .canvas-container {{
      background: #020617;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      align-items: center;
      position: relative;
    }}
    canvas {{
      image-rendering: pixelated;
      max-width: 100%;
      height: auto;
      border: 1px solid var(--border);
    }}
    .hover-hud {{
      margin-top: 12px;
      font-size: 13px;
      font-family: monospace;
      color: var(--accent);
      min-height: 20px;
    }}
    .stats-card {{
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
    }}
    .stats-card h3 {{
      font-size: 14px;
      margin-bottom: 12px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 6px;
      color: var(--accent);
    }}
    .stat-row {{
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      padding: 6px 0;
      border-bottom: 1px solid rgba(255,255,255,0.05);
    }}
    .stat-row:last-child {{ border-bottom: none; }}
    .stat-label {{ color: var(--muted); }}
    .stat-val {{ font-weight: 600; font-family: monospace; }}
    .gallery-section {{
      margin-top: 24px;
      border-top: 1px solid var(--border);
      padding-top: 20px;
    }}
    .gallery-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(90px, 1fr));
      gap: 10px;
      margin-top: 12px;
      max-height: 280px;
      overflow-y: auto;
      padding: 8px;
      background: rgba(15, 23, 42, 0.4);
      border-radius: 8px;
    }}
    .gallery-item {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 6px;
      text-align: center;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .gallery-item:hover, .gallery-item.selected {{
      border-color: var(--accent);
      background: rgba(56, 189, 248, 0.1);
    }}
    .gallery-item canvas {{ width: 100%; height: auto; }}
    .gallery-label {{ font-size: 10px; margin-top: 4px; color: var(--muted); }}
    footer {{
      margin-top: 24px;
      font-size: 12px;
      color: var(--muted);
      display: flex;
      justify-content: space-between;
      border-top: 1px solid var(--border);
      padding-top: 12px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="badge">SeeDL · Vision Intelligence 0.4.0</div>
      <h1>{title}</h1>
      <div class="summary">{summary}</div>
    </header>

    <div class="controls">
      <div class="control-group">
        <label for="layerSelect">Layer</label>
        <select id="layerSelect"></select>
      </div>

      <div class="control-group">
        <label for="channelSelect">Channel</label>
        <select id="channelSelect"></select>
      </div>

      <div class="control-group">
        <label>Normalization Mode</label>
        <div style="display:flex; gap:6px;">
          <button id="btnNormChannel" class="btn-active" onclick="setNorm('per_channel')">Channel</button>
          <button id="btnNormGlobal" onclick="setNorm('layer_global')">Global</button>
          <button id="btnNormZero" onclick="setNorm('zero_centered')">Zero-Centered</button>
        </div>
      </div>
    </div>

    <div class="view-grid">
      <div class="canvas-container">
        <canvas id="featureCanvas"></canvas>
        <div id="hoverHud" class="hover-hud">Hover over map to inspect coordinates and activation</div>
      </div>

      <div class="stats-card">
        <h3>Channel Diagnostics</h3>
        <div class="stat-row"><span class="stat-label">Layer Name</span><span id="stLayer" class="stat-val">—</span></div>
        <div class="stat-row"><span class="stat-label">Layer Type</span><span id="stType" class="stat-val">—</span></div>
        <div class="stat-row"><span class="stat-label">Original Shape</span><span id="stShape" class="stat-val">—</span></div>
        <div class="stat-row"><span class="stat-label">Spatial Resolution</span><span id="stSpatial" class="stat-val">—</span></div>
        <div class="stat-row"><span class="stat-label">Channel Index</span><span id="stChannel" class="stat-val">—</span></div>
        <div class="stat-row"><span class="stat-label">Min Activation</span><span id="stMin" class="stat-val">—</span></div>
        <div class="stat-row"><span class="stat-label">Max Activation</span><span id="stMax" class="stat-val">—</span></div>
        <div class="stat-row"><span class="stat-label">Mean Activation</span><span id="stMean" class="stat-val">—</span></div>
        <div class="stat-row"><span class="stat-label">Std. Deviation</span><span id="stStd" class="stat-val">—</span></div>
        <div class="stat-row"><span class="stat-label">Zero/Dead Units</span><span id="stSparsity" class="stat-val">—</span></div>
        <div class="stat-row"><span class="stat-label">Positive Fraction</span><span id="stPositive" class="stat-val">—</span></div>
        <div class="stat-row"><span class="stat-label">Downsampled</span><span id="stDownsampled" class="stat-val">—</span></div>
      </div>
    </div>

    <div class="gallery-section">
      <h3 style="font-size:14px; color:var(--text); margin-bottom:8px;">Channel Gallery Preview</h3>
      <div id="galleryGrid" class="gallery-grid"></div>
    </div>

    <footer>
      <span>Exported by SeeDL</span>
      <span>Authentic model activations · No synthetic demonstrations</span>
    </footer>
  </div>

  <script>
    const DATA = {payload_json};
    let currentLayerIdx = 0;
    let currentChannelIdx = 0;
    let normMode = "per_channel";

    const layerSelect = document.getElementById("layerSelect");
    const channelSelect = document.getElementById("channelSelect");
    const canvas = document.getElementById("featureCanvas");
    const ctx = canvas.getContext("2d");
    const hoverHud = document.getElementById("hoverHud");
    const galleryGrid = document.getElementById("galleryGrid");

    function init() {{
      DATA.layers.forEach((l, idx) => {{
        const opt = document.createElement("option");
        opt.value = idx;
        opt.textContent = `${{l.name}} (${{l.type}})`;
        layerSelect.appendChild(opt);
      }});

      layerSelect.addEventListener("change", (e) => {{
        currentLayerIdx = parseInt(e.target.value);
        currentChannelIdx = 0;
        updateChannelSelect();
        render();
      }});

      channelSelect.addEventListener("change", (e) => {{
        currentChannelIdx = parseInt(e.target.value);
        render();
      }});

      updateChannelSelect();
      render();
    }}

    function updateChannelSelect() {{
      channelSelect.innerHTML = "";
      const l = DATA.layers[currentLayerIdx];
      l.channels.forEach((ch, idx) => {{
        const opt = document.createElement("option");
        opt.value = idx;
        opt.textContent = `Ch ${{ch.index}} (mean: ${{ch.stats.mean.toFixed(2)}})`;
        channelSelect.appendChild(opt);
      }});
      channelSelect.value = currentChannelIdx;
    }}

    function setNorm(mode) {{
      normMode = mode;
      document.getElementById("btnNormChannel").className = mode === "per_channel" ? "btn-active" : "";
      document.getElementById("btnNormGlobal").className = mode === "layer_global" ? "btn-active" : "";
      document.getElementById("btnNormZero").className = mode === "zero_centered" ? "btn-active" : "";
      render();
    }}

    function getColor(val, minV, maxV, isSigned) {{
      let norm = 0.5;
      if (normMode === "zero_centered" || isSigned) {{
        const maxAbs = Math.max(Math.abs(minV), Math.abs(maxV)) || 1.0;
        const normSym = Math.max(-1.0, Math.min(1.0, val / maxAbs));
        if (normSym < 0) {{
          const t = Math.abs(normSym);
          return [Math.round(56 + (1 - t) * 100), Math.round(189 + (1 - t) * 40), 248];
        }} else {{
          const t = normSym;
          return [244, Math.round(63 + (1 - t) * 120), Math.round(94 + (1 - t) * 120)];
        }}
      }} else {{
        const rng = maxV - minV || 1.0;
        norm = Math.max(0.0, Math.min(1.0, (val - minV) / rng));
        return [Math.round(15 + norm * 35), Math.round(40 + norm * 180), Math.round(80 + norm * 160)];
      }}
    }}

    function render() {{
      const layer = DATA.layers[currentLayerIdx];
      const ch = layer.channels[currentChannelIdx];
      const grid = ch.grid;
      const gh = grid.length;
      const gw = gh > 0 ? grid[0].length : 0;
      const stats = ch.stats;

      // Update stats HUD
      document.getElementById("stLayer").textContent = layer.name;
      document.getElementById("stType").textContent = layer.type;
      document.getElementById("stShape").textContent = JSON.stringify(layer.original_shape);
      document.getElementById("stSpatial").textContent = `${{gh}} × ${{gw}} px`;
      document.getElementById("stChannel").textContent = `Ch ${{ch.index}} (of ${{layer.total_channels}})`;
      document.getElementById("stMin").textContent = stats.min.toFixed(4);
      document.getElementById("stMax").textContent = stats.max.toFixed(4);
      document.getElementById("stMean").textContent = stats.mean.toFixed(4);
      document.getElementById("stStd").textContent = stats.std.toFixed(4);
      document.getElementById("stSparsity").textContent = (stats.sparsity * 100).toFixed(1) + "%";
      document.getElementById("stPositive").textContent = (stats.positive_fraction * 100).toFixed(1) + "%";
      document.getElementById("stDownsampled").textContent = layer.downsampled ? "Yes (Preview)" : "No (Native)";

      // Setup canvas with singleton dimension support
      const maxCanvasPx = 360;
      let scaleX = Math.max(1, Math.floor(maxCanvasPx / Math.max(gw, 1)));
      let scaleY = Math.max(1, Math.floor(maxCanvasPx / Math.max(gh, 1)));
      if (gh === 1 && gw > 1) {{
        scaleY = Math.max(28, scaleX);
      }} else if (gw === 1 && gh > 1) {{
        scaleX = Math.max(28, scaleY);
      }} else if (gh === 1 && gw === 1) {{
        scaleX = 120;
        scaleY = 120;
      }} else {{
        const uniformScale = Math.min(scaleX, scaleY);
        scaleX = uniformScale;
        scaleY = uniformScale;
      }}
      canvas.width = gw * scaleX;
      canvas.height = gh * scaleY;

      let minV = stats.min;
      let maxV = stats.max;
      if (normMode === "layer_global") {{
        minV = layer.layer_stats.min;
        maxV = layer.layer_stats.max;
      }}

      for (let r = 0; r < gh; r++) {{
        for (let c = 0; c < gw; c++) {{
          const v = grid[r][c];
          const [cr, cg, cb] = getColor(v, minV, maxV, stats.has_negatives);
          ctx.fillStyle = `rgb(${{cr}},${{cg}},${{cb}})`;
          ctx.fillRect(c * scaleX, r * scaleY, scaleX, scaleY);
        }}
      }}

      // Interaction: mouse hover inspection
      canvas.onmousemove = (e) => {{
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        const col = Math.floor(mouseX / (canvas.clientWidth / gw));
        const row = Math.floor(mouseY / (canvas.clientHeight / gh));
        if (row >= 0 && row < gh && col >= 0 && col < gw) {{
          const v = grid[row][col];
          hoverHud.textContent = `Coord (y=${{row}}, x=${{col}}) · Activation: ${{v.toFixed(4)}}`;
        }}
      }};

      canvas.onmouseleave = () => {{
        hoverHud.textContent = "Hover over map to inspect coordinates and activation";
      }};

      // Render gallery thumbnails
      renderGallery();
    }}

    function renderGallery() {{
      galleryGrid.innerHTML = "";
      const layer = DATA.layers[currentLayerIdx];
      layer.channels.slice(0, 32).forEach((ch, idx) => {{
        const div = document.createElement("div");
        div.className = "gallery-item" + (idx === currentChannelIdx ? " selected" : "");
        div.onclick = () => {{
          currentChannelIdx = idx;
          channelSelect.value = idx;
          render();
        }};

        const thumb = document.createElement("canvas");
        const gh = ch.grid.length;
        const gw = gh > 0 ? ch.grid[0].length : 0;
        thumb.width = gw;
        thumb.height = gh;
        const tctx = thumb.getContext("2d");
        for (let r = 0; r < gh; r++) {{
          for (let c = 0; c < gw; c++) {{
            const v = ch.grid[r][c];
            const [cr, cg, cb] = getColor(v, ch.stats.min, ch.stats.max, ch.stats.has_negatives);
            tctx.fillStyle = `rgb(${{cr}},${{cg}},${{cb}})`;
            tctx.fillRect(c, r, 1, 1);
          }}
        }}

        const lbl = document.createElement("div");
        lbl.className = "gallery-label";
        lbl.textContent = `Ch ${{ch.index}}`;

        div.appendChild(thumb);
        div.appendChild(lbl);
        galleryGrid.appendChild(div);
      }});
    }}

    init();
  </script>
</body>
</html>"""
    return doc


def feature_comparison_html(view: Any) -> str:
    """Generate self-contained interactive HTML for CNN feature map comparison."""
    data = view.data
    title = html.escape(view.title)
    summary = html.escape(view.summary)
    names = data.get("names", ("State A", "State B"))
    name_a = html.escape(str(names[0]))
    name_b = html.escape(str(names[1]))

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{title} — SeeDL Feature Comparison</title>
  <style>
    body {{ font-family: -apple-system, system-ui, sans-serif; background: #0f172a; color: #f8fafc; padding: 24px; }}
    .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 24px; max-width: 1000px; margin: 0 auto; }}
    h1 {{ font-size: 20px; margin-bottom: 6px; }}
    .summary {{ color: #94a3b8; font-size: 14px; margin-bottom: 20px; }}
    .comparison-grid {{ display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; margin-top: 16px; }}
    .panel {{ background: #020617; border: 1px solid #334155; border-radius: 8px; padding: 16px; text-align: center; }}
    .panel h3 {{ font-size: 14px; margin-bottom: 12px; color: #38bdf8; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>{title}</h1>
    <div class="summary">{summary}</div>
    <div class="comparison-grid">
      <div class="panel">
        <h3>{name_a}</h3>
        <div>Shape: {data["layer_a"]["original_shape"]}</div>
      </div>
      <div class="panel">
        <h3>{name_b}</h3>
        <div>Shape: {data["layer_b"]["original_shape"]}</div>
      </div>
    </div>
  </div>
</body>
</html>"""
    return doc
