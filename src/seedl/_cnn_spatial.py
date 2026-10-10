"""3D Convolutional Neural Network (CNN) feature map exploration.

Captures real intermediate convolutional activations and renders spatial
feature-map channel planes in 3D along the depth dimension.
"""

from __future__ import annotations

import html
import math
from typing import Any

from .exceptions import MissingDependencyError, SeeDLError


def is_cnn_model(model: Any) -> bool:
    """Check if model contains convolutional layers."""
    try:
        import torch

        return any(
            isinstance(m, (torch.nn.Conv2d, torch.nn.modules.conv._ConvNd)) for m in model.modules()
        )
    except ImportError:
        return False


def inspect_visual_cnn(
    model: Any,
    example_input: Any,
    *,
    theme: str = "scientific",
    sample: int = 0,
    max_channels_per_layer: int = 6,
    max_spatial_dim: int = 20,
) -> tuple[dict[str, Any], str, list[str]]:
    """Capture real feature maps from a CNN forward pass and build 3D spatial scene."""
    import torch

    conv_layers: list[dict[str, Any]] = []
    handles = []
    captured_data: dict[str, Any] = {}

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.AvgPool2d)):
            specs: dict[str, Any] = {}
            if hasattr(module, "kernel_size"):
                specs["kernel_size"] = module.kernel_size
            if hasattr(module, "stride"):
                specs["stride"] = module.stride
            if hasattr(module, "padding"):
                specs["padding"] = module.padding
            if hasattr(module, "out_channels"):
                specs["out_channels"] = module.out_channels
            if hasattr(module, "in_channels"):
                specs["in_channels"] = module.in_channels

            conv_layers.append(
                dict(
                    name=name or "(conv)",
                    module=module,
                    type=type(module).__name__,
                    specs=specs,
                )
            )

    if not conv_layers:
        raise SeeDLError("Model does not contain 2D convolutional layers.")

    # Forward hook to capture real feature maps
    def make_hook(layer_name: str):
        def hook(_mod: Any, _args: Any, output: Any):
            if isinstance(output, torch.Tensor) and output.ndim == 4:
                # Shape: [batch, channels, H, W]
                b_size = output.shape[0]
                s_idx = min(sample, b_size - 1)
                sample_tensor = output[s_idx].detach().cpu()
                num_c = sample_tensor.shape[0]
                h, w = sample_tensor.shape[1], sample_tensor.shape[2]

                # Bounded preview: take up to max_channels_per_layer
                c_indices = list(range(min(num_c, max_channels_per_layer)))
                channels_data = []

                for c in c_indices:
                    ch_mat = sample_tensor[c]
                    # Downsample spatially if larger than max_spatial_dim
                    if h > max_spatial_dim or w > max_spatial_dim:
                        ch_mat = torch.nn.functional.adaptive_avg_pool2d(
                            ch_mat.unsqueeze(0).unsqueeze(0),
                            (min(h, max_spatial_dim), min(w, max_spatial_dim)),
                        ).squeeze()

                    ch_list = ch_mat.tolist()
                    vals = [v for row in ch_list for v in row if math.isfinite(v)]
                    min_v = min(vals) if vals else 0.0
                    max_v = max(vals) if vals else 0.0
                    mean_v = sum(vals) / len(vals) if vals else 0.0

                    channels_data.append(
                        dict(
                            channel_index=c,
                            grid=ch_list,
                            h=len(ch_list),
                            w=len(ch_list[0]) if ch_list else 0,
                            min_val=min_v,
                            max_val=max_v,
                            mean_val=mean_v,
                        )
                    )

                captured_data[layer_name] = dict(
                    original_shape=list(output.shape),
                    total_channels=num_c,
                    original_h=h,
                    original_w=w,
                    channels=channels_data,
                    downsampled=h > max_spatial_dim or w > max_spatial_dim,
                )

        return hook

    if example_input is not None:
        for layer_info in conv_layers:
            h = layer_info["module"].register_forward_hook(make_hook(layer_info["name"]))
            handles.append(h)

        try:
            with torch.no_grad():
                model.eval()
                if isinstance(example_input, dict):
                    model(**example_input)
                elif isinstance(example_input, (tuple, list)):
                    model(*example_input)
                else:
                    model(example_input)
        finally:
            for h in handles:
                h.remove()

    # Build layers payload
    layer_payloads = []
    for idx, layer_info in enumerate(conv_layers):
        name = layer_info["name"]
        c_info = captured_data.get(name, {})
        layer_payloads.append(
            dict(
                index=idx,
                name=name,
                type=layer_info["type"],
                specs=layer_info["specs"],
                captured=c_info,
            )
        )

    summary = f"CNN 3D Architecture: {len(conv_layers)} convolutional/pooling block(s)" + (
        f" · Captured {len(captured_data)} activation tensor(s)" if captured_data else ""
    )

    notes = [
        "Feature maps are rendered in 3D as spatial planes stacked along the network depth axis (X).",
        f"Bounded preview: up to {max_channels_per_layer} channels per block shown; spatial downsampling applied if > {max_spatial_dim}×{max_spatial_dim}.",
    ]
    if example_input is None:
        notes.append("Pass input=image to capture real intermediate feature maps.")

    data = dict(
        presentation="cnn_3d",
        dimensions=3,
        theme=theme,
        sample=sample,
        layers=layer_payloads,
        total_layers=len(conv_layers),
    )

    return data, summary, notes


def plotly_cnn_3d(view: Any) -> Any:
    """Render 3D CNN spatial feature map planes in Plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise MissingDependencyError("3D CNN visualization requires Plotly.") from exc

    data = view.data
    theme = data.get("theme", "scientific")
    bg = "#0f172a" if theme in {"dark", "scientific"} else "#f8fafc"
    fg = "#f8fafc" if theme in {"dark", "scientific"} else "#0f172a"

    fig = go.Figure()
    layers = data["layers"]

    layer_spacing = 3.5

    for l_idx, layer in enumerate(layers):
        lx = l_idx * layer_spacing
        name = layer["name"]
        l_type = layer["type"]
        specs = layer["specs"]
        captured = layer.get("captured", {})
        channels = captured.get("channels", [])

        # Label for the layer
        spec_str = ", ".join(f"{k}={v}" for k, v in specs.items())
        label_text = f"<b>{name}</b><br>{l_type}<br><sup>{spec_str}</sup>"

        fig.add_trace(
            go.Scatter3d(
                x=[lx],
                y=[0],
                z=[2.2],
                mode="text",
                text=[label_text],
                textfont=dict(size=12, color=fg),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        if not channels:
            # Render empty block outline
            fig.add_trace(
                go.Scatter3d(
                    x=[lx, lx, lx, lx, lx],
                    y=[-1.0, 1.0, 1.0, -1.0, -1.0],
                    z=[-1.0, -1.0, 1.0, 1.0, -1.0],
                    mode="lines",
                    line=dict(color="#38bdf8", width=3, dash="dot"),
                    name=name,
                    hoverinfo="text",
                    text=[f"{name} ({l_type})<br>{spec_str}"],
                )
            )
            continue

        # Render channel planes
        num_ch = len(channels)
        ch_spacing = 0.55
        start_y = -((num_ch - 1) * ch_spacing) / 2.0

        for c_idx, ch in enumerate(channels):
            cy = start_y + c_idx * ch_spacing
            grid = ch["grid"]
            gh = ch["h"]
            gw = ch["w"]

            # Generate coordinates for spatial grid points
            xs, ys, zs, colors, texts = [], [], [], [], []
            half_w = (gw - 1) / 2.0 if gw > 1 else 1.0
            half_h = (gh - 1) / 2.0 if gh > 1 else 1.0

            for r in range(gh):
                for c in range(gw):
                    val = float(grid[r][c])
                    xs.append(lx + (c - half_w) * (0.6 / max(gw, 1)))
                    ys.append(cy)
                    zs.append((r - half_h) * (1.8 / max(gh, 1)))
                    colors.append(val)
                    texts.append(
                        f"<b>{name} (Channel {ch['channel_index']})</b><br>"
                        f"Coord: ({r}, {c})<br>"
                        f"Activation: {val:.4f}<br>"
                        f"Mean: {ch['mean_val']:.3f}<br>"
                        f"Max: {ch['max_val']:.3f}"
                    )

            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="markers",
                    marker=dict(
                        size=3.5,
                        color=colors,
                        colorscale="Teal",
                        showscale=False,
                        opacity=0.85,
                    ),
                    name=f"{name} Ch{ch['channel_index']}",
                    hoverinfo="text",
                    hovertext=texts,
                    showlegend=False,
                )
            )

            # Outline frame for channel plane
            z_top = (gh - 1 - half_h) * (1.8 / max(gh, 1)) + 0.1
            z_bot = (0 - half_h) * (1.8 / max(gh, 1)) - 0.1
            y_box = cy
            fig.add_trace(
                go.Scatter3d(
                    x=[lx, lx, lx, lx, lx],
                    y=[y_box - 0.2, y_box + 0.2, y_box + 0.2, y_box - 0.2, y_box - 0.2],
                    z=[z_bot, z_bot, z_top, z_top, z_bot],
                    mode="lines",
                    line=dict(color="#38bdf8", width=1.5),
                    hoverinfo="skip",
                    showlegend=False,
                    opacity=0.4,
                )
            )

    fig.update_layout(
        title=dict(
            text=f"<b>{view.title}</b><br><sup>3D CNN Feature Maps & Spatial Architecture</sup>",
            x=0.04,
            y=0.96,
            font=dict(size=22, color=fg),
        ),
        paper_bgcolor=bg,
        font=dict(family="Inter, system-ui, sans-serif", color=fg),
        height=720,
        margin=dict(l=30, r=30, t=100, b=80),
        scene=dict(
            xaxis=dict(visible=False, showbackground=False),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            bgcolor=bg,
            aspectmode="manual",
            aspectratio=dict(x=max(2.0, len(layers) * 0.8), y=1.5, z=1.2),
            camera=dict(
                eye=dict(x=-1.8, y=-2.2, z=1.2),
                up=dict(x=0, y=0, z=1),
            ),
            dragmode="orbit",
        ),
        uirevision="seedl-cnn",
    )

    return fig


def svg_cnn_3d(view: Any) -> str:
    """Render 2D isometric architectural projection of CNN layers."""
    data = view.data
    theme = data.get("theme", "scientific")
    bg = "#0f172a" if theme in {"dark", "scientific"} else "#f8fafc"
    fg = "#f8fafc" if theme in {"dark", "scientific"} else "#0f172a"
    muted = "#94a3b8" if theme in {"dark", "scientific"} else "#64748b"
    border = "#1e293b" if theme in {"dark", "scientific"} else "#e2e8f0"

    layers = data["layers"]
    width = max(780, len(layers) * 160 + 120)
    height = 480

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        f'<rect width="{width}" height="{height}" fill="{bg}"/>',
        f'<text x="35" y="45" font-family="system-ui, sans-serif" font-size="12" fill="{muted}" font-weight="600">SeeDL / CNN 3D ARCHITECTURE</text>',
        f'<text x="35" y="80" font-family="system-ui, sans-serif" font-size="24" fill="{fg}" font-weight="700">{html.escape(view.title)}</text>',
        f'<text x="35" y="105" font-family="system-ui, sans-serif" font-size="13" fill="{muted}">{len(layers)} Convolutional Layer(s) · Spatial Feature Planes</text>',
    ]

    for i, layer in enumerate(layers):
        lx = 80 + i * 150
        ly = 240
        w_box = 80
        h_box = 100
        svg_lines += [
            f'<rect x="{lx}" y="{ly - h_box / 2}" width="{w_box}" height="{h_box}" rx="6" fill="{border}" stroke="#38bdf8" stroke-width="2" opacity="0.6"><title>{html.escape(layer["name"])}</title></rect>',
            f'<text x="{lx + w_box / 2}" y="{ly - 10}" font-family="system-ui, sans-serif" font-size="13" font-weight="700" fill="{fg}" text-anchor="middle">{html.escape(layer["name"])}</text>',
            f'<text x="{lx + w_box / 2}" y="{ly + 12}" font-family="system-ui, sans-serif" font-size="11" fill="{muted}" text-anchor="middle">{layer["type"]}</text>',
        ]
        if i < len(layers) - 1:
            svg_lines.append(
                f'<line x1="{lx + w_box}" y1="{ly}" x2="{lx + 150}" y2="{ly}" stroke="{muted}" stroke-width="2" stroke-dasharray="4,4"/>'
            )

    svg_lines.append("</svg>")
    return "\n".join(svg_lines)
