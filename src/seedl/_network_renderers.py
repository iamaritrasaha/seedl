"""Shared visual language for neuron diagrams and honest architecture fallbacks."""

from __future__ import annotations

import html
from typing import Any

from .exceptions import MissingDependencyError

PALETTE = ["#168f9c", "#6378d7", "#9670c9", "#d18a39"]
POSITIVE, NEGATIVE = "#119b9a", "#e17463"


def _theme(data: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        ("#101827", "#e5edf7", "#a5b5ce", "#23314a")
        if data.get("theme") == "dark"
        else ("#fbfcff", "#20324e", "#65758d", "#e8edf5")
    )


def _fmt(value: Any) -> str:
    return "unavailable" if value is None else f"{value:.5g}"


def _color_field(graph: dict[str, Any]) -> str | None:
    mode = graph["color_by"]
    return (
        "activation"
        if mode == "activation" or (mode == "auto" and graph["captured"])
        else "gradient"
        if mode == "gradient"
        else None
    )


def _scale(graph: dict[str, Any]) -> float:
    field = _color_field(graph)
    return (
        max((abs(n[field]) for n in graph["nodes"] if field and n[field] is not None), default=1)
        or 1
    )


def _color(node: dict[str, Any], graph: dict[str, Any]) -> str:
    field = _color_field(graph)
    if field is None:
        return PALETTE[node["layer"] % len(PALETTE)]
    value = node[field]
    if value is None:
        return "#8390a6"
    ratio = min(1, abs(value) / _scale(graph))
    start, end = (225, 231, 240), (225, 116, 99) if value < 0 else (17, 155, 154)
    return (
        "rgb("
        + ",".join(str(round(a + ratio * (b - a))) for a, b in zip(start, end, strict=True))
        + ")"
    )


def _operation(layer: dict[str, Any]) -> str:
    activation = layer["operation"]
    return "z = W @ a + b" + (f" → {activation}" if activation != "Linear" else "")


def _node_text(node: dict[str, Any], graph: dict[str, Any]) -> str:
    layer = graph["layers"][node["layer"]]
    heading = f"{layer['label']} · neuron {node['index']}"
    if graph["captured"]:
        heading += f" · sample {graph['sample']}"
    if node["layer"] == 0:
        return (
            f"{heading}\nInput feature: {_fmt(node['activation'])} ({node['activation_status']})"
            "\nInput features have no bias or parameter gradient."
        )
    return (
        f"{heading}\nModule: {layer['module']}\n"
        f"Operation: {_operation(layer)}\n"
        f"Activation: {_fmt(node['activation'])} ({node['activation_status']})\n"
        f"Pre-activation: {_fmt(node['preactivation'])}\nBias: {_fmt(node['bias'])}\n"
        f"Weight trainable: {node['trainable']} · bias trainable: {node['bias_trainable']}\n"
        f"Bias gradient: {_fmt(node['gradient'])} ({node['gradient_status']})"
    )


def _edge_text(edge: dict[str, Any]) -> str:
    return (
        f"{edge['source']} → {edge['target']}\nWeight: {_fmt(edge['weight'])}\n"
        f"Parameter: {edge['parameter']} [destination, source]\nTrainable: {edge['trainable']}\n"
        f"Weight gradient: {_fmt(edge['gradient'])} ({edge['gradient_status']})"
    )


def _weight_bin(weight: float, scale: float) -> int:
    return min(3, int(abs(weight) / scale * 4))


def _footer(graph: dict[str, Any]) -> str:
    field = _color_field(graph)
    color = (
        "layer identity"
        if field is None
        else f"{'bias gradient' if field == 'gradient' else field} · −{_scale(graph):.3g} to +{_scale(graph):.3g} · gray = unavailable"
    )
    return f"Node color: {color}. Connection width: |weight|."


def _vector_text(nodes: list[dict[str, Any]], field: str) -> str:
    values = ", ".join(_fmt(n[field]) for n in nodes[:6])
    suffix = f", … ({len(nodes)} total; hover for each value)" if len(nodes) > 6 else ""
    return "[" + values + suffix + "]"


def _step_text(graph: dict[str, Any], index: int) -> str:
    if index < 0:
        return "<b>Captured forward pass</b> · select a layer below; colors keep the same scale."
    layer = graph["layers"][index]
    output = _vector_text(layer["nodes"], "activation")
    if index == 0:
        return f"<b>Input · sample {graph['sample']}</b><br>Features = {output}"
    inputs = _vector_text(graph["layers"][index - 1]["nodes"], "activation")
    pre = _vector_text(layer["nodes"], "preactivation")
    operation = html.escape(_operation(layer))
    return (
        f"<b>{layer['label']} · {operation}</b>"
        f"<br>Input a = {inputs}<br>Pre-activation z = {pre}<br>Output = {output}"
    )


def plotly_network(view: Any) -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise MissingDependencyError(
            "Install `pip install 'seedl[plotly]'` for interactive networks."
        ) from exc
    graph = view.data.get("neuron_graph")
    if graph is None:
        return _plotly_architecture(view, go)
    bg, fg, muted, border = _theme(graph)
    spatial = graph["dimensions"] == 3
    trace_type = go.Scatter3d if spatial else go.Scatter
    figure = go.Figure()
    lookup = {n["id"]: n for n in graph["nodes"]}
    scale = max((abs(e["weight"]) for e in graph["edges"]), default=1) or 1
    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for edge in graph["edges"]:
        groups.setdefault(
            (
                edge["layer"],
                (1 if edge["weight"] > 0 else -1 if edge["weight"] < 0 else 0),
                _weight_bin(edge["weight"], scale),
            ),
            [],
        ).append(edge)
    legends = set()
    for (layer, sign_value, bucket), edges in groups.items():
        x: list[Any] = []
        y: list[Any] = []
        z: list[Any] = []
        text: list[str] = []
        for edge in edges:
            a, b = lookup[edge["source"]], lookup[edge["target"]]
            coords_a = (a["x"], a["y3"] if spatial else a["y"], a["z3"])
            coords_b = (b["x"], b["y3"] if spatial else b["y"], b["z3"])
            for values, start, end in zip((x, y, z), coords_a, coords_b, strict=True):
                values.extend([start, (start + end) / 2, end, None])
            text.extend([html.escape(_edge_text(edge)).replace("\n", "<br>")] * 3 + [""])
        sign = (
            "Positive weights"
            if sign_value > 0
            else "Negative weights"
            if sign_value < 0
            else "Zero weights"
        )
        kwargs = dict(
            x=x,
            y=y,
            mode="lines",
            line=dict(
                color=POSITIVE if sign_value > 0 else NEGATIVE if sign_value < 0 else "#8390a6",
                width=(1.2 if spatial else 0.8) + bucket * 0.8,
            ),
            opacity=graph["edge_opacity"],
            text=text,
            hovertemplate="%{text}<extra></extra>",
            name=sign,
            legendgroup=sign,
            legendrank={1: 0, -1: 1, 0: 2}[sign_value],
            showlegend=sign not in legends,
            meta=dict(layer=layer, role="edges"),
        )
        if spatial:
            kwargs["z"] = z
        figure.add_trace(trace_type(**kwargs))
        legends.add(sign)
    field = _color_field(graph)
    colorbar_added = False
    for index, layer in enumerate(graph["layers"]):
        nodes = layer["nodes"]
        # Separate missing values so None is never silently mapped to numeric zero.
        for present in (True, False):
            selected = [n for n in nodes if (field is None or n[field] is not None) == present]
            if not selected:
                continue
            marker: dict[str, Any] = dict(
                size=11 if spatial else max(10, min(28, 320 / max(graph["widths"]))),
                color=[_color(n, graph) for n in selected],
                line=dict(color=PALETTE[index % len(PALETTE)] if field else bg, width=1.8),
            )
            if field and present:
                marker.update(
                    color=[n[field] for n in selected],
                    cmin=-_scale(graph),
                    cmax=_scale(graph),
                    colorscale=[[0, NEGATIVE], [0.5, "#e1e7f0"], [1, POSITIVE]],
                    showscale=not colorbar_added,
                    colorbar=dict(
                        title="Activation" if field == "activation" else "Bias gradient",
                        thickness=12,
                        len=0.55,
                        outlinewidth=0,
                    ),
                )
                colorbar_added = True
            kwargs = dict(
                x=[n["x"] for n in selected],
                y=[n["y3"] if spatial else n["y"] for n in selected],
                mode="markers+text" if not spatial and max(graph["widths"]) <= 12 else "markers",
                textposition="middle right",
                textfont=dict(size=10, color=muted),
                marker=marker,
                name=layer["label"],
                showlegend=False,
                text=[str(n["index"]) for n in selected],
                hovertext=[
                    html.escape(_node_text(n, graph)).replace("\n", "<br>") for n in selected
                ],
                hovertemplate="%{hovertext}<extra></extra>",
                meta=dict(layer=index, role="nodes"),
            )
            if spatial:
                kwargs["z"] = [n["z3"] for n in selected]
            figure.add_trace(trace_type(**kwargs))
    annotations = []
    for index, layer in enumerate(graph["layers"]):
        label = f"<b>{layer['label']}</b><br>{layer['width']} neurons<br>{html.escape(layer['operation'])}"
        if spatial:
            ymin = min(n["y3"] for n in layer["nodes"]) - 0.4
            ymax = max(n["y3"] for n in layer["nodes"]) + 0.4
            zmin = min(n["z3"] for n in layer["nodes"]) - 0.4
            zmax = max(n["z3"] for n in layer["nodes"]) + 0.4
            figure.add_trace(
                go.Scatter3d(
                    x=[index * 3.5] * 5,
                    y=[ymin, ymax, ymax, ymin, ymin],
                    z=[zmin, zmin, zmax, zmax, zmin],
                    mode="lines",
                    line=dict(color=PALETTE[index % len(PALETTE)], width=2, dash="dot"),
                    opacity=0.35,
                    hoverinfo="skip",
                    showlegend=False,
                    meta=dict(role="guide", layer=index),
                )
            )
            figure.add_trace(
                go.Scatter3d(
                    x=[index * 3.5],
                    y=[0],
                    z=[max(n["z3"] for n in graph["nodes"]) + 0.9],
                    mode="text",
                    text=[label],
                    textfont=dict(size=12, color=fg),
                    showlegend=False,
                    hoverinfo="skip",
                    meta=dict(role="label", layer=index),
                )
            )
        else:
            figure.add_shape(
                type="rect",
                x0=index * 3.5 - 0.35,
                x1=index * 3.5 + 0.35,
                y0=-layer["width"] / 2,
                y1=layer["width"] / 2,
                fillcolor=PALETTE[index % len(PALETTE)],
                opacity=0.07,
                line=dict(width=0),
                layer="below",
            )
            annotations.append(
                dict(
                    x=index * 3.5,
                    y=1.02,
                    yref="paper",
                    yanchor="bottom",
                    text=label,
                    showarrow=False,
                    font=dict(size=13, color=fg),
                )
            )
    annotations.append(
        dict(
            x=0,
            y=-0.23 if spatial and graph.get("explore") else -0.17,
            xref="paper",
            yref="paper",
            text=_footer(graph)
            + ("<br>Dotted outlines mark layer planes, not connections." if spatial else ""),
            showarrow=False,
            xanchor="left",
            font=dict(size=11, color=muted),
        )
    )
    subtitle = graph["caption"]
    if graph["sampled"]:
        subtitle += " · sampled edges"
    if view.data.get("visual_notes"):
        subtitle += "<br>" + "<br>".join(html.escape(n) for n in view.data["visual_notes"])
    figure.update_layout(
        title=dict(
            text=f"<b>{view.title}</b><br><sup>{subtitle}</sup>",
            x=0.04,
            y=0.97,
            yanchor="top",
            font=dict(size=23),
        ),
        template="plotly_dark" if graph["theme"] == "dark" else "plotly_white",
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(family="Inter, system-ui, sans-serif", color=fg),
        height=700,
        margin=dict(
            l=45,
            r=75 if field else 40,
            t=(110 if spatial else 175) + (50 if view.data.get("visual_notes") else 0),
            b=125,
        ),
        legend=dict(
            orientation="h",
            x=0,
            y=-0.04,
            yanchor="top",
            font=dict(size=12),
            groupclick="togglegroup",
            title=dict(text="Connections · click to filter", font=dict(size=11, color=muted)),
        ),
        annotations=annotations,
        hoverlabel=dict(bgcolor=bg, font=dict(color=fg)),
        uirevision="seedl-network",
    )
    if spatial:
        axis = dict(visible=False, showbackground=False)
        camera = dict(eye=dict(x=-1.6, y=-2.6, z=0.55), up=dict(x=0, y=0, z=1))
        figure.update_layout(
            scene=dict(
                xaxis=axis,
                yaxis=axis,
                zaxis=axis,
                bgcolor=bg,
                aspectmode="manual",
                aspectratio=dict(x=max(1.8, len(graph["layers"]) * 0.6), y=1.6, z=1.25),
                camera=camera,
                dragmode="orbit",
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0,
                    y=1.09,
                    xanchor="left",
                    showactive=False,
                    bgcolor=bg,
                    bordercolor=border,
                    buttons=[
                        dict(label="Orbit", method="relayout", args=[{"scene.dragmode": "orbit"}]),
                        dict(label="Pan", method="relayout", args=[{"scene.dragmode": "pan"}]),
                        dict(
                            label="Reset view",
                            method="relayout",
                            args=[{"scene.camera": camera, "scene.dragmode": "orbit"}],
                        ),
                    ],
                )
            ],
        )
    else:
        figure.update_xaxes(
            visible=False, range=[-1.2, (len(graph["layers"]) - 1) * 3.5 + 1.2], fixedrange=False
        )
        figure.update_yaxes(
            visible=False,
            range=[-max(graph["widths"]) / 2 - 0.7, max(graph["widths"]) / 2 + 0.7],
            fixedrange=False,
        )
    if graph.get("explore"):
        if not graph["captured"]:
            figure.add_annotation(
                x=0,
                y=-0.25,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="left",
                text="Values unavailable. Supply input=… to capture a forward pass.",
                font=dict(color=muted, size=12),
            )
            return figure
        steps = []

        def detail(index: int) -> dict[str, Any]:
            return dict(
                x=0,
                y=-0.30,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="top",
                text=_step_text(graph, index),
                showarrow=False,
                align="left",
                font=dict(size=12, color=fg),
                bgcolor=bg,
                bordercolor=border,
                borderpad=10,
            )

        for current_layer in range(-1, len(graph["layers"])):
            opacities = []
            for trace in figure.data:
                meta = trace.meta or {}
                role = meta.get("role")
                active = meta.get("layer") == current_layer or (
                    role == "nodes" and meta.get("layer") == current_layer - 1
                )
                normal = (
                    graph["edge_opacity"] if role == "edges" else 0.35 if role == "guide" else 1
                )
                opacities.append(
                    normal
                    if current_layer == -1 or role == "label"
                    else (0.8 if role in ("edges", "guide") else 1)
                    if active
                    else (0.04 if role == "edges" else 0.22)
                )
            steps.append(
                dict(
                    method="update",
                    label="Reset" if current_layer < 0 else graph["layers"][current_layer]["label"],
                    args=[
                        {"opacity": opacities},
                        {"annotations": annotations + [detail(current_layer)]},
                    ],
                )
            )
        figure.update_layout(
            annotations=annotations + [detail(-1)],
            sliders=[
                dict(
                    active=0,
                    steps=steps,
                    x=0,
                    y=1.58,
                    len=0.95,
                    currentvalue=dict(prefix=f"Sample {graph['sample']} · ", font=dict(color=fg)),
                    bgcolor=border,
                    activebgcolor=PALETTE[0],
                    bordercolor=muted,
                    font=dict(color=fg),
                    transition=dict(duration=0),
                )
            ],
            margin=dict(t=260 if view.data.get("visual_notes") else 210, b=230),
            height=790 if view.data.get("visual_notes") else 740,
        )
    return figure


def svg_network(view: Any) -> str:
    graph = view.data.get("neuron_graph")
    if graph is None:
        return _svg_architecture(view)
    bg, fg, muted, border = _theme(graph)
    width = max(920, len(graph["layers"]) * 180)
    height = max(620, max(graph["widths"]) * 21 + 260)
    left, top = 85, 170
    gap = (width - 2 * left) / max(1, len(graph["layers"]) - 1)
    spacing = (height - 300) / max(8, max(graph["widths"]) - 1)
    positions = {
        n["id"]: (left + n["layer"] * gap, top + (height - 300) / 2 - n["y"] * spacing)
        for n in graph["nodes"]
    }
    body = [
        _svg_open(width, height, bg, fg, view.title),
        _text(35, 45, "SeeDL / NEURAL NETWORK", muted, 12),
        _text(35, 80, view.title, fg, 25),
        _text(35, 106, graph["caption"], muted, 13),
    ]
    scale = max((abs(e["weight"]) for e in graph["edges"]), default=1) or 1
    for edge in graph["edges"]:
        x1, y1 = positions[edge["source"]]
        x2, y2 = positions[edge["target"]]
        body.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{POSITIVE if edge["weight"] > 0 else NEGATIVE if edge["weight"] < 0 else "#8390a6"}" stroke-width="{0.6 + _weight_bin(edge["weight"], scale) * 0.65}" stroke-opacity="{graph["edge_opacity"]}"><title>{html.escape(_edge_text(edge))}</title></line>'
        )
    for node in graph["nodes"]:
        x, y = positions[node["id"]]
        radius = max(4, min(13, spacing * 0.36))
        body.append(f'<circle cx="{x}" cy="{y}" r="{radius + 3}" fill="{bg}"/>')
        body.append(
            f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{_color(node, graph)}" stroke="{fg}" stroke-width="0.5"><title>{html.escape(_node_text(node, graph))}</title></circle>'
        )
        if max(graph["widths"]) <= 12:
            body.append(_text(x + radius + 5, y + 3, str(node["index"]), muted, 10))
    for i, layer in enumerate(graph["layers"]):
        x = left + i * gap
        body += [
            _text(x, 140, f"{layer['label']} · {layer['width']}", fg, 14, anchor="middle"),
            _text(x, height - 100, layer["operation"], muted, 11, anchor="middle"),
        ]
    body += [
        _text(
            35,
            height - 60,
            "Positive weights (teal) · Negative weights (coral) · " + _footer(graph),
            muted,
            11,
        )
    ]
    note = " · ".join(view.data.get("visual_notes", []))
    if graph["dimensions"] == 3:
        note = "2D projection of the 3D network. Export HTML for rotation. " + note
    body.append(_text(35, height - 30, note, muted, 10))
    return "".join(body) + "</svg>"


def _svg_open(width: int, height: int, bg: str, fg: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{html.escape(title, quote=True)}" width="100%" viewBox="0 0 {width} {height}">'
        f'<title>{html.escape(title)}</title><rect width="100%" height="100%" rx="18" fill="{bg}"/>'
        "<style>text{font-family:Inter,system-ui,sans-serif}</style>"
    )


def _text(x: float, y: float, text: str, color: str, size: int, anchor: str = "start") -> str:
    return f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" text-anchor="{anchor}">{html.escape(text)}</text>'


def _block_positions(arch: dict[str, Any]) -> list[tuple[float, float]]:
    # Vertical flow leaves room for operation specs and tensor dimensions.
    return [
        (40 + (0 if arch["sequential"] else min(b["depth"], 6) * 32), 155 + i * 100)
        for i, b in enumerate(arch["blocks"])
    ]


def _svg_architecture(view: Any) -> str:
    arch = view.data["architecture"]
    bg, fg, muted, border = _theme(view.data)
    positions = _block_positions(arch)
    height = 210 + len(positions) * 100
    body = [
        _svg_open(1000, height, bg, fg, view.title),
        _text(40, 44, "SeeDL / ARCHITECTURE", muted, 12),
        _text(40, 80, view.title, fg, 25),
        _text(40, 110, arch["caption"], muted, 11),
    ]
    for a, b in arch["edges"]:
        ax, ay = positions[a]
        bx, by = positions[b]
        body.append(
            f'<path d="M {ax + 12} {ay + 75} V {by - 10} H {bx + 12} V {by}" fill="none" stroke="{POSITIVE}" stroke-width="2"/>'
        )
    for (x, y), block in zip(positions, arch["blocks"], strict=True):
        body.append(
            f'<rect x="{x}" y="{y}" width="{920 - x}" height="76" rx="12" fill="{border}"/>'
        )
        body += [
            _text(x + 18, y + 24, f"{block['name']} / {block['type']}", fg, 15),
            _text(x + 18, y + 45, block["specs"][:125], muted, 11),
            _text(
                x + 18,
                y + 63,
                f"Output: {block['shape']} · Parameters: {block['parameters']}",
                muted,
                11,
            ),
        ]
    for i, note in enumerate(view.data.get("visual_notes", [])[:2]):
        body.append(_text(40, height - 35 + 15 * i, note, muted, 11))
    return "".join(body) + "</svg>"


def _plotly_architecture(view: Any, go: Any) -> Any:
    arch = view.data["architecture"]
    bg, fg, muted, border = _theme(view.data)
    figure = go.Figure()
    positions = _block_positions(arch)
    for a, b in arch["edges"]:
        ax, ay = positions[a]
        bx, by = positions[b]
        figure.add_trace(
            go.Scatter(
                x=[ax + 12, ax + 12, bx + 12, bx + 12],
                y=[-ay - 75, -by + 10, -by + 10, -by],
                mode="lines",
                line=dict(color=POSITIVE, width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    for (x, y), block in zip(positions, arch["blocks"], strict=True):
        figure.add_shape(
            type="rect", x0=x, x1=930, y0=-y, y1=-y - 76, fillcolor=border, line=dict(width=0)
        )
        label = f"<b>{html.escape(block['name'])} / {block['type']}</b><br>{html.escape(block['specs'])}<br>Output: {html.escape(str(block['shape']))} · Parameters: {block['parameters']}"
        figure.add_annotation(
            x=x + 18,
            y=-y - 38,
            text=label,
            showarrow=False,
            xanchor="left",
            align="left",
            font=dict(size=12, color=fg),
        )
    figure.update_layout(
        title=dict(
            text=f"<b>{view.title}</b><br><sup>{html.escape(arch['caption'])}</sup>", x=0.04
        ),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(color=fg),
        height=max(420, len(positions) * 100 + 180),
        margin=dict(l=25, r=25, t=100, b=70),
    )
    for i, note in enumerate(view.data.get("visual_notes", [])[:2]):
        figure.add_annotation(
            x=0,
            y=-0.04 - i * 0.03,
            xref="paper",
            yref="paper",
            text=html.escape(note),
            showarrow=False,
            xanchor="left",
            font=dict(color=muted, size=11),
        )
    figure.update_xaxes(visible=False, range=[0, 980])
    figure.update_yaxes(visible=False, range=[-len(positions) * 100 - 140, -125])
    return figure


def matplotlib_network(view: Any) -> Any:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    graph = view.data.get("neuron_graph")
    bg, fg, muted, border = _theme(view.data)
    if graph is None:
        arch = view.data["architecture"]
        fig, ax = plt.subplots(figsize=(12, max(4, len(arch["blocks"]) * 1.1 + 2)))
        positions = _block_positions(arch)
        for a, b in arch["edges"]:
            ax.plot(
                [positions[a][0] + 10, positions[b][0] + 10],
                [-positions[a][1], -positions[b][1]],
                color=POSITIVE,
            )
        for (x, y), block in zip(positions, arch["blocks"], strict=True):
            ax.text(
                x,
                -y,
                f"{block['name']} / {block['type']}\n{block['specs']}\nOutput: {block['shape']}",
                fontsize=10,
                color=fg,
                bbox=dict(boxstyle="round,pad=0.8", fc=border, ec="none"),
            )
        ax.set(xlim=(0, 1000), ylim=(-len(positions) * 100 - 100, -60))
        fig.suptitle(view.title + "\n" + arch["caption"], color=fg, fontsize=12)
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        lookup = {n["id"]: n for n in graph["nodes"]}
        scale = max((abs(e["weight"]) for e in graph["edges"]), default=1) or 1
        segments = [
            [
                (lookup[e["source"]]["x"], lookup[e["source"]]["y"]),
                (lookup[e["target"]]["x"], lookup[e["target"]]["y"]),
            ]
            for e in graph["edges"]
        ]
        ax.add_collection(
            LineCollection(
                segments,
                colors=[
                    POSITIVE if e["weight"] > 0 else NEGATIVE if e["weight"] < 0 else "#8390a6"
                    for e in graph["edges"]
                ],
                linewidths=[0.6 + _weight_bin(e["weight"], scale) * 0.65 for e in graph["edges"]],
                alpha=graph["edge_opacity"],
            )
        )
        # Matplotlib accepts normalized RGB tuples, not CSS rgb() strings.
        colors = [_color(n, graph) for n in graph["nodes"]]
        converted = [
            tuple(int(v) / 255 for v in c[4:-1].split(",")) if c.startswith("rgb(") else c
            for c in colors
        ]
        ax.scatter(
            [n["x"] for n in graph["nodes"]],
            [n["y"] for n in graph["nodes"]],
            c=converted,
            s=160,
            edgecolors=bg,
            zorder=3,
        )
        for i, layer in enumerate(graph["layers"]):
            ax.text(
                i * 3.5,
                max(graph["widths"]) / 2 + 1,
                f"{layer['label']} · {layer['width']}\n{layer['operation']}",
                ha="center",
                color=fg,
                fontsize=10,
            )
        ax.set(
            xlim=(-1.3, (len(graph["layers"]) - 1) * 3.5 + 1.3),
            ylim=(-max(graph["widths"]) / 2 - 1, max(graph["widths"]) / 2 + 3),
        )
        fig.suptitle(view.title + "\n" + graph["caption"], color=fg)
        foot = _footer(graph) + (
            "  2D projection; HTML supports 3D rotation." if graph["dimensions"] == 3 else ""
        )
        fig.text(0.05, 0.025, foot, color=muted, fontsize=9)
    fig.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axis("off")
    return fig
