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


def _neuron_detail_card(node: dict[str, Any] | None, graph: dict[str, Any]) -> str:
    if node is None:
        return _footer(graph) + (
            "<br>Dotted outlines mark layer planes, not connections."
            if graph.get("dimensions") == 3
            else ""
        )
    layer = graph["layers"][node["layer"]]
    heading = (
        f"<b>NEURON INSPECTION: {layer['label']} · Neuron #{node['index']}</b> (id: {node['id']})"
    )
    if graph.get("captured"):
        heading += f" · Sample {graph['sample']}"

    lines = [heading]
    mod_name = layer["module"]
    op_str = html.escape(_operation(layer))
    lines.append(f"<b>Module:</b> {mod_name} · <b>Operation:</b> {op_str}")

    details = []
    if node["layer"] == 0:
        act_val = _fmt(node.get("activation"))
        details.append(
            f"<b>Input feature:</b> {act_val} ({node.get('activation_status', 'not captured')})"
        )
        details.append("No bias or parameter gradients (input features)")
    else:
        act_val = _fmt(node.get("activation"))
        pre_val = _fmt(node.get("preactivation"))
        bias_val = _fmt(node.get("bias"))
        bias_tr = (
            "trainable"
            if node.get("bias_trainable")
            else "frozen"
            if node.get("bias_trainable") is False
            else "none"
        )
        grad_val = _fmt(node.get("gradient"))
        grad_st = node.get("gradient_status", "missing")

        details.append(
            f"<b>Activation:</b> {act_val} ({node.get('activation_status', 'not captured')})"
        )
        details.append(f"<b>Pre-activation:</b> {pre_val}")
        details.append(f"<b>Bias:</b> {bias_val} ({bias_tr})")
        details.append(f"<b>Bias grad:</b> {grad_val} ({grad_st})")
    lines.append(" · ".join(details))

    # Incoming connections
    in_edges = node.get("incoming_edges", [])
    if in_edges:
        in_strs = []
        for e in in_edges[:6]:
            w_str = f"w={e['weight']:+.3g}"
            if e.get("gradient") is not None:
                w_str += f", ∇={e['gradient']:+.2g}"
            in_strs.append(f"[{e['source']}] {w_str}")
        in_desc = f"<b>Incoming ({len(in_edges)}):</b> " + ", ".join(in_strs)
        if len(in_edges) > 6:
            in_desc += f", … (+{len(in_edges) - 6} more; hover for each)"
    else:
        in_desc = "<b>Incoming:</b> None (input features)"
    lines.append(in_desc)

    # Outgoing connections
    out_edges = node.get("outgoing_edges", [])
    if out_edges:
        out_strs = []
        for e in out_edges[:6]:
            w_str = f"w={e['weight']:+.3g}"
            if e.get("gradient") is not None:
                w_str += f", ∇={e['gradient']:+.2g}"
            out_strs.append(f"[{e['target']}] {w_str}")
        out_desc = f"<b>Outgoing ({len(out_edges)}):</b> " + ", ".join(out_strs)
        if len(out_edges) > 6:
            out_desc += f", … (+{len(out_edges) - 6} more; hover for each)"
    else:
        out_desc = "<b>Outgoing:</b> None (output layer)"
    lines.append(out_desc)

    return "<br>".join(lines)


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

    # Highlight traces for interactive neuron selection
    def _edge_line_coords(
        edge_list: list[dict[str, Any]],
    ) -> tuple[list[Any], list[Any], list[Any], list[str]]:
        x_coords: list[Any] = []
        y_coords: list[Any] = []
        z_coords: list[Any] = []
        text_list: list[str] = []
        for edge in edge_list:
            a, b = lookup[edge["source"]], lookup[edge["target"]]
            coords_a = (a["x"], a["y3"] if spatial else a["y"], a["z3"])
            coords_b = (b["x"], b["y3"] if spatial else b["y"], b["z3"])
            for vals, start, end in zip(
                (x_coords, y_coords, z_coords), coords_a, coords_b, strict=True
            ):
                vals.extend([start, (start + end) / 2, end, None])
            text_list.extend([html.escape(_edge_text(edge)).replace("\n", "<br>")] * 3 + [""])
        return x_coords, y_coords, z_coords, text_list

    selected_id = graph.get("selected_neuron")
    selected_node = lookup.get(selected_id) if selected_id else None
    has_selection = selected_node is not None

    if selected_node is not None:
        init_in_x, init_in_y, init_in_z, init_in_text = _edge_line_coords(
            selected_node.get("incoming_edges", [])
        )
        init_out_x, init_out_y, init_out_z, init_out_text = _edge_line_coords(
            selected_node.get("outgoing_edges", [])
        )
        init_node_x = [selected_node["x"]]
        init_node_y = [selected_node["y3"] if spatial else selected_node["y"]]
        init_node_z = [selected_node["z3"]]
        init_node_text = [html.escape(_node_text(selected_node, graph)).replace("\n", "<br>")]
    else:
        init_in_x, init_in_y, init_in_z, init_in_text = [], [], [], []
        init_out_x, init_out_y, init_out_z, init_out_text = [], [], [], []
        init_node_x, init_node_y, init_node_z, init_node_text = [], [], [], []

    highlight_incoming_kw: dict[str, Any] = dict(
        x=init_in_x,
        y=init_in_y,
        mode="lines",
        line=dict(color="#00e5ff", width=3.5),
        opacity=0.95,
        name="Incoming to selected",
        hoverinfo="text",
        text=init_in_text,
        showlegend=False,
        visible=bool(init_in_x) if has_selection else False,
        meta=dict(role="highlight_incoming"),
    )
    if spatial:
        highlight_incoming_kw["z"] = init_in_z
    figure.add_trace(trace_type(**highlight_incoming_kw))

    highlight_outgoing_kw: dict[str, Any] = dict(
        x=init_out_x,
        y=init_out_y,
        mode="lines",
        line=dict(color="#fbbf24", width=3.5),
        opacity=0.95,
        name="Outgoing from selected",
        hoverinfo="text",
        text=init_out_text,
        showlegend=False,
        visible=bool(init_out_x) if has_selection else False,
        meta=dict(role="highlight_outgoing"),
    )
    if spatial:
        highlight_outgoing_kw["z"] = init_out_z
    figure.add_trace(trace_type(**highlight_outgoing_kw))

    highlight_node_kw: dict[str, Any] = dict(
        x=init_node_x,
        y=init_node_y,
        mode="markers",
        marker=dict(
            size=16 if spatial else 22,
            color="#fbbf24",
            line=dict(color="#ffffff", width=2.5),
        ),
        opacity=1.0,
        name="Selected neuron",
        hoverinfo="text",
        text=init_node_text,
        showlegend=False,
        visible=has_selection,
        meta=dict(role="highlight_node"),
    )
    if spatial:
        highlight_node_kw["z"] = init_node_z
    figure.add_trace(trace_type(**highlight_node_kw))

    if has_selection:
        for t in figure.data:
            role = t.meta.get("role") if t.meta else None
            if role == "edges":
                t.opacity = 0.04
            elif role == "nodes":
                t.opacity = 0.25

    footer_annot = dict(
        x=0,
        y=-0.23 if spatial and graph.get("explore") else -0.17,
        xref="paper",
        yref="paper",
        text=_neuron_detail_card(selected_node, graph),
        showarrow=False,
        xanchor="left",
        align="left",
        font=dict(size=11, color=fg if has_selection else muted),
        bgcolor=bg if has_selection else None,
        bordercolor=border if has_selection else None,
        borderpad=8 if has_selection else 0,
    )
    annotations.append(footer_annot)

    # Build Neuron Selection dropdown menu
    idx_incoming = len(figure.data) - 3
    idx_outgoing = len(figure.data) - 2
    idx_selected_node = len(figure.data) - 1

    overview_opacities = []
    for trace in figure.data:
        meta = trace.meta or {}
        role = meta.get("role")
        if role in ("highlight_incoming", "highlight_outgoing", "highlight_node"):
            overview_opacities.append(0.0)
        elif role == "edges":
            overview_opacities.append(graph["edge_opacity"])
        elif role == "guide":
            overview_opacities.append(0.35)
        else:
            overview_opacities.append(1.0)

    overview_x = [t.x for t in figure.data]
    overview_x[idx_incoming] = []
    overview_x[idx_outgoing] = []
    overview_x[idx_selected_node] = []

    overview_y = [t.y for t in figure.data]
    overview_y[idx_incoming] = []
    overview_y[idx_outgoing] = []
    overview_y[idx_selected_node] = []

    overview_text = [getattr(t, "text", []) for t in figure.data]
    overview_text[idx_incoming] = []
    overview_text[idx_outgoing] = []
    overview_text[idx_selected_node] = []

    overview_visible = [
        t.meta.get("role") not in ("highlight_incoming", "highlight_outgoing", "highlight_node")
        for t in figure.data
    ]

    overview_annots = annotations[:-1] + [
        dict(
            x=0,
            y=-0.23 if spatial and graph.get("explore") else -0.17,
            xref="paper",
            yref="paper",
            text=_neuron_detail_card(None, graph),
            showarrow=False,
            xanchor="left",
            font=dict(size=11, color=muted),
        )
    ]

    overview_args_data: dict[str, Any] = dict(
        opacity=overview_opacities,
        x=overview_x,
        y=overview_y,
        text=overview_text,
        visible=overview_visible,
    )
    if spatial:
        overview_z = [getattr(t, "z", []) for t in figure.data]
        overview_z[idx_incoming] = []
        overview_z[idx_outgoing] = []
        overview_z[idx_selected_node] = []
        overview_args_data["z"] = overview_z

    neuron_buttons = [
        dict(
            label="Overview (All Neurons)",
            method="update",
            args=[overview_args_data, dict(annotations=overview_annots)],
        )
    ]

    for n in graph["nodes"]:
        n_in_x, n_in_y, n_in_z, n_in_text = _edge_line_coords(n.get("incoming_edges", []))
        n_out_x, n_out_y, n_out_z, n_out_text = _edge_line_coords(n.get("outgoing_edges", []))
        n_opacities = []
        for trace in figure.data:
            meta = trace.meta or {}
            role = meta.get("role")
            if role == "edges":
                n_opacities.append(0.04)
            elif role == "nodes":
                n_opacities.append(0.25)
            elif role == "guide":
                n_opacities.append(0.1)
            elif role == "label":
                n_opacities.append(1.0)
            elif role in ("highlight_incoming", "highlight_outgoing", "highlight_node"):
                n_opacities.append(1.0)
            else:
                n_opacities.append(1.0)

        n_x = list(overview_x)
        n_x[idx_incoming] = n_in_x
        n_x[idx_outgoing] = n_out_x
        n_x[idx_selected_node] = [n["x"]]

        n_y = list(overview_y)
        n_y[idx_incoming] = n_in_y
        n_y[idx_outgoing] = n_out_y
        n_y[idx_selected_node] = [n["y3"] if spatial else n["y"]]

        n_text = list(overview_text)
        n_text[idx_incoming] = n_in_text
        n_text[idx_outgoing] = n_out_text
        n_text[idx_selected_node] = [html.escape(_node_text(n, graph)).replace("\n", "<br>")]

        n_vis = list(overview_visible)
        n_vis[idx_incoming] = bool(n_in_x)
        n_vis[idx_outgoing] = bool(n_out_x)
        n_vis[idx_selected_node] = True

        n_card_annot = annotations[:-1] + [
            dict(
                x=0,
                y=-0.23 if spatial and graph.get("explore") else -0.17,
                xref="paper",
                yref="paper",
                text=_neuron_detail_card(n, graph),
                showarrow=False,
                xanchor="left",
                align="left",
                font=dict(size=11, color=fg),
                bgcolor=bg,
                bordercolor=border,
                borderpad=8,
            )
        ]
        n_args_data: dict[str, Any] = dict(
            opacity=n_opacities,
            x=n_x,
            y=n_y,
            text=n_text,
            visible=n_vis,
        )
        if spatial:
            n_z_coords = list(overview_z)
            n_z_coords[idx_incoming] = n_in_z
            n_z_coords[idx_outgoing] = n_out_z
            n_z_coords[idx_selected_node] = [n["z3"]]
            n_args_data["z"] = n_z_coords

        neuron_buttons.append(
            dict(
                label=f"Neuron [{n['id']}] · {graph['layers'][n['layer']]['label']} #{n['index']}",
                method="update",
                args=[n_args_data, dict(annotations=n_card_annot)],
            )
        )

    active_idx = 0
    if selected_id:
        node_ids = [n["id"] for n in graph["nodes"]]
        if selected_id in node_ids:
            active_idx = 1 + node_ids.index(selected_id)

    neuron_menu = dict(
        type="dropdown",
        direction="down",
        active=active_idx,
        buttons=neuron_buttons,
        x=0.28 if spatial else 0.0,
        y=1.09,
        xanchor="left",
        yanchor="top",
        bgcolor=bg,
        bordercolor=border,
        font=dict(size=11, color=fg),
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
        height=720,
        margin=dict(
            l=45,
            r=75 if field else 40,
            t=(110 if spatial else 175) + (50 if view.data.get("visual_notes") else 0),
            b=135,
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

    camera = dict(eye=dict(x=-1.8, y=-2.4, z=0.85), up=dict(x=0, y=0, z=1))
    if spatial:
        axis = dict(visible=False, showbackground=False)
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
                ),
                neuron_menu,
            ],
        )
    else:
        figure.update_layout(updatemenus=[neuron_menu])
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

        frames = []
        for current_layer in range(-1, len(graph["layers"])):
            opacities = []
            for trace in figure.data:
                meta = trace.meta or {}
                role = meta.get("role")
                if role in ("highlight_incoming", "highlight_outgoing", "highlight_node"):
                    opacities.append(0.0)
                    continue
                active = meta.get("layer") == current_layer or (
                    role == "nodes" and meta.get("layer") == current_layer - 1
                )
                normal = (
                    graph["edge_opacity"] if role == "edges" else 0.35 if role == "guide" else 1.0
                )
                opacities.append(
                    normal
                    if current_layer == -1 or role == "label"
                    else (0.8 if role in ("edges", "guide") else 1.0)
                    if active
                    else (0.04 if role == "edges" else 0.22)
                )

            frame_name = "reset" if current_layer < 0 else f"step_{current_layer}"
            frames.append(
                go.Frame(
                    data=[dict(opacity=op) for op in opacities],
                    name=frame_name,
                    layout=dict(annotations=annotations[:-1] + [detail(current_layer)]),
                )
            )

            steps.append(
                dict(
                    method="update",
                    label="Reset" if current_layer < 0 else graph["layers"][current_layer]["label"],
                    args=[
                        {"opacity": opacities},
                        {"annotations": annotations[:-1] + [detail(current_layer)]},
                    ],
                )
            )

        figure.frames = frames

        anim_menu = dict(
            type="buttons",
            direction="right",
            x=0.56 if spatial else 0.34,
            y=1.09,
            xanchor="left",
            showactive=False,
            bgcolor=bg,
            bordercolor=border,
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[
                        [f"step_{i}" for i in range(len(graph["layers"]))],
                        dict(
                            frame=dict(duration=1400, redraw=True),
                            transition=dict(duration=350),
                            fromcurrent=True,
                            mode="immediate",
                        ),
                    ],
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[
                        [None],
                        dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate",
                            transition=dict(duration=0),
                        ),
                    ],
                ),
                dict(
                    label="⏮ Reset",
                    method="animate",
                    args=[
                        ["reset"],
                        dict(
                            frame=dict(duration=0, redraw=True),
                            mode="immediate",
                            transition=dict(duration=0),
                        ),
                    ],
                ),
            ],
        )

        speed_menu = dict(
            type="buttons",
            direction="right",
            x=0.86 if spatial else 0.68,
            y=1.09,
            xanchor="left",
            showactive=False,
            bgcolor=bg,
            bordercolor=border,
            buttons=[
                dict(
                    label="1x",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=1400, redraw=True),
                            transition=dict(duration=350),
                            mode="afterall",
                        ),
                    ],
                ),
                dict(
                    label="2x",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=700, redraw=True),
                            transition=dict(duration=175),
                            mode="afterall",
                        ),
                    ],
                ),
                dict(
                    label="0.5x",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=2800, redraw=True),
                            transition=dict(duration=700),
                            mode="afterall",
                        ),
                    ],
                ),
            ],
        )

        current_menus = list(figure.layout.updatemenus or [])
        current_menus.extend([anim_menu, speed_menu])

        figure.update_layout(
            updatemenus=current_menus,
            annotations=annotations[:-1] + [detail(-1)],
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
            margin=dict(t=260 if view.data.get("visual_notes") else 210, b=240),
            height=820 if view.data.get("visual_notes") else 760,
        )
    return figure


def svg_network(view: Any) -> str:
    graph = view.data.get("neuron_graph")
    if graph is None:
        return _svg_architecture(view)
    bg, fg, muted, border = _theme(graph)
    selected_id = graph.get("selected_neuron")
    selected_node = next((n for n in graph["nodes"] if n["id"] == selected_id), None)
    has_selection = selected_node is not None

    extra_h = 30 if has_selection else 0
    width = max(920, len(graph["layers"]) * 180)
    height = max(620 + extra_h, max(graph["widths"]) * 21 + 260 + extra_h)
    left, top = 85, 170
    gap = (width - 2 * left) / max(1, len(graph["layers"]) - 1)
    spacing = (height - (320 if has_selection else 300)) / max(8, max(graph["widths"]) - 1)
    positions = {
        n["id"]: (
            left + n["layer"] * gap,
            top + (height - (320 if has_selection else 300)) / 2 - n["y"] * spacing,
        )
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
        if has_selection:
            if edge["target"] == selected_id:
                body.append(
                    f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#00e5ff" stroke-width="2.5" stroke-opacity="0.95"><title>{html.escape(_edge_text(edge))}</title></line>'
                )
            elif edge["source"] == selected_id:
                body.append(
                    f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#fbbf24" stroke-width="2.5" stroke-opacity="0.95"><title>{html.escape(_edge_text(edge))}</title></line>'
                )
            else:
                color = (
                    POSITIVE
                    if edge["weight"] > 0
                    else NEGATIVE
                    if edge["weight"] < 0
                    else "#8390a6"
                )
                body.append(
                    f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{0.5 + _weight_bin(edge["weight"], scale) * 0.4}" stroke-opacity="0.06"><title>{html.escape(_edge_text(edge))}</title></line>'
                )
        else:
            body.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{POSITIVE if edge["weight"] > 0 else NEGATIVE if edge["weight"] < 0 else "#8390a6"}" stroke-width="{0.6 + _weight_bin(edge["weight"], scale) * 0.65}" stroke-opacity="{graph["edge_opacity"]}"><title>{html.escape(_edge_text(edge))}</title></line>'
            )
    for node in graph["nodes"]:
        x, y = positions[node["id"]]
        radius = max(4, min(13, spacing * 0.36))
        is_this_selected = has_selection and node["id"] == selected_id
        if is_this_selected:
            body.append(
                f'<circle cx="{x}" cy="{y}" r="{radius + 5}" fill="none" stroke="#fbbf24" stroke-width="2.5"/>'
            )
            body.append(f'<circle cx="{x}" cy="{y}" r="{radius + 2}" fill="{bg}"/>')
            body.append(
                f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{_color(node, graph)}" stroke="#ffffff" stroke-width="1.5"><title>{html.escape(_node_text(node, graph))}</title></circle>'
            )
        else:
            node_opacity = "0.35" if has_selection else "1.0"
            body.append(f'<circle cx="{x}" cy="{y}" r="{radius + 3}" fill="{bg}"/>')
            body.append(
                f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{_color(node, graph)}" fill-opacity="{node_opacity}" stroke="{fg}" stroke-opacity="{node_opacity}" stroke-width="0.5"><title>{html.escape(_node_text(node, graph))}</title></circle>'
            )
        if max(graph["widths"]) <= 12:
            lbl_color = fg if is_this_selected else muted
            body.append(_text(x + radius + 5, y + 3, str(node["index"]), lbl_color, 10))
    for i, layer in enumerate(graph["layers"]):
        x = left + i * gap
        body += [
            _text(x, 140, f"{layer['label']} · {layer['width']}", fg, 14, anchor="middle"),
            _text(
                x,
                height - (120 if has_selection else 100),
                layer["operation"],
                muted,
                11,
                anchor="middle",
            ),
        ]
    if has_selection and selected_node is not None:
        sel_layer = graph["layers"][selected_node["layer"]]
        in_count = len(selected_node.get("incoming_edges", []))
        out_count = len(selected_node.get("outgoing_edges", []))
        act_str = (
            f"Act: {_fmt(selected_node.get('activation'))}"
            if graph.get("captured")
            else "Act: not captured"
        )
        bias_str = (
            f"Bias: {_fmt(selected_node.get('bias'))}"
            if selected_node["layer"] > 0
            else "Input feature"
        )
        sel_summary = (
            f"Selected Neuron [{selected_node['id']}] · {sel_layer['label']} (#{selected_node['index']}) · "
            f"{act_str} · {bias_str} · Incoming (cyan): {in_count} · Outgoing (amber): {out_count}"
        )
        body += [
            _text(35, height - 76, sel_summary, fg, 12),
            _text(
                35,
                height - 56,
                "Highlighted connections: cyan = incoming, amber = outgoing · Dimmed = unrelated · "
                + _footer(graph),
                muted,
                11,
            ),
        ]
    else:
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
    body.append(_text(35, height - 25 if has_selection else height - 30, note, muted, 10))
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
