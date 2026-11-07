"""Classical ML panels in SeeDL's SVG/Plotly/Matplotlib rendering system."""

from __future__ import annotations

import html
import math
from typing import Any

from ._presentation import FONT, color_at, contrast_text, document_html, format_value
from .view import Visualization

COLORS = ("#31688e", "#1f9e89", "#855000", "#783b85", "#9f2424", "#52657d")


def _s(value: Any) -> str:
    return html.escape(format_value(value), quote=True)


def _short(value: Any, limit: int = 34) -> str:
    text = format_value(value)
    return _s(text if len(text) <= limit else text[: limit - 1] + "…")


def _number(value: Any) -> str:
    return _s(format_value(value))


def _table_html(panel: dict[str, Any]) -> str:
    heads = "".join(f'<th scope="col">{_s(c)}</th>' for c in panel["columns"])
    rows = "".join(
        "<tr>" + "".join(f"<td>{_s(v)}</td>" for v in row) + "</tr>" for row in panel["rows"]
    )
    return f'<div class="ml-table-wrap" tabindex="0" role="region" aria-label="{_s(panel["title"])}"><table><thead><tr>{heads}</tr></thead><tbody>{rows}</tbody></table></div>'


def _scatter_svg(panel: dict[str, Any], width: int = 920) -> str:
    from .renderers import _normalized

    series = panel.get("series", [])
    points = [
        (
            s,
            [
                (x, y)
                for x, y in zip(s["x"], s["y"], strict=True)
                if x is not None
                and y is not None
                and math.isfinite(float(x))
                and math.isfinite(float(y))
            ],
        )
        for s in series
    ]
    xs = [float(x) for _, pairs in points for x, _ in pairs]
    ys = [float(y) for _, pairs in points for _, y in pairs]
    if not xs:
        return '<text x="70" y="100" class="row">No finite values to plot.</text>'
    lo_x, hi_x, lo_y, hi_y = min(xs), max(xs), min(ys), max(ys)
    if panel.get("reference") == "zero":
        lo_y, hi_y = min(lo_y, 0), max(hi_y, 0)
    if panel.get("reference") == "identity":
        lo_x = lo_y = min(lo_x, lo_y)
        hi_x = hi_y = max(hi_x, hi_y)
    left, top, w, h = 100, 42, 720, 190

    def coord(x: float, y: float) -> tuple[float, float]:
        return left + _normalized(x, lo_x, hi_x) * w, top + (1 - _normalized(y, lo_y, hi_y)) * h

    parts = []
    for i in range(5):
        ratio = i / 4
        x = left + w * ratio
        y = top + h * (1 - ratio)
        xv = lo_x * (1 - ratio) + hi_x * ratio
        yv = lo_y * (1 - ratio) + hi_y * ratio
        parts.extend(
            [
                f'<line x1="{left}" y1="{y}" x2="{left + w}" y2="{y}" class="grid"/>',
                f'<text x="{x}" y="{top + h + 22}" text-anchor="middle" class="label">{_number(xv)}</text>',
                f'<text x="{left - 12}" y="{y + 4}" text-anchor="end" class="label">{_number(yv)}</text>',
            ]
        )
    if panel.get("reference") in {"identity", "zero"}:
        a, b = (
            (coord(lo_x, lo_y), coord(hi_x, hi_y))
            if panel["reference"] == "identity"
            else (coord(lo_x, 0), coord(hi_x, 0))
        )
        parts.append(
            f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="#52657d" stroke-dasharray="5 4"><title>{"Perfect prediction" if panel["reference"] == "identity" else "Zero residual"}</title></line>'
        )
    for j, (s, pairs) in enumerate(points):
        color = COLORS[j % len(COLORS)]
        if panel["type"] == "line":
            if panel.get("line_shape") == "hv" and pairs:
                stepped = [pairs[0]]
                for previous, current in zip(pairs[:-1], pairs[1:], strict=True):
                    stepped.extend([(current[0], previous[1]), current])
                pairs = stepped
            xy = " ".join(f"{coord(x, y)[0]:.3f},{coord(x, y)[1]:.3f}" for x, y in pairs)
            parts.append(
                f'<polyline points="{xy}" fill="none" stroke="{color}" stroke-width="2"><title>{_s(s["name"])}</title></polyline>'
            )
        else:
            for x, y in pairs:
                cx, cy = coord(x, y)
                parts.append(
                    f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="3" fill="{color}" fill-opacity="0.7"><title>{_s(s["name"])}: x={_s(x)}, y={_s(y)}</title></circle>'
                )
        if j < 8:
            parts.append(
                f'<text x="{left + (j % 4) * 190}" y="{298 + (j // 4) * 20}" class="label" style="fill:{color}">{_short(s["name"], 25)}</text>'
            )
    parts.extend(
        [
            f'<text x="{left + w / 2}" y="{top + h + 47}" text-anchor="middle" class="label">{_s(panel.get("x_label", ""))}</text>',
            f'<text x="20" y="{top + h / 2}" text-anchor="middle" class="label" transform="rotate(-90 20 {top + h / 2})">{_s(panel.get("y_label", ""))}</text>',
        ]
    )
    return "".join(parts)


def _pipeline_table(panel: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "table",
        "title": panel["title"],
        "columns": ["Stage", "Estimator / transformer", "Role", "Input columns"],
        "rows": [
            ["  " * n["depth"] + n["name"], n["estimator"], n["role"], n["columns"]]
            for n in panel["nodes"]
        ],
        "description": panel.get("description", ""),
    }


def _body(panel: dict[str, Any]) -> tuple[str, int]:
    from .renderers import _normalized

    kind = panel["type"]
    if kind == "pipeline":
        pieces = []
        nodes = {n["id"]: n for n in panel["nodes"]}
        for edge in panel["edges"]:
            a, b = nodes[edge["source"]], nodes[edge["target"]]
            ax, ay = 50 + a["depth"] * 65, 35 + a["id"] * 60
            bx, by = 50 + b["depth"] * 65, 35 + b["id"] * 60
            pieces.append(
                f'<path d="M {ax} {ay + 20} L {ax} {by} L {bx} {by}" stroke="#52657d" fill="none" stroke-dasharray="{"4 3" if edge["relation"] == "parallel column branch" else "none"}"><title>{_s(edge["relation"])}</title></path>'
            )
        for node in panel["nodes"]:
            x, y = 40 + node["depth"] * 65, 15 + node["id"] * 60
            label = node["name"] + " | " + str(node["estimator"])
            node_width = max(200, 800 - node["depth"] * 65)
            pieces.append(
                f'<rect x="{x}" y="{y}" width="{node_width}" height="40" rx="4" fill="#f8fafc" stroke="#dce4ed"/><text x="{x + 12}" y="{y + 16}" class="row"><title>{_s(label)}</title>{_short(label, 75)}</text><text x="{x + 12}" y="{y + 32}" class="label">{_short(node["role"], 40)} | columns: {_short(node["columns"], 40)}</text>'
            )
        return "".join(pieces), 60 * len(nodes) + 35
    if kind in {"scatter", "line"}:
        return _scatter_svg(panel), 340
    if kind == "table":
        columns, rows = panel["columns"], panel["rows"]
        col_width = 840 / max(1, len(columns))
        table_svg = "".join(
            f'<text x="{40 + i * col_width}" y="34" class="head">{_short(c, max(8, int(col_width / 7)))}</text>'
            for i, c in enumerate(columns)
        )
        for r, row in enumerate(rows[:20]):
            for c, value in enumerate(row):
                table_svg += f'<text x="{40 + c * col_width}" y="{62 + r * 24}" class="row"><title>{_s(value)}</title>{_short(value, max(6, int(col_width / 7)))}</text>'
        height = 100 + min(len(rows), 20) * 24
        if len(rows) > 20:
            table_svg += f'<text x="40" y="{height - 8}" class="label">Showing 20 of {len(rows)} rows. Full captured records remain in view.data and HTML tables.</text>'
        return table_svg, height
    if kind == "matrix":
        rows = panel["values"]
        count = max(len(rows), 1)
        cell = min(54, 560 / count)
        max_value = max((v for row in rows for v in row), default=1) or 1
        body = []
        for i, row in enumerate(rows):
            if count <= 20:
                body.append(
                    f'<text x="140" y="{54 + (i + 0.5) * cell}" text-anchor="end" class="label">{_short(panel["y_labels"][i], 18)}</text>'
                )
            for j, v in enumerate(row):
                color = color_at(float(v) / max_value)
                body.append(
                    f'<rect x="{155 + j * cell}" y="{42 + i * cell}" width="{cell}" height="{cell}" fill="{color}" stroke="white"><title>True {_s(panel["y_labels"][i])}, predicted {_s(panel["x_labels"][j])}: {_s(v)}</title></rect>'
                )
                if count <= 10:
                    body.append(
                        f'<text x="{155 + (j + 0.5) * cell}" y="{46 + (i + 0.5) * cell}" text-anchor="middle" class="row" style="fill:{contrast_text(color)}">{_number(v)}</text>'
                    )
        if count <= 20:
            for j, name in enumerate(panel["x_labels"]):
                body.append(
                    f'<text x="{155 + (j + 0.5) * cell}" y="{60 + count * cell}" text-anchor="middle" class="label">{_short(name, 10)}</text>'
                )
        body.append(
            f'<text x="155" y="{85 + count * cell}" class="label">Predicted class (columns); true class (rows). Color: purple 0 → yellow {_s(max_value)}.</text>'
        )
        return "".join(body), int(count * cell + 120)
    if kind in {"bar", "histogram"}:
        if kind == "histogram":
            labels = [
                f"{format_value(a)} to {format_value(b)}"
                for a, b in zip(panel["edges"][:-1], panel["edges"][1:], strict=True)
            ]
            values = panel["counts"]
        else:
            labels, values = panel["labels"], panel["values"]
        limit = min(len(values), 50)
        finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
        for value, error in zip(values, panel.get("errors", [None] * len(values)), strict=True):
            if value is not None and error is not None:
                finite.extend([float(value) - float(error), float(value) + float(error)])
        lo, hi = min([0.0, *finite]), max([0.0, *finite])
        zero = 290 + _normalized(0, lo, hi) * 480
        body = []
        for i, (name, v) in enumerate(zip(labels[:limit], values[:limit], strict=True)):
            y = 20 + i * 24
            body.append(
                f'<text x="275" y="{y + 14}" text-anchor="end" class="row"><title>{_s(name)}</title>{_short(name, 32)}</text>'
            )
            if v is not None:
                end = 290 + _normalized(float(v), lo, hi) * 480
                body.append(
                    f'<rect x="{min(zero, end)}" y="{y}" width="{abs(end - zero):.3f}" height="18" fill="{COLORS[0] if float(v) >= 0 else COLORS[4]}"><title>{_s(name)}: {_s(v)}</title></rect><text x="790" y="{y + 14}" class="row">{_number(v)}</text>'
                )
        for i, error in enumerate(panel.get("errors", [])[:limit]):
            value = values[i]
            if value is not None and error is not None:
                a = 290 + _normalized(float(value) - float(error), lo, hi) * 480
                b = 290 + _normalized(float(value) + float(error), lo, hi) * 480
                y = 29 + i * 24
                body.append(
                    f'<path d="M {a} {y} H {b} M {a} {y - 5} V {y + 5} M {b} {y - 5} V {y + 5}" stroke="#172b4d" fill="none"><title>Repeat/fold standard deviation: {_s(error)} (not a confidence interval)</title></path>'
                )
        body.append(
            f'<text x="290" y="{limit * 24 + 48}" class="label">{_s(panel.get("y_label", "Value"))} · top/displayed {limit} of {len(values)}</text>'
        )
        return "".join(body), limit * 24 + 70
    return '<text x="40" y="40" class="row">Panel data available in view.data.</text>', 100


def svg_ml(view: Visualization, *, chart_only: bool = False) -> str:
    from .renderers import _details, _svg_header

    parts, y = [], 86
    for panel in view.data.get("panels", {}).values():
        body, h = _body(panel)
        parts.append(
            f'<g transform="translate(0,{y})"><text x="32" y="-8" class="title" style="font-size:18px">{_s(panel["title"])}</text>{body}</g>'
        )
        y += h + 45
    if not chart_only:
        for line in _details(view):
            parts.append(f'<text x="32" y="{y}" class="note">{_s(line)}</text>')
            y += 18
    import json

    captured = html.escape(json.dumps(view.data, ensure_ascii=False, allow_nan=False))
    return (
        _svg_header(view, y + 20, chart_only=chart_only)
        + "".join(parts)
        + f'<metadata id="seedl-ml-data">{captured}</metadata></svg>'
    )


def structured_data_html(view: Visualization, markup: str) -> str:
    """Preserve captured values and provenance at full precision in offline HTML."""
    import json

    if 'id="seedl-ml-data"' in markup:
        return markup
    encoded = html.escape(json.dumps(view.data, ensure_ascii=False, allow_nan=False, indent=2))
    details = f'<details id="seedl-ml-data"><summary>Captured ML data and provenance (JSON, full precision)</summary><pre>{encoded}</pre></details>'
    for end in ("</main>", "</body>"):
        if end in markup:
            return markup.replace(end, details + end, 1)
    return markup + details


def html_ml(view: Visualization) -> str:
    """Semantic plot cards inside the shared offline document shell."""
    cards = []
    plotly_loaded = False
    for panel in view.data.get("panels", {}).values():
        if panel["type"] == "pipeline":
            content, height = _body(panel)
            body = (
                f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Pipeline diagram" viewBox="0 0 920 {height}" width="100%"><style>text{{font-family:{FONT};fill:#172b4d}}.row{{font-size:13px}}.label{{font-size:12px}}</style>{content}</svg>'
                + _table_html(_pipeline_table(panel))
            )
        elif panel["type"] == "table":
            body = _table_html(panel)
        elif panel["type"] == "matrix":
            table = {
                "title": panel["title"],
                "columns": ["True / Predicted", *panel["x_labels"]],
                "rows": [
                    [name, *values]
                    for name, values in zip(panel["y_labels"], panel["values"], strict=True)
                ],
            }
            largest = max((float(v) for row in panel["values"] for v in row), default=0) or 1.0
            headers = "".join(f'<th scope="col">{_s(v)}</th>' for v in table["columns"])
            rows = []
            for label, row in zip(panel["y_labels"], panel["values"], strict=True):
                cells = "".join(
                    f'<td style="background:{color_at(float(v) / largest)};color:{contrast_text(color_at(float(v) / largest))}">{_s(v)}</td>'
                    for v in row
                )
                rows.append(f'<tr><th scope="row">{_s(label)}</th>{cells}</tr>')
            body = f'<p>Color scale: 0 to {_s(largest)} (Viridis); all values are printed.</p><div class="ml-table-wrap" tabindex="0" role="region" aria-label="Confusion matrix values"><table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
        elif view.backend == "plotly":
            panel_view = Visualization(
                "ml_panel",
                panel["title"],
                panel.get("description", ""),
                {"panels": {"panel": panel}},
                backend="plotly",
            )
            figure = render_ml(panel_view, "plotly")
            figure.update_layout(
                title=None, height=360, margin={"l": 65, "r": 30, "t": 35, "b": 60}
            )
            body = figure.to_html(
                full_html=False,
                include_plotlyjs=not plotly_loaded,
                config={"responsive": True, "displaylogo": False},
            )
            plotly_loaded = True
        else:
            content, height = _body(panel)
            body = f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{_s(panel["title"])}" viewBox="0 0 920 {height}" width="100%"><title>{_s(panel["title"])}</title><style>text{{font-family:{FONT};fill:#172b4d}}.row{{font-size:13px;font-family:JetBrains Mono,monospace}}.label{{font-size:12px;fill:#52657d}}.head{{font-size:12px;font-weight:600}}.grid{{stroke:#dce4ed;stroke-width:1}}</style>{content}</svg>'
            if panel["type"] in {"scatter", "line"}:
                points = [
                    [s["name"], x, y]
                    for s in panel.get("series", [])
                    for x, y in list(zip(s["x"], s["y"], strict=True))[:20]
                ]
                table = {
                    "title": "Captured plot sample",
                    "columns": ["Series", panel.get("x_label", "x"), panel.get("y_label", "y")],
                    "rows": points,
                }
                body += f"<details><summary>Accessible plot values (first 20 captured points per series)</summary>{_table_html(table)}</details>"
        if "<svg " in body:
            # Keep labels legible on small screens without widening the document.
            end = body.index("</svg>") + len("</svg>")
            body = (
                '<p class="ml-scroll-hint">Chart: scroll horizontally for labels and values.</p>'
                f'<div class="ml-chart" tabindex="0" role="region" aria-label="{_s(panel["title"])} chart">'
                + body[:end]
                + "</div>"
                + body[end:]
            )
        cards.append(
            f'<section class="ml-panel"><h2>{_s(panel["title"])}</h2><p>{_s(panel.get("description", ""))}</p>{body}</section>'
        )
    metric_cards = "".join(
        f"<div><dt>{_s(name.replace('_', ' '))}</dt><dd>{_s(value)}</dd></div>"
        for name, value in view.data.get("metrics", {}).items()
    )
    metrics = (
        f'<dl class="ml-metrics" aria-label="Evaluation metrics">{metric_cards}</dl>'
        if metric_cards
        else ""
    )
    style = "<style>.ml-metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,150px),1fr));gap:12px;margin:16px 0}.ml-metrics div{background:#f8fafc;border:1px solid #dce4ed;border-radius:6px;padding:12px}.ml-metrics dt{font-size:12px;color:#52657d;text-transform:uppercase}.ml-metrics dd{font:600 20px JetBrains Mono,monospace;margin:6px 0 0;overflow-wrap:anywhere}.ml-panel{margin:24px 0;border-top:1px solid #dce4ed;padding-top:16px}.ml-panel h2{font-size:18px}.ml-table-wrap{overflow:auto;max-width:100%}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;text-align:left;border-bottom:1px solid #dce4ed;white-space:nowrap}td{font-family:JetBrains Mono,monospace}th{color:#52657d}.ml-panel svg{max-width:100%;height:auto}</style>"
    style += "<style>.ml-chart{overflow:auto;max-width:100%}.ml-scroll-hint{display:none}@media(max-width:600px){.ml-chart svg{width:920px;max-width:none}.ml-scroll-hint{display:block;font-size:12px;color:#52657d}}</style>"
    return structured_data_html(view, document_html(view, style + metrics + "".join(cards)))


def render_ml(view: Visualization, backend: str, *, chart_only: bool = False) -> Any:
    if backend == "svg":
        return svg_ml(view, chart_only=chart_only)
    panels = [
        _pipeline_table(p) if p["type"] == "pipeline" else p
        for p in view.data.get("panels", {}).values()
    ]
    if backend == "plotly":
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=max(1, len(panels)),
            cols=1,
            specs=[[{"type": "table" if p["type"] == "table" else "xy"}] for p in panels] or [[{}]],
            subplot_titles=[html.escape(p["title"]) for p in panels],
            vertical_spacing=min(0.08, 0.25 / max(1, len(panels))),
        )
        for row, p in enumerate(panels, 1):
            kind = p["type"]
            if kind == "table":
                fig.add_trace(
                    go.Table(
                        header={"values": [_s(c) for c in p["columns"]], "font": {"family": FONT}},
                        cells={
                            "font": {"family": "JetBrains Mono, monospace"},
                            "values": [
                                [_s(r[i]) for r in p["rows"]] for i in range(len(p["columns"]))
                            ],
                        },
                    ),
                    row=row,
                    col=1,
                )
            elif kind == "matrix":
                fig.add_trace(
                    go.Heatmap(
                        z=p["values"],
                        x=p["x_labels"],
                        y=p["y_labels"],
                        colorscale="Viridis",
                        zmin=0,
                        showscale=True,
                        colorbar={
                            "len": 0.6 / max(1, len(panels)),
                            "y": 1 - (row - 0.5) / max(1, len(panels)),
                        },
                        hovertemplate="True %{y}<br>Predicted %{x}<br>Value %{z}<extra></extra>",
                    ),
                    row=row,
                    col=1,
                )
                fig.update_yaxes(autorange="reversed", row=row, col=1)
            elif kind in {"line", "scatter"}:
                for i, s in enumerate(p.get("series", [])):
                    fig.add_trace(
                        go.Scatter(
                            x=s["x"],
                            y=s["y"],
                            name=html.escape(s["name"]),
                            mode="lines" if kind == "line" else "markers",
                            marker={"color": COLORS[i % len(COLORS)], "size": 5},
                            line={
                                "color": COLORS[i % len(COLORS)],
                                "shape": p.get("line_shape", "linear"),
                            },
                            showlegend=len(p.get("series", [])) > 1,
                            hovertemplate="x %{x:.6g}<br>y %{y:.6g}<extra>%{fullData.name}</extra>",
                        ),
                        row=row,
                        col=1,
                    )
                if p.get("reference") == "zero":
                    fig.add_hline(y=0, line_dash="dash", row=row, col=1)
                elif p.get("reference") == "identity":
                    values = [
                        v for s in p.get("series", []) for v in s["x"] + s["y"] if v is not None
                    ]
                    if values:
                        fig.add_trace(
                            go.Scatter(
                                x=[min(values), max(values)],
                                y=[min(values), max(values)],
                                mode="lines",
                                line={"color": "#52657d", "dash": "dash"},
                                name="Perfect prediction",
                                showlegend=False,
                            ),
                            row=row,
                            col=1,
                        )
            elif kind == "bar":
                fig.add_trace(
                    go.Bar(
                        x=p["labels"],
                        y=p["values"],
                        name=html.escape(p["title"]),
                        marker_color=COLORS[0],
                        error_y={
                            "type": "data",
                            "array": p.get("errors", []),
                            "visible": bool(p.get("errors")),
                        },
                        showlegend=False,
                    ),
                    row=row,
                    col=1,
                )
            elif kind == "histogram":
                edges = p["edges"]
                fig.add_trace(
                    go.Bar(
                        x=[a / 2 + b / 2 for a, b in zip(edges[:-1], edges[1:], strict=True)],
                        y=p["counts"],
                        width=[b - a for a, b in zip(edges[:-1], edges[1:], strict=True)],
                        marker_color=COLORS[0],
                        showlegend=False,
                    ),
                    row=row,
                    col=1,
                )
            if kind != "table":
                fig.update_xaxes(
                    title_text=html.escape(p.get("x_label", "")),
                    tickfont_family="JetBrains Mono, monospace",
                    row=row,
                    col=1,
                )
                fig.update_yaxes(
                    title_text=html.escape(p.get("y_label", "")),
                    tickfont_family="JetBrains Mono, monospace",
                    row=row,
                    col=1,
                )
        metric_text = " | ".join(
            f"{k.replace('_', ' ')}: {format_value(v)}"
            for k, v in list(view.data.get("metrics", {}).items())[:3]
        )
        fig.update_layout(
            title_text=html.escape(view.title)
            + (f"<br><sup>{html.escape(metric_text)}</sup>" if metric_text else ""),
            template="plotly_white",
            font={"family": FONT, "size": 13, "color": "#172b4d"},
            height=max(400, 360 * len(panels)),
            autosize=True,
            margin={"l": 65, "r": 30, "t": 90, "b": 55},
        )
        return fig
    if backend == "matplotlib":
        import matplotlib.pyplot as plt

        from .renderers import _mpl_table

        fig, axes = plt.subplots(
            max(1, len(panels)), 1, figsize=(9, 4 * max(1, len(panels))), squeeze=False
        )
        for axis, p in zip(axes[:, 0], panels, strict=True):
            axis.set_title(p["title"])
            if p["type"] == "table":
                _mpl_table(axis, p["rows"], p["columns"])
            elif p["type"] == "matrix":
                artist = axis.imshow(p["values"], cmap="viridis", vmin=0)
                axis.set_xticks(range(len(p["x_labels"])), p["x_labels"], rotation=45, ha="right")
                axis.set_yticks(range(len(p["y_labels"])), p["y_labels"])
                fig.colorbar(artist, ax=axis, label=p.get("description", "Value"))
            elif p["type"] in {"line", "scatter"}:
                for s in p.get("series", []):
                    if p["type"] == "line":
                        if p.get("line_shape") == "hv":
                            axis.step(s["x"], s["y"], where="post", label=s["name"])
                        else:
                            axis.plot(s["x"], s["y"], label=s["name"])
                    else:
                        axis.scatter(s["x"], s["y"], s=10, alpha=0.7, label=s["name"])
                if p.get("reference") == "zero":
                    axis.axhline(0, color="#52657d", linestyle="--")
                elif p.get("reference") == "identity":
                    limits = axis.get_xlim()
                    axis.plot(limits, limits, color="#52657d", linestyle="--")
                if len(p.get("series", [])) > 1:
                    axis.legend()
            elif p["type"] == "bar":
                axis.bar(p["labels"], p["values"], yerr=p.get("errors"), color=COLORS[0])
                axis.tick_params(axis="x", rotation=45)
            elif p["type"] == "histogram":
                edges = p["edges"]
                axis.bar(
                    [a / 2 + b / 2 for a, b in zip(edges[:-1], edges[1:], strict=True)],
                    p["counts"],
                    width=[b - a for a, b in zip(edges[:-1], edges[1:], strict=True)],
                    color=COLORS[0],
                )
            axis.set_xlabel(p.get("x_label", ""))
            axis.set_ylabel(p.get("y_label", ""))
        metric_text = " | ".join(
            f"{k.replace('_', ' ')}: {format_value(v)}"
            for k, v in list(view.data.get("metrics", {}).items())[:3]
        )
        fig.suptitle(view.title + ("\n" + metric_text if metric_text else ""))
        fig.tight_layout()
        return fig
    raise ValueError(f"Unknown backend {backend!r}.")
