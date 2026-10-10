"""Lazy renderers for SeeDL's compact, backend-neutral view data."""

from __future__ import annotations

import html
import importlib.util
import math
from collections.abc import Iterable
from typing import Any

from .exceptions import MissingDependencyError, SeeDLError
from .view import Visualization


def resolve_backend(backend: str) -> str:
    """Resolve ``auto`` to Plotly when installed, otherwise to built-in SVG."""
    if backend == "auto":
        return "plotly" if importlib.util.find_spec("plotly") else "svg"
    if backend not in {"plotly", "matplotlib", "svg"}:
        raise ValueError("backend must be 'auto', 'plotly', 'matplotlib', or 'svg'.")
    if backend == "plotly" and not importlib.util.find_spec("plotly"):
        raise MissingDependencyError(
            "Plotly rendering requires the optional dependency. "
            "Install it with `pip install 'seedl[plotly]'`."
        )
    if backend == "matplotlib" and not importlib.util.find_spec("matplotlib"):
        raise MissingDependencyError(
            "Matplotlib rendering requires the optional dependency. "
            "Install it with `pip install 'seedl[matplotlib]'`."
        )
    return backend


def render(view: Visualization, backend: str) -> Any:
    """Build a backend figure from a ``Visualization`` without changing its data."""
    if backend == "svg":
        return _svg(view)
    if backend == "plotly":
        return _plotly(view)
    if backend == "matplotlib":
        return _matplotlib(view)
    raise ValueError(f"Unknown renderer {backend!r}.")


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _finite(values: Iterable[Any]) -> list[float]:
    result = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def _svg_header(view: Visualization, height: int) -> str:
    title = _esc(view.title)
    summary = _esc(view.summary)
    notes = "".join(
        f'<text x="32" y="{height - 18 - i * 18}" class="note">{_esc(note)}</text>'
        for i, note in enumerate(view.notes[:3])
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{title}" '
        f'viewBox="0 0 920 {height}" width="100%" height="{height}">'
        "<style>text{font-family:Inter,ui-sans-serif,system-ui,sans-serif;fill:#172b4d}"
        ".title{font-size:23px;font-weight:650}.sub{font-size:13px;fill:#52657d}"
        ".note{font-size:12px;fill:#52657d}.axis{stroke:#b7c4d3;stroke-width:1}"
        ".grid{stroke:#e5ebf1;stroke-width:1}.label{font-size:11px;fill:#52657d}"
        ".value{font-size:44px;font-weight:650}.cell{stroke:white;stroke-width:.6}"
        ".row{font-size:12px}.head{font-size:11px;font-weight:650;fill:#52657d}"
        ".track{fill:#e6edf5}.bar{fill:#4b78c2}</style>"
        f'<text x="32" y="38" class="title">{title}</text>'
        f'<text x="32" y="61" class="sub">{summary}</text>'
        f"{notes}"
    )


def _svg_line_chart(
    traces: list[dict[str, Any]],
    *,
    top: int,
    height: int,
    title: str = "",
    x_label: str = "Index",
    y_label: str = "Value",
    width: int = 840,
    left: int = 54,
) -> str:
    bottom = top + height
    right = left + width
    data = [(t, _finite(t.get("values", []))) for t in traces]
    values = [value for _, series in data for value in series]
    if not values:
        return f'<text x="{left}" y="{top + 30}" class="label">No finite values to plot</text>'
    low, high = min(values), max(values)
    if low == high:
        padding = abs(low) * 0.1 or 1.0
        low -= padding
        high += padding
    x_values = []
    for trace, _series in data:
        series_values = trace.get("values", [])
        series_x = trace.get("x")
        for i, value in enumerate(series_values):
            try:
                x_value = float(series_x[i]) if series_x is not None else float(i + 1)
                y_value = float(value)
            except (IndexError, TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(x_value) and math.isfinite(y_value):
                x_values.append(x_value)
    x_low, x_high = (min(x_values), max(x_values)) if x_values else (0.0, 1.0)
    if x_low == x_high:
        x_high = x_low + 1.0
    if title:
        title_svg = f'<text x="{left}" y="{top - 10}" class="head">{_esc(title)}</text>'
    else:
        title_svg = ""
    grid = "".join(
        f'<line x1="{left}" y1="{top + i * height / 4}" x2="{right}" '
        f'y2="{top + i * height / 4}" class="grid"/>'
        for i in range(5)
    )
    paths: list[str] = []
    legend: list[str] = []
    colors = ["#315fbd", "#ef8734", "#24a079", "#a353a6", "#db5264", "#478e9c"]
    for index, (trace, series) in enumerate(data):
        if not series:
            continue
        color = trace.get("color", colors[index % len(colors)])
        original = trace.get("values", [])
        x_values_for_trace = trace.get("x")
        segments: list[list[tuple[float, float]]] = []
        current: list[tuple[float, float]] = []
        for i, value in enumerate(original):
            try:
                number = float(value)
                x_value = (
                    float(x_values_for_trace[i]) if x_values_for_trace is not None else float(i + 1)
                )
            except (IndexError, TypeError, ValueError, OverflowError):
                if current:
                    segments.append(current)
                    current = []
                continue
            if math.isfinite(number) and math.isfinite(x_value):
                x = left + ((x_value - x_low) / (x_high - x_low)) * width
                y = bottom - ((number - low) / (high - low)) * height
                current.append((x, y))
            elif current:
                segments.append(current)
                current = []
        if current:
            segments.append(current)
        if segments:
            for segment in segments:
                points = " ".join(f"{x:.1f},{y:.1f}" for x, y in segment)
                paths.append(
                    f'<polyline points="{points}" fill="none" stroke="{color}" '
                    'stroke-width="2.4" stroke-linejoin="round"/>'
                )
                paths.extend(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.3" fill="{color}"/>'
                    for x, y in segment
                )
            x = left + index * 145
            legend.append(
                f'<line x1="{x}" y1="{bottom + 25}" x2="{x + 18}" y2="{bottom + 25}" '
                f'stroke="{color}" stroke-width="3"/><text x="{x + 24}" y="{bottom + 29}" '
                f'class="label">{_esc(trace.get("name", "series"))}</text>'
            )
    high_label, low_label = _esc(f"{high:.4g}"), _esc(f"{low:.4g}")
    return (
        f'{title_svg}{grid}<line x1="{left}" y1="{bottom}" x2="{right}" '
        f'y2="{bottom}" class="axis"/><line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{bottom}" class="axis"/>'
        f'<text x="{left - 8}" y="{top + 3}" text-anchor="end" class="label">{high_label}</text>'
        f'<text x="{left - 8}" y="{bottom + 4}" text-anchor="end" class="label">{low_label}</text>'
        f'<text x="{left + width / 2}" y="{bottom + 46}" text-anchor="middle" '
        f'class="label">{_esc(x_label)}</text>'
        f'<text x="18" y="{top + height / 2}" text-anchor="middle" class="label" '
        f'transform="rotate(-90 18 {top + height / 2})">{_esc(y_label)}</text>'
        f"{''.join(paths)}{''.join(legend)}"
    )


def _svg(view: Visualization) -> str:
    kind, data = view.kind, view.data
    if kind == "scalar":
        value = _esc(data.get("display_value", data.get("value", "")))
        footer = _esc(data.get("caption", ""))
        body = (
            f'<text x="48" y="150" class="value">{value}</text>'
            f'<text x="50" y="180" class="sub">{footer}</text>'
        )
        height = 240
    elif kind == "sequence":
        body = _svg_line_chart(
            [{"name": data.get("label", "value"), "values": data["values"]}],
            top=98,
            height=145,
            x_label="Index",
            y_label=data.get("label", "Value"),
        )
        height = 330
    elif kind == "matrix":
        rows = data.get("values", [])
        row_count, col_count = len(rows), max((len(row) for row in rows), default=0)
        max_dim = max(row_count, col_count, 1)
        cell = min(20.0, 700.0 / max_dim)
        x0, y0 = 110, 98
        values = _finite(value for row in rows for value in row)
        low, high = (min(values), max(values)) if values else (0.0, 1.0)
        rects = []
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                try:
                    number = float(value)
                except (ValueError, TypeError, OverflowError):
                    continue
                if not math.isfinite(number):
                    color = "#dce4ed"
                else:
                    ratio = 0.5 if high == low else (number - low) / (high - low)
                    red = round(49 + 190 * ratio)
                    green = round(95 + 115 * (1 - abs(ratio - 0.5) * 2))
                    blue = round(189 - 120 * ratio)
                    color = f"rgb({red},{green},{blue})"
                rects.append(
                    f'<rect class="cell" x="{x0 + j * cell:.1f}" y="{y0 + i * cell:.1f}" '
                    f'width="{cell:.1f}" height="{cell:.1f}" fill="{color}"><title>'
                    f"row {i}, column {j}: {_esc(value)}</title></rect>"
                )
        body = (
            f'<text x="{x0}" y="87" class="label">columns</text>'
            f'<text x="42" y="{y0 + 14}" class="label">rows</text>{"".join(rects)}'
        )
        height = int(y0 + row_count * cell + 55)
    elif kind == "image":
        pixels = data.get("pixels", [])
        channels = data.get("channels", 1)
        h, w = len(pixels), max((len(row) for row in pixels), default=0)
        size = max(min(520 / max(w, 1), 360 / max(h, 1)), 1)
        pieces = []
        for y, row in enumerate(pixels):
            for x, pixel in enumerate(row):
                components = pixel if isinstance(pixel, (list, tuple)) else [pixel]
                if channels == 1 or len(components) == 1:
                    shade = round(float(components[0]) * 255)
                    color = f"rgb({shade}, {shade}, {shade})"
                else:
                    rgb = [
                        max(0, min(255, round(float(component) * 255)))
                        for component in components[:3]
                    ]
                    color = f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
                pieces.append(
                    f'<rect x="{200 + x * size:.1f}" y="{90 + y * size:.1f}" '
                    f'width="{size + 0.2:.1f}" height="{size + 0.2:.1f}" fill="{color}"/>'
                )
        body = (
            "".join(pieces) or '<text x="40" y="110" class="label">Image preview unavailable</text>'
        )
        height = int(110 + h * size + 60)
    elif kind == "distribution":
        distribution_values = _finite(data.get("sample", []))
        count = min(30, max(8, int(math.sqrt(max(len(distribution_values), 1)))))
        low, high = (
            (min(distribution_values), max(distribution_values)) if distribution_values else (0, 1)
        )
        bins = [0] * count
        for sample_value in distribution_values:
            index = (
                0
                if low == high
                else min(count - 1, int((sample_value - low) / (high - low) * count))
            )
            bins[index] += 1
        maximum = max(bins, default=1) or 1
        bars = []
        for i, bin_count in enumerate(bins):
            bar_h = 150 * bin_count / maximum
            bars.append(
                f'<rect x="{70 + i * 25}" y="{250 - bar_h:.1f}" width="21" '
                f'height="{bar_h:.1f}" rx="2" fill="#4b78c2"/>'
            )
        body = (
            "".join(bars)
            + '<line x1="70" y1="250" x2="825" y2="250" class="axis"/>'
            + '<text x="447" y="278" text-anchor="middle" class="label">Value</text>'
            + '<text x="30" y="165" text-anchor="middle" class="label" '
            'transform="rotate(-90 30 165)">Count</text>'
        )
        height = 310
    elif kind == "training":
        groups = data.get("groups", [])
        body_parts = []
        top = 105
        per_group = 175
        for group in groups:
            traces = group.get("series", [])
            body_parts.append(
                _svg_line_chart(
                    traces,
                    top=top + 18,
                    height=118,
                    title=group.get("name", "metric"),
                    x_label="Epoch",
                    y_label=group.get("name", "metric"),
                )
            )
            top += per_group
        body = "".join(body_parts) or '<text x="40" y="110" class="label">No metrics to plot</text>'
        height = max(220, top + 30)
    elif kind == "model":
        rows = data.get("layers", [])
        shown = rows[:100]
        labels = ["Layer", "Type", "Parameters", "Trainable", "Input shape", "Output shape"]
        positions = [32, 315, 485, 580, 675, 790]
        heading = "".join(
            f'<text x="{x}" y="102" class="head">{label}</text>'
            for x, label in zip(positions, labels, strict=True)
        )
        line = '<line x1="32" y1="112" x2="875" y2="112" class="axis"/>'
        rendered: list[str] = []
        for index, row in enumerate(shown):
            y = 132 + index * 22
            name = row.get("name") or "(root)"
            indent = min(22 * row.get("depth", 0), 88)
            fields = [
                (positions[0] + indent, name),
                (positions[1], row.get("type", "")),
                (positions[2], row.get("parameters", 0)),
                (positions[3], row.get("trainable", 0)),
                (positions[4], row.get("input_shape", "—")),
                (positions[5], row.get("output_shape", "—")),
            ]
            rendered.extend(
                f'<text x="{x}" y="{y}" class="row">{_esc(value)}</text>' for x, value in fields
            )
            rendered.append(f'<line x1="32" y1="{y + 8}" x2="875" y2="{y + 8}" class="grid"/>')
        body = heading + line + "".join(rendered)
        height = 150 + max(len(shown), 1) * 22
    elif kind in {"activations", "gradients"}:
        rows = data.get("layers", [])
        shown = rows[:100]
        labels = ["Layer", "Shape / status", "Mean", "Std", "Max |value|", "Finite"]
        positions = [32, 355, 540, 625, 710, 835]
        heading = "".join(
            f'<text x="{x}" y="102" class="head">{label}</text>'
            for x, label in zip(positions, labels, strict=True)
        )
        rendered = []
        for index, row in enumerate(shown):
            y = 130 + index * 23
            values = [
                row.get("name", ""),
                row.get("shape", row.get("status", "")),
                row.get("mean", "—"),
                row.get("std", "—"),
                row.get("max_abs", "—"),
                row.get("finite", "—"),
            ]
            rendered.extend(
                f'<text x="{x}" y="{y}" class="row">{_esc(value)}</text>'
                for x, value in zip(positions, values, strict=True)
            )
            rendered.append(f'<line x1="32" y1="{y + 8}" x2="875" y2="{y + 8}" class="grid"/>')
        body = heading + "".join(rendered)
        height = 145 + max(len(shown), 1) * 23
    else:
        body = f'<text x="32" y="110" class="label">{_esc(view.summary)}</text>'
        height = 210
    height += min(len(view.notes), 3) * 18
    return _svg_header(view, height) + body + "</svg>"


def _plotly(view: Visualization) -> Any:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise MissingDependencyError(
            "Plotly rendering requires the optional dependency. "
            "Install it with `pip install 'seedl[plotly]'`."
        ) from exc

    data = view.data
    layout = dict(title=view.title, template="plotly_white", margin=dict(l=50, r=25, t=65, b=50))
    if view.kind == "scalar":
        value = data.get("value", 0)
        try:
            is_finite = math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            is_finite = False
        if is_finite:
            figure = go.Figure(go.Indicator(mode="number", value=float(value)))
        else:
            figure = go.Figure()
            figure.add_annotation(
                text=_esc(data.get("display_value", "n/a")),
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=36),
            )
    elif view.kind == "sequence":
        values = data["values"]
        figure = go.Figure(
            go.Scatter(
                x=list(range(1, len(values) + 1)),
                y=values,
                mode="lines+markers",
                name=data.get("label", "value"),
            )
        )
        figure.update_xaxes(title="Index")
        figure.update_yaxes(title=data.get("label", "Value"))
    elif view.kind == "matrix":
        figure = go.Figure(go.Heatmap(z=data.get("values", []), colorbar=dict(title="Value")))
        figure.update_xaxes(title="Column")
        figure.update_yaxes(title="Row", autorange="reversed")
    elif view.kind == "image":
        figure = go.Figure(go.Image(z=data.get("pixels", [])))
        figure.update_yaxes(autorange="reversed")
    elif view.kind == "distribution":
        figure = go.Figure(go.Histogram(x=data.get("sample", []), nbinsx=40, name="sampled values"))
        figure.update_xaxes(title="Value")
        figure.update_yaxes(title="Count")
    elif view.kind == "training":
        groups = data.get("groups", [])
        figure = make_subplots(
            rows=max(len(groups), 1),
            cols=1,
            shared_xaxes=False,
            subplot_titles=[g["name"] for g in groups],
        )
        for row_index, group in enumerate(groups, start=1):
            for series in group.get("series", []):
                values = series.get("values", [])
                figure.add_trace(
                    go.Scatter(
                        x=series.get("x", list(range(1, len(values) + 1))),
                        y=values,
                        name=series.get("name"),
                        mode="lines+markers",
                    ),
                    row=row_index,
                    col=1,
                )
            figure.update_xaxes(title_text="Epoch", row=row_index, col=1)
            figure.update_yaxes(title_text=group["name"], row=row_index, col=1)
    elif view.kind == "model":
        rows = data.get("layers", [])[:300]
        headers = ["Layer", "Type", "Parameters", "Trainable", "Input shape", "Output shape"]
        cells = [
            [row.get("name") or "(root)" for row in rows],
            [row.get("type", "") for row in rows],
            [row.get("parameters", 0) for row in rows],
            [row.get("trainable", 0) for row in rows],
            [row.get("input_shape", "—") for row in rows],
            [row.get("output_shape", "—") for row in rows],
        ]
        figure = go.Figure(
            go.Table(header=dict(values=headers), cells=dict(values=cells, height=27))
        )
    elif view.kind == "activations":
        rows = data.get("layers", [])[:100]
        figure = go.Figure(
            go.Box(
                x=[row.get("name", "") for row in rows for _ in row.get("sample", [])],
                y=[value for row in rows for value in row.get("sample", [])],
                name="Activation values",
                boxpoints=False,
            )
        )
        if not any(row.get("sample") for row in rows):
            figure = go.Figure(
                go.Table(
                    header=dict(values=["Layer", "Shape"]),
                    cells=dict(
                        values=[[r.get("name") for r in rows], [r.get("shape") for r in rows]]
                    ),
                )
            )
        figure.update_xaxes(title="Layer")
        figure.update_yaxes(title="Activation")
    elif view.kind == "gradients":
        rows = data.get("layers", [])
        figure = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Maximum absolute gradient", "Sampled gradient values"),
            vertical_spacing=0.18,
        )
        magnitude_rows = [row for row in rows if row.get("max_abs") is not None]
        figure.add_trace(
            go.Bar(
                x=[row.get("name", "") for row in magnitude_rows],
                y=[row["max_abs"] for row in magnitude_rows],
                name="Maximum |gradient|",
            ),
            row=1,
            col=1,
        )
        sample_rows = [row for row in rows if row.get("sample")]
        sample_x = [row.get("name", "") for row in sample_rows for _ in row["sample"]]
        sample_y = [value for row in sample_rows for value in row["sample"]]
        if sample_y:
            figure.add_trace(
                go.Box(
                    x=sample_x,
                    y=sample_y,
                    name="Sampled values",
                    boxpoints=False,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
        figure.update_xaxes(title="Parameter", row=1, col=1)
        figure.update_yaxes(title="Maximum |gradient|", row=1, col=1)
        figure.update_xaxes(title="Parameter", row=2, col=1)
        figure.update_yaxes(title="Gradient", row=2, col=1)
    else:
        raise SeeDLError(f"The Plotly renderer does not support the {view.kind!r} view.")
    figure.update_layout(**layout)
    if view.notes:
        figure.add_annotation(
            text="<br>".join(view.notes),
            xref="paper",
            yref="paper",
            x=0,
            y=-0.2,
            showarrow=False,
            align="left",
        )
    return figure


def _matplotlib(view: Visualization) -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingDependencyError(
            "Matplotlib rendering requires the optional dependency. "
            "Install it with `pip install 'seedl[matplotlib]'`."
        ) from exc

    data = view.data
    figure, axis = plt.subplots(figsize=(10, 5))
    figure.suptitle(view.title)
    if view.kind == "scalar":
        axis.axis("off")
        axis.text(
            0.5,
            0.55,
            str(data.get("display_value", data.get("value", ""))),
            ha="center",
            va="center",
            fontsize=30,
        )
        axis.text(0.5, 0.38, data.get("caption", view.summary), ha="center", va="center")
    elif view.kind == "sequence":
        axis.plot(range(1, len(data["values"]) + 1), data["values"], marker="o", markersize=3)
        axis.set(xlabel="Index", ylabel=data.get("label", "Value"))
    elif view.kind == "matrix":
        axis.imshow(data.get("values", []), aspect="auto", interpolation="nearest", cmap="coolwarm")
        axis.set(xlabel="Column", ylabel="Row")
        figure.colorbar(axis.images[0], ax=axis, label="Value")
    elif view.kind == "image":
        axis.imshow(data.get("pixels", []), cmap="gray" if data.get("channels") == 1 else None)
        axis.axis("off")
    elif view.kind == "distribution":
        axis.hist(data.get("sample", []), bins=40, color="#4b78c2")
        axis.set(xlabel="Value", ylabel="Count")
    elif view.kind == "training":
        plt.close(figure)
        groups = data.get("groups", [])
        figure, axes = plt.subplots(
            max(len(groups), 1), 1, figsize=(10, 3.5 * max(len(groups), 1)), squeeze=False
        )
        figure.suptitle(view.title)
        for axis, group in zip(axes[:, 0], groups, strict=True):
            for series in group.get("series", []):
                axis.plot(
                    series.get("x", range(1, len(series.get("values", [])) + 1)),
                    series.get("values", []),
                    marker="o",
                    markersize=3,
                    label=series.get("name"),
                )
            axis.set(title=group.get("name"), xlabel="Epoch", ylabel=group.get("name"))
            axis.legend()
        if view.notes:
            figure.text(0.01, 0.01, "\n".join(view.notes), fontsize=8, color="#52657d")
        figure.tight_layout()
        return figure
    elif view.kind in {"model", "activations", "gradients"}:
        plt.close(figure)
        if view.kind == "gradients":
            rows = data.get("layers", [])
            figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
            figure.suptitle(view.title)
            magnitude_rows = [row for row in rows if row.get("max_abs") is not None]
            axes[0].bar(
                [row.get("name", "") for row in magnitude_rows],
                [row["max_abs"] for row in magnitude_rows],
                color="#4b78c2",
            )
            axes[0].set(ylabel="Maximum |gradient|", title="Layer-wise magnitude")
            for index, row in enumerate(rows):
                samples = row.get("sample", [])
                if samples:
                    jitter = [
                        ((sample_index % 7) - 3) * 0.025 for sample_index in range(len(samples))
                    ]
                    axes[1].scatter(
                        [index + offset for offset in jitter],
                        samples,
                        s=8,
                        alpha=0.45,
                        color="#315fbd",
                    )
            axes[1].set(
                xticks=range(len(rows)),
                xticklabels=[row.get("name", "") for row in rows],
                xlabel="Parameter",
                ylabel="Gradient",
                title="Sampled gradient values",
            )
            for axis in axes:
                axis.tick_params(axis="x", labelrotation=45)
            if view.notes:
                figure.text(0.01, 0.01, "\n".join(view.notes), fontsize=8, color="#52657d")
            figure.tight_layout(rect=(0, 0.08, 1, 0.95))
            return figure
        figure, axis = plt.subplots(figsize=(12, 0.8 + 0.34 * max(1, len(data.get("layers", [])))))
        axis.axis("off")
        if view.kind == "model":
            columns = ["Layer", "Type", "Parameters", "Trainable", "Input shape", "Output shape"]
            rows = [
                [
                    r.get("name") or "(root)",
                    r.get("type", ""),
                    r.get("parameters", 0),
                    r.get("trainable", 0),
                    r.get("input_shape", "—"),
                    r.get("output_shape", "—"),
                ]
                for r in data.get("layers", [])[:100]
            ]
        else:
            columns = ["Layer", "Shape / status", "Mean", "Std", "Max |value|", "Finite"]
            rows = [
                [
                    r.get("name", ""),
                    r.get("shape", r.get("status", "")),
                    r.get("mean", "—"),
                    r.get("std", "—"),
                    r.get("max_abs", "—"),
                    r.get("finite", "—"),
                ]
                for r in data.get("layers", [])[:100]
            ]
        axis.table(cellText=rows, colLabels=columns, loc="center", cellLoc="left")
        if view.notes:
            figure.text(0.01, 0.01, "\n".join(view.notes), fontsize=8, color="#52657d")
        figure.tight_layout(rect=(0, 0.08, 1, 0.95))
        return figure
    else:
        plt.close(figure)
        raise SeeDLError(f"The Matplotlib renderer does not support the {view.kind!r} view.")
    axis.set_title(view.summary)
    if view.notes:
        figure.text(0.01, 0.01, "\n".join(view.notes), fontsize=8, color="#52657d")
    figure.tight_layout()
    return figure
