"""Lazy renderers for SeeDL's compact, backend-neutral view data."""

from __future__ import annotations

import html
import importlib.util
import math
import textwrap
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
    if view.kind == "tensor_explorer":
        from ._tensor_explorer import svg_tensor_explorer

        resolve_backend(backend)
        return svg_tensor_explorer(view.data, view.metadata, view.notes)
    if view.kind == "training_playback":
        from ._playback import playback_figure, static_training

        resolve_backend(backend)
        return playback_figure(view) if backend == "plotly" else static_training(view, backend)
    if view.kind == "snapshot_comparison":
        from ._snapshot_compare import render_comparison

        resolve_backend(backend)
        return render_comparison(view, backend)
    if view.kind == "quantum_circuit" and view.data.get("simulate") and backend == "plotly":
        from ._playback import playback_figure

        resolve_backend(backend)
        return playback_figure(view)
    if view.kind in {"embeddings", "loss_landscape"} and view.data.get("dimensions") == 3:
        if backend != "plotly":
            raise SeeDLError("This spatial view requires Plotly; export HTML for interactive 3D.")
        from ._spatial import render_spatial

        resolve_backend("plotly")
        return render_spatial(view)
    if view.kind == "execution_trace":
        from ._trace_renderers import matplotlib_trace, plotly_trace, svg_trace

        resolve_backend(backend)
        if backend == "plotly":
            return plotly_trace(view)
        if backend == "matplotlib":
            return matplotlib_trace(view)
        return svg_trace(view)
    if view.kind == "attention_rollout":
        from ._transformer_observatory import svg_attention_rollout

        resolve_backend(backend)
        return svg_attention_rollout(view)
    if view.kind == "attribution_comparison":
        from ._explain_compare import svg_attribution_comparison

        resolve_backend(backend)
        return svg_attribution_comparison(view)
    if view.kind == "gradient_health":
        from ._gradient_health import svg_gradient_health

        resolve_backend(backend)
        return svg_gradient_health(view)
    if view.kind == "quantum_noise":
        from ._quantum_lab import svg_quantum_noise

        resolve_backend(backend)
        return svg_quantum_noise(view)
    if view.kind == "dataset_drift":
        from ._dataset_health import svg_dataset_drift

        resolve_backend(backend)
        return svg_dataset_drift(view)
    if view.kind == "run_comparison":
        from ._training_compare import svg_run_comparison

        resolve_backend(backend)
        return svg_run_comparison(view)
    if view.kind == "feature_maps":
        from ._features import matplotlib_features, plotly_features, svg_features

        resolve_backend(backend)
        if backend == "plotly":
            return plotly_features(view)
        if backend == "matplotlib":
            return matplotlib_features(view)
        return svg_features(view)
    if view.kind == "feature_comparison":
        from ._features import (
            matplotlib_feature_comparison,
            plotly_feature_comparison,
            svg_feature_comparison,
        )

        resolve_backend(backend)
        if backend == "plotly":
            return plotly_feature_comparison(view)
        if backend == "matplotlib":
            return matplotlib_feature_comparison(view)
        return svg_feature_comparison(view)
    if (
        view.kind in {"attention", "attention_comparison"}
        and view.data.get("presentation") == "attention"
    ):
        from ._transformers import (
            matplotlib_attention,
            plotly_attention,
            plotly_attention_comparison,
            svg_attention,
            svg_attention_comparison,
        )

        resolve_backend(backend)
        is_cmp = view.data.get("head_comparison") and view.data["head_comparison"].get("available")
        if backend == "plotly":
            return plotly_attention_comparison(view) if is_cmp else plotly_attention(view)
        if backend == "matplotlib":
            return matplotlib_attention(view)
        return svg_attention_comparison(view) if is_cmp else svg_attention(view)
    if view.kind == "model" and view.data.get("presentation") == "cnn_3d":
        from ._cnn_spatial import plotly_cnn_3d, svg_cnn_3d

        resolve_backend(backend)
        if backend == "plotly":
            return plotly_cnn_3d(view)
        return svg_cnn_3d(view)
    if view.kind == "bloch":
        from ._quantum_renderers import plotly_bloch, svg_bloch

        resolve_backend(backend)
        if backend == "plotly":
            return plotly_bloch(view)
        return svg_bloch(view)
    if view.kind == "quantum_circuit":
        from ._quantum_renderers import plotly_circuit, svg_circuit

        resolve_backend(backend)
        if backend == "plotly":
            return plotly_circuit(view)
        return svg_circuit(view)
    if view.kind == "statevector":
        from ._quantum_renderers import plotly_statevector, svg_statevector

        resolve_backend(backend)
        if backend == "plotly":
            return plotly_statevector(view)
        return svg_statevector(view)
    if view.kind == "entanglement":
        from ._quantum_renderers import plotly_entanglement, svg_entanglement

        resolve_backend(backend)
        if backend == "plotly":
            return plotly_entanglement(view)
        return svg_entanglement(view)
    if view.kind == "density_matrix":
        if len(view.data.get("density_matrix", [])) == 2:
            from ._quantum_renderers import plotly_bloch, svg_bloch

            resolve_backend(backend)
            if backend == "plotly":
                return plotly_bloch(view)
            return svg_bloch(view)
        else:
            from ._quantum_renderers import plotly_statevector, svg_statevector

            resolve_backend(backend)
            if backend == "plotly":
                return plotly_statevector(view)
            return svg_statevector(view)
    if view.kind == "model" and view.data.get("presentation") in {"neurons", "architecture"}:
        from ._network_renderers import matplotlib_network, plotly_network, svg_network

        resolve_backend(backend)
        return {"svg": svg_network, "plotly": plotly_network, "matplotlib": matplotlib_network}[
            backend
        ](view)
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


def _details(view: Visualization, width: int = 110) -> list[str]:
    lines = []
    items = list(view.metadata.items())
    for start in range(0, len(items), 3):
        lines.extend(
            textwrap.wrap(" · ".join(f"{k}: {v}" for k, v in items[start : start + 3]), width)
        )
    for note in view.notes:
        lines.extend(textwrap.wrap(note, width))
    if (
        view.kind in {"model", "activations", "gradients"}
        and len(view.data.get("layers", [])) > 100
    ):
        lines.append(
            f"Showing first 100 of {len(view.data['layers'])} rows; all rows remain in view.data."
        )
    return lines


def _reduce_attribution_matrix(attr: Any) -> list[Any]:
    """Reduce multi-dimensional attribution array to 2D matrix or 1D vector for display."""
    if not isinstance(attr, list) or not attr:
        return []
    if not isinstance(attr[0], list):
        return attr
    if not isinstance(attr[0][0], list):
        if len(attr) == 1:
            return attr[0]
        return attr
    if not isinstance(attr[0][0][0], list):
        h = len(attr[0])
        w = len(attr[0][0])
        out_2d = [[0.0] * w for _ in range(h)]
        num_slices = max(len(attr), 1)
        for s in attr:
            for r in range(h):
                for c in range(w):
                    out_2d[r][c] += abs(float(s[r][c])) / num_slices
        return out_2d
    return _reduce_attribution_matrix(attr[0])


def _normalized(value: float, low: float, high: float) -> float:
    if high == low:
        return 0.5
    scale = max(abs(low), abs(high)) or 1.0
    return (value / scale - low / scale) / (high / scale - low / scale)


def _table_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.5g}" if math.isfinite(value) else "n/a"
    return str(value)


def _stats_rows(data: dict[str, Any]) -> list[list[str]]:
    return [
        [
            _table_value(row.get(key, "—"))
            for key in ("name", "shape", "status", "mean", "std", "max_abs", "finite")
        ]
        for row in data.get("layers", [])[:100]
    ]


def _svg_table_text(x: int, y: int, value: Any, width: int) -> str:
    full = _table_value(value)
    limit = max(5, width // 7)
    short = full if len(full) <= limit else full[: limit - 1] + "…"
    if len(short) * 7 > width:
        short = short[: max(1, width // 7 - 1)] + "…"
    return f'<text x="{x}" y="{y}" class="row"><title>{_esc(full)}</title>{_esc(short)}</text>'


def _svg_header(view: Visualization, height: int) -> str:
    title = _esc(view.title)
    summary = _esc(view.summary)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{title}" '
        f'viewBox="0 0 920 {height}" width="100%" height="{height}" '
        'style="min-width:760px">'
        f"<title>{title}</title><desc>{_esc(str(view))}</desc>"
        "<style>text{font-family:Inter,ui-sans-serif,system-ui,sans-serif;fill:#172b4d}"
        ".title{font-size:23px;font-weight:650}.sub{font-size:13px;fill:#52657d}"
        ".note{font-size:12px;fill:#52657d}.axis{stroke:#b7c4d3;stroke-width:1}"
        ".grid{stroke:#e5ebf1;stroke-width:1}.label{font-size:11px;fill:#52657d}"
        ".value{font-size:44px;font-weight:650}.cell{stroke:white;stroke-width:.6}"
        ".row{font-size:12px}.head{font-size:11px;font-weight:650;fill:#52657d}"
        ".track{fill:#e6edf5}.bar{fill:#4b78c2}</style>"
        f'<text x="32" y="38" class="title">{title}</text>'
        f'<text x="32" y="61" class="sub">{summary}</text>'
    )


def _svg_line_chart(
    traces: list[dict[str, Any]],
    *,
    top: int,
    height: int,
    title: str = "",
    x_label: str = "Index",
    y_label: str = "Value",
    width: int = 810,
    left: int = 82,
) -> str:
    bottom = top + height
    right = left + width
    data = [(t, _finite(t.get("values", []))) for t in traces]
    values = [value for _, series in data for value in series]
    if not values:
        return (
            f'<text x="{left}" y="{top}" class="head">{_esc(title or y_label)}</text>'
            f'<text x="{left}" y="{top + 30}" class="label">No finite values to plot</text>'
        )
    low, high = min(values), max(values)
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
        dash = 'stroke-dasharray="6 4"' if trace.get("role") == "validation" else ""
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
                x = left + _normalized(x_value, x_low, x_high) * width
                y = bottom - _normalized(number, low, high) * height
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
                    f'stroke-width="2.4" stroke-linejoin="round" {dash}/>'
                )
                paths.extend(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.3" fill="{color}"/>'
                    for x, y in segment
                )
            x = left + (index % 4) * 200
            legend_y = bottom + 32 + (index // 4) * 20
            name = str(trace.get("name", "series"))
            name_label = name if len(name) <= 23 else name[:22] + "…"
            legend.append(
                f'<line x1="{x}" y1="{legend_y}" x2="{x + 18}" y2="{legend_y}" '
                f'stroke="{color}" stroke-width="3" {dash}/>'
                f'<text x="{x + 24}" y="{legend_y + 4}" '
                f'class="label"><title>{_esc(name)}</title>{_esc(name_label)}</text>'
            )
    high_label, low_label = _esc(f"{high:.4g}"), _esc(f"{low:.4g}")
    x_axis_y = bottom + 56 + max(0, math.ceil(len(traces) / 4) - 1) * 20
    return (
        f'{title_svg}{grid}<line x1="{left}" y1="{bottom}" x2="{right}" '
        f'y2="{bottom}" class="axis"/><line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{bottom}" class="axis"/>'
        f'<text x="{left - 8}" y="{top + 3}" text-anchor="end" class="label">{high_label}</text>'
        f'<text x="{left - 8}" y="{bottom + 4}" text-anchor="end" class="label">{low_label}</text>'
        f'<text x="{left + width / 2}" y="{x_axis_y}" text-anchor="middle" '
        f'class="label">{_esc(x_label)}</text>'
        f'<text x="18" y="{top + height / 2}" text-anchor="middle" class="label" '
        f'transform="rotate(-90 18 {top + height / 2})">{_esc(y_label)}</text>'
        f'<text x="{left}" y="{bottom + 16}" class="label">{x_low:.4g}</text>'
        f'<text x="{right}" y="{bottom + 16}" text-anchor="end" class="label">{x_high:.4g}</text>'
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
            [{"name": data.get("label", "value"), "values": data["values"], "x": data.get("x")}],
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
                    ratio = _normalized(number, low, high)
                    red = round(49 + 190 * ratio)
                    green = round(95 + 115 * (1 - abs(ratio - 0.5) * 2))
                    blue = round(189 - 120 * ratio)
                    color = f"rgb({red},{green},{blue})"
                rects.append(
                    f'<rect class="cell" x="{x0 + j * cell:.1f}" y="{y0 + i * cell:.1f}" '
                    f'width="{cell:.1f}" height="{cell:.1f}" fill="{color}"><title>'
                    f"row {data.get('row_indices', list(range(row_count)))[i]}, "
                    f"column {data.get('column_indices', list(range(col_count)))[j]}: "
                    f"{_esc(value)}</title></rect>"
                )
        body = (
            f'<text x="{x0}" y="87" class="label">columns</text>'
            f'<text x="42" y="{y0 + 14}" class="label">rows</text>{"".join(rects)}'
            f'<text x="{x0}" y="{y0 + row_count * cell + 22}" class="label">'
            f"Color scale: blue {low:.4g} → red {high:.4g}; gray = non-finite. "
            f"Row/column indices are zero-based; hover cells for values.</text>"
        )
        height = int(y0 + row_count * cell + 55)
    elif kind in {"image", "dataset_images"}:
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
                    f'width="{size if channels == 4 else size + 0.2:.1f}" '
                    f'height="{size if channels == 4 else size + 0.2:.1f}" fill="{color}" '
                    f'fill-opacity="{components[3] if channels == 4 else 1}" '
                    'shape-rendering="crispEdges"/>'
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
                else min(count - 1, int(_normalized(sample_value, low, high) * count))
            )
            bins[index] += 1
        maximum = max(bins, default=1) or 1
        bars = []
        for i, bin_count in enumerate(bins):
            bar_h = 150 * bin_count / maximum
            bars.append(
                f'<rect x="{70 + i * 750 / count}" y="{250 - bar_h:.1f}" '
                f'width="{750 / count - 2:.1f}" '
                f'height="{bar_h:.1f}" rx="2" fill="#4b78c2">'
                f"<title>{bin_count} sampled values</title></rect>"
            )
        body = (
            "".join(bars)
            + '<line x1="70" y1="250" x2="825" y2="250" class="axis"/>'
            + f'<text x="70" y="267" class="label">{low:.4g}</text>'
            + f'<text x="820" y="267" text-anchor="end" class="label">{high:.4g}</text>'
            + f'<text x="60" y="103" text-anchor="end" class="label">{maximum}</text>'
            + (
                '<text x="70" y="125" class="label">No finite values to plot</text>'
                if not distribution_values
                else ""
            )
            + '<text x="447" y="278" text-anchor="middle" class="label">Value</text>'
            + '<text x="30" y="165" text-anchor="middle" class="label" '
            'transform="rotate(-90 30 165)">Count</text>'
        )
        height = 310
    elif kind in {"training", "history_comparison"}:
        groups = data.get("groups", [])
        body_parts = []
        top = 105
        per_group = 205
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
            top += per_group + max(0, math.ceil(len(traces) / 4) - 1) * 20
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
                _svg_table_text(
                    x,
                    y,
                    value,
                    positions[min(i + 1, len(positions) - 1)] - x - 8
                    if i < len(positions) - 1
                    else 120,
                )
                for i, (x, value) in enumerate(fields)
            )
            rendered.append(f'<line x1="32" y1="{y + 8}" x2="875" y2="{y + 8}" class="grid"/>')
        body = heading + line + "".join(rendered)
        height = 150 + max(len(shown), 1) * 22
    elif kind in {"activations", "gradients"}:
        rows = data.get("layers", [])
        shown = rows[:100]
        labels = ["Layer", "Shape", "Status", "Mean", "Std", "Max |value|", "Finite"]
        positions = [32, 260, 405, 520, 615, 705, 825]
        heading = "".join(
            f'<text x="{x}" y="102" class="head">{label}</text>'
            for x, label in zip(positions, labels, strict=True)
        )
        rendered = []
        for index, row in enumerate(shown):
            y = 130 + index * 23
            values = [
                row.get("name", ""),
                row.get("shape", "—"),
                row.get("status", "available"),
                row.get("mean", "—"),
                row.get("std", "—"),
                row.get("max_abs", "—"),
                row.get("finite", "—"),
            ]
            rendered.extend(
                _svg_table_text(x, y, value, positions[i + 1] - x - 8 if i < 6 else 80)
                for i, (x, value) in enumerate(zip(positions, values, strict=True))
            )
            rendered.append(f'<line x1="32" y1="{y + 8}" x2="875" y2="{y + 8}" class="grid"/>')
        body = heading + "".join(rendered)
        height = 145 + max(len(shown), 1) * 23
    elif kind == "classification_predictions":
        ranked = data.get("ranked", [])
        bars = []
        for i, item in enumerate(ranked):
            prob = float(item.get("probability", 0.0))
            bar_w = max(2, int(prob * 450))
            y = 110 + i * 32
            bars.append(
                f'<text x="40" y="{y + 16}" class="row">{_esc(item.get("class", ""))}</text>'
                f'<rect x="180" y="{y}" width="{bar_w}" height="22" rx="3" fill="#4b78c2"/>'
                f'<text x="{190 + bar_w}" y="{y + 16}" class="label">{prob:.1%}</text>'
            )
        body = "".join(bars)
        height = 130 + len(ranked) * 32
    elif kind == "prediction_comparison":
        rows = data.get("rows", [])[:15]
        heading = (
            '<text x="40" y="105" class="head">Class</text>'
            f'<text x="240" y="105" class="head">{_esc(data.get("name_a", "Model A"))}</text>'
            f'<text x="400" y="105" class="head">{_esc(data.get("name_b", "Model B"))}</text>'
            '<text x="560" y="105" class="head">Difference</text>'
            '<line x1="40" y1="115" x2="720" y2="115" class="axis"/>'
        )
        rendered = []
        for i, row in enumerate(rows):
            y = 135 + i * 24
            delta = float(row.get("delta", 0.0))
            delta_str = f"{delta:+.1%}"
            color = "#24a079" if delta > 0 else "#db5264" if delta < 0 else "#52657d"
            rendered.append(
                f'<text x="40" y="{y}" class="row">{_esc(row.get("class", ""))}</text>'
                f'<text x="240" y="{y}" class="row">{float(row.get("score_a", 0.0)):.1%}</text>'
                f'<text x="400" y="{y}" class="row">{float(row.get("score_b", 0.0)):.1%}</text>'
                f'<text x="560" y="{y}" class="row" fill="{color}">{delta_str}</text>'
                f'<line x1="40" y1="{y + 6}" x2="720" y2="{y + 6}" class="grid"/>'
            )
        body = heading + "".join(rendered)
        height = 150 + len(rows) * 24
    elif kind == "model_comparison":
        rows = data.get("layer_rows", [])[:30]
        heading = (
            '<text x="40" y="105" class="head">Layer</text>'
            f'<text x="250" y="105" class="head">{_esc(data.get("name_a", "Model A"))} Params</text>'
            f'<text x="450" y="105" class="head">{_esc(data.get("name_b", "Model B"))} Params</text>'
            '<line x1="40" y1="115" x2="720" y2="115" class="axis"/>'
        )
        rendered = []
        for i, row in enumerate(rows):
            y = 135 + i * 22
            p_a = row.get("params_a", 0)
            p_b = row.get("params_b", 0)
            rendered.append(
                f'<text x="40" y="{y}" class="row">{_esc(row.get("layer", ""))}</text>'
                f'<text x="250" y="{y}" class="row">{p_a:, if isinstance(p_a, int) else p_a}</text>'
                f'<text x="450" y="{y}" class="row">{p_b:, if isinstance(p_b, int) else p_b}</text>'
                f'<line x1="40" y1="{y + 6}" x2="720" y2="{y + 6}" class="grid"/>'
            )
        body = heading + "".join(rendered)
        height = 150 + len(rows) * 22
    elif kind in {"attention", "educational_attention"}:
        matrix = data.get("matrix", [])
        tokens = data.get("tokens", [])
        n_tok = len(matrix)
        cell = min(36.0, 500.0 / max(n_tok, 1))
        x0, y0 = 120, 110
        rects = []
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                intensity = round(float(val) * 255)
                color = f"rgb({255 - intensity}, {255 - intensity // 2}, 255)"
                tok_from = tokens[i] if i < len(tokens) else str(i)
                tok_to = tokens[j] if j < len(tokens) else str(j)
                rects.append(
                    f'<rect class="cell" x="{x0 + j * cell:.1f}" y="{y0 + i * cell:.1f}" '
                    f'width="{cell:.1f}" height="{cell:.1f}" fill="{color}">'
                    f"<title>{_esc(tok_from)} -> {_esc(tok_to)}: {float(val):.3f}</title></rect>"
                )
        labels_svg = []
        for idx in range(min(n_tok, 20)):
            tok = tokens[idx] if idx < len(tokens) else str(idx)
            labels_svg.append(
                f'<text x="{x0 - 8}" y="{y0 + idx * cell + cell * 0.7:.1f}" text-anchor="end" class="label">{_esc(tok)}</text>'
            )
            labels_svg.append(
                f'<text x="{x0 + idx * cell + cell * 0.5:.1f}" y="{y0 - 8}" text-anchor="middle" class="label">{_esc(tok)}</text>'
            )
        body = "".join(labels_svg) + "".join(rects)
        height = int(y0 + n_tok * cell + 60)
    elif kind in {"educational_activations", "educational_losses"}:
        funcs = data.get("functions", {})
        if not funcs and "mse" in data:
            funcs = {k: data[k] for k in ("mse", "mae", "huber") if k in data}
        x_coords = data.get("x", data.get("errors", []))
        traces = [{"name": name, "values": vals, "x": x_coords} for name, vals in funcs.items()]
        body = _svg_line_chart(
            traces, top=105, height=180, title="Function Curves", x_label="x", y_label="f(x)"
        )
        height = 360
    elif kind == "educational_gradient_descent":
        traj = data.get("trajectory", [])
        traces = [
            {"name": "Loss", "values": [p["loss"] for p in traj], "x": [p["step"] for p in traj]},
            {
                "name": "Parameter w",
                "values": [p["w"] for p in traj],
                "x": [p["step"] for p in traj],
            },
        ]
        body = _svg_line_chart(
            traces,
            top=105,
            height=180,
            title="Optimization Progress",
            x_label="Step",
            y_label="Value",
        )
        height = 360
    elif kind == "educational_backpropagation":
        nodes = data.get("nodes", [])
        heading = (
            '<text x="40" y="105" class="head">Graph Node</text>'
            '<text x="320" y="105" class="head">Forward Output</text>'
            '<text x="520" y="105" class="head">Local Gradient (dL/d_)</text>'
            '<line x1="40" y1="115" x2="720" y2="115" class="axis"/>'
        )
        rendered = []
        for i, node in enumerate(nodes):
            y = 135 + i * 24
            rendered.append(
                f'<text x="40" y="{y}" class="row">{_esc(node.get("name", ""))}</text>'
                f'<text x="320" y="{y}" class="row">{float(node.get("forward", 0.0)):.4g}</text>'
                f'<text x="520" y="{y}" class="row" fill="#315fbd">{float(node.get("gradient", 0.0)):.4g}</text>'
                f'<line x1="40" y1="{y + 6}" x2="720" y2="{y + 6}" class="grid"/>'
            )
        body = heading + "".join(rendered)
        height = 150 + len(nodes) * 24
    elif kind == "embeddings":
        sim_mat = data.get("similarity_matrix", [])
        labels = data.get("labels", [])
        n_v = len(sim_mat)
        cell = min(28.0, 480.0 / max(n_v, 1))
        x0, y0 = 100, 105
        rects = []
        for i, row in enumerate(sim_mat):
            for j, val in enumerate(row):
                ratio = (float(val) + 1.0) / 2.0  # map [-1, 1] to [0, 1]
                red = round(49 + 190 * ratio)
                blue = round(189 - 120 * ratio)
                color = f"rgb({red}, 95, {blue})"
                lbl_i = labels[i] if i < len(labels) else str(i)
                lbl_j = labels[j] if j < len(labels) else str(j)
                rects.append(
                    f'<rect class="cell" x="{x0 + j * cell:.1f}" y="{y0 + i * cell:.1f}" '
                    f'width="{cell:.1f}" height="{cell:.1f}" fill="{color}">'
                    f"<title>{_esc(lbl_i)} · {_esc(lbl_j)}: {float(val):.3f}</title></rect>"
                )
        body = "".join(rects)
        height = int(y0 + n_v * cell + 60)
    else:
        body = f'<text x="32" y="110" class="label">{_esc(view.summary)}</text>'
        height = 210
    details = _details(view)
    footer = "".join(
        f'<text x="32" y="{height + 18 * i}" class="note">{_esc(line)}</text>'
        for i, line in enumerate(details)
    )
    height += len(details) * 18 + 20
    return _svg_header(view, height) + body + footer + "</svg>"


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
    layout: dict[str, Any] = dict(
        title=view.title, template="plotly_white", margin=dict(l=50, r=25, t=65, b=50)
    )
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
                x=data.get("x", list(range(len(values)))),
                y=values,
                mode="lines+markers",
                name=data.get("label", "value"),
            )
        )
        figure.update_xaxes(title="Index")
        figure.update_yaxes(title=data.get("label", "Value"))
    elif view.kind == "matrix":
        figure = go.Figure(
            go.Heatmap(
                z=data.get("values", []),
                x=data.get("column_indices"),
                y=data.get("row_indices"),
                colorbar=dict(title="Value"),
            )
        )
        figure.update_xaxes(title="Column")
        figure.update_yaxes(title="Row", autorange="reversed")
    elif view.kind in {"image", "dataset_images"}:
        pixels = data.get("pixels", [])
        if data.get("channels") == 1:
            figure = go.Figure(
                go.Heatmap(
                    z=[[p[0] for p in row] for row in pixels],
                    zmin=0,
                    zmax=1,
                    colorscale="gray",
                    showscale=False,
                )
            )
        else:
            channels = data.get("channels", 3)
            figure = go.Figure(
                go.Image(
                    z=pixels,
                    colormodel="rgba" if channels == 4 else "rgb",
                    zmin=[0] * 4,
                    zmax=[1] * 4,
                )
            )
        figure.update_xaxes(title="Preview column")
        figure.update_yaxes(title="Preview row")
        figure.update_yaxes(autorange="reversed")
    elif view.kind == "distribution":
        figure = go.Figure(
            go.Histogram(x=_finite(data.get("sample", [])), nbinsx=40, name="sampled values")
        )
        figure.update_xaxes(title="Value")
        figure.update_yaxes(title="Count")
    elif view.kind in {"training", "history_comparison"}:
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
                        line=dict(dash="dash" if series.get("role") == "validation" else "solid"),
                    ),
                    row=row_index,
                    col=1,
                )
            figure.update_xaxes(title_text="Epoch", row=row_index, col=1)
            figure.update_yaxes(title_text=group["name"], row=row_index, col=1)
    elif view.kind == "model":
        rows = data.get("layers", [])[:100]
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
            go.Table(
                header=dict(values=headers),
                cells=dict(values=cells, height=27),
                columnwidth=[2.2, 1.4, 1, 1, 1.8, 1.8],
            )
        )
    elif view.kind == "activations":
        rows = data.get("layers", [])[:20]
        figure = make_subplots(
            rows=2,
            cols=1,
            specs=[[{"type": "xy"}], [{"type": "table"}]],
            row_heights=[0.4, 0.6],
            vertical_spacing=0.1,
        )
        figure.add_trace(
            go.Box(
                x=[row.get("name", "") for row in rows for _ in row.get("sample", [])],
                y=[value for row in rows for value in row.get("sample", [])],
                name="Sampled activation values",
                boxpoints=False,
            )
        )
        figure.update_xaxes(title="Layer", type="category")
        figure.update_yaxes(title="Activation")
    elif view.kind == "gradients":
        rows = data.get("layers", [])[:20]
        figure = make_subplots(
            rows=3,
            cols=1,
            specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]],
            row_heights=[0.25, 0.25, 0.5],
            subplot_titles=("Maximum absolute gradient", "Sampled gradient values", "Summaries"),
            vertical_spacing=0.1,
        )
        magnitude_rows = [row for row in rows[:20] if row.get("max_abs") is not None]
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
        figure.update_xaxes(title="Parameter", type="category", row=1, col=1)
        figure.update_yaxes(title="Maximum |gradient|", row=1, col=1)
        figure.update_xaxes(title="Parameter", type="category", row=2, col=1)
        figure.update_yaxes(title="Gradient", row=2, col=1)
    elif view.kind == "classification_predictions":
        ranked = data.get("ranked", [])
        figure = go.Figure(
            go.Bar(
                x=[item.get("probability", 0.0) for item in ranked],
                y=[item.get("class", "") for item in ranked],
                orientation="h",
                marker=dict(color="#4b78c2"),
            )
        )
        figure.update_layout(
            xaxis=dict(title="Probability", tickformat=".0%"),
            yaxis=dict(autorange="reversed"),
        )
    elif view.kind == "prediction_comparison":
        rows = data.get("rows", [])[:15]
        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                name=data.get("name_a", "Model A"),
                x=[r.get("class", "") for r in rows],
                y=[r.get("score_a", 0.0) for r in rows],
            )
        )
        figure.add_trace(
            go.Bar(
                name=data.get("name_b", "Model B"),
                x=[r.get("class", "") for r in rows],
                y=[r.get("score_b", 0.0) for r in rows],
            )
        )
        figure.update_layout(barmode="group", yaxis=dict(title="Score", tickformat=".0%"))
    elif view.kind == "regression_predictions":
        preds = data.get("predictions", [])
        targets = data.get("targets")
        if targets:
            figure = go.Figure(
                go.Scatter(
                    x=targets,
                    y=preds,
                    mode="markers",
                    name="Predictions vs Targets",
                    marker=dict(color="#4b78c2", size=6),
                )
            )
            figure.update_xaxes(title="Target")
            figure.update_yaxes(title="Prediction")
        else:
            figure = go.Figure(
                go.Scatter(
                    y=preds,
                    mode="lines+markers",
                    name="Predictions",
                    marker=dict(color="#4b78c2", size=4),
                )
            )
            figure.update_xaxes(title="Sample Index")
            figure.update_yaxes(title="Prediction")
    elif view.kind == "model_comparison":
        rows = data.get("layer_rows", [])[:50]
        headers = [
            "Layer",
            f"{data.get('name_a', 'Model A')} Type",
            f"{data.get('name_b', 'Model B')} Type",
            f"{data.get('name_a', 'Model A')} Params",
            f"{data.get('name_b', 'Model B')} Params",
        ]
        cells = [
            [r.get("layer", "") for r in rows],
            [r.get("type_a", "—") for r in rows],
            [r.get("type_b", "—") for r in rows],
            [str(r.get("params_a", "—")) for r in rows],
            [str(r.get("params_b", "—")) for r in rows],
        ]
        figure = go.Figure(
            go.Table(
                header=dict(values=headers),
                cells=dict(values=cells, height=27),
            )
        )
    elif view.kind in {"attention", "educational_attention"}:
        figure = go.Figure(
            go.Heatmap(
                z=data.get("matrix", []),
                x=data.get("tokens"),
                y=data.get("tokens"),
                colorscale="Blues",
            )
        )
        figure.update_xaxes(title="Key Token")
        figure.update_yaxes(title="Query Token", autorange="reversed")
    elif view.kind == "embeddings":
        coords = data.get("coords_2d")
        labels = data.get("labels", [])
        if coords and len(coords[0]) >= 2:
            figure = go.Figure(
                go.Scatter(
                    x=[c[0] for c in coords],
                    y=[c[1] for c in coords],
                    text=labels,
                    mode="markers+text",
                    textposition="top center",
                    marker=dict(color="#4b78c2", size=8),
                )
            )
            figure.update_xaxes(title="Principal Component 1")
            figure.update_yaxes(title="Principal Component 2")
        else:
            figure = go.Figure(
                go.Heatmap(
                    z=data.get("similarity_matrix", []),
                    x=labels,
                    y=labels,
                    colorscale="Viridis",
                )
            )
    elif view.kind in {"educational_activations", "educational_losses"}:
        funcs = data.get("functions", {})
        if not funcs and "mse" in data:
            funcs = {k: data[k] for k in ("mse", "mae", "huber") if k in data}
        x_coords = data.get("x", data.get("errors", []))
        figure = go.Figure()
        for k, v in funcs.items():
            figure.add_trace(go.Scatter(x=x_coords, y=v, mode="lines", name=k))
        figure.update_xaxes(title="Input")
        figure.update_yaxes(title="Output")
    elif view.kind == "educational_gradient_descent":
        traj = data.get("trajectory", [])
        figure = go.Figure(
            go.Scatter(
                x=[p["step"] for p in traj],
                y=[p["loss"] for p in traj],
                mode="lines+markers",
                name="Loss",
                line=dict(color="#4b78c2"),
            )
        )
        figure.update_xaxes(title="Optimization Step")
        figure.update_yaxes(title="Objective Loss")
    elif view.kind == "educational_backpropagation":
        nodes = data.get("nodes", [])
        headers = ["Graph Node", "Forward Value", "Local Gradient"]
        cells = [
            [n.get("name", "") for n in nodes],
            [f"{float(n.get('forward', 0.0)):.4g}" for n in nodes],
            [f"{float(n.get('gradient', 0.0)):.4g}" for n in nodes],
        ]
        figure = go.Figure(
            go.Table(
                header=dict(values=headers),
                cells=dict(values=cells, height=27),
            )
        )
    elif view.kind == "dataset_tabular":
        f_stats = data.get("feature_stats", [])[:30]
        headers = ["Feature", "Min", "Max", "Mean"]
        cells = [
            [s.get("feature", "") for s in f_stats],
            [f"{s.get('min', 0.0):.4g}" for s in f_stats],
            [f"{s.get('max', 0.0):.4g}" for s in f_stats],
            [f"{s.get('mean', 0.0):.4g}" for s in f_stats],
        ]
        figure = go.Figure(
            go.Table(
                header=dict(values=headers),
                cells=dict(values=cells, height=27),
            )
        )
    elif view.kind in {"convolution_simulation", "educational_convolution"}:
        feat_map = data.get("feature_map", data.get("output", []))
        figure = go.Figure(
            go.Heatmap(
                z=feat_map,
                colorscale="Viridis",
            )
        )
        figure.update_xaxes(title="Output X")
        figure.update_yaxes(title="Output Y", autorange="reversed")
    elif view.kind == "attribution":
        raw_attr = data.get("attribution", [])
        attr = _reduce_attribution_matrix(raw_attr)
        if attr and isinstance(attr[0], list):
            figure = go.Figure(go.Heatmap(z=attr, colorscale="Hot"))
            figure.update_xaxes(title="Spatial X")
            figure.update_yaxes(title="Spatial Y", autorange="reversed")
        else:
            figure = go.Figure(go.Bar(y=attr, marker=dict(color="#4b78c2")))
            figure.update_xaxes(title="Feature Index")
            figure.update_yaxes(title="Attribution")
    elif view.kind == "educational_dropout":
        figure = go.Figure(
            go.Bar(
                x=list(range(len(data.get("inputs", [])))),
                y=data.get("train_outputs", []),
                name="Retained Outputs",
                marker=dict(color="#4b78c2"),
            )
        )
        figure.update_xaxes(title="Unit Index")
        figure.update_yaxes(title="Activation Value")
    elif view.kind == "educational_batch_norm":
        norm = data.get("normalized", [])
        figure = go.Figure(
            go.Scatter(
                y=norm,
                mode="lines+markers",
                name="Standardized Output",
                marker=dict(color="#4b78c2"),
            )
        )
        figure.update_xaxes(title="Sample Index")
        figure.update_yaxes(title="Standardized Value")
    else:
        raise SeeDLError(f"The Plotly renderer does not support the {view.kind!r} view.")
    if view.kind in {"activations", "gradients"}:
        stats = _stats_rows(data)
        headers = ["Layer", "Shape", "Status", "Mean", "Std", "Max |value|", "Finite"]
        figure.add_trace(
            go.Table(
                header=dict(values=headers),
                cells=dict(
                    values=list(zip(*stats, strict=True)) if stats else [() for _ in headers],
                    height=27,
                ),
                columnwidth=[2.4, 1.5, 1.3, 1, 1, 1.2, 1],
            ),
            row=2 if view.kind == "activations" else 3,
            col=1,
        )
    details = _details(view)
    base_height = (
        max(300, 150 + min(len(data.get("layers", [])), 100) * 27)
        if view.kind == "model"
        else max(700, 350 + min(len(data.get("layers", [])), 100) * 40)
        if view.kind in {"activations", "gradients"}
        else max(360, len(data.get("groups", [])) * 250)
        if view.kind == "training"
        else 360
    )
    footer_height = 18 * (len(details) + 2)
    layout.update(
        height=base_height + footer_height,
        margin=dict(l=65, r=30, t=95, b=footer_height + 65),
        title=dict(text=_esc(view.title) + "<br><sup>" + _esc(view.summary) + "</sup>"),
    )
    figure.update_layout(**layout)
    if view.kind in {"sequence", "distribution"} and not _finite(
        data.get("values", data.get("sample", []))
    ):
        figure.add_annotation(
            text="No finite values to plot",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
    if details:
        figure.add_annotation(
            text="<br>".join(_esc(line) for line in details),
            xref="paper",
            yref="paper",
            x=0,
            y=0,
            yshift=-65,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            align="left",
            font=dict(size=12),
        )
    return figure


def _mpl_table(axis: Any, rows: list[Any], columns: list[str]) -> None:
    if not rows:
        axis.text(0.5, 0.5, "No rows to show", ha="center", transform=axis.transAxes)
        return
    wrapped = [
        ["\n".join(textwrap.wrap(_table_value(v), 25 if i == 0 else 18)) for i, v in enumerate(row)]
        for row in rows
    ]
    table = axis.table(cellText=wrapped, colLabels=columns, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    row_lines = [max(v.count("\n") + 1 for v in row) for row in wrapped]
    total_lines = sum(row_lines) + 1
    for col in range(len(columns)):
        table[0, col].set_height(0.9 / total_lines)
    for index, line_count in enumerate(row_lines, 1):
        height = 0.9 * line_count / total_lines
        for col in range(len(columns)):
            table[index, col].set_height(height)


def _finish_matplotlib(figure: Any, view: Visualization) -> Any:
    lines = _details(view)
    height = figure.get_figheight()
    footer_inches = 0.18 * len(lines) + 0.2
    figure.set_figheight(height + footer_inches)
    if lines:
        figure.text(0.02, 0.02, "\n".join(lines), fontsize=9, color="#52657d")
    figure.suptitle(view.title + "\n" + view.summary, fontsize=12)
    figure.tight_layout(rect=(0, footer_inches / (height + footer_inches), 1, 0.93))
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
        axis.plot(
            data.get("x", range(len(data["values"]))), data["values"], marker="o", markersize=3
        )
        axis.set(xlabel="Index", ylabel=data.get("label", "Value"))
    elif view.kind == "matrix":
        axis.imshow(data.get("values", []), aspect="auto", interpolation="nearest", cmap="coolwarm")
        axis.set(xlabel="Column", ylabel="Row")
        for which, indices in (("x", data.get("column_indices")), ("y", data.get("row_indices"))):
            if indices:
                ticks = list(range(0, len(indices), max(1, math.ceil(len(indices) / 6))))
                getattr(axis, f"set_{which}ticks")(ticks, [indices[i] for i in ticks])
        figure.colorbar(axis.images[0], ax=axis, label="Value")
    elif view.kind in {"image", "dataset_images"}:
        axis.imshow(
            [[p[0] for p in row] for row in data.get("pixels", [])]
            if data.get("channels") == 1
            else data.get("pixels", []),
            cmap="gray" if data.get("channels") == 1 else None,
            vmin=0,
            vmax=1,
        )
        axis.axis("off")
    elif view.kind == "distribution":
        axis.hist(_finite(data.get("sample", [])), bins=40, color="#4b78c2")
        axis.set(xlabel="Value", ylabel="Count")
        if not _finite(data.get("sample", [])):
            axis.text(0.5, 0.5, "No finite values to plot", transform=axis.transAxes, ha="center")
    elif view.kind in {"training", "history_comparison"}:
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
                    linestyle="--" if series.get("role") == "validation" else "-",
                )
            axis.set(title=group.get("name"), xlabel="Epoch", ylabel=group.get("name"))
            axis.legend()
        return _finish_matplotlib(figure, view)
    elif view.kind in {"model", "activations", "gradients"}:
        plt.close(figure)
        if view.kind == "gradients":
            rows = data.get("layers", [])[:100]
            figure, axes = plt.subplots(
                3,
                1,
                figsize=(12, 7 + 0.3 * min(len(rows), 100)),
                gridspec_kw={"height_ratios": [3, 3, max(2, len(rows) * 0.3)]},
            )
            figure.suptitle(view.title)
            magnitude_rows = [row for row in rows[:20] if row.get("max_abs") is not None]
            axes[0].bar(
                [row.get("name", "") for row in magnitude_rows],
                [row["max_abs"] for row in magnitude_rows],
                color="#4b78c2",
            )
            axes[0].set(ylabel="Maximum |gradient|", title="Layer-wise magnitude")
            for index, row in enumerate(rows[:20]):
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
                xticks=range(min(len(rows), 20)),
                xticklabels=[row.get("name", "") for row in rows[:20]],
                xlabel="Parameter",
                ylabel="Gradient",
                title="Sampled gradient values",
            )
            axes[2].axis("off")
            _mpl_table(
                axes[2],
                _stats_rows(data),
                ["Layer", "Shape", "Status", "Mean", "Std", "Max |value|", "Finite"],
            )
            for axis in axes[:2]:
                axis.tick_params(axis="x", labelrotation=45)
            return _finish_matplotlib(figure, view)
        figure, axis = plt.subplots(
            figsize=(12, 0.8 + 0.34 * max(1, min(len(data.get("layers", [])), 100)))
        )
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
            columns = ["Layer", "Shape", "Status", "Mean", "Std", "Max |value|", "Finite"]
            rows = _stats_rows(data)
        _mpl_table(axis, rows, columns)
        return _finish_matplotlib(figure, view)
    elif view.kind == "classification_predictions":
        ranked = data.get("ranked", [])
        classes = [r.get("class", "") for r in reversed(ranked)]
        probs = [r.get("probability", 0.0) for r in reversed(ranked)]
        axis.barh(classes, probs, color="#4b78c2")
        axis.set_xlabel("Probability")
        return _finish_matplotlib(figure, view)
    elif view.kind == "prediction_comparison":
        rows = data.get("rows", [])[:10]
        cls_names = [r.get("class", "") for r in rows]
        sa = [r.get("score_a", 0.0) for r in rows]
        sb = [r.get("score_b", 0.0) for r in rows]
        indices = list(range(len(cls_names)))
        w = 0.35
        axis.bar(
            [i - w / 2 for i in indices],
            sa,
            w,
            label=data.get("name_a", "Model A"),
            color="#4b78c2",
        )
        axis.bar(
            [i + w / 2 for i in indices],
            sb,
            w,
            label=data.get("name_b", "Model B"),
            color="#ef8734",
        )
        axis.set_xticks(indices)
        axis.set_xticklabels(cls_names, rotation=35, ha="right")
        axis.legend()
        return _finish_matplotlib(figure, view)
    elif view.kind == "regression_predictions":
        preds = data.get("predictions", [])
        targets = data.get("targets")
        if targets:
            axis.scatter(targets, preds, color="#4b78c2", alpha=0.7)
            axis.set(xlabel="Target", ylabel="Prediction")
        else:
            axis.plot(preds, marker="o", markersize=3, color="#4b78c2")
            axis.set(xlabel="Sample Index", ylabel="Prediction")
        return _finish_matplotlib(figure, view)
    elif view.kind == "model_comparison":
        plt.close(figure)
        rows = data.get("layer_rows", [])[:50]
        figure, axis = plt.subplots(figsize=(12, 0.8 + 0.34 * max(1, min(len(rows), 50))))
        axis.axis("off")
        cols = [
            "Layer",
            f"{data.get('name_a', 'Model A')} Type",
            f"{data.get('name_b', 'Model B')} Type",
            f"{data.get('name_a', 'Model A')} Params",
            f"{data.get('name_b', 'Model B')} Params",
        ]
        row_vals = [
            [
                r.get("layer", ""),
                r.get("type_a", "—"),
                r.get("type_b", "—"),
                str(r.get("params_a", "—")),
                str(r.get("params_b", "—")),
            ]
            for r in rows
        ]
        _mpl_table(axis, row_vals, cols)
        return _finish_matplotlib(figure, view)
    elif view.kind in {"attention", "educational_attention"}:
        mat = data.get("matrix", [])
        toks = data.get("tokens", [])
        im = axis.imshow(mat, cmap="Blues", aspect="auto")
        figure.colorbar(im, ax=axis, label="Attention Weight")
        if toks:
            axis.set_xticks(range(len(toks)))
            axis.set_xticklabels(toks, rotation=45, ha="right")
            axis.set_yticks(range(len(toks)))
            axis.set_yticklabels(toks)
        axis.set(xlabel="Key Token", ylabel="Query Token")
        return _finish_matplotlib(figure, view)
    elif view.kind == "embeddings":
        coords = data.get("coords_2d")
        labels = data.get("labels", [])
        if coords and len(coords[0]) >= 2:
            x_vals = [c[0] for c in coords]
            y_vals = [c[1] for c in coords]
            axis.scatter(x_vals, y_vals, color="#4b78c2", alpha=0.8)
            for i, lbl in enumerate(labels):
                axis.annotate(lbl, (x_vals[i], y_vals[i]), fontsize=8)
            axis.set(xlabel="Principal Component 1", ylabel="Principal Component 2")
        else:
            im = axis.imshow(data.get("similarity_matrix", []), cmap="viridis", aspect="auto")
            figure.colorbar(im, ax=axis, label="Cosine Similarity")
        return _finish_matplotlib(figure, view)
    elif view.kind in {"educational_activations", "educational_losses"}:
        funcs = data.get("functions", {})
        if not funcs and "mse" in data:
            funcs = {k: data[k] for k in ("mse", "mae", "huber") if k in data}
        x_coords = data.get("x", data.get("errors", []))
        for k, v in funcs.items():
            axis.plot(x_coords, v, label=k)
        axis.legend()
        axis.set(xlabel="Input", ylabel="Output")
        return _finish_matplotlib(figure, view)
    elif view.kind == "educational_gradient_descent":
        traj = data.get("trajectory", [])
        axis.plot([p["step"] for p in traj], [p["loss"] for p in traj], marker="o", color="#4b78c2")
        axis.set(xlabel="Optimization Step", ylabel="Objective Loss")
        return _finish_matplotlib(figure, view)
    elif view.kind == "educational_backpropagation":
        plt.close(figure)
        nodes = data.get("nodes", [])
        figure, axis = plt.subplots(figsize=(10, 0.8 + 0.35 * max(1, len(nodes))))
        axis.axis("off")
        cols = ["Graph Node", "Forward Value", "Local Gradient"]
        row_vals = [
            [
                n.get("name", ""),
                f"{float(n.get('forward', 0.0)):.4g}",
                f"{float(n.get('gradient', 0.0)):.4g}",
            ]
            for n in nodes
        ]
        _mpl_table(axis, row_vals, cols)
        return _finish_matplotlib(figure, view)
    elif view.kind == "dataset_tabular":
        plt.close(figure)
        f_stats = data.get("feature_stats", [])[:30]
        figure, axis = plt.subplots(figsize=(10, 0.8 + 0.35 * max(1, len(f_stats))))
        axis.axis("off")
        cols = ["Feature", "Min", "Max", "Mean"]
        row_vals = [
            [
                s.get("feature", ""),
                f"{s.get('min', 0.0):.4g}",
                f"{s.get('max', 0.0):.4g}",
                f"{s.get('mean', 0.0):.4g}",
            ]
            for s in f_stats
        ]
        _mpl_table(axis, row_vals, cols)
        return _finish_matplotlib(figure, view)
    elif view.kind in {"convolution_simulation", "educational_convolution"}:
        feat_map = data.get("feature_map", data.get("output", []))
        im = axis.imshow(feat_map, cmap="viridis", interpolation="nearest")
        figure.colorbar(im, ax=axis, label="Activation")
        axis.set(xlabel="Output X", ylabel="Output Y")
        return _finish_matplotlib(figure, view)
    elif view.kind == "attribution":
        raw_attr = data.get("attribution", [])
        attr = _reduce_attribution_matrix(raw_attr)
        if attr and isinstance(attr[0], list):
            im = axis.imshow(attr, cmap="hot", interpolation="nearest")
            figure.colorbar(im, ax=axis, label="Attribution")
            axis.set(xlabel="Spatial X", ylabel="Spatial Y")
        else:
            axis.bar(range(len(attr)), attr, color="#4b78c2")
            axis.set(xlabel="Feature Index", ylabel="Attribution")
        return _finish_matplotlib(figure, view)
    elif view.kind == "educational_dropout":
        outs = data.get("train_outputs", [])
        axis.bar(range(len(outs)), outs, color="#4b78c2")
        axis.set(xlabel="Unit Index", ylabel="Activation Value")
        return _finish_matplotlib(figure, view)
    elif view.kind == "educational_batch_norm":
        norm = data.get("normalized", [])
        axis.plot(norm, marker="o", color="#4b78c2")
        axis.set(xlabel="Sample Index", ylabel="Standardized Value")
        return _finish_matplotlib(figure, view)
    else:
        plt.close(figure)
        raise SeeDLError(f"The Matplotlib renderer does not support the {view.kind!r} view.")
    return _finish_matplotlib(figure, view)
