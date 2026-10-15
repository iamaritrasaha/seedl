"""Shape- and index-aligned comparisons of bounded training observations."""

from __future__ import annotations

import html
import math
from typing import Any

from .exceptions import SeeDLError


def _stats(values: Any) -> dict[str, Any]:
    values = [v for v in (values or []) if v is not None]
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "max_abs": max((abs(v) for v in values), default=None),
        "l2": math.hypot(*values) if values else None,
    }


def _aligned(a: Any, b: Any) -> dict[str, Any]:
    if a is None or b is None:
        return {"status": "unavailable", "before": a, "after": b}
    if a["shape"] != b["shape"]:
        return {"status": "shape mismatch", "before_shape": a["shape"], "after_shape": b["shape"]}
    left, right = (
        dict(zip(a["indices"], a["values"], strict=True)),
        dict(zip(b["indices"], b["values"], strict=True)),
    )
    indices = sorted(left.keys() & right.keys())
    before, after = [left[i] for i in indices], [right[i] for i in indices]
    deltas = [
        y - x if x is not None and y is not None else None
        for x, y in zip(before, after, strict=True)
    ]
    return {
        "status": "aligned" if indices else "no shared sampled indices",
        "shape": a["shape"],
        "indices": indices,
        "before": before,
        "after": after,
        "delta": deltas,
        "before_stats": _stats(before),
        "after_stats": _stats(after),
        "delta_stats": _stats(deltas),
    }


def compare_snapshots(
    a: Any, b: Any, names: tuple[str, str]
) -> tuple[str, dict[str, Any], str, list[str]]:
    before, after = a.data, b.data
    rows = []
    for name in dict.fromkeys([*before["parameters"], *after["parameters"]]):
        pa, pb = before["parameters"].get(name), after["parameters"].get(name)
        aligned = _aligned(pa, pb)
        gradients = {
            "before": _stats(pa["gradients"] if pa else None),
            "after": _stats(pb["gradients"] if pb else None),
        }
        rows.append({"name": name, **aligned, "gradients": gradients})
    activations = []
    for layer in dict.fromkeys([*before["activations"], *after["activations"]]):
        va = before["activations"].get(layer, {}).get("activation")
        vb = after["activations"].get(layer, {}).get("activation")
        activations.append(
            {
                "layer": layer,
                "before": va,
                "after": vb,
                "before_stats": _stats(va),
                "after_stats": _stats(vb),
            }
        )
    data = {
        "names": names,
        "parameters": rows,
        "activations": activations,
        "outputs": _aligned(before["output"], after["output"]),
        "steps": [before["step"], after["step"]],
    }
    return (
        "snapshot_comparison",
        data,
        f"Recorded steps {before['step']} → {after['step']} · {len(rows)} parameter tensors",
        [
            "Values/statistics describe deterministic retained samples, not unsampled parameters or a full activation distribution.",
            "Deltas align parameter names, tensor shapes and original flat indices. Output changes can also reflect different training inputs.",
        ],
    )


def render_comparison(view: Any, backend: str) -> Any:
    rows = view.data["parameters"]
    if backend == "plotly":
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Sampled weight changes",
                "Sampled activation distributions",
                "Sampled gradient magnitudes",
                "Captured output changes",
            ),
        )
        for row in rows:
            if row["status"] == "aligned":
                fig.add_trace(
                    go.Scatter(x=row["indices"], y=row["delta"], mode="markers", name=row["name"]),
                    row=1,
                    col=1,
                )
        for activation in view.data["activations"]:
            for side, name in zip(("before", "after"), view.data["names"], strict=True):
                if activation[side] is not None:
                    fig.add_trace(
                        go.Histogram(
                            x=activation[side],
                            opacity=0.55,
                            name=f"{name} layer {activation['layer']}",
                        ),
                        row=1,
                        col=2,
                    )
        for side, name in zip(("before", "after"), view.data["names"], strict=True):
            fig.add_trace(
                go.Bar(
                    x=[r["name"] for r in rows],
                    y=[r["gradients"][side]["max_abs"] for r in rows],
                    name=f"{name} |gradient|max",
                ),
                row=2,
                col=1,
            )
        output = view.data["outputs"]
        if output["status"] == "aligned":
            for side, name in zip(("before", "after"), view.data["names"], strict=True):
                fig.add_trace(
                    go.Scatter(
                        x=output["indices"],
                        y=output[side],
                        mode="lines+markers",
                        name=f"{name} output",
                    ),
                    row=2,
                    col=2,
                )
        fig.update_layout(title_text=html.escape(view.title), height=780, barmode="overlay")
        return fig
    if backend == "matplotlib":
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots()
        for row in rows:
            if row["status"] == "aligned":
                axis.scatter(row["indices"], row["delta"], label=row["name"])
        axis.set(title=view.title, xlabel="Original flat index", ylabel="Sampled parameter change")
        axis.legend()
        return fig
    if backend != "svg":
        raise SeeDLError(f"Unknown backend {backend}.")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{130 + 25 * len(rows)}"><rect width="100%" height="100%" fill="#f8fafc"/><text x="25" y="35" font-family="system-ui" font-size="18">{html.escape(view.title)}</text><text x="25" y="70">Parameter · alignment · sampled delta L2 · max |gradient| before / after</text>'
    ]
    for i, row in enumerate(rows):
        delta = row.get("delta_stats", {}).get("l2")
        ga, gb = row["gradients"]["before"]["max_abs"], row["gradients"]["after"]["max_abs"]
        text = f"{row['name']} · {row['status']} · {delta} · {ga} / {gb}"
        lines.append(
            f'<text x="25" y="{105 + i * 25}" font-family="system-ui" font-size="12">{html.escape(text)}</text>'
        )
    return "".join(lines) + "</svg>"
