"""Optional, explicitly bounded spatial projections and parameter slices."""

from __future__ import annotations

import copy
import math
from typing import Any

from .exceptions import MissingDependencyError, SeeDLError


def embeddings_3d(
    vectors: Any, labels: list[str] | None, categories: list[str] | None, max_vectors: int
) -> tuple[dict[str, Any], list[str]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise MissingDependencyError(
            "3D PCA requires `pip install 'seedl[numpy,plotly]'`."
        ) from exc
    if (
        isinstance(max_vectors, bool)
        or not isinstance(max_vectors, int)
        or not 3 <= max_vectors <= 5000
    ):
        raise ValueError("3D max_vectors must be an integer between 3 and 5000.")
    n = len(vectors)
    if n < 3:
        raise SeeDLError("3D PCA requires at least three vectors and three features.")
    indices = np.linspace(0, n - 1, min(n, max_vectors), dtype=int).tolist()
    if hasattr(vectors, "detach"):
        selected = vectors.detach()[indices].cpu().tolist()
    else:
        selected = [vectors[i] for i in indices]
    try:
        matrix = np.asarray(selected)
        if np.iscomplexobj(matrix):
            raise ValueError("complex values")
        matrix = matrix.astype(float)
    except (TypeError, ValueError) as exc:
        raise SeeDLError("3D embeddings must be a rectangular real numeric matrix.") from exc
    if matrix.ndim != 2 or matrix.shape[1] < 3:
        raise SeeDLError("3D PCA requires at least three vectors and three features.")
    if not np.isfinite(matrix).all():
        raise SeeDLError("The sampled embeddings contain non-finite values.")
    for name, values in (("labels", labels), ("categories", categories)):
        if values is not None and len(values) != n:
            raise ValueError(f"{name} must have one entry per original vector.")
    # Scale before centering to avoid overflow for finite values near float limits.
    scale = float(np.max(np.abs(matrix))) or 1.0
    centered = matrix / scale - (matrix / scale).mean(axis=0)
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    for row in components:
        if row[int(np.argmax(np.abs(row)))] < 0:
            row *= -1
    projected = centered @ components[:3].T
    # Coordinates are in scaled units; the scale is explicit and recoverable.
    variance = singular**2
    ratios = (variance[:3] / variance.sum()).tolist() if variance.sum() > 0 else [0.0, 0.0, 0.0]
    item_categories = (
        [str(categories[i]) for i in indices] if categories else ["Vectors"] * len(indices)
    )
    if len(set(item_categories)) > 32:
        raise SeeDLError("3D embeddings support at most 32 displayed categories.")
    notes = [
        f"PCA fitted to {len(indices)} of {n} evenly spaced vectors; coordinates scaled by {scale:.5g}.",
        f"First three components explain {sum(ratios):.1%} of sampled variance. Projection can distort distances.",
    ]
    if np.linalg.matrix_rank(centered) < 3:
        notes.append(
            "Degenerate data: fewer than three independent components; some coordinates are zero (up to roundoff)."
        )
    return dict(
        dimensions=3,
        num_vectors=n,
        dim=matrix.shape[1],
        coords_3d=projected.tolist(),
        pca_variance_ratio=ratios,
        explained_variance=sum(ratios),
        coordinate_scale=scale,
        labels=[str(labels[i]) if labels else f"V{i}" for i in indices],
        categories=item_categories,
        indices=indices,
        summary=notes[0],
    ), notes


def loss_slice(
    model: Any,
    data: Any,
    loss_fn: Any,
    axes: Any,
    radius: float,
    steps: int,
    evaluate: bool,
    trajectory: Any,
) -> tuple[dict[str, Any], list[str]]:
    try:
        import torch
    except ImportError as exc:
        raise MissingDependencyError(
            "Loss slices require `pip install 'seedl[torch,plotly]'`."
        ) from exc

    from ._models import _clone_val

    if evaluate is not True:
        raise SeeDLError(
            "Loss slices run repeated forward passes. Set evaluate=True to explicitly authorize evaluation."
        )
    if not isinstance(model, torch.nn.Module):
        raise SeeDLError("loss_landscape expects a PyTorch model.")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 3 <= steps <= 31:
        raise ValueError("steps must be an integer between 3 and 31 (at most 961 evaluations).")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive.")
    if not callable(loss_fn):
        raise ValueError("Provide loss_fn(prediction, target) returning one scalar loss.")
    if not isinstance(data, (tuple, list)) or len(data) != 2:
        raise ValueError("data must be (input, target).")
    parameters = dict(model.named_parameters())
    if any(isinstance(p, torch.nn.parameter.UninitializedParameter) for p in parameters.values()):
        raise SeeDLError("Initialize lazy parameters before evaluating a loss slice.")
    if sum(p.numel() for p in parameters.values()) > 250_000:
        raise SeeDLError("Loss slices are limited to 250,000 parameters. Use a smaller model.")
    if any(p.device.type != "cpu" or not p.is_floating_point() for p in parameters.values()):
        raise SeeDLError("Loss slices currently require real floating-point CPU parameters.")
    if not isinstance(axes, (tuple, list)) or len(axes) != 2:
        raise ValueError("axes must contain two (parameter_name, flat_index) coordinates.")
    normalized = []
    for name, index in axes:
        if (
            name not in parameters
            or isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < parameters[name].numel()
        ):
            raise ValueError(
                "Each slice axis must name an existing parameter and a valid flat index."
            )
        normalized.append((name, index))
    if normalized[0] == normalized[1]:
        raise ValueError("Slice axes must be distinct parameter coordinates.")
    if trajectory is not None:
        if len(trajectory) > 100:
            raise ValueError("trajectory is limited to 100 pairs of slice offsets.")
        for a, b in trajectory:
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError("trajectory offsets must be finite.")
    offsets = [-radius + 2 * radius * i / (steps - 1) for i in range(steps)]
    # Deep copy isolates parameters, buffers and ordinary custom forward attributes.
    with torch.random.fork_rng(devices=[]):
        baseline_model = copy.deepcopy(model).eval()
        evaluation_rng = torch.random.get_rng_state()
        origin = [float(parameters[name].detach().reshape(-1)[index]) for name, index in normalized]
        if not all(math.isfinite(v) for v in origin):
            raise SeeDLError("Slice origins must be finite.")

        def evaluate_at(a: float, b: float) -> float:
            # Reset even non-persistent buffers and ordinary Python attributes at each point.
            clone = copy.deepcopy(baseline_model)
            cloned_parameters = dict(clone.named_parameters())
            torch.random.set_rng_state(evaluation_rng)
            with torch.no_grad():
                for (name, index), value, offset in zip(normalized, origin, (a, b), strict=True):
                    parameter = cloned_parameters[name]
                    coords = []
                    rest = index
                    for size in reversed(parameter.shape):
                        coords.append(rest % size)
                        rest //= size
                    parameter[tuple(reversed(coords))] = value + offset
                x, y = _clone_val(data, torch)
                prediction = (
                    clone(**x)
                    if isinstance(x, dict)
                    else clone(*x)
                    if isinstance(x, tuple)
                    else clone(x)
                )
                loss = loss_fn(prediction, y)
                if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                    raise SeeDLError("loss_fn must return one scalar tensor.")
                value = float(loss.item())
                if not math.isfinite(value):
                    raise SeeDLError("Non-finite loss encountered in the sampled slice.")
                return value

        losses = [[evaluate_at(a, b) for a in offsets] for b in offsets]
        trail = []
        if trajectory is not None:
            for a, b in trajectory:
                trail.append([float(a), float(b), evaluate_at(float(a), float(b))])
    notes = [
        f"A sampled two-dimensional coordinate slice, not the full loss landscape: {steps} × {steps} forward evaluations.",
        "All other parameters held fixed. Eval/no_grad on a deep copy; no backward pass. CPU RNG restored.",
        "Trajectory, when supplied, is evaluated in this same slice; it is not an automatically captured optimizer path.",
    ]
    return dict(
        dimensions=3,
        offsets=offsets,
        losses=losses,
        axes=normalized,
        origin=origin,
        trajectory=trail,
        evaluations=steps * steps + len(trail),
        summary=notes[0],
    ), notes


def render_spatial(view: Any) -> Any:
    import plotly.graph_objects as go

    data = view.data
    fig = go.Figure()
    if view.kind == "embeddings":
        for category in dict.fromkeys(data["categories"]):
            indices = [i for i, c in enumerate(data["categories"]) if c == category]
            points = [data["coords_3d"][i] for i in indices]
            fig.add_trace(
                go.Scatter3d(
                    x=[p[0] for p in points],
                    y=[p[1] for p in points],
                    z=[p[2] for p in points],
                    mode="markers",
                    name=category,
                    text=[data["labels"][i] for i in indices],
                    marker=dict(size=5, opacity=0.85),
                    hovertemplate="%{text}<br>PC1 %{x:.3g}<br>PC2 %{y:.3g}<br>PC3 %{z:.3g}<extra>%{fullData.name}</extra>",
                )
            )
        names = [f"PC{i + 1} ({r:.1%})" for i, r in enumerate(data["pca_variance_ratio"])]
    else:
        fig.add_trace(
            go.Surface(
                x=data["offsets"],
                y=data["offsets"],
                z=data["losses"],
                colorscale="Teal",
                colorbar=dict(title="Loss"),
                hovertemplate="Δ axis 1: %{x:.3g}<br>Δ axis 2: %{y:.3g}<br>Loss: %{z:.5g}<extra></extra>",
            )
        )
        if data["trajectory"]:
            fig.add_trace(
                go.Scatter3d(
                    x=[p[0] for p in data["trajectory"]],
                    y=[p[1] for p in data["trajectory"]],
                    z=[p[2] for p in data["trajectory"]],
                    mode="lines+markers",
                    name="Supplied slice trajectory",
                    line=dict(color="#e17463", width=5),
                )
            )
        names = [f"Δ {name}[{index}]" for name, index in data["axes"]] + ["Loss"]
    fig.update_layout(
        template="plotly_white",
        height=650,
        title=dict(text=f"<b>{view.title}</b><br><sup>{data['summary']}</sup>", x=0.03),
        margin=dict(l=20, r=30, t=90, b=20),
        scene=dict(
            xaxis_title=names[0],
            yaxis_title=names[1],
            zaxis_title=names[2],
            dragmode="orbit",
            camera=dict(eye=dict(x=1.5, y=-1.8, z=1.2)),
        ),
    )
    return fig
