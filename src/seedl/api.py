"""Public SeeDL operations and entry points."""

from __future__ import annotations

from typing import Any

from ._inspect import (
    inspect_object,
    is_torch_module,
)
from ._training import Tracker
from .exceptions import SeeDLError
from .renderers import resolve_backend
from .view import Visualization


def _validate_max_points(max_points: int) -> None:
    if isinstance(max_points, bool) or not isinstance(max_points, int) or max_points < 1:
        raise ValueError("max_points must be a positive integer.")


def _summary(kind: str, data: dict[str, Any], metadata: dict[str, Any]) -> str:
    if kind == "scalar":
        return str(data.get("caption", metadata.get("Type", "scalar")))
    if kind == "sequence":
        values = data.get("values", [])
        return f"{len(values):,} value(s)"
    if kind == "matrix":
        shape = data.get("shape", (len(data.get("values", [])), 0))
        return f"Matrix with shape {shape}"
    if kind in {"image", "dataset_images"}:
        return f"Image tensor with shape {data.get('shape')}"
    if kind == "distribution":
        return f"Sampled value distribution · shape {data.get('shape', 'unknown')}"
    if kind in {"model", "training", "history_comparison", "model_comparison"}:
        return str(data.get("summary", metadata.get("Summary", "")))
    if kind == "activations":
        return f"{len(data.get('layers', []))} layer output(s)"
    if kind == "gradients":
        return f"{len(data.get('layers', []))} parameter tensor(s)"
    if kind == "classification_predictions":
        return f"Top prediction: '{data.get('top_class')}' ({data.get('top_prob', 0):.1%})"
    if kind == "regression_predictions":
        return f"Regression: {data.get('count', len(data.get('predictions', [])))} prediction(s)"
    if kind == "prediction_comparison":
        return f"Prediction comparison: {data.get('name_a')} vs {data.get('name_b')}"
    if kind == "dataset_tabular":
        return f"Tabular dataset: {data.get('samples')} samples × {data.get('features')} features"
    if kind == "dataset_general":
        return f"Dataset with {data.get('count')} samples"
    if kind in {"attention", "educational_attention"}:
        return f"Attention matrix ({data.get('seq_len')} tokens)"
    if kind == "embeddings":
        return f"Embeddings: {data.get('num_vectors')} vectors in {data.get('dim')}D"
    if kind.startswith("educational_"):
        return str(data.get("concept", kind))
    return str(data.get("summary", metadata.get("Summary", "")))


def _show_result(view: Visualization) -> None:
    """Display in a terminal fallback; in notebooks, the return value renders automatically."""
    try:
        from IPython import get_ipython  # type: ignore[import-not-found]

        if get_ipython() is not None:
            # In IPython/Jupyter, Visualization instances render automatically when returned
            # as the final expression in a cell. Suppress explicit display() to avoid duplicates.
            return
    except ImportError:
        pass
    print(view)


def see(
    obj: Any,
    *,
    input: Any | None = None,
    slice: Any | None = None,
    detail: str = "summary",
    compare: Any | None = None,
    backend: str = "auto",
    interactive: bool | None = None,
    max_points: int = 2_048,
) -> Visualization:
    """Automatically visualize a supported deep-learning or numeric object.

    Supported objects include PyTorch modules and tensors, NumPy arrays, real
    numbers, numeric sequences, metric-history dictionaries, dataset batches,
    and prediction scores. Returns a :class:`Visualization`.

    Parameters:
        obj: The object to inspect and visualize.
        input: Used for PyTorch model inspection. The model is called once with that
            input under eval mode and without gradients; temporary hooks and buffer
            updates are restored.
        slice: Optional slice specification for higher-dimensional tensors.
        detail: Detail level for model hierarchy ('summary' or 'full'/'leaves').
        compare: Optional second object to compare against (e.g. model or history).
        backend: 'auto' (Plotly when installed, otherwise SVG), 'plotly', 'matplotlib', or 'svg'.
        interactive: True (forces Plotly), False (forces SVG if auto), or None.
        max_points: Bounded sampling limit for large arrays and sequences.
    """
    _validate_max_points(max_points)
    if interactive is not None and not isinstance(interactive, bool):
        raise TypeError("interactive must be True, False, or None.")
    if interactive is True:
        if backend == "auto":
            backend = "plotly"
        elif backend != "plotly":
            raise ValueError("interactive=True requires backend='plotly' or backend='auto'.")
    elif interactive is False and backend == "auto":
        backend = "svg"
    selected_backend = resolve_backend(backend)

    # Comparison shorthand: sd.see(model_a, compare=model_b)
    if compare is not None:
        from ._compare import compare_objects

        kind, data, summary_text, notes = compare_objects(obj, compare, max_points=max_points)
        view = Visualization(
            kind=kind,
            title="Comparison",
            summary=summary_text,
            data=data,
            notes=notes,
            backend=selected_backend,
        )
        _show_result(view)
        return view

    # Handle tracker instances
    if isinstance(obj, Tracker):
        obj = obj.history()

    if input is not None and not is_torch_module(obj):
        raise SeeDLError("The input= argument is only used when visualizing a PyTorch nn.Module.")

    # Slice support for tensors/arrays
    slice_note = None
    if slice is not None:
        try:
            orig_shape = getattr(obj, "shape", None)
            obj = obj[slice]
            slice_note = f"Showing slice {slice} of original shape {orig_shape}."
        except Exception as exc:
            raise SeeDLError(f"Failed to slice object with {slice}: {exc}") from exc

    kind, title, data, metadata, notes = inspect_object(
        obj, example_input=input, max_points=max_points
    )
    if slice_note:
        notes.insert(0, slice_note)

    view = Visualization(
        kind=kind,
        title=title,
        summary=_summary(kind, data, metadata),
        data=data,
        metadata=metadata,
        notes=notes,
        backend=selected_backend,
    )
    _show_result(view)
    return view


def activations(
    model: Any,
    *,
    input: Any,
    layers: str | list[str] = "all",
    backend: str = "auto",
    max_points: int = 2_048,
) -> Visualization:
    """Capture layer outputs and feature maps from one input-driven forward pass."""
    _validate_max_points(max_points)
    selected_backend = resolve_backend(backend)
    from ._activations import capture_activations

    data, summary, notes = capture_activations(model, input, layers=layers, max_points=max_points)
    view = Visualization(
        kind="activations",
        title=f"{type(model).__name__} activations",
        summary=summary,
        data=data,
        notes=notes,
        backend=selected_backend,
    )
    _show_result(view)
    return view


def gradients(
    model: Any,
    *,
    backend: str = "auto",
    max_points: int = 2_048,
) -> Visualization:
    """Visualize gradients that already exist on a PyTorch model's parameters.

    Observational and read-only. Never calls backward() or modifies weights.
    """
    _validate_max_points(max_points)
    selected_backend = resolve_backend(backend)
    from ._gradients import inspect_gradients

    data, summary, notes = inspect_gradients(model, max_points=max_points)
    view = Visualization(
        kind="gradients",
        title=f"{type(model).__name__} gradients",
        summary=summary,
        data=data,
        notes=notes,
        backend=selected_backend,
    )
    _show_result(view)
    return view


def compare(
    obj_a: Any,
    obj_b: Any,
    *,
    names: tuple[str, str] = ("Model A", "Model B"),
    backend: str = "auto",
    max_points: int = 2_048,
) -> Visualization:
    """Compare two compatible deep learning objects: models, training histories, or predictions."""
    _validate_max_points(max_points)
    selected_backend = resolve_backend(backend)
    from ._compare import compare_objects

    kind, data, summary, notes = compare_objects(obj_a, obj_b, names=names, max_points=max_points)
    view = Visualization(
        kind=kind,
        title=f"Comparison: {names[0]} vs {names[1]}",
        summary=summary,
        data=data,
        notes=notes,
        backend=selected_backend,
    )
    _show_result(view)
    return view


def explain(
    model: Any,
    *,
    input: Any,
    target: int | None = None,
    method: str = "saliency",
    steps: int = 20,
    baseline: Any | None = None,
    layer: str | None = None,
    abs: bool | None = None,
    backend: str = "auto",
) -> Visualization:
    """Generate input feature attributions using saliency, integrated gradients, or Grad-CAM.

    Methods:
    - 'saliency': Vanilla gradient saliency with respect to the input tensor.
    - 'integrated_gradients': Path-integrated gradients along a straight line from baseline
      using the composite midpoint integration rule (open Newton-Cotes formula with uniform weights,
      distinct from Gauss-Legendre quadrature which uses non-uniform polynomial roots).
      Attributions are signed by default to satisfy the Completeness axiom:
      sum(attribution) ≈ F(input) - F(baseline).
    - 'grad_cam': Coarse spatial activation mapping for convolutional networks.

    Parameters:
    - model: PyTorch nn.Module.
    - input: Input tensor.
    - target: Target class index (defaults to argmax predicted class).
    - method: Attribution method ('saliency', 'integrated_gradients', 'grad_cam').
    - steps: Number of interpolation steps for composite midpoint integration (default: 20).
    - baseline: Reference input tensor for integrated gradients (default: zeros).
    - layer: Target layer name or class for Grad-CAM (default: last conv layer).
    - abs: Whether to take absolute value of attributions (default: False for IG to preserve
      completeness, True for saliency).
    - backend: Visualization backend ('auto', 'svg', 'plotly', 'matplotlib').
    """
    selected_backend = resolve_backend(backend)
    from ._explain import explain_prediction

    data, summary, notes = explain_prediction(
        model,
        input,
        target=target,
        method=method,
        steps=steps,
        baseline=baseline,
        layer=layer,
        abs=abs,
    )
    view = Visualization(
        kind="attribution",
        title=f"{type(model).__name__} Attribution ({method})",
        summary=summary,
        data=data,
        notes=notes,
        backend=selected_backend,
    )
    _show_result(view)
    return view


def predictions(
    predictions: Any,
    *,
    targets: Any | None = None,
    labels: list[str] | None = None,
    task: str = "auto",
    backend: str = "auto",
    max_points: int = 2_048,
) -> Visualization:
    """Visualize classification scores, ranked classes, or regression residuals."""
    _validate_max_points(max_points)
    selected_backend = resolve_backend(backend)
    from ._predictions import inspect_predictions

    kind, data, summary, notes = inspect_predictions(
        predictions, targets=targets, labels=labels, task=task, max_points=max_points
    )
    view = Visualization(
        kind=kind,
        title="Prediction Analysis",
        summary=summary,
        data=data,
        notes=notes,
        backend=selected_backend,
    )
    _show_result(view)
    return view


def compare_predictions(
    preds_a: Any,
    preds_b: Any,
    *,
    labels: list[str] | None = None,
    names: tuple[str, str] = ("Model A", "Model B"),
    backend: str = "auto",
) -> Visualization:
    """Compare model predictions side by side on sample or batch."""
    selected_backend = resolve_backend(backend)
    from ._predictions import compare_predictions as comp_preds

    kind, data, summary, notes = comp_preds(preds_a, preds_b, labels=labels, names=names)
    view = Visualization(
        kind=kind,
        title=f"Prediction Comparison: {names[0]} vs {names[1]}",
        summary=summary,
        data=data,
        notes=notes,
        backend=selected_backend,
    )
    _show_result(view)
    return view


def dataset(
    data: Any,
    *,
    labels: Any | None = None,
    max_samples: int = 64,
    backend: str = "auto",
) -> Visualization:
    """Inspect a dataset, DataLoader batch, or sample collection safely with bounded sampling."""
    selected_backend = resolve_backend(backend)
    from ._datasets import inspect_dataset

    kind, data_payload, summary, notes = inspect_dataset(
        data, labels=labels, max_samples=max_samples
    )
    view = Visualization(
        kind=kind,
        title="Dataset Inspection",
        summary=summary,
        data=data_payload,
        notes=notes,
        backend=selected_backend,
    )
    _show_result(view)
    return view


def learn(
    concept: str,
    *,
    backend: str = "auto",
    **kwargs: Any,
) -> Visualization:
    """Interactive, mathematically verified educational demonstration of a core deep learning concept."""
    selected_backend = resolve_backend(backend)
    from ._educational import learn_concept

    kind, data, summary, notes = learn_concept(concept, **kwargs)
    view = Visualization(
        kind=kind,
        title=f"Learning: {concept.replace('_', ' ').title()}",
        summary=summary,
        data=data,
        notes=notes,
        backend=selected_backend,
    )
    _show_result(view)
    return view


def embeddings(
    vectors: Any,
    *,
    labels: list[str] | None = None,
    method: str = "pca",
    backend: str = "auto",
    max_vectors: int = 128,
) -> Visualization:
    """Visualize vector embeddings, pairwise cosine similarity matrix, and 2D PCA projection."""
    selected_backend = resolve_backend(backend)
    from ._embeddings import inspect_embeddings

    data, summary, notes = inspect_embeddings(
        vectors, labels=labels, method=method, max_vectors=max_vectors
    )
    view = Visualization(
        kind="embeddings",
        title="Vector Embeddings",
        summary=summary,
        data=data,
        notes=notes,
        backend=selected_backend,
    )
    _show_result(view)
    return view


def watch(model: Any | None = None, name: str = "run") -> Tracker:
    """Create a lightweight metric and gradient tracker."""
    return Tracker(name=name)


def track(model: Any | None = None, name: str = "run") -> Tracker:
    """Context manager for tracking training metrics across epochs or steps."""
    return Tracker(name=name)


def attention(
    weights: Any,
    *,
    tokens: list[str] | None = None,
    head: int | None = None,
    backend: str = "auto",
) -> Visualization:
    """Visualize multi-head self-attention heatmaps, query-key tokens, and causal structure."""
    selected_backend = resolve_backend(backend)
    from ._transformers import inspect_attention

    data, summary, notes = inspect_attention(weights, tokens=tokens, head=head)
    view = Visualization(
        kind="attention",
        title="Transformer Attention",
        summary=summary,
        data=data,
        notes=notes,
        backend=selected_backend,
    )
    _show_result(view)
    return view


def convolution(
    image: Any | None = None,
    kernel: Any | None = None,
    *,
    stride: int = 1,
    padding: int = 0,
    backend: str = "auto",
) -> Visualization:
    """Simulate and visualize 2D convolution operation, receptive field, and feature map arithmetic."""
    selected_backend = resolve_backend(backend)
    from ._vision import inspect_convolution

    data, summary, notes = inspect_convolution(
        image=image, kernel=kernel, stride=stride, padding=padding
    )
    view = Visualization(
        kind="convolution_simulation",
        title="2D Convolution Simulation",
        summary=summary,
        data=data,
        notes=notes,
        backend=selected_backend,
    )
    _show_result(view)
    return view
