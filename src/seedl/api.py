"""Public SeeDL operations and entry points."""

from __future__ import annotations

from typing import Any

from ._inspect import (
    inspect_object,
    is_numpy_array,
    is_torch_module,
    is_torch_tensor,
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
    if kind in {"feature_maps", "feature_comparison"}:
        return str(
            data.get("summary", metadata.get("Summary", f"{len(data.get('layers', []))} layer(s)"))
        )
    if kind in {"attention", "attention_comparison", "educational_attention"}:
        return f"Attention matrix ({data.get('seq_len')} tokens)"
    if kind == "embeddings":
        return f"Embeddings: {data.get('num_vectors')} vectors in {data.get('dim')}D"
    if kind == "execution_trace":
        nodes = data.get("nodes", [])
        return (
            f"Execution trace: {len(nodes)} operations ({data.get('total_parameters', 0):,} params)"
        )
    if kind == "attention_rollout":
        return f"Attention rollout across {data.get('num_layers')} layers ({data.get('seq_len')} tokens)"
    if kind == "attribution_comparison":
        return f"Attribution comparison: {len(data.get('methods', []))} method(s)"
    if kind == "gradient_health":
        return f"Gradient health: {data.get('total_parameters_inspected')} parameters inspected"
    if kind == "quantum_noise":
        return f"Quantum noise: {data.get('channel')} channel (p={data.get('param', 0):.2f})"
    if kind == "dataset_drift":
        return f"Dataset drift: PSI={data.get('psi', 0):.4f} ({data.get('drift_level', 'Unknown')})"
    if kind == "dataset_health":
        return f"Dataset health: {data.get('sample_count')} samples inspected"
    if kind == "run_comparison":
        return f"Run comparison: {data.get('name_a')} vs {data.get('name_b')}"
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
    layout: str | None = None,
    detail: str = "summary",
    compare: Any | None = None,
    backend: str = "auto",
    interactive: bool | None = None,
    max_points: int = 2_048,
    view: str = "auto",
    theme: str = "light",
    color_by: str = "auto",
    sample: int = 0,
    max_neurons: int = 100,
    max_connections: int = 1_500,
    min_weight: float = 0.0,
    edge_opacity: float = 0.45,
    select_neuron: tuple[int, int] | str | None = None,
    simulate: bool = False,
) -> Visualization:
    """Automatically visualize a supported deep-learning or numeric object.

    Supported objects include PyTorch modules and tensors, NumPy arrays, real
    numbers, numeric sequences, metric-history dictionaries, dataset batches,
    prediction scores, and quantum circuits/states. Returns a :class:`Visualization`.

    Parameters:
        obj: The object to inspect and visualize.
        input: Used for PyTorch model inspection. The model is called once with that
            input under eval mode and without gradients; temporary hooks and buffer
            updates are restored.
        slice: Optional slice specification for higher-dimensional tensors. Accepts
            integers, slice objects, tuples, or string syntax like '0, :, :'.
        layout: Optional explicit channel layout for 3D/4D tensors ('CHW', 'HWC', 'NCHW', 'NHWC').
        detail: Detail level for model hierarchy ('summary' or 'full'/'leaves').
        compare: Optional second object to compare against (e.g. model or history).
        backend: 'auto' (Plotly when installed, otherwise SVG), 'plotly', 'matplotlib', or 'svg'.
        interactive: True (forces Plotly), False (forces SVG if auto), or None.
        max_points: Bounded sampling limit for large arrays and sequences.
        view: Model display: auto, neurons, 3d (requires Plotly), architecture, summary.
        theme: Model graph theme: scientific, dark, light, minimal, or presentation.
        color_by: auto (activations when captured), layer, activation, or bias gradient.
        sample: Zero-based batch sample for neuron activations; accepts 1D or 2D input.
        max_neurons: Total neuron limit before architecture fallback (default 100).
        max_connections: Deterministic connection preview budget (default 1500).
        min_weight: Hide sampled connections below this absolute weight.
        edge_opacity: Connection opacity between zero and one.
        select_neuron: Highlight and inspect a specific neuron as (layer, index) or 'layer:index'.
        simulate: For quantum circuits, compute step-by-step statevector evolution.
    """
    _validate_max_points(max_points)
    if view not in {
        "auto",
        "neurons",
        "3d",
        "architecture",
        "summary",
        "features",
        "feature_maps",
        "flow",
        "trace",
    }:
        raise ValueError(
            "view must be auto, neurons, 3d, architecture, summary, features, feature_maps, flow, or trace."
        )
    if theme not in {"light", "dark", "scientific", "minimal", "presentation"}:
        raise ValueError("theme must be light, dark, scientific, minimal, or presentation.")
    if color_by not in {"auto", "layer", "activation", "gradient"}:
        raise ValueError("color_by must be auto, layer, activation, or gradient.")
    if detail not in {"summary", "full", "leaves"}:
        raise ValueError("detail must be summary, full, or leaves.")
    for name, value, upper in (
        ("max_neurons", max_neurons, 1000),
        ("max_connections", max_connections, 20000),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
            raise ValueError(f"{name} must be an integer between 1 and {upper}.")
    if isinstance(sample, bool) or not isinstance(sample, int) or sample < 0:
        raise ValueError("sample must be a nonnegative integer.")
    import math

    if not math.isfinite(min_weight) or min_weight < 0:
        raise ValueError("min_weight must be finite and nonnegative.")
    if not math.isfinite(edge_opacity) or not 0 <= edge_opacity <= 1:
        raise ValueError("edge_opacity must be between 0 and 1.")
    from ._execution_graph import ExecutionTrace

    if isinstance(obj, ExecutionTrace):
        return flow(obj, backend=backend)
    if view in {"flow", "trace"}:
        return flow(obj, input=input, backend=backend)

    from ._quantum import _is_qiskit_circuit, _is_qiskit_state

    if _is_qiskit_circuit(obj):
        return circuit(obj, simulate=simulate, theme=theme, backend=backend)
    if _is_qiskit_state(obj):
        data_arr = getattr(obj, "data", obj)
        dim = len(data_arr) if hasattr(data_arr, "__len__") else 0
        if dim == 2:
            return bloch(
                obj,
                view=view if view in {"3d", "auto"} else "3d",
                theme=theme,
                backend=backend,
            )
        return statevector(obj, theme=theme, backend=backend)
    if isinstance(obj, str) and obj.strip().lower() in {
        "0",
        "1",
        "+",
        "-",
        "+i",
        "-i",
        "|0>",
        "|1>",
        "|+>",
        "|->",
        "|+i>",
        "|-i>",
        "zero",
        "one",
        "plus",
        "minus",
    }:
        return bloch(
            obj,
            view=view if view in {"3d", "auto"} else "3d",
            theme=theme,
            backend=backend,
        )

    if (
        view != "auto"
        and view not in {"features", "feature_maps"}
        and (not is_torch_module(obj) or compare is not None)
    ):
        raise ValueError("Explicit view options require a model without compare=.")
    if view == "3d":
        if backend not in {"auto", "plotly"} or interactive is False:
            raise ValueError("view='3d' requires Plotly and interactive rendering.")
        backend = "plotly"
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

    if view in {"features", "feature_maps"}:
        return features(
            obj,
            input=input,
            batch_index=sample,
            theme=theme,
            backend=selected_backend,
        )

    # Comparison shorthand: sd.see(model_a, compare=model_b)
    if compare is not None:
        from ._compare import compare_objects

        kind, data, summary_text, notes = compare_objects(obj, compare, max_points=max_points)
        result = Visualization(
            kind=kind,
            title="Comparison",
            summary=summary_text,
            data=data,
            notes=notes,
            backend=selected_backend,
        )
        _show_result(result)
        return result

    # Handle tracker instances
    if isinstance(obj, Tracker):
        if obj.snapshots:
            if input is not None or slice is not None:
                raise SeeDLError("Recorded playback does not accept input= or slice=.")
            result = obj.playback(
                backend=selected_backend,
                color_by="gradient" if color_by == "gradient" else "activation",
            )
            _show_result(result)
            return result
        obj = obj.history()

    if input is not None and not is_torch_module(obj):
        raise SeeDLError("The input= argument is only used when visualizing a PyTorch nn.Module.")

    if is_torch_module(obj):
        if slice is not None:
            raise SeeDLError(
                "The slice= argument is only supported for PyTorch tensors and NumPy arrays, not nn.Module models."
            )
        if layout is not None:
            raise SeeDLError(
                "The layout= argument is only supported for PyTorch tensors and NumPy arrays, not nn.Module models."
            )
    elif layout is not None and not (is_torch_tensor(obj) or is_numpy_array(obj)):
        raise SeeDLError(
            f"The layout= argument is only supported for PyTorch tensors and NumPy arrays, got {type(obj).__name__}."
        )

    # Slice support for tensors/arrays
    slice_note = None
    orig_shape = getattr(obj, "shape", None)
    if slice is not None:
        try:
            from ._tensors import parse_slice_spec

            shape_tuple = tuple(int(s) for s in orig_shape) if orig_shape is not None else None
            parsed_slice = parse_slice_spec(slice, shape_tuple)
            obj = obj[parsed_slice]
            slice_note = f"Showing slice {slice} of original shape {orig_shape}."
        except SeeDLError:
            raise
        except Exception as exc:
            raise SeeDLError(f"Failed to slice object with {slice}: {exc}") from exc

    if is_torch_module(obj):
        if view == "3d":
            from ._cnn_spatial import inspect_visual_cnn, is_cnn_model

            if is_cnn_model(obj):
                cnn_data, cnn_sum, cnn_notes = inspect_visual_cnn(
                    obj, input, theme=theme, sample=sample
                )
                cnn_res = Visualization(
                    kind="model",
                    title=f"{type(obj).__name__} · 3D CNN",
                    summary=cnn_sum,
                    data=cnn_data,
                    notes=cnn_notes,
                    backend=selected_backend,
                )
                _show_result(cnn_res)
                return cnn_res

        from ._neurons import inspect_visual_model

        data, metadata, notes = inspect_visual_model(
            obj,
            input,
            view=view,
            theme=theme,
            color_by=color_by,
            sample=sample,
            max_neurons=max_neurons,
            max_connections=max_connections,
            min_weight=min_weight,
            edge_opacity=edge_opacity,
            detail=detail,
            select_neuron=select_neuron,
        )
        kind, title = "model", f"{type(obj).__name__} architecture"
        if data.get("neuron_graph"):
            title = "Neural network" + (" · 3D" if view == "3d" else "")
    else:
        kind, title, data, metadata, notes = inspect_object(
            obj, example_input=input, max_points=max_points, layout=layout
        )
    if slice_note:
        notes.insert(0, slice_note)
        if orig_shape is not None:
            metadata["Original shape"] = str(orig_shape)
            data["slice"] = str(slice)
            data["original_shape"] = tuple(int(s) for s in orig_shape)

    result = Visualization(
        kind=kind,
        title=title,
        summary=_summary(kind, data, metadata),
        data=data,
        metadata=metadata,
        notes=notes,
        backend=selected_backend,
    )
    _show_result(result)
    return result


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
    samples: int = 25,
    stdev_spread: float = 0.15,
    seed: int | None = None,
    backend: str = "auto",
) -> Visualization:
    """Generate input feature attributions using saliency, integrated gradients, Grad-CAM, or SmoothGrad.

    Methods:
    - 'saliency': Vanilla gradient saliency with respect to the input tensor.
    - 'integrated_gradients': Path-integrated gradients along a straight line from baseline
      using the composite midpoint integration rule (open Newton-Cotes formula with uniform weights,
      distinct from Gauss-Legendre quadrature which uses non-uniform polynomial roots).
      Attributions are signed by default to satisfy the Completeness axiom:
      sum(attribution) ≈ F(input) - F(baseline).
    - 'grad_cam': Coarse spatial activation mapping for convolutional networks.
    - 'smoothgrad': Averages gradient saliency over Gaussian noise perturbed inputs.

    Parameters:
    - model: PyTorch nn.Module.
    - input: Input tensor.
    - target: Target class index (defaults to argmax predicted class).
    - method: Attribution method ('saliency', 'integrated_gradients', 'grad_cam', 'smoothgrad').
    - steps: Number of interpolation steps for composite midpoint integration (default: 20).
    - baseline: Reference input tensor for integrated gradients (default: zeros).
    - layer: Target layer name or class for Grad-CAM (default: last conv layer).
    - abs: Whether to take absolute value of attributions (default: False for IG to preserve
      completeness, True for saliency and SmoothGrad).
    - samples: Number of noisy perturbations for SmoothGrad (default: 25).
    - stdev_spread: Noise standard deviation fraction relative to input range for SmoothGrad (default: 0.15).
    - seed: Optional deterministic PRNG seed for SmoothGrad perturbations.
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
        samples=samples,
        stdev_spread=stdev_spread,
        seed=seed,
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
    dimensions: int = 2,
    categories: list[str] | None = None,
) -> Visualization:
    """Visualize vector embeddings, pairwise cosine similarity matrix, and 2D PCA projection."""
    if dimensions not in {2, 3} or isinstance(dimensions, bool):
        raise ValueError("dimensions must be 2 or 3.")
    if dimensions == 3:
        if backend not in {"auto", "plotly"}:
            raise ValueError("3D embeddings require the Plotly backend.")
        selected_backend = resolve_backend("plotly")
        if method != "pca":
            raise ValueError("3D embeddings support method='pca'.")
        from ._spatial import embeddings_3d

        data, notes = embeddings_3d(vectors, labels, categories, max_vectors)
        result = Visualization(
            kind="embeddings",
            title="Embeddings · 3D PCA",
            summary=data["summary"],
            data=data,
            notes=notes,
            backend=selected_backend,
        )
        _show_result(result)
        return result
    if categories is not None:
        raise ValueError("categories currently requires dimensions=3.")
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


def watch(model: Any | None = None, name: str = "run", **capture_options: Any) -> Tracker:
    """Create a metric tracker; with a model, also record bounded training snapshots.

    Capture options: max_snapshots=64, max_bytes=8 MiB, every=1,
    max_values=2048, max_neurons=100, max_connections=1500, sample=0.
    Use as a context manager and call capture() before optimizer.step().
    """
    return Tracker(name=name, model=model, **capture_options)


def track(model: Any | None = None, name: str = "run", **capture_options: Any) -> Tracker:
    """Alias of watch(); existing metric-only tracking remains supported."""
    return watch(model=model, name=name, **capture_options)


def attention(
    weights: Any,
    *,
    input: Any = None,
    tokens: list[str] | None = None,
    query_tokens: list[str] | None = None,
    key_tokens: list[str] | None = None,
    head: int | None = None,
    compare_heads: tuple[int, int] | None = None,
    batch_index: int = 0,
    is_batched: bool | None = None,
    mask: Any = None,
    max_seq_len: int = 256,
    max_heads: int = 32,
    theme: str = "scientific",
    backend: str = "auto",
) -> Visualization:
    """Visualize multi-head self-attention heatmaps, query-key tokens, and causal structure.

    Parameters:
        weights: PyTorch tensor [heads, L, S], [B, heads, L, S], [L, S], or
            torch.nn.MultiheadAttention module (with input=...).
        input: Required when passing an nn.MultiheadAttention module. PyTorch MHA
            defaults to average_attn_weights=True; SeeDL requests per-head weights
            (average_attn_weights=False) when supported to expose individual heads.
        tokens: Token labels for square self-attention sequences.
        query_tokens: Token labels for query sequence (rows).
        key_tokens: Token labels for key sequence (columns).
        head: Selected attention head index (default None displays average of heads).
        compare_heads: Optional tuple of (head_a, head_b) for head comparison.
        batch_index: Batch index to inspect for batched tensors (default 0).
        is_batched: Disambiguate 3D tensors: True for [B, L, S] (averaged across heads),
            False for [heads, L, S] (per-head attention).
        mask: Optional attention mask matrix [L, S] or key padding mask [S].
        max_seq_len: Bounded sequence length limit (default 256).
        max_heads: Maximum heads to capture/preview (default 32).
        theme: UI theme ('scientific', 'light', 'dark').
        backend: Rendering backend ('auto', 'plotly', 'svg', 'matplotlib').
    """
    selected_backend = resolve_backend(backend)
    from ._transformers import inspect_attention

    data, summary, notes = inspect_attention(
        weights,
        input=input,
        tokens=tokens,
        query_tokens=query_tokens,
        key_tokens=key_tokens,
        head=head,
        compare_heads=compare_heads,
        batch_index=batch_index,
        is_batched=is_batched,
        mask=mask,
        max_seq_len=max_seq_len,
        max_heads=max_heads,
        theme=theme,
    )
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


def features(
    model_or_tensor: Any,
    *,
    input: Any = None,
    layers: str | list[str] | None = None,
    channel: int | None = None,
    batch_index: int = 0,
    max_channels: int = 64,
    max_spatial_dim: int = 128,
    theme: str = "scientific",
    backend: str = "auto",
) -> Visualization:
    """Capture and interactively explore CNN feature maps, channel galleries, and spatial activations.

    Supports PyTorch Conv2d modules, nested Sequential models, custom CNN modules,
    and direct 4D/3D activation tensors [B, C, H, W] or [C, H, W].
    """
    selected_backend = resolve_backend(backend)
    from ._features import capture_feature_maps

    data, summary, notes = capture_feature_maps(
        model_or_tensor,
        example_input=input,
        layers=layers,
        channel=channel,
        batch_index=batch_index,
        max_channels=max_channels,
        max_spatial_dim=max_spatial_dim,
        theme=theme,
    )
    title = (
        f"{type(model_or_tensor).__name__} Feature Maps"
        if not hasattr(model_or_tensor, "ndim")
        else "CNN Feature Maps"
    )
    view = Visualization(
        kind="feature_maps",
        title=title,
        summary=summary,
        data=data,
        notes=notes,
        backend=selected_backend,
    )
    _show_result(view)
    return view


feature_maps = features


def vit_attention(
    attention_weights: Any,
    patch_grid: tuple[int, int],
    *,
    from_token: int = 0,
    has_cls: bool = True,
    head: int | None = None,
    theme: str = "scientific",
    backend: str = "auto",
) -> Visualization:
    """Reshape Vision Transformer token attention slice into a 2D spatial patch attention map."""
    selected_backend = resolve_backend(backend)
    from ._transformers import inspect_vit_attention

    data, summary, notes = inspect_vit_attention(
        attention_weights,
        patch_grid=patch_grid,
        from_token=from_token,
        has_cls=has_cls,
        head=head,
        theme=theme,
    )
    view = Visualization(
        kind="feature_maps",
        title="ViT Patch Attention",
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


def loss_landscape(
    model: Any,
    data: Any,
    *,
    loss_fn: Any,
    axes: tuple[tuple[str, int], tuple[str, int]],
    radius: float = 1.0,
    steps: int = 15,
    evaluate: bool = False,
    trajectory: Any = None,
) -> Visualization:
    """Evaluate a bounded 2D parameter-coordinate slice and render a 3D loss surface.

    Requires explicit evaluate=True, a scalar loss_fn, and two distinct named
    parameter coordinates. CPU models up to 250,000 parameters; 3–31 grid steps.
    Original model is preserved by evaluation on a deep copy. Trajectory points
    are optional (axis-1 offset, axis-2 offset) pairs in this same slice.
    """
    backend = resolve_backend("plotly")
    from ._spatial import loss_slice

    payload, notes = loss_slice(model, data, loss_fn, axes, radius, steps, evaluate, trajectory)
    result = Visualization(
        kind="loss_landscape",
        title="Loss landscape · coordinate slice",
        summary=payload["summary"],
        data=payload,
        notes=notes,
        backend=backend,
    )
    _show_result(result)
    return result


def bloch(
    state: Any,
    *,
    view: str = "3d",
    theme: str = "scientific",
    backend: str = "auto",
) -> Visualization:
    """Visualize a single-qubit quantum state on an interactive 3D Bloch sphere.

    Calculates exact Bloch vector coordinates x = Tr(ρ X), y = Tr(ρ Y), z = Tr(ρ Z).
    Supports pure states (|r| = 1 on sphere surface) and mixed states (|r| < 1 inside sphere).
    Accepts named states ('0', '1', '+', '-', '+i', '-i'), 1D complex statevectors,
    2x2 density matrices, or Qiskit Statevector/DensityMatrix objects.
    """
    if view not in {"3d", "auto"}:
        raise ValueError("view must be '3d' or 'auto' for Bloch sphere.")
    if theme not in {"light", "dark", "scientific", "minimal", "presentation"}:
        raise ValueError("theme must be light, dark, scientific, minimal, or presentation.")
    import math

    from ._quantum import inspect_bloch_state

    data = inspect_bloch_state(state)
    data["theme"] = theme
    selected_backend = resolve_backend(
        backend if backend != "auto" else ("plotly" if view == "3d" else "svg")
    )
    state_label = f"|ψ⟩ = ({data['x']:+.2f}, {data['y']:+.2f}, {data['z']:+.2f})"
    summary = (
        f"Bloch Vector: ({data['x']:+.3f}, {data['y']:+.3f}, {data['z']:+.3f}) · "
        f"|r| = {data['r']:.3f} ({'Pure' if data['is_pure'] else 'Mixed'}) · "
        f"P(|0⟩)={data['prob_0']:.1%}, P(|1⟩)={data['prob_1']:.1%}"
    )
    notes = [
        "Bloch sphere geometrically represents single-qubit density operators ρ = ½(I + x X + y Y + z Z).",
        f"Coordinates: x = Tr(ρ X) = {data['x']:+.4f}, y = Tr(ρ Y) = {data['y']:+.4f}, z = Tr(ρ Z) = {data['z']:+.4f}.",
        f"Purity: Tr(ρ²) = {data['purity']:.4f} ({'pure state' if data['is_pure'] else 'mixed state'}).",
        f"Spherical angles: θ = {math.degrees(data['theta']):.1f}°, φ = {math.degrees(data['phi']):.1f}°.",
    ]
    result = Visualization(
        kind="bloch",
        title=f"Bloch Sphere · {state_label}",
        summary=summary,
        data=data,
        notes=notes,
        backend=selected_backend,
    )
    _show_result(result)
    return result


def circuit(
    circuit: Any,
    *,
    simulate: bool = False,
    max_qubits: int = 10,
    max_ops: int = 500,
    max_bytes: int = 32 * 1024 * 1024,
    theme: str = "scientific",
    backend: str = "auto",
) -> Visualization:
    """Visualize a quantum circuit diagram, registers, gates, and optional state evolution.

    Accepts Qiskit QuantumCircuit instances. When simulate=True, performs
    step-by-step statevector simulation up to 10 qubits.
    """
    if theme not in {"light", "dark", "scientific", "minimal", "presentation"}:
        raise ValueError("theme must be light, dark, scientific, minimal, or presentation.")
    from ._quantum import inspect_circuit, simulate_circuit_evolution

    data = inspect_circuit(circuit)
    data["theme"] = theme
    data["simulate"] = simulate
    if simulate:
        data["evolution"] = simulate_circuit_evolution(
            circuit, max_qubits=max_qubits, max_ops=max_ops, max_bytes=max_bytes
        )
    selected_backend = resolve_backend(backend)
    summary = f"Quantum Circuit: {data['num_qubits']} qubits · Depth {data['depth']} · {data['num_ops']} gates"
    notes = [
        f"Circuit operations: {data['num_ops']} total instructions across {data['num_qubits']} qubit wire(s).",
        f"Circuit depth: {data['depth']}.",
    ]
    if simulate:
        notes.append(
            f"Statevector evolution calculated across {len(data['evolution'])} circuit steps. "
            "Playback shows before/after states; post-halt states are unavailable. "
            "Basis charts show the first 64 states (full bounded state retained in data)."
        )
    result = Visualization(
        kind="quantum_circuit",
        title=f"Quantum Circuit ({data['num_qubits']} Qubits)",
        summary=summary,
        data=data,
        notes=notes,
        backend=selected_backend,
    )
    _show_result(result)
    return result


def statevector(
    state: Any,
    *,
    theme: str = "scientific",
    backend: str = "auto",
    max_states: int = 64,
) -> Visualization:
    """Visualize quantum statevector amplitudes, measurement probabilities, and complex phases."""
    if theme not in {"light", "dark", "scientific", "minimal", "presentation"}:
        raise ValueError("theme must be light, dark, scientific, minimal, or presentation.")
    if isinstance(max_states, bool) or not isinstance(max_states, int) or max_states < 1:
        raise ValueError("max_states must be a positive integer.")
    import cmath
    import math

    from ._quantum import parse_statevector

    vec = parse_statevector(state)
    dim = len(vec)
    num_qubits = int(math.log2(dim))
    probs = [abs(c) ** 2 for c in vec]
    phases = [cmath.phase(c) for c in vec]
    basis_labels = [format(i, f"0{num_qubits}b") for i in range(dim)]
    amplitudes = [
        {
            "state": basis_labels[i],
            "real": c.real,
            "imag": c.imag,
            "mag": abs(c),
            "phase": phases[i],
        }
        for i, c in enumerate(vec)
    ]

    if dim > max_states:
        # Retain the top basis states by measurement probability
        ranked_indices = sorted(range(dim), key=lambda i: probs[i], reverse=True)[:max_states]
        selected_indices = sorted(ranked_indices)
        probs_view = [probs[i] for i in selected_indices]
        phases_view = [phases[i] for i in selected_indices]
        basis_labels_view = [basis_labels[i] for i in selected_indices]
        amplitudes_view = [amplitudes[i] for i in selected_indices]
        truncated = True
    else:
        probs_view = probs
        phases_view = phases
        basis_labels_view = basis_labels
        amplitudes_view = amplitudes
        truncated = False

    data = {
        "theme": theme,
        "dim": dim,
        "num_qubits": num_qubits,
        "probabilities": probs_view,
        "phases": phases_view,
        "basis_states": basis_labels_view,
        "amplitudes": amplitudes_view,
        "truncated": truncated,
        "total_states": dim,
    }
    selected_backend = resolve_backend(backend)
    summary = f"Statevector: {num_qubits} qubits ({dim} computational basis states) · Normalized"
    if truncated:
        summary += f" · Top {max_states} states displayed"
    notes = [
        f"Probabilities P(i) = |c_i|² sum to {sum(probs):.6f}.",
        "Phase angles mapped to continuous colormap in radians [-π, +π].",
    ]
    if truncated:
        notes.append(
            f"Visualizing top {max_states} of {dim} computational basis states by probability "
            f"({dim - max_states} states truncated for visualization performance)."
        )
    result = Visualization(
        kind="statevector",
        title=f"Quantum Statevector ({num_qubits} Qubits)",
        summary=summary,
        data=data,
        notes=notes,
        backend=selected_backend,
    )
    _show_result(result)
    return result


def density_matrix(
    rho: Any,
    *,
    theme: str = "scientific",
    backend: str = "auto",
) -> Visualization:
    """Inspect a quantum density operator (purity, trace, Hermiticity, and components)."""
    from ._quantum import parse_density_matrix

    mat = parse_density_matrix(rho)
    dim = len(mat)
    purity = sum((mat[i][j] * mat[j][i]).real for i in range(dim) for j in range(dim))
    tr = sum(mat[i][i].real for i in range(dim))
    if dim == 2:
        return bloch(mat, theme=theme, backend=backend)
    data = {
        "theme": theme,
        "dim": dim,
        "purity": purity,
        "trace": tr,
        "density_matrix": [[{"real": c.real, "imag": c.imag} for c in row] for row in mat],
    }
    selected_backend = resolve_backend(backend)
    summary = f"Density Matrix: {dim}×{dim} · Tr(ρ) = {tr:.4f} · Purity = {purity:.4f}"
    notes = [
        f"Hermitian positive semi-definite density operator of dimension {dim}×{dim}.",
        f"Purity Tr(ρ²) = {purity:.4f} ({'Pure state' if abs(purity - 1.0) < 1e-4 else 'Mixed state'}).",
    ]
    result = Visualization(
        kind="density_matrix",
        title=f"Density Matrix ({dim}×{dim})",
        summary=summary,
        data=data,
        notes=notes,
        backend=selected_backend,
    )
    _show_result(result)
    return result


def entanglement(
    state: Any,
    *,
    theme: str = "scientific",
    backend: str = "auto",
) -> Visualization:
    """Analyze multi-qubit entanglement, reduced density matrices, and von Neumann entropy."""
    import math

    from ._quantum import (
        compute_bloch_vector,
        parse_statevector,
        partial_trace_2qubit,
        statevector_to_density_matrix,
        von_neumann_entropy_single_qubit,
    )

    vec = parse_statevector(state)
    if len(vec) != 4:
        raise ValueError("entanglement analysis currently requires a 2-qubit state (dim 4).")
    rho_4x4 = statevector_to_density_matrix(vec)
    rho_A = partial_trace_2qubit(rho_4x4, trace_out=1)
    rho_B = partial_trace_2qubit(rho_4x4, trace_out=0)
    bv_A = compute_bloch_vector(rho_A)
    bv_B = compute_bloch_vector(rho_B)
    s_A = von_neumann_entropy_single_qubit(rho_A)
    s_B = von_neumann_entropy_single_qubit(rho_B)
    basis = ["00", "01", "10", "11"]
    probs = [abs(c) ** 2 for c in vec]
    data = {
        "theme": theme,
        "basis_states": basis,
        "probabilities": probs,
        "entropy_q0": s_A,
        "entropy_q1": s_B,
        "bloch_q0": bv_A,
        "bloch_q1": bv_B,
    }
    selected_backend = resolve_backend(backend)
    is_bell = abs(s_A - 1.0) < 1e-4 and abs(s_B - 1.0) < 1e-4
    summary = (
        f"Entanglement: S(ρ_A) = {s_A:.3f} bit, S(ρ_B) = {s_B:.3f} bit · "
        f"{'Maximally Entangled (Bell)' if is_bell else 'Separable' if s_A < 1e-4 else 'Partially Entangled'}"
    )
    notes = [
        "Bipartite entanglement diagnosed via partial trace and subsystem von Neumann entropy.",
        f"Reduced Qubit 0 Bloch vector length: {math.sqrt(sum(c**2 for c in bv_A[:3])):.3f}.",
        f"Reduced Qubit 1 Bloch vector length: {math.sqrt(sum(c**2 for c in bv_B[:3])):.3f}.",
    ]
    if is_bell:
        notes.append(
            "Reduced states are maximally mixed (I/2); reduced Bloch vectors are at the exact center (0, 0, 0)."
        )
    result = Visualization(
        kind="entanglement",
        title="Quantum Entanglement (2 Qubits)",
        summary=summary,
        data=data,
        notes=notes,
        backend=selected_backend,
    )
    _show_result(result)
    return result


def trace(
    model: Any,
    input: Any,
    *,
    record: tuple[str, ...] | list[str] = ("shapes", "activations"),
    max_nodes: int = 256,
    use_fx: bool | None = None,
    max_samples: int = 32,
) -> Any:
    """Record an exact, verified neural execution trace of a model forward pass."""
    from ._trace import trace as _trace_impl

    return _trace_impl(
        model,
        input,
        record=record,
        max_nodes=max_nodes,
        use_fx=use_fx,
        max_samples=max_samples,
    )


def flow(
    trace_or_model: Any,
    input: Any = None,
    *,
    record: tuple[str, ...] | list[str] = ("shapes", "activations"),
    max_nodes: int = 256,
    backend: str = "auto",
    title: str | None = None,
    mode: str = "graph",
) -> Visualization:
    """Create an interactive Tensor Flow and Neural Execution visualization."""
    from ._flow import flow as _flow_impl

    res = _flow_impl(
        trace_or_model,
        input=input,
        record=record,
        max_nodes=max_nodes,
        backend=backend,
        title=title,
        mode=mode,
    )
    _show_result(res)
    return res


def attention_rollout(
    attention_tensors: list[Any] | Any,
    *,
    tokens: list[str] | None = None,
    include_residual: bool = True,
    head_reduction: str = "mean",
    backend: str = "auto",
    title: str | None = None,
) -> Visualization:
    """Compute and visualize cumulative Attention Rollout across transformer layers."""
    from ._transformer_observatory import attention_rollout as _rollout_impl

    res = _rollout_impl(
        attention_tensors,
        tokens=tokens,
        include_residual=include_residual,
        head_reduction=head_reduction,
        backend=backend,
        title=title,
    )
    _show_result(res)
    return res


def attention_layers(
    attention_tensors: list[Any] | Any,
    *,
    tokens: list[str] | None = None,
    backend: str = "auto",
    title: str | None = None,
) -> Visualization:
    """Multi-layer Transformer Attention Explorer."""
    from ._transformer_observatory import attention_layers as _layers_impl

    res = _layers_impl(
        attention_tensors,
        tokens=tokens,
        backend=backend,
        title=title,
    )
    _show_result(res)
    return res


def gradient_health(
    model: Any,
    *,
    max_parameters: int = 256,
    backend: str = "auto",
    title: str | None = None,
) -> Visualization:
    """Analyze and visualize gradient norms, distributions, and numerical health."""
    from ._gradient_health import gradient_health as _grad_impl

    res = _grad_impl(model, max_parameters=max_parameters, backend=backend, title=title)
    _show_result(res)
    return res


def dataset_health(
    dataset: Any,
    *,
    max_samples: int = 1000,
    backend: str = "auto",
    title: str | None = None,
) -> Visualization:
    """Analyze dataset quality, missing values, nonfinite numbers, outliers, and balance."""
    from ._dataset_health import dataset_health as _ds_health_impl

    res = _ds_health_impl(dataset, max_samples=max_samples, backend=backend, title=title)
    _show_result(res)
    return res


def dataset_drift(
    reference_data: Any,
    comparison_data: Any,
    *,
    max_samples: int = 1000,
    bins: int = 10,
    backend: str = "auto",
    title: str | None = None,
) -> Visualization:
    """Measure empirical distribution drift between reference and production dataset samples."""
    from ._dataset_health import dataset_drift as _drift_impl

    res = _drift_impl(
        reference_data,
        comparison_data,
        max_samples=max_samples,
        bins=bins,
        backend=backend,
        title=title,
    )
    _show_result(res)
    return res


def compare_attributions(
    model: Any,
    input: Any,
    *,
    target: int | None = None,
    methods: tuple[str, ...] | list[str] = (
        "saliency",
        "integrated_gradients",
        "smoothgrad",
    ),
    steps: int = 20,
    samples: int = 25,
    stdev_spread: float = 0.15,
    baseline: Any | None = None,
    layer: str | None = None,
    abs: bool | None = None,
    seed: int | None = 42,
    backend: str = "auto",
) -> Visualization:
    """Compare multiple feature attribution methods on the same model and input."""
    from ._explain_compare import compare_attributions as _cmp_attr_impl

    res = _cmp_attr_impl(
        model,
        input,
        target=target,
        methods=methods,
        steps=steps,
        samples=samples,
        stdev_spread=stdev_spread,
        baseline=baseline,
        layer=layer,
        abs=abs,
        seed=seed,
        backend=backend,
    )
    _show_result(res)
    return res


def compare_runs(
    run_a: Any,
    run_b: Any,
    *,
    name_a: str = "Run A",
    name_b: str = "Run B",
    metrics: list[str] | None = None,
    backend: str = "auto",
    title: str | None = None,
) -> Visualization:
    """Compare two training runs or experiment histories with step alignment."""
    from ._training_compare import compare_runs as _cmp_runs_impl

    res = _cmp_runs_impl(
        run_a,
        run_b,
        name_a=name_a,
        name_b=name_b,
        metrics=metrics,
        backend=backend,
        title=title,
    )
    _show_result(res)
    return res


def quantum_noise(
    state: Any = None,
    *,
    channel: str = "depolarizing",
    param: float = 0.3,
    steps: int = 10,
    backend: str = "auto",
    title: str | None = None,
) -> Visualization:
    """Analyze single-qubit quantum state evolution under Kraus noise channels."""
    from ._quantum_lab import quantum_noise_explorer as _noise_impl

    res = _noise_impl(
        state=state,
        channel=channel,
        param=param,
        steps=steps,
        backend=backend,
        title=title,
    )
    _show_result(res)
    return res


def report(
    title: str,
    views: Any = None,
    *,
    subtitle: str = "",
    abstract: str = "",
    author: str = "",
    affiliation: str = "",
    theme: str = "dark",
    max_size_mb: float = 25.0,
) -> Any:
    """Create a publication-grade scientific report combining multiple SeeDL visualizations."""
    from ._reports import report as _report_impl

    return _report_impl(
        title=title,
        views=views,
        subtitle=subtitle,
        abstract=abstract,
        author=author,
        affiliation=affiliation,
        theme=theme,
        max_size_mb=max_size_mb,
    )


def save_analysis(
    views: Any,
    filepath: Any,
    *,
    indent: int = 2,
) -> Any:
    """Save Visualization instances to a safe, reproducible JSON artifact without pickle."""
    from ._serialization import save_analysis as _save_impl

    return _save_impl(views, filepath, indent=indent)


def load_analysis(filepath: Any) -> Any:
    """Load reproducible Visualization instances from a safe JSON artifact."""
    from ._serialization import load_analysis as _load_impl

    return _load_impl(filepath)
