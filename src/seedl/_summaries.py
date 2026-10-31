"""Concise summaries for SeeDL's semantic visualization payloads."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

SummaryFormatter = Callable[[dict[str, Any], dict[str, Any]], str]


def _format_scalar(data: dict[str, Any], metadata: dict[str, Any]) -> str:
    return str(data.get("caption", metadata.get("Type", "scalar")))


def _format_sequence(data: dict[str, Any], metadata: dict[str, Any]) -> str:
    values = data.get("values", [])
    return f"{len(values):,} value(s)"


def _format_matrix(data: dict[str, Any], metadata: dict[str, Any]) -> str:
    shape = data.get("shape", (len(data.get("values", [])), 0))
    return f"Matrix with shape {shape}"


def _format_image(data: dict[str, Any], metadata: dict[str, Any]) -> str:
    return f"Image tensor with shape {data.get('shape')}"


def _format_distribution(data: dict[str, Any], metadata: dict[str, Any]) -> str:
    return f"Sampled value distribution · shape {data.get('shape', 'unknown')}"


def _format_summary_metadata(data: dict[str, Any], metadata: dict[str, Any]) -> str:
    return str(data.get("summary", metadata.get("Summary", "")))


def _format_activations(data: dict[str, Any], metadata: dict[str, Any]) -> str:
    return f"{len(data.get('layers', []))} layer output(s)"


def _format_gradients(data: dict[str, Any], metadata: dict[str, Any]) -> str:
    return f"{len(data.get('layers', []))} parameter tensor(s)"


def _format_classification_predictions(data: dict[str, Any], metadata: dict[str, Any]) -> str:
    return f"Top prediction: '{data.get('top_class')}' ({data.get('top_prob', 0):.1%})"


def _format_regression_predictions(data: dict[str, Any], metadata: dict[str, Any]) -> str:
    count = data.get("count", len(data.get("predictions", [])))
    return f"Regression: {count} prediction(s)"


def _format_prediction_comparison(data: dict[str, Any], metadata: dict[str, Any]) -> str:
    return f"Prediction comparison: {data.get('name_a')} vs {data.get('name_b')}"


def _format_dataset_tabular(data: dict[str, Any], metadata: dict[str, Any]) -> str:
    return f"Tabular dataset: {data.get('samples')} samples × {data.get('features')} features"


def _format_dataset_general(data: dict[str, Any], metadata: dict[str, Any]) -> str:
    return f"Dataset with {data.get('count')} samples"


def _format_feature_maps(data: dict[str, Any], metadata: dict[str, Any]) -> str:
    fallback = f"{len(data.get('layers', []))} layer(s)"
    return str(data.get("summary", metadata.get("Summary", fallback)))


def _format_attention(data: dict[str, Any], metadata: dict[str, Any]) -> str:
    return f"Attention matrix ({data.get('seq_len')} tokens)"


def _format_embeddings(data: dict[str, Any], metadata: dict[str, Any]) -> str:
    return f"Embeddings: {data.get('num_vectors')} vectors in {data.get('dim')}D"


def _format_execution_trace(data: dict[str, Any], metadata: dict[str, Any]) -> str:
    nodes = data.get("nodes", [])
    total_params = data.get("total_parameters", 0)
    return f"Execution trace: {len(nodes)} operations ({total_params:,} params)"


def _format_attention_rollout(data: dict[str, Any], metadata: dict[str, Any]) -> str:
    return (
        f"Attention rollout across {data.get('num_layers')} layers ({data.get('seq_len')} tokens)"
    )


def _format_attribution_comparison(data: dict[str, Any], metadata: dict[str, Any]) -> str:
    return f"Attribution comparison: {len(data.get('methods', []))} method(s)"


def _format_gradient_health(data: dict[str, Any], metadata: dict[str, Any]) -> str:
    return f"Gradient health: {data.get('total_parameters_inspected')} parameters inspected"


def _format_quantum_noise(data: dict[str, Any], metadata: dict[str, Any]) -> str:
    param = data.get("param", 0)
    return f"Quantum noise: {data.get('channel')} channel (p={param:.2f})"


def _format_dataset_drift(data: dict[str, Any], metadata: dict[str, Any]) -> str:
    psi = data.get("psi", 0)
    drift_level = data.get("drift_level", "Unknown")
    return f"Dataset drift: PSI={psi:.4f} ({drift_level})"


def _format_dataset_health(data: dict[str, Any], metadata: dict[str, Any]) -> str:
    return f"Dataset health: {data.get('sample_count')} samples inspected"


def _format_run_comparison(data: dict[str, Any], metadata: dict[str, Any]) -> str:
    return f"Run comparison: {data.get('name_a')} vs {data.get('name_b')}"


def _format_tensor_explorer(data: dict[str, Any], metadata: dict[str, Any]) -> str:
    shape = tuple(data.get("shape", ()))
    axes_count = len(data.get("axes", []))
    return f"Interactive exploration · shape {shape} · {axes_count} slicing axes"


def _format_educational(kind: str, data: dict[str, Any]) -> str:
    return str(data.get("concept", kind))


_SUMMARY_FORMATTERS: dict[str, SummaryFormatter] = {
    # Scalars, sequences, and matrices
    "scalar": _format_scalar,
    "sequence": _format_sequence,
    "matrix": _format_matrix,
    # Image tensors (shared handler)
    "image": _format_image,
    "dataset_images": _format_image,
    # Distributions
    "distribution": _format_distribution,
    # Model and training summaries (shared handler)
    "model": _format_summary_metadata,
    "training": _format_summary_metadata,
    "history_comparison": _format_summary_metadata,
    "model_comparison": _format_summary_metadata,
    # Activations and parameter gradients
    "activations": _format_activations,
    "gradients": _format_gradients,
    # Predictions and prediction comparisons
    "classification_predictions": _format_classification_predictions,
    "regression_predictions": _format_regression_predictions,
    "prediction_comparison": _format_prediction_comparison,
    # Datasets
    "dataset_tabular": _format_dataset_tabular,
    "dataset_general": _format_dataset_general,
    # Features (shared handler)
    "feature_maps": _format_feature_maps,
    "feature_comparison": _format_feature_maps,
    # Attention mechanisms (shared handler)
    "attention": _format_attention,
    "attention_comparison": _format_attention,
    "educational_attention": _format_attention,
    # Embeddings and traces
    "embeddings": _format_embeddings,
    "execution_trace": _format_execution_trace,
    "attention_rollout": _format_attention_rollout,
    "attribution_comparison": _format_attribution_comparison,
    # Diagnostic and health checks
    "gradient_health": _format_gradient_health,
    "quantum_noise": _format_quantum_noise,
    "dataset_drift": _format_dataset_drift,
    "dataset_health": _format_dataset_health,
    "run_comparison": _format_run_comparison,
    "tensor_explorer": _format_tensor_explorer,
}


def register_summary_formatter(
    kind: str | Iterable[str],
    formatter: SummaryFormatter,
) -> None:
    """Register a summary formatter for one or more visualization kinds."""
    if isinstance(kind, str):
        _SUMMARY_FORMATTERS[kind] = formatter
    else:
        for k in kind:
            _SUMMARY_FORMATTERS[k] = formatter


def _summary(kind: str, data: dict[str, Any], metadata: dict[str, Any]) -> str:
    formatter = _SUMMARY_FORMATTERS.get(kind)
    if formatter is not None:
        return formatter(data, metadata)
    if kind.startswith("educational_"):
        return _format_educational(kind, data)
    return _format_summary_metadata(data, metadata)
