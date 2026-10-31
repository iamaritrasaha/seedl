"""Concise summaries for SeeDL's semantic visualization payloads."""

from __future__ import annotations

from typing import Any


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
    if kind == "tensor_explorer":
        shape = tuple(data.get("shape", ()))
        axes_count = len(data.get("axes", []))
        return f"Interactive exploration · shape {shape} · {axes_count} slicing axes"
    if kind.startswith("educational_"):
        return str(data.get("concept", kind))
    return str(data.get("summary", metadata.get("Summary", "")))
