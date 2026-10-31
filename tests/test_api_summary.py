"""Regression tests for SeeDL's user-facing visualization summaries."""

from __future__ import annotations

import pytest

from seedl._summaries import _SUMMARY_FORMATTERS, register_summary_formatter
from seedl.api import _summary


@pytest.mark.parametrize(
    ("kind", "data", "metadata", "expected"),
    [
        # Scalars
        ("scalar", {"caption": "float"}, {}, "float"),
        ("scalar", {}, {"Type": "int"}, "int"),
        ("scalar", {}, {}, "scalar"),
        # Sequences
        ("sequence", {"values": [1, 2, 3]}, {}, "3 value(s)"),
        ("sequence", {}, {}, "0 value(s)"),
        # Matrices
        ("matrix", {"shape": (2, 3)}, {}, "Matrix with shape (2, 3)"),
        ("matrix", {"values": [[1, 2], [3, 4]]}, {}, "Matrix with shape (2, 0)"),
        ("matrix", {}, {}, "Matrix with shape (0, 0)"),
        # Images (shared handler for image and dataset_images)
        ("image", {"shape": (3, 8, 8)}, {}, "Image tensor with shape (3, 8, 8)"),
        ("image", {}, {}, "Image tensor with shape None"),
        (
            "dataset_images",
            {"shape": (16, 3, 32, 32)},
            {},
            "Image tensor with shape (16, 3, 32, 32)",
        ),
        # Distributions
        ("distribution", {"shape": (100,)}, {}, "Sampled value distribution · shape (100,)"),
        ("distribution", {}, {}, "Sampled value distribution · shape unknown"),
        # Models and training (shared handler)
        ("model", {"summary": "Two layers"}, {}, "Two layers"),
        ("model", {}, {}, ""),
        ("training", {"summary": "Epoch 10/10"}, {}, "Epoch 10/10"),
        ("history_comparison", {}, {"Summary": "Run A vs Run B"}, "Run A vs Run B"),
        ("model_comparison", {}, {"Summary": "ResNet vs VGG"}, "ResNet vs VGG"),
        # Activations and gradients
        ("activations", {"layers": [{}, {}]}, {}, "2 layer output(s)"),
        ("activations", {}, {}, "0 layer output(s)"),
        ("gradients", {"layers": [{}]}, {}, "1 parameter tensor(s)"),
        ("gradients", {}, {}, "0 parameter tensor(s)"),
        # Predictions
        (
            "classification_predictions",
            {"top_class": "cat", "top_prob": 0.876},
            {},
            "Top prediction: 'cat' (87.6%)",
        ),
        (
            "classification_predictions",
            {"top_class": "dog"},
            {},
            "Top prediction: 'dog' (0.0%)",
        ),
        ("regression_predictions", {"count": 3}, {}, "Regression: 3 prediction(s)"),
        (
            "regression_predictions",
            {"predictions": [1.0, 2.0, 3.5]},
            {},
            "Regression: 3 prediction(s)",
        ),
        ("regression_predictions", {}, {}, "Regression: 0 prediction(s)"),
        (
            "prediction_comparison",
            {"name_a": "model_1", "name_b": "model_2"},
            {},
            "Prediction comparison: model_1 vs model_2",
        ),
        # Datasets
        (
            "dataset_tabular",
            {"samples": 50, "features": 4},
            {},
            "Tabular dataset: 50 samples × 4 features",
        ),
        ("dataset_general", {"count": 100}, {}, "Dataset with 100 samples"),
        # Features (shared handler)
        ("feature_maps", {"summary": "CNN feature map"}, {}, "CNN feature map"),
        ("feature_maps", {}, {"Summary": "Metadata map"}, "Metadata map"),
        ("feature_maps", {"layers": [{}, {}]}, {}, "2 layer(s)"),
        ("feature_comparison", {}, {}, "0 layer(s)"),
        # Attention (shared handler)
        ("attention", {"seq_len": 128}, {}, "Attention matrix (128 tokens)"),
        ("attention_comparison", {"seq_len": 64}, {}, "Attention matrix (64 tokens)"),
        ("educational_attention", {"seq_len": 32}, {}, "Attention matrix (32 tokens)"),
        # Embeddings and traces
        ("embeddings", {"num_vectors": 9, "dim": 2}, {}, "Embeddings: 9 vectors in 2D"),
        (
            "execution_trace",
            {"nodes": [{}, {}, {}], "total_parameters": 1234},
            {},
            "Execution trace: 3 operations (1,234 params)",
        ),
        ("execution_trace", {}, {}, "Execution trace: 0 operations (0 params)"),
        (
            "attention_rollout",
            {"num_layers": 12, "seq_len": 256},
            {},
            "Attention rollout across 12 layers (256 tokens)",
        ),
        (
            "attribution_comparison",
            {"methods": ["IntegratedGradients", "DeepLIFT"]},
            {},
            "Attribution comparison: 2 method(s)",
        ),
        ("attribution_comparison", {}, {}, "Attribution comparison: 0 method(s)"),
        # Diagnostics and health
        (
            "gradient_health",
            {"total_parameters_inspected": 256},
            {},
            "Gradient health: 256 parameters inspected",
        ),
        (
            "quantum_noise",
            {"channel": "bitflip", "param": 0.05},
            {},
            "Quantum noise: bitflip channel (p=0.05)",
        ),
        (
            "quantum_noise",
            {"channel": "depolarizing"},
            {},
            "Quantum noise: depolarizing channel (p=0.00)",
        ),
        (
            "dataset_drift",
            {"psi": 0.123456, "drift_level": "Moderate"},
            {},
            "Dataset drift: PSI=0.1235 (Moderate)",
        ),
        ("dataset_drift", {}, {}, "Dataset drift: PSI=0.0000 (Unknown)"),
        (
            "dataset_health",
            {"sample_count": 500},
            {},
            "Dataset health: 500 samples inspected",
        ),
        (
            "run_comparison",
            {"name_a": "exp1", "name_b": "exp2"},
            {},
            "Run comparison: exp1 vs exp2",
        ),
        (
            "tensor_explorer",
            {"shape": [2, 3, 4], "axes": [0, 1]},
            {},
            "Interactive exploration · shape (2, 3, 4) · 2 slicing axes",
        ),
        (
            "tensor_explorer",
            {},
            {},
            "Interactive exploration · shape () · 0 slicing axes",
        ),
        # Educational prefix fallback
        ("educational_attention_demo", {"concept": "Attention"}, {}, "Attention"),
        ("educational_loss", {}, {}, "educational_loss"),
        # Unknown visualization kinds fallback
        ("unknown", {}, {"Summary": "Fallback summary"}, "Fallback summary"),
        ("unknown_custom", {"summary": "Direct summary"}, {}, "Direct summary"),
        ("unknown_custom", {}, {}, ""),
    ],
)
def test_summary_formatting(kind, data, metadata, expected) -> None:
    assert _summary(kind, data, metadata) == expected


def test_register_summary_formatter_single() -> None:
    custom_kind = "test_custom_kind_single"

    def _custom_formatter(data: dict, metadata: dict) -> str:
        return f"Custom: {data.get('metric', 0)}"

    register_summary_formatter(custom_kind, _custom_formatter)
    try:
        assert _summary(custom_kind, {"metric": 42}, {}) == "Custom: 42"
    finally:
        _SUMMARY_FORMATTERS.pop(custom_kind, None)


def test_register_summary_formatter_multiple() -> None:
    custom_kinds = ("test_custom_kind_a", "test_custom_kind_b")

    def _shared_custom_formatter(data: dict, metadata: dict) -> str:
        return f"Shared: {data.get('val')}"

    register_summary_formatter(custom_kinds, _shared_custom_formatter)
    try:
        assert _summary("test_custom_kind_a", {"val": "alpha"}, {}) == "Shared: alpha"
        assert _summary("test_custom_kind_b", {"val": "beta"}, {}) == "Shared: beta"
    finally:
        for k in custom_kinds:
            _SUMMARY_FORMATTERS.pop(k, None)


def test_all_formatters_callable() -> None:
    for kind, formatter in _SUMMARY_FORMATTERS.items():
        assert callable(formatter), f"Formatter for {kind} must be callable"
        # Verify formatter runs without crashing on empty dicts
        res = formatter({}, {})
        assert isinstance(res, str), f"Formatter for {kind} must return str, got {type(res)}"
