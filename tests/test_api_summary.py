"""Regression tests for SeeDL's user-facing visualization summaries."""

from __future__ import annotations

import pytest

from seedl.api import _summary


@pytest.mark.parametrize(
    ("kind", "data", "metadata", "expected"),
    [
        ("scalar", {"caption": "float"}, {}, "float"),
        ("scalar", {}, {"Type": "int"}, "int"),
        ("sequence", {"values": [1, 2, 3]}, {}, "3 value(s)"),
        ("matrix", {"shape": (2, 3)}, {}, "Matrix with shape (2, 3)"),
        ("image", {"shape": (3, 8, 8)}, {}, "Image tensor with shape (3, 8, 8)"),
        ("model", {"summary": "Two layers"}, {}, "Two layers"),
        ("activations", {"layers": [{}, {}]}, {}, "2 layer output(s)"),
        ("gradients", {"layers": [{}]}, {}, "1 parameter tensor(s)"),
        (
            "classification_predictions",
            {"top_class": "cat", "top_prob": 0.876},
            {},
            "Top prediction: 'cat' (87.6%)",
        ),
        ("regression_predictions", {"count": 3}, {}, "Regression: 3 prediction(s)"),
        ("dataset_tabular", {"samples": 50, "features": 4}, {}, "Tabular dataset: 50 samples × 4 features"),
        ("embeddings", {"num_vectors": 9, "dim": 2}, {}, "Embeddings: 9 vectors in 2D"),
        (
            "execution_trace",
            {"nodes": [{}, {}, {}], "total_parameters": 1234},
            {},
            "Execution trace: 3 operations (1,234 params)",
        ),
        ("gradient_health", {"total_parameters_inspected": 256}, {}, "Gradient health: 256 parameters inspected"),
        (
            "dataset_drift",
            {"psi": 0.123456, "drift_level": "Moderate"},
            {},
            "Dataset drift: PSI=0.1235 (Moderate)",
        ),
        (
            "tensor_explorer",
            {"shape": [2, 3, 4], "axes": [0, 1]},
            {},
            "Interactive exploration · shape (2, 3, 4) · 2 slicing axes",
        ),
        ("educational_attention_demo", {"concept": "Attention"}, {}, "Attention"),
        ("unknown", {}, {"Summary": "Fallback summary"}, "Fallback summary"),
    ],
)
def test_summary_formatting(kind, data, metadata, expected) -> None:
    assert _summary(kind, data, metadata) == expected
