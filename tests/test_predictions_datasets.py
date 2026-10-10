"""Tests for prediction inspection, ranking, residuals, comparison, and dataset inspection."""

from __future__ import annotations

import pytest

import seedl as sd
from seedl._predictions import inspect_predictions


def test_classification_predictions_and_logits():
    # Raw logits: [2.5, 0.1, -1.2]
    logits = [2.5, 0.1, -1.2]
    labels = ["cat", "dog", "bird"]
    kind, data, summary, notes = inspect_predictions(logits, labels=labels, task="classification")

    assert kind == "classification_predictions"
    assert data["top_class"] == "cat"
    assert data["top_prob"] > 0.8
    assert any("raw logits" in note.lower() for note in notes)

    # Normalized probabilities should not trigger logits note
    probs = [0.85, 0.10, 0.05]
    _, data_probs, _, notes_probs = inspect_predictions(probs, labels=labels, task="classification")
    assert not any("raw logits" in note.lower() for note in notes_probs)


def test_regression_predictions_and_residuals():
    preds = [10.0, 20.0, 30.0]
    targets = [12.0, 19.0, 33.0]

    view = sd.predictions(preds, targets=targets, task="regression", backend="svg")
    assert view.kind == "regression_predictions"
    assert "mae" in view.data
    assert "rmse" in view.data
    # Errors: [-2.0, 1.0, -3.0], MAE = (2 + 1 + 3) / 3 = 2.0
    assert abs(view.data["mae"] - 2.0) < 1e-4
    svg = view.to_figure()
    assert "<svg" in svg


def test_compare_predictions_agreement_and_disagreement():
    preds_a = [0.8, 0.15, 0.05]  # Class 0
    preds_b = [0.2, 0.70, 0.10]  # Class 1
    labels = ["alpha", "beta", "gamma"]

    view = sd.compare_predictions(
        preds_a, preds_b, labels=labels, names=("Model A", "Model B"), backend="svg"
    )
    assert view.kind == "prediction_comparison"
    assert view.data["agreement"] is False
    assert view.data["top_a"] == "alpha"
    assert view.data["top_b"] == "beta"


def test_dataset_and_dataloader_inspection():
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, TensorDataset

    # Create synthetic image dataset: (B, C, H, W)
    images = torch.randn(16, 3, 8, 8)
    labels = torch.randint(0, 4, (16,))
    ds = TensorDataset(images, labels)
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    view = sd.dataset(loader, backend="svg")
    assert view.kind == "dataset_images"
    assert view.data["num_samples"] == 8
    assert "class_counts" in view.data
    assert len(view.data["grid"]) > 0

    svg = view.to_figure()
    assert "<svg" in svg


def test_tabular_dataset_inspection():
    torch = pytest.importorskip("torch")
    from torch.utils.data import TensorDataset

    # 2D features (tabular): (32, 4)
    features = torch.randn(32, 4)
    ds = TensorDataset(features)

    view = sd.dataset(ds, backend="svg")
    assert view.kind == "dataset_tabular"
    assert view.data["features"] == 4
    assert len(view.data["feature_stats"]) == 4
