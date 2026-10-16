"""Tests for Dataset Intelligence and Drift Analytics in SeeDL 0.5.0."""

from __future__ import annotations

import pytest

import seedl as sd
from seedl._dataset_health import dataset_drift, dataset_health

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def test_dataset_drift_identical_distributions():
    """Identical distributions have near-zero PSI and JSD."""
    data = [float(i % 10) for i in range(500)]
    viz = dataset_drift(data, data)

    assert viz.kind == "dataset_drift"
    assert viz.data["psi"] < 0.05
    assert viz.data["jsd"] < 0.05
    assert "Negligible Drift" in viz.data["drift_level"]


def test_dataset_drift_shifted_distributions():
    """Strongly shifted distributions produce high PSI (> 0.25) and JSD (> 0.3 bits)."""
    # Baseline around 0, comparison around 10
    ref = [float(i % 5) for i in range(500)]
    comp = [float((i % 5) + 15) for i in range(500)]

    viz = dataset_drift(ref, comp)

    assert viz.kind == "dataset_drift"
    assert viz.data["psi"] > 0.25
    assert viz.data["jsd"] > 0.3
    assert "Significant Drift" in viz.data["drift_level"]


def test_dataset_health_audit_metrics_and_outliers():
    """Dataset health correctly detects IQR outliers, summary stats, and nonfinite values."""
    # 100 values with 2 extreme outliers and 2 NaNs
    vals = [10.0 + (i % 5) for i in range(100)]
    vals.append(1000.0)  # extreme upper outlier
    vals.append(-1000.0)  # extreme lower outlier
    vals.append(float("nan"))
    vals.append(float("inf"))

    viz = dataset_health(vals)

    assert viz.kind == "dataset_health"
    data = viz.data
    assert data["sample_count"] == 104
    assert data["nonfinite_count"] == 2
    assert data["outlier_count"] >= 2
    assert data["outlier_fraction"] > 0.01


def test_dataset_health_constant_warning():
    """Constant dataset produces a warning note."""
    vals = [42.0] * 50
    viz = dataset_health(vals)

    assert viz.kind == "dataset_health"
    assert viz.data["is_constant"] is True
    assert any("constant" in note.lower() for note in viz.notes)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_dataset_drift_and_health_torch_tensors():
    """Dataset health and drift support PyTorch tensors directly."""
    t1 = torch.randn(200)
    t2 = torch.randn(200) + 3.0

    viz_drift = sd.dataset_drift(t1, t2)
    assert viz_drift.kind == "dataset_drift"
    assert viz_drift.data["psi"] > 0.1

    viz_health = sd.dataset_health(t1)
    assert viz_health.kind == "dataset_health"
    assert viz_health.data["sample_count"] == 200
