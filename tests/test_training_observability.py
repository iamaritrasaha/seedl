"""Tests for training history parsing, diagnostic indicators, and experiment tracking."""

from __future__ import annotations

import math

import pytest

import seedl as sd
from seedl._training import (
    Tracker,
    exponential_smoothing,
    parse_training_history,
)


def test_exponential_smoothing_behavior():
    values = [1.0, 2.0, 3.0, 4.0]
    smoothed = exponential_smoothing(values, weight=0.5)
    assert len(smoothed) == 4
    assert smoothed[0] == 1.0
    # EMA check: 1.0*0.5 + 2.0*0.5 = 1.5
    assert abs(smoothed[1] - 1.5) < 1e-6
    # With NaN:
    vals_nan = [1.0, math.nan, 3.0]
    smoothed_nan = exponential_smoothing(vals_nan, weight=0.5)
    assert len(smoothed_nan) == 3
    assert math.isnan(smoothed_nan[1])


def test_training_diagnostics_divergence_and_plateau():
    # Diverging loss: 2.0 -> 10.0 -> 50.0
    diverging = {"loss": [2.0, 10.0, 50.0]}
    data, summary, notes = parse_training_history(diverging)
    assert any("diverging loss" in note.lower() for note in notes)

    # Plateau loss: 0.5, 0.5, 0.5, 0.5, 0.5
    plateau = {"loss": [0.5, 0.5, 0.5, 0.5, 0.5]}
    data2, summary2, notes2 = parse_training_history(plateau)
    assert any("plateau" in note.lower() for note in notes2)


def test_training_history_comparison_and_sd_compare():
    hist_a = {"epochs": [1, 2, 3], "loss": [1.0, 0.6, 0.4]}
    hist_b = {"epochs": [1, 2, 3], "loss": [1.0, 0.8, 0.7]}

    view = sd.compare(hist_a, hist_b, names=("Adam", "SGD"), backend="svg")
    assert view.kind == "history_comparison"
    assert "loss" in view.data["groups"][0]["name"]
    # Check that differences are noted
    assert len(view.data.get("differences", [])) >= 1
    svg = view.to_figure()
    assert "<svg" in svg


def test_tracker_logging_history_and_serialization(tmp_path):
    tracker = Tracker("baseline_experiment")
    tracker.log(epoch=1, loss=0.85, accuracy=0.62)
    tracker.log(epoch=2, loss=0.55, accuracy=0.79)
    tracker.log(epoch=3, loss=0.32, accuracy=0.91)

    hist = tracker.history()
    assert hist["epochs"] == [1.0, 2.0, 3.0]
    assert len(hist["loss"]) == 3
    assert len(hist["accuracy"]) == 3

    # Test JSON save and load
    save_path = tmp_path / "tracker_run.json"
    tracker.save(str(save_path))
    assert save_path.exists()

    loaded = Tracker.load(str(save_path))
    assert loaded.history() == tracker.history()

    # Test sd.see(tracker) direct visualization
    view = sd.see(tracker, backend="svg")
    assert view.kind == "training"
    assert len(view.data["groups"]) == 2  # loss and accuracy


def test_training_renderers_plotly_and_matplotlib():
    pytest.importorskip("plotly")
    plt = pytest.importorskip("matplotlib.pyplot")

    hist = {"loss": [0.9, 0.5, 0.3], "val_loss": [1.0, 0.6, 0.4]}
    view_plotly = sd.see(hist, backend="plotly")
    fig = view_plotly.to_figure()
    assert hasattr(fig, "data")
    assert len(fig.data) >= 2

    view_mpl = sd.see(hist, backend="matplotlib")
    fig_mpl = view_mpl.to_figure()
    assert hasattr(fig_mpl, "axes")
    plt.close(fig_mpl)


def test_exponential_smoothing_leading_nan_not_poisoning():
    # If first value is NaN, subsequent values must not become NaN!
    vals = [float("nan"), 2.0, 4.0]
    smoothed = exponential_smoothing(vals, weight=0.5)
    assert len(smoothed) == 3
    assert math.isnan(smoothed[0])
    # Index 1 should be 2.0
    assert abs(smoothed[1] - 2.0) < 1e-6
    # Index 2 should be 2.0 * 0.5 + 4.0 * 0.5 = 3.0
    assert abs(smoothed[2] - 3.0) < 1e-6


def test_tracker_load_malformed_and_name_preservation(tmp_path):
    from seedl.exceptions import SeeDLError

    # Test name preservation
    tracker = Tracker("my_custom_run")
    tracker.log(epoch=1, loss=0.5)
    p = tmp_path / "tracker_run.json"
    tracker.save(str(p))

    loaded = Tracker.load(str(p))
    assert loaded.name == "my_custom_run"

    # Test malformed JSON (array instead of dict)
    p_bad = tmp_path / "bad.json"
    p_bad.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(SeeDLError, match="Invalid tracker data"):
        Tracker.load(str(p_bad))


def test_single_epoch_summary():
    data, summary, _ = parse_training_history({"loss": [0.42]})
    assert "1 epoch(s)" in summary
