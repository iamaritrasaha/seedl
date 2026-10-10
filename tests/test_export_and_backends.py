"""Tests for visualization export (.save to HTML, SVG, PNG), explore(), and multi-backend consistency."""

from __future__ import annotations

import pytest

import seedl as sd
from seedl.exceptions import SeeDLError


def test_save_html_svg_png(tmp_path):
    pytest.importorskip("matplotlib")

    view = sd.see([1.0, 2.0, 3.0, 4.0], backend="svg")

    # Save SVG
    svg_path = tmp_path / "plot.svg"
    view.save(svg_path)
    assert svg_path.exists()
    content = svg_path.read_text(encoding="utf-8")
    assert "<svg" in content

    # Save HTML
    html_path = tmp_path / "plot.html"
    view.save(html_path)
    assert html_path.exists()
    html_content = html_path.read_text(encoding="utf-8")
    assert "<html" in html_content
    assert "Exported by SeeDL" in html_content

    # Save PNG
    png_path = tmp_path / "plot.png"
    view.save(png_path)
    assert png_path.exists()
    header = png_path.read_bytes()[:8]
    assert header == b"\x89PNG\r\n\x1a\n"

    # Unsupported format
    with pytest.raises(SeeDLError, match="Unsupported export format"):
        view.save(tmp_path / "plot.pdf")


def test_explore_switches_to_interactive():
    pytest.importorskip("plotly")

    view = sd.see([10.0, 20.0, 30.0], backend="svg")
    assert view.backend == "svg"

    fig = view.explore()
    assert view.backend == "plotly"
    assert hasattr(fig, "data")


def test_multi_backend_consistency_for_new_visualizations():
    pytest.importorskip("plotly")
    plt = pytest.importorskip("matplotlib.pyplot")

    # 1. Prediction classification
    p_view = sd.predictions([2.0, -1.0, 0.5], labels=["A", "B", "C"], task="classification")
    for b in ("svg", "plotly", "matplotlib"):
        fig = p_view.to_figure(b)
        assert fig is not None
        if b == "matplotlib":
            plt.close(fig)

    # 2. Embeddings
    e_view = sd.embeddings([[1.0, 0.0], [0.0, 1.0]], labels=["v1", "v2"])
    for b in ("svg", "plotly", "matplotlib"):
        fig = e_view.to_figure(b)
        assert fig is not None
        if b == "matplotlib":
            plt.close(fig)

    # 3. Attention
    a_view = sd.attention([[1.0, 0.0], [0.5, 0.5]])
    for b in ("svg", "plotly", "matplotlib"):
        fig = a_view.to_figure(b)
        assert fig is not None
        if b == "matplotlib":
            plt.close(fig)

    # 4. Convolution simulation
    c_view = sd.convolution()
    for b in ("svg", "plotly", "matplotlib"):
        fig = c_view.to_figure(b)
        assert fig is not None
        if b == "matplotlib":
            plt.close(fig)
