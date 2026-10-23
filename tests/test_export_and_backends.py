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


def test_export_unicode_and_default_windows_encoding(tmp_path):
    """Regression test ensuring exports with Unicode characters preserve fidelity without relying on system locale."""
    unicode_title = "Neural α-Flow & λ-Attributions ∑(Δθ)"
    unicode_notes = ["Learning rate η ≈ 0.001", "Gradient norm ||∇L|| ≤ 1.0"]

    view = sd.see([0.1, 0.5, 0.9], backend="svg")
    view.title = unicode_title
    view.notes = unicode_notes

    # SVG export
    svg_path = tmp_path / "unicode_plot.svg"
    view.save(svg_path)
    svg_text = svg_path.read_text(encoding="utf-8")
    assert "α-Flow" in svg_text
    assert "λ-Attributions" in svg_text

    # HTML export
    html_path = tmp_path / "unicode_plot.html"
    view.save(html_path)
    html_text = html_path.read_text(encoding="utf-8")
    assert "α-Flow" in html_text
    assert "λ-Attributions" in html_text
    assert "η ≈ 0.001" in html_text

    # Direct view.to_html() contract
    direct_html = view.to_html()
    assert "α-Flow" in direct_html
    assert "Exported by SeeDL" in direct_html
