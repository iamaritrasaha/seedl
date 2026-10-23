"""Tests for SeeDL's Interactive Multidimensional Tensor Explorer.

Validates client-side slider scrubbing, dimension axis switching, bounded memory transfers,
semantic dimension names, zero-dependency HTML rendering, SVG fallback, and XSS sanitization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import seedl as sd
from seedl._tensor_explorer import (
    build_tensor_explorer_payload,
    validate_dim_names,
)
from seedl.exceptions import SeeDLError

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ==============================================================================
# Unit Tests: validate_dim_names
# ==============================================================================


def test_validate_dim_names_valid():
    """Verify validation of dimension names across list and tuple sequences."""
    names = ["batch", "channels", "height", "width"]
    assert validate_dim_names(names, 4, (2, 3, 32, 32)) == names
    assert validate_dim_names(tuple(names), 4, (2, 3, 32, 32)) == names


def test_validate_dim_names_none_defaults():
    """Verify default generated names when dim_names is None."""
    assert validate_dim_names(None, 3, (10, 20, 30)) == ["dim_0", "dim_1", "dim_2"]


def test_validate_dim_names_strips_whitespace():
    """Verify whitespace is stripped from dimension names."""
    result = validate_dim_names([" batch ", " channel "], 2, (10, 20))
    assert result == ["batch", "channel"]


def test_validate_dim_names_rejects_non_sequence():
    """Verify rejection of non-list/tuple inputs like strings or dicts."""
    with pytest.raises(SeeDLError, match="dim_names must be a list or tuple of strings"):
        validate_dim_names("batch, channels", 2, (10, 20))
    with pytest.raises(SeeDLError, match="dim_names must be a list or tuple of strings"):
        validate_dim_names({"b": 1, "c": 2}, 2, (10, 20))


def test_validate_dim_names_rank_mismatch():
    """Verify rejection when the count of dimension names does not match tensor rank."""
    with pytest.raises(SeeDLError, match="dim_names has 3 names, but tensor has rank 4"):
        validate_dim_names(["b", "c", "h"], 4, (1, 3, 32, 32))


def test_validate_dim_names_invalid_elements():
    """Verify rejection of empty strings or non-string elements."""
    with pytest.raises(SeeDLError, match="must be a non-empty string"):
        validate_dim_names(["batch", ""], 2, (10, 20))
    with pytest.raises(SeeDLError, match="must be a non-empty string"):
        validate_dim_names(["batch", "   "], 2, (10, 20))
    with pytest.raises(SeeDLError, match="must be a non-empty string"):
        validate_dim_names(["batch", 123], 2, (10, 20))  # type: ignore[list-item]


# ==============================================================================
# Unit Tests: build_tensor_explorer_payload
# ==============================================================================


def test_build_tensor_explorer_payload_numpy_3d():
    """Verify bounded slice payload extraction for 3D NumPy array."""
    arr = np.linspace(-10.0, 10.0, 1000, dtype=np.float32).reshape(10, 10, 10)
    data, metadata, notes = build_tensor_explorer_payload(
        arr,
        dim_names=["depth", "height", "width"],
        max_frames=5,
    )

    assert data["interactive"] is True
    assert data["rank"] == 3
    assert data["shape"] == [10, 10, 10]
    assert data["dim_names"] == ["depth", "height", "width"]
    assert len(data["axes"]) == 3
    for ax in data["axes"]:
        assert ax["sampled_count"] <= 5
        assert len(ax["frames"]) <= 5
        for fr in ax["frames"]:
            assert "index" in fr
            assert "min" in fr
            assert "max" in fr
            assert "mean" in fr
            assert "values" in fr

    assert metadata["Rank"] == 3
    assert metadata["Shape"] == "(10, 10, 10)"
    assert metadata["Dimension names"] == "depth, height, width"
    assert any("Interactive slice explorer" in n for n in notes)


def test_build_tensor_explorer_payload_sparsity_and_nan():
    """Verify payload detects sparsity and non-finite values properly."""
    arr = np.zeros((10, 20, 20), dtype=np.float32)
    arr[0, 0, 0] = np.nan
    data, metadata, notes = build_tensor_explorer_payload(arr)

    assert metadata["NaN values"] == 1
    assert data["global_stats"]["nan_count"] == 1
    assert any("NaN value(s)" in n for n in notes)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_build_tensor_explorer_payload_torch_4d():
    """Verify 4D image batch payload with PyTorch tensor."""
    tensor = torch.randn(4, 3, 32, 32)
    data, metadata, notes = build_tensor_explorer_payload(
        tensor,
        dim_names=["batch", "channels", "height", "width"],
        max_frames=4,
    )

    assert data["rank"] == 4
    assert data["shape"] == [4, 3, 32, 32]
    # Slicing along Axis 0 (batch) produces (3, 32, 32) -> treated as image slices
    batch_axis = data["axes"][0]
    assert batch_axis["name"] == "batch"
    assert len(batch_axis["frames"]) == 4
    for fr in batch_axis["frames"]:
        assert fr["type"] == "image"
        assert "pixels" in fr


@pytest.mark.skipif(not HAS_TORCH or not torch.cuda.is_available(), reason="GPU not available")
def test_build_tensor_explorer_payload_gpu_device():
    """Verify tensor explorer on ROCm/CUDA GPU tensor preserves device and bounds memory."""
    device = torch.device("cuda:0")
    tensor = torch.ones(8, 16, 16, device=device)
    data, metadata, notes = build_tensor_explorer_payload(tensor, max_frames=8)

    assert "cuda" in data["device"].lower()
    assert metadata["Device"] == str(device)
    assert len(data["axes"]) >= 1


def test_build_tensor_explorer_payload_rejects_rank_less_than_two():
    """Verify payload builder rejects 1D arrays or scalars."""
    with pytest.raises(SeeDLError, match="requires at least 2 dimensions"):
        build_tensor_explorer_payload(np.array([1, 2, 3]))


# ==============================================================================
# Integration Tests: sd.see(interactive=True / view="interactive")
# ==============================================================================


def test_see_tensor_interactive_flag():
    """Verify sd.see(tensor, interactive=True) launches the Interactive Tensor Explorer."""
    arr = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    view = sd.see(arr, interactive=True)

    assert view.kind == "tensor_explorer"
    assert "Interactive exploration" in view.summary
    assert "shape (4, 4, 4)" in view.summary
    assert view.data["interactive"] is True
    assert "axes" in view.data


def test_see_tensor_view_interactive():
    """Verify sd.see(tensor, view='interactive') launches the explorer."""
    arr = np.zeros((5, 10, 10), dtype=np.float32)
    view = sd.see(arr, view="interactive")

    assert view.kind == "tensor_explorer"
    assert view.metadata["Shape"] == "(5, 10, 10)"
    assert view.data["shape"] == [5, 10, 10]


def test_see_tensor_view_explorer_alias():
    """Verify view='explorer' is a recognized alias for interactive tensor exploration."""
    arr = np.zeros((3, 12, 12), dtype=np.float32)
    view = sd.see(arr, view="explorer")
    assert view.kind == "tensor_explorer"


def test_see_tensor_interactive_with_custom_dim_names():
    """Verify semantic dimension names pass through to metadata and explorer payload."""
    arr = np.ones((2, 6, 8, 8), dtype=np.float32)
    names = ["trial", "feature", "y", "x"]
    view = sd.see(arr, interactive=True, dim_names=names)

    assert view.metadata["Dimension names"] == "trial, feature, y, x"
    assert view.data["dim_names"] == names
    assert view.data["axes"][0]["name"] == "trial"
    assert view.data["axes"][1]["name"] == "feature"


def test_see_tensor_static_with_dim_names():
    """Verify dim_names attaches to static inspection metadata without interactive=True."""
    arr = np.zeros((3, 10, 10), dtype=np.float32)
    view = sd.see(arr, dim_names=["depth", "height", "width"])

    assert view.kind != "tensor_explorer"
    assert view.metadata["Dimension names"] == "depth, height, width"
    assert view.data["dim_names"] == ["depth", "height", "width"]


def test_see_tensor_interactive_with_slice():
    """Verify slice= combines cleanly with interactive exploration when remaining rank >= 2."""
    arr = np.zeros((4, 6, 12, 12), dtype=np.float32)
    view = sd.see(arr, slice="0, :, :, :", interactive=True)

    assert view.kind == "tensor_explorer"
    assert view.metadata["Shape"] == "(6, 12, 12)"
    assert view.metadata["Original shape"] == "(4, 6, 12, 12)"
    assert any("Showing slice" in n for n in view.notes)


def test_see_tensor_interactive_slice_collapse_rank():
    """Verify slicing down to 1D rejects interactive exploration with clear error."""
    arr = np.zeros((4, 4, 4), dtype=np.float32)
    with pytest.raises(SeeDLError, match="requires at least 2 dimensions"):
        sd.see(arr, slice="0, 0, :", view="interactive")


def test_see_tensor_invalid_max_frames():
    """Verify max_frames <= 0 or non-int raises ValueError."""
    arr = np.zeros((4, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="max_frames must be a positive integer"):
        sd.see(arr, interactive=True, max_frames=0)
    with pytest.raises(ValueError, match="max_frames must be a positive integer"):
        sd.see(arr, interactive=True, max_frames=-5)


def test_see_rejects_dim_names_on_non_tensor():
    """Verify dim_names raises SeeDLError on non-tensor inputs."""
    with pytest.raises(SeeDLError, match="The dim_names= argument is only supported"):
        sd.see({"key": "val"}, dim_names=["a", "b"])

    with pytest.raises(SeeDLError, match="The dim_names= argument is only supported"):
        sd.see([1, 2, 3], dim_names=["a"])


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_see_rejects_dim_names_and_interactive_view_on_module():
    """Verify dim_names and view='interactive' reject PyTorch nn.Module."""
    mod = torch.nn.Linear(5, 2)
    with pytest.raises(SeeDLError, match="The dim_names= argument is only supported"):
        sd.see(mod, dim_names=["in", "out"])
    with pytest.raises(SeeDLError, match="view='interactive' is only supported"):
        sd.see(mod, view="interactive")


# ==============================================================================
# HTML Export, Canvas Rendering & Mimebundle Tests
# ==============================================================================


def test_tensor_explorer_to_html_contains_canvas_and_controls():
    """Verify complete HTML export includes HTML5 Canvas, sliders, and zero external JS."""
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    view = sd.see(arr, interactive=True)
    html_content = view.to_html()

    assert "<!DOCTYPE html>" in html_content
    assert "<canvas" in html_content
    assert 'type="range"' in html_content
    assert "ste-axis-buttons" in html_content
    assert "ste-select" in html_content
    assert "Viridis" in html_content
    assert "Coolwarm" in html_content
    assert "Interactive Tensor Explorer" in html_content
    # Crucial requirement: zero external script tags
    assert 'src="http' not in html_content
    assert "plotly.js" not in html_content


def test_tensor_explorer_file_save(tmp_path: Path):
    """Verify saving to HTML file on disk."""
    arr = np.zeros((3, 8, 8), dtype=np.float32)
    view = sd.see(arr, interactive=True)
    target = tmp_path / "explorer.html"
    saved = view.save(target)

    assert saved == target
    assert target.exists()
    assert target.stat().st_size > 1000
    content = target.read_text(encoding="utf-8")
    assert "seedl-tensor-explorer" in content


def test_tensor_explorer_notebook_representations():
    """Verify Jupyter notebook MIME bundles and HTML representations."""
    arr = np.zeros((4, 8, 8), dtype=np.float32)
    view = sd.see(arr, interactive=True)

    bundle = view._repr_mimebundle_()
    assert "text/html" in bundle
    assert "text/plain" in bundle
    assert '<div id="seedl-tensor-explorer' in bundle["text/html"]

    repr_html = view._repr_html_()
    assert repr_html is not None
    assert "<canvas" in repr_html


def test_tensor_explorer_svg_fallback():
    """Verify vector SVG fallback renders without error."""
    arr = np.arange(36, dtype=np.float32).reshape(6, 6)
    view = sd.see(arr, interactive=True)
    svg_str = view.to_figure("svg")

    assert svg_str.startswith("<svg")
    assert "</svg>" in svg_str
    assert "Interactive Tensor Explorer" in svg_str


def test_tensor_explorer_xss_protection():
    """Verify XSS strings in dimension names or notes are escaped in HTML export."""
    arr = np.zeros((4, 4), dtype=np.float32)
    malicious_name = "<script>alert('xss')</script>"
    view = sd.see(arr, interactive=True, dim_names=[malicious_name, "safe_dim"])
    html_content = view.to_html()

    # The raw unescaped script tag should NOT exist in the HTML structure
    assert "<script>alert('xss')</script>" not in html_content
    # It must be either JSON-encoded (\\u003c) or HTML-escaped (&lt;)
    assert "\\u003cscript\\u003e" in html_content or "&lt;script&gt;" in html_content
