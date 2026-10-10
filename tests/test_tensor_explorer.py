"""Tests for multidimensional tensor exploration, image grids, and memory estimation."""

from __future__ import annotations

import pytest

import seedl as sd
from seedl._tensors import (
    detect_image_layout,
    format_memory_bytes,
    format_sparsity,
    make_channel_grid,
    make_image_grid,
)


def test_format_memory_bytes():
    assert format_memory_bytes(500) == "500 B"
    assert format_memory_bytes(2048) == "2.0 KB"
    assert format_memory_bytes(5 * 1024 * 1024) == "5.0 MB"
    assert format_memory_bytes(3 * 1024 * 1024 * 1024) == "3.00 GB"


def test_format_sparsity():
    assert format_sparsity(0, 100) == "0.0% zeros"
    assert format_sparsity(50, 100) == "50.0% zeros"
    assert format_sparsity(10, 0) == "0.0% zeros"


def test_detect_image_layout():
    assert detect_image_layout((3, 32, 32)) == ("channels_first", 3)
    assert detect_image_layout((32, 32, 3)) == ("channels_last", 3)
    assert detect_image_layout((8, 3, 32, 32)) == ("batch_channels_first", 3)
    assert detect_image_layout((8, 32, 32, 3)) == ("batch_channels_last", 3)
    assert detect_image_layout((10, 20)) is None
    assert detect_image_layout((2, 5, 10, 20, 30)) is None


def test_make_image_grid():
    # 4 images of 4x4 grayscale
    images = [[[0.5 for _ in range(4)] for _ in range(4)] for _ in range(4)]
    grid, channels, rows, cols = make_image_grid(images, 1, max_images=4)
    assert rows == 2
    assert cols == 2
    # grid height should be 2 * 4 + 1 = 9
    assert len(grid) == 9
    assert len(grid[0]) == 9


def test_make_channel_grid():
    # 1 image of 4x4 RGB
    img = [[[0.2, 0.4, 0.6] for _ in range(4)] for _ in range(4)]
    grid, channels, rows, cols = make_channel_grid(img, 3)
    assert channels == 1
    assert rows >= 1
    assert cols >= 1


def test_tensor_slicing_with_see():
    np = pytest.importorskip("numpy")
    arr = np.arange(24).reshape(2, 3, 4)
    # View a 2D slice
    view = sd.see(arr, slice=(0, slice(None), slice(None)), backend="svg")
    assert view.kind == "matrix"
    assert any("Showing slice" in note for note in view.notes)


def test_torch_tensor_4d_batch_image():
    torch = pytest.importorskip("torch")
    # Batch of 4 RGB images
    batch = torch.rand(4, 3, 16, 16)
    view = sd.see(batch, backend="svg")
    # Should render as image or distribution preview
    assert view.kind in {"image", "dataset_images", "distribution"}
    assert view.data is not None
