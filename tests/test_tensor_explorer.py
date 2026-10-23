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


def test_tensor_rank_and_dimension_metadata():
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")

    # PyTorch 3D
    t = torch.randn(2, 5, 7)
    v_t = sd.see(t, backend="svg")
    assert v_t.metadata["Rank"] == 3
    assert v_t.metadata["Shape"] == "(2, 5, 7)"
    assert v_t.metadata["Elements"] == 70
    assert "Estimated memory" in v_t.metadata

    # NumPy 4D
    arr = np.zeros((2, 3, 4, 5), dtype=np.float32)
    v_arr = sd.see(arr, backend="svg")
    assert v_arr.metadata["Rank"] == 4
    assert v_arr.metadata["Shape"] == "(2, 3, 4, 5)"
    assert v_arr.metadata["Elements"] == 120
    assert v_arr.metadata["Sparsity"] == "100.0%"
    assert any("High sparsity" in note for note in v_arr.notes)


def test_tensor_slice_selection_and_string_syntax():
    np = pytest.importorskip("numpy")
    arr = np.arange(60).reshape(3, 4, 5)

    # String slice selecting a 2D matrix
    v1 = sd.see(arr, slice="1, :, :", backend="svg")
    assert v1.kind == "matrix"
    assert v1.metadata["Rank"] == 2
    assert v1.metadata["Shape"] == "(4, 5)"
    assert v1.metadata["Original shape"] == "(3, 4, 5)"
    assert v1.data["original_shape"] == (3, 4, 5)

    # String slice selecting a 1D sequence
    v2 = sd.see(arr, slice="0, 1, :", backend="svg")
    assert v2.kind == "sequence"
    assert v2.metadata["Rank"] == 1
    assert v2.metadata["Shape"] == "(5,)"

    # Ellipsis slice syntax
    v3 = sd.see(arr, slice="..., 0", backend="svg")
    assert v3.kind == "matrix"
    assert v3.metadata["Shape"] == "(3, 4)"


def test_tensor_slice_errors_bounds_and_dimensions():
    torch = pytest.importorskip("torch")
    from seedl.exceptions import SeeDLError

    t = torch.zeros(4, 8)

    # Out of bounds on axis 0
    with pytest.raises(SeeDLError, match="out of bounds"):
        sd.see(t, slice="10, :")

    # Negative out of bounds
    with pytest.raises(SeeDLError, match="out of bounds"):
        sd.see(t, slice="-5, :")

    # Too many slice dimensions for rank
    with pytest.raises(SeeDLError, match="Too many slice indices"):
        sd.see(t, slice="0, 0, 0")

    # Invalid slice string syntax
    with pytest.raises(SeeDLError, match="Expected an integer or slice syntax"):
        sd.see(t, slice="invalid_index")


def test_explicit_channel_layouts_chw_and_hwc():
    torch = pytest.importorskip("torch")
    from seedl.exceptions import SeeDLError

    # 3D CHW
    t_chw = torch.rand(3, 20, 20)
    v_chw = sd.see(t_chw, layout="CHW", backend="svg")
    assert v_chw.kind == "image"
    assert v_chw.data["channels"] == 3
    assert v_chw.data["layout"] == "channels_first"

    # 3D HWC
    t_hwc = torch.rand(20, 20, 3)
    v_hwc = sd.see(t_hwc, layout="HWC", backend="svg")
    assert v_hwc.kind == "image"
    assert v_hwc.data["channels"] == 3
    assert v_hwc.data["layout"] == "channels_last"

    # 4D NCHW batch
    t_nchw = torch.rand(4, 3, 16, 16)
    v_nchw = sd.see(t_nchw, layout="NCHW", backend="svg")
    assert v_nchw.kind == "image"
    assert v_nchw.data["channels"] == 3
    assert "batch_grid" in v_nchw.data

    # Error when layout rank does not match tensor rank
    with pytest.raises(SeeDLError, match="requires a 4D batch tensor"):
        sd.see(t_chw, layout="NCHW")

    with pytest.raises(SeeDLError, match="requires a 3D tensor"):
        sd.see(t_nchw, layout="CHW")

    # Error when channel count is unsupported for images (e.g. 7 channels)
    t_7chan = torch.rand(7, 16, 16)
    with pytest.raises(SeeDLError, match="image visualization requires 1"):
        sd.see(t_7chan, layout="CHW")

    # Error on unknown layout name
    with pytest.raises(SeeDLError, match="Unsupported layout"):
        sd.see(t_chw, layout="XYZ")


def test_ambiguous_channel_layout_handling():
    torch = pytest.importorskip("torch")

    # Shape (3, 32, 3) is ambiguous: axis 0 is 3 and axis 2 is 3
    ambiguous = torch.rand(3, 32, 3)

    # Without explicit layout, SeeDL must NOT silently guess!
    v = sd.see(ambiguous, backend="svg")
    assert v.kind != "image"  # Falls back to distribution rather than guessing
    assert any("ambiguous" in note.lower() for note in v.notes)

    # With explicit layout, ambiguity is resolved cleanly
    v_explicit = sd.see(ambiguous, layout="CHW", backend="svg")
    assert v_explicit.kind == "image"
    assert v_explicit.data["layout"] == "channels_first"


def test_tensor_statistics_nan_inf_sparsity():
    torch = pytest.importorskip("torch")

    # Create tensor with 2 NaNs, 3 Infs, 10 zeros, 5 regular numbers
    data = [float("nan")] * 2 + [float("inf")] * 2 + [float("-inf")] * 1 + [0.0] * 10 + [1.5] * 5
    t = torch.tensor(data, dtype=torch.float32)

    v = sd.see(t, backend="svg")
    assert v.metadata["Non-finite values"] == 5
    assert v.metadata["NaN values"] == 2
    assert v.metadata["Infinite values"] == 3
    assert v.metadata["Sparsity"] == "50.0%"
    assert any("2 NaN, 3 Inf" in note for note in v.notes)


def test_rocm_gpu_tensor_explorer_behavior():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm GPU device not available")

    gpu_t = torch.tensor([[1.0, 0.0, 3.0], [0.0, 5.0, 0.0]], device="cuda:0")
    v = sd.see(gpu_t, backend="svg")

    assert v.metadata["Device"] == "cuda:0"
    assert v.metadata["Rank"] == 2
    assert v.metadata["Shape"] == "(2, 3)"
    assert v.metadata["Sparsity"] == "50.0%"
    assert gpu_t.device.type == "cuda"

    # Slice on GPU
    v_sliced = sd.see(gpu_t, slice="0, :", backend="svg")
    assert v_sliced.kind == "sequence"
    assert v_sliced.metadata["Rank"] == 1
    assert v_sliced.metadata["Device"] == "cuda:0"
    assert gpu_t.device.type == "cuda"


def test_large_tensor_chunked_statistics_and_memory():
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")

    # 2.5 million elements exceeds the 1M chunk size in _torch_chunks
    t_large = torch.ones(2_500_000, dtype=torch.float32)
    t_large[0] = 5.0
    t_large[-1] = -3.0

    v = sd.see(t_large, backend="svg")
    assert v.metadata["Elements"] == 2_500_000
    assert v.metadata["Rank"] == 1
    assert "9.5 MB" in v.metadata["Estimated memory"]
    assert float(v.metadata["Minimum"]) == -3.0
    assert float(v.metadata["Maximum"]) == 5.0
    assert float(v.metadata["Mean"]) == pytest.approx(1.0, rel=1e-3)
    assert any("evenly spaced sample values" in note for note in v.notes)

    # NumPy large array
    arr_large = np.zeros(1_500_000, dtype=np.int32)
    v_np = sd.see(arr_large, backend="svg")
    assert v_np.metadata["Elements"] == 1_500_000
    assert v_np.metadata["Sparsity"] == "100.0%"
    assert "5.7 MB" in v_np.metadata["Estimated memory"]


