"""Tensor audit: original storage, rank/slicing, complex values and framework safety."""

import html
import math
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import seedl as sd
from seedl._dataset_health import _to_flat_floats
from seedl._quantum import apply_gate_to_state, parse_density_matrix, parse_statevector
from seedl._tensor_explorer import build_tensor_explorer_payload, svg_tensor_explorer
from seedl._tensor_utils import memory_offset, tensor_layout
from seedl._tensors import parse_slice_spec

SHAPES = [(), (5,), (2, 3), (2, 3, 4), (2, 3, 4, 5), (2, 2, 3, 4, 5), (0, 4), (2, 0, 3)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("framework", ["numpy", "torch"])
def test_rank_matrix_static_and_interactive(shape, framework):
    array = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    value = array if framework == "numpy" else pytest.importorskip("torch").from_numpy(array)
    for interactive in (False, True):
        view = sd.see(value, interactive=interactive, backend="svg")
        assert view.metadata["Rank"] == len(shape)
        assert view.metadata["Shape"] == str(shape)
        assert "<svg" in view.to_figure()
        if interactive:
            assert view.kind == "tensor_explorer"
            assert "<script" in view._repr_html_()
    sliced = sd.see(value, slice=Ellipsis, backend="svg")
    assert sliced.metadata["Shape"] == str(shape)


@pytest.mark.parametrize(
    "make",
    [
        lambda a: a,
        np.asfortranarray,
        lambda a: a.T,
        lambda a: a[1:, ::2],
        lambda a: a[::-1, ::-2],
        lambda a: np.broadcast_to(a[0, :1], (4, 6)),
    ],
)
def test_numpy_original_storage_layout(make):
    base = np.arange(24, dtype=np.float64).reshape(4, 6)
    value = make(base)
    view = sd.see(value, backend="svg")
    layout = view.metadata
    assert layout["Strides"] == tuple(s // value.itemsize for s in value.strides)
    assert layout["C contiguous"] == value.flags.c_contiguous
    assert layout["Fortran contiguous"] == value.flags.f_contiguous
    offsets = []
    for indices in np.ndindex(value.shape):
        offset = memory_offset(value, indices)
        root_pointer = (
            value.__array_interface__["data"][0] - layout["Storage offset"] * value.itemsize
        )
        pointer = root_pointer + offset * value.itemsize
        expected = value.__array_interface__["data"][0] + sum(
            i * s for i, s in zip(indices, value.strides, strict=True)
        )
        assert pointer == expected
        offsets.append(offset)
    assert layout["Memory offset range"] == (min(offsets), max(offsets))
    data, meta, _ = build_tensor_explorer_payload(value)
    assert meta["Strides"] == layout["Strides"]
    assert data["shape"] == list(value.shape)


def test_empty_and_scalar_storage_offsets():
    empty = np.empty((0, 4))
    assert tensor_layout(empty)["Memory offset range"] is None
    assert memory_offset(np.array(3), ()) == 0
    with pytest.raises(IndexError, match="out of bounds"):
        memory_offset(empty, (0, 0))
    with pytest.raises(IndexError, match="Expected 2 indices"):
        memory_offset(empty, (0,))


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32", "float64"])
@pytest.mark.parametrize("layout_name", ["transpose", "permute", "stepped", "expand"])
def test_torch_original_metadata_before_conversion(dtype_name, layout_name):
    torch = pytest.importorskip("torch")
    base = torch.arange(48, dtype=getattr(torch, dtype_name)).reshape(2, 4, 6).requires_grad_()
    value = {
        "transpose": lambda: base.transpose(0, 2),
        "permute": lambda: base.permute(2, 0, 1),
        "stepped": lambda: base[1:, 1:, ::2],
        "expand": lambda: base[:1, :1, :1].expand(2, 4, 6),
    }[layout_name]()
    before = value.detach().clone()
    for interactive in (False, True):
        view = sd.see(value, interactive=interactive, backend="svg")
        meta = view.metadata
        assert meta["Data type"] == dtype_name
        assert meta["Requires gradients"] is True
        assert meta["Device"] == str(value.device)
        assert meta["Strides"] == value.stride()
        assert meta["Storage offset"] == value.storage_offset()
        assert meta["Is contiguous"] == value.is_contiguous()
        assert "<svg" in view.to_figure()
    assert memory_offset(value, (-1, -1, -1)) == value.storage_offset() + sum(
        (s - 1) * st for s, st in zip(value.shape, value.stride(), strict=True)
    )
    torch.testing.assert_close(value.detach(), before)
    assert base.grad is None


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_accelerator_metadata_preserved(device):
    torch = pytest.importorskip("torch")
    available = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f"{device} accelerator unavailable")
    value = torch.ones((2, 3), device=device, requires_grad=True).T
    for interactive in (False, True):
        view = sd.see(value, interactive=interactive, backend="svg")
        assert view.metadata["Device"] == str(value.device)
        assert view.metadata["Requires gradients"] is True
        assert view.metadata["Strides"] == value.stride()


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((2, 3)),
        np.array([[7.0]]),
        np.array([[np.nan, 1.0], [2.0, 3.0]]),
        np.array([[np.inf, -np.inf], [1.0, 1.0]]),
        np.full((2, 2), np.nan),
        np.array([[-1e308, 1e308]]),
    ],
)
@pytest.mark.parametrize("framework", ["numpy", "torch"])
def test_numerical_cells_and_summaries(values, framework):
    value = values if framework == "numpy" else pytest.importorskip("torch").from_numpy(values)
    view = sd.see(value, backend="svg")
    assert "<svg" in view.to_figure()
    data, _, _ = build_tensor_explorer_payload(value)
    for axis in data["axes"]:
        for frame in axis["frames"]:
            assert all(math.isfinite(frame[key]) for key in ("min", "max", "mean"))


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("shape", [(), (4,), (2, 2), (1, 2, 2), (0, 4)])
@pytest.mark.parametrize("framework", ["numpy", "torch"])
def test_complex_components_are_preserved(dtype, shape, framework):
    array = np.resize(np.array([0, 1j, -1j, 1 + 1j], dtype=dtype), math.prod(shape)).reshape(shape)
    value = (
        array
        if framework == "numpy"
        else pytest.importorskip("torch").from_numpy(array).requires_grad_()
    )
    view = sd.see(value, backend="svg")
    assert view.metadata["Data type"] == np.dtype(dtype).name
    assert "<svg" in view.to_figure()
    components = view.data["complex_components"]
    np.testing.assert_allclose(components["magnitude"], np.abs(array).reshape(-1), atol=1e-6)
    np.testing.assert_allclose(components["probability"], np.abs(array).reshape(-1) ** 2, atol=1e-6)
    if array.size:
        assert components["phase"][0] == 0.0
    data, metadata, _ = build_tensor_explorer_payload(value)
    assert metadata["Display values"] == "magnitude"
    for axis in data["axes"]:
        for frame in axis["frames"]:
            assert "phase" in frame["complex_components"]


@pytest.mark.parametrize(
    "spec",
    [
        1.2,
        True,
        object(),
        np.array([1, 2]),
        slice(None, None, 0),
        slice(0, 2, 1.5),
        slice("bad", 2),
        (Ellipsis, Ellipsis),
        "1:2:0",
        "broken",
    ],
)
def test_invalid_slice_types_are_descriptive_value_errors(spec):
    with pytest.raises(ValueError, match="slice|Slice|index|integer|ellipsis"):
        parse_slice_spec(spec, (3, 4))


@pytest.mark.parametrize("spec", [(3,), (-4,), (0, 0, 0)])
def test_slice_bounds_are_index_errors(spec):
    with pytest.raises(IndexError, match="bounds|Too many"):
        parse_slice_spec(spec, (3, 4))


@pytest.mark.parametrize(
    "spec",
    [
        (),
        Ellipsis,
        -1,
        (Ellipsis, -1),
        (slice(None, None, -1), slice(None, None, -2)),
        (np.int64(-1), slice(None)),
        (slice(1, 3), slice(None, None, -1)),
    ],
)
def test_numpy_and_torch_slice_values_agree(spec):
    torch = pytest.importorskip("torch")
    array = np.arange(12).reshape(3, 4)
    tensor = torch.from_numpy(array)
    expected = np.asarray(array[spec])
    for value in (array, tensor):
        view = sd.see(value, slice=spec, backend="svg")
        assert view.metadata["Shape"] == str(expected.shape)
        actual = view.data.get("values", view.data.get("value"))
        np.testing.assert_array_equal(actual, expected)


def test_low_precision_helpers_do_not_need_tensor_numpy():
    torch = pytest.importorskip("torch")
    for dtype in (torch.bfloat16, torch.float16):
        value = torch.ones((2, 2), dtype=dtype, requires_grad=True)
        assert _to_flat_floats(value) == [1.0] * 4
        rollout = sd.attention_rollout(value.reshape(1, 1, 2, 2), backend="svg")
        np.testing.assert_allclose(rollout.data["rollout_matrix"], [[0.75, 0.25], [0.25, 0.75]])
    assert _to_flat_floats(4.0) == [4.0]


def test_quantum_tensor_ingestion_and_tolerances():
    torch = pytest.importorskip("torch")
    value = torch.tensor([1.0, 0.0], dtype=torch.bfloat16, requires_grad=True)
    assert parse_statevector(value) == [1 + 0j, 0j]
    assert parse_density_matrix(torch.diag(value)) == [[1 + 0j, 0j], [0j, 0j]]
    assert parse_statevector([math.sqrt(1 + 5e-6), 0])
    with pytest.raises(ValueError, match="normalized"):
        parse_statevector([math.sqrt(1 + 3e-5), 0])
    with pytest.raises(ValueError, match="unitary"):
        apply_gate_to_state([1, 0], [[2, 0], [0, 1]])
    with pytest.raises(ValueError, match="2x2"):
        apply_gate_to_state([1, 0], [[1], [0]])


def test_explorer_svg_and_html_escape_user_labels():
    attack = '</text><script>alert("x")</script><text>'
    view = sd.see(np.ones((2, 2, 2)), interactive=True, dim_names=[attack, "b", "c"], backend="svg")
    svg = svg_tensor_explorer(view.data, {**view.metadata, "Title": attack}, view.notes)
    ET.fromstring(svg)
    assert attack not in svg
    assert html.escape(attack) in svg
    assert attack not in view._repr_html_()


def test_no_optional_framework_imports_for_python_inputs():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split('.')[0] in ('torch','numpy','qiskit','plotly','matplotlib'):
            raise ModuleNotFoundError(name=name)
sys.meta_path.insert(0, Block())
import seedl
for value in (42, [1,2,3], [[1,2],[3,4]]):
    assert '<svg' in seedl.see(value, backend='svg').to_figure()
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "values",
    [
        np.empty((0, 4)),
        np.ones((2, 3)),
        np.array([[np.nan, np.inf], [1, -np.inf]]),
        np.array([[0j, 1j], [-1j, 0j]]),
    ],
)
@pytest.mark.parametrize("framework", ["numpy", "torch"])
def test_dataset_arrays_keep_full_shape_and_empty_columns(values, framework):
    from seedl._datasets import inspect_dataset

    value = values if framework == "numpy" else pytest.importorskip("torch").from_numpy(values)
    kind, data, _, _ = inspect_dataset(value)
    assert kind == "dataset_tabular"
    assert data["num_samples"] == values.shape[0]
    assert data["features"] == values.shape[1]
    assert all(
        math.isfinite(item[key]) for item in data["feature_stats"] for key in ("mean", "min", "max")
    )


def test_complex_trace_statistics_use_magnitude():
    torch = pytest.importorskip("torch")
    from seedl._execution_graph import compute_tensor_stats

    value = torch.tensor([0, 1j, -2j], dtype=torch.complex64, requires_grad=True)
    stats, sample, nonfinite, _ = compute_tensor_stats(value, torch=torch)
    assert stats["min"] == 0
    assert stats["max"] == 2
    assert nonfinite == 0
    assert sample == [0, 1, 2]


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("framework", ["numpy", "torch"])
def test_dataset_image_normalization_and_channel_layout(channels, framework):
    from seedl._datasets import inspect_dataset

    array = np.zeros((2, channels, 8, 8), dtype=np.float32)
    array[0, 0, 0, 0] = np.nan
    value = array if framework == "numpy" else pytest.importorskip("torch").from_numpy(array)
    kind, data, _, _ = inspect_dataset(value)
    assert kind == "dataset_images"
    grid = np.asarray(data["pixels"])
    assert grid.shape == ((8, 17, 3) if channels == 3 else (8, 17))
    assert np.isfinite(grid).all()


@pytest.mark.parametrize("samples", [[1e308] * 3, [-1e308, 1e308], [np.nan, 0, np.inf], []])
def test_histograms_constant_extreme_and_nonfinite(samples):
    from seedl._dataset_health import compute_histograms

    p, q, edges = compute_histograms(samples, samples)
    assert sum(p) == pytest.approx(1.0)
    assert sum(q) == pytest.approx(1.0)
    assert all(math.isfinite(v) for v in p + q + edges)


def test_buffer_backed_array_keeps_storage_offset():
    array = np.frombuffer(bytearray(80), dtype=np.float64)[3::2]
    assert tensor_layout(array)["Storage offset"] == 3
    assert memory_offset(array, (1,)) == 5


@pytest.mark.parametrize("framework", ["numpy", "torch"])
def test_complex_nonfinite_counters(framework):
    array = np.array([[complex("nan"), complex("inf")]], dtype=np.complex128)
    value = array if framework == "numpy" else pytest.importorskip("torch").from_numpy(array)
    for view in (sd.see(value, backend="svg"), sd.see(value, interactive=True, backend="svg")):
        assert view.metadata["NaN values"] == 1
        assert view.metadata["Infinite values"] == 1
        assert view.metadata["Non-finite values"] == 2


@pytest.mark.parametrize("value, shape", [(3.0, []), ([1, 2, 3], [3]), ([[1, 2], [3, 4]], [2, 2])])
def test_explorer_payload_python_inputs(value, shape):
    data, _, _ = build_tensor_explorer_payload(value)
    assert data["shape"] == shape


@pytest.mark.parametrize("name", ["max_frames", "max_slice_points"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_explorer_rejects_invalid_sampling_limits(name, value):
    with pytest.raises(ValueError, match="positive integer"):
        build_tensor_explorer_payload(np.ones((2, 2)), **{name: value})


@pytest.mark.parametrize("value", [1j, [0j, 1j, -2j], [[0j, 1j], [-2j, 0j]]])
def test_python_complex_inputs_keep_components(value):
    view = sd.see(value, backend="svg")
    array = np.asarray(value)
    np.testing.assert_allclose(
        view.data["complex_components"]["magnitude"], np.abs(array).reshape(-1)
    )
    assert "<svg" in view.to_figure()


@pytest.mark.parametrize("framework", ["numpy", "torch"])
def test_interactive_image_scales_nonfinite_pixels(framework):
    array = np.full((2, 3, 8, 8), 12.0, dtype=np.float32)
    array[0, 0, 0, 0] = np.nan
    value = array if framework == "numpy" else pytest.importorskip("torch").from_numpy(array)
    data, _, _ = build_tensor_explorer_payload(value)
    pixels = np.asarray(data["axes"][0]["frames"][0]["pixels"])
    assert np.isfinite(pixels).all()
    assert pixels.min() >= 0 and pixels.max() <= 1


def test_extreme_finite_quantum_amplitude_gives_normalization_error():
    with pytest.raises(ValueError, match="normalized"):
        parse_statevector([1e308, 0])
