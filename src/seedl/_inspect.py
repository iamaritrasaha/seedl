"""Object adapters used by the public API. Optional frameworks load on demand."""

from __future__ import annotations

import importlib
import math
import numbers
from collections.abc import Mapping, Sequence
from itertools import islice
from typing import Any

from .exceptions import MissingDependencyError, SeeDLError, UnsupportedObjectError


def _torch_module() -> Any | None:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            return None
        raise


def _numpy_module() -> Any | None:
    try:
        return importlib.import_module("numpy")
    except ModuleNotFoundError as exc:
        if exc.name == "numpy":
            return None
        raise


def _framework_type(value: Any, prefix: str) -> bool:
    return any(cls.__module__.split(".")[0] == prefix for cls in type(value).__mro__)


def is_torch_tensor(value: Any, torch: Any | None = None) -> bool:
    if torch is None and not _framework_type(value, "torch"):
        return False
    torch = torch if torch is not None else _torch_module()
    return bool(torch is not None and isinstance(value, torch.Tensor))


def is_torch_module(value: Any, torch: Any | None = None) -> bool:
    if torch is None and not _framework_type(value, "torch"):
        return False
    torch = torch if torch is not None else _torch_module()
    return bool(torch is not None and isinstance(value, torch.nn.Module))


def is_numpy_array(value: Any, np: Any | None = None) -> bool:
    if np is None and not _framework_type(value, "numpy"):
        return False
    np = np if np is not None else _numpy_module()
    return bool(np is not None and isinstance(value, np.ndarray))


def _fmt(number: Any, precision: int = 5) -> str:
    if number is None:
        return "n/a"
    try:
        value = float(number)
    except (TypeError, ValueError, OverflowError):
        return str(number)
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{precision}g}"


def _display_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "−∞" if number < 0 else "∞"
    return _fmt(number)


def _float_value(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, numbers.Real):
        raise SeeDLError(f"Expected a real number, received {type(value).__name__}.")
    return float(value)


def _metadata_from_stats(
    *,
    shape: tuple[int, ...],
    count: int,
    dtype: str,
    device: str,
    requires_grad: bool,
    minimum: float | None,
    maximum: float | None,
    mean: float | None,
    std: float | None,
    nonfinite: int,
    rank: int | None = None,
    memory_bytes: int | None = None,
    nan_count: int | None = None,
    inf_count: int | None = None,
    sparsity_pct: float | None = None,
) -> dict[str, Any]:
    from ._tensors import format_memory_bytes

    meta: dict[str, Any] = {
        "Shape": str(shape),
        "Elements": count,
        "Data type": dtype,
        "Device": device,
        "Minimum": _fmt(minimum),
        "Maximum": _fmt(maximum),
        "Mean": _fmt(mean),
        "Std. deviation": _fmt(std),
        "Requires gradients": requires_grad,
        "Non-finite values": nonfinite,
    }
    meta["Rank"] = rank if rank is not None else len(shape)
    if memory_bytes is not None:
        meta["Estimated memory"] = format_memory_bytes(memory_bytes)
    if sparsity_pct is not None:
        meta["Sparsity"] = f"{sparsity_pct:.1f}%"
    if nan_count is not None and nan_count > 0:
        meta["NaN values"] = nan_count
    if inf_count is not None and inf_count > 0:
        meta["Infinite values"] = inf_count
    return meta


def _torch_chunks(tensor: Any, chunk_size: int = 1_048_576):
    """Yield bounded flattened views, including for non-contiguous tensors."""
    count = int(tensor.numel())
    if count <= chunk_size or tensor.ndim == 0:
        yield tensor.reshape(-1)
        return
    axis = max(range(tensor.ndim), key=lambda index: int(tensor.shape[index]))
    other_elements = count // int(tensor.shape[axis])
    width = max(1, chunk_size // max(other_elements, 1))
    for start in range(0, int(tensor.shape[axis]), width):
        slices = [slice(None)] * tensor.ndim
        slices[axis] = slice(start, start + width)
        yield from _torch_chunks(tensor[tuple(slices)], chunk_size)


def _torch_statistics(
    tensor: Any,
    chunk_size: int = 1_048_576,
    *,
    return_counts: bool = False,
) -> tuple[Any, ...]:
    """Compute exact-shape summaries in bounded chunks without a full-size temporary."""
    torch = _torch_module()
    assert torch is not None
    is_fp = bool(tensor.dtype.is_floating_point)
    stats_dtype = (
        torch.float64
        if tensor.dtype == torch.float64 or not is_fp
        else torch.float32
    )
    finite_count = 0
    nonfinite = 0
    nan_count = 0
    inf_count = 0
    zero_count = 0
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    m2: float | None = None
    scale = 0.0
    for chunk in _torch_chunks(tensor, chunk_size=chunk_size):
        zero_count += int((chunk == 0).sum().item())
        if is_fp:
            nan_count += int(torch.isnan(chunk).sum().item())
            inf_count += int(torch.isinf(chunk).sum().item())
        values = chunk.to(dtype=stats_dtype)
        mask = torch.isfinite(values)
        nonfinite += int(values.numel()) - int(mask.sum().item())
        valid = values[mask]
        size = int(valid.numel())
        if not size:
            continue
        chunk_min = float(valid.min().item())
        chunk_max = float(valid.max().item())
        chunk_scale = float(valid.abs().max().item())
        if chunk_scale > scale:
            factor = scale / chunk_scale if scale else 0.0
            if finite_count:
                assert mean is not None and m2 is not None
                mean *= factor
                m2 *= factor * factor
            scale = chunk_scale
        normalized = valid / scale if scale else valid
        chunk_mean = float(normalized.mean().item())
        chunk_m2 = float((normalized - chunk_mean).square().sum().item())
        minimum = chunk_min if minimum is None else min(minimum, chunk_min)
        maximum = chunk_max if maximum is None else max(maximum, chunk_max)
        if finite_count == 0:
            mean, m2 = chunk_mean, chunk_m2
            finite_count = size
        else:
            assert mean is not None and m2 is not None
            combined = finite_count + size
            delta = chunk_mean - mean
            m2 += chunk_m2 + delta * delta * finite_count * size / combined
            mean += delta * size / combined
            finite_count = combined
    if nonfinite:
        if return_counts:
            return None, None, None, None, None, nonfinite, nan_count, inf_count, zero_count
        return None, None, None, None, None, nonfinite
    std = None
    if finite_count:
        assert m2 is not None
        assert mean is not None
        mean *= scale
        std = math.sqrt(max(m2 / finite_count, 0.0)) * scale
    if return_counts:
        return minimum, maximum, mean, std, scale if finite_count else None, 0, nan_count, inf_count, zero_count
    return minimum, maximum, mean, std, scale if finite_count else None, 0


def _torch_sample(tensor: Any, max_points: int) -> list[Any]:
    torch = _torch_module()
    assert torch is not None
    count = int(tensor.numel())
    sample_count = min(max_points, count)
    if not sample_count:
        return []
    step = max(1, math.ceil(count / sample_count))
    if tensor.is_contiguous():
        selected = tensor.view(-1)[::step][:sample_count]
    else:
        flat_indices = torch.arange(0, count, step, device=tensor.device)[:sample_count]
        coordinates = []
        for size in reversed(tensor.shape):
            coordinates.append(torch.remainder(flat_indices, int(size)))
            flat_indices = torch.div(flat_indices, int(size), rounding_mode="floor")
        selected = tensor[tuple(reversed(coordinates))]
    return selected.to(device="cpu").tolist()


def _torch_tensor_data(
    tensor: Any, max_points: int, layout: str | None = None
) -> tuple[str, dict[str, Any], dict[str, Any], list[str]]:
    torch = _torch_module()
    assert torch is not None
    if tensor.is_quantized:
        raise SeeDLError(
            "Quantized PyTorch tensors are not supported yet; call dequantize() first."
        )
    if tensor.is_complex():
        raise SeeDLError(
            "Complex tensors are not visualized yet; convert real and imaginary parts separately."
        )
    detached = tensor.detach()
    shape = tuple(int(dim) for dim in detached.shape)
    count = int(detached.numel())
    sparse = detached.layout != torch.strided
    if sparse:
        sparse_tensor = (
            detached if detached.layout == torch.sparse_coo else detached.to_sparse_coo()
        )
        values = sparse_tensor.coalesce().values()
        notes = [
            "Sparse tensor view summarizes stored values only; implicit zeros are not "
            "included in the distribution."
        ]
    else:
        values = detached
        notes = []
    dtype = str(detached.dtype).replace("torch.", "")
    device = str(detached.device)
    requires_grad = bool(tensor.requires_grad)
    (
        minimum,
        maximum,
        mean,
        std,
        max_abs,
        nonfinite,
        nan_count,
        inf_count,
        zero_count,
    ) = _torch_statistics(values, return_counts=True)
    if nonfinite:
        if nan_count and inf_count:
            notes.append(
                f"{nonfinite} non-finite value(s) present ({nan_count} NaN, {inf_count} Inf); "
                "aggregate statistics are unavailable."
            )
        elif nan_count:
            notes.append(
                f"{nan_count} NaN value(s) are present; aggregate statistics are unavailable."
            )
        elif inf_count:
            notes.append(
                f"{inf_count} infinite value(s) are present; aggregate statistics are unavailable."
            )
        else:
            notes.append(
                f"{nonfinite} non-finite value(s) are present; aggregate statistics are unavailable."
            )

    from ._tensors import (
        format_sparsity,
        is_ambiguous_image_layout,
        make_image_grid,
        resolve_image_layout,
    )

    sparsity_pct = (zero_count / count) * 100.0 if count else 0.0
    if sparsity_pct >= 50.0 and count > 10 and not sparse:
        notes.append(f"High sparsity: {format_sparsity(zero_count, count)} are zero.")

    total_bytes = count * int(detached.element_size()) if hasattr(detached, "element_size") else count * 4

    metadata = _metadata_from_stats(
        shape=shape,
        count=count,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        std=std,
        nonfinite=nonfinite,
        rank=len(shape),
        memory_bytes=total_bytes,
        nan_count=nan_count,
        inf_count=inf_count,
        sparsity_pct=sparsity_pct,
    )
    metadata["Maximum absolute value"] = _fmt(max_abs)
    if sparse:
        metadata["Stored values"] = int(values.numel())
    if count == 0:
        notes.append("This tensor is empty, so there are no values to plot.")
        return "distribution", {"sample": [], "shape": shape}, metadata, notes

    if sparse:
        sample = _torch_sample(values, max_points) if values.numel() else []
        notes.append("Sparse tensors are shown as a distribution rather than a dense heatmap.")
        return "distribution", {"sample": sample, "shape": shape}, metadata, notes

    if len(shape) == 0:
        return (
            "scalar",
            {
                "value": float(detached.item()),
                "display_value": _display_scalar(detached.item()),
                "caption": f"0-dimensional tensor · {dtype} · {device}",
            },
            metadata,
            notes,
        )

    image_candidate = resolve_image_layout(shape, layout)
    if image_candidate is not None:
        mode, channels = image_candidate
        if mode in ("channels_first", "channels_last"):
            pixels, channels = _torch_image_preview(detached, (mode, channels), max_points)
            notes.append(
                "Image layout was inferred from a 1, 3, or 4 channel axis. "
                "Values are scaled for display; tensor data is unchanged."
                if layout is None
                else f"Visualized using explicit {layout.upper()} image layout."
            )
            notes.append(
                f"Preview: {len(pixels)} rows × {len(pixels[0])} columns; "
                "float [0, 1] values retained, integers [0, 255] divided by 255, "
                "other ranges min-max scaled; non-finite components shown as mid-gray."
            )
            return (
                "image",
                {"pixels": pixels, "channels": channels, "shape": shape, "layout": mode},
                metadata,
                notes,
            )
        if mode in ("batch_channels_first", "batch_channels_last"):
            batch_size = shape[0]
            preview_count = min(batch_size, 16)
            batch_slice = detached[:preview_count]
            if mode == "batch_channels_first":
                batch_slice = batch_slice.permute(0, 2, 3, 1)
            sub_h, sub_w = int(batch_slice.shape[1]), int(batch_slice.shape[2])
            sub_pts = max(1, max_points // max(1, preview_count))
            ystep, xstep = _preview_steps(sub_h, sub_w, 64, sub_pts)
            downsampled = batch_slice[:, ::ystep, ::xstep][:, :64, :64]
            scaled = _image_scale(downsampled, detached.dtype).to(device="cpu").tolist()
            grid_pixels, out_chan, rows, cols = make_image_grid(
                scaled, channels, max_images=preview_count
            )
            notes.append(
                f"Batch image grid: showing {preview_count} of {batch_size} images tiled in a {rows}x{cols} grid."
            )
            return (
                "image",
                {
                    "pixels": grid_pixels,
                    "channels": channels,
                    "shape": shape,
                    "layout": mode,
                    "batch_grid": (rows, cols),
                },
                metadata,
                notes,
            )

    sampled = _torch_sample(detached, max_points)
    if len(shape) == 1:
        if count > len(sampled):
            notes.append(f"Showing {len(sampled):,} evenly spaced sample values out of {count:,}.")
        return (
            "sequence",
            {
                "values": sampled,
                "x": list(range(0, count, math.ceil(count / max_points))),
                "label": "Tensor value",
                "shape": shape,
            },
            metadata,
            notes,
        )
    if len(shape) == 2:
        row_step, col_step = _preview_steps(shape[0], shape[1], 80, max_points)
        preview = detached[::row_step, ::col_step][:80, :80].to(device="cpu").tolist()
        if len(preview) != shape[0] or (preview and len(preview[0]) != shape[1]):
            notes.append("The heatmap is a downsampled preview of the full matrix.")
        return (
            "matrix",
            {
                "values": preview,
                "shape": shape,
                "row_indices": list(range(0, shape[0], row_step)),
                "column_indices": list(range(0, shape[1], col_step)),
            },
            metadata,
            notes,
        )

    if layout is None and is_ambiguous_image_layout(shape):
        if len(shape) == 4:
            notes.append(
                f"Shape {shape} has an ambiguous channel position (both first and last dimensions are valid channels); "
                "specify layout='NCHW' or layout='NHWC' explicitly to render as an image."
            )
        else:
            notes.append(
                f"Shape {shape} has an ambiguous channel position (both first and last dimensions are valid channels); "
                "specify layout='CHW' or layout='HWC' explicitly to render as an image."
            )

    if count > len(sampled):
        notes.append(f"Showing {len(sampled):,} evenly spaced sample values out of {count:,}.")
    notes.append(
        "Higher-dimensional data is shown as a value distribution from an evenly spaced sample."
    )
    return "distribution", {"sample": sampled, "shape": shape}, metadata, notes


def _preview_steps(height: int, width: int, side_limit: int, max_points: int) -> tuple[int, int]:
    ystep, xstep = max(1, math.ceil(height / side_limit)), max(1, math.ceil(width / side_limit))
    while math.ceil(height / ystep) * math.ceil(width / xstep) > max_points:
        if math.ceil(height / ystep) >= math.ceil(width / xstep):
            ystep += 1
        else:
            xstep += 1
    return ystep, xstep


def _image_scale(values: Any, dtype: Any) -> Any:
    torch = _torch_module()
    assert torch is not None
    work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    values = values.to(dtype=work_dtype)
    if not values.numel():
        return values
    if (
        dtype != torch.bool
        and not dtype.is_floating_point
        and bool(((values >= 0) & (values <= 255)).all())
    ):
        return values / 255.0
    finite = torch.isfinite(values)
    if not bool(finite.all().item()):
        valid = values[finite]
        if not valid.numel():
            return torch.full_like(values, 0.5)
    else:
        valid = values
    low, high = float(valid.min().item()), float(valid.max().item())
    if low >= 0 and high <= 1:
        scaled = values
    elif high == low:
        scaled = torch.full_like(values, 0.5)
    else:
        scale = max(abs(low), abs(high))
        scaled = (values / scale - low / scale) / (high / scale - low / scale)
    return torch.where(finite, scaled, torch.full_like(values, 0.5))


def _torch_image_preview(
    tensor: Any, layout: tuple[str, int], max_points: int
) -> tuple[list[Any], int]:
    mode, channels = layout
    if mode == "channels_first":
        tensor = tensor.permute(1, 2, 0)
    height, width = int(tensor.shape[0]), int(tensor.shape[1])
    ystep, xstep = _preview_steps(height, width, 128, max_points)
    pixels = tensor[::ystep, ::xstep][:128, :128]
    return _image_scale(pixels, tensor.dtype).to(device="cpu").tolist(), channels


def _numpy_array_data(
    array: Any, max_points: int, layout: str | None = None
) -> tuple[str, dict[str, Any], dict[str, Any], list[str]]:
    np = _numpy_module()
    assert np is not None
    if np.iscomplexobj(array):
        raise SeeDLError(
            "Complex arrays are not visualized yet; convert real and imaginary parts separately."
        )
    shape = tuple(int(dim) for dim in array.shape)
    count = int(array.size)
    dtype = str(array.dtype)
    notes: list[str] = []
    if not (np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)):
        raise SeeDLError(f"Only real numeric NumPy arrays are supported, received dtype {dtype!r}.")
    (
        minimum,
        maximum,
        mean,
        std,
        nonfinite,
        nan_count,
        inf_count,
        zero_count,
    ) = _numpy_statistics(array, return_counts=True)
    if nonfinite:
        if nan_count and inf_count:
            notes.append(
                f"{nonfinite} non-finite value(s) present ({nan_count} NaN, {inf_count} Inf); "
                "aggregate statistics are unavailable."
            )
        elif nan_count:
            notes.append(
                f"{nan_count} NaN value(s) are present; aggregate statistics are unavailable."
            )
        elif inf_count:
            notes.append(
                f"{inf_count} infinite value(s) are present; aggregate statistics are unavailable."
            )
        else:
            notes.append(
                f"{nonfinite} non-finite value(s) are present; aggregate statistics are unavailable."
            )

    from ._tensors import (
        format_sparsity,
        is_ambiguous_image_layout,
        make_image_grid,
        resolve_image_layout,
    )

    sparsity_pct = (zero_count / count) * 100.0 if count else 0.0
    if sparsity_pct >= 50.0 and count > 10:
        notes.append(f"High sparsity: {format_sparsity(zero_count, count)} are zero.")

    total_bytes = count * int(array.itemsize)

    metadata = _metadata_from_stats(
        shape=shape,
        count=count,
        dtype=dtype,
        device="CPU",
        requires_grad=False,
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        std=std,
        nonfinite=nonfinite,
        rank=len(shape),
        memory_bytes=total_bytes,
        nan_count=nan_count,
        inf_count=inf_count,
        sparsity_pct=sparsity_pct,
    )
    if count == 0:
        notes.append("This array is empty, so there are no values to plot.")
        return "distribution", {"sample": [], "shape": shape}, metadata, notes
    if len(shape) == 0:
        return (
            "scalar",
            {
                "value": float(array.item()),
                "display_value": _display_scalar(array.item()),
                "caption": f"0-dimensional NumPy array · {dtype}",
            },
            metadata,
            notes,
        )

    image_candidate = resolve_image_layout(shape, layout)
    if image_candidate is not None:
        mode, channels = image_candidate
        if mode in ("channels_first", "channels_last"):
            pixels, channels = _numpy_image_preview(array, (mode, channels), max_points)
            notes.append(
                "Image layout was inferred from a 1, 3, or 4 channel axis. "
                "Values are scaled for display; array data is unchanged."
                if layout is None
                else f"Visualized using explicit {layout.upper()} image layout."
            )
            notes.append(
                f"Preview: {len(pixels)} rows × {len(pixels[0])} columns; "
                "float [0, 1] values retained, integers [0, 255] divided by 255, "
                "other ranges min-max scaled; non-finite components shown as mid-gray."
            )
            return (
                "image",
                {"pixels": pixels, "channels": channels, "shape": shape, "layout": mode},
                metadata,
                notes,
            )
        if mode in ("batch_channels_first", "batch_channels_last"):
            batch_size = shape[0]
            preview_count = min(batch_size, 16)
            batch_slice = array[:preview_count]
            if mode == "batch_channels_first":
                batch_slice = np.moveaxis(batch_slice, 1, -1)
            sub_h, sub_w = int(batch_slice.shape[1]), int(batch_slice.shape[2])
            sub_pts = max(1, max_points // max(1, preview_count))
            ystep, xstep = _preview_steps(sub_h, sub_w, 64, sub_pts)
            downsampled = batch_slice[:, ::ystep, ::xstep][:, :64, :64].astype(np.float64, copy=False)
            scaled = _numpy_image_scale(downsampled, array.dtype).tolist()
            grid_pixels, out_chan, rows, cols = make_image_grid(
                scaled, channels, max_images=preview_count
            )
            notes.append(
                f"Batch image grid: showing {preview_count} of {batch_size} images tiled in a {rows}x{cols} grid."
            )
            return (
                "image",
                {
                    "pixels": grid_pixels,
                    "channels": channels,
                    "shape": shape,
                    "layout": mode,
                    "batch_grid": (rows, cols),
                },
                metadata,
                notes,
            )

    if len(shape) == 1:
        step = max(1, math.ceil(count / max_points))
        sample = array.flat[::step][:max_points].tolist()
        if len(sample) < count:
            notes.append(f"Showing {len(sample):,} evenly spaced sample values out of {count:,}.")
        return (
            "sequence",
            {
                "values": sample,
                "x": list(range(0, count, step)),
                "label": "Array value",
                "shape": shape,
            },
            metadata,
            notes,
        )
    if len(shape) == 2:
        step_y, step_x = _preview_steps(shape[0], shape[1], 80, max_points)
        preview = array[::step_y, ::step_x][:80, :80].tolist()
        if step_y > 1 or step_x > 1:
            notes.append("The heatmap is a downsampled preview of the full matrix.")
        return (
            "matrix",
            {
                "values": preview,
                "shape": shape,
                "row_indices": list(range(0, shape[0], step_y)),
                "column_indices": list(range(0, shape[1], step_x)),
            },
            metadata,
            notes,
        )

    if layout is None and is_ambiguous_image_layout(shape):
        if len(shape) == 4:
            notes.append(
                f"Array shape {shape} has an ambiguous image layout (both first and last dimensions can be channels); "
                "specify layout='NCHW' or layout='NHWC' to visualize as an image."
            )
        else:
            notes.append(
                f"Array shape {shape} has an ambiguous image layout (both first and last dimensions can be channels); "
                "specify layout='CHW' or layout='HWC' to visualize as an image."
            )

    step = max(1, math.ceil(count / max_points))
    sample = array.flat[::step][:max_points].tolist()
    notes.append(
        "Higher-dimensional data is shown as a value distribution from an evenly spaced sample."
    )
    if len(sample) < count:
        notes.append(f"Showing {len(sample):,} sampled values out of {count:,}.")
    return "distribution", {"sample": sample, "shape": shape}, metadata, notes


def _numpy_statistics(
    array: Any,
    chunk_size: int = 1_048_576,
    *,
    return_counts: bool = False,
) -> tuple[Any, ...]:
    """Compute numeric summaries with bounded temporary arrays."""
    np = _numpy_module()
    assert np is not None
    count = int(array.size)
    if not count:
        if return_counts:
            return None, None, None, None, 0, 0, 0, 0
        return None, None, None, None, 0
    if array.ndim == 0 or count <= chunk_size:
        chunks = (array.reshape(-1),)
    else:

        def iter_chunks(part):
            if part.size <= chunk_size:
                yield part.reshape(-1)
                return
            split_axis = max(range(part.ndim), key=lambda i: int(part.shape[i]))
            part_width = max(1, chunk_size // max(int(part.size) // int(part.shape[split_axis]), 1))
            for start in range(0, int(part.shape[split_axis]), part_width):
                index = [slice(None)] * part.ndim
                index[split_axis] = slice(start, start + part_width)
                yield from iter_chunks(part[tuple(index)])

        chunks = iter_chunks(array)
    finite_count = nonfinite = nan_count = inf_count = zero_count = 0
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    m2: float | None = None
    scale = 0.0
    for chunk in chunks:
        zero_count += int(np.count_nonzero(chunk == 0))
        if np.issubdtype(chunk.dtype, np.floating):
            nan_count += int(np.isnan(chunk).sum())
            inf_count += int(np.isinf(chunk).sum())
        values = np.asarray(chunk, dtype=np.float64)
        finite = np.isfinite(values)
        nonfinite += int(values.size - int(finite.sum()))
        valid = values[finite]
        size = int(valid.size)
        if not size:
            continue
        chunk_min = float(valid.min())
        chunk_max = float(valid.max())
        chunk_scale = float(np.abs(valid).max())
        if chunk_scale > scale:
            factor = scale / chunk_scale if scale else 0.0
            if finite_count:
                assert mean is not None and m2 is not None
                mean *= factor
                m2 *= factor * factor
            scale = chunk_scale
        normalized = valid / scale if scale else valid
        chunk_mean = float(normalized.mean())
        chunk_m2 = float(np.square(normalized - chunk_mean).sum())
        minimum = chunk_min if minimum is None else min(minimum, chunk_min)
        maximum = chunk_max if maximum is None else max(maximum, chunk_max)
        if finite_count == 0:
            mean, m2 = chunk_mean, chunk_m2
            finite_count = size
        else:
            assert mean is not None and m2 is not None
            combined = finite_count + size
            delta = chunk_mean - mean
            m2 += chunk_m2 + delta * delta * finite_count * size / combined
            mean += delta * size / combined
            finite_count = combined
    if nonfinite:
        if return_counts:
            return None, None, None, None, nonfinite, nan_count, inf_count, zero_count
        return None, None, None, None, nonfinite
    std = None
    if finite_count:
        assert m2 is not None
        assert mean is not None
        mean *= scale
        std = math.sqrt(max(m2 / finite_count, 0.0)) * scale
    if return_counts:
        return minimum, maximum, mean, std, 0, nan_count, inf_count, zero_count
    return minimum, maximum, mean, std, 0


def _numpy_image_scale(array: Any, dtype: Any) -> Any:
    np = _numpy_module()
    assert np is not None
    integer_image = np.issubdtype(dtype, np.integer)
    finite = np.isfinite(array)
    if not finite.any():
        return np.full_like(array, 0.5)
    valid = array[finite]
    low, high = float(np.min(valid)), float(np.max(valid))
    if integer_image and low >= 0 and high <= 255:
        scaled = array / 255.0
    elif low >= 0 and high <= 1:
        scaled = array
    elif high == low:
        scaled = np.full_like(array, 0.5)
    else:
        scale = max(abs(low), abs(high))
        scaled = (array / scale - low / scale) / (high / scale - low / scale)
    return np.where(finite, scaled, 0.5)


def _numpy_image_preview(
    array: Any, layout: tuple[str, int], max_points: int
) -> tuple[list[Any], int]:
    np = _numpy_module()
    assert np is not None
    orig_dtype = array.dtype
    mode, channels = layout
    if mode == "channels_first":
        array = np.moveaxis(array, 0, -1)
    height, width = array.shape[:2]
    ystep, xstep = _preview_steps(height, width, 128, max_points)
    array = array[::ystep, ::xstep][:128, :128].astype(np.float64, copy=False)
    scaled = _numpy_image_scale(array, orig_dtype)
    return scaled.tolist(), channels


def _numeric_sequence(values: Any, *, label: str) -> list[float]:
    torch = _torch_module() if _framework_type(values, "torch") else None
    if torch is not None and isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise SeeDLError(
                f"{label} must be one-dimensional, received shape {tuple(values.shape)}."
            )
        values = values.detach().to(device="cpu").tolist()
    np = _numpy_module() if _framework_type(values, "numpy") else None
    if np is not None and isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise SeeDLError(f"{label} must be one-dimensional, received shape {values.shape}.")
        values = values.tolist()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise SeeDLError(f"{label} must be a sequence of real numbers.")
    result = [_float_value(value) for value in values]
    return result


def _history_data(
    history: Mapping[Any, Any], max_points: int
) -> tuple[dict[str, Any], str, list[str]]:
    metrics: dict[str, list[float]] = {}
    epochs: list[float] | None = None
    notes = ["Curves are diagnostic clues; they are not guarantees about model quality."]
    for key, values in history.items():
        if not isinstance(key, str):
            raise SeeDLError(
                "Training history keys must be strings such as 'loss' or 'val_accuracy'."
            )
        lowered = key.lower()
        if lowered in {"epoch", "epochs"}:
            epochs = _numeric_sequence(values, label=f"history[{key!r}]")
            continue
        metrics[key] = _numeric_sequence(values, label=f"history[{key!r}]")
    if not metrics:
        raise SeeDLError(
            "Training history is empty; add one or more metric sequences such as {'loss': [...]}."
        )
    if any(not values for values in metrics.values()):
        raise SeeDLError("Training history metric sequences must contain at least one value.")
    if epochs is not None and not epochs:
        raise SeeDLError("The epoch sequence must contain at least one value.")
    if epochs is not None and any(not math.isfinite(value) for value in epochs):
        raise SeeDLError("Epoch labels must be finite real numbers.")
    if any(not math.isfinite(value) for values in metrics.values() for value in values):
        notes.append("Non-finite metric values are kept as gaps in the curves.")
    groups: dict[str, list[dict[str, Any]]] = {}
    metric_names = set(metrics)
    for name, values in metrics.items():
        low = name.lower()
        if low.startswith("validation_"):
            base, label = name[len("validation_") :], "validation"
        elif low.startswith("val_"):
            base, label = name[4:], "validation"
        elif low.endswith("_validation"):
            base, label = name[:-11], "validation"
        elif low.endswith("_val"):
            base, label = name[:-4], "validation"
        else:
            base, label = name, "training"
        # Preserve standalone validation metrics as their own plot group.
        group = base if base in metric_names or label == "training" else name
        if group not in metric_names and label == "validation":
            group = base or name
        x = (
            list(epochs[: len(values)])
            if epochs is not None and len(epochs) == len(values)
            else list(range(1, len(values) + 1))
        )
        if epochs is not None and len(epochs) != len(values):
            notes.append(
                f"'{name}' has {len(values)} points; epoch labels do not match, "
                "so index labels are used."
            )
        groups.setdefault(group, []).append({"name": name, "role": label, "x": x, "values": values})
    output_groups: list[dict[str, Any]] = [
        {"name": name, "series": series} for name, series in groups.items()
    ]
    for metric_group in output_groups:
        metric_name = metric_group["name"].lower()
        if "loss" not in metric_name and "error" not in metric_name:
            continue
        training = next(
            (series for series in metric_group["series"] if series["role"] == "training"),
            None,
        )
        validation = next(
            (series for series in metric_group["series"] if series["role"] == "validation"),
            None,
        )
        if training is None or validation is None:
            continue
        paired = [
            (train, valid)
            for train, valid in zip(training["values"], validation["values"], strict=False)
            if math.isfinite(train) and math.isfinite(valid)
        ]
        if training["x"] != validation["x"] or len(paired) < 3:
            continue
        train_start, train_end = paired[0][0], paired[-1][0]
        validation_values = [pair[1] for pair in paired]
        validation_minimum = min(validation_values[:-1])
        rise = validation_values[-1] - validation_minimum
        train_fell = train_end < train_start
        validation_rose = rise > max(abs(validation_minimum) * 0.05, 1e-8)
        if train_fell and validation_rose:
            metric_label = "loss" if "loss" in metric_name else "error"
            notes.append(
                f"Possible overfitting clue in {metric_group['name']}: training {metric_label} "
                f"fell while validation {metric_label} rose after a lower point. "
                "Check the curves and validation setup."
            )
            break
    for group_data in output_groups:
        for series in group_data["series"]:
            if len(series["values"]) > max_points:
                step = math.ceil(len(series["values"]) / max_points)
                series["values"] = series["values"][::step]
                series["x"] = series["x"][::step]
                notes.append(
                    f"'{series['name']}' is an evenly spaced preview; spikes may be missed."
                )
    lengths = [len(values) for values in metrics.values()]
    summary = f"{len(metrics)} metric(s), {min(lengths)}–{max(lengths)} epochs"
    return {"groups": output_groups, "metrics": metrics}, summary, notes


def _shape_text(value: Any, torch: Any) -> str:
    if isinstance(value, torch.Tensor):
        return str(tuple(int(size) for size in value.shape))
    if isinstance(value, Mapping):
        parts = [f"{key}: {_shape_text(item, torch)}" for key, item in islice(value.items(), 4)]
        return "{" + ", ".join(parts) + (", …" if len(value) > 4 else "") + "}"
    if isinstance(value, tuple):
        parts = [_shape_text(item, torch) for item in value[:4]]
        if len(value) == 1:
            parts[0] += ","
        return "(" + ", ".join(parts) + (", …" if len(value) > 4 else "") + ")"
    if isinstance(value, list):
        parts = [_shape_text(item, torch) for item in value[:4]]
        return "[" + ", ".join(parts) + (", …" if len(value) > 4 else "") + "]"
    return type(value).__name__


def _clone_input(value: Any, torch: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_input(item, torch) for item in value)
    if isinstance(value, list):
        return [_clone_input(item, torch) for item in value]
    if isinstance(value, dict):
        return {key: _clone_input(item, torch) for key, item in value.items()}
    return value


def _call_model(model: Any, example_input: Any) -> Any:
    if isinstance(example_input, dict):
        return model(**example_input)
    if isinstance(example_input, tuple):
        return model(*example_input)
    return model(example_input)


def _parameter_numel(parameter: Any) -> int | None:
    try:
        return int(parameter.numel())
    except (RuntimeError, ValueError):
        return None


def _run_model_observation(
    model: Any, example_input: Any, *, capture_values: bool, max_points: int
) -> dict[int, dict[str, Any]]:
    torch = _torch_module()
    if torch is None:
        raise MissingDependencyError(
            "Model and activation inspection require PyTorch. "
            "Install it with `pip install 'seedl[torch]'`."
        )
    if any(
        isinstance(parameter, torch.nn.parameter.UninitializedParameter)
        for parameter in model.parameters()
    ):
        raise SeeDLError(
            "This model has uninitialized lazy parameters. Run it once yourself to "
            "initialize them, then call SeeDL again."
        )
    modules = list(model.named_modules())
    sample_limit = max(1, max_points // max(len(modules), 1)) if capture_values else max_points
    remaining_sample_points = max_points
    states = [(module, bool(module.training)) for _, module in modules]
    buffers = [
        (
            module,
            dict(module._buffers),
            set(module._non_persistent_buffers_set),
            {
                name: buffer.detach().clone() if buffer is not None else None
                for name, buffer in module._buffers.items()
            },
        )
        for _, module in modules
    ]
    observations: dict[int, dict[str, Any]] = {}
    handles = []

    def make_pre(module_id: int):
        def hook(_module: Any, args: Any) -> None:
            observation = observations.setdefault(module_id, {})
            observation["input_shape"] = _shape_text(args, torch)

        return hook

    def make_post(module_id: int, name: str, module: Any):
        def hook(_module: Any, _args: Any, output: Any) -> None:
            nonlocal remaining_sample_points
            observation = observations.setdefault(module_id, {})
            observation["output_shape"] = _shape_text(output, torch)
            if capture_values:
                tensor = _first_tensor(output, torch)
                if tensor is not None:
                    old_sample_count = len(observation.get("sample", []))
                    available = remaining_sample_points + old_sample_count
                    current_limit = min(sample_limit, available)
                    payload = _torch_tensor_data(tensor, max(current_limit, 1))
                    sample = (
                        _torch_sample(
                            tensor.detach()
                            if tensor.layout == torch.strided
                            else tensor.detach().to_sparse_coo().coalesce().values(),
                            current_limit,
                        )
                        if current_limit
                        else []
                    )
                    remaining_sample_points = available - len(sample)
                    observation["sample"] = sample
                    observation["stats"] = payload[2]
                    observation["tensor_shape"] = tuple(int(size) for size in tensor.shape)
                    observation["module_name"] = name
                    observation["module_type"] = type(module).__name__

                    # Diagnostics: dead units and saturation
                    dead_fraction = 0.0
                    sat_note = None
                    if tensor.numel() > 0:
                        try:
                            t_flat = tensor.detach().reshape(-1).to(dtype=torch.float32)
                            t_fin = t_flat[torch.isfinite(t_flat)]
                            if t_fin.numel() > 0:
                                zeros = int((t_fin == 0).sum().item())
                                dead_fraction = zeros / float(t_fin.numel())
                                mod_lower = type(module).__name__.lower()
                                if "relu" in mod_lower and dead_fraction > 0.8:
                                    sat_note = f"{dead_fraction * 100:.1f}% zero/dead units"
                                elif "sigmoid" in mod_lower:
                                    s_cnt = int(((t_fin < 0.05) | (t_fin > 0.95)).sum().item())
                                    if s_cnt / float(t_fin.numel()) > 0.5:
                                        sat_note = f"{(s_cnt / float(t_fin.numel())) * 100:.1f}% saturated (<0.05 or >0.95)"
                                elif "tanh" in mod_lower:
                                    t_cnt = int(((t_fin < -0.95) | (t_fin > 0.95)).sum().item())
                                    if t_cnt / float(t_fin.numel()) > 0.5:
                                        sat_note = f"{(t_cnt / float(t_fin.numel())) * 100:.1f}% saturated (<-0.95 or >0.95)"
                        except Exception:
                            pass
                    observation["dead_fraction"] = dead_fraction
                    observation["saturation_note"] = sat_note

                    # Feature maps for 4D conv tensors (N, C, H, W)
                    feat_preview = None
                    if tensor.ndim == 4 and tensor.shape[1] >= 1 and min(tensor.shape[2:]) >= 4:
                        try:
                            from ._tensors import make_image_grid

                            t_det = tensor.detach()
                            num_c = min(int(tensor.shape[1]), 16)
                            c_slices = []
                            for c in range(num_c):
                                s2d = t_det[0, c].to(dtype=torch.float32)
                                s_min, s_max = float(s2d.min().item()), float(s2d.max().item())
                                if s_max > s_min:
                                    norm = ((s2d - s_min) / (s_max - s_min)).cpu().tolist()
                                else:
                                    norm = torch.full_like(s2d, 0.5).cpu().tolist()
                                c_slices.append(norm)
                            grid, _, _, _ = make_image_grid(c_slices, 1, max_images=16)
                            feat_preview = grid
                        except Exception:
                            pass
                    observation["feature_maps"] = feat_preview
                else:
                    old_sample_count = len(observation.pop("sample", []))
                    remaining_sample_points += old_sample_count
                    for key in ("stats", "tensor_shape", "module_name", "module_type"):
                        observation.pop(key, None)

        return hook

    try:
        for name, module in modules:
            handles.append(module.register_forward_pre_hook(make_pre(id(module))))
            handles.append(module.register_forward_hook(make_post(id(module), name, module)))
        model.eval()
        copied_input = _clone_input(example_input, torch)
        with torch.no_grad():
            _call_model(model, copied_input)
    except SeeDLError:
        raise
    except Exception as exc:
        raise SeeDLError(
            "Could not inspect this model with the supplied input. "
            "Check the input structure, shapes, dtype, and device. "
            f"Original error: {exc}"
        ) from exc
    finally:
        try:
            for handle in handles:
                handle.remove()
        finally:
            try:
                with torch.no_grad():
                    for module, original_buffers, non_persistent, saved_values in buffers:
                        module._non_persistent_buffers_set = non_persistent
                        module._buffers = original_buffers
                        for name, saved in saved_values.items():
                            buffer = original_buffers[name]
                            if buffer is not None and saved is not None:
                                if buffer.shape != saved.shape:
                                    buffer.resize_(saved.shape)
                                buffer.copy_(saved)
            finally:
                for module, training in states:
                    module.training = training
    return observations


def _first_tensor(value: Any, torch: Any) -> Any | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor(item, torch)
            if found is not None:
                return found
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item, torch)
            if found is not None:
                return found
    return None


def _model_data(model: Any, example_input: Any | None) -> tuple[dict[str, Any], str, list[str]]:
    torch = _torch_module()
    assert torch is not None
    observations = (
        _run_model_observation(model, example_input, capture_values=False, max_points=1)
        if example_input is not None
        else {}
    )
    direct_counts: dict[int, tuple[int | None, int | None]] = {}
    for _, module in model.named_modules():
        parameters = list(module.parameters(recurse=False))
        counts = [_parameter_numel(parameter) for parameter in parameters]
        direct_counts[id(module)] = (
            None
            if any(count is None for count in counts)
            else sum(count for count in counts if count is not None),
            None
            if any(
                count is None
                for count, parameter in zip(counts, parameters, strict=True)
                if parameter.requires_grad
            )
            else sum(
                count
                for count, parameter in zip(counts, parameters, strict=True)
                if parameter.requires_grad and count is not None
            ),
        )
    layers = []
    all_params = list(model.parameters())
    seen: set[int] = set()
    total_counts: list[int | None] = []
    trainable_counts: list[int | None] = []
    for parameter in all_params:
        if id(parameter) not in seen:
            seen.add(id(parameter))
            count = _parameter_numel(parameter)
            total_counts.append(count)
            if parameter.requires_grad:
                trainable_counts.append(count)
    total = (
        None
        if any(count is None for count in total_counts)
        else sum(count for count in total_counts if count is not None)
    )
    trainable = (
        None
        if any(count is None for count in trainable_counts)
        else sum(count for count in trainable_counts if count is not None)
    )
    for name, module in model.named_modules():
        param_count, trainable_count = direct_counts[id(module)]
        observation = observations.get(id(module), {})
        layers.append(
            {
                "name": name,
                "depth": name.count(".") + (1 if name else 0),
                "type": type(module).__name__,
                "parameters": param_count,
                "trainable": trainable_count,
                "input_shape": observation.get("input_shape", "—"),
                "output_shape": observation.get("output_shape", "—"),
            }
        )
    total_label = f"{total:,}" if total is not None else "unknown number of"
    trainable_label = f"{trainable:,}" if trainable is not None else "unknown number of"
    summary = f"{len(layers)} module(s) · {total_label} parameter(s) · {trainable_label} trainable"
    notes = [
        "Per-layer counts include direct parameters only; shared parameters may appear in "
        "multiple rows. Total counts deduplicate shared parameters."
    ]
    if example_input is None:
        notes.append(
            "Only the module hierarchy is shown. Pass input=... to record input and output shapes."
        )
        if total is None:
            notes.append(
                "Parameter counts are unknown until lazy layers are initialized by your "
                "own forward pass."
            )
    else:
        notes.append(
            "An eval-mode forward pass under no_grad() was run once with the supplied input; "
            "temporary hooks were removed and ordinary buffer updates were restored."
        )
        notes.append(
            "This is a module hierarchy, not an execution graph. Shapes describe the last "
            "observed call for this example; keyword input shapes are not captured."
        )
    if len(layers) > 100:
        notes.append(
            "All backends show the first 100 modules; all rows remain available in view.data."
        )
    return {"layers": layers, "parameters": total, "trainable": trainable}, summary, notes


def _activation_data(
    model: Any, example_input: Any, max_points: int
) -> tuple[dict[str, Any], str, list[str]]:
    observations = _run_model_observation(
        model, example_input, capture_values=True, max_points=max_points
    )
    layers = []
    for name, module in model.named_modules():
        observation = observations.get(id(module), {})
        if "tensor_shape" not in observation:
            continue
        stats = observation["stats"]
        layers.append(
            {
                "name": name or "(root)",
                "type": type(module).__name__,
                "shape": str(observation["tensor_shape"]),
                "sample": observation.get("sample", [])[:max_points],
                "mean": stats.get("Mean", "n/a"),
                "std": stats.get("Std. deviation", "n/a"),
                "max_abs": stats.get("Maximum absolute value"),
                "finite": "yes"
                if stats.get("Non-finite values") == 0
                else f"no ({stats.get('Non-finite values')})",
                "status": "available",
                "dead_fraction": observation.get("dead_fraction", 0.0),
                "saturation_note": observation.get("saturation_note"),
                "feature_maps": observation.get("feature_maps"),
            }
        )
    notes = [
        "Captured in one eval-mode forward pass under no_grad(); training flags and "
        "ordinary buffer updates were restored. Custom forward side effects are not sandboxed.",
        "Each row shows the first tensor output from that module; repeated uses show the last "
        "observed output. Sample values share a max_points budget across modules.",
    ]
    dead_layers = [
        layer["name"]
        for layer in layers
        if layer.get("dead_fraction", 0.0) > 0.8
        and any(
            act in layer.get("type", "").lower()
            for act in ("relu", "gelu", "silu", "sigmoid", "tanh")
        )
    ]
    if dead_layers:
        notes.append(
            f"Dead unit alert (heuristic): {len(dead_layers)} activation layer(s) have >80% inactive zeros: {', '.join(dead_layers[:5])}"
        )
    sat_layers = [layer["name"] for layer in layers if layer.get("saturation_note")]
    if sat_layers:
        notes.append(
            f"Saturation detected in {len(sat_layers)} layer(s): {', '.join(sat_layers[:5])}"
        )
    if not layers:
        notes.append("No tensor outputs were captured for this input.")
    if len(layers) > 20:
        notes.append("Activation plots show the first 20 modules; tables show the first 100.")
    return {"layers": layers}, f"Captured {len(layers)} module activation(s)", notes


def _gradient_data(model: Any, max_points: int) -> tuple[dict[str, Any], str, list[str]]:
    torch = _torch_module()
    if torch is None:
        raise MissingDependencyError(
            "Gradient inspection requires PyTorch. Install it with `pip install 'seedl[torch]'`."
        )
    layers = []
    parameters = list(model.named_parameters())
    sample_limit = max(1, max_points // max(len(parameters), 1))
    remaining_sample_points = max_points
    present = 0
    sparse_count = 0
    zero_count = 0
    vanishing_count = 0
    exploding_count = 0
    for name, parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            layers.append(
                {
                    "name": name,
                    "shape": str(tuple(parameter.shape)),
                    "status": "no gradient",
                    "mean": None,
                    "std": None,
                    "max_abs": None,
                    "norm": None,
                    "l2_norm": None,
                    "finite": "—",
                    "trainable": bool(parameter.requires_grad),
                    "sample": [],
                }
            )
            continue
        present += 1
        gradient = gradient.detach()
        if gradient.is_complex():
            raise SeeDLError(
                f"Gradient for {name!r} is complex; convert its real and imaginary parts first."
            )
        if gradient.layout == torch.sparse_coo:
            sparse_count += 1
            values = gradient.coalesce().values().reshape(-1)
        elif gradient.layout != torch.strided:
            sparse_count += 1
            values = gradient.to_sparse_coo().coalesce().values().reshape(-1)
        else:
            values = gradient
        count = int(values.numel())
        current_limit = min(sample_limit, remaining_sample_points)
        sample = _torch_sample(values, current_limit) if count and current_limit else []
        remaining_sample_points -= len(sample)
        _, _, mean, std, max_abs, nonfinite, *_ = _torch_statistics(values)
        l2_norm_val = None
        if nonfinite == 0 and count:
            mean_value, std_value, max_abs_value = mean, std, max_abs
            try:
                l2_norm_val = float(values.float().norm().item())
            except Exception:
                l2_norm_val = None
            if max_abs_value is not None:
                if max_abs_value == 0.0:
                    zero_count += 1
                elif max_abs_value < 1e-7:
                    vanishing_count += 1
                elif max_abs_value > 100.0:
                    exploding_count += 1
        else:
            mean_value = std_value = max_abs_value = None
        layers.append(
            {
                "name": name,
                "shape": str(tuple(parameter.shape)),
                "status": "available",
                "mean": _fmt(mean_value),
                "std": _fmt(std_value),
                "max_abs": max_abs_value,
                "norm": l2_norm_val,
                "l2_norm": _fmt(l2_norm_val),
                "finite": "yes" if nonfinite == 0 else f"no ({nonfinite})",
                "trainable": bool(parameter.requires_grad),
                "sample": sample,
            }
        )
    notes = (
        []
        if present
        else [
            "No parameter gradients are available. Run loss.backward() before calling "
            "sd.gradients(model)."
        ]
    )
    if present:
        notes.append(
            "Gradient inspection is read-only and never calls backward() or an optimizer step. "
            "Sample values share a max_points budget across parameters."
        )
    nonfinite_param_count = sum(1 for layer in layers if "no (" in str(layer.get("finite", "")))
    if nonfinite_param_count > 0:
        notes.append(
            f"Non-finite gradients detected: {nonfinite_param_count} parameter tensor(s) contain NaN or Inf values."
        )
    if vanishing_count > 0:
        notes.append(
            f"Vanishing gradients detected (heuristic): {vanishing_count} parameter tensor(s) have maximum absolute gradient below 1e-7."
        )
    if exploding_count > 0:
        notes.append(
            f"Exploding gradients detected (heuristic): {exploding_count} parameter tensor(s) have maximum absolute gradient exceeding 100."
        )
    if zero_count > 0:
        notes.append(f"{zero_count} parameter tensor(s) have all-zero gradients.")
    if sparse_count:
        notes.append(
            "Sparse gradient summaries use stored values; implicit zeros are not included."
        )
    if len(layers) > 20:
        notes.append(
            "Gradient plots show the first 20 parameter tensors; tables show the first 100."
        )
    return (
        {
            "layers": layers,
            "present_count": present,
            "zero_count": zero_count,
            "vanishing_count": vanishing_count,
            "exploding_count": exploding_count,
        },
        f"Gradients available for {present} of {len(layers)} parameter tensor(s)",
        notes,
    )


def inspect_object(
    obj: Any, *, example_input: Any | None, max_points: int, layout: str | None = None
) -> tuple[str, str, dict[str, Any], dict[str, Any], list[str]]:
    """Return the semantic view payload for a supported object."""
    torch = _torch_module() if _framework_type(obj, "torch") else None
    if torch is not None and isinstance(obj, torch.nn.Module):
        data, summary, notes = _model_data(obj, example_input)
        data["summary"] = summary
        return "model", f"{type(obj).__name__} architecture", data, {"Summary": summary}, notes
    if torch is not None and isinstance(obj, torch.Tensor):
        kind, data, metadata, notes = _torch_tensor_data(obj, max_points, layout=layout)
        return kind, "Tensor visualization", data, metadata, notes
    np = _numpy_module() if _framework_type(obj, "numpy") else None
    if np is not None and isinstance(obj, np.ndarray):
        kind, data, metadata, notes = _numpy_array_data(obj, max_points, layout=layout)
        return kind, "NumPy array visualization", data, metadata, notes
    if np is not None and isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, Mapping):
        data, summary, notes = _history_data(obj, max_points)
        data["summary"] = summary
        return "training", "Training history", data, {}, notes
    if isinstance(obj, numbers.Real):
        value = float(obj)
        metadata = {"Type": type(obj).__name__}
        if math.isfinite(value):
            metadata["Status"] = "finite"
        else:
            metadata["Status"] = "non-finite"
        return (
            "scalar",
            "Numeric value",
            {
                "value": value,
                "display_value": _display_scalar(obj),
                "caption": type(obj).__name__,
            },
            metadata,
            [] if math.isfinite(value) else ["The value is not finite; it is displayed as text."],
        )
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        if not obj:
            raise SeeDLError("An empty sequence has no values to visualize.")
        if all(isinstance(item, Sequence) and not isinstance(item, (str, bytes)) for item in obj):
            rows = [_numeric_sequence(row, label=f"row {index}") for index, row in enumerate(obj)]
            widths = {len(row) for row in rows}
            if len(widths) != 1:
                raise SeeDLError("Nested numeric sequences must form a rectangular matrix.")
            if not rows[0]:
                raise SeeDLError("A numeric matrix must have at least one column.")
            row_step, col_step = _preview_steps(len(rows), len(rows[0]), 80, max_points)
            row_indices = list(range(0, len(rows), row_step))
            column_indices = list(range(0, len(rows[0]), col_step))
            if row_step > 1 or col_step > 1:
                rows = [row[::col_step] for row in rows[::row_step]]
                notes = ["The heatmap is an evenly spaced preview of this nested sequence."]
            else:
                notes = []
            return (
                "matrix",
                "Numeric matrix",
                {
                    "values": rows,
                    "shape": (len(obj), len(obj[0])),
                    "row_indices": row_indices,
                    "column_indices": column_indices,
                },
                {},
                notes,
            )
        values = _numeric_sequence(obj, label="sequence")
        indices = list(range(len(values)))
        if len(values) > max_points:
            stride = math.ceil(len(values) / max_points)
            values = values[::stride][:max_points]
            indices = indices[::stride][:max_points]
            notes = [f"Showing {len(values):,} evenly spaced values from the sequence."]
        else:
            notes = []
        return (
            "sequence",
            "Numeric sequence",
            {"values": values, "x": indices, "label": "Value"},
            {},
            notes,
        )
    supported = (
        "PyTorch tensors and modules, NumPy arrays, real numbers, numeric sequences, "
        "and metric dictionaries"
    )
    raise UnsupportedObjectError(
        f"Cannot visualize {type(obj).__name__}. SeeDL supports {supported}; "
        "install optional dependencies for PyTorch or NumPy objects."
    )


def activation_payload(
    model: Any, example_input: Any, max_points: int
) -> tuple[dict[str, Any], str, list[str]]:
    torch = _torch_module()
    if torch is None or not isinstance(model, torch.nn.Module):
        raise SeeDLError("sd.activations() expects a PyTorch nn.Module as its first argument.")
    return _activation_data(model, example_input, max_points)


def gradient_payload(model: Any, max_points: int) -> tuple[dict[str, Any], str, list[str]]:
    torch = _torch_module()
    if torch is None or not isinstance(model, torch.nn.Module):
        raise SeeDLError("sd.gradients() expects a PyTorch nn.Module.")
    return _gradient_data(model, max_points)
