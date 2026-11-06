"""Framework-neutral conversion and original storage-layout inspection."""

from __future__ import annotations

import math
import operator
from typing import Any


def to_native(value: Any) -> Any:
    """Detach and transfer tensors before conversion, including bfloat16/complex.

    Native lists avoid NumPy's unsupported PyTorch dtype conversions entirely.
    """
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "resolve_conj"):
        value = value.resolve_conj().resolve_neg()
    return value.tolist() if hasattr(value, "tolist") else value


def _contiguous(shape: tuple[int, ...], strides: tuple[int, ...], *, fortran: bool) -> bool:
    if 0 in shape:
        return True
    expected = 1
    axes = range(len(shape)) if fortran else reversed(range(len(shape)))
    for axis in axes:
        if shape[axis] > 1:
            if strides[axis] != expected:
                return False
            expected *= shape[axis]
    return True


def tensor_layout(value: Any) -> dict[str, Any]:
    """Read layout from the original object; offsets and strides are in elements.

    The offset range describes the backing storage span, never a flattened copy.
    Empty tensors have no addressable offsets. Broadcast axes can alias storage.
    """
    shape = tuple(int(s) for s in value.shape)
    if hasattr(value, "stride"):
        # Sparse tensor layouts have no dense strides or storage offsets.
        if str(value.layout) != "torch.strided":
            return {"Layout": str(value.layout)}
        strides = tuple(int(s) for s in value.stride())
        itemsize = int(value.element_size())
        offset = int(value.storage_offset())
        c_contiguous = bool(value.is_contiguous())
        f_contiguous = _contiguous(shape, strides, fortran=True)
    else:
        itemsize = int(value.itemsize)
        strides = tuple(int(s) // itemsize for s in value.strides)
        root = value
        pointer = int(value.__array_interface__["data"][0])
        root_pointer = pointer
        seen: set[int] = set()
        while getattr(root, "base", None) is not None and id(root) not in seen:
            seen.add(id(root))
            root = root.base
            if hasattr(root, "__array_interface__"):
                root_pointer = int(root.__array_interface__["data"][0])
        offset = (pointer - root_pointer) // itemsize
        c_contiguous = bool(value.flags.c_contiguous)
        f_contiguous = bool(value.flags.f_contiguous)
    span = None
    if math.prod(shape):
        extents = [(size - 1) * stride for size, stride in zip(shape, strides, strict=True)]
        span = (offset + sum(min(0, e) for e in extents), offset + sum(max(0, e) for e in extents))
    return {
        "Strides": strides,
        "Byte strides": tuple(s * itemsize for s in strides),
        "Storage offset": offset,
        "Is contiguous": c_contiguous,
        "C contiguous": c_contiguous,
        "Fortran contiguous": f_contiguous,
        "Memory offset range": span,
    }


def memory_offset(value: Any, indices: tuple[int, ...]) -> int:
    """Return a validated element offset in original backing storage."""
    if len(indices) != len(value.shape):
        raise IndexError(f"Expected {len(value.shape)} indices for shape {tuple(value.shape)}.")
    layout = tensor_layout(value)
    offset = layout["Storage offset"]
    for axis, (index, size, stride) in enumerate(
        zip(indices, value.shape, layout["Strides"], strict=True)
    ):
        index = operator.index(index)
        if not -size <= index < size:
            raise IndexError(f"Index {index} is out of bounds for axis {axis} with size {size}.")
        offset += (index % size) * stride
    return int(offset)


def complex_components(values: list[complex]) -> dict[str, list[float]]:
    """Expose bounded complex samples without comparing complex values."""
    return {
        "real": [z.real for z in values],
        "imaginary": [z.imag for z in values],
        "magnitude": [abs(z) for z in values],
        "probability": [abs(z) * abs(z) for z in values],
        "phase": [math.atan2(z.imag, z.real) if abs(z) else 0.0 for z in values],
    }
