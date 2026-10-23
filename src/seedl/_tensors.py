"""Multidimensional tensor and array inspection utilities for SeeDL."""

from __future__ import annotations

import math
from typing import Any

from .exceptions import SeeDLError


def format_memory_bytes(total_bytes: int) -> str:
    """Format byte count into a readable memory estimate."""
    if total_bytes < 1024:
        return f"{total_bytes} B"
    if total_bytes < 1024 * 1024:
        return f"{total_bytes / 1024:.1f} KB"
    if total_bytes < 1024 * 1024 * 1024:
        return f"{total_bytes / (1024 * 1024):.1f} MB"
    return f"{total_bytes / (1024 * 1024 * 1024):.2f} GB"


def format_sparsity(zeros: int, total: int) -> str:
    """Format sparsity percentage."""
    if total <= 0:
        return "0.0% zeros"
    percentage = (zeros / total) * 100.0
    return f"{percentage:.1f}% zeros"


def is_ambiguous_image_layout(shape: tuple[int, ...]) -> bool:
    """Determine whether a tensor shape has ambiguous channel positions.

    A shape is ambiguous if both channels-first and channels-last heuristics match
    simultaneously without explicit user layout guidance (e.g. (3, 32, 3)).
    """
    if len(shape) == 3:
        first_is_chan = shape[0] in (1, 3, 4)
        last_is_chan = shape[-1] in (1, 3, 4)
        return bool(first_is_chan and last_is_chan and shape[1] >= 1)
    if len(shape) == 4:
        first_is_chan = shape[1] in (1, 3, 4)
        last_is_chan = shape[-1] in (1, 3, 4)
        return bool(first_is_chan and last_is_chan and min(shape[2:3]) >= 1)
    return False


def detect_image_layout(
    shape: tuple[int, ...], min_spatial_size: int = 16
) -> tuple[str, int] | None:
    """Detect image or image-batch layout from shape dimensions.

    Returns:
        - ("channels_first", channels) for unambiguous 3D (C, H, W)
        - ("channels_last", channels) for unambiguous 3D (H, W, C)
        - ("batch_channels_first", channels) for unambiguous 4D (N, C, H, W)
        - ("batch_channels_last", channels) for unambiguous 4D (N, H, W, C)
        - None if ambiguous, spatial dimensions below min_spatial_size, or not image-like.
    """
    if is_ambiguous_image_layout(shape):
        return None

    if len(shape) == 3:
        if shape[0] in (1, 3, 4) and min(shape[1:]) >= min_spatial_size:
            return "channels_first", shape[0]
        if shape[-1] in (1, 3, 4) and min(shape[:2]) >= min_spatial_size:
            return "channels_last", shape[-1]
    elif len(shape) == 4:
        batch_size = shape[0]
        if 1 <= batch_size <= 256:
            if shape[1] in (1, 3, 4) and min(shape[2:]) >= min_spatial_size:
                return "batch_channels_first", shape[1]
            if shape[-1] in (1, 3, 4) and min(shape[1:3]) >= min_spatial_size:
                return "batch_channels_last", shape[-1]
    return None


def resolve_image_layout(
    shape: tuple[int, ...],
    layout: str | None = None,
    *,
    strict: bool = False,
) -> tuple[str, int] | None:
    """Resolve channel layout for 3D or 4D tensors, respecting explicit layouts and avoiding guesses.

    Supported explicit layout strings:
        - 'CHW', 'channels_first'
        - 'HWC', 'channels_last'
        - 'NCHW', 'batch_channels_first'
        - 'NHWC', 'batch_channels_last'

    Raises:
        SeeDLError: If an explicit layout is invalid, has mismatched rank/channel count,
                    or if an ambiguous layout is encountered under strict=True.
    """
    if layout is not None:
        norm = layout.strip().lower().replace("-", "_")
        if norm in {"chw", "channels_first"}:
            if len(shape) != 3:
                raise SeeDLError(
                    f"Layout {layout!r} requires a 3D tensor (C, H, W), but received shape {shape} of rank {len(shape)}."
                )
            channels = shape[0]
            if channels not in (1, 3, 4):
                raise SeeDLError(
                    f"Layout {layout!r} specifies {channels} channels for shape {shape}; image visualization requires 1 (grayscale), 3 (RGB), or 4 (RGBA) channels."
                )
            return "channels_first", channels

        if norm in {"hwc", "channels_last"}:
            if len(shape) != 3:
                raise SeeDLError(
                    f"Layout {layout!r} requires a 3D tensor (H, W, C), but received shape {shape} of rank {len(shape)}."
                )
            channels = shape[-1]
            if channels not in (1, 3, 4):
                raise SeeDLError(
                    f"Layout {layout!r} specifies {channels} channels for shape {shape}; image visualization requires 1 (grayscale), 3 (RGB), or 4 (RGBA) channels."
                )
            return "channels_last", channels

        if norm in {"nchw", "batch_channels_first"}:
            if len(shape) != 4:
                raise SeeDLError(
                    f"Layout {layout!r} requires a 4D batch tensor (N, C, H, W), but received shape {shape} of rank {len(shape)}."
                )
            channels = shape[1]
            if channels not in (1, 3, 4):
                raise SeeDLError(
                    f"Layout {layout!r} specifies {channels} channels for shape {shape}; image visualization requires 1 (grayscale), 3 (RGB), or 4 (RGBA) channels."
                )
            return "batch_channels_first", channels

        if norm in {"nhwc", "batch_channels_last"}:
            if len(shape) != 4:
                raise SeeDLError(
                    f"Layout {layout!r} requires a 4D batch tensor (N, H, W, C), but received shape {shape} of rank {len(shape)}."
                )
            channels = shape[-1]
            if channels not in (1, 3, 4):
                raise SeeDLError(
                    f"Layout {layout!r} specifies {channels} channels for shape {shape}; image visualization requires 1 (grayscale), 3 (RGB), or 4 (RGBA) channels."
                )
            return "batch_channels_last", channels

        raise SeeDLError(
            f"Unsupported layout {layout!r}. Supported layouts are 'CHW', 'HWC', 'NCHW', 'NHWC'."
        )

    # When layout is None: detect unambiguous layout or check ambiguity
    if is_ambiguous_image_layout(shape):
        if strict:
            raise SeeDLError(
                f"Ambiguous channel layout for tensor with shape {shape}: both channels-first and channels-last positions have valid channel counts. Specify layout='CHW' or layout='HWC' explicitly."
            )
        return None

    return detect_image_layout(shape)


def parse_slice_spec(slice_spec: Any, shape: tuple[int, ...] | None = None) -> Any:
    """Parse, normalize, and bounds-check a slice specification for tensor slicing.

    Accepts:
        - Integers (e.g. 0, -1)
        - Standard slice objects (e.g. slice(0, 10))
        - Tuples or lists of slices/ints/Ellipsis
        - Strings (e.g. "0", "0, :, :", ":, 0:10, 2", "0, ...")

    Raises:
        SeeDLError: For invalid slice syntax, excessive dimensions, or out-of-bounds indices.
    """
    if slice_spec is None:
        return slice(None)

    parsed_tuple: tuple[Any, ...]

    if isinstance(slice_spec, str):
        cleaned = slice_spec.strip().strip("[]()")
        if not cleaned:
            return slice(None)
        parts = [p.strip() for p in cleaned.split(",")]
        parsed_items: list[Any] = []
        for p in parts:
            if p == "...":
                parsed_items.append(Ellipsis)
            elif p == ":":
                parsed_items.append(slice(None))
            elif ":" in p:
                sub_parts = p.split(":")
                if len(sub_parts) == 2:
                    start_s, stop_s = sub_parts
                    try:
                        start = int(start_s) if start_s.strip() else None
                        stop = int(stop_s) if stop_s.strip() else None
                    except ValueError as err:
                        raise SeeDLError(f"Invalid integer in slice range {p!r}: {err}") from err
                    parsed_items.append(slice(start, stop))
                elif len(sub_parts) == 3:
                    start_s, stop_s, step_s = sub_parts
                    try:
                        start = int(start_s) if start_s.strip() else None
                        stop = int(stop_s) if stop_s.strip() else None
                        step = int(step_s) if step_s.strip() else None
                    except ValueError as err:
                        raise SeeDLError(f"Invalid integer in slice range {p!r}: {err}") from err
                    parsed_items.append(slice(start, stop, step))
                else:
                    raise SeeDLError(f"Invalid slice syntax: {p!r}")
            else:
                try:
                    parsed_items.append(int(p))
                except ValueError as err:
                    raise SeeDLError(
                        f"Expected an integer or slice syntax in {slice_spec!r}, got {p!r}."
                    ) from err
        parsed_tuple = tuple(parsed_items)
    elif isinstance(slice_spec, (tuple, list)):
        parsed_tuple = tuple(slice_spec)
    elif isinstance(slice_spec, (int, slice)) or slice_spec is Ellipsis:
        parsed_tuple = (slice_spec,)
    else:
        parsed_tuple = (slice_spec,)

    # Dimension and bounds checks when shape is known
    if shape is not None:
        # Handle Ellipsis expansion
        if Ellipsis in parsed_tuple:
            ellipsis_count = parsed_tuple.count(Ellipsis)
            if ellipsis_count > 1:
                raise SeeDLError("An index can only have a single ellipsis ('...')")
            non_ellipsis_count = len(parsed_tuple) - 1
            if non_ellipsis_count > len(shape):
                raise SeeDLError(
                    f"Too many slice indices ({len(parsed_tuple)}) for tensor of rank {len(shape)} with shape {shape}."
                )
            expanded_len = len(shape) - non_ellipsis_count
            ellipsis_idx = parsed_tuple.index(Ellipsis)
            parsed_tuple = (
                parsed_tuple[:ellipsis_idx]
                + (slice(None),) * expanded_len
                + parsed_tuple[ellipsis_idx + 1 :]
            )

        if len(parsed_tuple) > len(shape):
            raise SeeDLError(
                f"Too many slice indices ({len(parsed_tuple)}) for tensor of rank {len(shape)} with shape {shape}."
            )

        for axis, (item, size) in enumerate(zip(parsed_tuple, shape, strict=False)):
            if isinstance(item, int):
                if item < -size or item >= size:
                    raise SeeDLError(
                        f"Slice index {item} is out of bounds for axis {axis} with size {size} in shape {shape}."
                    )

    return parsed_tuple if len(parsed_tuple) > 1 else parsed_tuple[0]


def make_image_grid(
    images: list[list[list[Any]]],
    channels: int,
    *,
    max_images: int = 16,
    border: float = 0.85,
) -> tuple[list[list[Any]], int, int, int]:
    """Stitch a list of 2D pixel grids (each HxW or HxWxC) into a tiled grid.

    Returns:
        (grid_pixels, out_channels, grid_rows, grid_cols)
    """
    count = min(len(images), max_images)
    if count == 0:
        return [], channels, 0, 0
    selected = images[:count]
    cols = math.isqrt(count)
    if cols * cols < count:
        cols += 1
    rows = math.ceil(count / cols)

    h = len(selected[0])
    w = len(selected[0][0]) if h > 0 else 0
    if h == 0 or w == 0:
        return [], channels, rows, cols

    grid_h = rows * h + (rows - 1)
    grid_w = cols * w + (cols - 1)

    # Initialize empty grid with border color
    border_val = [border] * channels if channels > 1 else [border]
    grid: list[list[Any]] = [
        [list(border_val) if channels > 1 else border for _ in range(grid_w)] for _ in range(grid_h)
    ]

    for idx, img in enumerate(selected):
        r = idx // cols
        c = idx % cols
        y_offset = r * (h + 1)
        x_offset = c * (w + 1)
        for y in range(min(h, len(img))):
            for x in range(min(w, len(img[y]))):
                grid[y_offset + y][x_offset + x] = img[y][x]

    return grid, channels, rows, cols


def make_channel_grid(
    pixel_img: list[list[Any]],
    channels: int,
) -> tuple[list[list[Any]], int, int, int]:
    """Split a multi-channel image into individual grayscale channel tiles."""
    h = len(pixel_img)
    w = len(pixel_img[0]) if h > 0 else 0
    if h == 0 or w == 0 or channels <= 1:
        return pixel_img, channels, 1, 1

    channel_images: list[list[list[float]]] = []
    for c in range(channels):
        c_img = []
        for row in pixel_img:
            c_row = []
            for px in row:
                if isinstance(px, (list, tuple)) and len(px) > c:
                    c_row.append(float(px[c]))
                else:
                    c_row.append(float(px) if not isinstance(px, (list, tuple)) else 0.0)
            c_img.append(c_row)
        channel_images.append(c_img)

    return make_image_grid(channel_images, 1, max_images=channels)
