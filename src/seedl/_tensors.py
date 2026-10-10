"""Multidimensional tensor and array inspection utilities for SeeDL."""

from __future__ import annotations

import math
from typing import Any


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


def detect_image_layout(shape: tuple[int, ...]) -> tuple[str, int] | None:
    """Detect image or image-batch layout from shape dimensions.

    Returns:
        - ("channels_first", channels) for 3D (C, H, W)
        - ("channels_last", channels) for 3D (H, W, C)
        - ("batch_channels_first", channels) for 4D (N, C, H, W)
        - ("batch_channels_last", channels) for 4D (N, H, W, C)
        - None if not image-like
    """
    if len(shape) == 3:
        if shape[0] in (1, 3, 4) and min(shape[1:]) >= 4:
            return "channels_first", shape[0]
        if shape[-1] in (1, 3, 4) and min(shape[:2]) >= 4:
            return "channels_last", shape[-1]
    elif len(shape) == 4:
        batch_size = shape[0]
        if 1 <= batch_size <= 256:
            if shape[1] in (1, 3, 4) and min(shape[2:]) >= 4:
                return "batch_channels_first", shape[1]
            if shape[-1] in (1, 3, 4) and min(shape[1:3]) >= 4:
                return "batch_channels_last", shape[-1]
    return None


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
