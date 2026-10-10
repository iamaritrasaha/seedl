"""Computer vision deep learning tools: convolution operations and kernel filters."""

from __future__ import annotations

import math
from typing import Any

from .exceptions import SeeDLError


def inspect_convolution(
    image: list[list[float]] | Any | None = None,
    kernel: list[list[float]] | Any | None = None,
    *,
    stride: int = 1,
    padding: int = 0,
) -> tuple[dict[str, Any], str, list[str]]:
    """Simulate and visualize an exact 2D convolution operation with kernel and arithmetic."""
    # Default 8x8 input image if not provided
    if image is None:
        # A simple pattern with vertical and horizontal edges
        image = [
            [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ]
    elif hasattr(image, "detach") and hasattr(image, "tolist"):
        img_arr = image.detach().cpu().squeeze().tolist()
        image = img_arr if isinstance(img_arr[0], list) else [img_arr]
    elif hasattr(image, "tolist"):
        image = image.tolist()

    # Default 3x3 Sobel vertical edge detector kernel
    if kernel is None:
        kernel = [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ]
    elif hasattr(kernel, "detach") and hasattr(kernel, "tolist"):
        k_arr = kernel.detach().cpu().squeeze().tolist()
        kernel = k_arr if isinstance(k_arr[0], list) else [k_arr]
    elif hasattr(kernel, "tolist"):
        kernel = kernel.tolist()

    h_in, w_in = len(image), len(image[0])
    k_h, k_w = len(kernel), len(kernel[0])

    if stride < 1:
        raise SeeDLError("stride must be a positive integer.")
    if padding < 0:
        raise SeeDLError("padding must be non-negative.")

    # Apply padding
    padded_img = [[0.0] * (w_in + 2 * padding) for _ in range(h_in + 2 * padding)]
    for y in range(h_in):
        for x in range(w_in):
            padded_img[y + padding][x + padding] = float(image[y][x])

    # Compute output dimensions
    h_out = math.floor((h_in + 2 * padding - k_h) / stride + 1)
    w_out = math.floor((w_in + 2 * padding - k_w) / stride + 1)

    if h_out <= 0 or w_out <= 0:
        raise SeeDLError(
            f"Kernel size ({k_h}×{k_w}) with padding {padding} and stride {stride} is larger than input ({h_in}×{w_in})."
        )

    # Perform exact 2D convolution
    out_map: list[list[float]] = []
    for y in range(h_out):
        out_row: list[float] = []
        for x in range(w_out):
            val = 0.0
            in_y = y * stride
            in_x = x * stride
            for ky in range(k_h):
                for kx in range(k_w):
                    val += padded_img[in_y + ky][in_x + kx] * float(kernel[ky][kx])
            out_row.append(val)
        out_map.append(out_row)

    summary = (
        f"2D Convolution: Input ({h_in}×{w_in}) * Kernel ({k_h}×{k_w}) "
        f"-> Feature Map ({h_out}×{w_out}) [stride={stride}, pad={padding}]"
    )

    notes = [
        f"Exact 2D discrete cross-correlation calculated: Output size = floor((N + 2p - k)/s + 1) = {h_out}×{w_out}.",
        f"Kernel dimensions: {k_h} rows × {k_w} columns.",
        "Demonstrates spatial filtering, edge detection, and feature extraction in CNN layers.",
    ]

    data = {
        "input_shape": (h_in, w_in),
        "kernel_shape": (k_h, k_w),
        "output_shape": (h_out, w_out),
        "h_out": h_out,
        "w_out": w_out,
        "stride": stride,
        "padding": padding,
        "input_image": image,
        "kernel": kernel,
        "feature_map": out_map,
        "output": out_map,
    }

    return data, summary, notes
