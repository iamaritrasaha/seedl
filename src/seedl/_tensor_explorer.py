"""Interactive Multidimensional Tensor Explorer for SeeDL.

Provides bounded slice extraction, responsive browser-based slice navigation with
sliders, dimension switching, and zero-external-dependency HTML/Canvas rendering.
"""

from __future__ import annotations

import html
import json
import math
import uuid
from collections.abc import Sequence
from typing import Any

from ._inspect import (
    _metadata_from_stats,
    _numpy_module,
    _numpy_statistics,
    _preview_steps,
    _torch_module,
    _torch_statistics,
    is_numpy_array,
    is_torch_tensor,
)
from ._tensors import (
    format_sparsity,
)
from .exceptions import SeeDLError


def validate_dim_names(dim_names: Any, rank: int, shape: tuple[int, ...]) -> list[str]:
    """Validate and normalize semantic dimension names.

    Raises:
        SeeDLError: If dim_names is not a sequence of non-empty strings or if the count
                    does not match the tensor rank.
    """
    if dim_names is None:
        return [f"dim_{i}" for i in range(rank)]
    if isinstance(dim_names, (str, bytes)) or not isinstance(dim_names, (list, tuple)):
        raise SeeDLError("dim_names must be a list or tuple of strings.")
    if len(dim_names) != rank:
        raise SeeDLError(
            f"dim_names has {len(dim_names)} names, but tensor has rank {rank} with shape {shape}."
        )
    result: list[str] = []
    for i, name in enumerate(dim_names):
        if not isinstance(name, str) or not name.strip():
            raise SeeDLError(
                f"dim_names element at index {i} must be a non-empty string, got {name!r}."
            )
        result.append(name.strip())
    return result


def _default_dim_names(rank: int, layout: str | None, shape: tuple[int, ...]) -> list[str]:
    if layout is not None:
        normalized = layout.strip().upper()
        if normalized == "CHW" and rank == 3:
            return ["channel", "height", "width"]
        if normalized == "HWC" and rank == 3:
            return ["height", "width", "channel"]
        if normalized in {"NCHW", "BCHW"} and rank == 4:
            return ["batch", "channel", "height", "width"]
        if normalized in {"NHWC", "BHWC"} and rank == 4:
            return ["batch", "height", "width", "channel"]

    if rank == 4:
        if shape[1] in (1, 3, 4):
            return ["batch", "channel", "height", "width"]
        if shape[-1] in (1, 3, 4):
            return ["batch", "height", "width", "channel"]
        return ["batch", "dim_1", "dim_2", "dim_3"]

    if rank == 3:
        if shape[0] in (1, 3, 4):
            return ["channel", "height", "width"]
        if shape[-1] in (1, 3, 4):
            return ["height", "width", "channel"]
        return ["depth", "height", "width"]

    return [f"dim_{i}" for i in range(rank)]


def build_tensor_explorer_payload(
    tensor: Any,
    *,
    dim_names: Sequence[str] | None = None,
    layout: str | None = None,
    max_frames: int = 32,
    max_slice_points: int = 4096,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Extract bounded multidimensional slice frames and statistics for interactive exploration.

    Transfers only downsampled slice previews from GPU to CPU to preserve memory.
    """
    if not (is_torch_tensor(tensor) or is_numpy_array(tensor)):
        raise SeeDLError(
            "Interactive tensor exploration requires a PyTorch tensor or NumPy ndarray, "
            f"got {type(tensor).__name__}."
        )

    shape = tuple(int(s) for s in tensor.shape)
    rank = len(shape)
    count = int(math.prod(shape))

    if rank < 2:
        raise SeeDLError(
            f"Interactive tensor exploration requires at least 2 dimensions, got rank {rank} with shape {shape}."
        )

    # Validate or generate dimension names
    if dim_names is not None:
        validated_dims = validate_dim_names(dim_names, rank, shape)
    else:
        validated_dims = _default_dim_names(rank, layout, shape)

    is_torch = is_torch_tensor(tensor)
    device = str(tensor.device) if is_torch else "CPU"
    dtype_str = str(tensor.dtype).replace("torch.", "") if is_torch else str(tensor.dtype)
    requires_grad = bool(tensor.requires_grad) if is_torch else False

    # Compute global tensor statistics safely via chunked Welford
    notes: list[str] = []
    if is_torch:
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
        ) = _torch_statistics(tensor, return_counts=True)
        itemsize = int(tensor.element_size())
    else:
        (
            minimum,
            maximum,
            mean,
            std,
            nonfinite,
            nan_count,
            inf_count,
            zero_count,
        ) = _numpy_statistics(tensor, return_counts=True)
        itemsize = int(tensor.itemsize)

    total_bytes = count * itemsize
    sparsity_pct = (zero_count / count) * 100.0 if count else 0.0

    metadata = _metadata_from_stats(
        shape=shape,
        count=count,
        dtype=dtype_str,
        device=device,
        requires_grad=requires_grad,
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        std=std,
        nonfinite=nonfinite,
        rank=rank,
        memory_bytes=total_bytes,
        nan_count=nan_count,
        inf_count=inf_count,
        sparsity_pct=sparsity_pct,
    )
    metadata["Dimension names"] = ", ".join(validated_dims)

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

    if sparsity_pct >= 50.0 and count > 10:
        notes.append(f"High sparsity: {format_sparsity(zero_count, count)} are zero.")

    # Determine slicing axes (support slicing along first axes that have size > 1)
    # For rank 3 (D, H, W): slice along Axis 0, 1, 2
    # For rank 4 (N, C, H, W): slice along Axis 0 (batch), Axis 1 (channels)
    slicing_axes: list[int] = []
    if rank == 3:
        slicing_axes = [0, 1, 2]
    elif rank == 4:
        slicing_axes = [0, 1]
    else:
        slicing_axes = [i for i in range(min(rank, 3)) if shape[i] > 1]
        if not slicing_axes:
            slicing_axes = [0]

    axes_payload: list[dict[str, Any]] = []

    for axis in slicing_axes:
        axis_size = shape[axis]
        step = max(1, math.ceil(axis_size / max_frames))
        frame_indices = list(range(0, axis_size, step))[:max_frames]
        axis_name = validated_dims[axis]

        frames: list[dict[str, Any]] = []

        for idx in frame_indices:
            # Extract slice on tensor (performed on GPU first if torch tensor to bound transfer)
            if is_torch:
                torch_mod = _torch_module()
                assert torch_mod is not None
                sub = tensor.select(axis, idx).detach()
                sub_shape = tuple(int(s) for s in sub.shape)

                # Check if sub-tensor is an image (e.g. 3D with 3 channels)
                is_sub_image = False
                if len(sub_shape) == 3 and sub_shape[0] in (1, 3, 4) and min(sub_shape[1:]) >= 8:
                    is_sub_image = True
                    # CHW -> HWC
                    h, w = sub_shape[1], sub_shape[2]
                    ystep, xstep = _preview_steps(h, w, 64, max_slice_points)
                    sampled = sub[:, ::ystep, ::xstep][:, :64, :64].permute(1, 2, 0)
                    sampled_cpu = sampled.to(device="cpu", dtype=torch_mod.float32)
                    frame_pixels = sampled_cpu.tolist()
                    fmin = float(sampled_cpu.min().item()) if sampled_cpu.numel() else 0.0
                    fmax = float(sampled_cpu.max().item()) if sampled_cpu.numel() else 0.0
                    fmean = float(sampled_cpu.mean().item()) if sampled_cpu.numel() else 0.0
                    frames.append(
                        {
                            "index": idx,
                            "type": "image",
                            "shape": [int(sampled.shape[0]), int(sampled.shape[1]), int(sampled.shape[2])],
                            "pixels": frame_pixels,
                            "min": fmin,
                            "max": fmax,
                            "mean": fmean,
                        }
                    )
                elif len(sub_shape) == 3 and sub_shape[-1] in (1, 3, 4) and min(sub_shape[:2]) >= 8:
                    is_sub_image = True
                    h, w = sub_shape[0], sub_shape[1]
                    ystep, xstep = _preview_steps(h, w, 64, max_slice_points)
                    sampled = sub[::ystep, ::xstep][:64, :64]
                    sampled_cpu = sampled.to(device="cpu", dtype=torch_mod.float32)
                    frame_pixels = sampled_cpu.tolist()
                    fmin = float(sampled_cpu.min().item()) if sampled_cpu.numel() else 0.0
                    fmax = float(sampled_cpu.max().item()) if sampled_cpu.numel() else 0.0
                    fmean = float(sampled_cpu.mean().item()) if sampled_cpu.numel() else 0.0
                    frames.append(
                        {
                            "index": idx,
                            "type": "image",
                            "shape": [int(sampled.shape[0]), int(sampled.shape[1]), int(sampled.shape[2])],
                            "pixels": frame_pixels,
                            "min": fmin,
                            "max": fmax,
                            "mean": fmean,
                        }
                    )

                if not is_sub_image:
                    # 2D heatmap or higher-rank flatten to 2D
                    if len(sub_shape) == 2:
                        h, w = sub_shape[0], sub_shape[1]
                        ystep, xstep = _preview_steps(h, w, 64, max_slice_points)
                        sampled = sub[::ystep, ::xstep][:64, :64]
                    elif len(sub_shape) == 1:
                        step_1d = max(1, math.ceil(sub_shape[0] / 64))
                        sampled = sub[::step_1d][:64].unsqueeze(0)
                    else:
                        # Flatten remaining inner dimensions to 2D
                        first_dim = sub_shape[0]
                        rest = math.prod(sub_shape[1:])
                        flat_2d = sub.reshape(first_dim, rest)
                        ystep, xstep = _preview_steps(first_dim, rest, 64, max_slice_points)
                        sampled = flat_2d[::ystep, ::xstep][:64, :64]

                    sampled_cpu = sampled.to(device="cpu", dtype=torch_mod.float32)
                    fmin = float(sampled_cpu.min().item()) if sampled_cpu.numel() else 0.0
                    fmax = float(sampled_cpu.max().item()) if sampled_cpu.numel() else 0.0
                    fmean = float(sampled_cpu.mean().item()) if sampled_cpu.numel() else 0.0
                    frames.append(
                        {
                            "index": idx,
                            "type": "heatmap",
                            "shape": [int(sampled.shape[0]), int(sampled.shape[1])],
                            "values": sampled_cpu.tolist(),
                            "min": fmin,
                            "max": fmax,
                            "mean": fmean,
                        }
                    )
            else:
                np = _numpy_module()
                assert np is not None
                sub = np.take(tensor, idx, axis=axis)
                sub_shape = sub.shape

                is_sub_image = False
                if len(sub_shape) == 3 and sub_shape[0] in (1, 3, 4) and min(sub_shape[1:]) >= 8:
                    is_sub_image = True
                    h, w = sub_shape[1], sub_shape[2]
                    ystep, xstep = _preview_steps(h, w, 64, max_slice_points)
                    sampled = np.moveaxis(sub[:, ::ystep, ::xstep][:, :64, :64], 0, -1).astype(np.float32)
                    fmin = float(np.min(sampled)) if sampled.size else 0.0
                    fmax = float(np.max(sampled)) if sampled.size else 0.0
                    fmean = float(np.mean(sampled)) if sampled.size else 0.0
                    frames.append(
                        {
                            "index": idx,
                            "type": "image",
                            "shape": list(sampled.shape),
                            "pixels": sampled.tolist(),
                            "min": fmin,
                            "max": fmax,
                            "mean": fmean,
                        }
                    )
                elif len(sub_shape) == 3 and sub_shape[-1] in (1, 3, 4) and min(sub_shape[:2]) >= 8:
                    is_sub_image = True
                    h, w = sub_shape[0], sub_shape[1]
                    ystep, xstep = _preview_steps(h, w, 64, max_slice_points)
                    sampled = sub[::ystep, ::xstep][:64, :64].astype(np.float32)
                    fmin = float(np.min(sampled)) if sampled.size else 0.0
                    fmax = float(np.max(sampled)) if sampled.size else 0.0
                    fmean = float(np.mean(sampled)) if sampled.size else 0.0
                    frames.append(
                        {
                            "index": idx,
                            "type": "image",
                            "shape": list(sampled.shape),
                            "pixels": sampled.tolist(),
                            "min": fmin,
                            "max": fmax,
                            "mean": fmean,
                        }
                    )

                if not is_sub_image:
                    if len(sub_shape) == 2:
                        h, w = sub_shape[0], sub_shape[1]
                        ystep, xstep = _preview_steps(h, w, 64, max_slice_points)
                        sampled = sub[::ystep, ::xstep][:64, :64].astype(np.float32)
                    elif len(sub_shape) == 1:
                        step_1d = max(1, math.ceil(sub_shape[0] / 64))
                        sampled = np.expand_dims(sub[::step_1d][:64], 0).astype(np.float32)
                    else:
                        first_dim = sub_shape[0]
                        rest = math.prod(sub_shape[1:])
                        flat_2d = sub.reshape(first_dim, rest)
                        ystep, xstep = _preview_steps(first_dim, rest, 64, max_slice_points)
                        sampled = flat_2d[::ystep, ::xstep][:64, :64].astype(np.float32)

                    fmin = float(np.min(sampled)) if sampled.size else 0.0
                    fmax = float(np.max(sampled)) if sampled.size else 0.0
                    fmean = float(np.mean(sampled)) if sampled.size else 0.0
                    frames.append(
                        {
                            "index": idx,
                            "type": "heatmap",
                            "shape": list(sampled.shape),
                            "values": sampled.tolist(),
                            "min": fmin,
                            "max": fmax,
                            "mean": fmean,
                        }
                    )

        axes_payload.append(
            {
                "axis": axis,
                "name": axis_name,
                "size": axis_size,
                "step": step,
                "total_slices": axis_size,
                "sampled_count": len(frames),
                "frames": frames,
            }
        )

    notes.append(
        f"Interactive slice explorer: scrub through {len(axes_payload)} dimension axes with client-side slider."
    )

    data: dict[str, Any] = {
        "interactive": True,
        "rank": rank,
        "shape": list(shape),
        "dim_names": validated_dims,
        "dtype": dtype_str,
        "device": device,
        "axes": axes_payload,
        "global_stats": {
            "minimum": minimum,
            "maximum": maximum,
            "mean": mean,
            "std": std,
            "sparsity_pct": sparsity_pct,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "nonfinite": nonfinite,
        },
    }

    return data, metadata, notes


def tensor_explorer_html(view: Any, *, full_html: bool = True) -> str:
    """Generate self-contained interactive HTML for the Multidimensional Tensor Explorer."""
    data = view.data
    uid = uuid.uuid4().hex[:8]
    container_id = f"seedl-tensor-explorer-{uid}"

    title_escaped = html.escape(str(view.title))
    summary_escaped = html.escape(str(view.summary))
    shape_str = html.escape(str(view.metadata.get("Shape", data.get("shape"))))
    rank_str = html.escape(str(view.metadata.get("Rank", data.get("rank"))))
    device_str = html.escape(str(view.metadata.get("Device", data.get("device"))))
    dtype_str = html.escape(str(view.metadata.get("Data type", data.get("dtype"))))
    memory_str = html.escape(str(view.metadata.get("Estimated memory", "n/a")))
    sparsity_str = html.escape(str(view.metadata.get("Sparsity", "0.0%")))

    meta_chips = [
        f'<span class="ste-chip"><strong>Rank:</strong> {rank_str}</span>',
        f'<span class="ste-chip"><strong>Shape:</strong> {shape_str}</span>',
        f'<span class="ste-chip"><strong>Dtype:</strong> {dtype_str}</span>',
        f'<span class="ste-chip"><strong>Device:</strong> {device_str}</span>',
        f'<span class="ste-chip"><strong>Memory:</strong> {memory_str}</span>',
        f'<span class="ste-chip"><strong>Sparsity:</strong> {sparsity_str}</span>',
    ]

    nan_count = view.metadata.get("NaN values")
    if nan_count:
        meta_chips.append(
            f'<span class="ste-chip ste-chip-warn"><strong>NaNs:</strong> {html.escape(str(nan_count))}</span>'
        )
    inf_count = view.metadata.get("Infinite values")
    if inf_count:
        meta_chips.append(
            f'<span class="ste-chip ste-chip-warn"><strong>Infs:</strong> {html.escape(str(inf_count))}</span>'
        )

    meta_html = "".join(meta_chips)

    notes_html = "".join(
        f"<li>{html.escape(str(note))}</li>" for note in view.notes
    )

    # Serialize JSON payload safely for embedding in script tag
    payload_json = (
        json.dumps(data)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )

    widget_body = f"""
<div id="{container_id}" class="seedl-tensor-explorer-widget">
  <div class="ste-header">
    <div class="ste-title-row">
      <h2 class="ste-title">{title_escaped}</h2>
      <span class="ste-badge">Interactive</span>
    </div>
    <div class="ste-summary">{summary_escaped}</div>
    <div class="ste-chips-bar">{meta_html}</div>
  </div>

  <div class="ste-controls">
    <div class="ste-axis-selector-bar">
      <span class="ste-label">Slicing Dimension:</span>
      <div class="ste-axis-buttons" id="{container_id}-axis-buttons"></div>
    </div>

    <div class="ste-slider-row">
      <div class="ste-playback-buttons">
        <button type="button" class="ste-btn" id="{container_id}-prev-btn" title="Previous slice">◀</button>
        <button type="button" class="ste-btn ste-btn-primary" id="{container_id}-play-btn" title="Play/Pause playback">▶ Play</button>
        <button type="button" class="ste-btn" id="{container_id}-next-btn" title="Next slice">▶</button>
      </div>
      <input type="range" class="ste-range" id="{container_id}-slider" min="0" max="0" value="0" />
      <span class="ste-index-badge" id="{container_id}-index-badge">Slice: 0 / 0</span>
    </div>
  </div>

  <div class="ste-viewport">
    <div class="ste-canvas-container">
      <canvas id="{container_id}-canvas" class="ste-canvas" width="256" height="256"></canvas>
      <div id="{container_id}-tooltip" class="ste-tooltip" style="display:none;"></div>
    </div>
    <div class="ste-slice-meta" id="{container_id}-slice-meta">
      <div class="ste-meta-card">
        <div class="ste-card-title">Slice Information</div>
        <div id="{container_id}-slice-stats" class="ste-slice-stats-content">Loading...</div>
      </div>
      <div class="ste-palette-selector">
        <span class="ste-sublabel">Colormap:</span>
        <select id="{container_id}-colormap-select" class="ste-select">
          <option value="viridis" selected>Viridis</option>
          <option value="coolwarm">Coolwarm</option>
          <option value="grayscale">Grayscale</option>
          <option value="spectral">Spectral</option>
        </select>
      </div>
    </div>
  </div>

  <div class="ste-footer">
    <ul class="ste-notes">{notes_html}</ul>
    <div class="ste-brand">SeeDL Interactive Tensor Observatory</div>
  </div>
</div>

<script>
(function() {{
  const payload = {payload_json};
  const root = document.getElementById("{container_id}");
  if (!root) return;

  const axisButtonsContainer = document.getElementById("{container_id}-axis-buttons");
  const slider = document.getElementById("{container_id}-slider");
  const indexBadge = document.getElementById("{container_id}-index-badge");
  const prevBtn = document.getElementById("{container_id}-prev-btn");
  const nextBtn = document.getElementById("{container_id}-next-btn");
  const playBtn = document.getElementById("{container_id}-play-btn");
  const canvas = document.getElementById("{container_id}-canvas");
  const ctx = canvas ? canvas.getContext("2d") : null;
  const tooltip = document.getElementById("{container_id}-tooltip");
  const sliceStats = document.getElementById("{container_id}-slice-stats");
  const colormapSelect = document.getElementById("{container_id}-colormap-select");

  if (!payload.axes || payload.axes.length === 0) return;

  let currentAxisIdx = 0;
  let currentFrameIdx = 0;
  let isPlaying = false;
  let playInterval = null;
  let activeColormap = "viridis";

  function colormapViridis(t) {{
    // t in [0, 1]
    const r = Math.max(0, Math.min(255, Math.round(255 * (0.2 + 0.8 * Math.pow(t, 2)))));
    const g = Math.max(0, Math.min(255, Math.round(255 * Math.sin(t * Math.PI))));
    const b = Math.max(0, Math.min(255, Math.round(255 * (0.6 - 0.4 * t))));
    return [r, g, b];
  }}

  function colormapCoolwarm(t) {{
    const r = Math.round(255 * Math.min(1, Math.max(0, 2 * t)));
    const g = Math.round(255 * (1 - Math.abs(2 * t - 1)));
    const b = Math.round(255 * Math.min(1, Math.max(0, 2 * (1 - t))));
    return [r, g, b];
  }}

  function colormapGrayscale(t) {{
    const v = Math.round(255 * Math.max(0, Math.min(1, t)));
    return [v, v, v];
  }}

  function colormapSpectral(t) {{
    const hue = (1.0 - t) * 240; // blue to red
    const c = 1.0;
    const x = c * (1 - Math.abs(((hue / 60) % 2) - 1));
    let r1=0, g1=0, b1=0;
    if (hue < 60) {{ r1=c; g1=x; }}
    else if (hue < 120) {{ r1=x; g1=c; }}
    else if (hue < 180) {{ g1=c; b1=x; }}
    else if (hue < 240) {{ b1=c; }}
    return [Math.round(r1 * 255), Math.round(g1 * 255), Math.round(b1 * 255)];
  }}

  function getColor(val, min, max, map) {{
    if (isNaN(val) || !isFinite(val)) return [128, 128, 128, 255]; // Gray for non-finite
    const range = (max - min) || 1.0;
    const t = Math.max(0, Math.min(1, (val - min) / range));
    let rgb;
    if (map === "coolwarm") rgb = colormapCoolwarm(t);
    else if (map === "grayscale") rgb = colormapGrayscale(t);
    else if (map === "spectral") rgb = colormapSpectral(t);
    else rgb = colormapViridis(t);
    return [rgb[0], rgb[1], rgb[2], 255];
  }}

  function renderCurrentFrame() {{
    const axisData = payload.axes[currentAxisIdx];
    if (!axisData || !axisData.frames || axisData.frames.length === 0) return;
    const frame = axisData.frames[currentFrameIdx];
    if (!frame || !ctx) return;

    slider.value = currentFrameIdx;
    indexBadge.textContent = `${{axisData.name}} = ${{frame.index}} (${{currentFrameIdx + 1}}/${{axisData.frames.length}})`;

    sliceStats.innerHTML = `
      <div class="ste-stat-row"><span>Dimension:</span> <strong>${{axisData.name}} (Axis ${{axisData.axis}})</strong></div>
      <div class="ste-stat-row"><span>Index in Tensor:</span> <strong>${{frame.index}} / ${{axisData.total_slices - 1}}</strong></div>
      <div class="ste-stat-row"><span>Frame Shape:</span> <strong>${{frame.shape ? frame.shape.join(' × ') : '2D'}}</strong></div>
      <div class="ste-stat-row"><span>Slice Min:</span> <strong>${{frame.min != null ? frame.min.toFixed(4) : 'n/a'}}</strong></div>
      <div class="ste-stat-row"><span>Slice Max:</span> <strong>${{frame.max != null ? frame.max.toFixed(4) : 'n/a'}}</strong></div>
      <div class="ste-stat-row"><span>Slice Mean:</span> <strong>${{frame.mean != null ? frame.mean.toFixed(4) : 'n/a'}}</strong></div>
    `;

    const canvasWidth = 256;
    const canvasHeight = 256;
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;

    if (frame.type === "image" && frame.pixels) {{
      const pixels = frame.pixels;
      const rows = pixels.length;
      const cols = pixels[0].length;
      const imgData = ctx.createImageData(cols, rows);
      for (let y = 0; y < rows; y++) {{
        for (let x = 0; x < cols; x++) {{
          const p = pixels[y][x];
          const idx = (y * cols + x) * 4;
          if (Array.isArray(p)) {{
            imgData.data[idx] = Math.round(Math.max(0, Math.min(1, p[0])) * 255);
            imgData.data[idx + 1] = Math.round(Math.max(0, Math.min(1, p[1] || p[0])) * 255);
            imgData.data[idx + 2] = Math.round(Math.max(0, Math.min(1, p[2] || p[0])) * 255);
            imgData.data[idx + 3] = 255;
          }} else {{
            const c = Math.round(Math.max(0, Math.min(1, p)) * 255);
            imgData.data[idx] = c;
            imgData.data[idx + 1] = c;
            imgData.data[idx + 2] = c;
            imgData.data[idx + 3] = 255;
          }}
        }}
      }}
      // Scale up crisply to 256x256
      const offCanvas = document.createElement("canvas");
      offCanvas.width = cols;
      offCanvas.height = rows;
      offCanvas.getContext("2d").putImageData(imgData, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(offCanvas, 0, 0, canvasWidth, canvasHeight);
    }} else if (frame.values) {{
      const values = frame.values;
      const rows = values.length;
      const cols = values[0].length;
      const imgData = ctx.createImageData(cols, rows);
      for (let y = 0; y < rows; y++) {{
        for (let x = 0; x < cols; x++) {{
          const val = values[y][x];
          const idx = (y * cols + x) * 4;
          const [r, g, b, a] = getColor(val, frame.min, frame.max, activeColormap);
          imgData.data[idx] = r;
          imgData.data[idx + 1] = g;
          imgData.data[idx + 2] = b;
          imgData.data[idx + 3] = a;
        }}
      }}
      const offCanvas = document.createElement("canvas");
      offCanvas.width = cols;
      offCanvas.height = rows;
      offCanvas.getContext("2d").putImageData(imgData, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(offCanvas, 0, 0, canvasWidth, canvasHeight);
    }}
  }}

  function switchAxis(idx) {{
    currentAxisIdx = idx;
    currentFrameIdx = 0;
    const axisData = payload.axes[currentAxisIdx];
    slider.max = (axisData.frames.length - 1);
    slider.value = 0;
    updateAxisButtons();
    renderCurrentFrame();
  }}

  function updateAxisButtons() {{
    axisButtonsContainer.innerHTML = "";
    payload.axes.forEach((ax, idx) => {{
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `ste-btn ${{idx === currentAxisIdx ? 'ste-btn-active' : ''}}`;
      btn.textContent = `${{ax.name}} (Axis ${{ax.axis}}, size ${{ax.size}})`;
      btn.addEventListener("click", () => switchAxis(idx));
      axisButtonsContainer.appendChild(btn);
    }});
  }}

  function togglePlay() {{
    if (isPlaying) {{
      clearInterval(playInterval);
      playInterval = null;
      isPlaying = false;
      playBtn.textContent = "▶ Play";
    }} else {{
      isPlaying = true;
      playBtn.textContent = "⏸ Pause";
      const axisData = payload.axes[currentAxisIdx];
      playInterval = setInterval(() => {{
        currentFrameIdx = (currentFrameIdx + 1) % axisData.frames.length;
        renderCurrentFrame();
      }}, 400);
    }}
  }}

  // Event Listeners
  slider.addEventListener("input", (e) => {{
    currentFrameIdx = parseInt(e.target.value, 10);
    renderCurrentFrame();
  }});

  prevBtn.addEventListener("click", () => {{
    const axisData = payload.axes[currentAxisIdx];
    currentFrameIdx = (currentFrameIdx - 1 + axisData.frames.length) % axisData.frames.length;
    renderCurrentFrame();
  }});

  nextBtn.addEventListener("click", () => {{
    const axisData = payload.axes[currentAxisIdx];
    currentFrameIdx = (currentFrameIdx + 1) % axisData.frames.length;
    renderCurrentFrame();
  }});

  playBtn.addEventListener("click", togglePlay);

  colormapSelect.addEventListener("change", (e) => {{
    activeColormap = e.target.value;
    renderCurrentFrame();
  }});

  // Tooltip on canvas hover
  canvas.addEventListener("mousemove", (e) => {{
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / (rect.width / canvas.width));
    const y = Math.floor((e.clientY - rect.top) / (rect.height / canvas.height));
    const axisData = payload.axes[currentAxisIdx];
    const frame = axisData.frames[currentFrameIdx];
    if (frame && frame.values && frame.values[y] && frame.values[y][x] !== undefined) {{
      tooltip.style.display = "block";
      tooltip.style.left = `${{e.clientX - rect.left + 15}}px`;
      tooltip.style.top = `${{e.clientY - rect.top + 10}}px`;
      tooltip.textContent = `[${{y}}, ${{x}}]: ${{Number(frame.values[y][x]).toFixed(4)}}`;
    }} else {{
      tooltip.style.display = "none";
    }}
  }});

  canvas.addEventListener("mouseleave", () => {{
    tooltip.style.display = "none";
  }});

  // Initial render
  updateAxisButtons();
  switchAxis(0);
}})();
</script>
"""

    css = """
<style>
.seedl-tensor-explorer-widget {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: var(--ste-bg, #0f172a);
  color: var(--ste-text, #f8fafc);
  border: 1px solid var(--ste-border, #334155);
  border-radius: 10px;
  padding: 20px;
  max-width: 960px;
  margin: 12px auto;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
  box-sizing: border-box;
}
.seedl-tensor-explorer-widget * {
  box-sizing: border-box;
}
.ste-header {
  border-bottom: 1px solid var(--ste-border, #334155);
  padding-bottom: 12px;
  margin-bottom: 16px;
}
.ste-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 4px;
}
.ste-title {
  font-size: 19px;
  font-weight: 600;
  color: var(--ste-accent, #38bdf8);
  margin: 0;
}
.ste-badge {
  background: #0284c7;
  color: #fff;
  font-size: 11px;
  font-weight: bold;
  padding: 3px 8px;
  border-radius: 9999px;
  text-transform: uppercase;
}
.ste-summary {
  font-size: 13px;
  color: var(--ste-muted, #94a3b8);
  margin-bottom: 10px;
}
.ste-chips-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.ste-chip {
  background: var(--ste-chip-bg, #1e293b);
  border: 1px solid var(--ste-border, #334155);
  padding: 3px 8px;
  border-radius: 6px;
  font-size: 12px;
}
.ste-chip-warn {
  border-color: #f59e0b;
  color: #fbbf24;
}
.ste-controls {
  background: var(--ste-card-bg, #1e293b);
  border: 1px solid var(--ste-border, #334155);
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 16px;
}
.ste-axis-selector-bar {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 12px;
}
.ste-label {
  font-size: 13px;
  font-weight: 600;
  color: var(--ste-muted, #94a3b8);
}
.ste-axis-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.ste-btn {
  background: var(--ste-btn-bg, #334155);
  color: var(--ste-text, #f8fafc);
  border: 1px solid transparent;
  padding: 4px 10px;
  border-radius: 5px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.15s ease;
}
.ste-btn:hover {
  background: #475569;
}
.ste-btn-active {
  background: #0284c7;
  color: #ffffff;
  font-weight: 600;
}
.ste-btn-primary {
  background: #0284c7;
}
.ste-slider-row {
  display: flex;
  align-items: center;
  gap: 12px;
}
.ste-playback-buttons {
  display: flex;
  gap: 4px;
}
.ste-range {
  flex: 1;
  cursor: pointer;
  accent-color: #38bdf8;
}
.ste-index-badge {
  font-size: 12px;
  font-weight: 600;
  min-width: 140px;
  text-align: right;
  color: var(--ste-accent, #38bdf8);
}
.ste-viewport {
  display: grid;
  grid-template-columns: 280px 1fr;
  gap: 16px;
  align-items: start;
}
@media (max-width: 640px) {
  .ste-viewport {
    grid-template-columns: 1fr;
  }
}
.ste-canvas-container {
  position: relative;
  background: #000000;
  border: 1px solid var(--ste-border, #334155);
  border-radius: 8px;
  padding: 8px;
  display: flex;
  justify-content: center;
  align-items: center;
}
.ste-canvas {
  image-rendering: pixelated;
  image-rendering: -moz-crisp-edges;
  image-rendering: crisp-edges;
  max-width: 100%;
  height: auto;
  cursor: crosshair;
}
.ste-tooltip {
  position: absolute;
  background: rgba(15, 23, 42, 0.9);
  color: #38bdf8;
  border: 1px solid #38bdf8;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 11px;
  pointer-events: none;
  z-index: 10;
  white-space: nowrap;
}
.ste-slice-meta {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.ste-meta-card {
  background: var(--ste-card-bg, #1e293b);
  border: 1px solid var(--ste-border, #334155);
  border-radius: 8px;
  padding: 12px;
}
.ste-card-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--ste-muted, #94a3b8);
  border-bottom: 1px solid var(--ste-border, #334155);
  padding-bottom: 6px;
  margin-bottom: 8px;
}
.ste-slice-stats-content {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
}
.ste-stat-row {
  display: flex;
  justify-content: space-between;
}
.ste-palette-selector {
  display: flex;
  align-items: center;
  gap: 8px;
}
.ste-sublabel {
  font-size: 12px;
  color: var(--ste-muted, #94a3b8);
}
.ste-select {
  background: var(--ste-card-bg, #1e293b);
  color: var(--ste-text, #f8fafc);
  border: 1px solid var(--ste-border, #334155);
  border-radius: 4px;
  padding: 3px 8px;
  font-size: 12px;
}
.ste-footer {
  margin-top: 16px;
  padding-top: 12px;
  border-top: 1px solid var(--ste-border, #334155);
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  flex-wrap: wrap;
  gap: 12px;
}
.ste-notes {
  margin: 0;
  padding-left: 20px;
  color: var(--ste-muted, #94a3b8);
  font-size: 12px;
  max-width: 700px;
}
.ste-brand {
  font-size: 11px;
  color: #64748b;
  text-align: right;
  flex: 1;
}
</style>
"""

    if not full_html:
        return f"{css}\n{widget_body}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{title_escaped} — SeeDL Interactive Tensor Explorer</title>
  {css}
</head>
<body style="margin: 0; padding: 24px; background: #020617;">
  {widget_body}
</body>
</html>"""


def svg_tensor_explorer(data: dict[str, Any], metadata: dict[str, Any], notes: list[str]) -> str:
    """Generate static SVG figure representing the tensor explorer state for vector outputs."""
    title = html.escape(str(metadata.get("Title", "Interactive Tensor Explorer")))
    shape_str = html.escape(str(metadata.get("Shape", data.get("shape"))))
    rank_str = html.escape(str(metadata.get("Rank", data.get("rank"))))
    dtype_str = html.escape(str(metadata.get("Data type", data.get("dtype"))))
    dim_names_str = html.escape(str(metadata.get("Dimension names", ", ".join(data.get("dim_names", [])))))

    axes = data.get("axes", [])
    primary_axis = axes[0] if axes else {}
    frames = primary_axis.get("frames", [])
    slice_count = len(frames)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 380" width="100%" height="100%">
  <rect width="100%" height="100%" fill="#0f172a" rx="10"/>
  <rect x="20" y="20" width="720" height="340" fill="#1e293b" rx="8" stroke="#334155" stroke-width="1"/>
  
  <text x="40" y="55" fill="#38bdf8" font-family="-apple-system, sans-serif" font-size="18" font-weight="bold">{title}</text>
  <text x="40" y="80" fill="#94a3b8" font-family="-apple-system, sans-serif" font-size="13">Shape: {shape_str} | Rank: {rank_str} | Dtype: {dtype_str}</text>
  <text x="40" y="102" fill="#94a3b8" font-family="-apple-system, sans-serif" font-size="13">Dimensions: {dim_names_str}</text>
  
  <!-- Slice viewport representation -->
  <rect x="40" y="125" width="200" height="200" fill="#000000" rx="6" stroke="#38bdf8" stroke-width="1.5"/>
  <text x="140" y="225" fill="#38bdf8" font-family="-apple-system, sans-serif" font-size="13" text-anchor="middle">Interactive Slice Canvas</text>
  <text x="140" y="245" fill="#64748b" font-family="-apple-system, sans-serif" font-size="11" text-anchor="middle">(Export to HTML to explore)</text>
  
  <!-- Controls representation -->
  <rect x="260" y="125" width="460" height="200" fill="#0f172a" rx="6" stroke="#334155" stroke-width="1"/>
  <text x="280" y="160" fill="#f8fafc" font-family="-apple-system, sans-serif" font-size="14" font-weight="bold">Slice Navigation Controls</text>
  <text x="280" y="190" fill="#94a3b8" font-family="-apple-system, sans-serif" font-size="12">Primary Axis: {primary_axis.get("name", "dim_0")} (Axis {primary_axis.get("axis", 0)})</text>
  <text x="280" y="215" fill="#94a3b8" font-family="-apple-system, sans-serif" font-size="12">Total Slices: {primary_axis.get("total_slices", "n/a")} | Sampled Frames: {slice_count}</text>
  
  <!-- Slider graphic -->
  <line x1="280" y1="260" x2="680" y2="260" stroke="#475569" stroke-width="6" stroke-linecap="round"/>
  <circle cx="340" cy="260" r="10" fill="#38bdf8" stroke="#ffffff" stroke-width="2"/>
  <text x="280" y="295" fill="#64748b" font-family="-apple-system, sans-serif" font-size="11">Index 0</text>
  <text x="680" y="295" fill="#64748b" font-family="-apple-system, sans-serif" font-size="11" text-anchor="end">Index {primary_axis.get("total_slices", 1) - 1}</text>
</svg>"""
    return svg
