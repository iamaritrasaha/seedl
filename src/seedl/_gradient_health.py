"""Gradient health diagnostics and numerical stability intelligence."""

from __future__ import annotations

import html
import json
import math
from typing import Any

from .exceptions import MissingDependencyError
from .renderers import resolve_backend
from .view import Visualization


def _torch():
    try:
        import torch

        return torch
    except ModuleNotFoundError:
        return None


def inspect_gradient_health(
    model: Any,
    *,
    max_parameters: int = 256,
) -> dict[str, Any]:
    """Inspect the actual numerical behavior and norms of gradients across all model parameters.

    Invariants:
    - Deduplicates tied/shared parameters to avoid double-counting.
    - Preserves existing parameter gradients without alteration.
    - Evaluates exact L1, L2 norms, min/max, zero fraction, nonfinite counts.
    - Computes gradient-to-weight norm ratio ||∇W|| / ||W||.
    """
    torch = _torch()
    if torch is None or not isinstance(model, torch.nn.Module):
        raise MissingDependencyError("Gradient health inspection requires PyTorch.")

    seen_ptrs = set()
    param_records: list[dict[str, Any]] = []
    layer_groups: dict[str, list[dict[str, Any]]] = {}

    tot_l1 = 0.0
    tot_l2_sq = 0.0
    total_zero_elements = 0
    total_elements = 0
    total_nonfinite = 0
    missing_grad_params: list[str] = []

    # Iterate with remove_duplicate=False to detect tied/shared parameter aliases
    named_params = (
        model.named_parameters(remove_duplicate=False) if hasattr(model, "named_parameters") else []
    )
    for name, param in named_params:
        if len(param_records) >= max_parameters:
            break

        # Check for tied parameters
        ptr = (
            param.data_ptr() if hasattr(param, "data_ptr") and param.data_ptr() != 0 else id(param)
        )
        is_tied = ptr in seen_ptrs
        if not is_tied:
            seen_ptrs.add(ptr)

        num_elements = param.numel()
        shape = list(param.shape)
        weight_norm = float(param.data.norm(2).item()) if param.data.numel() > 0 else 0.0

        if param.grad is None:
            if param.requires_grad:
                missing_grad_params.append(name)
            record = {
                "name": name,
                "shape": shape,
                "num_elements": num_elements,
                "is_tied": is_tied,
                "has_grad": False,
                "requires_grad": param.requires_grad,
                "weight_norm": weight_norm,
                "l1_norm": None,
                "l2_norm": None,
                "max_abs": None,
                "mean": None,
                "std": None,
                "zero_fraction": None,
                "nonfinite_count": 0,
                "grad_to_weight_ratio": None,
            }
        else:
            grad = param.grad.detach()
            if grad.is_sparse:
                grad = grad.to_dense()
            float_grad = grad.float()

            is_finite = torch.isfinite(float_grad)
            nonfinite_cnt = int((~is_finite).sum().item())
            total_nonfinite += nonfinite_cnt

            if nonfinite_cnt == 0:
                l1 = float(float_grad.norm(1).item())
                l2 = float(float_grad.norm(2).item())
                max_abs = float(float_grad.abs().max().item())
                mean_val = float(float_grad.mean().item())
                std_val = float(float_grad.std().item()) if num_elements > 1 else 0.0
                zero_cnt = int((float_grad == 0.0).sum().item())
                zero_frac = zero_cnt / float(num_elements) if num_elements > 0 else 0.0

                if not is_tied:
                    tot_l1 += l1
                    tot_l2_sq += l2**2
                    total_zero_elements += zero_cnt
                    total_elements += num_elements
            else:
                l1, l2, max_abs, mean_val, std_val = (
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                )
                zero_frac = float("nan")

            ratio = (l2 / weight_norm) if (weight_norm > 1e-12 and math.isfinite(l2)) else None

            record = {
                "name": name,
                "shape": shape,
                "num_elements": num_elements,
                "is_tied": is_tied,
                "has_grad": True,
                "requires_grad": param.requires_grad,
                "weight_norm": weight_norm,
                "l1_norm": l1 if math.isfinite(l1) else None,
                "l2_norm": l2 if math.isfinite(l2) else None,
                "max_abs": max_abs if math.isfinite(max_abs) else None,
                "mean": mean_val if math.isfinite(mean_val) else None,
                "std": std_val if math.isfinite(std_val) else None,
                "zero_fraction": zero_frac if math.isfinite(zero_frac) else None,
                "nonfinite_count": nonfinite_cnt,
                "grad_to_weight_ratio": ratio,
            }

        param_records.append(record)

        # Layer grouping
        prefix = name.rsplit(".", 1)[0] if "." in name else "root"
        layer_groups.setdefault(prefix, []).append(record)

    global_l2 = math.sqrt(tot_l2_sq) if math.isfinite(tot_l2_sq) else float("nan")
    global_zero_frac = (total_zero_elements / float(total_elements)) if total_elements > 0 else 0.0

    # Aggregate layer stats
    layer_summaries = []
    for prefix, group in layer_groups.items():
        grp_l2_sq = sum(r["l2_norm"] ** 2 for r in group if r["l2_norm"] is not None)
        grp_l2 = math.sqrt(grp_l2_sq)
        grp_params = sum(r["num_elements"] for r in group)
        layer_summaries.append(
            {
                "layer": prefix,
                "num_tensors": len(group),
                "num_elements": grp_params,
                "l2_norm": grp_l2,
            }
        )

    # Descriptive scientific diagnostics
    observations: list[str] = []
    if total_nonfinite > 0:
        observations.append(
            f"Nonfinite gradients observed ({total_nonfinite} NaN/Inf values). Model updates are unstable."
        )
    if global_l2 < 1e-7 and total_elements > 0:
        observations.append(
            f"Global gradient L2 norm is near-zero ({global_l2:.2e}), consistent with vanishing gradients or unpropagated loss."
        )
    if global_l2 > 100.0:
        observations.append(
            f"Global gradient L2 norm is high ({global_l2:.2f}); consider gradient clipping."
        )
    if global_zero_frac > 0.8:
        observations.append(
            f"High fraction of zero gradients ({global_zero_frac:.1%}), common with sparse inputs or dead ReLU activations."
        )
    if missing_grad_params:
        observations.append(
            f"{len(missing_grad_params)} parameter(s) with requires_grad=True had grad=None."
        )

    return {
        "parameters": param_records,
        "layers": layer_summaries,
        "global_l1": tot_l1 if math.isfinite(tot_l1) else None,
        "global_l2": global_l2 if math.isfinite(global_l2) else None,
        "global_zero_fraction": global_zero_frac,
        "total_nonfinite": total_nonfinite,
        "total_parameters_inspected": len(param_records),
        "missing_grad_count": len(missing_grad_params),
        "missing_grad_parameters": missing_grad_params,
        "observations": observations,
    }


def gradient_health(
    model: Any,
    *,
    max_parameters: int = 256,
    backend: str = "auto",
    title: str | None = None,
) -> Visualization:
    """Analyze and visualize gradient norms, distributions, and numerical health."""
    resolved_backend = resolve_backend(backend)
    data = inspect_gradient_health(model, max_parameters=max_parameters)

    vis_title = title or "Gradient Health Intelligence"
    tot_params = data["total_parameters_inspected"]
    l2_str = (
        f"||∇||₂ = {data['global_l2']:.4e}" if data["global_l2"] is not None else "||∇||₂ = NaN"
    )
    summary = f"Inspected {tot_params} parameter tensor(s) · {l2_str} · {data['global_zero_fraction']:.1%} zeros"

    notes = [
        "Parameter gradients were read directly from existing autograd buffers without mutation.",
        "Shared/tied parameters are deduplicated to avoid overcounting gradient norms.",
    ]
    notes.extend(data["observations"])

    metadata = {
        "Inspected Parameters": tot_params,
        "Global L2 Norm": f"{data['global_l2']:.4e}" if data["global_l2"] is not None else "NaN",
        "Zero Gradients": f"{data['global_zero_fraction']:.1%}",
        "Nonfinite Gradients": data["total_nonfinite"],
    }

    return Visualization(
        kind="gradient_health",
        title=vis_title,
        summary=summary,
        data=data,
        metadata=metadata,
        notes=notes,
        backend=resolved_backend,
    )


def gradient_health_html(view: Visualization) -> str:
    """Generate self-contained interactive HTML for gradient health dashboard."""
    data = view.data
    title = html.escape(view.title)
    summary = html.escape(view.summary)
    payload_json = (
        json.dumps(data).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{title} — SeeDL Gradient Health</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card: #1e293b;
      --card-alt: #334155;
      --border: #475569;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --warning: #fbbf24;
      --danger: #f87171;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 24px;
      line-height: 1.5;
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 24px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }}
    header {{
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
      margin-bottom: 20px;
    }}
    header h1 {{ font-size: 22px; color: var(--accent); }}
    header p {{ color: var(--muted); font-size: 14px; margin-top: 4px; }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .stat-card {{
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px;
    }}
    .stat-title {{ font-size: 12px; color: var(--muted); text-transform: uppercase; }}
    .stat-val {{ font-size: 20px; font-weight: 700; margin-top: 4px; font-family: monospace; }}
    table.data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    table.data-table th, table.data-table td {{
      padding: 8px 12px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }}
    table.data-table th {{ background: var(--card-alt); color: var(--text); }}
  </style>
</head>
<body>
<div class="container">
  <header>
    <h1>{title}</h1>
    <p>{summary}</p>
  </header>

  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-title">Global L2 Norm</div>
      <div class="stat-val" id="stat-l2">-</div>
    </div>
    <div class="stat-card">
      <div class="stat-title">Zero Gradients</div>
      <div class="stat-val" id="stat-zeros">-</div>
    </div>
    <div class="stat-card">
      <div class="stat-title">Nonfinite Count</div>
      <div class="stat-val" id="stat-nonfinite">-</div>
    </div>
    <div class="stat-card">
      <div class="stat-title">Missing Grads</div>
      <div class="stat-val" id="stat-missing">-</div>
    </div>
  </div>

  <h2 style="font-size: 16px; margin-bottom: 12px; color: var(--accent);">Layer Gradient Breakdown</h2>
  <div style="overflow-x: auto; margin-bottom: 24px;">
    <table class="data-table">
      <thead>
        <tr>
          <th>Layer / Module</th>
          <th>Tensors</th>
          <th>Total Elements</th>
          <th>L2 Norm</th>
        </tr>
      </thead>
      <tbody id="layers-body"></tbody>
    </table>
  </div>

  <h2 style="font-size: 16px; margin-bottom: 12px; color: var(--accent);">Parameter Details</h2>
  <div style="overflow-x: auto;">
    <table class="data-table">
      <thead>
        <tr>
          <th>Parameter</th>
          <th>Shape</th>
          <th>Weight ||W||</th>
          <th>Grad ||∇||</th>
          <th>Ratio ||∇||/||W||</th>
          <th>Zero %</th>
        </tr>
      </thead>
      <tbody id="params-body"></tbody>
    </table>
  </div>
</div>

<script>
  const payload = {payload_json};
  document.getElementById('stat-l2').textContent = payload.global_l2 != null ? payload.global_l2.toExponential(3) : 'NaN';
  document.getElementById('stat-zeros').textContent = (payload.global_zero_fraction * 100).toFixed(1) + '%';
  document.getElementById('stat-nonfinite').textContent = payload.total_nonfinite;
  document.getElementById('stat-missing').textContent = payload.missing_grad_count;

  const lBody = document.getElementById('layers-body');
  (payload.layers || []).forEach(l => {{
    const row = document.createElement('tr');
    row.innerHTML = `
      <td><strong>${{l.layer}}</strong></td>
      <td>${{l.num_tensors}}</td>
      <td>${{l.num_elements.toLocaleString()}}</td>
      <td><code>${{l.l2_norm.toExponential(3)}}</code></td>
    `;
    lBody.appendChild(row);
  }});

  const pBody = document.getElementById('params-body');
  (payload.parameters || []).forEach(p => {{
    const row = document.createElement('tr');
    const ratioStr = p.grad_to_weight_ratio != null ? p.grad_to_weight_ratio.toExponential(2) : '-';
    const l2Str = p.l2_norm != null ? p.l2_norm.toExponential(2) : (p.has_grad ? 'NaN' : 'None');
    const zeroStr = p.zero_fraction != null ? (p.zero_fraction * 100).toFixed(1) + '%' : '-';
    row.innerHTML = `
      <td>${{p.name}} ${{p.is_tied ? '<span style="color:var(--warning); font-size:11px;">(Tied)</span>' : ''}}</td>
      <td><code>${{JSON.stringify(p.shape)}}</code></td>
      <td>${{p.weight_norm.toFixed(2)}}</td>
      <td><code>${{l2Str}}</code></td>
      <td>${{ratioStr}}</td>
      <td>${{zeroStr}}</td>
    `;
    pBody.appendChild(row);
  }});

  window.__seedl_grad_payload = payload;
</script>
</body>
</html>
"""


def svg_gradient_health(view: Visualization) -> str:
    """Render static SVG of gradient health summary."""
    data = view.data
    layers = data.get("layers", [])
    width = 800
    height = max(360, 120 + len(layers) * 45)

    title = html.escape(view.title)
    summary = html.escape(view.summary)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        f'<rect width="{width}" height="{height}" fill="#0f172a"/>',
        '<text x="40" y="38" font-family="system-ui, sans-serif" font-size="12" fill="#94a3b8" font-weight="600">SeeDL / GRADIENT HEALTH</text>',
        f'<text x="40" y="68" font-family="system-ui, sans-serif" font-size="20" fill="#f8fafc" font-weight="700">{title}</text>',
        f'<text x="40" y="92" font-family="system-ui, sans-serif" font-size="13" fill="#94a3b8">{summary}</text>',
    ]

    y = 130
    for idx, lyr in enumerate(layers[:16]):
        name = html.escape(lyr.get("layer", f"Layer {idx}"))
        l2 = lyr.get("l2_norm", 0.0)
        lines.extend(
            [
                f'<g transform="translate(60, {y})">',
                '<rect width="680" height="36" rx="6" fill="#1e293b" stroke="#334155" stroke-width="1"/>',
                f'<text x="20" y="23" font-family="system-ui, sans-serif" font-size="13" font-weight="600" fill="#38bdf8">{name}</text>',
                f'<text x="660" y="23" font-family="monospace" font-size="12" fill="#34d399" text-anchor="end">||∇||₂ = {l2:.4e}</text>',
                "</g>",
            ]
        )
        y += 45

    lines.append("</svg>")
    return "\n".join(lines)
