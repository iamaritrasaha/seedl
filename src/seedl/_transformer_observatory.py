"""Transformer Observatory 2.0: Multi-layer Attention, Mathematically Rigorous Rollout, and ViT Flow."""

from __future__ import annotations

import html
import json
import math
from typing import Any

from .exceptions import SeeDLError
from .renderers import resolve_backend
from .view import Visualization


def _to_numpy_or_list(obj: Any) -> list[Any]:
    """Convert tensor or array to native Python nested lists."""
    if hasattr(obj, "detach"):
        obj = obj.detach()
    if hasattr(obj, "cpu"):
        obj = obj.cpu()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "tolist"):
        return list(obj)
    return obj


def _matrix_multiply_2d(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Multiply two 2D square matrices a @ b in pure Python."""
    n = len(a)
    m = len(b[0]) if b else 0
    k_len = len(b)
    res = [[0.0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for k in range(k_len):
            aik = a[i][k]
            if aik != 0.0:
                for j in range(m):
                    res[i][j] += aik * b[k][j]
    return res


def _normalize_rows(mat: list[list[float]]) -> list[list[float]]:
    """Ensure each row sums to 1.0 (row-stochastic)."""
    normed: list[list[float]] = []
    for row in mat:
        r_sum = sum(row)
        if r_sum > 1e-12:
            normed.append([v / r_sum for v in row])
        else:
            # Uniform fallback for zero row
            n = len(row)
            normed.append([1.0 / n for _ in range(n)])
    return normed


def _numpy():
    try:
        import numpy as np

        return np
    except ImportError:
        return None


def _compute_attention_rollout_numpy(
    attention_layers: list[Any] | Any,
    np: Any,
    include_residual: bool = True,
    head_reduction: str = "mean",
) -> tuple[list[list[float]], list[list[list[float]]]]:
    # Stack or extract layers
    if isinstance(attention_layers, list):
        layers = [np.asarray(layer, dtype=float) for layer in attention_layers]
    else:
        arr = np.asarray(attention_layers, dtype=float)
        if arr.ndim == 4:
            layers = [arr[i] for i in range(arr.shape[0])]
        elif arr.ndim == 3:
            layers = [arr]
        else:
            layers = [arr]

    processed = []
    for l_idx, layer_arr in enumerate(layers):
        while layer_arr.ndim > 3 and layer_arr.shape[0] == 1:
            layer_arr = layer_arr[0]
        if layer_arr.ndim == 3:
            if head_reduction == "mean":
                mat = layer_arr.mean(axis=0)
            elif head_reduction == "min":
                mat = layer_arr.min(axis=0)
            elif head_reduction == "max":
                mat = layer_arr.max(axis=0)
            else:
                raise ValueError(f"Unsupported head_reduction {head_reduction!r}.")
        elif layer_arr.ndim == 2:
            mat = layer_arr
        else:
            raise SeeDLError(f"Invalid attention layer dimensions: {layer_arr.shape}")

        if mat.shape[0] != mat.shape[1]:
            raise SeeDLError(
                f"Attention rollout requires square attention matrices; layer {l_idx} has shape {mat.shape}."
            )

        row_sums = mat.sum(axis=-1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mat = mat / row_sums

        if include_residual:
            seq_len = mat.shape[0]
            mat = 0.5 * (mat + np.eye(seq_len))
            row_sums = mat.sum(axis=-1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            mat = mat / row_sums

        processed.append(mat)

    intermediates = []
    current_r = processed[0]
    intermediates.append(current_r.tolist())

    for l_idx in range(1, len(processed)):
        current_r = processed[l_idx] @ current_r
        row_sums = current_r.sum(axis=-1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        current_r = current_r / row_sums
        intermediates.append(current_r.tolist())

    return current_r.tolist(), intermediates


def compute_attention_rollout(
    attention_layers: list[Any] | Any,
    *,
    include_residual: bool = True,
    head_reduction: str = "mean",  # "mean", "min", "max"
) -> tuple[list[list[float]], list[list[list[float]]]]:
    """Compute cumulative attention rollout across transformer layers.

    Mathematical Invariant:
    Let A_l be the row-normalized attention matrix at layer l in {1, ..., L}.
    With residual connection, the effective attention matrix is:
        A'_l = 0.5 * (A_l + I)
    Cumulative rollout is computed as:
        R_1 = A'_1
        R_l = A'_l @ R_{l-1}
        R = A'_L @ A'_{L-1} @ ... @ A'_1

    Returns:
        (final_rollout_matrix, intermediate_rollouts_per_layer)
    """
    if attention_layers is None or len(attention_layers) == 0:
        raise SeeDLError("attention_layers cannot be empty.")

    np = _numpy()
    if np is not None:
        try:
            return _compute_attention_rollout_numpy(
                attention_layers,
                np,
                include_residual=include_residual,
                head_reduction=head_reduction,
            )
        except (ValueError, SeeDLError):
            raise
        except Exception:
            # Fall back to pure Python implementation if any unexpected conversion issue occurs
            pass

    processed_layers: list[list[list[float]]] = []

    for l_idx, raw_layer in enumerate(attention_layers):
        data = _to_numpy_or_list(raw_layer)
        # Handle 4D [batch, heads, seq, seq] or 3D [heads, seq, seq] or 2D [seq, seq]
        while (
            isinstance(data, list)
            and len(data) == 1
            and isinstance(data[0], list)
            and isinstance(data[0][0], list)
            and isinstance(data[0][0][0], list)
        ):
            data = data[0]  # strip batch dim 1

        if not isinstance(data, list) or not data:
            raise SeeDLError(f"Invalid attention layer at index {l_idx}.")

        # Check dimension
        if isinstance(data[0][0], list):
            # Multi-head: [num_heads, seq_len, seq_len]
            num_heads = len(data)
            seq_len_q = len(data[0])
            seq_len_k = len(data[0][0])
            if seq_len_q != seq_len_k:
                raise SeeDLError(
                    f"Attention rollout requires square attention matrices; layer {l_idx} has shape [{num_heads}, {seq_len_q}, {seq_len_k}]."
                )

            # Reduce heads
            aggregated = [[0.0 for _ in range(seq_len_k)] for _ in range(seq_len_q)]
            if head_reduction == "mean":
                for h in range(num_heads):
                    for i in range(seq_len_q):
                        for j in range(seq_len_k):
                            aggregated[i][j] += data[h][i][j]
                for i in range(seq_len_q):
                    for j in range(seq_len_k):
                        aggregated[i][j] /= float(num_heads)
            elif head_reduction == "min":
                for i in range(seq_len_q):
                    for j in range(seq_len_k):
                        aggregated[i][j] = min(data[h][i][j] for h in range(num_heads))
            elif head_reduction == "max":
                for i in range(seq_len_q):
                    for j in range(seq_len_k):
                        aggregated[i][j] = max(data[h][i][j] for h in range(num_heads))
            else:
                raise ValueError(f"Unsupported head_reduction {head_reduction!r}.")
            mat = aggregated
        else:
            # 2D [seq_len, seq_len]
            seq_len_q = len(data)
            seq_len_k = len(data[0]) if seq_len_q > 0 else 0
            if seq_len_q != seq_len_k:
                raise SeeDLError(
                    f"Attention rollout requires square attention matrices; layer {l_idx} has shape [{seq_len_q}, {seq_len_k}]."
                )
            mat = data

        # Normalize rows to guarantee row-stochasticity
        mat = _normalize_rows(mat)

        # Include residual connection: A'_l = 0.5 * (A_l + I)
        seq_len = len(mat)
        if include_residual:
            a_prime = [[0.0 for _ in range(seq_len)] for _ in range(seq_len)]
            for i in range(seq_len):
                for j in range(seq_len):
                    diag = 1.0 if i == j else 0.0
                    a_prime[i][j] = 0.5 * (mat[i][j] + diag)
            mat = _normalize_rows(a_prime)

        processed_layers.append(mat)

    # Accumulate rollout: R = A'_L @ A'_{L-1} @ ... @ A'_1
    intermediates: list[list[list[float]]] = []
    current_r = processed_layers[0]
    intermediates.append(current_r)

    for l_idx in range(1, len(processed_layers)):
        current_r = _matrix_multiply_2d(processed_layers[l_idx], current_r)
        current_r = _normalize_rows(current_r)
        intermediates.append(current_r)

    return current_r, intermediates


def attention_rollout(
    attention_tensors: list[Any] | Any,
    *,
    tokens: list[str] | None = None,
    include_residual: bool = True,
    head_reduction: str = "mean",
    backend: str = "auto",
    title: str | None = None,
) -> Visualization:
    """Compute and visualize cumulative Attention Rollout across transformer layers."""
    resolved_backend = resolve_backend(backend)

    if not isinstance(attention_tensors, (list, tuple)):
        # Could be 4D tensor [layers, heads, seq, seq] or numpy array
        if hasattr(attention_tensors, "detach"):
            attention_tensors = attention_tensors.detach().cpu().numpy()
        elif hasattr(attention_tensors, "shape"):
            pass  # keep numpy array intact
        else:
            t_list = _to_numpy_or_list(attention_tensors)
            if isinstance(t_list, list) and t_list and isinstance(t_list[0], list):
                attention_tensors = t_list
            else:
                raise SeeDLError("attention_tensors must be a list of layer attention matrices.")

    rollout_matrix, intermediates = compute_attention_rollout(
        attention_tensors,
        include_residual=include_residual,
        head_reduction=head_reduction,
    )

    seq_len = len(rollout_matrix)
    num_layers = len(intermediates)

    # Default tokens
    if tokens is None or len(tokens) != seq_len:
        tokens = [f"T{i}" for i in range(seq_len)]

    vis_title = title or "Transformer Attention Rollout"
    res_str = (
        "with residual connections (0.5 * (A + I))"
        if include_residual
        else "without residual connections"
    )
    summary = f"Cumulative attention rollout across {num_layers} layers ({seq_len} tokens, {head_reduction} heads, {res_str})"

    notes = [
        f"Rollout computed across {num_layers} transformer layers with {head_reduction} head aggregation.",
        f"Residual connections: {'Included' if include_residual else 'Excluded'}.",
        "Rollout quantifies structural information routing; it is not a causal proof of model predictions.",
    ]

    data = {
        "presentation": "attention_rollout",
        "num_layers": num_layers,
        "seq_len": seq_len,
        "tokens": tokens,
        "rollout_matrix": rollout_matrix,
        "intermediates": intermediates,
        "include_residual": include_residual,
        "head_reduction": head_reduction,
    }

    metadata = {
        "Layers": num_layers,
        "Sequence Length": seq_len,
        "Residual": "Yes" if include_residual else "No",
        "Head Reduction": head_reduction,
    }

    return Visualization(
        kind="attention_rollout",
        title=vis_title,
        summary=summary,
        data=data,
        metadata=metadata,
        notes=notes,
        backend=resolved_backend,
    )


def attention_layers(
    attention_tensors: list[Any] | Any,
    *,
    tokens: list[str] | None = None,
    backend: str = "auto",
    title: str | None = None,
) -> Visualization:
    """Multi-layer Transformer Attention Explorer."""
    resolved_backend = resolve_backend(backend)

    if not isinstance(attention_tensors, list):
        t_list = _to_numpy_or_list(attention_tensors)
        if isinstance(t_list, list) and t_list and isinstance(t_list[0], list):
            attention_tensors = t_list
        else:
            raise SeeDLError("attention_tensors must be a list of layer attention matrices.")

    parsed_layers: list[dict[str, Any]] = []
    layer_entropies: list[float] = []

    for l_idx, raw_l in enumerate(attention_tensors):
        data = _to_numpy_or_list(raw_l)
        while (
            isinstance(data, list)
            and len(data) == 1
            and isinstance(data[0], list)
            and isinstance(data[0][0], list)
            and isinstance(data[0][0][0], list)
        ):
            data = data[0]

        if isinstance(data[0][0], list):
            # [heads, seq, seq]
            heads = data
        else:
            # [seq, seq] -> wrap in 1 head
            heads = [data]

        num_heads = len(heads)
        seq_len = len(heads[0])
        # Compute head entropies: - sum p * log2(p)
        head_ent = []
        for h in range(num_heads):
            tot_h_ent = 0.0
            for row in heads[h]:
                for p in row:
                    if p > 1e-12:
                        tot_h_ent -= p * math.log2(p)
            head_ent.append(tot_h_ent / float(seq_len) if seq_len > 0 else 0.0)

        layer_entropies.append(sum(head_ent) / len(head_ent) if head_ent else 0.0)
        parsed_layers.append(
            {
                "layer_index": l_idx,
                "num_heads": num_heads,
                "seq_len": seq_len,
                "heads": heads,
                "head_entropies": head_ent,
            }
        )

    num_layers = len(parsed_layers)
    seq_len = int(parsed_layers[0]["seq_len"]) if parsed_layers else 0
    if tokens is None or len(tokens) != seq_len:
        tokens = [f"T{i}" for i in range(seq_len)]

    vis_title = title or "Multi-Layer Transformer Attention Observatory"
    heads_count = parsed_layers[0]["num_heads"] if parsed_layers else 0
    summary = f"Observing {num_layers} layers ({heads_count} heads/layer, {seq_len} tokens)"

    vis_data: dict[str, Any] = {
        "presentation": "attention_layers",
        "num_layers": num_layers,
        "seq_len": seq_len,
        "tokens": tokens,
        "layers": parsed_layers,
        "layer_entropies": layer_entropies,
    }

    notes = [
        f"Multi-layer attention captured across {num_layers} layers.",
        "Average layer attention entropy tracks dispersal vs concentration across depth.",
    ]

    return Visualization(
        kind="attention_layers",
        title=vis_title,
        summary=summary,
        data=vis_data,
        metadata={"Layers": num_layers, "Sequence Length": seq_len},
        notes=notes,
        backend=resolved_backend,
    )


def attention_rollout_html(view: Visualization) -> str:
    """Generate self-contained interactive HTML for Attention Rollout."""
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
  <title>{title} — SeeDL Attention Rollout</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card: #1e293b;
      --card-alt: #334155;
      --border: #475569;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-hover: #0284c7;
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
    .controls {{
      display: flex;
      gap: 16px;
      margin-bottom: 20px;
      align-items: center;
      flex-wrap: wrap;
    }}
    select, button {{
      background: var(--bg);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 8px 14px;
      border-radius: 6px;
      font-size: 13px;
      cursor: pointer;
    }}
    .main-grid {{
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 20px;
    }}
    @media (max-width: 900px) {{
      .main-grid {{ grid-template-columns: 1fr; }}
    }}
    .matrix-box {{
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      overflow: auto;
    }}
    .side-box {{
      background: var(--card-alt);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
    }}
    table.matrix-table {{
      border-collapse: collapse;
      font-size: 12px;
    }}
    table.matrix-table th, table.matrix-table td {{
      padding: 6px 10px;
      text-align: center;
      border: 1px solid var(--border);
    }}
    table.matrix-table th {{ background: var(--card-alt); color: var(--accent); }}
  </style>
</head>
<body>
<div class="container">
  <header>
    <h1>{title}</h1>
    <p>{summary}</p>
  </header>

  <div class="controls">
    <label style="font-size: 13px; color: var(--muted);">View Step:</label>
    <select id="layer-select" onchange="renderMatrix()">
      <option value="final">Final Cumulative Rollout</option>
    </select>
  </div>

  <div class="main-grid">
    <div class="matrix-box">
      <div id="table-container"></div>
    </div>

    <div class="side-box">
      <h3 style="font-size: 16px; color: var(--accent); margin-bottom: 12px;">Rollout Diagnostics</h3>
      <p style="font-size: 13px; color: var(--muted); margin-bottom: 8px;"><strong>Residual Connections:</strong> {data.get("include_residual")}</p>
      <p style="font-size: 13px; color: var(--muted); margin-bottom: 8px;"><strong>Head Aggregation:</strong> {data.get("head_reduction")}</p>
      <p style="font-size: 13px; color: var(--muted); margin-bottom: 8px;"><strong>Total Layers:</strong> {data.get("num_layers")}</p>
      <p style="font-size: 13px; color: var(--muted); margin-bottom: 8px;"><strong>Sequence Tokens:</strong> {data.get("seq_len")}</p>
      <div style="margin-top: 16px; font-size: 12px; color: var(--muted); line-height: 1.6;">
        Attention rollout simulates token information propagation by recursively multiplying row-stochastic attention matrices across layers:
        <br/><br/>
        <code style="background: rgba(0,0,0,0.3); padding: 4px 6px; border-radius: 4px;">R = A'_L @ ... @ A'_1</code>
      </div>
    </div>
  </div>
</div>

<script>
  const payload = {payload_json};
  const select = document.getElementById('layer-select');

  // Populate intermediate steps
  for (let i = 0; i < payload.num_layers; i++) {{
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = `Layer ${{i + 1}} Intermediate Rollout`;
    select.appendChild(opt);
  }}

  function renderMatrix() {{
    const val = select.value;
    const mat = val === 'final' ? payload.rollout_matrix : payload.intermediates[parseInt(val)];
    const tokens = payload.tokens;
    const n = tokens.length;

    let html = '<table class="matrix-table"><thead><tr><th>Query \\ Key</th>';
    tokens.forEach(t => {{ html += `<th>${{t}}</th>`; }});
    html += '</tr></thead><tbody>';

    for (let r = 0; r < n; r++) {{
      html += `<tr><th>${{tokens[r]}}</th>`;
      for (let c = 0; c < n; c++) {{
        const weight = mat[r][c];
        const intensity = Math.min(1.0, weight);
        const bg = `rgba(56, 189, 248, ${{intensity * 0.85 + 0.05}})`;
        const col = intensity > 0.5 ? '#0f172a' : '#f8fafc';
        html += `<td style="background: ${{bg}}; color: ${{col}};" title="${{tokens[r]}} -> ${{tokens[c]}}: ${{weight.toFixed(4)}}">${{weight.toFixed(3)}}</td>`;
      }}
      html += '</tr>';
    }}
    html += '</tbody></table>';
    document.getElementById('table-container').innerHTML = html;
  }}

  renderMatrix();
  window.__seedl_rollout_payload = payload;
</script>
</body>
</html>
"""


def svg_attention_rollout(view: Visualization) -> str:
    """Render static SVG of Attention Rollout matrix."""
    data = view.data
    tokens = data.get("tokens", [])
    matrix = data.get("rollout_matrix", [])
    n = len(tokens)
    width = 750
    height = max(400, 140 + n * 30)

    title = html.escape(view.title)
    summary = html.escape(view.summary)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        f'<rect width="{width}" height="{height}" fill="#0f172a"/>',
        '<text x="40" y="38" font-family="system-ui, sans-serif" font-size="12" fill="#94a3b8" font-weight="600">SeeDL / ATTENTION ROLLOUT</text>',
        f'<text x="40" y="68" font-family="system-ui, sans-serif" font-size="20" fill="#f8fafc" font-weight="700">{title}</text>',
        f'<text x="40" y="92" font-family="system-ui, sans-serif" font-size="13" fill="#94a3b8">{summary}</text>',
    ]

    cell_size = min(36, int(500 / max(n, 1)))
    gx = 120
    gy = 140

    # Draw column token headers
    for c in range(min(n, 14)):
        t_label = html.escape(tokens[c])
        lines.append(
            f'<text x="{gx + c * cell_size + cell_size // 2}" y="{gy - 8}" font-family="monospace" font-size="11" fill="#38bdf8" text-anchor="middle">{t_label}</text>'
        )

    for r in range(min(n, 14)):
        t_label = html.escape(tokens[r])
        lines.append(
            f'<text x="{gx - 10}" y="{gy + r * cell_size + cell_size // 2 + 4}" font-family="monospace" font-size="11" fill="#38bdf8" text-anchor="end">{t_label}</text>'
        )
        for c in range(min(n, 14)):
            w = matrix[r][c] if r < len(matrix) and c < len(matrix[r]) else 0.0
            col_val = int(min(255, 30 + w * 220))
            lines.append(
                f'<rect x="{gx + c * cell_size}" y="{gy + r * cell_size}" width="{cell_size - 1}" height="{cell_size - 1}" fill="rgb(15,{col_val},{int(col_val * 0.9)})"/>'
            )

    lines.append("</svg>")
    return "\n".join(lines)
