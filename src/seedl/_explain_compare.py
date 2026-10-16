"""Cross-method attribution comparison and interpretability alignment."""

from __future__ import annotations

import html
import json
import math
from typing import Any

from ._explain import explain_prediction
from .renderers import resolve_backend
from .view import Visualization


def _flatten_numbers(obj: Any) -> list[float]:
    """Recursively flatten nested lists/numbers into a 1D float list."""
    res: list[float] = []
    if isinstance(obj, (int, float)):
        res.append(float(obj))
    elif isinstance(obj, list):
        for item in obj:
            res.extend(_flatten_numbers(item))
    return res


def _compute_similarity(a: list[float], b: list[float]) -> dict[str, float | None]:
    """Compute Pearson correlation and Cosine similarity between two aligned 1D numeric vectors."""
    if len(a) != len(b) or len(a) < 2:
        return {"pearson": None, "cosine": None}

    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n

    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    var_a = sum((a[i] - mean_a) ** 2 for i in range(n))
    var_b = sum((b[i] - mean_b) ** 2 for i in range(n))

    denom_pearson = math.sqrt(var_a * var_b)
    pearson = cov / denom_pearson if denom_pearson > 1e-12 else 0.0

    dot = sum(a[i] * b[i] for i in range(n))
    norm_a = math.sqrt(sum(a[i] ** 2 for i in range(n)))
    norm_b = math.sqrt(sum(b[i] ** 2 for i in range(n)))
    denom_cosine = norm_a * norm_b
    cosine = dot / denom_cosine if denom_cosine > 1e-12 else 0.0

    return {
        "pearson": float(pearson) if math.isfinite(pearson) else None,
        "cosine": float(cosine) if math.isfinite(cosine) else None,
    }


def compare_attributions(
    model: Any,
    input: Any,
    *,
    target: int | None = None,
    methods: tuple[str, ...] | list[str] = (
        "saliency",
        "integrated_gradients",
        "smoothgrad",
    ),
    steps: int = 20,
    samples: int = 25,
    stdev_spread: float = 0.15,
    baseline: Any | None = None,
    layer: str | None = None,
    abs: bool | None = None,
    seed: int | None = 42,
    backend: str = "auto",
) -> Visualization:
    """Compare multiple feature attribution methods on the same model and input.

    Provides aligned statistics, pairwise similarity metrics (Pearson correlation,
    Cosine similarity) where shapes match, and side-by-side visualization.
    """
    resolved_backend = resolve_backend(backend)

    results: list[dict[str, Any]] = []
    vector_map: dict[str, list[float]] = {}
    shapes_map: dict[str, list[int]] = {}

    for method in methods:
        try:
            data, _, _ = explain_prediction(
                model,
                input,
                target=target,
                method=method,
                steps=steps,
                samples=samples,
                stdev_spread=stdev_spread,
                baseline=baseline,
                layer=layer,
                abs=abs,
                seed=seed,
            )
            raw_attr = data["attribution"]
            flat = _flatten_numbers(raw_attr)
            total = len(flat)

            # Basic stats
            min_v = min(flat) if flat else 0.0
            max_v = max(flat) if flat else 0.0
            mean_v = sum(flat) / total if total > 0 else 0.0
            zero_cnt = sum(1 for v in flat if v == 0.0)
            sparsity = zero_cnt / total if total > 0 else 0.0

            # Determine shape
            curr = raw_attr
            shape = []
            while isinstance(curr, list) and curr:
                shape.append(len(curr))
                curr = curr[0]

            results.append(
                {
                    "method": method,
                    "target_class": data["target_class"],
                    "predicted_score": data["predicted_score"],
                    "shape": shape,
                    "num_elements": total,
                    "min": min_v,
                    "max": max_v,
                    "mean": mean_v,
                    "sparsity": sparsity,
                    "attribution": raw_attr,
                    "diagnostics": data.get("diagnostics", {}),
                }
            )
            vector_map[method] = flat
            shapes_map[method] = shape
        except Exception as exc:
            results.append(
                {
                    "method": method,
                    "error": str(exc),
                }
            )

    # Compute pairwise similarities for methods with identical shapes
    similarity_matrix: dict[str, dict[str, Any]] = {}
    valid_methods = [r["method"] for r in results if "attribution" in r]
    for m1 in valid_methods:
        similarity_matrix[m1] = {}
        for m2 in valid_methods:
            if m1 == m2:
                similarity_matrix[m1][m2] = {"pearson": 1.0, "cosine": 1.0, "shape_match": True}
            elif shapes_map[m1] == shapes_map[m2]:
                sim = _compute_similarity(vector_map[m1], vector_map[m2])
                sim["shape_match"] = True
                similarity_matrix[m1][m2] = sim
            else:
                similarity_matrix[m1][m2] = {
                    "pearson": None,
                    "cosine": None,
                    "shape_match": False,
                    "note": f"Incomparable shapes: {shapes_map[m1]} vs {shapes_map[m2]}",
                }

    target_cls = results[0].get("target_class", "auto") if results else "auto"
    title = f"Attribution Comparison (Target Class: {target_cls})"
    summary = f"Compared {len(valid_methods)} attribution method(s) on identical model input"

    notes = [
        "Feature attribution highlights input sensitivity; it does not establish causal proof of model decisions.",
        "Pairwise correlations are computed only between methods whose spatial attribution shapes match exactly.",
    ]

    data_payload = {
        "presentation": "attribution_comparison",
        "methods": results,
        "similarity_matrix": similarity_matrix,
    }

    return Visualization(
        kind="attribution_comparison",
        title=title,
        summary=summary,
        data=data_payload,
        metadata={"Methods": ", ".join(valid_methods), "Target Class": str(target_cls)},
        notes=notes,
        backend=resolved_backend,
    )


def attribution_comparison_html(view: Visualization) -> str:
    """Generate self-contained interactive HTML for comparing feature attributions."""
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
  <title>{title} — SeeDL Attribution Comparison</title>
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
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 20px;
      margin-bottom: 24px;
    }}
    .method-card {{
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
    }}
    .method-card h3 {{
      font-size: 16px;
      color: var(--accent);
      margin-bottom: 8px;
      text-transform: capitalize;
    }}
    .metric-row {{
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      padding: 4px 0;
      border-bottom: 1px dashed rgba(255,255,255,0.08);
    }}
    .metric-label {{ color: var(--muted); }}
    .metric-value {{ font-family: monospace; font-weight: 600; }}
    table.sim-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      margin-top: 12px;
    }}
    table.sim-table th, table.sim-table td {{
      padding: 8px 12px;
      border: 1px solid var(--border);
      text-align: center;
    }}
    table.sim-table th {{ background: var(--card-alt); color: var(--text); }}
  </style>
</head>
<body>
<div class="container">
  <header>
    <h1>{title}</h1>
    <p>{summary}</p>
  </header>

  <h2 style="font-size: 16px; margin-bottom: 12px; color: var(--accent);">Method Profiles & Attribution Statistics</h2>
  <div class="grid" id="cards-container"></div>

  <h2 style="font-size: 16px; margin-bottom: 12px; color: var(--accent);">Pairwise Attribution Similarity (Pearson Correlation)</h2>
  <div style="overflow-x: auto;">
    <table class="sim-table" id="sim-table"></table>
  </div>
</div>

<script>
  const payload = {payload_json};
  const cardsContainer = document.getElementById('cards-container');
  const methods = payload.methods || [];

  methods.forEach(m => {{
    if (m.error) {{
      cardsContainer.innerHTML += `
        <div class="method-card">
          <h3>${{m.method}}</h3>
          <p style="color: #f87171; font-size: 13px;">Error: ${{m.error}}</p>
        </div>
      `;
      return;
    }}
    cardsContainer.innerHTML += `
      <div class="method-card">
        <h3>${{m.method.replace('_', ' ')}}</h3>
        <div class="metric-row"><span class="metric-label">Target Class</span><span class="metric-value">${{m.target_class}}</span></div>
        <div class="metric-row"><span class="metric-label">Predicted Score</span><span class="metric-value">${{m.predicted_score.toFixed(4)}}</span></div>
        <div class="metric-row"><span class="metric-label">Shape</span><span class="metric-value">[${{m.shape.join(', ')}}]</span></div>
        <div class="metric-row"><span class="metric-label">Mean Attribution</span><span class="metric-value">${{m.mean.toFixed(5)}}</span></div>
        <div class="metric-row"><span class="metric-label">Min / Max</span><span class="metric-value">${{m.min.toFixed(4)}} / ${{m.max.toFixed(4)}}</span></div>
        <div class="metric-row"><span class="metric-label">Sparsity (Zero %)</span><span class="metric-value">${{(m.sparsity * 100).toFixed(1)}}%</span></div>
      </div>
    `;
  }});

  // Similarity Table
  const sim = payload.similarity_matrix || {{}};
  const keys = Object.keys(sim);
  if (keys.length > 0) {{
    const table = document.getElementById('sim-table');
    let headHtml = '<thead><tr><th>Method</th>' + keys.map(k => `<th>${{k.replace('_', ' ')}}</th>`).join('') + '</tr></thead>';
    let bodyHtml = '<tbody>';
    keys.forEach(k1 => {{
      bodyHtml += `<tr><td><strong>${{k1.replace('_', ' ')}}</strong></td>`;
      keys.forEach(k2 => {{
        const cell = sim[k1][k2];
        if (cell.shape_match && cell.pearson != null) {{
          bodyHtml += `<td>${{cell.pearson.toFixed(3)}}</td>`;
        }} else {{
          bodyHtml += `<td style="color: var(--muted); font-size: 11px;">Incomparable</td>`;
        }}
      }});
      bodyHtml += '</tr>';
    }});
    bodyHtml += '</tbody>';
    table.innerHTML = headHtml + bodyHtml;
  }}
</script>
</body>
</html>
"""


def svg_attribution_comparison(view: Visualization) -> str:
    """Render static SVG comparison of attribution methods."""
    data = view.data
    methods = [m for m in data.get("methods", []) if not m.get("error")]
    width = 800
    height = max(360, 120 + len(methods) * 55)

    title = html.escape(view.title)
    summary = html.escape(view.summary)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        f'<rect width="{width}" height="{height}" fill="#0f172a"/>',
        '<text x="40" y="38" font-family="system-ui, sans-serif" font-size="12" fill="#94a3b8" font-weight="600">SeeDL / ATTRIBUTION COMPARISON</text>',
        f'<text x="40" y="68" font-family="system-ui, sans-serif" font-size="20" fill="#f8fafc" font-weight="700">{title}</text>',
        f'<text x="40" y="92" font-family="system-ui, sans-serif" font-size="13" fill="#94a3b8">{summary}</text>',
    ]

    y = 130
    for idx, m in enumerate(methods):
        name = html.escape(m.get("method", f"Method {idx}").replace("_", " ").title())
        shape_str = str(m.get("shape", []))
        mean_v = m.get("mean", 0.0)

        lines.extend(
            [
                f'<g transform="translate(60, {y})">',
                '<rect width="680" height="42" rx="6" fill="#1e293b" stroke="#334155" stroke-width="1"/>',
                f'<text x="20" y="26" font-family="system-ui, sans-serif" font-size="14" font-weight="600" fill="#38bdf8">{name}</text>',
                f'<text x="320" y="26" font-family="monospace" font-size="12" fill="#94a3b8">Shape {shape_str}</text>',
                f'<text x="660" y="26" font-family="monospace" font-size="12" fill="#34d399" text-anchor="end">Mean: {mean_v:.4e}</text>',
                "</g>",
            ]
        )
        y += 55

    lines.append("</svg>")
    return "\n".join(lines)
