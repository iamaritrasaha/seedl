"""Training run and experiment comparison diagnostics."""

from __future__ import annotations

import html
import json
import math
from typing import Any

from ._training import Tracker
from .exceptions import SeeDLError
from .renderers import resolve_backend
from .view import Visualization


def _extract_history_dict(run: Any) -> dict[str, Any]:
    """Extract metric history dictionary from a Tracker, TrainingRecorder, or dict."""
    if isinstance(run, Tracker):
        return run.history()
    if hasattr(run, "history"):
        hist = run.history
        return hist() if callable(hist) else hist
    if isinstance(run, dict):
        return run
    raise SeeDLError(
        f"Cannot extract training history from {type(run).__name__}; expected Tracker or dict."
    )


def compare_runs(
    run_a: Any,
    run_b: Any,
    *,
    name_a: str = "Run A",
    name_b: str = "Run B",
    metrics: list[str] | None = None,
    backend: str = "auto",
    title: str | None = None,
) -> Visualization:
    """Compare two training runs or experiment histories with step alignment.

    Invariants:
    - Never silently assumes nonaligned steps are equivalent; reports step count differences.
    - Compares loss progression, metric divergences, and final delta.
    - Warns explicitly on missing metrics between runs.
    """
    resolved_backend = resolve_backend(backend)

    hist_a = _extract_history_dict(run_a)
    hist_b = _extract_history_dict(run_b)

    keys_a = set(hist_a.keys())
    keys_b = set(hist_b.keys())
    common_keys = sorted(keys_a.intersection(keys_b))

    if metrics is not None:
        common_keys = [k for k in common_keys if k in metrics]

    if not common_keys:
        raise SeeDLError(
            f"No common metrics found between runs. Run A metrics: {sorted(keys_a)}, Run B metrics: {sorted(keys_b)}."
        )

    missing_in_b = sorted(keys_a - keys_b)
    missing_in_a = sorted(keys_b - keys_a)

    comparisons: list[dict[str, Any]] = []
    warnings: list[str] = []

    if missing_in_b:
        warnings.append(f"Metrics in {name_a} missing from {name_b}: {', '.join(missing_in_b)}.")
    if missing_in_a:
        warnings.append(f"Metrics in {name_b} missing from {name_a}: {', '.join(missing_in_a)}.")

    for m in common_keys:
        seq_a = [float(v) for v in hist_a[m] if math.isfinite(float(v))]
        seq_b = [float(v) for v in hist_b[m] if math.isfinite(float(v))]

        len_a = len(seq_a)
        len_b = len(seq_b)
        aligned_len = min(len_a, len_b)

        if len_a != len_b:
            warnings.append(
                f"Metric '{m}' step count mismatch: {name_a} has {len_a} steps, {name_b} has {len_b} steps. Aligning first {aligned_len} steps."
            )

        final_a = seq_a[-1] if seq_a else None
        final_b = seq_b[-1] if seq_b else None
        delta = (final_b - final_a) if (final_a is not None and final_b is not None) else None

        min_a = min(seq_a) if seq_a else None
        min_b = min(seq_b) if seq_b else None

        # Pointwise absolute differences
        diffs = [abs(seq_b[i] - seq_a[i]) for i in range(aligned_len)]
        mean_diff = sum(diffs) / len(diffs) if diffs else 0.0

        comparisons.append(
            {
                "metric": m,
                "steps_a": len_a,
                "steps_b": len_b,
                "final_a": final_a,
                "final_b": final_b,
                "min_a": min_a,
                "min_b": min_b,
                "delta": delta,
                "mean_abs_difference": mean_diff,
                "history_a": seq_a,
                "history_b": seq_b,
            }
        )

    vis_title = title or f"Training Run Comparison: {name_a} vs {name_b}"
    summary = f"Comparing {len(common_keys)} shared metric(s) across {name_a} and {name_b}"

    notes = [
        f"Compared runs: {name_a} vs {name_b}.",
        f"Shared metrics: {', '.join(common_keys)}.",
    ]
    notes.extend(warnings)

    data = {
        "presentation": "run_comparison",
        "name_a": name_a,
        "name_b": name_b,
        "metrics": comparisons,
        "missing_in_a": missing_in_a,
        "missing_in_b": missing_in_b,
    }

    metadata = {
        "Run A": name_a,
        "Run B": name_b,
        "Shared Metrics": len(common_keys),
    }

    return Visualization(
        kind="run_comparison",
        title=vis_title,
        summary=summary,
        data=data,
        metadata=metadata,
        notes=notes,
        backend=resolved_backend,
    )


def run_comparison_html(view: Visualization) -> str:
    """Generate self-contained interactive HTML for training run comparison."""
    data = view.data
    title = html.escape(view.title)
    summary = html.escape(view.summary)
    payload_json = (
        json.dumps(data).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )

    name_a = html.escape(data.get("name_a", "Run A"))
    name_b = html.escape(data.get("name_b", "Run B"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{title} — SeeDL Run Comparison</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card: #1e293b;
      --card-alt: #334155;
      --border: #475569;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --success: #34d399;
      --warning: #fbbf24;
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
      max-width: 1100px;
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
    table.comp-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      margin-bottom: 24px;
    }}
    table.comp-table th, table.comp-table td {{
      padding: 10px 14px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }}
    table.comp-table th {{ background: var(--card-alt); color: var(--text); }}
  </style>
</head>
<body>
<div class="container">
  <header>
    <h1>{title}</h1>
    <p>{summary}</p>
  </header>

  <h2 style="font-size: 16px; margin-bottom: 12px; color: var(--accent);">Metric Comparisons</h2>
  <div style="overflow-x: auto;">
    <table class="comp-table">
      <thead>
        <tr>
          <th>Metric</th>
          <th>{name_a} Final</th>
          <th>{name_b} Final</th>
          <th>Δ ({name_b} - {name_a})</th>
          <th>Mean |Diff|</th>
        </tr>
      </thead>
      <tbody id="metrics-body"></tbody>
    </table>
  </div>
</div>

<script>
  const payload = {payload_json};
  const body = document.getElementById('metrics-body');
  (payload.metrics || []).forEach(m => {{
    const row = document.createElement('tr');
    const fA = m.final_a != null ? m.final_a.toFixed(4) : '-';
    const fB = m.final_b != null ? m.final_b.toFixed(4) : '-';
    const dStr = m.delta != null ? (m.delta > 0 ? '+' : '') + m.delta.toFixed(4) : '-';
    const dCol = m.delta != null ? (m.delta < 0 ? '#34d399' : '#f87171') : 'inherit';
    row.innerHTML = `
      <td><strong>${{m.metric}}</strong></td>
      <td><code>${{fA}}</code></td>
      <td><code>${{fB}}</code></td>
      <td style="color:${{dCol}}; font-weight:600;"><code>${{dStr}}</code></td>
      <td><code>${{m.mean_abs_difference.toFixed(4)}}</code></td>
    `;
    body.appendChild(row);
  }});
</script>
</body>
</html>
"""


def svg_run_comparison(view: Visualization) -> str:
    """Render static SVG for training run comparison."""
    data = view.data
    metrics = data.get("metrics", [])
    width = 750
    height = max(340, 120 + len(metrics) * 50)

    title = html.escape(view.title)
    summary = html.escape(view.summary)
    name_a = html.escape(data.get("name_a", "Run A"))
    name_b = html.escape(data.get("name_b", "Run B"))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        f'<rect width="{width}" height="{height}" fill="#0f172a"/>',
        '<text x="40" y="38" font-family="system-ui, sans-serif" font-size="12" fill="#94a3b8" font-weight="600">SeeDL / RUN COMPARISON</text>',
        f'<text x="40" y="68" font-family="system-ui, sans-serif" font-size="20" fill="#f8fafc" font-weight="700">{title}</text>',
        f'<text x="40" y="92" font-family="system-ui, sans-serif" font-size="13" fill="#94a3b8">{summary}</text>',
    ]

    y = 130
    for idx, m in enumerate(metrics[:12]):
        m_name = html.escape(m.get("metric", f"Metric {idx}"))
        f_a = m.get("final_a", 0.0)
        f_b = m.get("final_b", 0.0)
        lines.extend(
            [
                f'<g transform="translate(60, {y})">',
                '<rect width="630" height="40" rx="6" fill="#1e293b" stroke="#334155"/>',
                f'<text x="20" y="25" font-family="system-ui, sans-serif" font-size="13" font-weight="600" fill="#38bdf8">{m_name}</text>',
                f'<text x="260" y="25" font-family="monospace" font-size="12" fill="#94a3b8">{name_a}: {f_a:.4g}</text>',
                f'<text x="450" y="25" font-family="monospace" font-size="12" fill="#34d399">{name_b}: {f_b:.4g}</text>',
                "</g>",
            ]
        )
        y += 50

    lines.append("</svg>")
    return "\n".join(lines)
