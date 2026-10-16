"""Dataset Intelligence & Drift Analytics: Quality audits, distribution shifts, and population stability."""

from __future__ import annotations

import html
import json
import math
from typing import Any

from .exceptions import SeeDLError
from .renderers import resolve_backend
from .view import Visualization


def _to_flat_floats(data: Any, max_samples: int = 1000) -> list[float]:
    """Convert tensor, array, or list to flat list of floats with bounded sampling."""
    if hasattr(data, "detach"):
        data = data.detach()
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "numpy"):
        data = data.numpy()

    flat: list[float] = []
    if hasattr(data, "reshape"):
        data = data.reshape(-1)
        step = max(1, len(data) // max_samples) if len(data) > max_samples else 1
        for val in data[::step][:max_samples]:
            v = float(val)
            flat.append(v)
    elif isinstance(data, (list, tuple)):
        # Flatten
        stack = list(data)
        while stack and len(flat) < max_samples:
            item = stack.pop(0)
            if isinstance(item, (list, tuple)):
                stack = list(item) + stack
            elif isinstance(item, (int, float)):
                flat.append(float(item))
    return flat


def compute_histograms(
    p_vals: list[float],
    q_vals: list[float],
    bins: int = 10,
    epsilon: float = 1e-6,
) -> tuple[list[float], list[float], list[float]]:
    """Compute smoothed, normalized histogram probabilities for two empirical samples."""
    all_vals = [v for v in p_vals + q_vals if math.isfinite(v)]
    if not all_vals:
        return [1.0 / bins] * bins, [1.0 / bins] * bins, [0.0] * (bins + 1)

    min_v = min(all_vals)
    max_v = max(all_vals)
    if min_v == max_v:
        max_v += 1.0

    step = (max_v - min_v) / float(bins)
    bin_edges = [min_v + i * step for i in range(bins + 1)]

    def get_bin(val: float) -> int:
        idx = int((val - min_v) / step)
        return min(bins - 1, max(0, idx))

    p_counts = [0] * bins
    for v in p_vals:
        if math.isfinite(v):
            p_counts[get_bin(v)] += 1

    q_counts = [0] * bins
    for v in q_vals:
        if math.isfinite(v):
            q_counts[get_bin(v)] += 1

    # Laplace smoothing
    tot_p = len(p_vals) + bins * epsilon
    tot_q = len(q_vals) + bins * epsilon
    p_probs = [(c + epsilon) / tot_p for c in p_counts]
    q_probs = [(c + epsilon) / tot_q for c in q_counts]

    return p_probs, q_probs, bin_edges


def compute_psi(p_probs: list[float], q_probs: list[float]) -> float:
    """Compute Population Stability Index (PSI) between baseline P and current Q."""
    psi = 0.0
    for p, q in zip(p_probs, q_probs, strict=False):
        if p > 0 and q > 0:
            psi += (q - p) * math.log(q / p)
    return max(0.0, float(psi))


def compute_jensen_shannon_divergence(p_probs: list[float], q_probs: list[float]) -> float:
    """Compute Jensen-Shannon Divergence D_JS(P || Q) in bits [0.0, 1.0]."""
    m_probs = [0.5 * (p + q) for p, q in zip(p_probs, q_probs, strict=False)]

    kl_pm = sum(
        p * math.log2(p / m) for p, m in zip(p_probs, m_probs, strict=False) if p > 0 and m > 0
    )
    kl_qm = sum(
        q * math.log2(q / m) for q, m in zip(q_probs, m_probs, strict=False) if q > 0 and m > 0
    )

    return max(0.0, min(1.0, 0.5 * (kl_pm + kl_qm)))


def dataset_drift(
    reference_data: Any,
    comparison_data: Any,
    *,
    max_samples: int = 1000,
    bins: int = 10,
    backend: str = "auto",
    title: str | None = None,
) -> Visualization:
    """Measure empirical distribution drift between reference and production dataset samples.

    Evaluates:
    - Jensen-Shannon Divergence (JSD)
    - Population Stability Index (PSI)
    - Mean and variance shifts
    - Relative histogram distance
    """
    resolved_backend = resolve_backend(backend)

    p_vals = _to_flat_floats(reference_data, max_samples=max_samples)
    q_vals = _to_flat_floats(comparison_data, max_samples=max_samples)

    if not p_vals or not q_vals:
        raise SeeDLError(
            "Both reference_data and comparison_data must contain finite numeric elements."
        )

    p_probs, q_probs, bin_edges = compute_histograms(p_vals, q_vals, bins=bins)
    psi_val = compute_psi(p_probs, q_probs)
    jsd_val = compute_jensen_shannon_divergence(p_probs, q_probs)

    mean_p = sum(p_vals) / len(p_vals)
    mean_q = sum(q_vals) / len(q_vals)
    delta_mean = mean_q - mean_p

    var_p = sum((v - mean_p) ** 2 for v in p_vals) / len(p_vals)
    var_q = sum((v - mean_q) ** 2 for v in q_vals) / len(q_vals)
    std_p = math.sqrt(var_p)
    std_q = math.sqrt(var_q)

    # Classify drift severity based on standard PSI benchmarks
    if psi_val < 0.1:
        drift_level = "Negligible Drift"
        status_color = "#34d399"
    elif psi_val < 0.25:
        drift_level = "Moderate Drift (Monitoring Recommended)"
        status_color = "#fbbf24"
    else:
        drift_level = "Significant Drift (Model Retraining Advisable)"
        status_color = "#f87171"

    vis_title = title or "Dataset Distribution Drift Analysis"
    summary = (
        f"PSI: {psi_val:.4f} ({drift_level}) · JSD: {jsd_val:.4f} bits · ΔMean: {delta_mean:+.3e}"
    )

    notes = [
        f"Population Stability Index (PSI = {psi_val:.4f}): standard industry benchmark (<0.10: negligible, 0.10-0.25: moderate, >0.25: significant).",
        f"Jensen-Shannon Divergence (JSD = {jsd_val:.4f} bits): symmetric, smoothed information-theoretic distance.",
        f"Sample sizes: Reference={len(p_vals)}, Comparison={len(q_vals)} (bounded at {max_samples}).",
    ]

    data = {
        "presentation": "dataset_drift",
        "psi": psi_val,
        "jsd": jsd_val,
        "drift_level": drift_level,
        "status_color": status_color,
        "reference": {
            "samples": len(p_vals),
            "mean": mean_p,
            "std": std_p,
            "probabilities": p_probs,
        },
        "comparison": {
            "samples": len(q_vals),
            "mean": mean_q,
            "std": std_q,
            "probabilities": q_probs,
        },
        "bin_edges": bin_edges,
    }

    metadata = {
        "PSI": f"{psi_val:.4f}",
        "JSD": f"{jsd_val:.4f} bits",
        "Assessment": drift_level,
        "Ref Samples": len(p_vals),
        "Comp Samples": len(q_vals),
    }

    return Visualization(
        kind="dataset_drift",
        title=vis_title,
        summary=summary,
        data=data,
        metadata=metadata,
        notes=notes,
        backend=resolved_backend,
    )


def dataset_health(
    dataset: Any,
    *,
    max_samples: int = 1000,
    backend: str = "auto",
    title: str | None = None,
) -> Visualization:
    """Analyze dataset quality, missing values, nonfinite numbers, outliers, and balance."""
    resolved_backend = resolve_backend(backend)
    vals = _to_flat_floats(dataset, max_samples=max_samples)

    if not vals:
        raise SeeDLError("Dataset contains no numeric samples.")

    n = len(vals)
    finite_vals = [v for v in vals if math.isfinite(v)]
    nonfinite_cnt = n - len(finite_vals)

    min_v = min(finite_vals) if finite_vals else 0.0
    max_v = max(finite_vals) if finite_vals else 0.0
    mean_v = sum(finite_vals) / len(finite_vals) if finite_vals else 0.0
    var_v = sum((v - mean_v) ** 2 for v in finite_vals) / len(finite_vals) if finite_vals else 0.0
    std_v = math.sqrt(var_v)

    # Outlier detection via IQR
    sorted_v = sorted(finite_vals)
    q1 = sorted_v[len(sorted_v) // 4] if sorted_v else 0.0
    q3 = sorted_v[(3 * len(sorted_v)) // 4] if sorted_v else 0.0
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = [v for v in finite_vals if v < lower_bound or v > upper_bound]
    outlier_fraction = len(outliers) / len(finite_vals) if finite_vals else 0.0

    is_constant = (max_v - min_v) < 1e-12

    vis_title = title or "Dataset Health & Quality Audit"
    summary = f"Audited {n} sample values · Min: {min_v:.3f}, Max: {max_v:.3f}, Outliers: {outlier_fraction:.1%}"

    notes = [
        f"Audited {n} bounded samples from dataset.",
        f"Nonfinite (NaN/Inf) count: {nonfinite_cnt}.",
        f"IQR-detected outliers: {len(outliers)} ({outlier_fraction:.1%}).",
    ]
    if is_constant:
        notes.append("Warning: Data values are constant (zero variance).")

    data = {
        "presentation": "dataset_health",
        "sample_count": n,
        "nonfinite_count": nonfinite_cnt,
        "min": min_v,
        "max": max_v,
        "mean": mean_v,
        "std": std_v,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "outlier_count": len(outliers),
        "outlier_fraction": outlier_fraction,
        "is_constant": is_constant,
    }

    return Visualization(
        kind="dataset_health",
        title=vis_title,
        summary=summary,
        data=data,
        metadata={"Samples": n, "Nonfinite": nonfinite_cnt, "Outliers": f"{outlier_fraction:.1%}"},
        notes=notes,
        backend=resolved_backend,
    )


def dataset_drift_html(view: Visualization) -> str:
    """Generate self-contained interactive HTML for dataset drift."""
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
  <title>{title} — SeeDL Dataset Drift</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card: #1e293b;
      --card-alt: #334155;
      --border: #475569;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
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
      max-width: 1050px;
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
    .stats-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .stat-card {{
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
    }}
    .stat-label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; }}
    .stat-val {{ font-size: 22px; font-weight: 700; margin-top: 4px; font-family: monospace; }}
  </style>
</head>
<body>
<div class="container">
  <header>
    <h1>{title}</h1>
    <p>{summary}</p>
  </header>

  <div class="stats-row">
    <div class="stat-card">
      <div class="stat-label">Population Stability Index</div>
      <div class="stat-val" style="color: {data.get("status_color", "#38bdf8")};">{data.get("psi", 0):.4f}</div>
      <div style="font-size: 12px; color: var(--muted); margin-top: 4px;">{data.get("drift_level")}</div>
    </div>

    <div class="stat-card">
      <div class="stat-label">Jensen-Shannon Divergence</div>
      <div class="stat-val" style="color: var(--accent);">{data.get("jsd", 0):.4f} bits</div>
      <div style="font-size: 12px; color: var(--muted); margin-top: 4px;">Information-theoretic shift</div>
    </div>

    <div class="stat-card">
      <div class="stat-label">Mean Shift (Δμ)</div>
      <div class="stat-val">{(data.get("comparison", {}).get("mean", 0) - data.get("reference", {}).get("mean", 0)):+.3e}</div>
      <div style="font-size: 12px; color: var(--muted); margin-top: 4px;">Ref: {data.get("reference", {}).get("mean", 0):.2f} → Comp: {data.get("comparison", {}).get("mean", 0):.2f}</div>
    </div>
  </div>
</div>
<script>
  window.__seedl_drift_payload = {payload_json};
</script>
</body>
</html>
"""


def svg_dataset_drift(view: Visualization) -> str:
    """Render static SVG for Dataset Drift."""
    data = view.data
    width = 750
    height = 320
    title = html.escape(view.title)
    summary = html.escape(view.summary)

    psi = data.get("psi", 0.0)
    jsd = data.get("jsd", 0.0)
    level = html.escape(str(data.get("drift_level", "Unknown")))
    color = data.get("status_color", "#38bdf8")

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        f'<rect width="{width}" height="{height}" fill="#0f172a"/>',
        '<text x="40" y="38" font-family="system-ui, sans-serif" font-size="12" fill="#94a3b8" font-weight="600">SeeDL / DATASET DRIFT INTELLIGENCE</text>',
        f'<text x="40" y="68" font-family="system-ui, sans-serif" font-size="20" fill="#f8fafc" font-weight="700">{title}</text>',
        f'<text x="40" y="92" font-family="system-ui, sans-serif" font-size="13" fill="#94a3b8">{summary}</text>',
        # Card
        '<g transform="translate(60, 130)">',
        '<rect width="630" height="150" rx="8" fill="#1e293b" stroke="#334155"/>',
        '<text x="30" y="40" font-family="system-ui, sans-serif" font-size="14" fill="#94a3b8">Population Stability Index (PSI):</text>',
        f'<text x="320" y="40" font-family="monospace" font-size="18" font-weight="700" fill="{color}">{psi:.4f} ({level})</text>',
        '<text x="30" y="85" font-family="system-ui, sans-serif" font-size="14" fill="#94a3b8">Jensen-Shannon Divergence:</text>',
        f'<text x="320" y="85" font-family="monospace" font-size="16" font-weight="700" fill="#38bdf8">{jsd:.4f} bits</text>',
        '<text x="30" y="125" font-family="system-ui, sans-serif" font-size="12" fill="#64748b">Benchmark: PSI &lt; 0.10: Stable · 0.10 - 0.25: Moderate · &gt; 0.25: Significant</text>',
        "</g>",
        "</svg>",
    ]
    return "\n".join(lines)
