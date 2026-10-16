"""Renderers for Neural Execution Tracer and Interactive Tensor Flow Explorer."""

from __future__ import annotations

import html
import json
from typing import Any

from .exceptions import MissingDependencyError


def _sanitize_json(obj: Any) -> str:
    """Safely serialize JSON payload to prevent script injection."""
    return json.dumps(obj).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def trace_html(view: Any) -> str:
    """Generate self-contained, offline interactive HTML for Neural Execution Trace and Tensor Flow Explorer."""
    data = view.data
    title = html.escape(view.title)
    summary = html.escape(view.summary)
    payload_json = _sanitize_json(data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{title} — SeeDL Neural Observatory</title>
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
      --success: #34d399;
      --warning: #fbbf24;
      --danger: #f87171;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 20px;
      line-height: 1.5;
    }}
    .container {{
      max-width: 1300px;
      margin: 0 auto;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 24px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
      margin-bottom: 20px;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .title-group h1 {{
      font-size: 22px;
      font-weight: 700;
      color: var(--accent);
    }}
    .title-group p {{
      color: var(--muted);
      font-size: 14px;
      margin-top: 4px;
    }}
    .meta-badges {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .badge {{
      background: var(--card-alt);
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      border: 1px solid var(--border);
    }}
    .tab-bar {{
      display: flex;
      gap: 8px;
      margin-bottom: 20px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
    }}
    .tab-btn {{
      background: transparent;
      border: 1px solid transparent;
      color: var(--muted);
      padding: 8px 16px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 14px;
      font-weight: 600;
      transition: all 0.2s;
    }}
    .tab-btn.active {{
      background: var(--accent);
      color: #0f172a;
    }}
    .tab-btn:hover:not(.active) {{
      background: var(--card-alt);
      color: var(--text);
    }}
    .controls {{
      display: flex;
      gap: 12px;
      margin-bottom: 16px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .search-input {{
      background: var(--bg);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 8px 14px;
      border-radius: 6px;
      font-size: 13px;
      flex: 1;
      min-width: 200px;
    }}
    .filter-btn {{
      background: var(--card-alt);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 8px 14px;
      border-radius: 6px;
      font-size: 13px;
      cursor: pointer;
    }}
    .filter-btn.active {{
      border-color: var(--accent);
      color: var(--accent);
    }}
    .main-layout {{
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 20px;
      min-height: 540px;
    }}
    @media (max-width: 900px) {{
      .main-layout {{ grid-template-columns: 1fr; }}
    }}
    .viewport {{
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      overflow: auto;
      position: relative;
    }}
    .inspector {{
      background: var(--card-alt);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .inspector h3 {{
      font-size: 16px;
      color: var(--accent);
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
    }}
    .metric-row {{
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      padding: 4px 0;
      border-bottom: 1px dashed rgba(255,255,255,0.08);
    }}
    .metric-label {{ color: var(--muted); }}
    .metric-value {{ font-weight: 600; font-family: monospace; }}
    .node-list {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .flow-node {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px 14px;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      transition: border-color 0.15s, background 0.15s;
    }}
    .flow-node:hover {{
      border-color: var(--accent);
      background: rgba(56, 189, 248, 0.08);
    }}
    .flow-node.selected {{
      border-color: var(--accent);
      background: rgba(56, 189, 248, 0.16);
      box-shadow: 0 0 10px rgba(56, 189, 248, 0.2);
    }}
    .node-info {{
      display: flex;
      flex-direction: column;
      gap: 2px;
    }}
    .node-name {{ font-weight: 600; font-size: 14px; }}
    .node-type {{ font-size: 12px; color: var(--muted); }}
    .node-shape {{ font-family: monospace; font-size: 12px; color: var(--accent); }}
    .bar-chart {{
      display: flex;
      align-items: flex-end;
      gap: 2px;
      height: 60px;
      padding: 8px 0;
      border-bottom: 1px solid var(--border);
    }}
    .bar-col {{
      background: var(--accent);
      flex: 1;
      min-height: 2px;
      border-radius: 1px;
    }}
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
    table.data-table th {{
      color: var(--muted);
      font-weight: 600;
      background: var(--card-alt);
    }}
    table.data-table tr:hover td {{
      background: rgba(255,255,255,0.03);
    }}
    .hidden {{ display: none !important; }}
    .badge-leaf {{ background: #065f46; color: #6ee7b7; padding: 2px 6px; border-radius: 4px; font-size: 11px; }}
    .badge-container {{ background: #1e3a8a; color: #93c5fd; padding: 2px 6px; border-radius: 4px; font-size: 11px; }}
    .arrow-down {{
      text-align: center;
      color: var(--muted);
      font-size: 16px;
      line-height: 1;
      padding: 2px 0;
    }}
  </style>
</head>
<body>
<div class="container">
  <header>
    <div class="title-group">
      <h1>{title}</h1>
      <p>{summary}</p>
    </div>
    <div class="meta-badges">
      <span class="badge" id="badge-nodes">Nodes: ...</span>
      <span class="badge" id="badge-params">Params: ...</span>
      <span class="badge" id="badge-mode">Mode: ...</span>
    </div>
  </header>

  <div class="tab-bar">
    <button class="tab-btn active" onclick="switchTab('graph')">Flow Graph</button>
    <button class="tab-btn" onclick="switchTab('timeline')">Timeline</button>
    <button class="tab-btn" onclick="switchTab('table')">Tensor Table</button>
    <button class="tab-btn" onclick="switchTab('summary')">Summary</button>
  </div>

  <div class="controls">
    <input type="text" class="search-input" id="search-box" placeholder="Search modules by name or type..." oninput="filterNodes()"/>
    <button class="filter-btn active" id="btn-leaf-only" onclick="toggleLeafOnly()">Leaf Modules Only</button>
  </div>

  <div class="main-layout">
    <!-- Viewport Views -->
    <div class="viewport" id="tab-graph">
      <div id="flow-container" class="node-list"></div>
    </div>

    <div class="viewport hidden" id="tab-timeline">
      <div id="timeline-container" class="node-list"></div>
    </div>

    <div class="viewport hidden" id="tab-table">
      <table class="data-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Name</th>
            <th>Type</th>
            <th>In Shape</th>
            <th>Out Shape</th>
            <th>Dtype</th>
            <th>Params</th>
            <th>Sparsity</th>
          </tr>
        </thead>
        <tbody id="table-body"></tbody>
      </table>
    </div>

    <div class="viewport hidden" id="tab-summary">
      <div id="summary-content" style="padding: 10px; font-size: 14px; line-height: 1.7;"></div>
    </div>

    <!-- Inspector Side Panel -->
    <div class="inspector" id="inspector-panel">
      <h3 id="insp-title">Node Inspector</h3>
      <div class="metric-row"><span class="metric-label">Module Type</span><span class="metric-value" id="insp-type">-</span></div>
      <div class="metric-row"><span class="metric-label">Execution Order</span><span class="metric-value" id="insp-order">-</span></div>
      <div class="metric-row"><span class="metric-label">Input Shape</span><span class="metric-value" id="insp-in">-</span></div>
      <div class="metric-row"><span class="metric-label">Output Shape</span><span class="metric-value" id="insp-out">-</span></div>
      <div class="metric-row"><span class="metric-label">Dtype</span><span class="metric-value" id="insp-dtype">-</span></div>
      <div class="metric-row"><span class="metric-label">Device</span><span class="metric-value" id="insp-device">-</span></div>
      <div class="metric-row"><span class="metric-label">Parameters</span><span class="metric-value" id="insp-params">-</span></div>
      <div class="metric-row"><span class="metric-label">Activation Mean</span><span class="metric-value" id="insp-mean">-</span></div>
      <div class="metric-row"><span class="metric-label">Activation Std</span><span class="metric-value" id="insp-std">-</span></div>
      <div class="metric-row"><span class="metric-label">Min / Max</span><span class="metric-value" id="insp-minmax">-</span></div>
      <div class="metric-row"><span class="metric-label">Sparsity (Zero %)</span><span class="metric-value" id="insp-sparsity">-</span></div>
      <div class="metric-row"><span class="metric-label">Nonfinite (NaN/Inf)</span><span class="metric-value" id="insp-nonfinite">-</span></div>
      <div class="metric-row"><span class="metric-label">Duration (ms)</span><span class="metric-value" id="insp-duration">-</span></div>

      <div style="margin-top: 10px;">
        <span class="metric-label" style="font-size: 12px;">Sampled Value Distribution</span>
        <div class="bar-chart" id="insp-chart"></div>
      </div>
    </div>
  </div>
</div>

<script>
  const payload = {payload_json};
  let leafOnly = true;
  let selectedNodeId = null;

  document.getElementById('badge-nodes').textContent = `Nodes: ${{payload.nodes.length}}`;
  document.getElementById('badge-params').textContent = `Total Params: ${{payload.total_parameters.toLocaleString()}}`;
  document.getElementById('badge-mode').textContent = `Mode: ${{payload.trace_mode}}`;

  function switchTab(tabName) {{
    ['graph', 'timeline', 'table', 'summary'].forEach(t => {{
      document.getElementById('tab-' + t).classList.add('hidden');
    }});
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById('tab-' + tabName).classList.remove('hidden');
    event.target.classList.add('active');
  }}

  function toggleLeafOnly() {{
    leafOnly = !leafOnly;
    const btn = document.getElementById('btn-leaf-only');
    btn.classList.toggle('active', leafOnly);
    renderViews();
  }}

  function filterNodes() {{
    renderViews();
  }}

  function selectNode(nodeId) {{
    selectedNodeId = nodeId;
    document.querySelectorAll('.flow-node').forEach(el => {{
      el.classList.toggle('selected', el.getAttribute('data-id') === nodeId);
    }});
    const node = payload.nodes.find(n => n.id === nodeId);
    if (!node) return;

    document.getElementById('insp-title').textContent = node.name;
    document.getElementById('insp-type').textContent = node.module_type;
    document.getElementById('insp-order').textContent = '#' + node.execution_order;
    document.getElementById('insp-in').textContent = JSON.stringify(node.input_shapes);
    document.getElementById('insp-out').textContent = JSON.stringify(node.output_shapes);
    document.getElementById('insp-dtype').textContent = node.dtypes.join(', ') || '-';
    document.getElementById('insp-device').textContent = node.devices.join(', ') || '-';
    document.getElementById('insp-params').textContent = node.num_parameters.toLocaleString();

    const stats = node.activation_stats || {{}};
    document.getElementById('insp-mean').textContent = stats.mean != null ? stats.mean.toFixed(4) : '-';
    document.getElementById('insp-std').textContent = stats.std != null ? stats.std.toFixed(4) : '-';
    document.getElementById('insp-minmax').textContent = (stats.min != null && stats.max != null) ?
      `${{stats.min.toFixed(2)}} / ${{stats.max.toFixed(2)}}` : '-';
    document.getElementById('insp-sparsity').textContent = (node.sparsity * 100).toFixed(1) + '%';
    document.getElementById('insp-nonfinite').textContent = node.nonfinite_count;
    document.getElementById('insp-duration').textContent = node.duration_ms != null ? node.duration_ms.toFixed(3) : '-';

    // Render bar chart
    const chart = document.getElementById('insp-chart');
    chart.innerHTML = '';
    const samples = node.sampled_values || [];
    if (samples.length > 0) {{
      const minV = Math.min(...samples);
      const maxV = Math.max(...samples);
      const range = (maxV - minV) || 1.0;
      samples.forEach(v => {{
        const bar = document.createElement('div');
        bar.className = 'bar-col';
        const h = Math.max(4, Math.min(50, ((v - minV) / range) * 50));
        bar.style.height = h + 'px';
        bar.title = 'Value: ' + v.toFixed(3);
        chart.appendChild(bar);
      }});
    }}
  }}

  function renderViews() {{
    const query = document.getElementById('search-box').value.toLowerCase().trim();
    const visibleNodes = payload.nodes.filter(n => {{
      if (leafOnly && !n.is_leaf) return false;
      if (query && !n.name.toLowerCase().includes(query) && !n.module_type.toLowerCase().includes(query)) return false;
      return true;
    }});

    // Flow Graph
    const flowContainer = document.getElementById('flow-container');
    flowContainer.innerHTML = '';
    visibleNodes.forEach((n, idx) => {{
      const card = document.createElement('div');
      card.className = 'flow-node' + (n.id === selectedNodeId ? ' selected' : '');
      card.setAttribute('data-id', n.id);
      card.onclick = () => selectNode(n.id);
      card.innerHTML = `
        <div class="node-info">
          <div class="node-name">${{n.name}} <span class="${{n.is_leaf ? 'badge-leaf' : 'badge-container'}}">${{n.is_leaf ? 'Leaf' : 'Module'}}</span></div>
          <div class="node-type">${{n.module_type}} · Order #${{n.execution_order}}</div>
        </div>
        <div class="node-shape">${{n.output_shapes.length ? JSON.stringify(n.output_shapes[0]) : '[]'}}</div>
      `;
      flowContainer.appendChild(card);
      if (idx < visibleNodes.length - 1) {{
        const arrow = document.createElement('div');
        arrow.className = 'arrow-down';
        arrow.textContent = '↓';
        flowContainer.appendChild(arrow);
      }}
    }});

    // Timeline
    const timelineContainer = document.getElementById('timeline-container');
    timelineContainer.innerHTML = '';
    visibleNodes.forEach(n => {{
      const card = document.createElement('div');
      card.className = 'flow-node' + (n.id === selectedNodeId ? ' selected' : '');
      card.setAttribute('data-id', n.id);
      card.onclick = () => selectNode(n.id);
      card.innerHTML = `
        <div class="node-info">
          <div class="node-name">Step #${{n.execution_order}}: ${{n.name}}</div>
          <div class="node-type">${{n.module_type}} · ${{n.duration_ms != null ? n.duration_ms.toFixed(2) + ' ms' : ''}}</div>
        </div>
        <div class="node-shape">${{n.output_shapes.length ? JSON.stringify(n.output_shapes[0]) : ''}}</div>
      `;
      timelineContainer.appendChild(card);
    }});

    // Table
    const tableBody = document.getElementById('table-body');
    tableBody.innerHTML = '';
    visibleNodes.forEach(n => {{
      const row = document.createElement('tr');
      row.style.cursor = 'pointer';
      row.onclick = () => {{ selectNode(n.id); switchTab('graph'); }};
      row.innerHTML = `
        <td>${{n.execution_order}}</td>
        <td><strong>${{n.name}}</strong></td>
        <td>${{n.module_type}}</td>
        <td><code>${{JSON.stringify(n.input_shapes)}}</code></td>
        <td><code>${{JSON.stringify(n.output_shapes)}}</code></td>
        <td>${{n.dtypes.join(', ')}}</td>
        <td>${{n.num_parameters.toLocaleString()}}</td>
        <td>${{(n.sparsity * 100).toFixed(1)}}%</td>
      `;
      tableBody.appendChild(row);
    }});

    // Summary
    const summaryEl = document.getElementById('summary-content');
    summaryEl.innerHTML = `
      <p><strong>Total Executed Operations:</strong> ${{payload.nodes.length}}</p>
      <p><strong>Total Model Parameters:</strong> ${{payload.total_parameters.toLocaleString()}} (${{payload.trainable_parameters.toLocaleString()}} trainable)</p>
      <p><strong>Primary Device:</strong> ${{payload.device}}</p>
      <p><strong>Observed Duration:</strong> ${{payload.duration_ms.toFixed(2)}} ms</p>
      <p><strong>Input Shapes:</strong> ${{JSON.stringify(payload.input_metadata.map(m => m.shape))}}</p>
      <p><strong>Output Shapes:</strong> ${{JSON.stringify(payload.output_metadata.map(m => m.shape))}}</p>
      <br/>
      <p><strong>Scientific Observations & Invariants:</strong></p>
      <ul style="padding-left: 20px; color: var(--muted);">
        ${{payload.limitations.map(l => `<li>${{l}}</li>`).join('')}}
      </ul>
    `;

    if (!selectedNodeId && visibleNodes.length > 0) {{
      selectNode(visibleNodes[0].id);
    }}
  }}

  // Initialize
  renderViews();
  // Expose global bridge for automated testing
  window.__seedl_trace_select_node = selectNode;
  window.__seedl_trace_payload = payload;
</script>
</body>
</html>
"""


def svg_trace(view: Any) -> str:
    """Render a clean static SVG representation of the neural execution trace."""
    data = view.data
    nodes = [n for n in data.get("nodes", []) if n.get("is_leaf", True)]
    if not nodes:
        nodes = data.get("nodes", [])

    width = 800
    row_height = 55
    height = max(400, 140 + len(nodes) * row_height)

    title = html.escape(view.title)
    summary = html.escape(view.summary)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        f'<rect width="{width}" height="{height}" fill="#0f172a"/>',
        '<text x="40" y="38" font-family="system-ui, sans-serif" font-size="12" fill="#94a3b8" font-weight="600">SeeDL / NEURAL EXECUTION TRACER</text>',
        f'<text x="40" y="68" font-family="system-ui, sans-serif" font-size="20" fill="#f8fafc" font-weight="700">{title}</text>',
        f'<text x="40" y="92" font-family="system-ui, sans-serif" font-size="13" fill="#94a3b8">{summary}</text>',
    ]

    # Draw nodes
    y = 130
    for idx, node in enumerate(nodes[:24]):
        name = html.escape(node.get("name", f"Node {idx}"))
        mtype = html.escape(node.get("module_type", "Module"))
        out_shapes = node.get("output_shapes", [])
        shape_str = str(out_shapes[0]) if out_shapes else "[]"

        lines.extend(
            [
                f'<g transform="translate(60, {y})">',
                '<rect width="680" height="42" rx="6" fill="#1e293b" stroke="#334155" stroke-width="1"/>',
                f'<text x="20" y="26" font-family="system-ui, sans-serif" font-size="14" font-weight="600" fill="#38bdf8">{idx + 1}. {name}</text>',
                f'<text x="320" y="26" font-family="system-ui, sans-serif" font-size="12" fill="#94a3b8">{mtype}</text>',
                f'<text x="660" y="26" font-family="monospace" font-size="12" fill="#34d399" text-anchor="end">{shape_str}</text>',
                "</g>",
            ]
        )
        if idx < len(nodes[:24]) - 1:
            lines.append(
                f'<line x1="400" y1="{y + 42}" x2="400" y2="{y + row_height}" stroke="#475569" stroke-width="2" stroke-dasharray="3,3"/>'
            )
        y += row_height

    if len(nodes) > 24:
        lines.append(
            f'<text x="400" y="{y + 20}" font-family="system-ui, sans-serif" font-size="13" fill="#94a3b8" text-anchor="middle">... and {len(nodes) - 24} more operations</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


def matplotlib_trace(view: Any) -> Any:
    """Render execution trace using Matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingDependencyError("Matplotlib rendering requires matplotlib.") from exc

    data = view.data
    nodes = [n for n in data.get("nodes", []) if n.get("is_leaf", True)]
    if not nodes:
        nodes = data.get("nodes", [])

    names = [n.get("name", "") for n in nodes[:20]]
    params = [n.get("num_parameters", 0) for n in nodes[:20]]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.35)))
    y_pos = range(len(names))
    ax.barh(y_pos, params, color="#0284c7", edgecolor="#0369a1", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Parameter Count")
    ax.set_title(f"{view.title}\n{view.summary}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plotly_trace(view: Any) -> Any:
    """Render execution trace using Plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise MissingDependencyError("Plotly rendering requires plotly.") from exc

    data = view.data
    nodes = [n for n in data.get("nodes", []) if n.get("is_leaf", True)]
    if not nodes:
        nodes = data.get("nodes", [])

    names = [
        f"#{n.get('execution_order', i)} {n.get('name', '')} ({n.get('module_type', '')})"
        for i, n in enumerate(nodes)
    ]
    params = [n.get("num_parameters", 0) for n in nodes]
    shapes = [str(n.get("output_shapes", [""])) for n in nodes]

    fig = go.Figure(
        data=[
            go.Bar(
                x=params,
                y=names,
                orientation="h",
                marker=dict(color="#0284c7"),
                text=shapes,
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title=f"<b>{view.title}</b><br><sup>{view.summary}</sup>",
        xaxis_title="Parameter Count",
        yaxis_title="Observed Operation",
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
        height=max(450, len(nodes) * 28),
    )
    return fig
