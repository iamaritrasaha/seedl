"""Interactive offline HTML dashboard for educational ML and DL concepts in SeeDL.

100% offline, self-contained, responsive from 320px to 1920px, accessible mathematical
markup, embedded Google Sans and JetBrains Mono typography, zero external dependencies.
"""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING, Any

from ._typography import FONT, MONO, embed_fonts

if TYPE_CHECKING:
    from .view import Visualization

# Theme Palette (Consistent with SeeDL Design System)
BLUE = "#2563eb"
INDIGO = "#1e40af"
SLATE = "#172b4d"
MUTED = "#52657d"
BORDER = "#dce4ed"
BG = "#f8fafc"
CARD_BG = "#ffffff"
CORAL = "#ef4444"
EMERALD = "#059669"
AMBER = "#d97706"
PURPLE = "#7c3aed"


def _escape(val: Any) -> str:
    return html.escape(str(val), quote=True)


def educational_html(view: Visualization) -> str:
    """Generate a self-contained, interactive offline educational learning dashboard."""
    concept = str(view.data.get("concept", view.kind.replace("educational_", "")))
    data_json = html.escape(json.dumps(view.data, ensure_ascii=False, allow_nan=False))

    lesson_content = _build_lesson_content(concept, view.data, view.title, view.summary, view.notes)

    markup = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape(view.title)} — SeeDL Learn</title>
  <style>
    :root {{
      --font-sans: {FONT};
      --font-mono: {MONO};
      --color-bg: {BG};
      --color-card: {CARD_BG};
      --color-text: {SLATE};
      --color-muted: {MUTED};
      --color-border: {BORDER};
      --color-primary: {BLUE};
      --color-primary-dark: {INDIGO};
      --color-emerald: {EMERALD};
      --color-amber: {AMBER};
      --color-coral: {CORAL};
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--color-bg);
      color: var(--color-text);
      font-family: var(--font-sans);
      font-size: 14px;
      line-height: 1.6;
      padding: clamp(12px, 3vw, 32px);
    }}
    .seedl-container {{
      max-width: 1200px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }}
    header.lesson-header {{
      background: var(--color-card);
      border: 1px solid var(--color-border);
      border-radius: 12px;
      padding: clamp(16px, 3vw, 28px);
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
    }}
    .badge-bar {{
      display: flex;
      gap: 8px;
      margin-bottom: 8px;
      flex-wrap: wrap;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }}
    .pill-blue {{ background: #eff6ff; color: var(--color-primary-dark); border: 1px solid #bfdbfe; }}
    .pill-emerald {{ background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }}
    .pill-amber {{ background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }}
    h1.lesson-title {{
      font-size: clamp(22px, 3.5vw, 30px);
      font-weight: 700;
      color: var(--color-text);
      line-height: 1.25;
      margin-bottom: 8px;
    }}
    p.lesson-summary {{
      font-size: 15px;
      color: var(--color-muted);
      max-width: 900px;
    }}

    /* Main Grid Layout */
    .lesson-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 24px;
    }}
    @media (min-width: 1024px) {{
      .lesson-grid {{
        grid-template-columns: 1.15fr 0.85fr;
      }}
    }}

    /* Section Cards */
    .card {{
      background: var(--color-card);
      border: 1px solid var(--color-border);
      border-radius: 12px;
      padding: clamp(16px, 2.5vw, 24px);
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
    }}
    .card-title {{
      font-size: 16px;
      font-weight: 700;
      color: var(--color-text);
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    /* Interactive Lab Panel */
    .lab-panel {{
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .visual-wrapper {{
      position: relative;
      background: #fafbfc;
      border: 1px solid var(--color-border);
      border-radius: 8px;
      overflow: hidden;
      min-height: 380px;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .visual-wrapper svg {{
      width: 100%;
      height: 100%;
      max-height: 480px;
      display: block;
    }}

    /* Controls */
    .controls-panel {{
      display: flex;
      flex-direction: column;
      gap: 14px;
      background: #f1f5f9;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 16px;
    }}
    .control-row {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .control-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 13px;
      font-weight: 600;
    }}
    .control-value {{
      font-family: var(--font-mono);
      color: var(--color-primary-dark);
      background: white;
      padding: 2px 8px;
      border-radius: 4px;
      border: 1px solid var(--color-border);
      font-size: 12px;
    }}
    input[type=range] {{
      -webkit-appearance: none;
      width: 100%;
      height: 6px;
      background: #cbd5e1;
      border-radius: 4px;
      outline: none;
      cursor: pointer;
    }}
    input[type=range]::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 20px;
      height: 20px;
      border-radius: 50%;
      background: var(--color-primary);
      cursor: pointer;
      box-shadow: 0 1px 3px rgba(0,0,0,0.3);
      transition: transform 0.1s ease;
    }}
    input[type=range]::-webkit-slider-thumb:hover {{
      transform: scale(1.15);
    }}
    input[type=range]:focus-visible {{
      outline: 2px solid var(--color-primary);
    }}

    /* Buttons */
    .btn-group {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    button.seedl-btn {{
      background: white;
      color: var(--color-text);
      border: 1px solid var(--color-border);
      padding: 8px 14px;
      border-radius: 6px;
      font-family: var(--font-sans);
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    button.seedl-btn:hover {{
      background: #f8fafc;
      border-color: #94a3b8;
    }}
    button.seedl-btn.active {{
      background: var(--color-primary);
      color: white;
      border-color: var(--color-primary-dark);
    }}
    button.seedl-btn:focus-visible {{
      outline: 2px solid var(--color-primary);
    }}

    /* Metrics Grid */
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
      gap: 12px;
    }}
    .metric-card {{
      background: white;
      border: 1px solid var(--color-border);
      border-radius: 8px;
      padding: 10px 14px;
      display: flex;
      flex-direction: column;
      gap: 2px;
    }}
    .metric-label {{
      font-size: 11px;
      font-weight: 600;
      color: var(--color-muted);
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}
    .metric-val {{
      font-family: var(--font-mono);
      font-size: 18px;
      font-weight: 700;
      color: var(--color-text);
    }}

    /* Mathematical Notation */
    .math-block {{
      background: #f8fafc;
      border: 1px solid var(--color-border);
      border-radius: 8px;
      padding: 14px 18px;
      font-size: 14px;
      line-height: 1.8;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .math-eq {{
      font-size: 15px;
      letter-spacing: 0.02em;
    }}
    .math-symbol {{
      font-style: italic;
      font-family: var(--font-sans);
      font-weight: 600;
    }}
    .math-live {{
      font-family: var(--font-mono);
      font-weight: 700;
      color: var(--color-primary-dark);
      background: #eff6ff;
      padding: 1px 6px;
      border-radius: 4px;
      border: 1px solid #bfdbfe;
    }}

    /* Code Block */
    pre.code-box {{
      background: #0f172a;
      color: #f8fafc;
      font-family: var(--font-mono);
      font-size: 12px;
      line-height: 1.6;
      padding: 16px;
      border-radius: 8px;
      overflow-x: auto;
      white-space: pre;
    }}

    /* Explanations & Notes */
    .note-list {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding-left: 20px;
    }}
    .note-list li {{
      color: var(--color-text);
    }}

    .limitations-box {{
      background: #fffbeb;
      border: 1px solid #fde68a;
      border-left: 4px solid var(--color-amber);
      border-radius: 6px;
      padding: 12px 16px;
      color: #92400e;
      font-size: 13px;
    }}

    footer.lesson-footer {{
      border-top: 1px solid var(--color-border);
      padding-top: 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      color: var(--color-muted);
      font-size: 12px;
      flex-wrap: wrap;
      gap: 8px;
    }}
  </style>
</head>
<body>
  <div class="seedl-container">
    <header class="lesson-header">
      <div class="badge-bar">
        <span class="pill pill-blue">SeeDL Learn</span>
        <span class="pill pill-emerald">100% Offline</span>
        <span class="pill pill-amber">{_escape(concept.replace("_", " ").title())}</span>
      </div>
      <h1 class="lesson-title">{_escape(view.title)}</h1>
      <p class="lesson-summary">{_escape(view.summary)}</p>
    </header>

    {lesson_content}

    <footer class="lesson-footer">
      <span>SeeDL v0.7.0 Educational System · Pure offline mathematical simulation</span>
      <span>No external network requests or CDNs · Google Sans & JetBrains Mono typography</span>
    </footer>
  </div>
  <script id="seedl-lesson-data" type="application/json">{data_json}</script>
</body>
</html>"""

    return embed_fonts(markup)


def _build_lesson_content(
    concept: str, data: dict[str, Any], title: str, summary: str, notes: list[str]
) -> str:
    """Build the 7-section structured lesson content for the given concept."""
    if concept == "linear_regression":
        return _render_linear_regression(data)
    elif concept == "polynomial_regression":
        return _render_polynomial_regression(data)
    elif concept == "gradient_descent":
        return _render_gradient_descent(data)
    elif concept == "logistic_regression":
        return _render_logistic_regression(data)
    elif concept == "knn":
        return _render_knn(data)
    elif concept == "decision_tree":
        return _render_decision_tree(data)
    elif concept == "kmeans":
        return _render_kmeans(data)
    elif concept == "pca":
        return _render_pca(data)
    else:
        return _render_generic_educational(concept, data, notes)


# ---------------------------------------------------------------------------
# 1. Linear Regression Renderer
# ---------------------------------------------------------------------------


def _render_linear_regression(data: dict[str, Any]) -> str:
    slope = data["slope"]
    intercept = data["intercept"]
    ols = data["ols"]
    mse = data["mse"]
    rmse = data["rmse"]
    mae = data["mae"]
    r2 = data["r2"]

    py_code = """import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Prepare deterministic observations
X = np.array([-2.0, -1.7, -1.4, -1.1, -0.8, -0.5, -0.2, 0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 3.1, 3.4, 3.7]).reshape(-1, 1)
y = 2.5 * X.ravel() + 1.2 + np.array([0.42, -0.65, 0.88, -0.32, 0.15, -0.74, 0.55, -0.18, 0.62, -0.45, 0.28, -0.81, 0.35, -0.22, 0.71, -0.58, 0.44, -0.39, 0.12, -0.04])

# 2. Fit Ordinary Least Squares (OLS) model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

print(f"Optimal line: y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}")
print(f"MSE: {mean_squared_error(y, y_pred):.4f}, R2: {r2_score(y, y_pred):.4f}")
"""

    return f"""
    <div class="lesson-grid">
      <!-- Left Column: Visual Lab & Controls -->
      <div class="lab-panel">
        <div class="card">
          <div class="card-title">Interactive Visual Lab</div>
          <div class="visual-wrapper" id="vis-wrap">
            <svg id="lr-svg" viewBox="0 0 600 400" preserveAspectRatio="xMidYMid meet">
              <defs>
                <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                  <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#f1f5f9" stroke-width="1"/>
                </pattern>
              </defs>
              <rect width="600" height="400" fill="url(#grid)"/>
              <!-- Axes -->
              <line x1="60" y1="340" x2="560" y2="340" stroke="#cbd5e1" stroke-width="2"/>
              <line x1="60" y1="40" x2="60" y2="340" stroke="#cbd5e1" stroke-width="2"/>
              <text x="560" y="360" font-size="12" fill="#64748b" text-anchor="end">Feature x</text>
              <text x="45" y="45" font-size="12" fill="#64748b" text-anchor="middle">y</text>
              <!-- Residual Lines Group -->
              <g id="residuals-group"></g>
              <!-- Fitted Line -->
              <line id="model-line" stroke="{BLUE}" stroke-width="3"/>
              <!-- OLS Reference Line (dashed) -->
              <line id="ols-line" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4 4"/>
              <!-- Data Points Group -->
              <g id="points-group"></g>
            </svg>
          </div>

          <!-- Controls -->
          <div class="controls-panel" style="margin-top: 16px;">
            <div class="control-row">
              <div class="control-header">
                <span>Slope (w)</span>
                <span class="control-value" id="val-slope">{slope:.2f}</span>
              </div>
              <input type="range" id="input-slope" min="-2.0" max="5.0" step="0.05" value="{slope}">
            </div>
            <div class="control-row">
              <div class="control-header">
                <span>Intercept (b)</span>
                <span class="control-value" id="val-intercept">{intercept:.2f}</span>
              </div>
              <input type="range" id="input-intercept" min="-5.0" max="10.0" step="0.1" value="{intercept}">
            </div>
            <div class="btn-group" style="margin-top: 4px;">
              <button class="seedl-btn" id="btn-snap-ols">⚡ Snap to OLS Fit (w*={ols["slope"]:.2f}, b*={ols["intercept"]:.2f})</button>
              <button class="seedl-btn" id="btn-reset-lr">Reset Initial</button>
            </div>
          </div>
        </div>

        <!-- Metric Cards -->
        <div class="metrics-grid">
          <div class="metric-card">
            <span class="metric-label">MSE (Loss)</span>
            <span class="metric-val" id="metric-mse">{mse:.4f}</span>
          </div>
          <div class="metric-card">
            <span class="metric-label">RMSE</span>
            <span class="metric-val" id="metric-rmse">{rmse:.4f}</span>
          </div>
          <div class="metric-card">
            <span class="metric-label">MAE</span>
            <span class="metric-val" id="metric-mae">{mae:.4f}</span>
          </div>
          <div class="metric-card">
            <span class="metric-label">R² Score</span>
            <span class="metric-val" id="metric-r2">{r2:.3f}</span>
          </div>
        </div>
      </div>

      <!-- Right Column: Mathematical Equations, Intuition & Code -->
      <div style="display: flex; flex-direction: column; gap: 20px;">
        <div class="card">
          <div class="card-title">1. Concept & Intuition</div>
          <p style="color: var(--color-muted); margin-bottom: 12px;">
            Linear regression finds the linear relationship that best predicts dependent target <em>y</em> from independent feature <em>x</em>.
            The model computes predictions <span class="math-symbol">y&#770; = w&#183;x + b</span> and minimizes the sum of squared vertical residual distances.
          </p>
          <div class="card-title" style="font-size: 14px; margin-top: 14px;">2. Real-World Relevance</div>
          <p style="color: var(--color-muted);">
            Widely used across financial econometrics (CAPM beta calculation), real estate appraisal, trend line extrapolation, and sensor calibration.
          </p>
        </div>

        <div class="card">
          <div class="card-title">3. Live Mathematical Equations</div>
          <div class="math-block">
            <div class="math-eq">
              <strong>Model:</strong> y&#770; = <span class="math-symbol">w</span>&#183;x + <span class="math-symbol">b</span>
              &nbsp;&#8594;&nbsp; y&#770; = <span class="math-live" id="eq-w">{slope:.2f}</span>&#183;x + <span class="math-live" id="eq-b">{intercept:.2f}</span>
            </div>
            <div class="math-eq">
              <strong>Residual Error:</strong> e<sub>i</sub> = y<sub>i</sub> &#8722; y&#770;<sub>i</sub>
            </div>
            <div class="math-eq">
              <strong>Loss (MSE):</strong> L(w, b) = <sup>1</sup>/<sub>N</sub> &#8721; e<sub>i</sub><sup>2</sup>
              = <span class="math-live" id="eq-mse">{mse:.4f}</span>
            </div>
            <div class="math-eq">
              <strong>Analytical OLS Minimum:</strong>
              w* = <sup>Cov(x, y)</sup>/<sub>Var(x)</sub> = <strong>{ols["slope"]:.3f}</strong>,
              b* = y&#773; &#8722; w*&#183;x&#773; = <strong>{ols["intercept"]:.3f}</strong>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">4. Walkthrough & Limitations</div>
          <ul class="note-list" style="margin-bottom: 12px;">
            <li>Adjusting <strong>slope w</strong> tilts the line around the intercept. Notice how distant residuals grow quadratically.</li>
            <li>Adjusting <strong>intercept b</strong> translates the line vertically without changing its slope.</li>
            <li>Clicking <strong>Snap to OLS Fit</strong> reaches the mathematical global minimum where &#8706;L/&#8706;w = 0 and &#8706;L/&#8706;b = 0.</li>
          </ul>
          <div class="limitations-box">
            <strong>Scientific Limitations:</strong> Highly sensitive to leverage points and outliers because squared residuals penalize large errors quadratically. Assumes homoscedasticity (constant noise variance) and strictly linear true relationships.
          </div>
        </div>

        <div class="card">
          <div class="card-title">5. Reproducible scikit-learn Code</div>
          <pre class="code-box"><code>{html.escape(py_code)}</code></pre>
        </div>
      </div>
    </div>

    <script>
    (function() {{
      const data = JSON.parse(document.getElementById('seedl-lesson-data').textContent);
      const xs = data.x;
      const ys = data.y;
      const n = xs.length;
      const ols = data.ols;

      // Coordinate scales
      const xMin = -2.5, xMax = 4.0;
      const yMin = -6.0, yMax = 12.0;
      const svgW = 600, svgH = 400;
      const padL = 60, padR = 40, padT = 40, padB = 60;

      function toSvgX(x) {{
        return padL + ((x - xMin) / (xMax - xMin)) * (svgW - padL - padR);
      }}
      function toSvgY(y) {{
        return (svgH - padB) - ((y - yMin) / (yMax - yMin)) * (svgH - padT - padB);
      }}

      const pointsGroup = document.getElementById('points-group');
      const resGroup = document.getElementById('residuals-group');
      const modelLine = document.getElementById('model-line');
      const olsLine = document.getElementById('ols-line');

      // Draw OLS reference line
      olsLine.setAttribute('x1', toSvgX(xMin));
      olsLine.setAttribute('y1', toSvgY(ols.slope * xMin + ols.intercept));
      olsLine.setAttribute('x2', toSvgX(xMax));
      olsLine.setAttribute('y2', toSvgY(ols.slope * xMax + ols.intercept));

      // Draw Data Points
      xs.forEach((x, i) => {{
        const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        c.setAttribute('cx', toSvgX(x));
        c.setAttribute('cy', toSvgY(ys[i]));
        c.setAttribute('r', '5');
        c.setAttribute('fill', '{INDIGO}');
        c.setAttribute('stroke', '#ffffff');
        c.setAttribute('stroke-width', '1.5');
        const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        title.textContent = `(${{x.toFixed(2)}}, ${{ys[i].toFixed(2)}})`;
        c.appendChild(title);
        pointsGroup.appendChild(c);
      }});

      // UI Sliders
      const inputSlope = document.getElementById('input-slope');
      const inputIntercept = document.getElementById('input-intercept');
      const valSlope = document.getElementById('val-slope');
      const valIntercept = document.getElementById('val-intercept');
      const eqW = document.getElementById('eq-w');
      const eqB = document.getElementById('eq-b');
      const eqMse = document.getElementById('eq-mse');

      const metricMse = document.getElementById('metric-mse');
      const metricRmse = document.getElementById('metric-rmse');
      const metricMae = document.getElementById('metric-mae');
      const metricR2 = document.getElementById('metric-r2');

      const yMean = ys.reduce((a, b) => a + b, 0) / n;
      const ssTot = ys.reduce((a, y) => a + Math.pow(y - yMean, 2), 0);

      function update(w, b) {{
        valSlope.textContent = w.toFixed(2);
        valIntercept.textContent = b.toFixed(2);
        eqW.textContent = w.toFixed(2);
        eqB.textContent = b.toFixed(2);

        // Update model line
        modelLine.setAttribute('x1', toSvgX(xMin));
        modelLine.setAttribute('y1', toSvgY(w * xMin + b));
        modelLine.setAttribute('x2', toSvgX(xMax));
        modelLine.setAttribute('y2', toSvgY(w * xMax + b));

        // Update residuals
        resGroup.innerHTML = '';
        let ssRes = 0;
        let sae = 0;
        xs.forEach((x, i) => {{
          const yHat = w * x + b;
          const e = ys[i] - yHat;
          ssRes += e * e;
          sae += Math.abs(e);

          const rLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
          rLine.setAttribute('x1', toSvgX(x));
          rLine.setAttribute('y1', toSvgY(ys[i]));
          rLine.setAttribute('x2', toSvgX(x));
          rLine.setAttribute('y2', toSvgY(yHat));
          rLine.setAttribute('stroke', '{CORAL}');
          rLine.setAttribute('stroke-width', '1.5');
          rLine.setAttribute('stroke-dasharray', '2 2');
          resGroup.appendChild(rLine);
        }});

        const mseVal = ssRes / n;
        const rmseVal = Math.sqrt(mseVal);
        const maeVal = sae / n;
        const r2Val = ssTot > 0 ? 1.0 - (ssRes / ssTot) : 0.0;

        metricMse.textContent = mseVal.toFixed(4);
        metricRmse.textContent = rmseVal.toFixed(4);
        metricMae.textContent = maeVal.toFixed(4);
        metricR2.textContent = r2Val.toFixed(3);
        eqMse.textContent = mseVal.toFixed(4);
      }}

      inputSlope.addEventListener('input', () => update(parseFloat(inputSlope.value), parseFloat(inputIntercept.value)));
      inputIntercept.addEventListener('input', () => update(parseFloat(inputSlope.value), parseFloat(inputIntercept.value)));

      document.getElementById('btn-snap-ols').addEventListener('click', () => {{
        inputSlope.value = ols.slope;
        inputIntercept.value = ols.intercept;
        update(ols.slope, ols.intercept);
      }});

      document.getElementById('btn-reset-lr').addEventListener('click', () => {{
        inputSlope.value = {slope};
        inputIntercept.value = {intercept};
        update({slope}, {intercept});
      }});

      update(parseFloat(inputSlope.value), parseFloat(inputIntercept.value));
    }})();
    </script>
    """


# ---------------------------------------------------------------------------
# 2. Polynomial Regression Renderer
# ---------------------------------------------------------------------------


def _render_polynomial_regression(data: dict[str, Any]) -> str:
    degree = data["degree"]
    reg = data["regularization"]
    regime = data["regime"]
    mse_train = data["mse_train"]
    mse_val = data["mse_val"]

    py_code = """import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# 1. Non-linear sinusoidal signal with noise
X_train = np.linspace(-0.95, 0.98, 14).reshape(-1, 1)
y_train = np.sin(np.pi * X_train.ravel()) + np.random.normal(0, 0.1, len(X_train))

X_val = np.linspace(-0.88, 0.95, 14).reshape(-1, 1)
y_val = np.sin(np.pi * X_val.ravel()) + np.random.normal(0, 0.1, len(X_val))

# 2. Pipeline: Polynomial expansion + L2 Ridge Regularization
degree = 3
model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=0.0))
model.fit(X_train, y_train)

print(f"Degree {degree} Train MSE: {mean_squared_error(y_train, model.predict(X_train)):.4f}")
print(f"Degree {degree} Val MSE:   {mean_squared_error(y_val, model.predict(X_val)):.4f}")
"""

    return f"""
    <div class="lesson-grid">
      <div class="lab-panel">
        <div class="card">
          <div class="card-title">Interactive Polynomial Curve & Overfitting Lab</div>
          <div class="visual-wrapper">
            <svg id="poly-svg" viewBox="0 0 600 400" preserveAspectRatio="xMidYMid meet">
              <!-- Grid and Axes -->
              <line x1="50" y1="200" x2="550" y2="200" stroke="#e2e8f0" stroke-width="1.5" stroke-dasharray="2 2"/>
              <line x1="300" y1="40" x2="300" y2="360" stroke="#e2e8f0" stroke-width="1.5" stroke-dasharray="2 2"/>
              <!-- Ground truth curve (dashed) -->
              <path id="ground-truth-path" fill="none" stroke="#94a3b8" stroke-width="2" stroke-dasharray="4 4"/>
              <!-- Fitted polynomial curve -->
              <path id="poly-path" fill="none" stroke="{EMERALD}" stroke-width="3"/>
              <!-- Points -->
              <g id="train-points"></g>
              <g id="val-points"></g>
            </svg>
          </div>

          <div class="controls-panel" style="margin-top: 16px;">
            <div class="control-row">
              <div class="control-header">
                <span>Polynomial Degree (d)</span>
                <span class="control-value" id="val-degree">{degree}</span>
              </div>
              <input type="range" id="input-degree" min="1" max="10" step="1" value="{degree}">
            </div>
            <div class="control-row">
              <div class="control-header">
                <span>L2 Regularization (&alpha;)</span>
                <span class="control-value" id="val-alpha">{reg:.2f}</span>
              </div>
              <input type="range" id="input-alpha" min="0.0" max="0.1" step="0.005" value="{reg}">
            </div>
            <div class="pill pill-amber" id="regime-pill" style="align-self: flex-start; margin-top: 4px;">{regime}</div>
          </div>
        </div>

        <div class="metrics-grid">
          <div class="metric-card">
            <span class="metric-label">Degree (d)</span>
            <span class="metric-val" id="metric-deg">{degree}</span>
          </div>
          <div class="metric-card">
            <span class="metric-label">Train MSE</span>
            <span class="metric-val" id="metric-tr-mse">{mse_train:.4f}</span>
          </div>
          <div class="metric-card">
            <span class="metric-label">Validation MSE</span>
            <span class="metric-val" id="metric-vl-mse">{mse_val:.4f}</span>
          </div>
        </div>
      </div>

      <div style="display: flex; flex-direction: column; gap: 20px;">
        <div class="card">
          <div class="card-title">1. Intuition: Bias-Variance Trade-off</div>
          <p style="color: var(--color-muted); margin-bottom: 10px;">
            As degree <em>d</em> increases, model capacity expands. With <em>d &#8804; 2</em>, the model suffers from <strong>high bias (underfitting)</strong>.
            At <em>d &#8776; 3&#8211;4</em>, it balances capacity and generalization. At <em>d &#8805; 5</em>, it exhibits <strong>high variance (overfitting)</strong>, memorizing noise while validation error explodes.
          </p>
        </div>

        <div class="card">
          <div class="card-title">2. Live Mathematical Equations</div>
          <div class="math-block">
            <div class="math-eq">
              <strong>Polynomial Model:</strong> y&#770; = &#8721;<sub>j=0</sub><sup>d</sup> w<sub>j</sub> &#183; x<sup>j</sup>
            </div>
            <div class="math-eq">
              <strong>Objective:</strong> min<sub>w</sub> [ <sup>1</sup>/<sub>N</sub> &#8721; (y<sub>i</sub> &#8722; y&#770;<sub>i</sub>)<sup>2</sup> + &alpha; &#8721;<sub>j=1</sub><sup>d</sup> w<sub>j</sub><sup>2</sup> ]
            </div>
            <div class="math-eq">
              <strong>Current State:</strong> d = <span class="math-live" id="live-deg">{degree}</span>,
              Train MSE = <span class="math-live" id="live-trmse">{mse_train:.4f}</span>,
              Val MSE = <span class="math-live" id="live-vlmse">{mse_val:.4f}</span>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">3. Explanatory Walkthrough & Limitations</div>
          <ul class="note-list" style="margin-bottom: 12px;">
            <li>Drag <strong>Degree</strong> to 1: observe that a straight line cannot capture the sine peak and valley.</li>
            <li>Drag <strong>Degree</strong> to 9 or 10: watch the green curve oscillate wildly near the endpoints (Runge's phenomenon).</li>
            <li>Increase <strong>Regularization &alpha;</strong>: watch how L2 penalty shrinks the polynomial weights, taming the wild oscillations!</li>
          </ul>
          <div class="limitations-box">
            <strong>Scientific Limitations:</strong> Polynomial features extrapolate catastrophically outside the training interval and create ill-conditioned Vandermonde design matrices. Splines or radial basis functions are preferred in production.
          </div>
        </div>

        <div class="card">
          <div class="card-title">4. Reproducible scikit-learn Code</div>
          <pre class="code-box"><code>{html.escape(py_code)}</code></pre>
        </div>
      </div>
    </div>

    <script>
    (function() {{
      const data = JSON.parse(document.getElementById('seedl-lesson-data').textContent);
      const xTr = data.x_train;
      const yTr = data.y_train;
      const xVl = data.x_val;
      const yVl = data.y_val;
      const curveX = data.curve_x;
      const gtY = data.ground_truth_y;

      const svgW = 600, svgH = 400;
      const padL = 50, padR = 50, padT = 40, padB = 40;

      function toSvgX(x) {{ return padL + ((x - (-1.1)) / 2.2) * (svgW - padL - padR); }}
      function toSvgY(y) {{ return (svgH - padB) - ((y - (-1.8)) / 3.6) * (svgH - padT - padB); }}

      // Draw Ground Truth Path
      const gtPath = document.getElementById('ground-truth-path');
      let gtD = `M ${{toSvgX(curveX[0])}} ${{toSvgY(gtY[0])}}`;
      for (let i = 1; i < curveX.length; i++) {{
        gtD += ` L ${{toSvgX(curveX[i])}} ${{toSvgY(gtY[i])}}`;
      }}
      gtPath.setAttribute('d', gtD);

      // Draw Training Points
      const trGroup = document.getElementById('train-points');
      xTr.forEach((x, i) => {{
        const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        c.setAttribute('cx', toSvgX(x));
        c.setAttribute('cy', toSvgY(yTr[i]));
        c.setAttribute('r', '5');
        c.setAttribute('fill', '{INDIGO}');
        const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        title.textContent = `Train point (${{x.toFixed(2)}}, ${{yTr[i].toFixed(2)}})`;
        c.appendChild(title);
        trGroup.appendChild(c);
      }});

      // Draw Validation Points
      const vlGroup = document.getElementById('val-points');
      xVl.forEach((x, i) => {{
        const r = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        r.setAttribute('x', toSvgX(x) - 4);
        r.setAttribute('y', toSvgY(yVl[i]) - 4);
        r.setAttribute('width', '8');
        r.setAttribute('height', '8');
        r.setAttribute('fill', '{AMBER}');
        r.setAttribute('stroke', '#ffffff');
        r.setAttribute('stroke-width', '1');
        const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        title.textContent = `Validation point (${{x.toFixed(2)}}, ${{yVl[i].toFixed(2)}})`;
        r.appendChild(title);
        vlGroup.appendChild(r);
      }});

      // Solver for live interactive polynomial fit
      function solve(deg, alpha) {{
        const m = deg + 1;
        const xtx = Array.from({{length: m}}, () => Array(m).fill(0));
        const xty = Array(m).fill(0);
        for (let i = 0; i < xTr.length; i++) {{
          const x = xTr[i], y = yTr[i];
          const p = [1];
          for (let j = 1; j < m; j++) p.push(p[j - 1] * x);
          for (let j = 0; j < m; j++) {{
            xty[j] += p[j] * y;
            for (let k = 0; k < m; k++) xtx[j][k] += p[j] * p[k];
          }}
        }}
        for (let j = 1; j < m; j++) xtx[j][j] += alpha;

        // Gaussian elimination
        for (let i = 0; i < m; i++) {{
          let maxRow = i;
          for (let r = i + 1; r < m; r++) if (Math.abs(xtx[r][i]) > Math.abs(xtx[maxRow][i])) maxRow = r;
          [xtx[i], xtx[maxRow]] = [xtx[maxRow], xtx[i]];
          [xty[i], xty[maxRow]] = [xty[maxRow], xty[i]];
          const diag = Math.abs(xtx[i][i]) < 1e-12 ? 1e-12 : xtx[i][i];
          for (let c = i; c < m; c++) xtx[i][c] /= diag;
          xty[i] /= diag;
          for (let r = 0; r < m; r++) {{
            if (r !== i) {{
              const f = xtx[r][i];
              for (let c = i; c < m; c++) xtx[r][c] -= f * xtx[i][c];
              xty[r] -= f * xty[i];
            }}
          }}
        }}
        return xty;
      }}

      function evalP(w, x) {{
        let res = 0, px = 1;
        for (let i = 0; i < w.length; i++) {{ res += w[i] * px; px *= x; }}
        return res;
      }}

      const polyPath = document.getElementById('poly-path');
      const inputDeg = document.getElementById('input-degree');
      const inputAlpha = document.getElementById('input-alpha');
      const valDeg = document.getElementById('val-degree');
      const valAlpha = document.getElementById('val-alpha');
      const regimePill = document.getElementById('regime-pill');

      const metricDeg = document.getElementById('metric-deg');
      const metricTrMse = document.getElementById('metric-tr-mse');
      const metricVlMse = document.getElementById('metric-vl-mse');
      const liveDeg = document.getElementById('live-deg');
      const liveTrMse = document.getElementById('live-trmse');
      const liveVlMse = document.getElementById('live-vlmse');

      function update() {{
        const d = parseInt(inputDeg.value, 10);
        const a = parseFloat(inputAlpha.value);
        valDeg.textContent = d;
        valAlpha.textContent = a.toFixed(3);
        liveDeg.textContent = d;
        metricDeg.textContent = d;

        const w = solve(d, a);

        // Compute MSEs
        let trErr = 0;
        xTr.forEach((x, i) => trErr += Math.pow(yTr[i] - evalP(w, x), 2));
        trErr /= xTr.length;

        let vlErr = 0;
        xVl.forEach((x, i) => vlErr += Math.pow(yVl[i] - evalP(w, x), 2));
        vlErr /= xVl.length;

        metricTrMse.textContent = trErr.toFixed(4);
        metricVlMse.textContent = vlErr.toFixed(4);
        liveTrMse.textContent = trErr.toFixed(4);
        liveVlMse.textContent = vlErr.toFixed(4);

        // Update regime
        if (d <= 2) {{
          regimePill.textContent = "Underfitting (High Bias)";
          regimePill.className = "pill pill-amber";
        }} else if (d <= 4) {{
          regimePill.textContent = "Balanced Fit (Optimal)";
          regimePill.className = "pill pill-emerald";
        }} else {{
          regimePill.textContent = "Overfitting (High Variance)";
          regimePill.className = "pill pill-coral";
        }}

        // Redraw curve
        let dStr = `M ${{toSvgX(curveX[0])}} ${{toSvgY(Math.max(-2.5, Math.min(2.5, evalP(w, curveX[0]))))}}`;
        for (let i = 1; i < curveX.length; i++) {{
          const yVal = Math.max(-2.5, Math.min(2.5, evalP(w, curveX[i])));
          dStr += ` L ${{toSvgX(curveX[i])}} ${{toSvgY(yVal)}}`;
        }}
        polyPath.setAttribute('d', dStr);
      }}

      inputDeg.addEventListener('input', update);
      inputAlpha.addEventListener('input', update);
      update();
    }})();
    </script>
    """


# ---------------------------------------------------------------------------
# 3. Gradient Descent Renderer (1D & 2D)
# ---------------------------------------------------------------------------


def _render_gradient_descent(data: dict[str, Any]) -> str:
    objective = data.get("objective", "1d_convex")
    lr = data["lr"]

    py_code = """import numpy as np
from sklearn.linear_model import SGDRegressor

# Demonstrate gradient descent optimization
X = np.random.randn(100, 2)
y = X @ np.array([2.0, -1.0]) + 0.5

# SGD with constant learning rate
model = SGDRegressor(loss='squared_error', learning_rate='constant', eta0=0.1, max_iter=50)
model.fit(X, y)

print(f"Learned weights: {model.coef_}, Intercept: {model.intercept_}")
"""

    return f"""
    <div class="lesson-grid">
      <div class="lab-panel">
        <div class="card">
          <div class="card-title">Interactive Optimization Trajectory Lab</div>
          <div class="visual-wrapper">
            <svg id="gd-svg" viewBox="0 0 600 400" preserveAspectRatio="xMidYMid meet">
              <path id="objective-curve" fill="none" stroke="#64748b" stroke-width="2"/>
              <path id="trajectory-line" fill="none" stroke="{BLUE}" stroke-width="2" stroke-dasharray="3 3"/>
              <g id="trajectory-points"></g>
              <circle id="current-point" r="7" fill="{CORAL}" stroke="#ffffff" stroke-width="2"/>
            </svg>
          </div>

          <div class="controls-panel" style="margin-top: 16px;">
            <div class="control-row">
              <div class="control-header">
                <span>Objective Surface</span>
                <span class="control-value" id="val-obj">{objective}</span>
              </div>
              <div class="btn-group">
                <button class="seedl-btn {"active" if objective == "1d_convex" else ""}" id="btn-obj-convex">1D Convex</button>
                <button class="seedl-btn {"active" if objective == "1d_double_well" else ""}" id="btn-obj-double">1D Double-Well</button>
                <button class="seedl-btn {"active" if objective == "2d_valley" else ""}" id="btn-obj-2d">2D Valley</button>
              </div>
            </div>
            <div class="control-row">
              <div class="control-header">
                <span>Learning Rate (&alpha;)</span>
                <span class="control-value" id="val-lr">{lr:.2f}</span>
              </div>
              <input type="range" id="input-lr" min="0.02" max="2.2" step="0.02" value="{lr}">
            </div>
            <div class="btn-group" style="margin-top: 4px;">
              <button class="seedl-btn" id="btn-play">&#9654; Play</button>
              <button class="seedl-btn" id="btn-step">Step &#8594;</button>
              <button class="seedl-btn" id="btn-reset">Reset</button>
            </div>
          </div>
        </div>

        <div class="metrics-grid">
          <div class="metric-card">
            <span class="metric-label">Step</span>
            <span class="metric-val" id="metric-step">0</span>
          </div>
          <div class="metric-card">
            <span class="metric-label">Position w</span>
            <span class="metric-val" id="metric-w">--</span>
          </div>
          <div class="metric-card">
            <span class="metric-label">Loss L(w)</span>
            <span class="metric-val" id="metric-loss">--</span>
          </div>
          <div class="metric-card">
            <span class="metric-label">Gradient &#8711;L</span>
            <span class="metric-val" id="metric-grad">--</span>
          </div>
        </div>
      </div>

      <div style="display: flex; flex-direction: column; gap: 20px;">
        <div class="card">
          <div class="card-title">1. Intuition & Dynamics</div>
          <p style="color: var(--color-muted); margin-bottom: 8px;">
            Gradient descent steps proportionally to the negative gradient: <span class="math-symbol">w<sub>t+1</sub> = w<sub>t</sub> &#8722; &alpha; &#8711;L(w<sub>t</sub>)</span>.
          </p>
          <ul class="note-list">
            <li><strong>&alpha; &le; 1/L:</strong> Monotonic convergence directly towards the minimum.</li>
            <li><strong>1/L &lt; &alpha; &lt; 2/L:</strong> Oscillatory overshoot, bouncing across the walls.</li>
            <li><strong>&alpha; &ge; 2/L:</strong> Divergence &#8212; loss cascades to infinity!</li>
          </ul>
        </div>

        <div class="card">
          <div class="card-title">2. Live Mathematical Equations</div>
          <div class="math-block">
            <div class="math-eq">
              <strong>Update Rule:</strong> w<sub>t+1</sub> = w<sub>t</sub> &#8722; &alpha; &#183; &#8711;L(w<sub>t</sub>)
            </div>
            <div class="math-eq">
              <strong>Live Step:</strong>
              w = <span class="math-live" id="live-w">--</span>,
              &#8711;L = <span class="math-live" id="live-grad">--</span>,
              Loss = <span class="math-live" id="live-loss">--</span>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">3. Scientific Limitations</div>
          <div class="limitations-box">
            Vanilla SGD lacks adaptive momentum (unlike Adam), making it vulnerable to ill-conditioned anisotropic valleys and saddle-point stagnation in non-convex loss surfaces.
          </div>
        </div>

        <div class="card">
          <div class="card-title">4. Reproducible Code</div>
          <pre class="code-box"><code>{html.escape(py_code)}</code></pre>
        </div>
      </div>
    </div>

    <script>
    (function() {{
      const data = JSON.parse(document.getElementById('seedl-lesson-data').textContent);
      let traj = data.trajectory;
      let currentIdx = 0;
      let timer = null;

      const objCurve = document.getElementById('objective-curve');
      const trajLine = document.getElementById('trajectory-line');
      const trajPoints = document.getElementById('trajectory-points');
      const currPoint = document.getElementById('current-point');

      const metricStep = document.getElementById('metric-step');
      const metricW = document.getElementById('metric-w');
      const metricLoss = document.getElementById('metric-loss');
      const metricGrad = document.getElementById('metric-grad');
      const liveW = document.getElementById('live-w');
      const liveGrad = document.getElementById('live-grad');
      const liveLoss = document.getElementById('live-loss');

      const inputLr = document.getElementById('input-lr');
      const valLr = document.getElementById('val-lr');
      const btnPlay = document.getElementById('btn-play');
      const btnStep = document.getElementById('btn-step');
      const btnReset = document.getElementById('btn-reset');

      function toSvgX(w) {{ return 300 + w * 55; }}
      function toSvgY(loss) {{ return 350 - loss * 35; }}

      // Draw 1D Quadratic Loss surface
      let curveD = "M 20 350";
      for (let w = -4.5; w <= 4.5; w += 0.1) {{
        const loss = 0.5 * w * w;
        curveD += ` L ${{toSvgX(w)}} ${{toSvgY(loss)}}`;
      }}
      objCurve.setAttribute('d', curveD);

      function recompute(lr) {{
        traj = [];
        let w = 3.5;
        for (let s = 0; s <= 20; s++) {{
          const loss = 0.5 * w * w;
          const g = w;
          traj.push({{ step: s, w: w, loss: loss, grad: g }});
          w = w - lr * g;
          if (Math.abs(w) > 50) break;
        }}
        renderStep(0);
      }}

      function renderStep(idx) {{
        currentIdx = Math.min(idx, traj.length - 1);
        const p = traj[currentIdx];

        metricStep.textContent = p.step;
        const wStr = typeof p.w === 'number' ? p.w.toFixed(3) : JSON.stringify(p.w);
        metricW.textContent = wStr;
        liveW.textContent = wStr;
        metricLoss.textContent = p.loss.toFixed(4);
        liveLoss.textContent = p.loss.toFixed(4);
        const gStr = typeof p.grad === 'number' ? p.grad.toFixed(3) : JSON.stringify(p.grad);
        metricGrad.textContent = gStr;
        liveGrad.textContent = gStr;

        const cx = toSvgX(typeof p.w === 'number' ? p.w : p.w[0]);
        const cy = toSvgY(p.loss);
        currPoint.setAttribute('cx', cx);
        currPoint.setAttribute('cy', cy);

        // Path up to currentIdx
        let d = `M ${{toSvgX(typeof traj[0].w === 'number' ? traj[0].w : traj[0].w[0])}} ${{toSvgY(traj[0].loss)}}`;
        for (let i = 1; i <= currentIdx; i++) {{
          d += ` L ${{toSvgX(typeof traj[i].w === 'number' ? traj[i].w : traj[i].w[0])}} ${{toSvgY(traj[i].loss)}}`;
        }}
        trajLine.setAttribute('d', d);
      }}

      inputLr.addEventListener('input', () => {{
        const lrVal = parseFloat(inputLr.value);
        valLr.textContent = lrVal.toFixed(2);
        recompute(lrVal);
      }});

      btnStep.addEventListener('click', () => {{
        if (currentIdx < traj.length - 1) renderStep(currentIdx + 1);
      }});

      btnReset.addEventListener('click', () => {{
        if (timer) clearInterval(timer);
        btnPlay.innerHTML = "&#9654; Play";
        renderStep(0);
      }});

      btnPlay.addEventListener('click', () => {{
        if (timer) {{
          clearInterval(timer);
          timer = null;
          btnPlay.innerHTML = "&#9654; Play";
        }} else {{
          btnPlay.innerHTML = "&#10074;&#10074; Pause";
          timer = setInterval(() => {{
            if (currentIdx < traj.length - 1) {{
              renderStep(currentIdx + 1);
            }} else {{
              clearInterval(timer);
              timer = null;
              btnPlay.innerHTML = "&#9654; Play";
            }}
          }}, 250);
        }}
      }});

      renderStep(0);
    }})();
    </script>
    """


# ---------------------------------------------------------------------------
# 4. Logistic Regression Renderer
# ---------------------------------------------------------------------------


def _render_logistic_regression(data: dict[str, Any]) -> str:
    thresh = data["threshold"]
    w1 = data["w1"]
    w2 = data["w2"]
    b = data["b"]
    cm = data["confusion_matrix"]
    acc = data["accuracy"]
    f1 = data["f1"]
    prec = data["precision"]
    rec = data["recall"]
    bce = data["bce_loss"]

    py_code = """from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Binary classification training
X = np.random.randn(30, 2)
y = (X[:, 0] * 1.4 + X[:, 1] * 1.1 - 0.4 > 0).astype(int)

clf = LogisticRegression()
clf.fit(X, y)
probs = clf.predict_proba(X)[:, 1]

print(f"Weights: {clf.coef_}, Intercept: {clf.intercept_}")
print("Confusion Matrix:\\n", confusion_matrix(y, (probs >= 0.5).astype(int)))
"""

    return f"""
    <div class="lesson-grid">
      <div class="lab-panel">
        <div class="card">
          <div class="card-title">Interactive 2D Decision Boundary & Probability Lab</div>
          <div class="visual-wrapper">
            <svg id="logreg-svg" viewBox="0 0 600 400" preserveAspectRatio="xMidYMid meet">
              <!-- Center axes -->
              <line x1="60" y1="200" x2="540" y2="200" stroke="#cbd5e1" stroke-width="1.5"/>
              <line x1="300" y1="40" x2="300" y2="360" stroke="#cbd5e1" stroke-width="1.5"/>
              <!-- Separating line -->
              <line id="boundary-line" stroke="{PURPLE}" stroke-width="3"/>
              <!-- Points -->
              <g id="logreg-points"></g>
            </svg>
          </div>

          <div class="controls-panel" style="margin-top: 16px;">
            <div class="control-row">
              <div class="control-header">
                <span>Decision Threshold (&tau;)</span>
                <span class="control-value" id="val-tau">{thresh:.2f}</span>
              </div>
              <input type="range" id="input-tau" min="0.05" max="0.95" step="0.05" value="{thresh}">
            </div>
            <div class="control-row">
              <div class="control-header">
                <span>Weight w1</span>
                <span class="control-value" id="val-w1">{w1:.2f}</span>
              </div>
              <input type="range" id="input-w1" min="-3.0" max="3.0" step="0.1" value="{w1}">
            </div>
            <div class="control-row">
              <div class="control-header">
                <span>Weight w2</span>
                <span class="control-value" id="val-w2">{w2:.2f}</span>
              </div>
              <input type="range" id="input-w2" min="-3.0" max="3.0" step="0.1" value="{w2}">
            </div>
            <div class="control-row">
              <div class="control-header">
                <span>Bias b</span>
                <span class="control-value" id="val-b">{b:.2f}</span>
              </div>
              <input type="range" id="input-b" min="-3.0" max="3.0" step="0.1" value="{b}">
            </div>
          </div>
        </div>

        <div class="metrics-grid">
          <div class="metric-card">
            <span class="metric-label">Accuracy</span>
            <span class="metric-val" id="metric-acc">{acc:.1%}</span>
          </div>
          <div class="metric-card">
            <span class="metric-label">F1 Score</span>
            <span class="metric-val" id="metric-f1">{f1:.3f}</span>
          </div>
          <div class="metric-card">
            <span class="metric-label">Precision</span>
            <span class="metric-val" id="metric-prec">{prec:.3f}</span>
          </div>
          <div class="metric-card">
            <span class="metric-label">Recall</span>
            <span class="metric-val" id="metric-rec">{rec:.3f}</span>
          </div>
        </div>
      </div>

      <div style="display: flex; flex-direction: column; gap: 20px;">
        <div class="card">
          <div class="card-title">1. Sigmoid Mapping & Decision Boundary</div>
          <p style="color: var(--color-muted); margin-bottom: 8px;">
            Logistic regression models <span class="math-symbol">P(Y=1|x) = &sigma;(w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub> + b)</span> where &sigma;(z) = 1 / (1 + e<sup>&#8722;z</sup>).
            Points are classified as Class 1 when the predicted probability meets or exceeds threshold <span class="math-symbol">&tau;</span>.
          </p>
        </div>

        <div class="card">
          <div class="card-title">2. Confusion Matrix & BCE Loss</div>
          <div class="metrics-grid" style="grid-template-columns: 1fr 1fr; margin-bottom: 12px;">
            <div class="metric-card" style="border-left: 4px solid {EMERALD};">
              <span class="metric-label">True Positives (TP)</span>
              <span class="metric-val" id="cm-tp">{cm["tp"]}</span>
            </div>
            <div class="metric-card" style="border-left: 4px solid {CORAL};">
              <span class="metric-label">False Positives (FP)</span>
              <span class="metric-val" id="cm-fp">{cm["fp"]}</span>
            </div>
            <div class="metric-card" style="border-left: 4px solid {CORAL};">
              <span class="metric-label">False Negatives (FN)</span>
              <span class="metric-val" id="cm-fn">{cm["fn"]}</span>
            </div>
            <div class="metric-card" style="border-left: 4px solid {EMERALD};">
              <span class="metric-label">True Negatives (TN)</span>
              <span class="metric-val" id="cm-tn">{cm["tn"]}</span>
            </div>
          </div>
          <div class="math-block">
            <div class="math-eq">
              <strong>Binary Cross-Entropy:</strong>
              L = &#8722;<sup>1</sup>/<sub>N</sub> &#8721; [ y ln(p) + (1&#8722;y) ln(1&#8722;p) ]
              = <span class="math-live" id="live-bce">{bce:.4f}</span>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">3. Walkthrough & Limitations</div>
          <ul class="note-list" style="margin-bottom: 12px;">
            <li>Raising <strong>Threshold &tau;</strong> requires higher confidence to predict Class 1 (increases Precision, lowers Recall).</li>
            <li>Lowering <strong>Threshold &tau;</strong> catches more true positives, but flags more false alarms.</li>
          </ul>
          <div class="limitations-box">
            <strong>Scientific Limitations:</strong> Logistic regression is inherently a linear classifier. It cannot solve non-linear separation problems (like XOR) without polynomial features or kernel trick.
          </div>
        </div>

        <div class="card">
          <div class="card-title">4. Reproducible Code</div>
          <pre class="code-box"><code>{html.escape(py_code)}</code></pre>
        </div>
      </div>
    </div>

    <script>
    (function() {{
      const data = JSON.parse(document.getElementById('seedl-lesson-data').textContent);
      const samples = data.samples;
      const ptsGroup = document.getElementById('logreg-points');
      const bLine = document.getElementById('boundary-line');

      function toSvgX(x) {{ return 300 + x * 75; }}
      function toSvgY(y) {{ return 200 - y * 50; }}

      // Draw initial points
      samples.forEach((s) => {{
        const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        c.setAttribute('cx', toSvgX(s.x1));
        c.setAttribute('cy', toSvgY(s.x2));
        c.setAttribute('r', '6');
        c.setAttribute('fill', s.y === 1 ? '{CORAL}' : '{INDIGO}');
        c.setAttribute('stroke', '#ffffff');
        c.setAttribute('stroke-width', '1.5');
        ptsGroup.appendChild(c);
      }});

      const inputTau = document.getElementById('input-tau');
      const inputW1 = document.getElementById('input-w1');
      const inputW2 = document.getElementById('input-w2');
      const inputB = document.getElementById('input-b');

      const valTau = document.getElementById('val-tau');
      const valW1 = document.getElementById('val-w1');
      const valW2 = document.getElementById('val-w2');
      const valB = document.getElementById('val-b');

      const cmTp = document.getElementById('cm-tp');
      const cmFp = document.getElementById('cm-fp');
      const cmTn = document.getElementById('cm-tn');
      const cmFn = document.getElementById('cm-fn');
      const mAcc = document.getElementById('metric-acc');
      const mF1 = document.getElementById('metric-f1');
      const mPrec = document.getElementById('metric-prec');
      const mRec = document.getElementById('metric-rec');
      const liveBce = document.getElementById('live-bce');

      function update() {{
        const tau = parseFloat(inputTau.value);
        const w1 = parseFloat(inputW1.value);
        const w2 = parseFloat(inputW2.value);
        const b = parseFloat(inputB.value);

        valTau.textContent = tau.toFixed(2);
        valW1.textContent = w1.toFixed(2);
        valW2.textContent = w2.toFixed(2);
        valB.textContent = b.toFixed(2);

        // Update boundary line: w1*x1 + w2*x2 + b = logit(tau)
        const logitTau = Math.log(tau / (1.0 - tau));
        if (Math.abs(w2) > 1e-4) {{
          const x1a = -3.2, x1b = 3.2;
          const x2a = (logitTau - b - w1 * x1a) / w2;
          const x2b = (logitTau - b - w1 * x1b) / w2;
          bLine.setAttribute('x1', toSvgX(x1a));
          bLine.setAttribute('y1', toSvgY(x2a));
          bLine.setAttribute('x2', toSvgX(x1b));
          bLine.setAttribute('y2', toSvgY(x2b));
        }}

        // Recompute metrics
        let tp = 0, fp = 0, tn = 0, fn = 0, bce = 0;
        samples.forEach(s => {{
          const z = w1 * s.x1 + w2 * s.x2 + b;
          const p = 1.0 / (1.0 + Math.exp(-z));
          const yHat = p >= tau ? 1 : 0;
          const pClip = Math.min(Math.max(p, 1e-12), 1.0 - 1e-12);
          bce -= (s.y * Math.log(pClip) + (1 - s.y) * Math.log(1 - pClip));

          if (s.y === 1 && yHat === 1) tp++;
          else if (s.y === 0 && yHat === 1) fp++;
          else if (s.y === 0 && yHat === 0) tn++;
          else fn++;
        }});

        bce /= samples.length;
        const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
        const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
        const f1 = prec + rec > 0 ? 2 * prec * rec / (prec + rec) : 0;
        const acc = (tp + tn) / samples.length;

        cmTp.textContent = tp;
        cmFp.textContent = fp;
        cmTn.textContent = tn;
        cmFn.textContent = fn;
        mAcc.textContent = (acc * 100).toFixed(1) + '%';
        mF1.textContent = f1.toFixed(3);
        mPrec.textContent = prec.toFixed(3);
        mRec.textContent = rec.toFixed(3);
        liveBce.textContent = bce.toFixed(4);
      }}

      inputTau.addEventListener('input', update);
      inputW1.addEventListener('input', update);
      inputW2.addEventListener('input', update);
      inputB.addEventListener('input', update);
      update();
    }})();
    </script>
    """


# ---------------------------------------------------------------------------
# 5. KNN Renderer
# ---------------------------------------------------------------------------


def _render_knn(data: dict[str, Any]) -> str:
    k = data["k"]
    metric = data["metric"]
    weights = data["weights"]
    qx, qy = data["query_point"]
    pred_cls = data["predicted_class"]

    py_code = """from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# KNN with feature scaling
X = np.random.randn(30, 2)
y = np.random.choice([0, 1, 2], size=30)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='distance')
knn.fit(X_scaled, y)

query = scaler.transform([[0.2, 0.5]])
print("Predicted Class:", knn.predict(query)[0])
print("Class Probabilities:", knn.predict_proba(query)[0])
"""

    return f"""
    <div class="lesson-grid">
      <div class="lab-panel">
        <div class="card">
          <div class="card-title">Interactive K-Nearest Neighbors Lab</div>
          <div class="visual-wrapper">
            <svg id="knn-svg" viewBox="0 0 600 400" preserveAspectRatio="xMidYMid meet">
              <g id="knn-lines"></g>
              <g id="knn-data-points"></g>
              <!-- Query Point -->
              <circle id="query-pt" r="8" fill="{PURPLE}" stroke="#ffffff" stroke-width="2.5" cursor="pointer"/>
            </svg>
          </div>

          <div class="controls-panel" style="margin-top: 16px;">
            <div class="control-row">
              <div class="control-header">
                <span>Number of Neighbors (k)</span>
                <span class="control-value" id="val-k">{k}</span>
              </div>
              <input type="range" id="input-k" min="1" max="15" step="1" value="{k}">
            </div>
            <div class="control-row">
              <div class="control-header">
                <span>Distance Metric</span>
              </div>
              <div class="btn-group">
                <button class="seedl-btn {"active" if metric == "euclidean" else ""}" id="btn-euclidean">Euclidean</button>
                <button class="seedl-btn {"active" if metric == "manhattan" else ""}" id="btn-manhattan">Manhattan</button>
              </div>
            </div>
            <div class="control-row">
              <div class="control-header">
                <span>Voting Weighting</span>
              </div>
              <div class="btn-group">
                <button class="seedl-btn {"active" if weights == "uniform" else ""}" id="btn-uniform">Uniform</button>
                <button class="seedl-btn {"active" if weights == "distance" else ""}" id="btn-distance">Distance-Weighted</button>
              </div>
            </div>
          </div>
        </div>

        <div class="metrics-grid">
          <div class="metric-card">
            <span class="metric-label">Predicted Class</span>
            <span class="metric-val" id="metric-pred-class">Class {pred_cls}</span>
          </div>
          <div class="metric-card">
            <span class="metric-label">Neighbors (k)</span>
            <span class="metric-val" id="metric-k-val">{k}</span>
          </div>
        </div>
      </div>

      <div style="display: flex; flex-direction: column; gap: 20px;">
        <div class="card">
          <div class="card-title">1. Intuition & Voting Mechanism</div>
          <p style="color: var(--color-muted); margin-bottom: 8px;">
            K-Nearest Neighbors is a non-parametric instance-based classifier. It assigns the class with majority support among the <em>k</em> closest points to the query.
          </p>
        </div>

        <div class="card">
          <div class="card-title">2. Live Mathematical Equations</div>
          <div class="math-block">
            <div class="math-eq">
              <strong>Euclidean:</strong> d(q, x) = &#8730;[ (q<sub>1</sub> &#8722; x<sub>1</sub>)<sup>2</sup> + (q<sub>2</sub> &#8722; x<sub>2</sub>)<sup>2</sup> ]
            </div>
            <div class="math-eq">
              <strong>Manhattan:</strong> d(q, x) = |q<sub>1</sub> &#8722; x<sub>1</sub>| + |q<sub>2</sub> &#8722; x<sub>2</sub>|
            </div>
            <div class="math-eq">
              <strong>Prediction:</strong> y&#770;<sub>q</sub> = argmax<sub>c</sub> &#8721; w<sub>i</sub>
              &nbsp;&#8594;&nbsp; <span class="math-live" id="live-pred-cls">Class {pred_cls}</span>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">3. Scientific Limitations & Scaling</div>
          <div class="limitations-box">
            <strong>Curse of Dimensionality:</strong> In high dimensions, pairwise distances concentrate and neighborhood locality breaks down. <strong>Feature Scaling</strong> (StandardScaler) is mandatory whenever features possess different units or scales.
          </div>
        </div>

        <div class="card">
          <div class="card-title">4. Reproducible Code</div>
          <pre class="code-box"><code>{html.escape(py_code)}</code></pre>
        </div>
      </div>
    </div>

    <script>
    (function() {{
      const data = JSON.parse(document.getElementById('seedl-lesson-data').textContent);
      const points = data.points;
      const ptsGroup = document.getElementById('knn-data-points');
      const linesGroup = document.getElementById('knn-lines');
      const queryPt = document.getElementById('query-pt');

      let currentMetric = "{metric}";
      let currentWeights = "{weights}";
      let qx = {qx}, qy = {qy};

      function toSvgX(x) {{ return 300 + x * 100; }}
      function toSvgY(y) {{ return 200 - y * 80; }}

      const colors = ['{CORAL}', '{BLUE}', '{EMERALD}'];

      points.forEach(p => {{
        const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        c.setAttribute('cx', toSvgX(p[0]));
        c.setAttribute('cy', toSvgY(p[1]));
        c.setAttribute('r', '6');
        c.setAttribute('fill', colors[p[2]]);
        c.setAttribute('stroke', '#ffffff');
        c.setAttribute('stroke-width', '1.5');
        ptsGroup.appendChild(c);
      }});

      const inputK = document.getElementById('input-k');
      const valK = document.getElementById('val-k');
      const metricKVal = document.getElementById('metric-k-val');
      const metricPredClass = document.getElementById('metric-pred-class');
      const livePredCls = document.getElementById('live-pred-cls');

      function update() {{
        const k = parseInt(inputK.value, 10);
        valK.textContent = k;
        metricKVal.textContent = k;

        queryPt.setAttribute('cx', toSvgX(qx));
        queryPt.setAttribute('cy', toSvgY(qy));

        // Distances
        const dists = points.map((p, idx) => {{
          let d = 0;
          if (currentMetric === 'euclidean') {{
            d = Math.sqrt(Math.pow(p[0] - qx, 2) + Math.pow(p[1] - qy, 2));
          }} else {{
            d = Math.abs(p[0] - qx) + Math.abs(p[1] - qy);
          }}
          return {{ p, d, idx }};
        }});

        dists.sort((a, b) => a.d - b.d);
        const topK = dists.slice(0, k);

        // Draw connecting lines
        linesGroup.innerHTML = '';
        const votes = [0, 0, 0];
        topK.forEach(item => {{
          const l = document.createElementNS('http://www.w3.org/2000/svg', 'line');
          l.setAttribute('x1', toSvgX(qx));
          l.setAttribute('y1', toSvgY(qy));
          l.setAttribute('x2', toSvgX(item.p[0]));
          l.setAttribute('y2', toSvgY(item.p[1]));
          l.setAttribute('stroke', colors[item.p[2]]);
          l.setAttribute('stroke-width', '2');
          l.setAttribute('stroke-dasharray', '3 3');
          linesGroup.appendChild(l);

          const w = currentWeights === 'uniform' ? 1.0 : (1.0 / (item.d + 1e-5));
          votes[item.p[2]] += w;
        }});

        const bestCls = votes.indexOf(Math.max(...votes));
        metricPredClass.textContent = `Class ${{bestCls}}`;
        livePredCls.textContent = `Class ${{bestCls}}`;
      }}

      inputK.addEventListener('input', update);

      document.getElementById('btn-euclidean').addEventListener('click', function() {{
        currentMetric = 'euclidean';
        this.classList.add('active');
        document.getElementById('btn-manhattan').classList.remove('active');
        update();
      }});
      document.getElementById('btn-manhattan').addEventListener('click', function() {{
        currentMetric = 'manhattan';
        this.classList.add('active');
        document.getElementById('btn-euclidean').classList.remove('active');
        update();
      }});
      document.getElementById('btn-uniform').addEventListener('click', function() {{
        currentWeights = 'uniform';
        this.classList.add('active');
        document.getElementById('btn-distance').classList.remove('active');
        update();
      }});
      document.getElementById('btn-distance').addEventListener('click', function() {{
        currentWeights = 'distance';
        this.classList.add('active');
        document.getElementById('btn-uniform').classList.remove('active');
        update();
      }});

      update();
    }})();
    </script>
    """


# ---------------------------------------------------------------------------
# 6. Decision Tree Renderer
# ---------------------------------------------------------------------------


def _render_decision_tree(data: dict[str, Any]) -> str:
    max_depth = data["max_depth"]
    criterion = data["criterion"]
    tree = data["tree"]
    regions = data["regions"]

    py_code = """from sklearn.tree import DecisionTreeClassifier, export_text
import numpy as np

# Recursive orthogonal partitioning
X = np.random.rand(32, 2) * 10
y = ((X[:, 0] > 3) & (X[:, 0] < 8) & (X[:, 1] > 2) & (X[:, 1] < 7)).astype(int)

clf = DecisionTreeClassifier(max_depth=2, criterion='gini')
clf.fit(X, y)

print(export_text(clf, feature_names=['x1', 'x2']))
"""

    return f"""
    <div class="lesson-grid">
      <div class="lab-panel">
        <div class="card">
          <div class="card-title">Dual View: 2D Partitions & Hierarchical Tree</div>
          <div class="visual-wrapper">
            <svg id="tree-svg" viewBox="0 0 600 400" preserveAspectRatio="xMidYMid meet">
              <g id="partition-regions"></g>
              <g id="dt-points"></g>
            </svg>
          </div>

          <div class="controls-panel" style="margin-top: 16px;">
            <div class="control-row">
              <div class="control-header">
                <span>Maximum Tree Depth</span>
                <span class="control-value" id="val-depth">{max_depth}</span>
              </div>
              <input type="range" id="input-depth" min="1" max="4" step="1" value="{max_depth}">
            </div>
            <div class="control-row">
              <div class="control-header">
                <span>Split Criterion</span>
              </div>
              <div class="btn-group">
                <button class="seedl-btn {"active" if criterion == "gini" else ""}" id="btn-gini">Gini Impurity</button>
                <button class="seedl-btn {"active" if criterion == "entropy" else ""}" id="btn-entropy">Entropy</button>
              </div>
            </div>
          </div>
        </div>

        <div class="metrics-grid">
          <div class="metric-card">
            <span class="metric-label">Max Depth</span>
            <span class="metric-val">{max_depth}</span>
          </div>
          <div class="metric-card">
            <span class="metric-label">Leaf Regions</span>
            <span class="metric-val">{len(regions)}</span>
          </div>
          <div class="metric-card">
            <span class="metric-label">Root Impurity</span>
            <span class="metric-val">{tree["impurity"]:.3f}</span>
          </div>
        </div>
      </div>

      <div style="display: flex; flex-direction: column; gap: 20px;">
        <div class="card">
          <div class="card-title">1. Recursive Binary Partitioning</div>
          <p style="color: var(--color-muted); margin-bottom: 8px;">
            Decision trees recursively select the feature and threshold that maximizes Information Gain &Delta;I.
            The resulting partitions are orthogonal hyper-rectangles in feature space.
          </p>
        </div>

        <div class="card">
          <div class="card-title">2. Live Mathematical Equations</div>
          <div class="math-block">
            <div class="math-eq">
              <strong>Gini:</strong> I<sub>G</sub> = 1 &#8722; &#8721; p<sub>k</sub><sup>2</sup>
            </div>
            <div class="math-eq">
              <strong>Entropy:</strong> H = &#8722;&#8721; p<sub>k</sub> log<sub>2</sub>(p<sub>k</sub>)
            </div>
            <div class="math-eq">
              <strong>Information Gain:</strong> &Delta;I = I(D) &#8722; <sup>|D<sub>L</sub>|</sup>/<sub>|D|</sub> I(D<sub>L</sub>) &#8722; <sup>|D<sub>R</sub>|</sup>/<sub>|D|</sub> I(D<sub>R</sub>)
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">3. Walkthrough & Limitations</div>
          <div class="limitations-box">
            <strong>High Variance:</strong> Decision trees are greedy estimators; minor changes in data can drastically alter the root split. Ensembles (Random Forests, Gradient Boosting) reduce variance.
          </div>
        </div>

        <div class="card">
          <div class="card-title">4. Reproducible Code</div>
          <pre class="code-box"><code>{html.escape(py_code)}</code></pre>
        </div>
      </div>
    </div>

    <script>
    (function() {{
      const data = JSON.parse(document.getElementById('seedl-lesson-data').textContent);
      const dataset = data.dataset;
      const regions = data.regions;

      const regGroup = document.getElementById('partition-regions');
      const ptsGroup = document.getElementById('dt-points');

      function toSvgX(x) {{ return 40 + (x / 10.0) * 520; }}
      function toSvgY(y) {{ return 360 - (y / 10.0) * 320; }}

      // Draw regions
      regions.forEach(r => {{
        const [xmin, xmax, ymin, ymax] = r.bounds;
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', toSvgX(xmin));
        rect.setAttribute('y', toSvgY(ymax));
        rect.setAttribute('width', toSvgX(xmax) - toSvgX(xmin));
        rect.setAttribute('height', toSvgY(ymin) - toSvgY(ymax));
        rect.setAttribute('fill', r.class === 1 ? '#fee2e2' : '#e0e7ff');
        rect.setAttribute('stroke', '#cbd5e1');
        rect.setAttribute('stroke-width', '1.5');
        regGroup.appendChild(rect);
      }});

      // Draw points
      dataset.forEach(p => {{
        const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        c.setAttribute('cx', toSvgX(p[0]));
        c.setAttribute('cy', toSvgY(p[1]));
        c.setAttribute('r', '5');
        c.setAttribute('fill', p[2] === 1 ? '{CORAL}' : '{INDIGO}');
        c.setAttribute('stroke', '#ffffff');
        c.setAttribute('stroke-width', '1.5');
        ptsGroup.appendChild(c);
      }});
    }})();
    </script>
    """


# ---------------------------------------------------------------------------
# 7. KMeans Renderer
# ---------------------------------------------------------------------------


def _render_kmeans(data: dict[str, Any]) -> str:
    k = data["n_clusters"]
    init = data["init"]
    inertia = data["final_inertia"]
    steps = data["convergence_step"]

    py_code = """from sklearn.cluster import KMeans
import numpy as np

# KMeans++ clustering
X = np.random.randn(36, 2)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(X)

print("Centroids:\\n", kmeans.cluster_centers_)
print(f"Final Inertia (WCSS): {kmeans.inertia_:.4f}")
"""

    return f"""
    <div class="lesson-grid">
      <div class="lab-panel">
        <div class="card">
          <div class="card-title">Interactive KMeans Clustering & Inertia Lab</div>
          <div class="visual-wrapper">
            <svg id="km-svg" viewBox="0 0 600 400" preserveAspectRatio="xMidYMid meet">
              <g id="km-points"></g>
              <g id="km-centroids"></g>
            </svg>
          </div>

          <div class="controls-panel" style="margin-top: 16px;">
            <div class="control-row">
              <div class="control-header">
                <span>Number of Clusters (K)</span>
                <span class="control-value" id="val-k">{k}</span>
              </div>
              <input type="range" id="input-k" min="2" max="5" step="1" value="{k}">
            </div>
            <div class="control-row">
              <div class="control-header">
                <span>Initialization Scheme</span>
              </div>
              <div class="btn-group">
                <button class="seedl-btn {"active" if init == "kmeans++" else ""}" id="btn-kpp">KMeans++</button>
                <button class="seedl-btn {"active" if init == "random" else ""}" id="btn-rand">Random</button>
              </div>
            </div>
          </div>
        </div>

        <div class="metrics-grid">
          <div class="metric-card">
            <span class="metric-label">Clusters K</span>
            <span class="metric-val">{k}</span>
          </div>
          <div class="metric-card">
            <span class="metric-label">Inertia (WCSS)</span>
            <span class="metric-val">{inertia:.3f}</span>
          </div>
          <div class="metric-card">
            <span class="metric-label">Iterations</span>
            <span class="metric-val">{steps}</span>
          </div>
        </div>
      </div>

      <div style="display: flex; flex-direction: column; gap: 20px;">
        <div class="card">
          <div class="card-title">1. Alternating Optimization</div>
          <p style="color: var(--color-muted); margin-bottom: 8px;">
            KMeans alternates between <strong>Assignment Step</strong> (assign each point to nearest centroid) and <strong>Centroid Update Step</strong> (move centroid to the mean of assigned points).
          </p>
        </div>

        <div class="card">
          <div class="card-title">2. Live Mathematical Equations</div>
          <div class="math-block">
            <div class="math-eq">
              <strong>Objective (Inertia):</strong> J = &#8721;<sub>k=1</sub><sup>K</sup> &#8721;<sub>i &isin; C<sub>k</sub></sub> ||x<sub>i</sub> &#8722; &mu;<sub>k</sub>||<sup>2</sup>
            </div>
            <div class="math-eq">
              <strong>Centroid Update:</strong> &mu;<sub>k</sub> = <sup>1</sup>/<sub>|C<sub>k</sub>|</sub> &#8721;<sub>i &isin; C<sub>k</sub></sub> x<sub>i</sub>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">3. Limitations</div>
          <div class="limitations-box">
            Assumes isotropic, spherical clusters of roughly equal density. Fails on non-convex geometric shapes like concentric rings.
          </div>
        </div>

        <div class="card">
          <div class="card-title">4. Reproducible Code</div>
          <pre class="code-box"><code>{html.escape(py_code)}</code></pre>
        </div>
      </div>
    </div>

    <script>
    (function() {{
      const data = JSON.parse(document.getElementById('seedl-lesson-data').textContent);
      const points = data.points;
      const history = data.history;

      const ptsGroup = document.getElementById('km-points');
      const centGroup = document.getElementById('km-centroids');

      function toSvgX(x) {{ return 300 + x * 75; }}
      function toSvgY(y) {{ return 200 - y * 55; }}

      const colors = ['{BLUE}', '{CORAL}', '{EMERALD}', '{PURPLE}', '{AMBER}'];

      const finalStep = history[history.length - 1];

      // Draw points
      points.forEach((p, idx) => {{
        const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        c.setAttribute('cx', toSvgX(p[0]));
        c.setAttribute('cy', toSvgY(p[1]));
        c.setAttribute('r', '5');
        const clusterIdx = finalStep.assignments[idx];
        c.setAttribute('fill', colors[clusterIdx % colors.length]);
        c.setAttribute('stroke', '#ffffff');
        c.setAttribute('stroke-width', '1.5');
        ptsGroup.appendChild(c);
      }});

      // Draw centroids
      finalStep.centroids.forEach((c, idx) => {{
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        const cx = toSvgX(c[0]), cy = toSvgY(c[1]);
        const pts = `${{cx}},${{cy - 10}} ${{cx + 10}},${{cy}} ${{cx}},${{cy + 10}} ${{cx - 10}},${{cy}}`;
        rect.setAttribute('points', pts);
        rect.setAttribute('fill', colors[idx % colors.length]);
        rect.setAttribute('stroke', '#0f172a');
        rect.setAttribute('stroke-width', '2');
        centGroup.appendChild(rect);
      }});
    }})();
    </script>
    """


# ---------------------------------------------------------------------------
# 8. PCA Renderer
# ---------------------------------------------------------------------------


def _render_pca(data: dict[str, Any]) -> str:
    angle = data["angle"]
    opt_angle = data["optimal_angle"]
    ratio1, ratio2 = data["explained_variance_ratio"]
    proj_var = data["projected_variance"]
    rec_err = data["mean_reconstruction_error"]

    py_code = """from sklearn.decomposition import PCA
import numpy as np

# Principal Component Analysis
X = np.random.multivariate_normal([0, 0], [[2.5, 1.8], [1.8, 1.5]], size=50)

pca = PCA(n_components=2)
pca.fit(X)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Principal Vectors (Components):\\n", pca.components_)
"""

    return f"""
    <div class="lesson-grid">
      <div class="lab-panel">
        <div class="card">
          <div class="card-title">Interactive PCA & Projection Axis Lab</div>
          <div class="visual-wrapper">
            <svg id="pca-svg" viewBox="0 0 600 400" preserveAspectRatio="xMidYMid meet">
              <g id="pca-rays"></g>
              <line id="pca-axis" stroke="{PURPLE}" stroke-width="2.5"/>
              <line id="pc1-line" stroke="{EMERALD}" stroke-width="2" stroke-dasharray="4 4"/>
              <g id="pca-pts"></g>
            </svg>
          </div>

          <div class="controls-panel" style="margin-top: 16px;">
            <div class="control-row">
              <div class="control-header">
                <span>Projection Axis Angle (&theta;)</span>
                <span class="control-value" id="val-theta">{angle:.1f}&deg;</span>
              </div>
              <input type="range" id="input-theta" min="0" max="180" step="1" value="{angle}">
            </div>
            <div class="btn-group" style="margin-top: 4px;">
              <button class="seedl-btn" id="btn-snap-pc1">⚡ Snap to PC1 ({opt_angle:.1f}&deg;)</button>
              <button class="seedl-btn" id="btn-snap-pc2">Snap to PC2 ({(opt_angle + 90) % 180:.1f}&deg;)</button>
            </div>
          </div>
        </div>

        <div class="metrics-grid">
          <div class="metric-card">
            <span class="metric-label">Projected Variance</span>
            <span class="metric-val" id="metric-proj-var">{proj_var:.3f}</span>
          </div>
          <div class="metric-card">
            <span class="metric-label">Reconstruction Error</span>
            <span class="metric-val" id="metric-rec-err">{rec_err:.4f}</span>
          </div>
          <div class="metric-card">
            <span class="metric-label">PC1 Explains</span>
            <span class="metric-val">{ratio1:.1%}</span>
          </div>
        </div>
      </div>

      <div style="display: flex; flex-direction: column; gap: 20px;">
        <div class="card">
          <div class="card-title">1. Fundamental Theorem of PCA</div>
          <p style="color: var(--color-muted); margin-bottom: 8px;">
            <strong>Maximizing variance is mathematically equivalent to minimizing reconstruction error!</strong>
            Rotate the angle slider and observe: as the projected points spread out to their maximum variance, the perpendicular error rays reach their minimum length.
          </p>
        </div>

        <div class="card">
          <div class="card-title">2. Live Mathematical Equations</div>
          <div class="math-block">
            <div class="math-eq">
              <strong>Covariance:</strong> &Sigma; = <sup>1</sup>/<sub>N&#8722;1</sub> X<sub>c</sub><sup>T</sup> X<sub>c</sub>
            </div>
            <div class="math-eq">
              <strong>Projected Variance:</strong> Var(z) = u<sup>T</sup> &Sigma; u
              = <span class="math-live" id="live-proj-var">{proj_var:.3f}</span>
            </div>
            <div class="math-eq">
              <strong>Reconstruction MSE:</strong> <sup>1</sup>/<sub>N</sub> &#8721; ||x<sub>i</sub> &#8722; x&#770;<sub>i</sub>||<sup>2</sup>
              = <span class="math-live" id="live-rec-err">{rec_err:.4f}</span>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">3. Scientific Limitations</div>
          <div class="limitations-box">
            PCA is strictly a linear transformation. For curved, non-linear manifolds (such as Swiss rolls or single-cell trajectories), non-linear techniques like t-SNE or UMAP are required.
          </div>
        </div>

        <div class="card">
          <div class="card-title">4. Reproducible Code</div>
          <pre class="code-box"><code>{html.escape(py_code)}</code></pre>
        </div>
      </div>
    </div>

    <script>
    (function() {{
      const data = JSON.parse(document.getElementById('seedl-lesson-data').textContent);
      const points = data.projections.map(p => p.orig);
      const mean = data.mean;
      const optAngle = data.optimal_angle;

      const ptsGroup = document.getElementById('pca-pts');
      const raysGroup = document.getElementById('pca-rays');
      const pcaAxis = document.getElementById('pca-axis');
      const pc1Line = document.getElementById('pc1-line');

      function toSvgX(x) {{ return 300 + x * 65; }}
      function toSvgY(y) {{ return 200 - y * 65; }}

      // Draw initial points
      points.forEach(p => {{
        const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        c.setAttribute('cx', toSvgX(p[0]));
        c.setAttribute('cy', toSvgY(p[1]));
        c.setAttribute('r', '5');
        c.setAttribute('fill', '{INDIGO}');
        c.setAttribute('stroke', '#ffffff');
        c.setAttribute('stroke-width', '1.5');
        ptsGroup.appendChild(c);
      }});

      // Draw PC1 reference line
      const optRad = optAngle * Math.PI / 180;
      pc1Line.setAttribute('x1', toSvgX(mean[0] - 4 * Math.cos(optRad)));
      pc1Line.setAttribute('y1', toSvgY(mean[1] - 4 * Math.sin(optRad)));
      pc1Line.setAttribute('x2', toSvgX(mean[0] + 4 * Math.cos(optRad)));
      pc1Line.setAttribute('y2', toSvgY(mean[1] + 4 * Math.sin(optRad)));

      const inputTheta = document.getElementById('input-theta');
      const valTheta = document.getElementById('val-theta');
      const metricProjVar = document.getElementById('metric-proj-var');
      const metricRecErr = document.getElementById('metric-rec-err');
      const liveProjVar = document.getElementById('live-proj-var');
      const liveRecErr = document.getElementById('live-rec-err');

      function update(deg) {{
        valTheta.textContent = deg.toFixed(1) + '°';
        const rad = deg * Math.PI / 180;
        const ux = Math.cos(rad);
        const uy = Math.sin(rad);

        pcaAxis.setAttribute('x1', toSvgX(mean[0] - 4.5 * ux));
        pcaAxis.setAttribute('y1', toSvgY(mean[1] - 4.5 * uy));
        pcaAxis.setAttribute('x2', toSvgX(mean[0] + 4.5 * ux));
        pcaAxis.setAttribute('y2', toSvgY(mean[1] + 4.5 * uy));

        // Rays & Metrics
        raysGroup.innerHTML = '';
        let projScalars = [];
        let recErrSq = 0;

        points.forEach(p => {{
          const xc = p[0] - mean[0];
          const yc = p[1] - mean[1];
          const z = xc * ux + yc * uy;
          projScalars.push(z);

          const rx = mean[0] + z * ux;
          const ry = mean[1] + z * uy;
          recErrSq += Math.pow(p[0] - rx, 2) + Math.pow(p[1] - ry, 2);

          const l = document.createElementNS('http://www.w3.org/2000/svg', 'line');
          l.setAttribute('x1', toSvgX(p[0]));
          l.setAttribute('y1', toSvgY(p[1]));
          l.setAttribute('x2', toSvgX(rx));
          l.setAttribute('y2', toSvgY(ry));
          l.setAttribute('stroke', '{CORAL}');
          l.setAttribute('stroke-width', '1.5');
          l.setAttribute('stroke-dasharray', '2 2');
          raysGroup.appendChild(l);
        }});

        const v = projScalars.reduce((a, z) => a + z * z, 0) / (points.length - 1);
        const err = recErrSq / points.length;

        metricProjVar.textContent = v.toFixed(3);
        metricRecErr.textContent = err.toFixed(4);
        liveProjVar.textContent = v.toFixed(3);
        liveRecErr.textContent = err.toFixed(4);
      }}

      inputTheta.addEventListener('input', () => update(parseFloat(inputTheta.value)));
      document.getElementById('btn-snap-pc1').addEventListener('click', () => {{
        inputTheta.value = optAngle;
        update(optAngle);
      }});
      document.getElementById('btn-snap-pc2').addEventListener('click', () => {{
        const pc2Angle = (optAngle + 90) % 180;
        inputTheta.value = pc2Angle;
        update(pc2Angle);
      }});

      update(parseFloat(inputTheta.value));
    }})();
    </script>
    """


# ---------------------------------------------------------------------------
# Fallback / Deep Learning concepts renderer
# ---------------------------------------------------------------------------


def _render_generic_educational(concept: str, data: dict[str, Any], notes: list[str]) -> str:
    notes_html = "".join(f"<li>{_escape(n)}</li>" for n in notes)
    data_pretty = html.escape(json.dumps(data, indent=2))

    return f"""
    <div class="card">
      <div class="card-title">Educational Deep Learning Simulation: {_escape(concept.replace("_", " ").title())}</div>
      <ul class="note-list" style="margin-bottom: 16px;">{notes_html}</ul>
      <div class="card-title" style="font-size: 14px;">Structured Lesson Payload</div>
      <pre class="code-box"><code>{data_pretty}</code></pre>
    </div>
    """
