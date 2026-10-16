# SeeDL 0.5.0 — Neural Observatory Official Website Overhaul Plan

## 1. Executive Summary & Mission

SeeDL 0.5.0 ("The Neural Observatory") introduces a comprehensive suite of scientific visualization and diagnostic capabilities for PyTorch deep neural networks, Transformers, explainability algorithms, training trajectories, open quantum systems, dataset drift, and multi-component scientific reports.

The mission of this overhaul is to elevate the official SeeDL website (`https://iamaritrasaha.github.io/seedl/`) into a modern, responsive, high-performance, and mathematically rigorous interactive scientific research lab.

### Key Objectives
1. **Preserve All Existing Capabilities**: Maintain 100% backward compatibility for all 10 existing interactive demonstrations from SeeDL 0.4.0 (CNN feature maps, multi-head attention, head comparison, ViT patch attention, 3D neural network, training playback, training comparison, Bloch sphere, quantum circuit, and quantum evolution).
2. **Expand to 19 Scientific Demonstrations**: Add 9 new flagship demonstrations highlighting SeeDL 0.5.0 Neural Observatory capabilities, far exceeding the $\ge 17$ requirement.
3. **Playground Architecture 2.0**: Implement a categorized, searchable, accessible, and deep-linkable playground navigation system with 7 distinct scientific domains.
4. **Authentic Provenance & Real Data**: Generate every demonstration asset using the published `seedl==0.5.0` package from PyPI with deterministic seeds and real numerical tensors — zero fabricated graphs or synthetic mockups.
5. **Rigorous Quality & Security**: Ensure zero runtime CDNs, self-contained offline execution, iframe sandboxing, WCAG 2.1 AA accessibility, and automated browser verification via Playwright in Chromium.
6. **Strict Release Constraints**: Maintain manual-only GitHub Pages deployment (`workflow_dispatch`), pin `seedl==0.5.0`, and ensure all 6 CI matrix jobs pass green.

---

## 2. Visual Identity & Design System

The visual design is modeled as a **Scientific Instrument for Neural and Quantum Observatories**:
- **Palette**: Deep slate backgrounds (`#090e17` / `#0f172a`), refined borders (`#263244`), mint/cyan primary accents (`#8aebd7` / `#00e5ff`), and amber highlights (`#fbbf24`). High-contrast light mode (`#f5f7fa` with `#076c5a` accent) fully supported.
- **Typography**: Clean system sans-serif hierarchy (`Inter`, system UI) paired with monospace tabular numerals for scientific tensors, matrices, coordinates, and metrics.
- **Layout**: Flexible CSS grid and flexbox architectures that seamlessly adapt from 4K workstations down to 320px mobile displays.
- **Interactive Stages**: Responsive iframes with contained horizontal scrolling, HUD metadata badges, full-window breakout links, and interactive inspection sidebars.

---

## 3. Demonstration Catalog (19 Demonstrations)

| Index | Slug | Category | Scientific Method & API | Key Features & Provenance |
|---|---|---|---|---|
| 01 | `cnn` | Explainability | `sd.features(model, input)` | Preserved 0.4.0: 5 layers, spatial hover `(y, x)`, 3 normalization modes. |
| 02 | `attention` | Transformers | `sd.attention(weights, tokens)` | Preserved 0.4.0: 4 heads, 7 tokens, entropy calculation, row-stochastic. |
| 03 | `attention-cmp` | Transformers | `sd.attention(compare_heads=...)` | Preserved 0.4.0: Head A vs B, difference heatmap, MAE, Frobenius norm, cosine similarity. |
| 04 | `vit` | Transformers | `sd.vit_attention(weights, patch_grid)` | Preserved 0.4.0: [CLS] token spatial routing over 4×4 image patch grid. |
| 05 | `network` | Neural Networks | `sd.see(model, input, view="3d")` | Preserved 0.4.0: 3→5→3→2 MLP, WebGL 3D orbit, neuron selection, incoming/outgoing weights. |
| 06 | `training` | Training | `sd.watch(...) -> run.playback()` | Preserved 0.4.0: 16 recorded passes, loss timeline, neuron activation/gradient inspection. |
| 07 | `comparison` | Training | `sd.compare(snap_0, snap_15)` | Preserved 0.4.0: Step 0 vs 15 parameter deltas, gradient stats, output shifts. |
| 08 | `bloch` | Quantum | `sd.bloch(state, view="3d")` | Preserved 0.4.0: 6 basis/superposition presets (`0, 1, +, -, +i, -i`), unit sphere geometry. |
| 09 | `circuit` | Quantum | `sd.see(qc)` | Preserved 0.4.0: Two-qubit Bell circuit with H and CX, interactive gate hover. |
| 10 | `evolution` | Quantum | `sd.circuit(qc, simulate=True)` | Preserved 0.4.0: Gate-by-gate unitary evolution, statevector probabilities and Bloch vectors. |
| 11 | `trace` | Neural Networks | `sd.trace(model, input)` | **NEW in 0.5.0**: Execution tracer DAG with explicit tensor dependencies, aliases, shapes, execution order. |
| 12 | `flow` | Neural Networks | `sd.flow(model, input)` | **NEW in 0.5.0**: Tensor shape, dtype, and memory transition pipeline across layer boundaries. |
| 13 | `rollout` | Transformers | `sd.attention_rollout(weights)` | **NEW in 0.5.0**: Multi-layer residual attention propagation: $A'_l = \text{norm}(A_l + I)$, cumulative product. |
| 14 | `attribution` | Explainability | `sd.compare_attributions(model, ...)` | **NEW in 0.5.0**: Saliency vs Integrated Gradients vs SmoothGrad with Pearson and Cosine metrics. |
| 15 | `gradient-health` | Training | `sd.gradient_health(model)` | **NEW in 0.5.0**: Layer-level gradient norms (L1/L2), gradient-to-weight ratios, sparsity, non-finite check. |
| 16 | `quantum-noise` | Quantum | `sd.quantum_noise(channel, p)` | **NEW in 0.5.0**: Open quantum systems with Kraus channels (depolarizing, amplitude/phase damping), purity loss. |
| 17 | `dataset-drift` | Data Intelligence | `sd.dataset_drift(baseline, target)` | **NEW in 0.5.0**: Numerical feature shift diagnostics, Population Stability Index (PSI), Jensen-Shannon Divergence. |
| 18 | `report` | Scientific Reports | `sd.report(...)` | **NEW in 0.5.0**: Sandboxed offline diagnostic multi-card scientific report combining disparate observatory views. |
| 19 | `runs-compare` | Training | `sd.compare_runs(run_a, run_b)` | **NEW in 0.5.0**: Experiment-level training trajectory comparison, loss convergence curves, step alignment. |

---

## 4. Playground Architecture 2.0

### Categorized Navigation
The playground organizes 19 demos across 7 distinct categories:
- **Neural Networks**: 3D Network (`network`), Execution Tracer (`trace`), Tensor Flow (`flow`)
- **Transformers**: Multi-Head Attention (`attention`), Head Comparison (`attention-cmp`), ViT Patch Attention (`vit`), Attention Rollout (`rollout`)
- **Explainability**: Attribution Benchmark (`attribution`), CNN Feature Maps (`cnn`)
- **Training & Diagnostics**: Training Playback (`training`), Step Comparison (`comparison`), Run Comparison (`runs-compare`), Gradient Health (`gradient-health`)
- **Quantum Computing**: Bloch Sphere (`bloch`), Quantum Circuit (`circuit`), State Evolution (`evolution`), Quantum Noise (`quantum-noise`)
- **Data Intelligence**: Dataset Drift (`dataset-drift`)
- **Scientific Reports**: Multi-Section Report (`report`)

### Interaction & Accessibility Controls
- **Category Filter Pills**: Filter tabs by domain with single click/tap, or view "All Demos".
- **Instant Search Filter**: Client-side instant keyword filter matching demo titles, descriptions, and technologies.
- **Deep Linking**: Hash-based URL routing (`#trace`, `#rollout`, etc.) preserving backward compatibility with legacy bookmarks (`#cnn`, `#attention`, `#network`, etc.).
- **Keyboard Navigation**: Full WAI-ARIA tablist accessibility with `ArrowLeft`, `ArrowRight`, `Home`, and `End` navigation.
- **Offline & Responsive**: Lazy-loaded iframes with dynamic height calculation, zero CDN dependencies, and offline resilience.

---

## 5. Homepage 0.5.0 Architecture

The homepage (`website/index.html`) is updated with:
1. **Hero Section**: "Neural Observatory — Trace the computation. Understand the model. Visualize the science."
2. **Updated Installation Grid**:
   - `pip install seedl==0.5.0`
   - `pip install "seedl[torch,plotly]==0.5.0"`
   - `pip install "seedl[quantum,plotly]==0.5.0"`
   - `pip install "seedl[all]==0.5.0"`
3. **Observatory Showcase Cards**:
   - Execution Tracing & Tensor Flow
   - Transformer Attention & Rollout
   - Advanced Explainability (Integrated Gradients & SmoothGrad)
   - Gradient Health Diagnostics
   - Quantum Intelligence & Open Quantum Systems (Noise channels)
   - Dataset Drift & Health
   - Unified Scientific Reports
4. **Interactive Previews**: Links directly to the relevant playground demonstration stage.

---

## 6. Documentation Overhaul (`website/docs/index.html`)

The technical documentation documents the complete SeeDL 0.5.0 public API:
1. **Execution Tracing & Tensor Flow**: `sd.trace`, `sd.flow`, dependency edge semantics (`explicit_tensor`, `inferred_sequential`, etc.).
2. **Transformer Intelligence**: `sd.attention`, `sd.attention_rollout`, mathematical formulation of residual attention accumulation.
3. **Explainability**: `sd.explain`, `sd.compare_attributions`, composite midpoint numerical integration in Integrated Gradients, SmoothGrad Gaussian noise, Pearson correlation and Cosine similarity metrics.
4. **Gradient Health**: `sd.gradient_health`, layer L1/L2 norms, gradient-to-weight ratios, sparsity diagnostics.
5. **Training Trajectories**: `sd.watch`, `sd.compare` (snapshots), `sd.compare_runs` (multi-run experiments).
6. **Quantum Systems**: `sd.bloch`, `sd.circuit`, `sd.quantum_noise` (depolarizing, amplitude damping, phase damping, Kraus operators).
7. **Dataset Intelligence**: `sd.dataset_health`, `sd.dataset_drift`, Population Stability Index (PSI) and Jensen-Shannon Divergence (JSD).
8. **Scientific Reports**: `sd.report`, offline sandboxing, report composition.
9. **Analysis Integrity & Serialization**: `sd.save_analysis`, `sd.load_analysis`, SHA-256 integrity validation, deterministic canonical JSON.

---

## 7. Build, Asset Generation & CI Pipeline

1. **Asset Generation (`website/generate.py`)**:
   - Uses `seedl==0.5.0` installed in site-packages.
   - Deterministic model definitions with CPU-only execution and fixed random seeds.
   - Outputs standalone HTML viewers and SVG fallback assets to `website/demos/` and `website/assets/`.
   - Computes SHA-256 digests and numerical provenance into `website/demos/manifest.json`.
2. **Validation Engine (`website/check.py`)**:
   - Verifies 100% of internal links, anchor IDs, asset existence, and payload sizes.
   - Asserts manifest version is `0.5.0` and all demo SHA-256 digests match.
   - Validates that every code snippet in documentation executes cleanly without errors.
3. **Browser Acceptance Suite (`scripts/verify_website.py`)**:
   - Playwright Chromium test suite serving the site under `/seedl/`.
   - Tests navigation, theme toggling, mobile responsiveness, tab switching, and user interactions across all 19 demos.
   - Verifies offline operation, zero browser console errors, and zero external network requests.
4. **GitHub Pages Deployment (`.github/workflows/pages.yml`)**:
   - Pinned to `seedl==0.5.0`.
   - Restricted to `workflow_dispatch` (manual trigger only). No automatic deployment on push or PR.
