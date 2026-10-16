# SeeDL official website

Static HTML/CSS/JavaScript for SeeDL **0.4.0. Vision & Attention Intelligence**.
No JavaScript build step, external CDN, analytics, browser Python, or remote inference.
The Python library and package version are unchanged.

## Build and preview

From the repository root, using Python 3.10+:

```bash
uv sync --extra torch --extra plotly --extra quantum --extra dev
.venv/bin/python website/generate.py
.venv/bin/python website/check.py
.venv/bin/python -m http.server 8000 --directory website --bind 127.0.0.1
```

Open `http://127.0.0.1:8000/`. The build generates nineteen HTML demonstrations,
five static SVG previews, a copied neural-network image, provenance
JSON with SHA-256 hashes, and one shared local Plotly bundle. Generated assets
are ignored by Git and rebuilt before deployment. The generator uses SeeDL's
own Plotly figures, its existing neuron interaction bridge, its 0.4.0
lightweight offline HTML viewers, and its playback exporter with
`include_plotlyjs=False`; no scientific engine is reimplemented.

The source HTML is edited directly. The generator only updates demonstration
assets. Fixed PyTorch seeds make demonstrations repeatable; exact floating-point
bytes can vary across Python/platform versions. The manual workflow pins its
scientific dependencies; the manifest records the versions used by each build.

## Vision & Attention Demos (New in 0.4.0)

1. **CNN Feature Map Explorer**: Captures genuine intermediate activations from
   a deterministic PyTorch CNN (`Conv2d`, `ReLU`, `MaxPool2d`). Inspects layers,
   channels, spatial coordinates, channel thumbnails, and normalization modes
   (`per_channel`, `layer_global`, and `zero_centered`).

2. **Transformer Attention Explorer**: Visualizes authentic multi-head routing
   weights from `torch.nn.MultiheadAttention` over a 7-token sequence. Supports
   individual heads, average routing, query-token selection, key-token inspection,
   and dynamic 1D query attention distributions. Values are observational routing
   weights ($P(K_j \mid Q_i)$), not causal proofs of feature importance.

3. **Multi-Head Attention Comparison**: Compares routing shifts between attention
   heads side-by-side with difference heatmaps and divergence metrics (Frobenius
   norm, MAE, and cosine similarity).

4. **Vision Transformer (ViT) Patch Attention**: Reshapes 1D token attention
   slices (`[CLS]` token 0) into 2D $4 \times 4$ spatial image patch heatmaps over
   the image grid.

## Preserved 0.3.0 Demonstrations

- **Training Playback**: 16 seeded, CPU-only SGD steps on a 2 → 4 → 1 PyTorch
  network. `sd.watch()` records each completed pass with `run.capture()`
  **before** `optimizer.step()`.
- **Training Comparison**: `sd.compare(run.snapshots[0], run.snapshots[-1])`
  renders sampled weight changes, activation distributions, gradient statistics,
  and captured output differences.
- **Quantum Evolution**: `sd.circuit(qc, simulate=True)` exports a Bell circuit
  and a Bell + RZ phase variant. Every gate's amplitudes, probabilities, phases,
  and reduced-qubit Bloch vectors are checked against Qiskit.
- **3D Neural Network**: Interactive WebGL rendering of neuron connections and
  activations with orbit, zoom, and neuron inspection.
- **Bloch Sphere**: Six exact single-qubit quantum states on the unit sphere.

## Verification

```bash
.venv/bin/python website/check.py
# Uses existing Playwright + Chromium; no browser installation during tests.
.venv/bin/python scripts/verify_website.py
```

The browser check serves the site under `/seedl/`, exercises CNN layer/channel
selection, hover coordinates, normalization toggles, Transformer head and query
token selection, attention cell hover, head comparison pairs, ViT patch grid
hover, physical neuron clicks, incoming/outgoing highlighting, camera drag/zoom/reset,
all six Bloch states, Bell gate inspection, theme persistence, keyboard tabs,
navigation, mobile layout, recorded playback controls, training neuron clicks,
loss/activation/weight/gradient updates, circuit gate selection, Bell/phase
values, sampled comparisons, failed requests, and browser errors. It writes
desktop, mobile, and demo screenshots to `artifacts/website/`.

## Deploy. only when authorized

1. Push the reviewed branch and merge through the normal review process.
2. In repository **Settings → Pages**, choose **GitHub Actions** as the source.
3. In **Actions → Official website (manual deployment)**, choose **Run workflow**
   on the reviewed branch/ref. Configure an environment reviewer for
   `github-pages` if a separate publication approval is desired.
4. After a successful run, visit `https://iamaritrasaha.github.io/seedl/`.

There are no push/release triggers, secrets, or automatic publication. The
build job has read-only repository access. Only the deployment job receives
`pages: write` and `id-token: write`. PyPI publishing is untouched. Relative
paths support both a project site under `/seedl/` and a root-level local preview.

## Viewer layout

Canvas and token viewers fit the available width and their frames follow the
content height. Multi-panel Plotly figures keep a readable width on small screens;
scroll within the view or use its full-window link. Switching tabs resizes Plotly
once the panel is visible. The website adds presentation styles during generation,
before calculating demo hashes; captured values and library exports are preserved.
