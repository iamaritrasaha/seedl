# SeeDL official website

Static HTML/CSS/JavaScript for SeeDL **0.3.0**. No JavaScript build step, CDN,
analytics, browser Python, or remote inference. The Python library and package
version are unchanged.

## Build and preview

From the repository root, using Python 3.10+:

```bash
uv sync --extra torch --extra plotly --extra quantum --extra dev
.venv/bin/python website/generate.py
.venv/bin/python website/check.py
.venv/bin/python -m http.server 8000 --directory website --bind 127.0.0.1
```

Open `http://127.0.0.1:8000/`. The build generates twelve HTML demonstrations,
two static SVG previews, a copied existing neural-network image, provenance
JSON with SHA-256 hashes, and one shared local Plotly bundle. Generated assets
are ignored by Git and rebuilt before deployment. The generator uses SeeDL's
own Plotly figures, its existing neuron interaction bridge, and its 0.3.0
playback exporter with `include_plotlyjs=False`; no scientific engine is
reimplemented. The Bloch gate shortcut menu is hidden so each named
state export keeps its vector, label, and measurement probabilities consistent.

The source HTML is edited directly. The generator only updates demonstration
assets. A fixed PyTorch seed makes the demonstration repeatable with the same
versions; exact floating-point bytes can vary across Python/platform versions.
The manual workflow pins its scientific dependencies; the manifest records the
versions used by each build.

## Recorded demos

The training demo performs 16 seeded, CPU-only SGD steps on a 2 → 4 → 1
PyTorch network during the build. `sd.watch()` records each completed
forward/backward pass with `run.capture()` **before** `optimizer.step()`.
`run.playback()` supplies the original controls and inspection behavior.
Parameter previews retain 9/17 values; the graph records all 12 connections.
Captures are capped at 16 snapshots and 512,000 encoded bytes. Build assertions
check captured outputs, parameter samples and gradients against the live tensors.

`sd.compare(run.snapshots[0], run.snapshots[-1])` renders sampled weight changes,
activation distributions, gradient statistics, and captured output differences.
The final snapshot precedes the final optimizer update. The website does not
implement a comparison engine.

`sd.circuit(qc, simulate=True)` exports a Bell circuit and a Bell + RZ phase
variant. Every gate's amplitudes, probabilities, phases, and reduced-qubit
Bloch vectors are checked against Qiskit's `quantum_info`. These circuits have
no measurement. SeeDL halts evolution for measurement, reset, conditional, and
unsupported operations; unavailable post-halt states are never measurement
outcomes. Phase of a zero amplitude is undefined.

`assets/playback.css` only adapts presentation. The small generation adapter
loads the shared local Plotly runtime before the untouched library export;
it never strips scripts or rewrites controller HTML. Each demo stays below
1 MB, excluding the single shared bundle. Inactive demos load on first use;
changing tabs or hiding the page pauses playback. No automatic playback runs.
“Live” means interaction with recorded snapshots, without a training server.

## Verification

```bash
.venv/bin/python website/check.py
# Uses existing Playwright + Chromium; no browser installation during tests.
.venv/bin/python scripts/verify_website.py
```

The browser check serves the site under `/seedl/`, exercises physical neuron
clicks, incoming/outgoing highlighting, camera drag/zoom/reset, all six Bloch
states, Bell gate inspection, theme persistence, keyboard tabs, navigation,
mobile layout, recorded playback controls, training neuron clicks,
loss/activation/weight/gradient updates, circuit gate selection, Bell/phase
values, sampled comparisons, failed requests, and browser errors. It writes
desktop, mobile, and demo screenshots
to `artifacts/website/`. Use `SEEDL_BROWSER=/path/to/chrome` if Playwright's
bundled Chromium is unavailable.

Plots use a dark scientific palette in both site themes. On narrow screens,
plots can scroll horizontally to preserve readable Plotly controls; surrounding
navigation and demo controls remain responsive. A WebGL-capable browser and
JavaScript are required for 3D demos. Homepage previews and documentation are
available without JavaScript. All browser interactions use precomputed data.

## Deploy — only when authorized

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
