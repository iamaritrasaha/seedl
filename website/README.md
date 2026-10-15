# SeeDL official website

Static HTML/CSS/JavaScript for SeeDL **0.2.0**. No JavaScript build step, CDN,
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

Open http://127.0.0.1:8000/. The build generates eight HTML demonstrations,
two static SVG previews, a copied existing neural-network image, provenance
JSON with SHA-256 hashes, and one shared local Plotly bundle. Generated assets
are ignored by Git and rebuilt before deployment. The generator uses SeeDL's
own Plotly figures and its existing neuron interaction bridge; no scientific
engine is reimplemented. The Bloch gate shortcut menu is hidden so each named
state export keeps its vector, label, and measurement probabilities consistent.

The source HTML is edited directly. The generator only updates demonstration
assets. A fixed PyTorch seed makes the demonstration repeatable with the same
versions; exact floating-point bytes can vary across Python/platform versions.
The manual workflow pins its scientific dependencies; the manifest records the
versions used by each build.

## Verification

```bash
.venv/bin/python website/check.py
# Uses existing Playwright + Chromium; no browser installation during tests.
.venv/bin/python scripts/verify_website.py
```

The browser check serves the site under `/seedl/`, exercises physical neuron
clicks, incoming/outgoing highlighting, camera drag/zoom/reset, all six Bloch
states, Bell gate inspection, theme persistence, keyboard tabs, navigation,
mobile layout, failed requests, and browser errors. It writes two screenshots
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
