# SeeDL visual redesign checklist

## Baseline
- [x] Inspect clean Git status and recent commits (starting at 38189f3).
- [x] Read public API, view/export contracts, renderers, model/activation inspection.
- [x] Inspect existing visualization families, README, showcase notebook and CI/publishing.
- [x] Baseline: 118 passed, 1 accelerator-only skip.

## Implementation
- [x] Bounded, deterministic neuron graph from exact Linear / Sequential structure.
- [x] Actual weight indexing, biases, parameter sharing, trainability and gradient availability.
- [x] One safe example-input pass, selected-sample activations, no automatic backward.
- [x] Layered 2D Plotly, SVG fallback, Matplotlib compatibility, light/dark themes.
- [x] Real Scatter3d neurons and connections, camera, hover, legend, standalone HTML.
- [x] Accurate forward-layer exploration controls with captured values.
- [x] Architecture blocks / hierarchy for CNNs, complex and large networks; no inferred skip edges.
- [x] Regression, mathematical mapping, bounds, missing-dependency and export tests.
- [x] Execute visual showcase notebook and inspect browser output and controls.
- [x] Save real screenshots and update README with implemented contracts and limitations.
- [x] Implement and visually inspect secondary 3D embeddings and bounded coordinate loss slices after neuron acceptance.
- [x] Run full tests, Ruff lint/format, mypy, lock check, package build, diff check.

Publishing workflows, version metadata, tags and package publication stay untouched.

Final result: 148 passed, 1 existing accelerator skip. See VISUAL_ACCEPTANCE.md for commands, screenshots and limits.
