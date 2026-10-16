# CI repair and visualization polish: 10 October 2026

## CI findings and fixes

The inspected [failed run](https://github.com/iamaritrasaha/seedl/actions/runs/38048142288)
used commit `5a71eef1fa3dc835d0cc2cd8ffd7247e0c1f2f49`. The two failing jobs rejected
`main` as a release version: `verify_release.py` implicitly defaulted `--tag` to
`GITHUB_REF_NAME`. Its other metadata checks had already passed.

- `--tag` now defaults to `None`, independent of the environment. Metadata verification
  still runs, and `--check-dist` still requires both archives and checks wheel contents.
- An explicit tag is always checked, including an explicitly empty string. `v0.1.0` and
  `refs/tags/v0.1.0` pass; mismatches and branch names fail. Removed the unused `os` import.
- Both explicit `--tag` checks remain in the publishing workflow. Release trigger,
  repository restriction, PyPI environment, OIDC permissions and publishing job are unchanged.
- A second issue was confirmed in the failed run's logs: the job labeled Python 3.10
  actually ran CPython 3.13.16. `.python-version` overrode setup-python's interpreter.
  Each CI job now sets `UV_PYTHON`; the test job uses its matrix version. The publishing
  build also explicitly selects its advertised Python 3.10.
- Added conditional `tomli` to the development extra for Python 3.10. The package-validation
  job explicitly supplies `tomli`, since it does not install development extras. Core runtime
  dependencies remain empty. The verifier is included in the sdist alongside its tests.

The interpreter choice follows [uv's Python version configuration](https://docs.astral.sh/uv/concepts/python-versions/).
No release version was changed. Nothing was committed, pushed, published, tagged or released.
The existing failed remote run is not reported as fixed remotely: a new run must be checked
at the pushed commit after the user authorizes a push.

## Regression coverage and local verification

22 CLI cases cover absent tags with and without branch environment variables, explicit valid
and invalid tags (also with distribution checking), empty tags, metadata mismatches, missing
wheel/sdist files and invalid wheel metadata. Four visualization tests cover exact captured
readouts, missing-input disclosure, bounded vector previews, real 3D guides, camera reset
arguments, edge grouping/counts and slider behavior without changing numeric colors/values.

The full suite passed on **Python 3.10.22, 3.11.17, 3.12.3 and 3.13.16**: **174 passed,
1 skipped on each version**. The skip requires an available CUDA/ROCm accelerator.
Existing tests were retained.

Passed local checks:

```text
uv run --extra dev --extra all pytest
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev mypy
uv lock --check
uv build --no-sources
git diff --check
```

Additional verification:

- Actual `UV_PYTHON`-selected interpreter runs, in separate temporary project environments.
- Python 3.10 mypy and branch metadata verification.
- Python 3.10 wheel and sdist build; strict Twine checks on both artifacts.
- Distribution verification with `GITHUB_REF_NAME=main` and no tag, and with explicit `v0.1.0`.
- Clean-room wheel and sdist installations on Python 3.10: correct site-packages import,
  core SVG rendering, no torch/numpy/plotly/matplotlib imported or installed.
- Showcase notebook executed top-to-bottom: 14 code cells, no error outputs.

## Visual changes and observed results

- Small 2D diagrams now have subtle layer bands and visible neuron indices. Colored outlines
  preserve layer identity when the node fill shows activation or bias-gradient values.
- Increased connection width and default opacity from 0.32 to 0.45. Explicit `edge_opacity`
  is still honored; positive, negative and zero connections keep their real weight mapping.
- 3D uses a more oblique initial camera and wider depth aspect, separating neurons that
  overlapped in the old camera. Dotted plane outlines clarify layers and are explicitly
  identified as layout guides, not learned connections. All traces remain Scatter3d.
- Added visible Orbit, Pan and Reset view controls. Reset restores both the designed camera
  and orbit mode. The sign legend says “Connections · click to filter” and has stable ordering.
- Hover text distinguishes input features from learned neurons, identifies the captured sample,
  and avoids misleading input bias/trainability fields. Bias-gradient coloring is named precisely.
- Forward controls sit above the graph. Selected transformations retain strong edges and
  adjacent nodes while other layers remain visible as context. An immediate readout shows
  captured input, pre-activation and output values, first six entries for wider layers with
  an explicit truncation notice. Hover retains every node's available value. Reset and backward
  selection work without another model call, interpolation, or changing the numeric color scale.
- Missing input produces an explicit unavailable-values explanation instead of pretend activity.
- Existing neuron/edge budgets and architecture fallbacks are unchanged. A medium example
  shows all 80 neurons and 220 of 1,536 connections. A 3,082-neuron MLP collapses to three blocks.

## Browser evidence

The real Codex in-app browser was used at desktop panel widths of approximately 805–1265 px.
Rendered images were inspected, not inferred from successful Figure construction.

| Artifact | What was inspected |
| --- | --- |
| [2D light](visuals/network-2d-light.jpg) | Layer colors, indices, spacing, positive/negative connections |
| [2D activations](visuals/network-2d.jpg) | Real sample-0 outputs, fixed symmetric color scale, layer outlines |
| [3D dark](visuals/network-3d.jpg), [3D light](visuals/network-3d-light.jpg) | Initial camera, separated neuron grids, layer guides, contrast |
| [Rotated 3D](visuals/network-3d-orbit.jpg) | Orbit visibly changes the object; real connection hover |
| [Legend filtering](visuals/legend-filter.jpg) | Positive edges hidden, negative edges retained |
| [Forward step](visuals/forward-step.jpg), [3D step](visuals/forward-3d-step.jpg) | Hidden 1 highlighting, actual vectors, clearly positioned slider |
| [Sampled forward view](visuals/forward-sampled.jpg) | Sampling disclosure and slider have separate space |
| [Medium 2D](visuals/medium.jpg), [medium 3D](visuals/medium-3d.jpg) | 80 nodes and bounded sampled edges, disclosure retained |
| [Filtered weights](visuals/weighted.jpg) | 34 of 92 true edges, threshold disclosure |
| [Large model](visuals/large.jpg) | Architecture collapse rather than millions of lines |
| [SVG](visuals/small-light.svg), [browser screenshot](visuals/svg-fallback.jpg) | Static circles, indices and actual weighted edges remain usable |
| [Browser-exported PNG](visuals/network-3d-export.png) | Downloaded via Plotly toolbar, opened and visually inspected |

Actually exercised: orbit drag, zoom drag, pan drag, Reset view, weight hover, input-neuron hover,
sign legend hide/show, forward selection, backward-to-Input, Reset, and browser PNG download.
The self-contained HTML exports were the pages used for these interactions. No CDN script is
needed; export tests check embedded Plotly and actual Scatter3d traces. Browser PNG export
works without Python Kaleido; Python-side `save('.png')` still needs its optional setup.

Reproduce the notebook and HTML artifacts:

```bash
uv run --extra all --with nbformat --with nbclient --with nbconvert --with ipykernel \
  python scripts/build_visual_showcase.py
python3 -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/docs/visuals/showcase.html`. Individual pages include `3d.html`,
`forward.html`, `forward-3d.html`, `forward-sampled.html`, `3d-light.html` and `medium-3d.html`.
Generated HTML is intentionally ignored by Git; the executed notebook, SVGs and screenshots
are reviewable, and the builder reproduces the interactive files.

## Limits

- Dense networks still have many crossings. Sign filtering, magnitude filtering and edge budgets
  reduce clutter; sampling is deterministic, not a guarantee that the strongest edges are kept.
- Extreme rotation or zoom can overlap or clip 3D labels. Small panels can require scrolling.
- Forward exploration is discrete captured-layer inspection, not temporal animation. No fabricated
  transitional activations, new backward pass, or clickable module expansion was introduced.
- Complex models retain the existing Sequential flow or explicitly labeled containment fallback;
  exact custom/residual/transformer execution wiring remains outside this patch.
- A Jupyter kernel executed the notebook; interactive controls were tested in the in-app browser,
  not separately in JupyterLab/Colab, on mobile, or across all GPUs/WebGL implementations.
- No performance frame-rate benchmark or Python-side Kaleido export was claimed.

## Modified files

Code/configuration: `.github/workflows/ci.yml`, `.github/workflows/publish.yml`, `pyproject.toml`,
`uv.lock`, `scripts/verify_release.py`, `src/seedl/api.py`, `src/seedl/_network_renderers.py`,
`tests/test_verify_release.py`, `tests/test_neuron_graph.py`.

Documentation/showcase: `README.md`, `scripts/build_visual_showcase.py`,
`examples/neural_network_explorer.ipynb`, `docs/VISUAL_ACCEPTANCE.md`,
`docs/CI_VISUAL_POLISH_CHECKLIST.md`, this report, and the generated SVG/screenshots listed above.
