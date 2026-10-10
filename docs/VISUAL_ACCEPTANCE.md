# Visual redesign acceptance record

Follow-up: [CI repair and visual polish report](CI_VISUAL_POLISH_REPORT.md) records the
current 174-test results across Python 3.10–3.13, updated 14-cell notebook, interaction checks
and new screenshots. The original implementation record below is retained for context.

Verified on 10 October 2026 in the local repository, using CPU PyTorch, Python 3.13,
and the Codex in-app browser. No publishing workflow or release version was changed.

## What was implemented

- Default neuron diagrams for exact Linear/Sequential MLPs, including nested containers
  and repeated uses of shared Linear modules.
- Real weight[destination, source] edges; biases, trainability and existing gradients in
  hover details; actual selected-sample pre/post-activation values.
- Plotly 2D and real Scatter3d 3D, light/dark themes, deterministic layer layouts,
  signed/magnitude-weighted edges, SVG fallback and Matplotlib 2D compatibility.
- Captured forward-pass exploration via an operational layer slider with backward selection
  and Reset, plus animated playback (**▶ Play**, **⏸ Pause**, **⏮ Reset**) with speed controls
  (**0.5x**, **1x**, **2x**). No additional forward/backward calls during exploration.
- Interactive neuron selection and inspection: isolate any neuron via `select_neuron=(layer, idx)`
  or the in-plot dropdown menu; incoming connections highlight in cyan, outgoing connections highlight
  in amber, unrelated edges dim, and a dedicated inspection panel displays exact layer identity, operation,
  bias, trainability, activation, pre-activation, and parameter gradients.
- SVG neuron selection support with cyan incoming and amber outgoing connections, outer highlight ring,
  and footer inspection summary.
- Sequential CNN operation blocks with channels/kernel/stride and observed tensor shapes.
  Transformer/custom/residual modules use an explicitly labeled containment hierarchy.
- Bounded edge sampling/filtering; architecture collapse before weight extraction for
  oversized MLPs. Architecture rendering is limited to 40 blocks.
- 3D PCA scatter with category coloring, bounded sampling and variance reporting.
- Bounded 3D loss Surface with explicitly named parameter coordinates, evaluation consent,
  state isolation, and optional user-supplied slice trajectory.
- Offline self-contained Plotly HTML exports. Large generated HTML files are ignored by
  Git and reproducible from the checked-in notebook builder; screenshots, SVGs and the
  executed notebook remain reviewable artifacts.

## Visual and interaction evidence

The notebook was executed top-to-bottom (12 code cells), retained Plotly MIME outputs,
and converted into an HTML preview with embedded Plotly figures. Its title/setup and first
rendered figure were inspected; each visualization family was also opened directly as its
exported HTML/SVG in the real browser.

| Example | Evidence and observation |
| --- | --- |
| Small MLP | [2D screenshot](visuals/network-2d.jpg): 4 → 8 → 6 → 2, 20 visible nodes and 92 true connections |
| SVG fallback | [Screenshot](visuals/svg-fallback.jpg), [SVG](visuals/small-light.svg): circular nodes, signed edges, readable labels |
| Weight filtering | [Screenshot](visuals/weighted.jpg): 34 of 92 edges at minimum magnitude 0.25, explicit disclosure |
| Medium MLP | [Screenshot](visuals/medium.jpg): 80 nodes, 220 of 1,536 edges at budget 250; bounded preview and readable layer labels |
| Activations | [2D screenshot](visuals/network-2d.jpg): real sample 0, symmetric activation colorbar; numerical output checked against PyTorch |
| 3D MLP | [Initial camera](visuals/network-3d.jpg), [rotated view with weight hover](visuals/network-3d-orbit.jpg): real depth, orbit, zoom, pan and saved-camera reset tested |
| Forward exploration | [Screenshot](visuals/forward-step.jpg): Hidden 1 highlights its incoming edges and adjacent nodes; backward-to-Input and Reset verified in the browser |
| CNN | [Screenshot](visuals/cnn.jpg): channel counts, kernel/stride, spatial outputs and Sequential connectors |
| Large network | [Screenshot](visuals/large.jpg): 3,082 neurons collapse into operation blocks; browser opens without rendering millions of edges |
| Transformer | [Screenshot](visuals/transformer.jpg): attention, feed-forward and normalization hierarchy; containment caption |
| PCA | [Screenshot](visuals/embeddings-3d.jpg): 48 example vectors, three PC axes with variance percentages, supplied category legend |
| Loss slice | [Screenshot](visuals/loss-slice.jpg): actual sampled Surface, weight/bias offset axes, loss colorbar and supplied path |

Legend hit targets were clicked to hide positive connections. A connection hover displayed
its real source/destination, weight, parameter identity, trainability and missing-gradient status.
Camera drag and zoom visibly changed the 3D object. Slider changes visibly updated highlights
and the operation text; backward and Reset changed the displayed state correctly.

Visual QA caught and corrected an overlapping legend/subtitle, reversed initial 3D orientation,
unsupported HTML in 3D labels, and medium-network label/legend crowding.

Plotly API references consulted: [Scatter3d](https://plotly.com/python/reference/scatter3d/)
and [3D camera controls](https://plotly.com/python/3d-camera-controls/).

## Verification

Baseline: **118 passed, 1 skipped**. Latest: **181 passed, 1 skipped** (182 collected).
The skip is the existing CUDA/ROCm-only test because this execution has no available accelerator.
All existing tests were retained.

Passed:

```text
uv run --extra dev --extra all pytest
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev mypy
uv lock --check
uv build --no-sources
uv run python scripts/verify_release.py --check-dist
git diff --check
```

Both `seedl-0.1.0-py3-none-any.whl` and `seedl-0.1.0.tar.gz` built successfully and passed
Twine's strict metadata check. An isolated wheel installation with no optional dependencies
passed import and SVG rendering, with torch/numpy/plotly/matplotlib absent from `sys.modules`.

New tests independently check node/edge counts, actual weight indexing and biases, activation
values across batch sizes, shared occurrences, frozen parameters, missing/zero gradients,
state cleanup on failure, deterministic layout/sampling, collapse bounds, Scatter3d depth,
missing Plotly, exports, custom-model fallback, analytical PCA variance and an analytical
quadratic loss slice. Loss evaluation also resets non-persistent buffers and Python counters
for each point and preserves original CPU RNG and existing gradients.

Reproduce notebook and browser artifacts from the repository root:

```bash
uv run --extra all --with nbformat --with nbclient --with nbconvert --with ipykernel \
  python scripts/build_visual_showcase.py
python3 -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/docs/visuals/showcase.html` or individual exported files.

## Limits and deliberately deferred behavior

- Custom `forward` data-flow tracing and exact residual/transformer skip edges are deferred.
  The current hierarchy does not imply that those edges were traced.
- Progressive clickable module expansion and temporal propagation animation are deferred.
  The existing controls are hover, legend toggles and captured-layer stepping.
- Default bounds are 100 neurons, 48 per layer, 12 layers and 1,500 connections. Sampling
  can miss strong connections; filtering occurs after sampling. Spatial layout is not a
  learned embedding of neurons.
- Neuron activation capture accepts one-dimensional feature vectors or two-dimensional
  batches. No automatic batch/token flattening or aggregation of activation values.
- SVG/Matplotlib export of a 3D neuron view is explicitly a 2D projection. PCA/loss spatial
  views require Plotly. Genuine 3D raster export needs optional Kaleido and was not verified.
- GPU execution, Colab/JupyterLab interaction, mobile browsers, and all WebGL implementations
  were not tested. The notebook executed in a real Jupyter kernel; browser controls were
  checked in the in-app browser. Some notebook viewers do not execute Plotly MIME output.
- Rotating to edge-on angles or zooming far in can overlap or clip 3D labels.
- Loss surfaces are CPU-only coordinate slices, at most 250,000 parameters and 961 grid
  evaluations plus up to 100 supplied path points. Input size and forward cost still matter.
  The path is not an automatically captured optimizer trajectory. Arbitrary external side
  effects inside user Python callbacks cannot be sandboxed.

Nothing was published, pushed, tagged or released.
