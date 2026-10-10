# SeeDL

SeeDL turns a PyTorch model into a visual network you can inspect.

```python
import seedl as sd

sd.see(model)
```

For a supported small MLP, this draws actual neurons and connections from the model's weights.
Rotate the network in three dimensions with:

```python
sd.see(model, view="3d")
```

![Actual 2D neuron diagram with captured activations](docs/visuals/network-2d.jpg)
![Actual interactive 3D neuron diagram](docs/visuals/network-3d.jpg)

These screenshots show an initialized 4 → 8 → 6 → 2 PyTorch model, not an artist's mockup.
See the [executed visual showcase](examples/neural_network_explorer.ipynb),
[visual acceptance record](docs/VISUAL_ACCEPTANCE.md), and
[reproduction script](scripts/build_visual_showcase.py).

The zero-dependency core still supports SVG. Plotly, PyTorch, NumPy and Matplotlib remain
optional extras. Existing tensor, training, prediction, gradient, activation, attention,
explainability and educational APIs remain available.

## Explore a network

```python
import torch
from torch import nn
import seedl as sd

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 6),
    nn.ReLU(),
    nn.Linear(6, 2),
)
x = torch.tensor([[0.3, 0.8, -0.2, 1.0]])

sd.see(model)  # Neurons and actual weighted edges
sd.see(model, select_neuron=(1, 2))  # Highlight incoming (cyan) & outgoing (amber) connections
sd.see(model, view="3d")  # Plotly: orbit, zoom, pan
sd.see(model, input=x, theme="dark")  # Actual post-activation node values
sd.see(model, input=x).explore()  # Animated forward pass (Play/Pause/Reset) and layer-step slider
sd.see(model, view="architecture")  # Operation blocks or module hierarchy
sd.see(model, view="summary")  # Existing detailed inspection tables

view = sd.see(model, view="3d", input=x, select_neuron=(1, 2))
view.save("network.html")  # Self-contained Plotly JS; opens offline
```

In notebooks, return the view as the last expression in a cell. In a script, call
`sd.see(model).show()` to open Plotly; the initial `see()` call prints a summary in terminals.

### What the picture means

- **Neurons:** Exact built-in `nn.Linear` and nested `nn.Sequential` networks are supported.
  Supported intervening operations are ReLU, Tanh, Sigmoid, GELU, SiLU, LeakyReLU, ELU and
  Identity. Activations annotate the preceding transformation instead of adding neuron layers.
  Custom subclasses, hooks, functional operations inside custom `forward` methods and other
  structures fall back to architecture inspection.
- **Connections:** An edge from input `i` to output `j` reads `weight[j, i]`. Teal means positive,
  coral means negative, gray means exactly zero. Four line-width bins encode absolute weight
  magnitude relative to the displayed edges. Hover includes the exact weight, parameter name,
  trainability and existing weight gradient. Biases appear on neuron hover.
- **Activations:** `input=x` explicitly requests one eval/no-grad forward pass with a cloned
  input. `sample=0` selects a zero-based batch row. Supported activation inputs have shape
  `[features]` or `[batch, features]`. Each node shows the actual output after its activation;
  hover also includes the Linear pre-activation. No input means no activation values.
  Training flags, hooks and ordinary buffers are restored. Custom forward side effects are
  not sandboxed in general architecture inspection.
- **Sharing and gradients:** Reused Linear modules appear at each Sequential occurrence;
  parameter totals remain deduplicated. Missing gradients differ from zero gradients.
  `color_by="gradient"` colors nodes by their existing **bias gradients**, with missing values
  gray; edge colors still represent weights. SeeDL never calls backward automatically.
- **Coordinates:** Layer positions are deterministic layout coordinates, not learned embeddings.
  In 3D each layer occupies a grid in a plane, making depth visible and the network rotatable.

### Model types and bounds

| Model | Automatic representation |
| --- | --- |
| Small supported MLP | Neurons and weighted connections |
| Sequential CNN | Connected operation blocks, channels, kernels, captured spatial dimensions |
| Transformer or custom/residual model | Module hierarchy; connectors mean containment |
| Large MLP | Architecture blocks, with explicit collapse disclosure |
| Other unsupported model | Existing informative inspection fallback |

Registration order does **not** establish custom execution order. SeeDL does not invent residual
or attention wiring, nor represent convolutional feature maps as fully connected neurons.
Architecture views show up to 40 blocks; use `view="summary"` for table inspection and `view.data`
for all rows. Progressive clickable module expansion is not implemented.

Defaults are **100 total neurons**, **48 per layer**, **12 layers**, and **1,500 connections**.
Models exceeding a neuron or layer limit fall back before extracting weight values. Configure
`max_neurons` (1–1,000) and `max_connections` (1–20,000); the per-layer and layer-count limits
remain in force. Edges are sampled evenly across flattened matrices using a deterministic
stride, then `min_weight` filters by absolute magnitude. This is a preview, not a strongest-edge
ranking. The plot discloses displayed versus total connections. Connections are batched into
sign/magnitude/layer traces, rather than creating a Plotly trace per edge.

```python
sd.see(model, max_neurons=100, max_connections=500, min_weight=0.1, edge_opacity=0.45)
sd.see(model, color_by="layer", theme="light")
sd.see(model, input=x, sample=0, color_by="activation")
```

### Interactive and static output

Install `pip install "seedl[torch,plotly]"` for interactive model exploration. Plotly 3D uses
actual `Scatter3d` traces and a WebGL browser. Hover, sign-legend toggles, orbit, zoom and pan
are supported. **Orbit**, **Pan**, and **Reset view** controls sit above the graph. Reset view
restores the designed camera and orbit mode. Dotted outlines identify layer planes, not weights.
Colored node outlines preserve layer identity while the fill shows activation values.
Edge-on camera angles can overlap labels.

`.explore()` provides animated forward propagation (**▶ Play**, **⏸ Pause**, **⏮ Reset**) with
speed controls (**0.5x**, **1x**, **2x**) and a layer-step slider. It visually steps through computational
transformations in execution order, highlighting active transformations and displaying actual captured
tensors without inventing intermediate values. Without a captured input, exploration explicitly indicates
that values are unavailable.

Users can also select and inspect individual neurons via `select_neuron=(layer, idx)` or through the
interactive dropdown menu. Selecting a neuron isolates its incoming connections (cyan) and outgoing
connections (amber), dims unrelated edges, and provides a full inspection panel with module identity,
operation, bias, trainability, activation, pre-activation, and gradient values. There is no external
server requirement, notebook widget dependency, or fake particle physics.

Without Plotly, `sd.see(model)` renders SVG circles, edges, labels and native title tooltips.
Explicit `view="3d"` requires Plotly and raises an installation hint if it is absent. Unsupported
3D model requests use an explicitly disclosed 2D architecture fallback. SVG/Matplotlib export
of a 3D neuron network is labeled as a **2D projection**. Python-side 3D raster export requires
Plotly's optional Kaleido setup. The browser toolbar can download a genuine 3D PNG without
Kaleido; this was tested and [the output inspected](docs/visuals/network-3d-export.png).
HTML is the portable interactive export.

Notebooks must trust the output and support Plotly MIME rendering. GitHub's notebook viewer
may not execute interactive output; use Jupyter or generated HTML. Small panels can require
scrolling. Browser inspection was completed in the Codex in-app browser; other frontends and
GPU/WebGL configurations have not all been tested.

### More spatial views

```python
sd.embeddings(vectors, dimensions=3, labels=names, categories=groups)
```

Three-component PCA requires NumPy and Plotly, at least three vectors and three features.
It fits an evenly sampled preview (default 128 vectors; configurable up to 5,000), reports
per-component explained variance, and handles rank-deficient data explicitly. Labels and
categories refer to the original rows. Coordinates use a disclosed common input scale to avoid
numeric overflow; `view.data["coordinate_scale"]` records it. These coordinates can distort
high-dimensional distances. Existing `dimensions=2` behavior is preserved.

```python
regressor = nn.Linear(1, 1)
features = torch.linspace(-1, 1, 20).reshape(-1, 1)
targets = 2 * features + 0.5
sd.loss_landscape(
    regressor,
    (features, targets),
    loss_fn=nn.functional.mse_loss,
    axes=(("weight", 0), ("bias", 0)),
    radius=1.5,
    steps=15,
    evaluate=True,
)
```

This is a **sampled two-dimensional coordinate slice of parameter space**, rendered with a
Plotly `Surface`. Axis indices address flattened parameter coordinates; all other parameters
stay fixed. `evaluate=True` is required because the call runs repeated forward evaluations on
a deep copy. CPU models up to 250,000 parameters and grids of 3–31 steps per axis are supported
(default 225 grid evaluations, maximum 961). A scalar `loss_fn` is required. Optional
`trajectory=[(axis1_offset, axis2_offset), ...]` evaluates up to 100 extra points in the same
slice; this does not capture an optimizer automatically. Original model parameters, gradients,
buffers, training flags and CPU RNG are preserved. External side effects of arbitrary custom
Python callbacks cannot be sandboxed. The cost also depends on the supplied input and model.

---

## Key Capabilities

- **Multidimensional Tensor Explorer:** Scalars, 1D signals, 2D matrices, image tensors (CHW/HWC), batch grids, channel decomposition, memory footprints, and value distributions.
- **Model Architecture & Shape Tracing:** Clean module hierarchy, parameter counting (trainable vs frozen), shared parameter deduplication, and hook-based shape tracing.
- **Activations & Feature Maps:** Layer outputs, convolutional feature map grids, saturation diagnostics, and heuristic dead unit alerts (>80% inactive zeros in ReLU/GELU).
- **Gradient Flow Observability:** Observational, non-intrusive parameter gradient inspection, L2 norm metrics, and heuristic alerts for vanishing (<1e-7), exploding (>100), non-finite (NaN/Inf), or all-zero layers without modifying model parameters.
- **Training Observability & Tracking:** Exponential moving average (EMA) smoothing, multi-metric grouping, divergence/plateau/overfitting alerts, lightweight `Tracker` class, and run comparison (`sd.compare`).
- **Prediction Analysis:** Top-k classification ranking, logit vs normalized probability detection, regression residuals ($y_{pred} - y_{true}$), MAE/MSE/RMSE metrics, and multi-model comparison (`sd.compare_predictions`).
- **Dataset & DataLoader Inspection:** Bounded inspection of PyTorch `Dataset` and `DataLoader` batches, image tile grids, tabular column distributions, and class frequencies without exhausting streams or downloading full datasets.
- **CNNs & Transformers:** 2D discrete cross-correlation simulations (following deep learning Conv2d conventions for kernel, stride, padding, and receptive field) and multi-head attention heatmaps (self- and cross-attention) with query-key token labels and causal mask detection.
- **Vector Embeddings:** Pairwise cosine similarity matrix and analytical 2D PCA projections with explained variance ratio.
- **Explainability & Attribution:** Feature attribution using Vanilla Saliency, Integrated Gradients (with interpolation steps), and Grad-CAM for convolutional networks, with strict state and gradient preservation.
- **Educational Concept Simulations:** Interactive visual simulations of core deep learning concepts via `sd.learn(...)` (Gradient Descent, Backpropagation chain rule, Attention, Activation functions, Loss functions, Dropout, Batch Normalization, and Convolution).
- **Flexible Export & Multi-Backend:** Built-in SVG fallback (zero dependencies), interactive Plotly figures, static Matplotlib figures, `.save("path.html" | ".svg" | ".png")`, and interactive switching via `.explore()`.

---

## Installation

Python 3.10 or newer is supported. The core package operates purely on the Python standard library:

```bash
pip install seedl
```

Install optional dependencies on demand:

```bash
pip install "seedl[torch]"       # PyTorch models, tensors, activations, explainability
pip install "seedl[numpy]"       # NumPy arrays and fast SVD
pip install "seedl[plotly]"      # Interactive web visualizations (recommended)
pip install "seedl[matplotlib]"  # Static plotting and PNG exports
pip install "seedl[all]"         # All capabilities and backends
```

Optional dependencies are imported strictly on demand. Importing `seedl` does not eagerly load PyTorch, NumPy, Plotly, or Matplotlib.

---

## Quick Start

```python
import seedl as sd

# 1. Visualize a scalar with custom unit / note
sd.see(0.0035)

# 2. Visualize a numeric sequence or loss signal
sd.see([1.2, 2.5, 3.1, 2.8, 4.0, 5.2, 4.9])

# 3. Visualize a 2D matrix or correlation table
matrix = [
    [1.0, 0.5, 0.2],
    [0.5, 2.0, 0.8],
    [0.2, 0.8, 1.5],
]
view = sd.see(matrix)

# 4. Export visualization directly to HTML, SVG, or PNG
view.save("matrix_view.html")
view.save("matrix_view.svg")
```

In a Jupyter/Colab notebook, visualizations render as rich interactive cell outputs. In a terminal, SeeDL displays a concise, informative text summary.

---

## Core Workflows

### 1. PyTorch Model Inspection & Shape Tracing

```python
import torch
import torch.nn as nn
import seedl as sd

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((4, 4)),
    nn.Flatten(),
    nn.Linear(16 * 4 * 4, 10),
)

# Hierarchy and parameter counts:
sd.see(model)

# Trace tensor shapes across layers with an example input:
example = torch.randn(2, 3, 32, 32)
sd.see(model, input=example)
```

### 2. Activations and Convolution Feature Maps

Capture layer activations from an input-driven forward pass:

```python
act_view = sd.activations(model, input=example)
# act_view includes intermediate stats, feature map grid previews, and dead-unit diagnostics
```

### 3. Gradient Flow Diagnostics

Inspect gradients that already exist on a model after `loss.backward()`:

```python
output = model(example).sum()
output.backward()

grad_view = sd.gradients(model)
# Warns automatically about vanishing gradients (<1e-7), exploding gradients (>100), or all-zero layers.
```

### 4. Training Observability and Experiment Tracking

Track metrics during training with the lightweight `Tracker`:

```python
tracker = sd.watch(name="baseline_run")

for epoch in range(1, 11):
    # ... training step ...
    tracker.log(epoch=epoch, loss=train_loss, val_loss=val_loss, accuracy=acc)

# Visualize training curves with exponential moving average (EMA) smoothing:
sd.see(tracker)

# Compare two runs side by side:
sd.compare(run_a_history, run_b_history, names=("SGD", "AdamW"))
```

### 5. Prediction Inspection & Error Residuals

```python
# Multiclass classification: Top-k ranking and probability normalization:
logits = [2.4, 0.2, -1.5, 0.9]
labels = ["cat", "dog", "bird", "fish"]
sd.predictions(logits, labels=labels, task="classification")

# Regression: Residual error analysis, MAE, MSE, RMSE:
sd.predictions(y_pred, targets=y_true, task="regression")

# Model prediction agreement / disagreement comparison:
sd.compare_predictions(preds_model_a, preds_model_b, labels=labels)
```

### 6. Dataset & DataLoader Batch Inspection

Safely preview a dataset or DataLoader batch without exhausting generators:

```python
from torch.utils.data import DataLoader

loader = DataLoader(my_dataset, batch_size=16)
sd.dataset(loader)  # Automatically shows tiled image grid or tabular feature summaries
```

### 7. Feature Attribution & Explainability

Inspect which input features influenced a model's prediction:

```python
# Saliency maps (input gradient sensitivity):
sd.explain(model, input=sample, method="saliency")

# Integrated Gradients (path-integrated attribution satisfying completeness via composite midpoint rule):
sd.explain(model, input=sample, method="integrated_gradients", steps=25)

# Grad-CAM (coarse spatial activation mapping for convnets):
sd.explain(conv_model, input=sample_image, method="grad_cam")
```

*Guaranteed state safety:* `sd.explain` preserves original model training flags, temporary hooks, and pre-existing parameter gradients.

### 8. Convolution & Transformer Attention

```python
# Exact 2D discrete cross-correlation simulation with arithmetic:
sd.convolution(image=my_image, kernel=sobel_kernel, stride=1, padding=1)

# Multi-head attention heatmaps with query/key token alignment:
sd.attention(attention_matrix, tokens=["The", "model", "learns", "fast"])
```

### 9. Vector Embeddings & PCA

```python
# Pairwise cosine similarity matrix and analytical 2D PCA projection:
sd.embeddings(embedding_vectors, labels=["doc_1", "doc_2", "query_1"])
```

### 10. Educational Concept Simulations (`sd.learn`)

Explore fundamental deep learning mechanisms interactively:

```python
sd.learn("gradient_descent", lr=0.15, steps=12)
sd.learn("backpropagation")
sd.learn("attention")
sd.learn("activation_functions")
sd.learn("loss_functions")
sd.learn("dropout", p=0.5)
sd.learn("batch_norm")
sd.learn("convolution")
```

---

## Visualization Views & Export

Every SeeDL operation returns a `Visualization` instance:

```python
view = sd.see(obj)

# Switch to interactive Plotly exploration mode:
fig = view.explore()

# Export to standalone files:
view.save("report.html")  # Self-contained HTML with embedded graphics
view.save("figure.svg")  # Clean vector SVG
view.save("figure.png")  # Raster PNG via Matplotlib
```

---

## Development & Verification

Run the test suite and quality checks:

```bash
uv sync --extra dev --extra all
uv run --extra dev --extra all pytest
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev mypy
uv build --no-sources
```

---

## Documentation & Releases

- [Release Notes (v0.1.0)](docs/RELEASE_NOTES_0.1.0.md): Complete release summary of verified capabilities and limitations.
- [Publishing Guide](docs/PUBLISHING.md): Instructions for building, verifying, and publishing to PyPI via OIDC Trusted Publishing.

---

## License

Released under the [MIT License](LICENSE).
