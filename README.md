# SeeDL

SeeDL is a small Python library for seeing what tensors, models, and training
metrics are doing. Start with one call:

```python
import seedl as sd

sd.see(tensor)
```

SeeDL picks a useful view, labels the important details, and gives a static SVG
fallback when an optional plotting package is not installed. In a notebook the
view renders as the cell output. In a terminal SeeDL prints a concise summary;
the returned view can also be opened or inspected directly.

## Install

Python 3.10 or newer is supported. The core package uses only the Python
standard library:

```bash
pip install seedl
```

Install support for the libraries you use:

```bash
pip install "seedl[torch]"       # PyTorch models and tensors
pip install "seedl[numpy]"       # NumPy arrays
pip install "seedl[plotly]"      # interactive charts (recommended)
pip install "seedl[matplotlib]"  # static plotting backend
pip install "seedl[all]"         # all optional dependencies
```

The optional dependencies are imported only when needed. Importing `seedl`
does not import PyTorch, NumPy, Plotly, or Matplotlib.

## Quick start

```python
import seedl as sd

sd.see(0.73)
sd.see([0.9, 0.62, 0.41, 0.28])
sd.see({"loss": [1.2, 0.8, 0.5], "val_loss": [1.3, 0.9, 0.7]})
```

Each call returns a `Visualization`. In a notebook it is displayed automatically;
in a normal Python process, use `view.show()` to open the selected figure, or
`print(view)` for its text summary.

## Tensors and arrays

```python
import torch
import seedl as sd

weights = torch.randn(12, 8)
view = sd.see(weights)
```

Scalars are shown as values, vectors as line charts, matrices as heatmaps, and
image-like arrays as image previews. The view includes shape, element count,
dtype, device, summary statistics, gradient tracking, and non-finite values.
Large vectors and images are sampled or downsampled for display, with the
sampling noted in the view. The original tensor or array is not modified.

Three-dimensional data is treated as an image only when a leading or trailing
axis has 1, 3, or 4 channels and both other dimensions are at least 16. Float
images in `[0, 1]` are displayed as-is; integer images up to 255 are scaled to
that range. Other image values are scaled for display and the view states that
assumption. Other higher-dimensional tensors are shown as sampled value
distributions; SeeDL does not label these as images.

NumPy arrays are supported when NumPy is installed. Core numeric Python values,
flat sequences, and rectangular nested sequences do not require NumPy.

## Models

```python
import torch.nn as nn
import seedl as sd

model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
sd.see(model)
```

Without an example input, SeeDL shows the module hierarchy and per-layer and
total parameter counts. To record the shapes observed on one input:

```python
import torch

example = torch.randn(4, 8)
sd.see(model, input=example)
```

The input may be a tensor, a tuple of positional inputs, or a dictionary of
keyword inputs. SeeDL runs one `eval()`-mode forward pass under `no_grad()`,
clones tensor inputs, removes its temporary hooks, restores each module's
training flag, and restores registered buffers from temporary copies. This pass
can still execute user-defined Python code in `forward`; use it only when that
behavior is safe. Dynamic control flow is represented only for the supplied
example.

## Training history

```python
history = {
    "loss": [1.2, 0.8, 0.5, 0.3],
    "val_loss": [1.3, 0.9, 0.7, 0.8],
    "accuracy": [0.45, 0.65, 0.78, 0.86],
    "val_accuracy": [0.42, 0.61, 0.72],
}
sd.see(history)
```

Training and validation series are grouped by metric, with epochs on the
horizontal axis. Missing validation curves and unequal metric lengths are
supported. You can provide an `"epoch"` or `"epochs"` sequence; if its length
does not match a metric, that metric uses its index as the horizontal axis.
Non-finite values appear as gaps. Curves are diagnostic clues, not guarantees
about generalization or model quality.

## Activations and gradients

Capture intermediate outputs with one explicit forward pass:

```python
view = sd.activations(model, input=example)
```

SeeDL captures the first tensor output from each module and shows its shape and
sampled values. It removes temporary hooks and restores module flags and
registered buffers after the pass.

Inspect gradients that already exist after your own backward pass:

```python
loss.backward()
sd.gradients(model)
```

This view reports per-parameter gradient summaries. It never runs
`backward()` or an optimizer step. Parameters with no gradient are called out.

## Backends and views

`backend="auto"` uses Plotly when installed and otherwise the built-in SVG
renderer. Choose a backend explicitly when needed:

```python
view = sd.see(weights, backend="plotly")
static_view = sd.see(weights, backend="svg")
matplotlib_view = sd.see(weights, backend="matplotlib")
```

`interactive=True` selects Plotly and gives an install hint if Plotly is
missing. The returned `Visualization` exposes `kind`, `title`, `summary`,
`metadata`, `notes`, `data`, `figure`, `to_figure(backend=...)`, and `show()`.
The built-in SVG view works without a plotting library.

## Limits and troubleshooting

- Complex tensors and arrays are not supported yet. Split real and imaginary
  components before visualizing them.
- An input-dependent model view requires PyTorch and an explicit `input=`.
- Model execution can trigger custom forward-method side effects beyond module
  flags and registered buffers; inspect unfamiliar models without `input=` first.
- Plotly and Matplotlib raise an actionable optional-dependency error if they
  are explicitly selected but unavailable.
- Exact statistics scan the full tensor or array in bounded chunks; on a GPU the
  reductions can synchronize with the device. Rendered samples are capped by
  `max_points` (default 2,048).
- TensorFlow/Keras adapters, attention maps, convolution filters, graph tracing,
  and exported reports are planned, not implemented in this release.

## Development

```bash
uv sync --extra dev --extra numpy --extra plotly --extra matplotlib
uv run --extra dev --extra numpy --extra plotly --extra matplotlib pytest
uv run --extra dev ruff check .
uv run --extra dev mypy src/seedl
uv build
```

The automated tests do not need a GPU, display server, or network. Tests for
optional packages run only when those packages are installed.

## Project status and license

SeeDL is an early, functional MVP. Its stable starting point is `sd.see()` and
the returned `Visualization`; its limitations are listed above. Contributions
that improve correctness, readability, and the beginner's path from a picture
to a useful interpretation are welcome.

Released under the MIT License. See [LICENSE](LICENSE).
