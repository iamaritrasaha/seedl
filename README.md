# SeeDL

SeeDL turns tensors, neural networks, classical ML estimators, training diagnostics, and quantum states
into readable summaries and visualizations. Inspect existing Python objects,
then explore them in a notebook or save a self-contained, offline HTML file.

Python **3.10–3.13** · MIT · **Zero mandatory runtime dependencies**

## Install

```bash
python -m pip install seedl
```

The core handles Python numbers, sequences, matrices, and SVG/HTML exports.
Install integrations as needed:

```bash
python -m pip install "seedl[sklearn]"  # scikit-learn (new in this checkout)
python -m pip install "seedl[numpy]"    # NumPy tensors
python -m pip install "seedl[torch]"    # PyTorch models and diagnostics
python -m pip install "seedl[plotly]"   # interactive charts and 3D
python -m pip install "seedl[quantum]"  # Qiskit circuits and quantum arrays
```

For CUDA or ROCm, use the [PyTorch installation instructions](https://pytorch.org/get-started/locally/)
for your platform. SeeDL uses your existing tensor device.

> The CLI and presentation improvements described here are in development.
> To try this checkout, run `python -m pip install -e .` from the repository.
> Published 0.6.0 predates these changes, including the new `sklearn` extra. For the
> classical ML APIs, install this checkout with `python -m pip install -e ".[sklearn]"`.
> The package version is unchanged.

## Discover

```bash
seedl                  # compact command guide
seedl help see         # Python signature, documentation, runnable example
seedl list             # capabilities organized by task
seedl doctor           # environment and optional integrations
seedl examples         # copyable Python examples
seedl demo             # write seedl-demo.html with labeled UCI Iris data
```

`python -m seedl` provides the same commands. Use `seedl --help` for options,
`--plain` or `NO_COLOR=1` for output without terminal colors. Help and doctor do
not load PyTorch or Qiskit. The CLI describes Python APIs; use Python to inspect
your in-memory objects. Demo writes a new file and never launches a browser.

## Visualize in a few lines

```python
import seedl as sd

view = sd.see([[1, 2], [3, 4]], backend="svg")
view.show()
view.save("result.html")
```

The example is synthetic. SVG needs no optional packages. Omit `backend` to use
Plotly when installed, with SVG as the fallback.

In scripts, visualization functions print a compact summary. Calling SVG
`.show()` immediately afterward does not repeat it. In notebooks, put `view` on
the last line or call `view.show()` to display it. Explicit `.show()` with Plotly
or Matplotlib uses that backend's display behavior. `.save()` never opens a browser.

```python
print(sd.guide())  # discover APIs inside Python
print(sd.guide("activations"))
print(view.to_text(detail="full"))
# view.data, view.metadata and view.notes retain the structured results.
```

## Choose a workflow

| Task | Python API | Example |
| --- | --- | --- |
| Evaluate a fitted scikit-learn model | `sd.evaluate(model, X_test, y_test, split="test")` | [Classical ML quickstart](docs/CLASSICAL_ML.md) |
| Inspect a tensor and its slices | `sd.see(tensor, interactive=True)` | [Tensor exploration](docs/GETTING_STARTED.md#intermediate-tensors-and-models) |
| View a network | `sd.see(model, input=batch)` | [Model inspection](examples/02_model_and_training.py) |
| Examine activations | `sd.activations(model, input=batch)` | [Activations and gradients](examples/03_activations_and_gradients.py) |
| Diagnose existing gradients | `sd.gradients(model)`, `sd.gradient_health(model)` | [Gradient health](examples/gradient_health_demo.py) |
| Visualize attention weights | `sd.attention(weights, tokens=tokens)` | [Attention](docs/GETTING_STARTED.md#intermediate-attention-and-quantum) |
| Explore quantum states and circuits | `sd.bloch("+")`, `sd.circuit(qc)` | [Quantum examples](docs/GETTING_STARTED.md#intermediate-attention-and-quantum) |

Inspection preserves sampling disclosures and nonfinite warnings. Model
inspection with `input=` uses one temporary eval/no-grad forward pass; existing
gradient inspection never calls backward. Arbitrary side effects inside custom
model forward functions cannot be sandboxed. Attention weights describe routing,
not causal importance. See the [scientific correctness notes](docs/SCIENTIFIC_CORRECTNESS_0.5.0.md).

## Go further

- [Classical ML](docs/CLASSICAL_ML.md): evaluation, pipelines, importance, clusters, and executable notebooks.
- [Getting started](docs/GETTING_STARTED.md): beginner, intermediate, and advanced paths.
- [Command reference](docs/CLI.md): all commands, output handling, and demo provenance.
- [Typography](docs/TYPOGRAPHY.md): offline Google Sans and JetBrains Mono assets.
- [Extended feature reference and gallery](docs/REFERENCE.md): existing APIs and advanced options.
- [Neural observatory](docs/NEURAL_OBSERVATORY.md), [execution tracing](docs/EXECUTION_TRACING.md), and [scientific reports](docs/SCIENTIFIC_REPORTS.md).
- [Examples](examples/) and [release notes](docs/RELEASE_NOTES_0.6.0.md).

## Development

```bash
python -m pip install -e ".[dev,all]"
ruff check .
ruff format --check .
mypy
pytest
python -m build
python scripts/verify_release.py --check-dist
```

Browser tests use optional Playwright/Chromium. Accelerator tests skip when the
required hardware is unavailable. See [the DX audit and validation](docs/developer-experience/AUDIT.md)
and [validation report](docs/developer-experience/VALIDATION.md) for the scope and results of this overhaul.
