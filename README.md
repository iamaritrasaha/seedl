# SeeDL

SeeDL turns tensors, neural networks, classical ML estimators, training diagnostics, and quantum states
into readable summaries and self-contained visualizations. Inspect existing Python objects,
then explore them in a notebook or save an offline HTML file.

Python **3.10–3.13** · MIT · **Zero mandatory runtime dependencies**

---

### What is SeeDL?
SeeDL is a lightweight, dependency-free scientific visualization and model diagnostic library for Python. It inspects existing Python objects—NumPy arrays, PyTorch models, scikit-learn estimators, and Qiskit circuits—without modifying them or requiring background web servers.

### Why use it?
- **Zero mandatory dependencies**: Core runs in pure standard-library Python with offline SVG and HTML generation.
- **Scientific integrity**: Transparent data provenance, non-destructive inspection, and explicit disclosures for numerical edge cases (NaN/Inf) and sampling.
- **Offline typography & portability**: Fully self-contained exports bundled with Google Sans and JetBrains Mono fonts; no external CDN calls or tracking.
- **Complete lifecycle**: From interactive ML concept simulations to in-flight training diagnostics, attention rollouts, and multi-metric model evaluations.

---

## Quick FAQ & How-To

### How do I install it?
Install the core library (pure Python, zero dependencies):
```bash
pip install seedl
```
Or install with all optional integrations (scikit-learn, PyTorch, NumPy, Plotly, Matplotlib, Qiskit):
```bash
pip install "seedl[all]"
```
Individual extras:
```bash
pip install "seedl[sklearn]"  # scikit-learn evaluation & interpretation
pip install "seedl[numpy]"    # NumPy arrays & multidimensional tensors
pip install "seedl[torch]"    # PyTorch networks, activations & gradients
pip install "seedl[plotly]"   # interactive 3D WebGL charts
pip install "seedl[quantum]"  # Qiskit quantum circuits & Bloch spheres
```

### How do I inspect a tensor?
```python
import numpy as np
import seedl as sd

tensor = np.arange(24).reshape(2, 3, 4)
view = sd.see(tensor, interactive=True, dim_names=["batch", "row", "col"])
view.show()
```

### How do I evaluate a trained scikit-learn model?
Train your model in your own scikit-learn code, then pass the fitted estimator to SeeDL (`sd.evaluate` never calls `.fit()`):
```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seedl as sd

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

# Comprehensive evaluation dossier (confusion matrix, ROC, PR, metrics)
result = sd.evaluate(model, X_test, y_test, split="test")
result.show()
result.save("evaluation.html")
```

### How do I learn a machine-learning concept visually?
Use `sd.learn()` in Python or `seedl learn` from your terminal:
```python
import seedl as sd

# Explore ordinary least squares regression
view = sd.learn("linear_regression")
view.show()
view.save("linear_regression.html")

# Explore gradient descent optimization steps
gd = sd.learn("gradient_descent", lr=0.15, steps=10)
gd.save("gradient_descent.html")
```
From the command line:
```bash
seedl learn list                         # list all 15 educational lessons
seedl learn linear_regression            # export linear_regression.html
seedl learn kmeans --output clusters.svg # export as standalone SVG
```

### How do I save and share a visualization?
All SeeDL visualizations return a `Visualization` or `Evaluation` object with `.save()`:
```python
view.save("report.html")  # self-contained interactive offline HTML
view.save("figure.svg")  # standalone SVG with embedded typography
```
Generated HTML and SVG files are 100% self-contained—open them in any modern browser or share them without running a server.

---

## Discover

```bash
seedl                                # compact command guide
seedl help see                       # Python signature, documentation, runnable example
seedl help evaluate                  # classical ML evaluation documentation
seedl help learn                     # educational simulation documentation
seedl list                           # capabilities organized by task
seedl doctor                         # environment and optional integrations
seedl examples                       # copyable Python examples
seedl demo                           # write seedl-demo.html with labeled UCI Iris data
seedl learn list                     # browse all interactive ML/DL learning lessons
```

`python -m seedl` provides the same commands. Use `seedl --help` for options,
`--plain` or `NO_COLOR=1` for output without terminal colors. Help and doctor do
not load PyTorch or Qiskit. The CLI describes Python APIs; use Python to inspect
your in-memory objects. Demo and learn write new files and never launch a browser.

---

## Choose a workflow

| Task | Python API | Example |
| --- | --- | --- |
| Evaluate a fitted scikit-learn model | `sd.evaluate(model, X_test, y_test, split="test")` | [Classical ML quickstart](docs/CLASSICAL_ML.md) |
| Learn an ML/DL concept visually | `sd.learn("gradient_descent", lr=0.1)` | [Interactive ML Learning](docs/INTERACTIVE_ML_LEARNING.md) |
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

---

## Go further

- [Interactive ML Learning](docs/INTERACTIVE_ML_LEARNING.md): 15 interactive simulations for classical ML and deep learning.
- [Classical ML](docs/CLASSICAL_ML.md): evaluation, pipelines, importance, clusters, and executable notebooks.
- [Getting started](docs/GETTING_STARTED.md): beginner, intermediate, and advanced paths.
- [Command reference](docs/CLI.md): all commands, output handling, and demo provenance.
- [Typography](docs/TYPOGRAPHY.md): offline Google Sans and JetBrains Mono assets.
- [Extended feature reference and gallery](docs/REFERENCE.md): existing APIs and advanced options.
- [Neural observatory](docs/NEURAL_OBSERVATORY.md), [execution tracing](docs/EXECUTION_TRACING.md), and [scientific reports](docs/SCIENTIFIC_REPORTS.md).
- [Examples](examples/) and [release notes](docs/RELEASE_NOTES_0.7.0.md).

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
