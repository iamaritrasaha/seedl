# Getting started

## Beginner: inspect, understand, save

Install the core with `python -m pip install seedl`. For the unreleased CLI and
presentation changes, install this checkout with `python -m pip install -e .`.
Run `seedl`, then `seedl help see` to discover the Python interface.

```python
import seedl as sd

view = sd.see([[1, 2], [3, 4]], backend="svg")  # synthetic example
view.show()
view.save("result.html")
print(view.to_text(detail="full"))
```

The SVG core is dependency-free. HTML files are self-contained and work offline.
`backend="auto"` (the default) uses Plotly if installed; use `"svg"` for a
predictable dependency-free output. PNG export uses an optional image backend.
`view.to_text()` does not render a figure or load optional plotting libraries.

Text summaries show the title, description, existing diagnostic observations,
aligned metadata and layer statistics. WARNING marks existing numerical
observations such as NaN/Inf; CAUTION marks heuristic observations such as
saturation. These labels do not add a new diagnostic algorithm. All sampling and
provenance notes remain visible. Missing statistics are unavailable, never zero.
Full detail expands metadata and layer rows; original structured results remain
in `view.data`, `view.metadata`, and `view.notes` without extra rounding.

Scripts print a summary when you call a visualization API. An immediate SVG
`view.show()` suppresses the identical repeat; `print(view.to_text())` always
prints. A changed summary displays again. Notebook kernels suppress implicit
printing: use the returned object as the last expression, or `view.show()`.
Terminal IPython behaves like other terminals. Explicit Plotly/Matplotlib
`.show()` retains those backends' normal display behavior.

`print(sd.guide())` lists tasks; `print(sd.guide("see"))` shows the live Python
signature and documentation. These helpers return strings without displaying
or executing anything themselves.

## Intermediate: tensors and models

Install `seedl[numpy]` for the first example and `seedl[torch]` for the second.
These inputs are synthetic.

```python
import numpy as np
import seedl as sd

tensor = np.arange(24).reshape(2, 3, 4)
view = sd.see(tensor, interactive=True, dim_names=["batch", "row", "column"])
view.save("tensor.html")
```

The explorer shows the original shape, dtype, device, sampled slices and
statistics. Sliders navigate captured frames, not a live Python tensor. Read the
sampling notes before interpreting a preview. Color ranges rescale per slice;
compare the displayed minimum and maximum when navigating. Focus the canvas and
use arrow keys to inspect captured cell values with a keyboard. RGB images use
the disclosed channel scaling instead of the numeric heatmap palette.

```python
import torch
import seedl as sd

torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())
x = torch.ones(2, 4)
view = sd.see(model, input=x, backend="svg")
view.save("model.html")
activations = sd.activations(model, input=x, backend="svg")
activations.save("activations.html")

# Your training code produces gradients; inspection only reads existing gradients.
model(x).sum().backward()
gradients = sd.gradient_health(model, backend="svg")
gradients.save("gradients.html")
```

Each input-driven inspection runs its own single forward pass. Reusing and
exporting the returned view does not rerun the model. Do not invoke a second
inspection when you only need a second export. Input-driven inspection restores
module training modes, hooks, gradients and ordinary buffer updates; arbitrary
custom forward side effects are outside this guarantee. Gradient health and
`sd.gradients(model)` do not call forward or backward.

## Intermediate: attention and quantum

```python
import seedl as sd

weights = [[0.8, 0.2], [0.3, 0.7]]  # synthetic routing probabilities
attention = sd.attention(weights, tokens=["query", "key"], backend="svg")
attention.save("attention.html")
state = sd.bloch("+", backend="svg")  # analytic single-qubit |+> state
state.save("state.html")
```

Attention describes observed routing, not causal importance. Circuit inspection
requires `seedl[quantum]`:

```python
from qiskit import QuantumCircuit
import seedl as sd

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
view = sd.circuit(qc, backend="svg")
view.save("circuit.html")
```

## Advanced: targeted inspection and scientific reports

- Use `slice=`, `max_points=`, `max_frames=` and `dim_names=` for bounded tensor previews.
- Use `layers=` in `sd.activations` to focus on relevant executed modules.
- Explore [execution tracing](EXECUTION_TRACING.md) and [attention rollout](TRANSFORMER_OBSERVATORY.md).
- Record snapshots with `sd.watch` and compare training runs; see [the observatory guide](NEURAL_OBSERVATORY.md).
- Combine existing views with [scientific reports](SCIENTIFIC_REPORTS.md), or preserve analyses using `sd.save_analysis`.
- Find all existing parameters and extended examples in [the reference](REFERENCE.md).

No public advanced options were removed for the simplified entry points.
[Examples](../examples/) include focused scripts and complete notebooks.
