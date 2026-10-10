"""Build and execute the visual acceptance notebook (optional authoring dependencies).

Run from the repository root:
uv run --extra all --with nbformat --with nbclient --with nbconvert --with ipykernel \
    python scripts/build_visual_showcase.py
"""

import copy
from pathlib import Path

import nbformat
import plotly.graph_objects as go
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs" / "visuals"
DEST.mkdir(parents=True, exist_ok=True)
cells = []


def markdown(text):
    cells.append(nbformat.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbformat.v4.new_code_cell(text))


markdown("""# SeeDL · Neural network explorer

Real neurons, real weights, real activations. This executed notebook demonstrates the visual
redesign on deterministic, randomly initialized toy models (these are not trained models).

## Setup
Install `seedl[all]` and a Jupyter kernel. Run every cell in order. No data is downloaded.
Each example exports a self-contained interactive HTML file to `docs/visuals`.
The 3D views require a WebGL-capable browser and trusted notebook output.
""")
code("""from pathlib import Path

import plotly.io as pio
import torch
from torch import nn

import seedl as sd

pio.renderers.default = "plotly_mimetype"
torch.manual_seed(7)
root = Path.cwd()
if root.name == "examples":
    root = root.parent
out = root / "docs" / "visuals"
out.mkdir(parents=True, exist_ok=True)
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 2))
x = torch.tensor([[0.3, 0.8, -0.2, 1.0], [-0.5, 0.1, 0.7, 0.2]])
print("SeeDL", sd.__version__, "| PyTorch", torch.__version__)""")
markdown("""## 1. Recognize the network
Four input neurons, eight and six hidden neurons, two output neurons. ReLU belongs to its
transformation, so there are **20 nodes and 92 weighted connections**, not six neuron layers.
Colors identify layers when no input is supplied. Hover on a connection to read its real weight.
""")
code("""small = sd.see(model)
assert len(small.data["neuron_graph"]["nodes"]) == 20
assert len(small.data["neuron_graph"]["edges"]) == 92
small.save(out / "small-light.html")
small.save(out / "small-light.svg")
small""")
markdown("""## 2. A medium MLP with bounded connections
This 80-neuron model has 1,536 connections. The preview budget samples at most 250 edges,
evenly across matrix indices. The disclosure stays in the chart; unshown edges still exist.
""")
code("""medium_model = nn.Sequential(nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 16))
medium = sd.see(medium_model, max_connections=250)
medium.save(out / "medium.html")
medium""")
markdown("""## 3. Filter by real weight magnitude
Teal connections are positive and coral connections are negative. Width uses four magnitude
bins within the displayed weights. Filtering makes a clearer picture, but hides weak connections.
""")
code("""weighted = sd.see(model, min_weight=0.25, edge_opacity=0.6)
weighted.save(out / "weighted.html")
weighted""")
markdown("""## 4. Observe one real sample
Node colors now encode actual outputs from sample 0 of the supplied batch. Hidden nodes show
the output **after ReLU**; hover exposes the pre-activation and bias too. Zero is a real value;
missing or non-finite values are not replaced by zero. One eval/no-grad pass uses cloned input.
""")
code("""active = sd.see(model, input=x, sample=0, theme="dark")
expected = model(x)[0].detach().tolist()
actual = [node["activation"] for node in active.data["neuron_graph"]["layers"][-1]["nodes"]]
assert actual == expected
active.save(out / "neurons.html")
active.save(out / "neurons.svg")
active""")
markdown("""## 5. Rotate a real 3D network
Neurons occupy deterministic grids within layer planes. Drag to orbit; choose zoom or pan in
the toolbar. **Orbit**, **Pan**, and **Reset view** are also available above the graph.
Dotted outlines identify layer planes; they are not connections. Labels
can overlap at edge-on camera angles. Spatial positions are layout coordinates, not learned values.
""")
code("""spatial = sd.see(model, input=x, view="3d", theme="dark")
assert all(trace.type == "scatter3d" for trace in spatial.figure.data)
spatial.save(out / "3d.html")
spatial.save(out / "3d.svg")
spatial""")
markdown("""### Compare light mode and a denser 3D view
The same camera and layer guides work in both themes. The medium model still uses the edge
budget; all 80 neurons are present while the caption discloses sampled connections.
""")
code("""spatial_light = sd.see(model, input=x, view="3d", theme="light")
spatial_light.save(out / "3d-light.html")
spatial_light""")
code("""medium_spatial = sd.see(medium_model, view="3d", max_connections=250)
medium_spatial.save(out / "medium-3d.html")
medium_spatial""")
markdown("""## 6. Follow the forward pass
The slider highlights a transformation and its input/output nodes. Select layers in either
direction; Reset restores the whole network. This revisits captured values, with no additional
model calls. The readout shows actual input, pre-activation and output vectors (first six values
for wide layers; hover for every neuron). Values keep a fixed color scale while stepping.
The slider is above the graph. There is no simulated time evolution or clickable expansion.
""")
code("""forward = sd.see(model, input=x, theme="dark")
forward_figure = forward.explore()
forward.save(out / "forward.html")
forward_3d = sd.see(model, input=x, view="3d", theme="dark")
forward_3d.explore()
forward_3d.save(out / "forward-3d.html")
forward_sampled = sd.see(model, input=x, max_connections=40)
forward_sampled.explore()
forward_sampled.save(out / "forward-sampled.html")
forward_figure""")
markdown("""## 7. Convolutional data flow
Convolution blocks show channel counts, kernel/stride and captured spatial dimensions.
These are operations and feature-map shapes, not fully connected neuron wiring.
""")
code("""cnn = nn.Sequential(
    nn.Conv2d(3, 8, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((2, 2)),
    nn.Flatten(),
    nn.Linear(32, 3),
)
conv = sd.see(cnn, input=torch.ones(1, 3, 16, 16))
conv.save(out / "cnn.html")
conv.save(out / "cnn.svg")
conv""")
markdown("""## 8. Large models collapse before extracting weights
A 1,024 → 2,048 → 10 model has over two million weights. It becomes a handful of architecture
blocks. No million-edge graph is constructed. `view="summary"` retains detailed inspection.
""")
code("""large_model = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, 10))
large = sd.see(large_model)
assert "neuron_graph" not in large.data
large.save(out / "large.html")
large""")
markdown("""## 9. Transformer hierarchy
Custom/residual structures use containment connectors. Registration order does not prove
execution order; this view does not invent skip connections. Attention, feed-forward and
normalization modules remain visible, with deeper details in `view="summary"`.
""")
code("""transformer = nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=32, batch_first=True)
hierarchy = sd.see(transformer)
hierarchy.save(out / "transformer.html")
hierarchy""")
markdown("""## 10. Three-component PCA
These are explicitly generated example vectors, not real documents. PCA fits the displayed
sample and reports explained variance per axis. Category colors use supplied labels. Coordinates
are scaled by the recorded `coordinate_scale` before PCA to avoid numeric overflow.
""")
code("""vectors = torch.randn(48, 6)
vectors[24:] += torch.tensor([2.0, -1.0, 0.0, 1.0, 0.0, 0.0])
embedded = sd.embeddings(
    vectors,
    dimensions=3,
    labels=[f"Example {i}" for i in range(48)],
    categories=["Group A"] * 24 + ["Group B"] * 24,
)
embedded.save(out / "embeddings-3d.html")
embedded""")
markdown("""## 11. A defined loss slice
For a one-input linear regressor, vary weight[0] and bias[0] around the current model. All other
parameters stay fixed. `evaluate=True` explicitly authorizes repeated evaluation on a deep copy.
This is a sampled two-dimensional coordinate slice of parameter space, not the full loss function.
The supplied path is an illustrative slice trajectory, not a recorded optimizer run.
""")
code("""regressor = nn.Linear(1, 1)
with torch.no_grad():
    regressor.weight.fill_(1.0)
    regressor.bias.fill_(0.0)
features = torch.linspace(-1, 1, 20).reshape(-1, 1)
targets = 2 * features + 0.5
surface = sd.loss_landscape(
    regressor,
    (features, targets),
    loss_fn=nn.functional.mse_loss,
    axes=(("weight", 0), ("bias", 0)),
    radius=1.5,
    steps=15,
    evaluate=True,
    trajectory=[(0, 0), (0.5, 0.25), (1, 0.5)],
)
surface.save(out / "loss-slice.html")
surface""")
markdown("""## Checks and limits

- Neuron detail: exact built-in Linear/Sequential plus supported elementwise activations; no custom hooks.
- Default limits: 100 total neurons, 48 per layer, 12 layers, 1,500 sampled connections.
- Higher-rank MLP activations require selecting a `[features]` or `[batch, features]` input first.
- SVG remains usable without Plotly; a 3D neuron SVG is explicitly labeled as a 2D projection.
- Complex execution graphs, clickable expansion, animated propagation and automatic optimizer-path capture are not implemented.
- All other SeeDL entry points remain available; this notebook focuses on the visual redesign.
""")
notebook = nbformat.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
notebook.metadata.language_info = {"name": "python", "version": "3"}
nbformat.validate(notebook)
NotebookClient(
    notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
).execute()
path = ROOT / "examples" / "neural_network_explorer.ipynb"
nbformat.write(notebook, path)
# nbconvert does not natively render Plotly MIME; give the HTML preview real figures.

preview = copy.deepcopy(notebook)
first_plot = True
for cell in preview.cells:
    for output in cell.get("outputs", []):
        payload = output.get("data", {})
        if "application/vnd.plotly.v1+json" in payload:
            figure = go.Figure(payload["application/vnd.plotly.v1+json"])
            output["data"] = {
                "text/html": figure.to_html(
                    full_html=False,
                    include_plotlyjs=first_plot,
                    config={"responsive": True, "scrollZoom": True},
                )
            }
            first_plot = False
body, _ = HTMLExporter().from_notebook_node(preview)
body = body.replace(
    "</head>", "<style>pre{white-space:pre-wrap!important;overflow-wrap:anywhere}</style></head>"
)
(DEST / "showcase.html").write_text(body, encoding="utf-8")
print(f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} cells: {path}")
