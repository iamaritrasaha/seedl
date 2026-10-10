# SeeDL

SeeDL is a comprehensive, deep-learning-aware visualization and observability framework for Python.
Start with one simple, effortless call:

```python
import seedl as sd

sd.see(tensor)
sd.see(model)
sd.see(history)
sd.see(predictions)
sd.see(gradients)
```

SeeDL determines the nature of supported objects, chooses a meaningful visualization, displays useful diagnostic information automatically, and provides optional interactive exploration. When optional plotting libraries are absent, SeeDL renders self-contained, publication-ready SVGs with zero required third-party dependencies.

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
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src/seedl
uv build
```

---

## License

Released under the [MIT License](LICENSE).
