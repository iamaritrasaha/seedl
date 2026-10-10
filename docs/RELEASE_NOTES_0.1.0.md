# SeeDL 0.1.0 Release Notes

**Initial Public MVP Release** — 10 October 2026

SeeDL (`seedl`) is a deep-learning-aware visualization and observability library for Python. It provides automated, diagnostic inspection for tensors, neural network models, activations, gradients, training curves, predictions, and embeddings.

---

## Highlights & Verified Capabilities

### 1. Automatic Visualization Dispatch (`sd.see`)
- Inspects scalars, 1D sequences, 2D matrices, image tensors (CHW/HWC), value distributions, PyTorch modules, and training trackers with a single entry point.
- Zero-dependency core: generates clean, publication-ready vector SVGs with no mandatory third-party dependencies.

### 2. PyTorch Model Architecture & Shape Tracing
- Module hierarchy display with trainable vs. frozen parameter breakdown and shared parameter deduplication.
- Non-intrusive hook-based shape tracing across all layers when provided with an example input (`sd.see(model, input=example)`), cleanly preserving model state and training flags.

### 3. Layer Activations & Feature Maps (`sd.activations`)
- Captures intermediate layer activations during an input-driven forward pass.
- Distribution summaries and heuristic dead-unit detection (>80% inactive zeros in threshold activation layers like ReLU, GELU, and SiLU).

### 4. Gradient Flow Observability (`sd.gradients`)
- Observes pre-existing parameter gradients following `.backward()` without calling backward or altering optimizer states.
- Reports parameter gradient L2 norms and advisory warnings for vanishing (<1e-7), exploding (>100), non-finite (`NaN`/`Inf`), or all-zero layers.

### 5. Training Observability & Run Comparison (`sd.watch`, `sd.compare`)
- Lightweight `Tracker` for logging epoch/step metrics.
- Exponential moving average (EMA) curve smoothing resilient against leading NaNs.
- Advisory plateau, divergence, and overfitting indicators.
- Side-by-side run comparison (`sd.compare(run_a, run_b)`).

### 6. Prediction Analysis & Calibration (`sd.predictions`, `sd.compare_predictions`)
- Multiclass classification top-k ranking and normalized probability distributions.
- Regression residual error analysis ($y_{pred} - y_{true}$) with MAE, MSE, and RMSE metrics.
- Model-to-model prediction agreement and divergence analysis (`sd.compare_predictions`).

### 7. Attention & Computer Vision Tools
- Multi-head attention heatmaps (`sd.attention`) supporting both self-attention and rectangular cross-attention matrices ($T_Q \neq T_K$) with token labels and causal mask detection.
- 2D discrete convolution simulation (`sd.convolution`) with kernel, stride, padding, and receptive field arithmetic (following standard deep learning cross-correlation conventions).

### 8. Vector Embeddings (`sd.embeddings`)
- Pairwise cosine similarity matrix heatmap.
- Analytical 2D PCA projections with explained variance ratio, supporting degenerate or rank-1 embedding sets without coordinate degradation.

### 9. Feature Attribution & Explainability (`sd.explain`)
- **Vanilla Saliency:** Input gradient sensitivity with signed or absolute attributions.
- **Integrated Gradients:** Path-integrated attributions using the uniform composite midpoint integration rule (open Newton–Cotes formula), strictly satisfying the Completeness axiom ($\sum \text{Attr} \approx F(x) - F(x_0)$).
- **Grad-CAM:** Coarse spatial activation mapping for convolutional networks.
- Multi-dimensional attribution arrays automatically reduce to 2D spatial heatmaps for image inputs across all backends.

### 10. Educational Concept Simulations (`sd.learn`)
- Mathematically grounded interactive visual demonstrations of core deep learning concepts:
  - `gradient_descent`, `backpropagation`, `attention`, `activation_functions`, `loss_functions`, `dropout`, `batch_norm`, and `convolution`.

### 11. Multi-Backend Rendering & Export
- Built-in SVG fallback (standard library only).
- Interactive Plotly figures via `.explore()` or `backend="plotly"`.
- Static Matplotlib figures via `backend="matplotlib"`.
- Direct file exports to `.html`, `.svg`, and `.png` via `.save(...)`.
- Native Jupyter notebook rendering via `_repr_mimebundle_` without duplicate cell outputs.

---

## Installation

Install the zero-dependency core:
```bash
pip install seedl
```

Or with optional integrations:
```bash
pip install "seedl[torch]"       # PyTorch models, tensors, activations, explainability
pip install "seedl[plotly]"      # Interactive web visualizations
pip install "seedl[matplotlib]"  # Static plotting and PNG exports
pip install "seedl[all]"         # All optional capabilities
```

With `uv`:
```bash
uv add seedl
uv add "seedl[all]"
```

---

## Known Limitations

- **Hardware Acceleration:** Verified on Linux x86_64 CPU. Dedicated CUDA / ROCm GPU accelerators are handled via PyTorch device preservation, but hardware-accelerator execution was not benchmarked in headless CI runners.
- **Large Dataset Streaming:** DataLoader inspection performs bounded sampling (up to `max_samples`); infinite or non-indexable remote streams should be wrapped with standard sampling before passing to `sd.dataset`.
- **Heuristic Signals:** Vanishing/exploding gradient and dead-unit alerts are advisory heuristic signals, not formal mathematical proofs.
- **Future Capabilities:** 3D volumetric rendering and quantum tensor network visualizations are planned for future major releases and are not part of 0.1.0.
