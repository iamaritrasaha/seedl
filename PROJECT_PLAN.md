# SeeDL Engineering Evolution Plan

## 1. Vision & Architecture

SeeDL evolves from its initial 0.1.0 MVP into a comprehensive, scientific deep learning visualization and observability framework. The core philosophy remains: **"If an operation, object, process, or behavior can be understood better by seeing it, SeeDL should make that visualization effortless."**

The primary user interface remains:
```python
import seedl as sd

sd.see(obj)
```
with specialized, high-leverage entry points:
- `sd.activations(model, input=...)`
- `sd.gradients(model)`
- `sd.compare(a, b)`
- `sd.explain(model, input=..., method="saliency"|"integrated_gradients"|"grad_cam")`
- `sd.predictions(preds, targets=..., labels=...)`
- `sd.dataset(dataset_or_batch, ...)`
- `sd.learn(concept, ...)`
- `sd.watch(model)` / `sd.track(model)`

And view object capabilities:
- `view.save("output.html" | "output.svg" | "output.png")`
- `view.explore()`

### Design Principles
1. **Zero-mandatory third-party runtime dependencies**: Core runs on standard library Python; PyTorch, NumPy, Plotly, and Matplotlib load on demand.
2. **Scientific rigor & safety**: No silent state mutation, exact vs approximate disclosures, bounded sampling, chunked statistics, strict exception-safe hook cleanup, and gradient/RNG state preservation.
3. **Multi-backend consistency**: Seamless support across built-in SVG (lightweight fallback/static), Plotly (interactive exploration), and Matplotlib (publication figures).

---

## 2. Phase-by-Phase Roadmap & Prioritized Checklist

### Phase 1 — Baseline & Architecture Review
- [x] Baseline test verification and environment inspection.
- [ ] Establish modular package structure (`tensors.py`, `models.py`, `activations.py`, `gradients.py`, `training.py`, `predictions.py`, `datasets.py`, `vision.py`, `transformers.py`, `embeddings.py`, `explain.py`, `compare.py`, `educational.py`, `export.py`).
- [ ] Maintain 100% backward compatibility for all existing public functions and view contracts.

### Phase 2 — Multidimensional Tensor Explorer (P0)
- [ ] Multidimensional tensor slicing, index exploration, rank/shape metadata.
- [ ] Enhanced image tensor handling: CHW / HWC auto-detection, batch handling, channel-wise comparisons, value scaling safety.
- [ ] NaN, infinity, and sparsity indicators with accurate aggregate statistics.
- [ ] Bounded chunked statistics and sampling memory limits.

### Phase 3 — Model Architecture & Computation Graphs (P0)
- [ ] Expand `sd.see(model)`: hierarchical module navigation, parameter breakdown (trainable vs frozen, shared parameters).
- [ ] Input-driven shape tracing: input/output dimensions for nested, sequential, residual, and branching modules.
- [ ] Residual and branch connection detection heuristics with clear educational indicators.
- [ ] Safety preservation: training/eval modes, buffer recovery, hook cleanup.

### Phase 4 — Activations & Gradient Tools (P0)
- [ ] `sd.activations(model, input=...)`: layer filtering (`layers="all"|list`), feature map preview grids, saturation and dead-neuron diagnostic metrics.
- [ ] `sd.gradients(model)`: distinction between no-gradient, zero-gradient, and non-finite; vanishing and exploding gradient heuristics with explicit evidence-based caveats.
- [ ] Educational backpropagation view showing forward -> loss -> backward dataflow.

### Phase 5 — Training & Experiment Observability (P0/P1)
- [ ] Multi-metric training dashboard with epoch/step curves, smoothing toggles, and missing-data handling.
- [ ] Diagnostic indicators: overfitting, underfitting, diverging loss, learning-rate sensitivity, stalled training.
- [ ] Lightweight training trackers: `sd.watch(model)` and `with sd.track(model) as tracker:`.
- [ ] Experiment comparison: `sd.compare(history_a, history_b)`.

### Phase 6 — Predictions & Dataset Visualization (P1)
- [ ] `sd.predictions(preds, labels=..., targets=...)`: classification logits vs calibrated/normalized probabilities, top-k ranking, regression residual plots.
- [ ] `sd.compare_predictions(a, b, ...)`: side-by-side comparison of model outputs.
- [ ] `sd.dataset(data, labels=...)`: bounded inspection of DataLoader batches, PyTorch datasets, NumPy arrays, class balance, and feature distributions.

### Phase 7 — Computer Vision & Transformers (P1/P2)
- [ ] CNN inspection: interactive convolution demonstrations (kernel, stride, padding, input-output arithmetic), learned filter grids.
- [ ] Transformer inspection: multi-head attention weight heatmaps (query-key coordinates), sequence length / mask exploration.
- [ ] Embeddings: vector similarity matrices, cosine distances, PCA projections.

### Phase 8 — Explainability & Interpretability (P2)
- [ ] `sd.explain(model, input=..., method=...)`: Saliency maps, Integrated Gradients, Grad-CAM with analytical verification, gradient cleanup, and clear causality limitations.

### Phase 9 — Educational Subsystem (P1)
- [ ] `sd.learn(concept)`: interactive, mathematically verified simulations of core concepts:
  - `gradient_descent`
  - `backpropagation`
  - `attention`
  - `convolution`
  - `activation_functions`
  - `loss_functions`
  - `dropout`
  - `batch_norm`

### Phase 10 — Backends, Export & Usability (P0/P1)
- [ ] `view.save(path)` supporting `.html`, `.svg`, `.png` with clear capability checks.
- [ ] `view.explore()` for launching interactive views.
- [ ] Unified styling and high-resolution rendering across SVG, Plotly, and Matplotlib.

### Phase 11 — Validation, Testing & Documentation (P0)
- [ ] Comprehensive unit and integration test suite covering all new capabilities.
- [ ] Numerical edge-case testing: NaN/inf, extreme floats, non-contiguous arrays, zero sizes.
- [ ] Runnable examples in `examples/`.
- [ ] Updated `README.md` and API documentation.
- [ ] Packaging validation: `uv lock --check`, ruff, mypy, `uv build`.
