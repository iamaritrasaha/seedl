# SeeDL 0.6.0 — Multidimensional Tensor Explorer & Model Observability

> **Navigate high-dimensional tensors.**  
> **Diagnose activation regimes and gradient health.**  
> **Preserve memory and execution integrity.**

**Status:** Prepared for release; publication pending.  
*(Note for maintainers: When publishing the GitHub Release, update this status line to `**Status:** Published`.)*

---

## 1. Introduction

SeeDL 0.6.0 is a major milestone delivering interactive multidimensional tensor visualization, deep layer-level activation diagnostics, and rigorous memory safety across CPU, CUDA, and ROCm platforms.

Building upon the foundations of SeeDL 0.5.0 ("The Neural Observatory"), version 0.6.0 introduces two major innovations:
1. **Interactive Multidimensional Tensor Explorer**: A zero-dependency, client-side HTML Canvas slice visualizer enabling continuous navigation, dimension swapping, and live statistical telemetry across 3D, 4D, 5D, and higher-order tensors.
2. **Model Observability & Layer-Level Activation Diagnostics**: A comprehensive diagnostic suite that instruments neural network layers to track activation distributions, non-finite occurrences, ReLU inactivity, GELU negative regimes, Sigmoid/Tanh saturation boundaries, and individual invocations of repeated submodules.

As with all SeeDL releases, version 0.6.0 strictly adheres to core scientific software invariants:
- **Zero Mandatory Runtime Dependencies**: The core package runs on standard Python 3.10–3.13 without requiring PyTorch, Plotly, or NumPy unless those specific integrations are exercised.
- **Authentic Data Only**: Visualizations represent authentic tensor activations, gradients, states, and graph topologies — never synthetic demonstrations or mocked outputs.
- **State Preservation Guarantees**: Model forward execution guarantees full preservation of user models, module buffers, training modes, and autograd gradient states with guaranteed cleanup in `try...finally` blocks.
- **Offline by Default**: Interactive HTML documents, Canvas widgets, and SVG graphics require zero external CDN connections or runtime servers.

---

## 2. Release Highlights

- **Interactive Multidimensional Tensor Explorer (`sd.see(tensor, interactive=True)`)**:
  - Interactive client-side slice scrubber for 3D, 4D, 5D, and arbitrary high-dimensional tensors.
  - Dynamic dimension selector buttons (Batch, Channel, Depth, Height, Width) with real-time continuous slice sliders.
  - Multi-colormap support (`viridis`, `plasma`, `coolwarm`, `grayscale`), interactive zoom/pan, dynamic cursor inspection, and instant slice summary statistics.
  - Zero-dependency client-side HTML Canvas renderer operating completely offline with no CDN dependencies.
- **Layer-Level Activation Diagnostics (`sd.activations(model, input=...)`)**:
  - Mean, standard deviation, minimum, maximum, and bounded distribution samples across executed layers.
  - Exact zero fractions, non-finite (NaN/Inf) detection, and activation magnitude instability warnings.
  - Activation-regime diagnostics: ReLU inactivity, GELU negative-regime vs zero tracking, Sigmoid saturation near 0 and 1, Tanh saturation near -1 and +1.
  - Transparent invocation tracking (`layer#1`, `layer#2`) for repeated module executions in RNNs, weight-tied layers, and looped architectures.
  - Filtered layer selection (`include_types`, `exclude_types`, `layer_names`) with consistent summary statistics computed over filtered layers.
- **Memory-Safe Tensor Statistics (`compute_tensor_stats()`)**:
  - Bounded chunked processing on massive tensors ($O(\text{chunk\_size})$ intermediate memory) without allocating full-sized boolean masks, float copies, or blocking GPU synchronizations.
- **Memory-Safe Sparse Gradient Diagnostics (`sd.gradient_health()`)**:
  - Native sparse-gradient statistics on `values()` without calling `.to_dense()`, preventing out-of-memory crashes on massive embedding tables while correctly handling implicit zeros, duplicate sparse indices, and tied parameters.
- **Cross-Platform Robustness & Windows Compatibility**:
  - CP1252 / legacy Windows console safe formatting: ASCII fallback (`||grad||_2` instead of unicode `\u2207`) avoiding `UnicodeEncodeError`.
  - UTF-8 forced file encoding across all HTML and SVG exports.
  - Robust MIOpen kernel-cache collision avoidance for ROCm/AMD GPU environments.

---

## 3. Interactive Multidimensional Tensor Explorer

Understanding high-dimensional tensors (feature activations, convolutional feature pyramids, multi-head attention blocks, and volumetric medical scans) has traditionally required writing ad-hoc slicing code. SeeDL 0.6.0 introduces the **Multidimensional Tensor Explorer**:

### 3.1 Capabilities

- **High-Dimensional Slicing**: Effortlessly visualizes tensors of dimension $D \ge 3$. Users can designate any two axes as the 2D spatial display plane while scrubbing through remaining dimensions via interactive sliders.
- **Dimension Swapping**: Switch which dimensions map to spatial axes on the fly directly in the browser interface.
- **Color Mapping & Contrast**: Interactive choice of perceptual colormaps (`viridis`, `plasma`, `coolwarm`, `grayscale`) with automatic contrast normalization.
- **Dynamic Telemetry Overlay**: Displays min, max, mean, standard deviation, and sparsity for the currently viewed 2D slice alongside global tensor metadata.
- **Zero-Dependency Architecture**: Built using standard HTML5 Canvas and vanilla JavaScript (< 50 KB bundled). Does not require WebGL or external CDN script loading.

### 3.2 Usage Example

```python
import torch
import seedl as sd

# Inspect a 4D convolutional activation tensor [Batch, Channel, Height, Width]
features = torch.randn(4, 16, 64, 64)

# Launch interactive explorer in notebook or export to standalone HTML
view = sd.see(features, interactive=True)
view.save("tensor_explorer.html")
```

---

## 4. Model Observability & Layer Activation Diagnostics

While SeeDL 0.5.0 introduced forward execution tracing via `sd.flow()`, SeeDL 0.6.0 provides deep, layer-by-layer activation health diagnostics via `sd.activations()`:

### 4.1 Capabilities

- **Distribution & Health Telemetry**: Measures layer-wise mean, standard deviation, minimum, maximum, and bounded distribution samples.
- **Non-Finite Detection**: Immediately flags layers producing `NaN` or `+/-Inf` values, pinpointing numerical instability before it propagates through the network.
- **Activation-Regime Diagnostics**:
  - **ReLU Inactivity**: Tracks the fraction of outputs at exact zero ($x = 0$), identifying dormant neurons under the current input.
  - **GELU Diagnostics**: Separates the smooth negative regime ($x < 0$) from exact zeros to accurately reflect Gaussian Error Linear Unit behavior.
  - **Sigmoid Saturation**: Tracks outputs saturating near lower ($x \le \epsilon$) and upper ($x \ge 1 - \epsilon$) asymptotes where gradients vanish.
  - **Tanh Saturation**: Tracks outputs saturating near lower ($x \le -1 + \epsilon$) and upper ($x \ge 1 - \epsilon$) limits.
- **Magnitude Warnings**: Flags layers producing unusually large activations ($|x| > 1000$) that risk numerical overflow or FP16 underflow.
- **Repeated Module Invocation Tracking**:
  - Modules invoked multiple times during a single forward pass (e.g. RNN unrolling, looped blocks, shared attention layers) are assigned distinct invocation identifiers (`module#1`, `module#2`).
  - Ensures each execution step is reported with its own distinct activation statistics and shape profile.
- **Layer Filtering**:
  - Filter by module types (`include_types=[nn.Conv2d]`, `exclude_types=[nn.Dropout]`) or layer names.
  - Summary metrics and notes are accurately computed over the filtered subset rather than the full model.

### 4.2 Usage Example

```python
import torch
import torch.nn as nn
import seedl as sd


class RecurrentNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.Linear(32, 32)
        self.relu = nn.ReLU()
        self.out = nn.Linear(32, 2)

    def forward(self, x):
        h = x
        for _ in range(3):  # Repeated module invocations
            h = self.relu(self.cell(h))
        return self.out(h)


model = RecurrentNetwork()
x = torch.randn(8, 32)

# Diagnose activation health across all layers and repeated invocations
diag = sd.activations(model, input=x, zero_threshold=0.8)
diag.save("activation_diagnostics.html")
print(diag.summary)
```

---

## 5. Memory Safety & Numerically Stable Telemetry

Accurate diagnostics must not exhaust memory on production-scale models. SeeDL 0.6.0 hardens core telemetry routines against out-of-memory (OOM) failures:

### 5.1 Bounded Chunked Tensor Statistics

In previous versions, computing min, max, non-finite counts, and zero fractions on large tensors could allocate full-sized boolean mask tensors or trigger full floating-point copies.

SeeDL 0.6.0 introduces bounded chunked processing in `compute_tensor_stats()`:
- Tensors exceeding the chunk threshold ($10^6$ elements) are processed in $O(\text{chunk\_size})$ blocks.
- Intermediate boolean masks and reductions remain bounded within a small working memory window.
- Avoids large temporary allocations on GPU and avoids blocking synchronizations.

### 5.2 Memory-Safe Sparse Gradient Statistics

Large recommendation models and NLP systems often use sparse embedding tables (`nn.Embedding(..., sparse=True)`):
- Calling `.to_dense()` on a sparse gradient with shape $[500000, 1024]$ requires $\approx 2 \text{ GB}$ of RAM, frequently triggering OOM crashes.
- SeeDL 0.6.0 inspects `grad.values()` directly, computing norm and distribution statistics on non-zero stored values without densification.
- Implicit zeros, duplicate sparse indices (via `coalesce()`), and tied parameters are handled with mathematical correctness.

---

## 6. Cross-Platform Robustness & Windows Compatibility

SeeDL 0.6.0 resolves multiple Windows-specific platform issues:
- **Encoding Safety**: Windows consoles using legacy code pages (e.g. CP1252) previously raised `UnicodeEncodeError` when printing mathematical symbols such as `\u2207` (nabla). SeeDL 0.6.0 provides ASCII fallbacks (`||grad||_2`) and ensures safe stdout handling across all environments.
- **UTF-8 File Exports**: All HTML and SVG exports explicitly enforce `encoding="utf-8"`, preventing encoding corruption on Windows systems.
- **ROCm / AMD GPU Stability**: MIOpen kernel-cache collisions on Windows ROCm configurations are mitigated through isolated environment verification.

---

## 7. Verification & Scientific Benchmarks

Every feature in SeeDL 0.6.0 has undergone independent scientific and regression audits:

| Test Suite | Tests Passed | Tests Skipped | Tests Failed |
| :--- | :--- | :--- | :--- |
| Core Unit & Integration Suite | **530** | 0 | 0 |
| Playwright Headless Browser Suite | **24** | 0 | 0 |
| Native Windows AMD ROCm Smoke Tests | **Verified** | 0 | 0 |
| Ruff Linter & Formatter | **Passed** | 0 | 0 |
| Mypy Static Type Checker | **Passed** | 0 | 0 |
| Package Metadata & Clean-Room Install | **Verified** | 0 | 0 |

---

## 8. Migration & Upgrading

SeeDL 0.6.0 maintains **100% backward compatibility** with all existing SeeDL APIs. Upgrading requires no code modifications:

```bash
pip install --upgrade seedl
```

Or with optional integrations:
```bash
pip install --upgrade "seedl[all]"
```
