# SeeDL 0.5.0 — Neural Observatory

> **Trace the computation.**  
> **Understand the model.**  
> **Visualize the science.**

**Status:** Prepared for release; publication pending.  
*(Note for maintainers: When publishing the GitHub Release, update this status line to `**Status:** Published`.)*

---

## 1. Introduction

SeeDL 0.5.0 — **The Neural Observatory** — is a major milestone in open-source scientific visualization, inspection, and interpretability for deep learning and quantum computing models.

Building upon SeeDL's lightweight, zero-dependency core, version 0.5.0 transforms SeeDL into an advanced, laboratory-grade neural observatory. It allows researchers, machine learning engineers, educators, and scientists to record forward execution graphs, analyze multi-layer Transformer information propagation, audit gradient health across deep networks, compare multi-method attribution landscapes, investigate quantum noise channels, track dataset quality and drift, and compile self-contained, offline scientific reports.

As with all SeeDL releases, version 0.5.0 adheres to strict scientific software engineering principles:
- **Zero Mandatory Runtime Dependencies**: The core package runs on standard Python 3.10–3.13 without forcing heavy framework installations.
- **Authentic Data Only**: Visualizations represent authentic tensor activations, gradients, states, and graph topologies — never synthetic demonstrations or mocked outputs.
- **State Preservation Guarantees**: Model execution preserves user models, registered module buffers, mixed training modes, and parameter gradients.
- **Offline by Default**: Interactive HTML documents, SVG graphics, and reports require zero external CDN connections or runtime servers.

---

## 2. Highlights

- **Neural Execution Tracer (`sd.trace()`)**: Non-destructive forward execution graph recording with runtime hook interception and optional `torch.fx` symbolic topology analysis.
- **Tensor Flow Explorer (`sd.flow()`)**: Interactive browser-based Directed Acyclic Graph (DAG) visualizing module invocations, tensor shapes, memory footprints, and data dependencies.
- **Transformer Observatory (`sd.attention_rollout()`, `sd.attention_layers()`)**: Mathematical attention rollout across multi-layer Transformers with residual weighting and layer-by-layer attention matrix inspection.
- **Advanced Explainability (`sd.compare_attributions()`)**: Side-by-side attribution comparison between Saliency Maps, Integrated Gradients, and SmoothGrad with completeness axiom verification, pairwise Pearson correlation, and Cosine similarity.
- **Gradient Health Observatory (`sd.gradient_health()`)**: Layer-by-layer gradient norm diagnostics, gradient-to-weight ratios, vanishing/high gradient alerts, tied-parameter deduplication, and zero-gradient sparsity detection.
- **Training Run Comparison (`sd.compare_runs()`)**: Two-run training comparison with step alignment, final deltas, pointwise absolute differences, and missing metric alerts.
- **Quantum Noise Lab (`sd.quantum_noise()`)**: Single-qubit open quantum system simulation evaluating depolarizing, amplitude damping, and phase damping channels via Kraus operators, Bloch sphere trajectories, and von Neumann entropy.
- **Dataset Intelligence (`sd.dataset_health()`, `sd.dataset_drift()`)**: Distribution profiling, numerical condition checks, Population Stability Index (PSI), and Jensen-Shannon Divergence (JSD).
- **Scientific Report Builder (`sd.report()`)**: Offline, publication-ready multi-section scientific reports embedding isolated, interactive visualization cards via sandboxed iframes.
- **Safe Analysis Serialization (`sd.save_analysis()`, `sd.load_analysis()`)**: Pickle-free, pure JSON artifact serialization protected by SHA-256 canonical payload integrity verification and strict schema validation.

---

## 3. Neural Execution Tracer

The Neural Execution Tracer (`sd.trace(model, input)`) records an authentic execution graph of a model's forward pass:

- **Topology & Invocations**: Captures exact invocation sequences for both leaf submodules and hierarchical containers, including multiple invocations of shared or reused modules (`mod#0`, `mod#1`).
- **Data Dependency Tracking**: Explicitly distinguishes four edge categories:
  - `explicit_tensor`: Direct verified tensor object pointer identity between module output and downstream input.
  - `storage_alias`: Shared memory storage view identified via matching underlying storage data pointers (`data_ptr()`).
  - `symbolic_fx`: Verified data dependencies extracted from the symbolic computational graph via `torch.fx.symbolic_trace` (when available).
  - `inferred_sequential`: Heuristic execution-order fallback edges between consecutive leaf modules when intermediate functional operations obscure explicit tensor pointers (flagged `is_uncertain=True`).
- **Tensor Metadata**: Measures input/output tensor shapes, data types (`float32`, `bfloat16`, etc.), target devices, and total element counts per layer.
- **Activation Statistics**: Samples activation distributions non-destructively, recording min, max, mean, standard deviation, sparsity, and nonfinite counts.
- **Symbolic Tracing Boundaries**: Models with dynamic data-dependent control flow (e.g. `if x.sum() > 0: ...`), non-tensor Python primitives, or custom C++ operators not registered with FX gracefully fall back to hook-based tracing.

---

## 4. Tensor Flow Explorer

The Tensor Flow Explorer (`sd.flow(trace)` or `sd.see(model, input, view="flow")`) renders interactive execution DAGs:

- **Browser-Native Interactive Graph**: Nodes colored by module type, execution order, or memory footprint with responsive pan and zoom controls.
- **Inspection Sidebar**: Clicking any node opens immediate scientific telemetry detailing tensor shapes, memory overhead, layer parameters, duration in milliseconds, and activation statistics.
- **Edge Highlighting**: Hovering over tensor dependencies highlights upstream producers and downstream consumers.
- **Static SVG Fallback**: Built-in SVG renderer generates publication-quality vectorized graph diagrams without requiring browser runtimes or JavaScript.

---

## 5. Transformer Observatory

SeeDL 0.5.0 extends Transformer attention analysis from single layers to multi-layer network architectures:

- **Attention Rollout (`sd.attention_rollout()`)**: Implements the Abnar & Zuidema (2020) attention rollout algorithm:
  $$\tilde{A}_l = \left(0.5 A_l + 0.5 I\right) \tilde{A}_{l-1}$$
  tracking the propagation of token information through residual connections across successive self-attention blocks.
- **Multi-Layer Attention Inspection (`sd.attention_layers()`)**: Inspects attention weight matrices across all Transformer layers simultaneously with head-level breakdown and token-to-token routing heatmaps.
- **Head Entropy & Dispersion Diagnostics**: Computes Shannon entropy and attention dispersion per head and per layer to quantify focal vs. diffuse routing behaviors.

---

## 6. Advanced Explainability

SeeDL 0.5.0 introduces comprehensive attribution benchmarking via `sd.compare_attributions(model, input, target_class)`:

- **Integrated Gradients (Sundararajan et al., 2017)**: Path-integral attribution using composite midpoint numerical integration ($\alpha_k = (k - 0.5)/m$ for $k=1,\dots,m$), baseline selection, and numerical verification of the Completeness Axiom:
  $$\sum_i \text{Attribution}_i \approx F(x) - F(x') \pm \epsilon$$
- **SmoothGrad (Smilkov et al., 2017)**: Monte Carlo noise averaging over Gaussian-perturbed inputs to suppress gradient noise and isolate robust saliency signals.
- **Attribution Comparison Matrix**: Computes pairwise Pearson correlation, Cosine similarity, and summary statistics (min, max, mean, sparsity) across aligned attribution vectors to measure consensus.

---

## 7. Gradient Health

The Gradient Health Observatory (`sd.gradient_health(model)`) audits backpropagated gradients prior to optimizer steps:

- **Gradient Norms**: Computes exact $L_1$, $L_2$, and $L_\infty$ (max absolute value) norms per parameter tensor and aggregated global L2 norm ($\sqrt{\sum \|g_i\|_2^2}$).
- **Heuristic Pathology Alerts**: Reports observational warnings when global gradient L2 norm is near-zero ($\|g\|_2 < 10^{-7}$, vanishing gradient warning), high ($\|g\|_2 > 100.0$, suggesting gradient clipping), or contains nonfinite values (`NaN`, `Inf`).
- **Tied Parameter Transparency**: Deduplicates shared parameters by tracking memory storage pointers (`param.data_ptr()`), preventing artificial inflation of global gradient norms.
- **Gradient-to-Weight Ratio**: Evaluates $\|\nabla W\|_2 / \|W\|_2$ per parameter to quantify relative update magnitudes.
- **Sparsity Telemetry**: Flags layers with high fractions of exact zero gradients ($>80\%$), serving as an empirical heuristic indicator of sparse inputs or dead ReLU activations.

---

## 8. Training Run Comparison

`sd.compare_runs(run_a, run_b, *, name_a="Run A", name_b="Run B", metrics=None)` provides aligned comparative analysis between two training runs or experiment histories:

- **Step Alignment**: Aligns common metric sequences step-by-step; explicitly detects and warns on step count mismatches between runs without silently discarding data.
- **Metric Comparison**: Computes final values, final delta ($\Delta = \text{final}_B - \text{final}_A$), minimum achieved values, and mean absolute differences across aligned steps.
- **Missing Metric Detection**: Highlights metrics present in one run but missing from the other.
- **Flexible Inputs**: Accepts SeeDL `Tracker` objects, objects exposing a `.history` mapping or method, or raw metric dictionaries.

---

## 9. Quantum Intelligence

`sd.quantum_noise(state=None, channel="depolarizing", param=0.3, steps=10)` expands SeeDL's quantum visualization capabilities into noisy, single-qubit open quantum systems:

- **Kraus Channel Simulation**: Models depolarizing noise, amplitude damping (energy relaxation $T_1$), and phase damping (dephasing $T_2$) channels via valid trace-preserving Kraus operators:
  $$\rho' = \sum_k K_k \rho K_k^\dagger, \quad \sum_k K_k^\dagger K_k = I$$
- **Bloch Sphere Trajectories**: Simulates and plots the contraction of the Bloch vector $r = (x, y, z)$ under progressive noise parameter sweeps ($p \in [0, 1]$), illustrating state mixedness as the vector contracts into the interior of the Bloch sphere ($\|r\| < 1$).
- **Quantum Information Metrics**: Evaluates Purity ($\gamma = \text{Tr}(\rho^2) = \frac{1 + \|r\|^2}{2}$), Von Neumann Entropy in bits ($S = -\sum_i \lambda_i \log_2 \lambda_i$), and Quantum Coherence ($C = \sum_{i \neq j} |\rho_{ij}|$).
- **Trace Distance**: Provides exact trace distance computation $D(\rho, \sigma) = \frac{1}{2}\text{Tr}|\rho - \sigma|$ via `compute_trace_distance(rho, sigma)`.
- **Two-Qubit Entanglement Negativity**: Computes exact Peres-Horodecki Entanglement Negativity $\mathcal{N}(\rho) = \sum_{\mu_i < 0} |\mu_i| = \frac{\|\rho^{T_A}\|_1 - 1}{2} \in [0, 0.5]$ for bipartite states via `compute_two_qubit_negativity(rho)`. Note that this represents standard negativity $\mathcal{N}(\rho)$ (vanishing for separable states, $0.5$ for maximally entangled Bell states), distinct from logarithmic negativity $E_N(\rho) = \log_2 \|\rho^{T_A}\|_1$.

---

## 10. Dataset Intelligence

SeeDL 0.5.0 introduces automated dataset sanity and drift diagnostics:

- **Dataset Health (`sd.dataset_health(data)`)**: Computes summary statistics, missing-value fractions, constant columns, duplicate detection, and numerical condition number estimates with bounded sampling.
- **Dataset Drift (`sd.dataset_drift(reference_data, current_data)`)**: Detects empirical feature distribution shifts between baseline reference and current production data using:
  - **Population Stability Index (PSI)**: $\text{PSI} = \sum_i (q_i - p_i) \ln(q_i / p_i)$ with Laplace smoothing.
  - **Jensen-Shannon Divergence (JSD)**: Symmetric information-theoretic divergence $D_{JS}(P \parallel Q) \in [0.0, 1.0]$ bits.
  - **Distribution Shifts**: Mean shift ($\Delta\mu$), variance shift ($\Delta\sigma^2$), and relative histogram distance across normalized empirical bins.

---

## 11. Scientific Report Builder

`sd.report()` compiles multiple SeeDL visualizations into self-contained, publication-ready offline HTML reports:

- **Sandboxed Interactive Iframe Cards**: Interactive visualization views are embedded as isolated HTML documents via the HTML5 `srcdoc` attribute with `sandbox="allow-scripts"`.
- **Zero Namespace Collisions**: Each figure runs in its own execution context, preventing global CSS and JavaScript variable clashes between cards.
- **Pure Offline Guarantee**: Reports contain zero external CDN scripts or stylesheet links (`http://` / `https://`).
- **Static Fallbacks**: Matplotlib backends fall back to base64-encoded PNGs; static SVGs are rendered cleanly inline; passing `interactive=False` generates a completely script-free static document.

---

## 12. Reproducible Analysis Serialization

SeeDL 0.5.0 introduces safe, non-executable artifact storage via `sd.save_analysis()` and `sd.load_analysis()`:

- **Pickle-Free Security**: Strict JSON serialization disallows arbitrary Python bytecode, preventing arbitrary code execution vulnerabilities.
- **SHA-256 Canonical Integrity Verification**: Generates deterministic canonical UTF-8 JSON bytes (lexicographically sorted keys, compact separators, float standardization) to compute a cryptographic checksum. Modifying any numerical value or metadata triggers an immediate integrity verification failure.
- **Schema Versioning**: Strictly validates against schema version `1.0.0` and rejects unsupported or malformed artifact specifications.
- **DoS Safeguards**: Enforces a 50 MB file size limit before reading into memory to prevent resource exhaustion attacks.

---

## 13. Scientific Correctness

All 0.5.0 mathematical implementations have undergone rigorous numerical verification:
- **Kraus Completeness**: Formally verified that $\sum_k K_k^\dagger K_k = I$ to machine precision across all implemented quantum noise channels.
- **Density Matrix Hermiticity**: Guaranteed $\rho = \rho^\dagger$ and $\text{Tr}(\rho) = 1$ with positive semi-definite eigenvalues.
- **Completeness Axiom in Integrated Gradients**: Verified numerical convergence to within $0.1\%$ of the actual model logit difference on test architectures.
- **Deterministic Attention Rollout**: Mathematically equivalent results between NumPy and PyTorch implementations with validated row stochasticity ($\sum_j A_{ij} = 1$).

---

## 14. Security and State Preservation

SeeDL is designed to operate safely inside production research environments:
- **Full Module Buffer Restoration**: All registered PyTorch buffers (including non-persistent buffers and `None`-valued buffers of arbitrary size) are snapshotted prior to execution and restored under `torch.no_grad()` in guaranteed `finally` blocks.
- **Mixed Training Mode Preservation**: Submodules in mixed `train()` and `eval()` states retain their individual modes after tracing.
- **Parameter Gradient Protection**: Existing parameter `.grad` tensors are untouched and restored.
- **Caller Input Non-Mutation**: Input arguments are cloned and detached recursively; in-place forward operations by user models cannot mutate caller tensors.
- **Guaranteed Hook Removal**: All forward hooks and pre-hooks are removed in `finally` blocks, even when models raise runtime exceptions.
- **Sandboxing Boundaries**: While PyTorch buffers, parameters, gradients, modes, and input tensors are fully restored, arbitrary custom Python code in forward methods cannot be sandboxed against external side effects (such as mutations of global Python variables, file I/O, or non-PyTorch external state).

---

## 15. Performance

- **Zero Startup Overhead**: Importing `seedl` executes in under 20 milliseconds on standard systems because all heavy third-party packages (`torch`, `plotly`, `matplotlib`, `qiskit`) are lazily imported only when required.
- **Bounded Sampling**: Intermediate activation statistics are computed with configurable upper bounds (`max_samples=32`), avoiding out-of-memory errors on massive feature maps.
- **Lightweight SVG Output**: Static SVG charts typically require fewer than 30 KB of memory.

---

## 16. Installation

SeeDL 0.5.0 requires **Python 3.10, 3.11, 3.12, or 3.13**.

```bash
# Core package (zero mandatory runtime dependencies)
pip install seedl

# PyTorch deep-learning integrations (tracing, rollout, explainability, gradients)
pip install "seedl[torch]"

# Interactive browser visualizations (Plotly)
pip install "seedl[plotly]"

# Quantum computing integrations (Qiskit)
pip install "seedl[quantum]"

# Complete laboratory suite (all integrations)
pip install "seedl[all]"
```

---

## 17. Usage Examples

### Neural Execution Tracing
```python
import torch
import torch.nn as nn
import seedl as sd

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
x = torch.randn(4, 10)

# Record authentic execution trace
trace = sd.trace(model, x)

# Generate interactive browser visualization or export HTML
view = sd.flow(trace)
view.save("trace.html")
```

### Transformer Attention Rollout
```python
import torch
import seedl as sd

# Simulated multi-layer attention weight matrices [batch, heads, seq, seq]
tokens = ["[CLS]", "The", "neural", "observatory", "is", "active", "[SEP]"]
seq_len = len(tokens)
attention_matrices = [torch.softmax(torch.randn(1, 4, seq_len, seq_len), dim=-1) for _ in range(3)]

rollout_view = sd.attention_rollout(attention_matrices, tokens=tokens)
rollout_view.save("rollout.html")
```

### Gradient Health Audit
```python
import torch
import torch.nn as nn
import seedl as sd

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
x = torch.randn(4, 10)
loss = model(x).sum()
loss.backward()

# Audit backpropagated gradients across all layers
health_view = sd.gradient_health(model)
print(health_view.summary)
```

### Quantum Noise Simulation
```python
import seedl as sd

# Simulate amplitude damping channel on default pure state |+>
noise_view = sd.quantum_noise(channel="amplitude_damping", param=0.25, steps=10)
noise_view.save("quantum_noise.html")
```

### Training Run Comparison
```python
import seedl as sd

tracker_a = sd.track()
tracker_b = sd.track()
tracker_a.log(loss=0.50, accuracy=0.82)
tracker_a.log(loss=0.30, accuracy=0.91)
tracker_b.log(loss=0.60, accuracy=0.78)
tracker_b.log(loss=0.25, accuracy=0.93)

compare_view = sd.compare_runs(tracker_a, tracker_b, name_a="Baseline", name_b="Experimental")
compare_view.save("run_comparison.html")
```

### Dataset Drift Detection
```python
import seedl as sd

ref_data = [float(i) for i in range(100)]
cur_data = [float(i) + 5.0 for i in range(100)]

drift_view = sd.dataset_drift(ref_data, cur_data)
print(drift_view.summary)
```

### Reproducible Analysis Serialization
```python
import seedl as sd

# Save analysis view with SHA-256 integrity hash
sd.save_analysis(health_view, "run_audit.seedl.json")

# Safely load and verify integrity
verified_view = sd.load_analysis("run_audit.seedl.json")
```

---

## 18. Backward Compatibility

SeeDL 0.5.0 preserves backward compatibility across all verified public API functions and signatures introduced in versions 0.1.0 through 0.4.0:
- `sd.see()`
- `sd.compare()`
- `sd.explain()`
- `sd.features()`
- `sd.attention()`
- `sd.vit_attention()`
- `sd.watch()`
- `sd.track()`
- `sd.activations()`
- `sd.gradients()`
- `sd.bloch()`
- `sd.circuit()`
- `sd.statevector()`
- `sd.density_matrix()`
- `sd.entanglement()`

All existing visualization backends (HTML, SVG, Matplotlib, Plotly) and core data structures (`Visualization`, `Tracker`) remain fully supported without breaking changes.

---

## 19. Limitations

To maintain scientific integrity, the following limitations are explicitly noted:
- **Heuristic vs. Causal Explanations**: Attention rollout and gradient attribution methods quantify linear signal propagation and output sensitivity; they do **not** establish causal proof of model reasoning.
- **Execution Dependency Inference**: In models with custom unhooked Python functions or data-dependent dynamic control flow, some intermediate dependencies are labeled as `inferred_sequential` rather than verified explicit pointers.
- **Sandboxing Boundary**: Custom Python code executed inside user forward methods cannot be completely isolated from unmanaged external side effects (e.g. mutations to global Python variables, file I/O, or non-PyTorch system state).
- **Buffer Snapshot Memory**: Snapshotting registered model buffers creates detached clones before execution, which temporarily increases memory usage proportional to the total buffer size during tracing.
- **Hardware Timing**: Asynchronous GPU kernel execution means layer timing reflects CPU launch latency unless explicit CUDA synchronization is enabled.
- **Large Graph Rendering**: Browser rendering of execution traces containing more than 500 nodes may experience latency; the `max_nodes` parameter should be used to restrict visualization depth on massive architectures.

---

## 20. Full Changelog

### Added
- Neural Execution Tracer (`sd.trace()`) recording module invocation graphs and tensor shapes.
- Tensor Flow Explorer (`sd.flow()`) with interactive pan/zoom DAG navigation and node telemetry.
- Transformer Attention Rollout (`sd.attention_rollout()`) and Multi-Layer Attention Viewer (`sd.attention_layers()`).
- Attribution Comparison Laboratory (`sd.compare_attributions()`) supporting Saliency, Integrated Gradients, and SmoothGrad with Pearson correlation and Cosine similarity.
- Gradient Health Diagnostics (`sd.gradient_health()`) auditing norm distributions, sparsity, and nonfinite values.
- Training Run Comparator (`sd.compare_runs()`) with step alignment, delta metrics, and missing metric alerts.
- Quantum Noise Simulation (`sd.quantum_noise()`) modeling single-qubit open quantum systems via Kraus channels.
- Dataset Health and Drift Assessment (`sd.dataset_health()`, `sd.dataset_drift()`) with Population Stability Index and Jensen-Shannon Divergence.
- Scientific Report Builder (`sd.report()`) compiling multi-view offline reports in sandboxed iframes.
- Analysis Artifact Serialization (`sd.save_analysis()`, `sd.load_analysis()`) with SHA-256 cryptographic verification.

### Improved
- Model state preservation: registered buffers of arbitrary size, mixed train/eval modes, and gradients are restored.
- Caller input tensors are protected against in-place mutations.
- Report interactivity embeds authentic HTML views while maintaining zero CDN dependencies and XSS protection.
- Python 3.13 support verified across all optional extras.

### Fixed
- Analysis loader integrity validation: load now cryptographically verifies payload SHA-256 checksums and enforces schema versions.
- Execution tracer buffer mutation: registered buffers modified during forward pass are restored.
- Scientific report figure interactivity: interactive controls are preserved using sandboxed `<iframe>` `srcdoc` containers.
