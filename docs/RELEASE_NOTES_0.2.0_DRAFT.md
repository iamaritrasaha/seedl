# SeeDL 0.2.0 Draft Release Notes — The Spatial & Quantum Revolution

**Milestone Target:** `0.2.0`  
**Current Core Version:** `0.1.0` (Unbumped per non-negotiable release rules until maintainer authorization)  
**Branch:** `feature/advanced-3d-quantum`  

---

## Overview

SeeDL 0.2.0 transforms the library from a classical neural network explainability tool into a unified scientific visualization platform spanning classical deep learning architectures and quantum systems.

---

## 🚀 Track A — Advanced Interactive 3D Neural Networks

### 1. Direct 3D Viewport Neuron Selection
- **Client-Side Event Bridge**: Embedded JavaScript post-script attaching `plotly_click` and click-to-clear listeners directly to the WebGL graph canvas.
- **Customdata Tagging**: Every 3D neuron marker carries its unique layer and index coordinates (`customdata = "l:n"`).
- **Interactive Dendrite Isolation**: Clicking any neuron highlights incoming connections in cyan (`#00e5ff`) and outgoing connections in warm amber (`#fbbf24`), dimming unrelated connections to `opacity: 0.04`.
- **Camera Viewport Preservation**: Dynamically caches `scene.camera` during Plotly restyle and relayout operations, preventing jarring camera jumps when switching neuron selections.
- **Parametric HUD Card**: Real-time inspection panel displaying neuron operation ($z = W \cdot a + b$), true post-activation, bias, pre-activation, parameter trainability, and sorted weights.

### 2. Premium 3D Visual Themes & Spatial Depth
- **Five Tested Themes**: `scientific` (deep space navy default), `dark`, `light`, `minimal`, and `presentation`.
- **Spatial Scene Abstraction (`_scene.py`)**: Reusable decoupled scene representation separating scientific telemetry from visual rendering primitives.
- **High-Density Scaling**: Adaptive marker sizing and aggregated cluster bounding planes for deeper architectures.

### 3. CNN 3D Spatial Feature Map Stacks (`_cnn_spatial.py`)
- **Layer Hook Telemetry**: Captures intermediate activations across convolutional layers without modifying model weights.
- **Stacked 3D Feature Map Planes**: Renders $(H, W)$ channel activation grids stacked along the depth axis with teal/viridis color intensity mapping.

---

## ⚛️ Track B — The Quantum Visualization Subsystem

### 1. Interactive 3D Bloch Sphere (`sd.bloch`)
- **Mathematical Rigor**: Computes Bloch vectors via exact Pauli expectation traces:
  $$r_i = \text{Tr}(\rho \sigma_i), \quad i \in \{x, y, z\}$$
- **Pure and Mixed States**: Surface vectors ($r = 1.0$) for pure statevectors; interior vectors ($r < 1.0$) for mixed density matrices.
- **Spatial Reference Framework**: High-detail wireframe equator, parallels ($z = \pm 0.5$), prime meridians, antipodal basis poles ($|0\rangle, |1\rangle, |+\rangle, |-\rangle, |+i\rangle, |-i\rangle$), and coordinate axes with directional arrows.
- **Noise Channels**: Exact simulation of dephasing, depolarizing, and amplitude damping channels.

### 2. Quantum Circuit Diagrams & State Evolution (`sd.circuit`)
- **Interactive Wire Diagrams**: Clean multi-qubit circuit grid with distinct gate badges ($H, X, Y, Z, S, T, R_x, R_y, R_z$) and controlled operations (CNOT / CX, CZ).
- **Unitary Statevector Evolution**: Sequential matrix multiplication tracking intermediate states, basis probabilities, and per-qubit reduced Bloch vectors.
- **Qiskit Integration**: Seamless parsing of Qiskit `QuantumCircuit` objects with automatic fallback to pure Python state manipulation.

### 3. Bipartite Entanglement Diagnostics (`sd.entanglement`)
- **Automated Partial Traces**: Computes reduced density matrices $\rho_A = \text{Tr}_B(\rho)$ and $\rho_B = \text{Tr}_A(\rho)$.
- **von Neumann Entropy**:
  $$S(\rho_A) = -\text{Tr}(\rho_A \log_2 \rho_A) = -\sum_i \lambda_i \log_2 \lambda_i$$
- **Benchmark Controls**: Rigorously verified on Bell pairs ($S = 1.0000$ bit), GHZ states, Werner mixed states, and separable product states ($S = 0.0000$ bit).

### 4. Continuous Phase & Probability Visualizations
- **Statevector Amplitudes (`sd.statevector`)**: High-fidelity probability bar charts with continuous phase colorwheel encoding ($\text{arg}(c_i) \in [-\pi, \pi]$).
- **Density Matrix Heatmaps (`sd.density_matrix`)**: Dual real and imaginary matrix heatmaps with Hermiticity, trace ($\text{Tr}(\rho) = 1$), and purity ($\gamma = \text{Tr}(\rho^2)$) diagnostics.

## 📋 Version & Feature Availability Matrix

| Feature / Subsystem | PyPI 0.1.0 (Released) | Unreleased Source (dev/main) | Planned for 0.2.0 | Status / Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Classical Model 2D Architecture** (`sd.see`) | ✅ Yes | ✅ Yes | ✅ Yes | Production-ready |
| **Tensor & Array Inspection** | ✅ Yes | ✅ Yes | ✅ Yes | Scaled with `max_points` |
| **Training History Observability** (`sd.watch`, `sd.track`) | ✅ Yes | ✅ Yes | ✅ Yes | Multi-run comparison |
| **Layer Activations & Gradients** | ✅ Yes | ✅ Yes | ✅ Yes | Non-destructive hooks |
| **Explainability (Saliency, IG, Grad-CAM)** | ✅ Yes | ✅ Yes | ✅ Yes | Complete axioms |
| **Interactive 3D Neural Networks** (`view="3d"`) | ❌ No | ✅ Yes | ✅ Yes | Requires `[plotly]` |
| **Direct 3D Viewport Neuron Clicking** | ❌ No | ✅ Yes | ✅ Yes | WebGL event bridge |
| **3D CNN Spatial Stacks** | ❌ No | ✅ Yes | ✅ Yes | Interactive depth |
| **3D Bloch Sphere** (`sd.bloch`) | ❌ No | ✅ Yes | ✅ Yes | Pure & mixed states |
| **Quantum Circuit Evolution** (`sd.circuit`) | ❌ No | ✅ Yes | ✅ Yes | Audited measurement halt |
| **Entanglement Diagnostics** (`sd.entanglement`) | ❌ No | ✅ Yes | ✅ Yes | 2-qubit von Neumann |
| **Statevector Phase Bars** (`sd.statevector`) | ❌ No | ✅ Yes | ✅ Yes | Bounded at 64 states |
| **Density Matrix Validation** (`sd.density_matrix`) | ❌ No | ✅ Yes | ✅ Yes | NumPy eigvalsh PSD |
| **Multiparty Entanglement (> 2 qubits)** | ❌ No | ❌ No | ⏳ Future | Planned for v0.3.0 |
| **Lindblad Master Equation Solvers** | ❌ No | ❌ No | ⏳ Future | Noise channels only |
| **Pulse-Level Circuit Compilation** | ❌ No | ❌ No | ⏳ Future | Gate level only |

> [!IMPORTANT]
> **PyPI 0.1.0 vs. 0.2.0 Notice**: `pip install seedl==0.1.0` contains the classical deep learning features only. The quantum APIs (`sd.bloch`, `sd.circuit`, `sd.entanglement`, `sd.statevector`, `sd.density_matrix`) and 3D neuron click interaction are part of the unreleased 0.2.0 development milestone. To use them prior to the official 0.2.0 release, install from the development branch: `pip install "seedl[all] @ git+https://github.com/iamaritrasaha/seedl.git"`.

---

## 🛡️ Scientific Correctness Audit & Repairs (0.2.0 Development Milestone)

Following an independent scientific computing and quantum information audit, the following critical repairs were implemented:

1. **Circuit Measurement Semantics Halting**:
   - *Previous behavior*: `simulate_circuit_evolution()` silently treated `MEASURE` instructions as no-ops and continued applying subsequent unitary gates to an uncollapsed state.
   - *Repaired behavior*: Measurement now faithfully records pre-measurement superposition probabilities, logs the target classical register bits, and explicitly halts subsequent unitary evolution with `post_halt=True` and `measurement_halt=True` flags, preventing scientifically invalid superposition claims after projective measurement.
   - Non-unitary `RESET`, conditional gates (`c_if`, `if_else`), and unsupported gates are similarly halted with precise diagnostic metadata.

2. **Full Density Matrix Positive Semidefiniteness (PSD) Validation**:
   - *Previous behavior*: `parse_density_matrix()` checked positive semidefiniteness exclusively for $2 \times 2$ matrices; $4 \times 4$ and $8 \times 8$ matrices with negative eigenvalues were silently accepted.
   - *Repaired behavior*: Integrated NumPy's Hermitian eigensolver `np.linalg.eigvalsh()` to enforce non-negative eigenvalues ($\lambda_{\min} \ge -10^{-5}$) across all square matrix dimensions ($2 \times 2, 4 \times 4, 8 \times 8, \dots$). Includes non-finite (NaN/Inf) element rejection.

3. **Universal Qiskit Statevector Invariant Enforcement**:
   - *Previous behavior*: Qiskit `Statevector` objects bypassed normalization, non-empty, and power-of-two dimension checks.
   - *Repaired behavior*: Unified through `_validate_statevector_vec()`, guaranteeing that all input representations (Qiskit objects, NumPy arrays, native lists) satisfy identical mathematical invariants.

4. **Resource Safety & Bounded Payloads**:
   - Bounded statevector dimension at $2^{14} = 16,384$ (14 qubits) to prevent memory exhaustion.
   - Bounded density matrix dimension at $256 \times 256$ (8 qubits) to prevent quadratic memory blowup.
   - Bounded circuit simulation operations at 500 instructions (`max_ops=500`).
   - Bounded rendered basis states in `sd.statevector()` at 64 states by default (`max_states=64`), sorting by probability and transparently disclosing truncation in telemetry notes.
   - Fixed SVG bar width and spacing geometry to eliminate visual bar overlap.

5. **3D Neuron Interaction Security & Robustness**:
   - Protected client-side JavaScript serialization against script-tag breakout (`</script>` escaping).
   - Escaped HTML entities in parametric HUD cards (`escHtml`), preventing rendering breakage on models with custom `<...>` layer labels.
   - Validated camera preservation and repeated click lifecycle via headless Chromium browser tests.

---

## 📦 Packaging & Architecture

- **Optional Quantum Dependencies**: Added `[project.optional-dependencies] quantum = ["qiskit>=1.0", "numpy>=1.23"]`.
- **Zero Core Dependency Guarantee**: Pure Python core remains completely intact. Importing `seedl` performs zero eager imports of `qiskit`, `torch`, or `plotly`.
- **Self-Contained HTML Export**: `view.to_html()` and `view.save("output.html")` embed full WebGL figures and interactive JavaScript callbacks without external dependencies.
