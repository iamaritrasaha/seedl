# SeeDL 0.2.0 Release Notes: The Spatial & Quantum Revolution

**Version:** `0.2.0`  
**Release Date:** October 2026  
**License:** MIT  

---

## 🌟 Executive Summary

SeeDL 0.2.0 unifies classical deep learning and quantum information science into a single, high-performance scientific visualization library.

This release introduces two flagship systems:
1. **Interactive 3D Neural Networks**: Real-time WebGL orbit, pan, zoom, and direct pointer-click neuron selection with dynamic dendrite isolation and parametric inspection HUD.
2. **Quantum Computing & Information**: Interactive 3D Bloch spheres, quantum circuit diagrams with step-by-step unitary evolution, bipartite entanglement diagnostics via partial trace and von Neumann entropy, continuous phase probability bars, and density matrix heatmaps.

Throughout this milestone, SeeDL maintains its foundational architectural commitment: **zero mandatory third-party dependencies** for the core pure Python and SVG engine.

---

## 🚀 Track A: Advanced Interactive 3D Neural Networks

### 1. Direct 3D Viewport Pointer Interaction
- **Dual Event Binding**: Supports both native WebGL canvas click detection (via GPU color picking and raycast coordinates) and Plotly event dispatch (`plotly_click`).
- **Neuron Isolation**: Clicking any neuron highlights incoming synaptic connections in vibrant cyan (`#00e5ff`) and outgoing connections in warm amber (`#fbbf24`), dimming unrelated network edges to `0.04` opacity.
- **Camera Preservation**: Caches `scene.camera` during restyle and relayout operations, allowing smooth inspection without jarring viewport resets.
- **Parametric HUD Card**: Real-time HUD annotation card displays mathematical operations ($z = W \cdot a + b$), post-activation, bias, pre-activation, parameter trainability, and incoming/outgoing connection summaries.
- **Overlapping Layout Repair**: Restructured annotation anchors (`yanchor="top"`, `y=-0.15`) to eliminate collision with horizontal Plotly legends.

### 2. Premium 3D Visual Themes & Spatial Depth
- **Five Built-in Themes**: `scientific` (deep space navy default), `dark`, `light`, `minimal`, and `presentation`.
- **Decoupled Scene Graph (`_scene.py`)**: Clean architectural separation between scientific telemetry extraction and 3D rendering primitives.
- **High-Density Network Scaling**: Adaptive marker sizing, guide planes, and depth-axis spacing for deep multilayer architectures.

### 3. CNN 3D Spatial Feature Map Stacks (`_cnn_spatial.py`)
- **Non-Destructive Activation Hooks**: Safely intercepts intermediate activations across 2D convolutional layers under `eval()` mode.
- **Stacked 3D Feature Planes**: Visualizes $(H, W)$ channel activation grids stacked along the depth axis with viridis/teal intensity mapping.

---

## ⚛️ Track B: Quantum Information Visualization

### 1. Interactive 3D Bloch Sphere (`sd.bloch`)
- **Pauli Expectation Traces**: Computes exact Bloch vectors $r_i = \text{Tr}(\rho \sigma_i)$ for $i \in \{x, y, z\}$.
- **Pure vs. Mixed States**: Surface vectors ($|r| = 1.0$) for pure states; interior vectors ($|r| < 1.0$) for mixed states.
- **Convenient Input Parsing**: Accepts statevectors, $2 \times 2$ density matrices, Qiskit `Statevector` / `DensityMatrix`, and string labels (`"+"`, `"-"`, `"0"`, `"1"`, `"+i"`, `"-i"`).
- **Physical Reference Frame**: Equator wireframe, latitude parallels, prime meridian, coordinate axes, and antipodal pole markers.

### 2. Quantum Circuit Diagrams & State Evolution (`sd.circuit`, `sd.see(qc)`)
- **Multi-Qubit Circuit Grid**: Gate badges for $H, X, Y, Z, S, T, R_x, R_y, R_z$ and multi-qubit gates (CX, CZ).
- **Step-by-Step Unitary Simulation**: Tracks statevector evolution, basis probabilities, and per-qubit reduced Bloch vectors.
- **Measurement Halting Policy**: Non-unitary measurement and reset operations cleanly halt unitary evolution (`measurement_halt=True`), freezing pre-measurement probabilities and marking subsequent gates with `post_halt=True`.
- **Qiskit Integration**: Direct parsing of Qiskit `QuantumCircuit` objects.

### 3. Bipartite Entanglement Diagnostics (`sd.entanglement`)
- **Automated Partial Traces**: Computes reduced density operators $\rho_A = \text{Tr}_B(\rho)$ and $\rho_B = \text{Tr}_A(\rho)$ with verified little-endian qubit ordering.
- **von Neumann Entropy**: Calculates $S(\rho_A) = -\text{Tr}(\rho_A \log_2 \rho_A) = -\sum_i \lambda_i \log_2 \lambda_i$.
- **Benchmark Controls**: Verified against Bell pairs ($S = 1.0000$ bit), GHZ states, Werner states, and separable states ($S = 0.0000$ bit).

### 4. Statevectors & Density Matrices (`sd.statevector`, `sd.density_matrix`)
- **Continuous Phase Bar Charts**: Statevector probability bars color-coded by continuous complex phase $\text{arg}(c_i) \in [-\pi, \pi]$.
- **Hermitian Eigensolver Validation**: Validates positive semidefiniteness ($\lambda_i \ge -10^{-7}$) across arbitrary qubit dimensions ($2 \times 2, 4 \times 4, 8 \times 8$).

---

## 🛡️ Scientific Rigor, Security & Resource Safety

1. **Measurement Semantics**: Eliminates unphysical post-measurement states (e.g. $|0\rangle \to H \to \text{Measure} \to H$ yielding $|0\rangle$).
2. **Positive Semidefiniteness**: Rejects non-positive density matrices of any power-of-two dimension using NumPy Hermitian eigensolver.
3. **Qiskit Invariants**: Universal validation for power-of-two dimensions, finite amplitudes, and $L_2$ normalization across all input channels.
4. **HTML/JSON Security Hardening**: Sanitized user labels with `html.escape()`. Embedded script JSON fortified with Unicode escaping (`\u003c`, `\u003e`, `\u0026`) to prevent script breakout attacks (such as `</sCrIpT>`).
5. **Resource Limits**:
   - `MAX_STATEVECTOR_DIM = 16_384` (14 qubits)
   - `MAX_DENSITY_MATRIX_DIM = 256` (8 qubits)
   - `MAX_CIRCUIT_OPS = 500` operations

---

## 📦 Installation & Optional Dependencies

| Extra | Command | Purpose |
| :--- | :--- | :--- |
| **Core (Zero Deps)** | `pip install seedl` | Pure Python SVG engine |
| **Quantum + 3D** | `pip install "seedl[quantum,plotly]"` | 3D Bloch spheres, circuits & WebGL neural networks |
| **Deep Learning + 3D** | `pip install "seedl[torch,plotly]"` | PyTorch activation capture & 3D WebGL neural networks |
| **All Extras** | `pip install "seedl[all]"` | PyTorch, NumPy, Plotly, Matplotlib, Qiskit |
