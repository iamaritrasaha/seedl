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

---

## 📦 Packaging & Architecture

- **Optional Quantum Dependencies**: Added `[project.optional-dependencies] quantum = ["qiskit>=1.0", "numpy>=1.23"]`.
- **Zero Core Dependency Guarantee**: Pure Python core remains completely intact. Importing `seedl` performs zero eager imports of `qiskit`, `torch`, or `plotly`.
- **Self-Contained HTML Export**: `view.to_html()` and `view.save("output.html")` embed full WebGL figures and interactive JavaScript callbacks without external dependencies.
