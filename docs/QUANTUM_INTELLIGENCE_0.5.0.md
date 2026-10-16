# Quantum Intelligence 2.0: Open Quantum Systems, Kraus Noise, & Entanglement

## 1. Overview

While SeeDL 0.4.0 provided foundational quantum circuit and statevector visualization, **Quantum Intelligence 2.0** elevates SeeDL into a **research-grade open quantum system laboratory**:

1. **Mixed State Analysis**: Density matrices $\rho$, quantum purity $\gamma = \text{Tr}(\rho^2)$, and von Neumann entropy $S(\rho) = -\text{Tr}(\rho \log_2 \rho)$.
2. **CPTP Kraus Noise Channels**: Exact simulation of quantum decoherence under Depolarizing, Phase Damping (Dephasing), and Amplitude Damping channels.
3. **Bloch Vector Decoherence Trajectories**: Real-time continuous visualization of Bloch vector contraction inside the Bloch sphere.
4. **Two-Qubit Entanglement Negativity**: Exact Peres-Horodecki partial transpose evaluation for 2-qubit mixed states.

---

## 2. Kraus Representation of Quantum Noise

Any physical open quantum system interacting with an environment can be modeled by a Completely Positive Trace-Preserving (CPTP) map:
$$\mathcal{E}(\rho) = \sum_{k} K_k \rho K_k^\dagger$$
satisfying the completeness condition:
$$\sum_{k} K_k^\dagger K_k = I$$

### Implemented Channels

#### 1. Depolarizing Channel
Represents isotropic decoherence where the qubit state is replaced with the completely mixed state $\frac{I}{2}$ with probability $p$:
$$K_0 = \sqrt{1 - \frac{3p}{4}} I, \quad K_1 = \sqrt{\frac{p}{4}} X, \quad K_2 = \sqrt{\frac{p}{4}} Y, \quad K_3 = \sqrt{\frac{p}{4}} Z$$
Bloch vector shrinks uniformly: $\vec{r}' = (1 - p) \vec{r}$.

#### 2. Phase Damping (Dephasing) Channel
Models loss of quantum coherence without energy exchange (decay of transverse off-diagonal elements):
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1 - \lambda} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & 0 \\ 0 & \sqrt{\lambda} \end{pmatrix}$$
Coherence decays as $\rho_{01}' = \sqrt{1 - \lambda} \rho_{01}$, flattening the Bloch sphere along the $Z$ axis.

#### 3. Amplitude Damping Channel
Models energy dissipation (spontaneous emission from excited state $|1\rangle$ to ground state $|0\rangle$):
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1 - \gamma} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$
The Bloch sphere shrinks and shifts upward toward the North pole ($|0\rangle$).

---

## 3. Entanglement Negativity for 2-Qubit Mixed States

To quantify quantum entanglement in mixed states $\rho$, SeeDL computes the **Negativity** $\mathcal{N}(\rho)$ based on the Peres-Horodecki Positive Partial Transpose (PPT) criterion:
$$\mathcal{N}(\rho) = \frac{\|\rho^{T_A}\|_1 - 1}{2} = \sum_{i} \frac{|\mu_i| - \mu_i}{2}$$
where $\mu_i$ are the eigenvalues of the partial transpose $\rho^{T_A}$ taken with respect to subsystem $A$:
- **Separable States**: All $\mu_i \ge 0 \implies \mathcal{N}(\rho) = 0.0$.
- **Maximally Entangled Bell States**: $\mathcal{N}(|\Phi^+\rangle) = 0.5$.

---

## 4. Usage Example

```python
import seedl as sd

# Simulate single-qubit decoherence under depolarizing noise
viz = sd.quantum_noise(
    channel="depolarizing",
    param=0.4,
    steps=10,
)

# Export interactive HTML with real-time continuous parameter slider
viz.save("quantum_decoherence.html")
```
