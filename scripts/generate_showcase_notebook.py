"""Generate and execute the comprehensive 3D and Quantum showcase notebook.

Preserves compact committed notebook size by exporting heavy HTML artifacts
separately to examples/artifacts/.
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path("/media/hrik/Hrik/Projects/seedl/examples/advanced_3d_quantum_showcase.ipynb")
ARTIFACTS_DIR = Path("/media/hrik/Hrik/Projects/seedl/examples/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# SeeDL 0.2.0 · The Spatial & Quantum Revolution\n",
            "\n",
            "This showcase demonstrates interactive 3D scientific visualization for deep neural networks\n",
            "and quantum information systems.\n",
            "\n",
            "## Architecture\n",
            "- **Track A: Advanced Interactive 3D Neural Networks**: Direct neuron selection, connection neighborhood highlighting, forward propagation animation, CNN 3D feature maps, and transformer attention.\n",
            "- **Track B: Quantum Computing**: Interactive 3D Bloch sphere, single-qubit gate actions, quantum circuits, statevectors, density matrices, and multi-qubit entanglement diagnostics.\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import math\n",
            "from pathlib import Path\n",
            "import torch\n",
            "from torch import nn\n",
            "import seedl as sd\n",
            "\n",
            "print('SeeDL Version:', sd.__version__)\n",
            "print('PyTorch Version:', torch.__version__)\n",
            "try:\n",
            "    import qiskit\n",
            "    print('Qiskit Version:', qiskit.__version__)\n",
            "except ImportError:\n",
            "    print('Qiskit not installed (using pure Python quantum engine)')\n",
            "\n",
            "out_dir = Path('artifacts')\n",
            "out_dir.mkdir(parents=True, exist_ok=True)\n",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## SECTION A — NEURAL NETWORKS IN 3D\n",
            "Explore real network geometry, deterministic graph layout, direct click selection, and captured forward passes.",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Small MLP in 3D\n",
            "torch.manual_seed(42)\n",
            "small_mlp = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 2))\n",
            "v_small = sd.see(small_mlp, view='3d', theme='scientific')\n",
            "v_small.save(out_dir / '01_small_mlp_3d.html')\n",
            "print(v_small)\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Medium MLP in 3D\n",
            "med_mlp = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 3))\n",
            "v_med = sd.see(med_mlp, view='3d', theme='scientific')\n",
            "v_med.save(out_dir / '02_med_mlp_3d.html')\n",
            "print(v_med)\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3 & 4. Direct Neuron Inspection and Connection Highlighting\n",
            "# Select neuron at layer 1, index 2\n",
            "x = torch.tensor([[0.5, -1.2]])\n",
            "v_sel = sd.see(small_mlp, input=x, view='3d', select_neuron=(1, 2), theme='scientific')\n",
            "v_sel.save(out_dir / '03_selected_neuron_3d.html')\n",
            "print(v_sel)\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 5 & 6. Forward Propagation Playback and Activation Coloring\n",
            "explorer = sd.see(small_mlp, input=x, view='3d').explore()\n",
            "print('Interactive 3D explorer ready with Play, Pause, and Step controls.')\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 7. CNN Feature Maps in 3D\n",
            "class DemoCNN(nn.Module):\n",
            "    def __init__(self):\n",
            "        super().__init__()\n",
            "        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)\n",
            "        self.relu = nn.ReLU()\n",
            "        self.pool = nn.MaxPool2d(2)\n",
            "        self.conv2 = nn.Conv2d(4, 6, kernel_size=3, padding=1)\n",
            "    def forward(self, x):\n",
            "        return self.conv2(self.pool(self.relu(self.conv1(x))))\n",
            "\n",
            "cnn = DemoCNN()\n",
            "img = torch.randn(1, 1, 16, 16)\n",
            "v_cnn = sd.see(cnn, input=img, view='3d', theme='scientific')\n",
            "v_cnn.save(out_dir / '04_cnn_feature_maps_3d.html')\n",
            "print(v_cnn)\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 8. Transformer Attention Explorer\n",
            "import torch.nn.functional as F\n",
            "tokens = ['Quantum', 'Computing', 'Transforms', 'Deep', 'Learning']\n",
            "q = torch.randn(1, 4, 5, 16)\n",
            "k = torch.randn(1, 4, 5, 16)\n",
            "attn_weights = F.softmax((q @ k.transpose(-2, -1)) / 4.0, dim=-1)\n",
            "v_attn = sd.attention(attn_weights, tokens=tokens)\n",
            "v_attn.save(out_dir / '05_transformer_attention.html')\n",
            "print(v_attn)\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 9. Dense Model Fallback (Honest Architecture Hierarchy)\n",
            "dense_model = nn.Sequential(*[nn.Linear(64, 64) for _ in range(15)])\n",
            "v_dense = sd.see(dense_model, view='3d')\n",
            "print(v_dense)\n",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## SECTION B — QUANTUM COMPUTING VISUALIZATION\n",
            "Explore quantum geometry, Bloch spheres, gates, circuits, and entanglement diagnostics.",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1 & 2. Bloch Sphere for |0> and |+>\n",
            "v_b0 = sd.bloch('0', view='3d', theme='scientific')\n",
            "v_b0.save(out_dir / '06_bloch_zero.html')\n",
            "print('State |0>:', v_b0.summary)\n",
            "\n",
            "v_bplus = sd.bloch('+', view='3d', theme='scientific')\n",
            "v_bplus.save(out_dir / '07_bloch_plus.html')\n",
            "print('State |+>:', v_bplus.summary)\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3, 4, 5. Single-Qubit Gates and Unitary State Evolution\n",
            "# H |0> = |+>\n",
            "# X |0> = |1>\n",
            "# Ry(pi/2) |0> = |+>\n",
            "from seedl._quantum import get_single_qubit_gate, apply_gate_to_state\n",
            "state_0 = [1.0 + 0j, 0.0 + 0j]\n",
            "h_gate = get_single_qubit_gate('H')\n",
            "state_plus = apply_gate_to_state(state_0, h_gate)\n",
            "v_gate = sd.bloch(state_plus, view='3d', theme='scientific')\n",
            "print('After Hadamard gate:', v_gate.summary)\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 6. Single-Qubit Circuit\n",
            "from qiskit import QuantumCircuit\n",
            "qc_single = QuantumCircuit(1)\n",
            "qc_single.h(0)\n",
            "qc_single.t(0)\n",
            "v_qc1 = sd.see(qc_single, theme='scientific')\n",
            "v_qc1.save(out_dir / '08_single_qubit_circuit.html')\n",
            "print(v_qc1)\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 7. Two-Qubit Bell Circuit & State Evolution\n",
            "qc_bell = QuantumCircuit(2)\n",
            "qc_bell.h(0)\n",
            "qc_bell.cx(0, 1)\n",
            "v_bell_sim = sd.see(qc_bell, simulate=True, theme='scientific')\n",
            "v_bell_sim.save(out_dir / '09_bell_circuit_evolution.html')\n",
            "print(v_bell_sim)\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 8, 9, 10. Joint Probabilities, Reduced Density Matrices & Entanglement Entropy\n",
            "bell_state = [1.0 / math.sqrt(2.0), 0.0, 0.0, 1.0 / math.sqrt(2.0)]\n",
            "v_ent = sd.entanglement(bell_state, theme='scientific')\n",
            "v_ent.save(out_dir / '10_bell_entanglement_diagnostic.html')\n",
            "print(v_ent)\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 11. Statevector Amplitudes and Phases (3-qubit GHZ state)\n",
            "ghz_state = [1.0 / math.sqrt(2.0), 0, 0, 0, 0, 0, 0, 1.0 / math.sqrt(2.0)]\n",
            "v_ghz = sd.statevector(ghz_state, theme='scientific')\n",
            "v_ghz.save(out_dir / '11_ghz_statevector.html')\n",
            "print(v_ghz)\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 12. Density Matrix Inspection (Mixed State)\n",
            "rho_mixed = [[0.7 + 0j, 0.2 + 0.1j], [0.2 - 0.1j, 0.3 + 0j]]\n",
            "v_rho = sd.density_matrix(rho_mixed, theme='scientific')\n",
            "v_rho.save(out_dir / '12_mixed_density_matrix.html')\n",
            "print(v_rho)\n",
            "print('\\nAll 12 showcase demonstrations completed and exported successfully!')\n",
        ],
    },
]

notebook_content = {
    "cells": cells,
    "metadata": {
        "language_info": {"name": "python", "version": "3.13"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook_content, indent=1), encoding="utf-8")
print(f"Generated {NOTEBOOK_PATH}")
