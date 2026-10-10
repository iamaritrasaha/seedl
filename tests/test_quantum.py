"""Rigorous mathematical correctness and integration tests for SeeDL quantum features."""

import cmath
import math

import pytest

import seedl as sd
from seedl._quantum import (
    apply_gate_to_state,
    apply_noise_channel,
    compute_bloch_vector,
    get_single_qubit_gate,
    parse_density_matrix,
    parse_statevector,
    partial_trace_2qubit,
    statevector_to_density_matrix,
    von_neumann_entropy_single_qubit,
)

# =========================================================================
# 1. BLOCH VECTOR REFERENCE STATES (P0)
# =========================================================================


@pytest.mark.parametrize(
    "state_input,expected_bloch",
    [
        ("0", (0.0, 0.0, 1.0)),
        ("1", (0.0, 0.0, -1.0)),
        ("+", (1.0, 0.0, 0.0)),
        ("-", (-1.0, 0.0, 0.0)),
        ("+i", (0.0, 1.0, 0.0)),
        ("-i", (0.0, -1.0, 0.0)),
        ([1.0, 0.0], (0.0, 0.0, 1.0)),
        ([0.0, 1.0], (0.0, 0.0, -1.0)),
        ([1 / math.sqrt(2), 1 / math.sqrt(2)], (1.0, 0.0, 0.0)),
        ([1 / math.sqrt(2), -1 / math.sqrt(2)], (-1.0, 0.0, 0.0)),
        ([1 / math.sqrt(2), 1j / math.sqrt(2)], (0.0, 1.0, 0.0)),
        ([1 / math.sqrt(2), -1j / math.sqrt(2)], (0.0, -1.0, 0.0)),
    ],
)
def test_bloch_vectors_exact_reference_states(state_input, expected_bloch):
    x, y, z, purity = compute_bloch_vector(state_input)
    assert math.isclose(x, expected_bloch[0], abs_tol=1e-5)
    assert math.isclose(y, expected_bloch[1], abs_tol=1e-5)
    assert math.isclose(z, expected_bloch[2], abs_tol=1e-5)
    assert math.isclose(purity, 1.0, abs_tol=1e-5)
    r = math.sqrt(x**2 + y**2 + z**2)
    assert math.isclose(r, 1.0, abs_tol=1e-5)


def test_global_phase_invariance():
    """Physical Bloch vector must remain strictly invariant under global phase e^{iγ}."""
    vec = parse_statevector("+")
    for gamma in [0.1, 0.5, 1.234, math.pi / 3, 2.5]:
        phase_factor = cmath.exp(1j * gamma)
        phased_vec = [c * phase_factor for c in vec]
        x, y, z, purity = compute_bloch_vector(phased_vec)
        assert math.isclose(x, 1.0, abs_tol=1e-5)
        assert math.isclose(y, 0.0, abs_tol=1e-5)
        assert math.isclose(z, 0.0, abs_tol=1e-5)
        assert math.isclose(purity, 1.0, abs_tol=1e-5)


def test_mixed_state_bloch_vector_length():
    """Mixed state density operators must have Bloch vector length r < 1."""
    # Maximally mixed state: rho = I/2
    rho_mixed = [[0.5 + 0j, 0j], [0j, 0.5 + 0j]]
    x, y, z, purity = compute_bloch_vector(rho_mixed)
    assert math.isclose(x, 0.0, abs_tol=1e-5)
    assert math.isclose(y, 0.0, abs_tol=1e-5)
    assert math.isclose(z, 0.0, abs_tol=1e-5)
    assert math.isclose(purity, 0.5, abs_tol=1e-5)
    r = math.sqrt(x**2 + y**2 + z**2)
    assert math.isclose(r, 0.0, abs_tol=1e-5)

    # Partially mixed state: 0.75 |0><0| + 0.25 |1><1|
    rho_partial = [[0.75 + 0j, 0j], [0j, 0.25 + 0j]]
    x, y, z, purity = compute_bloch_vector(rho_partial)
    assert math.isclose(x, 0.0, abs_tol=1e-5)
    assert math.isclose(y, 0.0, abs_tol=1e-5)
    assert math.isclose(z, 0.5, abs_tol=1e-5)
    assert math.isclose(purity, 0.75**2 + 0.25**2, abs_tol=1e-5)
    r = math.sqrt(x**2 + y**2 + z**2)
    assert math.isclose(r, 0.5, abs_tol=1e-5)


# =========================================================================
# 2. QUANTUM GATES AND UNITARY EVOLUTION
# =========================================================================


def test_pauli_and_hadamard_gate_actions():
    """Verify matrix operations of single-qubit gates on statevectors."""
    state_0 = [1.0 + 0j, 0.0 + 0j]

    # X |0> = |1>
    x_gate = get_single_qubit_gate("X")
    state_1 = apply_gate_to_state(state_0, x_gate)
    bx, by, bz, _ = compute_bloch_vector(state_1)
    assert math.isclose(bz, -1.0, abs_tol=1e-5)

    # H |0> = |+>
    h_gate = get_single_qubit_gate("H")
    state_plus = apply_gate_to_state(state_0, h_gate)
    bx, by, bz, _ = compute_bloch_vector(state_plus)
    assert math.isclose(bx, 1.0, abs_tol=1e-5)

    # H |+> = |0>
    state_restored = apply_gate_to_state(state_plus, h_gate)
    bx, by, bz, _ = compute_bloch_vector(state_restored)
    assert math.isclose(bz, 1.0, abs_tol=1e-5)

    # S |+> = |+i>
    s_gate = get_single_qubit_gate("S")
    state_plus_i = apply_gate_to_state(state_plus, s_gate)
    bx, by, bz, _ = compute_bloch_vector(state_plus_i)
    assert math.isclose(by, 1.0, abs_tol=1e-5)

    # Z |+> = |->
    z_gate = get_single_qubit_gate("Z")
    state_minus = apply_gate_to_state(state_plus, z_gate)
    bx, by, bz, _ = compute_bloch_vector(state_minus)
    assert math.isclose(bx, -1.0, abs_tol=1e-5)


def test_rotation_gates():
    """Test Rx, Ry, Rz rotation gates with specific angles."""
    state_0 = [1.0 + 0j, 0.0 + 0j]

    # Ry(pi) |0> = |1> (up to global phase)
    ry_pi = get_single_qubit_gate("RY", parameter=math.pi)
    res_1 = apply_gate_to_state(state_0, ry_pi)
    bx, by, bz, _ = compute_bloch_vector(res_1)
    assert math.isclose(bz, -1.0, abs_tol=1e-5)

    # Ry(pi/2) |0> = |+>
    ry_half = get_single_qubit_gate("RY", parameter=math.pi / 2.0)
    res_plus = apply_gate_to_state(state_0, ry_half)
    bx, by, bz, _ = compute_bloch_vector(res_plus)
    assert math.isclose(bx, 1.0, abs_tol=1e-5)

    # Rx(pi/2) |0> -> vector along -Y
    rx_half = get_single_qubit_gate("RX", parameter=math.pi / 2.0)
    res_rx = apply_gate_to_state(state_0, rx_half)
    bx, by, bz, _ = compute_bloch_vector(res_rx)
    assert math.isclose(by, -1.0, abs_tol=1e-5)


# =========================================================================
# 3. DENSITY MATRIX VALIDATION AND NOISE CHANNELS
# =========================================================================


def test_density_matrix_validation():
    """Verify checks for Hermiticity, trace, and positivity."""
    # Non-Hermitian matrix
    non_herm = [[1.0 + 0j, 0.5 + 0j], [0.0 + 0j, 0.0 + 0j]]
    with pytest.raises(ValueError, match="not Hermitian"):
        parse_density_matrix(non_herm)

    # Trace not 1.0
    bad_tr = [[0.8 + 0j, 0j], [0j, 0.8 + 0j]]
    with pytest.raises(ValueError, match="trace must be approximately 1.0"):
        parse_density_matrix(bad_tr)

    # Negative eigenvalue
    bad_pos = [[1.2 + 0j, 0j], [0j, -0.2 + 0j]]
    with pytest.raises(ValueError, match="positive semi-definite"):
        parse_density_matrix(bad_pos)


def test_noise_channels():
    """Test dephasing and depolarizing noise channels on density operators."""
    # Pure |+> state density matrix
    rho_plus = [[0.5 + 0j, 0.5 + 0j], [0.5 + 0j, 0.5 + 0j]]

    # Full dephasing (p = 1.0) eliminates off-diagonal elements -> maximally mixed in z
    rho_dephased = apply_noise_channel(rho_plus, "dephasing", parameter=1.0)
    assert math.isclose(abs(rho_dephased[0][1]), 0.0, abs_tol=1e-5)
    bx, by, bz, _ = compute_bloch_vector(rho_dephased)
    assert math.isclose(bx, 0.0, abs_tol=1e-5)
    assert math.isclose(by, 0.0, abs_tol=1e-5)
    assert math.isclose(bz, 0.0, abs_tol=1e-5)

    # Depolarizing (p = 1.0) -> I/2
    rho_depol = apply_noise_channel(rho_plus, "depolarizing", parameter=1.0)
    bx, by, bz, purity = compute_bloch_vector(rho_depol)
    assert math.isclose(purity, 0.5, abs_tol=1e-5)
    assert math.isclose(bx, 0.0, abs_tol=1e-5)


# =========================================================================
# 4. ENTANGLEMENT, BELL STATES & PARTIAL TRACE
# =========================================================================


def test_bell_state_reduced_states_and_entropy():
    """For a Bell state |Phi+>, reduced density matrices are I/2 and entropy is 1.0 bit."""
    # Bell state: |Phi+> = (|00> + |11>) / sqrt(2)
    bell_vec = [1.0 / math.sqrt(2.0), 0.0, 0.0, 1.0 / math.sqrt(2.0)]
    rho_4x4 = statevector_to_density_matrix(bell_vec)

    # Partial trace over qubit 1 (MSB) -> leaves qubit 0 (LSB)
    rho_q0 = partial_trace_2qubit(rho_4x4, trace_out=1)
    assert math.isclose(rho_q0[0][0].real, 0.5, abs_tol=1e-5)
    assert math.isclose(rho_q0[1][1].real, 0.5, abs_tol=1e-5)
    assert math.isclose(abs(rho_q0[0][1]), 0.0, abs_tol=1e-5)

    # Partial trace over qubit 0 (LSB) -> leaves qubit 1 (MSB)
    rho_q1 = partial_trace_2qubit(rho_4x4, trace_out=0)
    assert math.isclose(rho_q1[0][0].real, 0.5, abs_tol=1e-5)
    assert math.isclose(rho_q1[1][1].real, 0.5, abs_tol=1e-5)
    assert math.isclose(abs(rho_q1[0][1]), 0.0, abs_tol=1e-5)

    # Reduced Bloch vectors must be (0, 0, 0)
    bx0, by0, bz0, _ = compute_bloch_vector(rho_q0)
    bx1, by1, bz1, _ = compute_bloch_vector(rho_q1)
    assert math.isclose(bx0, 0.0, abs_tol=1e-5)
    assert math.isclose(by0, 0.0, abs_tol=1e-5)
    assert math.isclose(bz0, 0.0, abs_tol=1e-5)
    assert math.isclose(bx1, 0.0, abs_tol=1e-5)
    assert math.isclose(by1, 0.0, abs_tol=1e-5)
    assert math.isclose(bz1, 0.0, abs_tol=1e-5)

    # Entanglement entropy must be exactly 1.0 bit
    s0 = von_neumann_entropy_single_qubit(rho_q0)
    s1 = von_neumann_entropy_single_qubit(rho_q1)
    assert math.isclose(s0, 1.0, abs_tol=1e-5)
    assert math.isclose(s1, 1.0, abs_tol=1e-5)


def test_product_state_zero_entropy():
    """Separable product states have zero entanglement entropy."""
    # |00>
    prod_vec = [1.0, 0.0, 0.0, 0.0]
    rho_4x4 = statevector_to_density_matrix(prod_vec)
    rho_q0 = partial_trace_2qubit(rho_4x4, trace_out=1)
    rho_q1 = partial_trace_2qubit(rho_4x4, trace_out=0)
    s0 = von_neumann_entropy_single_qubit(rho_q0)
    s1 = von_neumann_entropy_single_qubit(rho_q1)
    assert math.isclose(s0, 0.0, abs_tol=1e-5)
    assert math.isclose(s1, 0.0, abs_tol=1e-5)


# =========================================================================
# 5. PUBLIC API & VISUALIZATION DISPATCH (P0 / P1)
# =========================================================================


def test_public_bloch_api():
    """Test sd.bloch() with named and array inputs."""
    view = sd.bloch("+", view="3d")
    assert view.kind == "bloch"
    assert "Bloch Sphere" in view.title
    assert math.isclose(view.data["x"], 1.0, abs_tol=1e-5)

    # Plotly figure construction
    fig = view.figure
    assert fig is not None
    assert len(fig.data) > 0


def test_public_entanglement_api():
    """Test sd.entanglement() on Bell state."""
    bell = [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)]
    view = sd.entanglement(bell)
    assert view.kind == "entanglement"
    assert math.isclose(view.data["entropy_q0"], 1.0, abs_tol=1e-5)
    assert math.isclose(view.data["entropy_q1"], 1.0, abs_tol=1e-5)


def test_public_statevector_api():
    """Test sd.statevector() for 3-qubit GHZ state."""
    # GHZ: (|000> + |111>) / sqrt(2)
    ghz = [1 / math.sqrt(2), 0, 0, 0, 0, 0, 0, 1 / math.sqrt(2)]
    view = sd.statevector(ghz)
    assert view.kind == "statevector"
    assert view.data["num_qubits"] == 3
    assert math.isclose(view.data["probabilities"][0], 0.5, abs_tol=1e-5)
    assert math.isclose(view.data["probabilities"][7], 0.5, abs_tol=1e-5)


def test_public_density_matrix_api():
    """Test sd.density_matrix() for 2x2 density operator."""
    rho = [[0.7 + 0j, 0.2 + 0.1j], [0.2 - 0.1j, 0.3 + 0j]]
    view = sd.density_matrix(rho)
    # 2x2 defaults to bloch representation
    assert view.kind == "bloch"


# =========================================================================
# 6. QISKIT INTEGRATION & CIRCUIT VISUALIZATION
# =========================================================================


def test_qiskit_circuit_visualization():
    """Verify QuantumCircuit inspection and simulation with Qiskit."""
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    view = sd.see(qc)
    assert view.kind == "quantum_circuit"
    assert view.data["num_qubits"] == 2
    assert view.data["num_ops"] == 2
    assert view.data["instructions"][0]["name"] == "h"
    assert view.data["instructions"][1]["name"] == "cx"

    # Circuit simulation
    view_sim = sd.see(qc, simulate=True)
    assert view_sim.data["simulate"] is True
    assert "evolution" in view_sim.data
    # 3 steps: Init, H, CX
    evo = view_sim.data["evolution"]
    assert len(evo) == 3
    # Step 0: |00> has prob 1.0
    assert math.isclose(evo[0]["probabilities"][0], 1.0, abs_tol=1e-5)
    # Step 2: Bell state (|00> and |11> have prob 0.5)
    assert math.isclose(evo[2]["probabilities"][0], 0.5, abs_tol=1e-5)
    assert math.isclose(evo[2]["probabilities"][3], 0.5, abs_tol=1e-5)
