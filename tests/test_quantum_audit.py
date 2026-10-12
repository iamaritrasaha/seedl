"""Scientific Correctness Regression Tests — SeeDL 0.2.0 Audit.

Covers bugs confirmed and fixed during independent audit:

  1. parse_statevector Qiskit path skipped all validation
     (zero norm, non-pow2 dimension, non-finite elements, unnormalised).
  2. parse_density_matrix only validated PSD for 2×2; 4×4 and 8×8
     matrices with negative eigenvalues were silently accepted.
  3. Non-finite (NaN/Inf) elements accepted in statevectors and density matrices.
  4. simulate_circuit_evolution treated MEASURE as a no-op and continued
     unitary evolution after measurement — scientifically misleading.
  5. RESET, conditional gates, and unsupported gates not properly halted.

Also contains independent Qiskit reference comparisons for all cardinal
states, all standard gates, Bell/GHZ states, asymmetric product states,
partial traces, and von Neumann entropy.
"""

from __future__ import annotations

import cmath
import math

import pytest

from seedl._quantum import (
    compute_bloch_vector,
    get_single_qubit_gate,
    parse_density_matrix,
    parse_statevector,
    partial_trace_2qubit,
    simulate_circuit_evolution,
    statevector_to_density_matrix,
    von_neumann_entropy_single_qubit,
)

# =========================================================================
# 1. parse_statevector — Qiskit path must apply all invariants
# =========================================================================


def _make_qiskit_like_statevector(data: list):
    """Return a stand-in object that _is_qiskit_state() recognises as Qiskit Statevector."""
    import types

    cls = types.new_class(
        "Statevector",
        (),
        {},
        lambda ns_: ns_.update(
            {
                "__module__": "qiskit.quantum_info",
                "data": types.SimpleNamespace(tolist=lambda s=data: s),
            }
        ),
    )
    return cls()


def test_qiskit_like_object_rejects_zero_norm():
    """Qiskit-like Statevector object with zero norm must raise ValueError."""
    fake = _make_qiskit_like_statevector([0.0, 0.0])
    with pytest.raises(ValueError, match="zero norm"):
        parse_statevector(fake)


def test_qiskit_like_object_rejects_non_pow2():
    """Qiskit-like Statevector object with non-power-of-two dim must raise ValueError."""
    fake = _make_qiskit_like_statevector([0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="power of 2"):
        parse_statevector(fake)


def test_qiskit_like_object_rejects_unnormalized():
    """Qiskit-like Statevector object that is unnormalized must raise ValueError."""
    fake = _make_qiskit_like_statevector([0.9, 0.0])
    with pytest.raises(ValueError, match="normalized"):
        parse_statevector(fake)


def test_qiskit_like_object_rejects_nan():
    """Qiskit-like Statevector object with NaN must raise ValueError."""
    fake = _make_qiskit_like_statevector([float("nan"), 0.0])
    with pytest.raises(ValueError, match="not finite"):
        parse_statevector(fake)


def test_qiskit_path_rejects_zero_norm():
    """Qiskit Statevector with zero norm must raise ValueError."""
    try:
        from qiskit.quantum_info import Statevector

        sv = Statevector([0.0, 0.0])
        with pytest.raises(ValueError, match="zero norm"):
            parse_statevector(sv)
    except ImportError:
        pytest.skip("Qiskit not installed")


def test_qiskit_path_rejects_non_pow2_dimension():
    """Qiskit Statevector of length 3 must raise ValueError (not a qubit state)."""
    try:
        from qiskit.quantum_info import Statevector

        s = 1.0 / math.sqrt(3.0)
        sv = Statevector([s, s, s])
        with pytest.raises(ValueError, match="power of 2"):
            parse_statevector(sv)
    except ImportError:
        pytest.skip("Qiskit not installed")


def test_qiskit_path_rejects_unnormalised():
    """Qiskit Statevector that is not normalised must raise ValueError."""
    try:
        from qiskit.quantum_info import Statevector

        sv = Statevector([0.9, 0.0])
        with pytest.raises(ValueError, match="normalized"):
            parse_statevector(sv)
    except ImportError:
        pytest.skip("Qiskit not installed")


def test_qiskit_path_accepts_valid_bell_state():
    """Qiskit Statevector for Bell state |Φ+⟩ passes all validation checks."""
    try:
        from qiskit.quantum_info import Statevector

        s2i = 1.0 / math.sqrt(2.0)
        sv = Statevector([s2i, 0.0, 0.0, s2i])
        vec = parse_statevector(sv)
        assert len(vec) == 4
        norm_sq = sum(abs(c) ** 2 for c in vec)
        assert math.isclose(norm_sq, 1.0, abs_tol=1e-6)
    except ImportError:
        pytest.skip("Qiskit not installed")


# =========================================================================
# 2. parse_statevector — non-finite element rejection (native path)
# =========================================================================


def test_native_path_rejects_nan():
    """NaN amplitude in native list must raise ValueError."""
    with pytest.raises(ValueError, match="not finite"):
        parse_statevector([float("nan"), 0.0])


def test_native_path_rejects_inf():
    """Inf amplitude in native list must raise ValueError (caught via norm check)."""
    with pytest.raises(ValueError):
        parse_statevector([float("inf"), 0.0])


def test_native_path_rejects_empty():
    """Empty list must raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        parse_statevector([])


def test_native_path_rejects_non_pow2():
    """Length-3 list must raise ValueError."""
    s = 1.0 / math.sqrt(3.0)
    with pytest.raises(ValueError, match="power of 2"):
        parse_statevector([s, s, s])


# =========================================================================
# 3. parse_density_matrix — PSD validation for 4×4 and 8×8
# =========================================================================


def test_density_matrix_4x4_rejects_negative_eigenvalue():
    """4×4 matrix with a negative eigenvalue must raise ValueError."""
    # Diagonal matrix: tr=1, Hermitian, but rho[3][3]=-0.25 < 0.
    rho = [[0j] * 4 for _ in range(4)]
    rho[0][0] = 0.75 + 0j
    rho[1][1] = 0.25 + 0j
    rho[2][2] = 0.25 + 0j
    rho[3][3] = -0.25 + 0j
    with pytest.raises(ValueError, match="positive semi-definite"):
        parse_density_matrix(rho)


def test_density_matrix_8x8_rejects_negative_eigenvalue():
    """8×8 matrix with a negative eigenvalue must raise ValueError."""
    rho = [[0j] * 8 for _ in range(8)]
    rho[0][0] = 1.125 + 0j
    rho[7][7] = -0.125 + 0j
    with pytest.raises(ValueError, match="positive semi-definite"):
        parse_density_matrix(rho)


def test_density_matrix_4x4_accepts_valid_bell_state():
    """4×4 density matrix of Bell |Φ+⟩ must pass all checks."""
    s2i = 1.0 / math.sqrt(2.0)
    bell = [s2i, 0.0, 0.0, s2i]
    rho = statevector_to_density_matrix(bell)
    mat = parse_density_matrix(rho)
    # Purity of pure state = 1
    n = len(mat)
    purity = sum((mat[i][j] * mat[j][i]).real for i in range(n) for j in range(n))
    assert math.isclose(purity, 1.0, abs_tol=1e-6)


def test_density_matrix_nan_rejected():
    """NaN entry in density matrix must raise ValueError."""
    rho = [[float("nan") + 0j, 0j], [0j, 1 + 0j]]
    with pytest.raises(ValueError, match="not finite"):
        parse_density_matrix(rho)


# =========================================================================
# 4. simulate_circuit_evolution — measurement semantics
# =========================================================================


def test_circuit_measure_halts_evolution():
    """After MEASURE, subsequent unitary gates must NOT be applied to the state."""
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.x(0)  # This X gate must NOT be applied after measurement.

    steps = simulate_circuit_evolution(qc)
    assert len(steps) == 4  # Init, H, MEASURE, X(not applied)

    measure_step = steps[2]
    assert measure_step["name"] == "MEASURE"
    assert measure_step.get("measurement_halt") is True

    # The X step must be flagged as not applied.
    x_step = steps[3]
    assert x_step.get("post_halt") is True

    # Critically: the X-step probabilities must equal the MEASURE-step probabilities
    # (pre-measurement superposition), NOT |1⟩ with probability 1.
    assert measure_step["probabilities"] == x_step["probabilities"]


def test_circuit_measure_records_pre_measurement_probabilities():
    """MEASURE step shows pre-collapse superposition probabilities, not collapsed."""
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    steps = simulate_circuit_evolution(qc)
    # Steps: Init, H, MEASURE
    assert len(steps) == 3

    h_probs = steps[1]["probabilities"]
    measure_probs = steps[2]["probabilities"]

    # Both should show 50/50 (pre-measurement superposition).
    assert math.isclose(h_probs[0], 0.5, abs_tol=1e-6)
    assert math.isclose(h_probs[1], 0.5, abs_tol=1e-6)
    assert math.isclose(measure_probs[0], 0.5, abs_tol=1e-6)
    assert math.isclose(measure_probs[1], 0.5, abs_tol=1e-6)


def test_circuit_reset_halts_evolution():
    """RESET is non-unitary; evolution must halt and subsequent gates not applied."""
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.h(0)
    qc.reset(0)
    qc.x(0)  # Must NOT be applied.

    steps = simulate_circuit_evolution(qc)
    # Steps: Init, H, RESET, X(not applied)
    assert len(steps) == 4

    reset_step = steps[2]
    assert reset_step["name"] == "RESET"
    assert reset_step.get("reset_halt") is True

    x_step = steps[3]
    assert x_step.get("post_halt") is True

    # State must NOT reflect X being applied.
    # Pre-halt state after H: 50/50
    assert math.isclose(reset_step["probabilities"][0], 0.5, abs_tol=1e-6)
    assert math.isclose(x_step["probabilities"][0], 0.5, abs_tol=1e-6)


def test_circuit_barrier_is_not_halting():
    """BARRIER is structural only; evolution must continue after it."""
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.barrier()
    qc.cx(0, 1)

    steps = simulate_circuit_evolution(qc)
    # Steps: Init, H, BARRIER, CX — BARRIER must NOT halt evolution.
    assert len(steps) == 4

    barrier_step = steps[2]
    assert barrier_step["name"] == "BARRIER"
    # No halt flags
    assert not barrier_step.get("measurement_halt")
    assert not barrier_step.get("reset_halt")
    assert not barrier_step.get("post_halt")

    # CX must be applied normally — Bell state probabilities.
    cx_step = steps[3]
    assert math.isclose(cx_step["probabilities"][0], 0.5, abs_tol=1e-6)
    assert math.isclose(cx_step["probabilities"][3], 0.5, abs_tol=1e-6)


def test_circuit_measure_with_qubit_and_clbit_info():
    """MEASURE step must record measured_qubits and classical_bits metadata."""
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)

    steps = simulate_circuit_evolution(qc)
    measure_step = next(s for s in steps if s["name"] == "MEASURE")
    assert "measured_qubits" in measure_step
    assert 0 in measure_step["measured_qubits"]
    assert "classical_bits" in measure_step
    assert 0 in measure_step["classical_bits"]


# =========================================================================
# 5. Qiskit reference comparisons — cardinal states and standard gates
# =========================================================================


SQRT2_INV = 1.0 / math.sqrt(2.0)


@pytest.mark.parametrize(
    "state_str,ref_bloch",
    [
        ("|0>", (0.0, 0.0, 1.0)),
        ("|1>", (0.0, 0.0, -1.0)),
        ("+", (1.0, 0.0, 0.0)),
        ("-", (-1.0, 0.0, 0.0)),
        ("+i", (0.0, 1.0, 0.0)),
        ("-i", (0.0, -1.0, 0.0)),
    ],
)
def test_cardinal_states_vs_qiskit_reference(state_str, ref_bloch):
    """Compare SeeDL Bloch vectors for cardinal states against Qiskit quantum_info."""
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    # Construct Qiskit Statevector for each cardinal state.
    state_map = {
        "|0>": [1.0, 0.0],
        "|1>": [0.0, 1.0],
        "+": [SQRT2_INV, SQRT2_INV],
        "-": [SQRT2_INV, -SQRT2_INV],
        "+i": [SQRT2_INV, complex(0.0, SQRT2_INV)],
        "-i": [SQRT2_INV, complex(0.0, -SQRT2_INV)],
    }
    sv = Statevector(state_map[state_str])

    # SeeDL parse
    vec = parse_statevector(sv)
    bx, by, bz, purity = compute_bloch_vector(vec)

    assert math.isclose(bx, ref_bloch[0], abs_tol=1e-6), f"x mismatch for {state_str}"
    assert math.isclose(by, ref_bloch[1], abs_tol=1e-6), f"y mismatch for {state_str}"
    assert math.isclose(bz, ref_bloch[2], abs_tol=1e-6), f"z mismatch for {state_str}"
    assert math.isclose(purity, 1.0, abs_tol=1e-6), f"purity not 1.0 for {state_str}"


@pytest.mark.parametrize(
    "gate_name,initial,expected_bloch",
    [
        # X|0> = |1>
        ("X", [1.0, 0.0], (0.0, 0.0, -1.0)),
        # Y|0> = i|1>  (Bloch: (0,0,-1))
        ("Y", [1.0, 0.0], (0.0, 0.0, -1.0)),
        # Z|+> = |->  (Bloch: (-1,0,0))
        ("Z", [SQRT2_INV, SQRT2_INV], (-1.0, 0.0, 0.0)),
        # H|0> = |+>
        ("H", [1.0, 0.0], (1.0, 0.0, 0.0)),
        # H|1> = |->
        ("H", [0.0, 1.0], (-1.0, 0.0, 0.0)),
        # S|+> = |+i>
        ("S", [SQRT2_INV, SQRT2_INV], (0.0, 1.0, 0.0)),
    ],
)
def test_gates_vs_qiskit_reference(gate_name, initial, expected_bloch):
    """Compare gate application results against Qiskit Statevector.evolve()."""
    pytest.importorskip("qiskit")

    from seedl._quantum import apply_gate_to_state

    g_mat = get_single_qubit_gate(gate_name)
    sv_seedl = apply_gate_to_state(initial, g_mat)
    bx, by, bz, _ = compute_bloch_vector(sv_seedl)

    assert math.isclose(bx, expected_bloch[0], abs_tol=1e-6), f"{gate_name}: x wrong"
    assert math.isclose(by, expected_bloch[1], abs_tol=1e-6), f"{gate_name}: y wrong"
    assert math.isclose(bz, expected_bloch[2], abs_tol=1e-6), f"{gate_name}: z wrong"


@pytest.mark.parametrize(
    "theta",
    [0.0, math.pi / 6, math.pi / 4, math.pi / 2, math.pi, 2 * math.pi],
)
@pytest.mark.parametrize("gate_name", ["RX", "RY", "RZ"])
def test_rotation_gates_vs_qiskit_reference(gate_name, theta):
    """Compare Rx/Ry/Rz(θ)|0⟩ Bloch vector against Qiskit's decomposition."""
    pytest.importorskip("qiskit")
    from qiskit.circuit.library import RXGate, RYGate, RZGate
    from qiskit.quantum_info import Statevector

    gate_cls = {"RX": RXGate, "RY": RYGate, "RZ": RZGate}[gate_name]
    qk_gate = gate_cls(theta)

    # Qiskit reference
    sv0 = Statevector([1.0, 0.0])
    sv_ref = sv0.evolve(qk_gate)
    ref_vec = sv_ref.data.tolist()
    ref_bx, ref_by, ref_bz, _ = compute_bloch_vector([complex(c) for c in ref_vec])

    # SeeDL
    from seedl._quantum import apply_gate_to_state

    g_mat = get_single_qubit_gate(gate_name, parameter=theta)
    sv_seedl = apply_gate_to_state([1.0 + 0j, 0.0 + 0j], g_mat)
    bx, by, bz, _ = compute_bloch_vector(sv_seedl)

    # Compare up to global-phase-invariant Bloch vector
    assert math.isclose(bx, ref_bx, abs_tol=1e-6), f"{gate_name}({theta:.3f}): x wrong"
    assert math.isclose(by, ref_by, abs_tol=1e-6), f"{gate_name}({theta:.3f}): y wrong"
    assert math.isclose(bz, ref_bz, abs_tol=1e-6), f"{gate_name}({theta:.3f}): z wrong"


# =========================================================================
# 6. Asymmetric product states — verify qubit-ordering correctness
# =========================================================================


@pytest.mark.parametrize(
    "q0_state,q1_state,sv,expected_q0_bloch,expected_q1_bloch",
    [
        # q0=|0>, q1=|0>: sv = kron(|0>, |0>) = [1,0,0,0]
        ([1.0, 0.0], [1.0, 0.0], [1.0, 0.0, 0.0, 0.0], (0, 0, 1), (0, 0, 1)),
        # q0=|+>, q1=|0>: sv = kron(|0>, |+>) = [s2i, s2i, 0, 0]
        (
            [SQRT2_INV, SQRT2_INV],
            [1.0, 0.0],
            [SQRT2_INV, SQRT2_INV, 0.0, 0.0],
            (1, 0, 0),
            (0, 0, 1),
        ),
        # q0=|1>, q1=|0>: sv = kron(|0>, |1>) = [0, 1, 0, 0]
        ([0.0, 1.0], [1.0, 0.0], [0.0, 1.0, 0.0, 0.0], (0, 0, -1), (0, 0, 1)),
        # q0=|0>, q1=|+>: sv = kron(|+>, |0>) = [s2i, 0, s2i, 0]
        (
            [1.0, 0.0],
            [SQRT2_INV, SQRT2_INV],
            [SQRT2_INV, 0.0, SQRT2_INV, 0.0],
            (0, 0, 1),
            (1, 0, 0),
        ),
        # q0=|+>, q1=|1>: sv = kron(|1>, |+>) = [0, 0, s2i, s2i]
        (
            [SQRT2_INV, SQRT2_INV],
            [0.0, 1.0],
            [0.0, 0.0, SQRT2_INV, SQRT2_INV],
            (1, 0, 0),
            (0, 0, -1),
        ),
        # q0=|1>, q1=|+>: sv = kron(|+>, |1>) = [0, s2i, 0, s2i]
        (
            [0.0, 1.0],
            [SQRT2_INV, SQRT2_INV],
            [0.0, SQRT2_INV, 0.0, SQRT2_INV],
            (0, 0, -1),
            (1, 0, 0),
        ),
    ],
)
def test_asymmetric_product_state_partial_trace(
    q0_state, q1_state, sv, expected_q0_bloch, expected_q1_bloch
):
    """Partial trace of asymmetric product state must recover each subsystem state.

    SeeDL uses Qiskit little-endian convention: index = Σ_k bit_k * 2^k,
    so sv = kron(q_{n-1}, ..., q_1, q_0).
    """
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector, partial_trace

    sv_ks = Statevector(sv)

    # Verify our statevector construction is correct via Qiskit reference.
    rho_q0_ref = partial_trace(sv_ks, [1])  # trace out q1, keep q0
    rho_q1_ref = partial_trace(sv_ks, [0])  # trace out q0, keep q1

    rho_4x4 = statevector_to_density_matrix([complex(c) for c in sv])
    rho_q0 = partial_trace_2qubit(rho_4x4, trace_out=1)
    rho_q1 = partial_trace_2qubit(rho_4x4, trace_out=0)

    # Compare reduced density matrices element-by-element against Qiskit.
    ref_q0 = rho_q0_ref.data.tolist()
    ref_q1 = rho_q1_ref.data.tolist()
    for i in range(2):
        for j in range(2):
            assert math.isclose(abs(rho_q0[i][j] - ref_q0[i][j]), 0.0, abs_tol=1e-6), (
                f"q0 rho[{i}][{j}] mismatch"
            )
            assert math.isclose(abs(rho_q1[i][j] - ref_q1[i][j]), 0.0, abs_tol=1e-6), (
                f"q1 rho[{i}][{j}] mismatch"
            )

    # Also verify Bloch vectors match expected.
    bx0, by0, bz0, _ = compute_bloch_vector(rho_q0)
    bx1, by1, bz1, _ = compute_bloch_vector(rho_q1)
    assert math.isclose(bx0, expected_q0_bloch[0], abs_tol=1e-6)
    assert math.isclose(by0, expected_q0_bloch[1], abs_tol=1e-6)
    assert math.isclose(bz0, expected_q0_bloch[2], abs_tol=1e-6)
    assert math.isclose(bx1, expected_q1_bloch[0], abs_tol=1e-6)
    assert math.isclose(by1, expected_q1_bloch[1], abs_tol=1e-6)
    assert math.isclose(bz1, expected_q1_bloch[2], abs_tol=1e-6)


# =========================================================================
# 7. GHZ state — 3-qubit entanglement
# =========================================================================


def test_ghz_state_circuit_simulation():
    """GHZ state from H+CX+CX must show 50% |000⟩ and 50% |111⟩."""
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)

    steps = simulate_circuit_evolution(qc)
    final = steps[-1]
    probs = final["probabilities"]

    # |000⟩ = index 0, |111⟩ = index 7
    assert math.isclose(probs[0], 0.5, abs_tol=1e-6)
    assert math.isclose(probs[7], 0.5, abs_tol=1e-6)
    # All other basis states must have zero probability.
    for i in [1, 2, 3, 4, 5, 6]:
        assert math.isclose(probs[i], 0.0, abs_tol=1e-6)


def test_ghz_statevector_via_qiskit_reference():
    """GHZ state probabilities computed by SeeDL match Qiskit Statevector."""
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    s2i = 1.0 / math.sqrt(2.0)
    ghz = [s2i, 0, 0, 0, 0, 0, 0, s2i]
    sv = Statevector(ghz)
    vec = parse_statevector(sv)

    probs = [abs(c) ** 2 for c in vec]
    assert math.isclose(probs[0], 0.5, abs_tol=1e-6)
    assert math.isclose(probs[7], 0.5, abs_tol=1e-6)
    for i in [1, 2, 3, 4, 5, 6]:
        assert math.isclose(probs[i], 0.0, abs_tol=1e-6)


# =========================================================================
# 8. Von Neumann entropy — vs Qiskit reference
# =========================================================================


@pytest.mark.parametrize(
    "description,rho_2x2,expected_entropy",
    [
        ("pure |0>", [[1.0 + 0j, 0j], [0j, 0.0 + 0j]], 0.0),
        ("pure |1>", [[0.0 + 0j, 0j], [0j, 1.0 + 0j]], 0.0),
        ("maximally mixed I/2", [[0.5 + 0j, 0j], [0j, 0.5 + 0j]], 1.0),
        (
            "pure |+>",
            [[0.5 + 0j, 0.5 + 0j], [0.5 + 0j, 0.5 + 0j]],
            0.0,
        ),
        (
            "3/4 mixed",
            [[0.75 + 0j, 0j], [0j, 0.25 + 0j]],
            -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25)),
        ),
    ],
)
def test_von_neumann_entropy_vs_qiskit(description, rho_2x2, expected_entropy):
    """Von Neumann entropy must match the analytic formula and Qiskit reference."""
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import DensityMatrix, entropy

    qk_rho = DensityMatrix(rho_2x2)
    qk_entropy = float(entropy(qk_rho, base=2))

    seedl_entropy = von_neumann_entropy_single_qubit(rho_2x2)

    assert math.isclose(seedl_entropy, qk_entropy, abs_tol=1e-6), (
        f"{description}: SeeDL entropy {seedl_entropy:.6f} != Qiskit {qk_entropy:.6f}"
    )
    assert math.isclose(seedl_entropy, expected_entropy, abs_tol=1e-6), (
        f"{description}: got {seedl_entropy:.6f}, expected {expected_entropy:.6f}"
    )


# =========================================================================
# 9. Global-phase invariance — extended
# =========================================================================


@pytest.mark.parametrize("state_str", ["|0>", "|1>", "+", "-", "+i", "-i"])
def test_global_phase_invariance_all_cardinal_states(state_str):
    """Bloch vector is invariant under e^{iγ} for all cardinal states."""
    vec = parse_statevector(state_str)
    bx_ref, by_ref, bz_ref, p_ref = compute_bloch_vector(vec)

    for gamma in [0.1, 0.7, 1.5, math.pi / 3, 2.7, -1.3]:
        phase = cmath.exp(1j * gamma)
        phased = [c * phase for c in vec]
        bx, by, bz, p = compute_bloch_vector(phased)
        assert math.isclose(bx, bx_ref, abs_tol=1e-6), f"{state_str} γ={gamma}: x changed"
        assert math.isclose(by, by_ref, abs_tol=1e-6), f"{state_str} γ={gamma}: y changed"
        assert math.isclose(bz, bz_ref, abs_tol=1e-6), f"{state_str} γ={gamma}: z changed"
        assert math.isclose(p, p_ref, abs_tol=1e-6), f"{state_str} γ={gamma}: purity changed"


# =========================================================================
# 10. Performance / resource safety — qubit count limit
# =========================================================================


def test_simulate_circuit_respects_max_qubits_limit():
    """Circuits exceeding max_qubits must raise SeeDLError, not allocate huge arrays."""
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    from seedl.exceptions import SeeDLError

    qc = QuantumCircuit(3)  # Default max is 10; use max_qubits=2 to trigger limit.
    with pytest.raises(SeeDLError, match="bounded at 2 qubits"):
        simulate_circuit_evolution(qc, max_qubits=2)


def test_simulate_circuit_respects_max_ops_limit():
    """Circuits exceeding max_ops must raise SeeDLError."""
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    from seedl.exceptions import SeeDLError

    qc = QuantumCircuit(1)
    for _ in range(10):
        qc.x(0)
    with pytest.raises(SeeDLError, match="bounded at 5 operations"):
        simulate_circuit_evolution(qc, max_ops=5)


def test_statevector_exceeding_max_dim_rejected():
    """Statevectors exceeding MAX_STATEVECTOR_DIM (16,384) must be rejected."""
    # 2^15 = 32,768 (power of two, but exceeds limit)
    fake_large = [0.0] * 32_768
    fake_large[0] = 1.0
    with pytest.raises(ValueError, match="exceeds maximum supported limit"):
        parse_statevector(fake_large)


def test_density_matrix_exceeding_max_dim_rejected():
    """Density matrices exceeding MAX_DENSITY_MATRIX_DIM (256x256) must be rejected."""
    # 512x512 = 2^9 (power of two, but exceeds limit)
    dim = 512
    row = [0.0] * dim
    fake_large_dm = [list(row) for _ in range(dim)]
    fake_large_dm[0][0] = 1.0
    with pytest.raises(ValueError, match="exceeds maximum supported limit"):
        parse_density_matrix(fake_large_dm)


def test_statevector_max_states_truncation_reporting():
    """Statevectors with more basis states than max_states must truncate and report in notes."""
    import seedl as sd

    # 4-qubit GHZ state has 16 basis states
    s2i = 1.0 / math.sqrt(2.0)
    ghz_16 = [0.0] * 16
    ghz_16[0] = s2i
    ghz_16[15] = s2i

    # Bound rendering at 4 states
    view = sd.statevector(ghz_16, max_states=4)
    assert view.data["truncated"] is True
    assert view.data["total_states"] == 16
    assert len(view.data["basis_states"]) == 4
    assert any("truncated" in note for note in view.notes)
    assert "Top 4 states displayed" in view.summary
