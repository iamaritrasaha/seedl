"""Quantum information mathematics and data model for SeeDL.

All major integrations (Qiskit, NumPy) remain optional.
Mathematical operations on pure and mixed states, Bloch vectors,
unitary gate evolution, partial traces, and entanglement entropy
are implemented with verified numerical precision.
"""

from __future__ import annotations

import cmath
import math
from typing import Any

from .exceptions import MissingDependencyError, SeeDLError

# Pauli matrices
PAULI_X = [[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]]
PAULI_Y = [[0.0 + 0.0j, 0.0 - 1.0j], [0.0 + 1.0j, 0.0 + 0.0j]]
PAULI_Z = [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]]
IDENTITY_2 = [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]]

SQRT2_INV = 1.0 / math.sqrt(2.0)


def _is_qiskit_circuit(obj: Any) -> bool:
    """Check if object is a Qiskit QuantumCircuit without requiring Qiskit at import."""
    cls_name = type(obj).__name__
    mod_name = getattr(type(obj), "__module__", "")
    return cls_name == "QuantumCircuit" and "qiskit" in mod_name


def _is_qiskit_state(obj: Any) -> bool:
    """Check if object is a Qiskit Statevector or DensityMatrix."""
    cls_name = type(obj).__name__
    mod_name = getattr(type(obj), "__module__", "")
    return cls_name in {"Statevector", "DensityMatrix"} and "qiskit" in mod_name


MAX_STATEVECTOR_DIM: int = 16_384  # 14 qubits
MAX_DENSITY_MATRIX_DIM: int = 256  # 8 qubits (256 x 256 = 65,536 elements)
MAX_CIRCUIT_OPS: int = 500


def _validate_statevector_vec(vec: list[complex]) -> list[complex]:
    """Apply all scientific invariants to an already-parsed list of complex amplitudes.

    Validates: non-empty, power-of-two dimension, finite elements, normalisation.
    Raises ValueError with a precise diagnostic on any violation.
    """
    if not vec:
        raise ValueError("Statevector cannot be empty.")

    n = len(vec)
    if (n & (n - 1)) != 0:
        raise ValueError(f"Statevector dimension ({n}) must be a power of 2 for qubit systems.")

    if n > MAX_STATEVECTOR_DIM:
        raise ValueError(
            f"Statevector dimension ({n}) exceeds maximum supported limit of "
            f"{MAX_STATEVECTOR_DIM} (14 qubits) to prevent memory exhaustion."
        )

    # Check for non-finite amplitudes (NaN / Inf)
    for i, c in enumerate(vec):
        if not (math.isfinite(c.real) and math.isfinite(c.imag)):
            raise ValueError(
                f"Statevector element [{i}] is not finite: {c!r}. "
                "NaN and Inf amplitudes are not physically valid."
            )

    norm_sq = sum(abs(c) ** 2 for c in vec)
    if norm_sq < 1e-12:
        raise ValueError("Statevector has zero norm and cannot represent a quantum state.")
    if abs(norm_sq - 1.0) > 1e-5:
        raise ValueError(
            f"Statevector is not normalized (sum of |c_i|² = {norm_sq:.6f} ≠ 1.0). "
            "All valid quantum states must satisfy ⟨ψ|ψ⟩ = 1."
        )
    return vec


def parse_statevector(state: Any) -> list[complex]:
    """Parse and validate a statevector from complex numbers, arrays, or Qiskit objects.

    Every supported input representation is subject to the same scientific invariants:
      - Non-empty.
      - Dimension is a power of 2 (qubit system).
      - All amplitudes are finite (no NaN or Inf).
      - State is normalised: sum(|c_i|²) ≈ 1.
    """
    if _is_qiskit_state(state):
        # Extract raw data from Qiskit Statevector, then apply the SAME validation
        # as the native path. Enforce dimension limits before expensive conversions.
        data = getattr(state, "data", state)
        if hasattr(data, "__len__") and len(data) > MAX_STATEVECTOR_DIM:
            raise ValueError(
                f"Statevector dimension ({len(data)}) exceeds maximum supported limit of "
                f"{MAX_STATEVECTOR_DIM} (14 qubits) to prevent memory exhaustion."
            )
        if hasattr(data, "tolist"):
            data = data.tolist()
        try:
            vec = [complex(c) for c in data]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Qiskit Statevector contains non-numeric amplitudes: {exc}") from exc
        return _validate_statevector_vec(vec)

    if isinstance(state, str):
        named = state.strip().lower()
        if named in {"0", "|0>", "zero"}:
            return [1.0 + 0.0j, 0.0 + 0.0j]
        if named in {"1", "|1>", "one"}:
            return [0.0 + 0.0j, 1.0 + 0.0j]
        if named in {"+", "|+>", "plus"}:
            return [SQRT2_INV + 0.0j, SQRT2_INV + 0.0j]
        if named in {"-", "|->", "minus"}:
            return [SQRT2_INV + 0.0j, -SQRT2_INV + 0.0j]
        if named in {"+i", "|+i>", "+j", "i"}:
            return [SQRT2_INV + 0.0j, complex(0.0, SQRT2_INV)]
        if named in {"-i", "|-i>", "-j"}:
            return [SQRT2_INV + 0.0j, complex(0.0, -SQRT2_INV)]
        raise ValueError(
            f"Unrecognized state label {state!r}. Use '0', '1', '+', '-', '+i', or '-i'."
        )

    if hasattr(state, "__len__") and len(state) > MAX_STATEVECTOR_DIM:
        raise ValueError(
            f"Statevector dimension ({len(state)}) exceeds maximum supported limit of "
            f"{MAX_STATEVECTOR_DIM} (14 qubits) to prevent memory exhaustion."
        )

    if hasattr(state, "tolist"):
        state = state.tolist()

    if isinstance(state, (list, tuple)):
        # Check if 1D statevector or 2D density matrix
        if state and isinstance(state[0], (list, tuple)):
            raise ValueError(
                "Provided input is a 2D matrix. Use parse_density_matrix or sd.density_matrix()."
            )
        if len(state) > MAX_STATEVECTOR_DIM:
            raise ValueError(
                f"Statevector dimension ({len(state)}) exceeds maximum supported limit of "
                f"{MAX_STATEVECTOR_DIM} (14 qubits) to prevent memory exhaustion."
            )
        try:
            vec = [complex(c) for c in state]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Statevector elements must be numeric complex numbers: {exc}"
            ) from exc
        return _validate_statevector_vec(vec)

    raise TypeError(
        f"Unsupported state input type {type(state).__name__}. "
        "Provide a 1D list/array of complex amplitudes, a Qiskit Statevector, or a named state string ('0', '+', etc.)."
    )


def statevector_to_density_matrix(vec: list[complex]) -> list[list[complex]]:
    """Compute pure state density matrix rho = |psi><psi|."""
    n = len(vec)
    return [[vec[i] * vec[j].conjugate() for j in range(n)] for i in range(n)]


def parse_density_matrix(rho: Any) -> list[list[complex]]:
    """Parse, validate, and check physical properties of a density matrix."""
    if _is_qiskit_state(rho):
        data = getattr(rho, "data", rho)
        if hasattr(data, "__len__") and len(data) > MAX_DENSITY_MATRIX_DIM:
            raise ValueError(
                f"Density matrix dimension ({len(data)}x{len(data)}) exceeds maximum supported limit of "
                f"{MAX_DENSITY_MATRIX_DIM}x{MAX_DENSITY_MATRIX_DIM} (8 qubits) to prevent resource exhaustion."
            )
        if hasattr(data, "tolist"):
            data = data.tolist()
        rho = data

    if hasattr(rho, "__len__") and len(rho) > MAX_DENSITY_MATRIX_DIM:
        raise ValueError(
            f"Density matrix dimension ({len(rho)}x{len(rho)}) exceeds maximum supported limit of "
            f"{MAX_DENSITY_MATRIX_DIM}x{MAX_DENSITY_MATRIX_DIM} (8 qubits) to prevent resource exhaustion."
        )

    if hasattr(rho, "tolist"):
        rho = rho.tolist()

    if not isinstance(rho, (list, tuple)) or not rho or not isinstance(rho[0], (list, tuple)):
        raise ValueError("Density matrix must be a 2D square matrix of complex numbers.")

    n = len(rho)
    if n > MAX_DENSITY_MATRIX_DIM:
        raise ValueError(
            f"Density matrix dimension ({n}x{n}) exceeds maximum supported limit of "
            f"{MAX_DENSITY_MATRIX_DIM}x{MAX_DENSITY_MATRIX_DIM} (8 qubits) to prevent resource exhaustion."
        )
    for row in rho:
        if len(row) != n:
            raise ValueError(f"Density matrix must be square (got shape {n} x {len(row)}).")

    if (n & (n - 1)) != 0:
        raise ValueError(f"Density matrix dimension ({n}) must be a power of 2.")

    mat: list[list[complex]] = []
    for r in range(n):
        row_c = []
        for c in range(n):
            try:
                val = complex(rho[r][c])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Matrix element [{r}][{c}] is not numeric: {exc}") from exc
            if not (math.isfinite(val.real) and math.isfinite(val.imag)):
                raise ValueError(
                    f"Matrix element [{r}][{c}] is not finite: {val!r}. "
                    "NaN and Inf are not valid density matrix entries."
                )
            row_c.append(val)
        mat.append(row_c)

    # Check Hermiticity: rho == rho^dagger
    for i in range(n):
        for j in range(i, n):
            diff = abs(mat[i][j] - mat[j][i].conjugate())
            if diff > 1e-5:
                raise ValueError(
                    f"Density matrix is not Hermitian at [{i}][{j}]: "
                    f"{mat[i][j]} != {mat[j][i].conjugate()} (diff = {diff:.2e})."
                )

    # Check Trace: Tr(rho) == 1
    tr = sum(mat[i][i].real for i in range(n))
    tr_imag = sum(abs(mat[i][i].imag) for i in range(n))
    if abs(tr - 1.0) > 1e-4 or tr_imag > 1e-4:
        raise ValueError(
            f"Density matrix trace must be approximately 1.0 (got Tr(rho) = {tr:.5f} + {tr_imag:.5f}i)."
        )

    # Positive semidefiniteness: all eigenvalues >= -eps.
    # Use NumPy's Hermitian eigensolver (eigvalsh) for all sizes; it exploits
    # the Hermitian structure and is numerically stable.  For 2x2 matrices we
    # could use the analytic formula, but eigvalsh is correct and consistent.
    try:
        import numpy as np

        arr = np.array(mat, dtype=complex)
        eigenvalues = np.linalg.eigvalsh(arr)  # real by Hermitian construction
        min_eig = float(eigenvalues.min())
        if min_eig < -1e-5:
            raise ValueError(
                f"Density matrix is not positive semi-definite "
                f"(minimum eigenvalue = {min_eig:.6f} < 0). "
                "Physical density operators must have non-negative eigenvalues."
            )
    except ImportError:
        # NumPy unavailable: fall back to analytic 2x2 formula.
        # For larger matrices without NumPy we cannot validate PSD — emit a note
        # but do not block.  NumPy is a dependency of the [quantum] extra.
        if n == 2:
            det = (mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]).real
            disc = tr * tr - 4.0 * det
            if disc < -1e-6:
                raise ValueError(  # noqa: B904
                    "Density matrix has non-real eigenvalues (complex discriminant)."
                ) from None
            disc = max(0.0, disc)
            eig1 = (tr + math.sqrt(disc)) / 2.0
            eig2 = (tr - math.sqrt(disc)) / 2.0
            if eig1 < -1e-5 or eig2 < -1e-5:
                raise ValueError(  # noqa: B904
                    f"Density matrix must be positive semi-definite (eigenvalues: {eig1:.4f}, {eig2:.4f})."
                ) from None

    return mat


def compute_bloch_vector(state: Any) -> tuple[float, float, float, float]:
    """Compute (x, y, z, purity) Bloch vector components from a single-qubit state.

    Calculates:
      x = Tr(rho * X)
      y = Tr(rho * Y)
      z = Tr(rho * Z)
      purity = Tr(rho^2)

    For pure states: x^2 + y^2 + z^2 = 1, purity = 1.
    For mixed states: x^2 + y^2 + z^2 < 1, purity < 1.
    """
    if isinstance(state, (list, tuple)) and state and isinstance(state[0], (list, tuple)):
        rho = parse_density_matrix(state)
        if len(rho) != 2:
            raise ValueError("Bloch sphere requires a single-qubit (2x2) density matrix.")
    elif hasattr(state, "data") and state.data.ndim == 2:
        rho = parse_density_matrix(state)
        if len(rho) != 2:
            raise ValueError("Bloch sphere requires a single-qubit (2x2) density matrix.")
    else:
        vec = parse_statevector(state)
        if len(vec) != 2:
            raise ValueError(
                f"Bloch sphere represents single-qubit states (dim 2). Provided state has dimension {len(vec)}."
            )
        rho = statevector_to_density_matrix(vec)

    # x = Tr(rho * X) = rho_01 + rho_10 = 2 * Re(rho_01)
    x = 2.0 * rho[0][1].real
    # y = Tr(rho * Y) = i * (rho_01 - rho_10) = -2 * Im(rho_01)
    y = -2.0 * rho[0][1].imag
    # z = Tr(rho * Z) = rho_00 - rho_11
    z = rho[0][0].real - rho[1][1].real

    # Purity = Tr(rho^2) = rho_00^2 + rho_11^2 + 2 * |rho_01|^2
    purity = (
        rho[0][0].real ** 2
        + rho[1][1].real ** 2
        + 2.0 * (rho[0][1].real ** 2 + rho[0][1].imag ** 2)
    )

    # Clamp tiny floating point errors
    x = 0.0 if abs(x) < 1e-12 else x
    y = 0.0 if abs(y) < 1e-12 else y
    z = 0.0 if abs(z) < 1e-12 else z

    return (x, y, z, purity)


def get_single_qubit_gate(name: str, parameter: float | None = None) -> list[list[complex]]:
    """Return the 2x2 complex unitary matrix for standard single-qubit gates."""
    gate_upper = name.strip().upper()
    if gate_upper == "X":
        return PAULI_X
    if gate_upper == "Y":
        return PAULI_Y
    if gate_upper == "Z":
        return PAULI_Z
    if gate_upper == "H":
        return [
            [SQRT2_INV + 0.0j, SQRT2_INV + 0.0j],
            [SQRT2_INV + 0.0j, -SQRT2_INV + 0.0j],
        ]
    if gate_upper == "S":
        return [
            [1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 1.0j],
        ]
    if gate_upper == "T":
        phase = cmath.exp(1j * math.pi / 4.0)
        return [
            [1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, phase],
        ]
    if gate_upper in {"RX", "RY", "RZ"}:
        theta = parameter if parameter is not None else math.pi / 2.0
        half = theta / 2.0
        c = math.cos(half)
        s = math.sin(half)
        if gate_upper == "RX":
            return [
                [complex(c, 0.0), complex(0.0, -s)],
                [complex(0.0, -s), complex(c, 0.0)],
            ]
        if gate_upper == "RY":
            return [
                [complex(c, 0.0), complex(-s, 0.0)],
                [complex(s, 0.0), complex(c, 0.0)],
            ]
        if gate_upper == "RZ":
            return [
                [cmath.exp(complex(0.0, -half)), 0.0 + 0.0j],
                [0.0 + 0.0j, cmath.exp(complex(0.0, half))],
            ]

    raise ValueError(
        f"Unsupported single-qubit gate {name!r}. Supported: X, Y, Z, H, S, T, Rx, Ry, Rz."
    )


def apply_gate_to_state(state: list[complex], gate_matrix: list[list[complex]]) -> list[complex]:
    """Apply a 2x2 unitary matrix to a 1-qubit statevector."""
    if len(state) != 2 or len(gate_matrix) != 2:
        raise ValueError("apply_gate_to_state requires single-qubit state and 2x2 gate.")
    out_0 = gate_matrix[0][0] * state[0] + gate_matrix[0][1] * state[1]
    out_1 = gate_matrix[1][0] * state[0] + gate_matrix[1][1] * state[1]
    return [out_0, out_1]


def apply_gate_to_density_matrix(
    rho: list[list[complex]], gate_matrix: list[list[complex]]
) -> list[list[complex]]:
    """Apply unitary U to density matrix: rho' = U rho U^dagger."""
    # First: M = U * rho
    m = [
        [gate_matrix[i][0] * rho[0][j] + gate_matrix[i][1] * rho[1][j] for j in range(2)]
        for i in range(2)
    ]
    # Second: out = M * U^dagger
    u_dag = [
        [gate_matrix[0][0].conjugate(), gate_matrix[1][0].conjugate()],
        [gate_matrix[0][1].conjugate(), gate_matrix[1][1].conjugate()],
    ]
    out = [[m[i][0] * u_dag[0][j] + m[i][1] * u_dag[1][j] for j in range(2)] for i in range(2)]
    return out


def apply_noise_channel(
    rho: list[list[complex]], channel: str, parameter: float
) -> list[list[complex]]:
    """Apply standard quantum noise channel to single-qubit density matrix."""
    if not 0.0 <= parameter <= 1.0:
        raise ValueError("Noise parameter must be between 0.0 and 1.0.")
    p = parameter
    ch = channel.strip().lower()

    if ch in {"dephasing", "phase_damping"}:
        # Shrinks off-diagonal coherences by (1 - p)
        return [
            [rho[0][0], rho[0][1] * (1.0 - p)],
            [rho[1][0] * (1.0 - p), rho[1][1]],
        ]
    if ch in {"depolarizing", "depolarization"}:
        # rho' = (1 - p)*rho + (p/2)*I
        return [
            [(1.0 - p) * rho[0][0] + p * 0.5, (1.0 - p) * rho[0][1]],
            [(1.0 - p) * rho[1][0], (1.0 - p) * rho[1][1] + p * 0.5],
        ]
    if ch in {"amplitude_damping", "damping"}:
        # K0 = [[1, 0], [0, sqrt(1 - p)]]
        # K1 = [[0, sqrt(p)], [0, 0]]
        # rho' = K0 rho K0^dagger + K1 rho K1^dagger
        k0_11 = math.sqrt(max(0.0, 1.0 - p))
        return [
            [rho[0][0] + p * rho[1][1], rho[0][1] * k0_11],
            [rho[1][0] * k0_11, (1.0 - p) * rho[1][1]],
        ]

    raise ValueError(
        f"Unsupported noise channel {channel!r}. Supported: dephasing, depolarizing, amplitude_damping."
    )


def partial_trace_2qubit(rho_4x4: list[list[complex]], trace_out: int) -> list[list[complex]]:
    """Compute reduced density matrix for a 2-qubit system.

    Parameters:
      rho_4x4: 4x4 density matrix.
      trace_out: 0 (trace out qubit 0 to get qubit 1) or 1 (trace out qubit 1 to get qubit 0).
      Qiskit convention: qubit 0 is the least significant bit (index: 2*q1 + q0).
    """
    if len(rho_4x4) != 4 or len(rho_4x4[0]) != 4:
        raise ValueError("partial_trace_2qubit requires a 4x4 density matrix.")

    reduced = [[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]]

    if trace_out == 1:
        # Trace out qubit 1 (most significant) -> leaves qubit 0
        for q0 in range(2):
            for q0_p in range(2):
                reduced[q0][q0_p] = sum(rho_4x4[2 * q1 + q0][2 * q1 + q0_p] for q1 in range(2))
    elif trace_out == 0:
        # Trace out qubit 0 (least significant) -> leaves qubit 1
        for q1 in range(2):
            for q1_p in range(2):
                reduced[q1][q1_p] = sum(rho_4x4[2 * q1 + q0][2 * q1_p + q0] for q0 in range(2))
    else:
        raise ValueError("trace_out must be 0 or 1 for a 2-qubit system.")

    return reduced


def von_neumann_entropy_single_qubit(rho_2x2: list[list[complex]]) -> float:
    """Calculate von Neumann entropy S(rho) = -Tr(rho log2 rho) for a 2x2 density matrix in bits."""
    tr = (rho_2x2[0][0] + rho_2x2[1][1]).real
    det = (rho_2x2[0][0] * rho_2x2[1][1] - rho_2x2[0][1] * rho_2x2[1][0]).real
    disc = max(0.0, tr * tr - 4.0 * det)
    eig1 = max(0.0, (tr + math.sqrt(disc)) / 2.0)
    eig2 = max(0.0, (tr - math.sqrt(disc)) / 2.0)

    entropy = 0.0
    for eig in (eig1, eig2):
        if eig > 1e-12:
            entropy -= eig * math.log2(eig)

    return float(entropy)


def inspect_bloch_state(state: Any) -> dict[str, Any]:
    """Inspect and extract complete mathematical representation for a single-qubit Bloch state."""
    # Determine pure vs mixed
    is_matrix = (
        isinstance(state, (list, tuple)) and state and isinstance(state[0], (list, tuple))
    ) or (hasattr(state, "data") and state.data.ndim == 2)

    if is_matrix:
        rho = parse_density_matrix(state)
        x, y, z, purity = compute_bloch_vector(rho)
        is_pure = abs(purity - 1.0) < 1e-4
        amplitudes = None
        global_phase = None
        prob_0 = rho[0][0].real
        prob_1 = rho[1][1].real
    else:
        vec = parse_statevector(state)
        rho = statevector_to_density_matrix(vec)
        x, y, z, purity = compute_bloch_vector(vec)
        is_pure = True
        amplitudes = [
            {"real": c.real, "imag": c.imag, "mag": abs(c), "phase": cmath.phase(c)} for c in vec
        ]
        prob_0 = abs(vec[0]) ** 2
        prob_1 = abs(vec[1]) ** 2
        global_phase = cmath.phase(vec[0]) if abs(vec[0]) > 1e-6 else cmath.phase(vec[1])

    r = math.sqrt(x * x + y * y + z * z)
    # Theta in [0, pi], Phi in [-pi, pi]
    theta = math.acos(max(-1.0, min(1.0, z / r))) if r > 1e-6 else 0.0
    phi = math.atan2(y, x) if r > 1e-6 else 0.0

    return dict(
        x=x,
        y=y,
        z=z,
        r=r,
        theta=theta,
        phi=phi,
        purity=purity,
        is_pure=is_pure,
        prob_0=prob_0,
        prob_1=prob_1,
        amplitudes=amplitudes,
        global_phase=global_phase,
        density_matrix=[[{"real": c.real, "imag": c.imag} for c in row] for row in rho],
    )


def inspect_circuit(circuit: Any) -> dict[str, Any]:
    """Inspect a Qiskit QuantumCircuit without executing expensive simulation."""
    if not _is_qiskit_circuit(circuit):
        raise TypeError(f"Expected a Qiskit QuantumCircuit, got {type(circuit).__name__}.")

    num_qubits = circuit.num_qubits
    num_clbits = getattr(circuit, "num_clbits", 0)
    depth = circuit.depth()
    num_ops = len(circuit.data)

    qubit_labels = [f"q_{i}" for i in range(num_qubits)]
    clbit_labels = [f"c_{i}" for i in range(num_clbits)]

    instructions: list[dict[str, Any]] = []
    qubit_timeline: list[list[int]] = [[] for _ in range(num_qubits)]

    for step_idx, operation in enumerate(circuit.data):
        # Qiskit circuit data item: CircuitInstruction(operation, qubits, clbits)
        op = getattr(
            operation, "operation", operation[0] if isinstance(operation, tuple) else operation
        )
        raw_qargs = getattr(
            operation, "qubits", operation[1] if isinstance(operation, tuple) else []
        )
        qargs = list(raw_qargs) if raw_qargs is not None else []
        raw_cargs = getattr(
            operation, "clbits", operation[2] if isinstance(operation, tuple) else []
        )
        cargs = list(raw_cargs) if raw_cargs is not None else []

        op_name = getattr(op, "name", type(op).__name__)
        params = [
            float(p) if isinstance(p, (int, float)) else str(p) for p in getattr(op, "params", [])
        ]

        target_qubits = [
            circuit.find_bit(q).index if hasattr(circuit, "find_bit") else circuit.qubits.index(q)
            for q in qargs
        ]
        target_clbits = [
            circuit.find_bit(c).index if hasattr(circuit, "find_bit") else circuit.clbits.index(c)
            for c in cargs
        ]

        inst_data = dict(
            step=step_idx,
            name=op_name,
            params=params,
            qubits=target_qubits,
            clbits=target_clbits,
            num_qubits=len(target_qubits),
            label=f"{op_name.upper()}"
            + (
                f"({', '.join(f'{p:.3g}' if isinstance(p, float) else str(p) for p in params)})"
                if params
                else ""
            ),
        )
        instructions.append(inst_data)

        for q in target_qubits:
            qubit_timeline[q].append(step_idx)

    return dict(
        num_qubits=num_qubits,
        num_clbits=num_clbits,
        depth=depth,
        num_ops=num_ops,
        qubit_labels=qubit_labels,
        clbit_labels=clbit_labels,
        instructions=instructions,
        timeline=qubit_timeline,
    )


def simulate_circuit_evolution(
    circuit: Any, max_qubits: int = 10, max_ops: int = 500, max_bytes: int = 32 * 1024 * 1024
) -> list[dict[str, Any]]:
    """Step-by-step statevector simulation of unitary operations in a QuantumCircuit."""
    if not _is_qiskit_circuit(circuit):
        raise TypeError("Simulation requires a Qiskit QuantumCircuit.")

    from ._recording import _positive

    _positive("max_qubits", max_qubits, 16)
    _positive("max_ops", max_ops, 10_000)
    _positive("max_bytes", max_bytes, 256 * 1024 * 1024)
    num_qubits = circuit.num_qubits
    if num_qubits > max_qubits:
        raise SeeDLError(
            f"Circuit simulation bounded at {max_qubits} qubits (circuit has {num_qubits})."
        )

    num_ops = len(circuit.data)
    if num_ops > max_ops:
        raise SeeDLError(
            f"Circuit simulation bounded at {max_ops} operations (circuit has {num_ops}). "
            "Set max_ops higher if you intentionally wish to simulate large circuits."
        )

    try:
        from qiskit.quantum_info import Statevector
    except ImportError as exc:
        raise MissingDependencyError(
            "Circuit simulation requires Qiskit. Install `pip install 'seedl[quantum]'`."
        ) from exc

    # Initial state |0...0>
    current_state = Statevector.from_int(0, 2**num_qubits)
    steps: list[dict[str, Any]] = []

    encoded_bytes = 0

    # Record Step 0: initial state
    def record_step(name: str, step_num: int, inst_desc: str) -> dict[str, Any]:
        data_vec = current_state.data.tolist()
        probs = [abs(c) ** 2 for c in data_vec]
        dim = len(data_vec)

        # Reduced 1-qubit Bloch vectors
        qubit_bloch = []
        for q in range(num_qubits):
            if num_qubits == 1:
                bv = compute_bloch_vector(data_vec)
            elif num_qubits == 2:
                rho = statevector_to_density_matrix([complex(c) for c in data_vec])
                # In Qiskit convention, qubit 0 is LSB, qubit 1 is MSB
                # To get qubit q, trace out (1 - q)
                red = partial_trace_2qubit(rho, trace_out=1 - q)
                bv = compute_bloch_vector(red)
            else:
                # For >2 qubits, use Qiskit's partial_trace if available
                try:
                    from qiskit.quantum_info import partial_trace

                    q_indices = [i for i in range(num_qubits) if i != q]
                    red_rho = partial_trace(current_state, q_indices).data.tolist()
                    bv = compute_bloch_vector(red_rho)
                except Exception as exc:
                    raise SeeDLError("Could not calculate a reduced qubit state.") from exc
            qubit_bloch.append(bv)

        # Basis state labels (e.g. |00>, |01>)
        basis_labels = [format(i, f"0{num_qubits}b") for i in range(dim)]
        recorded = dict(
            step=step_num,
            name=name,
            description=inst_desc,
            probabilities=probs,
            amplitudes=[
                {
                    "state": basis_labels[i],
                    "real": c.real,
                    "imag": c.imag,
                    "mag": abs(c),
                    "phase": cmath.phase(c),
                }
                for i, c in enumerate(data_vec)
            ],
            bloch_vectors=qubit_bloch,
        )
        return recorded

    def append_step(step: dict[str, Any]) -> None:
        nonlocal encoded_bytes
        import json

        halted = (bool(steps) and not steps[-1]["statevector_valid"]) or any(
            step.get(key)
            for key in (
                "measurement_halt",
                "reset_halt",
                "conditional_halt",
                "unsupported_halt",
                "post_halt",
            )
        )
        step["statevector_valid"] = not halted
        step["state_role"] = "last valid pre-halt state" if halted else "after operation"
        encoded_bytes += len(json.dumps(step, allow_nan=False).encode())
        if encoded_bytes > max_bytes:
            raise SeeDLError(
                "Circuit evolution exceeds max_bytes. Reduce max_qubits/max_ops or use a smaller circuit."
            )
        steps.append(step)

    append_step(record_step("Init", 0, "Initial state |" + "0" * num_qubits + ">"))

    # Track whether the statevector simulation has been halted by a non-unitary
    # instruction (measurement, reset, conditional).  Once halted, we record
    # the step but do NOT apply further unitary gates — doing so would operate on
    # an invalid (un-collapsed) state and produce scientifically misleading output.
    evolution_halted: bool = False
    halt_reason: str = ""

    # Non-unitary instruction names that collapse the statevector.
    # "barrier" is a purely structural hint and does NOT modify the state.
    _MEASUREMENT_OPS = {"measure"}
    _RESET_OPS = {"reset"}
    _CONDITIONAL_OPS = {"if_else", "c_if"}

    for idx, inst in enumerate(circuit.data):
        op = getattr(inst, "operation", inst[0] if isinstance(inst, tuple) else inst)
        raw_qargs = getattr(inst, "qubits", inst[1] if isinstance(inst, tuple) else [])
        qargs = list(raw_qargs) if raw_qargs is not None else []
        raw_cargs = getattr(inst, "clbits", inst[2] if isinstance(inst, tuple) else [])
        cargs = list(raw_cargs) if raw_cargs is not None else []
        op_name = getattr(op, "name", type(op).__name__).lower()

        q_ids = [
            circuit.find_bit(q).index if hasattr(circuit, "find_bit") else circuit.qubits.index(q)
            for q in qargs
        ]

        if op_name == "barrier":
            # Barrier is a structural hint with no physical effect on the state.
            append_step(
                record_step(
                    "BARRIER",
                    idx + 1,
                    f"Barrier (structural only, no state change) on qubits {q_ids}",
                )
            )
            continue

        if op_name in _MEASUREMENT_OPS:
            # Record the pre-measurement superposition probabilities, then halt.
            # Continuing unitary evolution after measurement would use a state that
            # has been classically collapsed, producing scientifically wrong results.
            c_ids = [
                circuit.find_bit(c).index
                if hasattr(circuit, "find_bit")
                else circuit.clbits.index(c)
                for c in cargs
            ]
            step = record_step(
                "MEASURE",
                idx + 1,
                (
                    f"Measurement of qubit(s) {q_ids} → classical bit(s) {c_ids}. "
                    "Pre-measurement superposition probabilities shown. "
                    "Unitary evolution halted: subsequent gates would operate on "
                    "a classically collapsed state that cannot be faithfully represented "
                    "by a single statevector."
                ),
            )
            step["measurement_halt"] = True
            step["measured_qubits"] = q_ids
            step["classical_bits"] = c_ids
            append_step(step)
            if not evolution_halted:
                evolution_halted = True
                halt_reason = f"MEASURE on qubit(s) {q_ids} at step {idx + 1}"
            continue

        if op_name in _RESET_OPS:
            # Reset collapses the qubit to |0⟩ non-unitarily.
            step = record_step(
                "RESET",
                idx + 1,
                (
                    f"Reset of qubit(s) {q_ids} to |0⟩. "
                    "This is a non-unitary operation. "
                    "Unitary evolution halted: reset discards quantum information "
                    "and cannot be represented by a unitary statevector evolution."
                ),
            )
            step["reset_halt"] = True
            step["reset_qubits"] = q_ids
            append_step(step)
            if not evolution_halted:
                evolution_halted = True
                halt_reason = f"RESET on qubit(s) {q_ids} at step {idx + 1}"
            continue

        # Check for conditional gates (c_if / if_else)
        condition = getattr(op, "condition", None)
        if condition is not None or op_name in _CONDITIONAL_OPS:
            step = record_step(
                op_name.upper(),
                idx + 1,
                (
                    f"Conditional gate {op_name.upper()} on qubit(s) {q_ids}. "
                    "Conditional operations depend on classical measurement outcomes "
                    "and cannot be simulated by pure statevector evolution. "
                    "Unitary evolution halted."
                ),
            )
            step["conditional_halt"] = True
            append_step(step)
            if not evolution_halted:
                evolution_halted = True
                halt_reason = f"conditional gate {op_name.upper()} at step {idx + 1}"
            continue

        # If evolution was already halted by a prior non-unitary op, record the
        # remaining steps but do NOT apply them to the state.
        if evolution_halted:
            step = record_step(
                op_name.upper(),
                idx + 1,
                (
                    f"Gate {op_name.upper()} on qubit(s) {q_ids} NOT applied — "
                    f"simulation was halted by {halt_reason}. "
                    "Shown probabilities reflect the pre-halt state."
                ),
            )
            step["post_halt"] = True
            step["halt_reason"] = halt_reason
            append_step(step)
            continue

        try:
            # Evolve statevector with the unitary instruction
            current_state = current_state.evolve(
                op,
                q_ids,
            )
        except Exception as exc:
            # If instruction is non-unitary or custom (e.g. parameterised gates
            # that Qiskit cannot automatically convert), record the step with a
            # diagnostic and halt.  Do NOT silently skip and continue evolution
            # on a potentially invalid state.
            step = record_step(
                op_name.upper(),
                idx + 1,
                (
                    f"Could not apply {op_name.upper()} to qubit(s) {q_ids}: {exc}. "
                    "This gate is unsupported, non-unitary, or requires unbound parameters. "
                    "Unitary evolution halted to avoid producing misleading results."
                ),
            )
            step["unsupported_halt"] = True
            step["error"] = str(exc)
            append_step(step)
            if not evolution_halted:
                evolution_halted = True
                halt_reason = f"unsupported gate {op_name.upper()} at step {idx + 1}"
            continue

        append_step(
            record_step(
                op_name.upper(),
                idx + 1,
                f"Apply {op_name.upper()} to qubit(s) {q_ids}",
            )
        )

    return steps
