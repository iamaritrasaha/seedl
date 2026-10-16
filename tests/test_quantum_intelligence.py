"""Tests for Quantum Intelligence 2.0 (Kraus noise, purity, entropy, negativity) in SeeDL 0.5.0."""

from __future__ import annotations

import math

import seedl as sd
from seedl._quantum_lab import (
    apply_kraus_channel,
    compute_bloch_vector_from_density_matrix,
    compute_coherence,
    compute_purity,
    compute_two_qubit_negativity,
    compute_von_neumann_entropy,
    density_matrix_from_statevector,
    get_noise_kraus_operators,
)


def _hermitian_transpose(mat: list[list[complex]]) -> list[list[complex]]:
    n = len(mat)
    m = len(mat[0]) if n > 0 else 0
    return [[mat[j][i].conjugate() for j in range(n)] for i in range(m)]


def _matrix_multiply(a: list[list[complex]], b: list[list[complex]]) -> list[list[complex]]:
    n = len(a)
    m = len(b[0]) if a else 0
    k_len = len(b)
    res = [[0.0 + 0.0j for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for k in range(k_len):
            aik = a[i][k]
            if aik != 0.0:
                for j in range(m):
                    res[i][j] += aik * b[k][j]
    return res


def test_kraus_channel_completeness():
    """Verify that all noise channels satisfy the Kraus completeness relation: sum_k K_k^dagger K_k = I."""
    channels = ["depolarizing", "dephasing", "amplitude_damping"]
    test_params = [0.0, 0.25, 0.5, 0.75, 1.0]

    for ch in channels:
        for p in test_params:
            kraus_ops = get_noise_kraus_operators(ch, p)
            identity_sum = [[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]]
            for K in kraus_ops:
                k_dag = _hermitian_transpose(K)
                prod = _matrix_multiply(k_dag, K)
                identity_sum[0][0] += prod[0][0]
                identity_sum[0][1] += prod[0][1]
                identity_sum[1][0] += prod[1][0]
                identity_sum[1][1] += prod[1][1]

            assert math.isclose(identity_sum[0][0].real, 1.0, abs_tol=1e-7)
            assert math.isclose(identity_sum[1][1].real, 1.0, abs_tol=1e-7)
            assert math.isclose(abs(identity_sum[0][1]), 0.0, abs_tol=1e-7)
            assert math.isclose(abs(identity_sum[1][0]), 0.0, abs_tol=1e-7)


def test_kraus_trace_preservation_and_hermiticity():
    """Applying Kraus channels preserves unit trace and Hermiticity."""
    # Start with pure state |+>
    inv = 1.0 / math.sqrt(2.0)
    rho = density_matrix_from_statevector([complex(inv, 0), complex(inv, 0)])

    channels = ["depolarizing", "dephasing", "amplitude_damping"]
    for ch in channels:
        kraus_ops = get_noise_kraus_operators(ch, 0.4)
        rho_out = apply_kraus_channel(rho, kraus_ops)

        # Trace preservation: Tr(rho') = 1.0
        tr = rho_out[0][0].real + rho_out[1][1].real
        assert math.isclose(tr, 1.0, abs_tol=1e-7)

        # Hermiticity: rho' == rho'^dagger
        assert math.isclose(rho_out[0][0].imag, 0.0, abs_tol=1e-7)
        assert math.isclose(rho_out[1][1].imag, 0.0, abs_tol=1e-7)
        assert math.isclose(rho_out[0][1].real, rho_out[1][0].real, abs_tol=1e-7)
        assert math.isclose(rho_out[0][1].imag, -rho_out[1][0].imag, abs_tol=1e-7)


def test_purity_and_entropy_extremes():
    """Verify purity and von Neumann entropy on pure and maximally mixed states."""
    # Pure state |0>
    rho_pure = density_matrix_from_statevector([1.0 + 0j, 0.0 + 0j])
    assert math.isclose(compute_purity(rho_pure), 1.0, abs_tol=1e-7)
    assert math.isclose(compute_von_neumann_entropy(rho_pure), 0.0, abs_tol=1e-7)

    # Completely mixed single-qubit state: rho = I / 2
    rho_mixed = [[0.5 + 0j, 0j], [0j, 0.5 + 0j]]
    assert math.isclose(compute_purity(rho_mixed), 0.5, abs_tol=1e-7)
    assert math.isclose(compute_von_neumann_entropy(rho_mixed), 1.0, abs_tol=1e-7)


def test_bloch_vector_and_coherence():
    """Verify Bloch coordinates and coherence l1-norm for |+> state."""
    # |+> state has Bloch vector (1, 0, 0)
    inv = 1.0 / math.sqrt(2.0)
    rho = density_matrix_from_statevector([complex(inv, 0), complex(inv, 0)])
    rx, ry, rz = compute_bloch_vector_from_density_matrix(rho)

    assert math.isclose(rx, 1.0, abs_tol=1e-6)
    assert math.isclose(ry, 0.0, abs_tol=1e-6)
    assert math.isclose(rz, 0.0, abs_tol=1e-6)

    # Coherence l1-norm of |+> is |rho_01| + |rho_10| = 0.5 + 0.5 = 1.0
    assert math.isclose(compute_coherence(rho), 1.0, abs_tol=1e-6)


def test_two_qubit_entanglement_negativity():
    """Verify Negativity is 0.0 for separable states and 0.5 for Bell state |Phi+>."""
    # Separable state |00> = |0> (x) |0>
    rho_sep = density_matrix_from_statevector([1.0 + 0j, 0j, 0j, 0j])
    neg_sep = compute_two_qubit_negativity(rho_sep)
    assert math.isclose(neg_sep, 0.0, abs_tol=1e-6)

    # Maximally entangled Bell state |Phi+> = (|00> + |11>) / sqrt(2)
    inv = 1.0 / math.sqrt(2.0)
    rho_bell = density_matrix_from_statevector([complex(inv, 0), 0j, 0j, complex(inv, 0)])
    neg_bell = compute_two_qubit_negativity(rho_bell)
    assert math.isclose(neg_bell, 0.5, abs_tol=1e-6)


def test_sd_quantum_noise_api_and_export():
    """sd.quantum_noise() produces a valid visualization and renders without error."""
    viz = sd.quantum_noise(channel="depolarizing", param=0.4, steps=5)
    assert viz.kind == "quantum_noise"
    assert "trajectory" in viz.data
    trajectory = viz.data["trajectory"]
    assert len(trajectory) == 6  # initial step + 5 steps

    # Check purity decreases under depolarizing noise
    purity_start = trajectory[0]["purity"]
    purity_end = trajectory[-1]["purity"]
    assert purity_start >= purity_end

    # Test SVG and HTML rendering
    from seedl._quantum_lab import quantum_noise_html, svg_quantum_noise

    html_markup = quantum_noise_html(viz)
    assert "<!DOCTYPE html>" in html_markup
    assert "Quantum Noise Explorer" in html_markup

    svg_markup = svg_quantum_noise(viz)
    assert "<svg" in svg_markup


def test_trace_distance_properties():
    """Verify trace distance D(rho, sigma) properties: symmetry, identity, orthogonality, bounds."""
    from seedl._quantum_lab import compute_trace_distance

    rho_0 = density_matrix_from_statevector([1.0 + 0j, 0.0 + 0j])  # |0>
    rho_1 = density_matrix_from_statevector([0.0 + 0j, 1.0 + 0j])  # |1>
    inv = 1.0 / math.sqrt(2.0)
    rho_plus = density_matrix_from_statevector([complex(inv, 0), complex(inv, 0)])  # |+>

    # Identity
    assert math.isclose(compute_trace_distance(rho_0, rho_0), 0.0, abs_tol=1e-7)

    # Orthogonal states have trace distance 1.0
    assert math.isclose(compute_trace_distance(rho_0, rho_1), 1.0, abs_tol=1e-7)

    # Symmetry
    d_0_plus = compute_trace_distance(rho_0, rho_plus)
    d_plus_0 = compute_trace_distance(rho_plus, rho_0)
    assert math.isclose(d_0_plus, d_plus_0, abs_tol=1e-7)
    # Between |0> (rz=1) and |+> (rx=1), Euclidean distance in Bloch sphere is sqrt(2), so D = sqrt(2)/2 ≈ 0.7071
    assert math.isclose(d_0_plus, 1.0 / math.sqrt(2.0), abs_tol=1e-6)

    # Bounds
    assert 0.0 <= d_0_plus <= 1.0


def test_maximally_mixed_state_negativity():
    """Maximally mixed 2-qubit state has exactly zero entanglement negativity."""
    # rho = I_4 / 4
    rho_mixed_4 = [[0.25 + 0j if i == j else 0j for j in range(4)] for i in range(4)]
    neg = compute_two_qubit_negativity(rho_mixed_4)
    assert math.isclose(neg, 0.0, abs_tol=1e-7)
