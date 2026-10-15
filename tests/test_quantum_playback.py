"""Gate-by-gate state and reduced-qubit checks against quantum_info."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Pauli, Statevector, partial_trace

import seedl as sd
from seedl._playback import quantum_figure, quantum_pair
from seedl.exceptions import SeeDLError


@pytest.mark.parametrize("qubits", [1, 2, 3, 4])
def test_evolution_matches_qiskit_at_every_gate(qubits):
    qc = QuantumCircuit(qubits)
    for q in range(qubits):
        qc.h(q)
        qc.rx(0.13 * (q + 1), q)
        qc.ry(0.7, q)
        qc.rz(-0.37, q)
    for q in range(qubits - 1):
        qc.cx(q, q + 1)
    qc.barrier()
    view = sd.circuit(qc, simulate=True, backend="svg")
    prefix = QuantumCircuit(qubits)
    for index, step in enumerate(view.data["evolution"]):
        if index:
            instruction = qc.data[index - 1]
            prefix.append(instruction.operation, [qc.find_bit(q).index for q in instruction.qubits])
        expected = Statevector.from_instruction(prefix)
        actual = [complex(a["real"], a["imag"]) for a in step["amplitudes"]]
        np.testing.assert_allclose(actual, expected.data, atol=1e-12)
        np.testing.assert_allclose(step["probabilities"], expected.probabilities(), atol=1e-12)
        np.testing.assert_allclose(
            [a["phase"] for a in step["amplitudes"]], np.angle(expected.data), atol=1e-12
        )
        for q, vector in enumerate(step["bloch_vectors"]):
            reduced = partial_trace(expected, [i for i in range(qubits) if i != q])
            xyz = [reduced.expectation_value(Pauli(axis)).real for axis in ("X", "Y", "Z")]
            np.testing.assert_allclose(vector[:3], xyz, atol=1e-12)
        before, after = quantum_pair(view, index)
        assert before is not None and after is step
        assert step["statevector_valid"] is True


@pytest.mark.parametrize("halt", ["measure", "reset", "conditional", "unsupported"])
def test_no_invalid_state_is_displayed_after_halt(halt):
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    if halt == "measure":
        qc.measure(0, 0)
    elif halt == "reset":
        qc.reset(0)
    elif halt == "conditional":
        with qc.if_test((qc.clbits[0], True)):
            qc.x(0)
    else:
        qc.rx(Parameter("theta"), 0)
    qc.barrier()
    qc.x(0)
    view = sd.circuit(qc, simulate=True, backend="plotly")
    before, after = quantum_pair(view, 2)
    assert before is not None and after is None
    assert view.data["evolution"][2]["statevector_valid"] is False
    for index in (3, 4):
        assert quantum_pair(view, index) == (None, None)
        figure = quantum_figure(view, index)
        assert not figure.data  # Neither stale amplitudes nor a fake zero vector.
        assert "unavailable" in figure.layout.annotations[-1].text
    # Before the halt, both panes are valid.
    assert all(state is not None for state in quantum_pair(view, 1))


def test_quantum_payload_limit_and_static_fallback(tmp_path):
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    with pytest.raises(SeeDLError, match="max_bytes"):
        sd.circuit(qc, simulate=True, max_bytes=100, backend="svg")
    view = sd.circuit(qc, simulate=True, backend="plotly")
    assert "<svg" in view.to_figure("svg")
    source = view.to_html()
    assert "seedl-playback" in source
    assert source.count("plotly.js v") == 1
    assert "Before ·" in source or "Before \\u00b7" in source
