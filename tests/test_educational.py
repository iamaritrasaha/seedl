"""Tests for educational deep learning concept simulations (sd.learn)."""

from __future__ import annotations

import pytest

import seedl as sd
from seedl._educational import SUPPORTED_CONCEPTS, learn_concept
from seedl.exceptions import SeeDLError


@pytest.mark.parametrize("concept", sorted(SUPPORTED_CONCEPTS))
def test_all_supported_concepts_run_and_render_svg(concept):
    view = sd.learn(concept, backend="svg")
    assert view.kind.startswith("educational_") or view.kind in {
        "educational_gradient_descent",
        "educational_backpropagation",
        "educational_activations",
        "educational_loss",
        "educational_attention",
        "educational_dropout",
        "educational_batch_norm",
        "educational_convolution",
    }
    assert len(view.summary) > 0
    assert len(view.notes) > 0
    svg = view.to_figure()
    assert "<svg" in svg


def test_gradient_descent_mathematics():
    # L(w) = 0.5 * w^2 -> dL/dw = w
    # Update: w_{t+1} = w_t - lr * w_t = w_t * (1 - lr)
    # With lr = 0.2 and w_0 = 10.0:
    # w_1 = 8.0, w_2 = 6.4, w_3 = 5.12
    kind, data, summary, notes = learn_concept("gradient_descent", lr=0.2, steps=3, w_init=10.0)
    traj = data["trajectory"]
    assert len(traj) == 4
    assert abs(traj[0]["w"] - 10.0) < 1e-5
    assert abs(traj[1]["w"] - 8.0) < 1e-5
    assert abs(traj[2]["w"] - 6.4) < 1e-5
    assert abs(traj[3]["w"] - 5.12) < 1e-5


def test_backpropagation_chain_rule_mathematics():
    # Model: y = w1*x, z = relu(y), L = 0.5*(z - target)^2
    kind, data, summary, notes = learn_concept("backpropagation")
    nodes = data["nodes"]
    assert len(nodes) == 9
    # Check chain rule explanation note
    assert any("chain rule" in note.lower() for note in notes)


def test_activation_functions_mathematics():
    kind, data, summary, notes = learn_concept("activation_functions")
    funcs = data["functions"]

    # Find x=0 index
    x_vals = data["x"]
    idx_zero = min(range(len(x_vals)), key=lambda i: abs(x_vals[i] - 0.0))

    # ReLU(0) = 0
    assert abs(funcs["ReLU"][idx_zero] - 0.0) < 1e-4
    # Sigmoid(0) = 0.5
    assert abs(funcs["Sigmoid"][idx_zero] - 0.5) < 1e-4
    # Tanh(0) = 0.0
    assert abs(funcs["Tanh"][idx_zero] - 0.0) < 1e-4


def test_dropout_scaling_and_mask():
    kind, data, summary, notes = learn_concept("dropout", p=0.5)
    # Check scale factor: 1 / (1 - 0.5) = 2.0
    assert abs(data["scale"] - 2.0) < 1e-4
    # Expected value preservation note
    assert any("inverted dropout" in note.lower() for note in notes)


def test_batch_norm_standardization():
    kind, data, summary, notes = learn_concept("batch_norm")
    norm_vals = data["normalized"]
    # Mean of normalized values should be approximately 0.0
    mean_norm = sum(norm_vals) / len(norm_vals)
    assert abs(mean_norm) < 1e-4
    # Variance of normalized values should be approximately 1.0
    var_norm = sum((x - mean_norm) ** 2 for x in norm_vals) / len(norm_vals)
    assert abs(var_norm - 1.0) < 1e-2


def test_unknown_concept_error_lists_options():
    with pytest.raises(SeeDLError, match="Supported concepts"):
        sd.learn("quantum_entangled_attention")


def test_educational_invalid_parameters():
    # Dropout p out of range
    with pytest.raises(SeeDLError, match="Dropout probability"):
        sd.learn("dropout", p=-0.1)

    with pytest.raises(SeeDLError, match="Dropout probability"):
        sd.learn("dropout", p=1.0)

    # Gradient descent invalid steps
    with pytest.raises(SeeDLError, match="steps must be a positive integer"):
        sd.learn("gradient_descent", steps=0)
