"""Interactive and mathematically accurate educational simulations of deep learning concepts."""

from __future__ import annotations

import math
from typing import Any

from ._vision import inspect_convolution
from .exceptions import SeeDLError

SUPPORTED_CONCEPTS = {
    "gradient_descent",
    "backpropagation",
    "attention",
    "activation_functions",
    "loss_functions",
    "dropout",
    "batch_norm",
    "convolution",
}


def learn_concept(
    concept: str,
    **kwargs: Any,
) -> tuple[str, dict[str, Any], str, list[str]]:
    """Generate an educational visualization for a core deep learning concept."""
    norm_concept = concept.lower().strip().replace(" ", "_").replace("-", "_")
    if norm_concept not in SUPPORTED_CONCEPTS:
        available = ", ".join(f"'{c}'" for c in sorted(SUPPORTED_CONCEPTS))
        raise SeeDLError(
            f"Unknown educational concept {concept!r}. Supported concepts: {available}."
        )

    if norm_concept == "gradient_descent":
        return _learn_gradient_descent(**kwargs)
    elif norm_concept == "backpropagation":
        return _learn_backpropagation(**kwargs)
    elif norm_concept == "activation_functions":
        return _learn_activation_functions(**kwargs)
    elif norm_concept == "loss_functions":
        return _learn_loss_functions(**kwargs)
    elif norm_concept == "attention":
        return _learn_attention(**kwargs)
    elif norm_concept == "dropout":
        return _learn_dropout(**kwargs)
    elif norm_concept == "batch_norm":
        return _learn_batch_norm(**kwargs)
    else:  # convolution
        data, s, n = inspect_convolution(**kwargs)
        return "educational_convolution", data, s, n


def _learn_gradient_descent(
    lr: float = 0.15,
    steps: int = 15,
    w_init: float = 3.5,
) -> tuple[str, dict[str, Any], str, list[str]]:
    """Simulate gradient descent on quadratic loss surface L(w) = 0.5 * w^2."""
    w = float(w_init)
    trajectory: list[dict[str, float]] = []

    for step in range(steps + 1):
        loss = 0.5 * w * w
        grad = w  # dL/dw = w
        trajectory.append({"step": step, "w": w, "loss": loss, "grad": grad})
        w = w - lr * grad

    final_loss = trajectory[-1]["loss"]
    final_w = trajectory[-1]["w"]

    summary = (
        f"Gradient Descent Simulation (lr={lr}): w={w_init:.2f} -> {final_w:.4f} "
        f"across {steps} steps (final loss: {final_loss:.4e})"
    )

    notes = [
        "Objective function: L(w) = 0.5 * w^2 with gradient dL/dw = w.",
        f"Learning rate: {lr}. Update rule: w_new = w - lr * grad.",
    ]
    if lr >= 2.0:
        notes.append("Alert: Learning rate is too high (lr >= 2.0), causing loss to diverge!")
    elif lr > 1.0:
        notes.append("Notice: Oscillatory behavior observed (1.0 < lr < 2.0).")
    else:
        notes.append("Convergence: Stable monotonic reduction towards global minimum at w=0.")

    data = {
        "concept": "gradient_descent",
        "lr": lr,
        "steps": steps,
        "trajectory": trajectory,
        "final_w": final_w,
        "final_loss": final_loss,
    }

    return "educational_gradient_descent", data, summary, notes


def _learn_backpropagation(
    x: float = 2.0,
    w1: float = 1.5,
    b1: float = 0.2,
    w2: float = -0.8,
    b2: float = 0.5,
    target: float = 1.0,
) -> tuple[str, dict[str, Any], str, list[str]]:
    """Simulate analytical forward and backward pass through a 2-layer MLP node graph."""
    # Forward Pass
    z1 = w1 * x + b1
    a1 = max(0.0, z1)  # ReLU
    y_pred = w2 * a1 + b2
    loss = 0.5 * (y_pred - target) ** 2  # MSE

    # Backward Pass (Chain Rule)
    dL_dy = y_pred - target
    dL_dw2 = dL_dy * a1
    dL_db2 = dL_dy * 1.0
    dL_da1 = dL_dy * w2
    da1_dz1 = 1.0 if z1 > 0 else 0.0  # ReLU derivative
    dL_dz1 = dL_da1 * da1_dz1
    dL_dw1 = dL_dz1 * x
    dL_db1 = dL_dz1 * 1.0
    dL_dx = dL_dz1 * w1

    summary = f"Backpropagation Dataflow: Input x={x} -> Pred={y_pred:.4f} (target={target}) -> Loss={loss:.4f}"

    nodes = [
        {"name": "Input (x)", "forward": x, "gradient": dL_dx},
        {"name": "Linear 1 (z1 = w1*x + b1)", "forward": z1, "gradient": dL_dz1},
        {"name": "Weight 1 (w1)", "forward": w1, "gradient": dL_dw1},
        {"name": "Bias 1 (b1)", "forward": b1, "gradient": dL_db1},
        {"name": "ReLU Activation (a1)", "forward": a1, "gradient": dL_da1},
        {"name": "Linear 2 (y_pred)", "forward": y_pred, "gradient": dL_dy},
        {"name": "Weight 2 (w2)", "forward": w2, "gradient": dL_dw2},
        {"name": "Bias 2 (b2)", "forward": b2, "gradient": dL_db2},
        {"name": "Loss L = 0.5*(y - t)^2", "forward": loss, "gradient": 1.0},
    ]

    notes = [
        "Analytical chain rule verification: dL/dw1 = (dL/dy) * (dy/da1) * (da1/dz1) * (dz1/dw1).",
        f"Forward prediction: {y_pred:.4f}; Loss: {loss:.4f}.",
        f"Calculated gradients: dL/dw1 = {dL_dw1:.4f}, dL/dw2 = {dL_dw2:.4f}.",
    ]

    data = {
        "concept": "backpropagation",
        "nodes": nodes,
        "x": x,
        "target": target,
        "pred": y_pred,
        "loss": loss,
    }

    return "educational_backpropagation", data, summary, notes


def _learn_activation_functions() -> tuple[str, dict[str, Any], str, list[str]]:
    """Compute mathematical curves and derivatives for popular activation functions."""
    xs = [round(-4.0 + i * 0.1, 2) for i in range(81)]  # [-4, 4]
    funcs = {
        "ReLU": [max(0.0, x) for x in xs],
        "Sigmoid": [1.0 / (1.0 + math.exp(-x)) for x in xs],
        "Tanh": [math.tanh(x) for x in xs],
        "GELU": [
            0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x**3))))
            for x in xs
        ],
        "LeakyReLU": [x if x > 0 else 0.1 * x for x in xs],
    }

    derivatives = {
        "ReLU_grad": [1.0 if x > 0 else 0.0 for x in xs],
        "Sigmoid_grad": [
            (1.0 / (1.0 + math.exp(-x))) * (1.0 - (1.0 / (1.0 + math.exp(-x)))) for x in xs
        ],
        "Tanh_grad": [1.0 - math.tanh(x) ** 2 for x in xs],
    }

    summary = (
        "Mathematical Comparison of Activation Functions (ReLU, Sigmoid, Tanh, GELU, LeakyReLU)"
    )
    notes = [
        "Sigmoid & Tanh saturate at extreme inputs (|x| > 2), resulting in vanishing gradients (deriv -> 0).",
        "ReLU maintains a constant gradient of 1.0 for positive inputs, mitigating vanishing gradients.",
        "GELU provides a smooth, non-monotonic curve with negative curvature used in modern transformers.",
    ]

    data = {
        "concept": "activation_functions",
        "x": xs,
        "functions": funcs,
        "derivatives": derivatives,
    }

    return "educational_activations", data, summary, notes


def _learn_loss_functions() -> tuple[str, dict[str, Any], str, list[str]]:
    """Compare loss curves for regression and classification."""
    errors = [round(-3.0 + i * 0.1, 2) for i in range(61)]  # [-3, 3]
    mse = [0.5 * e * e for e in errors]
    mae = [abs(e) for e in errors]
    huber = [0.5 * e * e if abs(e) <= 1.0 else abs(e) - 0.5 for e in errors]

    summary = "Loss Functions: Mean Squared Error (MSE), Mean Absolute Error (MAE), Huber Loss"
    notes = [
        "MSE heavily penalizes large outlier errors quadratically, but has weak gradients for small errors.",
        "MAE has constant gradient magnitude (robust to outliers), but is non-smooth at 0.",
        "Huber loss combines quadratic behavior for small errors (|e| <= 1) and linear behavior for large errors.",
    ]

    data = {
        "concept": "loss_functions",
        "errors": errors,
        "mse": mse,
        "mae": mae,
        "huber": huber,
    }

    return "educational_losses", data, summary, notes


def _learn_attention() -> tuple[str, dict[str, Any], str, list[str]]:
    """Simulate scaled dot-product attention calculation step-by-step."""
    # 3 query tokens, 3 key tokens in d_k = 2
    q = [[1.0, 0.0], [0.5, 0.8], [0.0, 1.0]]
    k = [[1.0, 0.2], [0.8, 0.5], [0.2, 1.0]]
    d_k = 2.0
    scale = math.sqrt(d_k)

    # Dot product Q * K^T
    raw_scores = []
    scaled_scores = []
    softmax_weights = []

    for qi in q:
        raw_row = [qi[0] * ki[0] + qi[1] * ki[1] for ki in k]
        scaled_row = [s / scale for s in raw_row]
        # softmax
        max_s = max(scaled_row)
        exps = [math.exp(s - max_s) for s in scaled_row]
        sum_e = sum(exps)
        prob_row = [e / sum_e for e in exps]

        raw_scores.append(raw_row)
        scaled_scores.append(scaled_row)
        softmax_weights.append(prob_row)

    summary = "Scaled Dot-Product Attention: Softmax(Q K^T / sqrt(d_k))"
    notes = [
        f"Scaling factor: 1/sqrt(d_k) = 1/sqrt({d_k}) = {1.0 / scale:.3f} prevents extreme softmax saturation.",
        "Rows of the attention weight matrix represent normalized query-key probabilities summing to 1.0.",
    ]

    data = {
        "concept": "attention",
        "raw_scores": raw_scores,
        "scaled_scores": scaled_scores,
        "weights": softmax_weights,
        "d_k": d_k,
    }

    return "educational_attention", data, summary, notes


def _learn_dropout(p: float = 0.5) -> tuple[str, dict[str, Any], str, list[str]]:
    """Demonstrate inverted dropout unit masking and scaling."""
    inputs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    # Deterministic mask pattern for demonstration: alternate drops
    mask = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    scale = 1.0 / (1.0 - p) if p < 1.0 else 0.0
    train_outputs = [x * m * scale for x, m in zip(inputs, mask, strict=True)]
    eval_outputs = list(inputs)

    summary = f"Inverted Dropout (p={p}): Active scaling factor = 1/(1-p) = {scale:.2f}"
    notes = [
        f"Dropout probability: p={p}. Inverted dropout scales retained activations by {scale:.2f} during training.",
        "Expected value of activations is preserved between training and evaluation without inference-time modification.",
    ]

    data = {
        "concept": "dropout",
        "p": p,
        "scale": scale,
        "inputs": inputs,
        "mask": mask,
        "train_outputs": train_outputs,
        "eval_outputs": eval_outputs,
    }

    return "educational_dropout", data, summary, notes


def _learn_batch_norm() -> tuple[str, dict[str, Any], str, list[str]]:
    """Demonstrate mini-batch normalization forward pass."""
    batch = [2.0, 4.0, 6.0, 8.0, 10.0]
    n = len(batch)
    mean = sum(batch) / float(n)
    var = sum((x - mean) ** 2 for x in batch) / float(n)
    std = math.sqrt(var + 1e-5)
    normalized = [(x - mean) / std for x in batch]
    gamma, beta = 1.5, 0.5
    transformed = [gamma * norm + beta for norm in normalized]

    summary = (
        f"Batch Normalization: Batch Mean={mean:.2f}, Variance={var:.2f} -> Norm(0, 1) -> Affine"
    )
    notes = [
        "Normalizes layer inputs to zero mean and unit variance across the mini-batch.",
        f"Learnable affine parameters gamma={gamma} and beta={beta} restore representational capacity.",
    ]

    data = {
        "concept": "batch_norm",
        "batch": batch,
        "mean": mean,
        "variance": var,
        "normalized": normalized,
        "output": transformed,
        "gamma": gamma,
        "beta": beta,
    }

    return "educational_batch_norm", data, summary, notes
