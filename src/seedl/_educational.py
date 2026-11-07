"""Interactive and mathematically accurate educational simulations of deep learning and classical ML concepts."""

from __future__ import annotations

import difflib
import math
from typing import Any

from ._vision import inspect_convolution
from .exceptions import SeeDLError

CLASSICAL_ML_CONCEPTS = (
    "linear_regression",
    "polynomial_regression",
    "gradient_descent",
    "logistic_regression",
    "knn",
    "decision_tree",
    "kmeans",
    "pca",
)

DEEP_LEARNING_CONCEPTS = (
    "backpropagation",
    "activation_functions",
    "loss_functions",
    "attention",
    "dropout",
    "batch_norm",
    "convolution",
)

EDUCATIONAL_GROUPS = {
    "Classical Machine Learning": CLASSICAL_ML_CONCEPTS,
    "Deep Learning": DEEP_LEARNING_CONCEPTS,
}

EDUCATIONAL_DESCRIPTIONS = {
    "linear_regression": "Ordinary least squares regression with analytical fit",
    "polynomial_regression": "Non-linear curve fitting via polynomial basis expansion",
    "gradient_descent": "First-order optimization on quadratic loss surface",
    "logistic_regression": "Binary classification and sigmoid probability mapping",
    "knn": "K-nearest neighbors instance-based classification",
    "decision_tree": "Recursive binary splitting via impurity reduction",
    "kmeans": "Iterative centroid assignment and cluster variance",
    "pca": "Orthogonal variance-maximizing dimensionality reduction",
    "backpropagation": "Analytical chain rule gradients across MLP graph",
    "activation_functions": "Curves and derivatives (ReLU, Sigmoid, Tanh, GELU)",
    "loss_functions": "Regression and classification losses (MSE, MAE, Huber)",
    "attention": "Scaled dot-product attention step-by-step",
    "dropout": "Inverted dropout masking and activation scaling",
    "batch_norm": "Mini-batch standardization and affine transformation",
    "convolution": "2D spatial convolution kernel and feature map",
}

SUPPORTED_CONCEPTS = set(CLASSICAL_ML_CONCEPTS) | set(DEEP_LEARNING_CONCEPTS)


def learn_concept(
    concept: str,
    **kwargs: Any,
) -> tuple[str, dict[str, Any], str, list[str]]:
    """Generate an educational visualization for a core deep learning or classical ML concept."""
    norm_concept = concept.lower().strip().replace(" ", "_").replace("-", "_")
    if norm_concept == "k_nearest_neighbors":
        norm_concept = "knn"

    if norm_concept not in SUPPORTED_CONCEPTS:
        matches = difflib.get_close_matches(norm_concept, sorted(SUPPORTED_CONCEPTS), n=1)
        hint = f" Did you mean '{matches[0]}'?" if matches else ""
        available = ", ".join(f"'{c}'" for c in sorted(SUPPORTED_CONCEPTS))
        raise SeeDLError(
            f"Unknown educational concept {concept!r}.{hint} Supported concepts: {available}."
        )

    if norm_concept == "linear_regression":
        return _learn_linear_regression(**kwargs)
    elif norm_concept == "polynomial_regression":
        return _learn_polynomial_regression(**kwargs)
    elif norm_concept == "gradient_descent":
        return _learn_gradient_descent(**kwargs)
    elif norm_concept == "logistic_regression":
        return _learn_logistic_regression(**kwargs)
    elif norm_concept == "knn":
        return _learn_knn(**kwargs)
    elif norm_concept == "decision_tree":
        return _learn_decision_tree(**kwargs)
    elif norm_concept == "kmeans":
        return _learn_kmeans(**kwargs)
    elif norm_concept == "pca":
        return _learn_pca(**kwargs)
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


# ---------------------------------------------------------------------------
# 1. Linear Regression Explorer
# ---------------------------------------------------------------------------


def _learn_linear_regression(
    slope: float = 1.8,
    intercept: float = 0.8,
    **kwargs: Any,
) -> tuple[str, dict[str, Any], str, list[str]]:
    """Explore simple linear regression, OLS optimality, residuals, and goodness-of-fit."""
    del kwargs
    try:
        slope = float(slope)
        intercept = float(intercept)
    except (TypeError, ValueError) as exc:
        raise SeeDLError(f"slope and intercept must be valid floats: {exc}") from exc
    if not (math.isfinite(slope) and math.isfinite(intercept)):
        raise SeeDLError("slope and intercept must be finite real numbers.")

    # Synthetic dataset with declared ground truth: y = 2.5*x + 1.2 + epsilon
    x_vals = [
        -2.0,
        -1.7,
        -1.4,
        -1.1,
        -0.8,
        -0.5,
        -0.2,
        0.1,
        0.4,
        0.7,
        1.0,
        1.3,
        1.6,
        1.9,
        2.2,
        2.5,
        2.8,
        3.1,
        3.4,
        3.7,
    ]
    noise = [
        0.42,
        -0.65,
        0.88,
        -0.32,
        0.15,
        -0.74,
        0.55,
        -0.18,
        0.62,
        -0.45,
        0.28,
        -0.81,
        0.35,
        -0.22,
        0.71,
        -0.58,
        0.44,
        -0.39,
        0.12,
        -0.04,
    ]
    y_vals = [round(2.5 * x + 1.2 + eps, 4) for x, eps in zip(x_vals, noise, strict=True)]
    n = len(x_vals)

    # Analytical OLS solution: w* = Cov(x, y) / Var(x), b* = y_bar - w* * x_bar
    x_bar = sum(x_vals) / n
    y_bar = sum(y_vals) / n
    ss_xx = sum((x - x_bar) ** 2 for x in x_vals)
    ss_xy = sum((x - x_bar) * (y - y_bar) for x, y in zip(x_vals, y_vals, strict=True))
    w_ols = ss_xy / ss_xx
    b_ols = y_bar - w_ols * x_bar

    # Predictions and residuals for user's candidate line
    y_pred = [round(slope * x + intercept, 4) for x in x_vals]
    residuals = [round(y - y_hat, 4) for y, y_hat in zip(y_vals, y_pred, strict=True)]

    # Metrics
    mse = sum(e * e for e in residuals) / n
    rmse = math.sqrt(mse)
    mae = sum(abs(e) for e in residuals) / n
    ss_tot = sum((y - y_bar) ** 2 for y in y_vals)
    ss_res = sum(e * e for e in residuals)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    # OLS optimal metrics
    y_ols_pred = [round(w_ols * x + b_ols, 4) for x in x_vals]
    ols_res = [y - yp for y, yp in zip(y_vals, y_ols_pred, strict=True)]
    ols_mse = sum(e * e for e in ols_res) / n
    ols_r2 = 1.0 - (sum(e * e for e in ols_res) / ss_tot)

    summary = (
        f"Linear Regression: y_hat = {slope:.2f}x + {intercept:.2f} "
        f"(MSE={mse:.4f}, R²={r2:.3f}) | OLS optimal: w*={w_ols:.2f}, b*={b_ols:.2f} (MSE*={ols_mse:.4f})"
    )

    notes = [
        "Objective: Minimize Mean Squared Error MSE = (1/N) * sum((y_i - (w*x_i + b))^2).",
        f"Candidate line parameters: slope w = {slope:.3f}, intercept b = {intercept:.3f}.",
        f"Analytical OLS closed-form fit: w* = {w_ols:.3f}, b* = {b_ols:.3f} (R² = {ols_r2:.3f}).",
        f"Fit diagnostics: MSE = {mse:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}, R² = {r2:.3f}.",
    ]
    if r2 >= 0.90:
        notes.append("High correlation: The candidate model captures over 90% of variance.")
    elif r2 < 0.50:
        notes.append(
            "Suboptimal fit: The candidate parameters deviate significantly from the OLS minimum."
        )

    data = {
        "concept": "linear_regression",
        "x": x_vals,
        "y": y_vals,
        "slope": slope,
        "intercept": intercept,
        "y_pred": y_pred,
        "residuals": residuals,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "ground_truth": {"slope": 2.5, "intercept": 1.2},
        "ols": {
            "slope": w_ols,
            "intercept": b_ols,
            "mse": ols_mse,
            "r2": ols_r2,
            "y_pred": y_ols_pred,
        },
    }

    return "educational_linear_regression", data, summary, notes


# ---------------------------------------------------------------------------
# 2. Polynomial Regression & Overfitting Explorer
# ---------------------------------------------------------------------------


def _solve_linear_system(a: list[list[float]], b: list[float]) -> list[float]:
    """Solve A x = b via Gaussian elimination with partial pivoting."""
    n = len(b)
    # Augmented matrix copy
    aug = [list(row) + [b[i]] for i, row in enumerate(a)]
    for i in range(n):
        # Pivot
        max_row = max(range(i, n), key=lambda r: abs(aug[r][i]))
        aug[i], aug[max_row] = aug[max_row], aug[i]
        diag = aug[i][i]
        if abs(diag) < 1e-12:
            diag = 1e-12  # regularize degenerate pivot
        # Normalize
        for j in range(i, n + 1):
            aug[i][j] /= diag
        # Eliminate
        for r in range(n):
            if r != i:
                factor = aug[r][i]
                for j in range(i, n + 1):
                    aug[r][j] -= factor * aug[i][j]
    return [aug[i][n] for i in range(n)]


def _fit_polynomial(
    xs: list[float], ys: list[float], degree: int, alpha: float = 0.0
) -> list[float]:
    """Fit polynomial coefficients w = [w0, w1, ..., wd] via normal equations (X^T X + alpha*I) w = X^T y."""
    m = degree + 1
    # Build X^T X and X^T y
    xtx = [[0.0] * m for _ in range(m)]
    xty = [0.0] * m
    for x, y in zip(xs, ys, strict=True):
        powers = [x**p for p in range(m)]
        for i in range(m):
            xty[i] += powers[i] * y
            for j in range(m):
                xtx[i][j] += powers[i] * powers[j]
    # L2 ridge penalty on non-intercept coefficients
    for i in range(1, m):
        xtx[i][i] += alpha
    return _solve_linear_system(xtx, xty)


def _eval_polynomial(w: list[float], x: float) -> float:
    return sum(coeff * (x**p) for p, coeff in enumerate(w))


def _learn_polynomial_regression(
    degree: int = 3,
    regularization: float = 0.0,
    **kwargs: Any,
) -> tuple[str, dict[str, Any], str, list[str]]:
    """Explore polynomial regression, model capacity, training vs validation error, and overfitting."""
    del kwargs
    if not isinstance(degree, int) or not (1 <= degree <= 10):
        raise SeeDLError("degree must be an integer between 1 and 10.")
    try:
        regularization = float(regularization)
    except (TypeError, ValueError) as exc:
        raise SeeDLError(f"regularization must be a valid float: {exc}") from exc
    if regularization < 0.0 or not math.isfinite(regularization):
        raise SeeDLError("regularization must be non-negative.")

    # Authentic synthetic dataset: ground truth y = sin(pi * x)
    x_train = [
        -0.95,
        -0.80,
        -0.65,
        -0.50,
        -0.35,
        -0.20,
        -0.05,
        0.10,
        0.25,
        0.40,
        0.55,
        0.70,
        0.85,
        0.98,
    ]
    noise_tr = [
        0.08,
        -0.12,
        0.15,
        -0.05,
        0.10,
        -0.08,
        0.05,
        -0.14,
        0.11,
        -0.07,
        0.13,
        -0.10,
        0.06,
        -0.09,
    ]
    y_train = [
        round(math.sin(math.pi * x) + eps, 4) for x, eps in zip(x_train, noise_tr, strict=True)
    ]

    x_val = [
        -0.88,
        -0.72,
        -0.58,
        -0.42,
        -0.28,
        -0.12,
        0.02,
        0.18,
        0.32,
        0.48,
        0.62,
        0.78,
        0.90,
        0.95,
    ]
    noise_vl = [
        -0.10,
        0.09,
        -0.07,
        0.12,
        -0.11,
        0.06,
        -0.05,
        0.08,
        -0.13,
        0.07,
        -0.09,
        0.11,
        -0.08,
        0.05,
    ]
    y_val = [round(math.sin(math.pi * x) + eps, 4) for x, eps in zip(x_val, noise_vl, strict=True)]

    # Fit requested model
    weights = _fit_polynomial(x_train, y_train, degree, regularization)
    y_tr_pred = [_eval_polynomial(weights, x) for x in x_train]
    y_vl_pred = [_eval_polynomial(weights, x) for x in x_val]

    mse_train = sum((y - p) ** 2 for y, p in zip(y_train, y_tr_pred, strict=True)) / len(y_train)
    mse_val = sum((y - p) ** 2 for y, p in zip(y_val, y_vl_pred, strict=True)) / len(y_val)

    # Generate smooth fitted curve
    curve_x = [round(-1.0 + i * 0.02, 2) for i in range(101)]
    curve_y = [_eval_polynomial(weights, x) for x in curve_x]
    ground_truth_y = [math.sin(math.pi * x) for x in curve_x]

    # Precalculate degree sweep (1 to 10) for error curve
    degree_sweep: list[dict[str, Any]] = []
    for d in range(1, 11):
        w_d = _fit_polynomial(x_train, y_train, d, regularization)
        tr_err = sum(
            (y - _eval_polynomial(w_d, x)) ** 2 for x, y in zip(x_train, y_train, strict=True)
        ) / len(y_train)
        vl_err = sum(
            (y - _eval_polynomial(w_d, x)) ** 2 for x, y in zip(x_val, y_val, strict=True)
        ) / len(y_val)
        degree_sweep.append(
            {
                "degree": d,
                "mse_train": round(tr_err, 5),
                "mse_val": round(min(vl_err, 50.0), 5),
            }
        )

    # Diagnostic regime
    if degree <= 2:
        regime = "Underfitting (High Bias)"
        regime_desc = (
            "Model complexity is too low to capture the true non-linear curvature of sin(pi * x)."
        )
    elif degree in {3, 4}:
        regime = "Balanced Fit (Optimal Trade-off)"
        regime_desc = (
            "Model captures the sine shape accurately with low validation generalization error."
        )
    else:
        regime = "Overfitting (High Variance)"
        regime_desc = f"Degree {degree} polynomial oscillates through noise points; validation MSE rises sharply."

    summary = (
        f"Polynomial Regression (degree={degree}, alpha={regularization}): "
        f"Train MSE={mse_train:.4f}, Val MSE={mse_val:.4f} [{regime}]"
    )

    notes = [
        "Ground truth signal: y = sin(pi * x) on [-1, 1] observed with Gaussian noise.",
        f"Current model degree: d = {degree} with L2 regularization alpha = {regularization:.2e}.",
        f"Empirical performance: Train MSE = {mse_train:.4f}, Validation MSE = {mse_val:.4f}.",
        f"Bias-variance diagnosis: {regime} - {regime_desc}",
    ]

    data = {
        "concept": "polynomial_regression",
        "degree": degree,
        "regularization": regularization,
        "weights": weights,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "y_train_pred": y_tr_pred,
        "y_val_pred": y_vl_pred,
        "mse_train": mse_train,
        "mse_val": mse_val,
        "curve_x": curve_x,
        "curve_y": curve_y,
        "ground_truth_y": ground_truth_y,
        "regime": regime,
        "degree_sweep": degree_sweep,
    }

    return "educational_polynomial_regression", data, summary, notes


# ---------------------------------------------------------------------------
# 3. Gradient Descent Laboratory (1D & 2D surfaces)
# ---------------------------------------------------------------------------


def _learn_gradient_descent(
    lr: float = 0.15,
    steps: int = 15,
    w_init: float | list[float] = 3.5,
    objective: str = "1d_convex",
    **kwargs: Any,
) -> tuple[str, dict[str, Any], str, list[str]]:
    """Simulate gradient descent on 1D convex, 1D double-well, or 2D anisotropic quadratic valley."""
    del kwargs
    if not isinstance(steps, int) or steps < 1:
        raise SeeDLError("steps must be a positive integer.")
    try:
        lr = float(lr)
    except (TypeError, ValueError) as exc:
        raise SeeDLError(f"lr must be a valid float: {exc}") from exc
    if lr <= 0 or not math.isfinite(lr):
        raise SeeDLError("lr must be a positive number.")

    objective = objective.lower().strip()
    if objective not in {"1d_convex", "1d_double_well", "2d_valley"}:
        raise SeeDLError("objective must be '1d_convex', '1d_double_well', or '2d_valley'.")

    trajectory: list[dict[str, Any]] = []

    if objective == "2d_valley":
        # Anisotropic quadratic valley: L(w1, w2) = 0.5 * (w1^2 + 10 * w2^2)
        # Hessian has eigenvalues 1.0 and 10.0. Max stable learning rate is 2 / 10 = 0.2.
        if isinstance(w_init, (list, tuple)) and len(w_init) >= 2:
            w1, w2 = float(w_init[0]), float(w_init[1])
        else:
            w1, w2 = 3.0, 1.5

        for step in range(steps + 1):
            loss = 0.5 * (w1 * w1 + 10.0 * w2 * w2)
            g1 = w1
            g2 = 10.0 * w2
            trajectory.append(
                {
                    "step": step,
                    "w": [round(w1, 5), round(w2, 5)],
                    "w1": round(w1, 5),
                    "w2": round(w2, 5),
                    "loss": loss,
                    "grad": [round(g1, 5), round(g2, 5)],
                }
            )
            w1 = w1 - lr * g1
            w2 = w2 - lr * g2
            # Bound numeric explosion
            if abs(w1) > 1e6 or abs(w2) > 1e6:
                break

        final_loss = trajectory[-1]["loss"]
        final_w = trajectory[-1]["w"]

        # Diagnostics for 2D valley:
        # L_max = 10.0 (steep axis), L_min = 1.0 (flat axis)
        if lr >= 0.2:
            diag_status = "divergence"
            diag_msg = "Alert: Learning rate lr >= 0.2 (2/L_max = 2/10), causing violent divergence along w2!"
        elif lr > 0.1:
            diag_status = "oscillation"
            diag_msg = "Notice: Oscillatory zigzagging observed across the steep valley walls (0.1 < lr < 0.2)."
        else:
            diag_status = "monotonic_convergence"
            diag_msg = "Convergence: Stable descent into the global minimum (0, 0)."

        summary = (
            f"Gradient Descent [2D Valley] (lr={lr}): w={trajectory[0]['w']} -> {final_w} "
            f"across {len(trajectory) - 1} steps (final loss: {final_loss:.4e})"
        )
        notes = [
            "Objective: Anisotropic valley L(w1, w2) = 0.5 * (w1^2 + 10 * w2^2) with condition number kappa = 10.",
            "Gradients: dL/dw1 = w1, dL/dw2 = 10*w2. Maximum stable learning rate: 2/L_max = 2/10 = 0.20.",
            f"Learning rate: {lr}. Trajectory: {len(trajectory)} evaluated points.",
            diag_msg,
        ]

    elif objective == "1d_double_well":
        # Non-convex double-well potential: L(w) = 0.25*w^4 - 0.5*w^2 + 0.25
        # dL/dw = w^3 - w. Minima at w = +/-1.0 (loss=0), local maximum / saddle at w=0 (loss=0.25)
        w_val = float(w_init if isinstance(w_init, (int, float)) else 1.8)
        for step in range(steps + 1):
            loss = 0.25 * (w_val**4) - 0.5 * (w_val**2) + 0.25
            grad = w_val**3 - w_val
            trajectory.append(
                {
                    "step": step,
                    "w": w_val,
                    "loss": loss,
                    "grad": grad,
                }
            )
            w_val = w_val - lr * grad
            if abs(w_val) > 1e6:
                break

        final_loss = trajectory[-1]["loss"]
        final_w = trajectory[-1]["w"]
        summary = (
            f"Gradient Descent [1D Double-Well] (lr={lr}): w={trajectory[0]['w']:.2f} -> {final_w:.4f} "
            f"across {len(trajectory) - 1} steps (final loss: {final_loss:.4e})"
        )
        notes = [
            "Objective: Non-convex double-well L(w) = 0.25*w^4 - 0.5*w^2 + 0.25 with minima at w = -1 and w = +1.",
            "Gradient: dL/dw = w^3 - w. Demonstrates sensitivity to initial position and saddle point at w = 0.",
            f"Learning rate: {lr}. Final coordinate: w = {final_w:.4f} with loss {final_loss:.4e}.",
        ]

    else:
        # 1D Convex quadratic: L(w) = 0.5 * w^2 with dL/dw = w
        w_val = float(w_init if isinstance(w_init, (int, float)) else 3.5)
        for step in range(steps + 1):
            loss = 0.5 * w_val * w_val
            grad = w_val
            trajectory.append({"step": step, "w": w_val, "loss": loss, "grad": grad})
            w_val = w_val - lr * grad
            if abs(w_val) > 1e6:
                break

        final_loss = trajectory[-1]["loss"]
        final_w = trajectory[-1]["w"]

        summary = (
            f"Gradient Descent Simulation (lr={lr}): w={trajectory[0]['w']:.2f} -> {final_w:.4f} "
            f"across {len(trajectory) - 1} steps (final loss: {final_loss:.4e})"
        )

        notes = [
            "Objective function: L(w) = 0.5 * w^2 with analytical gradient dL/dw = w (Lipschitz L=1.0).",
            f"Learning rate: {lr}. Update rule: w_new = w - lr * grad.",
        ]
        if lr >= 2.0:
            notes.append(
                "Alert: Learning rate is too high (lr >= 2.0 = 2/L), causing loss to diverge!"
            )
        elif lr > 1.0:
            notes.append("Notice: Oscillatory behavior observed (1.0 < lr < 2.0).")
        else:
            notes.append("Convergence: Stable monotonic reduction towards global minimum at w=0.")

    data = {
        "concept": "gradient_descent",
        "objective": objective,
        "lr": lr,
        "steps": steps,
        "trajectory": trajectory,
        "final_w": final_w,
        "final_loss": final_loss,
        "stability": diag_status
        if objective == "2d_valley"
        else (
            "divergence" if lr >= 2.0 else ("oscillation" if lr > 1.0 else "monotonic_convergence")
        ),
    }

    return "educational_gradient_descent", data, summary, notes


# ---------------------------------------------------------------------------
# 4. Logistic Regression & Decision Boundary Explorer
# ---------------------------------------------------------------------------


def _learn_logistic_regression(
    threshold: float = 0.5,
    w1: float = 1.4,
    w2: float = 1.1,
    b: float = -0.4,
    **kwargs: Any,
) -> tuple[str, dict[str, Any], str, list[str]]:
    """Explore logistic regression, Sigmoid mapping, decision boundary, and classification metrics."""
    del kwargs
    try:
        threshold = float(threshold)
        w1 = float(w1)
        w2 = float(w2)
        b = float(b)
    except (TypeError, ValueError) as exc:
        raise SeeDLError(f"Parameters must be valid numbers: {exc}") from exc
    if not (0.01 <= threshold <= 0.99):
        raise SeeDLError("threshold must be in the range [0.01, 0.99].")
    if not (math.isfinite(w1) and math.isfinite(w2) and math.isfinite(b)):
        raise SeeDLError("Weights and bias must be finite real numbers.")

    # 30 authentic 2D points (15 class 0, 15 class 1)
    points = [
        # Class 0 (centered around -1.0, -1.0)
        (-1.8, -1.5, 0),
        (-1.4, -0.8, 0),
        (-1.2, -1.9, 0),
        (-0.9, -1.1, 0),
        (-1.6, -0.4, 0),
        (-0.5, -1.6, 0),
        (-1.1, -1.4, 0),
        (-0.8, -0.7, 0),
        (-1.5, -1.7, 0),
        (-0.3, -1.0, 0),
        (-2.1, -1.2, 0),
        (-0.7, -1.8, 0),
        (-1.3, -0.6, 0),
        (-0.6, -1.3, 0),
        (-1.7, -2.0, 0),
        # Class 1 (centered around +1.0, +1.0)
        (1.8, 1.5, 1),
        (1.4, 0.8, 1),
        (1.2, 1.9, 1),
        (0.9, 1.1, 1),
        (1.6, 0.4, 1),
        (0.5, 1.6, 1),
        (1.1, 1.4, 1),
        (0.8, 0.7, 1),
        (1.5, 1.7, 1),
        (0.3, 1.0, 1),
        (2.1, 1.2, 1),
        (0.7, 1.8, 1),
        (1.3, 0.6, 1),
        (0.6, 1.3, 1),
        (1.7, 2.0, 1),
    ]

    samples: list[dict[str, Any]] = []
    tp, fp, tn, fn = 0, 0, 0, 0
    bce_sum = 0.0

    for x1, x2, y in points:
        z = w1 * x1 + w2 * x2 + b
        # Stable Sigmoid
        if z >= 0:
            prob = 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            prob = ez / (1.0 + ez)
        y_hat = 1 if prob >= threshold else 0

        # Loss: - [y * log(p) + (1-y) * log(1-p)]
        eps = 1e-12
        p_clipped = min(max(prob, eps), 1.0 - eps)
        loss_i = -(y * math.log(p_clipped) + (1.0 - y) * math.log(1.0 - p_clipped))
        bce_sum += loss_i

        if y == 1 and y_hat == 1:
            tp += 1
        elif y == 0 and y_hat == 1:
            fp += 1
        elif y == 0 and y_hat == 0:
            tn += 1
        else:
            fn += 1

        samples.append(
            {
                "x1": x1,
                "x2": x2,
                "y": y,
                "z": round(z, 4),
                "prob": round(prob, 4),
                "y_hat": y_hat,
            }
        )

    n = len(points)
    bce_loss = bce_sum / n
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Decision boundary line in 2D: w1 * x1 + w2 * x2 + b = logit(tau)
    logit_tau = math.log(threshold / (1.0 - threshold))
    boundary_points = []
    if abs(w2) > 1e-5:
        for x1_val in (-2.5, 2.5):
            x2_val = (logit_tau - b - w1 * x1_val) / w2
            boundary_points.append({"x1": x1_val, "x2": round(x2_val, 4)})

    # Inset Sigmoid curve
    sigmoid_curve = []
    for step in range(-60, 61):
        z_v = step * 0.1
        sig = 1.0 / (1.0 + math.exp(-z_v))
        sigmoid_curve.append({"z": round(z_v, 1), "p": round(sig, 4)})

    summary = (
        f"Logistic Regression (threshold={threshold:.2f}): "
        f"Accuracy={accuracy:.1%}, F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, BCE={bce_loss:.4f}"
    )

    notes = [
        "Model: P(Y=1|x) = sigma(w1*x1 + w2*x2 + b) = 1 / (1 + e^-(w^T x + b)).",
        f"Decision threshold tau = {threshold:.2f} maps to logit threshold logit(tau) = {logit_tau:.3f}.",
        f"Decision boundary equation: {w1:.2f}*x1 + {w2:.2f}*x2 + {b:.2f} = {logit_tau:.2f}.",
        f"Confusion matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn} | Binary Cross-Entropy = {bce_loss:.4f}.",
    ]

    data = {
        "concept": "logistic_regression",
        "threshold": threshold,
        "w1": w1,
        "w2": w2,
        "b": b,
        "logit_tau": logit_tau,
        "samples": samples,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "bce_loss": bce_loss,
        "boundary_points": boundary_points,
        "sigmoid_curve": sigmoid_curve,
    }

    return "educational_logistic_regression", data, summary, notes


# ---------------------------------------------------------------------------
# 5. K-Nearest Neighbors Explorer
# ---------------------------------------------------------------------------


def _learn_knn(
    k: int = 5,
    metric: str = "euclidean",
    weights: str = "uniform",
    query_point: tuple[float, float] | list[float] | None = None,
    **kwargs: Any,
) -> tuple[str, dict[str, Any], str, list[str]]:
    """Explore K-Nearest Neighbors, distance metrics, neighbor weighting, and decision voting."""
    del kwargs
    if not isinstance(k, int) or k < 1 or k > 25:
        raise SeeDLError("k must be an integer between 1 and 25.")
    metric = metric.lower().strip()
    if metric not in {"euclidean", "manhattan"}:
        raise SeeDLError("metric must be 'euclidean' or 'manhattan'.")
    weights = weights.lower().strip()
    if weights not in {"uniform", "distance"}:
        raise SeeDLError("weights must be 'uniform' or 'distance'.")

    if query_point is None:
        qx, qy = 0.2, 0.5
    else:
        try:
            qx, qy = float(query_point[0]), float(query_point[1])
        except (IndexError, TypeError, ValueError) as exc:
            raise SeeDLError(f"query_point must be a pair of floats: {exc}") from exc

    # Multi-class 2D synthetic dataset (3 classes, 10 samples each = 30 samples)
    points = [
        # Class 0: Coral / Red
        (-1.5, 1.2, 0),
        (-1.2, 0.8, 0),
        (-1.8, 0.5, 0),
        (-0.9, 1.4, 0),
        (-1.4, 1.6, 0),
        (-0.6, 0.9, 0),
        (-1.1, 0.3, 0),
        (-1.6, 0.9, 0),
        (-0.8, 1.7, 0),
        (-1.3, 1.1, 0),
        # Class 1: Sky / Blue
        (1.5, 1.2, 1),
        (1.2, 0.8, 1),
        (1.8, 0.5, 1),
        (0.9, 1.4, 1),
        (1.4, 1.6, 1),
        (0.6, 0.9, 1),
        (1.1, 0.3, 1),
        (1.6, 0.9, 1),
        (0.8, 1.7, 1),
        (1.3, 1.1, 1),
        # Class 2: Emerald / Green
        (0.0, -1.2, 2),
        (-0.4, -0.8, 2),
        (0.5, -1.5, 2),
        (-0.8, -1.4, 2),
        (0.3, -0.9, 2),
        (-0.2, -1.6, 2),
        (0.6, -1.1, 2),
        (-0.5, -1.1, 2),
        (0.1, -1.8, 2),
        (-0.1, -0.6, 2),
    ]

    # Calculate distance to query point
    distances: list[dict[str, Any]] = []
    for i, (x, y, cls) in enumerate(points):
        if metric == "euclidean":
            dist = math.sqrt((x - qx) ** 2 + (y - qy) ** 2)
        else:  # manhattan
            dist = abs(x - qx) + abs(y - qy)
        distances.append(
            {
                "id": i,
                "x": x,
                "y": y,
                "class": cls,
                "distance": round(dist, 4),
            }
        )

    # Sort and pick top k
    distances.sort(key=lambda d: d["distance"])
    neighbors = distances[:k]

    # Voting tally
    class_votes: dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0}
    for neighbor in neighbors:
        if weights == "uniform":
            w = 1.0
        else:  # distance weighted (inverse distance)
            w = 1.0 / (neighbor["distance"] + 1e-5)
        neighbor["weight"] = round(w, 4)
        class_votes[neighbor["class"]] += w

    total_votes = sum(class_votes.values())
    class_probs = {
        c: round(v / total_votes, 4) if total_votes > 0 else 0.0 for c, v in class_votes.items()
    }
    predicted_class = max(class_votes.keys(), key=lambda c: class_votes[c])

    class_names = ["Class 0 (Coral)", "Class 1 (Blue)", "Class 2 (Green)"]
    summary = (
        f"K-Nearest Neighbors (k={k}, metric={metric}, weights={weights}): "
        f"Query ({qx:.2f}, {qy:.2f}) -> {class_names[predicted_class]} (confidence: {class_probs[predicted_class]:.1%})"
    )

    notes = [
        f"Selected metric: {metric.capitalize()} distance with {weights} vote weighting.",
        f"Nearest neighbors evaluated: {k} points. Class distribution: { {class_names[c]: round(v, 2) for c, v in class_votes.items()} }.",
        "Curse of dimensionality: KNN relies on local density; in high dimensions, Euclidean distances concentrate and contrast collapses.",
        "Vital requirement: Feature scaling (e.g. StandardScaler) is strictly required when feature scales differ.",
    ]

    data = {
        "concept": "knn",
        "k": k,
        "metric": metric,
        "weights": weights,
        "query_point": [qx, qy],
        "points": points,
        "neighbors": neighbors,
        "class_votes": class_votes,
        "class_probabilities": class_probs,
        "predicted_class": predicted_class,
    }

    return "educational_knn", data, summary, notes


# ---------------------------------------------------------------------------
# 6. Decision Tree Learning Explorer
# ---------------------------------------------------------------------------


def _gini_impurity(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    return 1.0 - sum((c / total) ** 2 for c in counts)


def _entropy_impurity(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


def _learn_decision_tree(
    max_depth: int = 2,
    criterion: str = "gini",
    **kwargs: Any,
) -> tuple[str, dict[str, Any], str, list[str]]:
    """Explore decision tree recursive partitioning, Gini impurity, entropy, and information gain."""
    del kwargs
    if not isinstance(max_depth, int) or not (1 <= max_depth <= 4):
        raise SeeDLError("max_depth must be an integer between 1 and 4.")
    criterion = criterion.lower().strip()
    if criterion not in {"gini", "entropy"}:
        raise SeeDLError("criterion must be 'gini' or 'entropy'.")

    impurity_fn = _gini_impurity if criterion == "gini" else _entropy_impurity

    # 32 points in [0, 10] x [0, 10] with two classes
    # Structured with non-linear partitions: e.g. class 1 concentrated in [3, 8] x [2, 7]
    dataset = [
        (1.0, 2.0, 0),
        (2.0, 8.0, 0),
        (1.5, 5.0, 0),
        (2.5, 1.5, 0),
        (8.5, 8.0, 0),
        (9.0, 2.0, 0),
        (8.0, 1.0, 0),
        (9.2, 5.5, 0),
        (0.8, 9.0, 0),
        (8.8, 9.2, 0),
        (4.0, 9.0, 0),
        (6.0, 1.0, 0),
        (5.0, 0.5, 0),
        (5.5, 9.5, 0),
        (1.2, 7.0, 0),
        (8.2, 3.5, 0),
        # Class 1
        (4.0, 4.0, 1),
        (5.0, 5.0, 1),
        (6.0, 4.5, 1),
        (4.5, 6.0, 1),
        (6.5, 5.5, 1),
        (5.5, 3.5, 1),
        (3.5, 5.0, 1),
        (7.0, 4.0, 1),
        (4.8, 4.2, 1),
        (5.2, 6.2, 1),
        (6.2, 3.8, 1),
        (3.8, 3.2, 1),
        (6.8, 6.0, 1),
        (4.2, 5.8, 1),
        (5.8, 5.0, 1),
        (4.9, 4.9, 1),
    ]

    # Recursive Tree Builder
    node_counter = 0

    def build_tree(
        data: list[tuple[float, float, int]],
        depth: int,
        bounds: tuple[float, float, float, float],
    ) -> dict[str, Any]:
        nonlocal node_counter
        nid = node_counter
        node_counter += 1

        counts = [sum(1 for p in data if p[2] == 0), sum(1 for p in data if p[2] == 1)]
        current_imp = impurity_fn(counts)
        majority = 0 if counts[0] >= counts[1] else 1

        node: dict[str, Any] = {
            "id": nid,
            "depth": depth,
            "samples": len(data),
            "counts": counts,
            "impurity": round(current_imp, 4),
            "majority": majority,
            "bounds": list(bounds),  # [x_min, x_max, y_min, y_max]
        }

        # Stopping conditions
        if depth >= max_depth or len(data) <= 2 or current_imp == 0.0:
            node["leaf"] = True
            return node

        # Find best split across features x1 (0) and x2 (1)
        best_gain = -1.0
        best_split: dict[str, Any] | None = None

        for feat_idx in (0, 1):
            values = sorted({p[feat_idx] for p in data})
            # Midpoints as candidate thresholds
            for i in range(len(values) - 1):
                thresh = (values[i] + values[i + 1]) / 2.0
                left_pts = [p for p in data if p[feat_idx] <= thresh]
                right_pts = [p for p in data if p[feat_idx] > thresh]
                if not left_pts or not right_pts:
                    continue

                c_left = [
                    sum(1 for p in left_pts if p[2] == 0),
                    sum(1 for p in left_pts if p[2] == 1),
                ]
                c_right = [
                    sum(1 for p in right_pts if p[2] == 0),
                    sum(1 for p in right_pts if p[2] == 1),
                ]
                imp_left = impurity_fn(c_left)
                imp_right = impurity_fn(c_right)

                w_left = len(left_pts) / len(data)
                w_right = len(right_pts) / len(data)
                gain = current_imp - (w_left * imp_left + w_right * imp_right)

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        "feature": feat_idx,
                        "feature_name": "x1" if feat_idx == 0 else "x2",
                        "threshold": round(thresh, 3),
                        "gain": round(gain, 4),
                        "left_pts": left_pts,
                        "right_pts": right_pts,
                    }

        if best_split is None or best_split["gain"] <= 1e-5:
            node["leaf"] = True
            return node

        node["leaf"] = False
        node["feature"] = best_split["feature"]
        node["feature_name"] = best_split["feature_name"]
        node["threshold"] = best_split["threshold"]
        node["gain"] = best_split["gain"]

        # Child bounds
        xmin, xmax, ymin, ymax = bounds
        if best_split["feature"] == 0:
            b_left = (xmin, best_split["threshold"], ymin, ymax)
            b_right = (best_split["threshold"], xmax, ymin, ymax)
        else:
            b_left = (xmin, xmax, ymin, best_split["threshold"])
            b_right = (xmin, xmax, best_split["threshold"], ymax)

        node["left"] = build_tree(best_split["left_pts"], depth + 1, b_left)
        node["right"] = build_tree(best_split["right_pts"], depth + 1, b_right)
        return node

    root = build_tree(dataset, depth=0, bounds=(0.0, 10.0, 0.0, 10.0))

    # Collect all leaf regions for 2D visualization
    regions: list[dict[str, Any]] = []

    def collect_regions(n: dict[str, Any]) -> None:
        if n.get("leaf"):
            regions.append(
                {
                    "id": n["id"],
                    "bounds": n["bounds"],
                    "class": n["majority"],
                    "impurity": n["impurity"],
                    "samples": n["samples"],
                }
            )
        else:
            collect_regions(n["left"])
            collect_regions(n["right"])

    collect_regions(root)

    root_gain = root.get("gain", 0.0)
    root_imp = root["impurity"]

    summary = (
        f"Decision Tree (max_depth={max_depth}, criterion={criterion}): "
        f"Root impurity={root_imp:.3f}, primary split {root.get('feature_name', 'none')} <= {root.get('threshold', 0)} "
        f"(Info Gain Delta_I={root_gain:.3f}), leaves={len(regions)}"
    )

    notes = [
        f"Impurity metric: {criterion.capitalize()} (Root I = {root_imp:.3f}).",
        f"Recursive partitioning produced {len(regions)} orthogonal hyper-rectangular decision regions.",
        f"Root split on {root.get('feature_name', 'x1')} at threshold {root.get('threshold', 0)} yields Information Gain {root_gain:.3f}.",
        "Pruning / depth limits prevent memorization and excessive tree depth on noisy data.",
    ]

    data = {
        "concept": "decision_tree",
        "max_depth": max_depth,
        "criterion": criterion,
        "dataset": dataset,
        "tree": root,
        "regions": regions,
        "total_nodes": node_counter,
    }

    return "educational_decision_tree", data, summary, notes


# ---------------------------------------------------------------------------
# 7. KMeans Clustering Explorer
# ---------------------------------------------------------------------------


def _learn_kmeans(
    n_clusters: int = 3,
    init: str = "kmeans++",
    max_iter: int = 10,
    seed: int = 42,
    **kwargs: Any,
) -> tuple[str, dict[str, Any], str, list[str]]:
    """Explore KMeans clustering, Voronoi partitions, alternating centroid updates, and inertia (WCSS)."""
    del kwargs
    if not isinstance(n_clusters, int) or not (2 <= n_clusters <= 5):
        raise SeeDLError("n_clusters must be an integer between 2 and 5.")
    init = init.lower().strip()
    if init not in {"kmeans++", "random"}:
        raise SeeDLError("init must be 'kmeans++' or 'random'.")

    # 36 authentic 2D points from 3 true clusters
    points = [
        # Cluster A (-2, 2)
        (-2.2, 2.3),
        (-1.8, 1.9),
        (-2.5, 2.1),
        (-1.6, 2.6),
        (-2.0, 1.6),
        (-2.4, 2.7),
        (-1.7, 2.2),
        (-2.1, 2.5),
        (-1.9, 1.8),
        (-2.3, 2.0),
        (-1.5, 2.4),
        (-2.2, 1.7),
        # Cluster B (2, 2)
        (2.2, 2.3),
        (1.8, 1.9),
        (2.5, 2.1),
        (1.6, 2.6),
        (2.0, 1.6),
        (2.4, 2.7),
        (1.7, 2.2),
        (2.1, 2.5),
        (1.9, 1.8),
        (2.3, 2.0),
        (1.5, 2.4),
        (2.2, 1.7),
        # Cluster C (0, -2)
        (0.2, -1.8),
        (-0.3, -2.1),
        (0.5, -2.3),
        (-0.4, -1.7),
        (0.1, -2.4),
        (-0.2, -2.2),
        (0.4, -1.9),
        (-0.5, -2.0),
        (0.0, -2.5),
        (0.3, -2.1),
        (-0.1, -1.6),
        (0.2, -2.2),
    ]

    # Deterministic centroid initialization
    import random

    rng = random.Random(seed)

    if init == "random":
        initial_centroids = [list(pt) for pt in rng.sample(points, n_clusters)]
    else:  # kmeans++
        first_pt = points[0]
        initial_centroids = [list(first_pt)]
        while len(initial_centroids) < n_clusters:
            # Distance squared to nearest existing centroid
            d2_list = []
            for p in points:
                min_d2 = min((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 for c in initial_centroids)
                d2_list.append(min_d2)
            total_d2 = sum(d2_list)
            # Pick next centroid with probability proportional to D^2
            threshold_rnd = rng.uniform(0, total_d2)
            cum = 0.0
            chosen = points[-1]
            for p, d2 in zip(points, d2_list, strict=True):
                cum += d2
                if cum >= threshold_rnd:
                    chosen = p
                    break
            initial_centroids.append(list(chosen))

    centroids = [[round(c[0], 4), round(c[1], 4)] for c in initial_centroids]
    history: list[dict[str, Any]] = []

    # Iteration loop
    converged = False
    convergence_step = max_iter

    for step in range(max_iter):
        # 1. Assignment Step
        assignments: list[int] = []
        point_distances: list[float] = []
        wcss = 0.0

        for p in points:
            dists = [(p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 for c in centroids]
            best_k = min(range(n_clusters), key=lambda idx: dists[idx])
            assignments.append(best_k)
            point_distances.append(math.sqrt(dists[best_k]))
            wcss += dists[best_k]

        history.append(
            {
                "step": step,
                "centroids": [[round(c[0], 4), round(c[1], 4)] for c in centroids],
                "assignments": list(assignments),
                "inertia": round(wcss, 4),
            }
        )

        # 2. Centroid Update Step
        new_centroids: list[list[float]] = []
        max_shift = 0.0

        for k in range(n_clusters):
            assigned_pts = [p for p, a in zip(points, assignments, strict=True) if a == k]
            if assigned_pts:
                new_x = sum(p[0] for p in assigned_pts) / len(assigned_pts)
                new_y = sum(p[1] for p in assigned_pts) / len(assigned_pts)
            else:
                new_x, new_y = centroids[k]
            shift = math.sqrt((new_x - centroids[k][0]) ** 2 + (new_y - centroids[k][1]) ** 2)
            max_shift = max(max_shift, shift)
            new_centroids.append([round(new_x, 4), round(new_y, 4)])

        centroids = new_centroids
        if max_shift < 1e-4 and not converged:
            converged = True
            convergence_step = step + 1

    final_inertia = history[-1]["inertia"]

    summary = (
        f"KMeans Clustering (k={n_clusters}, init={init}): "
        f"Inertia WCSS={final_inertia:.3f} | Converged in {convergence_step} iterations"
    )

    notes = [
        "Objective: Minimize Within-Cluster Sum of Squares (WCSS Inertia) sum(||x_i - mu_c||^2).",
        f"Centroid initialization scheme: {init}. Number of clusters K = {n_clusters}.",
        f"Algorithm convergence: {'Achieved' if converged else 'Max iterations reached'} at step {convergence_step} with final inertia {final_inertia:.3f}.",
        "Sensitivity note: Standard KMeans assumes spherical, equally sized clusters; density variations or elongated shapes can lead to suboptimal local minima.",
    ]

    data = {
        "concept": "kmeans",
        "n_clusters": n_clusters,
        "init": init,
        "points": points,
        "initial_centroids": initial_centroids,
        "final_centroids": centroids,
        "history": history,
        "final_inertia": final_inertia,
        "converged": converged,
        "convergence_step": convergence_step,
    }

    return "educational_kmeans", data, summary, notes


# ---------------------------------------------------------------------------
# 8. Principal Component Analysis (PCA) Explorer
# ---------------------------------------------------------------------------


def _learn_pca(
    angle: float = 33.0,
    show_pc2: bool = True,
    show_reconstruction: bool = True,
    **kwargs: Any,
) -> tuple[str, dict[str, Any], str, list[str]]:
    """Explore Principal Component Analysis, covariance matrix, eigenvectors, variance maximization, and projection error."""
    del kwargs
    try:
        angle = float(angle)
    except (TypeError, ValueError) as exc:
        raise SeeDLError(f"angle must be a valid float: {exc}") from exc
    if not math.isfinite(angle):
        raise SeeDLError("angle must be a finite number.")

    # 24 authentic correlated 2D points (major axis along roughly 33 degrees)
    points = [
        (-2.8, -1.8),
        (-2.4, -1.2),
        (-2.0, -1.4),
        (-1.8, -0.8),
        (-1.5, -1.1),
        (-1.2, -0.6),
        (-1.0, -0.9),
        (-0.7, -0.3),
        (-0.4, -0.5),
        (-0.2, -0.1),
        (0.1, -0.2),
        (0.3, 0.2),
        (0.5, 0.0),
        (0.8, 0.5),
        (1.1, 0.4),
        (1.3, 0.9),
        (1.6, 0.7),
        (1.9, 1.2),
        (2.1, 1.0),
        (2.4, 1.5),
        (2.6, 1.3),
        (2.9, 1.8),
        (3.1, 1.7),
        (3.4, 2.1),
    ]
    n = len(points)

    # 1. Mean Centering
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    centered = [(p[0] - mean_x, p[1] - mean_y) for p in points]

    # 2. Sample Covariance Matrix Sigma = (1/(N-1)) * X_c^T X_c
    s_xx = sum(x * x for x, y in centered) / (n - 1)
    s_yy = sum(y * y for x, y in centered) / (n - 1)
    s_xy = sum(x * y for x, y in centered) / (n - 1)

    # 3. Analytical 2x2 Eigensystem
    # Characteristic equation: lambda^2 - Tr(Sigma)*lambda + Det(Sigma) = 0
    trace_sig = s_xx + s_yy
    det_sig = s_xx * s_yy - s_xy * s_xy
    delta = math.sqrt(max(0.0, (trace_sig / 2.0) ** 2 - det_sig))
    lambda1 = trace_sig / 2.0 + delta
    lambda2 = trace_sig / 2.0 - delta

    # Eigenvector for lambda1: orientation angle theta*
    # 2*s_xy / (s_xx - s_yy) = tan(2*theta)
    theta_opt_rad = 0.5 * math.atan2(2 * s_xy, s_xx - s_yy)
    theta_opt_deg = (math.degrees(theta_opt_rad)) % 180.0

    v1 = [math.cos(theta_opt_rad), math.sin(theta_opt_rad)]
    v2 = [-math.sin(theta_opt_rad), math.cos(theta_opt_rad)]

    total_var = lambda1 + lambda2
    ratio1 = lambda1 / total_var if total_var > 0 else 0.0
    ratio2 = lambda2 / total_var if total_var > 0 else 0.0

    # 4. Projections onto user's candidate axis at `angle`
    cand_rad = math.radians(angle)
    u_cand = [math.cos(cand_rad), math.sin(cand_rad)]

    projections: list[dict[str, Any]] = []
    proj_scalars: list[float] = []
    rec_errors_sq: list[float] = []

    for orig, cent in zip(points, centered, strict=True):
        # Scalar projection z_i = x_c * cos(theta) + y_c * sin(theta)
        z = cent[0] * u_cand[0] + cent[1] * u_cand[1]
        proj_scalars.append(z)
        # Reconstructed 2D point (in original coordinate frame)
        recon_x = mean_x + z * u_cand[0]
        recon_y = mean_y + z * u_cand[1]
        err_sq = (orig[0] - recon_x) ** 2 + (orig[1] - recon_y) ** 2
        rec_errors_sq.append(err_sq)
        projections.append(
            {
                "orig": [orig[0], orig[1]],
                "centered": [round(cent[0], 4), round(cent[1], 4)],
                "z": round(z, 4),
                "recon": [round(recon_x, 4), round(recon_y, 4)],
                "error_sq": round(err_sq, 4),
            }
        )

    proj_var = sum(z * z for z in proj_scalars) / (n - 1)
    mean_rec_error = sum(rec_errors_sq) / n

    # 5. Angular Sweep: Variance and Reconstruction Error vs Angle (0 to 180 degrees)
    angular_sweep: list[dict[str, Any]] = []
    for deg in range(0, 185, 5):
        rad = math.radians(deg)
        ux, uy = math.cos(rad), math.sin(rad)
        v_proj = sum((c[0] * ux + c[1] * uy) ** 2 for c in centered) / (n - 1)
        r_err = (
            sum(
                (c[0] - (c[0] * ux + c[1] * uy) * ux) ** 2
                + (c[1] - (c[0] * ux + c[1] * uy) * uy) ** 2
                for c in centered
            )
            / n
        )
        angular_sweep.append(
            {
                "angle": deg,
                "variance": round(v_proj, 4),
                "reconstruction_error": round(r_err, 4),
            }
        )

    summary = (
        f"PCA Explorer: PC1 optimal angle={theta_opt_deg:.1f}° "
        f"(explains {ratio1:.1%} variance, lambda1={lambda1:.3f}) | "
        f"Candidate axis angle={angle:.1f}° (projected variance={proj_var:.3f}, MSE_rec={mean_rec_error:.4f})"
    )

    notes = [
        f"Analytical Covariance Matrix: s_xx = {s_xx:.3f}, s_yy = {s_yy:.3f}, s_xy = {s_xy:.3f}.",
        f"Eigenvalues: lambda1 = {lambda1:.3f} ({ratio1:.1%}), lambda2 = {lambda2:.3f} ({ratio2:.1%}).",
        f"Principal Direction PC1: Angle theta* = {theta_opt_deg:.1f}° with unit vector v1 = [{v1[0]:.3f}, {v1[1]:.3f}].",
        "Fundamental Theorem of PCA: Maximizing projected variance is mathematically equivalent to minimizing orthogonal reconstruction error.",
    ]

    data = {
        "concept": "pca",
        "angle": angle,
        "show_pc2": show_pc2,
        "show_reconstruction": show_reconstruction,
        "mean": [round(mean_x, 4), round(mean_y, 4)],
        "covariance": {"s_xx": s_xx, "s_yy": s_yy, "s_xy": s_xy},
        "eigenvalues": [lambda1, lambda2],
        "explained_variance_ratio": [ratio1, ratio2],
        "optimal_angle": theta_opt_deg,
        "pc1_vector": v1,
        "pc2_vector": v2,
        "projected_variance": proj_var,
        "mean_reconstruction_error": mean_rec_error,
        "projections": projections,
        "angular_sweep": angular_sweep,
    }

    return "educational_pca", data, summary, notes


# ---------------------------------------------------------------------------
# Deep Learning Concepts (Existing)
# ---------------------------------------------------------------------------


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
    if not (0.0 <= p < 1.0):
        raise SeeDLError("Dropout probability p must be in the range [0.0, 1.0).")
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
