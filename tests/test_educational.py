"""Tests for educational deep learning and classical machine learning simulations (sd.learn)."""

from __future__ import annotations

import subprocess
import sys

import pytest

import seedl as sd
from seedl._educational import SUPPORTED_CONCEPTS, learn_concept
from seedl.exceptions import SeeDLError


@pytest.mark.parametrize("concept", sorted(SUPPORTED_CONCEPTS))
def test_all_supported_concepts_run_and_render_svg(concept):
    view = sd.learn(concept, backend="svg")
    assert view.kind.startswith("educational_")
    assert len(view.summary) > 0
    assert len(view.notes) > 0
    svg = view.to_figure()
    assert "<svg" in svg
    assert "</svg>" in svg


@pytest.mark.parametrize("concept", sorted(SUPPORTED_CONCEPTS))
def test_all_supported_concepts_export_html(concept, tmp_path):
    view = sd.learn(concept)
    html_content = view.to_html()
    assert "<!DOCTYPE html>" in html_content
    assert "Google Sans" in html_content
    assert "JetBrains Mono" in html_content
    assert "data:font/woff2;base64" in html_content
    assert "</script>" in html_content

    # Save to disk
    out_file = tmp_path / f"{concept}.html"
    view.save(out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 1000


def test_alias_k_nearest_neighbors():
    view1 = sd.learn("knn")
    view2 = sd.learn("k_nearest_neighbors")
    assert view1.kind == view2.kind == "educational_knn"
    assert view1.data["predicted_class"] == view2.data["predicted_class"]


# ---------------------------------------------------------------------------
# Classical ML Tests
# ---------------------------------------------------------------------------


def test_linear_regression_mathematics():
    # Test default model
    kind, data, summary, notes = learn_concept("linear_regression")
    assert kind == "educational_linear_regression"
    xs = data["x"]
    ys = data["y"]
    ols = data["ols"]
    assert len(xs) == 20
    assert len(ys) == 20

    # Verify analytical OLS solution Cov(x, y) / Var(x)
    n = len(xs)
    x_bar = sum(xs) / n
    y_bar = sum(ys) / n
    cov_xy = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys, strict=True))
    var_x = sum((x - x_bar) ** 2 for x in xs)
    expected_w = cov_xy / var_x
    expected_b = y_bar - expected_w * x_bar

    assert abs(ols["slope"] - expected_w) < 1e-4
    assert abs(ols["intercept"] - expected_b) < 1e-4
    assert ols["r2"] > 0.90  # Strong linear correlation

    # Test user specified parameters
    _, data_custom, _, _ = learn_concept("linear_regression", slope=2.5, intercept=1.2)
    assert data_custom["slope"] == 2.5
    assert data_custom["intercept"] == 1.2
    assert data_custom["mse"] < 1.0
    assert data_custom["r2"] > 0.85

    # Check residuals calculation: e_i = y_i - y_hat_i
    for y, y_hat, e in zip(
        data_custom["y"], data_custom["y_pred"], data_custom["residuals"], strict=True
    ):
        assert abs((y - y_hat) - e) < 1e-4


def test_polynomial_regression_mathematics():
    # Degree 3 fit
    kind, data, summary, notes = learn_concept(
        "polynomial_regression", degree=3, regularization=0.0
    )
    assert kind == "educational_polynomial_regression"
    assert data["degree"] == 3
    assert data["regime"] == "Balanced Fit (Optimal Trade-off)"
    assert data["mse_train"] < 0.05
    assert data["mse_val"] < 0.05

    # Degree 1 (Underfitting)
    _, data_d1, _, _ = learn_concept("polynomial_regression", degree=1)
    assert data_d1["regime"] == "Underfitting (High Bias)"
    assert data_d1["mse_train"] > data["mse_train"]

    # Degree 9 (Overfitting)
    _, data_d9, _, _ = learn_concept("polynomial_regression", degree=9)
    assert data_d9["regime"] == "Overfitting (High Variance)"
    # Training error should be lower than degree 3
    assert data_d9["mse_train"] < data["mse_train"]
    # Validation error should increase relative to degree 3
    assert data_d9["mse_val"] > data["mse_val"]

    # Check degree sweep table has 10 entries
    sweep = data["degree_sweep"]
    assert len(sweep) == 10
    assert sweep[0]["degree"] == 1
    assert sweep[9]["degree"] == 10


def test_gradient_descent_extended_surfaces():
    # 1D Convex
    _, d_1d, _, notes_1d = learn_concept("gradient_descent", lr=0.1, steps=5, objective="1d_convex")
    assert d_1d["stability"] == "monotonic_convergence"
    assert any("monotonic" in n.lower() for n in notes_1d)

    # 1D Double Well
    _, d_dw, _, _ = learn_concept(
        "gradient_descent", lr=0.05, steps=10, w_init=1.5, objective="1d_double_well"
    )
    assert d_dw["objective"] == "1d_double_well"
    assert len(d_dw["trajectory"]) == 11
    # Should move towards minimum at w = 1.0
    assert abs(d_dw["final_w"] - 1.0) < 0.5

    # 2D Valley
    _, d_2d, _, notes_2d = learn_concept(
        "gradient_descent", lr=0.05, steps=10, objective="2d_valley"
    )
    assert d_2d["objective"] == "2d_valley"
    assert d_2d["stability"] == "monotonic_convergence"

    # 2D Valley divergence test (lr >= 0.2)
    _, d_div, _, notes_div = learn_concept(
        "gradient_descent", lr=0.25, steps=5, objective="2d_valley"
    )
    assert d_div["stability"] == "divergence"
    assert any("divergence" in n.lower() or "alert" in n.lower() for n in notes_div)


def test_logistic_regression_mathematics():
    kind, data, summary, notes = learn_concept(
        "logistic_regression", threshold=0.5, w1=1.4, w2=1.1, b=-0.4
    )
    assert kind == "educational_logistic_regression"
    samples = data["samples"]
    assert len(samples) == 30

    # Check Sigmoid output range
    for s in samples:
        assert 0.0 <= s["prob"] <= 1.0
        assert s["y_hat"] in (0, 1)

    # Confusion matrix sum must equal total samples
    cm = data["confusion_matrix"]
    assert cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"] == 30
    assert data["accuracy"] > 0.85
    assert data["f1"] > 0.80

    # Changing threshold changes prediction counts
    _, d_high, _, _ = learn_concept("logistic_regression", threshold=0.9)
    # Higher threshold requires higher probability, reducing positive predictions
    pos_orig = cm["tp"] + cm["fp"]
    cm_high = d_high["confusion_matrix"]
    pos_high = cm_high["tp"] + cm_high["fp"]
    assert pos_high <= pos_orig


def test_knn_mathematics():
    # Euclidean uniform
    kind, data_euc, summary, notes = learn_concept(
        "knn", k=5, metric="euclidean", weights="uniform"
    )
    assert kind == "educational_knn"
    assert data_euc["k"] == 5
    assert len(data_euc["neighbors"]) == 5

    # Verify nearest neighbors are sorted by distance
    distances = [n["distance"] for n in data_euc["neighbors"]]
    assert distances == sorted(distances)

    # Manhattan distance
    _, data_man, _, _ = learn_concept("knn", k=5, metric="manhattan")
    assert data_man["metric"] == "manhattan"

    # Distance-weighted voting
    _, data_dist, _, _ = learn_concept("knn", k=5, weights="distance")
    assert data_dist["weights"] == "distance"


def test_decision_tree_mathematics():
    kind, data, summary, notes = learn_concept("decision_tree", max_depth=2, criterion="gini")
    assert kind == "educational_decision_tree"
    tree = data["tree"]
    assert tree["depth"] == 0
    assert tree["samples"] == 32
    # Gini impurity for balanced binary dataset is ~0.5
    assert abs(tree["impurity"] - 0.5) < 0.05
    assert "gain" in tree
    assert tree["gain"] > 0.0

    # Leaf regions partition space
    regions = data["regions"]
    assert len(regions) >= 2
    for r in regions:
        assert len(r["bounds"]) == 4
        assert r["class"] in (0, 1)

    # Entropy criterion
    _, data_ent, _, _ = learn_concept("decision_tree", max_depth=2, criterion="entropy")
    assert data_ent["criterion"] == "entropy"


def test_kmeans_clustering_mathematics():
    kind, data, summary, notes = learn_concept("kmeans", n_clusters=3, init="kmeans++", max_iter=10)
    assert kind == "educational_kmeans"
    assert data["n_clusters"] == 3
    assert len(data["final_centroids"]) == 3
    history = data["history"]
    assert len(history) >= 2

    # Verify inertia decreases or remains equal over iterations
    inertias = [h["inertia"] for h in history]
    for i in range(len(inertias) - 1):
        assert inertias[i + 1] <= inertias[i] + 1e-4

    # Random initialization
    _, data_rnd, _, _ = learn_concept("kmeans", n_clusters=4, init="random")
    assert data_rnd["n_clusters"] == 4


def test_pca_mathematics():
    kind, data, summary, notes = learn_concept("pca")
    assert kind == "educational_pca"
    mean = data["mean"]
    cov = data["covariance"]
    assert len(mean) == 2
    assert cov["s_xx"] > 0
    assert cov["s_yy"] > 0

    # Check analytical eigenvalues
    l1, l2 = data["eigenvalues"]
    assert l1 >= l2 >= 0
    r1, r2 = data["explained_variance_ratio"]
    assert abs((r1 + r2) - 1.0) < 1e-4
    assert r1 > 0.85  # PC1 explains majority of variance

    # Candidate axis at optimal angle maximizes variance and minimizes reconstruction error
    opt_angle = data["optimal_angle"]
    _, data_opt, _, _ = learn_concept("pca", angle=opt_angle)
    # At perpendicular angle (PC2)
    _, data_perp, _, _ = learn_concept("pca", angle=(opt_angle + 90) % 180)

    assert data_opt["projected_variance"] > data_perp["projected_variance"]
    assert data_opt["mean_reconstruction_error"] < data_perp["mean_reconstruction_error"]

    # Verify fundamental identity: Projected Var + Reconstruction Error == Total Variance (approx)
    total_var = cov["s_xx"] + cov["s_yy"]
    assert (
        abs((data_opt["projected_variance"] + data_opt["mean_reconstruction_error"]) - total_var)
        < 0.25
    )


# ---------------------------------------------------------------------------
# Deep Learning Tests (Preserved)
# ---------------------------------------------------------------------------


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
    assert abs(data["scale"] - 2.0) < 1e-4
    assert any("inverted dropout" in note.lower() for note in notes)


def test_batch_norm_standardization():
    kind, data, summary, notes = learn_concept("batch_norm")
    norm_vals = data["normalized"]
    mean_norm = sum(norm_vals) / len(norm_vals)
    assert abs(mean_norm) < 1e-4
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

    with pytest.raises(SeeDLError, match="lr must be a positive number"):
        sd.learn("gradient_descent", lr=-0.5)

    # Linear regression invalid parameters
    with pytest.raises(SeeDLError, match="finite real numbers"):
        sd.learn("linear_regression", slope=float("nan"))

    # Polynomial regression degree bounds
    with pytest.raises(SeeDLError, match="degree must be an integer between 1 and 10"):
        sd.learn("polynomial_regression", degree=0)

    with pytest.raises(SeeDLError, match="degree must be an integer between 1 and 10"):
        sd.learn("polynomial_regression", degree=11)

    with pytest.raises(SeeDLError, match="regularization must be non-negative"):
        sd.learn("polynomial_regression", regularization=-1.0)

    # Logistic regression threshold bounds
    with pytest.raises(SeeDLError, match="threshold must be in the range"):
        sd.learn("logistic_regression", threshold=0.0)

    with pytest.raises(SeeDLError, match="threshold must be in the range"):
        sd.learn("logistic_regression", threshold=1.0)

    # KNN k bounds and metric
    with pytest.raises(SeeDLError, match="k must be an integer between 1 and 25"):
        sd.learn("knn", k=0)

    with pytest.raises(SeeDLError, match="metric must be 'euclidean' or 'manhattan'"):
        sd.learn("knn", metric="cosine")

    # Decision tree bounds
    with pytest.raises(SeeDLError, match="max_depth must be an integer between 1 and 4"):
        sd.learn("decision_tree", max_depth=5)

    with pytest.raises(SeeDLError, match="criterion must be 'gini' or 'entropy'"):
        sd.learn("decision_tree", criterion="mse")

    # KMeans clusters bounds
    with pytest.raises(SeeDLError, match="n_clusters must be an integer between 2 and 5"):
        sd.learn("kmeans", n_clusters=1)

    with pytest.raises(SeeDLError, match="init must be 'kmeans\\+\\+' or 'random'"):
        sd.learn("kmeans", init="spectral")


def test_cli_learn_commands(tmp_path):
    # Test CLI `seedl learn list`
    result_list = subprocess.run(
        [sys.executable, "-m", "seedl", "learn", "list"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result_list.returncode == 0
    assert "Classical Machine Learning" in result_list.stdout
    assert "linear_regression" in result_list.stdout
    assert "pca" in result_list.stdout
    assert "Deep Learning" in result_list.stdout

    # Test CLI `seedl learn linear_regression --output ...`
    out_file = tmp_path / "test_cli_lr.html"
    result_export = subprocess.run(
        [sys.executable, "-m", "seedl", "learn", "linear_regression", "--output", str(out_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result_export.returncode == 0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "Linear Regression" in content
