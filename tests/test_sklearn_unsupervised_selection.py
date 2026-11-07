"""No-refit clustering/PCA/CV and explicitly opted-in clone refitting."""

from __future__ import annotations

# ruff: noqa: E402
import pickle

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_iris, make_blobs
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV, cross_validate, learning_curve, validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import seedl as sd
from seedl.exceptions import SeeDLError


def test_kmeans_cluster_sizes_and_silhouette():
    X, _ = make_blobs(n_samples=150, centers=3, n_features=4, random_state=10)
    model = KMeans(n_clusters=3, random_state=0, n_init=5).fit(X)
    before = pickle.dumps(model)
    result = sd.evaluate(
        model, X, split="training", max_points=17, silhouette_max_samples=100, backend="svg"
    )
    labels = model.predict(X)
    chosen = np.asarray(result.data["provenance"]["silhouette_sample_indices"])
    assert result.metrics["silhouette"] == pytest.approx(
        silhouette_score(X[chosen], labels[chosen])
    )
    assert result.metrics["clusters"] == 3
    assert sum(result.data["panels"]["cluster_sizes"]["values"]) == len(X)
    assert sum(len(s["x"]) for s in result.data["panels"]["clusters"]["series"]) == 17
    assert any("not a fitted PCA" in n for n in result.notes)
    assert pickle.dumps(model) == before


def test_dbscan_explicit_alignment_noise_and_distance():
    X, _ = make_blobs(n_samples=120, centers=2, cluster_std=0.25, random_state=0)
    X = np.vstack([X, [[50.0, 50.0]]])
    model = DBSCAN(eps=0.6, min_samples=4, metric="manhattan").fit(X)
    before = pickle.dumps(model)
    with pytest.raises(SeeDLError, match="cannot predict"):
        sd.evaluate(model, X, backend="svg")
    with pytest.raises(SeeDLError, match="training"):
        sd.evaluate(model, X, labels=model.labels_, split="test", backend="svg")
    result = sd.evaluate(model, X, labels=model.labels_, split="training", backend="svg")
    keep = model.labels_ != -1
    assert result.metrics["noise_samples"] == int((~keep).sum())
    assert "Noise (-1)" in result.data["panels"]["cluster_sizes"]["labels"]
    assert any(s.get("noise") for s in result.data["panels"]["clusters"]["series"])
    assert result.metrics["silhouette"] == pytest.approx(
        silhouette_score(X[keep], model.labels_[keep], metric="manhattan")
    )
    assert result.data["provenance"]["prediction_calls"] == 0
    assert pickle.dumps(model) == before
    assert "cluster_sizes" in sd.see(model, backend="svg").data["panels"]
    with pytest.raises(SeeDLError, match="aligned"):
        sd.evaluate(model, X, labels=model.labels_[:-1], split="training")


@pytest.mark.parametrize("mode", ["one_cluster", "all_noise", "singletons"])
def test_invalid_silhouette_is_unavailable(mode):
    X = np.array([[0.0, 0.0], [1, 1], [2, 2]])
    model = KMeans(n_clusters=1, random_state=0, n_init=1).fit(X)
    labels = {"one_cluster": [0, 0, 0], "all_noise": [-1, -1, -1], "singletons": [0, 1, 2]}[mode]
    result = sd.evaluate(model, X, labels=labels, backend="svg")
    assert result.metrics["silhouette"] is None
    assert "silhouette" in result.unavailable_plots
    with pytest.raises(SeeDLError, match="Silhouette requires"):
        result.plot("silhouette")


def test_pipeline_silhouette_uses_transformed_feature_space():
    X, _ = make_blobs(n_samples=100, n_features=3, random_state=0)
    X[:, 0] *= 1000
    model = make_pipeline(StandardScaler(copy=False), KMeans(3, random_state=0, n_init=5)).fit(
        X.copy()
    )
    safe = model[0].transform(X.copy())
    labels = model[-1].predict(safe)
    original = X.copy()
    before = pickle.dumps(model)
    result = sd.evaluate(model, X, backend="svg")
    assert result.metrics["silhouette"] == pytest.approx(silhouette_score(safe, labels))
    assert "transformed" in result.data["panels"]["silhouette"]["description"]
    np.testing.assert_array_equal(original, X)
    assert pickle.dumps(model) == before


def test_fitted_pca_projection_and_no_automatic_fit(monkeypatch):
    X, _ = load_iris(return_X_y=True)
    pca = PCA(2).fit(X)
    cluster = KMeans(3, random_state=0, n_init=5).fit(X)
    expected = pca.transform(X)
    before = pickle.dumps(pca)

    def fail(*args, **kwargs):
        pytest.fail("implicit fitting")

    monkeypatch.setattr(PCA, "fit", fail)
    monkeypatch.setattr(PCA, "fit_transform", fail)
    projection = sd.evaluate(pca, X, backend="svg")
    series = projection.data["panels"]["pca_projection"]["series"][0]
    np.testing.assert_allclose(series["x"], expected[:, 0])
    np.testing.assert_allclose(series["y"], expected[:, 1])
    np.testing.assert_allclose(
        projection.data["panels"]["explained_variance"]["values"], pca.explained_variance_ratio_
    )
    display = sd.evaluate(cluster, X, projection=pca, backend="svg")
    assert "PCA projection" in display.data["panels"]["clusters"]["description"]
    assert display.metrics["silhouette"] == pytest.approx(silhouette_score(X, cluster.predict(X)))
    assert pickle.dumps(pca) == before


def test_sparse_cluster_projection_without_full_densification():
    from scipy.sparse import csr_matrix

    class SmallOnly(csr_matrix):
        def toarray(self, *a, **k):
            assert self.shape[1] <= 2, "full feature matrix densified"
            return super().toarray(*a, **k)

    X, _ = make_blobs(n_samples=40, n_features=10, centers=2, random_state=0)
    model = KMeans(2, random_state=0, n_init=3).fit(csr_matrix(X))
    result = sd.evaluate(model, SmallOnly(X), backend="svg")
    assert result.metrics["silhouette"] is not None


def test_existing_gridsearch_and_cv_do_not_fit(monkeypatch):
    X, y = load_iris(return_X_y=True)
    search = GridSearchCV(
        LogisticRegression(max_iter=1000),
        {"C": [0.1, 1, 10]},
        scoring={"accuracy": "accuracy", "f1": "f1_macro"},
        refit="accuracy",
        cv=3,
    ).fit(X, y)
    captured = cross_validate(
        LogisticRegression(max_iter=1000), X, y, cv=3, return_train_score=True
    )
    before = pickle.dumps(search)

    def fail(*a, **k):
        pytest.fail("refit")

    monkeypatch.setattr(LogisticRegression, "fit", fail)
    view = sd.cv_results(search, metric="accuracy", backend="svg")
    rows = view.data["panels"]["cv_candidates"]["rows"]
    for i, params, mean, std, rank in rows:
        assert mean == search.cv_results_["mean_test_accuracy"][i]
        assert std == search.cv_results_["std_test_accuracy"][i]
        assert params == search.cv_results_["params"][i]
        assert rank == search.cv_results_["rank_test_accuracy"][i]
    assert "cv_scores" in sd.see(search, backend="svg").data["panels"]
    folds = sd.cv_results(captured, backend="svg")
    np.testing.assert_allclose(folds.data["panels"]["cv_scores"]["values"], captured["test_score"])
    assert any("not a separate untouched" in n for n in folds.notes)
    assert pickle.dumps(search) == before
    with pytest.raises(SeeDLError, match="Choose one"):
        sd.cv_results(search, metric="missing")


def test_captured_failed_cv_scores_remain_missing():
    result = sd.cv_results({"test_score": [0.5, np.nan, 0.7]}, backend="svg")
    assert result.data["panels"]["cv_scores"]["values"] == [0.5, None, 0.7]
    assert result.metrics["mean_cv_score"] is None
    assert any("WARNING" in n for n in result.notes)


def test_curves_require_explicit_refitting():
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=1000)
    with pytest.raises(SeeDLError, match="allow_refit=True"):
        sd.learning_curve(model, X, y)
    with pytest.raises(SeeDLError, match="allow_refit=True"):
        sd.validation_curve(model, X, y, param_name="C", param_range=[0.1, 1])
    assert not hasattr(model, "coef_")


def test_opt_in_learning_curve_matches_sklearn_preserves_model():
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=1000)
    before = pickle.dumps(model)
    # Capture uses all rows sorted by original index when n <= max_samples.
    sizes, train, valid = learning_curve(
        model,
        X,
        y,
        cv=3,
        train_sizes=[0.5, 1.0],
        scoring="accuracy",
        shuffle=True,
        random_state=0,
        n_jobs=1,
    )
    result = sd.learning_curve(
        model,
        X,
        y,
        allow_refit=True,
        cv=3,
        train_sizes=[0.5, 1.0],
        scoring="accuracy",
        backend="svg",
    )
    np.testing.assert_allclose(result.data["train_scores"], train)
    np.testing.assert_allclose(result.data["validation_scores"], valid)
    assert result.data["provenance"]["max_clone_fits"] == 6
    assert pickle.dumps(model) == before
    assert not hasattr(model, "coef_")


def test_opt_in_validation_curve_matches_sklearn():
    X = np.arange(30, dtype=float).reshape(15, 2)
    y = X[:, 0] + np.sin(X[:, 1])
    model = Ridge().fit(X, y)
    before = pickle.dumps(model)
    train, valid = validation_curve(
        model,
        X,
        y,
        param_name="alpha",
        param_range=[0.1, 1, 10],
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=1,
    )
    result = sd.validation_curve(
        model,
        X,
        y,
        param_name="alpha",
        param_range=[0.1, 1, 10],
        allow_refit=True,
        cv=3,
        scoring="neg_mean_squared_error",
        backend="svg",
    )
    np.testing.assert_allclose(result.data["train_scores"], train)
    np.testing.assert_allclose(result.data["validation_scores"], valid)
    assert all(v <= 0 for v in result.data["panels"]["curve"]["series"][1]["y"])
    assert pickle.dumps(model) == before


def test_unsupported_unsupervised_inputs_and_bounds():
    X, _ = make_blobs(n_samples=20, n_features=3, random_state=0)
    model = KMeans(2, n_init=1, random_state=0).fit(X)
    with pytest.raises(SeeDLError, match="does not treat y"):
        sd.evaluate(model, X, np.ones(len(X)))
    with pytest.raises(SeeDLError, match="previously fitted PCA"):
        sd.evaluate(model, X, projection=StandardScaler().fit(X))
    with pytest.raises(ValueError):
        sd.evaluate(model, X, silhouette_max_samples=100000)
    with pytest.raises(SeeDLError, match="unfitted"):
        sd.evaluate(model, X, projection=PCA(2))
