"""Reference metrics, data safety and optional-dependency tests for classical ML."""

from __future__ import annotations

# Optional integrations must skip before their imports.
# ruff: noqa: E402
import json
import pickle
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

np = pytest.importorskip("numpy")
sklearn = pytest.importorskip("sklearn")
from sklearn import metrics
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import seedl as sd
from seedl.exceptions import MissingDependencyError, SeeDLError


@pytest.fixture
def binary_data():
    ds = load_breast_cancer()
    return train_test_split(ds.data, ds.target, test_size=0.25, stratify=ds.target, random_state=42)


@pytest.fixture
def regression_data():
    ds = load_diabetes()
    return train_test_split(ds.data, ds.target, random_state=42)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        lambda: DecisionTreeClassifier(max_depth=3, random_state=0),
        lambda: RandomForestClassifier(n_estimators=10, random_state=0),
        lambda: make_pipeline(StandardScaler(), SVC()),
    ],
)
def test_binary_metrics_and_scores(factory, binary_data):
    Xtr, X, ytr, y = binary_data
    model = factory().fit(Xtr, ytr)
    before = pickle.dumps(model)
    result = sd.evaluate(model, X, y, split="test", backend="svg")
    pred = model.predict(X)
    assert result.metrics["accuracy"] == metrics.accuracy_score(y, pred)
    assert result.metrics["f1_macro"] == pytest.approx(metrics.f1_score(y, pred, average="macro"))
    scores = (
        model.predict_proba(X)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X)
    )
    assert result.metrics["roc_auc"] == pytest.approx(metrics.roc_auc_score(y, scores))
    assert result.metrics["average_precision"] == pytest.approx(
        metrics.average_precision_score(y, scores)
    )
    assert (
        result.data["panels"]["confusion_matrix"]["values"]
        == metrics.confusion_matrix(y, pred).tolist()
    )
    assert result.plot("roc").data["provenance"] == result.data["provenance"]
    assert set(result.available_plots) == {
        "class_report",
        "confusion_matrix",
        "roc",
        "precision_recall",
    }
    assert pickle.dumps(model) == before
    json.dumps(result.data, allow_nan=False)
    ET.fromstring(result.to_figure())


@pytest.mark.parametrize(
    "factory",
    [
        LinearRegression,
        lambda: make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
        lambda: make_pipeline(StandardScaler(copy=False), LinearRegression()),
    ],
)
def test_regression_reference_and_input_preservation(factory, regression_data):
    Xtr, X, ytr, y = regression_data
    model = factory().fit(Xtr.copy(), ytr)
    original_X = X.copy()
    before = pickle.dumps(model)
    result = sd.evaluate(model, X, y, split="test", max_points=13, backend="svg")
    pred = model.predict(X.copy())
    assert result.metrics["mae"] == pytest.approx(metrics.mean_absolute_error(y, pred))
    assert result.metrics["mse"] == pytest.approx(metrics.mean_squared_error(y, pred))
    assert result.metrics["rmse"] == pytest.approx(np.sqrt(metrics.mean_squared_error(y, pred)))
    assert result.metrics["r2"] == pytest.approx(metrics.r2_score(y, pred))
    assert len(result.data["panels"]["actual_predicted"]["series"][0]["x"]) == 13
    assert sum(result.data["panels"]["residual_distribution"]["counts"]) == len(y)
    np.testing.assert_array_equal(X, original_X)
    assert pickle.dumps(model) == before


def test_no_implicit_fit_and_capture_reused(binary_data, monkeypatch, tmp_path):
    Xtr, X, ytr, y = binary_data
    model = LogisticRegression(max_iter=10000).fit(Xtr, ytr)
    counts = {"predict": 0, "predict_proba": 0}
    original_predict = LogisticRegression.predict
    original_probability = LogisticRegression.predict_proba

    def predict(self, X):
        counts["predict"] += 1
        return original_predict(self, X)

    def proba(self, X):
        counts["predict_proba"] += 1
        return original_probability(self, X)

    def fail(*args, **kwargs):
        pytest.fail("fit() must not be called")

    monkeypatch.setattr(LogisticRegression, "fit", fail)
    monkeypatch.setattr(LogisticRegression, "predict", predict)
    monkeypatch.setattr(LogisticRegression, "predict_proba", proba)
    sd.see(model, backend="svg")
    assert counts == {"predict": 0, "predict_proba": 0}
    result = sd.evaluate(model, X, y, backend="svg")
    result.plot("roc").save(tmp_path / "roc.html")
    result.save(tmp_path / "evaluation.svg")
    result.to_text()
    result._repr_mimebundle_()
    assert counts == {"predict": 1, "predict_proba": 1}


def test_multiclass_report_is_not_binary():
    ds = load_iris()
    Xtr, X, ytr, y = train_test_split(ds.data, ds.target, stratify=ds.target, random_state=1)
    model = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    result = sd.evaluate(model, X, y, class_names=ds.target_names, normalize="true", backend="svg")
    assert "roc_auc" not in result.metrics
    assert "Multiclass" in result.unavailable_plots["roc"]
    assert np.allclose(
        result.data["panels"]["confusion_matrix"]["values"],
        metrics.confusion_matrix(y, model.predict(X), normalize="true"),
    )
    assert result.data["panels"]["confusion_matrix"]["x_labels"] == ds.target_names.tolist()
    assert sum(row[-1] for row in result.data["panels"]["class_report"]["rows"]) == len(y)
    with pytest.raises(SeeDLError, match="exactly two"):
        result.plot("roc")


def test_positive_class_orientation(binary_data):
    Xtr, X, ytr, y = binary_data
    model = make_pipeline(StandardScaler(), SVC()).fit(Xtr, ytr)
    result = sd.evaluate(model, X, y, positive_label=0, backend="svg")
    assert result.metrics["roc_auc"] == pytest.approx(
        metrics.roc_auc_score(y == 0, -model.decision_function(X))
    )
    assert "not probability" in " ".join(result.notes)


def test_no_score_method_and_single_target_class(binary_data):
    from sklearn.base import BaseEstimator, ClassifierMixin

    class LabelsOnly(ClassifierMixin, BaseEstimator):
        def __init__(self):
            self.classes_ = np.array([0, 1])
            self.n_features_in_ = 2

        def fit(self, X, y=None):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    result = sd.evaluate(LabelsOnly(), [[1, 2], [3, 4]], [0, 1], backend="svg")
    assert "neither predict_proba" in result.unavailable_plots["roc"]
    assert result.data["panels"]["class_report"]["rows"][1][2] is None
    assert any("undefined" in n for n in result.notes)
    Xtr, X, ytr, y = binary_data
    model = DummyClassifier().fit(Xtr, ytr)
    result = sd.evaluate(model, X[y == 0], y[y == 0], backend="svg")
    assert "roc" not in result.available_plots
    with pytest.raises(SeeDLError, match="unavailable"):
        sd.evaluate(model, X[y == 0], y[y == 0], plots="roc", backend="svg")


def test_small_constant_regression_warns():
    model = DummyRegressor().fit([[0], [1]], [4.0, 4.0])
    result = sd.evaluate(model, [[2]], [4.0], backend="svg")
    assert result.metrics["r2"] is None
    assert result.metrics["rmse"] == 0
    assert any("fewer than 20" in n for n in result.notes)
    assert "WARNING" in result.to_text()
    json.dumps(result.data, allow_nan=False)


def test_sparse_inputs_without_densifying(regression_data):
    from scipy.sparse import csr_matrix

    class NoDense(csr_matrix):
        def toarray(self, *a, **k):
            pytest.fail("input densification")

        def todense(self, *a, **k):
            pytest.fail("input densification")

    Xtr, X, ytr, y = regression_data
    model = LinearRegression().fit(csr_matrix(Xtr), ytr)
    result = sd.evaluate(model, NoDense(X), y, backend="svg")
    assert result.metrics["mae"] == pytest.approx(
        metrics.mean_absolute_error(y, model.predict(csr_matrix(X)))
    )


def test_dataframe_alignment_and_features(regression_data):
    pd = pytest.importorskip("pandas")
    Xtr, X, ytr, y = regression_data
    names = [f"column-{i}" for i in range(X.shape[1])]
    model = LinearRegression().fit(pd.DataFrame(Xtr, columns=names), ytr)
    frame = pd.DataFrame(X, columns=names)
    target = pd.Series(y)
    result = sd.evaluate(model, frame, target, backend="svg")
    assert result.metrics["r2"] is not None
    with pytest.raises(SeeDLError, match="indices"):
        sd.evaluate(model, frame, target.iloc[::-1])
    with pytest.raises(SeeDLError, match="predict.*failed"):
        sd.evaluate(model, frame[names[::-1]], target)
    inspection = sd.see(model, backend="svg")
    assert inspection.data["input_feature_names"] == names


def test_estimator_inspection_coefficients_and_tree_bounds(regression_data):
    Xtr, X, ytr, y = regression_data
    model = LinearRegression().fit(Xtr, ytr)
    inspection = sd.see(model, backend="svg")
    rows = inspection.data["panels"]["coefficients"]["rows"]
    assert sorted(row[2] for row in rows) == pytest.approx(sorted(model.coef_))
    assert inspection.data["intercept"] == model.intercept_
    tree = DecisionTreeClassifier(random_state=0).fit(*load_iris(return_X_y=True))
    info = sd.inspect_estimator(tree, max_depth=1, max_nodes=3, backend="svg")
    assert len(info.data["tree_nodes"]) <= 3
    assert max(n["depth"] for n in info.data["tree_nodes"]) <= 1
    root = info.data["tree_nodes"][0]
    assert root["threshold"] == tree.tree_.threshold[0]
    assert root["samples"] == tree.tree_.n_node_samples[0]
    assert not sd.see(LinearRegression(), backend="svg").data["fitted"]


def test_pipeline_names_use_transformed_space(regression_data):
    Xtr, X, ytr, y = regression_data
    model = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(Xtr, ytr)
    inspection = sd.see(model, backend="svg")
    assert "transformed" in inspection.data["coefficient_space"]
    names = [r[1] for r in inspection.data["panels"]["coefficients"]["rows"]]
    assert any("^2" in name or " " in name for name in names)


@pytest.mark.parametrize("backend", ["svg", "plotly", "matplotlib"])
def test_backends_and_safe_export(backend, regression_data, tmp_path):
    if backend != "svg":
        pytest.importorskip(backend)
    Xtr, X, ytr, y = regression_data
    model = LinearRegression().fit(Xtr, ytr)
    result = sd.evaluate(model, X, y, backend=backend, plots="actual_predicted")
    result.title = '<script>alert("user")</script>'
    result.notes.append('<img src=x onerror="alert(1)">')
    result.save(tmp_path / f"{backend}.html")
    doc = (tmp_path / f"{backend}.html").read_text()
    assert "&lt;script&gt;" in doc
    assert "<img src=x" not in doc
    assert "Google Sans" in doc and "JetBrains Mono" in doc
    if backend == "matplotlib":
        import matplotlib.pyplot as plt

        plt.close(result.figure)


@pytest.mark.parametrize("change", ["missing", "length", "features", "multilabel", "nan", "task"])
def test_invalid_targets_and_dimensions(change, regression_data):
    Xtr, X, ytr, y = regression_data
    model = LinearRegression().fit(Xtr, ytr)
    with pytest.raises((SeeDLError, ValueError)):
        if change == "missing":
            sd.evaluate(model, X)
        elif change == "length":
            sd.evaluate(model, X, y[:-1])
        elif change == "features":
            sd.evaluate(model, X[:, :-1], y)
        elif change == "multilabel":
            sd.evaluate(model, X, np.column_stack([y, y]))
        elif change == "nan":
            sd.evaluate(model, X, np.full(len(y), np.nan))
        else:
            sd.evaluate(model, X, y, task="classification")


def test_unfitted_and_existing_see_options():
    with pytest.raises(SeeDLError, match="unfitted"):
        sd.evaluate(LinearRegression(), [[1]], [1])
    with pytest.raises(SeeDLError, match="belong to"):
        sd.see(LinearRegression(), input=[[1]])
    assert sd.see([[1, 2], [3, 4]], backend="svg").kind == "matrix"


def test_missing_dependency_and_lazy_discovery(monkeypatch):
    import builtins

    from seedl._sklearn_validation import dependencies

    original_import = builtins.__import__

    def reject(name, *args, **kwargs):
        if name == "sklearn":
            raise ImportError("missing scikit-learn")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject)
    with pytest.raises(MissingDependencyError, match=r"seedl\[sklearn\]"):
        dependencies()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import seedl, sys; seedl.guide(); assert not any(n in sys.modules for n in ('sklearn','pandas','scipy','numpy','torch','qiskit'))",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_regression_numeric_overflow_underflow_are_disclosed():
    X = np.array([[0.0], [1.0]])
    model = DummyRegressor(strategy="constant", constant=0.0).fit(X, [0.0, 0.0])
    tiny = sd.evaluate(model, X, [1e-200, -1e-200], backend="svg")
    assert tiny.metrics["mse"] == 0  # exact sklearn result, with explicit underflow warning
    assert any("underflow" in n for n in tiny.notes)
    assert "WARNING:" in tiny.to_text()
    large = sd.evaluate(model, X, [1e308, -1e308], backend="svg")
    assert large.metrics["mse"] is None
    assert any("overflow" in n for n in large.notes)
    assert "residual_distribution" in large.unavailable_plots
    json.dumps(large.data, allow_nan=False)
    ET.fromstring(large.to_figure())


def test_expanded_features_are_bounded_before_prediction():
    model = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(
        [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], [0.0, 1.0, 2.0]
    )
    # Simulate known fitted output width without allocating a massive matrix.
    model[-1].n_features_in_ = 1_000_000
    with pytest.raises(SeeDLError, match="transformed feature matrix"):
        sd.evaluate(model, np.zeros((20, 2)), np.zeros(20))


def test_wrong_predictions_have_zero_f1_not_undefined():
    from sklearn.base import BaseEstimator, ClassifierMixin

    class Wrong(ClassifierMixin, BaseEstimator):
        def fit(self, X, y):
            self.classes_ = np.array([0, 1])
            self.n_features_in_ = 1
            return self

        def predict(self, X):
            return np.array([1, 0])

    result = sd.evaluate(Wrong().fit([[0], [1]], [0, 1]), [[0], [1]], [0, 1], backend="svg")
    assert result.metrics["f1_macro"] == 0
    assert all(row[4] == 0 for row in result.data["panels"]["class_report"]["rows"])


def test_incompatible_installed_sklearn_fails_actionably(monkeypatch):
    import sklearn

    from seedl._sklearn_validation import dependencies

    monkeypatch.setattr(sklearn, "__version__", "1.3.2")
    with pytest.raises(MissingDependencyError, match="scikit-learn>=1.4.2"):
        dependencies()


def test_complex_targets_are_not_silently_cast():
    model = LinearRegression().fit([[0.0], [1.0]], [0.0, 1.0])
    with pytest.raises(SeeDLError, match="complex"):
        sd.evaluate(model, [[0.0], [1.0]], [1 + 2j, 3 + 4j])


def test_framework_arrays_rejected_before_copy():
    class FrameworkArray:
        __module__ = "torch"
        shape = (2, 1)

        def __deepcopy__(self, memo):
            pytest.fail("framework tensor copy")

    model = LinearRegression().fit([[0.0], [1.0]], [0.0, 1.0])
    with pytest.raises(SeeDLError, match="CPU arrays explicitly"):
        sd.evaluate(model, FrameworkArray(), [0.0, 1.0])
