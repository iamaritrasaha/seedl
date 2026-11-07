"""Pipeline feature provenance and sklearn interpretation reference checks."""

from __future__ import annotations

# Optional integrations must skip before their imports.
# ruff: noqa: E402
import copy
import pickle
import xml.etree.ElementTree as ET

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

import seedl as sd
from seedl.exceptions import SeeDLError


@pytest.fixture
def mixed_pipeline():
    pd = pytest.importorskip("pandas")
    ds = load_iris()
    frame = pd.DataFrame(ds.data, columns=ds.feature_names)
    frame["sepal band"] = np.where(frame.iloc[:, 0] > 5, "wide", "narrow")
    Xtr, X, ytr, y = train_test_split(frame, ds.target, stratify=ds.target, random_state=0)
    pre = ColumnTransformer(
        [
            ("numeric", StandardScaler(), ds.feature_names),
            ("band", OneHotEncoder(handle_unknown="ignore"), ["sepal band"]),
        ]
    )
    model = make_pipeline(pre, LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
    return model, X, y


def test_column_transformer_diagram_and_expanded_names(mixed_pipeline, tmp_path):
    model, X, y = mixed_pipeline
    before = pickle.dumps(model)
    result = sd.see(model, backend="svg")
    assert "pipeline" in result.data["panels"]
    nodes = result.data["panels"]["pipeline"]["nodes"]
    assert any(n["role"] == "parallel column branch" and n["name"] == "band" for n in nodes)
    assert any("sepal band" in str(n["columns"]) for n in nodes)
    names = result.data["transformed_feature_names"]
    assert any(name.startswith("band__") for name in names)
    coefficient_names = {r[1] for r in result.data["panels"]["coefficients"]["rows"]}
    assert coefficient_names <= set(names)
    assert "sepal band" not in coefficient_names
    assert pickle.dumps(model) == before
    result.save(tmp_path / "pipeline.html")
    result.save(tmp_path / "pipeline.svg")
    ET.fromstring((tmp_path / "pipeline.svg").read_text())
    evaluation = sd.evaluate(model, X, y, split="test", backend="svg")
    assert evaluation.metrics["accuracy"] > 0.8
    assert pickle.dumps(model) == before


def test_unknown_transformed_names_are_not_original_names():
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"original A": [1.0, 2, 3, 4], "original B": [4.0, 3, 2, 1]})
    model = make_pipeline(FunctionTransformer(np.square), LinearRegression()).fit(X, [1, 2, 3, 4])
    view = sd.see(model, backend="svg")
    names = [row[1] for row in view.data["panels"]["coefficients"]["rows"]]
    assert all(name.startswith("transformed feature") for name in names)
    assert any("not equated" in n for n in view.notes)


def test_unfitted_pipeline_and_transformer_inspection():
    pipeline = make_pipeline(StandardScaler(), LinearRegression())
    view = sd.see(pipeline, backend="svg")
    assert not view.data["fitted"]
    assert "pipeline" in view.data["panels"]
    assert sd.see(StandardScaler(), backend="svg").data["fitted"] is False


def test_native_importance_matches_forest():
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(n_estimators=8, random_state=0).fit(X, y)
    before = pickle.dumps(model)
    view = sd.feature_importance(model, backend="svg")
    values = view.data["panels"]["feature_importance"]["values"]
    np.testing.assert_allclose(values, sorted(model.feature_importances_, reverse=True))
    assert "Impurity" in view.data["panels"]["feature_importance"]["title"]
    assert any("causal" in n for n in view.notes)
    assert pickle.dumps(model) == before


def test_permutation_reference_and_original_columns(mixed_pipeline):
    model, X, y = mixed_pipeline
    before = pickle.dumps(model)
    expected = permutation_importance(
        model, X, y, scoring="accuracy", n_repeats=3, random_state=0, n_jobs=1
    )
    view = sd.feature_importance(
        model,
        X,
        y,
        method="permutation",
        scoring="accuracy",
        n_repeats=3,
        split="test",
        backend="svg",
    )
    detail = view.data["panels"]["permutation_details"]["rows"]
    for name, mean, std in detail:
        i = list(X.columns).index(name)
        assert mean == pytest.approx(expected.importances_mean[i])
        assert std == pytest.approx(expected.importances_std[i])
    assert view.data["provenance"]["input_feature_space"] == "original input columns"
    assert len(view.data["importances"]) == X.shape[1]
    assert pickle.dumps(model) == before


def test_pdp_and_ice_match_sklearn():
    X, y = load_diabetes(return_X_y=True)
    model = LinearRegression().fit(X, y)
    before = pickle.dumps(model)
    reference = partial_dependence(
        model, X[:30], [0], method="brute", kind="both", grid_resolution=12
    )
    view = sd.partial_dependence(
        model,
        X[:30],
        [np.int64(0)],
        kind="both",
        max_samples=30,
        grid_resolution=12,
        max_ice=5,
        backend="svg",
    )
    series = view.data["panels"]["pdp_0"]["series"]
    grid = reference.get("grid_values", reference.get("values"))[0]
    np.testing.assert_allclose(series[0]["x"], grid)
    np.testing.assert_allclose(series[0]["y"], reference["average"][0])
    assert len(series) == 6
    np.testing.assert_allclose(series[1]["y"], reference["individual"][0][0])
    assert pickle.dumps(model) == before


def test_multiclass_pdp_requires_target(mixed_pipeline):
    model, X, y = mixed_pipeline
    with pytest.raises(SeeDLError, match="explicit target"):
        sd.partial_dependence(model, X, [0])
    view = sd.partial_dependence(
        model, X, ["sepal length (cm)"], target=2, grid_resolution=8, backend="svg"
    )
    assert "class 2" in view.data["panels"]["pdp_0"]["y_label"]
    with pytest.raises(SeeDLError, match="numeric"):
        sd.partial_dependence(model, X, ["sepal band"], target=2)


def test_inplace_preprocessing_does_not_corrupt_repeated_diagnostics():
    X, y = load_diabetes(return_X_y=True)
    model = make_pipeline(StandardScaler(copy=False), LinearRegression()).fit(X.copy(), y)
    safe_model = copy.deepcopy(model)
    safe_model[0].copy = True
    before = pickle.dumps(model)
    original_X = X.copy()
    result = sd.feature_importance(
        model, X[:40], y[:40], method="permutation", n_repeats=2, scoring="r2", backend="svg"
    )
    expected = permutation_importance(
        safe_model, X[:40], y[:40], n_repeats=2, scoring="r2", random_state=0
    )
    rows = result.data["panels"]["permutation_details"]["rows"]
    for name, mean, std in rows:
        index = int(name.split()[-1])
        assert mean == pytest.approx(expected.importances_mean[index])
        assert std == pytest.approx(expected.importances_std[index])
    dependence = sd.partial_dependence(model, X[:20], [0], grid_resolution=8, backend="svg")
    expected_pdp = partial_dependence(safe_model, X[:20], [0], grid_resolution=8, method="brute")
    np.testing.assert_allclose(
        dependence.data["panels"]["pdp_0"]["series"][0]["y"], expected_pdp["average"][0]
    )
    np.testing.assert_array_equal(X, original_X)
    assert pickle.dumps(model) == before


def test_binary_scores_with_inplace_preprocessing():
    X, y = load_iris(return_X_y=True)
    keep = y < 2
    model = make_pipeline(StandardScaler(copy=False), LogisticRegression()).fit(
        X[keep].copy(), y[keep]
    )
    safe_model = copy.deepcopy(model)
    safe_model[0].copy = True
    original = X[keep].copy()
    from sklearn.metrics import roc_auc_score

    result = sd.evaluate(model, original, y[keep], backend="svg")
    assert result.metrics["roc_auc"] == pytest.approx(
        roc_auc_score(y[keep], safe_model.predict_proba(X[keep])[:, 1])
    )
    np.testing.assert_array_equal(original, X[keep])


def test_interpretation_does_not_fit(mixed_pipeline, monkeypatch):
    model, X, y = mixed_pipeline

    def fail(*a, **k):
        pytest.fail("implicit fit")

    monkeypatch.setattr(LogisticRegression, "fit", fail)
    sd.feature_importance(model, X, y, method="permutation", n_repeats=1, backend="svg")
    sd.partial_dependence(model, X, [0], target=1, grid_resolution=3, backend="svg")


def test_unavailable_importance_and_bounds():
    X, y = load_iris(return_X_y=True)
    model = SVC().fit(X, y)
    with pytest.raises(SeeDLError, match="no native"):
        sd.feature_importance(model)
    with pytest.raises(SeeDLError, match="requires"):
        sd.feature_importance(model, X, method="permutation")
    with pytest.raises(ValueError):
        sd.partial_dependence(model, X, [0], grid_resolution=500, target=1)
    with pytest.raises(SeeDLError, match="one to four"):
        sd.partial_dependence(model, X, [0, 1, 2, 3, 4], target=1)
