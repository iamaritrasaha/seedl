"""Build authentic ML examples and inspect offline exports with headless Chromium.

Requires seedl[sklearn,plotly], pandas, and optional Playwright + Chromium for
screenshots. Default datasets are bundled with sklearn (no downloads).
--california explicitly enables California Housing's initial dataset download.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_diabetes, load_iris
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import seedl as sd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--california",
        action="store_true",
        help="explicitly fetch/cache California Housing for regression",
    )
    parser.add_argument("--data-home", type=Path)
    parser.add_argument("--screenshots", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    binary = load_breast_cancer(as_frame=True)
    Xt, X, yt, y = train_test_split(
        binary.data, binary.target, test_size=0.25, stratify=binary.target, random_state=42
    )
    classifier = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)
    ).fit(Xt, yt)
    classification = sd.evaluate(
        classifier,
        X,
        y,
        split="test",
        class_names=dict(enumerate(binary.target_names)),
        positive_label=0,
        backend="svg",
    )
    classification.title = "Breast Cancer | held-out classification"
    classification.notes.append(
        "Authentic Wisconsin Diagnostic Breast Cancer data, bundled with sklearn; positive label 0 = malignant. Educational benchmark, not clinical validation."
    )
    regression = (
        fetch_california_housing(data_home=args.data_home, as_frame=True)
        if args.california
        else load_diabetes(as_frame=True)
    )
    Rt, R, rt, r = train_test_split(
        regression.data, regression.target, test_size=0.2, random_state=42
    )
    regressor = make_pipeline(StandardScaler(), LinearRegression()).fit(Rt, rt)
    evaluation = sd.evaluate(regressor, R, r, split="test", backend="svg")
    evaluation.title = (
        "California Housing" if args.california else "Diabetes"
    ) + " | held-out regression"
    evaluation.metadata["Dataset"] = (
        "California Housing (1990 census)" if args.california else "sklearn Diabetes dataset"
    )
    if args.california:
        evaluation.metadata["Target units"] = "100,000 USD"
        evaluation.notes.append(
            "Authentic California Housing benchmark: target units 100,000 USD, historical 1990 census, source target cap applies. This is not a current real-estate valuation model."
        )
    iris = load_iris(as_frame=True)
    mixed = iris.data.copy()
    mixed["sepal band"] = np.where(mixed.iloc[:, 0] > 5, "wide", "narrow")
    pre = ColumnTransformer(
        [
            ("numeric", StandardScaler(), iris.feature_names),
            ("band", OneHotEncoder(handle_unknown="ignore"), ["sepal band"]),
        ]
    )
    pipeline = make_pipeline(pre, LogisticRegression(max_iter=1000)).fit(mixed, iris.target)
    cluster = KMeans(3, n_init=10, random_state=0).fit(iris.data)
    dbscan = DBSCAN(eps=0.5).fit(iris.data)
    pca = PCA(2).fit(iris.data)
    search = GridSearchCV(LogisticRegression(max_iter=1000), {"C": [0.1, 1.0, 10.0]}, cv=3).fit(
        iris.data, iris.target
    )
    views = {
        "classification": classification,
        "regression": evaluation,
        "roc": classification.plot("roc"),
        "pipeline": sd.see(pipeline, backend="svg"),
        "permutation": sd.feature_importance(
            classifier,
            X,
            y,
            method="permutation",
            split="test",
            scoring="accuracy",
            n_repeats=3,
            backend="svg",
        ),
        "dependence": sd.partial_dependence(
            regressor,
            R,
            [0],
            kind="both",
            max_samples=128,
            max_ice=6,
            grid_resolution=20,
            backend="svg",
        ),
        "clusters": sd.evaluate(
            cluster, iris.data, projection=pca, split="training", backend="svg"
        ),
        "dbscan": sd.evaluate(
            dbscan, iris.data, labels=dbscan.labels_, split="training", backend="svg"
        ),
        "pca": sd.evaluate(pca, iris.data, split="training", backend="svg"),
        "search": sd.cv_results(search, backend="svg"),
    }
    for name, view in views.items():
        view.save(root / f"{name}.html")
        view.save(root / f"{name}.svg")
        (root / f"{name}.txt").write_text(view.to_text(detail="full"), encoding="utf-8")
    (root / "metrics.json").write_text(
        json.dumps(
            {"classification": classification.metrics, "regression": evaluation.metrics}, indent=2
        )
    )
    if args.screenshots:
        from playwright.sync_api import sync_playwright

        observations = []
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for width in (320, 390, 768, 1280):
                page = browser.new_page(viewport={"width": width, "height": 1000})
                errors = []
                requests = []
                page.on("pageerror", lambda e, errors=errors: errors.append(str(e)))
                page.on("request", lambda req, requests=requests: requests.append(req.url))
                for name in views:
                    page.goto((root / f"{name}.html").as_uri())
                    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (
                        name,
                        width,
                    )
                    page.screenshot(path=str(root / f"{name}-{width}.png"), full_page=True)
                    observations.append({"view": name, "width": width, "no_page_overflow": True})
                assert not errors, errors
                assert not any(url.startswith(("http:", "https:")) for url in requests), requests
                page.close()
            browser.close()
        (root / "browser-observations.json").write_text(json.dumps(observations, indent=2))
    print(f"Verified outputs: {root}")


if __name__ == "__main__":
    main()
