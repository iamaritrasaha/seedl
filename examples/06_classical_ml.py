"""Offline classical ML demo: python examples/06_classical_ml.py --output artifacts/ml.

Authentic bundled Breast Cancer and Iris data; no downloads. All fitting is
explicit sklearn example code. SeeDL only inspects/evaluates fitted objects.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import seedl as sd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/classical-ml"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ds = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        ds.data, ds.target, test_size=0.25, stratify=ds.target, random_state=42
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    model.fit(X_train, y_train)  # User's explicit training step, outside SeeDL.
    result = sd.evaluate(
        model,
        X_test,
        y_test,
        split="test",
        class_names=dict(enumerate(ds.target_names)),
        positive_label=0,
        backend="svg",
    )
    result.notes.append(
        "Authentic Wisconsin Diagnostic Breast Cancer dataset, bundled with scikit-learn. Positive label 0 = malignant. Educational held-out evaluation, not a clinical validation."
    )
    result.save(args.output / "classification.html")
    result.plot("confusion_matrix").save(args.output / "confusion-matrix.svg")
    sd.see(model, backend="svg").save(args.output / "pipeline.html")
    sd.feature_importance(
        model,
        X_test,
        y_test,
        method="permutation",
        split="test",
        scoring="accuracy",
        n_repeats=3,
        backend="svg",
    ).save(args.output / "permutation.html")
    X = load_iris().data
    cluster = KMeans(3, n_init=10, random_state=0).fit(X)  # Explicit example fitting.
    projection = PCA(2).fit(X)  # Optional display projection fit by the caller.
    sd.evaluate(cluster, X, projection=projection, split="training", backend="svg").save(
        args.output / "clusters.html"
    )
    dbscan = DBSCAN(eps=0.5).fit(X)
    sd.evaluate(dbscan, X, labels=dbscan.labels_, split="training", backend="svg").save(
        args.output / "dbscan.html"
    )
    print(
        f"Offline examples written to {args.output.resolve()}; open classification.html yourself."
    )


if __name__ == "__main__":
    main()
