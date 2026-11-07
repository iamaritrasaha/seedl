"""Offline interactive ML learning demo: python examples/07_interactive_ml_learning.py --output artifacts/interactive-ml.

Demonstrates SeeDL's educational learning modules across 7 foundational classical ML concepts:
1. Linear Regression (OLS closed-form fit and residuals)
2. Polynomial Regression (capacity sweep and overfitting)
3. Gradient Descent (convex optimization on quadratic loss surface)
4. Logistic Regression (binary classification decision boundary)
5. Decision Tree (recursive binary splitting with Gini impurity)
6. KMeans Clustering (iterative centroid updates and Voronoi partitioning)
7. Principal Component Analysis (orthogonal variance projection)

All lessons are mathematically verified and self-contained with zero runtime dependencies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import seedl as sd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/interactive-ml"),
        help="Directory to save generated interactive lessons (default: artifacts/interactive-ml)",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("Generating SeeDL v0.7.0 Interactive ML Learning lessons...")

    # 1. Linear Regression Explorer
    print("1. Linear Regression Explorer (OLS solution)")
    lr_view = sd.learn("linear_regression", slope=2.2, intercept=1.0, backend="svg")
    lr_view.save(args.output / "01_linear_regression.html")
    lr_view.save(args.output / "01_linear_regression.svg")
    print(f"   {lr_view.summary}")

    # 2. Polynomial Regression & Overfitting Explorer
    print("\n2. Polynomial Regression Explorer (degree=3, degree sweep 1-10)")
    poly_view = sd.learn("polynomial_regression", degree=3, regularization=0.0, backend="svg")
    poly_view.save(args.output / "02_polynomial_regression.html")
    print(f"   {poly_view.summary}")

    # 3. Gradient Descent Laboratory
    print("\n3. Gradient Descent Laboratory (lr=0.15, steps=10)")
    gd_view = sd.learn(
        "gradient_descent", lr=0.15, steps=10, start_x=-3.5, start_y=2.5, backend="svg"
    )
    gd_view.save(args.output / "03_gradient_descent.html")
    print(f"   {gd_view.summary}")

    # 4. Logistic Regression Explorer
    print("\n4. Logistic Regression Explorer (binary cross-entropy)")
    log_view = sd.learn("logistic_regression", w1=1.8, w2=-1.2, bias=0.2, backend="svg")
    log_view.save(args.output / "04_logistic_regression.html")
    print(f"   {log_view.summary}")

    # 5. Decision Tree Explorer
    print("\n5. Decision Tree Explorer (max_depth=2, Gini impurity)")
    tree_view = sd.learn("decision_tree", max_depth=2, criterion="gini", backend="svg")
    tree_view.save(args.output / "05_decision_tree.html")
    print(f"   {tree_view.summary}")

    # 6. KMeans Clustering Explorer
    print("\n6. KMeans Clustering Explorer (k=3, 5 iterations)")
    kmeans_view = sd.learn("kmeans", k=3, iterations=5, seed=42, backend="svg")
    kmeans_view.save(args.output / "06_kmeans.html")
    print(f"   {kmeans_view.summary}")

    # 7. Principal Component Analysis Explorer
    print("\n7. Principal Component Analysis Explorer (2 components)")
    pca_view = sd.learn("pca", n_components=2, backend="svg")
    pca_view.save(args.output / "07_pca.html")
    print(f"   {pca_view.summary}")

    print(
        f"\nAll 7 interactive lessons saved to {args.output.resolve()}; open any HTML file in a browser."
    )


if __name__ == "__main__":
    main()
