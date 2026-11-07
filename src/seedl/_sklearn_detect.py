"""Estimator detection without importing scikit-learn or any numeric framework."""

from __future__ import annotations

from typing import Any


def is_sklearn_estimator(value: Any) -> bool:
    return any(
        cls.__name__ == "BaseEstimator" and cls.__module__ == "sklearn.base"
        for cls in type(value).__mro__
    )
