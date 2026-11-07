"""Estimator detection without importing scikit-learn or any numeric framework."""

from __future__ import annotations

from typing import Any


def is_sklearn_estimator(value: Any) -> bool:
    return any(
        cls.__name__ == "BaseEstimator" and cls.__module__ == "sklearn.base"
        for cls in type(value).__mro__
    )


def supported_sklearn_version(version: str) -> bool:
    import re

    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    return bool(match and (1, 4, 2) <= tuple(int(v) for v in match.groups()) < (2, 0, 0))
