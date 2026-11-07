"""Lazy scikit-learn validation and bounded, non-densifying input handling."""

from __future__ import annotations

import copy
from typing import Any

from ._sklearn_detect import is_sklearn_estimator
from .exceptions import MissingDependencyError, SeeDLError

MAX_ROWS = 1_000_000
MAX_DENSE_CELLS = 10_000_000
MAX_POINTS = 10_000


def dependencies() -> tuple[Any, Any]:
    try:
        import numpy as np
        import sklearn
    except ImportError as exc:
        raise MissingDependencyError(
            'Classical ML requires scikit-learn. Install with python -m pip install "seedl[sklearn]".'
        ) from exc
    return np, sklearn


def require_estimator(model: Any, *, fitted: bool = True) -> None:
    dependencies()
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    if not is_sklearn_estimator(model):
        raise SeeDLError(
            "Expected a scikit-learn BaseEstimator instance; use sd.predictions() for already computed predictions."
        )
    if fitted:
        try:
            check_is_fitted(model)
        except (NotFittedError, TypeError) as exc:
            raise SeeDLError(
                "Estimator is unfitted. Fit it in your own training code before evaluation; SeeDL never calls fit()."
            ) from exc


def fitted_status(model: Any) -> bool:
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    try:
        check_is_fitted(model)
        return True
    except (NotFittedError, TypeError):
        return False


def bounded_int(name: str, value: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
        raise ValueError(f"{name} must be an integer between 1 and {upper}.")
    return value


def native(value: Any) -> Any:
    """JSON-safe scalars, preserving undefined values as unavailable, never zero."""
    import math

    if hasattr(value, "tolist"):
        return native(value.tolist())
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)[:200]


def rows_at(data: Any, indices: Any) -> Any:
    if hasattr(data, "iloc"):
        return data.iloc[indices].copy()
    try:
        return data[indices].copy()
    except (TypeError, AttributeError):
        return [copy.deepcopy(data[int(i)]) for i in indices]


def indices(count: int, limit: int) -> Any:
    import numpy as np

    return np.arange(count) if count <= limit else np.linspace(0, count - 1, limit, dtype=int)


def input_data(
    model: Any, X: Any, y: Any = None, *, require_y: bool = False, max_rows: int = MAX_ROWS
) -> tuple[Any, Any, int, int]:
    np, _ = dependencies()
    from scipy import sparse
    from sklearn.utils.validation import check_consistent_length

    if X is None:
        raise SeeDLError("Supply the evaluation feature matrix X explicitly.")
    if require_y and y is None:
        raise SeeDLError(
            "Supervised evaluation requires aligned target labels y. SeeDL never infers them."
        )
    if y is not None and hasattr(X, "index") and hasattr(y, "index"):
        if not callable(X.index) and not callable(y.index) and not X.index.equals(y.index):
            raise SeeDLError(
                "X and y pandas indices differ or are reordered; align rows explicitly before evaluation."
            )
    shape = getattr(X, "shape", None)
    if shape is None:
        X = np.asarray(X)
        shape = X.shape
    if len(shape) != 2 or shape[0] == 0 or shape[1] == 0:
        raise SeeDLError("X must be a nonempty two-dimensional sample-by-feature matrix.")
    n, p = int(shape[0]), int(shape[1])
    if n > max_rows:
        raise SeeDLError(
            f"X has {n:,} rows; this operation is bounded to {max_rows:,}. Supply an explicit evaluation subset."
        )
    if not sparse.issparse(X) and n * p > MAX_DENSE_CELLS:
        raise SeeDLError(
            "Dense X exceeds the 10,000,000-cell safety limit; supply an explicit subset or a suitable sparse matrix."
        )
    if sparse.issparse(X) and X.nnz > MAX_DENSE_CELLS:
        raise SeeDLError(
            "Sparse X exceeds the 10,000,000 stored-value safety limit; supply an explicit subset."
        )
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and int(expected) != p:
        raise SeeDLError(f"X has {p} features; estimator expects {expected} input features.")
    # Copy preserves callers' data for pipelines with in-place transformers (copy=False).
    X = X.copy() if hasattr(X, "copy") else copy.deepcopy(X)
    if y is not None:
        check_consistent_length(X, y)
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        if y.ndim != 1:
            raise SeeDLError(
                "Only single-output targets are supported; evaluate each output explicitly."
            )
    return X, y, n, p


def input_names(model: Any, X: Any = None) -> list[str]:
    names = list(X.columns) if hasattr(X, "columns") else getattr(model, "feature_names_in_", None)
    count = getattr(model, "n_features_in_", None)
    if names is not None:
        return [str(n) for n in names]
    if count is None and X is not None:
        count = X.shape[1]
    return [f"feature {i}" for i in range(int(count or 0))]


def predict_once(model: Any, X: Any, n: int) -> Any:
    import numpy as np

    try:
        predicted = np.asarray(model.predict(X))
    except Exception as exc:
        raise SeeDLError(f"Estimator predict() failed: {exc}") from exc
    if predicted.ndim == 2 and predicted.shape[1] == 1:
        predicted = predicted[:, 0]
    if predicted.ndim != 1 or len(predicted) != n:
        raise SeeDLError("Estimator predictions must be a single aligned output per sample.")
    return predicted
