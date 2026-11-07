"""Classical ML orchestration. Optional frameworks are imported inside operations."""

from __future__ import annotations

from typing import Any

from ._sklearn_validation import (
    MAX_POINTS,
    bounded_int,
    dependencies,
    input_data,
    native,
    predict_once,
    require_estimator,
)
from .evaluation import Evaluation
from .exceptions import SeeDLError
from .renderers import resolve_backend
from .view import Visualization


def inspect_estimator(
    model: Any,
    *,
    backend: str = "auto",
    max_depth: int = 3,
    max_nodes: int = 63,
    max_features: int = 50,
) -> Visualization:
    dependencies()
    from ._sklearn_metadata import inspect_estimator as extract

    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or not 0 <= max_depth <= 8:
        raise ValueError("max_depth must be an integer between 0 and 8.")
    bounded_int("max_nodes", max_nodes, 255)
    bounded_int("max_features", max_features, 100)
    data, metadata, notes = extract(
        model, max_depth=max_depth, max_nodes=max_nodes, max_features=max_features
    )
    return Visualization(
        kind="ml_estimator",
        title=f"{type(model).__name__} inspection",
        summary=f"{'Fitted' if data['fitted'] else 'Unfitted'} scikit-learn estimator | read-only inspection",
        data=data,
        metadata=metadata,
        notes=notes,
        backend=resolve_backend(backend),
    )


def evaluate(
    model: Any,
    X: Any,
    y: Any = None,
    *,
    task: str = "auto",
    split: str = "unspecified",
    plots: str | list[str] | None = None,
    normalize: str | None = None,
    class_names: Any = None,
    positive_label: Any = None,
    labels: Any = None,
    projection: Any = None,
    silhouette_max_samples: int = 1000,
    max_points: int = 2048,
    backend: str = "auto",
) -> Evaluation:
    np, sklearn = dependencies()
    from sklearn.utils.multiclass import type_of_target

    from ._sklearn_metrics import classification, regression
    from ._sklearn_unsupervised import clustering, task_for
    from ._sklearn_unsupervised import projection as pca_projection

    bounded_int("max_points", max_points, MAX_POINTS)
    selected_backend = resolve_backend(backend)
    require_estimator(model)
    if split not in {"unspecified", "training", "validation", "test"}:
        raise ValueError("split must be 'unspecified', 'training', 'validation', or 'test'.")
    if task not in {"auto", "classification", "regression", "clustering", "projection"}:
        raise ValueError(
            "task must be auto, classification, regression, clustering, or projection."
        )
    inferred = task_for(model)
    if inferred is None or task not in {"auto", inferred}:
        raise SeeDLError(
            "Estimator task is unsupported or conflicts with task=. Supported: classifiers, regressors, clusterers and fitted PCA. Use sd.see(model) for other estimator inspection."
        )
    if inferred != "clustering" and (labels is not None or projection is not None):
        raise SeeDLError(
            "labels and projection are clustering-only options; labels are explicit aligned cluster IDs, not supervised targets."
        )
    if inferred != "classification" and (
        normalize is not None or class_names is not None or positive_label is not None
    ):
        raise SeeDLError("normalize, class_names and positive_label apply to classification only.")
    supervised = inferred in {"classification", "regression"}
    if not supervised and y is not None:
        raise SeeDLError(
            "Unsupervised evaluation does not treat y as cluster predictions or targets. Use labels= for explicit aligned cluster assignments."
        )
    X, y, n, p = input_data(model, X, y, require_y=supervised)
    extra: dict[str, Any] = {"prediction_calls": 1 if supervised else 0}
    if inferred == "classification":
        if type_of_target(y) not in {"binary", "multiclass"}:
            raise SeeDLError(
                "Classification requires finite single-output binary or multiclass target labels; multilabel and continuous targets are unsupported."
            )
        predicted = predict_once(model, X, n)
        metrics, panels, unavailable, notes = classification(
            model,
            X,
            y,
            predicted,
            max_points=max_points,
            normalize=normalize,
            class_names=class_names,
            positive_label=positive_label,
        )
    elif inferred == "regression":
        if np.iscomplexobj(y):
            raise SeeDLError(
                "Regression requires real numeric targets; complex values are not silently cast to real."
            )
        try:
            y = np.asarray(y, dtype=float)
        except (ValueError, TypeError) as exc:
            raise SeeDLError("Regression requires numeric target values.") from exc
        if not np.isfinite(y).all():
            raise SeeDLError(
                "Regression targets contain NaN or Inf; align and clean targets explicitly before evaluation."
            )
        predicted = predict_once(model, X, n)
        if np.iscomplexobj(predicted):
            raise SeeDLError(
                "Regression predictions are complex; metrics require real predictions."
            )
        metrics, panels, unavailable, notes = regression(
            y, np.asarray(predicted, dtype=float), max_points=max_points
        )
    elif inferred == "clustering":
        metrics, panels, unavailable, notes, extra = clustering(
            model,
            X,
            labels=labels,
            split=split,
            projection=projection,
            max_points=max_points,
            silhouette_max_samples=silhouette_max_samples,
        )
    else:
        metrics, panels, unavailable, notes, extra = pca_projection(model, X, max_points=max_points)
    reasons: dict[str, list[str]] = {}
    for name, reason in unavailable.items():
        reasons.setdefault(reason, []).append(name)
    for reason, names in reasons.items():
        notes.append(f"Unavailable plots ({', '.join(names)}): {reason}")
    available = list(panels)
    if plots is not None:
        requested = [plots] if isinstance(plots, str) else list(plots)
        if not requested:
            raise SeeDLError("plots must select at least one plot.")
        for name in requested:
            if name not in panels:
                raise SeeDLError(
                    f"Plot {name!r} unavailable: {unavailable.get(name, 'unknown plot for this task')}. Available: {', '.join(panels)}."
                )
        panels = {name: panels[name] for name in requested}
    metadata = {
        "Estimator": type(model).__name__,
        "Task": inferred,
        "Split": split,
        "Samples": n,
        "Input features": p,
        **metrics,
    }
    if split == "unspecified":
        notes.insert(
            0,
            "Evaluation split is unspecified. Set split='training', 'validation' or 'test'; SeeDL cannot infer whether data were used for fitting.",
        )
    elif split == "training" and supervised:
        notes.insert(
            0, "CAUTION: training-set performance is not an estimate of unseen/test performance."
        )
    notes.append(
        "Inspection/evaluation never calls fit(). Selecting, displaying or exporting a captured plot never reruns the estimator."
    )
    data = native(
        {
            "task": inferred,
            "metrics": metrics,
            "panels": panels,
            "unavailable_plots": unavailable,
            "available_at_capture": available,
            "provenance": {
                "split": split,
                "samples": n,
                "input_features": p,
                "sklearn_version": sklearn.__version__,
                **extra,
            },
        }
    )
    return Evaluation(
        kind="ml_evaluation",
        title=f"{type(model).__name__} {inferred} evaluation",
        summary=f"{n:,} {split} samples | {len(panels)} captured plots | scikit-learn diagnostics",
        data=data,
        metadata=native(metadata),
        notes=notes,
        backend=selected_backend,
    )
