"""Supervised evaluation through scikit-learn's established metric implementations."""

from __future__ import annotations

import warnings
from typing import Any

from ._sklearn_validation import indices, native
from .exceptions import SeeDLError


def _table(
    title: str, columns: list[str], rows: list[list[Any]], description: str = ""
) -> dict[str, Any]:
    return {
        "type": "table",
        "title": title,
        "columns": columns,
        "rows": native(rows),
        "description": description,
    }


def classification(
    model: Any,
    X: Any,
    y: Any,
    pred: Any,
    *,
    max_points: int,
    normalize: str | None,
    class_names: Any,
    positive_label: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], list[str]]:
    import numpy as np
    from sklearn import metrics
    from sklearn.utils.multiclass import unique_labels

    labels = unique_labels(y, pred, getattr(model, "classes_", y))
    if len(labels) > 100:
        raise SeeDLError(
            "Classification display is bounded to 100 classes; evaluate an explicit class subset or use scikit-learn metrics directly."
        )
    if normalize not in {None, "true", "pred", "all"}:
        raise ValueError("normalize must be None, 'true', 'pred', or 'all'.")
    if class_names is None:
        names = [str(v) for v in labels]
    elif isinstance(class_names, dict):
        if any(v not in class_names for v in labels):
            raise SeeDLError("class_names mapping must cover every observed/model class.")
        names = [str(class_names[v]) for v in labels]
    else:
        if len(class_names) != len(labels):
            raise SeeDLError("class_names must have one name for each sorted observed/model label.")
        names = [str(v) for v in class_names]
    if len(set(names)) != len(names):
        raise SeeDLError("Class display names must be unique.")
    precision, recall, f1, support = metrics.precision_recall_fscore_support(
        y, pred, labels=labels, zero_division=np.nan
    )
    result: dict[str, Any] = {"accuracy": float(metrics.accuracy_score(y, pred))}
    for average in ("macro", "weighted"):
        values = metrics.precision_recall_fscore_support(
            y, pred, labels=labels, average=average, zero_division=np.nan
        )
        result.update(
            {
                f"{metric}_{average}": native(v)
                for metric, v in zip(("precision", "recall", "f1"), values[:3], strict=True)
            }
        )
    notes = [
        "Classification metrics use scikit-learn on every supplied row. Class names follow the displayed label order.",
        "Accuracy is the fraction correct. Precision measures correctness among a class's predicted samples; recall measures recovery among its true samples; F1 balances precision and recall. Support is the true sample count. Macro treats classes equally; weighted uses true support.",
        "Precision, recall and F1 with undefined denominators are unavailable (not zero); scikit-learn excludes undefined values from macro/weighted averages.",
    ]
    if any(not np.isfinite(v).all() for v in (precision, recall, f1)):
        notes.append(
            "WARNING: some per-class metrics are undefined because a class has no predicted or true samples; see support and unavailable values."
        )
    cm = metrics.confusion_matrix(y, pred, labels=labels, normalize=normalize)
    matrix = {
        "type": "matrix",
        "title": "Confusion matrix",
        "x_label": "Predicted class",
        "y_label": "True class",
        "x_labels": names,
        "y_labels": names,
        "values": native(cm),
        "description": "Counts"
        if normalize is None
        else f"Normalized by {normalize}: true = row, pred = column, all = total. Empty denominators follow scikit-learn's zero convention.",
    }
    panels = {
        "confusion_matrix": matrix,
        "class_report": _table(
            "Classification report",
            ["Class", "Label", "Precision", "Recall", "F1", "Support"],
            [
                [name, label, p, r, f, int(s)]
                for name, label, p, r, f, s in zip(
                    names, labels, precision, recall, f1, support, strict=True
                )
            ],
        ),
    }
    unavailable: dict[str, str] = {}
    model_classes = np.asarray(getattr(model, "classes_", []))
    reason = "Binary ROC/precision-recall curves require exactly two fitted classes and both classes in the supplied targets. Multiclass results use per-class and macro/weighted metrics."
    if len(model_classes) != 2 or len(unique_labels(y)) != 2 or not np.isin(y, model_classes).all():
        if positive_label is not None and len(model_classes) != 2:
            raise SeeDLError("positive_label applies only to binary classifiers.")
        unavailable.update(roc=reason, precision_recall=reason)
        return result, panels, unavailable, notes
    positive = model_classes[1] if positive_label is None else positive_label
    match = np.flatnonzero(model_classes == positive)
    if len(match) != 1:
        raise SeeDLError("positive_label must be one of the estimator's two classes_.")
    pos_index = int(match[0])
    positive_name = names[int(np.flatnonzero(labels == positive)[0])]
    scores = None
    source = ""
    try:
        if callable(getattr(model, "predict_proba", None)):
            probability = np.asarray(model.predict_proba(X))
            if (
                probability.shape != (len(y), 2)
                or not np.isfinite(probability).all()
                or np.any(probability < 0)
                or np.any(probability > 1)
                or not np.allclose(probability.sum(axis=1), 1, atol=1e-6)
            ):
                raise ValueError("predict_proba did not return valid aligned binary probabilities")
            scores, source = probability[:, pos_index], "predict_proba"
        elif callable(getattr(model, "decision_function", None)):
            scores = np.asarray(model.decision_function(X))
            if scores.shape != (len(y),) or not np.isfinite(scores).all():
                raise ValueError("decision_function did not return aligned finite binary scores")
            if pos_index == 0:
                scores = -scores
            source = "decision_function (uncalibrated margin, not probability)"
    except (ValueError, TypeError, AttributeError, NotImplementedError) as exc:
        unavailable.update(roc=str(exc), precision_recall=str(exc))
        notes.append(f"WARNING: score-based plots unavailable: {exc}.")
        return result, panels, unavailable, notes
    if scores is None:
        reason = "Estimator exposes neither predict_proba nor decision_function; probabilities are not manufactured from predicted labels."
        unavailable.update(roc=reason, precision_recall=reason)
        return result, panels, unavailable, notes
    binary_y = y == positive
    fpr, tpr, _ = metrics.roc_curve(binary_y, scores)
    pr_precision, pr_recall, _ = metrics.precision_recall_curve(binary_y, scores)
    result["roc_auc"] = float(metrics.roc_auc_score(binary_y, scores))
    result["average_precision"] = float(metrics.average_precision_score(binary_y, scores))
    for name, title, x, yy, xlabel, ylabel in (
        ("roc", "Binary ROC curve", fpr, tpr, "False positive rate", "True positive rate"),
        (
            "precision_recall",
            "Binary precision-recall curve",
            pr_recall,
            pr_precision,
            "Recall",
            "Precision",
        ),
    ):
        take = indices(len(x), max_points)
        panels[name] = {
            "type": "line",
            "line_shape": "hv" if name == "precision_recall" else "linear",
            "title": title,
            "x_label": xlabel,
            "y_label": ylabel,
            "series": [
                {
                    "name": f"Positive class: {positive_name} (label {positive})",
                    "x": native(x[take]),
                    "y": native(yy[take]),
                }
            ],
            "description": f"Positive class {positive!s}; score source: {source}. All metrics use the full curve; displayed curve vertices: {len(take)} of {len(x)}.",
        }
    notes.append(
        f"Binary positive label: {positive!s} ({positive_name}); scores from {source}. ROC-AUC and average precision are computed before display sampling. Average precision is not trapezoidal PR area."
    )
    return result, panels, unavailable, notes


def regression(
    y: Any, pred: Any, *, max_points: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], list[str]]:
    import numpy as np
    from sklearn import metrics

    if not np.isfinite(pred).all():
        raise SeeDLError(
            "Regression predictions contain NaN or Inf; metrics are unavailable. Inspect the estimator and data."
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with np.errstate(over="warn", invalid="warn", divide="warn"):
            rmse = (
                metrics.root_mean_squared_error(y, pred)
                if hasattr(metrics, "root_mean_squared_error")
                else metrics.mean_squared_error(y, pred, squared=False)
            )
            r2 = metrics.r2_score(y, pred, force_finite=False)
            result = {
                "mae": float(metrics.mean_absolute_error(y, pred)),
                "mse": float(metrics.mean_squared_error(y, pred)),
                "rmse": float(rmse),
                "r2": native(r2),
            }
    notes = [
        f"Metrics use scikit-learn on all {len(y):,} supplied rows. Residual = actual minus predicted; MAE/RMSE use target units, MSE uses squared target units; R2 is dimensionless."
    ]
    notes.append(
        "MAE/RMSE summarize target-sized errors; lower is better. R2=1 is perfect on these rows, R2=0 matches the constant mean-target baseline, and negative R2 is worse than that baseline. These are not confidence estimates."
    )
    notes.extend(f"WARNING: {w.message}" for w in caught)
    if not np.isfinite(r2):
        notes.append(
            f"WARNING: R2 is undefined/non-finite ({r2}); constant or fewer than two targets do not establish explained variance. It is shown as unavailable, not zero."
        )
    if len(y) < 20:
        notes.append(
            "CAUTION: fewer than 20 evaluation samples; metrics may be unstable and should not be generalized without more evidence."
        )
    take = indices(len(y), max_points)
    with np.errstate(over="ignore", invalid="ignore"):
        residual = y - pred
    nonfinite_metrics = [k for k, v in result.items() if v is not None and not np.isfinite(v)]
    if nonfinite_metrics:
        notes.append(
            f"WARNING: non-finite metrics ({', '.join(nonfinite_metrics)}) from numerical overflow are unavailable, not zero. Consider rescaling target units explicitly."
        )
    if result["mse"] == 0 and np.any(residual != 0):
        notes.append(
            "WARNING: squared-error calculation underflowed to zero for nonzero residuals in scikit-learn. Rescale target units explicitly before interpreting MSE/RMSE."
        )
    if not np.isfinite(residual).all():
        notes.append(
            "WARNING: residual subtraction overflowed despite finite inputs; non-finite plotted values are unavailable, not zero."
        )
    panels: dict[str, Any] = {}
    for name, title, x, yy, xlabel, ylabel in (
        (
            "actual_predicted",
            "Actual versus predicted",
            y,
            pred,
            "Actual target (target units)",
            "Predicted target (target units)",
        ),
        (
            "residuals",
            "Residual versus predicted",
            pred,
            residual,
            "Predicted target (target units)",
            "Residual: actual - predicted (target units)",
        ),
    ):
        panels[name] = {
            "type": "scatter",
            "title": title,
            "x_label": xlabel,
            "y_label": ylabel,
            "series": [{"name": "Evaluation samples", "x": native(x[take]), "y": native(yy[take])}],
            "reference": "identity" if name == "actual_predicted" else "zero",
            "description": f"Showing {len(take):,} of {len(y):,} aligned rows; metrics use all rows.",
        }
    unavailable: dict[str, str] = {}
    try:
        if not np.isfinite(residual).all():
            raise ValueError("residuals contain non-finite values")
        span = float(np.max(residual)) - float(np.min(residual))
        if not np.isfinite(span):
            raise ValueError("residual range overflows floating-point bin coordinates")
        with warnings.catch_warnings(record=True) as histogram_warnings:
            warnings.simplefilter("always")
            counts, edges = np.histogram(residual, bins=min(30, max(1, int(np.sqrt(len(y))))))
        if not np.isfinite(edges).all():
            raise ValueError("histogram bin edges overflowed")
        panels["residual_distribution"] = {
            "type": "histogram",
            "title": "Residual distribution",
            "x_label": "Residual (target units)",
            "y_label": "Sample count",
            "edges": native(edges),
            "counts": native(counts),
            "description": "Histogram uses every supplied evaluation residual; bins are display summaries, not a probability density.",
        }
        notes.extend(f"WARNING: {w.message}" for w in histogram_warnings)
        if np.all(residual == residual[0]):
            notes.append(
                f"Every residual equals {residual[0]!r}. Histogram bin width is a display interval, not observed dispersion or uncertainty."
            )
    except (ValueError, OverflowError, IndexError) as exc:
        unavailable["residual_distribution"] = (
            f"Residuals cannot be safely binned at the current numeric scale: {exc}. Rescale target units explicitly."
        )
        notes.append("WARNING: residual histogram unavailable at this numeric scale.")
    return native(result), panels, unavailable, notes
