"""Captured clustering and fitted PCA diagnostics; no model fitting or automatic PCA."""

from __future__ import annotations

from typing import Any

from ._sklearn_validation import (
    bounded_int,
    indices,
    input_names,
    native,
    predict_once,
    require_estimator,
    rows_at,
)
from .exceptions import SeeDLError


def task_for(model: Any) -> str | None:
    from sklearn.base import ClusterMixin, is_classifier, is_regressor
    from sklearn.decomposition import PCA

    from ._sklearn_metadata import final_estimator

    if is_classifier(model):
        return "classification"
    if is_regressor(model):
        return "regression"
    final = final_estimator(model)
    if isinstance(final, ClusterMixin):
        return "clustering"
    if isinstance(final, PCA):
        return "projection"
    return None


def _feature_space(model: Any, sample: Any) -> tuple[Any, list[str], str]:
    from sklearn.pipeline import Pipeline

    from ._sklearn_metadata import feature_names

    fitted = getattr(model, "best_estimator_", model)
    if isinstance(fitted, Pipeline):
        transformed = fitted[:-1].transform(sample.copy())
        if not hasattr(transformed, "shape") or len(transformed.shape) != 2:
            raise SeeDLError("Preprocessing must produce a two-dimensional feature matrix.")
        if transformed.shape[1] <= 10000:
            names, space = feature_names(model, int(transformed.shape[1]))
        else:
            names, space = (
                ["transformed feature 0", "transformed feature 1"],
                "transformed features (names omitted for bounded preview)",
            )
        return transformed, names, space
    return sample, input_names(model, sample, limit=2), "estimator input feature space"


def _projection_series(
    coordinates: Any, groups: Any, max_points: int, names: list[str]
) -> tuple[list[dict[str, Any]], str, str]:
    import numpy as np
    from scipy import sparse

    if coordinates.shape[1] < 1:
        raise SeeDLError("No projection features are available.")
    first = (
        coordinates.iloc[:, : min(2, coordinates.shape[1])]
        if hasattr(coordinates, "iloc")
        else coordinates[:, : min(2, coordinates.shape[1])]
    )
    if sparse.issparse(first):
        first = first.toarray()  # bounded preview, <= max_points x 2
    first = np.asarray(first, dtype=float)
    if not np.isfinite(first).all():
        raise SeeDLError("Displayed coordinates contain NaN/Inf; a finite projection is required.")
    one = first.shape[1] == 1
    xx = np.arange(len(first)) if one else first[:, 0]
    yy = first[:, 0] if one else first[:, 1]
    series = []
    for label in np.unique(groups):
        selected = groups == label
        series.append(
            {
                "name": "Noise (-1)" if label == -1 else f"Cluster {label}",
                "x": native(xx[selected]),
                "y": native(yy[selected]),
                "noise": bool(label == -1),
            }
        )
    return series, "Preview sample index" if one else names[0], names[0] if one else names[1]


def clustering(
    model: Any,
    X: Any,
    *,
    labels: Any,
    split: str,
    projection: Any,
    max_points: int,
    silhouette_max_samples: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], list[str], dict[str, Any]]:
    import numpy as np
    from scipy import sparse
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_samples, silhouette_score
    from sklearn.utils.multiclass import type_of_target

    from ._sklearn_metadata import final_estimator

    bounded_int("silhouette_max_samples", silhouette_max_samples, 1000)
    final = final_estimator(model)
    if getattr(final, "metric", None) == "precomputed":
        raise SeeDLError(
            "Precomputed-distance clustering is inspectable with sd.see(), but feature-space projection/evaluation is unsupported. Use scikit-learn's precomputed-distance metrics explicitly."
        )
    n = X.shape[0]
    predicted = labels is None
    if labels is None:
        if not callable(getattr(model, "predict", None)):
            raise SeeDLError(
                "This clusterer (including DBSCAN) cannot predict labels for new rows. Supply labels=model.labels_ with X in the exact original training-row order and split='training'. SeeDL cannot recover or infer that alignment."
            )
        groups = predict_once(model, X, n)
    else:
        if (
            hasattr(X, "index")
            and hasattr(labels, "index")
            and not callable(labels.index)
            and not X.index.equals(labels.index)
        ):
            raise SeeDLError("X and cluster-label indices are not aligned.")
        groups = np.asarray(labels)
        if not callable(getattr(model, "predict", None)) and split != "training":
            raise SeeDLError(
                "Clusterers without predict(), such as DBSCAN, require split='training' with explicitly aligned labels_."
            )
    if (
        groups.ndim != 1
        or len(groups) != n
        or type_of_target(groups) not in {"binary", "multiclass"}
    ):
        raise SeeDLError("Cluster labels must be one aligned finite integer cluster ID per sample.")
    if not np.issubdtype(groups.dtype, np.integer):
        raise SeeDLError("Cluster IDs must be integers, with -1 reserved for noise.")
    unique, counts = np.unique(groups, return_counts=True)
    if len(unique) > 100:
        raise SeeDLError(
            "Clustering charts support at most 100 cluster IDs. Use sd.see(model) for a bounded label-count summary, or an explicit suitable subset."
        )
    notes = [
        f"Cluster assignment source: {'one predict() call on supplied rows' if predicted else 'explicit caller-supplied labels; caller must preserve training-row alignment'}. No fit() or automatic dimensionality-reduction fit is performed."
    ]
    take = indices(n, max_points)
    plot_sample = rows_at(X, take)
    if projection is not None:
        require_estimator(projection)
        if not isinstance(final_estimator(projection), PCA):
            raise SeeDLError(
                "projection must be a previously fitted PCA or a fitted pipeline ending in PCA; no automatic projection fitting is permitted."
            )
        expected = getattr(projection, "n_features_in_", None)
        if expected is not None and int(expected) != X.shape[1]:
            raise SeeDLError("Projection expects a different original input feature count.")
        projected = projection.transform(plot_sample.copy())
        names = [f"PC {i + 1} (projected coordinate)" for i in range(projected.shape[1])]
        plot_space = "fitted PCA projection (not original-space distances)"
        notes.append(
            "PCA projection is a display only. Distances, cluster boundaries and silhouette scores are not measured in the 2D PCA display."
        )
    else:
        projected, names, plot_space = _feature_space(model, plot_sample)
        notes.append(
            "Cluster plot shows the first one/two coordinates of the estimator's feature space. Additional dimensions are omitted; this is not a fitted PCA projection."
        )
    series, xlabel, ylabel = _projection_series(projected, groups[take], max_points, names)
    panels = {
        "clusters": {
            "type": "scatter",
            "title": "Cluster assignments",
            "x_label": xlabel,
            "y_label": ylabel,
            "series": series,
            "description": f"{plot_space}; {len(take)} of {n} rows shown. Noise is labeled -1; cluster sizes use every row.",
        },
        "cluster_sizes": {
            "type": "bar",
            "title": "Cluster sizes (all supplied rows)",
            "labels": ["Noise (-1)" if v == -1 else f"Cluster {v}" for v in unique],
            "values": native(counts),
            "x_label": "Cluster ID",
            "y_label": "Sample count",
            "description": "Counts include noise. Cluster IDs are nominal labels, not ordered quality scores.",
        },
    }
    result: dict[str, Any] = {
        "clusters": int(np.sum(unique != -1)),
        "noise_samples": int(np.sum(groups == -1)),
    }
    unavailable = {}
    eligible = np.flatnonzero(groups != -1)
    rng = np.random.RandomState(0)
    chosen = (
        np.sort(rng.choice(eligible, min(len(eligible), silhouette_max_samples), replace=False))
        if len(eligible)
        else eligible
    )
    sil_groups = groups[chosen]
    if len(np.unique(sil_groups)) < 2 or len(np.unique(sil_groups)) >= len(sil_groups):
        reason = "Silhouette requires 2 <= non-noise clusters <= non-noise samples - 1. The bounded sample may omit rare clusters."
        unavailable["silhouette"] = reason
        result["silhouette"] = None
        notes.append("Silhouette unavailable: " + reason)
    else:
        metric_X, _, metric_space = _feature_space(model, rows_at(X, chosen))
        if (sparse.issparse(metric_X) and metric_X.nnz > 2_000_000) or (
            not sparse.issparse(metric_X) and metric_X.shape[0] * metric_X.shape[1] > 2_000_000
        ):
            raise SeeDLError(
                "Silhouette feature matrix exceeds the 2,000,000-value safety bound; supply fewer rows/features or use explicit sklearn computation."
            )
        metric = getattr(final, "metric", "euclidean")
        kwargs = dict(getattr(final, "metric_params", None) or {})
        if metric == "minkowski" and getattr(final, "p", None) is not None:
            kwargs.setdefault("p", final.p)
        try:
            result["silhouette"] = float(
                silhouette_score(metric_X, sil_groups, metric=metric, **kwargs)
            )
            scores = silhouette_samples(metric_X, sil_groups, metric=metric, **kwargs)
        except (ValueError, TypeError) as exc:
            raise SeeDLError(
                f"Silhouette is unsupported for this feature/distance combination: {exc}"
            ) from exc
        series_sil = []
        offset = 0
        for group in np.unique(sil_groups):
            values = np.sort(scores[sil_groups == group])
            xx = np.arange(offset, offset + len(values))
            offset += len(values)
            series_sil.append({"name": f"Cluster {group}", "x": native(xx), "y": native(values)})
        panels["silhouette"] = {
            "type": "scatter",
            "title": "Silhouette analysis (bounded non-noise sample)",
            "x_label": "Samples sorted within cluster (not original row order)",
            "y_label": "Silhouette coefficient (dimensionless)",
            "series": series_sil,
            "reference": "zero",
            "description": f"scikit-learn silhouette, distance={metric}, space={metric_space}; {len(chosen)} of {len(eligible)} non-noise rows, random_state=0. Noise excluded; metrics do not use display PCA coordinates.",
        }
        notes.append(
            f"Silhouette uses {len(chosen)} of {len(eligible)} non-noise rows, sampled without replacement with random_state=0, in {metric_space} using {metric}. It measures cluster separation, not external-label accuracy."
        )
    extra = {
        "cluster_ids": native(unique),
        "cluster_counts": native(counts),
        "display_sample_indices": native(take),
        "silhouette_sample_indices": native(chosen),
        "prediction_calls": 1 if predicted else 0,
    }
    return result, panels, unavailable, notes, extra


def projection(
    model: Any, X: Any, *, max_points: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], list[str], dict[str, Any]]:
    import numpy as np

    from ._sklearn_metadata import final_estimator

    final = final_estimator(model)
    take = indices(X.shape[0], max_points)
    transformed = model.transform(rows_at(X, take))
    if len(transformed.shape) != 2 or transformed.shape[1] < 1:
        raise SeeDLError("PCA transform must expose at least one component.")
    series, xlabel, ylabel = _projection_series(
        transformed,
        np.zeros(len(take), dtype=int),
        max_points,
        [f"PC {i + 1} (projected coordinate)" for i in range(transformed.shape[1])],
    )
    series[0]["name"] = "Projected samples"
    variance = np.asarray(final.explained_variance_ratio_)
    panels = {
        "pca_projection": {
            "type": "scatter",
            "title": "Fitted PCA projection",
            "x_label": xlabel,
            "y_label": ylabel,
            "series": series,
            "description": f"First {min(2, transformed.shape[1])} of {transformed.shape[1]} fitted components; {len(take)} of {X.shape[0]} rows displayed. Projected distances are not original feature-space distances.",
        },
        "explained_variance": {
            "type": "bar",
            "title": "Explained variance of fitted PCA",
            "labels": [f"PC {i + 1}" for i in range(min(100, len(variance)))],
            "values": native(variance[:100]),
            "x_label": "Fitted principal component",
            "y_label": "Explained variance ratio (dimensionless)",
            "description": f"Existing explained_variance_ratio_, not re-estimated on evaluation rows. First {min(100, len(variance))} of {len(variance)} components shown.",
        },
    }
    notes = [
        "PCA uses the fitted transform() only, on bounded display rows. No fit(), fit_transform(), or new PCA model is called.",
        "Variance ratios describe the PCA fitting population. A 2D projection omits other components and must not be interpreted as preserving original-space distances.",
    ]
    if not np.isfinite(variance).all():
        notes.append(
            "WARNING: fitted PCA variance ratios are non-finite (for example constant training data); unavailable entries are not replaced with zero."
        )
    return (
        {
            "components": int(len(variance)),
            "displayed_variance_ratio": native(np.sum(variance[:2])),
        },
        panels,
        {},
        notes,
        {"display_sample_indices": native(take), "prediction_calls": 0, "transform_calls": 1},
    )
