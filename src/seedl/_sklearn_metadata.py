"""Read estimator metadata without fitting, predicting, or transforming input data."""

from __future__ import annotations

from typing import Any

from ._sklearn_metrics import _table
from ._sklearn_validation import fitted_status, input_names, native, require_estimator


def _parameter(value: Any, depth: int = 0) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return native(value[:200] if isinstance(value, str) else value)
    if depth < 2 and isinstance(value, (list, tuple)):
        return [_parameter(v, depth + 1) for v in value[:8]]
    if depth < 2 and isinstance(value, dict):
        return {str(k)[:80]: _parameter(v, depth + 1) for k, v in list(value.items())[:8]}
    if isinstance(value, slice):
        return f"slice({value.start}, {value.stop}, {value.step})"
    if hasattr(value, "shape"):
        return {"type": type(value).__name__, "shape": native(value.shape)}
    return f"<{type(value).__module__}.{type(value).__name__}>"


def final_estimator(model: Any) -> Any:
    from sklearn.pipeline import Pipeline

    model = getattr(model, "best_estimator_", model)
    if isinstance(model, Pipeline):
        model = model.steps[-1][1]
    return model


def feature_names(model: Any, count: int) -> tuple[list[str], str]:
    """Never substitute original names for transformed model coefficients."""
    from sklearn.pipeline import Pipeline

    model = getattr(model, "best_estimator_", model)
    if isinstance(model, Pipeline):
        try:
            names = model[:-1].get_feature_names_out()
            if len(names) == count:
                return [
                    str(v) for v in names
                ], "transformed features from fitted preprocessing get_feature_names_out()"
        except (AttributeError, ValueError, TypeError):
            pass
        return [
            f"transformed feature {i}" for i in range(count)
        ], "transformed features; names unavailable (original columns are not substituted)"
    names = input_names(model)
    return (
        names if len(names) == count else [f"feature {i}" for i in range(count)]
    ), "estimator input features"


def inspect_estimator(
    model: Any, *, max_depth: int = 3, max_nodes: int = 63, max_features: int = 50
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    import numpy as np

    require_estimator(model, fitted=False)
    fitted = fitted_status(model)
    model_type = f"{type(model).__module__}.{type(model).__name__}"
    params = model.get_params(deep=False)
    rows = [[str(k), _parameter(v)] for k, v in list(params.items())[:100]]
    panels: dict[str, Any] = {
        "parameters": _table(
            "Estimator parameters",
            ["Parameter", "Value"],
            rows,
            "Constructor settings; no fitting or prediction is performed.",
        )
    }
    data: dict[str, Any] = {
        "task": "inspection",
        "fitted": fitted,
        "estimator_type": model_type,
        "parameters": {str(k): _parameter(v) for k, v in list(params.items())[:100]},
        "panels": panels,
    }
    meta = {"Estimator": type(model).__name__, "Fitted": fitted}
    notes = [
        "Inspection reads existing attributes only; fit(), predict(), and transform() are not called."
    ]
    if len(params) > 100:
        notes.append(f"Showing 100 of {len(params)} constructor parameters.")
    if not fitted:
        notes.append(
            "Estimator is unfitted; fitted coefficients, importances and tree structure are unavailable. Train it in your own code."
        )
    count = getattr(model, "n_features_in_", None)
    if count is not None:
        meta["Input features"] = int(count)
        data["input_feature_names"] = input_names(model, limit=500)
        if int(count) > 500:
            notes.append(f"Input feature-name preview capped at 500 of {count} features.")
    final = final_estimator(model)
    coeff = getattr(final, "coef_", None) if fitted else None
    importance = getattr(final, "feature_importances_", None) if fitted else None
    if coeff is not None:
        from scipy import sparse

        if sparse.issparse(coeff):
            if coeff.shape[0] * coeff.shape[1] > 100_000:
                coeff = None
                notes.append(
                    "Coefficient display unavailable: sparse coefficient matrix exceeds the 100,000-cell display bound; no densification performed."
                )
            else:
                coeff = coeff.toarray()
        if coeff is not None:
            coeff = np.atleast_2d(coeff)
            if coeff.size > 100_000:
                notes.append(
                    "Coefficient display unavailable: coefficient matrix exceeds the 100,000-cell bound."
                )
            else:
                names, space = feature_names(model, coeff.shape[1])
                classes = getattr(final, "classes_", None)
                coeff_rows = []
                for j, values in enumerate(coeff[:10]):
                    target = (
                        str(classes[1])
                        if classes is not None and len(classes) == 2 and coeff.shape[0] == 1
                        else (
                            str(classes[j])
                            if classes is not None and len(classes) == coeff.shape[0]
                            else f"output {j}"
                        )
                    )
                    take = np.argsort(-np.abs(values), kind="stable")[:max_features]
                    coeff_rows.extend([[target, names[int(i)], float(values[i])] for i in take])
                panels["coefficients"] = _table(
                    "Linear / logistic coefficients",
                    ["Output or class", "Feature", "Coefficient"],
                    coeff_rows,
                    f"{space}; signed coefficients, ranked by absolute magnitude per output. Binary logistic coefficients describe classes_[1] log-odds, not probabilities.",
                )
                data["coefficient_space"] = space
                data["intercept"] = native(getattr(final, "intercept_", None))
                meta["Intercept"] = data["intercept"]
                notes.append(
                    f"Coefficient preview: at most {max_features} features per output and 10 outputs. Coefficients depend on encoding/scaling and are associations, not causal effects. Space: {space}."
                )
    if importance is not None:
        values = np.asarray(importance)
        if values.ndim == 1 and len(values) <= 100_000:
            names, space = feature_names(model, len(values))
            take = np.argsort(-values, kind="stable")[:max_features]
            panels["feature_importance"] = {
                "type": "bar",
                "title": "Impurity-based feature importance",
                "labels": [names[int(i)] for i in take],
                "values": native(values[take]),
                "x_label": "Feature",
                "y_label": "Mean impurity decrease (dimensionless)",
                "description": f"{space}; top {len(take)} of {len(values)}. Impurity importance is biased toward high-cardinality features and does not measure causality or held-out predictive value.",
            }
            notes.append(
                "Impurity-based importance uses existing feature_importances_; no permutation or refitting is implied. It can favor high-cardinality features and does not establish causality or held-out predictive value."
            )
    tree = getattr(final, "tree_", None) if fitted else None
    if tree is not None:
        names, space = feature_names(model, int(final.n_features_in_))
        records: list[dict[str, Any]] = []
        queue = [(0, 0)]
        while queue and len(records) < max_nodes:
            node, depth = queue.pop(0)
            leaf = int(tree.children_left[node]) == int(tree.children_right[node])
            feature = "leaf" if leaf else names[int(tree.feature[node])]
            records.append(
                {
                    "node": node,
                    "depth": depth,
                    "feature": feature,
                    "threshold": None if leaf else float(tree.threshold[node]),
                    "left": None if leaf else int(tree.children_left[node]),
                    "right": None if leaf else int(tree.children_right[node]),
                    "samples": int(tree.n_node_samples[node]),
                    "impurity": float(tree.impurity[node]),
                }
            )
            if not leaf and depth < max_depth:
                queue.extend(
                    [
                        (int(tree.children_left[node]), depth + 1),
                        (int(tree.children_right[node]), depth + 1),
                    ]
                )
        data["tree_nodes"] = records
        panels["tree"] = _table(
            "Decision tree split inspection",
            [
                "Node",
                "Depth",
                "Feature",
                "Threshold (<= left)",
                "Left child",
                "Right child",
                "Samples",
                "Impurity",
            ],
            [
                [
                    r[k]
                    for k in (
                        "node",
                        "depth",
                        "feature",
                        "threshold",
                        "left",
                        "right",
                        "samples",
                        "impurity",
                    )
                ]
                for r in records
            ],
            f"{space}. Showing {len(records)} of {tree.node_count} nodes, depth <= {max_depth}; children beyond the preview remain in the estimator. Missing-value routing, where supported, follows the estimator.",
        )
    from ._sklearn_interpret import pipeline_structure

    diagram, diagram_notes = pipeline_structure(model, max_nodes=max_nodes)
    if diagram is not None:
        panels["pipeline"] = diagram
        notes.extend(diagram_notes)
        for candidate, key in ((model, "transformed_feature_names"),):
            try:
                from sklearn.pipeline import Pipeline

                prefix = candidate[:-1] if isinstance(candidate, Pipeline) else candidate
                expanded = prefix.get_feature_names_out()
                data[key] = [str(v) for v in expanded[:500]]
                if len(expanded) > 500:
                    notes.append(
                        f"Expanded feature-name preview capped at 500 of {len(expanded)} transformed names."
                    )
            except (AttributeError, ValueError, TypeError):
                notes.append(
                    "Expanded preprocessing feature names are unavailable; transformed columns are not equated with original inputs."
                )
    if fitted and hasattr(final, "labels_"):
        labels, counts = np.unique(final.labels_, return_counts=True)
        take = np.argsort(-counts, kind="stable")[:max_features]
        panels["cluster_sizes"] = _table(
            "Existing training cluster sizes",
            ["Cluster ID", "Training samples"],
            [[int(labels[i]), int(counts[i])] for i in take],
            f"Existing labels_ only; top {len(take)} of {len(labels)} labels. -1 denotes noise. These labels cannot be aligned to new rows automatically.",
        )
    if fitted and hasattr(final, "explained_variance_ratio_"):
        ratio = np.asarray(final.explained_variance_ratio_)
        panels["explained_variance"] = {
            "type": "bar",
            "title": "Fitted explained variance",
            "labels": [f"PC {i + 1}" for i in range(min(len(ratio), max_features))],
            "values": native(ratio[:max_features]),
            "x_label": "Fitted component",
            "y_label": "Explained variance ratio (dimensionless)",
            "description": "Existing explained_variance_ratio_; not re-estimated on evaluation data. See sd.evaluate(pca, X) for a projection without refitting.",
        }
        if not np.isfinite(ratio).all():
            notes.append(
                "WARNING: fitted explained variance contains undefined/non-finite values; unavailable values are not zero."
            )
    if fitted and hasattr(model, "cv_results_"):
        from ._sklearn_selection import cv_results

        keys = [
            k.removeprefix("mean_test_") for k in model.cv_results_ if k.startswith("mean_test_")
        ]
        chosen_metric = "score" if "score" in keys else (keys[0] if keys else None)
        if chosen_metric is not None:
            cv = cv_results(model, metric=chosen_metric, max_candidates=max_features, backend="svg")
            panels.update(cv.data["panels"])
            notes.extend(cv.notes)
            notes.append(
                f"Search preview metric: {chosen_metric}. Other captured metrics can be selected with sd.cv_results(search, metric=...)."
            )
    return native(data), native(meta), notes
