"""Explicit, bounded interpretation through scikit-learn; never refits estimators."""

from __future__ import annotations

import warnings
from typing import Any

from ._sklearn_validation import (
    bounded_int,
    dependencies,
    indices,
    input_data,
    input_names,
    native,
    read_only_response,
    require_estimator,
    rows_at,
)
from .evaluation import Evaluation
from .exceptions import SeeDLError
from .renderers import resolve_backend


def pipeline_structure(
    model: Any, *, max_nodes: int = 63
) -> tuple[dict[str, Any] | None, list[str]]:
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    from ._sklearn_metadata import _parameter

    model = getattr(model, "best_estimator_", model)
    if not isinstance(model, (Pipeline, ColumnTransformer)):
        return None, []
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    def visit(
        obj: Any, name: str, parent: int | None, depth: int, role: str, columns: Any = None
    ) -> int | None:
        if len(nodes) >= max_nodes or depth > 8:
            return None
        ident = len(nodes)
        nodes.append(
            {
                "id": ident,
                "name": name,
                "estimator": obj if isinstance(obj, str) else type(obj).__name__,
                "parent": parent,
                "depth": depth,
                "role": role,
                "columns": _parameter(columns),
            }
        )
        if parent is not None:
            edges.append({"source": parent, "target": ident, "relation": role})
        if isinstance(obj, Pipeline):
            previous = ident
            for step, transformer in obj.steps:
                child = visit(transformer, step, ident, depth + 1, "sequential stage")
                if child is not None and previous != ident:
                    edges.append(
                        {"source": previous, "target": child, "relation": "sequential flow"}
                    )
                if child is not None:
                    previous = child
        elif isinstance(obj, ColumnTransformer):
            for step, transformer, cols in getattr(obj, "transformers_", obj.transformers):
                visit(transformer, step, ident, depth + 1, "parallel column branch", cols)
        return ident

    visit(
        model,
        "input → fitted pipeline" if hasattr(model, "n_features_in_") else "pipeline (unfitted)",
        None,
        0,
        "root",
    )
    notes = [
        f"Pipeline diagram shows declared sequential stages and parallel ColumnTransformer branches (not a runtime trace). Preview bounded to {max_nodes} nodes and 8 nesting levels. Branch column selectors refer to original input columns; branch outputs are concatenated in declared order."
    ]
    return {
        "type": "pipeline",
        "title": "Pipeline and column transformations",
        "nodes": nodes,
        "edges": edges,
        "description": "Indentation shows containment; labeled edges distinguish sequential flow from parallel column branches. Feature names are expanded only when the fitted preprocessing exposes a reliable mapping.",
    }, notes


def feature_importance(
    model: Any,
    X: Any = None,
    y: Any = None,
    *,
    method: str = "native",
    split: str = "unspecified",
    scoring: Any = None,
    n_repeats: int = 5,
    max_samples: int = 512,
    max_features: int = 50,
    random_state: int = 0,
    backend: str = "auto",
) -> Evaluation:
    np, sklearn = dependencies()
    from sklearn.base import is_classifier, is_regressor
    from sklearn.inspection import permutation_importance

    from ._sklearn_metadata import inspect_estimator
    from ._sklearn_metrics import _table

    require_estimator(model)
    bounded_int("max_features", max_features, 100)
    bounded_int("max_samples", max_samples, 2000)
    bounded_int("n_repeats", n_repeats, 20)
    if split not in {"unspecified", "training", "validation", "test"}:
        raise ValueError("split must be unspecified, training, validation, or test.")
    notes: list[str] = []
    if method == "native":
        if X is not None or y is not None:
            raise SeeDLError(
                "Native importance uses fitted attributes only. Select method='permutation' to use supplied evaluation data."
            )
        data, meta, notes = inspect_estimator(model, max_features=max_features)
        panels = {
            k: p for k, p in data["panels"].items() if k in {"feature_importance", "coefficients"}
        }
        if not panels:
            raise SeeDLError(
                "Estimator has no native feature_importances_ or coef_. Use method='permutation' with explicit X and y when supported."
            )
        data["panels"] = panels
        data["method"] = "native fitted attributes"
    elif method == "permutation":
        X, y, n, p = input_data(model, X, y, require_y=is_classifier(model) or is_regressor(model))
        if p > 200:
            raise SeeDLError(
                "Permutation importance is bounded to 200 input features; this method permutes every input feature. Use suitable upstream feature selection or scikit-learn directly for a larger explicit computation."
            )
        selected = np.sort(
            np.random.RandomState(random_state).choice(n, min(n, max_samples), replace=False)
        )
        sample = rows_at(X, selected)
        target = y[selected] if y is not None else None
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = permutation_importance(
                    read_only_response(model),
                    sample,
                    target,
                    scoring=scoring,
                    n_repeats=n_repeats,
                    random_state=random_state,
                    n_jobs=1,
                )
        except (ValueError, TypeError) as exc:
            raise SeeDLError(
                f"scikit-learn permutation importance does not support this estimator/data/scorer combination: {exc}. Sparse data are never silently densified."
            ) from exc
        if isinstance(result, dict) and "importances_mean" not in result:
            raise SeeDLError("Choose one scoring metric, not a multi-metric scoring collection.")
        names = input_names(model, X)
        take = np.argsort(-result.importances_mean, kind="stable")[:max_features]
        mean, std = result.importances_mean, result.importances_std
        panels = {
            "permutation_importance": {
                "type": "bar",
                "title": "Permutation feature importance",
                "labels": [names[int(i)] for i in take],
                "values": native(mean[take]),
                "errors": native(std[take]),
                "x_label": "Original input feature",
                "y_label": "Decrease in evaluation score (scorer units)",
                "description": f"Mean score decrease over {n_repeats} permutations; error values are repeat standard deviations, not confidence intervals. Score: {scoring or 'estimator.score'}. Negative values mean a higher score after permutation.",
            },
            "permutation_details": _table(
                "Permutation details",
                ["Input feature", "Mean score decrease", "Repeat std. deviation"],
                [[names[int(i)], mean[i], std[i]] for i in take],
            ),
        }
        data = {
            "task": "interpretation",
            "method": "permutation",
            "panels": panels,
            "importances": native(result.importances[take]),
            "sample_indices": native(selected),
            "provenance": {
                "sklearn_version": sklearn.__version__,
                "split": split,
                "scoring": str(scoring or "estimator.score"),
                "n_repeats": n_repeats,
                "input_feature_space": "original input columns",
            },
        }
        meta = {
            "Estimator": type(model).__name__,
            "Method": "permutation (scikit-learn)",
            "Split": split,
            "Reference samples": len(selected),
            "Input features": p,
            "Permutations per feature": n_repeats,
        }
        notes = [
            f"Permutation importance uses {len(selected)} of {n} supplied rows, sampled without replacement with random_state={random_state}. Every input column is permuted through the entire estimator/pipeline; no transformed features are relabeled as original columns.",
            "Importance depends on the scorer, sample and correlated features; it is not a causal effect. Negative values and repeat uncertainty are preserved.",
            "No fit() call is made. Permutation invokes the scorer repeatedly; this is an explicit opt-in diagnostic.",
        ]
        notes.extend(f"WARNING: {w.message}" for w in caught)
        if not np.isfinite(mean).all():
            notes.append(
                "WARNING: some permutation scores are non-finite and are shown as unavailable, not zero."
            )
        if split == "training":
            notes.append(
                "CAUTION: permutation importance was measured on training data; use a suitable held-out validation/test set to assess generalization."
            )
        if split == "unspecified":
            notes.append(
                "Reference split is unspecified; set split= explicitly to preserve training/validation/test provenance."
            )
    else:
        raise ValueError("method must be 'native' or 'permutation'.")
    return Evaluation(
        kind="ml_importance",
        title=f"{type(model).__name__} feature interpretation",
        summary=f"{method} method | no fitting | feature space disclosed in each plot",
        data=native(data),
        metadata=native(meta),
        notes=notes,
        backend=resolve_backend(backend),
    )


def partial_dependence(
    model: Any,
    X: Any,
    features: Any,
    *,
    kind: str = "average",
    target: Any = None,
    grid_resolution: int = 30,
    max_samples: int = 256,
    max_ice: int = 20,
    backend: str = "auto",
) -> Evaluation:
    from numbers import Integral

    np, sklearn = dependencies()
    from sklearn.base import is_classifier, is_regressor
    from sklearn.inspection import partial_dependence as compute

    require_estimator(model)
    if kind not in {"average", "individual", "both"}:
        raise ValueError("kind must be average, individual, or both.")
    bounded_int("grid_resolution", grid_resolution, 50)
    if grid_resolution < 2:
        raise ValueError("grid_resolution must be at least 2.")
    bounded_int("max_samples", max_samples, 512)
    bounded_int("max_ice", max_ice, 50)
    if not (is_classifier(model) or is_regressor(model)):
        raise SeeDLError(
            "Partial dependence requires a fitted classifier or regressor supported by scikit-learn."
        )
    X, _, n, p = input_data(model, X)
    names = input_names(model, X)
    chosen = [features] if isinstance(features, (str, Integral)) else list(features)
    if not 1 <= len(chosen) <= 4:
        raise SeeDLError(
            "Select one to four individual numeric input features; interaction surfaces are not supported."
        )
    columns = []
    for feature in chosen:
        if isinstance(feature, str):
            if feature not in names:
                raise SeeDLError(f"Unknown input feature {feature!r}.")
            index = names.index(feature)
        elif (
            isinstance(feature, Integral)
            and not isinstance(feature, bool)
            and 0 <= int(feature) < p
        ):
            index = int(feature)
        else:
            raise SeeDLError(
                "Features must be valid original-input column names or integer indices."
            )
        columns.append(index)
    classes = getattr(model, "classes_", None)
    output = 0
    if is_classifier(model):
        if classes is None or len(classes) < 2 or len(classes) > 20:
            raise SeeDLError("Partial-dependence classification supports 2–20 fitted classes.")
        if len(classes) > 2 and target is None:
            raise SeeDLError(
                "Multiclass partial dependence requires an explicit target class label."
            )
        label = classes[1] if target is None else target
        found = np.flatnonzero(np.asarray(classes) == label)
        if len(found) != 1 or (len(classes) == 2 and int(found[0]) != 1):
            raise SeeDLError(
                "Binary partial dependence follows scikit-learn's classes_[1] response; multiclass target must be a fitted class label."
            )
        output = 0 if len(classes) == 2 else int(found[0])
        response = (
            "predicted probability"
            if callable(getattr(model, "predict_proba", None))
            else "uncalibrated decision score (not probability)"
        )
        ylabel = (
            f"Mean {response}, class {label}"
            if kind != "individual"
            else f"{response}, class {label}"
        )
    else:
        if target is not None:
            raise SeeDLError(
                "Single-output regression partial dependence does not accept a class target."
            )
        ylabel = "Predicted target (target units)"
    selected = indices(n, max_samples)
    sample = rows_at(X, selected)
    panels = {}
    notes = [
        f"Partial dependence uses scikit-learn method='brute' on {len(selected)} of {n} reference rows (evenly spaced when bounded), grid_resolution={grid_resolution}, default 5th–95th percentile grid. No model fitting is performed.",
        "PDP/ICE vary one original input feature through the entire pipeline. These are model-response summaries, not causal effects; correlated features may produce unrealistic combinations.",
    ]
    for index in columns:
        column = sample.iloc[:, index] if hasattr(sample, "iloc") else sample[:, index]
        from scipy import sparse

        if sparse.issparse(column):
            column = column.toarray()  # bounded single reference column, at most 512 values
        try:
            np.asarray(column, dtype=float)
        except (ValueError, TypeError) as exc:
            raise SeeDLError(
                "PDP/ICE currently support numeric selected input features. Other columns may be categorical inside a pipeline."
            ) from exc
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = compute(
                    read_only_response(model),
                    sample,
                    [index],
                    kind=kind,
                    method="brute",
                    grid_resolution=grid_resolution,
                )
        except (ValueError, TypeError, NotImplementedError) as exc:
            raise SeeDLError(
                f"scikit-learn partial dependence is unsupported for this estimator/data combination: {exc}"
            ) from exc
        notes.extend(f"WARNING: {w.message}" for w in caught)
        grid = result.get("grid_values", result.get("values"))[0]
        series = []
        if "average" in result:
            if result["average"].shape[0] <= output:
                raise SeeDLError(
                    "Partial-dependence output does not match the selected response; multi-output estimators require a separate single-output model."
                )
            series.append(
                {
                    "name": "Partial dependence (reference mean)",
                    "x": native(grid),
                    "y": native(result["average"][output]),
                }
            )
        if "individual" in result:
            individual = result["individual"][output]
            curves = indices(len(individual), max_ice)
            for row in curves:
                series.append(
                    {
                        "name": f"ICE: input row {int(selected[row])}",
                        "x": native(grid),
                        "y": native(individual[row]),
                    }
                )
            notes.append(
                f"Feature {names[index]}: displaying {len(curves)} of {len(individual)} computed ICE curves; curves use the same response/feature space as the PDP."
            )
        panels[f"pdp_{index}"] = {
            "type": "line",
            "title": f"{kind.title()} response: {names[index]}",
            "x_label": f"Original input: {names[index]}",
            "y_label": ylabel,
            "series": series,
            "description": "Reference sample and response units are disclosed; output is conditional on this fitted model and preprocessing.",
        }
    data = {
        "task": "interpretation",
        "method": "sklearn.partial_dependence (brute)",
        "panels": panels,
        "sample_indices": native(selected),
        "provenance": {
            "sklearn_version": sklearn.__version__,
            "reference_rows": len(selected),
            "original_feature_indices": columns,
        },
    }
    return Evaluation(
        kind="ml_dependence",
        title=f"{type(model).__name__} partial dependence / ICE",
        summary=f"{len(columns)} input features | {len(selected)} reference rows | no fitting",
        data=native(data),
        metadata={
            "Estimator": type(model).__name__,
            "Method": "partial dependence / ICE",
            "Reference samples": len(selected),
            "Feature space": "original input columns",
        },
        notes=notes,
        backend=resolve_backend(backend),
    )
