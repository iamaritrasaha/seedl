"""Existing CV results and explicit bounded refitting of sklearn estimator clones."""

from __future__ import annotations

import warnings
from typing import Any

from ._sklearn_validation import (
    bounded_int,
    dependencies,
    input_data,
    native,
    require_estimator,
    rows_at,
)
from .evaluation import Evaluation
from .exceptions import SeeDLError
from .renderers import resolve_backend


def cv_results(
    results: Any, *, metric: str = "score", max_candidates: int = 50, backend: str = "auto"
) -> Evaluation:
    np, sklearn = dependencies()
    from ._sklearn_metadata import _parameter
    from ._sklearn_metrics import _table

    bounded_int("max_candidates", max_candidates, 100)
    search = results if hasattr(results, "cv_results_") else None
    if search is not None:
        require_estimator(search)
        captured = search.cv_results_
    elif isinstance(results, dict):
        captured = results
    else:
        raise SeeDLError(
            "Supply a fitted search with cv_results_ or an existing cross_validate result dictionary; no search or CV fitting is run automatically."
        )
    key = f"mean_test_{metric}"
    notes = [
        "Existing cross-validation results only; no fit(), predict(), cross-validation or search execution is performed.",
        "scikit-learn names held-out fold scores 'test'; here they are CV validation scores, not a separate untouched final test set. Scorer values retain their sign; negative loss scorers are not positive losses.",
    ]
    if "params" in captured:
        if key not in captured:
            choices = [k.removeprefix("mean_test_") for k in captured if k.startswith("mean_test_")]
            raise SeeDLError(f"CV metric {metric!r} unavailable. Choose one of {choices}.")
        if not 1 <= len(captured["params"]) <= 100000:
            raise SeeDLError(
                "Existing searches must contain 1–100,000 captured candidates for bounded display."
            )
        means = np.asarray(captured[key], dtype=float)
        std = np.asarray(
            captured.get(f"std_test_{metric}", np.full(len(means), np.nan)), dtype=float
        )
        if len(means) != len(captured["params"]) or len(std) != len(means):
            raise SeeDLError(
                "CV candidates, means and standard deviations must have aligned lengths."
            )
        rank = np.asarray(captured.get(f"rank_test_{metric}", np.arange(len(means)) + 1))
        order = np.argsort(rank, kind="stable")[:max_candidates]
        panels = {
            "cv_scores": {
                "type": "bar",
                "title": "Existing search CV validation scores",
                "labels": [f"Candidate {int(i)}" for i in order],
                "values": native(means[order]),
                "errors": native(std[order]),
                "x_label": "Original candidate index (rank order)",
                "y_label": f"CV validation {metric} (scorer units)",
                "description": f"Top {len(order)} of {len(means)} candidates. Error values are fold standard deviations, not confidence intervals; hyperparameters are in the candidate table.",
            },
            "cv_candidates": _table(
                "Search candidate details",
                ["Candidate", "Parameters", "Mean CV validation", "Fold std. deviation", "Rank"],
                [
                    [int(i), _parameter(captured["params"][i]), means[i], std[i], rank[i]]
                    for i in order
                ],
            ),
        }
        scores = {
            "candidates": len(means),
            "best_displayed_cv_score": native(means[order[0]]) if len(order) else None,
        }
    else:
        key = f"test_{metric}"
        if key not in captured:
            choices = [k.removeprefix("test_") for k in captured if k.startswith("test_")]
            raise SeeDLError(f"Fold metric {metric!r} unavailable. Choose one of {choices}.")
        values = np.asarray(captured[key], dtype=float)
        if values.ndim != 1 or not 1 <= len(values) <= 100:
            raise SeeDLError("Existing fold score arrays must contain 1–100 scalar scores.")
        rows = [[i + 1, v] for i, v in enumerate(values)]
        panels = {
            "cv_scores": {
                "type": "bar",
                "title": "Existing cross-validation fold scores",
                "labels": [f"Fold {i + 1}" for i in range(len(values))],
                "values": native(values),
                "x_label": "Cross-validation fold",
                "y_label": f"CV validation {metric} (scorer units)",
                "description": "Captured held-out fold scores; no independent final test set is implied.",
            },
            "cv_folds": _table("Captured fold values", ["Fold", "CV validation score"], rows),
        }
        scores = {
            "folds": len(values),
            "mean_cv_score": native(np.mean(values)),
            "std_cv_score": native(np.std(values)),
        }
        notes.append(
            "Displayed fold mean/std are descriptive aggregates (population std, ddof=0), not generalization confidence intervals."
        )
        means = values
    if not np.isfinite(means).all():
        notes.append(
            "WARNING: non-finite captured scores (for example failed folds) remain unavailable; they are not replaced with zero or excluded from means."
        )
    data = {
        "task": "model_selection",
        "metrics": scores,
        "panels": panels,
        "provenance": {
            "sklearn_version": sklearn.__version__,
            "source": "existing cv_results_"
            if search is not None
            else "existing cross_validate dictionary",
            "metric": metric,
            "refitted": False,
        },
    }
    return Evaluation(
        kind="ml_model_selection",
        title="Existing cross-validation results",
        summary="Captured model-selection scores | no fitting | CV validation, not final test",
        data=native(data),
        metadata={"Metric": metric, "Source": "existing results", **native(scores)},
        notes=notes,
        backend=resolve_backend(backend),
    )


def _curve(
    model: Any,
    X: Any,
    y: Any,
    *,
    mode: str,
    allow_refit: bool,
    cv: int,
    scoring: Any,
    max_samples: int,
    random_state: int,
    backend: str,
    train_sizes: Any = None,
    param_name: str | None = None,
    param_range: Any = None,
) -> Evaluation:
    np, sklearn = dependencies()
    from sklearn.model_selection import learning_curve, validation_curve

    from ._sklearn_metrics import _table

    if allow_refit is not True:
        raise SeeDLError(
            "Learning/validation curves refit estimator clones. Explicitly set allow_refit=True and supply training data; no fitting has been performed."
        )
    require_estimator(model, fitted=False)
    from sklearn.base import is_classifier, is_regressor

    if not (is_classifier(model) or is_regressor(model)):
        raise SeeDLError("Learning/validation curves require a classifier or regressor.")
    if hasattr(model, "param_grid") or hasattr(model, "param_distributions"):
        raise SeeDLError(
            "Nested search refitting is not bounded here. Pass an explicitly selected search.best_estimator_ instead."
        )
    bounded_int("cv", cv, 5)
    if cv < 2:
        raise ValueError("cv must be between 2 and 5.")
    bounded_int("max_samples", max_samples, 5000)
    X, y, n, p = input_data(model, X, y, require_y=True)
    take = np.sort(
        np.random.RandomState(random_state).choice(n, min(n, max_samples), replace=False)
    )
    X_sample, y_sample = rows_at(X, take), y[take]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if mode == "learning":
            sizes = np.asarray(train_sizes if train_sizes is not None else (0.25, 0.5, 0.75, 1.0))
            if sizes.ndim != 1 or not 1 <= len(sizes) <= 6:
                raise SeeDLError("Choose 1–6 train_sizes to bound clone fits.")
            x, train, valid = learning_curve(
                model,
                X_sample,
                y_sample,
                train_sizes=sizes,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                shuffle=True,
                random_state=random_state,
                error_score=np.nan,
            )
            xlabel = "Training samples per CV fold"
            labels = native(x)
        else:
            values = list(param_range) if param_range is not None else []
            if not param_name or param_name not in model.get_params():
                raise SeeDLError("param_name must name an existing estimator/pipeline parameter.")
            if not 1 <= len(values) <= 6:
                raise SeeDLError("Choose 1–6 param_range candidates to bound clone fits.")
            train, valid = validation_curve(
                model,
                X_sample,
                y_sample,
                param_name=param_name,
                param_range=values,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                error_score=np.nan,
            )
            numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
            x = np.asarray(values) if numeric else np.arange(len(values))
            xlabel = param_name if numeric else f"{param_name} candidate index (see table)"
            labels = values
    train_mean, valid_mean = np.mean(train, axis=1), np.mean(valid, axis=1)
    train_std, valid_std = np.std(train, axis=1), np.std(valid, axis=1)
    panels = {
        "curve": {
            "type": "line",
            "title": "Learning curve" if mode == "learning" else "Validation curve",
            "x_label": xlabel,
            "y_label": f"{scoring or 'estimator.score'} (scorer units)",
            "series": [
                {"name": "CV training mean", "x": native(x), "y": native(train_mean)},
                {"name": "CV validation mean", "x": native(x), "y": native(valid_mean)},
            ],
            "description": f"Explicitly refitted clones, {cv} folds; training/validation means only. Score signs are preserved. Fold dispersion is in the table, not a confidence band.",
        },
        "curve_details": _table(
            "Cross-validation curve details",
            [
                "Size / parameter",
                "Training mean",
                "Training fold std",
                "Validation mean",
                "Validation fold std",
            ],
            [
                [v, a, b, c, d]
                for v, a, b, c, d in zip(
                    labels, train_mean, train_std, valid_mean, valid_std, strict=True
                )
            ],
        ),
    }
    notes = [
        f"Explicit opt-in fitting: scikit-learn refits estimator clones for {cv} folds x {len(x)} settings (at most {cv * len(x)} outer clone fit calls; estimator internals may fit submodels). The original estimator is not fit or modified.",
        f"Reference data: {len(take)} of {n} supplied rows, sampled without replacement with random_state={random_state}. Use training data only; curves are model-selection evidence, not final-test evaluation.",
        "Aggregates are fold means and population standard deviations (ddof=0). Failed folds remain unavailable and are not silently dropped.",
    ]
    notes.extend(f"WARNING: {w.message}" for w in caught)
    if not np.isfinite(train).all() or not np.isfinite(valid).all():
        notes.append(
            "WARNING: failed/undefined fold scores are unavailable. Do not interpret these curves without investigating the failed folds."
        )
    data = {
        "task": "model_selection",
        "panels": panels,
        "train_scores": native(train),
        "validation_scores": native(valid),
        "sample_indices": native(take),
        "provenance": {
            "sklearn_version": sklearn.__version__,
            "refitted_clones": True,
            "cv": cv,
            "max_clone_fits": cv * len(x),
            "scoring": str(scoring or "estimator.score"),
        },
    }
    return Evaluation(
        kind="ml_model_selection",
        title=f"Explicit {mode} curve",
        summary=f"{cv} CV folds | {len(x)} settings | estimator clones refit by explicit request",
        data=native(data),
        metadata={
            "Estimator": type(model).__name__,
            "Refitted": "clones only (explicit allow_refit=True)",
            "CV folds": cv,
            "Reference samples": len(take),
        },
        notes=notes,
        backend=resolve_backend(backend),
    )


def learning_curve(
    model: Any,
    X: Any,
    y: Any,
    *,
    allow_refit: bool = False,
    cv: int = 3,
    train_sizes: Any = None,
    scoring: Any = None,
    max_samples: int = 2000,
    random_state: int = 0,
    backend: str = "auto",
) -> Evaluation:
    return _curve(
        model,
        X,
        y,
        mode="learning",
        allow_refit=allow_refit,
        cv=cv,
        train_sizes=train_sizes,
        scoring=scoring,
        max_samples=max_samples,
        random_state=random_state,
        backend=backend,
    )


def validation_curve(
    model: Any,
    X: Any,
    y: Any,
    *,
    param_name: str,
    param_range: Any,
    allow_refit: bool = False,
    cv: int = 3,
    scoring: Any = None,
    max_samples: int = 2000,
    random_state: int = 0,
    backend: str = "auto",
) -> Evaluation:
    return _curve(
        model,
        X,
        y,
        mode="validation",
        allow_refit=allow_refit,
        cv=cv,
        param_name=param_name,
        param_range=param_range,
        scoring=scoring,
        max_samples=max_samples,
        random_state=random_state,
        backend=backend,
    )
