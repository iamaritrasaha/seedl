"""Model prediction inspection and comparison for classification and regression."""

from __future__ import annotations

import math
from typing import Any

from .exceptions import SeeDLError


def inspect_predictions(
    predictions: Any,
    *,
    targets: Any | None = None,
    labels: list[str] | None = None,
    task: str = "auto",
    max_points: int = 2_048,
) -> tuple[str, dict[str, Any], str, list[str]]:
    """Inspect model predictions, probabilities, residuals, or ranking."""
    preds_list = _to_list(predictions)
    if preds_list is None or preds_list == []:
        raise SeeDLError("Cannot inspect empty predictions.")
    if isinstance(preds_list, (int, float)):
        preds_list = [float(preds_list)]

    targets_list = _to_list(targets) if targets is not None else None
    if targets_list is not None and isinstance(targets_list, (int, float)):
        targets_list = [float(targets_list)]

    # Determine dimensionality: 1D (single sample or 1D regression) or 2D (batch or multiclass)
    is_2d = isinstance(preds_list, list) and len(preds_list) > 0 and isinstance(preds_list[0], list)

    # Task detection: classification vs regression
    if task == "auto":
        if is_2d and len(preds_list[0]) > 1:
            task = "classification"
        elif (
            not is_2d
            and len(preds_list) > 1
            and all(isinstance(v, (int, float)) for v in preds_list)
        ):
            # Check if values look like classification probabilities or regression targets
            is_prob = (
                all(0.0 <= float(v) <= 1.0 for v in preds_list)
                and abs(sum(float(v) for v in preds_list) - 1.0) < 0.05
            )
            task = "classification" if is_prob else "regression"
        else:
            task = "regression"

    if task == "classification":
        return _inspect_classification(preds_list, labels=labels, targets=targets_list)
    else:
        return _inspect_regression(preds_list, targets=targets_list)


def compare_predictions(
    preds_a: Any,
    preds_b: Any,
    *,
    labels: list[str] | None = None,
    names: tuple[str, str] = ("Model A", "Model B"),
) -> tuple[str, dict[str, Any], str, list[str]]:
    """Compare predictions from two models on the same sample or batch."""
    p_a = _to_list(preds_a)
    p_b = _to_list(preds_b)

    if p_a is None or p_a == [] or p_b is None or p_b == []:
        raise SeeDLError("Cannot compare empty predictions.")

    # Assume single sample multiclass for comparison if 1D or single row
    row_a = p_a[0] if (isinstance(p_a, list) and p_a and isinstance(p_a[0], list)) else p_a
    row_b = p_b[0] if (isinstance(p_b, list) and p_b and isinstance(p_b[0], list)) else p_b

    if isinstance(row_a, (int, float)):
        row_a = [row_a]
    if isinstance(row_b, (int, float)):
        row_b = [row_b]

    probs_a, is_logits_a = _softmax_if_needed([float(v) for v in row_a])
    probs_b, is_logits_b = _softmax_if_needed([float(v) for v in row_b])

    num_classes = max(len(probs_a), len(probs_b))
    if num_classes == 0:
        raise SeeDLError("Cannot compare empty predictions.")
    class_names = [
        labels[i] if labels and i < len(labels) else f"Class {i}" for i in range(num_classes)
    ]

    top_a = max(range(len(probs_a)), key=lambda i: probs_a[i]) if probs_a else 0
    top_b = max(range(len(probs_b)), key=lambda i: probs_b[i]) if probs_b else 0

    agree = (top_a == top_b) and (len(probs_a) > 0 and len(probs_b) > 0)

    rows = []
    for i in range(num_classes):
        val_a = probs_a[i] if i < len(probs_a) else 0.0
        val_b = probs_b[i] if i < len(probs_b) else 0.0
        delta = val_b - val_a
        rows.append(
            {
                "class": class_names[i],
                "score_a": val_a,
                "score_b": val_b,
                "delta": delta,
            }
        )

    # Sort by absolute delta
    rows.sort(key=lambda r: abs(float(str(r["delta"]))), reverse=True)

    name_a, name_b = names
    summary = (
        f"{name_a} and {name_b} {'agree' if agree else 'disagree'} on top class: "
        f"'{class_names[top_a]}' ({probs_a[top_a]:.1%}) vs '{class_names[top_b]}' ({probs_b[top_b]:.1%})"
    )

    notes = [
        f"Comparing predictions from {name_a} and {name_b}.",
        f"Agreement on top prediction: {'Yes' if agree else 'No'}.",
    ]
    if is_logits_a or is_logits_b:
        notes.append(
            "Raw inputs were converted to normalized probabilities via softmax for comparison."
        )

    data = {
        "classes": class_names,
        "rows": rows,
        "name_a": name_a,
        "name_b": name_b,
        "top_a": class_names[top_a],
        "top_b": class_names[top_b],
        "top_score_a": probs_a[top_a],
        "top_score_b": probs_b[top_b],
        "agreement": agree,
    }

    return "prediction_comparison", data, summary, notes


def _inspect_classification(
    preds_list: Any,
    labels: list[str] | None = None,
    targets: list[Any] | None = None,
) -> tuple[str, dict[str, Any], str, list[str]]:
    if not preds_list:
        raise SeeDLError("Cannot inspect empty predictions.")
    # If 2D (batch), take first sample for detailed top-k, and summarize batch
    is_batch = isinstance(preds_list, list) and preds_list and isinstance(preds_list[0], list)
    single_sample = preds_list[0] if is_batch else preds_list

    if not single_sample:
        raise SeeDLError("Cannot inspect empty predictions sample.")

    raw_scores = [float(v) for v in single_sample]
    if len(raw_scores) == 1 and 0.0 <= raw_scores[0] <= 1.0:
        # Single probability represents binary classification
        p = raw_scores[0]
        probs = [1.0 - p, p]
        was_logits = False
        num_classes = 2
        class_names = labels[:2] if (labels and len(labels) >= 2) else ["Class 0", "Class 1"]
        raw_scores = probs
    else:
        probs, was_logits = _softmax_if_needed(raw_scores)
        num_classes = len(probs)
        class_names = [
            labels[i] if labels and i < len(labels) else f"Class {i}" for i in range(num_classes)
        ]

    # Rank classes
    ranked_indices = sorted(range(num_classes), key=lambda i: probs[i], reverse=True)
    top_k = min(5, num_classes)
    ranked = [
        {
            "rank": r + 1,
            "class": class_names[idx],
            "index": idx,
            "probability": probs[idx],
            "raw_score": raw_scores[idx],
        }
        for r, idx in enumerate(ranked_indices[:top_k])
    ]

    top_class = ranked[0]["class"]
    top_prob = ranked[0]["probability"]

    summary = f"Top prediction: '{top_class}' with {top_prob:.1%} confidence"
    notes = []
    if was_logits:
        notes.append(
            "Raw inputs were identified as raw logits and normalized with softmax. "
            "Scores represent relative model confidence; probability normalization does not guarantee empirical calibration."
        )
    else:
        notes.append("Inputs were already normalized probabilities summing to 1.")

    if is_batch:
        notes.append(f"Visualizing first sample of batch ({len(preds_list)} total samples).")

    data = {
        "ranked": ranked,
        "top_class": top_class,
        "top_prob": top_prob,
        "all_probs": probs,
        "classes": class_names,
        "was_logits": was_logits,
        "is_batch": is_batch,
        "batch_size": len(preds_list) if is_batch else 1,
    }

    return "classification_predictions", data, summary, notes


def _inspect_regression(
    preds_list: list[Any],
    targets: list[Any] | None = None,
) -> tuple[str, dict[str, Any], str, list[str]]:
    if not preds_list:
        raise SeeDLError("Cannot inspect empty predictions.")
    flat_preds = [
        float(v)
        for v in (
            preds_list if not isinstance(preds_list[0], list) else [row[0] for row in preds_list]
        )
    ]
    n = len(flat_preds)
    if n == 0:
        raise SeeDLError("Cannot inspect empty predictions.")

    data: dict[str, Any] = {
        "predictions": flat_preds,
        "count": n,
        "pred_min": min(flat_preds),
        "pred_max": max(flat_preds),
        "pred_mean": sum(flat_preds) / max(n, 1),
    }

    notes = [
        f"Regression output: {n} prediction value(s) ranging from {data['pred_min']:.4g} to {data['pred_max']:.4g}."
    ]

    if targets is not None:
        flat_targets = [
            float(v)
            for v in (targets if not isinstance(targets[0], list) else [row[0] for row in targets])
        ]
        if len(flat_targets) != n:
            raise SeeDLError(
                f"Shape mismatch: {n} predictions but {len(flat_targets)} targets provided."
            )
        residuals = [p - t for p, t in zip(flat_preds, flat_targets, strict=True)]
        mae = sum(abs(r) for r in residuals) / max(len(residuals), 1)
        mse = sum(r * r for r in residuals) / max(len(residuals), 1)
        rmse = math.sqrt(max(mse, 0.0))

        data.update(
            {
                "targets": flat_targets,
                "residuals": residuals,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
            }
        )
        summary = f"Regression: {n} sample(s) · MAE: {mae:.4g} · RMSE: {rmse:.4g}"
        notes.append(f"Residual analysis available: MAE = {mae:.4g}, RMSE = {rmse:.4g}.")
    else:
        summary = f"Regression: {n} prediction value(s)"
        notes.append(
            "Pass targets=... to display residuals, scatter comparison, and error metrics."
        )

    return "regression_predictions", data, summary, notes


def _softmax_if_needed(values: list[float]) -> tuple[list[float], bool]:
    if not values:
        return [], False
    all_between_0_1 = all(0.0 <= v <= 1.0 for v in values)
    sums_to_1 = abs(sum(values) - 1.0) < 0.05
    if all_between_0_1 and sums_to_1:
        return values, False

    # Apply numerically stable softmax
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    sum_exps = sum(exps)
    probs = [e / sum_exps for e in exps]
    return probs, True


def _to_list(val: Any) -> Any:
    if hasattr(val, "detach") and hasattr(val, "cpu") and hasattr(val, "tolist"):
        return val.detach().cpu().tolist()
    if hasattr(val, "tolist"):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return [_to_list(item) if isinstance(item, (list, tuple)) else item for item in val]
    return val
