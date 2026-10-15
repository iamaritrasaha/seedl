"""Model, training history, and prediction comparison utilities."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from ._models import inspect_model_architecture
from ._predictions import compare_predictions
from ._training import compare_histories
from .exceptions import SeeDLError


def _torch():
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError:
        return None


def compare_objects(
    obj_a: Any,
    obj_b: Any,
    *,
    names: tuple[str, str] = ("Model A", "Model B"),
    max_points: int = 2_048,
) -> tuple[str, dict[str, Any], str, list[str]]:
    """Compare two compatible deep learning objects: models, training histories, or predictions."""
    from ._recording import TrainingSnapshot

    if isinstance(obj_a, TrainingSnapshot) and isinstance(obj_b, TrainingSnapshot):
        from ._snapshot_compare import compare_snapshots

        return compare_snapshots(obj_a, obj_b, names)

    from .view import Visualization

    if isinstance(obj_a, Visualization) and isinstance(obj_b, Visualization):
        if obj_a.kind == "feature_maps" and obj_b.kind == "feature_maps":
            from ._features import compare_feature_maps

            data, summary, notes = compare_feature_maps(obj_a, obj_b, names=names)
            return "feature_comparison", data, summary, notes

        if obj_a.kind in {"attention", "educational_attention"} and obj_b.kind in {
            "attention",
            "educational_attention",
        }:
            from ._transformers import compare_attention_heads

            mat_a = obj_a.data.get("matrix", [])
            mat_b = obj_b.data.get("matrix", [])
            q_toks = obj_a.data.get("query_tokens", obj_a.data.get("tokens", []))
            k_toks = obj_a.data.get("key_tokens", obj_a.data.get("tokens", []))
            cmp_res = compare_attention_heads([mat_a, mat_b], 0, 1, q_toks, k_toks, False)
            cmp_res["names"] = names
            summary = (
                f"Attention Comparison: {names[0]} vs {names[1]} · "
                f"MAE: {cmp_res['mae']:.4f} · Cosine Sim: {cmp_res['cosine_similarity']:.4f}"
            )
            notes = [
                f"Comparing attention matrices: {names[0]} vs {names[1]}.",
                f"Frobenius norm of difference: {cmp_res['frobenius_norm']:.4f}.",
                f"Mean absolute difference: {cmp_res['mae']:.4f}.",
            ]
            payload = {
                "presentation": "attention",
                "matrix": cmp_res["diff_matrix"],
                "all_heads": [mat_a, mat_b],
                "query_tokens": q_toks,
                "key_tokens": k_toks,
                "head_comparison": cmp_res,
                "num_heads": 2,
                "captured_heads": 2,
            }
            return "attention_comparison", payload, summary, notes

    # 4D CNN feature map tensors
    ndim_a = getattr(obj_a, "ndim", None)
    ndim_b = getattr(obj_b, "ndim", None)
    if ndim_a == 4 and ndim_b == 4:
        from ._features import compare_feature_maps

        data, summary, notes = compare_feature_maps(obj_a, obj_b, names=names)
        return "feature_comparison", data, summary, notes

    torch = _torch()

    # Model comparison
    is_mod_a = torch is not None and isinstance(obj_a, torch.nn.Module)
    is_mod_b = torch is not None and isinstance(obj_b, torch.nn.Module)

    if is_mod_a and is_mod_b:
        return _compare_models(obj_a, obj_b, names)

    # History comparison
    if isinstance(obj_a, Mapping) and isinstance(obj_b, Mapping):
        hist_data, sum_str, notes = compare_histories(obj_a, obj_b, names, max_points)
        return "history_comparison", hist_data, sum_str, notes

    # Prediction comparison
    is_pred_a = hasattr(obj_a, "tolist") or isinstance(obj_a, (list, tuple))
    is_pred_b = hasattr(obj_b, "tolist") or isinstance(obj_b, (list, tuple))

    if is_pred_a and is_pred_b:
        kind, pred_data, sum_str, notes = compare_predictions(obj_a, obj_b, names=names)
        return kind, pred_data, sum_str, notes

    raise SeeDLError(
        f"Cannot compare objects of type {type(obj_a).__name__} and {type(obj_b).__name__}. "
        "sd.compare() supports pairs of PyTorch nn.Modules, training history dicts, or prediction tensors."
    )


def _compare_models(
    model_a: Any,
    model_b: Any,
    names: tuple[str, str],
) -> tuple[str, dict[str, Any], str, list[str]]:
    name_a, name_b = names
    data_a, _, _ = inspect_model_architecture(model_a)
    data_b, _, _ = inspect_model_architecture(model_b)

    param_a = data_a["parameters"]
    param_b = data_b["parameters"]
    diff_params = param_b - param_a
    diff_sign = "+" if diff_params > 0 else ""

    train_a = data_a["trainable"]
    train_b = data_b["trainable"]

    layers_a = data_a["layers"]
    layers_b = data_b["layers"]

    summary = (
        f"{name_a} ({param_a:,} params) vs {name_b} ({param_b:,} params) · "
        f"Diff: {diff_sign}{diff_params:,} ({diff_params / max(param_a, 1) * 100:+.1f}%)"
    )

    notes = [
        f"Architecture comparison between {name_a} and {name_b}.",
        f"Total parameter difference: {diff_sign}{diff_params:,} parameters.",
        f"Trainable parameters: {train_a:,} vs {train_b:,}.",
        f"Module count: {len(layers_a)} vs {len(layers_b)}.",
    ]

    # Align layers by name
    names_a = {layer["name"]: layer for layer in layers_a}
    names_b = {layer["name"]: layer for layer in layers_b}
    all_names = list(dict.fromkeys(list(names_a) + list(names_b)))

    comparison_rows = []
    for l_name in all_names[:50]:  # Up to 50 rows
        la = names_a.get(l_name)
        lb = names_b.get(l_name)
        p_la = la["parameters"] if la else "—"
        p_lb = lb["parameters"] if lb else "—"
        type_a = la["type"] if la else "—"
        type_b = lb["type"] if lb else "—"
        match = la is not None and lb is not None and la["type"] == lb["type"]

        comparison_rows.append(
            {
                "layer": l_name or "(root)",
                "type_a": type_a,
                "type_b": type_b,
                "params_a": p_la,
                "params_b": p_lb,
                "matched": match,
            }
        )

    data = {
        "name_a": name_a,
        "name_b": name_b,
        "params_a": param_a,
        "params_b": param_b,
        "trainable_a": train_a,
        "trainable_b": train_b,
        "diff_params": diff_params,
        "layer_rows": comparison_rows,
    }

    return "model_comparison", data, summary, notes
