"""Feature attribution and model interpretability with strict state preservation."""

from __future__ import annotations

import importlib
from typing import Any

from .exceptions import MissingDependencyError, SeeDLError


def _torch():
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError:
        return None


def explain_prediction(
    model: Any,
    example_input: Any,
    *,
    target: int | None = None,
    method: str = "saliency",
    steps: int = 20,
) -> tuple[dict[str, Any], str, list[str]]:
    """Compute feature attribution using saliency, integrated gradients, or Grad-CAM.

    State preservation:
    - Model is set to eval mode during computation and original training flag is restored.
    - Temporary hooks are safely removed in a finally block.
    - Existing parameter gradients are saved and restored; no parameters or optimizer states are modified.
    - The input tensor is detached and cloned; original user input is never mutated.
    """
    torch = _torch()
    if torch is None or not isinstance(model, torch.nn.Module):
        raise MissingDependencyError(
            "Explainability tools require PyTorch. Install it with `pip install 'seedl[torch]'`."
        )

    if not isinstance(example_input, torch.Tensor):
        raise SeeDLError("example_input must be a PyTorch Tensor.")

    valid_methods = {"saliency", "integrated_gradients", "grad_cam"}
    if method not in valid_methods:
        raise SeeDLError(
            f"Unsupported explain method {method!r}. Choose from: {', '.join(sorted(valid_methods))}."
        )

    # Save original training modes and parameter gradients
    states = [(m, bool(m.training)) for _, m in model.named_modules()]
    saved_grads = {
        name: param.grad.clone() if param.grad is not None else None
        for name, param in model.named_parameters()
    }

    try:
        model.eval()
        # Zero out existing gradients temporarily to avoid interference
        model.zero_grad(set_to_none=True)

        if method == "saliency":
            attribution, target_class, pred_score = _compute_saliency(
                model, example_input, target, torch
            )
            method_desc = "Vanilla Saliency (Input Gradient Magnitude)"
            limitation = (
                "Saliency measures instantaneous gradient sensitivity at the input; "
                "it does not account for non-linear saturation and does not prove causality."
            )
        elif method == "integrated_gradients":
            attribution, target_class, pred_score = _compute_integrated_gradients(
                model, example_input, target, steps, torch
            )
            method_desc = f"Integrated Gradients ({steps} interpolation steps)"
            limitation = (
                "Integrated Gradients satisfies completeness along a straight-line path from baseline; "
                "results depend on the choice of baseline (default: zeros)."
            )
        else:  # grad_cam
            attribution, target_class, pred_score = _compute_grad_cam(
                model, example_input, target, torch
            )
            method_desc = "Grad-CAM (Class Activation Mapping)"
            limitation = (
                "Grad-CAM visualizes coarse spatial activation weights from the last convolutional layer; "
                "it highlights general regions of interest rather than fine-grained pixel boundaries."
            )

    finally:
        # Restore model training modes
        for m, training in states:
            m.training = training
        # Restore original parameter gradients
        for name, param in model.named_parameters():
            saved = saved_grads.get(name)
            if saved is not None:
                param.grad = saved
            else:
                param.grad = None

    attr_list = attribution.cpu().tolist()
    summary = (
        f"Attribution ({method_desc}) for target class {target_class} (score: {pred_score:.4g})"
    )
    notes = [
        f"Method: {method_desc}.",
        f"Target output class index: {target_class}.",
        limitation,
        "All model states and parameter gradients were cleanly preserved during inspection.",
    ]

    data = {
        "method": method,
        "target_class": target_class,
        "predicted_score": pred_score,
        "attribution": attr_list,
        "input_shape": tuple(int(s) for s in example_input.shape),
    }

    return data, summary, notes


def _compute_saliency(model: Any, input_tensor: Any, target: int | None, torch: Any):
    cloned_input = input_tensor.detach().clone().requires_grad_(True)
    output = model(cloned_input)

    flat_out = output.reshape(-1)
    if target is None:
        target = int(flat_out.argmax().item())

    score = flat_out[target]
    score_val = float(score.item())

    # Compute gradient w.r.t input
    grads = torch.autograd.grad(score, cloned_input, create_graph=False, retain_graph=False)[0]
    attribution = grads.abs()

    return attribution, target, score_val


def _compute_integrated_gradients(
    model: Any, input_tensor: Any, target: int | None, steps: int, torch: Any
):
    cloned_input = input_tensor.detach().clone()
    baseline = torch.zeros_like(cloned_input)

    # First determine target class from original input
    with torch.no_grad():
        orig_out = model(cloned_input).reshape(-1)
        if target is None:
            target = int(orig_out.argmax().item())
        score_val = float(orig_out[target].item())

    # Accumulate gradients along path from baseline to input
    accum_grads = torch.zeros_like(cloned_input)
    delta = cloned_input - baseline

    for step in range(1, steps + 1):
        alpha = float(step) / float(steps)
        interpolated = (baseline + alpha * delta).detach().clone().requires_grad_(True)
        step_out = model(interpolated).reshape(-1)
        step_score = step_out[target]
        step_grad = torch.autograd.grad(
            step_score, interpolated, create_graph=False, retain_graph=False
        )[0]
        accum_grads += step_grad

    avg_grad = accum_grads / float(steps)
    ig_attribution = (delta * avg_grad).abs()

    return ig_attribution, target, score_val


def _compute_grad_cam(model: Any, input_tensor: Any, target: int | None, torch: Any):
    # Find last convolutional layer
    conv_layer = None
    for _, module in reversed(list(model.named_modules())):
        if "conv" in type(module).__name__.lower():
            conv_layer = module
            break

    if conv_layer is None:
        raise SeeDLError("Grad-CAM requires at least one convolutional layer in the model.")

    feature_maps = None
    gradients = None

    def fwd_hook(_m, _i, output):
        nonlocal feature_maps
        feature_maps = output

    def bwd_hook(_m, _grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    h1 = conv_layer.register_forward_hook(fwd_hook)
    h2 = conv_layer.register_full_backward_hook(bwd_hook)

    try:
        cloned_input = input_tensor.detach().clone().requires_grad_(True)
        output = model(cloned_input)
        flat_out = output.reshape(-1)
        if target is None:
            target = int(flat_out.argmax().item())
        score = flat_out[target]
        score_val = float(score.item())

        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)

        assert feature_maps is not None and gradients is not None
        # Global average pooling of gradients over spatial dimensions
        weights = gradients.mean(dim=(-2, -1), keepdim=True)
        cam = (weights * feature_maps).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)  # ReLU on weighted sum

        # Normalize to [0, 1]
        c_min, c_max = cam.min(), cam.max()
        if c_max > c_min:
            cam = (cam - c_min) / (c_max - c_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.squeeze(), target, score_val
    finally:
        h1.remove()
        h2.remove()
