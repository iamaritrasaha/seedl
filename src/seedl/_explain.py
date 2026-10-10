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
    baseline: Any | None = None,
    layer: str | None = None,
    abs: bool | None = None,
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

    if not isinstance(steps, int) or steps < 1:
        raise SeeDLError("steps must be a positive integer.")

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
                model, example_input, target, torch, abs=abs
            )
            abs_mode = "Absolute Magnitude" if (abs is True or abs is None) else "Signed"
            method_desc = f"Vanilla Saliency ({abs_mode})"
            limitation = (
                "Saliency measures instantaneous gradient sensitivity at the input; "
                "it does not account for non-linear saturation and does not prove causality."
            )
        elif method == "integrated_gradients":
            attribution, target_class, pred_score = _compute_integrated_gradients(
                model, example_input, target, steps, torch, baseline=baseline, abs=abs
            )
            method_desc = f"Integrated Gradients ({steps} steps via composite midpoint rule)"
            limitation = (
                "Integrated Gradients approximates the path integral via the composite midpoint "
                "integration rule (open Newton-Cotes formula with uniform step weights, distinct from "
                "Gauss-Legendre quadrature which uses non-uniform polynomial roots). Attributions "
                "are signed by default to satisfy the Completeness axiom."
            )
        else:  # grad_cam
            attribution, target_class, pred_score = _compute_grad_cam(
                model, example_input, target, torch, layer=layer
            )
            method_desc = "Grad-CAM (Class Activation Mapping)"
            limitation = (
                "Grad-CAM visualizes coarse spatial activation weights from the target convolutional layer; "
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


def _compute_saliency(
    model: Any,
    input_tensor: Any,
    target: int | None,
    torch: Any,
    abs: bool | None = None,
):
    cloned_input = input_tensor.detach().clone().requires_grad_(True)
    output = model(cloned_input)

    flat_out = output.reshape(-1)
    if flat_out.numel() == 0:
        raise SeeDLError("Model produced empty output.")
    if target is None:
        target = int(flat_out.argmax().item())
    elif target < 0 or target >= flat_out.numel():
        raise SeeDLError(
            f"Target index {target} is out of bounds for model output with {flat_out.numel()} element(s)."
        )

    score = flat_out[target]
    score_val = float(score.item())

    # Compute gradient w.r.t input
    grads = torch.autograd.grad(score, cloned_input, create_graph=False, retain_graph=False)[0]
    attribution = grads.abs() if (abs is True or abs is None) else grads

    return attribution, target, score_val


def _compute_integrated_gradients(
    model: Any,
    input_tensor: Any,
    target: int | None,
    steps: int,
    torch: Any,
    baseline: Any | None = None,
    abs: bool | None = None,
):
    cloned_input = input_tensor.detach().clone()
    if baseline is None:
        baseline_t = torch.zeros_like(cloned_input)
    elif isinstance(baseline, torch.Tensor):
        baseline_t = (
            baseline.detach().clone().to(device=cloned_input.device, dtype=cloned_input.dtype)
        )
    else:
        try:
            baseline_t = torch.tensor(
                baseline, dtype=cloned_input.dtype, device=cloned_input.device
            )
        except Exception as exc:
            raise SeeDLError(f"Invalid baseline: {exc}") from exc

    if baseline_t.shape != cloned_input.shape:
        raise SeeDLError(
            f"Baseline shape {tuple(baseline_t.shape)} must match input shape {tuple(cloned_input.shape)}."
        )

    # First determine target class and score from original input
    with torch.no_grad():
        orig_out = model(cloned_input).reshape(-1)
        if orig_out.numel() == 0:
            raise SeeDLError("Model produced empty output.")
        if target is None:
            target = int(orig_out.argmax().item())
        elif target < 0 or target >= orig_out.numel():
            raise SeeDLError(
                f"Target index {target} is out of bounds for model output with {orig_out.numel()} element(s)."
            )
        score_val = float(orig_out[target].item())

    # Accumulate gradients along path from baseline to input using the uniform composite
    # midpoint integration rule (open Newton-Cotes formula with alpha_k = (k - 0.5) / steps).
    # Note: This is composite midpoint integration, distinct from Gauss-Legendre quadrature
    # which uses non-uniform roots of Legendre polynomials and non-uniform weights.
    accum_grads = torch.zeros_like(cloned_input)
    delta = cloned_input - baseline_t

    for step in range(1, steps + 1):
        alpha = (float(step) - 0.5) / float(steps)
        interpolated = (baseline_t + alpha * delta).detach().clone().requires_grad_(True)
        step_out = model(interpolated).reshape(-1)
        step_score = step_out[target]
        step_grad = torch.autograd.grad(
            step_score, interpolated, create_graph=False, retain_graph=False
        )[0]
        accum_grads += step_grad

    avg_grad = accum_grads / float(steps)
    ig_attribution = delta * avg_grad
    if abs is True:
        ig_attribution = ig_attribution.abs()

    return ig_attribution, target, score_val


def _compute_grad_cam(
    model: Any,
    input_tensor: Any,
    target: int | None,
    torch: Any,
    layer: str | None = None,
):
    conv_layer = None
    if layer is not None:
        for name, module in model.named_modules():
            if name == layer or type(module).__name__ == layer:
                conv_layer = module
                break
        if conv_layer is None:
            raise SeeDLError(f"Target layer {layer!r} not found in model.")
    else:
        # Find last convolutional layer
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
        if flat_out.numel() == 0:
            raise SeeDLError("Model produced empty output.")
        if target is None:
            target = int(flat_out.argmax().item())
        elif target < 0 or target >= flat_out.numel():
            raise SeeDLError(
                f"Target index {target} is out of bounds for model output with {flat_out.numel()} element(s)."
            )
        score = flat_out[target]
        score_val = float(score.item())

        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)

        if feature_maps is None or gradients is None:
            raise SeeDLError(
                "Failed to capture feature maps or gradients for Grad-CAM. "
                "Ensure the target layer is connected to the model output."
            )

        # Global average pooling of gradients over spatial dimensions (dims >= 2)
        spatial_dims = tuple(range(2, gradients.ndim)) if gradients.ndim > 2 else ()
        weights = gradients.mean(dim=spatial_dims, keepdim=True) if spatial_dims else gradients
        cam = (weights * feature_maps).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)  # ReLU on weighted sum

        # Normalize to [0, 1]
        c_min, c_max = cam.min(), cam.max()
        if c_max > c_min:
            cam = (cam - c_min) / (c_max - c_min)
        else:
            cam = torch.zeros_like(cam)

        if cam.ndim >= 2:
            if cam.shape[0] == 1 and cam.shape[1] == 1:
                result_cam = cam[0, 0]
            elif cam.shape[1] == 1:
                result_cam = cam.squeeze(1)
            else:
                result_cam = cam.squeeze()
        else:
            result_cam = cam

        if result_cam.ndim == 0:
            result_cam = result_cam.reshape(1, 1)

        return result_cam, target, score_val
    finally:
        h1.remove()
        h2.remove()
