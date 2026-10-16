"""Feature attribution and model interpretability with strict state preservation."""

from __future__ import annotations

import importlib
import math
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
    samples: int = 25,
    stdev_spread: float = 0.15,
    seed: int | None = None,
) -> tuple[dict[str, Any], str, list[str]]:
    """Compute feature attribution using saliency, integrated gradients, Grad-CAM, or SmoothGrad.

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

    valid_methods = {"saliency", "integrated_gradients", "grad_cam", "smoothgrad"}
    if method not in valid_methods:
        raise SeeDLError(
            f"Unsupported explain method {method!r}. Choose from: {', '.join(sorted(valid_methods))}."
        )

    if not isinstance(steps, int) or steps < 1:
        raise SeeDLError("steps must be a positive integer.")
    if not isinstance(samples, int) or samples < 1:
        raise SeeDLError("samples must be a positive integer.")
    if not isinstance(stdev_spread, (int, float)) or stdev_spread <= 0:
        raise SeeDLError("stdev_spread must be a positive number.")

    # Save original training modes and parameter gradients
    states = [(m, bool(m.training)) for _, m in model.named_modules()]
    saved_grads = {
        name: param.grad.clone() if param.grad is not None else None
        for name, param in model.named_parameters()
    }

    diagnostics: dict[str, Any] = {}

    try:
        model.eval()
        # Zero out existing gradients temporarily to avoid interference
        model.zero_grad(set_to_none=True)

        if method == "saliency":
            attribution, target_class, pred_score = _compute_saliency(
                model, example_input, target, torch, abs_mode=abs
            )
            abs_mode = "Absolute Magnitude" if (abs is True or abs is None) else "Signed"
            method_desc = f"Vanilla Saliency ({abs_mode})"
            limitation = (
                "Saliency measures instantaneous gradient sensitivity at the input; "
                "it does not account for non-linear saturation and does not prove causality."
            )
        elif method == "integrated_gradients":
            attribution, target_class, pred_score, diagnostics = _compute_integrated_gradients(
                model, example_input, target, steps, torch, baseline=baseline, abs_mode=abs
            )
            method_desc = f"Integrated Gradients ({steps} steps via composite midpoint rule)"
            limitation = (
                "Integrated Gradients approximates the path integral via the composite midpoint "
                "integration rule (open Newton-Cotes formula with uniform step weights). Attributions "
                "are signed by default to satisfy the Completeness axiom."
            )
        elif method == "smoothgrad":
            attribution, target_class, pred_score, diagnostics = _compute_smoothgrad(
                model,
                example_input,
                target,
                samples,
                float(stdev_spread),
                torch,
                seed=seed,
                abs_mode=abs,
            )
            abs_mode = "Absolute Magnitude" if (abs is True or abs is None) else "Signed"
            method_desc = f"SmoothGrad ({samples} noise samples, σ={stdev_spread:.2f}, {abs_mode})"
            limitation = (
                "SmoothGrad averages gradients over noisy perturbations to reduce visual noise; "
                "it remains an empirical sensitivity heuristic and does not guarantee causality."
            )
        else:  # grad_cam
            attribution, target_class, pred_score, diagnostics = _compute_grad_cam(
                model, example_input, target, torch, layer=layer
            )
            layer_name = diagnostics.get("target_layer", "last_conv")
            method_desc = f"Grad-CAM (Layer: {layer_name})"
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
    if "completeness_error" in diagnostics:
        err = diagnostics["completeness_error"]
        notes.append(f"Completeness approximation error: {err:.6g} (IG sum vs ΔF).")

    data = {
        "method": method,
        "target_class": target_class,
        "predicted_score": pred_score,
        "attribution": attr_list,
        "input_shape": tuple(int(s) for s in example_input.shape),
        "diagnostics": diagnostics,
    }

    return data, summary, notes


def _compute_saliency(
    model: Any,
    input_tensor: Any,
    target: int | None,
    torch: Any,
    abs_mode: bool | None = None,
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
    attribution = grads.abs() if (abs_mode is True or abs_mode is None) else grads

    return attribution, target, score_val


def _compute_integrated_gradients(
    model: Any,
    input_tensor: Any,
    target: int | None,
    steps: int,
    torch: Any,
    baseline: Any | None = None,
    abs_mode: bool | None = None,
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

    # First determine target class and score from original input and baseline
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

        base_out = model(baseline_t).reshape(-1)
        base_score_val = float(base_out[target].item())

    accum_grads = torch.zeros_like(cloned_input)
    delta = cloned_input - baseline_t
    convergence_curve: list[float] = []

    for step in range(1, steps + 1):
        alpha = (float(step) - 0.5) / float(steps)
        interpolated = (baseline_t + alpha * delta).detach().clone().requires_grad_(True)
        step_out = model(interpolated).reshape(-1)
        step_score = step_out[target]
        step_grad = torch.autograd.grad(
            step_score, interpolated, create_graph=False, retain_graph=False
        )[0]
        accum_grads += step_grad
        cur_approx = (delta * (accum_grads / float(step))).sum().item()
        convergence_curve.append(float(cur_approx))

    avg_grad = accum_grads / float(steps)
    ig_attribution = delta * avg_grad

    # Completeness verification: sum(IG(x)) ≈ F(x) - F(x_0)
    ig_sum = float(ig_attribution.sum().item())
    expected_diff = score_val - base_score_val
    completeness_err = float(math.fabs(ig_sum - expected_diff))
    denom = max(float(math.fabs(expected_diff)), 1e-7)
    rel_completeness_err = completeness_err / denom

    if abs_mode is True:
        ig_attribution = ig_attribution.abs()

    diagnostics = {
        "completeness_error": completeness_err,
        "relative_completeness_error": rel_completeness_err,
        "ig_sum": ig_sum,
        "target_score": score_val,
        "baseline_score": base_score_val,
        "expected_score_difference": expected_diff,
        "steps": steps,
        "convergence_curve": convergence_curve,
    }

    return ig_attribution, target, score_val, diagnostics


def _compute_smoothgrad(
    model: Any,
    input_tensor: Any,
    target: int | None,
    samples: int,
    stdev_spread: float,
    torch: Any,
    seed: int | None = None,
    abs_mode: bool | None = None,
):
    cloned_input = input_tensor.detach().clone()
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

    # Noise level: sigma = stdev_spread * (max(x) - min(x))
    min_v = float(cloned_input.min().item())
    max_v = float(cloned_input.max().item())
    spread = max_v - min_v if max_v > min_v else 1.0
    sigma = stdev_spread * spread

    gen = None
    if seed is not None:
        gen = torch.Generator(device=cloned_input.device)
        gen.manual_seed(seed)

    accum_grads = torch.zeros_like(cloned_input)
    grad_norms: list[float] = []

    for _ in range(samples):
        if gen is not None:
            noise = (
                torch.randn(
                    cloned_input.shape,
                    generator=gen,
                    device=cloned_input.device,
                    dtype=cloned_input.dtype,
                )
                * sigma
            )
        else:
            noise = torch.randn_like(cloned_input) * sigma

        noisy_input = (cloned_input + noise).detach().clone().requires_grad_(True)
        step_out = model(noisy_input).reshape(-1)
        step_score = step_out[target]
        step_grad = torch.autograd.grad(
            step_score, noisy_input, create_graph=False, retain_graph=False
        )[0]

        if abs_mode is True or abs_mode is None:
            accum_grads += step_grad.abs()
        else:
            accum_grads += step_grad
        grad_norms.append(float(step_grad.norm().item()))

    attribution = accum_grads / float(samples)

    diagnostics = {
        "samples": samples,
        "stdev_spread": stdev_spread,
        "noise_sigma": sigma,
        "seed": seed,
        "mean_sample_gradient_norm": sum(grad_norms) / len(grad_norms) if grad_norms else 0.0,
    }

    return attribution, target, score_val, diagnostics


def _compute_grad_cam(
    model: Any,
    input_tensor: Any,
    target: int | None,
    torch: Any,
    layer: str | None = None,
):
    conv_layer = None
    chosen_layer_name = ""

    if layer is not None:
        for name, module in model.named_modules():
            if name == layer or type(module).__name__ == layer:
                conv_layer = module
                chosen_layer_name = name or type(module).__name__
                break
        if conv_layer is None:
            raise SeeDLError(f"Target layer {layer!r} not found in model.")
    else:
        # Find last convolutional layer
        for name, module in reversed(list(model.named_modules())):
            if "conv" in type(module).__name__.lower():
                conv_layer = module
                chosen_layer_name = name or type(module).__name__
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

        diagnostics = {
            "target_layer": chosen_layer_name,
            "feature_map_shape": list(feature_maps.shape),
            "gradients_shape": list(gradients.shape),
            "normalization": "min-max [0, 1] with ReLU rectification",
        }

        return result_cam, target, score_val, diagnostics
    finally:
        h1.remove()
        h2.remove()
