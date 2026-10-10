"""Bounded neuron graphs for structurally proven feed-forward PyTorch networks.

Only exact built-in Linear/Sequential modules are accepted. Module registration
order in arbitrary user modules is never treated as data flow.
"""

from __future__ import annotations

import math
from typing import Any

from ._inspect import _model_data
from .exceptions import SeeDLError


def sequential_leaves(model: Any, torch: Any) -> list[tuple[str, Any]] | None:
    """Flatten exact Sequential containers, retaining repeated module occurrences."""
    result: list[tuple[str, Any]] = []

    def visit(module: Any, path: str) -> bool:
        if type(module) is torch.nn.Sequential:
            return all(
                visit(child, f"{path}.{name}".strip(".")) for name, child in module._modules.items()
            )
        if list(module.children()):
            return False
        result.append((path or "(root)", module))
        return True

    return result if visit(model, "") else None


def _plan(model: Any, torch: Any) -> tuple[list[tuple[str, Any]], list[dict[str, Any]]] | None:
    leaves = sequential_leaves(model, torch)
    if not leaves or type(model) not in {torch.nn.Sequential, torch.nn.Linear}:
        return None
    # Hooks can replace outputs, invalidating the algebra represented by the graph.
    if any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()):
        return None
    activations = {
        torch.nn.ReLU,
        torch.nn.Tanh,
        torch.nn.Sigmoid,
        torch.nn.GELU,
        torch.nn.SiLU,
        torch.nn.LeakyReLU,
        torch.nn.ELU,
        torch.nn.Identity,
    }
    stages: list[dict[str, Any]] = []
    for name, module in leaves:
        if type(module) is torch.nn.Linear:
            if module.in_features < 1 or module.out_features < 1:
                return None
            if stages and stages[-1]["module"].out_features != module.in_features:
                return None
            if module.weight.device.type == "meta" or module.weight.is_complex():
                return None
            stages.append({"name": name, "module": module, "operations": []})
        elif type(module) in activations and stages:
            stages[-1]["operations"].append(repr(module))
        else:
            return None
    return (leaves, stages) if stages else None


def _number(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _normalize_select_neuron(select_neuron: Any, widths: list[int]) -> str | None:
    if select_neuron is None:
        return None
    if isinstance(select_neuron, (tuple, list)) and len(select_neuron) == 2:
        layer_val, neuron_val = select_neuron
        if isinstance(layer_val, bool) or isinstance(neuron_val, bool):
            raise ValueError(f"select_neuron indices must be integers, got {select_neuron!r}.")
        if not isinstance(layer_val, int) or not isinstance(neuron_val, int):
            raise ValueError(f"select_neuron indices must be integers, got {select_neuron!r}.")
        layer_idx, neuron_idx = layer_val, neuron_val
    elif isinstance(select_neuron, str) and ":" in select_neuron:
        parts = select_neuron.strip().split(":")
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            raise ValueError(
                f"select_neuron string must be formatted as 'layer:neuron', got {select_neuron!r}."
            )
        layer_idx, neuron_idx = int(parts[0]), int(parts[1])
    else:
        raise ValueError(
            f"select_neuron must be (layer_idx, neuron_idx) or 'layer:neuron', got {select_neuron!r}."
        )

    if not 0 <= layer_idx < len(widths):
        raise ValueError(
            f"select_neuron layer index {layer_idx} is out of bounds (model has {len(widths)} layers: 0..{len(widths) - 1})."
        )
    if not 0 <= neuron_idx < widths[layer_idx]:
        raise ValueError(
            f"select_neuron neuron index {neuron_idx} is out of bounds for layer {layer_idx} (width: {widths[layer_idx]})."
        )
    return f"{layer_idx}:{neuron_idx}"


def inspect_visual_model(
    model: Any,
    example_input: Any,
    *,
    view: str,
    theme: str,
    color_by: str,
    sample: int,
    max_neurons: int,
    max_connections: int,
    min_weight: float,
    edge_opacity: float,
    detail: str,
    select_neuron: tuple[int, int] | str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    import torch

    planned = _plan(model, torch) if view not in {"summary", "architecture"} else None
    graph: dict[str, Any] | None = None
    reason = None
    if planned:
        leaves, stages = planned
        widths = [stages[0]["module"].in_features] + [s["module"].out_features for s in stages]
        if sum(widths) > max_neurons or max(widths) > 48 or len(widths) > 12:
            reason = (
                f"Neuron detail collapsed: {sum(widths):,} neurons; limit {max_neurons}, "
                "48 per layer and 12 layers. Showing architecture blocks."
            )
        else:
            graph = _build_graph(stages, widths, max_connections, min_weight)
            graph.update(
                theme=theme,
                color_by=color_by,
                sample=sample,
                dimensions=3 if view == "3d" else 2,
                edge_opacity=edge_opacity,
                selected_neuron=_normalize_select_neuron(select_neuron, widths)
                if select_neuron is not None
                else None,
            )
    elif view in {"neurons", "3d"}:
        reason = "Neuron wiring unavailable: requires exact Linear/Sequential layers with supported activations and no custom hooks. Showing architecture."

    handles = []
    cursor = 0
    stage_index = 0

    def capture(_module: Any, args: Any, kwargs: Any, output: Any) -> None:
        nonlocal cursor, stage_index
        assert graph is not None and planned is not None
        expected = planned[0][cursor][1]
        if expected is not _module:
            raise SeeDLError(
                "Observed execution does not match the supported Sequential structure."
            )
        if type(_module) is torch.nn.Linear:
            if stage_index == 0:
                _assign_values(graph, 0, args[0] if args else kwargs["input"], sample, "activation")
            stage_index += 1
            _assign_values(graph, stage_index, output, sample, "preactivation")
        _assign_values(graph, stage_index, output, sample, "activation")
        cursor += 1

    try:
        if graph is not None and example_input is not None:
            assert planned is not None
            for module in dict.fromkeys(m for _, m in planned[0]):
                handles.append(module.register_forward_hook(capture, with_kwargs=True))
        data, summary, notes = _model_data(model, example_input)
    finally:
        for handle in handles:
            handle.remove()

    data.update(summary=summary, theme=theme, presentation="summary")
    if detail == "leaves":
        names = {name for name, module in model.named_modules() if not list(module.children())}
        data["layers"] = [row for row in data["layers"] if row["name"] in names]
    if graph is not None:
        data.update(neuron_graph=graph, presentation="neurons")
        count = len(graph["edges"])
        graph["caption"] = (
            f"{len(graph['nodes'])} neurons · {count:,} / {graph['total_connections']:,} connections"
            + (f" · sample {sample}" if example_input is not None else " · weights only")
        )
        notes = [n for n in notes if "hierarchy" not in n]
        notes += [
            "Edges use weight[destination, source]. Teal is positive; coral is negative; gray is zero; width encodes magnitude in four bins.",
            "Node activation is the captured post-activation output; bias and pre-activation appear on hover. No backward pass is run.",
        ]
        if graph["sampled"]:
            notes.append(
                f"Connections are deterministically sampled before magnitude filtering: showing {count:,} of {graph['total_connections']:,}."
            )
        if min_weight:
            notes.append(f"Connections with |weight| < {min_weight:g} are hidden after sampling.")
        if graph["nonfinite_weights"]:
            notes.append(
                "Non-finite sampled weights are omitted; they are not represented as zero."
            )
        if example_input is not None:
            notes.append(
                f"Activations show sample {sample} (zero-based), from one eval/no_grad pass on a cloned input."
            )
        else:
            notes.append("No activations captured. Pass input=... to observe a forward pass.")
        if color_by == "gradient":
            notes.append(
                "Node color is the existing bias gradient; absent gradients are gray. Edge color remains weight sign."
            )
    elif view != "summary" and (
        reason or _has_architecture(model, torch) or view == "architecture"
    ):
        notes = [n for n in notes if "hierarchy" not in n]
        data["presentation"] = "architecture"
        data["architecture"] = architecture_payload(model, data["layers"], torch)
        notes.append(data["architecture"]["caption"])
        if view == "3d":
            notes.append(
                "3D neuron view is unavailable for this model; this is a 2D architecture fallback."
            )
    if reason:
        notes.append(reason)
    data["visual_notes"] = [
        n
        for n in notes
        if any(
            term in n for term in ("collapsed", "sampled", "hidden", "unavailable", "Non-finite")
        )
    ]
    return data, {"Summary": summary}, notes


def _assign_values(graph: dict[str, Any], layer: int, tensor: Any, sample: int, key: str) -> None:
    width = graph["widths"][layer]
    if tensor.ndim not in {1, 2} or tensor.shape[-1] != width:
        raise SeeDLError(
            "Neuron activation capture supports [features] or [batch, features] tensors only."
        )
    count = 1 if tensor.ndim == 1 else tensor.shape[0]
    if sample >= count:
        raise SeeDLError(f"sample={sample} is outside the batch of {count} sample(s).")
    values = (tensor if tensor.ndim == 1 else tensor[sample]).detach().cpu().tolist()
    for node, value in zip(graph["layers"][layer]["nodes"], values, strict=True):
        node[key] = _number(value)
        node[key + "_status"] = "available" if node[key] is not None else "non-finite"
    graph["captured"] = True


def _build_graph(
    stages: list[dict[str, Any]], widths: list[int], limit: int, threshold: float
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    layers = []
    parameter_ids: dict[int, str] = {}
    for index, width in enumerate(widths):
        stage = stages[index - 1] if index else None
        module = stage["module"] if stage else None
        biases = (
            module.bias.detach().cpu().tolist()
            if module is not None and module.bias is not None
            else [None] * width
        )
        gradients = (
            module.bias.grad.detach().cpu().tolist()
            if module is not None and module.bias is not None and module.bias.grad is not None
            else [None] * width
        )
        label = (
            "Input" if index == 0 else "Output" if index == len(widths) - 1 else f"Hidden {index}"
        )
        columns = max(1, math.ceil(math.sqrt(width / 2)))
        rows = math.ceil(width / columns)
        current = []
        for j in range(width):
            node = dict(
                id=f"{index}:{j}",
                layer=index,
                index=j,
                x=index * 3.5,
                y=(width - 1) / 2 - j,
                z=(rows - 1) / 2 - j // columns,
                y3=(columns - 1) / 2 - j % columns,
                z3=(rows - 1) / 2 - j // columns,
                bias=_number(biases[j]) if biases[j] is not None else None,
                gradient=_number(gradients[j]) if gradients[j] is not None else None,
                gradient_status="missing"
                if gradients[j] is None
                else "available"
                if _number(gradients[j]) is not None
                else "non-finite",
                activation=None,
                activation_status="not captured",
                preactivation=None,
                trainable=bool(module.weight.requires_grad) if module is not None else None,
                bias_trainable=bool(module.bias.requires_grad)
                if module is not None and module.bias is not None
                else None,
                incoming_edges=[],
                outgoing_edges=[],
            )
            current.append(node)
            nodes.append(node)
        layers.append(
            dict(
                label=label,
                width=width,
                nodes=current,
                module=stage["name"] if stage else "input",
                operation=" → ".join(stage["operations"])
                if stage and stage["operations"]
                else "Linear"
                if stage
                else "Features",
            )
        )
    total = sum(a * b for a, b in zip(widths[:-1], widths[1:], strict=True))
    stride = max(1, math.ceil(total / limit))
    edges = []
    offset = 0
    invalid = 0
    for index, stage in enumerate(stages):
        module = stage["module"]
        weight = module.weight.detach()
        identity = parameter_ids.setdefault(id(module.weight), stage["name"] + ".weight")
        size = module.in_features * module.out_features
        indices = range((-offset) % stride, size, stride)
        # Gather only the bounded preview, even for large original parameter tensors.
        import torch

        flat = torch.tensor(list(indices), device=weight.device, dtype=torch.long)
        dest, source = flat // module.in_features, flat % module.in_features
        values = weight[dest, source].cpu().tolist()
        grad = module.weight.grad
        gradients = (
            grad.detach()[dest, source].cpu().tolist()
            if grad is not None and grad.layout == torch.strided
            else [None] * len(values)
        )
        for k, value, gradient in zip(indices, values, gradients, strict=True):
            if not math.isfinite(value):
                invalid += 1
                continue
            if abs(value) < threshold:
                continue
            edges.append(
                dict(
                    source=f"{index}:{k % module.in_features}",
                    target=f"{index + 1}:{k // module.in_features}",
                    layer=index + 1,
                    weight=value,
                    gradient=_number(gradient) if gradient is not None else None,
                    gradient_status="missing"
                    if gradient is None
                    else "available"
                    if math.isfinite(gradient)
                    else "non-finite",
                    parameter=identity,
                    trainable=bool(module.weight.requires_grad),
                )
            )
        offset += size

    node_map = {n["id"]: n for n in nodes}
    for edge in edges:
        if edge["target"] in node_map:
            node_map[edge["target"]]["incoming_edges"].append(edge)
        if edge["source"] in node_map:
            node_map[edge["source"]]["outgoing_edges"].append(edge)

    return dict(
        nodes=nodes,
        node_map=node_map,
        layers=layers,
        edges=edges,
        widths=widths,
        total_connections=total,
        sampled=stride > 1,
        nonfinite_weights=invalid,
        captured=False,
    )


def _has_architecture(model: Any, torch: Any) -> bool:
    return any(
        isinstance(
            m,
            (
                torch.nn.Linear,
                torch.nn.modules.conv._ConvNd,
                torch.nn.MultiheadAttention,
                torch.nn.TransformerEncoderLayer,
                torch.nn.TransformerDecoderLayer,
            ),
        )
        for m in model.modules()
    )


def architecture_payload(model: Any, rows: list[dict[str, Any]], torch: Any) -> dict[str, Any]:
    leaves = sequential_leaves(model, torch) if type(model) is torch.nn.Sequential else None
    row_map = {r["name"] or "(root)": r for r in rows}
    modules = leaves if leaves else [(name or "(root)", m) for name, m in model.named_modules()]
    blocks = []
    for name, module in modules[:40]:
        row = row_map.get(name, {})
        specs = []
        for attr in (
            "in_features",
            "out_features",
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "groups",
            "embed_dim",
            "num_heads",
        ):
            if hasattr(module, attr):
                specs.append(f"{attr}={getattr(module, attr)}")
        blocks.append(
            dict(
                name=name,
                type=type(module).__name__,
                depth=name.count(".") + (name != "(root)"),
                specs=", ".join(specs),
                shape=row.get("output_shape", "—"),
                parameters=row.get("parameters"),
            )
        )
    edges = []
    if leaves:
        edges = [(i, i + 1) for i in range(len(blocks) - 1)]
    else:
        names = {b["name"]: i for i, b in enumerate(blocks)}
        for i, block in enumerate(blocks):
            parent = block["name"].rsplit(".", 1)[0] if "." in block["name"] else "(root)"
            if block["name"] != "(root)" and parent in names:
                edges.append((names[parent], i))
    caption = (
        "Sequential data flow · blocks represent operations, not individual neurons."
        if leaves
        else "Module hierarchy · connectors show containment, not execution or residual paths."
    )
    if len(modules) > 40:
        caption += f" Showing first 40 of {len(modules)} modules; full inspection: view='summary'."
    return dict(blocks=blocks, edges=edges, sequential=bool(leaves), caption=caption)
