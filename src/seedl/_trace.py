"""Neural Execution Tracer with rigorous state preservation and exact topology observation."""

from __future__ import annotations

import importlib
import time
from typing import Any

from ._execution_graph import ExecutionTrace, TraceEdge, TraceNode, compute_tensor_stats
from .exceptions import MissingDependencyError


def _torch():
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError:
        return None


def _extract_tensors(obj: Any, torch: Any) -> list[Any]:
    """Recursively extract all PyTorch tensors from nested tuples, lists, and dicts."""
    tensors = []
    if isinstance(obj, torch.Tensor):
        tensors.append(obj)
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            tensors.extend(_extract_tensors(item, torch))
    elif isinstance(obj, dict):
        for val in obj.values():
            tensors.extend(_extract_tensors(val, torch))
    return tensors


def _get_tensor_id(tensor: Any) -> int:
    """Return storage data pointer if available, else object identity."""
    try:
        if hasattr(tensor, "data_ptr") and tensor.data_ptr() != 0:
            return tensor.data_ptr()
    except Exception:
        pass
    return id(tensor)


def _clone_input(obj: Any, torch: Any) -> Any:
    """Recursively clone and detach tensors from nested structures so user input is never mutated."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if isinstance(obj, tuple):
        return tuple(_clone_input(item, torch) for item in obj)
    if isinstance(obj, list):
        return [_clone_input(item, torch) for item in obj]
    if isinstance(obj, dict):
        return {k: _clone_input(v, torch) for k, v in obj.items()}
    return obj


def trace(
    model: Any,
    input: Any,
    *,
    record: tuple[str, ...] | list[str] = ("shapes", "activations"),
    max_nodes: int = 256,
    use_fx: bool | None = None,
    max_samples: int = 32,
) -> ExecutionTrace:
    """Record an exact, verified neural execution trace of a model forward pass.

    State Preservation Guarantees:
    - Model training modes (eval/train, including mixed submodule modes) are preserved and restored after execution.
    - Registered parameter gradients are untouched and restored.
    - All registered module buffers (including non-persistent and None-valued buffers of arbitrary size) are snapshotted and restored to their original values and registrations.
    - All forward hooks are removed in a guaranteed finally block, even on exceptions.
    - User input tensor is cloned and detached; original user input is guaranteed unmutated.
    - Retained activations are detached from autograd graphs.

    Memory Overhead & Sandboxing Notice:
    - Buffer snapshotting creates detached clones of registered module buffers. For models with very large buffers, this temporarily uses memory proportional to buffer size, which is released upon trace completion.
    - Sandboxing Limitation: Arbitrary custom Python code in forward methods cannot be completely sandboxed. While model buffers, parameters, gradients, modes, and input tensors are fully restored, unmanaged side effects (such as mutations of global Python variables, file I/O, or non-PyTorch external state) cannot be intercepted.
    """
    torch = _torch()
    if torch is None or not isinstance(model, torch.nn.Module):
        raise MissingDependencyError(
            "Neural Execution Tracer requires PyTorch. Install with `pip install 'seedl[torch]'`."
        )

    if not isinstance(max_nodes, int) or max_nodes < 1:
        raise ValueError("max_nodes must be a positive integer.")

    # 1. Save original training mode for every submodule to preserve mixed modes
    orig_training_modes = [(module, bool(module.training)) for module in model.modules()]

    # 2. Save original parameter gradients
    saved_grads = {
        name: param.grad.clone() if param.grad is not None else None
        for name, param in model.named_parameters()
    }

    # 3. Snapshot registered buffers across all modules (including None-valued and non-persistent sets)
    saved_buffers = [
        (
            module,
            dict(module._buffers),
            set(module._non_persistent_buffers_set),
            {
                b_name: b_val.detach().clone() if b_val is not None else None
                for b_name, b_val in module._buffers.items()
            },
        )
        for _, module in model.named_modules()
        if module._buffers
    ]

    hooks: list[Any] = []
    observed_nodes: list[TraceNode] = []
    producer_by_obj_id: dict[int, str] = {}  # id(tensor) -> node_id
    producer_by_data_ptr: dict[int, str] = {}  # tensor.data_ptr() -> node_id
    raw_edges: list[TraceEdge] = []
    module_call_counts: dict[str, int] = {}
    pre_times: dict[str, float] = {}
    _retained_tensors: list[
        Any
    ] = []  # Prevents intermediate tensor deallocation and allocator address reuse

    # Clone user inputs to ensure model forward mutations never affect caller's input tensors
    forward_input = _clone_input(input, torch)
    forward_input_tensors = _extract_tensors(forward_input, torch)
    _retained_tensors.extend(forward_input_tensors)
    input_metadata = []
    for idx, inp in enumerate(forward_input_tensors):
        producer_by_obj_id[id(inp)] = "INPUT"
        try:
            if hasattr(inp, "data_ptr") and inp.data_ptr() != 0:
                producer_by_data_ptr[inp.data_ptr()] = "INPUT"
        except Exception:
            pass
        input_metadata.append(
            {
                "index": idx,
                "shape": list(inp.shape),
                "dtype": str(inp.dtype).replace("torch.", ""),
                "device": str(inp.device),
                "num_elements": inp.numel(),
            }
        )

    execution_counter = 0

    # Build hierarchical name map
    named_modules = dict(model.named_modules())
    leaf_modules = set()
    for name, mod in named_modules.items():
        if len(list(mod.children())) == 0:
            leaf_modules.add(name)

    # Forward pre-hook to measure timing
    def pre_hook(mod: Any, args: Any, mod_name: str):
        pre_times[mod_name] = time.perf_counter()

    # Forward hook to capture observed invocation
    def post_hook(mod: Any, args: Any, output: Any, mod_name: str):
        nonlocal execution_counter
        if len(observed_nodes) >= max_nodes:
            return

        end_time = time.perf_counter()
        start_time = pre_times.pop(mod_name, end_time)
        duration_ms = (end_time - start_time) * 1000.0

        is_leaf = mod_name in leaf_modules
        call_idx = module_call_counts.get(mod_name, 0)
        module_call_counts[mod_name] = call_idx + 1

        node_id = f"{mod_name}#{call_idx}" if mod_name else f"root#{call_idx}"
        exec_order = execution_counter
        execution_counter += 1

        # Extract input shapes and dtypes
        in_tensors = _extract_tensors(args, torch)
        in_shapes = [list(t.shape) for t in in_tensors]
        dtypes = [str(t.dtype).replace("torch.", "") for t in in_tensors]
        devices = [str(t.device) for t in in_tensors]

        # Extract output shapes
        out_tensors = _extract_tensors(output, torch)
        out_shapes = [list(t.shape) for t in out_tensors]
        if not dtypes and out_tensors:
            dtypes = [str(t.dtype).replace("torch.", "") for t in out_tensors]
        if not devices and out_tensors:
            devices = [str(t.device) for t in out_tensors]

        # Retain intermediate tensors until end of forward pass to prevent allocator address reuse
        _retained_tensors.extend(in_tensors)
        _retained_tensors.extend(out_tensors)

        total_elems = sum(t.numel() for t in out_tensors)

        # Parameter counts for this module (direct parameters)
        direct_params = sum(p.numel() for p in mod.parameters(recurse=False))
        trainable_params = sum(p.numel() for p in mod.parameters(recurse=False) if p.requires_grad)

        # Activation stats
        stats: dict[str, Any] = {}
        samples: list[float] = []
        nonfinite_cnt = 0
        sparsity_val = 0.0
        is_sampled = False

        if "activations" in record and out_tensors:
            first_out = out_tensors[0]
            stats, samples, nonfinite_cnt, sparsity_val = compute_tensor_stats(
                first_out, max_samples=max_samples, torch=torch
            )
            is_sampled = first_out.numel() > max_samples

        # Trace data dependencies from input tensors
        for inp_t in in_tensors:
            obj_id = id(inp_t)
            dptr = None
            try:
                if hasattr(inp_t, "data_ptr") and inp_t.data_ptr() != 0:
                    dptr = inp_t.data_ptr()
            except Exception:
                pass

            if obj_id in producer_by_obj_id:
                source_id = producer_by_obj_id[obj_id]
                if source_id != node_id:
                    raw_edges.append(
                        TraceEdge(
                            source=source_id,
                            target=node_id,
                            edge_type="explicit_tensor",
                            tensor_shape=list(inp_t.shape),
                            tensor_dtype=str(inp_t.dtype).replace("torch.", ""),
                            description="Direct verified tensor object dependency",
                            is_uncertain=False,
                        )
                    )
            elif dptr is not None and dptr in producer_by_data_ptr:
                source_id = producer_by_data_ptr[dptr]
                if source_id != node_id:
                    raw_edges.append(
                        TraceEdge(
                            source=source_id,
                            target=node_id,
                            edge_type="storage_alias",
                            tensor_shape=list(inp_t.shape),
                            tensor_dtype=str(inp_t.dtype).replace("torch.", ""),
                            description="Shared storage alias / view of upstream tensor",
                            is_uncertain=False,
                        )
                    )

        # Record output tensors for downstream consumers
        for out_t in out_tensors:
            producer_by_obj_id[id(out_t)] = node_id
            try:
                if hasattr(out_t, "data_ptr") and out_t.data_ptr() != 0:
                    producer_by_data_ptr[out_t.data_ptr()] = node_id
            except Exception:
                pass

        parent_name = mod_name.rsplit(".", 1)[0] if "." in mod_name else ""

        node = TraceNode(
            id=node_id,
            name=mod_name or "model",
            module_type=type(mod).__name__,
            invocation_index=call_idx,
            execution_order=exec_order,
            parent_name=parent_name,
            is_leaf=is_leaf,
            input_shapes=in_shapes,
            output_shapes=out_shapes,
            dtypes=dtypes,
            devices=devices,
            num_elements=total_elems,
            num_parameters=direct_params,
            trainable_parameters=trainable_params,
            activation_stats=stats,
            duration_ms=duration_ms,
            is_reused=call_idx > 0,
            sampled_values=samples,
            is_sampled=is_sampled,
            sample_size=len(samples),
            nonfinite_count=nonfinite_cnt,
            sparsity=sparsity_val,
        )
        observed_nodes.append(node)

    # Register hooks on all named modules
    for name, mod in model.named_modules():
        # Avoid registering on root if model has submodules
        h_pre = mod.register_forward_pre_hook(lambda m, a, nm=name: pre_hook(m, a, nm))
        h_post = mod.register_forward_hook(lambda m, a, o, nm=name: post_hook(m, a, o, nm))
        hooks.extend([h_pre, h_post])

    trace_start = time.perf_counter()
    output_metadata = []
    limitations: list[str] = [
        "Execution timing on GPU reflects asynchronous launch overhead unless CUDA synchronization was explicitly invoked.",
        "Module hooks observe module boundaries; in-place Python tensor operations outside submodules may require torch.fx.",
        "Executing arbitrary custom Python forward methods cannot be completely sandboxed against external side effects (e.g. global variable mutations or disk I/O).",
    ]
    warnings: list[str] = []
    trace_mode = "hooks"
    fx_edges: list[TraceEdge] = []

    try:
        model.eval()
        with torch.no_grad():
            # Format inputs appropriately
            if isinstance(forward_input, tuple):
                res = model(*forward_input)
            elif isinstance(forward_input, dict):
                res = model(**forward_input)
            else:
                res = model(forward_input)

        out_tensors = _extract_tensors(res, torch)
        _retained_tensors.extend(out_tensors)
        for idx, ot in enumerate(out_tensors):
            output_metadata.append(
                {
                    "index": idx,
                    "shape": list(ot.shape),
                    "dtype": str(ot.dtype).replace("torch.", ""),
                    "device": str(ot.device),
                    "num_elements": ot.numel(),
                }
            )
            # Mark final output connections
            obj_id = id(ot)
            dptr = None
            try:
                if hasattr(ot, "data_ptr") and ot.data_ptr() != 0:
                    dptr = ot.data_ptr()
            except Exception:
                pass

            if obj_id in producer_by_obj_id:
                source_id = producer_by_obj_id[obj_id]
                raw_edges.append(
                    TraceEdge(
                        source=source_id,
                        target="OUTPUT",
                        edge_type="explicit_tensor",
                        tensor_shape=list(ot.shape),
                        tensor_dtype=str(ot.dtype).replace("torch.", ""),
                        description="Model final output (verified object)",
                        is_uncertain=False,
                    )
                )
            elif dptr is not None and dptr in producer_by_data_ptr:
                source_id = producer_by_data_ptr[dptr]
                raw_edges.append(
                    TraceEdge(
                        source=source_id,
                        target="OUTPUT",
                        edge_type="storage_alias",
                        tensor_shape=list(ot.shape),
                        tensor_dtype=str(ot.dtype).replace("torch.", ""),
                        description="Model final output (shared storage alias)",
                        is_uncertain=False,
                    )
                )

        # Remove forward hooks before optional torch.fx trace so hooks do not fire during symbolic trace
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass
        hooks.clear()

        # Optional torch.fx analysis with multi-input and residual skip propagation
        if use_fx is not False:
            try:
                import torch.fx as fx

                fx_graph = fx.symbolic_trace(model)
                trace_mode = "torch.fx+hooks"

                fx_node_sources: dict[Any, set[str]] = {}
                fx_mod_call_counts: dict[str, int] = {}

                for node in fx_graph.graph.nodes:
                    if node.op == "placeholder":
                        fx_node_sources[node] = {"INPUT"}
                    elif node.op == "call_module":
                        target_name = str(node.target)
                        call_idx = fx_mod_call_counts.get(target_name, 0)
                        fx_mod_call_counts[target_name] = call_idx + 1
                        mod_id = f"{target_name}#{call_idx}"

                        for inp in node.all_input_nodes:
                            for src in fx_node_sources.get(inp, set()):
                                if src != mod_id:
                                    fx_edges.append(
                                        TraceEdge(
                                            source=src,
                                            target=mod_id,
                                            edge_type="symbolic_fx",
                                            description="Symbolic FX verified data dependency",
                                            is_uncertain=False,
                                        )
                                    )
                        fx_node_sources[node] = {mod_id}
                    elif node.op in ("call_function", "call_method", "get_attr"):
                        # Propagate sources through functions/methods/attributes (e.g. add, cat, view, slice)
                        srcs: set[str] = set()
                        for inp in node.all_input_nodes:
                            srcs.update(fx_node_sources.get(inp, set()))
                        fx_node_sources[node] = srcs
                    elif node.op == "output":
                        for inp in node.all_input_nodes:
                            for src in fx_node_sources.get(inp, set()):
                                fx_edges.append(
                                    TraceEdge(
                                        source=src,
                                        target="OUTPUT",
                                        edge_type="symbolic_fx",
                                        description="Symbolic FX output dependency",
                                        is_uncertain=False,
                                    )
                                )
            except Exception as exc:
                if use_fx is True:
                    warnings.append(f"torch.fx symbolic trace failed: {exc}")
                trace_mode = "hooks"

    finally:
        # Clear retained tensors to free memory and prevent autograd retention
        _retained_tensors.clear()
        # Guarantee hook removal (if any remain)
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass
        # Guarantee buffer restoration across all modules
        with torch.no_grad():
            for module, orig_buffers, non_persistent, saved_vals in saved_buffers:
                module._non_persistent_buffers_set = non_persistent
                module._buffers = orig_buffers
                for b_name, b_saved in saved_vals.items():
                    if b_saved is None:
                        orig_buffers[b_name] = None
                        continue
                    current_buf = orig_buffers.get(b_name)
                    if current_buf is None or current_buf.shape != b_saved.shape:
                        orig_buffers[b_name] = b_saved.clone()
                    else:
                        current_buf.copy_(b_saved)
        # Guarantee training state restoration (preserves mixed modes)
        for m, orig_mode in orig_training_modes:
            m.training = orig_mode
        # Guarantee parameter gradient restoration
        for name, param in model.named_parameters():
            saved = saved_grads.get(name)
            if saved is not None:
                param.grad = saved
            else:
                param.grad = None

    total_duration_ms = (time.perf_counter() - trace_start) * 1000.0

    # Merge edges with deduplication and metadata enrichment
    final_edges: list[TraceEdge] = []
    seen_edges = set()

    # Prioritize hook edges that have observed shapes/dtypes
    for e in raw_edges:
        key = (e.source, e.target)
        seen_edges.add(key)
        final_edges.append(e)

    # Add non-redundant FX edges (e.g. residual skips, branches where intermediate ops were unhooked functions)
    for e in fx_edges:
        key = (e.source, e.target)
        if key not in seen_edges:
            seen_edges.add(key)
            final_edges.append(e)

    # If no explicit edges were captured for consecutive leaf nodes, add inferred sequential edges
    leaf_nodes = [n for n in observed_nodes if n.is_leaf]
    if len(leaf_nodes) > 1:
        has_any_leaf_edges = any(e.source != "INPUT" and e.target != "OUTPUT" for e in final_edges)
        if not has_any_leaf_edges:
            for i in range(len(leaf_nodes) - 1):
                src = leaf_nodes[i].id
                tgt = leaf_nodes[i + 1].id
                final_edges.append(
                    TraceEdge(
                        source=src,
                        target=tgt,
                        edge_type="inferred_sequential",
                        description="Inferred sequential execution order (data dependency unverified)",
                        is_uncertain=True,
                    )
                )
            limitations.append(
                "Some edges are inferred based on execution order; explicit tensor pointer matching was incomplete."
            )

    # Deduplicate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_tot = sum(p.numel() for p in model.parameters() if p.requires_grad)

    primary_device = str(next(model.parameters()).device) if list(model.parameters()) else "cpu"

    return ExecutionTrace(
        nodes=observed_nodes,
        edges=final_edges,
        input_metadata=input_metadata,
        output_metadata=output_metadata,
        total_parameters=total_params,
        trainable_parameters=trainable_tot,
        duration_ms=total_duration_ms,
        device=primary_device,
        trace_mode=trace_mode,
        warnings=warnings,
        limitations=limitations,
    )
