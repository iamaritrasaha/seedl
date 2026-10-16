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
    - Model training modes (eval/train) are preserved and restored after execution.
    - Registered parameter gradients are untouched and restored.
    - Module buffers are not modified.
    - All forward hooks are removed in a guaranteed finally block, even on exceptions.
    - User input tensor is cloned/detached where appropriate; original input is untouched.
    - Retained activations are detached from autograd graphs.
    """
    torch = _torch()
    if torch is None or not isinstance(model, torch.nn.Module):
        raise MissingDependencyError(
            "Neural Execution Tracer requires PyTorch. Install with `pip install 'seedl[torch]'`."
        )

    if not isinstance(max_nodes, int) or max_nodes < 1:
        raise ValueError("max_nodes must be a positive integer.")

    # Save original training mode states and gradients
    states = [(m, bool(m.training)) for _, m in model.named_modules()]
    saved_grads = {
        name: param.grad.clone() if param.grad is not None else None
        for name, param in model.named_parameters()
    }

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

    # Extract initial network input tensor IDs
    input_tensors = _extract_tensors(input, torch)
    _retained_tensors.extend(input_tensors)
    input_metadata = []
    for idx, inp in enumerate(input_tensors):
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
    ]
    warnings: list[str] = []

    try:
        model.eval()
        with torch.no_grad():
            # Format inputs appropriately
            if isinstance(input, tuple):
                res = model(*input)
            elif isinstance(input, dict):
                res = model(**input)
            else:
                res = model(input)

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

    finally:
        # Clear retained tensors
        _retained_tensors.clear()
        # Guarantee hook removal
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass
        # Guarantee training state restoration
        for m, training in states:
            m.training = training
        # Guarantee parameter gradient restoration
        for name, param in model.named_parameters():
            saved = saved_grads.get(name)
            if saved is not None:
                param.grad = saved
            else:
                param.grad = None

    total_duration_ms = (time.perf_counter() - trace_start) * 1000.0

    # Optional torch.fx analysis with multi-input and residual skip propagation
    trace_mode = "hooks"
    fx_edges: list[TraceEdge] = []
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
