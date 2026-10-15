"""Bounded, detached observations of user-owned PyTorch training passes."""

from __future__ import annotations

import json
import math
import weakref
from collections import deque
from dataclasses import dataclass
from typing import Any

from .exceptions import MissingDependencyError, SeeDLError


def _positive(name: str, value: int, upper: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
        raise ValueError(f"{name} must be an integer between 1 and {upper}.")


def _finite(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class TrainingSnapshot:
    """Immutable encoded observation; ``data`` returns an independent decoded copy."""

    _payload: bytes

    @property
    def data(self) -> dict[str, Any]:
        return json.loads(self._payload)

    @property
    def nbytes(self) -> int:
        return len(self._payload)


class TrainingRecorder:
    """Keep no tensors, model ownership, or autograd graphs in recorded snapshots."""

    def __init__(
        self,
        model: Any,
        *,
        max_snapshots: int = 64,
        max_bytes: int = 8 * 1024 * 1024,
        every: int = 1,
        max_values: int = 2048,
        max_neurons: int = 100,
        max_connections: int = 1500,
        sample: int = 0,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise MissingDependencyError(
                "Training capture requires `pip install 'seedl[torch]'`."
            ) from exc
        if not isinstance(model, torch.nn.Module):
            raise SeeDLError("Training capture requires a PyTorch nn.Module.")
        for name, value, upper in (
            ("max_snapshots", max_snapshots, 1000),
            ("max_bytes", max_bytes, 256 * 1024 * 1024),
            ("every", every, 1_000_000),
            ("max_values", max_values, 100_000),
            ("max_neurons", max_neurons, 1000),
            ("max_connections", max_connections, 20_000),
        ):
            _positive(name, value, upper)
        if isinstance(sample, bool) or not isinstance(sample, int) or sample < 0:
            raise ValueError("sample must be a nonnegative integer.")
        self._model = weakref.ref(model)
        self.max_snapshots, self.max_bytes, self.every = max_snapshots, max_bytes, every
        self.max_values, self.max_neurons, self.max_connections = (
            max_values,
            max_neurons,
            max_connections,
        )
        self.sample = sample
        self.snapshots: deque[TrainingSnapshot] = deque()
        self.nbytes = 0
        self.dropped = 0
        self._calls = 0
        self._last_step = -1
        self._handles: list[Any] = []
        self._pending: dict[str, Any] | None = None
        self._versions: tuple[Any, ...] | None = None
        self._cursor = 0
        self._stage = 0
        self._plan: Any = None

    def _parameters(self) -> list[tuple[str, Any]]:
        model = self._model()
        if model is None:
            raise SeeDLError("The observed model is no longer available.")
        import torch

        result: list[tuple[str, Any]] = []
        for name, parameter in model.named_parameters():
            if len(result) >= 256:
                raise SeeDLError("Training capture supports up to 256 parameter tensors.")
            if isinstance(parameter, torch.nn.parameter.UninitializedParameter):
                raise SeeDLError("Initialize lazy parameters with a forward pass before recording.")
            if (
                parameter.device.type == "meta"
                or parameter.is_complex()
                or parameter.layout != torch.strided
            ):
                raise SeeDLError("Training capture requires initialized, dense real parameters.")
            result.append((name, parameter))
        return result

    def _signature(self) -> tuple[Any, ...]:
        return tuple((name, id(p), p._version, tuple(p.shape)) for name, p in self._parameters())

    def start(self) -> None:
        if self._handles:
            raise SeeDLError("This tracker is already observing a training context.")
        import torch

        from ._neurons import _plan

        model = self._model()
        self._parameters()
        assert model is not None
        self._plan = _plan(model, torch)
        if self._plan:
            widths = [self._plan[1][0]["module"].in_features] + [
                s["module"].out_features for s in self._plan[1]
            ]
            if sum(widths) > self.max_neurons or max(widths) > 48 or len(widths) > 12:
                self._plan = None
        self._handles.append(model.register_forward_pre_hook(self._begin, with_kwargs=True))
        if self._plan:
            for module in dict.fromkeys(m for _, m in self._plan[0]):
                self._handles.append(
                    module.register_forward_hook(self._activation, with_kwargs=True)
                )
        self._handles.append(model.register_forward_hook(self._end, with_kwargs=True))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._pending = None
        self._versions = None
        self._plan = None

    def _begin(self, module: Any, args: Any, kwargs: Any) -> None:
        self._pending = None
        self._versions = self._signature()
        self._cursor = self._stage = 0
        self._pending = {"activations": {}, "output": None, "complete": False}

    def _activation(self, module: Any, args: Any, kwargs: Any, output: Any) -> None:
        if self._pending is None:
            return
        import torch

        from ._neurons import _number

        leaves, _ = self._plan
        if self._cursor >= len(leaves) or leaves[self._cursor][1] is not module:
            self._pending = None
            return

        def values(tensor: Any) -> list[float | None] | None:
            if not isinstance(tensor, torch.Tensor) or tensor.ndim not in {1, 2}:
                return None
            if tensor.ndim == 2:
                if self.sample >= tensor.shape[0]:
                    return None
                tensor = tensor[self.sample]
            elif self.sample:
                return None
            return [_number(v) for v in tensor.detach().cpu().tolist()]

        if type(module) is torch.nn.Linear:
            if self._stage == 0:
                self._pending["activations"]["0"] = {
                    "activation": values(args[0] if args else kwargs["input"])
                }
            self._stage += 1
            self._pending["activations"][str(self._stage)] = {"preactivation": values(output)}
        self._pending["activations"][str(self._stage)]["activation"] = values(output)
        self._cursor += 1

    def _sample_tensor(self, tensor: Any, indices: list[int]) -> list[float | None]:
        import torch

        # Index before copying: no full GPU tensor or noncontiguous reshape copy.
        if not indices:
            return []
        flat = torch.tensor(indices, dtype=torch.long, device=tensor.device)
        coords = []
        for size in reversed(tensor.shape):
            coords.append(flat % size)
            flat = flat // size
        selected = (
            tensor.detach()[tuple(reversed(coords))]
            if coords
            else tensor.detach().expand(len(indices))
        )
        return [_finite(v) for v in selected.cpu().tolist()]

    def _tensor_preview(self, tensor: Any, limit: int) -> dict[str, Any] | None:
        import torch

        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.is_complex()
            or tensor.layout != torch.strided
        ):
            return None
        count = tensor.numel()
        stride = max(1, math.ceil(count / limit))
        indices = list(range(0, count, stride))
        return {
            "shape": list(tensor.shape),
            "indices": indices,
            "values": self._sample_tensor(tensor, indices),
        }

    def _end(self, module: Any, args: Any, kwargs: Any, output: Any) -> None:
        if self._pending is not None:
            self._pending["complete"] = True
            self._pending["output"] = self._tensor_preview(output, self.max_values)
            if self._versions != self._signature():
                self._pending = None

    def capture(
        self, *, step: int | None = None, epoch: int | float | None = None, loss: Any = None
    ) -> TrainingSnapshot | None:
        if not self._handles:
            raise SeeDLError("Use `with sd.watch(model) as run:` before calling run.capture().")
        step = self._last_step + 1 if step is None else step
        if isinstance(step, bool) or not isinstance(step, int) or step <= self._last_step:
            raise ValueError("step must be a strictly increasing nonnegative integer.")
        epoch_value = _finite(epoch) if epoch is not None else None
        if epoch is not None and epoch_value is None:
            raise ValueError("epoch must be finite.")
        if hasattr(loss, "detach"):
            loss = loss.detach().item()
        loss_value = _finite(loss) if loss is not None else None
        if (
            self._pending is None
            or not self._pending["complete"]
            or self._versions != self._signature()
        ):
            raise SeeDLError(
                "Capture after a completed forward/backward pass and before optimizer.step(); parameters changed or no valid forward was observed."
            )
        self._calls += 1
        self._last_step = step
        if (self._calls - 1) % self.every:
            self._pending = None
            return None
        parameters = self._parameters()
        total = sum(p.numel() for _, p in parameters)
        stride = max(1, math.ceil(total / self.max_values))
        offset = 0
        rows = {}
        import torch

        for name, p in parameters:
            indices = list(range((-offset) % stride, p.numel(), stride))
            grad = p.grad
            rows[name] = {
                "shape": list(p.shape),
                "indices": indices,
                "values": self._sample_tensor(p, indices),
                "gradients": self._sample_tensor(grad, indices)
                if grad is not None and grad.layout == torch.strided
                else None,
            }
            offset += p.numel()
        graph = None
        if self._plan and self._cursor == len(self._plan[0]):
            from ._neurons import _build_graph

            stages = self._plan[1]
            widths = [stages[0]["module"].in_features] + [s["module"].out_features for s in stages]
            graph = _build_graph(stages, widths, self.max_connections, 0)
            graph.update(
                theme="scientific",
                color_by="activation",
                sample=self.sample,
                dimensions=3,
                edge_opacity=0.45,
                selected_neuron=None,
                playback=True,
                caption="Recorded training pass",
            )
            for layer, captured in self._pending["activations"].items():
                for field, values in captured.items():
                    if values is not None:
                        for node, value in zip(
                            graph["layers"][int(layer)]["nodes"], values, strict=True
                        ):
                            node[field] = value
                            node[field + "_status"] = (
                                "available" if value is not None else "non-finite"
                            )
            graph["captured"] = True
            # Reconstruct aliases only while rendering, not in the bounded store.
            graph.pop("node_map")
            for node in graph["nodes"]:
                node.pop("incoming_edges")
                node.pop("outgoing_edges")
            for layer in graph["layers"]:
                layer["nodes"] = [n["id"] for n in layer["nodes"]]
        payload = {
            "step": step,
            "epoch": epoch_value,
            "loss": loss_value,
            "parameters": rows,
            **self._pending,
            "graph": graph,
        }
        encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        self._pending = None
        if len(encoded) > self.max_bytes:
            raise SeeDLError(
                f"Snapshot needs {len(encoded)} bytes; max_bytes={self.max_bytes}. Reduce capture limits."
            )
        snapshot = TrainingSnapshot(encoded)
        while self.snapshots and (
            len(self.snapshots) >= self.max_snapshots
            or self.nbytes + snapshot.nbytes > self.max_bytes
        ):
            self.nbytes -= self.snapshots.popleft().nbytes
            self.dropped += 1
        self.snapshots.append(snapshot)
        self.nbytes += snapshot.nbytes
        return snapshot


def restore_graph(snapshot: dict[str, Any], color_by: str = "activation") -> dict[str, Any] | None:
    graph = snapshot["graph"]
    if graph is None:
        return None
    nodes = {n["id"]: n for n in graph["nodes"]}
    graph["node_map"] = nodes
    graph["color_by"] = color_by
    for node in nodes.values():
        node.update(incoming_edges=[], outgoing_edges=[])
    for edge in graph["edges"]:
        nodes[edge["source"]]["outgoing_edges"].append(edge)
        nodes[edge["target"]]["incoming_edges"].append(edge)
    for layer in graph["layers"]:
        layer["nodes"] = [nodes[n] for n in layer["nodes"]]
    return graph
