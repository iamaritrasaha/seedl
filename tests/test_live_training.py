"""Real training provenance, capture boundaries, and bounded payload regressions."""

from __future__ import annotations

import gc
import json
import weakref
from dataclasses import replace

import pytest
import torch
from torch import nn

import seedl as sd
from seedl._recording import restore_graph
from seedl.exceptions import SeeDLError


def train(steps=4, **options):
    torch.manual_seed(17)
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
    x = torch.tensor([[0.5, -1.0], [1.0, 0.0]])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with sd.watch(model, **options) as tracker:
        for step in range(steps):
            optimizer.zero_grad()
            output = model(x)
            loss = (output - 0.4).square().mean()
            loss.backward()
            tracker.capture(step=step * 2, epoch=step // 2, loss=loss)
            optimizer.step()
    return model, tracker


def test_exact_captured_values_shapes_and_weight_indexing():
    torch.manual_seed(4)
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
    x = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    with sd.track(model, sample=1) as tracker:
        output = model(x)
        loss = output.square().sum()
        loss.backward()
        gradients = {n: p.grad.clone() for n, p in model.named_parameters()}
        snapshot = tracker.capture(step=8, epoch=2, loss=loss)
        assert snapshot is not None
        data = snapshot.data
        assert data["output"]["shape"] == [2, 1]
        assert data["output"]["values"] == output.detach().flatten().tolist()
        assert data["loss"] == loss.item()
        graph = restore_graph(data)
        for edge in graph["edges"]:
            source = int(edge["source"].split(":")[1])
            destination = int(edge["target"].split(":")[1])
            assert edge["weight"] == before[edge["parameter"]][destination, source].item()
            assert edge["gradient"] == gradients[edge["parameter"]][destination, source].item()
        expected = torch.relu(model[0](x))[1].detach().tolist()
        assert [n["activation"] for n in graph["layers"][1]["nodes"]] == expected
        for name, parameter in model.named_parameters():
            assert torch.equal(parameter, before[name])
            assert torch.equal(parameter.grad, gradients[name])
    assert not model._forward_hooks and not model._forward_pre_hooks
    assert all(not module._forward_hooks for module in model.modules())
    assert tracker._recorder._pending is None
    assert tracker._recorder._plan is None
    # Mutation of a decoded copy cannot corrupt a recording.
    data["loss"] = -1
    assert snapshot.data["loss"] == loss.item()


def test_capture_preserves_rng_modes_buffers_and_gradients():
    model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2), nn.Dropout(0.2))
    model[1].eval()  # Mixed mode must survive observation.
    with sd.watch(model) as run:
        y = model(torch.ones(3, 2))
        y.sum().backward()
        modes = [m.training for m in model.modules()]
        rng = torch.random.get_rng_state().clone()
        buffers = {n: b.clone() for n, b in model.named_buffers()}
        gradients = [p.grad.clone() for p in model.parameters()]
        parameters = [p.clone() for p in model.parameters()]
        run.capture()
        assert torch.equal(rng, torch.random.get_rng_state())
        assert modes == [m.training for m in model.modules()]
        assert all(torch.equal(b, buffers[n]) for n, b in model.named_buffers())
        assert all(torch.equal(p, v) for p, v in zip(model.parameters(), parameters, strict=True))
        assert all(
            torch.equal(p.grad, v) for p, v in zip(model.parameters(), gradients, strict=True)
        )


def test_parameter_change_rejected_and_hooks_removed_on_error():
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(SeeDLError, match="before optimizer.step"):
        with sd.watch(model) as tracker:
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            tracker.capture()
    assert not model._forward_hooks and not model._forward_pre_hooks
    assert tracker._recorder._pending is None
    with pytest.raises(SeeDLError, match="with sd.watch"):
        tracker.capture()


def test_failed_forward_never_becomes_a_snapshot():
    class Fails(nn.Module):
        def forward(self, x):
            raise RuntimeError("failed forward")

    model = Fails()
    with sd.watch(model) as run:
        with pytest.raises(RuntimeError):
            model(torch.ones(1))
        with pytest.raises(SeeDLError, match="completed forward"):
            run.capture()


def test_no_tensor_or_graph_retention_and_model_can_be_collected():
    model = nn.Linear(2, 1)
    ref = weakref.ref(model)
    with sd.watch(model) as run:
        output = model(torch.ones(1, 2))
        output_ref = weakref.ref(output)
        output.sum().backward()
        run.capture()
        del output
        gc.collect()
        assert output_ref() is None
    assert isinstance(run.snapshots[0]._payload, bytes)
    json.loads(run.snapshots[0]._payload)
    del model
    gc.collect()
    assert ref() is None


def test_count_byte_and_every_bounds_are_independent():
    _, tracker = train(30, max_snapshots=3, every=2)
    assert [s.data["step"] for s in tracker.snapshots] == [48, 52, 56]
    assert tracker.recording_stats["dropped"] == 12
    assert tracker.recording_stats["bytes"] == sum(s.nbytes for s in tracker.snapshots)
    _, one = train(1)
    budget = one.snapshots[0].nbytes * 2 + 100
    _, bounded = train(20, max_snapshots=100, max_bytes=budget)
    assert bounded.recording_stats["bytes"] <= budget
    assert 1 <= len(bounded.snapshots) <= 2
    model = nn.Linear(2, 1)
    with sd.watch(model, max_bytes=100) as run:
        model(torch.ones(1, 2))
        with pytest.raises(SeeDLError, match="max_bytes"):
            run.capture()
        assert run.snapshots == ()


def test_global_parameter_sampling_keeps_original_indices_and_noncontiguous_values():
    model = nn.Linear(100, 80, bias=False)
    model.weight = nn.Parameter(torch.arange(8000.0).reshape(100, 80).T)
    with sd.watch(model, max_values=11) as run:
        y = model(torch.ones(1, 100))
        y.sum().backward()
        data = run.capture().data
    p = data["parameters"]["weight"]
    assert len(p["indices"]) <= 11
    assert p["values"] == model.weight.detach().reshape(-1)[p["indices"]].tolist()
    assert p["shape"] == [80, 100]
    assert data["graph"] is None


def test_absent_nonfinite_and_sparse_gradients_are_explicit():
    model = nn.Linear(2, 1, bias=False)
    with sd.watch(model) as run:
        model(torch.ones(1, 2))
        first = run.capture()
        assert first.data["parameters"]["weight"]["gradients"] is None
        model.weight.grad = torch.full_like(model.weight, float("nan"))
        model(torch.ones(1, 2))
        last = run.capture(loss=float("inf"))
        assert last.data["parameters"]["weight"]["gradients"] == [None, None]
        assert last.data["loss"] is None
    # A bias-free node has no invented bias gradient.
    assert last.data["graph"]["nodes"][-1]["gradient"] is None


def test_snapshot_compare_alignment_and_rendering(tmp_path):
    _, tracker = train()
    a, b = tracker.snapshots[0], tracker.snapshots[-1]
    view = sd.compare(a, b, backend="svg")
    for row in view.data["parameters"]:
        assert row["status"] == "aligned"
        pa, pb = a.data["parameters"][row["name"]], b.data["parameters"][row["name"]]
        assert row["delta"] == [y - x for x, y in zip(pa["values"], pb["values"], strict=True)]
        assert row["shape"] == pa["shape"]
    assert view.data["outputs"]["status"] == "aligned"
    assert view.data["activations"]
    assert "<svg" in view.figure
    assert len(view.to_figure("plotly").data) > 4
    data = b.data
    data["parameters"]["0.weight"]["shape"] = [6]
    mismatch = sd.compare(a, replace(b, _payload=json.dumps(data).encode()), backend="svg")
    assert mismatch.data["parameters"][0]["status"] == "shape mismatch"
    data["parameters"]["0.weight"]["shape"] = a.data["parameters"]["0.weight"]["shape"]
    data["parameters"]["0.weight"]["indices"] = [1, 3, 5]
    data["parameters"]["0.weight"]["values"] = [100.0, 200.0, 300.0]
    data["parameters"]["0.weight"]["gradients"] = None
    subset = sd.compare(a, replace(b, _payload=json.dumps(data).encode()), backend="svg")
    assert subset.data["parameters"][0]["indices"] == [1, 3, 5]
    assert subset.data["parameters"][0]["after"] == [100.0, 200.0, 300.0]


def test_playback_exports_svg_and_notebook_html(tmp_path):
    _, run = train()
    view = sd.see(run, backend="plotly")
    assert view.kind == "training_playback"
    assert view.figure.layout.uirevision == "seedl-training"
    assert "<svg" in view.to_figure("svg")
    target = view.save(tmp_path / "training.html")
    source = target.read_text(encoding="utf-8")
    assert source.count("plotly.js v") == 1
    assert "setTimeout" in source and "seedl-playback" in source
    assert "cdn.plot.ly/" not in source.split("Plotly.newPlot")[-1]
    bundle = view._repr_mimebundle_()
    assert "seedl-playback" in bundle["text/html"]
    assert "text/plain" in bundle
    assert len(source) < 6_000_000
    # Decoding/rendering does not attach aliases to the immutable payload.
    assert "node_map" not in run.snapshots[0].data["graph"]


@pytest.mark.parametrize(
    "options",
    [{"max_snapshots": 0}, {"max_bytes": True}, {"every": -1}, {"sample": -1}, {"max_values": 0}],
)
def test_capture_option_validation(options):
    with pytest.raises(ValueError):
        sd.watch(nn.Linear(2, 1), **options)


def test_metric_only_behavior_stays_compatible():
    with sd.watch() as run:
        run.log(epoch=1, loss=0.5)
    assert sd.see(run, backend="svg").kind == "training"
    with pytest.raises(SeeDLError, match="Pass a model"):
        run.capture()
    with pytest.raises(SeeDLError, match="No training snapshots"):
        run.playback()


def test_root_linear_names_sparse_gradients_and_backend_flags():
    model = nn.Linear(2, 1)
    with sd.watch(model) as run:
        model(torch.ones(1, 2)).sum().backward()
        data = run.capture().data
    assert {edge["parameter"] for edge in data["graph"]["edges"]} == {"weight"}
    assert sd.see(run, interactive=False).backend == "svg"
    with pytest.raises(ValueError, match="color_by"):
        sd.see(run, color_by="invalid")
    embedding = nn.Embedding(20, 3, sparse=True)
    with sd.watch(embedding) as sparse_run:
        embedding(torch.tensor([1, 4])).sum().backward()
        sparse_data = sparse_run.capture().data
    assert sparse_data["parameters"]["weight"]["gradients"] is None
    assert embedding.weight.grad.is_sparse


def test_export_frame_payload_limit(monkeypatch):
    from seedl import _playback

    _, run = train()
    monkeypatch.setattr(_playback, "MAX_EXPORT_BYTES", 100)
    with pytest.raises(SeeDLError, match="Playback payload exceeds"):
        run.playback().to_html()


def test_timeline_color_and_weight_scales_stay_fixed():
    _, run = train()
    view = run.playback()
    from seedl._playback import training_figure

    first, last = training_figure(view, 0), training_figure(view, len(run.snapshots) - 1)

    def scales(figure):
        return {
            (trace.meta["layer"], trace.marker.cmin, trace.marker.cmax)
            for trace in figure.data
            if trace.meta and trace.meta.get("role") == "nodes" and trace.marker.cmin is not None
        }

    assert scales(first) == scales(last)
    assert view.data["weight_scale"] == max(
        abs(edge["weight"])
        for snapshot in run.snapshots
        for edge in snapshot.data["graph"]["edges"]
    )


def test_automatic_step_continues_after_explicit_step_with_sampling():
    model = nn.Linear(2, 1)
    with sd.watch(model, every=2) as run:
        model(torch.ones(1, 2))
        run.capture(step=10)
        model(torch.ones(1, 2))
        assert run.capture() is None
        model(torch.ones(1, 2))
        snapshot = run.capture()
    assert snapshot.data["step"] == 12
    assert [s.data["step"] for s in run.snapshots] == [10, 12]


def test_large_model_gradient_preview_uses_gradients_and_preserves_missingness():
    model = nn.Linear(80, 80, bias=False)
    with sd.watch(model, max_values=4) as run:
        model(torch.full((1, 80), 2.0)).sum().backward()
        first = run.capture()
        model.zero_grad(set_to_none=True)
        model(torch.ones(1, 80))
        run.capture()
    from seedl._playback import training_figure

    view = run.playback(color_by="gradient")
    assert (
        list(view.figure.data[0].y) == first.data["parameters"]["weight"]["gradients"] == [2.0] * 4
    )
    assert view.figure.layout.yaxis.title.text == "Gradient"
    assert list(training_figure(view, 1).data[0].y) == [None] * 4


def test_html_limit_counts_actual_serialization_and_initial_figure(monkeypatch):
    from seedl import _playback

    _, run = train(1)
    view = run.playback()
    view.title = "<" * 3000
    content = _playback.playback_html(view, include_plotlyjs=False)
    monkeypatch.setattr(_playback, "MAX_EXPORT_BYTES", len(content.encode()) - 100)
    with pytest.raises(SeeDLError, match="Playback payload exceeds"):
        _playback.playback_html(view, include_plotlyjs=False)
