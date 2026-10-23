from __future__ import annotations

import pytest

import seedl as sd
from seedl.exceptions import SeeDLError


def test_three_component_pca_matches_covariance_and_sampling():
    np = pytest.importorskip("numpy")
    pytest.importorskip("plotly")
    x = np.array([[3, 0, 0], [-3, 0, 0], [0, 2, 0], [0, -2, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    v = sd.embeddings(x, dimensions=3, labels=list("abcdef"), categories=["a"] * 3 + ["b"] * 3)
    assert v.data["pca_variance_ratio"] == pytest.approx([9 / 14, 4 / 14, 1 / 14])
    assert v.data["explained_variance"] == pytest.approx(1)
    assert np.asarray(v.data["coords_3d"]) * v.data["coordinate_scale"] == pytest.approx(x)
    assert all(t.type == "scatter3d" for t in v.figure.data)
    sampled = sd.embeddings(x, dimensions=3, max_vectors=3)
    assert sampled.data["indices"] == [0, 2, 5]
    assert sampled.data["labels"] == ["V0", "V2", "V5"]


def test_degenerate_and_invalid_embedding_data():
    pytest.importorskip("numpy")
    pytest.importorskip("plotly")
    v = sd.embeddings([[1, 1, 1]] * 4, dimensions=3)
    assert v.data["explained_variance"] == 0
    assert v.data["coords_3d"] == [[0, 0, 0]] * 4
    for x in ([[1, 2, 3]], [[1, 2]] * 3, [[float("nan"), 0, 1]] * 3, [[1, 2, 3], [1], [2, 3, 4]]):
        with pytest.raises(SeeDLError):
            sd.embeddings(x, dimensions=3)
    with pytest.raises(ValueError):
        sd.embeddings([[1, 2, 3]] * 3, dimensions=3, labels=["a"])
    with pytest.raises(ValueError):
        sd.embeddings([[1, 2, 3]] * 3, dimensions=3, backend="svg")


def test_loss_surface_matches_analytic_quadratic_and_preserves_state(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("plotly")
    m = torch.nn.Linear(1, 1)
    with torch.no_grad():
        m.weight.fill_(2)
        m.bias.fill_(1)
    m.weight.grad = torch.full_like(m.weight, 7)
    x = torch.tensor([[1.0], [2.0]], requires_grad=True)
    y = torch.tensor([[3.0], [5.0]])
    original = {k: v.clone() for k, v in m.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    v = sd.loss_landscape(
        m,
        (x, y),
        loss_fn=torch.nn.functional.mse_loss,
        axes=(("weight", 0), ("bias", 0)),
        steps=3,
        radius=1,
        evaluate=True,
        trajectory=[(0, 0), (0.5, 0.5)],
    )
    for row, b in zip(v.data["losses"], [-1, 0, 1], strict=True):
        assert row == pytest.approx([((a + b) ** 2 + (2 * a + b) ** 2) / 2 for a in [-1, 0, 1]])
    assert v.figure.data[0].type == "surface"
    assert v.data["trajectory"][0] == [0, 0, 0]
    assert v.data["evaluations"] == 11
    assert m.training and m.weight.grad.item() == 7 and m.bias.grad is None and x.grad is None
    assert all(torch.equal(original[k], val) for k, val in m.state_dict().items())
    assert torch.equal(rng, torch.random.get_rng_state())
    assert "surface" in v.save(tmp_path / "slice.html").read_text(encoding="utf-8")


def test_loss_requires_consent_and_valid_axes_and_cleans_failure():
    torch = pytest.importorskip("torch")
    pytest.importorskip("plotly")
    m = torch.nn.Linear(1, 1)
    opts = dict(loss_fn=torch.nn.functional.mse_loss, axes=(("weight", 0), ("bias", 0)), steps=3)
    data = (torch.ones(1, 1), torch.ones(1, 1))
    with pytest.raises(SeeDLError, match="evaluate=True"):
        sd.loss_landscape(m, data, **opts)
    opts["axes"] = (("weight", 0), ("weight", 0))
    with pytest.raises(ValueError, match="distinct"):
        sd.loss_landscape(m, data, evaluate=True, **opts)
    opts["axes"] = (("weight", 0), ("bias", 0))
    original = m.weight.clone()

    def bad_loss(prediction, target):
        raise RuntimeError("failure")

    opts["loss_fn"] = bad_loss
    with pytest.raises(RuntimeError, match="failure"):
        sd.loss_landscape(m, data, evaluate=True, **opts)
    assert torch.equal(m.weight, original) and m.training


def test_loss_slice_resets_nonpersistent_buffers_and_python_state_per_point():
    torch = pytest.importorskip("torch")
    pytest.importorskip("plotly")

    class Stateful(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.bias = torch.nn.Parameter(torch.zeros(1))
            self.register_buffer("counter", torch.zeros(1), persistent=False)
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            self.counter.add_(1)
            return x * self.weight + self.bias + self.counter + self.calls

    model = Stateful()
    view = sd.loss_landscape(
        model,
        (torch.ones(1), torch.zeros(1)),
        loss_fn=torch.nn.functional.mse_loss,
        axes=(("weight", 0), ("bias", 0)),
        steps=3,
        evaluate=True,
    )
    for row, b in zip(view.data["losses"], [-1, 0, 1], strict=True):
        assert row == pytest.approx([(2 + a + b) ** 2 for a in [-1, 0, 1]])
    assert model.counter.item() == 0 and model.calls == 0
