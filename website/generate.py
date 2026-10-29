"""Build deterministic SeeDL 0.6.0 demos; no server or CDN is needed at runtime.

Run from the repository root using the published SeeDL package:
    .venv/bin/python website/generate.py
"""

from __future__ import annotations

import contextlib
import hashlib
import html
import io
import json
import math
import shutil
from pathlib import Path

import numpy as np
import plotly
import plotly.offline
import qiskit
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector, partial_trace

import seedl as sd
from seedl._network_renderers import build_network_interactive_js
from seedl._playback import playback_html

ROOT = Path(__file__).resolve().parent


class DemoCNN(torch.nn.Module):
    """Small deterministic CNN with Conv2d, ReLU, and MaxPool2d."""

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        return self.relu2(self.conv2(self.pool1(self.relu1(self.conv1(x)))))


class ResidualMLP(torch.nn.Module):
    """Small deterministic MLP with a residual skip connection."""

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 6)
        self.relu1 = torch.nn.ReLU()
        self.res_fc = torch.nn.Linear(6, 6)
        self.relu2 = torch.nn.ReLU()
        self.fc_out = torch.nn.Linear(6, 2)

    def forward(self, x):
        h1 = self.relu1(self.fc1(x))
        h2 = self.relu2(h1 + self.res_fc(h1))
        return self.fc_out(h2)


def training_example():
    """Small CPU-only regression; captures belong to the pass before each update."""
    torch.manual_seed(41)
    model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1)).cpu()
    x = torch.tensor([[1.0, -2.0]])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.08)
    with sd.watch(
        model,
        name="Tiny regression",
        max_snapshots=16,
        max_bytes=512_000,
        max_values=12,
        max_neurons=12,
        max_connections=16,
    ) as run:
        for step in range(16):
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            loss = (output - 0.5).square().mean()
            loss.backward()
            snapshot = run.capture(step=step, epoch=step // 4, loss=loss)
            # Verify provenance at the capture boundary, before optimizer.step().
            assert snapshot.data["output"]["values"] == output.detach().flatten().tolist()
            for name, parameter in model.named_parameters():
                captured = snapshot.data["parameters"][name]
                indices = captured["indices"]
                assert captured["values"] == parameter.detach().flatten()[indices].tolist()
                assert captured["gradients"] == parameter.grad.flatten()[indices].tolist()
            optimizer.step()
    assert run.snapshots[-1].data["loss"] < run.snapshots[0].data["loss"]
    assert run.recording_stats["dropped"] == 0
    return run


def verify_evolution(view, qc):
    """Check every retained amplitude, probability, phase and Bloch vector with Qiskit."""
    prefix = QuantumCircuit(qc.num_qubits)
    for index, step in enumerate(view.data["evolution"]):
        if index:
            op = qc.data[index - 1]
            prefix.append(op.operation, [qc.find_bit(q).index for q in op.qubits])
        expected = Statevector.from_instruction(prefix)
        assert step["statevector_valid"]
        np.testing.assert_allclose(
            [complex(a["real"], a["imag"]) for a in step["amplitudes"]],
            expected.data,
            atol=1e-12,
        )
        np.testing.assert_allclose(step["probabilities"], expected.probabilities(), atol=1e-12)
        np.testing.assert_allclose(
            [a["phase"] for a in step["amplitudes"]], np.angle(expected.data), atol=1e-12
        )
        for q, vector in enumerate(step["bloch_vectors"]):
            reduced = partial_trace(expected, [i for i in range(qc.num_qubits) if i != q])
            np.testing.assert_allclose(
                vector[:3],
                [reduced.expectation_value(Pauli(a)).real for a in "XYZ"],
                atol=1e-12,
            )


def export_playback(view, name):
    """Use the library export option; retain its controller without HTML surgery."""
    body = playback_html(view, full_html=False, include_plotlyjs=False)
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{name} · SeeDL 0.6.0</title>"
        '<script src="../assets/plotly.min.js"></script>'
        '<link rel="stylesheet" href="../assets/playback.css">'
        f"</head><body>{body}</body></html>"
    )
    (ROOT / "demos" / f"{name}.html").write_text(document)


def export(view, name: str, *, network: bool = False) -> None:
    """Keep SeeDL's own interaction bridge and share one local Plotly runtime."""
    fig = view.figure
    fig.update_layout(autosize=True, height=None, width=None)
    post_script = None
    if network:
        post_script = build_network_interactive_js(view.data["neuron_graph"], True)
    else:
        # Preset-only Bloch demos: gate shortcuts change the arrow without all
        # the associated annotations. Keep each export self-consistent.
        if view.kind == "bloch":
            fig.update_layout(updatemenus=[])
    body = fig.to_html(
        include_plotlyjs="../assets/plotly.min.js",
        full_html=False,
        div_id="seedl-demo",
        config={"responsive": True, "scrollZoom": True, "displaylogo": False},
        post_script=post_script,
    )
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{view.kind} · SeeDL 0.6.0</title>"
        "<style>html,body{margin:0;background:#0f172a;height:100%;}"
        ".plotly-graph-div{width:100%!important;height:100%!important;min-height:560px;}"
        "body>div{height:100%}*:focus-visible{outline:2px solid #00e5ff}</style>"
        f"</head><body>{body}</body></html>"
    )
    (ROOT / "demos" / f"{name}.html").write_text(document)


def build_report_preview_svg(report: object) -> str:
    """Generate a deterministic SVG overview representing the multi-section scientific report."""
    title = html.escape(str(getattr(report, "title", "Scientific Diagnostic Report")))
    author = html.escape(str(getattr(report, "author", "SeeDL Scientific Verification Team")))
    sections = list(getattr(report, "sections", []))

    sec_meta = [
        {"color": "#38bdf8", "tag": "EXECUTION GRAPH & PIPELINE", "badge": "DAG"},
        {"color": "#a855f7", "tag": "TRANSFORMER ATTENTION", "badge": "ROLLOUT"},
        {"color": "#fbbf24", "tag": "AUTOGRAD TELEMETRY", "badge": "GRADIENTS"},
        {"color": "#34d399", "tag": "OPEN QUANTUM SYSTEMS", "badge": "KRAUS"},
        {"color": "#f87171", "tag": "DISTRIBUTION DIVERGENCE", "badge": "DRIFT"},
    ]

    cards_svg: list[str] = []
    y_start = 140
    card_h = 56
    gap = 12

    for idx, sec in enumerate(sections[:5]):
        y = y_start + idx * (card_h + gap)
        sec_title = html.escape(str(getattr(sec, "title", f"Section {idx + 1}")))
        meta = (
            sec_meta[idx]
            if idx < len(sec_meta)
            else {"color": "#38bdf8", "tag": "REPORT SECTION", "badge": "VIEW"}
        )
        color = meta["color"]
        tag = html.escape(str(meta["tag"]))
        badge = html.escape(str(meta["badge"]))
        cards_svg.append(f"""  <g transform="translate(40, {y})">
    <rect width="700" height="{card_h}" rx="8" fill="#1e293b" stroke="#334155" stroke-width="1"/>
    <rect x="0" y="0" width="6" height="{card_h}" rx="3" fill="{color}"/>
    <text x="24" y="23" font-family="system-ui, -apple-system, sans-serif" font-size="11" font-weight="700" fill="{color}" letter-spacing="0.5">{tag}</text>
    <text x="24" y="43" font-family="system-ui, -apple-system, sans-serif" font-size="14" font-weight="600" fill="#f8fafc">{idx + 1}. {sec_title}</text>
    <rect x="580" y="15" width="96" height="26" rx="5" fill="#0f172a" stroke="#334155" stroke-width="1"/>
    <text x="628" y="32" font-family="monospace" font-size="11" font-weight="700" fill="{color}" text-anchor="middle">{badge}</text>
  </g>""")

    sections_rendered = "\n".join(cards_svg)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 780 520" width="780" height="520">
  <rect width="780" height="520" fill="#0f172a"/>
  <rect x="20" y="20" width="740" height="480" rx="12" fill="#0b1120" stroke="#334155" stroke-width="1.5"/>

  <!-- Dossier Header -->
  <text x="40" y="55" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#38bdf8" font-weight="700" letter-spacing="1">SeeDL / MULTI-COMPONENT SCIENTIFIC DOSSIER</text>
  <text x="40" y="85" font-family="system-ui, -apple-system, sans-serif" font-size="18" fill="#f8fafc" font-weight="700">{title}</text>
  <text x="40" y="110" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#94a3b8">{len(sections)} Report Sections · {author} · Self-Contained Offline Report</text>

  <!-- Report Section Cards -->
{sections_rendered}

  <!-- Footer dossier metadata -->
  <text x="40" y="475" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#64748b">Structure: Standalone HTML with sidebar TOC navigation, embedded interactive figures, and telemetry dossiers.</text>
</svg>
"""


def main() -> None:
    if sd.__version__ != "0.6.0":
        raise RuntimeError(f"Build these demos with SeeDL 0.6.0, found {sd.__version__}.")
    assert "site-packages" in str(Path(sd.__file__).resolve()), (
        f"SeeDL must be imported from installed package, got {sd.__file__}"
    )

    (ROOT / "demos").mkdir(exist_ok=True)
    (ROOT / "assets" / "plotly.min.js").write_text(plotly.offline.get_plotlyjs())
    if (ROOT.parent / "docs/visuals/network-3d.jpg").exists():
        shutil.copyfile(ROOT.parent / "docs/visuals/network-3d.jpg", ROOT / "assets/network.jpg")
    (ROOT / "assets/plotly-LICENSE.txt").write_text(
        "Plotly.js is distributed under the MIT license.\n"
        "The copyright and license notice are included in plotly.min.js.\n"
        "https://github.com/plotly/plotly.js/blob/master/LICENSE\n"
    )

    manifest: dict[str, object] = {
        "seedl": sd.__version__,
        "torch": torch.__version__,
        "plotly": plotly.__version__,
        "qiskit": qiskit.__version__,
        "seed": 20,
    }

    # =========================================================================
    # 1. 3D Neural Network (MLP) [Preserved 0.4.0]
    # =========================================================================
    torch.manual_seed(20)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 2),
    ).eval()
    sample = torch.tensor([[0.8, -0.4, 0.6]])
    neural = sd.see(model, input=sample, view="3d", theme="scientific", backend="plotly")
    export(neural, "network", network=True)

    manifest["network"] = {
        "nodes": len(neural.data["neuron_graph"]["nodes"]),
        "edges": len(neural.data["neuron_graph"]["edges"]),
        "input": sample.tolist(),
        "output": model(sample).detach().tolist(),
        "model": str(model),
    }

    # =========================================================================
    # 2. Bloch Spheres (6 presets) [Preserved 0.4.0]
    # =========================================================================
    expected = {
        "0": (0, 0, 1),
        "1": (0, 0, -1),
        "+": (1, 0, 0),
        "-": (-1, 0, 0),
        "+i": (0, 1, 0),
        "-i": (0, -1, 0),
    }
    manifest["bloch"] = {}
    for name, slug in [
        ("0", "zero"),
        ("1", "one"),
        ("+", "plus"),
        ("-", "minus"),
        ("+i", "plus-i"),
        ("-i", "minus-i"),
    ]:
        view = sd.bloch(name, backend="plotly")
        vector = [view.data[k] for k in ("x", "y", "z")]
        assert all(
            math.isclose(a, b, abs_tol=1e-12) for a, b in zip(vector, expected[name], strict=True)
        )
        export(view, f"bloch-{slug}")
        if name == "+":
            (ROOT / "assets/bloch.svg").write_text(view.to_figure("svg"))
        manifest["bloch"][slug] = {
            "state": name,
            "vector": vector,
            "prob_0": view.data["prob_0"],
            "prob_1": view.data["prob_1"],
        }

    # =========================================================================
    # 3. Quantum Circuit & Entanglement [Preserved 0.4.0]
    # =========================================================================
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    circuit = sd.see(qc, backend="plotly", theme="scientific")
    export(circuit, "bell")
    (ROOT / "assets/bell.svg").write_text(circuit.to_figure("svg"))
    manifest["circuit"] = circuit.data
    bell = [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)]
    ent = sd.entanglement(bell, backend="svg")
    manifest["bell"] = {"amplitudes": bell, "entanglement": ent.data}

    # =========================================================================
    # 4. Training Playback & Comparison [Preserved 0.4.0]
    # =========================================================================
    run = training_example()
    export_playback(run.playback(backend="plotly"), "training")
    comparison = sd.compare(
        run.snapshots[0], run.snapshots[-1], names=("Step 0", "Step 15"), backend="plotly"
    )
    comparison.figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font_color="#edf3fa",
    )
    export(comparison, "comparison")
    manifest["training"] = {
        "seed": 41,
        "device": "cpu",
        "input": [[1.0, -2.0]],
        "target": 0.5,
        "capture_boundary": "after backward, before optimizer.step",
        "stats": run.recording_stats,
        "snapshots": [snapshot.data for snapshot in run.snapshots],
        "comparison": comparison.data,
    }

    # =========================================================================
    # 5. Quantum State Evolution [Preserved 0.4.0]
    # =========================================================================
    evolution = sd.circuit(qc, simulate=True, backend="plotly", theme="scientific")
    verify_evolution(evolution, qc)
    export_playback(evolution, "evolution")
    phased = qc.copy()
    phased.rz(math.pi / 2, 1)
    phase_view = sd.circuit(phased, simulate=True, backend="plotly", theme="scientific")
    verify_evolution(phase_view, phased)
    export_playback(phase_view, "evolution-phase")
    manifest["evolution"] = {"bell": evolution.data, "phase": phase_view.data}

    # =========================================================================
    # 6. CNN Feature Maps Explorer [Preserved 0.4.0]
    # =========================================================================
    torch.manual_seed(42)
    cnn_model = DemoCNN().eval()
    cnn_sample = torch.randn(1, 3, 24, 24)
    cnn_view = sd.features(cnn_model, input=cnn_sample)
    cnn_view.save(ROOT / "demos/cnn-features.html")
    (ROOT / "assets/cnn-features.svg").write_text(cnn_view.to_figure("svg"))

    manifest["cnn"] = {
        "seed": 42,
        "input_shape": list(cnn_sample.shape),
        "captured_layers": len(cnn_view.data["layers"]),
        "layers": [
            {
                "name": layer["name"],
                "type": layer["type"],
                "shape": list(layer["original_shape"]),
                "channels": len(layer["channels"]),
                "downsampled": layer["downsampled"],
                "stats": layer["layer_stats"],
            }
            for layer in cnn_view.data["layers"]
        ],
    }

    # =========================================================================
    # 7. Transformer Attention Explorer [Preserved 0.4.0]
    # =========================================================================
    torch.manual_seed(42)
    tokens = ["The", "neural", "network", "attends", "to", "these", "tokens"]
    mha = torch.nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
    attn_x = torch.randn(1, len(tokens), 16)
    _, attn_weights = mha(attn_x, attn_x, attn_x, need_weights=True, average_attn_weights=False)
    attn_view = sd.attention(attn_weights, tokens=tokens)
    attn_view.save(ROOT / "demos/attention.html")
    (ROOT / "assets/attention.svg").write_text(attn_view.to_figure("svg"))

    manifest["attention"] = {
        "seed": 42,
        "tokens": tokens,
        "num_heads": 4,
        "weights_shape": list(attn_weights.shape),
        "mean_entropy": attn_view.data["mean_entropy"],
        "is_causal": attn_view.data["is_causal"],
        "is_softmax": attn_view.data["is_softmax"],
        "head_stats": attn_view.data["head_stats"],
    }

    # =========================================================================
    # 8. Multi-Head Attention Comparisons [Preserved 0.4.0]
    # =========================================================================
    cmp_pairs = [(0, 1), (0, 2), (1, 2), (2, 3)]
    manifest["attention_comparison"] = {"pairs": {}}
    for ha, hb in cmp_pairs:
        cmp_v = sd.attention(attn_weights, tokens=tokens, compare_heads=(ha, hb), backend="plotly")
        slug = "attention-comparison" if (ha, hb) == (0, 1) else f"attention-comparison-{ha}-{hb}"
        cmp_v.figure.update_layout(margin=dict(l=85, r=100, t=110, b=85))
        cmp_v.figure.update_traces(
            colorbar=dict(x=1.04, thickness=14), selector=dict(showscale=True)
        )
        cmp_v.figure.data[-1].colorbar.update(x=1.04, thickness=14)
        cmp_v.figure.update_yaxes(automargin=True)
        export(cmp_v, slug)
        h_cmp = cmp_v.data["head_comparison"]
        manifest["attention_comparison"]["pairs"][f"{ha}-{hb}"] = {
            "head_a": h_cmp["head_a"],
            "head_b": h_cmp["head_b"],
            "mae": h_cmp["mae"],
            "frobenius_norm": h_cmp["frobenius_norm"],
            "cosine_similarity": h_cmp["cosine_similarity"],
            "max_abs_diff": h_cmp["max_abs_diff"],
        }

    # =========================================================================
    # 9. Vision Transformer Patch Attention [Preserved 0.4.0]
    # =========================================================================
    torch.manual_seed(42)
    mha_vit = torch.nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
    vit_q = torch.randn(1, 17, 16)
    _, vit_weights = mha_vit(vit_q, vit_q, vit_q, need_weights=True, average_attn_weights=False)
    vit_view = sd.vit_attention(vit_weights, patch_grid=(4, 4), from_token=0, has_cls=True)
    vit_view.save(ROOT / "demos/vit-attention.html")
    (ROOT / "assets/vit-attention.svg").write_text(vit_view.to_figure("svg"))

    manifest["vit"] = {
        "seed": 42,
        "patch_grid": [4, 4],
        "from_token": 0,
        "has_cls": True,
        "attention_shape": list(vit_weights.shape),
        "stats": vit_view.data["layers"][0]["channels"][0]["stats"],
        "grid": vit_view.data["layers"][0]["channels"][0]["grid"],
    }

    # =========================================================================
    # 10. Neural Execution Tracer DAG (NEW in SeeDL 0.5.0)
    # =========================================================================
    torch.manual_seed(42)
    res_model = ResidualMLP().eval()
    res_x = torch.tensor([[0.5, -1.2, 0.8, 0.3]])
    trace_view = sd.flow(
        res_model,
        res_x,
        mode="graph",
        title="Neural Execution Tracer: Residual MLP",
    )
    trace_view.save(ROOT / "demos/trace.html")
    (ROOT / "assets/trace.svg").write_text(trace_view.to_figure("svg"))

    manifest["trace"] = {
        "seed": 42,
        "input_shape": list(res_x.shape),
        "total_nodes": len(trace_view.data["nodes"]),
        "total_edges": len(trace_view.data["edges"]),
        "total_parameters": trace_view.data["total_parameters"],
        "trace_mode": trace_view.data["trace_mode"],
        "nodes": [
            {
                "id": n["id"],
                "name": n["name"],
                "module_type": n["module_type"],
                "input_shapes": n["input_shapes"],
                "output_shapes": n["output_shapes"],
                "dtypes": n["dtypes"],
                "is_leaf": n["is_leaf"],
            }
            for n in trace_view.data["nodes"]
        ],
    }

    # =========================================================================
    # 11. Tensor Flow Explorer (NEW in SeeDL 0.5.0)
    # =========================================================================
    flow_view = sd.flow(
        cnn_model,
        cnn_sample,
        mode="table",
        title="Tensor Flow Explorer: CNN Feature Pipeline",
    )
    flow_view.save(ROOT / "demos/flow.html")
    (ROOT / "assets/flow.svg").write_text(flow_view.to_figure("svg"))

    manifest["flow"] = {
        "seed": 42,
        "input_shape": list(cnn_sample.shape),
        "total_nodes": len(flow_view.data["nodes"]),
        "nodes": [
            {
                "id": n["id"],
                "name": n["name"],
                "module_type": n["module_type"],
                "input_shapes": n["input_shapes"],
                "output_shapes": n["output_shapes"],
                "dtypes": n["dtypes"],
                "num_elements": n["num_elements"],
                "num_parameters": n["num_parameters"],
            }
            for n in flow_view.data["nodes"]
        ],
    }

    # =========================================================================
    # 12. Transformer Attention Rollout Lab (NEW in SeeDL 0.5.0)
    # =========================================================================
    torch.manual_seed(42)
    rollout_tokens = ["[CLS]", "The", "neural", "observatory", "attends", "to", "[SEP]"]
    rollout_layers = [
        torch.softmax(torch.randn(1, 4, len(rollout_tokens), len(rollout_tokens)), dim=-1)
        for _ in range(3)
    ]
    rollout_view = sd.attention_rollout(
        rollout_layers,
        tokens=rollout_tokens,
        include_residual=True,
        head_reduction="mean",
        title="Transformer Attention Rollout: 3 Layers",
    )
    rollout_view.save(ROOT / "demos/rollout.html")
    (ROOT / "assets/rollout.svg").write_text(rollout_view.to_figure("svg"))

    manifest["rollout"] = {
        "seed": 42,
        "tokens": rollout_tokens,
        "num_layers": rollout_view.data["num_layers"],
        "seq_len": rollout_view.data["seq_len"],
        "include_residual": rollout_view.data["include_residual"],
        "head_reduction": rollout_view.data["head_reduction"],
        "rollout_matrix": rollout_view.data["rollout_matrix"],
    }

    # =========================================================================
    # 13. Advanced Explainability Lab: Attribution Benchmark (NEW in SeeDL 0.5.0)
    # =========================================================================
    torch.manual_seed(42)
    attr_model = torch.nn.Sequential(
        torch.nn.Linear(6, 12),
        torch.nn.Tanh(),
        torch.nn.Linear(12, 2),
    ).eval()
    attr_x = torch.tensor([[0.7, -0.5, 1.2, -0.3, 0.4, -0.9]])
    attr_view = sd.compare_attributions(
        attr_model,
        attr_x,
        target=0,
        methods=("saliency", "integrated_gradients", "smoothgrad"),
        steps=10,
        samples=10,
        seed=42,
    )
    attr_view.save(ROOT / "demos/attribution.html")
    (ROOT / "assets/attribution.svg").write_text(attr_view.to_figure("svg"))

    manifest["attribution"] = {
        "seed": 42,
        "input_shape": list(attr_x.shape),
        "target_class": 0,
        "methods": [
            {
                "method": m["method"],
                "min": m["min"],
                "max": m["max"],
                "mean": m["mean"],
                "sparsity": m["sparsity"],
            }
            for m in attr_view.data["methods"]
        ],
        "similarity_matrix": attr_view.data["similarity_matrix"],
    }

    # =========================================================================
    # 14. Gradient Health Dashboard (NEW in SeeDL 0.5.0)
    # =========================================================================
    torch.manual_seed(42)
    grad_model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2),
    )
    grad_x = torch.randn(2, 4)
    grad_y = torch.tensor([0, 1])
    grad_out = grad_model(grad_x)
    grad_loss = torch.nn.functional.cross_entropy(grad_out, grad_y)
    grad_loss.backward()
    grad_view = sd.gradient_health(grad_model, title="Gradient Diagnostics: 2-Layer Perceptron")
    grad_view.save(ROOT / "demos/gradient-health.html")
    (ROOT / "assets/gradient-health.svg").write_text(grad_view.to_figure("svg"))

    manifest["gradient_health"] = {
        "seed": 42,
        "global_l1": grad_view.data["global_l1"],
        "global_l2": grad_view.data["global_l2"],
        "global_zero_fraction": grad_view.data["global_zero_fraction"],
        "total_nonfinite": grad_view.data["total_nonfinite"],
        "total_parameters_inspected": grad_view.data["total_parameters_inspected"],
        "parameters": [
            {
                "name": p["name"],
                "shape": p["shape"],
                "l1_norm": p["l1_norm"],
                "l2_norm": p["l2_norm"],
                "zero_fraction": p["zero_fraction"],
                "nonfinite_count": p["nonfinite_count"],
            }
            for p in grad_view.data["parameters"]
        ],
    }

    # =========================================================================
    # 15. Quantum Noise Lab (NEW in SeeDL 0.5.0)
    # =========================================================================
    noise_channels = [
        ("depolarizing", "quantum-noise"),
        ("amplitude_damping", "quantum-noise-amplitude"),
        ("phase_damping", "quantum-noise-phase"),
    ]
    manifest["quantum_noise"] = {"channels": {}}
    for ch, slug in noise_channels:
        q_noise_view = sd.quantum_noise(
            channel=ch,
            param=0.4,
            steps=8,
            title=f"Quantum Noise Explorer: {ch.replace('_', ' ').title()} Channel",
        )
        q_noise_view.save(ROOT / f"demos/{slug}.html")
        if ch == "depolarizing":
            (ROOT / "assets/quantum-noise.svg").write_text(q_noise_view.to_figure("svg"))
        manifest["quantum_noise"]["channels"][ch] = {
            "channel": ch,
            "param": 0.4,
            "steps": 8,
            "trajectory_len": len(q_noise_view.data["trajectory"]),
            "initial_purity": q_noise_view.data["trajectory"][0]["purity"],
            "final_purity": q_noise_view.data["trajectory"][-1]["purity"],
            "initial_entropy": q_noise_view.data["trajectory"][0]["entropy"],
            "final_entropy": q_noise_view.data["trajectory"][-1]["entropy"],
        }

    # =========================================================================
    # 16. Dataset Drift Observatory (NEW in SeeDL 0.5.0)
    # =========================================================================
    np.random.seed(42)
    ref_data = np.random.normal(0.0, 1.0, size=(200, 3))
    comp_data = np.random.normal(0.4, 1.2, size=(200, 3))
    drift_view = sd.dataset_drift(
        ref_data,
        comp_data,
        title="Dataset Distribution Drift Analysis: Reference vs Production Distribution",
    )
    drift_view.save(ROOT / "demos/dataset-drift.html")
    (ROOT / "assets/dataset-drift.svg").write_text(drift_view.to_figure("svg"))

    manifest["dataset_drift"] = {
        "seed": 42,
        "psi": drift_view.data["psi"],
        "jsd": drift_view.data["jsd"],
        "drift_level": drift_view.data["drift_level"],
        "status_color": drift_view.data["status_color"],
        "ref_samples": drift_view.data["reference"]["samples"],
        "comp_samples": drift_view.data["comparison"]["samples"],
    }

    # =========================================================================
    # 17. Scientific Diagnostic Report (NEW in SeeDL 0.5.0)
    # =========================================================================
    sci_report = sd.report(
        title="Neural Observatory Multi-Component Diagnostic Report",
        subtitle="End-to-End Scientific Architecture Assessment",
        abstract=(
            "Comprehensive diagnostic audit covering execution flow, attention rollout, "
            "gradient health, quantum noise, and dataset shift."
        ),
        author="SeeDL Scientific Verification Team",
        affiliation="SeeDL Neural Observatory",
        theme="dark",
    )
    sci_report.add_section(
        "Execution Trace & Tensor Flow",
        trace_view,
        narrative="Observed tensor transitions across linear and activation layers in residual MLP.",
    )
    sci_report.add_section(
        "Transformer Attention Rollout",
        rollout_view,
        narrative="Cumulative attention flow across 3 transformer layers with residual connections.",
    )
    sci_report.add_section(
        "Gradient Norms & Health",
        grad_view,
        narrative="Layer-level gradient L1/L2 norms and numerical stability diagnostics.",
    )
    sci_report.add_section(
        "Open Quantum System Noise",
        sd.quantum_noise(channel="depolarizing", param=0.3, steps=6),
        narrative="Kraus channel depolarizing noise decoherence trajectory.",
    )
    sci_report.add_section(
        "Dataset Distribution Drift",
        drift_view,
        narrative="Population Stability Index and Jensen-Shannon Divergence against reference distribution.",
    )
    sci_report.save(ROOT / "demos/report.html")
    (ROOT / "assets/report.svg").write_text(build_report_preview_svg(sci_report))

    manifest["report"] = {
        "title": sci_report.title,
        "author": sci_report.author,
        "sections": len(sci_report.sections),
    }

    # =========================================================================
    # 18. Training Run Trajectory Comparison (NEW in SeeDL 0.5.0)
    # =========================================================================
    # Authentic deterministic CPU training comparing SGD and AdamW on a small
    # synthetic demonstration dataset. Identical weights, loss, and batch order.
    # Note: relative optimizer performance depends on architecture and problem;
    # neither optimizer is universally superior.
    torch.manual_seed(42)
    X_train = torch.randn(32, 4)
    y_train = ((X_train[:, 0] * 1.5 - X_train[:, 1] * 0.8 + X_train[:, 2] * 0.5) > 0.0).long()

    def make_run_model():
        torch.manual_seed(100)
        return torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2),
        )

    model_sgd = make_run_model()
    model_adamw = make_run_model()
    model_adamw.load_state_dict(model_sgd.state_dict())

    criterion_run = torch.nn.CrossEntropyLoss()
    opt_sgd = torch.optim.SGD(model_sgd.parameters(), lr=0.1)
    opt_adamw = torch.optim.AdamW(model_adamw.parameters(), lr=0.05, weight_decay=1e-4)

    run_sgd: dict[str, list[float]] = {"loss": [], "accuracy": []}
    run_adamw: dict[str, list[float]] = {"loss": [], "accuracy": []}

    num_training_steps = 4
    for _ in range(num_training_steps):
        # Step SGD
        opt_sgd.zero_grad()
        out_sgd = model_sgd(X_train)
        loss_sgd = criterion_run(out_sgd, y_train)
        loss_sgd.backward()
        opt_sgd.step()
        with torch.no_grad():
            preds_sgd = model_sgd(X_train).argmax(dim=-1)
            acc_sgd = float((preds_sgd == y_train).float().mean().item())
            l_val_sgd = float(loss_sgd.item())
            assert math.isfinite(l_val_sgd) and math.isfinite(acc_sgd)
            run_sgd["loss"].append(round(l_val_sgd, 4))
            run_sgd["accuracy"].append(round(acc_sgd, 4))

        # Step AdamW
        opt_adamw.zero_grad()
        out_adamw = model_adamw(X_train)
        loss_adamw = criterion_run(out_adamw, y_train)
        loss_adamw.backward()
        opt_adamw.step()
        with torch.no_grad():
            preds_adamw = model_adamw(X_train).argmax(dim=-1)
            acc_adamw = float((preds_adamw == y_train).float().mean().item())
            l_val_adamw = float(loss_adamw.item())
            assert math.isfinite(l_val_adamw) and math.isfinite(acc_adamw)
            run_adamw["loss"].append(round(l_val_adamw, 4))
            run_adamw["accuracy"].append(round(acc_adamw, 4))

    runs_view = sd.compare_runs(
        run_sgd,
        run_adamw,
        name_a="Baseline SGD",
        name_b="AdamW Optimizer",
        title="Training Run Comparison: Baseline SGD vs AdamW Optimizer",
    )
    runs_view.save(ROOT / "demos/runs-compare.html")
    (ROOT / "assets/runs-compare.svg").write_text(runs_view.to_figure("svg"))

    manifest["runs_compare"] = {
        "name_a": runs_view.data["name_a"],
        "name_b": runs_view.data["name_b"],
        "metrics": runs_view.data["metrics"],
        "provenance": {
            "dataset": "Synthetic binary classification (32 samples, 4 features)",
            "dataset_seed": 42,
            "model_seed": 100,
            "architecture": "Sequential(Linear(4, 8), ReLU(), Linear(8, 2))",
            "loss_fn": "CrossEntropyLoss",
            "steps": num_training_steps,
            "optimizer_a": {"name": "SGD", "lr": 0.1},
            "optimizer_b": {"name": "AdamW", "lr": 0.05, "weight_decay": 1e-4},
            "measurement_timing": "Step loss recorded during backward; accuracy evaluated post-step on synthetic batch",
            "verified_finite": True,
            "demonstration_type": "Small synthetic demonstration; relative performance depends on problem and hyperparameters",
        },
    }

    # =========================================================================
    # Apply Presentation Fixes & Link CSS
    # =========================================================================
    native_stems = {
        "cnn-features",
        "attention",
        "vit-attention",
        "trace",
        "flow",
        "rollout",
        "attribution",
        "gradient-health",
        "quantum-noise",
        "quantum-noise-amplitude",
        "quantum-noise-phase",
        "dataset-drift",
        "runs-compare",
    }
    for path in (ROOT / "demos").glob("*.html"):
        document = path.read_text().replace("\u2014", ": ")
        if path.stem in native_stems:
            if '<link rel="stylesheet" href="../assets/viewer.css">' not in document:
                document = document.replace(
                    "</head>",
                    '<link rel="stylesheet" href="../assets/viewer.css"></head>',
                )
        path.write_text(document)

    # Hash all generated HTML demos
    manifest["sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "demos").glob("*.html"))
    }
    (ROOT / "demos/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (ROOT / ".nojekyll").touch()
    print(f"Built {len(manifest['sha256'])} authentic SeeDL 0.6.0 demos; one shared Plotly bundle.")


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        main()
    print(
        "Built authentic SeeDL 0.6.0 demonstrations, training snapshots, and shared Plotly bundle."
    )
