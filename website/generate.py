"""Build deterministic SeeDL 0.2.0 demos; no server or CDN is needed at runtime.

Run from the repository root: .venv/bin/python website/generate.py
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import shutil
from pathlib import Path

import plotly
import plotly.offline
import torch
from qiskit import QuantumCircuit

import seedl as sd
from seedl._network_renderers import build_network_interactive_js

ROOT = Path(__file__).resolve().parent


def export(view, name: str, *, network: bool = False) -> None:
    """Keep SeeDL's own interaction bridge and share one local Plotly runtime."""
    fig = view.figure
    fig.update_layout(autosize=True, height=None, width=None)
    post_script = None
    if network:
        post_script = build_network_interactive_js(view.data["neuron_graph"], True)
    else:
        # Preset-only Bloch demos: gate shortcuts change the arrow without all
        # the associated annotations in 0.2.0. Keep each export self-consistent.
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
        f"<title>{view.kind} · SeeDL</title>"
        "<style>html,body{margin:0;background:#0f172a;height:100%;}"
        ".plotly-graph-div{width:100%!important;height:100%!important;min-height:560px;}"
        "body>div{height:100%}*:focus-visible{outline:2px solid #00e5ff}</style>"
        f"</head><body>{body}</body></html>"
    )
    (ROOT / "demos" / f"{name}.html").write_text(document)


def main() -> None:
    if sd.__version__ != "0.2.0":
        raise RuntimeError("Build these demos with the SeeDL 0.2.0 checkout.")
    (ROOT / "demos").mkdir(exist_ok=True)
    (ROOT / "assets" / "plotly.min.js").write_text(plotly.offline.get_plotlyjs())
    shutil.copyfile(ROOT.parent / "docs/visuals/network-3d.jpg", ROOT / "assets/network.jpg")
    (ROOT / "assets/plotly-LICENSE.txt").write_text(
        "Plotly.js is distributed under the MIT license.\n"
        "The copyright and license notice are included in plotly.min.js.\n"
        "https://github.com/plotly/plotly.js/blob/master/LICENSE\n"
    )
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
    manifest = {
        "seedl": sd.__version__,
        "torch": torch.__version__,
        "plotly": plotly.__version__,
        "seed": 20,
        "input": sample.tolist(),
        "output": model(sample).detach().tolist(),
        "model": str(model),
        "bloch": {},
        "network": {
            "nodes": len(neural.data["neuron_graph"]["nodes"]),
            "edges": len(neural.data["neuron_graph"]["edges"]),
        },
    }
    expected = {
        "0": (0, 0, 1),
        "1": (0, 0, -1),
        "+": (1, 0, 0),
        "-": (-1, 0, 0),
        "+i": (0, 1, 0),
        "-i": (0, -1, 0),
    }
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
    manifest["sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "demos").glob("*.html"))
    }
    (ROOT / "demos/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (ROOT / ".nojekyll").touch()
    print(f"Built {len(manifest['sha256'])} authentic SeeDL demos; one shared Plotly bundle.")


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        main()
    print("Built 8 SeeDL demos, two SVG previews, and one shared local Plotly bundle.")
