"""Check built assets, internal links, science provenance, and Python doc snippets."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import torch

ROOT = Path(__file__).resolve().parent


class Page(HTMLParser):
    def __init__(self, text: str):
        super().__init__()
        self.links = []
        self.ids = set()
        self.snippets = []
        self.in_code = False
        self.code = ""
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for attr in ("src", "href"):
            if attrs.get(attr):
                self.links.append(attrs[attr])
        if tag == "pre":
            self.in_code = True
            self.code = ""

    def handle_endtag(self, tag):
        if tag == "pre":
            self.in_code = False
            self.snippets.append(self.code)

    def handle_data(self, data):
        if self.in_code:
            self.code += data


def main() -> None:
    paths = [ROOT / "index.html", ROOT / "playground.html", ROOT / "docs/index.html"]
    pages = {path: Page(path.read_text()) for path in paths}
    checked = 0
    for path, page in pages.items():
        for link in page.links:
            parts = urlsplit(link)
            if parts.scheme or parts.netloc:
                continue
            assert not parts.path.startswith("/"), f"Root-relative link breaks /seedl/: {link}"
            target = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            assert target.is_relative_to(ROOT), f"Link escapes site: {link}"
            assert target.is_file(), f"Missing asset: {path.name}: {link}"
            if parts.fragment:
                assert target in pages and parts.fragment in pages[target].ids, (
                    f"Missing anchor in {path.name}: {link}"
                )
            checked += 1

    manifest = json.loads((ROOT / "demos/manifest.json").read_text())
    assert manifest["seedl"] == "0.5.0", f"Expected manifest version 0.5.0, got {manifest['seedl']}"
    demo_files = list((ROOT / "demos").glob("*.html"))
    assert len(manifest["sha256"]) == len(demo_files)
    assert len(manifest["sha256"]) >= 30, (
        f"Expected at least 30 demos, got {len(manifest['sha256'])}"
    )

    for path in paths:
        text = path.read_text()
        assert 'class="version">0.5.0' in text, f"Outdated version badge in {path.name}"
        # Current API links use 0.5.0; historical release notes links are preserved
        assert "blob/v0.4.0/src/" not in text, f"Stale 0.4.0 API link in {path.name}"
        assert "blob/v0.3.0/src/" not in text, f"Stale 0.3.0 API link in {path.name}"
        assert "blob/v0.2.0/src/" not in text, f"Stale 0.2.0 API link in {path.name}"

    # 1. Preserved 0.3.0 Training Provenance
    training = manifest["training"]
    snapshots = training["snapshots"]
    assert len(snapshots) == 16 and training["stats"]["dropped"] == 0
    assert training["device"] == "cpu"
    assert [s["step"] for s in snapshots] == list(range(16))
    assert snapshots[-1]["loss"] < snapshots[0]["loss"]
    assert all(s["graph"] and s["complete"] for s in snapshots)
    assert all(p["gradients"] is not None for s in snapshots for p in s["parameters"].values())
    assert training["capture_boundary"] == "after backward, before optimizer.step"
    assert training["comparison"]["steps"] == [0, 15]
    for row in training["comparison"]["parameters"]:
        assert row["status"] == "aligned"
        assert row["delta"] == [b - a for a, b in zip(row["before"], row["after"], strict=True)]

    # 2. Preserved 0.3.0 Quantum Provenance
    from types import SimpleNamespace

    from generate import verify_evolution
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    verify_evolution(SimpleNamespace(data=manifest["evolution"]["bell"]), qc)
    qc.rz(math.pi / 2, 1)
    verify_evolution(SimpleNamespace(data=manifest["evolution"]["phase"]), qc)

    assert len(manifest["bloch"]) == 6
    for slug, data in manifest["bloch"].items():
        assert math.isclose(sum(n * n for n in data["vector"]), 1, abs_tol=1e-12)
        assert math.isclose(data["prob_0"] + data["prob_1"], 1, abs_tol=1e-12)
        assert (ROOT / f"demos/bloch-{slug}.html").is_file()

    assert [i["name"] for i in manifest["circuit"]["instructions"]] == ["h", "cx"]
    assert manifest["circuit"]["instructions"][1]["qubits"] == [0, 1]
    assert all(
        math.isclose(a, b, abs_tol=1e-12)
        for a, b in zip(
            manifest["bell"]["entanglement"]["probabilities"], [0.5, 0.0, 0.0, 0.5], strict=True
        )
    )
    assert math.isclose(manifest["bell"]["entanglement"]["entropy_q0"], 1, abs_tol=1e-12)

    # 3. Preserved 0.4.0 CNN Provenance
    cnn = manifest["cnn"]
    assert cnn["seed"] == 42
    assert cnn["captured_layers"] == 5
    assert cnn["layers"][0]["name"] == "conv1"
    assert cnn["layers"][0]["shape"] == [1, 8, 24, 24]
    assert cnn["layers"][2]["name"] == "pool1"
    assert cnn["layers"][2]["shape"] == [1, 8, 12, 12]
    assert cnn["layers"][2]["downsampled"] is False
    assert (ROOT / "demos/cnn-features.html").is_file()
    assert (ROOT / "assets/cnn-features.svg").is_file()

    # 4. Preserved 0.4.0 Transformer Attention Provenance
    attn = manifest["attention"]
    assert attn["seed"] == 42
    assert attn["num_heads"] == 4
    assert len(attn["tokens"]) == 7
    assert attn["mean_entropy"] > 0
    assert attn["is_softmax"] is True
    assert (ROOT / "demos/attention.html").is_file()
    assert (ROOT / "assets/attention.svg").is_file()

    # 5. Preserved 0.4.0 Attention Comparison Provenance
    cmp_pair = manifest["attention_comparison"]["pairs"]["0-1"]
    assert cmp_pair["head_a"] == 0 and cmp_pair["head_b"] == 1
    assert cmp_pair["mae"] > 0
    assert cmp_pair["frobenius_norm"] > 0
    assert 0 <= cmp_pair["cosine_similarity"] <= 1.0
    assert (ROOT / "demos/attention-comparison.html").is_file()

    # 6. Preserved 0.4.0 ViT Patch Attention Provenance
    vit = manifest["vit"]
    assert vit["seed"] == 42
    assert vit["patch_grid"] == [4, 4]
    assert vit["from_token"] == 0
    assert vit["has_cls"] is True
    assert len(vit["grid"]) == 4 and len(vit["grid"][0]) == 4
    assert (ROOT / "demos/vit-attention.html").is_file()
    assert (ROOT / "assets/vit-attention.svg").is_file()

    # 7. NEW in 0.5.0: Neural Execution Tracer DAG Provenance
    trace = manifest["trace"]
    assert trace["seed"] == 42
    assert trace["total_nodes"] >= 5
    assert trace["total_edges"] >= 6
    assert trace["total_parameters"] > 0
    assert (ROOT / "demos/trace.html").is_file()
    assert (ROOT / "assets/trace.svg").is_file()

    # 8. NEW in 0.5.0: Tensor Flow Explorer Provenance
    flow = manifest["flow"]
    assert flow["seed"] == 42
    assert flow["total_nodes"] >= 5
    assert (ROOT / "demos/flow.html").is_file()
    assert (ROOT / "assets/flow.svg").is_file()

    # 9. NEW in 0.5.0: Transformer Attention Rollout Provenance
    rollout = manifest["rollout"]
    assert rollout["seed"] == 42
    assert rollout["num_layers"] == 3
    assert rollout["seq_len"] == 7
    assert len(rollout["rollout_matrix"]) == 7
    assert (ROOT / "demos/rollout.html").is_file()
    assert (ROOT / "assets/rollout.svg").is_file()

    # 10. NEW in 0.5.0: Attribution Benchmark Provenance
    attr = manifest["attribution"]
    assert attr["seed"] == 42
    assert len(attr["methods"]) == 3
    sim = attr["similarity_matrix"]
    assert "saliency" in sim and "integrated_gradients" in sim and "smoothgrad" in sim
    assert (ROOT / "demos/attribution.html").is_file()
    assert (ROOT / "assets/attribution.svg").is_file()

    # 11. NEW in 0.5.0: Gradient Health Dashboard Provenance
    grad = manifest["gradient_health"]
    assert grad["seed"] == 42
    assert grad["global_l2"] > 0
    assert grad["total_nonfinite"] == 0
    assert len(grad["parameters"]) >= 4
    assert (ROOT / "demos/gradient-health.html").is_file()
    assert (ROOT / "assets/gradient-health.svg").is_file()

    # 12. NEW in 0.5.0: Quantum Noise Lab Provenance
    qnoise = manifest["quantum_noise"]
    assert len(qnoise["channels"]) == 3
    depol = qnoise["channels"]["depolarizing"]
    assert depol["final_purity"] < depol["initial_purity"]
    assert depol["final_entropy"] > depol["initial_entropy"]
    assert (ROOT / "demos/quantum-noise.html").is_file()
    assert (ROOT / "demos/quantum-noise-amplitude.html").is_file()
    assert (ROOT / "demos/quantum-noise-phase.html").is_file()
    assert (ROOT / "assets/quantum-noise.svg").is_file()

    # 13. NEW in 0.5.0: Dataset Drift Observatory Provenance
    drift = manifest["dataset_drift"]
    assert drift["seed"] == 42
    assert drift["psi"] > 0
    assert drift["jsd"] > 0
    assert (ROOT / "demos/dataset-drift.html").is_file()
    assert (ROOT / "assets/dataset-drift.svg").is_file()

    # 14. NEW in 0.5.0: Scientific Diagnostic Report Provenance
    rep = manifest["report"]
    assert rep["sections"] >= 5
    assert (ROOT / "demos/report.html").is_file()
    report_svg_path = ROOT / "assets/report.svg"
    drift_svg_path = ROOT / "assets/dataset-drift.svg"
    assert report_svg_path.is_file() and drift_svg_path.is_file()
    report_svg = report_svg_path.read_text(encoding="utf-8")
    drift_svg = drift_svg_path.read_text(encoding="utf-8")
    assert report_svg != drift_svg, "report.svg must not be identical to dataset-drift.svg"
    assert (
        "Execution Trace &amp; Tensor Flow" in report_svg
        or "Execution Trace & Tensor Flow" in report_svg
    )
    assert "Transformer Attention Rollout" in report_svg
    assert "Gradient Norms &amp; Health" in report_svg or "Gradient Norms & Health" in report_svg
    assert "Open Quantum System Noise" in report_svg
    assert "Dataset Distribution Drift" in report_svg
    assert "MULTI-COMPONENT SCIENTIFIC DOSSIER" in report_svg

    # 15. NEW in 0.5.0: Training Run Comparison Provenance
    runs = manifest["runs_compare"]
    assert len(runs["metrics"]) >= 2
    assert (ROOT / "demos/runs-compare.html").is_file()
    assert (ROOT / "assets/runs-compare.svg").is_file()
    prov = runs.get("provenance", {})
    assert prov.get("verified_finite") is True
    assert prov.get("dataset_seed") == 42
    assert prov.get("model_seed") == 100
    assert prov.get("steps") == 4

    # Verify that metrics originate from actual deterministic training routine
    torch.manual_seed(42)
    X_chk = torch.randn(32, 4)
    y_chk = ((X_chk[:, 0] * 1.5 - X_chk[:, 1] * 0.8 + X_chk[:, 2] * 0.5) > 0.0).long()

    def make_chk_model():
        torch.manual_seed(100)
        return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))

    m_sgd, m_adamw = make_chk_model(), make_chk_model()
    m_adamw.load_state_dict(m_sgd.state_dict())
    crit_chk = torch.nn.CrossEntropyLoss()
    o_sgd = torch.optim.SGD(m_sgd.parameters(), lr=0.1)
    o_adamw = torch.optim.AdamW(m_adamw.parameters(), lr=0.05, weight_decay=1e-4)

    expected_sgd: dict[str, list[float]] = {"loss": [], "accuracy": []}
    expected_adamw: dict[str, list[float]] = {"loss": [], "accuracy": []}
    for _ in range(4):
        o_sgd.zero_grad()
        l_s = crit_chk(m_sgd(X_chk), y_chk)
        l_s.backward()
        o_sgd.step()
        with torch.no_grad():
            expected_sgd["loss"].append(round(float(l_s.item()), 4))
            expected_sgd["accuracy"].append(
                round(float((m_sgd(X_chk).argmax(-1) == y_chk).float().mean().item()), 4)
            )

        o_adamw.zero_grad()
        l_a = crit_chk(m_adamw(X_chk), y_chk)
        l_a.backward()
        o_adamw.step()
        with torch.no_grad():
            expected_adamw["loss"].append(round(float(l_a.item()), 4))
            expected_adamw["accuracy"].append(
                round(float((m_adamw(X_chk).argmax(-1) == y_chk).float().mean().item()), 4)
            )

    metric_map = {m["metric"]: m for m in runs["metrics"]}
    assert "loss" in metric_map and "accuracy" in metric_map
    assert metric_map["loss"]["history_a"] == expected_sgd["loss"]
    assert metric_map["loss"]["history_b"] == expected_adamw["loss"]
    assert metric_map["accuracy"]["history_a"] == expected_sgd["accuracy"]
    assert metric_map["accuracy"]["history_b"] == expected_adamw["accuracy"]
    for m in runs["metrics"]:
        for val in m["history_a"] + m["history_b"]:
            assert math.isfinite(val)

    # Check all demo files against manifest sha256 and size/CDN bounds
    for name, digest in manifest["sha256"].items():
        path = ROOT / "demos" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
        html = path.read_text()
        assert "cdn.plot.ly" not in html
        assert "cdn." not in html
        assert "http://" not in html and "https://" not in html
        assert path.stat().st_size < 1_000_000, f"Unbounded demo payload: {name}"

        # Plotly demos share the single local Plotly bundle
        if any(
            token in name
            for token in (
                "network",
                "bloch",
                "bell",
                "training",
                "comparison",
                "evolution",
                "attention-comparison",
            )
        ):
            assert html.count('src="../assets/plotly.min.js"') == 1, (
                f"Missing shared Plotly in {name}"
            )
            assert "plotly.js v" not in html, f"Duplicated inline Plotly runtime in {name}"

        for link in Page(html).links:
            parts = urlsplit(link)
            if not parts.scheme and not parts.netloc and parts.path:
                assert (path.parent / parts.path).is_file(), link
                checked += 1

    for name in ("training", "evolution", "evolution-phase"):
        assert "seedl-playback" in (ROOT / f"demos/{name}.html").read_text()

    assert sum(p.stat().st_size for p in paths) < 95_000, "Initial HTML has grown unexpectedly"

    # Execute visible Python examples in temp directory
    count = 0
    previous = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="seedl-docs-") as temp:
            os.chdir(temp)
            for path, page in pages.items():
                namespace = {"__name__": "__website_example__"}
                for snippet in page.snippets:
                    if not any(
                        token in snippet
                        for token in (
                            "import ",
                            "view = sd.see",
                            "comparison = sd.compare",
                            "cmp_view = sd.attention",
                            "sd.flow",
                            "sd.attention_rollout",
                            "sd.compare_attributions",
                            "sd.gradient_health",
                            "sd.quantum_noise",
                            "sd.dataset_drift",
                            "sd.report",
                            "sd.compare_runs",
                            "sd.save_analysis",
                        )
                    ):
                        continue
                    with contextlib.redirect_stdout(io.StringIO()):
                        exec(compile(snippet, str(path), "exec"), namespace)
                    count += 1
    finally:
        os.chdir(previous)

    print(
        f"PASS: {checked} local links/assets, {len(manifest['sha256'])} demo hashes, "
        f"16 captured passes, Bell/Bloch/phase science, CNN/Attention/ViT science, "
        f"Trace/Flow/Rollout/Attribution/Gradient/Noise/Drift/Report/Runs science, {count} Python examples."
    )


if __name__ == "__main__":
    main()
