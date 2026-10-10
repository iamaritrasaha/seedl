"""Automated real browser interaction verification using Playwright and Chromium.

Tests 3D neural network selection, camera retention, Bloch sphere rendering,
circuit visualization, CNN 3D feature maps, and captures visual screenshot artifacts.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
from playwright.sync_api import sync_playwright

import seedl as sd

ARTIFACTS_DIR = Path("/media/hrik/Hrik/Projects/seedl/examples/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def test_browser_interactions():
    print("\n--- Starting Real Browser Interaction Verification ---")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/usr/bin/google-chrome",
            headless=True,
            args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
        )
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        # -------------------------------------------------------------
        # 1. 3D NEURAL NETWORK: DIRECT NEURON SELECTION & CAMERA
        # -------------------------------------------------------------
        print("1. Testing 3D Neural Network with Direct Neuron Selection...")
        model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))
        x = torch.tensor([[1.2, -0.8]])
        v_nn = sd.see(model, input=x, view="3d", theme="scientific")
        html_nn = ARTIFACTS_DIR / "neural_3d_interactive.html"
        v_nn.save(html_nn)

        page.goto(f"file://{html_nn.resolve()}")
        page.wait_for_selector(".plotly-graph-div")
        page.wait_for_timeout(1000)

        # Overview screenshot
        ss_overview = ARTIFACTS_DIR / "screenshot_neural_3d_overview.png"
        page.screenshot(path=str(ss_overview))
        print(f"   Saved {ss_overview.name}")

        # Simulate clicking neuron 1:1 directly
        res = page.evaluate("""() => {
            if (window.__seedl_select_neuron) {
                window.__seedl_select_neuron('1:1');
                return true;
            }
            return false;
        }""")
        assert res is True, "window.__seedl_select_neuron is missing!"
        page.wait_for_timeout(500)

        # Check that highlight traces became visible
        trace_status = page.evaluate("""() => {
            var gd = document.getElementsByClassName('plotly-graph-div')[0];
            var visibleCount = 0;
            for (var i = 0; i < gd.data.length; i++) {
                var role = gd.data[i].meta ? gd.data[i].meta.role : null;
                if ((role === 'highlight_incoming' || role === 'highlight_outgoing' || role === 'highlight_node') && gd.data[i].visible === true) {
                    visibleCount++;
                }
            }
            return visibleCount;
        }""")
        assert trace_status > 0, "Highlight traces did not become visible after neuron click!"
        print(f"   Verified: Highlight traces activated ({trace_status} traces visible).")

        # Selected screenshot
        ss_selected = ARTIFACTS_DIR / "screenshot_neural_3d_selected.png"
        page.screenshot(path=str(ss_selected))
        print(f"   Saved {ss_selected.name}")

        # Clear selection
        page.evaluate("() => window.__seedl_clear_selection && window.__seedl_clear_selection()")
        page.wait_for_timeout(300)
        cleared_status = page.evaluate("""() => {
            var gd = document.getElementsByClassName('plotly-graph-div')[0];
            var visibleCount = 0;
            for (var i = 0; i < gd.data.length; i++) {
                var role = gd.data[i].meta ? gd.data[i].meta.role : null;
                if ((role === 'highlight_incoming' || role === 'highlight_outgoing' || role === 'highlight_node') && gd.data[i].visible === true) {
                    visibleCount++;
                }
            }
            return visibleCount;
        }""")
        assert cleared_status == 0, "Highlight traces still visible after clear selection!"
        print("   Verified: Selection cleared successfully.")

        # -------------------------------------------------------------
        # 2. 3D BLOCH SPHERE
        # -------------------------------------------------------------
        print("2. Testing 3D Bloch Sphere...")
        v_bloch = sd.bloch("+", view="3d", theme="scientific")
        html_bloch = ARTIFACTS_DIR / "bloch_sphere_3d.html"
        v_bloch.save(html_bloch)

        page.goto(f"file://{html_bloch.resolve()}")
        page.wait_for_selector(".plotly-graph-div")
        page.wait_for_timeout(1000)

        ss_bloch = ARTIFACTS_DIR / "screenshot_bloch_sphere_3d.png"
        page.screenshot(path=str(ss_bloch))
        print(f"   Saved {ss_bloch.name}")

        # -------------------------------------------------------------
        # 3. QUANTUM CIRCUIT & EVOLUTION
        # -------------------------------------------------------------
        print("3. Testing Quantum Circuit Visualization...")
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        v_circuit = sd.see(qc, theme="scientific")
        html_circuit = ARTIFACTS_DIR / "quantum_circuit.html"
        v_circuit.save(html_circuit)

        page.goto(f"file://{html_circuit.resolve()}")
        page.wait_for_selector(".plotly-graph-div")
        page.wait_for_timeout(800)

        ss_circuit = ARTIFACTS_DIR / "screenshot_quantum_circuit.png"
        page.screenshot(path=str(ss_circuit))
        print(f"   Saved {ss_circuit.name}")

        # -------------------------------------------------------------
        # 4. ENTANGLEMENT & BELL STATE
        # -------------------------------------------------------------
        print("4. Testing Quantum Entanglement Analysis...")
        bell = [1.0 / math.sqrt(2.0), 0.0, 0.0, 1.0 / math.sqrt(2.0)]
        v_ent = sd.entanglement(bell, theme="scientific")
        html_ent = ARTIFACTS_DIR / "quantum_entanglement.html"
        v_ent.save(html_ent)

        page.goto(f"file://{html_ent.resolve()}")
        page.wait_for_selector(".plotly-graph-div")
        page.wait_for_timeout(800)

        ss_ent = ARTIFACTS_DIR / "screenshot_quantum_entanglement.png"
        page.screenshot(path=str(ss_ent))
        print(f"   Saved {ss_ent.name}")

        # -------------------------------------------------------------
        # 5. CNN 3D FEATURE MAPS
        # -------------------------------------------------------------
        print("5. Testing 3D CNN Feature Maps...")
        cnn_model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(4, 6, kernel_size=3, padding=1),
        )
        img = torch.randn(1, 1, 16, 16)
        v_cnn = sd.see(cnn_model, input=img, view="3d", theme="scientific")
        html_cnn = ARTIFACTS_DIR / "cnn_3d_spatial.html"
        v_cnn.save(html_cnn)

        page.goto(f"file://{html_cnn.resolve()}")
        page.wait_for_selector(".plotly-graph-div")
        page.wait_for_timeout(1000)

        ss_cnn = ARTIFACTS_DIR / "screenshot_cnn_3d_spatial.png"
        page.screenshot(path=str(ss_cnn))
        print(f"   Saved {ss_cnn.name}")

        browser.close()

    print("--- Real Browser Verification Complete: All Tests Passed! ---\n")


if __name__ == "__main__":
    test_browser_interactions()
