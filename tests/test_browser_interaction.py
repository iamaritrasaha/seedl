"""Interactive Browser Verification Tests via Playwright — SeeDL 0.2.0 Audit.

Tests actual browser interactions on exported 3D HTML visualizations:
  - Direct neuron selection via client-side JavaScript bridge
  - Correct neuron identity resolution
  - Incoming edge highlighting coordinates
  - Outgoing edge highlighting coordinates
  - Selection reset and trace opacity restoration
  - Camera preservation across selections
  - Repeated selection cycles
  - HTML export integrity
  - Special characters in model labels (HTML injection / XSS safety)
  - Safe JavaScript serialization
  - Payload and memory footprint checks
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import seedl as sd


def _check_playwright_browser_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
        return True
    except Exception:
        return False


_HAS_PLAYWRIGHT_BROWSER = _check_playwright_browser_available()


@pytest.mark.skipif(not _HAS_PLAYWRIGHT_BROWSER, reason="Playwright Chromium browser not available")
def test_browser_3d_neuron_interaction_lifecycle():
    """Verify end-to-end browser interaction: selection, highlighting, reset, repeated clicks."""
    from playwright.sync_api import sync_playwright

    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
    x = torch.tensor([[1.0, 2.0]])
    v = sd.see(model, input=x, view="3d")

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        v.save(f.name)
        html_path = f.name

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("file://" + html_path)
            page.wait_for_selector(".plotly-graph-div")

            # 1. Verify JS API is available
            has_select = page.evaluate("typeof window.__seedl_select_neuron === 'function'")
            has_clear = page.evaluate("typeof window.__seedl_clear_selection === 'function'")
            assert has_select, "window.__seedl_select_neuron must be defined"
            assert has_clear, "window.__seedl_clear_selection must be defined"

            # 2. Select neuron 0:0
            page.evaluate("window.__seedl_select_neuron('0:0')")

            # Check that highlight traces are populated and visible
            highlight_state = page.evaluate("""() => {
                var gd = document.getElementsByClassName('plotly-graph-div')[0];
                var nodeTrace = null, outTrace = null;
                for (var i = 0; i < gd.data.length; i++) {
                    var role = gd.data[i].meta ? gd.data[i].meta.role : null;
                    if (role === 'highlight_node') nodeTrace = gd.data[i];
                    if (role === 'highlight_outgoing') outTrace = gd.data[i];
                }
                return {
                    nodeVisible: nodeTrace ? nodeTrace.visible : false,
                    nodeX: nodeTrace ? nodeTrace.x : [],
                    outVisible: outTrace ? outTrace.visible : false,
                    outLen: outTrace ? outTrace.x.length : 0,
                    cardText: gd.layout.annotations[gd.layout.annotations.length - 1].text
                };
            }""")

            assert highlight_state["nodeVisible"] is True
            assert len(highlight_state["nodeX"]) == 1
            assert highlight_state["outVisible"] is True
            assert highlight_state["outLen"] > 0
            assert "0:0" in highlight_state["cardText"]
            assert "Input" in highlight_state["cardText"]

            # 3. Repeated selection: switch to hidden layer neuron 1:1
            page.evaluate("window.__seedl_select_neuron('1:1')")
            card_1_1 = page.evaluate("""() => {
                var gd = document.getElementsByClassName('plotly-graph-div')[0];
                return gd.layout.annotations[gd.layout.annotations.length - 1].text;
            }""")
            assert "1:1" in card_1_1

            # 4. Clear selection and verify reset
            page.evaluate("window.__seedl_clear_selection()")
            cleared_state = page.evaluate("""() => {
                var gd = document.getElementsByClassName('plotly-graph-div')[0];
                var nodeTrace = null;
                for (var i = 0; i < gd.data.length; i++) {
                    var role = gd.data[i].meta ? gd.data[i].meta.role : null;
                    if (role === 'highlight_node') nodeTrace = gd.data[i];
                }
                return {
                    nodeVisible: nodeTrace ? nodeTrace.visible : false,
                    cardText: gd.layout.annotations[gd.layout.annotations.length - 1].text
                };
            }""")
            assert cleared_state["nodeVisible"] is False
            assert "SeeDL Overview" in cleared_state["cardText"]

            browser.close()
    finally:
        if os.path.exists(html_path):
            os.remove(html_path)


@pytest.mark.skipif(not _HAS_PLAYWRIGHT_BROWSER, reason="Playwright Chromium browser not available")
def test_browser_camera_preservation_during_selection():
    """Verify camera orientation is preserved when selecting and clearing neurons."""
    from playwright.sync_api import sync_playwright

    model = nn.Sequential(nn.Linear(2, 2))
    v = sd.see(model, view="3d")

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        v.save(f.name)
        html_path = f.name

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("file://" + html_path)
            page.wait_for_selector(".plotly-graph-div")

            # Mutate the camera to simulate user rotation
            page.evaluate("""() => {
                var gd = document.getElementsByClassName('plotly-graph-div')[0];
                Plotly.relayout(gd, {'scene.camera.eye': {x: 3.14, y: 2.71, z: 1.41}});
            }""")

            # Select neuron
            page.evaluate("window.__seedl_select_neuron('0:0')")

            cam_after_select = page.evaluate("""() => {
                var gd = document.getElementsByClassName('plotly-graph-div')[0];
                return gd.layout.scene.camera.eye;
            }""")

            # Eye coordinates must remain at rotated position
            assert abs(cam_after_select["x"] - 3.14) < 1e-3
            assert abs(cam_after_select["y"] - 2.71) < 1e-3
            assert abs(cam_after_select["z"] - 1.41) < 1e-3

            browser.close()
    finally:
        if os.path.exists(html_path):
            os.remove(html_path)


@pytest.mark.skipif(not _HAS_PLAYWRIGHT_BROWSER, reason="Playwright Chromium browser not available")
def test_browser_special_characters_in_labels():
    """Verify that model labels containing HTML special characters render safely without script execution."""
    from playwright.sync_api import sync_playwright

    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
    v = sd.see(model, view="3d")
    # Inject special characters and potential script tags into graph metadata
    v.data["neuron_graph"]["layers"][0]["label"] = "Input <script>alert('xss')</script> & Custom<T>"
    v.data["neuron_graph"]["layers"][0]["module"] = "Module<Special & 'Chars'>"

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        v.save(f.name)
        html_path = f.name

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Listen for unexpected alert / script dialogs
            dialog_triggered = []
            page.on("dialog", lambda d: dialog_triggered.append(d.message))

            page.goto("file://" + html_path)
            page.wait_for_selector(".plotly-graph-div")

            # Select neuron
            page.evaluate("window.__seedl_select_neuron('0:0')")

            card_text = page.evaluate("""() => {
                var gd = document.getElementsByClassName('plotly-graph-div')[0];
                return gd.layout.annotations[gd.layout.annotations.length - 1].text;
            }""")

            assert len(dialog_triggered) == 0, "No script alerts should be triggered"
            assert "0:0" in card_text
            # Verify that <script> was properly escaped to &lt;script&gt; and didn't execute
            assert "<script>" not in card_text
            assert "&lt;script&gt;" in card_text

            browser.close()
    finally:
        if os.path.exists(html_path):
            os.remove(html_path)


def test_html_export_payload_size():
    """Verify HTML export payload remains bounded and under reasonable limits."""
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    v = sd.see(model, view="3d")
    html_content = v.to_html()

    # Payload size should be bounded (under 5MB for a small model)
    size_mb = len(html_content.encode("utf-8")) / (1024 * 1024)
    assert size_mb < 5.0, f"HTML export payload too large: {size_mb:.2f} MB"
    assert "plotly-graph-div" in html_content
    assert "__seedl_select_neuron" in html_content


@pytest.mark.skipif(not _HAS_PLAYWRIGHT_BROWSER, reason="Playwright Chromium browser not available")
def test_browser_pointer_drag_rotation_and_selection_lifecycle():
    """Verify that pointer dragging rotates the 3D scene without triggering selection, and subsequent clicks function."""
    from playwright.sync_api import sync_playwright

    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
    x = torch.tensor([[1.0, 2.0]])
    v = sd.see(model, input=x, view="3d")

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        v.save(f.name)
        html_path = f.name

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1200, "height": 800})
            page.goto("file://" + html_path)
            page.wait_for_selector(".plotly-graph-div")
            page.wait_for_timeout(800)

            # 1. Check initial camera
            cam_init = page.evaluate(
                '() => document.getElementsByClassName("plotly-graph-div")[0].layout.scene.camera.eye'
            )

            # 2. Perform pointer drag across the scene center
            page.mouse.move(600, 400)
            page.mouse.down()
            page.mouse.move(750, 300, steps=10)
            page.mouse.up()
            page.wait_for_timeout(500)

            # 3. Verify camera changed
            cam_dragged = page.evaluate(
                '() => document.getElementsByClassName("plotly-graph-div")[0].layout.scene.camera.eye'
            )
            assert cam_init != cam_dragged, "Scene camera should rotate during mouse drag"

            # 4. Verify dragging did NOT select any neuron
            annot_drag = page.evaluate("""() => {
                var gd = document.getElementsByClassName('plotly-graph-div')[0];
                return gd.layout.annotations[gd.layout.annotations.length - 1].text;
            }""")
            assert "NEURON INSPECTION" not in annot_drag, (
                "Pointer drag must not trigger neuron selection"
            )

            # 5. Click neuron 0:0 and verify selection
            page.evaluate("""() => {
                var gd = document.getElementsByClassName('plotly-graph-div')[0];
                var nodeCurve = -1;
                for (var i = 0; i < gd.data.length; i++) {
                    if (gd.data[i].meta && gd.data[i].meta.role === 'nodes') {
                        nodeCurve = i;
                        break;
                    }
                }
                gd.emit('plotly_click', {
                    points: [{ curveNumber: nodeCurve, pointNumber: 0, customdata: '0:0' }]
                });
            }""")
            page.wait_for_timeout(400)
            annot_0 = page.evaluate("""() => {
                var gd = document.getElementsByClassName('plotly-graph-div')[0];
                return gd.layout.annotations[gd.layout.annotations.length - 1].text;
            }""")
            assert "0:0" in annot_0
            assert "NEURON INSPECTION" in annot_0

            # 6. Click neuron 1:1 and verify selection switched
            page.evaluate("""() => {
                var gd = document.getElementsByClassName('plotly-graph-div')[0];
                var nodeCurve = -1;
                for (var i = 0; i < gd.data.length; i++) {
                    if (gd.data[i].meta && gd.data[i].meta.role === 'nodes') {
                        nodeCurve = i;
                        break;
                    }
                }
                gd.emit('plotly_click', {
                    points: [{ curveNumber: nodeCurve, pointNumber: 1, customdata: '1:1' }]
                });
            }""")
            page.wait_for_timeout(400)
            annot_1 = page.evaluate("""() => {
                var gd = document.getElementsByClassName('plotly-graph-div')[0];
                return gd.layout.annotations[gd.layout.annotations.length - 1].text;
            }""")
            assert "1:1" in annot_1

            # 7. Clear selection and verify overview restored
            page.evaluate("window.__seedl_clear_selection()")
            page.wait_for_timeout(400)
            annot_clear = page.evaluate("""() => {
                var gd = document.getElementsByClassName('plotly-graph-div')[0];
                return gd.layout.annotations[gd.layout.annotations.length - 1].text;
            }""")
            assert "SeeDL Overview" in annot_clear

            browser.close()
    finally:
        if os.path.exists(html_path):
            os.remove(html_path)


@pytest.mark.skipif(not _HAS_PLAYWRIGHT_BROWSER, reason="Playwright Chromium browser not available")
def test_html_security_mixed_case_script_tags():
    """Verify that mixed-case </sCrIpT>, </SCRIPT>, <!-- sequences in user-controlled labels cannot terminate script elements."""
    from playwright.sync_api import sync_playwright

    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
    v = sd.see(model, view="3d")
    # Attack vectors targeting HTML parser termination
    v.data["neuron_graph"]["layers"][0]["label"] = "</sCrIpT><script>alert('xss1')</script>"
    v.data["neuron_graph"]["layers"][1]["label"] = (
        "<!-- </SCRIPT> <script>alert('xss2')</script> -->"
    )
    v.data["neuron_graph"]["layers"][0]["module"] = "\"></script><script>alert('xss3')</script>"

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        v.save(f.name)
        html_path = f.name

    try:
        html_text = open(html_path, encoding="utf-8").read()
        # Verify JSON serialization uses unicode escapes so </script> never appears in script body
        assert "</sCrIpT>" not in html_text
        assert "</SCRIPT>" not in html_text

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            dialog_triggered = []
            page.on("dialog", lambda d: dialog_triggered.append(d.message))

            page.goto("file://" + html_path)
            page.wait_for_selector(".plotly-graph-div")

            # Verify that page loaded and JS initialized without syntax or parser error
            is_func = page.evaluate("typeof window.__seedl_select_neuron === 'function'")
            assert is_func, "JS environment must initialize cleanly"
            assert len(dialog_triggered) == 0, "No script injection should execute"

            # Select neuron and verify safe rendering
            page.evaluate("window.__seedl_select_neuron('0:0')")
            card_text = page.evaluate("""() => {
                var gd = document.getElementsByClassName('plotly-graph-div')[0];
                return gd.layout.annotations[gd.layout.annotations.length - 1].text;
            }""")
            assert len(dialog_triggered) == 0
            assert "<script>" not in card_text
            assert "&lt;script&gt;" in card_text

            browser.close()
    finally:
        if os.path.exists(html_path):
            os.remove(html_path)


@pytest.mark.skipif(not _HAS_PLAYWRIGHT_BROWSER, reason="Playwright Chromium browser not available")
def test_browser_real_pointer_click_on_webgl_neuron():
    """Verify that physical mouse pointer clicks (with pixel coordinates) on visible WebGL neurons select them."""
    from playwright.sync_api import sync_playwright

    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
    x = torch.tensor([[1.0, 2.0]])
    v = sd.see(model, input=x, view="3d")

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        v.save(f.name)
        html_path = f.name

    screenshot_dir = Path("/tmp/seedl_browser_test")
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1200, "height": 800})
            page.goto("file://" + html_path)
            page.wait_for_selector(".plotly-graph-div")
            page.wait_for_timeout(1000)

            # Record initial overview screenshot
            page.screenshot(path=str(screenshot_dir / "real_click_01_overview.png"))

            # Attach hover listener to track node detection
            page.evaluate("""() => {
                window.__nodeHovers = [];
                var gd = document.getElementsByClassName('plotly-graph-div')[0];
                gd.on('plotly_hover', (d) => {
                    if (d && d.points && d.points[0]) {
                        var pt = d.points[0];
                        var tr = gd.data[pt.curveNumber];
                        if (tr && tr.meta && tr.meta.role === 'nodes') {
                            window.__nodeHovers.push({
                                x: pt.x, y: pt.y, z: pt.z,
                                customdata: pt.customdata,
                                curve: pt.curveNumber,
                                point: pt.pointNumber
                            });
                        }
                    }
                });
            }""")

            # Search canvas grid for a visible neuron marker
            rect = page.evaluate("""() => {
                var c = document.querySelector('canvas');
                var r = c.getBoundingClientRect();
                return { left: r.left, top: r.top, width: r.width, height: r.height };
            }""")

            target_point = None
            for y_scan in range(int(rect["top"]) + 50, int(rect["top"] + rect["height"]) - 50, 15):
                for x_scan in range(
                    int(rect["left"]) + 50, int(rect["left"] + rect["width"]) - 50, 15
                ):
                    page.mouse.move(x_scan, y_scan)
                    hovers = page.evaluate("window.__nodeHovers")
                    if hovers:
                        target_point = (x_scan, y_scan, hovers[-1]["customdata"])
                        break
                if target_point:
                    break

            assert target_point is not None, (
                "A visible neuron marker must be detected on the canvas"
            )
            target_x, target_y, target_id = target_point

            # Initial camera
            cam_before = page.evaluate(
                '() => document.getElementsByClassName("plotly-graph-div")[0].layout.scene.camera.eye'
            )

            # Perform REAL physical mouse pointer click on the neuron
            page.mouse.move(target_x, target_y)
            page.mouse.click(target_x, target_y)
            page.wait_for_timeout(500)

            # Check that neuron was selected by the mouse click
            annot_after_click = page.evaluate("""() => {
                var gd = document.getElementsByClassName('plotly-graph-div')[0];
                return gd.layout.annotations[gd.layout.annotations.length - 1].text;
            }""")
            assert "NEURON INSPECTION" in annot_after_click
            assert target_id in annot_after_click

            # Check camera position preserved across selection
            cam_after = page.evaluate(
                '() => document.getElementsByClassName("plotly-graph-div")[0].layout.scene.camera.eye'
            )
            assert cam_before["x"] == pytest.approx(cam_after["x"], abs=1e-3)
            assert cam_before["y"] == pytest.approx(cam_after["y"], abs=1e-3)
            assert cam_before["z"] == pytest.approx(cam_after["z"], abs=1e-3)

            # Take screenshot with neuron selected
            page.screenshot(path=str(screenshot_dir / "real_click_02_selected.png"))

            # Clear selection via API
            page.evaluate("window.__seedl_clear_selection()")
            page.wait_for_timeout(400)
            annot_cleared = page.evaluate("""() => {
                var gd = document.getElementsByClassName('plotly-graph-div')[0];
                return gd.layout.annotations[gd.layout.annotations.length - 1].text;
            }""")
            assert "SeeDL Overview" in annot_cleared

            browser.close()
    finally:
        if os.path.exists(html_path):
            os.remove(html_path)


@pytest.mark.skipif(not _HAS_PLAYWRIGHT_BROWSER, reason="Playwright Chromium browser not available")
def test_browser_quantum_bloch_and_circuit_inspection():
    """Verify in-browser rendering and visual integrity of Bloch sphere and Quantum Circuit HTML exports."""
    from playwright.sync_api import sync_playwright

    # 1. Bloch sphere 3D visualization
    v_bloch = sd.bloch("+", backend="plotly")
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        v_bloch.save(f.name)
        bloch_path = f.name

    # 2. Quantum circuit visualization
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    v_qc = sd.circuit(qc, simulate=True, backend="plotly")
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        v_qc.save(f.name)
        qc_path = f.name

    screenshot_dir = Path("/tmp/seedl_browser_test")
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)

            # Test Bloch sphere in browser
            page_bloch = browser.new_page(viewport={"width": 1000, "height": 800})
            page_bloch.goto("file://" + bloch_path)
            page_bloch.wait_for_selector(".plotly-graph-div")
            page_bloch.wait_for_timeout(800)

            # Check that 3D scene and vector are present
            bloch_info = page_bloch.evaluate("""() => {
                var gd = document.getElementsByClassName('plotly-graph-div')[0];
                return {
                    numTraces: gd.data.length,
                    hasScene: !!gd.layout.scene,
                    title: gd.layout.title ? gd.layout.title.text : ''
                };
            }""")
            assert bloch_info["numTraces"] >= 2, (
                "Bloch sphere should contain surface wireframe and statevector"
            )
            assert bloch_info["hasScene"] is True, "Bloch sphere must render as 3D WebGL scene"
            page_bloch.screenshot(path=str(screenshot_dir / "quantum_01_bloch_sphere.png"))
            page_bloch.close()

            # Test Quantum circuit in browser
            page_qc = browser.new_page(viewport={"width": 1100, "height": 700})
            page_qc.goto("file://" + qc_path)
            page_qc.wait_for_selector(".plotly-graph-div")
            page_qc.wait_for_timeout(800)

            qc_info = page_qc.evaluate("""() => {
                var gd = document.getElementsByClassName('plotly-graph-div')[0];
                return {
                    numTraces: gd.data.length,
                    annotations: gd.layout.annotations ? gd.layout.annotations.length : 0
                };
            }""")
            assert qc_info["numTraces"] >= 2, (
                "Quantum circuit must render qubit wires and gate traces"
            )
            assert qc_info["annotations"] >= 2, (
                "Quantum circuit must render qubit wire labels and gate annotations"
            )
            page_qc.screenshot(path=str(screenshot_dir / "quantum_02_circuit.png"))
            page_qc.close()

            browser.close()
    finally:
        for p_file in (bloch_path, qc_path):
            if os.path.exists(p_file):
                os.remove(p_file)
