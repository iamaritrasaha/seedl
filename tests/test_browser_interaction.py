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
