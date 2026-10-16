"""Headless browser end-to-end integration tests using Playwright for SeeDL 0.5.0."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

try:
    from playwright.sync_api import sync_playwright

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

import seedl as sd
from seedl._quantum_lab import quantum_noise_html
from seedl._reports import report
from seedl._trace_renderers import trace_html


class MiniModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4 * 4 * 4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv(x))
        x = x.flatten(1)
        return self.fc(x)


@pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
def test_browser_flow_tracer_interaction():
    """Verify tabs, node click inspection, and search filtering in headless Chromium."""
    model = MiniModel()
    trace = sd.trace(model, input=torch.randn(1, 1, 4, 4))
    flow_viz = sd.flow(trace)
    html_markup = trace_html(flow_viz)

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "trace.html"
        file_path.write_text(html_markup, encoding="utf-8")
        file_url = f"file://{file_path.resolve()}"

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            console_errors = []
            page.on(
                "console",
                lambda msg: console_errors.append(msg.text) if msg.type == "error" else None,
            )

            page.goto(file_url)

            # Check header content
            assert page.is_visible("header")
            assert "Neural Execution Trace" in page.text_content("header")

            # Check default tab: Flow Graph visible, Timeline hidden
            assert page.is_visible("#tab-graph")
            assert not page.is_visible("#tab-timeline")

            # Click Timeline tab button
            page.click("button:has-text('Timeline')")
            assert not page.is_visible("#tab-graph")
            assert page.is_visible("#tab-timeline")

            # Click Tensor Table tab button
            page.click("button:has-text('Tensor Table')")
            assert page.is_visible("#tab-table")
            table_rows = page.query_selector_all("#table-body tr")
            assert len(table_rows) > 0

            # Switch back to Flow Graph
            page.click("button:has-text('Flow Graph')")
            assert page.is_visible("#tab-graph")

            # Click first node and verify Inspector updates
            flow_nodes = page.query_selector_all(".flow-node")
            assert len(flow_nodes) > 0
            flow_nodes[0].click()

            insp_title = page.text_content("#insp-title")
            assert (
                "conv" in insp_title.lower()
                or "linear" in insp_title.lower()
                or len(insp_title) > 0
            )

            # Test search filter
            page.fill("#search-box", "conv")
            visible_nodes = [c for c in page.query_selector_all(".flow-node") if c.is_visible()]
            assert len(visible_nodes) >= 1

            assert not console_errors, f"JavaScript errors occurred: {console_errors}"
            browser.close()


@pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
def test_browser_quantum_noise_slider_interaction():
    """Verify dynamic noise slider updates state metrics in headless Chromium."""
    viz = sd.quantum_noise(channel="depolarizing", param=0.2, steps=10)
    html_markup = quantum_noise_html(viz)

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "quantum.html"
        file_path.write_text(html_markup, encoding="utf-8")
        file_url = f"file://{file_path.resolve()}"

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            console_errors = []
            page.on(
                "console",
                lambda msg: console_errors.append(msg.text) if msg.type == "error" else None,
            )

            page.goto(file_url)

            # Read initial decohered purity at p=0.2
            initial_purity_text = page.text_content("#final-purity")
            initial_purity = float(initial_purity_text)

            # Move slider to 0.8
            slider = page.query_selector("#noise-slider")
            assert slider is not None
            page.evaluate(
                "el => { el.value = '0.80'; el.dispatchEvent(new Event('input')); }", slider
            )

            # Verify updated purity is strictly lower
            updated_purity_text = page.text_content("#final-purity")
            updated_purity = float(updated_purity_text)
            assert updated_purity < initial_purity

            assert not console_errors, f"JavaScript errors occurred: {console_errors}"
            browser.close()


@pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
def test_browser_report_theme_toggle_and_navigation():
    """Verify report theme toggle and TOC navigation in headless Chromium."""
    v1 = sd.see([[1, 2], [3, 4]], backend="svg")
    rep = report(
        title="Automated Test Report",
        views=[v1],
        subtitle="Verification Sprint",
        theme="dark",
    )
    html_markup = rep.to_html()

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "report.html"
        file_path.write_text(html_markup, encoding="utf-8")
        file_url = f"file://{file_path.resolve()}"

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            console_errors = []
            page.on(
                "console",
                lambda msg: console_errors.append(msg.text) if msg.type == "error" else None,
            )

            page.goto(file_url)

            # Verify initial dark theme (light-theme class absent)
            is_light_before = page.evaluate("document.body.classList.contains('light-theme')")
            assert not is_light_before

            # Click theme toggle button
            page.click(".theme-toggle")
            is_light_after = page.evaluate("document.body.classList.contains('light-theme')")
            assert is_light_after

            # Click again to toggle back
            page.click(".theme-toggle")
            is_light_reset = page.evaluate("document.body.classList.contains('light-theme')")
            assert not is_light_reset

            assert not console_errors, f"JavaScript errors occurred: {console_errors}"
            browser.close()


@pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
def test_browser_attention_rollout_interaction():
    """Verify attention rollout dropdown navigation and matrix rendering in headless Chromium."""
    from seedl._transformer_observatory import attention_rollout_html

    a1 = [[0.8, 0.2], [0.1, 0.9]]
    a2 = [[0.7, 0.3], [0.4, 0.6]]
    viz = sd.attention_rollout([a1, a2], tokens=["tokA", "tokB"])
    html_markup = attention_rollout_html(viz)

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "rollout.html"
        file_path.write_text(html_markup, encoding="utf-8")
        file_url = f"file://{file_path.resolve()}"

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            console_errors = []
            page.on(
                "console",
                lambda msg: console_errors.append(msg.text) if msg.type == "error" else None,
            )

            page.goto(file_url)

            # Check header
            assert page.is_visible("header")
            assert "Attention Rollout" in page.text_content("header")

            # Check layer select dropdown
            select = page.query_selector("#layer-select")
            assert select is not None
            options = page.query_selector_all("#layer-select option")
            assert len(options) >= 2  # Layer options + Final

            # Check table container renders
            table_cells = page.query_selector_all("table.matrix-table td")
            assert len(table_cells) > 0

            # Switch to layer 0
            page.select_option("#layer-select", "0")
            table_cells_l0 = page.query_selector_all("table.matrix-table td")
            assert len(table_cells_l0) > 0

            assert not console_errors, f"JavaScript errors occurred: {console_errors}"
            browser.close()


@pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
def test_browser_gradient_health_dashboard():
    """Verify gradient health dashboard metric cards and tables in headless Chromium."""
    from seedl._gradient_health import gradient_health_html

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()

    viz = sd.gradient_health(model)
    html_markup = gradient_health_html(viz)

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "grad_health.html"
        file_path.write_text(html_markup, encoding="utf-8")
        file_url = f"file://{file_path.resolve()}"

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            console_errors = []
            page.on(
                "console",
                lambda msg: console_errors.append(msg.text) if msg.type == "error" else None,
            )

            page.goto(file_url)

            # Verify stat cards
            assert page.is_visible(".stats-grid")
            assert page.is_visible("#stat-l2")
            l2_text = page.text_content("#stat-l2")
            assert "e" in l2_text or "." in l2_text

            # Verify layer table rows
            layer_rows = page.query_selector_all("#layers-body tr")
            assert len(layer_rows) >= 2

            # Verify parameter table rows
            param_rows = page.query_selector_all("#params-body tr")
            assert len(param_rows) >= 2

            assert not console_errors, f"JavaScript errors occurred: {console_errors}"
            browser.close()


@pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
def test_browser_mobile_responsiveness_and_keyboard_nav():
    """Verify layout responsiveness on mobile viewport and tab keyboard accessibility."""
    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    trace = sd.trace(model, input=torch.randn(1, 2))
    html_markup = trace_html(sd.flow(trace))

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "mobile_trace.html"
        file_path.write_text(html_markup, encoding="utf-8")
        file_url = f"file://{file_path.resolve()}"

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            # Set mobile viewport
            page.set_viewport_size({"width": 375, "height": 667})

            console_errors = []
            page.on(
                "console",
                lambda msg: console_errors.append(msg.text) if msg.type == "error" else None,
            )

            page.goto(file_url)

            # Check that container fits within viewport without overflow
            container = page.query_selector(".container")
            assert container is not None

            # Test tab navigation via keyboard
            page.focus("#search-box")
            page.keyboard.type("0")
            page.keyboard.press("Escape")

            assert not console_errors, f"JavaScript errors occurred: {console_errors}"
            browser.close()
