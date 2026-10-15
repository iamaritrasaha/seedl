"""Actual browser interaction tests for Vision & Attention Intelligence via Playwright Chromium.

Verifies interactive controls, selectors, client-side rendering, hover inspection,
and exact numerical agreement with source PyTorch tensors in offline HTML.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import seedl as sd


@pytest.fixture(scope="module")
def browser():
    api = pytest.importorskip("playwright.sync_api")
    with api.sync_playwright() as p:
        try:
            instance = p.chromium.launch(headless=True)
        except api.Error as error:
            pytest.skip(f"Playwright Chromium unavailable: {error}")
        yield instance
        instance.close()


def test_cnn_feature_map_browser_interaction(browser, tmp_path):
    """Verify CNN feature map interactive HTML in a real Chromium browser."""
    # Create simple ConvNet with known activation ranges
    model = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
    )
    # Fixed input
    torch.manual_seed(42)
    x = torch.randn(1, 1, 8, 8)

    view = sd.features(model, input=x)
    target = view.save(tmp_path / "cnn_explorer.html")

    page = browser.new_page(viewport={"width": 1280, "height": 900})
    page_errors: list[str] = []
    network_requests: list[str] = []
    page.on("pageerror", lambda err: page_errors.append(str(err)))
    page.on("request", lambda req: network_requests.append(req.url))

    page.goto(target.as_uri())
    page.wait_for_selector("#featureCanvas")

    # 1. Verify offline integrity (no external requests, no JS console errors)
    assert len(page_errors) == 0, f"Unexpected JS page errors: {page_errors}"
    external_reqs = [url for url in network_requests if not url.startswith("file://")]
    assert len(external_reqs) == 0, f"Unexpected external network requests: {external_reqs}"

    # 2. Verify layer selector contains all captured layers
    layer_options = page.eval_on_selector_all(
        "#layerSelect option", "opts => opts.map(o => o.textContent)"
    )
    assert len(layer_options) >= 2
    assert any("Conv2d" in opt for opt in layer_options)

    # 3. Check stats HUD values match first channel
    first_ch_stats = view.data["layers"][0]["channels"][0]["stats"]
    min_text = page.locator("#stMin").inner_text()
    max_text = page.locator("#stMax").inner_text()
    mean_text = page.locator("#stMean").inner_text()

    assert float(min_text) == pytest.approx(first_ch_stats["min"], abs=1e-3)
    assert float(max_text) == pytest.approx(first_ch_stats["max"], abs=1e-3)
    assert float(mean_text) == pytest.approx(first_ch_stats["mean"], abs=1e-3)

    # 4. Switch channel via select
    page.select_option("#channelSelect", "1")
    second_ch_stats = view.data["layers"][0]["channels"][1]["stats"]
    new_mean = page.locator("#stMean").inner_text()
    assert float(new_mean) == pytest.approx(second_ch_stats["mean"], abs=1e-3)

    # 5. Toggle normalization mode
    page.click("#btnNormZero")
    assert "btn-active" in page.get_attribute("#btnNormZero", "class")
    page.click("#btnNormGlobal")
    assert "btn-active" in page.get_attribute("#btnNormGlobal", "class")

    # 6. Hover over canvas to verify coordinate HUD
    canvas = page.locator("#featureCanvas")
    box = canvas.bounding_box()
    assert box is not None
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    hover_text = page.locator("#hoverHud").inner_text()
    assert "Coord" in hover_text
    assert "Activation:" in hover_text

    # 7. Channel gallery interaction
    gallery_items = page.locator(".gallery-item")
    assert gallery_items.count() >= 2
    gallery_items.nth(0).click()
    assert "selected" in gallery_items.nth(0).get_attribute("class")


def test_transformer_attention_browser_interaction(browser, tmp_path):
    """Verify Transformer Attention interactive HTML in a real Chromium browser."""
    # 2-head attention matrix over 4 tokens
    head0 = [
        [0.7, 0.2, 0.1, 0.0],
        [0.1, 0.6, 0.2, 0.1],
        [0.0, 0.2, 0.5, 0.3],
        [0.0, 0.0, 0.2, 0.8],
    ]
    head1 = [
        [0.4, 0.3, 0.2, 0.1],
        [0.2, 0.4, 0.3, 0.1],
        [0.1, 0.2, 0.4, 0.3],
        [0.1, 0.1, 0.2, 0.6],
    ]
    tokens = ["Deep", "Learning", "Attention", "Transformer"]
    weights = torch.tensor([head0, head1])

    view = sd.attention(weights, tokens=tokens)
    target = view.save(tmp_path / "attention_explorer.html")

    page = browser.new_page(viewport={"width": 1280, "height": 900})
    page_errors: list[str] = []
    network_requests: list[str] = []
    page.on("pageerror", lambda err: page_errors.append(str(err)))
    page.on("request", lambda req: network_requests.append(req.url))

    page.goto(target.as_uri())
    page.wait_for_selector(".token-table")

    # 1. Zero console errors, zero external network requests
    assert len(page_errors) == 0, f"Unexpected JS page errors: {page_errors}"
    external_reqs = [url for url in network_requests if not url.startswith("file://")]
    assert len(external_reqs) == 0, f"Unexpected external network requests: {external_reqs}"

    # 2. Check head select options (Head 0, Head 1, Average)
    head_opts = page.eval_on_selector_all(
        "#headSelect option", "opts => opts.map(o => o.textContent)"
    )
    assert "Head 0" in head_opts
    assert "Head 1" in head_opts
    assert any("Average" in opt for opt in head_opts)

    # 3. Check token table headers contain user token labels
    th_texts = page.eval_on_selector_all(".col-hdr", "ths => ths.map(t => t.textContent)")
    assert th_texts == tokens

    # 4. Verify query profile updates when query token is changed
    page.select_option("#querySelect", "2")  # Token "Attention"
    profile_title = page.locator("#profileTitle").inner_text()
    assert "Attention" in profile_title

    # 5. Hover over attention cell to check exact numerical weight
    cells = page.locator(".cell")
    assert cells.count() == 16
    cells.nth(0).hover()
    hover_hud = page.locator("#hoverHud").inner_text()
    assert "Deep" in hover_hud
    # Head 0, Query 0 -> Key 0 is 0.7000
    assert "0.7000" in hover_hud or "70.0%" in hover_hud

    # 6. Switch head to Head 1 and hover (0, 0) -> should show 0.4000
    page.select_option("#headSelect", "1")
    cells = page.locator(".cell")
    cells.nth(0).hover()
    new_hover = page.locator("#hoverHud").inner_text()
    assert "0.4000" in new_hover or "40.0%" in new_hover
