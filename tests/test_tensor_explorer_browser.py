"""Browser interaction tests for SeeDL's Interactive Multidimensional Tensor Explorer.

Executes in real Playwright Chromium to validate client-side slider scrubbing, dimension
switching, playback controls, colormap changes, hover coordinates, non-finite values,
multiple widget isolation, keyboard accessibility, and zero browser console errors.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

import seedl as sd

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from playwright.sync_api import sync_playwright

    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False


def _is_chromium_launchable() -> bool:
    if not HAS_PLAYWRIGHT:
        return False
    try:
        with sync_playwright() as p:
            b = p.chromium.launch(headless=True)
            b.close()
        return True
    except Exception:
        return False


_CHROMIUM_AVAILABLE = _is_chromium_launchable()
skip_without_browser = pytest.mark.skipif(
    not _CHROMIUM_AVAILABLE, reason="Playwright Chromium browser not available"
)

_ENV_ARTIFACTS = os.environ.get("SEEDL_TEST_ARTIFACTS_DIR")
ARTIFACTS_DIR = (
    Path(_ENV_ARTIFACTS)
    if _ENV_ARTIFACTS
    else Path(tempfile.gettempdir()) / "seedl_browser_artifacts"
)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Browser Interaction Tests
# ==============================================================================


@skip_without_browser
def test_browser_tensor_slider_and_frame_navigation(tmp_path: Path):
    """Verify that moving the slider changes displayed slice, badge, and statistics."""
    arr = np.linspace(-5.0, 5.0, 16 * 20 * 20, dtype=np.float32).reshape(16, 20, 20)
    view = sd.see(arr, interactive=True, dim_names=["depth", "height", "width"])
    html_path = tmp_path / "slider_test.html"
    view.save(html_path)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1024, "height": 768})
        console_errors: list[str] = []
        page.on("pageerror", lambda err: console_errors.append(str(err)))

        page.goto(html_path.as_uri())

        # Verify initial badge
        badge = page.locator(".ste-index-badge")
        assert "depth = 0 (1/16)" in badge.inner_text()

        # Update slider to index 8
        slider = page.locator(".ste-range")
        slider.evaluate("el => { el.value = 8; el.dispatchEvent(new Event('input')); }")
        page.wait_for_timeout(100)

        assert "depth = 8 (9/16)" in badge.inner_text()
        stats = page.locator(".ste-slice-stats-content").inner_text()
        assert "Index in Tensor" in stats
        assert "8 / 15" in stats
        assert "20" in stats

        # Capture screenshot
        page.screenshot(path=str(ARTIFACTS_DIR / "01_slider_navigation.png"))
        assert len(console_errors) == 0
        browser.close()


@skip_without_browser
def test_browser_tensor_axis_switching(tmp_path: Path):
    """Verify switching slicing dimensions updates active buttons, slider range, and slice."""
    arr = np.arange(10 * 15 * 20, dtype=np.float32).reshape(10, 15, 20)
    view = sd.see(arr, interactive=True, dim_names=["z", "y", "x"])
    html_path = tmp_path / "axis_switch.html"
    view.save(html_path)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1024, "height": 768})

        page.goto(html_path.as_uri())

        # Check 3 axis buttons exist
        buttons = page.locator(".ste-axis-buttons .ste-btn")
        assert buttons.count() == 3
        assert "ste-btn-active" in (buttons.nth(0).get_attribute("class") or "")

        # Switch to Axis 1 ('y')
        buttons.nth(1).click()
        page.wait_for_timeout(100)

        assert "ste-btn-active" in (buttons.nth(1).get_attribute("class") or "")
        assert "ste-btn-active" not in (buttons.nth(0).get_attribute("class") or "")

        badge = page.locator(".ste-index-badge").inner_text()
        assert "y = 0 (1/15)" in badge

        stats = page.locator(".ste-slice-stats-content").inner_text()
        assert "Dimension" in stats
        assert "y (Axis 1)" in stats

        # Capture screenshot
        page.screenshot(path=str(ARTIFACTS_DIR / "02_axis_switching.png"))
        browser.close()


@skip_without_browser
def test_browser_tensor_playback_controls_and_boundary_wrapping(tmp_path: Path):
    """Verify Prev/Next buttons wrap around boundaries and Play/Pause advances frames."""
    arr = np.zeros((6, 12, 12), dtype=np.float32)
    view = sd.see(arr, interactive=True)
    html_path = tmp_path / "playback.html"
    view.save(html_path)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1024, "height": 768})
        page.goto(html_path.as_uri())

        badge = page.locator(".ste-index-badge")
        next_btn = page.locator("button[title='Next slice']")
        prev_btn = page.locator("button[title='Previous slice']")
        play_btn = page.locator(".ste-playback-buttons button.ste-btn-primary")

        # 0 -> Next -> 1
        next_btn.click()
        assert "(2/6)" in badge.inner_text()

        # 1 -> Prev -> 0
        prev_btn.click()
        assert "(1/6)" in badge.inner_text()

        # 0 -> Prev -> wraps to 5 (last slice)
        prev_btn.click()
        assert "(6/6)" in badge.inner_text()

        # 5 -> Next -> wraps back to 0
        next_btn.click()
        assert "(1/6)" in badge.inner_text()

        # Test Play / Pause
        play_btn.click()
        assert "Pause" in play_btn.inner_text()
        # Wait 900ms (at least 2 ticks of 400ms interval)
        page.wait_for_timeout(900)
        # Should have advanced
        assert "(1/6)" not in badge.inner_text()

        # Pause
        play_btn.click()
        assert "Play" in play_btn.inner_text()
        frozen_text = badge.inner_text()
        page.wait_for_timeout(500)
        # Ensure timer is not runaway
        assert badge.inner_text() == frozen_text

        page.screenshot(path=str(ARTIFACTS_DIR / "03_playback_controls.png"))
        browser.close()


@skip_without_browser
def test_browser_tensor_colormap_switching(tmp_path: Path):
    """Verify colormap dropdown changes canvas pixel colors."""
    arr = np.linspace(0.0, 100.0, 8 * 16 * 16, dtype=np.float32).reshape(8, 16, 16)
    view = sd.see(arr, interactive=True)
    html_path = tmp_path / "colormap.html"
    view.save(html_path)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1024, "height": 768})
        page.goto(html_path.as_uri())

        select = page.locator(".ste-select")

        # Get initial pixel data (Viridis)
        viridis_data = page.evaluate("""() => {
            const canvas = document.querySelector('.ste-canvas');
            const ctx = canvas.getContext('2d');
            return Array.from(ctx.getImageData(128, 128, 1, 1).data);
        }""")

        # Select Coolwarm
        select.select_option("coolwarm")
        page.wait_for_timeout(100)
        coolwarm_data = page.evaluate("""() => {
            const canvas = document.querySelector('.ste-canvas');
            const ctx = canvas.getContext('2d');
            return Array.from(ctx.getImageData(128, 128, 1, 1).data);
        }""")
        page.screenshot(path=str(ARTIFACTS_DIR / "04_colormap_coolwarm.png"))

        # Select Spectral
        select.select_option("spectral")
        page.wait_for_timeout(100)
        spectral_data = page.evaluate("""() => {
            const canvas = document.querySelector('.ste-canvas');
            const ctx = canvas.getContext('2d');
            return Array.from(ctx.getImageData(128, 128, 1, 1).data);
        }""")
        page.screenshot(path=str(ARTIFACTS_DIR / "05_colormap_spectral.png"))

        # The colormap pixel values must differ across palettes
        assert viridis_data != coolwarm_data
        assert coolwarm_data != spectral_data

        browser.close()


@skip_without_browser
def test_browser_tensor_hover_tooltip_coordinates_and_values(tmp_path: Path):
    """Verify hovering over canvas renders tooltip with coordinates and source value."""
    # Constant matrix with value 42.5
    arr = np.full((4, 16, 16), 42.5, dtype=np.float32)
    view = sd.see(arr, interactive=True)
    html_path = tmp_path / "tooltip.html"
    view.save(html_path)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1024, "height": 768})
        page.goto(html_path.as_uri())

        canvas = page.locator(".ste-canvas")
        box = canvas.bounding_box()
        assert box is not None

        # Hover over center of canvas
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.wait_for_timeout(100)

        tooltip = page.locator(".ste-tooltip")
        assert tooltip.is_visible()
        tip_text = tooltip.inner_text()
        assert "42.5000" in tip_text

        # Mouse leave hides tooltip
        page.mouse.move(box["x"] - 50, box["y"] - 50)
        page.wait_for_timeout(100)
        assert not tooltip.is_visible()

        page.screenshot(path=str(ARTIFACTS_DIR / "06_hover_tooltip.png"))
        browser.close()


@skip_without_browser
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_browser_tensor_4d_rgb_image_batch(tmp_path: Path):
    """Verify 4D image batch renders as RGB image and tooltip shows RGB values."""
    batch = torch.zeros(4, 3, 24, 24, dtype=torch.float32)
    batch[:, 0, :, :] = 1.0  # Full Red
    batch[:, 1, :, :] = 0.5  # Half Green
    batch[:, 2, :, :] = 0.0  # No Blue

    view = sd.see(batch, interactive=True, dim_names=["batch", "channels", "height", "width"])
    html_path = tmp_path / "rgb_batch.html"
    view.save(html_path)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1024, "height": 768})
        page.goto(html_path.as_uri())

        stats = page.locator(".ste-slice-stats-content").inner_text()
        assert "Dimension" in stats
        assert "batch (Axis 0)" in stats

        # Hover to check RGB tooltip
        canvas = page.locator(".ste-canvas")
        box = canvas.bounding_box()
        assert box is not None
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.wait_for_timeout(100)

        tooltip = page.locator(".ste-tooltip")
        assert tooltip.is_visible()
        assert "RGB: (1.000, 0.500, 0.000)" in tooltip.inner_text()

        page.screenshot(path=str(ARTIFACTS_DIR / "07_4d_rgb_image.png"))
        browser.close()


@skip_without_browser
def test_browser_tensor_non_finite_nan_and_inf_handling(tmp_path: Path):
    """Verify non-finite numbers (NaN/Inf) render cleanly without canvas crashes."""
    arr = np.linspace(-2.0, 2.0, 4 * 16 * 16, dtype=np.float32).reshape(4, 16, 16)
    arr[0, 8, 8] = np.nan
    arr[0, 8, 9] = np.inf

    view = sd.see(arr, interactive=True)
    html_path = tmp_path / "nan_inf.html"
    view.save(html_path)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1024, "height": 768})
        console_errors: list[str] = []
        page.on("pageerror", lambda err: console_errors.append(str(err)))
        page.goto(html_path.as_uri())

        # Header warning chips
        assert page.locator(".ste-chip-warn").count() >= 1

        # Hover over NaN position (center)
        canvas = page.locator(".ste-canvas")
        box = canvas.bounding_box()
        assert box is not None
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.wait_for_timeout(100)

        tooltip = page.locator(".ste-tooltip")
        assert tooltip.is_visible()
        assert "NaN" in tooltip.inner_text() or "Inf" in tooltip.inner_text()

        page.screenshot(path=str(ARTIFACTS_DIR / "08_nan_inf_rendering.png"))
        assert len(console_errors) == 0
        browser.close()


@skip_without_browser
def test_browser_tensor_multiple_widgets_isolation(tmp_path: Path):
    """Verify multiple interactive widgets on the same page operate completely independently."""
    arr_a = np.zeros((4, 10, 10), dtype=np.float32)
    arr_b = np.ones((8, 12, 12), dtype=np.float32)
    view_a = sd.see(arr_a, interactive=True, dim_names=["sample_a", "h", "w"])
    view_b = sd.see(arr_b, interactive=True, dim_names=["sample_b", "y", "x"])

    html_a = view_a._repr_html_() or ""
    html_b = view_b._repr_html_() or ""

    full_page = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"/><title>Dual Widgets</title></head>
<body>
  <div id="wrapper-a">{html_a}</div>
  <hr/>
  <div id="wrapper-b">{html_b}</div>
</body>
</html>"""
    dual_html = tmp_path / "dual_widgets.html"
    dual_html.write_text(full_page, encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1024, "height": 900})
        page.goto(dual_html.as_uri())

        # Widget A controls
        slider_a = page.locator("#wrapper-a .ste-range")
        badge_a = page.locator("#wrapper-a .ste-index-badge")
        badge_b = page.locator("#wrapper-b .ste-index-badge")

        # Move Widget A to slice 2
        slider_a.evaluate("el => { el.value = 2; el.dispatchEvent(new Event('input')); }")
        page.wait_for_timeout(100)

        assert "sample_a = 2 (3/4)" in badge_a.inner_text()
        # Widget B must be unaffected (still slice 0)
        assert "sample_b = 0 (1/8)" in badge_b.inner_text()

        page.screenshot(path=str(ARTIFACTS_DIR / "09_multiple_widgets.png"))
        browser.close()


@skip_without_browser
def test_browser_tensor_unicode_and_keyboard_navigation(tmp_path: Path):
    """Verify Unicode Greek dimension names and keyboard accessibility (Arrow keys, Space)."""
    arr = np.linspace(0, 1, 6 * 10 * 10, dtype=np.float32).reshape(6, 10, 10)
    # Greek dimension names
    view = sd.see(arr, interactive=True, dim_names=["χρόνος", "ύψος", "πλάτος"])
    html_path = tmp_path / "unicode_keyboard.html"
    view.save(html_path)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1024, "height": 768})
        console_errors: list[str] = []
        page.on("pageerror", lambda err: console_errors.append(str(err)))
        page.goto(html_path.as_uri())

        # Verify Unicode text in badge and buttons
        badge = page.locator(".ste-index-badge")
        assert "χρόνος" in badge.inner_text()
        buttons = page.locator(".ste-axis-buttons .ste-btn")
        assert "ύψος" in buttons.nth(1).inner_text()
        assert "πλάτος" in buttons.nth(2).inner_text()

        # Focus widget container
        widget = page.locator(".seedl-tensor-explorer-widget")
        widget.focus()

        # Keyboard Right Arrow -> advances slice
        page.keyboard.press("ArrowRight")
        page.wait_for_timeout(100)
        assert "χρόνος = 1 (2/6)" in badge.inner_text()

        # Keyboard Left Arrow -> decrements slice
        page.keyboard.press("ArrowLeft")
        page.wait_for_timeout(100)
        assert "χρόνος = 0 (1/6)" in badge.inner_text()

        # Keyboard Spacebar -> toggles Play
        play_btn = page.locator(".ste-playback-buttons button.ste-btn-primary")
        page.keyboard.press("Space")
        page.wait_for_timeout(100)
        assert "Pause" in play_btn.inner_text()
        page.keyboard.press("Space")
        page.wait_for_timeout(100)
        assert "Play" in play_btn.inner_text()

        page.screenshot(path=str(ARTIFACTS_DIR / "10_unicode_keyboard.png"))
        assert len(console_errors) == 0
        browser.close()
