"""Real browser checks for responsive, offline and accessible presentation."""

from __future__ import annotations

import pytest

import seedl as sd


@pytest.fixture(scope="module")
def browser():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as p:
        try:
            instance = p.chromium.launch()
        except Exception as exc:
            pytest.skip(f"Playwright Chromium unavailable: {exc}")
        yield instance
        instance.close()


@pytest.mark.parametrize("width", [320, 390, 768, 1280])
def test_offline_matrix_export_readable_at_each_width(browser, tmp_path, width):
    view = sd.see([[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]], backend="svg")
    view.title = 'User <script>window.injected=true</script> & "labels"'
    path = view.save(tmp_path / "matrix.html")
    page = browser.new_page(viewport={"width": width, "height": 900})
    requests, errors = [], []
    page.on("request", lambda req: requests.append(req.url))
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(path.as_uri())
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert page.locator("h1").inner_text() == view.title
    assert page.evaluate("window.injected") is None
    if width < 600:
        assert page.locator(".matrix-values table").is_visible()
        assert page.locator(".matrix-values td").first.inner_text() == "5.1"
        assert (
            page.locator(".matrix-values td").first.evaluate(
                "el => parseFloat(getComputedStyle(el).fontSize)"
            )
            >= 12
        )
    page.get_by_text("Text alternative and full metadata").focus()
    page.keyboard.press("Enter")
    assert page.locator("details").get_attribute("open") is not None
    assert not errors
    assert not any(url.startswith(("http:", "https:")) for url in requests)
    page.close()


@pytest.mark.parametrize("width", [320, 390, 768, 1280])
def test_tensor_controls_palette_precision_and_keyboard(browser, tmp_path, width):
    np = pytest.importorskip("numpy")
    tensor = np.array([[[0.0, 1.0], [1e-200, 0.5]], [[1, 2], [3, 4]]])
    view = sd.see(tensor, interactive=True, dim_names=["batch", "row", "col"], backend="svg")
    path = view.save(tmp_path / "tensor.html")
    page = browser.new_page(viewport={"width": width, "height": 900})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(path.as_uri())
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    canvas = page.locator("canvas").first
    pixel = canvas.evaluate("el => Array.from(el.getContext('2d').getImageData(0,0,1,1).data)")
    assert pixel == [68, 1, 84, 255]  # actual Viridis lower endpoint
    canvas.focus()
    page.keyboard.press("ArrowDown")
    assert "1.00000e-200" in page.locator(".ste-value-readout").inner_text()
    # Canvas arrow inspection leaves the slice unchanged.
    assert "batch = 0" in page.locator(".ste-index-badge").inner_text()
    slider = page.locator('input[type="range"]')
    slider.focus()
    page.keyboard.press("ArrowRight")
    assert slider.input_value() == "1"
    assert "batch = 1" in page.locator(".ste-index-badge").inner_text()
    assert "Range rescales per slice" in page.locator(".ste-legend").inner_text()
    assert not errors
    page.close()


def test_gradient_names_escaped_and_observations_visible(browser, tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential()
    label = '<img src=x onerror="injected=1">'
    model.add_module(label, torch.nn.Linear(2, 1))
    model(torch.ones(1, 2)).sum().backward()
    view = sd.gradient_health(model, backend="svg")
    view.notes.append("Saturation detected (presentation test observation).")
    path = view.save(tmp_path / "gradients.html")
    page = browser.new_page(viewport={"width": 390, "height": 900})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(path.as_uri())
    assert page.locator("#params-body").inner_text().count(label) == 2
    assert page.locator("#params-body img, #layers-body img").count() == 0
    assert page.evaluate("window.injected") is None
    assert "CAUTION" in page.locator(".findings").first.inner_text()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert not errors
    page.close()


def test_plotly_export_offline_and_responsive(browser, tmp_path):
    pytest.importorskip("plotly")
    view = sd.see([1, 2, 3], backend="plotly")
    path = view.save(tmp_path / "plotly.html")
    page = browser.new_page(viewport={"width": 390, "height": 900})
    requests, errors = [], []
    page.on("request", lambda req: requests.append(req.url))
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(path.as_uri())
    page.wait_for_selector(".main-svg")
    assert (
        page.locator(".plotly-graph-div").evaluate("el => el.getBoundingClientRect().width") <= 390
    )
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert not any(url.startswith(("http:", "https:")) for url in requests)
    assert not errors
    page.close()


def test_bundled_fonts_load_offline_in_export_and_website(browser, tmp_path):
    from pathlib import Path

    view = sd.see([[1, 2], [3, 4]], backend="svg")
    path = view.save(tmp_path / "fonts.html")
    page = browser.new_page(viewport={"width": 390, "height": 900})
    requests = []
    page.on("request", lambda req: requests.append(req.url))
    page.goto(path.as_uri())
    loaded = page.evaluate("""async () => {
        const ui = await document.fonts.load('14px "Google Sans"');
        const mono = await document.fonts.load('14px "JetBrains Mono"');
        return [ui.length, mono.length, getComputedStyle(document.body).fontFamily,
            getComputedStyle(document.querySelector('.matrix-values td')).fontFamily];
    }""")
    assert loaded[0] > 0 and loaded[1] > 0
    assert "Google Sans" in loaded[2] and "JetBrains Mono" in loaded[3]
    assert not any(url.startswith(("https:", "http:")) for url in requests)
    root = Path(__file__).resolve().parents[1]
    page.goto((root / "website/docs/index.html").as_uri())
    website_fonts = page.evaluate("""async () => {
        const ui = await document.fonts.load('14px "Google Sans"');
        const mono = await document.fonts.load('14px "JetBrains Mono"');
        return [ui.length, mono.length, getComputedStyle(document.body).fontFamily,
            getComputedStyle(document.querySelector('code')).fontFamily];
    }""")
    assert website_fonts[0] > 0 and website_fonts[1] > 0
    assert "Google Sans" in website_fonts[2] and "JetBrains Mono" in website_fonts[3]
    assert not any(url.startswith(("https:", "http:")) for url in requests)
    for name in ("GoogleSans.woff2", "JetBrainsMono.woff2"):
        assert (root / "website/assets/fonts" / name).read_bytes() == (
            root / "src/seedl/assets/fonts" / name
        ).read_bytes()
    page.close()
