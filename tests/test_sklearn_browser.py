"""Actual offline browser verification of classical ML output and data provenance."""

from __future__ import annotations

# ruff: noqa: E402
import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import seedl as sd


@pytest.fixture(scope="module")
def browser():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:
            pytest.skip(f"Chromium unavailable: {exc}")
        yield b
        b.close()


@pytest.fixture(scope="module")
def models():
    data = load_breast_cancer()
    Xt, X, yt, y = train_test_split(data.data, data.target, stratify=data.target, random_state=0)
    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xt, yt)
    regression = load_diabetes()
    Rt, R, rt, r = train_test_split(regression.data, regression.target, random_state=0)
    return classifier, X, y, LinearRegression().fit(Rt, rt), R, r


@pytest.mark.parametrize("width", [320, 390, 768, 1280])
@pytest.mark.parametrize("backend", ["svg", "plotly"])
def test_responsive_offline_evaluation_with_safe_labels(browser, models, tmp_path, width, backend):
    if backend == "plotly":
        pytest.importorskip("plotly")
    classifier, X, y, regressor, R, r = models
    malicious = 'malignant <img src=x onerror="injected=1"> </script>'
    result = sd.evaluate(
        classifier,
        X,
        y,
        split="test",
        class_names={0: malicious, 1: "benign"},
        positive_label=0,
        backend=backend,
        max_points=25,
    )
    path = result.save(tmp_path / "classification.html")
    page = browser.new_page(viewport={"width": width, "height": 900})
    requests, errors, dialogs = [], [], []
    page.on("request", lambda req: requests.append(req.url))
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
    page.goto(path.as_uri())
    assert page.locator(".ml-metrics").is_visible()
    assert page.locator(".ml-metrics dd").count() == len(result.metrics)
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert page.locator("img[src=x]").count() == 0
    assert page.evaluate("window.injected") is None
    page.locator("#seedl-ml-data summary").click()
    captured = json.loads(page.locator("#seedl-ml-data pre").inner_text())
    assert captured == result.data
    page.locator("#seedl-ml-data summary").click()
    assert malicious in page.locator(".ml-panel").first.inner_text()
    if backend == "plotly":
        page.wait_for_selector(".plotly-graph-div")
        assert (
            page.locator(".plotly-graph-div").first.evaluate("el=>el.getBoundingClientRect().width")
            <= width
        )
    for family in ("Google Sans", "JetBrains Mono"):
        assert (
            page.evaluate(
                "async name => (await document.fonts.load('14px \\\"'+name+'\\\"')).length", family
            )
            > 0
        )
    regression_result = sd.evaluate(regressor, R, r, split="test", backend=backend, max_points=25)
    reg_path = regression_result.save(tmp_path / "regression.html")
    page.goto(reg_path.as_uri())
    assert "rmse" in page.locator(".ml-metrics").inner_text().lower()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert not errors and not dialogs
    assert not any(url.startswith(("http:", "https:")) for url in requests)
    page.close()


def test_serialization_report_and_full_precision(models, tmp_path):
    classifier, X, y, regressor, R, r = models
    result = sd.evaluate(regressor, R, r, backend="svg")
    saved = sd.save_analysis(result, tmp_path / "analysis.json")
    loaded = sd.load_analysis(saved)
    assert loaded.data == result.data
    doc = loaded.to_html()
    assert repr(result.metrics["rmse"]) in doc
    report = sd.report(title="Classical ML report", views=[result])
    report.save(tmp_path / "report.html")
    report_text = (tmp_path / "report.html").read_text()
    assert "sandbox=" in report_text and "ML data" in report_text
