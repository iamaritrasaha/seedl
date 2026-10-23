"""Actual rendered playback controls in Playwright Chromium, offline and kernel-free."""

from pathlib import Path

import pytest
import torch
from qiskit import QuantumCircuit
from torch import nn

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


def training_view():
    torch.manual_seed(41)
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.15)
    with sd.watch(model, name="Browser training") as run:
        for i in range(10):
            optimizer.zero_grad()
            y = model(torch.tensor([[1.0, -2.0]]))
            loss = (y - 0.5).square().mean()
            loss.backward()
            run.capture(step=i * 3, epoch=i // 2, loss=loss)
            optimizer.step()
    return run, run.playback(backend="plotly")


def wait_index(page, index):
    page.wait_for_function(
        '(index) => document.querySelector(".plotly-graph-div").dataset.seedlIndex === String(index)',
        arg=index,
    )


def test_training_controls_inspection_camera_offline(browser, tmp_path):
    run, view = training_view()
    target = view.save(tmp_path / "training.html")
    page = browser.new_page(viewport={"width": 1400, "height": 1100})
    errors, requests = [], []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("request", lambda request: requests.append(request.url))
    page.goto(target.as_uri())
    page.wait_for_selector(".seedl-playback")
    page.wait_for_function(
        'document.querySelector(".plotly-graph-div").seedlPlayback !== undefined'
    )
    panel = page.locator(".seedl-playback")
    panel.locator('[data-control="neuron"]').select_option("1:0")
    assert "1:0" in panel.locator("[data-inspection]").inner_text()
    initial_card = panel.locator("[data-inspection]").inner_text()
    page.evaluate(
        "Plotly.relayout(document.querySelector('.plotly-graph-div'), {'scene.camera.eye': {x:2.3,y:-1.1,z:1.7}})"
    )
    panel.locator('[data-action="next"]').click()
    wait_index(page, 1)
    assert panel.locator("[data-inspection]").inner_text() != initial_card
    camera = page.evaluate('document.querySelector(".plotly-graph-div").layout.scene.camera.eye')
    assert camera == pytest.approx({"x": 2.3, "y": -1.1, "z": 1.7})
    # The rendered trace values change to the exact recorded pass.
    actual = page.evaluate(
        """() => {const gd=document.querySelector('.plotly-graph-div');const t=gd.data.find(t=>t.meta && t.meta.role==='nodes' && t.meta.layer===1);return {ids:t.customdata, values:t.marker.color};}"""
    )
    nodes = {n["id"]: n for n in run.snapshots[1].data["graph"]["nodes"]}
    assert actual["values"] == pytest.approx([nodes[id]["activation"] for id in actual["ids"]])
    assert "Step 3" in panel.locator("[data-status]").inner_text()
    panel.locator('[data-action="previous"]').click()
    wait_index(page, 0)
    panel.locator('[data-control="epoch"]').select_option("6")
    wait_index(page, 6)
    panel.locator('[data-control="timeline"]').focus()
    page.keyboard.press("End")
    wait_index(page, 9)
    panel.locator('[data-action="rewind"]').click()
    wait_index(page, 0)
    panel.locator('[data-control="speed"]').select_option("4")
    panel.locator('[data-action="play"]').click()
    page.wait_for_function('document.querySelector(".plotly-graph-div").seedlPlayback.index >= 2')
    panel.locator('[data-action="pause"]').click()
    page.wait_for_timeout(500)
    paused = page.evaluate('document.querySelector(".plotly-graph-div").seedlPlayback.index')
    page.wait_for_timeout(500)
    assert (
        page.evaluate('document.querySelector(".plotly-graph-div").seedlPlayback.index') == paused
    )
    assert not page.evaluate('document.querySelector(".plotly-graph-div").seedlPlayback.playing')
    # Actual pointer click on a rendered WebGL node, after playback updates.
    panel.locator('[data-control="neuron"]').select_option("")
    page.evaluate(
        """() => {window.liveNodeHover=null;document.querySelector('.plotly-graph-div').on('plotly_hover', e=>{const p=e.points[0];if(p.data.meta && p.data.meta.role==='nodes') window.liveNodeHover=p.customdata;});}"""
    )
    rect = page.locator("canvas").first.bounding_box()
    point = None
    for y in range(int(rect["y"]) + 40, int(rect["y"] + rect["height"]) - 40, 18):
        for x in range(int(rect["x"]) + 40, int(rect["x"] + rect["width"]) - 40, 18):
            page.mouse.move(x, y)
            hovered = page.evaluate("window.liveNodeHover")
            if hovered:
                point = x, y, hovered
                break
        if point:
            break
    assert point, "A real WebGL neuron must be reachable by the pointer"
    page.mouse.click(point[0], point[1])
    page.wait_for_function(
        '(id)=>document.querySelector("[data-inspection]").textContent.includes(id)', arg=point[2]
    )
    assert not errors
    assert all(url.startswith(("file:", "data:", "blob:")) for url in requests)
    folder = Path("/tmp/seedl-live-verification")
    folder.mkdir(exist_ok=True)
    page.screenshot(path=str(folder / "training.png"), full_page=True)
    page.close()


def test_quantum_gate_selection_bloch_phase_and_halt(browser, tmp_path):
    qc = QuantumCircuit(2, 1)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.7, 1)
    qc.measure(0, 0)
    qc.x(1)
    view = sd.circuit(qc, simulate=True, backend="plotly")
    target = view.save(tmp_path / "quantum.html")
    page = browser.new_page(viewport={"width": 1300, "height": 1050})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(target.as_uri())
    page.wait_for_selector(".seedl-playback")
    panel = page.locator(".seedl-playback")
    # Select a gate by clicking its actual SVG marker in the preserved circuit diagram.
    gate = page.locator("[data-seedl-circuit] .scatterlayer .point").first
    gate.scroll_into_view_if_needed()
    gate_rect = gate.bounding_box()
    page.mouse.click(
        gate_rect["x"] + gate_rect["width"] / 2, gate_rect["y"] + gate_rect["height"] / 2
    )
    wait_index(page, 1)
    panel.locator('[data-control="step"]').select_option("2")
    wait_index(page, 2)
    values = page.evaluate(
        """() => document.querySelector('.plotly-graph-div').data.filter(t=>t.type==='bar').map(t=>({p:t.y, phases:t.marker.color, amplitudes:t.customdata}))"""
    )
    assert len(values) == 2
    assert values[0]["p"] == pytest.approx([0.5, 0.5, 0, 0])
    assert values[1]["p"] == pytest.approx([0.5, 0, 0, 0.5])
    panel.locator('[data-action="next"]').click()
    wait_index(page, 3)
    phases = page.evaluate(
        "document.querySelector('.plotly-graph-div').data.filter(t=>t.type==='bar')[1].marker.color"
    )
    assert phases[0] == pytest.approx(-0.35)
    assert phases[3] == pytest.approx(0.35)
    panel.locator('[data-action="next"]').click()
    wait_index(page, 4)
    assert (
        page.evaluate(
            "document.querySelector('.plotly-graph-div').data.filter(t=>t.type==='bar').length"
        )
        == 1
    )
    panel.locator('[data-action="next"]').click()
    wait_index(page, 5)
    assert page.evaluate("document.querySelector('.plotly-graph-div').data.length") == 0
    assert "unavailable" in page.locator(".plotly-graph-div").inner_text()
    panel.locator('[data-action="rewind"]').click()
    wait_index(page, 0)
    panel.locator('[data-control="step"]').select_option("3")
    wait_index(page, 3)
    folder = Path("/tmp/seedl-live-verification")
    folder.mkdir(exist_ok=True)
    page.screenshot(path=str(folder / "quantum.png"), full_page=True)
    assert not errors
    page.close()


def test_notebook_mime_multiple_views_are_independent(browser, tmp_path):
    _, view = training_view()
    first, second = view._repr_mimebundle_()["text/html"], view._repr_mimebundle_()["text/html"]
    target = tmp_path / "notebook-output.html"
    target.write_text(
        "<!DOCTYPE html><html><body>" + first + second + "</body></html>", encoding="utf-8"
    )
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(target.as_uri())
    page.wait_for_selector(".seedl-playback")
    assert page.locator(".seedl-playback").count() == 2
    page.locator(".seedl-playback").first.locator('[data-action="next"]').click()
    page.wait_for_function(
        'document.querySelectorAll(".plotly-graph-div")[0].dataset.seedlIndex === "1"'
    )
    assert page.evaluate(
        'Array.from(document.querySelectorAll(".plotly-graph-div")).filter(g=>g.seedlPlayback).map(g=>g.seedlPlayback.index)'
    ) == [1, 0]
    assert not errors
    page.close()


def test_playback_label_serialization_is_literal(browser, tmp_path):
    _, view = training_view()
    view.title = "__TRAINING__ __CIRCUIT__ </sCrIpT><script>window.seedlInjected=true</script>"
    target = view.save(tmp_path / "literal-labels.html")
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(target.as_uri())
    page.wait_for_selector(".seedl-playback")
    page.locator('[data-action="next"]').click()
    wait_index(page, 1)
    title = page.evaluate('document.querySelector(".plotly-graph-div").layout.title.text')
    assert "__TRAINING__ __CIRCUIT__" in title
    assert page.evaluate("window.seedlInjected === undefined")
    assert not errors
    page.close()


def test_export_reports_evictions_and_sampling_in_the_rendered_panel(browser, tmp_path):
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
    with sd.watch(model, max_values=3, max_connections=2, max_snapshots=1) as run:
        for step in range(2):
            model(torch.ones(1, 2)).sum().backward()
            run.capture(step=step)
    page = browser.new_page()
    page.goto(run.playback().save(tmp_path / "coverage.html").as_uri())
    page.wait_for_selector(".seedl-playback")
    assert "1 evicted" in page.locator("[data-recording]").inner_text()
    assert "1 retained snapshots" in page.locator("[data-recording]").inner_text()
    coverage = page.locator("[data-coverage]").inner_text()
    assert "Parameter values retained: 3/13" in coverage
    assert "Output values retained: 1/1" in coverage
    assert "Connections retained: 2/9" in coverage
    page.close()
