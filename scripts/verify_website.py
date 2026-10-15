"""Real Chromium verification of the static website, served under /seedl/."""

from __future__ import annotations

import io
import json
import os
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/website"


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if not self.path.startswith("/seedl/"):
            self.send_error(404)
            return
        self.path = self.path[len("/seedl") :]
        super().do_GET()

    def log_message(self, *_args):
        pass


def ready(frame):
    frame.wait_for_function(
        """() => { const gd = document.getElementById('seedl-demo');
        return gd && gd._fullLayout && gd._fullLayout._plots &&
            (!gd._fullLayout.scene || gd._fullLayout.scene._scene.glplot); }"""
    )


def node_centers(png: bytes):
    """Find rendered marker centers, so the test clicks real canvas pixels."""
    image = Image.open(io.BytesIO(png)).convert("RGB")
    width, height = image.size
    pixels = image.load()
    points = set()
    for y in range(int(height * 0.34), int(height * 0.74)):
        for x in range(int(width * 0.18), int(width * 0.76)):
            r, g, b = pixels[x, y]
            if r > 90 and g > 120 and b > 110:
                points.add((x, y))
    centers = []
    while points:
        stack = [points.pop()]
        component = []
        while stack:
            x, y = stack.pop()
            component.append((x, y))
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor in points:
                    points.remove(neighbor)
                    stack.append(neighbor)
        if len(component) > 70:
            centers.append(
                tuple(sum(point[i] for point in component) / len(component) for i in (0, 1))
            )
    return sorted(centers)


def playback_ready(frame):
    frame.wait_for_function(
        """() => {const gd=document.querySelector('.plotly-graph-div');
        return gd?.seedlPlayback && gd._fullLayout?.scene?._scene?.glplot;}"""
    )


def wait_index(frame, index):
    frame.wait_for_function(
        "i => document.querySelector('.plotly-graph-div').dataset.seedlIndex === String(i)",
        arg=index,
    )


def close_values(actual, expected):
    assert len(actual) == len(expected)
    assert all(abs(a - b) < 1e-10 for a, b in zip(actual, expected, strict=True)), (
        actual,
        expected,
    )


def verify_recorded_demos(page):
    manifest = json.loads((ROOT / "website/demos/manifest.json").read_text())
    page.locator("#tab-training").click()
    training = page.locator("#training iframe").element_handle().content_frame()
    playback_ready(training)
    panel = training.locator(".seedl-playback")
    snapshots = manifest["training"]["snapshots"]
    panel.locator('[data-control="neuron"]').select_option("1:0")
    initial_card = panel.locator("[data-inspection]").inner_text()
    assert "Activation" in initial_card and "gradient" in initial_card.lower()
    panel.locator('[data-action="next"]').click()
    wait_index(training, 1)
    assert panel.locator("[data-inspection]").inner_text() != initial_card
    # Exact rendered activations and hover weight/gradient data come from the recording.
    for index in (1, 15):
        panel.locator('[data-control="step"]').select_option(str(index))
        wait_index(training, index)
        actual = training.evaluate("""() => document.querySelector('.plotly-graph-div').data
            .filter(t=>t.meta?.role==='nodes').map(t=>({ids:t.customdata, colors:t.marker.color}))""")
        nodes = {n["id"]: n for n in snapshots[index]["graph"]["nodes"]}
        for trace in actual:
            if isinstance(trace["colors"], list):
                close_values(trace["colors"], [nodes[id]["activation"] for id in trace["ids"]])
        texts = training.evaluate(
            "document.querySelector('.plotly-graph-div').data.flatMap(t=>t.text || []).join(' ')"
        )
        assert "Weight" in texts and "gradient" in texts.lower()
        for edge in snapshots[index]["graph"]["edges"]:
            label = f"{edge['source']} → {edge['target']}<br>Weight: {edge['weight']:.5g}<br>"
            assert label in texts
            assert f"Weight gradient: {edge['gradient']:.5g}" in texts
        card = panel.locator("[data-inspection]").inner_text()
        assert f"Bias gradient: {nodes['1:0']['gradient']:.5g}" in card
    loss = training.evaluate("document.querySelector('[data-loss]').data[0].y")
    close_values(loss, [s["loss"] for s in snapshots])
    assert training.evaluate("document.querySelector('[data-loss]').data[1].x") == [15]
    panel.locator('[data-action="previous"]').click()
    wait_index(training, 14)
    panel.locator('[data-control="timeline"]').focus()
    page.keyboard.press("Home")
    wait_index(training, 0)
    panel.locator('[data-control="epoch"]').select_option("8")
    wait_index(training, 8)
    panel.locator('[data-action="rewind"]').click()
    wait_index(training, 0)
    panel.locator('[data-control="speed"]').select_option("4")
    panel.locator('[data-action="play"]').click()
    training.wait_for_function(
        "document.querySelector('.plotly-graph-div').seedlPlayback.index >= 2"
    )
    panel.locator('[data-action="pause"]').click()
    training.wait_for_timeout(350)
    paused = training.evaluate("document.querySelector('.plotly-graph-div').seedlPlayback.index")
    training.wait_for_timeout(350)
    assert (
        training.evaluate("document.querySelector('.plotly-graph-div').seedlPlayback.index")
        == paused
    )
    # Physical pointer selection on the actual training network, not a dispatched Plotly event.
    panel.locator('[data-control="neuron"]').select_option("")
    graph = training.locator(".plotly-graph-div").first
    graph.scroll_into_view_if_needed()
    training.evaluate("""() => { window.neuronHover=null;
      document.querySelector('.plotly-graph-div').on('plotly_hover', e=>{
        const p=e.points[0]; if(p.data.meta?.role==='nodes') window.neuronHover=p.customdata;
      }); }""")
    canvas = training.locator("canvas").first.bounding_box()
    point = None
    for y in range(int(canvas["y"]) + 35, int(canvas["y"] + canvas["height"]) - 35, 18):
        for x in range(int(canvas["x"]) + 35, int(canvas["x"] + canvas["width"]) - 35, 18):
            page.mouse.move(x, y)
            hovered = training.evaluate("window.neuronHover")
            if hovered:
                point = x, y, hovered
                break
        if point:
            break
    assert point, "No real training neuron reachable by pointer"
    page.mouse.click(point[0], point[1])
    training.wait_for_function(
        'id => document.querySelector("[data-control=neuron]").value === id', arg=point[2]
    )
    graph.screenshot(path=str(ARTIFACTS / "training-network.png"))
    # Navigation pauses a running recording without replacing the library's controller.
    panel.locator('[data-action="play"]').click()
    page.locator("#tab-evolution").click()
    assert not training.evaluate(
        "document.querySelector('.plotly-graph-div').seedlPlayback.playing"
    )
    quantum = page.locator("#evolution iframe").element_handle().content_frame()
    playback_ready(quantum)
    quantum.wait_for_selector("[data-seedl-circuit] .point")
    gate = quantum.locator("[data-seedl-circuit] .scatterlayer .point").first
    gate.scroll_into_view_if_needed()
    gate.click(force=True)
    wait_index(quantum, 1)
    qp = quantum.locator(".seedl-playback")
    qp.locator('[data-control="step"]').select_option("2")
    wait_index(quantum, 2)
    bars = quantum.evaluate(
        "document.querySelector('.plotly-graph-div').data.filter(t=>t.type==='bar').map(t=>t.y)"
    )
    close_values(bars[0], [0.5, 0.5, 0, 0])
    close_values(bars[1], [0.5, 0, 0, 0.5])
    vectors = quantum.evaluate("""document.querySelector('.plotly-graph-div').data
        .filter(t=>t.type==='scatter3d' && t.mode==='lines+markers+text' && t.scene==='scene2')
        .map(t=>[t.x[1]-t.x[0],t.y[1],t.z[1]])""")
    assert len(vectors) == 2
    for vector in vectors:
        close_values(vector, [0, 0, 0])
    with quantum.expect_navigation(wait_until="load"):
        page.locator("#evolution-circuit").select_option("evolution-phase")
    quantum = page.locator("#evolution iframe").element_handle().content_frame()
    playback_ready(quantum)
    quantum.locator('[data-control="step"]').select_option("3")
    wait_index(quantum, 3)
    actual = quantum.evaluate("""document.querySelector('.plotly-graph-div').data
        .filter(t=>t.type==='bar').map(t=>({p:t.y, phase:t.marker.color, amplitudes:t.customdata}))""")
    expected = manifest["evolution"]["phase"]["evolution"][3]
    close_values(actual[1]["p"], expected["probabilities"])
    close_values(actual[1]["phase"], [a["phase"] for a in expected["amplitudes"]])
    quantum.locator(".plotly-graph-div").first.screenshot(
        path=str(ARTIFACTS / "quantum-evolution.png")
    )
    page.locator("#tab-comparison").click()
    comparison = page.locator("#comparison iframe").element_handle().content_frame()
    ready(comparison)
    data = comparison.evaluate("document.getElementById('seedl-demo').data")
    expected = manifest["training"]["comparison"]
    for row in expected["parameters"]:
        trace = next(t for t in data if t["name"] == row["name"])
        close_values(trace["y"], row["delta"])
    for side, name in [("before", "Step 0"), ("after", "Step 15")]:
        trace = next(t for t in data if t["name"] == name + " |gradient|max")
        close_values(trace["y"], [r["gradients"][side]["max_abs"] for r in expected["parameters"]])
        trace = next(t for t in data if t["name"] == name + " output")
        close_values(trace["y"], expected["outputs"][side])
    # Actual legend toggling and data hover on the comparison plot.
    comparison.locator(".legendtoggle").first.click()
    comparison.wait_for_function(
        "document.getElementById('seedl-demo').data[0].visible === 'legendonly'"
    )
    comparison.locator(".legendtoggle").first.click()
    comparison.locator(".scatterlayer .point").first.hover(force=True)
    assert comparison.locator(".hoverlayer").text_content().strip()
    comparison.locator("#seedl-demo").screenshot(path=str(ARTIFACTS / "comparison.png"))
    # Once loaded, all three demos remain interactive with the network disabled.
    page.context.set_offline(True)
    try:
        page.locator("#tab-training").click()
        panel.locator('[data-control="step"]').select_option("7")
        wait_index(training, 7)
        page.locator("#tab-evolution").click()
        quantum.locator('[data-control="step"]').select_option("1")
        wait_index(quantum, 1)
        page.locator("#tab-comparison").click()
        comparison.locator(".scatterlayer .point").first.hover(force=True)
        assert comparison.locator(".hoverlayer").text_content().strip()
    finally:
        page.context.set_offline(False)
    print(
        "PASS: training controls, real neuron click, loss/activations/weights/gradients; quantum gate click, Bell probabilities/Bloch vectors/phases; comparison deltas, gradients, outputs and legend.",
        flush=True,
    )


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(Handler, directory=str(ROOT / "website"))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/seedl/"
    errors = []
    requests = []
    try:
        with sync_playwright() as playwright:
            options = {"headless": True, "args": ["--enable-unsafe-swiftshader"]}
            if os.environ.get("SEEDL_BROWSER"):
                options["executable_path"] = os.environ["SEEDL_BROWSER"]
            browser = playwright.chromium.launch(**options)
            context = browser.new_context(
                viewport={"width": 1440, "height": 1000}, reduced_motion="reduce"
            )
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
            page.on(
                "response",
                lambda response: (
                    errors.append(f"HTTP {response.status}: {response.url}")
                    if response.status >= 400
                    else None
                ),
            )
            page.on(
                "requestfailed", lambda request: errors.append(f"Request failed: {request.url}")
            )
            page.on("request", lambda request: requests.append(request.url))
            page.goto(base)
            page.locator(".quantum-card img").last.scroll_into_view_if_needed()
            assert all(
                page.locator("img").evaluate_all(
                    "(images) => images.map(i => i.complete && i.naturalWidth > 0)"
                )
            )
            assert not any("plotly.min.js" in url for url in requests), (
                "Homepage loaded the heavy demo runtime"
            )
            page.get_by_role("button", name="Use light theme").click()
            page.reload()
            assert page.locator("html").get_attribute("data-theme") == "light"
            page.get_by_role("button", name="Use dark theme").click()
            page.evaluate("scrollTo(0, 0)")
            page.screenshot(path=str(ARTIFACTS / "desktop.png"), full_page=True)
            page.locator("[data-preview]").click()
            preview = page.locator(".preview-live iframe").element_handle().content_frame()
            ready(preview)
            page.locator(".hero a.primary").click()
            assert page.url.endswith("docs/index.html#quickstart")
            # All documentation navigation targets exist and are visible when followed.
            for anchor in page.locator(".docnav a").all():
                anchor.click()
                assert page.locator(anchor.get_attribute("href")).is_visible()
            page.get_by_role("link", name="Playground", exact=True).click()
            print(
                "PASS: homepage resources, theme persistence, quickstart navigation, documentation anchors.",
                flush=True,
            )
            iframe = page.locator("#network iframe")
            frame = iframe.element_handle().content_frame()
            ready(frame)
            iframe.scroll_into_view_if_needed()
            frame.wait_for_timeout(350)
            graph = frame.locator("#seedl-demo")
            points = node_centers(graph.screenshot())
            box = graph.bounding_box()
            assert points, "No rendered neuron markers found"
            clicked = False
            for x, y in points:
                page.mouse.move(box["x"] + x, box["y"] + y)
                page.wait_for_timeout(100)
                page.mouse.click(box["x"] + x, box["y"] + y)
                try:
                    frame.wait_for_function(
                        """() => ['highlight_incoming', 'highlight_outgoing'].every(role =>
                        document.getElementById('seedl-demo').data.some(t => t.meta?.role === role && t.visible && t.x.length))""",
                        timeout=1200,
                    )
                    clicked = True
                    break
                except Exception:
                    continue
            assert clicked, (
                "Physical canvas click did not highlight incoming and outgoing connections"
            )
            assert page.locator("#neuron").input_value(), (
                "Selection did not reach accessible inspector"
            )
            assert frame.evaluate(
                "document.getElementById('seedl-demo').layout.annotations.at(-1).text.includes('Bias')"
            )
            camera_before = frame.evaluate(
                "JSON.stringify(document.getElementById('seedl-demo')._fullLayout.scene.camera)"
            )
            page.locator("#neuron").select_option("1:2")
            frame.wait_for_function(
                "document.getElementById('seedl-demo').layout.annotations.at(-1).text.includes('1:2')"
            )
            assert (
                frame.evaluate(
                    "JSON.stringify(document.getElementById('seedl-demo')._fullLayout.scene.camera)"
                )
                == camera_before
            )
            # Physical orbit and zoom gestures must change the actual Plotly camera.
            canvas = frame.locator("canvas").first.bounding_box()
            cx, cy = canvas["x"] + canvas["width"] * 0.6, canvas["y"] + canvas["height"] * 0.6
            page.mouse.move(cx, cy)
            page.mouse.down()
            page.mouse.move(cx + 75, cy + 30, steps=8)
            page.mouse.up()
            frame.wait_for_function(
                "JSON.stringify(document.getElementById('seedl-demo')._fullLayout.scene.camera) !== "
                + json.dumps(camera_before)
            )
            # Plotly's serialized layout can lag wheel zoom. Inspect the actual
            # WebGL camera eye used to render, rather than that layout snapshot.
            camera_rotated = frame.evaluate(
                "JSON.stringify(document.getElementById('seedl-demo')._fullLayout.scene._scene.glplot.camera.eye)"
            )
            page.mouse.wheel(0, -180)
            frame.wait_for_function(
                "JSON.stringify(document.getElementById('seedl-demo')._fullLayout.scene._scene.glplot.camera.eye) !== "
                + json.dumps(camera_rotated)
            )
            with frame.expect_navigation(wait_until="load"):
                page.locator("#reset-network").click()
            frame = iframe.element_handle().content_frame()
            ready(frame)
            assert frame.evaluate(
                "document.getElementById('seedl-demo').data.every(t => !t.meta?.role?.startsWith('highlight') || !t.visible)"
            )
            assert page.locator("#neuron").input_value() == ""
            # Keyboard-operated tab list.
            page.locator("#tab-network").focus()
            page.keyboard.press("ArrowRight")
            assert page.locator("#tab-training").get_attribute("aria-selected") == "true"
            page.keyboard.press("ArrowRight")
            assert page.locator("#tab-bloch").get_attribute("aria-selected") == "true"
            bloch = page.locator("#bloch iframe").element_handle().content_frame()
            ready(bloch)
            expected = {
                "zero": [0, 0, 1],
                "one": [0, 0, -1],
                "plus": [1, 0, 0],
                "minus": [-1, 0, 0],
                "plus-i": [0, 1, 0],
                "minus-i": [0, -1, 0],
            }
            for slug, vector in expected.items():
                with bloch.expect_navigation(wait_until="load"):
                    page.locator(f'[data-state="{slug}"]').click()
                page.wait_for_function(
                    "document.querySelector('#bloch iframe').src.endsWith("
                    + json.dumps(f"bloch-{slug}.html")
                    + ")"
                )
                bloch = page.locator("#bloch iframe").element_handle().content_frame()
                ready(bloch)
                arrow = bloch.evaluate("""() => { const traces = document.getElementById('seedl-demo').data;
                    const tip = traces.at(-1); return [tip.x[0], tip.y[0], tip.z[0]]; }""")
                assert all(abs(a - b) < 1e-12 for a, b in zip(arrow, vector, strict=True)), (
                    slug,
                    arrow,
                )
                assert (
                    page.locator("#bloch-vector").inner_text()
                    == "(" + ", ".join(str(v) for v in vector) + ")"
                )
            page.locator("#tab-circuit").click()
            circuit = page.locator("#circuit iframe").element_handle().content_frame()
            ready(circuit)
            assert circuit.locator(".main-svg").count() > 0
            assert "H" in circuit.locator("#seedl-demo").inner_text()
            assert "q_0" in circuit.locator("#seedl-demo").inner_text()
            # Hover a real gate marker and check the renderer's inspection tooltip.
            points = circuit.locator(".scatterlayer .point").all()
            assert points, "No circuit gate markers rendered"
            hovered = False
            for point in points:
                point.hover(force=True)
                if (circuit.locator(".hoverlayer").text_content() or "").strip():
                    hovered = True
                    break
            assert hovered, "Circuit gate inspection tooltip missing"
            print(
                "PASS: physical neuron click, highlights, inspector, camera orbit/zoom/reset, six Bloch vectors, Bell rendering/hover."
            )
            verify_recorded_demos(page)
            page.set_viewport_size({"width": 390, "height": 844})
            for route in ("index.html", "docs/index.html", "playground.html#network"):
                page.goto(base + route)
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (
                    f"Mobile overflow: {route}"
                )
                page.get_by_role("button", name="Toggle navigation").click()
                assert page.locator(".navlinks").is_visible()
                page.keyboard.press("Escape")
                assert not page.locator(".navlinks").is_visible()
            frame = page.locator("#network iframe").element_handle().content_frame()
            ready(frame)
            page.locator("#neuron").select_option("1:1")
            frame.wait_for_function(
                "document.getElementById('seedl-demo').layout.annotations.at(-1).text.includes('1:1')"
            )
            page.screenshot(path=str(ARTIFACTS / "mobile.png"), full_page=True)
            for name in ("training", "evolution", "comparison"):
                page.locator(f"#tab-{name}").click()
                iframe = page.locator(f"#{name} iframe")
                frame = iframe.element_handle().content_frame()
                if name == "comparison":
                    ready(frame)
                else:
                    playback_ready(frame)
                    panel = frame.locator(".seedl-playback")
                    panel.locator('[data-action="next"]').click()
                    wait_index(frame, 1)
                    assert frame.evaluate("document.documentElement.scrollWidth <= innerWidth")
                    assert frame.evaluate(
                        "document.querySelector('.plotly-graph-div')._fullLayout.width >= 700"
                    ), "Mobile plots must retain readable axes and panels"
                    if name == "evolution":
                        assert not panel.locator('[data-control="neuron"]').is_visible()
                        assert not panel.locator('[data-control="epoch"]').is_visible()
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                page.screenshot(path=str(ARTIFACTS / f"mobile-{name}.png"), full_page=True)
                frame.locator(".plotly-graph-div").first.scroll_into_view_if_needed()
                page.screenshot(path=str(ARTIFACTS / f"mobile-{name}-plot.png"))
            assert not errors, "\n".join(errors)
            assert all(
                url.startswith(base) or url.startswith("data:") or url.startswith("blob:")
                for url in requests
            ), "Unexpected remote resource"
            print(
                "PASS: /seedl/ paths, navigation, docs anchors, theme persistence, lazy runtime, mobile controls, zero browser/HTTP errors."
            )
            browser.close()
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
