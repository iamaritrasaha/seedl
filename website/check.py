"""Check built assets, internal links, science provenance, and Python doc snippets."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent


class Page(HTMLParser):
    def __init__(self, text: str):
        super().__init__()
        self.links = []
        self.ids = set()
        self.snippets = []
        self.in_code = False
        self.code = ""
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for attr in ("src", "href"):
            if attrs.get(attr):
                self.links.append(attrs[attr])
        if tag == "pre":
            self.in_code = True
            self.code = ""

    def handle_endtag(self, tag):
        if tag == "pre":
            self.in_code = False
            self.snippets.append(self.code)

    def handle_data(self, data):
        if self.in_code:
            self.code += data


def main() -> None:
    paths = [ROOT / "index.html", ROOT / "playground.html", ROOT / "docs/index.html"]
    pages = {path: Page(path.read_text()) for path in paths}
    checked = 0
    for path, page in pages.items():
        for link in page.links:
            parts = urlsplit(link)
            if parts.scheme or parts.netloc:
                continue
            assert not parts.path.startswith("/"), f"Root-relative link breaks /seedl/: {link}"
            target = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            assert target.is_relative_to(ROOT), f"Link escapes site: {link}"
            assert target.is_file(), f"Missing asset: {path.name}: {link}"
            if parts.fragment:
                assert target in pages and parts.fragment in pages[target].ids, (
                    f"Missing anchor: {link}"
                )
            checked += 1
    manifest = json.loads((ROOT / "demos/manifest.json").read_text())
    assert manifest["seedl"] == "0.2.0"
    assert len(manifest["bloch"]) == 6
    for slug, data in manifest["bloch"].items():
        assert math.isclose(sum(n * n for n in data["vector"]), 1, abs_tol=1e-12)
        assert math.isclose(data["prob_0"] + data["prob_1"], 1, abs_tol=1e-12)
        assert (ROOT / f"demos/bloch-{slug}.html").is_file()
    assert [i["name"] for i in manifest["circuit"]["instructions"]] == ["h", "cx"]
    assert manifest["circuit"]["instructions"][1]["qubits"] == [0, 1]
    assert all(
        math.isclose(a, b, abs_tol=1e-12)
        for a, b in zip(
            manifest["bell"]["entanglement"]["probabilities"], [0.5, 0.0, 0.0, 0.5], strict=True
        )
    )
    assert math.isclose(manifest["bell"]["entanglement"]["entropy_q0"], 1, abs_tol=1e-12)
    for name, digest in manifest["sha256"].items():
        path = ROOT / "demos" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
        html = path.read_text()
        assert html.count('src="../assets/plotly.min.js"') == 1
        assert "cdn.plot.ly" not in html
    assert sum(p.stat().st_size for p in paths) < 50_000, "Initial HTML has grown unexpectedly"
    # Execute visible Python examples as a reader would, preserving the model/x
    # variables shared by consecutive documentation examples. Exports go to temp.
    count = 0
    previous = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="seedl-docs-") as temp:
            os.chdir(temp)
            for path, page in pages.items():
                namespace = {"__name__": "__website_example__"}
                for snippet in page.snippets:
                    if "import " not in snippet and not snippet.startswith("view = sd.see"):
                        continue
                    with contextlib.redirect_stdout(io.StringIO()):
                        exec(compile(snippet, str(path), "exec"), namespace)
                    count += 1
    finally:
        os.chdir(previous)
    print(
        f"PASS: {checked} local links/assets, 8 demo hashes, Bell/Bloch science, {count} Python examples."
    )


if __name__ == "__main__":
    main()
