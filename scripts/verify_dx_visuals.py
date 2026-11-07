"""Capture reproducible DX screenshots. Requires numpy, torch and playwright.

Run from the repository: python scripts/verify_dx_visuals.py --output /tmp/seedl-dx
For a before comparison, set PYTHONPATH to an unmodified checkout's src directory.
No data downloads, browser launches outside headless Chromium, or model training.
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import numpy as np
import torch
from playwright.sync_api import sync_playwright

import seedl as sd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Authentic UCI Iris excerpt, first five setosa observations; all columns in cm.
    iris = np.array(
        [
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [4.7, 3.2, 1.3, 0.2],
            [4.6, 3.1, 1.5, 0.2],
            [5.0, 3.6, 1.4, 0.2],
        ]
    )
    with (
        (root / "terminal.txt").open("w", encoding="utf-8") as stream,
        contextlib.redirect_stdout(stream),
    ):
        views = {
            "matrix": sd.see(iris.tolist(), backend="svg"),
            "tensor": sd.see(
                iris.reshape(1, 5, 4),
                interactive=True,
                dim_names=["batch", "observation", "measurement"],
                backend="svg",
            ),
        }
        views["matrix"].title = "Iris setosa measurements (cm)"
        views["matrix"].notes.append(
            "UCI Iris dataset (CC BY 4.0), first five observations; columns: sepal length, sepal width, petal length, petal width. Source: https://archive.ics.uci.edu/dataset/53/iris"
        )
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 1))
        x = torch.tensor(iris, dtype=torch.float32)
        views["network"] = sd.see(model, input=x, backend="svg")
        views["activations"] = sd.activations(model, input=x, backend="svg")
        model(x).sum().backward()
        views["gradients"] = sd.gradient_health(model, backend="svg")
        # Analytic state and explicitly synthetic attention, alongside measured Iris data.
        views["quantum"] = sd.bloch("+", backend="svg")
        views["attention"] = sd.attention(
            [[0.8, 0.2], [0.3, 0.7]], tokens=["query", "key"], backend="svg"
        )
        for name, view in views.items():
            view.save(root / f"{name}.html")
            view.save(root / f"{name}.svg")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for width in (320, 390, 768, 1280):
            page = browser.new_page(viewport={"width": width, "height": 900})
            for name in (
                "matrix",
                "tensor",
                "gradients",
                "network",
                "activations",
                "attention",
                "quantum",
            ):
                page.goto((root / f"{name}.html").as_uri())
                page.screenshot(path=str(root / f"{name}-{width}.png"), full_page=True)
                print(
                    name,
                    width,
                    page.evaluate(
                        "({viewport:innerWidth,scroll:document.documentElement.scrollWidth})"
                    ),
                )
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
