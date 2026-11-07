"""Public discovery, terminal behavior and presentation regressions."""

from __future__ import annotations

import copy
import importlib.metadata
import io
import json
import os
import subprocess
import sys
import types
import xml.etree.ElementTree as ET

import pytest

import seedl as sd
from seedl._discovery import EXAMPLES, GROUPS
from seedl._presentation import color_at, format_value, write_terminal
from seedl.cli import main


def run_cli(*args: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "seedl", *args],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        check=False,
    )


@pytest.mark.parametrize(
    "args", [(), ("--help",), ("help",), ("help", "see"), ("list",), ("doctor",), ("examples",)]
)
def test_cli_read_only_commands(args, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "seedl", *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()
    assert "\x1b" not in result.stdout
    assert not list(tmp_path.iterdir())


def test_console_script_metadata():
    entry = next(
        e for e in importlib.metadata.distribution("seedl").entry_points if e.name == "seedl"
    )
    assert entry.group == "console_scripts"
    assert entry.value == "seedl.cli:main"


@pytest.mark.parametrize("command", [[], ["help", "see"], ["list"], ["doctor"], ["examples"]])
def test_discovery_does_not_import_frameworks(command):
    script = f"import sys; from seedl.cli import main; main({command!r}); assert not any(n in sys.modules for n in ('torch','numpy','qiskit','plotly','matplotlib'))"
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("width", [24, 40, 80])
def test_cli_narrow_and_plain(width):
    result = run_cli(
        "--plain", COLUMNS=str(width), NO_COLOR="1", TERM="dumb", PYTHONIOENCODING="ascii"
    )
    assert result.returncode == 0
    assert "\x1b" not in result.stdout
    assert max(map(len, result.stdout.splitlines())) <= width
    assert "seedl" in result.stdout


@pytest.mark.parametrize(
    "args,expected",
    [
        (["hlep"], "Did you mean 'help'"),
        (["help", "se"], "Did you mean 'see'"),
        (["help", "__dict__"], "seedl list"),
        (["demo", "--output", "wrong.png"], ".html or .svg"),
    ],
)
def test_cli_invalid_input(args, expected):
    result = run_cli(*args)
    assert result.returncode == 2
    assert expected in result.stderr
    assert "Traceback" not in result.stderr


def test_doctor_missing_optional_dependencies(monkeypatch, capsys):
    monkeypatch.setattr("seedl.cli.importlib.util.find_spec", lambda _: None)
    assert main(["doctor"]) == 0
    text = capsys.readouterr().out
    assert text.count("missing; install:") == 6
    assert "without extras" in text
    assert "seedl[quantum]" in text


def test_guide_covers_public_operations(capsys):
    names = {name for group in GROUPS.values() for name in group}
    public_operations = {name for name in sd.__all__ if name[0].islower() and name != "guide"}
    assert names == public_operations
    text = sd.guide()
    assert capsys.readouterr().out == ""
    assert all(f"sd.{name}" in text for name in names)
    assert "max_points" in sd.guide("see")
    assert "Runnable Python example" in sd.guide("activations")


def test_demo_provenance_and_no_overwrite(tmp_path, monkeypatch, capsys):
    import webbrowser

    monkeypatch.setattr(webbrowser, "open", lambda *a, **k: pytest.fail("browser launch"))
    path = tmp_path / "demo.html"
    assert main(["demo", "--output", str(path)]) == 0
    content = path.read_text()
    assert "UCI Iris" in content and "centimeters" in content and "sepal length" in content
    assert "CC BY 4.0" in content
    assert '<script src="http' not in content
    with pytest.raises(SystemExit) as caught:
        main(["demo", "--output", str(path)])
    assert caught.value.code == 2
    assert path.read_text() == content
    assert "already exists" in capsys.readouterr().err
    path_svg = tmp_path / "demo.svg"
    main(["demo", "--output", str(path_svg)])
    ET.fromstring(path_svg.read_text())


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "unavailable"),
        (float("nan"), "NaN"),
        (float("inf"), "+Inf"),
        (-float("inf"), "-Inf"),
        (1e-200, "1e-200"),
        (1e200, "1e+200"),
    ],
)
def test_numeric_presentation(value, expected):
    assert format_value(value) == expected


def test_text_findings_precision_and_data_preserved():
    view = sd.Visualization(
        "activations",
        "Activation diagnostics",
        "One measured layer",
        {
            "layers": [
                {
                    "name": "layer",
                    "shape": (1, 2),
                    "mean": 1e-200,
                    "std": None,
                    "max_abs": float("inf"),
                }
            ]
        },
        metadata={"Shape": (1, 2), "Tiny statistic": 1.234567891234e-100},
        notes=[
            "Sampled preview; 2 of 100 values captured.",
            "Saturation detected in one layer (heuristic).",
            "2 non-finite values present: NaN and Inf.",
        ],
    )
    original = copy.deepcopy(view.data)
    result = view.to_text(width=48, plain=True)
    assert result.index("WARNING") < result.index("CAUTION") < result.index("Summary")
    assert "NaN means undefined" in result
    assert "1e-200" in result and "+Inf" in result and "unavailable" in result
    assert max(map(len, result.splitlines())) <= 48
    assert "Sampled preview" in result
    assert "1.234567891234e-100" in view.to_text(detail="full")
    assert view.data == original
    assert str(view) == view.to_text()
    with pytest.raises(ValueError):
        view.to_text(detail="invalid")
    with pytest.raises(ValueError):
        view.to_text(width=5)


def test_text_full_metadata_and_rows_are_discoverable():
    view = sd.Visualization(
        "activations",
        "Layers",
        "20 layers",
        {"layers": [{"name": f"layer-{i}", "mean": float(i)} for i in range(20)]},
        metadata={f"field-{i}": i for i in range(20)},
    )
    assert "Showing 6 of 20 layers" in view.to_text()
    assert "layer-19" in view.to_text(detail="full")
    assert "field-19" in view.to_text(detail="full")


def test_windows_safe_output_and_controls():
    buffer = io.BytesIO()
    output = io.TextIOWrapper(buffer, encoding="cp1252")
    write_terminal("Tensor × 3 → |ψ⟩ ∞", output)
    output.flush()
    text = buffer.getvalue().decode("cp1252")
    assert "Tensor x 3 ->" in text and "Inf" in text
    view = sd.Visualization("scalar", "User\x1b[31m title\x00", "summary", {"value": 1})
    assert "\x1b" not in str(view) and "\x00" not in str(view)


def test_terminal_show_does_not_duplicate_or_render(monkeypatch, capsys):
    monkeypatch.setattr("seedl._presentation.in_notebook", lambda: False)
    view = sd.see([1, 2], backend="svg")
    assert "Numeric sequence" in capsys.readouterr().out
    monkeypatch.setattr(
        "seedl.renderers.render", lambda *a, **k: pytest.fail("text should not render")
    )
    view.show()
    assert capsys.readouterr().out == ""
    view.notes.append("New observation")
    view.show()
    assert "New observation" in capsys.readouterr().out
    assert view._cached_figure is None


def test_notebook_implicit_and_explicit_display(monkeypatch, capsys):
    displayed = []
    monkeypatch.setattr("seedl._presentation.in_notebook", lambda: True)
    monkeypatch.setitem(
        sys.modules, "IPython.display", types.SimpleNamespace(display=displayed.append)
    )
    view = sd.see([1, 2], backend="svg")
    assert capsys.readouterr().out == ""
    view.show()
    assert displayed == [view]
    bundle = view._repr_mimebundle_()
    assert "image/svg+xml" in bundle and bundle["text/plain"] == view.to_text()


def test_terminal_ipython_is_not_a_notebook(monkeypatch):
    from seedl._presentation import in_notebook

    monkeypatch.setitem(
        sys.modules, "IPython", types.SimpleNamespace(get_ipython=lambda: types.SimpleNamespace())
    )
    assert not in_notebook()
    monkeypatch.setitem(
        sys.modules,
        "IPython",
        types.SimpleNamespace(get_ipython=lambda: types.SimpleNamespace(kernel=object())),
    )
    assert in_notebook()


def test_svg_and_html_safe_responsive_export(tmp_path):
    view = sd.see([[1, 2], [3, 4]], backend="svg")
    view.title = '<script>alert("title")</script>'
    view.notes.append('<img src=x onerror="alert(1)">')
    svg = view.to_figure()
    ET.fromstring(svg)
    assert "min-width:760px" not in svg
    assert "Viridis" in svg and "#440154" in svg and "#fde725" in svg
    assert "<script>" not in svg
    path = tmp_path / "safe.html"
    view.save(path)
    doc = path.read_text()
    assert "&lt;script&gt;" in doc and "&lt;img" in doc
    assert "<img src=x" not in doc
    assert 'name="viewport"' in doc
    assert "Text alternative and full metadata" in doc
    assert doc.count("<h1>") == 1
    assert 'class="title"' not in doc  # no duplicate visual heading


def test_viridis_anchors():
    assert color_at(0) == "#440154"
    assert color_at(1) == "#fde725"
    assert color_at(-1) == color_at(0)


@pytest.mark.parametrize("topic", list(EXAMPLES))
def test_discovery_examples_execute(topic, tmp_path, monkeypatch):
    if topic == "tensor":
        pytest.importorskip("numpy")
    if topic in {"network", "activations", "gradients", "gradient_health"}:
        pytest.importorskip("torch")
    if topic == "evaluate":
        pytest.importorskip("sklearn")
    if topic == "circuit":
        pytest.importorskip("qiskit")
    if topic == "statevector":
        pytest.importorskip("numpy")
    monkeypatch.chdir(tmp_path)
    exec(compile(EXAMPLES[topic], f"seedl examples {topic}", "exec"), {})
    assert list(tmp_path.glob("*.html"))


def test_presentation_exports_reuse_capture_and_preserve_model(tmp_path):
    torch = pytest.importorskip("torch")

    class Counted(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.layer = torch.nn.Linear(2, 1)
            self.register_buffer("running", torch.zeros(1))

        def forward(self, x):
            self.calls += 1
            self.running.add_(1)
            return self.layer(x)

    model = Counted()
    model(torch.ones(1, 2)).sum().backward()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    grads = [p.grad.clone() for p in model.parameters()]
    old_calls = model.calls
    view = sd.activations(model, input=torch.ones(1, 2), backend="svg")
    assert model.calls == old_calls + 1
    captured = json.dumps(view.data, sort_keys=True)
    view.to_text(detail="full")
    view._repr_mimebundle_()
    view.save(tmp_path / "activations.html")
    view.save(tmp_path / "activations.svg")
    assert model.calls == old_calls + 1
    assert model.training is True
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(torch.equal(p.grad, old) for p, old in zip(model.parameters(), grads, strict=True))
    assert json.dumps(view.data, sort_keys=True) == captured


def test_font_assets_are_lazy_and_packaged():
    script = "from seedl.cli import main; from seedl._typography import font_css; main(['doctor']); assert font_css.cache_info().currsize == 0"
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    from seedl._typography import font_css

    css = font_css()
    assert "Google Sans" in css and "JetBrains Mono" in css
    assert "data:font/woff2;base64," in css
    assert "SIL OPEN FONT LICENSE" in css
