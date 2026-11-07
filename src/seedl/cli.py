"""SeeDL's standard-library command line; examples run only on explicit request."""

from __future__ import annotations

import argparse
import difflib
import importlib.metadata
import importlib.util
import os
import platform
import shutil
import sys
import textwrap
from pathlib import Path
from typing import NoReturn

from . import __version__
from ._discovery import EXAMPLES, capabilities, function_help
from ._presentation import clean_text, write_terminal

COMMANDS = {
    "help [api]": "Discover Python APIs and read function help",
    "list": "Browse capabilities by task",
    "doctor": "Check Python and optional integrations",
    "examples [topic]": "Print copyable Python examples",
    "demo [--output FILE]": "Export labeled UCI Iris example data offline",
}


class _Parser(argparse.ArgumentParser):
    def _print_message(self, message: str, file: object = None) -> None:
        if message:
            write_terminal(message.rstrip("\n"), file or sys.stderr)  # type: ignore[arg-type]

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(
            2,
            f"seedl: {clean_text(message)}\nRun 'seedl --help' for commands or 'seedl list' for Python APIs.\n",
        )


def _width() -> int:
    return max(24, shutil.get_terminal_size(fallback=(88, 24)).columns)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="seedl",
        description="SeeDL | Inspect tensors, networks, and quantum states.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, width=_width()),
        epilog="Use Python APIs inside Python: import seedl as sd; sd.see(data).",
    )
    parser.add_argument("--version", action="version", version=f"SeeDL {__version__}")
    parser.add_argument(
        "--plain",
        action="store_true",
        help="plain text without terminal colors (also respects NO_COLOR)",
    )
    sub = parser.add_subparsers(dest="command", title="commands")
    for syntax, description in COMMANDS.items():
        name = syntax.split()[0]
        child = sub.add_parser(
            name,
            help=description,
            description=description,
            formatter_class=lambda prog: argparse.HelpFormatter(prog, width=_width()),
        )
        child.add_argument(
            "--plain", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS
        )
        if name == "help":
            child.add_argument("api", nargs="?", help="Python function, e.g. see or activations")
        elif name == "examples":
            child.add_argument("topic", nargs="?", choices=tuple(EXAMPLES), help="example topic")
        elif name == "demo":
            child.add_argument(
                "--output",
                type=Path,
                default=Path("seedl-demo.html"),
                help="new .html or .svg file (default: seedl-demo.html); never overwrites",
            )
    return parser


def _compact_guide(width: int) -> str:
    lines = ["SeeDL | Scientific visualization in Python", "", "Get started"]
    for syntax, description in COMMANDS.items():
        label = f"seedl {syntax}"
        if width >= 76:
            lines.extend(
                textwrap.wrap(
                    description,
                    width=width,
                    initial_indent=f"  {label:<29}  ",
                    subsequent_indent=" " * 33,
                )
            )
        else:
            lines.append(f"  {label}")
            lines.extend(
                textwrap.wrap(
                    description, width=width, initial_indent="    ", subsequent_indent="    "
                )
            )
    lines.extend(
        (
            "",
            "In Python",
            "  import seedl as sd",
            '  view = sd.see([[1, 2], [3, 4]], backend="svg")',
            "  view.show()",
            '  view.save("result.html")',
            "",
            "Run seedl --help for command options.",
            "No browser opens automatically. Demo writes only the requested file.",
        )
    )
    if width < 60:
        start = lines.index("In Python")
        end = lines.index("Run seedl --help for command options.")
        lines[start:end] = ["Python quickstart", "  seedl examples see", ""]
    return "\n".join(lines)


def _doctor() -> str:
    supported = (3, 10) <= sys.version_info[:2] <= (3, 13)
    lines = [
        "SeeDL environment",
        f"  Package   {__version__}",
        f"  Python    {platform.python_version()} ({'supported' if supported else 'outside the tested 3.10-3.13 range'})",
        f"  Platform  {platform.system()} {platform.machine()}",
        "",
        "Optional integrations (presence only; frameworks are not imported)",
    ]
    for module, extra in (
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("plotly", "plotly"),
        ("matplotlib", "matplotlib"),
        ("qiskit", "quantum"),
    ):
        try:
            present = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            present = False
        try:
            version = importlib.metadata.version(module) if present else ""
        except importlib.metadata.PackageNotFoundError:
            version = "version unknown"
        status = (
            f"found {version}"
            if present
            else f'missing; install: python -m pip install "seedl[{extra}]"'
        )
        lines.append(f"  {module:<11} {status}")
    lines.extend(
        (
            "",
            "Core SVG and offline HTML work without extras.",
            "CUDA/ROCm availability and driver compatibility require a framework runtime check.",
            "Doctor does not initialize a GPU or verify installed binary compatibility.",
        )
    )
    return "\n".join(lines)


def _demo(path: Path) -> str:
    """Export authentic measurements without fetching data or launching a browser."""
    if path.suffix.lower() not in {".html", ".svg"}:
        raise ValueError("Demo output must end in .html or .svg.")
    if path.exists():
        raise ValueError(f"File already exists: {path}. Choose a new --output path.")
    from contextlib import redirect_stdout
    from io import StringIO

    from .api import see

    # Verified against UCI iris.data, first five Iris-setosa observations, centimeters.
    measurements = [
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2],
        [4.6, 3.1, 1.5, 0.2],
        [5.0, 3.6, 1.4, 0.2],
    ]
    with redirect_stdout(StringIO()):
        view = see(measurements, backend="svg")
    view.title = "Iris setosa measurements (cm)"
    view.summary = "Authentic example data | first 5 UCI Iris observations, 4 measurements in cm"
    view.metadata = {
        "Source": "UCI Iris dataset (Fisher, 1936)",
        "Units": "centimeters (cm)",
        "Rows": "first five Iris-setosa observations",
        "Columns": "sepal length, sepal width, petal length, petal width",
    }
    view.notes.append(
        "Example excerpt, not representative of the full dataset. Source: https://archive.ics.uci.edu/dataset/53/iris (CC BY 4.0). Values are unchanged from iris.data."
    )
    # Exclusive creation protects existing data even when another process creates the file.
    path.parent.mkdir(parents=True, exist_ok=True)
    from ._typography import embed_fonts

    content = (
        view.to_html() if path.suffix.lower() == ".html" else embed_fonts(view.to_figure("svg"))
    )
    with path.open("x", encoding="utf-8") as output:
        output.write(content)
    return f"Created {path}\nData: UCI Iris, first five setosa observations; measurements in cm.\nOpen the file in a browser when ready."


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args_in = list(sys.argv[1:] if argv is None else argv)
    # Improve argparse's invalid-command guidance without guessing an API invocation.
    if (
        args_in
        and not args_in[0].startswith("-")
        and args_in[0] not in {s.split()[0] for s in COMMANDS}
    ):
        matches = difflib.get_close_matches(args_in[0], [s.split()[0] for s in COMMANDS], n=1)
        hint = f" Did you mean '{matches[0]}'?" if matches else ""
        parser.error(f"unknown command {args_in[0]!r}.{hint}")
    args = parser.parse_args(args_in)
    try:
        if args.command is None:
            output = _compact_guide(_width())
        elif args.command == "help":
            output = function_help(args.api) if args.api else capabilities(width=_width())
        elif args.command == "list":
            output = capabilities(width=_width())
        elif args.command == "doctor":
            output = _doctor()
        elif args.command == "examples":
            topics = (
                [args.topic]
                if args.topic
                else [
                    "see",
                    "tensor",
                    "network",
                    "activations",
                    "gradient_health",
                    "attention",
                    "bloch",
                    "circuit",
                ]
            )
            output = "Copyable Python examples (run inside Python)\n\n" + "\n\n".join(
                f"{topic}\n{'-' * len(topic)}\n{EXAMPLES[topic]}" for topic in topics
            )
        else:
            output = _demo(args.output)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    output = clean_text(output)
    # Examples and signatures keep their whitespace so copy/paste remains runnable.
    if args.command not in {"examples", "help"}:
        output = "\n".join(
            textwrap.fill(line, width=_width(), subsequent_indent="    ")
            if len(line) > _width()
            else line
            for line in output.splitlines()
        )
    color = (
        not args.plain
        and "NO_COLOR" not in os.environ
        and sys.stdout.isatty()
        and os.environ.get("TERM") != "dumb"
    )
    if color:
        first, _, rest = output.partition("\n")
        output = f"\033[1;36m{first}\033[0m\n{rest}"
    write_terminal(output)
    return 0
