"""Dependency-free presentation utilities; never compute scientific diagnostics."""

from __future__ import annotations

import math
import re
import sys
import textwrap
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from .view import Visualization

from ._typography import FONT

# Published Viridis control colors (Matplotlib/Plotly); interpolate for compact SVG/Canvas.
VIRIDIS = [
    "#440154",
    "#482878",
    "#3e4989",
    "#31688e",
    "#26828e",
    "#1f9e89",
    "#35b779",
    "#6ece58",
    "#b5de2b",
    "#fde725",
]


def clean_text(value: Any) -> str:
    """Keep user labels readable without terminal escape/control sequences."""
    value = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(value))
    return "".join(
        c if c in "\n\t" or (ord(c) >= 32 and not 127 <= ord(c) <= 159) else "" for c in value
    )


def ascii_text(value: str) -> str:
    for original, replacement in {
        "×": "x",
        "·": "|",
        "−": "-",
        "—": "-",
        "–": "-",
        "→": "->",
        "∞": "Inf",
        "…": "...",
        "⟩": ">",
        "⟨": "<",
    }.items():
        value = value.replace(original, replacement)
    return value.encode("ascii", errors="backslashreplace").decode("ascii")


def write_terminal(value: str, stream: TextIO | None = None) -> None:
    """Write using the stream's encoding, including redirected Windows output."""
    stream = stream or sys.stdout
    encoding = stream.encoding or "utf-8"
    try:
        value.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        value = ascii_text(value)
    print(value, file=stream)


def in_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and getattr(shell, "kernel", None) is not None
    except ImportError:
        return False


def format_value(value: Any, *, full: bool = False) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "-Inf" if value < 0 else "+Inf"
        return repr(value) if full else f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(format_value(v, full=full) for v in value) + ")"
    return clean_text(value)


def note_level(note: str) -> str:
    """Label existing observations; do not invent a diagnosis or threshold."""
    text = note.lower()
    for explicit in ("WARNING", "CAUTION", "INFO"):
        if text.lstrip().startswith(explicit.lower() + ":"):
            return explicit
    if any(
        token in text
        for token in (
            "non-finite",
            "nonfinite",
            "nan value",
            "infinite value",
            "numerical instability",
        )
    ):
        return "WARNING"
    if any(
        token in text
        for token in (
            "alert",
            "saturation detected",
            "near-zero",
            "is high",
            "high fraction",
            "grad=none",
            "had grad=none",
            "possible overfitting",
            "warning",
        )
    ):
        return "CAUTION"
    return "INFO"


def ordered_notes(view: Visualization) -> list[tuple[str, str]]:
    notes = []
    for note in view.notes:
        text = str(note)
        level = note_level(text)
        if text.lstrip().upper().startswith(level + ":"):
            text = text.lstrip()[len(level) + 1 :].lstrip()
        notes.append((level, text))
    return sorted(notes, key=lambda item: {"WARNING": 0, "CAUTION": 1, "INFO": 2}[item[0]])


def color_at(ratio: float) -> str:
    index = max(0.0, min(1.0, ratio)) * (len(VIRIDIS) - 1)
    lo = int(index)
    hi = min(lo + 1, len(VIRIDIS) - 1)
    a, b = VIRIDIS[lo], VIRIDIS[hi]
    rgb = [
        round(int(a[i : i + 2], 16) * (1 - index + lo) + int(b[i : i + 2], 16) * (index - lo))
        for i in (1, 3, 5)
    ]
    return "#" + "".join(f"{v:02x}" for v in rgb)


def _wrap(text: Any, width: int, prefix: str = "") -> list[str]:
    return textwrap.wrap(
        clean_text(text), width=width, initial_indent=prefix, subsequent_indent=" " * len(prefix)
    ) or [prefix.rstrip()]


def to_text(
    view: Visualization, *, detail: str = "summary", width: int = 88, plain: bool = False
) -> str:
    if detail not in {"summary", "full"}:
        raise ValueError("detail must be 'summary' or 'full'")
    if isinstance(width, bool) or not isinstance(width, int) or width < 24:
        raise ValueError("width must be an integer of at least 24 columns")
    full = detail == "full"
    lines = _wrap(view.title, width)
    lines.append("=" * min(width, max(12, len(clean_text(view.title)))))
    lines.extend(_wrap(view.summary, width))
    notes = ordered_notes(view)
    findings = [(level, text) for level, text in notes if level != "INFO"]
    if findings:
        lines.append("")
        for level, text in findings:
            lines.extend(_wrap(text, width, f"{level}: "))
        if any(level == "WARNING" for level, _ in findings):
            lines.extend(
                _wrap(
                    "NaN means undefined; Inf means infinite. Unavailable statistics are not zero.",
                    width,
                )
            )
    items = list(view.metadata.items())
    if not items and view.data.get("shape") is not None:
        items.append(("Shape", view.data["shape"]))
    if view.kind == "scalar":
        items.insert(0, ("Value", view.data.get("value")))
    # Keep the common shape/device/statistics first; full metadata stays in .metadata.
    visible = items if full else items[:12]
    if visible:
        lines.extend(("", "Summary"))
        label_width = min(max(len(clean_text(k)) for k, _ in visible), max(8, width // 3))
        for key, value in visible:
            label = clean_text(key)
            lines.extend(
                _wrap(f"{label:<{label_width}}  {format_value(value, full=full)}", width, "  ")
            )
        if len(visible) < len(items):
            lines.extend(
                _wrap(
                    f"{len(items) - len(visible)} more metadata fields; use to_text(detail='full') or .metadata.",
                    width,
                    "  ",
                )
            )
    values = view.data.get("values", []) if view.kind in {"matrix", "sequence"} else []
    if values:
        lines.extend(("", "Values"))
        if view.kind == "matrix":
            row_indices = view.data.get("row_indices", list(range(len(values))))
            for i, row in enumerate(values if full else values[:6]):
                displayed = row if full else row[:8]
                text = ", ".join(format_value(v, full=full) for v in displayed)
                if len(displayed) < len(row):
                    text += f" ... ({len(row)} columns)"
                lines.extend(_wrap(f"row {row_indices[i]}: [{text}]", width, "  "))
            if not full and len(values) > 6:
                lines.extend(
                    _wrap(
                        f"Showing 6 of {len(values)} captured rows; all captured values in .data['values'].",
                        width,
                    )
                )
        else:
            displayed = values if full else values[:12]
            lines.extend(
                _wrap(", ".join(format_value(v, full=full) for v in displayed), width, "  ")
            )
            if len(displayed) < len(values):
                lines.extend(
                    _wrap(
                        f"Showing {len(displayed)} of {len(values)} captured values; full detail in .data['values'].",
                        width,
                    )
                )
    rows = view.data.get("layers", [])
    if rows and isinstance(rows, list) and isinstance(rows[0], dict):
        columns = [
            k
            for k in (
                "name",
                "layer",
                "num_tensors",
                "num_elements",
                "l2_norm",
                "type",
                "parameters",
                "trainable",
                "shape",
                "status",
                "mean",
                "std",
                "max_abs",
                "finite",
            )
            if any(k in r for r in rows)
        ]
        shown = rows if full else rows[:6]
        cells = [
            [
                format_value((r.get(k) or "(root)") if k == "name" else r.get(k), full=full)
                for k in columns
            ]
            for r in shown
        ]
        sizes = [max(len(k), *(len(row[i]) for row in cells)) for i, k in enumerate(columns)]
        lines.extend(("", "Layers"))
        if sum(sizes) + 2 * len(sizes) <= width:
            lines.append("  ".join(k.ljust(sizes[i]) for i, k in enumerate(columns)))
            for row in cells:
                lines.append(
                    "  ".join(
                        v.ljust(sizes[i]) if i == 0 else v.rjust(sizes[i])
                        for i, v in enumerate(row)
                    )
                )
        else:
            for row in cells:
                lines.extend(
                    _wrap(
                        "; ".join(f"{k}: {v}" for k, v in zip(columns, row, strict=True)),
                        width,
                        "  ",
                    )
                )
        if len(shown) < len(rows):
            lines.extend(
                _wrap(
                    f"Showing {len(shown)} of {len(rows)} layers; all records in .data['layers'].",
                    width,
                )
            )
    info = [text for level, text in notes if level == "INFO"]
    if info:
        lines.extend(("", "Notes"))
        # All provenance and sampling disclosures remain visible.
        for note in info:
            lines.extend(_wrap(note, width, "  - "))
    if full:
        lines.extend(
            ("", "Structured payload: .data (values, samples and diagnostics); .metadata; .notes.")
        )
    result = "\n".join(lines)
    if plain:
        result = ascii_text(result)
        result = "\n".join(part for line in result.splitlines() for part in _wrap(line, width))
    return result


def findings_html(
    view: Visualization, *, levels: tuple[str, ...] = ("WARNING", "CAUTION", "INFO")
) -> str:
    """Escaped, severity-labeled observations shared by offline documents."""
    import html

    items = "".join(
        f'<li class="finding {level.lower()}"><strong>{level}:</strong> {html.escape(text)}</li>'
        for level, text in ordered_notes(view)
        if level in levels
    )
    return (
        f'<section aria-label="Observations"><ul class="findings">{items}</ul></section>'
        if items
        else ""
    )


def document_html(view: Visualization, chart: str, *, heading: bool = True) -> str:
    """Shared offline document shell; plot-specific controls remain in the renderer."""
    import html

    title = html.escape(view.title)
    header = (
        f"<header><h1>{title}</h1><p>{html.escape(view.summary)}</p></header>" if heading else ""
    )
    alternative = html.escape(view.to_text(detail="full"))
    matrix = ""
    rows = view.data.get("values", []) if view.kind == "matrix" else []
    if rows and len(rows) <= 12 and max(map(len, rows)) <= 12:
        columns = view.data.get("column_indices", list(range(len(rows[0]))))
        row_indices = view.data.get("row_indices", list(range(len(rows))))
        heads = "".join(f'<th scope="col">{html.escape(str(c))}</th>' for c in columns)
        body = "".join(
            '<tr><th scope="row">'
            + html.escape(str(row_indices[i]))
            + "</th>"
            + "".join(f"<td>{html.escape(format_value(v))}</td>" for v in row)
            + "</tr>"
            for i, row in enumerate(rows)
        )
        matrix = f'<div class="matrix-values" tabindex="0" role="region" aria-label="Matrix values"><table><caption>Values by row and column (zero-based)</caption><thead><tr><th scope="col">Row / Col</th>{heads}</tr></thead><tbody>{body}</tbody></table></div>'
        chart = f'<div class="matrix-chart">{chart}</div>'
    findings = findings_html(view, levels=("WARNING", "CAUTION"))
    notes = findings_html(view, levels=("INFO",))
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — SeeDL</title>
<style>
:root {{ color-scheme: light; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; padding:clamp(12px,3vw,32px); background:#f8fafc; color:#172b4d; font:14px/1.6 {FONT}; }}
main {{ max-width:1040px; margin:auto; background:white; border:1px solid #dce4ed; border-radius:8px; padding:clamp(12px,3vw,28px); min-width:0; }}
h1 {{ font-size:clamp(20px,3vw,26px); line-height:1.25; margin:0 0 8px; overflow-wrap:anywhere; }}
header p {{ color:#52657d; margin:0 0 20px; }}
.chart {{ min-width:0; overflow:auto; }}
.chart svg {{ display:block; max-width:100%; height:auto; }}
.matrix-values {{ display:none; overflow:auto; }}
.matrix-values table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }}
.matrix-values th, .matrix-values td {{ border-bottom:1px solid #dce4ed; padding:8px; text-align:right; }}
.matrix-values th {{ color:#52657d; }}
.matrix-values td {{ font-family:JetBrains Mono,monospace; }}
.matrix-values caption {{ text-align:left; color:#52657d; padding-bottom:12px; }}
@media(max-width:600px) {{ .matrix-values {{ display:block; }} .matrix-chart {{ display:none; }} }}
.findings {{ padding-left:22px; overflow-wrap:anywhere; }}
.finding {{ margin:8px 0; }}
.warning {{ color:#9f2424; }} .caution {{ color:#855000; }}
.info {{ color:#52657d; }}
details {{ margin-top:20px; border-top:1px solid #dce4ed; padding-top:12px; }}
summary {{ cursor:pointer; font-weight:600; }}
pre {{ white-space:pre-wrap; overflow-wrap:anywhere; font:12px/1.6 JetBrains Mono,monospace; font-variant-numeric:tabular-nums; }}
:focus-visible {{ outline:3px solid #2563eb; outline-offset:3px; }}
footer {{ color:#52657d; font-size:12px; margin-top:20px; }}
</style></head><body><main>
{header}{findings}<div class="chart">{chart}{matrix}</div>{notes}
<details><summary>Text alternative and full metadata</summary><pre>{alternative}</pre></details>
<footer>Exported by SeeDL · Offline document · Original structured values remain in Python view.data</footer>
</main></body></html>"""


def contrast_text(color: str) -> str:
    """Choose dark or white text using WCAG relative luminance."""
    components = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in components]
    luminance = sum(v * weight for v, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))
    return "#000000" if luminance > 0.179 else "#ffffff"
