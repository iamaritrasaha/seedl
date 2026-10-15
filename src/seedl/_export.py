"""Export utilities for SeeDL visualizations (HTML, SVG, PNG)."""

from __future__ import annotations

import base64
import html
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from .exceptions import MissingDependencyError, SeeDLError

if TYPE_CHECKING:
    from .view import Visualization


def save_visualization(
    view: Visualization,
    filepath: str | Path,
    *,
    format: str | None = None,
) -> Path:
    """Save a visualization to HTML, SVG, or PNG format."""
    path = Path(filepath)
    ext = (format or path.suffix).lower().lstrip(".")

    if not ext:
        raise SeeDLError(
            "Cannot determine export format; specify a file extension like .html, .svg, or .png."
        )

    if ext in {"html", "htm"}:
        _save_html(view, path)
    elif ext == "svg":
        _save_svg(view, path)
    elif ext in {"png", "jpg", "jpeg"}:
        _save_image(view, path, ext)
    else:
        raise SeeDLError(
            f"Unsupported export format {ext!r}. Supported formats are .html, .svg, and .png."
        )

    return path


def _save_html(view: Visualization, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if view.backend == "plotly" and (
        view.kind == "training_playback"
        or (view.kind == "quantum_circuit" and view.data.get("simulate"))
    ):
        from ._playback import playback_html

        path.write_text(playback_html(view), encoding="utf-8")
        return
    if view.backend == "plotly":
        fig = view.figure
        if hasattr(fig, "write_html"):
            post_script = None
            if view.data.get("neuron_graph"):
                from ._network_renderers import build_network_interactive_js

                spatial = view.data["neuron_graph"].get("dimensions") == 3
                post_script = build_network_interactive_js(view.data["neuron_graph"], spatial)
            fig.write_html(
                str(path),
                include_plotlyjs=True,
                config={"responsive": True, "scrollZoom": True},
                post_script=post_script,
            )
            return

    # Self-contained HTML with embedded SVG or PNG
    title = html.escape(view.title)
    summary = html.escape(view.summary)

    if view.backend == "matplotlib":
        fig = view.figure
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        chart_html = f'<img src="data:image/png;base64,{b64}" alt="{title}" style="max-width:100%; height:auto;" />'
    else:
        # SVG backend
        svg_content = view.to_figure("svg")
        chart_html = f'<div style="overflow-x:auto;">{svg_content}</div>'

    meta_items = "".join(
        f'<span style="background:#eef2f7; padding:4px 8px; border-radius:4px; font-size:12px; margin-right:8px; display:inline-block; margin-bottom:4px;"><strong>{html.escape(k)}:</strong> {html.escape(str(v))}</span>'
        for k, v in view.metadata.items()
    )
    notes_items = "".join(
        f'<li style="color:#52657d; font-size:13px; margin-bottom:4px;">{html.escape(n)}</li>'
        for n in view.notes
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title} — SeeDL</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f8fafc; color: #1e293b; margin: 0; padding: 24px; }}
    .card {{ background: #ffffff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); max-width: 960px; margin: 0 auto; padding: 24px; }}
    h1 {{ font-size: 22px; margin-top: 0; margin-bottom: 6px; color: #0f172a; }}
    .summary {{ color: #64748b; font-size: 14px; margin-bottom: 16px; }}
    .meta {{ margin-bottom: 20px; }}
    .notes {{ margin-top: 20px; padding-left: 20px; }}
    footer {{ margin-top: 24px; font-size: 11px; color: #94a3b8; text-align: right; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>{title}</h1>
    <div class="summary">{summary}</div>
    {f'<div class="meta">{meta_items}</div>' if meta_items else ""}
    <div class="chart">{chart_html}</div>
    {f'<ul class="notes">{notes_items}</ul>' if notes_items else ""}
    <footer>Exported by SeeDL</footer>
  </div>
</body>
</html>"""
    path.write_text(doc, encoding="utf-8")


def _save_svg(view: Visualization, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg_str = view.to_figure("svg")
    path.write_text(svg_str, encoding="utf-8")


def _save_image(view: Visualization, path: Path, ext: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import importlib.util

    spatial = (
        view.data.get("dimensions") == 3 or view.data.get("neuron_graph", {}).get("dimensions") == 3
    )
    if importlib.util.find_spec("matplotlib") is not None and not spatial:
        fig = view.to_figure("matplotlib")
        fig.savefig(str(path), format=ext, bbox_inches="tight")
        return

    if view.backend == "plotly":
        fig = view.figure
        if hasattr(fig, "write_image"):
            try:
                fig.write_image(str(path))
                return
            except Exception as exc:
                raise MissingDependencyError(
                    f"Saving Plotly images as {ext} requires the 'kaleido' package. "
                    "Install it or export to .html / .svg instead."
                ) from exc

    raise MissingDependencyError(
        f"Exporting to .{ext} requires Matplotlib. Install it with `pip install 'seedl[matplotlib]'` "
        "or save as .html or .svg."
    )
