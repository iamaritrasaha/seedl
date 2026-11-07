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
    from ._typography import embed_fonts

    _write_html(view, path)
    markup = path.read_text(encoding="utf-8")
    if view.kind.startswith("ml_"):
        from ._ml_renderers import structured_data_html

        markup = structured_data_html(view, markup)
    path.write_text(embed_fonts(markup), encoding="utf-8")


def _write_html(view: Visualization, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if view.kind.startswith("ml_") and view.backend in {"svg", "plotly"}:
        from ._ml_renderers import html_ml

        path.write_text(html_ml(view), encoding="utf-8")
        return
    if view.kind == "tensor_explorer":
        from ._tensor_explorer import tensor_explorer_html

        path.write_text(tensor_explorer_html(view, full_html=True), encoding="utf-8")
        return
    if view.kind == "execution_trace":
        from ._trace_renderers import trace_html

        path.write_text(trace_html(view), encoding="utf-8")
        return
    if view.kind == "attention_rollout":
        from ._transformer_observatory import attention_rollout_html

        path.write_text(attention_rollout_html(view), encoding="utf-8")
        return
    if view.kind == "attribution_comparison":
        from ._explain_compare import attribution_comparison_html

        path.write_text(attribution_comparison_html(view), encoding="utf-8")
        return
    if view.kind == "gradient_health":
        from ._gradient_health import gradient_health_html

        path.write_text(gradient_health_html(view), encoding="utf-8")
        return
    if view.kind == "quantum_noise":
        from ._quantum_lab import quantum_noise_html

        path.write_text(quantum_noise_html(view), encoding="utf-8")
        return
    if view.kind == "dataset_drift":
        from ._dataset_health import dataset_drift_html

        path.write_text(dataset_drift_html(view), encoding="utf-8")
        return
    if view.kind == "run_comparison":
        from ._training_compare import run_comparison_html

        path.write_text(run_comparison_html(view), encoding="utf-8")
        return
    if view.kind == "feature_maps":
        from ._features import features_html

        path.write_text(features_html(view), encoding="utf-8")
        return
    if view.kind == "feature_comparison":
        from ._features import feature_comparison_html

        path.write_text(feature_comparison_html(view), encoding="utf-8")
        return
    if (
        view.kind in {"attention", "attention_comparison"}
        and view.data.get("presentation") == "attention"
    ):
        from ._transformers import attention_html

        path.write_text(attention_html(view), encoding="utf-8")
        return
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
            if view.data.get("neuron_graph"):
                # Preserve the full-viewport WebGL scene and its pointer/camera geometry.
                fig.write_html(
                    str(path),
                    include_plotlyjs=True,
                    config={"responsive": True, "scrollZoom": True},
                    post_script=post_script,
                )
                return
            from ._presentation import document_html

            chart = fig.to_html(
                full_html=False,
                include_plotlyjs=True,
                config={"responsive": True, "scrollZoom": True},
                post_script=post_script,
            )
            # Figure title already supplies the heading visually; HTML provides its accessible name.
            path.write_text(document_html(view, chart, heading=False), encoding="utf-8")
            return

    from ._presentation import document_html
    from .renderers import render

    if view.backend == "matplotlib":
        buf = BytesIO()
        view.figure.savefig(buf, format="png", bbox_inches="tight")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        chart = f'<img src="data:image/png;base64,{b64}" alt="{html.escape(view.title, quote=True)}" style="max-width:100%;height:auto">'
    else:
        chart = render(view, "svg", chart_only=True)
    generic_svg = 'data-seedl-chart-only="true"' in chart
    path.write_text(
        document_html(view, chart, heading=generic_svg or view.backend == "matplotlib"),
        encoding="utf-8",
    )


def _save_svg(view: Visualization, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from ._typography import embed_fonts

    svg_str = view.to_figure("svg")
    path.write_text(embed_fonts(svg_str), encoding="utf-8")


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
