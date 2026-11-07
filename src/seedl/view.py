"""The small, stable object returned by SeeDL's visualization functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Visualization:
    """A rendered view and the compact data used to create it.

    ``Visualization`` is returned by :func:`seedl.see`, :func:`seedl.activations`,
    and :func:`seedl.gradients`. In notebooks it renders when it is the final
    expression in a cell. Call :meth:`show` to display it from a regular script.
    """

    kind: str
    title: str
    summary: str
    data: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    backend: str = "svg"
    _cached_figure: Any = field(default=None, init=False, repr=False)

    _last_terminal_text: str | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def figure(self) -> Any:
        """Return the Plotly or Matplotlib figure, or SVG text for the SVG backend."""
        if self._cached_figure is None:
            from .renderers import render

            self._cached_figure = render(self, self.backend)
        return self._cached_figure

    def to_figure(self, backend: str | None = None) -> Any:
        """Render with ``plotly``, ``matplotlib``, or the built-in ``svg`` backend."""
        from .renderers import render

        return render(self, backend or self.backend)

    def show(self) -> None:
        """Display the visualization or print its concise summary in a terminal."""
        from ._presentation import in_notebook

        if in_notebook():
            from IPython.display import display

            display(self)
        elif self.backend in {"matplotlib", "plotly"}:
            self.figure.show()
        else:
            self._show_terminal()

    def _show_terminal(self) -> None:
        import shutil

        from ._presentation import write_terminal

        text = self.to_text(width=max(24, shutil.get_terminal_size(fallback=(88, 24)).columns))
        if text != self._last_terminal_text:
            write_terminal(text)
            self._last_terminal_text = text

    def to_text(self, *, detail: str = "summary", width: int = 88, plain: bool = False) -> str:
        """Return readable diagnostics without rendering or loading a plotting library.

        ``detail='full'`` includes all metadata/layer rows with full float precision.
        All sampling/provenance notes and warnings remain visible in summary mode.
        Original values remain in ``data`` and ``metadata``. ``plain=True`` uses ASCII.
        """
        from ._presentation import to_text

        return to_text(self, detail=detail, width=width, plain=plain)

    def save(self, filepath: str | Any, *, format: str | None = None) -> Any:
        """Export the visualization to an HTML, SVG, or PNG file.

        Supported extensions: ``.html``, ``.svg``, ``.png``.
        """
        from ._export import save_visualization

        return save_visualization(self, filepath, format=format)

    def to_html(self) -> str:
        """Return a complete, self-contained interactive HTML document for this visualization."""
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            self.save(tmp_path, format="html")
            return tmp_path.read_text(encoding="utf-8")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def explore(self, *, select_neuron: tuple[int, int] | str | None = None) -> Any:
        """Return an interactive figure with layer stepping, forward animation, and neuron inspection."""
        if self.data.get("neuron_graph"):
            self.data["neuron_graph"]["explore"] = True
            if select_neuron is not None:
                from ._neurons import _normalize_select_neuron

                self.data["neuron_graph"]["selected_neuron"] = _normalize_select_neuron(
                    select_neuron, self.data["neuron_graph"]["widths"]
                )
        try:
            import importlib.util

            if importlib.util.find_spec("plotly"):
                self.backend = "plotly"
                self._cached_figure = None
                return self.figure
        except Exception:
            pass
        return self.figure

    def __repr__(self) -> str:
        return f"Visualization(kind={self.kind!r}, title={self.title!r}, summary={self.summary!r})"

    def __str__(self) -> str:
        return self.to_text()

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> dict[str, Any]:
        """Let IPython display the selected backend without an extra server."""
        from ._typography import embed_fonts

        bundle = self._mimebundle(include, exclude)
        if "text/html" in bundle:
            bundle["text/html"] = embed_fonts(bundle["text/html"])
        if "image/svg+xml" in bundle:
            bundle["image/svg+xml"] = embed_fonts(bundle["image/svg+xml"])
        return bundle

    def _mimebundle(self, include: Any = None, exclude: Any = None) -> dict[str, Any]:
        del include, exclude
        if self.kind.startswith("ml_") and self.backend == "svg":
            from ._ml_renderers import html_ml

            return {"text/html": html_ml(self), "text/plain": str(self)}
        if self.kind == "tensor_explorer":
            from ._tensor_explorer import tensor_explorer_html

            return {
                "text/html": tensor_explorer_html(self, full_html=False),
                "text/plain": str(self),
            }
        if self.backend == "plotly" and (
            self.kind == "training_playback"
            or (self.kind == "quantum_circuit" and self.data.get("simulate"))
        ):
            from ._playback import playback_html

            return {"text/html": playback_html(self, full_html=False), "text/plain": str(self)}
        if self.backend == "plotly":
            figure = self.figure
            if hasattr(figure, "_repr_mimebundle_"):
                bundle = figure._repr_mimebundle_()
                if isinstance(bundle, tuple):
                    bundle = bundle[0]
                if isinstance(bundle, dict) and bundle:
                    res = dict(bundle)
                    res["text/plain"] = str(self)
                    return res
            if hasattr(figure, "_repr_html_"):
                html = figure._repr_html_()
                if html:
                    return {"text/html": html, "text/plain": str(self)}
        if self.backend == "matplotlib":
            from io import BytesIO

            buffer = BytesIO()
            self.figure.savefig(buffer, format="png", bbox_inches="tight")
            return {"image/png": buffer.getvalue(), "text/plain": str(self)}
        return {"image/svg+xml": self.figure, "text/plain": str(self)}

    def _repr_html_(self) -> str | None:
        """Expose HTML for SVG and interactive widgets; do not override the selected backend MIME type."""
        if self.kind == "tensor_explorer":
            from ._tensor_explorer import tensor_explorer_html
            from ._typography import embed_fonts

            return embed_fonts(tensor_explorer_html(self, full_html=False))
        if self.backend != "svg":
            return None
        from ._typography import embed_fonts
        from .renderers import render

        return embed_fonts(render(self, "svg"))
