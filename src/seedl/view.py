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
        try:
            from IPython import get_ipython  # type: ignore[import-not-found]

            if get_ipython() is not None:
                from IPython.display import display  # type: ignore[import-not-found]

                display(self)
                return
        except ImportError:
            pass

        figure = self.figure
        if self.backend == "matplotlib":
            figure.show()
        elif self.backend == "plotly":
            figure.show()
        else:
            print(self)

    def __str__(self) -> str:
        details = [f"{self.title}: {self.summary}"]
        if self.metadata:
            details.extend(f"  {key}: {value}" for key, value in self.metadata.items())
        details.extend(f"  Note: {note}" for note in self.notes)
        return "\n".join(details)

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> dict[str, Any]:
        """Let IPython display the selected backend without an extra server."""
        del include, exclude
        if self.backend == "plotly":
            figure = self.figure
            if hasattr(figure, "_repr_mimebundle_"):
                bundle = figure._repr_mimebundle_()
                if isinstance(bundle, tuple):
                    bundle = bundle[0]
                if isinstance(bundle, dict) and bundle:
                    return bundle
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
        """Expose HTML only for SVG; do not override the selected backend MIME type."""
        if self.backend != "svg":
            return None
        from .renderers import render

        return render(self, "svg")
