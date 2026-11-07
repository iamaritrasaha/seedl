"""Lightweight captured ML results; rendering never retains or reruns an estimator."""

from __future__ import annotations

from typing import Any

from .exceptions import SeeDLError
from .view import Visualization


class Evaluation(Visualization):
    """A Visualization with captured metrics and independently selectable plots."""

    @property
    def metrics(self) -> dict[str, Any]:
        return self.data.get("metrics", {})

    @property
    def available_plots(self) -> tuple[str, ...]:
        return tuple(self.data.get("panels", {}))

    @property
    def unavailable_plots(self) -> dict[str, str]:
        return self.data.get("unavailable_plots", {})

    def plot(self, name: str) -> Visualization:
        """Select an already captured plot, e.g. result.plot('confusion_matrix')."""
        panels = self.data.get("panels", {})
        if name not in panels:
            reason = self.unavailable_plots.get(name, "Unknown or uncaptured plot.")
            raise SeeDLError(
                f"Plot {name!r} is unavailable: {reason} Available: {', '.join(panels)}."
            )
        panel = panels[name]
        return Visualization(
            kind="ml_panel",
            title=panel["title"],
            summary=panel.get("description", self.summary),
            data={"panels": {name: panel}, "metrics": self.metrics, "task": self.data.get("task")},
            metadata=self.metadata.copy(),
            notes=self.notes.copy(),
            backend=self.backend,
        )
