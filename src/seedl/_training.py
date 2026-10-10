"""Training observability, diagnostic indicators, and lightweight experiment tracking."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .exceptions import SeeDLError


def exponential_smoothing(values: list[float], weight: float = 0.6) -> list[float]:
    """Compute exponential moving average of a numeric sequence."""
    if not values:
        return []
    smoothed = []
    last = values[0]
    for val in values:
        if math.isfinite(val):
            last = last * weight + (1 - weight) * val
            smoothed.append(last)
        else:
            smoothed.append(val)
    return smoothed


def parse_training_history(
    history: Mapping[Any, Any],
    max_points: int = 2_048,
    smooth: bool = True,
) -> tuple[dict[str, Any], str, list[str]]:
    """Parse a metric history dictionary and evaluate diagnostic indicators."""
    metrics: dict[str, list[float]] = {}
    epochs: list[float] | None = None
    notes = ["Curves are diagnostic clues; they are not guarantees about model performance."]

    for key, values in history.items():
        if not isinstance(key, str):
            raise SeeDLError("Training history keys must be strings such as 'loss' or 'val_loss'.")
        lowered = key.lower()
        if lowered in {"epoch", "epochs", "step", "steps"}:
            epochs = _extract_floats(values, f"history[{key!r}]")
            continue
        metrics[key] = _extract_floats(values, f"history[{key!r}]")

    if not metrics:
        raise SeeDLError("Training history is empty; add one or more metric sequences.")
    if any(not v for v in metrics.values()):
        raise SeeDLError("Training history metric sequences must contain at least one value.")
    if epochs is not None and not epochs:
        raise SeeDLError("Epoch sequence must contain at least one value.")
    if epochs is not None and any(not math.isfinite(v) for v in epochs):
        raise SeeDLError("Epoch labels must be finite real numbers.")

    has_nonfinite = any(not math.isfinite(v) for seq in metrics.values() for v in seq)
    if has_nonfinite:
        notes.append("Non-finite metric values are kept as gaps in the curves.")

    # Group metrics (train & val)
    groups: dict[str, list[dict[str, Any]]] = {}
    metric_names = set(metrics)

    for name, values in metrics.items():
        low = name.lower()
        if low.startswith("val_"):
            base, role = name[4:], "validation"
        elif low.startswith("validation_"):
            base, role = name[len("validation_") :], "validation"
        elif low.endswith("_val"):
            base, role = name[:-4], "validation"
        elif low.endswith("_validation"):
            base, role = name[:-11], "validation"
        else:
            base, role = name, "training"

        group_name = base if base in metric_names or role == "training" else name
        if group_name not in metric_names and role == "validation":
            group_name = base or name

        x_coords = (
            list(epochs[: len(values)])
            if epochs is not None and len(epochs) == len(values)
            else list(range(1, len(values) + 1))
        )
        if epochs is not None and len(epochs) != len(values):
            notes.append(
                f"'{name}' has {len(values)} points; epoch labels do not match, so index labels are used."
            )

        smoothed = exponential_smoothing(values) if smooth else values

        groups.setdefault(group_name, []).append(
            {
                "name": name,
                "role": role,
                "x": x_coords,
                "values": values,
                "smoothed": smoothed,
            }
        )

    output_groups: list[dict[str, Any]] = [
        {"name": name, "series": series} for name, series in groups.items()
    ]

    # Diagnostics: Overfitting, Divergence, Plateau, Learning Rate
    for group in output_groups:
        g_name = group["name"].lower()
        series_map = {s["role"]: s for s in group["series"]}
        train_s = series_map.get("training")
        val_s = series_map.get("validation")

        if "loss" in g_name or "error" in g_name:
            if train_s:
                t_vals = [v for v in train_s["values"] if math.isfinite(v)]
                if len(t_vals) >= 3:
                    # Diverging loss check
                    if t_vals[-1] > t_vals[0] * 3.0 and t_vals[-1] > 10.0:
                        notes.append(
                            f"Diverging loss alert in {group['name']}: final loss ({t_vals[-1]:.3g}) "
                            f"is significantly higher than initial loss ({t_vals[0]:.3g})."
                        )
                    # Plateau / Stalled check
                    recent = t_vals[-min(5, len(t_vals)) :]
                    if len(recent) >= 4 and abs(recent[-1] - recent[0]) < 1e-4:
                        notes.append(
                            f"Plateau notice in {group['name']}: loss has stalled across recent epochs."
                        )

            # Overfitting check
            if train_s and val_s:
                paired = [
                    (tr, va)
                    for tr, va in zip(train_s["values"], val_s["values"], strict=False)
                    if math.isfinite(tr) and math.isfinite(va)
                ]
                if len(paired) >= 3 and train_s["x"] == val_s["x"]:
                    t_start, t_end = paired[0][0], paired[-1][0]
                    v_vals = [p[1] for p in paired]
                    v_min = min(v_vals[:-1])
                    rise = v_vals[-1] - v_min
                    if t_end < t_start and rise > max(abs(v_min) * 0.05, 1e-8):
                        notes.append(
                            f"Possible overfitting clue in {group['name']}: training loss decreased while "
                            f"validation loss increased after its minimum point ({v_min:.3g} -> {v_vals[-1]:.3g})."
                        )

    # Subsampling for large series
    for g in output_groups:
        for s in g["series"]:
            if len(s["values"]) > max_points:
                step = math.ceil(len(s["values"]) / max_points)
                s["values"] = s["values"][::step]
                s["smoothed"] = s["smoothed"][::step]
                s["x"] = s["x"][::step]
                notes.append(
                    f"'{s['name']}' was downsampled for display; all points remain in data."
                )

    lengths = [len(v) for v in metrics.values()]
    summary = f"{len(metrics)} metric(s), {min(lengths)}–{max(lengths)} epochs"

    return {"groups": output_groups, "metrics": metrics}, summary, notes


def compare_histories(
    hist_a: Mapping[str, Any],
    hist_b: Mapping[str, Any],
    names: tuple[str, str] = ("Run A", "Run B"),
    max_points: int = 2_048,
) -> tuple[dict[str, Any], str, list[str]]:
    """Compare two training histories side by side."""
    data_a, sum_a, _ = parse_training_history(hist_a, max_points=max_points, smooth=False)
    data_b, sum_b, _ = parse_training_history(hist_b, max_points=max_points, smooth=False)

    name_a, name_b = names
    combined_groups: list[dict[str, Any]] = []
    all_group_names = list(
        dict.fromkeys([g["name"] for g in data_a["groups"]] + [g["name"] for g in data_b["groups"]])
    )

    diff_summaries: list[str] = []

    for g_name in all_group_names:
        series_list = []
        g_a = next((g for g in data_a["groups"] if g["name"] == g_name), None)
        g_b = next((g for g in data_b["groups"] if g["name"] == g_name), None)

        if g_a:
            for s in g_a["series"]:
                series_list.append(
                    {
                        "name": f"{name_a}: {s['name']}",
                        "role": s["role"],
                        "run": name_a,
                        "x": s["x"],
                        "values": s["values"],
                    }
                )
        if g_b:
            for s in g_b["series"]:
                series_list.append(
                    {
                        "name": f"{name_b}: {s['name']}",
                        "role": s["role"],
                        "run": name_b,
                        "x": s["x"],
                        "values": s["values"],
                    }
                )

        combined_groups.append({"name": g_name, "series": series_list})

        # Calculate final diff if both have same metric
        if g_a and g_b:
            last_a = g_a["series"][0]["values"][-1]
            last_b = g_b["series"][0]["values"][-1]
            if math.isfinite(last_a) and math.isfinite(last_b):
                diff = last_b - last_a
                sign = "+" if diff > 0 else ""
                diff_summaries.append(
                    f"{g_name}: {last_a:.4g} vs {last_b:.4g} (diff: {sign}{diff:.4g})"
                )

    summary = f"Comparison of {name_a} vs {name_b} across {len(combined_groups)} metric group(s)"
    notes = [
        f"Comparing {name_a} and {name_b}. Solid lines: {name_a}; Dashed lines: {name_b} (or role-specific).",
    ]
    if diff_summaries:
        notes.append("Final metric differences: " + " · ".join(diff_summaries))

    return (
        {
            "groups": combined_groups,
            "run_a": name_a,
            "run_b": name_b,
            "differences": diff_summaries,
        },
        summary,
        notes,
    )


def _extract_floats(values: Any, label: str) -> list[float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        # Check torch/numpy
        if hasattr(values, "detach") and hasattr(values, "tolist"):
            values = values.detach().cpu().tolist()
        elif hasattr(values, "tolist"):
            values = values.tolist()
        else:
            raise SeeDLError(f"{label} must be a sequence of real numbers.")
    result = []
    for v in values:
        try:
            val = float(v)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SeeDLError(f"{label} contains non-numeric value: {v!r}") from exc
        result.append(val)
    return result


class Tracker:
    """Lightweight in-memory experiment and training metric recorder."""

    def __init__(self, name: str = "run") -> None:
        self.name = name
        self._history: dict[str, list[float]] = {}
        self._epochs: list[float] = []

    def log(self, epoch: int | float | None = None, **metrics: float | int) -> None:
        """Record metrics for an epoch or step."""
        if epoch is not None:
            self._epochs.append(float(epoch))
        elif not self._epochs and metrics:
            self._epochs.append(1.0)
        elif (
            metrics
            and len(self._epochs) < max((len(v) for v in self._history.values()), default=0) + 1
        ):
            self._epochs.append(float(len(self._epochs) + 1))

        for k, v in metrics.items():
            try:
                val = float(v)
            except (TypeError, ValueError):
                continue
            self._history.setdefault(k, []).append(val)

    def history(self) -> dict[str, Any]:
        """Return a training history mapping compatible with sd.see()."""
        res = dict(self._history)
        if self._epochs:
            res["epochs"] = list(self._epochs)
        return res

    def save(self, filepath: str) -> None:
        """Save history dictionary to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.history(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> Tracker:
        """Load history from a JSON file."""
        tracker = cls()
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        epochs = data.pop("epochs", None)
        if epochs:
            tracker._epochs = [float(e) for e in epochs]
        tracker._history = {k: [float(v) for v in seq] for k, seq in data.items()}
        return tracker

    def __enter__(self) -> Tracker:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass
