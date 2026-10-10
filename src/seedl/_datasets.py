"""Dataset and DataLoader batch inspection with bounded sampling."""

from __future__ import annotations

import importlib
from collections import Counter
from typing import Any

from ._tensors import make_image_grid


def _torch():
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError:
        return None


def inspect_dataset(
    data: Any,
    *,
    labels: Any | None = None,
    max_samples: int = 64,
) -> tuple[str, dict[str, Any], str, list[str]]:
    """Inspect a dataset, DataLoader batch, or sample collection safely."""
    torch = _torch()

    # If it's a PyTorch DataLoader, safely extract first batch without consuming whole iterator
    samples = None
    extracted_labels = labels

    if hasattr(data, "__iter__") and not isinstance(data, (list, tuple, dict)):
        try:
            it = iter(data)
            first_batch = next(it)
            if isinstance(first_batch, (tuple, list)) and len(first_batch) >= 2:
                samples = first_batch[0]
                extracted_labels = first_batch[1]
            else:
                samples = first_batch
        except Exception:
            pass

    if samples is None:
        samples = data

    if (
        hasattr(samples, "__len__")
        and hasattr(samples, "__getitem__")
        and not hasattr(samples, "shape")
    ):
        try:
            count = len(samples)
            sample_items = [samples[i] for i in range(min(count, max_samples))]
            if sample_items and isinstance(sample_items[0], (tuple, list)):
                if len(sample_items[0]) >= 2:
                    raw_x = [item[0] for item in sample_items]
                    extracted_labels = [item[1] for item in sample_items]
                else:
                    raw_x = [item[0] for item in sample_items]
            else:
                raw_x = sample_items

            if torch is not None and raw_x and isinstance(raw_x[0], torch.Tensor):
                samples = torch.stack(raw_x)
            else:
                samples = raw_x
        except Exception:
            pass

    # Extract shape if tensor/numpy
    shape = getattr(samples, "shape", None)
    if shape is not None:
        shape = tuple(int(s) for s in shape)

    # Convert labels to list of strings/ints if present
    label_list = None
    if extracted_labels is not None:
        if hasattr(extracted_labels, "tolist"):
            label_list = extracted_labels.tolist()
        elif isinstance(extracted_labels, (list, tuple)):
            label_list = list(extracted_labels)

    class_counts: dict[str, int] = {}
    if label_list:
        counts = Counter(str(lbl) for lbl in label_list)
        class_counts = dict(counts.most_common(20))

    notes = [
        "Dataset inspection is bounded; no iterators were exhausted and no data was downloaded."
    ]

    # Check for image batch: 4D tensor (N, C, H, W) or (N, H, W, C)
    if shape is not None and len(shape) == 4 and shape[0] >= 1:
        n, c_or_h = shape[0], shape[1]
        is_chw = c_or_h in (1, 3, 4) and min(shape[2:]) >= 4
        is_hwc = shape[-1] in (1, 3, 4) and min(shape[1:3]) >= 4

        if is_chw or is_hwc:
            channels = c_or_h if is_chw else shape[-1]
            num_display = min(n, 16)
            image_tiles = []
            for i in range(num_display):
                img = samples[i]
                if is_chw and torch is not None and isinstance(img, torch.Tensor):
                    img = img.permute(1, 2, 0)
                if hasattr(img, "detach") and hasattr(img, "cpu"):
                    arr = img.detach().cpu().to(dtype=torch.float32)
                    # normalize
                    low, high = float(arr.min().item()), float(arr.max().item())
                    if high > low:
                        norm = ((arr - low) / (high - low)).tolist()
                    else:
                        norm = torch.full_like(arr, 0.5).tolist()
                    image_tiles.append(norm if channels > 1 else [[p[0]] for p in norm])
                elif hasattr(img, "tolist"):
                    image_tiles.append(img.tolist())

            grid, _, rows, cols = make_image_grid(image_tiles, channels, max_images=16)
            summary = f"Image dataset preview: {n} sample(s) with shape {shape[1:]}"
            notes.append(f"Showing {num_display} sample images arranged in a {rows}×{cols} grid.")
            if class_counts:
                notes.append(
                    f"Observed classes in sample: {', '.join(f'{k} ({v})' for k, v in list(class_counts.items())[:5])}."
                )

            data_payload = {
                "kind": "image_dataset",
                "pixels": grid,
                "grid": grid,
                "channels": channels,
                "shape": shape,
                "sample_count": n,
                "num_samples": n,
                "class_counts": class_counts,
                "labels": label_list[:num_display] if label_list else None,
            }
            return "dataset_images", data_payload, summary, notes

    # Tabular / 2D numerical inspection
    if shape is not None and len(shape) == 2:
        num_samples, num_features = shape
        summary = f"Tabular dataset: {num_samples} sample(s) × {num_features} feature(s)"
        notes.append(f"Shape: {num_samples} rows × {num_features} columns.")

        # Compute summary stats across columns (up to 20 columns)
        feature_stats = []
        for col_idx in range(min(num_features, 20)):
            if hasattr(samples, "shape"):
                col = samples[:, col_idx]  # type: ignore[call-overload]
            else:
                col = [row[col_idx] for row in samples if len(row) > col_idx]
            if hasattr(col, "detach") and hasattr(col, "cpu"):
                vals = col.detach().cpu().to(dtype=torch.float32)
                mean_v = float(vals.mean().item())
                min_v = float(vals.min().item())
                max_v = float(vals.max().item())
            elif hasattr(col, "mean"):
                mean_v = float(col.mean())
                min_v = float(col.min())
                max_v = float(col.max())
            else:
                mean_v, min_v, max_v = 0.0, 0.0, 0.0
            feature_stats.append(
                {
                    "feature": f"Feature {col_idx}",
                    "mean": mean_v,
                    "min": min_v,
                    "max": max_v,
                }
            )

        data_payload = {
            "kind": "tabular_dataset",
            "samples": num_samples,
            "num_samples": num_samples,
            "features": num_features,
            "feature_stats": feature_stats,
            "class_counts": class_counts,
        }
        return "dataset_tabular", data_payload, summary, notes

    # General sequence or item collection
    count = len(samples) if hasattr(samples, "__len__") else max_samples
    summary = f"Dataset collection: {count} sample(s)"
    data_payload = {
        "kind": "general_dataset",
        "count": count,
        "class_counts": class_counts,
    }
    return "dataset_general", data_payload, summary, notes
