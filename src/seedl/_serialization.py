"""Safe, reproducible analysis artifact serialization without pickle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .exceptions import SeeDLError
from .view import Visualization

SCHEMA_VERSION = "1.0.0"
MAX_ARTIFACT_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB safety limit


def _compute_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def save_analysis(
    views: Visualization | list[Visualization],
    filepath: str | Path,
    *,
    indent: int = 2,
) -> Path:
    """Save one or more Visualization instances to a safe, reproducible JSON artifact.

    Invariants:
    - Pure JSON format; absolutely no Python bytecode or pickle serialization.
    - Includes schema version, SeeDL version, and SHA-256 payload integrity hash.
    - Preserves all view data, metadata, notes, and backend specifications.
    """
    path = Path(filepath)
    view_list = [views] if isinstance(views, Visualization) else list(views)

    serialized_views = []
    for v in view_list:
        if not isinstance(v, Visualization):
            raise SeeDLError(f"Expected Visualization instance, got {type(v).__name__}.")
        serialized_views.append(
            {
                "kind": v.kind,
                "title": v.title,
                "summary": v.summary,
                "data": v.data,
                "metadata": v.metadata,
                "notes": v.notes,
                "backend": v.backend,
            }
        )

    payload = {
        "$schema": f"https://seedl.dev/schemas/analysis-v{SCHEMA_VERSION}.json",
        "schema_version": SCHEMA_VERSION,
        "seedl_version": "0.4.0",
        "views": serialized_views,
    }

    raw_json = json.dumps(payload, indent=indent, allow_nan=False)
    encoded = raw_json.encode("utf-8")

    if len(encoded) > MAX_ARTIFACT_SIZE_BYTES:
        raise SeeDLError(
            f"Artifact size ({len(encoded) / (1024 * 1024):.1f} MB) exceeds safety limit of {MAX_ARTIFACT_SIZE_BYTES / (1024 * 1024):.1f} MB."
        )

    integrity_hash = _compute_hash(encoded)
    payload["integrity_hash"] = integrity_hash

    final_json = json.dumps(payload, indent=indent, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(final_json, encoding="utf-8")
    return path


def load_analysis(filepath: str | Path) -> list[Visualization] | Visualization:
    """Load reproducible Visualization instances from a safe JSON artifact.

    Invariants:
    - Checks file size before reading to prevent out-of-memory denial-of-service.
    - Validates schema structure and required fields.
    - Reconstructs authentic, interactive Visualization objects.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Analysis file not found: {path}")

    # Check file size prior to reading
    file_size = path.stat().st_size
    if file_size > MAX_ARTIFACT_SIZE_BYTES:
        raise SeeDLError(
            f"File size ({file_size / (1024 * 1024):.1f} MB) exceeds maximum allowed artifact size ({MAX_ARTIFACT_SIZE_BYTES / (1024 * 1024):.1f} MB)."
        )

    raw_text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw_text)
    except Exception as exc:
        raise SeeDLError(f"Malformed analysis JSON file: {exc}") from exc

    if not isinstance(data, dict):
        raise SeeDLError("Invalid artifact: Root JSON must be an object.")

    schema_ver = data.get("schema_version")
    if not schema_ver:
        raise SeeDLError("Missing 'schema_version' in analysis artifact.")

    views_data = data.get("views")
    if not isinstance(views_data, list):
        raise SeeDLError("Invalid artifact: 'views' must be a list.")

    reconstructed: list[Visualization] = []
    for item in views_data:
        if not isinstance(item, dict) or "kind" not in item:
            raise SeeDLError("Invalid view specification in artifact.")
        v = Visualization(
            kind=item["kind"],
            title=item.get("title", ""),
            summary=item.get("summary", ""),
            data=item.get("data", {}),
            metadata=item.get("metadata", {}),
            notes=item.get("notes", []),
            backend=item.get("backend", "svg"),
        )
        reconstructed.append(v)

    return reconstructed[0] if len(reconstructed) == 1 else reconstructed
