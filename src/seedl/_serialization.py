"""Safe, reproducible analysis artifact serialization without pickle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ._version import __version__
from .exceptions import SeeDLError
from .view import Visualization

SCHEMA_VERSION = "1.0.0"
MAX_ARTIFACT_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB safety limit


def _canonical_json_bytes(data: dict[str, Any]) -> bytes:
    """Generate deterministic, canonical UTF-8 JSON bytes for hashing.

    Invariants:
    - Excludes the 'integrity_hash' field itself.
    - Keys are sorted lexicographically at all levels.
    - Compact separators without whitespace (',' and ':').
    - Disallows non-standard IEEE 754 float values (NaN, Infinity).
    - Preserves Unicode characters directly in UTF-8 (ensure_ascii=False).
    """
    cleaned = {k: v for k, v in data.items() if k != "integrity_hash"}
    return json.dumps(
        cleaned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


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
    - Hash is computed deterministically over canonical JSON bytes of all artifact data.
    - Preserves all view data, metadata, notes, and backend specifications.
    """
    path = Path(filepath)
    view_list = [views] if isinstance(views, Visualization) else list(views)
    if not view_list:
        raise SeeDLError("No visualizations provided to save_analysis.")

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

    payload: dict[str, Any] = {
        "$schema": f"https://seedl.dev/schemas/analysis-v{SCHEMA_VERSION}.json",
        "schema_version": SCHEMA_VERSION,
        "seedl_version": __version__,
        "views": serialized_views,
    }

    try:
        canonical_bytes = _canonical_json_bytes(payload)
    except Exception as exc:
        raise SeeDLError(f"Failed to serialize analysis data to canonical JSON: {exc}") from exc

    integrity_hash = hashlib.sha256(canonical_bytes).hexdigest()
    payload["integrity_hash"] = integrity_hash

    final_json = json.dumps(payload, indent=indent, ensure_ascii=False, allow_nan=False)
    encoded = final_json.encode("utf-8")

    if len(encoded) > MAX_ARTIFACT_SIZE_BYTES:
        raise SeeDLError(
            f"Artifact size ({len(encoded) / (1024 * 1024):.1f} MB) exceeds safety limit of {MAX_ARTIFACT_SIZE_BYTES / (1024 * 1024):.1f} MB."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(final_json, encoding="utf-8")
    return path


def load_analysis(filepath: str | Path) -> list[Visualization] | Visualization:
    """Load reproducible Visualization instances from a safe JSON artifact.

    Invariants:
    - Checks file size before reading to prevent out-of-memory denial-of-service.
    - Validates schema structure and enforces supported schema versions.
    - Verifies SHA-256 payload integrity hash against canonical bytes.
    - Validates view field types.
    - Reconstructs authentic, interactive Visualization objects without pickle.
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
    raw_bytes_len = len(raw_text.encode("utf-8"))
    if raw_bytes_len > MAX_ARTIFACT_SIZE_BYTES:
        raise SeeDLError(
            f"File size ({raw_bytes_len / (1024 * 1024):.1f} MB) exceeds maximum allowed artifact size ({MAX_ARTIFACT_SIZE_BYTES / (1024 * 1024):.1f} MB)."
        )

    try:
        data = json.loads(raw_text)
    except Exception as exc:
        raise SeeDLError(f"Malformed analysis JSON file: {exc}") from exc

    if not isinstance(data, dict):
        raise SeeDLError("Invalid artifact: Root JSON must be an object.")

    # 1. Schema version validation
    schema_ver = data.get("schema_version")
    if not schema_ver:
        raise SeeDLError("Missing 'schema_version' in analysis artifact.")
    if schema_ver != SCHEMA_VERSION:
        raise SeeDLError(
            f"Unsupported schema version: '{schema_ver}'. Supported version is '{SCHEMA_VERSION}'."
        )

    # 2. Integrity hash verification
    expected_hash = data.get("integrity_hash")
    if expected_hash is None:
        raise SeeDLError("Missing 'integrity_hash' in analysis artifact.")
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or not all(c in "0123456789abcdefABCDEF" for c in expected_hash)
    ):
        raise SeeDLError(
            "Malformed 'integrity_hash' in analysis artifact; expected 64-character hex string."
        )

    try:
        canonical_bytes = _canonical_json_bytes(data)
    except Exception as exc:
        raise SeeDLError(
            f"Failed to compute canonical JSON for integrity verification: {exc}"
        ) from exc

    actual_hash = hashlib.sha256(canonical_bytes).hexdigest()
    if actual_hash.lower() != expected_hash.lower():
        raise SeeDLError(
            "Integrity verification failed: artifact content does not match recorded SHA-256 integrity hash."
        )

    # 3. Views validation and reconstruction
    views_data = data.get("views")
    if not isinstance(views_data, list) or len(views_data) == 0:
        raise SeeDLError("Invalid artifact: 'views' must be a non-empty list.")

    reconstructed: list[Visualization] = []
    for idx, item in enumerate(views_data):
        if not isinstance(item, dict):
            raise SeeDLError(f"Invalid view at index {idx}: must be an object.")
        kind = item.get("kind")
        if not isinstance(kind, str) or not kind.strip():
            raise SeeDLError(f"Invalid view at index {idx}: 'kind' must be a non-empty string.")

        title = item.get("title", "")
        if not isinstance(title, str):
            raise SeeDLError(f"Invalid view at index {idx}: 'title' must be a string.")

        summary = item.get("summary", "")
        if not isinstance(summary, str):
            raise SeeDLError(f"Invalid view at index {idx}: 'summary' must be a string.")

        view_data = item.get("data", {})
        if not isinstance(view_data, dict):
            raise SeeDLError(f"Invalid view at index {idx}: 'data' must be an object.")

        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            raise SeeDLError(f"Invalid view at index {idx}: 'metadata' must be an object.")

        notes = item.get("notes", [])
        if not isinstance(notes, list) or not all(isinstance(n, str) for n in notes):
            raise SeeDLError(f"Invalid view at index {idx}: 'notes' must be a list of strings.")

        backend = item.get("backend", "svg")
        if not isinstance(backend, str):
            raise SeeDLError(f"Invalid view at index {idx}: 'backend' must be a string.")

        v = Visualization(
            kind=kind,
            title=title,
            summary=summary,
            data=view_data,
            metadata=metadata,
            notes=notes,
            backend=backend,
        )
        reconstructed.append(v)

    return reconstructed[0] if len(reconstructed) == 1 else reconstructed
