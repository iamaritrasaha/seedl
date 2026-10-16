"""Tests for safe, reproducible analysis artifact serialization in SeeDL 0.5.0."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import seedl as sd
from seedl._serialization import load_analysis, save_analysis
from seedl.exceptions import SeeDLError
from seedl.view import Visualization


def _create_sample_viz(title: str = "Test View") -> Visualization:
    return Visualization(
        kind="dataset_health",
        title=title,
        summary="Dataset audit report",
        data={"sample_count": 100, "outlier_fraction": 0.02, "min": -1.5, "max": 2.5},
        metadata={"Samples": 100, "Quality": "High"},
        notes=["Clean distribution", "No NaNs observed"],
        backend="svg",
    )


def test_serialization_single_view_roundtrip():
    """Serialize a single visualization to JSON and restore it authentically."""
    viz = _create_sample_viz("Roundtrip Single")

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "analysis.seedl.json"
        saved = save_analysis(viz, file_path)
        assert saved.exists()

        # Check pure JSON
        content = file_path.read_text(encoding="utf-8")
        parsed = json.loads(content)
        assert "integrity_hash" in parsed
        assert parsed["schema_version"] == "1.0.0"
        assert len(parsed["views"]) == 1

        # Reconstruct
        restored = load_analysis(file_path)
        assert isinstance(restored, Visualization)
        assert restored.kind == viz.kind
        assert restored.title == viz.title
        assert restored.summary == viz.summary
        assert restored.data == viz.data
        assert restored.metadata == viz.metadata
        assert restored.notes == viz.notes


def test_serialization_multiple_views_roundtrip():
    """Serialize multiple visualizations to JSON and restore them as a list."""
    v1 = _create_sample_viz("View A")
    v2 = _create_sample_viz("View B")

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "multi.seedl.json"
        save_analysis([v1, v2], file_path)

        restored = load_analysis(file_path)
        assert isinstance(restored, list)
        assert len(restored) == 2
        assert restored[0].title == "View A"
        assert restored[1].title == "View B"


def test_serialization_malformed_and_schema_validation():
    """load_analysis rejects malformed files or files missing schema version."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        bad_json = Path(tmp_dir) / "bad.json"
        bad_json.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(SeeDLError, match="Malformed analysis JSON"):
            load_analysis(bad_json)

        no_schema = Path(tmp_dir) / "no_schema.json"
        no_schema.write_text(json.dumps({"views": []}), encoding="utf-8")
        with pytest.raises(SeeDLError, match="Missing 'schema_version'"):
            load_analysis(no_schema)


def test_sd_public_api_save_and_load_analysis():
    """sd.save_analysis and sd.load_analysis work from the root package namespace."""
    viz = _create_sample_viz("Root API Test")

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "api_test.json"
        sd.save_analysis(viz, path)
        loaded = sd.load_analysis(path)
        assert isinstance(loaded, Visualization)
        assert loaded.title == "Root API Test"
