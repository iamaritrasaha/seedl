"""Tests for safe, reproducible analysis artifact serialization in SeeDL 0.5.0."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

import seedl as sd
from seedl._serialization import _canonical_json_bytes, load_analysis, save_analysis
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


def test_serialization_oversized_file_rejection(monkeypatch):
    """load_analysis rejects files exceeding MAX_ARTIFACT_SIZE_BYTES before reading."""
    import seedl._serialization as ser

    # Temporarily set max size to 50 bytes for testing
    monkeypatch.setattr(ser, "MAX_ARTIFACT_SIZE_BYTES", 50)

    with tempfile.TemporaryDirectory() as tmp_dir:
        large_file = Path(tmp_dir) / "large.json"
        large_file.write_text("x" * 200, encoding="utf-8")

        with pytest.raises(SeeDLError, match="exceeds maximum allowed artifact size"):
            load_analysis(large_file)


def test_serialization_tampered_numerical_value_rejection():
    """Modifying a numerical value in an artifact causes load_analysis to reject it."""
    viz = _create_sample_viz("Original Viz")
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "tampered_num.json"
        save_analysis(viz, path)

        raw = json.loads(path.read_text(encoding="utf-8"))
        # Tamper with numerical data
        raw["views"][0]["data"]["outlier_fraction"] = 0.99
        path.write_text(json.dumps(raw), encoding="utf-8")

        with pytest.raises(SeeDLError, match="Integrity verification failed"):
            load_analysis(path)


def test_serialization_tampered_title_rejection():
    """Modifying a title in an artifact causes load_analysis to reject it."""
    viz = _create_sample_viz("Original Viz")
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "tampered_title.json"
        save_analysis(viz, path)

        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["views"][0]["title"] = "Forged Title"
        path.write_text(json.dumps(raw), encoding="utf-8")

        with pytest.raises(SeeDLError, match="Integrity verification failed"):
            load_analysis(path)


def test_serialization_incorrect_and_missing_hash_rejection():
    """Rejects artifacts with incorrect, missing, or malformed integrity hashes."""
    viz = _create_sample_viz("Hash Test")
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "corrupt_hash.json"
        save_analysis(viz, path)

        raw = json.loads(path.read_text(encoding="utf-8"))

        # Incorrect hash (wrong 64-char hex)
        raw["integrity_hash"] = "0" * 64
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(SeeDLError, match="Integrity verification failed"):
            load_analysis(path)

        # Missing hash
        del raw["integrity_hash"]
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(SeeDLError, match="Missing 'integrity_hash'"):
            load_analysis(path)

        # Malformed hash (wrong length / non-hex)
        raw["integrity_hash"] = "not_a_valid_sha256"
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(SeeDLError, match="Malformed 'integrity_hash'"):
            load_analysis(path)


def test_serialization_unsupported_schema_version():
    """load_analysis rejects artifacts declaring an unsupported schema version."""
    viz = _create_sample_viz("Schema Test")
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "future_schema.json"
        save_analysis(viz, path)

        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["schema_version"] = "2.0.0"
        path.write_text(json.dumps(raw), encoding="utf-8")

        with pytest.raises(SeeDLError, match="Unsupported schema version: '2.0.0'"):
            load_analysis(path)


def test_serialization_unicode_roundtrip():
    """Unicode characters, mathematical symbols, and non-ASCII notes roundtrip accurately."""
    viz = Visualization(
        kind="dataset_health",
        title="λ_1 = 0.42 ∇θ — 日本語 — 🚀",
        summary="Spectral audit: ∑ α_i > 0.95",
        data={"metric_α": 1.618, "greek": "λ, μ, σ, ω"},
        metadata={"Project": "日本語テスト", "Symbol": "∂L/∂w"},
        notes=["Note with emojis 🧠⚡ and math: ∫ f(x) dx"],
        backend="svg",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "unicode.seedl.json"
        save_analysis(viz, path)

        restored = load_analysis(path)
        assert isinstance(restored, Visualization)
        assert restored.title == "λ_1 = 0.42 ∇θ — 日本語 — 🚀"
        assert restored.summary == "Spectral audit: ∑ α_i > 0.95"
        assert restored.data["greek"] == "λ, μ, σ, ω"
        assert restored.metadata["Project"] == "日本語テスト"
        assert restored.notes[0] == "Note with emojis 🧠⚡ and math: ∫ f(x) dx"


def test_serialization_view_field_validation():
    """load_analysis enforces strict field typing on reconstructed views."""

    def _sign(payload: dict) -> None:
        payload["integrity_hash"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()

    viz = _create_sample_viz("Field Type Test")
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "field_test.json"
        save_analysis(viz, path)

        raw = json.loads(path.read_text(encoding="utf-8"))

        # Empty views list
        raw["views"] = []
        _sign(raw)
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(SeeDLError, match="views' must be a non-empty list"):
            load_analysis(path)

        # Non-dict view item
        raw["views"] = ["not_a_dict"]
        _sign(raw)
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(SeeDLError, match="must be an object"):
            load_analysis(path)

        # Missing kind
        raw["views"] = [{"title": "Missing Kind"}]
        _sign(raw)
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(SeeDLError, match="'kind' must be a non-empty string"):
            load_analysis(path)

        # Non-dict data
        raw["views"] = [{"kind": "test", "data": "not_a_dict"}]
        _sign(raw)
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(SeeDLError, match="'data' must be an object"):
            load_analysis(path)

        # Non-list notes
        raw["views"] = [{"kind": "test", "notes": "not_a_list"}]
        _sign(raw)
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(SeeDLError, match="'notes' must be a list of strings"):
            load_analysis(path)
